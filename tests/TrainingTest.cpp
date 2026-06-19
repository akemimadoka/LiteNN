#include <gtest/gtest.h>

#include <LiteNN.h>

#include <cmath>
#include <optional>
#include <ranges>
#include <vector>

using namespace LiteNN;

namespace
{
	float ReadFloat(const Tensor<CPU>& tensor, std::size_t index)
	{
		const auto cpuTensor = tensor.CopyToDevice(CPU{});
		return static_cast<const float*>(cpuTensor.UnsafeRawData())[index];
	}

	float ReadVariableDataFloat(const Graph& graph, std::size_t variableIndex, std::size_t elementIndex)
	{
		const auto tensor = graph.GetVariable(variableIndex)->Data().CopyToDevice(CPU{});
		return ReadFloat(tensor, elementIndex);
	}

	float ReadVariableGradFloat(const Graph& graph, std::size_t variableIndex, std::size_t elementIndex)
	{
		const auto tensor = graph.GetVariable(variableIndex)->Grad().CopyToDevice(CPU{});
		return ReadFloat(tensor, elementIndex);
	}
} // namespace

TEST(Training, StepRunsForwardBackwardStoresGradientsAndUpdatesVariables)
{
	ModelGraph model;
	Graph& graph = model.UnsafeMutableGraph();
	const auto weightIndex = graph.AddVariable(Variable::Create(Tensor<CPU>({ 3.0f }, { 1 })));

	Subgraph sg;
	const auto x = sg.AddParam(DataType::Float32, { 1 });
	const auto weight = sg.AddNode(VariableRefNode{ weightIndex }, { OutputInfo{ DataType::Float32, { 1 } } });
	const auto y = sg.AddNode(BinaryOpNode{ BinaryOp::Multiply, { x, 0 }, { weight, 0 } },
	                          { OutputInfo{ DataType::Float32, { 1 } } });
	sg.SetResults({ { y, 0 } });
	graph.SetForward(graph.AddSubgraph(std::move(sg)));

	Training::Trainer<CPU, Optimizer::SGD> trainer(model, Optimizer::SGD(0.1f));
	EXPECT_EQ(trainer.ExecutionPolicy(), Training::TrainExecutionPolicy::Interpreter);
	EXPECT_NO_THROW(Training::ValidateTrainStepPlan(trainer.Plan()));
	const auto countAbiRole = [&](Training::TrainStepABIRole role) {
		return static_cast<std::size_t>(
		    std::ranges::count_if(trainer.Plan().abiBindings,
		                          [&](const Training::TrainStepABIBinding& binding) { return binding.role == role; }));
	};
	EXPECT_EQ(countAbiRole(Training::TrainStepABIRole::MutableParameter), 1u);
	EXPECT_EQ(countAbiRole(Training::TrainStepABIRole::Gradient), 1u);
	EXPECT_EQ(countAbiRole(Training::TrainStepABIRole::LossInput), 1u);
	EXPECT_EQ(trainer.Plan().abiBindings[0].runtimeState, std::optional<std::size_t>{ 0u });
	std::vector<Tensor<CPU>> inputs;
	inputs.emplace_back(Tensor<CPU>({ 2.0f }, { 1 }));
	std::vector<Tensor<CPU>> outputGradients;
	outputGradients.emplace_back(Tensor<CPU>({ 2.0f }, { 1 }));

	auto result = trainer.Step(inputs, outputGradients);

	ASSERT_EQ(result.outputs.size(), 1);
	ASSERT_EQ(result.backwardResults.size(), 2);
	EXPECT_FLOAT_EQ(ReadFloat(result.outputs[0], 0), 6.0f);
	EXPECT_FLOAT_EQ(ReadFloat(result.backwardResults[0], 0), 6.0f);
	EXPECT_FLOAT_EQ(ReadFloat(result.backwardResults[1], 0), 4.0f);
	EXPECT_FLOAT_EQ(ReadVariableGradFloat(graph, weightIndex, 0), 4.0f);
	EXPECT_FLOAT_EQ(ReadVariableDataFloat(graph, weightIndex, 0), 2.6f);
}

TEST(Training, TrainerExposesParameterSetAndStateDict)
{
	ModelGraph model;
	Graph& graph = model.UnsafeMutableGraph();
	const auto weightIndex = graph.AddVariable(Variable::Create(Tensor<CPU>({ 3.0f }, { 1 })));
	graph.SetVariableName(weightIndex, "linear.weight");

	Subgraph sg;
	const auto x = sg.AddParam(DataType::Float32, { 1 });
	const auto weight = sg.AddNode(VariableRefNode{ weightIndex }, { OutputInfo{ DataType::Float32, { 1 } } });
	const auto y = sg.AddNode(BinaryOpNode{ BinaryOp::Multiply, { x, 0 }, { weight, 0 } },
	                          { OutputInfo{ DataType::Float32, { 1 } } });
	sg.SetResults({ { y, 0 } });
	graph.SetForward(graph.AddSubgraph(std::move(sg)));

	Training::Trainer<CPU, Optimizer::SGD> trainer(model, Optimizer::SGD(0.1f));
	ASSERT_EQ(trainer.Parameters().Size(), 1);
	EXPECT_EQ(trainer.Parameters()[0].name, "linear.weight");

	auto state = trainer.SaveStateDict();
	ASSERT_EQ(state.parameters.size(), 1);
	EXPECT_EQ(state.parameters[0].name, "linear.weight");
	EXPECT_FLOAT_EQ(ReadFloat(state.parameters[0].value, 0), 3.0f);

	*static_cast<float*>(trainer.Parameters()[0].Parameter().UnsafeRawData()) = 42.0f;
	EXPECT_FLOAT_EQ(ReadVariableDataFloat(graph, weightIndex, 0), 42.0f);
	trainer.LoadStateDict(state);
	EXPECT_FLOAT_EQ(ReadVariableDataFloat(graph, weightIndex, 0), 3.0f);
}

#ifdef LITENN_ENABLE_TRAINING_AOT
TEST(Training, AOTPolicyRunsForwardBackwardAndRefreshesUpdatedWeights)
{
	ModelGraph model;
	Graph& graph = model.UnsafeMutableGraph();
	const auto weightIndex = graph.AddVariable(Variable::Create(Tensor<CPU>({ 3.0f }, { 1 })));

	Subgraph sg;
	const auto x = sg.AddParam(DataType::Float32, { 1 });
	const auto weight = sg.AddNode(VariableRefNode{ weightIndex }, { OutputInfo{ DataType::Float32, { 1 } } });
	const auto y = sg.AddNode(BinaryOpNode{ BinaryOp::Multiply, { x, 0 }, { weight, 0 } },
	                          { OutputInfo{ DataType::Float32, { 1 } } });
	sg.SetResults({ { y, 0 } });
	graph.SetForward(graph.AddSubgraph(std::move(sg)));

	Training::TrainerOptions options;
	options.executionPolicy = Training::TrainExecutionPolicy::AOT;
	Training::Trainer<CPU, Optimizer::SGD> trainer(model, Optimizer::SGD(0.1f), options);
	EXPECT_EQ(trainer.ExecutionPolicy(), Training::TrainExecutionPolicy::AOT);
	EXPECT_TRUE(trainer.UsesCompiledOptimizerUpdateEntries());

	std::vector<Tensor<CPU>> inputs;
	inputs.emplace_back(Tensor<CPU>({ 2.0f }, { 1 }));
	auto outputs = trainer.Forward(inputs);

	ASSERT_EQ(outputs.size(), 1);
	EXPECT_FLOAT_EQ(ReadFloat(outputs[0], 0), 6.0f);
	std::vector<Tensor<CPU>> outputGradients;
	outputGradients.emplace_back(Tensor<CPU>({ 2.0f }, { 1 }));

	auto firstStep = trainer.Step(inputs, outputGradients);
	ASSERT_EQ(firstStep.outputs.size(), 1);
	ASSERT_EQ(firstStep.backwardResults.size(), 2);
	EXPECT_FLOAT_EQ(ReadFloat(firstStep.outputs[0], 0), 6.0f);
	EXPECT_FLOAT_EQ(ReadFloat(firstStep.backwardResults[0], 0), 6.0f);
	EXPECT_FLOAT_EQ(ReadFloat(firstStep.backwardResults[1], 0), 4.0f);
	EXPECT_FLOAT_EQ(ReadVariableDataFloat(graph, weightIndex, 0), 2.6f);

	auto secondStep = trainer.Step(inputs, outputGradients);
	ASSERT_EQ(secondStep.outputs.size(), 1);
	ASSERT_EQ(secondStep.backwardResults.size(), 2);
	EXPECT_FLOAT_EQ(ReadFloat(secondStep.outputs[0], 0), 5.2f);
	EXPECT_FLOAT_EQ(ReadFloat(secondStep.backwardResults[0], 0), 5.2f);
	EXPECT_FLOAT_EQ(ReadFloat(secondStep.backwardResults[1], 0), 4.0f);
	EXPECT_FLOAT_EQ(ReadVariableDataFloat(graph, weightIndex, 0), 2.2f);
}

TEST(Training, AOTPolicyRunsAdamWCompiledOptimizerStateUpdate)
{
	ModelGraph model;
	Graph& graph = model.UnsafeMutableGraph();
	const auto weightIndex = graph.AddVariable(Variable::Create(Tensor<CPU>({ 3.0f }, { 1 })));

	Subgraph sg;
	const auto x = sg.AddParam(DataType::Float32, { 1 });
	const auto weight = sg.AddNode(VariableRefNode{ weightIndex }, { OutputInfo{ DataType::Float32, { 1 } } });
	const auto y = sg.AddNode(BinaryOpNode{ BinaryOp::Multiply, { x, 0 }, { weight, 0 } },
	                          { OutputInfo{ DataType::Float32, { 1 } } });
	sg.SetResults({ { y, 0 } });
	graph.SetForward(graph.AddSubgraph(std::move(sg)));

	Training::TrainerOptions options;
	options.executionPolicy = Training::TrainExecutionPolicy::AOT;
	Optimizer::AdamWOptions adamwOptions;
	adamwOptions.learningRate = 0.1f;
	adamwOptions.beta1 = 0.0f;
	adamwOptions.beta2 = 0.0f;
	adamwOptions.epsilon = 1.0e-8f;
	adamwOptions.weightDecay = 0.0f;
	Training::Trainer<CPU, Optimizer::AdamW> trainer(model, Optimizer::AdamW(adamwOptions), options);
	EXPECT_TRUE(trainer.UsesCompiledOptimizerUpdateEntries());

	std::vector<Tensor<CPU>> inputs;
	inputs.emplace_back(Tensor<CPU>({ 2.0f }, { 1 }));
	std::vector<Tensor<CPU>> outputGradients;
	outputGradients.emplace_back(Tensor<CPU>({ 2.0f }, { 1 }));

	auto firstStep = trainer.Step(inputs, outputGradients);
	ASSERT_EQ(firstStep.outputs.size(), 1);
	ASSERT_EQ(firstStep.backwardResults.size(), 2);
	EXPECT_FLOAT_EQ(ReadFloat(firstStep.outputs[0], 0), 6.0f);
	EXPECT_FLOAT_EQ(ReadFloat(firstStep.backwardResults[1], 0), 4.0f);
	EXPECT_NEAR(ReadVariableDataFloat(graph, weightIndex, 0), 2.9f, 1.0e-5f);
	EXPECT_FLOAT_EQ(ReadFloat(trainer.Optimizer().FirstMoment(0), 0), 4.0f);
	EXPECT_FLOAT_EQ(ReadFloat(trainer.Optimizer().SecondMoment(0), 0), 16.0f);

	auto secondStep = trainer.Step(inputs, outputGradients);
	ASSERT_EQ(secondStep.outputs.size(), 1);
	EXPECT_NEAR(ReadFloat(secondStep.outputs[0], 0), 5.8f, 1.0e-5f);
	EXPECT_NEAR(ReadVariableDataFloat(graph, weightIndex, 0), 2.8f, 1.0e-5f);
	EXPECT_EQ(trainer.Optimizer().StepIndex(), 2u);
}
#endif

TEST(Training, StepSoftmaxCrossEntropyComputesLossAndUpdatesVariables)
{
	ModelGraph model;
	Graph& graph = model.UnsafeMutableGraph();
	const auto logitsIndex = graph.AddVariable(Variable::Create(Tensor<CPU>({ 0.0f, 0.0f }, { 2 })));

	Subgraph sg;
	const auto logits = sg.AddNode(VariableRefNode{ logitsIndex }, { OutputInfo{ DataType::Float32, { 2 } } });
	sg.SetResults({ { logits, 0 } });
	graph.SetForward(graph.AddSubgraph(std::move(sg)));

	Training::Trainer<CPU, Optimizer::SGD> trainer(model, Optimizer::SGD(1.0f));
	std::vector<Tensor<CPU>> inputs;

	auto result = trainer.StepSoftmaxCrossEntropy(inputs, 1);

	ASSERT_EQ(result.outputs.size(), 1);
	ASSERT_EQ(result.backwardResults.size(), 1);
	EXPECT_NEAR(result.loss, std::log(2.0), 1.0e-6);
	EXPECT_FLOAT_EQ(ReadFloat(result.outputs[0], 0), 0.0f);
	EXPECT_FLOAT_EQ(ReadFloat(result.outputs[0], 1), 0.0f);
	EXPECT_FLOAT_EQ(ReadFloat(result.backwardResults[0], 0), 0.5f);
	EXPECT_FLOAT_EQ(ReadFloat(result.backwardResults[0], 1), -0.5f);
	EXPECT_FLOAT_EQ(ReadVariableGradFloat(graph, logitsIndex, 0), 0.5f);
	EXPECT_FLOAT_EQ(ReadVariableGradFloat(graph, logitsIndex, 1), -0.5f);
	EXPECT_FLOAT_EQ(ReadVariableDataFloat(graph, logitsIndex, 0), -0.5f);
	EXPECT_FLOAT_EQ(ReadVariableDataFloat(graph, logitsIndex, 1), 0.5f);
}

TEST(Training, StepSoftmaxCrossEntropyBatchAveragesLossAndGradients)
{
	ModelGraph model;
	Graph& graph = model.UnsafeMutableGraph();
	const auto logitsIndex = graph.AddVariable(Variable::Create(Tensor<CPU>({ 0.0f, 0.0f, 0.0f, 0.0f }, { 2, 2 })));

	Subgraph sg;
	const auto logits = sg.AddNode(VariableRefNode{ logitsIndex }, { OutputInfo{ DataType::Float32, { 2, 2 } } });
	sg.SetResults({ { logits, 0 } });
	graph.SetForward(graph.AddSubgraph(std::move(sg)));

	Training::Trainer<CPU, Optimizer::SGD> trainer(model, Optimizer::SGD(1.0f));
	std::vector<Tensor<CPU>> inputs;
	std::vector<std::size_t> targets = { 0, 1 };

	auto result = trainer.StepSoftmaxCrossEntropyBatch(inputs, targets);

	ASSERT_EQ(result.outputs.size(), 1);
	ASSERT_EQ(result.backwardResults.size(), 1);
	EXPECT_NEAR(result.loss, std::log(2.0), 1.0e-6);
	EXPECT_FLOAT_EQ(ReadFloat(result.backwardResults[0], 0), -0.25f);
	EXPECT_FLOAT_EQ(ReadFloat(result.backwardResults[0], 1), 0.25f);
	EXPECT_FLOAT_EQ(ReadFloat(result.backwardResults[0], 2), 0.25f);
	EXPECT_FLOAT_EQ(ReadFloat(result.backwardResults[0], 3), -0.25f);
	EXPECT_FLOAT_EQ(ReadVariableGradFloat(graph, logitsIndex, 0), -0.25f);
	EXPECT_FLOAT_EQ(ReadVariableGradFloat(graph, logitsIndex, 1), 0.25f);
	EXPECT_FLOAT_EQ(ReadVariableGradFloat(graph, logitsIndex, 2), 0.25f);
	EXPECT_FLOAT_EQ(ReadVariableGradFloat(graph, logitsIndex, 3), -0.25f);
	EXPECT_FLOAT_EQ(ReadVariableDataFloat(graph, logitsIndex, 0), 0.25f);
	EXPECT_FLOAT_EQ(ReadVariableDataFloat(graph, logitsIndex, 1), -0.25f);
	EXPECT_FLOAT_EQ(ReadVariableDataFloat(graph, logitsIndex, 2), -0.25f);
	EXPECT_FLOAT_EQ(ReadVariableDataFloat(graph, logitsIndex, 3), 0.25f);
}
