#include <gtest/gtest.h>

#include <LiteNN.h>
#include <LiteNN/Layer/Layer.h>

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

	void BuildInterpreterLocalBackwardStateModel(ModelGraph& model)
	{
		Graph& graph = model.UnsafeMutableGraph();
		const auto slot = graph.AddActivationSlot(TensorType::Dense(DataType::Float32, ShapeView{ 2 }));

		Subgraph forward;
		const auto input = forward.AddParam(DataType::Float32, { 2 });
		forward.SetResults({ { input, 0 } });
		graph.SetForward(graph.AddSubgraph(std::move(forward)));
		graph.SetInputNames({ "x" });
		graph.SetOutputNames({ "y" });

		Subgraph backward;
		const auto backwardInput = backward.AddParam(DataType::Float32, { 2 });
		[[maybe_unused]] const auto outputGradient = backward.AddParam(DataType::Float32, { 2 });
		const auto saved = backward.AddNode(SaveActivationNode{ { backwardInput, 0 }, slot },
		                                    { OutputInfo{ DataType::Float32, { 2 } } });
		backward.SetResults({ { saved, 0 } });
		graph.SetBackward(graph.AddSubgraph(std::move(backward)));
	}

	std::size_t BuildScalarMultiplyModel(ModelGraph& model)
	{
		Graph& graph = model.UnsafeMutableGraph();
		const auto weightIndex = graph.AddVariable(Variable::Create(Tensor<CPU>({ 3.0f }, { 1 })));

		Subgraph sg;
		const auto x = sg.AddParam(DataType::Float32, { 1 });
		const auto weight = sg.AddNode(VariableRefNode{ weightIndex }, { OutputInfo{ DataType::Float32, { 1 } } });
		const auto y = sg.AddNode(BinaryOpNode{ BinaryOp::Multiply, { x, 0 }, { weight, 0 } },
		                          { OutputInfo{ DataType::Float32, { 1 } } });
		sg.SetResults({ { y, 0 } });
		graph.SetForward(graph.AddSubgraph(std::move(sg)));
		return weightIndex;
	}

	std::size_t BuildBroadcastReduceSharedVariableModel(ModelGraph& model)
	{
		Graph& graph = model.UnsafeMutableGraph();
		const auto scaleIndex = graph.AddVariable(Variable::Create(Tensor<CPU>({ 0.5f, -0.25f }, { 1, 2 })));

		Subgraph sg;
		const auto input = sg.AddParam(DataType::Float32, { 2, 2 });
		const auto scaleForMultiply =
		    sg.AddNode(VariableRefNode{ scaleIndex }, { OutputInfo{ DataType::Float32, { 1, 2 } } });
		const auto scaled = sg.AddNode(BinaryOpNode{ BinaryOp::Multiply, { input, 0 }, { scaleForMultiply, 0 } },
		                               { OutputInfo{ DataType::Float32, { 2, 2 } } });
		const auto scaleForAdd =
		    sg.AddNode(VariableRefNode{ scaleIndex }, { OutputInfo{ DataType::Float32, { 1, 2 } } });
		const auto shifted = sg.AddNode(BinaryOpNode{ BinaryOp::Add, { scaled, 0 }, { scaleForAdd, 0 } },
		                                { OutputInfo{ DataType::Float32, { 2, 2 } } });
		const auto reduced =
		    sg.AddNode(ReduceOpNode{ ReduceOp::Sum, { shifted, 0 }, 1 }, { OutputInfo{ DataType::Float32, { 2 } } });
		sg.SetResults({ { reduced, 0 } });
		graph.SetForward(graph.AddSubgraph(std::move(sg)));
		graph.SetInputNames({ "x" });
		graph.SetOutputNames({ "row_sum" });
		return scaleIndex;
	}

	std::size_t BuildExplicitBroadcastToModel(ModelGraph& model)
	{
		Graph& graph = model.UnsafeMutableGraph();
		const auto scaleIndex = graph.AddVariable(Variable::Create(Tensor<CPU>({ 0.5f, -0.25f }, { 2 })));

		Subgraph sg;
		const auto input = sg.AddParam(DataType::Float32, { 2, 2 });
		const auto scale = sg.AddNode(VariableRefNode{ scaleIndex }, { OutputInfo{ DataType::Float32, { 2 } } });
		const auto broadcasted =
		    sg.AddNode(BroadcastToNode{ { scale, 0 }, { 2, 2 } }, { OutputInfo{ DataType::Float32, { 2, 2 } } });
		const auto output = sg.AddNode(BinaryOpNode{ BinaryOp::Multiply, { input, 0 }, { broadcasted, 0 } },
		                               { OutputInfo{ DataType::Float32, { 2, 2 } } });
		sg.SetResults({ { output, 0 } });
		graph.SetForward(graph.AddSubgraph(std::move(sg)));
		graph.SetInputNames({ "x" });
		graph.SetOutputNames({ "scaled" });
		return scaleIndex;
	}

	struct BatchLinearModelIndices
	{
		std::size_t weight{};
		std::size_t bias{};
	};

	BatchLinearModelIndices BuildBatchLinearClassifierModel(ModelGraph& model)
	{
		Graph& graph = model.UnsafeMutableGraph();
		const auto weightIndex =
		    graph.AddVariable(Variable::Create(Tensor<CPU>({ 0.25f, -0.5f, 0.75f, 0.125f }, { 2, 2 })));
		const auto biasIndex = graph.AddVariable(Variable::Create(Tensor<CPU>({ 0.1f, -0.2f }, { 1, 2 })));

		Subgraph sg;
		const auto x = sg.AddParam(DataType::Float32, { 2, 2 });
		const auto weight = sg.AddNode(VariableRefNode{ weightIndex }, { OutputInfo{ DataType::Float32, { 2, 2 } } });
		const auto matmul = sg.AddNode(BinaryOpNode{ BinaryOp::MatMul, { x, 0 }, { weight, 0 } },
		                               { OutputInfo{ DataType::Float32, { 2, 2 } } });
		const auto bias = sg.AddNode(VariableRefNode{ biasIndex }, { OutputInfo{ DataType::Float32, { 1, 2 } } });
		const auto logits = sg.AddNode(BinaryOpNode{ BinaryOp::Add, { matmul, 0 }, { bias, 0 } },
		                               { OutputInfo{ DataType::Float32, { 2, 2 } } });
		sg.SetResults({ { logits, 0 } });
		graph.SetForward(graph.AddSubgraph(std::move(sg)));
		graph.SetInputNames({ "x" });
		graph.SetOutputNames({ "logits" });
		return { weightIndex, biasIndex };
	}

	std::vector<std::size_t> BuildTinyLinearChainClassifierModel(ModelGraph& model)
	{
		ModelBuilder builder;
		auto layer0 = Layer::CreateLinear(builder, Tensor<CPU>({ 0.25f, -0.5f, 0.75f, 0.125f, -0.25f, 0.5f }, { 2, 3 }),
		                                  Tensor<CPU>({ 0.1f, -0.2f, 0.05f }, { 1, 3 }));
		auto layer1 =
		    Layer::CreateLinear(builder, Tensor<CPU>({ 0.5f, -0.25f, -0.125f, 0.375f, 0.25f, 0.125f }, { 3, 2 }),
		                        Tensor<CPU>({ 0.0f, 0.1f }, { 1, 2 }));
		Graph graph = builder.UnsafeTakeGraph();

		Subgraph sg;
		const auto input = sg.AddParam(DataType::Float32, { 2, 2 });
		const auto hidden = Layer::AddLinear(sg, layer0, { input, 0 });
		const auto logits = Layer::AddLinear(sg, layer1, hidden);
		sg.SetResults({ logits });
		graph.SetForward(graph.AddSubgraph(std::move(sg)));
		graph.SetInputNames({ "x" });
		graph.SetOutputNames({ "logits" });

		model = ModelGraph(std::move(graph));
		return { layer0.weightVariable, *layer0.biasVariable, layer1.weightVariable, *layer1.biasVariable };
	}

	std::vector<std::size_t> BuildTinyReLUMLPClassifierModel(ModelGraph& model)
	{
		ModelBuilder builder;
		auto layer0 = Layer::CreateLinear(builder, Tensor<CPU>({ 0.25f, -0.5f, 0.75f, 0.125f, -0.25f, 0.5f }, { 2, 3 }),
		                                  Tensor<CPU>({ 0.1f, -0.2f, 0.05f }, { 1, 3 }));
		auto layer1 =
		    Layer::CreateLinear(builder, Tensor<CPU>({ 0.5f, -0.25f, -0.125f, 0.375f, 0.25f, 0.125f }, { 3, 2 }),
		                        Tensor<CPU>({ 0.0f, 0.1f }, { 1, 2 }));
		Graph graph = builder.UnsafeTakeGraph();

		Subgraph sg;
		const auto input = sg.AddParam(DataType::Float32, { 2, 2 });
		const auto hidden = Layer::AddReLU(sg, Layer::AddLinear(sg, layer0, { input, 0 }));
		const auto logits = Layer::AddLinear(sg, layer1, hidden);
		sg.SetResults({ logits });
		graph.SetForward(graph.AddSubgraph(std::move(sg)));
		graph.SetInputNames({ "x" });
		graph.SetOutputNames({ "logits" });

		model = ModelGraph(std::move(graph));
		return { layer0.weightVariable, *layer0.biasVariable, layer1.weightVariable, *layer1.biasVariable };
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

TEST(Training, AOTPolicyRejectsInterpreterLocalBackwardStateBeforeRunnerInitialization)
{
	ModelGraph model;
	BuildInterpreterLocalBackwardStateModel(model);

	Training::TrainerOptions options;
	options.executionPolicy = Training::TrainExecutionPolicy::AOT;
	EXPECT_THROW((Training::Trainer<CPU, Optimizer::SGD>(model, Optimizer::SGD(0.1f), options)), std::runtime_error);
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

TEST(Training, AOTAndInterpreterSGDStepMatchForScalarGraph)
{
	ModelGraph interpreterModel;
	const auto interpreterWeight = BuildScalarMultiplyModel(interpreterModel);
	ModelGraph aotModel;
	const auto aotWeight = BuildScalarMultiplyModel(aotModel);

	Training::Trainer<CPU, Optimizer::SGD> interpreterTrainer(interpreterModel, Optimizer::SGD(0.1f));
	Training::TrainerOptions aotOptions;
	aotOptions.executionPolicy = Training::TrainExecutionPolicy::AOT;
	Training::Trainer<CPU, Optimizer::SGD> aotTrainer(aotModel, Optimizer::SGD(0.1f), aotOptions);

	std::vector<Tensor<CPU>> inputs;
	inputs.emplace_back(Tensor<CPU>({ 2.0f }, { 1 }));
	std::vector<Tensor<CPU>> outputGradients;
	outputGradients.emplace_back(Tensor<CPU>({ 2.0f }, { 1 }));

	const auto interpreterStep = interpreterTrainer.Step(inputs, outputGradients);
	const auto aotStep = aotTrainer.Step(inputs, outputGradients);

	ASSERT_EQ(aotStep.outputs.size(), interpreterStep.outputs.size());
	ASSERT_EQ(aotStep.backwardResults.size(), interpreterStep.backwardResults.size());
	EXPECT_FLOAT_EQ(ReadFloat(aotStep.outputs[0], 0), ReadFloat(interpreterStep.outputs[0], 0));
	EXPECT_FLOAT_EQ(ReadFloat(aotStep.backwardResults[0], 0), ReadFloat(interpreterStep.backwardResults[0], 0));
	EXPECT_FLOAT_EQ(ReadFloat(aotStep.backwardResults[1], 0), ReadFloat(interpreterStep.backwardResults[1], 0));
	EXPECT_FLOAT_EQ(ReadVariableGradFloat(aotModel.UnsafeMutableGraph(), aotWeight, 0),
	                ReadVariableGradFloat(interpreterModel.UnsafeMutableGraph(), interpreterWeight, 0));
	EXPECT_FLOAT_EQ(ReadVariableDataFloat(aotModel.UnsafeMutableGraph(), aotWeight, 0),
	                ReadVariableDataFloat(interpreterModel.UnsafeMutableGraph(), interpreterWeight, 0));
}

TEST(Training, AOTAndInterpreterBroadcastReduceSharedVariableGradientsMatch)
{
	ModelGraph interpreterModel;
	const auto interpreterScale = BuildBroadcastReduceSharedVariableModel(interpreterModel);
	ModelGraph aotModel;
	const auto aotScale = BuildBroadcastReduceSharedVariableModel(aotModel);

	Training::Trainer<CPU, Optimizer::SGD> interpreterTrainer(interpreterModel, Optimizer::SGD(0.05f));
	Training::TrainerOptions aotOptions;
	aotOptions.executionPolicy = Training::TrainExecutionPolicy::AOT;
	Training::Trainer<CPU, Optimizer::SGD> aotTrainer(aotModel, Optimizer::SGD(0.05f), aotOptions);

	std::vector<Tensor<CPU>> inputs;
	inputs.emplace_back(Tensor<CPU>({ 1.0f, 2.0f, -0.5f, 0.25f }, { 2, 2 }));
	std::vector<Tensor<CPU>> outputGradients;
	outputGradients.emplace_back(Tensor<CPU>({ 1.0f, -0.5f }, { 2 }));

	const auto interpreterStep = interpreterTrainer.Step(inputs, outputGradients);
	const auto aotStep = aotTrainer.Step(inputs, outputGradients);

	ASSERT_EQ(aotStep.outputs.size(), interpreterStep.outputs.size());
	ASSERT_EQ(aotStep.backwardResults.size(), interpreterStep.backwardResults.size());
	for (std::size_t i = 0; i < aotStep.outputs[0].NumElements(); ++i)
	{
		EXPECT_NEAR(ReadFloat(aotStep.outputs[0], i), ReadFloat(interpreterStep.outputs[0], i), 1.0e-5f);
	}
	for (std::size_t resultIndex = 0; resultIndex < aotStep.backwardResults.size(); ++resultIndex)
	{
		ASSERT_EQ(aotStep.backwardResults[resultIndex].NumElements(),
		          interpreterStep.backwardResults[resultIndex].NumElements());
		for (std::size_t i = 0; i < aotStep.backwardResults[resultIndex].NumElements(); ++i)
		{
			EXPECT_NEAR(ReadFloat(aotStep.backwardResults[resultIndex], i),
			            ReadFloat(interpreterStep.backwardResults[resultIndex], i), 1.0e-5f);
		}
	}
	for (std::size_t i = 0; i < 2; ++i)
	{
		EXPECT_NEAR(ReadVariableGradFloat(aotModel.UnsafeMutableGraph(), aotScale, i),
		            ReadVariableGradFloat(interpreterModel.UnsafeMutableGraph(), interpreterScale, i), 1.0e-5f);
		EXPECT_NEAR(ReadVariableDataFloat(aotModel.UnsafeMutableGraph(), aotScale, i),
		            ReadVariableDataFloat(interpreterModel.UnsafeMutableGraph(), interpreterScale, i), 1.0e-5f);
	}
}

TEST(Training, AOTAndInterpreterExplicitBroadcastToGradientsMatch)
{
	ModelGraph interpreterModel;
	const auto interpreterScale = BuildExplicitBroadcastToModel(interpreterModel);
	ModelGraph aotModel;
	const auto aotScale = BuildExplicitBroadcastToModel(aotModel);

	Training::Trainer<CPU, Optimizer::SGD> interpreterTrainer(interpreterModel, Optimizer::SGD(0.05f));
	Training::TrainerOptions aotOptions;
	aotOptions.executionPolicy = Training::TrainExecutionPolicy::AOT;
	Training::Trainer<CPU, Optimizer::SGD> aotTrainer(aotModel, Optimizer::SGD(0.05f), aotOptions);

	std::vector<Tensor<CPU>> inputs;
	inputs.emplace_back(Tensor<CPU>({ 1.0f, 2.0f, -0.5f, 0.25f }, { 2, 2 }));
	std::vector<Tensor<CPU>> outputGradients;
	outputGradients.emplace_back(Tensor<CPU>({ 1.0f, -0.5f, 0.25f, 2.0f }, { 2, 2 }));

	const auto interpreterStep = interpreterTrainer.Step(inputs, outputGradients);
	const auto aotStep = aotTrainer.Step(inputs, outputGradients);

	ASSERT_EQ(aotStep.outputs.size(), interpreterStep.outputs.size());
	ASSERT_EQ(aotStep.backwardResults.size(), interpreterStep.backwardResults.size());
	for (std::size_t resultIndex = 0; resultIndex < aotStep.backwardResults.size(); ++resultIndex)
	{
		ASSERT_EQ(aotStep.backwardResults[resultIndex].NumElements(),
		          interpreterStep.backwardResults[resultIndex].NumElements());
		for (std::size_t i = 0; i < aotStep.backwardResults[resultIndex].NumElements(); ++i)
		{
			EXPECT_NEAR(ReadFloat(aotStep.backwardResults[resultIndex], i),
			            ReadFloat(interpreterStep.backwardResults[resultIndex], i), 1.0e-5f);
		}
	}
	for (std::size_t i = 0; i < 2; ++i)
	{
		EXPECT_NEAR(ReadVariableGradFloat(aotModel.UnsafeMutableGraph(), aotScale, i),
		            ReadVariableGradFloat(interpreterModel.UnsafeMutableGraph(), interpreterScale, i), 1.0e-5f);
		EXPECT_NEAR(ReadVariableDataFloat(aotModel.UnsafeMutableGraph(), aotScale, i),
		            ReadVariableDataFloat(interpreterModel.UnsafeMutableGraph(), interpreterScale, i), 1.0e-5f);
	}
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

TEST(Training, AOTAndInterpreterAdamWStepsKeepOptimizerStateInParity)
{
	ModelGraph interpreterModel;
	const auto interpreterWeight = BuildScalarMultiplyModel(interpreterModel);
	ModelGraph aotModel;
	const auto aotWeight = BuildScalarMultiplyModel(aotModel);

	Optimizer::AdamWOptions adamwOptions;
	adamwOptions.learningRate = 0.05f;
	adamwOptions.beta1 = 0.8f;
	adamwOptions.beta2 = 0.95f;
	adamwOptions.epsilon = 1.0e-8f;
	adamwOptions.weightDecay = 0.01f;
	Training::Trainer<CPU, Optimizer::AdamW> interpreterTrainer(interpreterModel, Optimizer::AdamW(adamwOptions));
	Training::TrainerOptions aotOptions;
	aotOptions.executionPolicy = Training::TrainExecutionPolicy::AOT;
	Training::Trainer<CPU, Optimizer::AdamW> aotTrainer(aotModel, Optimizer::AdamW(adamwOptions), aotOptions);

	std::vector<Tensor<CPU>> inputs;
	inputs.emplace_back(Tensor<CPU>({ 2.0f }, { 1 }));
	std::vector<Tensor<CPU>> outputGradients;
	outputGradients.emplace_back(Tensor<CPU>({ 2.0f }, { 1 }));
	for (std::size_t step = 1; step <= 2; ++step)
	{
		const auto interpreterStep = interpreterTrainer.Step(inputs, outputGradients);
		const auto aotStep = aotTrainer.Step(inputs, outputGradients);

		ASSERT_EQ(aotStep.outputs.size(), interpreterStep.outputs.size());
		ASSERT_EQ(aotStep.backwardResults.size(), interpreterStep.backwardResults.size());
		EXPECT_NEAR(ReadFloat(aotStep.outputs[0], 0), ReadFloat(interpreterStep.outputs[0], 0), 1.0e-5f);
		EXPECT_NEAR(ReadFloat(aotStep.backwardResults[0], 0), ReadFloat(interpreterStep.backwardResults[0], 0),
		            1.0e-5f);
		EXPECT_NEAR(ReadFloat(aotStep.backwardResults[1], 0), ReadFloat(interpreterStep.backwardResults[1], 0),
		            1.0e-5f);
		EXPECT_NEAR(ReadVariableDataFloat(aotModel.UnsafeMutableGraph(), aotWeight, 0),
		            ReadVariableDataFloat(interpreterModel.UnsafeMutableGraph(), interpreterWeight, 0), 1.0e-5f);
		EXPECT_NEAR(ReadFloat(aotTrainer.Optimizer().FirstMoment(0), 0),
		            ReadFloat(interpreterTrainer.Optimizer().FirstMoment(0), 0), 1.0e-5f);
		EXPECT_NEAR(ReadFloat(aotTrainer.Optimizer().SecondMoment(0), 0),
		            ReadFloat(interpreterTrainer.Optimizer().SecondMoment(0), 0), 1.0e-5f);
		EXPECT_EQ(aotTrainer.Optimizer().StepIndex(), step);
		EXPECT_EQ(interpreterTrainer.Optimizer().StepIndex(), step);
	}
}

TEST(Training, AOTAndInterpreterSoftmaxCrossEntropyBatchLinearSGDMatch)
{
	ModelGraph interpreterModel;
	const auto interpreterVariables = BuildBatchLinearClassifierModel(interpreterModel);
	ModelGraph aotModel;
	const auto aotVariables = BuildBatchLinearClassifierModel(aotModel);

	Training::Trainer<CPU, Optimizer::SGD> interpreterTrainer(interpreterModel, Optimizer::SGD(0.05f));
	Training::TrainerOptions aotOptions;
	aotOptions.executionPolicy = Training::TrainExecutionPolicy::AOT;
	Training::Trainer<CPU, Optimizer::SGD> aotTrainer(aotModel, Optimizer::SGD(0.05f), aotOptions);

	std::vector<Tensor<CPU>> inputs;
	inputs.emplace_back(Tensor<CPU>({ 1.0f, 2.0f, -0.5f, 0.25f }, { 2, 2 }));
	std::vector<std::size_t> targets = { 1, 0 };

	const auto interpreterStep = interpreterTrainer.StepSoftmaxCrossEntropyBatch(inputs, targets);
	const auto aotStep = aotTrainer.StepSoftmaxCrossEntropyBatch(inputs, targets);

	EXPECT_NEAR(aotStep.loss, interpreterStep.loss, 1.0e-5);
	ASSERT_EQ(aotStep.outputs.size(), interpreterStep.outputs.size());
	ASSERT_EQ(aotStep.backwardResults.size(), interpreterStep.backwardResults.size());
	for (std::size_t i = 0; i < aotStep.outputs[0].NumElements(); ++i)
	{
		EXPECT_NEAR(ReadFloat(aotStep.outputs[0], i), ReadFloat(interpreterStep.outputs[0], i), 1.0e-5f);
	}
	for (std::size_t resultIndex = 0; resultIndex < aotStep.backwardResults.size(); ++resultIndex)
	{
		ASSERT_EQ(aotStep.backwardResults[resultIndex].NumElements(),
		          interpreterStep.backwardResults[resultIndex].NumElements());
		for (std::size_t i = 0; i < aotStep.backwardResults[resultIndex].NumElements(); ++i)
		{
			EXPECT_NEAR(ReadFloat(aotStep.backwardResults[resultIndex], i),
			            ReadFloat(interpreterStep.backwardResults[resultIndex], i), 1.0e-5f);
		}
	}
	for (std::size_t i = 0; i < 4; ++i)
	{
		EXPECT_NEAR(ReadVariableGradFloat(aotModel.UnsafeMutableGraph(), aotVariables.weight, i),
		            ReadVariableGradFloat(interpreterModel.UnsafeMutableGraph(), interpreterVariables.weight, i),
		            1.0e-5f);
		EXPECT_NEAR(ReadVariableDataFloat(aotModel.UnsafeMutableGraph(), aotVariables.weight, i),
		            ReadVariableDataFloat(interpreterModel.UnsafeMutableGraph(), interpreterVariables.weight, i),
		            1.0e-5f);
	}
	for (std::size_t i = 0; i < 2; ++i)
	{
		EXPECT_NEAR(ReadVariableGradFloat(aotModel.UnsafeMutableGraph(), aotVariables.bias, i),
		            ReadVariableGradFloat(interpreterModel.UnsafeMutableGraph(), interpreterVariables.bias, i),
		            1.0e-5f);
		EXPECT_NEAR(ReadVariableDataFloat(aotModel.UnsafeMutableGraph(), aotVariables.bias, i),
		            ReadVariableDataFloat(interpreterModel.UnsafeMutableGraph(), interpreterVariables.bias, i),
		            1.0e-5f);
	}
}

TEST(Training, AOTAndInterpreterSoftmaxCrossEntropyTinyLinearChainSGDMatch)
{
	ModelGraph interpreterModel;
	const auto interpreterVariables = BuildTinyLinearChainClassifierModel(interpreterModel);
	ModelGraph aotModel;
	const auto aotVariables = BuildTinyLinearChainClassifierModel(aotModel);

	Training::Trainer<CPU, Optimizer::SGD> interpreterTrainer(interpreterModel, Optimizer::SGD(0.05f));
	Training::TrainerOptions aotOptions;
	aotOptions.executionPolicy = Training::TrainExecutionPolicy::AOT;
	Training::Trainer<CPU, Optimizer::SGD> aotTrainer(aotModel, Optimizer::SGD(0.05f), aotOptions);

	std::vector<Tensor<CPU>> inputs;
	inputs.emplace_back(Tensor<CPU>({ 1.0f, 2.0f, -0.5f, 0.25f }, { 2, 2 }));
	std::vector<std::size_t> targets = { 1, 0 };

	const auto interpreterStep = interpreterTrainer.StepSoftmaxCrossEntropyBatch(inputs, targets);
	const auto aotStep = aotTrainer.StepSoftmaxCrossEntropyBatch(inputs, targets);

	EXPECT_NEAR(aotStep.loss, interpreterStep.loss, 1.0e-5);
	ASSERT_EQ(aotStep.outputs.size(), interpreterStep.outputs.size());
	ASSERT_EQ(aotStep.backwardResults.size(), interpreterStep.backwardResults.size());
	for (std::size_t i = 0; i < aotStep.outputs[0].NumElements(); ++i)
	{
		EXPECT_NEAR(ReadFloat(aotStep.outputs[0], i), ReadFloat(interpreterStep.outputs[0], i), 1.0e-5f);
	}
	for (std::size_t resultIndex = 0; resultIndex < aotStep.backwardResults.size(); ++resultIndex)
	{
		ASSERT_EQ(aotStep.backwardResults[resultIndex].NumElements(),
		          interpreterStep.backwardResults[resultIndex].NumElements());
		for (std::size_t i = 0; i < aotStep.backwardResults[resultIndex].NumElements(); ++i)
		{
			EXPECT_NEAR(ReadFloat(aotStep.backwardResults[resultIndex], i),
			            ReadFloat(interpreterStep.backwardResults[resultIndex], i), 1.0e-5f);
		}
	}
	for (std::size_t variableIndex = 0; variableIndex < aotVariables.size(); ++variableIndex)
	{
		const auto variable = aotVariables[variableIndex];
		const auto referenceVariable = interpreterVariables[variableIndex];
		const auto elementCount = aotModel.UnsafeGraphView().GetVariable(variable)->Data().NumElements();
		for (std::size_t i = 0; i < elementCount; ++i)
		{
			EXPECT_NEAR(ReadVariableGradFloat(aotModel.UnsafeMutableGraph(), variable, i),
			            ReadVariableGradFloat(interpreterModel.UnsafeMutableGraph(), referenceVariable, i), 1.0e-5f);
			EXPECT_NEAR(ReadVariableDataFloat(aotModel.UnsafeMutableGraph(), variable, i),
			            ReadVariableDataFloat(interpreterModel.UnsafeMutableGraph(), referenceVariable, i), 1.0e-5f);
		}
	}
}

TEST(Training, AOTAndInterpreterSoftmaxCrossEntropyTinyReLUMLPSGDMatch)
{
	ModelGraph interpreterModel;
	const auto interpreterVariables = BuildTinyReLUMLPClassifierModel(interpreterModel);
	ModelGraph aotModel;
	const auto aotVariables = BuildTinyReLUMLPClassifierModel(aotModel);

	Training::Trainer<CPU, Optimizer::SGD> interpreterTrainer(interpreterModel, Optimizer::SGD(0.05f));
	Training::TrainerOptions aotOptions;
	aotOptions.executionPolicy = Training::TrainExecutionPolicy::AOT;
	Training::Trainer<CPU, Optimizer::SGD> aotTrainer(aotModel, Optimizer::SGD(0.05f), aotOptions);

	std::vector<Tensor<CPU>> inputs;
	inputs.emplace_back(Tensor<CPU>({ 1.0f, 2.0f, -0.5f, 0.25f }, { 2, 2 }));
	std::vector<std::size_t> targets = { 1, 0 };

	const auto interpreterStep = interpreterTrainer.StepSoftmaxCrossEntropyBatch(inputs, targets);
	const auto aotStep = aotTrainer.StepSoftmaxCrossEntropyBatch(inputs, targets);

	EXPECT_NEAR(aotStep.loss, interpreterStep.loss, 1.0e-5);
	ASSERT_EQ(aotStep.outputs.size(), interpreterStep.outputs.size());
	ASSERT_EQ(aotStep.backwardResults.size(), interpreterStep.backwardResults.size());
	for (std::size_t i = 0; i < aotStep.outputs[0].NumElements(); ++i)
	{
		EXPECT_NEAR(ReadFloat(aotStep.outputs[0], i), ReadFloat(interpreterStep.outputs[0], i), 1.0e-5f);
	}
	for (std::size_t resultIndex = 0; resultIndex < aotStep.backwardResults.size(); ++resultIndex)
	{
		ASSERT_EQ(aotStep.backwardResults[resultIndex].NumElements(),
		          interpreterStep.backwardResults[resultIndex].NumElements());
		for (std::size_t i = 0; i < aotStep.backwardResults[resultIndex].NumElements(); ++i)
		{
			EXPECT_NEAR(ReadFloat(aotStep.backwardResults[resultIndex], i),
			            ReadFloat(interpreterStep.backwardResults[resultIndex], i), 1.0e-5f);
		}
	}
	for (std::size_t variableIndex = 0; variableIndex < aotVariables.size(); ++variableIndex)
	{
		const auto variable = aotVariables[variableIndex];
		const auto referenceVariable = interpreterVariables[variableIndex];
		const auto elementCount = aotModel.UnsafeGraphView().GetVariable(variable)->Data().NumElements();
		for (std::size_t i = 0; i < elementCount; ++i)
		{
			EXPECT_NEAR(ReadVariableGradFloat(aotModel.UnsafeMutableGraph(), variable, i),
			            ReadVariableGradFloat(interpreterModel.UnsafeMutableGraph(), referenceVariable, i), 1.0e-5f);
			EXPECT_NEAR(ReadVariableDataFloat(aotModel.UnsafeMutableGraph(), variable, i),
			            ReadVariableDataFloat(interpreterModel.UnsafeMutableGraph(), referenceVariable, i), 1.0e-5f);
		}
	}
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
