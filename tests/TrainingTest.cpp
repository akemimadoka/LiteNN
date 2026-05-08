#include <gtest/gtest.h>

#include <LiteNN.h>

#include <cmath>
#include <vector>

using namespace LiteNN;

namespace
{
	float ReadFloat(const Tensor<CPU>& tensor, std::size_t index)
	{
		const auto cpuTensor = tensor.CopyToDevice(CPU{});
		return static_cast<const float*>(cpuTensor.RawData())[index];
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
	Graph graph;
	const auto weightIndex = graph.AddVariable(Variable::Create(Tensor<CPU>({ 3.0f }, { 1 })));

	Subgraph sg;
	const auto x = sg.AddParam(DataType::Float32, { 1 });
	const auto weight = sg.AddNode(VariableRefNode{ weightIndex }, { OutputInfo{ DataType::Float32, { 1 } } });
	const auto y = sg.AddNode(BinaryOpNode{ BinaryOp::Multiply, { x, 0 }, { weight, 0 } },
	                          { OutputInfo{ DataType::Float32, { 1 } } });
	sg.SetResults({ { y, 0 } });
	graph.SetForward(graph.AddSubgraph(std::move(sg)));

	Training::CPUTrainer<Optimizer::SGD> trainer(graph, Optimizer::SGD(0.1f));
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

TEST(Training, StepSoftmaxCrossEntropyComputesLossAndUpdatesVariables)
{
	Graph graph;
	const auto logitsIndex = graph.AddVariable(Variable::Create(Tensor<CPU>({ 0.0f, 0.0f }, { 2 })));

	Subgraph sg;
	const auto logits = sg.AddNode(VariableRefNode{ logitsIndex }, { OutputInfo{ DataType::Float32, { 2 } } });
	sg.SetResults({ { logits, 0 } });
	graph.SetForward(graph.AddSubgraph(std::move(sg)));

	Training::CPUTrainer<Optimizer::SGD> trainer(graph, Optimizer::SGD(1.0f));
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
	Graph graph;
	const auto logitsIndex =
	    graph.AddVariable(Variable::Create(Tensor<CPU>({ 0.0f, 0.0f, 0.0f, 0.0f }, { 2, 2 })));

	Subgraph sg;
	const auto logits = sg.AddNode(VariableRefNode{ logitsIndex }, { OutputInfo{ DataType::Float32, { 2, 2 } } });
	sg.SetResults({ { logits, 0 } });
	graph.SetForward(graph.AddSubgraph(std::move(sg)));

	Training::CPUTrainer<Optimizer::SGD> trainer(graph, Optimizer::SGD(1.0f));
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
