#include <gtest/gtest.h>

#include <LiteNN/Debug/Dump.h>
#include <LiteNN/Layer/Loss.h>
#include <LiteNN/Pass/AutogradPass.h>
#include <LiteNN/Pass/ConstFoldPass.h>
#include <LiteNN/Runtime/Interpreter.h>
#include <LiteNN/Serialization/ModelIO.h>

#include <algorithm>
#include <chrono>
#include <cmath>
#include <filesystem>
#include <format>
#include <span>
#include <vector>

using namespace LiteNN;

namespace
{
	float ReadFloat(const Tensor<CPU>& tensor, std::size_t index)
	{
		return static_cast<const float*>(tensor.RawData())[index];
	}

	std::filesystem::path MakeTempPath(std::string_view stem)
	{
		const auto now = std::chrono::steady_clock::now().time_since_epoch().count();
		return std::filesystem::temp_directory_path() / std::format("{}_{}.ltnn", stem, now);
	}

	Tensor<CPU> MakeFloatTensor(std::span<const double> values, std::initializer_list<std::size_t> shape)
	{
		return Tensor<CPU>(values, ShapeView{ shape }, DataType::Float32);
	}

	Tensor<CPU> MakeFloatTensor(const std::vector<double>& values, std::initializer_list<std::size_t> shape)
	{
		return MakeFloatTensor(std::span<const double>{ values.data(), values.size() }, shape);
	}

	Tensor<CPU> MakeFloatTensor(std::initializer_list<double> values, std::initializer_list<std::size_t> shape)
	{
		return Tensor<CPU>(values, ShapeView{ shape }, DataType::Float32);
	}

	ConstantNode MakeFloatConstant(std::span<const double> values, std::initializer_list<std::size_t> shape)
	{
		auto tensor = MakeFloatTensor(values, shape);
		return ConstantNode{ tensor.CopyToDevice(PolymorphicDevice{ CPU{} }) };
	}

	ConstantNode MakeFloatConstant(std::initializer_list<double> values, std::initializer_list<std::size_t> shape)
	{
		auto tensor = MakeFloatTensor(values, shape);
		return ConstantNode{ tensor.CopyToDevice(PolymorphicDevice{ CPU{} }) };
	}

	double ExpectedCrossEntropy(std::span<const double> logits, std::span<const double> labels,
	                            std::size_t classCount)
	{
		const auto rows = logits.size() / classCount;
		double total = 0.0;
		for (auto row = 0uz; row < rows; ++row)
		{
			const auto offset = row * classCount;
			const auto* logitsRow = logits.data() + offset;
			const auto* labelsRow = labels.data() + offset;
			const auto maxLogit = *std::max_element(logitsRow, logitsRow + classCount);
			double sumExp = 0.0;
			for (auto col = 0uz; col < classCount; ++col)
			{
				sumExp += std::exp(static_cast<double>(logitsRow[col] - maxLogit));
			}
			const auto logSumExp = std::log(sumExp) + static_cast<double>(maxLogit);
			for (auto col = 0uz; col < classCount; ++col)
			{
				total -= static_cast<double>(labelsRow[col]) *
				         (static_cast<double>(logitsRow[col]) - logSumExp);
			}
		}
		return total / static_cast<double>(rows);
	}

	Graph BuildLossGraph()
	{
		Graph graph;
		Subgraph subgraph;
		const auto logits = subgraph.AddParam(DataType::Float32, { 2, 3 });
		const auto labels = subgraph.AddParam(DataType::Float32, { 2, 3 });
		const auto loss = Layer::AddCrossEntropyLoss(subgraph, { logits, 0 }, { labels, 0 });
		subgraph.SetResults({ loss });
		graph.SetForward(graph.AddSubgraph(std::move(subgraph)));
		return graph;
	}
}

TEST(LossNode, CrossEntropyLossMatchesGGMLStyleSoftLabels)
{
	auto graph = BuildLossGraph();
	Runtime::Interpreter<CPU> interpreter;
	const std::vector<double> logits = { 1.0, 2.0, 3.0, 1.0, 0.0, -1.0 };
	const std::vector<double> labels = { 0.0, 0.0, 1.0, 0.25, 0.75, 0.0 };
	std::vector<Tensor<CPU>> inputs;
	inputs.push_back(MakeFloatTensor(logits, { 2, 3 }));
	inputs.push_back(MakeFloatTensor(labels, { 2, 3 }));
	const auto outputs = interpreter.RunForward(BuildExecutablePlan(graph), inputs);

	ASSERT_EQ(outputs.size(), 1u);
	EXPECT_EQ(outputs[0].Shape().ToOwned(), std::vector<std::size_t>({ 1 }));
	EXPECT_NEAR(ReadFloat(outputs[0], 0), ExpectedCrossEntropy(logits, labels, 3), 1e-6);
}

TEST(LossNode, CrossEntropyBackwardMatchesSoftmaxMinusLabels)
{
	Graph graph;
	Subgraph subgraph;
	const auto grad = subgraph.AddParam(DataType::Float32, { 1 });
	const auto logits = subgraph.AddParam(DataType::Float32, { 2, 3 });
	const auto labels = subgraph.AddParam(DataType::Float32, { 2, 3 });
	const auto dx = Layer::AddCrossEntropyLossBackward(subgraph, { grad, 0 }, { logits, 0 }, { labels, 0 });
	subgraph.SetResults({ dx });
	graph.SetForward(graph.AddSubgraph(std::move(subgraph)));

	Runtime::Interpreter<CPU> interpreter;
	const std::vector<double> logitsData = { 1.0, 2.0, 3.0, 1.0, 0.0, -1.0 };
	const std::vector<double> labelsData = { 0.0, 0.0, 1.0, 0.25, 0.75, 0.0 };
	std::vector<Tensor<CPU>> inputs;
	inputs.push_back(MakeFloatTensor({ 2.0 }, { 1 }));
	inputs.push_back(MakeFloatTensor(logitsData, { 2, 3 }));
	inputs.push_back(MakeFloatTensor(labelsData, { 2, 3 }));
	const auto outputs = interpreter.RunForward(BuildExecutablePlan(graph), inputs);

	const std::vector<float> expected = {
	    0.0900306f, 0.244728f, -0.334759f,
	    0.415241f, -0.505272f, 0.0900306f,
	};
	ASSERT_EQ(outputs[0].NumElements(), expected.size());
	for (auto i = 0uz; i < expected.size(); ++i)
	{
		EXPECT_NEAR(ReadFloat(outputs[0], i), expected[i], 1e-5f) << i;
	}
}

TEST(LossNode, AutogradUsesCrossEntropyBackwardForLogits)
{
	auto graph = BuildLossGraph();
	AutogradPass{}.Run(graph);

	Runtime::Interpreter<CPU> interpreter;
	const std::vector<double> logits = { 1.0, 2.0, 3.0, 1.0, 0.0, -1.0 };
	const std::vector<double> labels = { 0.0, 0.0, 1.0, 0.25, 0.75, 0.0 };
	std::vector<Tensor<CPU>> forwardInputs;
	forwardInputs.push_back(MakeFloatTensor(logits, { 2, 3 }));
	forwardInputs.push_back(MakeFloatTensor(labels, { 2, 3 }));
	static_cast<void>(interpreter.RunForward(BuildExecutablePlan(graph), forwardInputs));
	std::vector<Tensor<CPU>> backwardInputs;
	backwardInputs.push_back(MakeFloatTensor(logits, { 2, 3 }));
	backwardInputs.push_back(MakeFloatTensor(labels, { 2, 3 }));
	backwardInputs.push_back(MakeFloatTensor({ 1.0 }, { 1 }));
	const auto gradients = interpreter.RunBackward(BuildExecutablePlan(graph), backwardInputs);

	ASSERT_GE(gradients.size(), 2u);
	EXPECT_NEAR(ReadFloat(gradients[0], 2), -0.167379f, 1e-5f);
	for (auto i = 0uz; i < gradients[1].NumElements(); ++i)
	{
		EXPECT_FLOAT_EQ(ReadFloat(gradients[1], i), 0.0f);
	}
}

TEST(LossNode, ConstFoldSerializationAndDumpKeepLossNodes)
{
	auto graph = BuildLossGraph();
	const auto path = MakeTempPath("litenn_loss_node");
	Serialization::SaveModel(graph, path);
	auto loaded = Serialization::LoadModel(path);
	std::filesystem::remove(path);

	const auto dump = Debug::DumpGraph(loaded);
	EXPECT_NE(dump.find("CrossEntropyLossNode"), std::string::npos);

	Graph constGraph;
	Subgraph subgraph;
	const auto logits = subgraph.AddNode(MakeFloatConstant({ 1.0, 2.0, 3.0 }, { 1, 3 }),
	                                     { OutputInfo{ DataType::Float32, { 1, 3 } } });
	const auto labels = subgraph.AddNode(MakeFloatConstant({ 0.0, 0.0, 1.0 }, { 1, 3 }),
	                                     { OutputInfo{ DataType::Float32, { 1, 3 } } });
	const auto loss = Layer::AddCrossEntropyLoss(subgraph, { logits, 0 }, { labels, 0 });
	subgraph.SetResults({ loss });
	constGraph.SetForward(constGraph.AddSubgraph(std::move(subgraph)));
	ConstFoldPass{}.Run(constGraph);

	Runtime::Interpreter<CPU> interpreter;
	const auto outputs = interpreter.RunForward(BuildExecutablePlan(constGraph), {});
	const std::vector<double> constLogits = { 1.0, 2.0, 3.0 };
	const std::vector<double> constLabels = { 0.0, 0.0, 1.0 };
	EXPECT_NEAR(ReadFloat(outputs[0], 0), ExpectedCrossEntropy(constLogits, constLabels, 3), 1e-6);
}
