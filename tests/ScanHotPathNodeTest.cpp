#include <gtest/gtest.h>

#include <LiteNN.h>
#include <LiteNN/Debug/Dump.h>
#include <LiteNN/Layer/BatchMatMul.h>
#include <LiteNN/Layer/Normalization.h>
#include <LiteNN/Layer/RWKVWKV.h>
#include <LiteNN/Layer/SSMScan.h>
#include <LiteNN/Layer/Scan.h>
#include <LiteNN/Layer/Softmax.h>
#include <LiteNN/Pass/ConstFoldPass.h>
#include <LiteNN/Runtime/Interpreter.h>
#include <LiteNN/Serialization/ModelIO.h>
#include <LiteNN/Validation/GraphValidator.h>

#include <cmath>
#include <filesystem>
#include <ranges>
#include <span>
#include <vector>

using namespace LiteNN;

namespace
{
	float ReadFloat(const Tensor<CPU>& tensor, std::size_t index)
	{
		return static_cast<const float*>(tensor.RawData())[index];
	}

	void ExpectShape(ShapeView shape, std::initializer_list<std::size_t> expected)
	{
		EXPECT_TRUE(std::ranges::equal(shape.Dims, expected));
	}

	void ExpectTensorNear(const Tensor<CPU>& tensor, std::initializer_list<float> expected, float tolerance = 1e-5F)
	{
		ASSERT_EQ(tensor.NumElements(), expected.size());
		std::size_t index = 0;
		for (const auto value : expected)
		{
			EXPECT_NEAR(ReadFloat(tensor, index), value, tolerance) << "index=" << index;
			++index;
		}
	}

	std::vector<Tensor<CPU>> RunGraph(Graph& graph, std::vector<Tensor<CPU>> inputs)
	{
		Runtime::Interpreter<CPU> interpreter;
		return interpreter.RunForward(graph, inputs);
	}

	NodeOutput AddFloatConstant(Subgraph& subgraph, std::initializer_list<double> values,
	                            std::initializer_list<std::size_t> shape)
	{
		auto tensor = Tensor<CPU>(values, shape);
		auto shapeVector = std::vector<std::size_t>(shape);
		const auto node = subgraph.AddNode(ConstantNode{ tensor.CopyToDevice(PolymorphicDevice{ CPU{} }) },
		                                   { OutputInfo{ DataType::Float32, std::move(shapeVector) } });
		return NodeOutput{ node, 0uz };
	}

	Graph BuildAllNewNodeGraph()
	{
		Graph graph;
		Subgraph subgraph;

		const auto x = subgraph.AddParam(DataType::Float32, { 2, 3 });
		const auto scale = subgraph.AddParam(DataType::Float32, { 1, 3 });
		const auto bias = subgraph.AddParam(DataType::Float32, { 1, 3 });
		const auto lhs = subgraph.AddParam(DataType::Float32, { 2, 2, 3 });
		const auto rhs = subgraph.AddParam(DataType::Float32, { 1, 3, 2 });
		const auto state = subgraph.AddParam(DataType::Float32, { 2, 2 });
		const auto dt = subgraph.AddParam(DataType::Float32, { 1 });
		const auto a = subgraph.AddParam(DataType::Float32, { 1 });
		const auto b = subgraph.AddParam(DataType::Float32, { 1 });
		const auto c = subgraph.AddParam(DataType::Float32, { 1 });
		const auto d = subgraph.AddParam(DataType::Float32, { 1 });
		const auto key = subgraph.AddParam(DataType::Float32, { 2, 2 });
		const auto value = subgraph.AddParam(DataType::Float32, { 2, 2 });
		const auto receptance = subgraph.AddParam(DataType::Float32, { 2, 2 });
		const auto timeDecay = subgraph.AddParam(DataType::Float32, { 1 });
		const auto timeFirst = subgraph.AddParam(DataType::Float32, { 1 });

		const auto scan = Layer::AddScan(subgraph, { x, 0 }, 1, ScanOp::Sum);
		const auto softmax = Layer::AddSoftmax(subgraph, { x, 0 }, 1);
		const auto norm = Layer::AddNormalization(subgraph, { x, 0 }, NormalizationMode::LayerNorm, 1, 1e-5,
		                                          NodeOutput{ scale, 0 }, NodeOutput{ bias, 0 });
		const auto batchMatMul = Layer::AddBatchMatMul(subgraph, { lhs, 0 }, { rhs, 0 });
		const auto ssm = Layer::AddSSMScan(subgraph, { state, 0 }, { dt, 0 }, { a, 0 }, { b, 0 }, { c, 0 },
		                                   NodeOutput{ d, 0 });
		const auto rwkv = Layer::AddRWKVWKV(subgraph, { key, 0 }, { value, 0 }, { receptance, 0 },
		                                    { timeDecay, 0 }, { timeFirst, 0 });

		subgraph.SetResults({ scan, softmax, norm, batchMatMul, ssm, rwkv });
		graph.SetForward(graph.AddSubgraph(std::move(subgraph)));
		return graph;
	}

	std::vector<Tensor<CPU>> MakeAllNewNodeInputs()
	{
		std::vector<Tensor<CPU>> inputs;
		inputs.emplace_back(Tensor<CPU>({ 1.0F, 3.0F, 2.0F, 4.0F, 0.0F, 5.0F }, { 2, 3 }));
		inputs.emplace_back(Tensor<CPU>({ 1.0F, 2.0F, 1.0F }, { 1, 3 }));
		inputs.emplace_back(Tensor<CPU>({ 0.0F, 0.5F, -1.0F }, { 1, 3 }));
		inputs.emplace_back(Tensor<CPU>({ 1.0F, 2.0F, 3.0F,
		                                  4.0F, 5.0F, 6.0F,
		                                  7.0F, 8.0F, 9.0F,
		                                  10.0F, 11.0F, 12.0F },
		                                { 2, 2, 3 }));
		inputs.emplace_back(Tensor<CPU>({ 1.0F, 0.0F,
		                                  0.0F, 1.0F,
		                                  1.0F, 1.0F },
		                                { 1, 3, 2 }));
		inputs.emplace_back(Tensor<CPU>({ 1.0F, 2.0F, 3.0F, 4.0F }, { 2, 2 }));
		inputs.emplace_back(Tensor<CPU>({ 1.0F }, { 1 }));
		inputs.emplace_back(Tensor<CPU>({ 0.0F }, { 1 }));
		inputs.emplace_back(Tensor<CPU>({ 1.0F }, { 1 }));
		inputs.emplace_back(Tensor<CPU>({ 1.0F }, { 1 }));
		inputs.emplace_back(Tensor<CPU>({ 0.5F }, { 1 }));
		inputs.emplace_back(Tensor<CPU>({ 0.0F, 0.0F, 0.0F, 0.0F }, { 2, 2 }));
		inputs.emplace_back(Tensor<CPU>({ 2.0F, 4.0F, 6.0F, 8.0F }, { 2, 2 }));
		inputs.emplace_back(Tensor<CPU>({ 1.0F, 0.5F, 1.0F, 0.25F }, { 2, 2 }));
		inputs.emplace_back(Tensor<CPU>({ 0.0F }, { 1 }));
		inputs.emplace_back(Tensor<CPU>({ 0.0F }, { 1 }));
		return inputs;
	}
} // namespace

TEST(ScanHotPathNode, ExecutesScanSoftmaxNormalizationAndBatchMatMul)
{
	Graph graph;
	Subgraph subgraph;
	const auto x = subgraph.AddParam(DataType::Float32, { 2, 3 });
	const auto scale = subgraph.AddParam(DataType::Float32, { 1, 3 });
	const auto bias = subgraph.AddParam(DataType::Float32, { 1, 3 });
	const auto lhs = subgraph.AddParam(DataType::Float32, { 2, 2, 3 });
	const auto rhs = subgraph.AddParam(DataType::Float32, { 1, 3, 2 });

	const auto scanSum = Layer::AddScan(subgraph, { x, 0 }, 1, ScanOp::Sum);
	const auto scanMax = Layer::AddScan(subgraph, { x, 0 }, 0, ScanOp::Max);
	const auto softmax = Layer::AddSoftmax(subgraph, { x, 0 }, 1);
	const auto norm = Layer::AddNormalization(subgraph, { x, 0 }, NormalizationMode::LayerNorm, 1, 1e-5,
	                                          NodeOutput{ scale, 0 }, NodeOutput{ bias, 0 });
	const auto batchMatMul = Layer::AddBatchMatMul(subgraph, { lhs, 0 }, { rhs, 0 });

	subgraph.SetResults({ scanSum, scanMax, softmax, norm, batchMatMul });
	graph.SetForward(graph.AddSubgraph(std::move(subgraph)));

	std::vector<Tensor<CPU>> inputs;
	inputs.emplace_back(Tensor<CPU>({ 1.0F, 3.0F, 2.0F, 4.0F, 0.0F, 5.0F }, { 2, 3 }));
	inputs.emplace_back(Tensor<CPU>({ 1.0F, 2.0F, 1.0F }, { 1, 3 }));
	inputs.emplace_back(Tensor<CPU>({ 0.0F, 0.5F, -1.0F }, { 1, 3 }));
	inputs.emplace_back(Tensor<CPU>({ 1.0F, 2.0F, 3.0F,
	                                  4.0F, 5.0F, 6.0F,
	                                  7.0F, 8.0F, 9.0F,
	                                  10.0F, 11.0F, 12.0F },
	                                { 2, 2, 3 }));
	inputs.emplace_back(Tensor<CPU>({ 1.0F, 0.0F,
	                                  0.0F, 1.0F,
	                                  1.0F, 1.0F },
	                                { 1, 3, 2 }));

	const auto outputs = RunGraph(graph, std::move(inputs));
	ASSERT_EQ(outputs.size(), 5u);

	ExpectTensorNear(outputs[0], { 1.0F, 4.0F, 6.0F, 4.0F, 4.0F, 9.0F });
	ExpectTensorNear(outputs[1], { 1.0F, 3.0F, 2.0F, 4.0F, 3.0F, 5.0F });

	const auto row0Denom = std::exp(-2.0F) + 1.0F + std::exp(-1.0F);
	const auto row1Denom = std::exp(-1.0F) + std::exp(-5.0F) + 1.0F;
	ExpectTensorNear(outputs[2],
	                 { std::exp(-2.0F) / row0Denom, 1.0F / row0Denom, std::exp(-1.0F) / row0Denom,
	                   std::exp(-1.0F) / row1Denom, std::exp(-5.0F) / row1Denom, 1.0F / row1Denom },
	                 1e-5F);

	const auto row0NormDenom = std::sqrt(2.0F / 3.0F + 1e-5F);
	const auto row1NormDenom = std::sqrt(14.0F / 3.0F + 1e-5F);
	ExpectTensorNear(outputs[3],
	                 { -1.0F / row0NormDenom, 0.5F + 2.0F / row0NormDenom, -1.0F,
	                   1.0F / row1NormDenom, 0.5F - 6.0F / row1NormDenom,
	                   -1.0F + 2.0F / row1NormDenom },
	                 1e-5F);

	ExpectShape(outputs[4].Shape(), { 2, 2, 2 });
	ExpectTensorNear(outputs[4], { 4.0F, 5.0F, 10.0F, 11.0F, 16.0F, 17.0F, 22.0F, 23.0F });
}

TEST(ScanHotPathNode, ExecutesRecurrenceReferenceKernels)
{
	Graph graph;
	Subgraph subgraph;
	const auto state = subgraph.AddParam(DataType::Float32, { 2, 2 });
	const auto dt = subgraph.AddParam(DataType::Float32, { 1 });
	const auto a = subgraph.AddParam(DataType::Float32, { 1 });
	const auto b = subgraph.AddParam(DataType::Float32, { 1 });
	const auto c = subgraph.AddParam(DataType::Float32, { 1 });
	const auto d = subgraph.AddParam(DataType::Float32, { 1 });
	const auto key = subgraph.AddParam(DataType::Float32, { 2, 2 });
	const auto value = subgraph.AddParam(DataType::Float32, { 2, 2 });
	const auto receptance = subgraph.AddParam(DataType::Float32, { 2, 2 });
	const auto timeDecay = subgraph.AddParam(DataType::Float32, { 1 });
	const auto timeFirst = subgraph.AddParam(DataType::Float32, { 1 });

	const auto ssm = Layer::AddSSMScan(subgraph, { state, 0 }, { dt, 0 }, { a, 0 }, { b, 0 }, { c, 0 },
	                                   NodeOutput{ d, 0 });
	const auto rwkv = Layer::AddRWKVWKV(subgraph, { key, 0 }, { value, 0 }, { receptance, 0 },
	                                    { timeDecay, 0 }, { timeFirst, 0 });
	subgraph.SetResults({ ssm, rwkv });
	graph.SetForward(graph.AddSubgraph(std::move(subgraph)));

	std::vector<Tensor<CPU>> inputs;
	inputs.emplace_back(Tensor<CPU>({ 1.0F, 2.0F, 3.0F, 4.0F }, { 2, 2 }));
	inputs.emplace_back(Tensor<CPU>({ 1.0F }, { 1 }));
	inputs.emplace_back(Tensor<CPU>({ 0.0F }, { 1 }));
	inputs.emplace_back(Tensor<CPU>({ 1.0F }, { 1 }));
	inputs.emplace_back(Tensor<CPU>({ 1.0F }, { 1 }));
	inputs.emplace_back(Tensor<CPU>({ 0.5F }, { 1 }));
	inputs.emplace_back(Tensor<CPU>({ 0.0F, 0.0F, 0.0F, 0.0F }, { 2, 2 }));
	inputs.emplace_back(Tensor<CPU>({ 2.0F, 4.0F, 6.0F, 8.0F }, { 2, 2 }));
	inputs.emplace_back(Tensor<CPU>({ 1.0F, 0.5F, 1.0F, 0.25F }, { 2, 2 }));
	inputs.emplace_back(Tensor<CPU>({ 0.0F }, { 1 }));
	inputs.emplace_back(Tensor<CPU>({ 0.0F }, { 1 }));

	const auto outputs = RunGraph(graph, std::move(inputs));
	ASSERT_EQ(outputs.size(), 2u);
	ExpectTensorNear(outputs[0], { 1.5F, 3.0F, 5.5F, 8.0F });
	ExpectTensorNear(outputs[1], { 2.0F, 2.0F, 4.0F, 1.5F });
}

TEST(ScanHotPathNode, ConstFoldHandlesG52G53Nodes)
{
	Graph graph;
	Subgraph subgraph;
	const auto x = AddFloatConstant(subgraph, { 1.0F, 3.0F, 2.0F, 4.0F, 0.0F, 5.0F }, { 2, 3 });
	const auto scale = AddFloatConstant(subgraph, { 1.0F, 2.0F, 1.0F }, { 1, 3 });
	const auto bias = AddFloatConstant(subgraph, { 0.0F, 0.5F, -1.0F }, { 1, 3 });
	const auto lhs = AddFloatConstant(subgraph,
	                                  { 1.0F, 2.0F, 3.0F,
	                                    4.0F, 5.0F, 6.0F,
	                                    7.0F, 8.0F, 9.0F,
	                                    10.0F, 11.0F, 12.0F },
	                                  { 2, 2, 3 });
	const auto rhs = AddFloatConstant(subgraph,
	                                  { 1.0F, 0.0F,
	                                    0.0F, 1.0F,
	                                    1.0F, 1.0F },
	                                  { 1, 3, 2 });
	const auto state = AddFloatConstant(subgraph, { 1.0F, 2.0F, 3.0F, 4.0F }, { 2, 2 });
	const auto dt = AddFloatConstant(subgraph, { 1.0F }, { 1 });
	const auto a = AddFloatConstant(subgraph, { 0.0F }, { 1 });
	const auto b = AddFloatConstant(subgraph, { 1.0F }, { 1 });
	const auto c = AddFloatConstant(subgraph, { 1.0F }, { 1 });
	const auto d = AddFloatConstant(subgraph, { 0.5F }, { 1 });
	const auto key = AddFloatConstant(subgraph, { 0.0F, 0.0F, 0.0F, 0.0F }, { 2, 2 });
	const auto value = AddFloatConstant(subgraph, { 2.0F, 4.0F, 6.0F, 8.0F }, { 2, 2 });
	const auto receptance = AddFloatConstant(subgraph, { 1.0F, 0.5F, 1.0F, 0.25F }, { 2, 2 });
	const auto timeDecay = AddFloatConstant(subgraph, { 0.0F }, { 1 });
	const auto timeFirst = AddFloatConstant(subgraph, { 0.0F }, { 1 });

	const auto scan = Layer::AddScan(subgraph, x, 1, ScanOp::Sum);
	const auto softmax = Layer::AddSoftmax(subgraph, x, 1);
	const auto norm =
	    Layer::AddNormalization(subgraph, x, NormalizationMode::LayerNorm, 1, 1e-5, scale, bias);
	const auto batchMatMul = Layer::AddBatchMatMul(subgraph, lhs, rhs);
	const auto ssm = Layer::AddSSMScan(subgraph, state, dt, a, b, c, d);
	const auto rwkv = Layer::AddRWKVWKV(subgraph, key, value, receptance, timeDecay, timeFirst);
	subgraph.SetResults({ scan, softmax, norm, batchMatMul, ssm, rwkv });
	graph.SetForward(graph.AddSubgraph(std::move(subgraph)));

	ConstFoldPass pass;
	pass.Run(graph);
	Validation::ValidateGraph(graph);

	const auto& folded = graph.GetSubgraph(graph.Forward());
	for (const auto result : folded.Results())
	{
		EXPECT_TRUE(std::holds_alternative<ConstantNode>(folded.GetNodeEntry(result.node).node));
	}

	const auto outputs = RunGraph(graph, {});
	ASSERT_EQ(outputs.size(), 6u);
	ExpectTensorNear(outputs[0], { 1.0F, 4.0F, 6.0F, 4.0F, 4.0F, 9.0F });
	ExpectTensorNear(outputs[3], { 4.0F, 5.0F, 10.0F, 11.0F, 16.0F, 17.0F, 22.0F, 23.0F });
	ExpectTensorNear(outputs[4], { 1.5F, 3.0F, 5.5F, 8.0F });
	ExpectTensorNear(outputs[5], { 2.0F, 2.0F, 4.0F, 1.5F });
}

TEST(ScanHotPathNode, SerializationRoundTripPreservesG52G53Nodes)
{
	auto graph = BuildAllNewNodeGraph();
	Validation::ValidateGraph(graph);

	const auto path = std::filesystem::path("litenn_g52_g53_nodes_roundtrip_test.ltnn");
	std::filesystem::remove(path);
	Serialization::SaveModel(graph, path);
	auto loaded = Serialization::LoadModel(path);
	std::filesystem::remove(path);

	auto expected = RunGraph(graph, MakeAllNewNodeInputs());
	auto actual = RunGraph(loaded, MakeAllNewNodeInputs());
	ASSERT_EQ(actual.size(), expected.size());
	for (std::size_t output = 0; output < actual.size(); ++output)
	{
		ASSERT_EQ(actual[output].NumElements(), expected[output].NumElements());
		for (std::size_t index = 0; index < actual[output].NumElements(); ++index)
		{
			EXPECT_NEAR(ReadFloat(actual[output], index), ReadFloat(expected[output], index), 1e-5F)
			    << "output=" << output << ", index=" << index;
		}
	}
}

TEST(ScanHotPathNode, DumpIncludesG52G53NodeKinds)
{
	auto graph = BuildAllNewNodeGraph();
	const auto dump = Debug::DumpGraph(graph);
	EXPECT_NE(dump.find("ScanNode"), std::string::npos);
	EXPECT_NE(dump.find("SoftmaxNode"), std::string::npos);
	EXPECT_NE(dump.find("NormalizationNode"), std::string::npos);
	EXPECT_NE(dump.find("BatchMatMulNode"), std::string::npos);
	EXPECT_NE(dump.find("SSMScanNode"), std::string::npos);
	EXPECT_NE(dump.find("RWKVWKVNode"), std::string::npos);
}
