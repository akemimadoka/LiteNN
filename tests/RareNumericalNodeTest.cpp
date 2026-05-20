#include <gtest/gtest.h>

#include <LiteNN.h>
#include <LiteNN/Debug/Dump.h>
#include <LiteNN/Layer/OutProd.h>
#include <LiteNN/Layer/SolveTri.h>
#include <LiteNN/Layer/TimestepEmbedding.h>
#include <LiteNN/Pass/ConstFoldPass.h>
#include <LiteNN/Runtime/Interpreter.h>
#include <LiteNN/Serialization/ModelIO.h>
#include <LiteNN/Validation/GraphValidator.h>

#include <cmath>
#include <filesystem>
#include <initializer_list>
#include <ranges>
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

	Graph BuildRareNumericalGraph()
	{
		Graph graph;
		Subgraph subgraph;

		const auto lhs = subgraph.AddParam(DataType::Float32, { 1, 2, 3 });
		const auto rhs = subgraph.AddParam(DataType::Float32, { 2, 2, 3 });
		const auto timesteps = subgraph.AddParam(DataType::Float32, { 2 });
		const auto a = subgraph.AddParam(DataType::Float32, { 2, 2 });
		const auto b = subgraph.AddParam(DataType::Float32, { 2, 2 });

		const auto outProd = Layer::AddOutProd(subgraph, { lhs, 0 }, { rhs, 0 });
		const auto embedding = Layer::AddTimestepEmbedding(subgraph, { timesteps, 0 }, 5, 1000);
		const auto solve = Layer::AddSolveTri(subgraph, { a, 0 }, { b, 0 });
		subgraph.SetResults({ outProd, embedding, solve });
		graph.SetForward(graph.AddSubgraph(std::move(subgraph)));
		return graph;
	}

	std::vector<Tensor<CPU>> MakeRareNumericalInputs()
	{
		std::vector<Tensor<CPU>> inputs;
		inputs.emplace_back(Tensor<CPU>({ 1.0F, 2.0F, 3.0F,
		                                  4.0F, 5.0F, 6.0F },
		                                { 1, 2, 3 }));
		inputs.emplace_back(Tensor<CPU>({ 1.0F, 0.0F, 1.0F,
		                                  0.0F, 1.0F, 1.0F,
		                                  2.0F, 1.0F, 0.0F,
		                                  1.0F, 1.0F, 1.0F },
		                                { 2, 2, 3 }));
		inputs.emplace_back(Tensor<CPU>({ 0.0F, 1.0F }, { 2 }));
		inputs.emplace_back(Tensor<CPU>({ 2.0F, 0.0F,
		                                  3.0F, 1.0F },
		                                { 2, 2 }));
		inputs.emplace_back(Tensor<CPU>({ 4.0F, 2.0F,
		                                  11.0F, 7.0F },
		                                { 2, 2 }));
		return inputs;
	}
} // namespace

TEST(RareNumericalNode, ExecutesOutProdTimestepEmbeddingAndSolveTri)
{
	auto graph = BuildRareNumericalGraph();
	Validation::ValidateGraph(graph);

	const auto outputs = RunGraph(graph, MakeRareNumericalInputs());
	ASSERT_EQ(outputs.size(), 3u);

	ExpectShape(outputs[0].Shape(), { 2, 2, 2 });
	ExpectTensorNear(outputs[0], { 4.0F, 5.0F,
	                               10.0F, 11.0F,
	                               4.0F, 6.0F,
	                               13.0F, 15.0F });

	const auto freq = std::exp(-std::log(1000.0F) / 2.0F);
	ExpectShape(outputs[1].Shape(), { 2, 5 });
	ExpectTensorNear(outputs[1], { 1.0F, 1.0F, 0.0F, 0.0F, 0.0F,
	                               std::cos(1.0F), std::cos(freq),
	                               std::sin(1.0F), std::sin(freq), 0.0F });

	ExpectShape(outputs[2].Shape(), { 2, 2 });
	ExpectTensorNear(outputs[2], { 2.0F, 1.0F,
	                               5.0F, 4.0F });
}

TEST(RareNumericalNode, ConstFoldFoldsRareNumericalNodes)
{
	Graph graph;
	Subgraph subgraph;
	const auto lhs = AddFloatConstant(subgraph, { 1.0F, 2.0F, 3.0F,
	                                              4.0F, 5.0F, 6.0F },
	                                  { 1, 2, 3 });
	const auto rhs = AddFloatConstant(subgraph, { 1.0F, 0.0F, 1.0F,
	                                              0.0F, 1.0F, 1.0F,
	                                              2.0F, 1.0F, 0.0F,
	                                              1.0F, 1.0F, 1.0F },
	                                  { 2, 2, 3 });
	const auto timesteps = AddFloatConstant(subgraph, { 0.0F, 1.0F }, { 2 });
	const auto a = AddFloatConstant(subgraph, { 2.0F, 0.0F,
	                                            3.0F, 1.0F },
	                                { 2, 2 });
	const auto b = AddFloatConstant(subgraph, { 4.0F, 2.0F,
	                                            11.0F, 7.0F },
	                                { 2, 2 });

	const auto outProd = Layer::AddOutProd(subgraph, lhs, rhs);
	const auto embedding = Layer::AddTimestepEmbedding(subgraph, timesteps, 5, 1000);
	const auto solve = Layer::AddSolveTri(subgraph, a, b);
	subgraph.SetResults({ outProd, embedding, solve });
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
	ASSERT_EQ(outputs.size(), 3u);
	ExpectTensorNear(outputs[0], { 4.0F, 5.0F, 10.0F, 11.0F, 4.0F, 6.0F, 13.0F, 15.0F });
	ExpectTensorNear(outputs[2], { 2.0F, 1.0F, 5.0F, 4.0F });
}

TEST(RareNumericalNode, SerializationRoundTripPreservesRareNumericalNodes)
{
	auto graph = BuildRareNumericalGraph();
	Validation::ValidateGraph(graph);

	const auto path = std::filesystem::path("litenn_rare_numerical_nodes_roundtrip_test.ltnn");
	std::filesystem::remove(path);
	Serialization::SaveModel(graph, path);
	auto loaded = Serialization::LoadModel(path);
	std::filesystem::remove(path);

	auto expected = RunGraph(graph, MakeRareNumericalInputs());
	auto actual = RunGraph(loaded, MakeRareNumericalInputs());
	ASSERT_EQ(actual.size(), expected.size());
	for (auto output = 0uz; output < actual.size(); ++output)
	{
		ASSERT_EQ(actual[output].NumElements(), expected[output].NumElements());
		for (auto index = 0uz; index < actual[output].NumElements(); ++index)
		{
			EXPECT_NEAR(ReadFloat(actual[output], index), ReadFloat(expected[output], index), 1e-5F)
			    << "output=" << output << ", index=" << index;
		}
	}
}

TEST(RareNumericalNode, DumpIncludesRareNumericalNodeKinds)
{
	auto graph = BuildRareNumericalGraph();
	const auto dump = Debug::DumpGraph(graph);
	EXPECT_NE(dump.find("OutProdNode"), std::string::npos);
	EXPECT_NE(dump.find("TimestepEmbeddingNode"), std::string::npos);
	EXPECT_NE(dump.find("SolveTriNode"), std::string::npos);
}
