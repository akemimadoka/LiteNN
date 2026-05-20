#include <gtest/gtest.h>

#include <LiteNN.h>
#include <LiteNN/Pass/ConstFoldPass.h>
#include <LiteNN/Runtime/Interpreter.h>

#include <algorithm>
#include <filesystem>
#include <initializer_list>
#include <optional>
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
		EXPECT_TRUE(std::equal(shape.Dims.begin(), shape.Dims.end(), expected.begin(), expected.end()));
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

	Graph BuildG54Graph()
	{
		Graph graph;
		Subgraph subgraph;
		const auto input = subgraph.AddParam(DataType::Float32, { 1, 1, 3, 3 });
		const auto convWeight = subgraph.AddParam(DataType::Float32, { 1, 1, 2, 2 });
		const auto convBias = subgraph.AddParam(DataType::Float32, { 1 });
		const auto deconvInput = subgraph.AddParam(DataType::Float32, { 1, 1, 2, 2 });
		const auto deconvWeight = subgraph.AddParam(DataType::Float32, { 1, 1, 2, 2 });

		const auto im2col = Layer::AddIm2Col(subgraph, { input, 0 }, { 2, 2 }, { 1, 1 }, { 1, 1 }, { 0, 0 },
		                                     { 0, 0 });
		const auto conv = Layer::AddConv2D(subgraph, { input, 0 }, { convWeight, 0 }, NodeOutput{ convBias, 0 });
		const auto maxPool = Layer::AddMaxPool2D(subgraph, { input, 0 }, { 2, 2 }, { 1, 1 }, { 0, 0 }, { 0, 0 });
		const auto avgPool = Layer::AddAveragePool2D(subgraph, { input, 0 }, { 2, 2 }, { 1, 1 }, { 0, 0 },
		                                             { 0, 0 }, false);
		const auto deconv = Layer::AddConvTranspose2D(subgraph, { deconvInput, 0 }, { deconvWeight, 0 });
		const auto nearest = Layer::AddNearestUpsample2D(subgraph, { deconvInput, 0 }, { 4, 4 });
		const auto bilinear = Layer::AddBilinearUpsample2D(subgraph, { deconvInput, 0 }, { 3, 3 }, true);
		const auto bicubic = Layer::AddBicubicUpsample2D(subgraph, { deconvInput, 0 }, { 2, 2 }, true);
		subgraph.SetResults({ im2col, conv, maxPool, avgPool, deconv, nearest, bilinear, bicubic });
		graph.SetForward(graph.AddSubgraph(std::move(subgraph)));
		return graph;
	}

	std::vector<Tensor<CPU>> MakeG54Inputs()
	{
		std::vector<Tensor<CPU>> inputs;
		inputs.emplace_back(Tensor<CPU>({ 1.0F, 2.0F, 3.0F,
		                                  4.0F, 5.0F, 6.0F,
		                                  7.0F, 8.0F, 9.0F },
		                                { 1, 1, 3, 3 }));
		inputs.emplace_back(Tensor<CPU>({ 1.0F, 0.0F, 0.0F, 1.0F }, { 1, 1, 2, 2 }));
		inputs.emplace_back(Tensor<CPU>({ 10.0F }, { 1 }));
		inputs.emplace_back(Tensor<CPU>({ 1.0F, 2.0F,
		                                  3.0F, 4.0F },
		                                { 1, 1, 2, 2 }));
		inputs.emplace_back(Tensor<CPU>({ 1.0F, 1.0F,
		                                  1.0F, 1.0F },
		                                { 1, 1, 2, 2 }));
		return inputs;
	}
} // namespace

TEST(ConvolutionPoolingNode, Im2ColConv2DAndPoolExecute)
{
	auto graph = BuildG54Graph();
	Validation::ValidateGraph(graph);

	const auto outputs = RunGraph(graph, MakeG54Inputs());
	ASSERT_EQ(outputs.size(), 8u);

	ExpectShape(outputs[0].Shape(), { 1, 4, 4 });
	ExpectTensorNear(outputs[0], { 1.0F, 2.0F, 4.0F, 5.0F,
	                               2.0F, 3.0F, 5.0F, 6.0F,
	                               4.0F, 5.0F, 7.0F, 8.0F,
	                               5.0F, 6.0F, 8.0F, 9.0F });

	ExpectShape(outputs[1].Shape(), { 1, 1, 2, 2 });
	ExpectTensorNear(outputs[1], { 16.0F, 18.0F, 22.0F, 24.0F });

	ExpectShape(outputs[2].Shape(), { 1, 1, 2, 2 });
	ExpectTensorNear(outputs[2], { 5.0F, 6.0F, 8.0F, 9.0F });

	ExpectShape(outputs[3].Shape(), { 1, 1, 2, 2 });
	ExpectTensorNear(outputs[3], { 3.0F, 4.0F, 6.0F, 7.0F });

	ExpectShape(outputs[4].Shape(), { 1, 1, 3, 3 });
	ExpectTensorNear(outputs[4], { 1.0F, 3.0F, 2.0F,
	                               4.0F, 10.0F, 6.0F,
	                               3.0F, 7.0F, 4.0F });

	ExpectShape(outputs[5].Shape(), { 1, 1, 4, 4 });
	ExpectTensorNear(outputs[5], { 1.0F, 1.0F, 2.0F, 2.0F,
	                               1.0F, 1.0F, 2.0F, 2.0F,
	                               3.0F, 3.0F, 4.0F, 4.0F,
	                               3.0F, 3.0F, 4.0F, 4.0F });

	ExpectShape(outputs[6].Shape(), { 1, 1, 3, 3 });
	ExpectTensorNear(outputs[6], { 1.0F, 1.5F, 2.0F,
	                               2.0F, 2.5F, 3.0F,
	                               3.0F, 3.5F, 4.0F });

	ExpectShape(outputs[7].Shape(), { 1, 1, 2, 2 });
	ExpectTensorNear(outputs[7], { 1.0F, 2.0F, 3.0F, 4.0F }, 1e-4F);
}

TEST(ConvolutionPoolingNode, GroupedConv2DExecutesPerGroup)
{
	Graph graph;
	Subgraph subgraph;
	const auto input = subgraph.AddParam(DataType::Float32, { 1, 2, 2, 2 });
	const auto weight = subgraph.AddParam(DataType::Float32, { 2, 1, 1, 1 });
	const auto conv = Layer::AddConv2D(subgraph, { input, 0 }, { weight, 0 }, std::nullopt,
	                                   { 1, 1 }, { 1, 1 }, { 0, 0 }, { 0, 0 }, 2);
	subgraph.SetResults({ conv });
	graph.SetForward(graph.AddSubgraph(std::move(subgraph)));
	Validation::ValidateGraph(graph);

	std::vector<Tensor<CPU>> inputs;
	inputs.emplace_back(Tensor<CPU>({ 1.0F, 2.0F, 3.0F, 4.0F,
	                                  10.0F, 12.0F, 14.0F, 16.0F },
	                                { 1, 2, 2, 2 }));
	inputs.emplace_back(Tensor<CPU>({ 2.0F, 3.0F }, { 2, 1, 1, 1 }));
	const auto outputs = RunGraph(graph, std::move(inputs));
	ASSERT_EQ(outputs.size(), 1u);

	ExpectShape(outputs[0].Shape(), { 1, 2, 2, 2 });
	ExpectTensorNear(outputs[0], { 2.0F, 4.0F, 6.0F, 8.0F,
	                               30.0F, 36.0F, 42.0F, 48.0F });
}

TEST(ConvolutionPoolingNode, AveragePoolPaddingControlsDenominator)
{
	Graph graph;
	Subgraph subgraph;
	const auto input = subgraph.AddParam(DataType::Float32, { 1, 1, 2, 2 });
	const auto excludePad = Layer::AddAveragePool2D(subgraph, { input, 0 }, { 2, 2 }, { 1, 1 }, { 1, 1 },
	                                                { 1, 1 }, false);
	const auto includePad = Layer::AddAveragePool2D(subgraph, { input, 0 }, { 2, 2 }, { 1, 1 }, { 1, 1 },
	                                                { 1, 1 }, true);
	subgraph.SetResults({ excludePad, includePad });
	graph.SetForward(graph.AddSubgraph(std::move(subgraph)));
	Validation::ValidateGraph(graph);

	std::vector<Tensor<CPU>> inputs;
	inputs.emplace_back(Tensor<CPU>({ 1.0F, 2.0F, 3.0F, 4.0F }, { 1, 1, 2, 2 }));
	const auto outputs = RunGraph(graph, std::move(inputs));
	ASSERT_EQ(outputs.size(), 2u);

	ExpectShape(outputs[0].Shape(), { 1, 1, 3, 3 });
	ExpectTensorNear(outputs[0], { 1.0F, 1.5F, 2.0F,
	                               2.0F, 2.5F, 3.0F,
	                               3.0F, 3.5F, 4.0F });

	ExpectShape(outputs[1].Shape(), { 1, 1, 3, 3 });
	ExpectTensorNear(outputs[1], { 0.25F, 0.75F, 0.5F,
	                               1.0F, 2.5F, 1.5F,
	                               0.75F, 1.75F, 1.0F });
}

TEST(ConvolutionPoolingNode, ConstFoldHandlesG54Nodes)
{
	Graph graph;
	Subgraph subgraph;
	const auto input = AddFloatConstant(subgraph,
	                                    { 1.0F, 2.0F, 3.0F,
	                                      4.0F, 5.0F, 6.0F,
	                                      7.0F, 8.0F, 9.0F },
	                                    { 1, 1, 3, 3 });
	const auto im2col = Layer::AddIm2Col(subgraph, input, { 2, 2 }, { 1, 1 }, { 1, 1 }, { 0, 0 }, { 0, 0 });
	const auto weight = AddFloatConstant(subgraph, { 1.0F, 0.0F, 0.0F, 1.0F }, { 1, 1, 2, 2 });
	const auto bias = AddFloatConstant(subgraph, { 10.0F }, { 1 });
	const auto conv = Layer::AddConv2D(subgraph, input, weight, bias);
	const auto maxPool = Layer::AddMaxPool2D(subgraph, input, { 2, 2 }, { 1, 1 }, { 0, 0 }, { 0, 0 });
	const auto deconvInput = AddFloatConstant(subgraph,
	                                          { 1.0F, 2.0F,
	                                            3.0F, 4.0F },
	                                          { 1, 1, 2, 2 });
	const auto deconvWeight = AddFloatConstant(subgraph,
	                                           { 1.0F, 1.0F,
	                                             1.0F, 1.0F },
	                                           { 1, 1, 2, 2 });
	const auto deconv = Layer::AddConvTranspose2D(subgraph, deconvInput, deconvWeight);
	const auto nearest = Layer::AddNearestUpsample2D(subgraph, deconvInput, { 4, 4 });
	subgraph.SetResults({ im2col, conv, maxPool, deconv, nearest });
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
	ASSERT_EQ(outputs.size(), 5u);
	ExpectTensorNear(outputs[0], { 1.0F, 2.0F, 4.0F, 5.0F,
	                               2.0F, 3.0F, 5.0F, 6.0F,
	                               4.0F, 5.0F, 7.0F, 8.0F,
	                               5.0F, 6.0F, 8.0F, 9.0F });
	ExpectTensorNear(outputs[1], { 16.0F, 18.0F, 22.0F, 24.0F });
	ExpectTensorNear(outputs[2], { 5.0F, 6.0F, 8.0F, 9.0F });
	ExpectTensorNear(outputs[3], { 1.0F, 3.0F, 2.0F,
	                               4.0F, 10.0F, 6.0F,
	                               3.0F, 7.0F, 4.0F });
	ExpectTensorNear(outputs[4], { 1.0F, 1.0F, 2.0F, 2.0F,
	                               1.0F, 1.0F, 2.0F, 2.0F,
	                               3.0F, 3.0F, 4.0F, 4.0F,
	                               3.0F, 3.0F, 4.0F, 4.0F });
}

TEST(ConvolutionPoolingNode, SerializationRoundTripAndDumpPreserveG54Nodes)
{
	auto graph = BuildG54Graph();
	Validation::ValidateGraph(graph);

	const auto dump = Debug::DumpGraph(graph);
	EXPECT_NE(dump.find("Im2ColNode"), std::string::npos);
	EXPECT_NE(dump.find("Conv2DNode"), std::string::npos);
	EXPECT_NE(dump.find("ConvTranspose2DNode"), std::string::npos);
	EXPECT_NE(dump.find("Pool2DNode"), std::string::npos);
	EXPECT_NE(dump.find("UpsampleNode"), std::string::npos);

	const auto path = std::filesystem::path("litenn_g54_nodes_roundtrip_test.ltnn");
	std::filesystem::remove(path);
	Serialization::SaveModel(graph, path);
	auto loaded = Serialization::LoadModel(path);
	std::filesystem::remove(path);

	auto expected = RunGraph(graph, MakeG54Inputs());
	auto actual = RunGraph(loaded, MakeG54Inputs());
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
