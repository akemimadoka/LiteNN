#include <gtest/gtest.h>

#include <LiteNN.h>

#include <ranges>
#include <stdexcept>
#include <vector>

using namespace LiteNN;

namespace
{
	Graph BuildNamedGraph()
	{
		Graph graph;
		Subgraph sg;
		const auto image = sg.AddParam(DataType::Float32, { 4, 8 });
		const auto scale = sg.AddParam(DataType::Float32, { 4, 8 });
		const auto y = sg.AddNode(BinaryOpNode{ BinaryOp::Multiply, { image, 0 }, { scale, 0 } },
		                          { OutputInfo{ DataType::Float32, { 4, 8 } } });
		sg.SetResults({ { y, 0 } });
		graph.SetForward(graph.AddSubgraph(std::move(sg)));
		graph.SetInputNames({ "image", "scale" });
		graph.SetOutputNames({ "logits" });
		return graph;
	}
} // namespace

TEST(Signature, GraphReturnsNamedInputAndOutputSpecs)
{
	auto graph = BuildNamedGraph();
	Validation::ValidateGraph(graph);

	ASSERT_EQ(graph.FindInput("image"), 0u);
	ASSERT_EQ(graph.FindInput("scale"), 1u);
	ASSERT_EQ(graph.FindOutput("logits"), 0u);
	EXPECT_FALSE(graph.FindInput("missing").has_value());

	auto inputs = graph.InputSignature();
	ASSERT_EQ(inputs.size(), 2);
	EXPECT_EQ(inputs[0].name, "image");
	EXPECT_EQ(inputs[0].dtype, DataType::Float32);
	EXPECT_TRUE(std::ranges::equal(inputs[0].shape, std::vector<std::size_t>{ 4, 8 }));
	EXPECT_EQ(inputs[1].name, "scale");

	auto outputs = graph.OutputSignature();
	ASSERT_EQ(outputs.size(), 1);
	EXPECT_EQ(outputs[0].name, "logits");
	EXPECT_EQ(outputs[0].dtype, DataType::Float32);
	EXPECT_TRUE(std::ranges::equal(outputs[0].shape, std::vector<std::size_t>{ 4, 8 }));
}

TEST(Signature, GraphUsesStableDefaultNamesWhenNamesAreNotSet)
{
	Graph graph;
	Subgraph sg;
	const auto x = sg.AddParam(DataType::Float32, { 2 });
	sg.SetResults({ { x, 0 } });
	graph.SetForward(graph.AddSubgraph(std::move(sg)));

	auto inputs = graph.InputSignature();
	auto outputs = graph.OutputSignature();
	ASSERT_EQ(inputs.size(), 1);
	ASSERT_EQ(outputs.size(), 1);
	EXPECT_EQ(inputs[0].name, "input0");
	EXPECT_EQ(outputs[0].name, "output0");
}

TEST(Signature, GraphValidatorRejectsInvalidSignatureNames)
{
	auto graph = BuildNamedGraph();
	graph.SetInputNames({ "dup", "dup" });

	try
	{
		Validation::ValidateGraph(graph);
		FAIL() << "expected duplicate input name to fail validation";
	}
	catch (const Validation::GraphValidationError& ex)
	{
		const std::string message = ex.what();
		EXPECT_NE(message.find("duplicate input name"), std::string::npos);
	}
}
