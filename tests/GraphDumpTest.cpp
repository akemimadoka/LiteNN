#include <gtest/gtest.h>

#include <LiteNN.h>
#include <LiteNN/Debug/Dump.h>

#include <string>

using namespace LiteNN;

namespace
{
	Graph BuildGraphForDump()
	{
		Graph graph;
		const auto variableIndex =
		    graph.AddVariable(Variable::Create(Tensor<CPU>(ShapeView{ 4, 8 }, DataType::Float32)));
		graph.AddActivationSlot({ DataType::Float32, { 4, 8 } });
		graph.AddTapeSlot({ DataType::Float32, { 4, 8 } });

		Subgraph subgraph;
		const auto image = subgraph.AddParam(DataType::Float32, { 4, 8 });
		const auto scale = subgraph.AddParam(DataType::Float32, { 4, 8 });
		const auto bias =
		    subgraph.AddNode(VariableRefNode{ variableIndex }, { OutputInfo{ DataType::Float32, { 4, 8 } } });
		const auto shifted = subgraph.AddNode(BinaryOpNode{ BinaryOp::Add, { image, 0 }, { bias, 0 } },
		                                     { OutputInfo{ DataType::Float32, { 4, 8 } } });
		const auto logits = subgraph.AddNode(BinaryOpNode{ BinaryOp::Multiply, { shifted, 0 }, { scale, 0 } },
		                                    { OutputInfo{ DataType::Float32, { 4, 8 } } });
		subgraph.SetResults({ { logits, 0 } });

		graph.SetForward(graph.AddSubgraph(std::move(subgraph)));
		graph.SetInputNames({ "image", "scale" });
		graph.SetOutputNames({ "logits" });
		return graph;
	}
} // namespace

TEST(GraphDumpTest, IncludesSignaturesNodesAndSlots)
{
	auto dump = Debug::DumpGraph(BuildGraphForDump());

	EXPECT_NE(dump.find("graph {"), std::string::npos);
	EXPECT_NE(dump.find("forward = @0"), std::string::npos);
	EXPECT_NE(dump.find("inputs = [image: Float32[4, 8], scale: Float32[4, 8]]"), std::string::npos);
	EXPECT_NE(dump.find("outputs = [logits: Float32[4, 8]]"), std::string::npos);
	EXPECT_NE(dump.find("variables = [var0: Float32[4, 8]]"), std::string::npos);
	EXPECT_NE(dump.find("activation_slots = [slot0: Float32[4, 8]]"), std::string::npos);
	EXPECT_NE(dump.find("tape_slots = [slot0: Float32[4, 8]]"), std::string::npos);
	EXPECT_NE(dump.find("subgraph @0 [forward]"), std::string::npos);
	EXPECT_NE(dump.find("BinaryOpNode(op=BinaryOp::Multiply"), std::string::npos);
	EXPECT_NE(dump.find("VariableRefNode(variable=0)"), std::string::npos);
	EXPECT_NE(dump.find("results = [logits="), std::string::npos);
}