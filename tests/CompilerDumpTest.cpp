#include <gtest/gtest.h>

#include <LiteNN.h>
#include <LiteNN/Compiler/Dump.h>

#include <string>

using namespace LiteNN;

namespace
{
	Graph BuildSimpleAddGraph()
	{
		Graph graph;
		Subgraph subgraph;
		const auto lhs = subgraph.AddParam(DataType::Float32, { 2, 2 });
		const auto rhs = subgraph.AddParam(DataType::Float32, { 2, 2 });
		const auto sum = subgraph.AddNode(BinaryOpNode{ BinaryOp::Add, { lhs, 0 }, { rhs, 0 } },
		                                 { OutputInfo{ DataType::Float32, { 2, 2 } } });
		subgraph.SetResults({ { sum, 0 } });
		graph.SetForward(graph.AddSubgraph(std::move(subgraph)));
		graph.SetInputNames({ "lhs", "rhs" });
		graph.SetOutputNames({ "sum" });
		return graph;
	}
} // namespace

TEST(CompilerDumpTest, DumpsInputDialectMlir)
{
	auto dump = Debug::DumpMLIR(BuildSimpleAddGraph(), Debug::MLIRDumpStage::InputDialect);

	EXPECT_NE(dump.find("litenn.func @subgraph_0"), std::string::npos);
	EXPECT_NE(dump.find("litenn.binary"), std::string::npos);
}

TEST(CompilerDumpTest, DumpsLoweredMlir)
{
	auto dump = Debug::DumpMLIR(BuildSimpleAddGraph(), Debug::MLIRDumpStage::AfterLowering);

	EXPECT_EQ(dump.find("litenn.binary"), std::string::npos);
	EXPECT_NE(dump.find("linalg.generic"), std::string::npos);
}

TEST(CompilerDumpTest, DumpsCompiledModuleMetadata)
{
	auto artifact = Compiler<CPU>::CompileArtifact(BuildSimpleAddGraph());
	auto dump = Debug::DumpCompiledModuleMetadata(artifact);

	EXPECT_NE(dump.find("compiled_module {"), std::string::npos);
	EXPECT_NE(dump.find("rodata_size = "), std::string::npos);
	EXPECT_NE(dump.find("instruction_size = "), std::string::npos);
	EXPECT_NE(dump.find("inputs = [lhs: Float32[2, 2], rhs: Float32[2, 2]]"), std::string::npos);
	EXPECT_NE(dump.find("outputs = [sum: Float32[2, 2]]"), std::string::npos);
}