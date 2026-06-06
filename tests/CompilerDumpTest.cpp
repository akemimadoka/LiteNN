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

	Graph BuildExternalLinearGraph()
	{
		Graph graph;
		const auto weightIndex = graph.AddVariable(Variable::Create(
		    Tensor<CPU>({ 0.5, -0.25, 0.75, 0.125, -0.5, 0.25 }, { 3, 2 }, DataType::Float32)));
		graph.SetVariableName(weightIndex, "projection.weight");

		Subgraph subgraph;
		const auto input = subgraph.AddParam(DataType::Float32, { 2, 3 });
		const auto weight =
		    subgraph.AddNode(VariableRefNode{ weightIndex }, { OutputInfo{ DataType::Float32, { 3, 2 } } });
		const auto product = subgraph.AddNode(BinaryOpNode{ BinaryOp::MatMul, { input, 0 }, { weight, 0 } },
		                                      { OutputInfo{ DataType::Float32, { 2, 2 } } });
		const auto biasTensor = Tensor<CPU>({ 0.1, -0.2 }, { 1, 2 }, DataType::Float32);
		const auto bias = subgraph.AddNode(
		    ConstantNode{ biasTensor.CopyToDevice(PolymorphicDevice{ CPU{} }) },
		    { OutputInfo{ DataType::Float32, { 1, 2 } } });
		const auto sum = subgraph.AddNode(BinaryOpNode{ BinaryOp::Add, { product, 0 }, { bias, 0 } },
		                                  { OutputInfo{ DataType::Float32, { 2, 2 } } });
		subgraph.SetResults({ { sum, 0 } });
		graph.SetForward(graph.AddSubgraph(std::move(subgraph)));
		graph.SetInputNames({ "input" });
		graph.SetOutputNames({ "logits" });
		return graph;
	}
} // namespace

TEST(CompilerDumpTest, DumpsInputDialectMlir)
{
	auto dump = Debug::DumpMLIR(Detail::BuildExecutablePlanFromGraph(BuildSimpleAddGraph()), Debug::MLIRDumpStage::InputDialect);

	EXPECT_NE(dump.find("litenn.func @subgraph_0"), std::string::npos);
	EXPECT_NE(dump.find("litenn.binary"), std::string::npos);
}

TEST(CompilerDumpTest, DumpsLoweredMlir)
{
	auto dump = Debug::DumpMLIR(Detail::BuildExecutablePlanFromGraph(BuildSimpleAddGraph()), Debug::MLIRDumpStage::AfterLowering);

	EXPECT_EQ(dump.find("litenn.binary"), std::string::npos);
	EXPECT_NE(dump.find("func.func"), std::string::npos);
}

TEST(CompilerDumpTest, DumpsCompiledModuleMetadata)
{
	auto artifact = Compiler<CPU>::CompileArtifact(Detail::BuildExecutablePlanFromGraph(BuildSimpleAddGraph()));
	auto dump = Debug::DumpCompiledModuleMetadata(artifact);

	EXPECT_NE(dump.find("compiled_module {"), std::string::npos);
	EXPECT_NE(dump.find("backend = cpu_native"), std::string::npos);
	EXPECT_NE(dump.find("rodata_size = "), std::string::npos);
	EXPECT_NE(dump.find("instruction_size = "), std::string::npos);
	EXPECT_NE(dump.find("inputs = [lhs: Float32[2, 2], rhs: Float32[2, 2]]"), std::string::npos);
	EXPECT_NE(dump.find("outputs = [sum: Float32[2, 2]]"), std::string::npos);
}

TEST(CompilerDumpTest, DumpsSeparatedArtifactExternalTensorMetadata)
{
	CompilerOptions options;
	options.cpuAOTThreadCount = 2;
	options.cpuAOTParallelMinFlops = 1;
	options.enableCPUAOTExternalRegions = true;

	auto artifact = Compiler<CPU>::CompileArtifact(Detail::BuildExecutablePlanFromGraph(BuildExternalLinearGraph()), options);
	auto separated = artifact.SeparateRodata();
	auto dump = Debug::DumpCompiledModuleMetadata(separated);

	EXPECT_NE(dump.find("compiled_module {"), std::string::npos);
	EXPECT_NE(dump.find("backend = cpu_native"), std::string::npos);
	EXPECT_NE(dump.find("metadata_size = "), std::string::npos);
	EXPECT_NE(dump.find("constants_size = "), std::string::npos);
	EXPECT_NE(dump.find("weights_size = "), std::string::npos);
	EXPECT_NE(dump.find("regions = [metadata:"), std::string::npos);
	EXPECT_NE(dump.find("external_tensors = ["), std::string::npos);
	EXPECT_NE(dump.find("projection.weight: region=weights"), std::string::npos);
	EXPECT_NE(dump.find("constant_"), std::string::npos);
	EXPECT_NE(dump.find("policy=exact_checksum"), std::string::npos);
	EXPECT_NE(dump.find("inputs = [input: Float32[2, 3]]"), std::string::npos);
	EXPECT_NE(dump.find("outputs = [logits: Float32[2, 2]]"), std::string::npos);
}
