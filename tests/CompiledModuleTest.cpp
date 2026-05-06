#include <gtest/gtest.h>

#include <LiteNN.h>
#include <LiteNN/Compiler/CompiledModule.h>

#include <array>
#include <filesystem>
#include <ranges>
#include <vector>

using namespace LiteNN;

namespace
{
	float ReadFloat(const Tensor<CPU>& t, std::size_t i)
	{
		return static_cast<const float*>(t.RawData())[i];
	}

	Graph BuildSimpleAddGraph()
	{
		Graph graph;
		Subgraph sg;
		const auto a = sg.AddParam(DataType::Float32, { 2, 2 });
		const auto b = sg.AddParam(DataType::Float32, { 2, 2 });
		const auto y = sg.AddNode(BinaryOpNode{ BinaryOp::Add, { a, 0 }, { b, 0 } },
		                          { OutputInfo{ DataType::Float32, { 2, 2 } } });
		sg.SetResults({ { y, 0 } });
		graph.AddSubgraph(std::move(sg));
		graph.SetForward(0);
		return graph;
	}
} // namespace

TEST(CompiledModuleTest, RunsAfterLoadingFromRodataAndInstructionAddresses)
{
	auto graph = BuildSimpleAddGraph();
	auto compiled = Compiler<CPU>::Compile(graph);

	ASSERT_GT(compiled.Rodata().size(), 0u);
	ASSERT_GT(compiled.Instructions().size(), 0u);

	auto loaded = CompiledModule<CPU>::Load(compiled.Image());
	Tensor<CPU> a({ 1, 2, 3, 4 }, { 2, 2 }, DataType::Float32);
	Tensor<CPU> b({ 10, 20, 30, 40 }, { 2, 2 }, DataType::Float32);
	std::array<Tensor<CPU>, 2> inputs = { std::move(a), std::move(b) };

	auto outputs = loaded.Run(inputs);
	ASSERT_EQ(outputs.size(), 1u);
	ASSERT_EQ(outputs[0].DType(), DataType::Float32);
	ASSERT_TRUE(std::ranges::equal(outputs[0].Shape().Dims, std::vector<std::size_t>{ 2, 2 }));
	EXPECT_FLOAT_EQ(ReadFloat(outputs[0], 0), 11.0f);
	EXPECT_FLOAT_EQ(ReadFloat(outputs[0], 1), 22.0f);
	EXPECT_FLOAT_EQ(ReadFloat(outputs[0], 2), 33.0f);
	EXPECT_FLOAT_EQ(ReadFloat(outputs[0], 3), 44.0f);
}

TEST(CompiledModuleTest, WritesCarrierObjectFile)
{
	auto graph = BuildSimpleAddGraph();
	auto compiled = Compiler<CPU>::Compile(graph);

	const auto path = std::filesystem::temp_directory_path() / "litenn_compiled_module_test.o";
	compiled.WriteObjectFile(path, "litenn_test_module");

	ASSERT_TRUE(std::filesystem::exists(path));
	EXPECT_GT(std::filesystem::file_size(path), 0u);
	std::filesystem::remove(path);
}
