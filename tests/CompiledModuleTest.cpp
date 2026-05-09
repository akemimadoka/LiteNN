#include <gtest/gtest.h>

#include <LiteNN.h>
#include <LiteNN/Compiler/CompiledModule.h>

#include <array>
#include <cstddef>
#include <format>
#include <filesystem>
#include <future>
#include <ranges>
#include <string>
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
		graph.SetInputNames({ "lhs", "rhs" });
		graph.SetOutputNames({ "sum" });
		return graph;
	}

	struct WorkerResult
	{
		bool ok{};
		std::string message;
	};

	WorkerResult RunCompiledModuleWorker(const CompiledModule<CPU>& module, int workerId)
	{
		try
		{
			for (int iteration = 0; iteration < 32; ++iteration)
			{
				const auto base = static_cast<float>(workerId * 100 + iteration);
				Tensor<CPU> a({ base + 1, base + 2, base + 3, base + 4 }, { 2, 2 }, DataType::Float32);
				Tensor<CPU> b({ 10, 20, 30, 40 }, { 2, 2 }, DataType::Float32);
				std::array<Tensor<CPU>, 2> inputs = { std::move(a), std::move(b) };

				auto outputs = module.Run(inputs);
				if (outputs.size() != 1 || outputs[0].NumElements() != 4)
				{
					return { false, std::format("worker {} output metadata mismatch", workerId) };
				}
				for (std::size_t i = 0; i < 4; ++i)
				{
					const auto expected = base + static_cast<float>(i + 1) + static_cast<float>((i + 1) * 10);
					if (ReadFloat(outputs[0], i) != expected)
					{
						return { false, std::format("worker {} output {} mismatch at iteration {}", workerId, i,
						                            iteration) };
					}
				}
			}
		}
		catch (const std::exception& ex)
		{
			return { false, std::format("worker {} threw: {}", workerId, ex.what()) };
		}
		return { true, {} };
	}
} // namespace

TEST(CompiledModuleTest, RunsAfterLoadingFromRodataAndInstructionAddresses)
{
	auto graph = BuildSimpleAddGraph();
	auto compiled = Compiler<CPU>::Compile(graph);

	ASSERT_GT(compiled.Rodata().size(), 0u);
	ASSERT_GT(compiled.Instructions().size(), 0u);
	ASSERT_EQ(compiled.InputSpecs().size(), 2u);
	ASSERT_EQ(compiled.OutputSpecs().size(), 1u);
	EXPECT_EQ(compiled.InputSpecs()[0].name, "lhs");
	EXPECT_EQ(compiled.InputSpecs()[1].name, "rhs");
	EXPECT_EQ(compiled.OutputSpecs()[0].name, "sum");
	EXPECT_EQ(compiled.FindInput("lhs"), 0u);
	EXPECT_EQ(compiled.FindInput("rhs"), 1u);
	EXPECT_EQ(compiled.FindOutput("sum"), 0u);

	auto loaded = CompiledModule<CPU>::Load(compiled.Image());
	ASSERT_EQ(loaded.InputSpecs().size(), 2u);
	ASSERT_EQ(loaded.OutputSpecs().size(), 1u);
	EXPECT_EQ(loaded.InputSpecs()[0].name, "lhs");
	EXPECT_EQ(loaded.InputSpecs()[1].name, "rhs");
	EXPECT_EQ(loaded.OutputSpecs()[0].name, "sum");
	EXPECT_EQ(loaded.FindInput("lhs"), 0u);
	EXPECT_EQ(loaded.FindInput("rhs"), 1u);
	EXPECT_EQ(loaded.FindOutput("sum"), 0u);
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

TEST(CompiledModuleTest, ReportsInputMismatchWithExpectedAndActualSignature)
{
	auto graph = BuildSimpleAddGraph();
	auto compiled = Compiler<CPU>::Compile(graph);

	Tensor<CPU> wrongA({ 1, 2, 3 }, { 3 }, DataType::Float32);
	Tensor<CPU> b({ 10, 20, 30, 40 }, { 2, 2 }, DataType::Float32);
	std::array<Tensor<CPU>, 2> inputs = { std::move(wrongA), std::move(b) };

	try
	{
		(void)compiled.Run(inputs);
		FAIL() << "expected CompiledModule input validation to throw";
	}
	catch (const std::runtime_error& ex)
	{
		const std::string message = ex.what();
		EXPECT_NE(message.find("CompiledModule input 0 ('lhs') mismatch"), std::string::npos);
		EXPECT_NE(message.find("expected Float32[2, 2]"), std::string::npos);
		EXPECT_NE(message.find("got Float32[3]"), std::string::npos);
	}
}

TEST(CompiledModuleTest, RunIntoWritesCallerProvidedOutputBuffer)
{
	auto graph = BuildSimpleAddGraph();
	auto compiled = Compiler<CPU>::Compile(graph);

	Tensor<CPU> a({ 1, 2, 3, 4 }, { 2, 2 }, DataType::Float32);
	Tensor<CPU> b({ 10, 20, 30, 40 }, { 2, 2 }, DataType::Float32);
	std::array<Tensor<CPU>, 2> inputs = { std::move(a), std::move(b) };
	std::array<Tensor<CPU>, 1> outputs = {
		Tensor<CPU>(Uninitialized, { 2, 2 }, DataType::Float32)
	};

	compiled.RunInto(inputs, outputs);

	EXPECT_FLOAT_EQ(ReadFloat(outputs[0], 0), 11.0f);
	EXPECT_FLOAT_EQ(ReadFloat(outputs[0], 1), 22.0f);
	EXPECT_FLOAT_EQ(ReadFloat(outputs[0], 2), 33.0f);
	EXPECT_FLOAT_EQ(ReadFloat(outputs[0], 3), 44.0f);
}

TEST(CompiledModuleTest, RejectsRodataWithMismatchedAbiMetadata)
{
	auto graph = BuildSimpleAddGraph();
	auto compiled = Compiler<CPU>::Compile(graph);

	std::vector<std::byte> rodata(compiled.Rodata().begin(), compiled.Rodata().end());
	ASSERT_GT(rodata.size(), 16u);
	rodata[12] = std::byte{ 0 };
	rodata[13] = std::byte{ 0 };
	rodata[14] = std::byte{ 0 };
	rodata[15] = std::byte{ 0 };

	const auto image = CompiledModuleImage{
	    .rodata = rodata.data(),
	    .rodataSize = rodata.size(),
	    .instructions = compiled.Instructions().data(),
	    .instructionSize = compiled.Instructions().size(),
	};

	try
	{
		(void)CompiledModule<CPU>::Load(image);
		FAIL() << "expected ABI metadata validation to throw";
	}
	catch (const std::runtime_error& ex)
	{
		const std::string message = ex.what();
		EXPECT_NE(message.find("pointer size"), std::string::npos);
	}
}

TEST(CompiledModuleTest, ConcurrentRunUsesIndependentInputAndOutputBuffers)
{
	auto graph = BuildSimpleAddGraph();
	auto compiled = Compiler<CPU>::Compile(graph);
	auto loaded = CompiledModule<CPU>::Load(compiled.Image());

	std::vector<std::future<WorkerResult>> futures;
	for (int workerId = 0; workerId < 4; ++workerId)
	{
		futures.push_back(std::async(std::launch::async, [&loaded, workerId] {
			return RunCompiledModuleWorker(loaded, workerId);
		}));
	}

	for (auto& future : futures)
	{
		auto result = future.get();
		EXPECT_TRUE(result.ok) << result.message;
	}
}
