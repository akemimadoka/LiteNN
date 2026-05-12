#include <gtest/gtest.h>

#include <LiteNN.h>
#include <LiteNN/Compiler/CompiledModule.h>

#include <array>
#include <cstddef>
#include <cstdint>
#include <format>
#include <filesystem>
#include <future>
#include <ranges>
#include <span>
#include <string>
#include <vector>

using namespace LiteNN;

namespace
{
	float ReadFloat(const Tensor<CPU>& t, std::size_t i)
	{
		return static_cast<const float*>(t.RawData())[i];
	}

	std::uint64_t ReadU64LE(std::span<const std::byte> bytes, std::size_t offset)
	{
		std::uint64_t value = 0;
		for (int i = 0; i < 8; ++i)
		{
			value |= std::to_integer<std::uint64_t>(bytes[offset + i]) << (i * 8);
		}
		return value;
	}

	std::size_t RodataBackendOffset(std::span<const std::byte> rodata)
	{
		constexpr std::size_t kMagicSize = 8;
		constexpr std::size_t kU32Size = 4;
		constexpr std::size_t kU64Size = 8;
		std::size_t offset = kMagicSize + kU32Size + kU32Size + kU32Size;
		const auto tripleSize = ReadU64LE(rodata, offset);
		offset += kU64Size + static_cast<std::size_t>(tripleSize);
		return offset;
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
	auto artifact = Compiler<CPU>::CompileArtifact(graph);

	const auto path = std::filesystem::temp_directory_path() / "litenn_compiled_module_test.o";
	artifact.WriteObjectFile(path, "litenn_test_module");

	ASSERT_TRUE(std::filesystem::exists(path));
	EXPECT_GT(std::filesystem::file_size(path), 0u);
	std::filesystem::remove(path);
}

TEST(CompiledModuleTest, CompileArtifactSeparatesObjectGenerationFromLoad)
{
	auto graph = BuildSimpleAddGraph();
	auto artifact = Compiler<CPU>::CompileArtifact(graph);

	EXPECT_EQ(artifact.Backend(), CompiledModuleBackend::CPUNative);
	ASSERT_GT(artifact.Rodata().size(), 0u);
	ASSERT_GT(artifact.Instructions().size(), 0u);
	ASSERT_EQ(artifact.InputSpecs().size(), 2u);
	ASSERT_EQ(artifact.OutputSpecs().size(), 1u);
	EXPECT_EQ(artifact.FindInput("lhs"), 0u);
	EXPECT_EQ(artifact.FindInput("rhs"), 1u);
	EXPECT_EQ(artifact.FindOutput("sum"), 0u);

	auto loaded = artifact.Load();
	EXPECT_EQ(loaded.Backend(), CompiledModuleBackend::CPUNative);
	Tensor<CPU> a({ 1, 2, 3, 4 }, { 2, 2 }, DataType::Float32);
	Tensor<CPU> b({ 10, 20, 30, 40 }, { 2, 2 }, DataType::Float32);
	std::array<Tensor<CPU>, 2> inputs = { std::move(a), std::move(b) };

	auto outputs = loaded.Run(inputs);
	ASSERT_EQ(outputs.size(), 1u);
	EXPECT_FLOAT_EQ(ReadFloat(outputs[0], 0), 11.0f);
	EXPECT_FLOAT_EQ(ReadFloat(outputs[0], 1), 22.0f);
	EXPECT_FLOAT_EQ(ReadFloat(outputs[0], 2), 33.0f);
	EXPECT_FLOAT_EQ(ReadFloat(outputs[0], 3), 44.0f);
}

TEST(CompiledModuleTest, ExposesBackendMetadataAcrossArtifactAndLoad)
{
	auto graph = BuildSimpleAddGraph();
	auto artifact = Compiler<CPU>::CompileArtifact(graph);
	auto copied = CompiledModuleArtifact::CopyFromImage(artifact.Image());
	auto compiled = Compiler<CPU>::Compile(graph);
	auto loaded = CompiledModule<CPU>::Load(compiled.Image());

	EXPECT_EQ(artifact.Backend(), CompiledModuleBackend::CPUNative);
	EXPECT_EQ(copied.Backend(), CompiledModuleBackend::CPUNative);
	EXPECT_EQ(compiled.Backend(), CompiledModuleBackend::CPUNative);
	EXPECT_EQ(loaded.Backend(), CompiledModuleBackend::CPUNative);
}

TEST(CompiledModuleTest, LoadsArtifactFromExportedSymbolAddresses)
{
	auto graph = BuildSimpleAddGraph();
	auto artifact = Compiler<CPU>::CompileArtifact(graph);

	const std::uint64_t rodataSize = artifact.Rodata().size();
	const std::uint64_t instructionSize = artifact.Instructions().size();
	auto exportedArtifact = CompiledModuleArtifact::FromExportedSymbols({
	    .rodata = artifact.Rodata().data(),
	    .rodataSize = &rodataSize,
	    .instructions = artifact.Instructions().data(),
	    .instructionSize = &instructionSize,
	});

	ASSERT_EQ(exportedArtifact.InputSpecs().size(), 2u);
	ASSERT_EQ(exportedArtifact.OutputSpecs().size(), 1u);

	auto loaded = exportedArtifact.Load();
	Tensor<CPU> a({ 1, 2, 3, 4 }, { 2, 2 }, DataType::Float32);
	Tensor<CPU> b({ 10, 20, 30, 40 }, { 2, 2 }, DataType::Float32);
	std::array<Tensor<CPU>, 2> inputs = { std::move(a), std::move(b) };

	auto outputs = loaded.Run(inputs);
	ASSERT_EQ(outputs.size(), 1u);
	EXPECT_FLOAT_EQ(ReadFloat(outputs[0], 0), 11.0f);
	EXPECT_FLOAT_EQ(ReadFloat(outputs[0], 1), 22.0f);
	EXPECT_FLOAT_EQ(ReadFloat(outputs[0], 2), 33.0f);
	EXPECT_FLOAT_EQ(ReadFloat(outputs[0], 3), 44.0f);
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

TEST(CompiledModuleTest, NarrowMatMulRowTileMatchesReference)
{
	Graph graph;
	const auto weightIndex = graph.AddVariable(Variable::Create(Tensor<CPU>(
	    { 1.0, -2.0, 0.5, 3.0, -1.0,
	      0.25, 4.0, -1.5, 2.0, 0.75,
	      -3.0, 1.0, 2.5, -0.5, 1.25 },
	    { 3, 5 }, DataType::Float32)));

	Subgraph sg;
	const auto input = sg.AddParam(DataType::Float32, { 16, 3 });
	const auto weight = sg.AddNode(VariableRefNode{ weightIndex },
	                               { OutputInfo{ DataType::Float32, { 3, 5 } } });
	const auto output = sg.AddNode(BinaryOpNode{ BinaryOp::MatMul, { input, 0 }, { weight, 0 } },
	                               { OutputInfo{ DataType::Float32, { 16, 5 } } });
	sg.SetResults({ { output, 0 } });
	graph.AddSubgraph(std::move(sg));
	graph.SetForward(0);

	std::vector<double> inputData(16 * 3);
	for (std::size_t row = 0; row < 16; ++row)
	{
		inputData[row * 3 + 0] = static_cast<double>(row + 1);
		inputData[row * 3 + 1] = static_cast<double>(row % 5) - 2.0;
		inputData[row * 3 + 2] = static_cast<double>(row % 7) * 0.5 + 1.0;
	}
	Tensor<CPU> x(std::span<const double>(inputData), { 16, 3 }, DataType::Float32);
	std::array<Tensor<CPU>, 1> inputs = { std::move(x) };
	std::array<Tensor<CPU>, 1> outputs = {
	    Tensor<CPU>(Uninitialized, { 16, 5 }, DataType::Float32)
	};

	auto compiled = Compiler<CPU>::Compile(graph);
	compiled.RunInto(inputs, outputs);

	const double weights[3][5] = {
	    { 1.0, -2.0, 0.5, 3.0, -1.0 },
	    { 0.25, 4.0, -1.5, 2.0, 0.75 },
	    { -3.0, 1.0, 2.5, -0.5, 1.25 },
	};
	for (std::size_t row = 0; row < 16; ++row)
	{
		for (std::size_t col = 0; col < 5; ++col)
		{
			double expected = 0.0;
			for (std::size_t k = 0; k < 3; ++k)
			{
				expected += inputData[row * 3 + k] * weights[k][col];
			}
			EXPECT_NEAR(ReadFloat(outputs[0], row * 5 + col), expected, 1e-5f);
		}
	}
}

TEST(CompiledModuleTest, PackedWideMatMulMatchesReference)
{
	Graph graph;
	std::vector<double> weightData(3 * 256);
	for (std::size_t k = 0; k < 3; ++k)
	{
		for (std::size_t col = 0; col < 256; ++col)
		{
			weightData[k * 256 + col] =
			    (static_cast<double>((col % 17) + 1) * 0.03125) - static_cast<double>(k) * 0.125;
		}
	}
	const auto weightIndex = graph.AddVariable(Variable::Create(
	    Tensor<CPU>(std::span<const double>(weightData), { 3, 256 }, DataType::Float32)));

	Subgraph sg;
	const auto input = sg.AddParam(DataType::Float32, { 8, 3 });
	const auto weight = sg.AddNode(VariableRefNode{ weightIndex },
	                               { OutputInfo{ DataType::Float32, { 3, 256 } } });
	const auto output = sg.AddNode(BinaryOpNode{ BinaryOp::MatMul, { input, 0 }, { weight, 0 } },
	                               { OutputInfo{ DataType::Float32, { 8, 256 } } });
	sg.SetResults({ { output, 0 } });
	graph.AddSubgraph(std::move(sg));
	graph.SetForward(0);

	std::vector<double> inputData(8 * 3);
	for (std::size_t row = 0; row < 8; ++row)
	{
		inputData[row * 3 + 0] = static_cast<double>(row + 1) * 0.5;
		inputData[row * 3 + 1] = static_cast<double>(row % 3) - 1.0;
		inputData[row * 3 + 2] = static_cast<double>(row % 5) * 0.25 + 0.75;
	}
	Tensor<CPU> x(std::span<const double>(inputData), { 8, 3 }, DataType::Float32);
	std::array<Tensor<CPU>, 1> inputs = { std::move(x) };
	std::array<Tensor<CPU>, 1> outputs = {
	    Tensor<CPU>(Uninitialized, { 8, 256 }, DataType::Float32)
	};

	auto compiled = Compiler<CPU>::Compile(graph);
	compiled.RunInto(inputs, outputs);

	for (std::size_t row = 0; row < 8; ++row)
	{
		for (std::size_t col = 0; col < 256; ++col)
		{
			double expected = 0.0;
			for (std::size_t k = 0; k < 3; ++k)
			{
				expected += inputData[row * 3 + k] * weightData[k * 256 + col];
			}
			EXPECT_NEAR(ReadFloat(outputs[0], row * 256 + col), expected, 1e-5f);
		}
	}
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

TEST(CompiledModuleTest, RejectsRodataWithInvalidBackendMetadata)
{
	auto graph = BuildSimpleAddGraph();
	auto compiled = Compiler<CPU>::Compile(graph);

	std::vector<std::byte> rodata(compiled.Rodata().begin(), compiled.Rodata().end());
	const auto backendOffset = RodataBackendOffset(rodata);
	ASSERT_LE(backendOffset + 4, rodata.size());
	rodata[backendOffset + 0] = std::byte{ 0xff };
	rodata[backendOffset + 1] = std::byte{ 0xff };
	rodata[backendOffset + 2] = std::byte{ 0xff };
	rodata[backendOffset + 3] = std::byte{ 0xff };

	const auto image = CompiledModuleImage{
	    .rodata = rodata.data(),
	    .rodataSize = rodata.size(),
	    .instructions = compiled.Instructions().data(),
	    .instructionSize = compiled.Instructions().size(),
	};

	try
	{
		(void)CompiledModule<CPU>::Load(image);
		FAIL() << "expected backend metadata validation to throw";
	}
	catch (const std::runtime_error& ex)
	{
		const std::string message = ex.what();
		EXPECT_NE(message.find("backend"), std::string::npos);
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

TEST(CompiledModuleTest, RunManyIntoRunsIndependentInvocationsConcurrently)
{
	auto graph = BuildSimpleAddGraph();
	auto compiled = Compiler<CPU>::Compile(graph);

	constexpr std::size_t kInvocationCount = 16;
	std::vector<std::array<Tensor<CPU>, 2>> inputs;
	std::vector<std::array<Tensor<CPU>, 1>> outputs;
	std::vector<CompiledModuleInvocation> invocations;
	inputs.reserve(kInvocationCount);
	outputs.reserve(kInvocationCount);
	invocations.reserve(kInvocationCount);

	for (std::size_t i = 0; i < kInvocationCount; ++i)
	{
		const auto base = static_cast<float>(i * 100);
		inputs.push_back({
		    Tensor<CPU>({ base + 1, base + 2, base + 3, base + 4 }, { 2, 2 }, DataType::Float32),
		    Tensor<CPU>({ 10, 20, 30, 40 }, { 2, 2 }, DataType::Float32),
		});
		outputs.push_back({
		    Tensor<CPU>(Uninitialized, { 2, 2 }, DataType::Float32),
		});
		invocations.push_back({
		    .inputs = std::span<const Tensor<CPU>>(inputs.back()),
		    .outputs = std::span<Tensor<CPU>>(outputs.back()),
		});
	}

	compiled.RunManyInto(invocations, 4);

	for (std::size_t i = 0; i < kInvocationCount; ++i)
	{
		const auto base = static_cast<float>(i * 100);
		for (std::size_t element = 0; element < 4; ++element)
		{
			const auto expected = base + static_cast<float>(element + 1) +
			                      static_cast<float>((element + 1) * 10);
			EXPECT_FLOAT_EQ(ReadFloat(outputs[i][0], element), expected);
		}
	}
}
