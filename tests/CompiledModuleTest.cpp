#include <gtest/gtest.h>

#include <LiteNN.h>
#include <LiteNN/Compiler/CUDANativePayload.h>
#include <LiteNN/Compiler/CompiledModule.h>
#include <LiteNN/Compiler/Dump.h>
#include <LiteNN/Pass/FusionPass.h>
#include <LiteNN/Runtime/Interpreter.h>

#ifdef LITENN_ENABLE_CUDA
#include <LiteNN/Compiler/CUDANativeCodegen.h>
#endif

#include <array>
#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <format>
#include <future>
#include <fstream>
#include <iterator>
#include <ranges>
#include <span>
#include <stdexcept>
#include <string>
#include <vector>

using namespace LiteNN;

namespace
{
	float ReadFloat(const Tensor<CPU>& t, std::size_t i)
	{
		return static_cast<const float*>(t.RawData())[i];
	}

	void ExpectTensorNear(const Tensor<CPU>& actual, const Tensor<CPU>& expected, float tolerance = 1e-5f)
	{
		ASSERT_EQ(actual.DType(), expected.DType());
		ASSERT_EQ(actual.Shape(), expected.Shape());
		ASSERT_EQ(actual.NumElements(), expected.NumElements());
		for (std::size_t i = 0; i < actual.NumElements(); ++i)
		{
			EXPECT_NEAR(ReadFloat(actual, i), ReadFloat(expected, i), tolerance);
		}
	}

	void ExpectCompiledMatchesInterpreter(const Graph& graph, std::span<const Tensor<CPU>> inputs,
	                                      float tolerance = 1e-5f)
	{
		Runtime::Interpreter<CPU> interpreter;
		const auto expected = interpreter.RunForward(graph, inputs);
		auto compiled = Compiler<CPU>::Compile(graph);
		const auto outputs = compiled.Run(inputs);

		ASSERT_EQ(outputs.size(), expected.size());
		for (std::size_t i = 0; i < outputs.size(); ++i)
		{
			ExpectTensorNear(outputs[i], expected[i], tolerance);
		}
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

	const CompiledModuleRegionInfo* FindRegionInfo(std::span<const CompiledModuleRegionInfo> infos,
	                                               std::string_view name)
	{
		for (const auto& info : infos)
		{
			if (info.name == name)
			{
				return &info;
			}
		}
		return nullptr;
	}

	std::vector<std::byte> ReadFileBytes(const std::filesystem::path& path)
	{
		std::ifstream in(path, std::ios::binary);
		if (!in)
		{
			throw std::runtime_error("failed to open test file");
		}
		const std::vector<char> chars{ std::istreambuf_iterator<char>(in), std::istreambuf_iterator<char>() };
		std::vector<std::byte> bytes(chars.size());
		std::memcpy(bytes.data(), chars.data(), chars.size());
		return bytes;
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

	Graph BuildQuantizedConstantOutputGraph()
	{
		Graph graph;
		Subgraph sg;
		auto params = PerTensorAffineQuantization(DataType::Int8, 0.25F, -3);
		Tensor<CPU> storage({ -3.0, 1.0, 5.0, 7.0 }, { 2, 2 }, DataType::Int8);
		const auto quantized =
		    sg.AddNode(QuantizedConstantNode{ storage.CopyToDevice(PolymorphicDevice{ CPU{} }), params },
		               { OutputInfo{ DataType::Int8, { 2, 2 } } });
		sg.SetResults({ { quantized, 0 } });
		graph.AddSubgraph(std::move(sg));
		graph.SetForward(0);
		graph.SetOutputNames({ "quantized_weight" });
		return graph;
	}

	Graph BuildGetRowsGraph()
	{
		Graph graph;
		const auto tableIndex = graph.AddVariable(Variable::Create(
		    Tensor<CPU>({ 10.0f, 11.0f, 20.0f, 21.0f, 30.0f, 31.0f, 40.0f, 41.0f }, { 4, 2 }, DataType::Float32)));

		Subgraph sg;
		const auto indices = sg.AddParam(DataType::Int32, { 3 });
		const auto table = sg.AddNode(VariableRefNode{ tableIndex }, { OutputInfo{ DataType::Float32, { 4, 2 } } });
		const auto gathered =
		    sg.AddNode(GetRowsNode{ { table, 0 }, { indices, 0 } }, { OutputInfo{ DataType::Float32, { 3, 2 } } });
		sg.SetResults({ { gathered, 0 } });
		graph.SetForward(graph.AddSubgraph(std::move(sg)));
		graph.SetInputNames({ "token_ids" });
		graph.SetOutputNames({ "embeddings" });
		return graph;
	}

	Graph BuildTinyLinearChainGraph(std::size_t batch)
	{
		Graph graph;
		const auto h1 = Layer::CreateLinear(
		    graph,
		    Tensor<CPU>({ 0.5, -0.25, 0.75, 0.125, -0.5, 0.25, 1.0, -1.0, 0.375, 0.625, -0.75, 0.5 }, { 3, 4 },
		                DataType::Float32),
		    Tensor<CPU>({ 0.1, -0.2, 0.3, -0.4 }, { 1, 4 }, DataType::Float32));
		const auto h2 = Layer::CreateLinear(
		    graph, Tensor<CPU>({ 0.25, -0.5, 0.75, 0.5, 0.125, -0.25, -0.375, 0.625 }, { 4, 2 }, DataType::Float32),
		    Tensor<CPU>({ 0.05, -0.15 }, { 1, 2 }, DataType::Float32));

		Subgraph sg;
		const auto input = sg.AddParam(DataType::Float32, { batch, 3 });
		const auto hidden = Layer::AddReLU(sg, Layer::AddLinear(sg, h1, { input, 0 }));
		sg.SetResults({ Layer::AddLinear(sg, h2, hidden) });
		graph.SetForward(graph.AddSubgraph(std::move(sg)));
		graph.SetInputNames({ "input" });
		graph.SetOutputNames({ "logits" });
		return graph;
	}

	std::vector<double> MakePatternValues(std::size_t count, double scale)
	{
		std::vector<double> values(count);
		for (std::size_t i = 0; i < count; ++i)
		{
			const auto centered = static_cast<int>(i % 17) - 8;
			values[i] = static_cast<double>(centered) * scale;
		}
		return values;
	}

	Graph BuildWideLinearChainGraph(std::size_t batch)
	{
		constexpr std::size_t kInput = 64;
		constexpr std::size_t kHidden = 64;
		constexpr std::size_t kOutput = 32;

		Graph graph;
		auto w1 = MakePatternValues(kInput * kHidden, 0.005);
		auto b1 = MakePatternValues(kHidden, 0.001);
		auto w2 = MakePatternValues(kHidden * kOutput, 0.004);
		auto b2 = MakePatternValues(kOutput, 0.001);
		const auto h1 =
		    Layer::CreateLinear(graph, Tensor<CPU>(std::span<const double>(w1), { kInput, kHidden }, DataType::Float32),
		                        Tensor<CPU>(std::span<const double>(b1), { 1, kHidden }, DataType::Float32));
		const auto h2 = Layer::CreateLinear(
		    graph, Tensor<CPU>(std::span<const double>(w2), { kHidden, kOutput }, DataType::Float32),
		    Tensor<CPU>(std::span<const double>(b2), { 1, kOutput }, DataType::Float32));

		Subgraph sg;
		const auto input = sg.AddParam(DataType::Float32, { batch, kInput });
		const auto hidden = Layer::AddReLU(sg, Layer::AddLinear(sg, h1, { input, 0 }));
		sg.SetResults({ Layer::AddLinear(sg, h2, hidden) });
		graph.SetForward(graph.AddSubgraph(std::move(sg)));
		graph.SetInputNames({ "input" });
		graph.SetOutputNames({ "logits" });
		return graph;
	}

	class ScopedEnvVar
	{
	public:
		ScopedEnvVar(const char* name, const char* value) : name_(name)
		{
			if (const char* oldValue = std::getenv(name))
			{
				oldValue_ = oldValue;
			}
			Set(value);
		}

		~ScopedEnvVar()
		{
			if (oldValue_.empty())
			{
				Unset();
			}
			else
			{
				Set(oldValue_.c_str());
			}
		}

		ScopedEnvVar(const ScopedEnvVar&) = delete;
		ScopedEnvVar& operator=(const ScopedEnvVar&) = delete;

	private:
		void Set(const char* value) const
		{
#ifdef _WIN32
			_putenv_s(name_, value);
#else
			setenv(name_, value, 1);
#endif
		}

		void Unset() const
		{
#ifdef _WIN32
			_putenv_s(name_, "");
#else
			unsetenv(name_);
#endif
		}

		const char* name_{};
		std::string oldValue_;
	};

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
						return { false,
							     std::format("worker {} output {} mismatch at iteration {}", workerId, i, iteration) };
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
	const std::vector<std::size_t> expectedShape{ 2, 2 };
	ASSERT_EQ(outputs[0].Shape(), ShapeView{ expectedShape });
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

TEST(CompiledModuleTest, LoadsSeparatedArtifactFromIndependentRegions)
{
	auto graph = BuildSimpleAddGraph();
	auto artifact = Compiler<CPU>::CompileArtifact(graph);
	auto separated = artifact.SeparateRodata();

	ASSERT_GT(separated.Metadata().size(), 0u);
	ASSERT_EQ(separated.Constants().size(), 0u);
	ASSERT_EQ(separated.Weights().size(), 0u);
	ASSERT_GT(separated.Instructions().size(), 0u);
	ASSERT_EQ(separated.InputSpecs().size(), 2u);
	ASSERT_EQ(separated.OutputSpecs().size(), 1u);
	EXPECT_EQ(separated.FindInput("lhs"), 0u);
	EXPECT_EQ(separated.FindOutput("sum"), 0u);

	const auto regionInfos = separated.RegionInfos();
	ASSERT_NE(FindRegionInfo(regionInfos, "metadata"), nullptr);
	ASSERT_NE(FindRegionInfo(regionInfos, "constants"), nullptr);
	ASSERT_NE(FindRegionInfo(regionInfos, "weights"), nullptr);
	ASSERT_NE(FindRegionInfo(regionInfos, "instructions"), nullptr);
	EXPECT_EQ(FindRegionInfo(regionInfos, "constants")->size, 0u);
	EXPECT_EQ(FindRegionInfo(regionInfos, "weights")->size, 0u);
	EXPECT_EQ(FindRegionInfo(regionInfos, "instructions")->size, separated.Instructions().size());

	auto loaded = CompiledModule<CPU>::Load(separated.Image());
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

TEST(CompiledModuleTest, LoadsSeparatedArtifactFromExportedSymbolAddresses)
{
	auto graph = BuildSimpleAddGraph();
	auto separated = Compiler<CPU>::CompileArtifact(graph).SeparateRodata();

	const std::uint64_t metadataSize = separated.Metadata().size();
	const std::uint64_t constantsSize = separated.Constants().size();
	const std::uint64_t weightsSize = separated.Weights().size();
	const std::uint64_t instructionSize = separated.Instructions().size();
	auto exported = CompiledModuleSeparatedArtifact::FromExportedSymbols({
	    .metadata = separated.Metadata().data(),
	    .metadataSize = &metadataSize,
	    .constants = separated.Constants().data(),
	    .constantsSize = &constantsSize,
	    .weights = separated.Weights().data(),
	    .weightsSize = &weightsSize,
	    .instructions = separated.Instructions().data(),
	    .instructionsSize = &instructionSize,
	});

	EXPECT_EQ(exported.Backend(), CompiledModuleBackend::CPUNative);
	ASSERT_EQ(exported.InputSpecs().size(), 2u);
	EXPECT_EQ(exported.FindInput("rhs"), 1u);
	auto loaded = exported.Load();
	Tensor<CPU> a({ 3, 4, 5, 6 }, { 2, 2 }, DataType::Float32);
	Tensor<CPU> b({ 7, 8, 9, 10 }, { 2, 2 }, DataType::Float32);
	std::array<Tensor<CPU>, 2> inputs = { std::move(a), std::move(b) };
	auto outputs = loaded.Run(inputs);
	ASSERT_EQ(outputs.size(), 1u);
	EXPECT_FLOAT_EQ(ReadFloat(outputs[0], 0), 10.0f);
	EXPECT_FLOAT_EQ(ReadFloat(outputs[0], 1), 12.0f);
	EXPECT_FLOAT_EQ(ReadFloat(outputs[0], 2), 14.0f);
	EXPECT_FLOAT_EQ(ReadFloat(outputs[0], 3), 16.0f);
}

TEST(CompiledModuleTest, SeparatedArtifactValidatesRegionCompatibility)
{
	auto graph = BuildSimpleAddGraph();
	auto separated = Compiler<CPU>::CompileArtifact(graph).SeparateRodata();

	EXPECT_NO_THROW((void) separated.WithReboundConstants({
	    .data = separated.Constants().data(),
	    .size = separated.Constants().size(),
	}));
	EXPECT_NO_THROW((void) separated.WithReboundWeights({
	    .data = separated.Weights().data(),
	    .size = separated.Weights().size(),
	}));

	std::array<std::byte, 1> wrongSize{ std::byte{ 1 } };
	EXPECT_THROW((void) separated.WithReboundWeights({ .data = wrongSize.data(), .size = wrongSize.size() }),
	             std::runtime_error);

	std::vector<std::byte> instructions(separated.Instructions().begin(), separated.Instructions().end());
	ASSERT_FALSE(instructions.empty());
	instructions[0] ^= std::byte{ 0xff };
	auto image = separated.Image();
	image.instructions = { .data = instructions.data(), .size = instructions.size() };
	try
	{
		(void) CompiledModuleSeparatedArtifact::CopyFromImage(image);
		FAIL() << "expected separated instruction checksum validation to throw";
	}
	catch (const std::runtime_error& ex)
	{
		const std::string message = ex.what();
		EXPECT_NE(message.find("checksum"), std::string::npos);
	}
}

TEST(CompiledModuleTest, WritesSeparatedCarrierObjects)
{
	auto graph = BuildSimpleAddGraph();
	auto separated = Compiler<CPU>::CompileArtifact(graph).SeparateRodata();

	const auto root = std::filesystem::temp_directory_path() / "litenn_separated_compiled_module_test";
	std::filesystem::remove_all(root);
	std::filesystem::create_directories(root);

	const auto combinedPath = root / "combined.o";
	separated.WriteObjectFile(combinedPath, "litenn_sep_test");
	ASSERT_TRUE(std::filesystem::exists(combinedPath));
	EXPECT_GT(std::filesystem::file_size(combinedPath), 0u);

	const auto splitDir = root / "split";
	separated.WriteObjectFiles(splitDir, "litenn_sep_test");
	for (const auto& name : { "litenn_sep_test_metadata.o", "litenn_sep_test_constants.o",
		                      "litenn_sep_test_weights.o", "litenn_sep_test_instructions.o" })
	{
		const auto path = splitDir / name;
		ASSERT_TRUE(std::filesystem::exists(path)) << path.string();
		EXPECT_GT(std::filesystem::file_size(path), 0u);
	}

	const auto rawDir = root / "raw";
	separated.WriteRegionFiles(rawDir, "litenn_sep_test");
	auto metadata = ReadFileBytes(rawDir / "litenn_sep_test.metadata.bin");
	auto constants = ReadFileBytes(rawDir / "litenn_sep_test.constants.bin");
	auto weights = ReadFileBytes(rawDir / "litenn_sep_test.weights.bin");
	auto instructions = ReadFileBytes(rawDir / "litenn_sep_test.instructions.bin");
	auto fileBacked = CompiledModuleSeparatedArtifact::CopyFromImage({
	    .metadata = { .data = metadata.data(), .size = metadata.size() },
	    .constants = { .data = constants.data(), .size = constants.size() },
	    .weights = { .data = weights.data(), .size = weights.size() },
	    .instructions = { .data = instructions.data(), .size = instructions.size() },
	});
	EXPECT_EQ(fileBacked.Backend(), CompiledModuleBackend::CPUNative);

	std::filesystem::remove_all(root);
}

TEST(CompiledModuleTest, CPUGetRowsArtifactMatchesInterpreter)
{
	auto graph = BuildGetRowsGraph();
	std::array<Tensor<CPU>, 1> inputs = { Tensor<CPU>({ 2, 0, 3 }, { 3 }, DataType::Int32) };

	Runtime::Interpreter<CPU> interpreter;
	const auto expected = interpreter.RunForward(graph, std::span<const Tensor<CPU>>(inputs));
	auto artifact = Compiler<CPU>::CompileArtifact(graph);
	auto loaded = artifact.Load();
	const auto outputs = loaded.Run(inputs);

	ASSERT_EQ(expected.size(), 1u);
	ASSERT_EQ(outputs.size(), 1u);
	ASSERT_EQ(outputs[0].NumElements(), expected[0].NumElements());
	for (std::size_t i = 0; i < outputs[0].NumElements(); ++i)
	{
		EXPECT_NEAR(ReadFloat(outputs[0], i), ReadFloat(expected[0], i), 1e-5f);
	}
}

TEST(CompiledModuleTest, CPUDataMovementSoftmaxArtifactMatchesInterpreter)
{
	Graph graph;
	Subgraph sg;
	const auto x = sg.AddParam(DataType::Float32, { 1, 2, 3 });
	const auto broadcast =
	    sg.AddNode(BroadcastToNode{ { x, 0 }, { 4, 2, 3 } }, { OutputInfo{ DataType::Float32, { 4, 2, 3 } } });
	const auto permute =
	    sg.AddNode(PermuteNode{ { broadcast, 0 }, { 1, 0, 2 } }, { OutputInfo{ DataType::Float32, { 2, 4, 3 } } });
	const auto softmax = sg.AddNode(SoftmaxNode{ { permute, 0 }, 2 }, { OutputInfo{ DataType::Float32, { 2, 4, 3 } } });
	sg.SetResults({ { softmax, 0 } });
	graph.SetForward(graph.AddSubgraph(std::move(sg)));
	graph.SetInputNames({ "logits" });
	graph.SetOutputNames({ "probabilities" });

	const std::vector<double> inputData = {
		1.0, -2.0, 0.5, 3.0, 0.25, -1.0,
	};
	std::array<Tensor<CPU>, 1> inputs = { Tensor<CPU>(std::span<const double>(inputData), { 1, 2, 3 },
		                                              DataType::Float32) };

	Runtime::Interpreter<CPU> interpreter;
	const auto expected = interpreter.RunForward(graph, std::span<const Tensor<CPU>>(inputs));
	auto compiled = Compiler<CPU>::Compile(graph);
	const auto outputs = compiled.Run(std::span<const Tensor<CPU>>(inputs));

	ASSERT_EQ(outputs.size(), 1u);
	ASSERT_EQ(expected.size(), 1u);
	ASSERT_EQ(outputs[0].Shape(), expected[0].Shape());
	for (std::size_t i = 0; i < outputs[0].NumElements(); ++i)
	{
		EXPECT_NEAR(ReadFloat(outputs[0], i), ReadFloat(expected[0], i), 1e-5f);
	}
}

TEST(CompiledModuleTest, CPUBatchMatMulArtifactMatchesInterpreter)
{
	Graph graph;
	Subgraph sg;
	const auto lhs = sg.AddParam(DataType::Float32, { 2, 1, 2, 3 });
	const auto rhs = sg.AddParam(DataType::Float32, { 1, 4, 3, 2 });
	const auto out = sg.AddNode(BatchMatMulNode{ { lhs, 0 }, { rhs, 0 } },
	                            { OutputInfo{ DataType::Float32, { 2, 4, 2, 2 } } });
	sg.SetResults({ { out, 0 } });
	graph.SetForward(graph.AddSubgraph(std::move(sg)));

	const std::vector<double> lhsData = {
		1.0,  -2.0, 0.5,  3.0,  0.25, -1.0,
		0.75, 1.5,  -0.5, 2.0,  -1.25, 0.125,
	};
	const std::vector<double> rhsData = {
		0.5,  -1.0,  1.5, 0.25, -0.75, 2.0,
		1.0,  0.5,   -1.0, 1.25, 0.75,  -0.25,
		-0.5, 2.0,   0.25, -1.5, 1.0,   0.75,
		1.25, -0.75, 0.5, 0.5,  -1.25, 1.5,
	};
	std::array<Tensor<CPU>, 2> inputs = {
		Tensor<CPU>(std::span<const double>(lhsData), { 2, 1, 2, 3 }, DataType::Float32),
		Tensor<CPU>(std::span<const double>(rhsData), { 1, 4, 3, 2 }, DataType::Float32),
	};

	ExpectCompiledMatchesInterpreter(graph, std::span<const Tensor<CPU>>(inputs), 1e-5f);
}

TEST(CompiledModuleTest, CPUConv2DArtifactMatchesInterpreter)
{
	Graph graph;
	std::vector<double> weightData;
	weightData.reserve(36);
	for (std::size_t i = 0; i < 36; ++i)
	{
		weightData.push_back((static_cast<double>(i % 7) - 3.0) * 0.125);
	}
	const auto weightIndex = graph.AddVariable(
	    Variable::Create(Tensor<CPU>(std::span<const double>(weightData), { 4, 1, 3, 3 }, DataType::Float32)));
	const auto biasIndex =
	    graph.AddVariable(Variable::Create(Tensor<CPU>({ 0.1, -0.2, 0.3, -0.4 }, { 4 }, DataType::Float32)));

	Subgraph sg;
	const auto input = sg.AddParam(DataType::Float32, { 1, 2, 3, 3 });
	const auto weight =
	    sg.AddNode(VariableRefNode{ weightIndex }, { OutputInfo{ DataType::Float32, { 4, 1, 3, 3 } } });
	const auto bias = sg.AddNode(VariableRefNode{ biasIndex }, { OutputInfo{ DataType::Float32, { 4 } } });
	const auto conv = Layer::AddConv2D(sg, { input, 0 }, { weight, 0 }, NodeOutput{ bias, 0 },
	                                   { 1, 1 }, { 1, 1 }, { 1, 1 }, { 1, 1 }, 2);
	sg.SetResults({ conv });
	graph.SetForward(graph.AddSubgraph(std::move(sg)));
	graph.SetInputNames({ "input" });
	graph.SetOutputNames({ "conv" });

	std::array<Tensor<CPU>, 1> inputs = {
		Tensor<CPU>({ 1.0, -2.0, 0.5, 3.0, 0.25, -1.0, 0.75, 1.5, -0.5,
		              2.0, -1.25, 0.125, 0.5, -0.75, 1.25, -1.5, 0.875, 0.25 },
		            { 1, 2, 3, 3 }, DataType::Float32),
	};

	ExpectCompiledMatchesInterpreter(graph, std::span<const Tensor<CPU>>(inputs), 1e-5f);
}

TEST(CompiledModuleTest, CPUNearestUpsampleArtifactMatchesInterpreter)
{
	Graph graph;
	Subgraph sg;
	const auto input = sg.AddParam(DataType::Float32, { 1, 1, 2, 3 });
	const auto upsample = Layer::AddNearestUpsample2D(sg, { input, 0 }, { 4, 6 });
	sg.SetResults({ upsample });
	graph.SetForward(graph.AddSubgraph(std::move(sg)));
	graph.SetInputNames({ "input" });
	graph.SetOutputNames({ "upsampled" });

	std::array<Tensor<CPU>, 1> inputs = {
		Tensor<CPU>({ 1.0, 2.0, 3.0, 4.0, 5.0, 6.0 }, { 1, 1, 2, 3 }, DataType::Float32),
	};

	ExpectCompiledMatchesInterpreter(graph, std::span<const Tensor<CPU>>(inputs), 1e-5f);
}

TEST(CompiledModuleTest, CPUTorchStyleGroupNormArtifactMatchesInterpreter)
{
	Graph graph;
	const auto scaleIndex =
	    graph.AddVariable(Variable::Create(Tensor<CPU>({ 1.5, -0.75 }, { 1, 2, 1, 1 }, DataType::Float32)));
	const auto biasIndex =
	    graph.AddVariable(Variable::Create(Tensor<CPU>({ 0.25, -0.5 }, { 1, 2, 1, 1 }, DataType::Float32)));

	Subgraph sg;
	const auto input = sg.AddParam(DataType::Float32, { 1, 2, 2, 2 });
	const auto grouped = Layer::AddReshape(sg, { input, 0 }, { 1, 2, 4 });
	const auto normalized = Layer::AddNormalization(sg, grouped, NormalizationMode::LayerNorm, 2, 1e-5);
	const auto nchw = Layer::AddReshape(sg, normalized, { 1, 2, 2, 2 });
	const auto scale =
	    sg.AddNode(VariableRefNode{ scaleIndex }, { OutputInfo{ DataType::Float32, { 1, 2, 1, 1 } } });
	const auto bias =
	    sg.AddNode(VariableRefNode{ biasIndex }, { OutputInfo{ DataType::Float32, { 1, 2, 1, 1 } } });
	const auto scaled = sg.AddNode(BinaryOpNode{ BinaryOp::Multiply, nchw, { scale, 0 } },
	                               { OutputInfo{ DataType::Float32, { 1, 2, 2, 2 } } });
	const auto shifted = sg.AddNode(BinaryOpNode{ BinaryOp::Add, { scaled, 0 }, { bias, 0 } },
	                                { OutputInfo{ DataType::Float32, { 1, 2, 2, 2 } } });
	sg.SetResults({ { shifted, 0 } });
	graph.SetForward(graph.AddSubgraph(std::move(sg)));

	std::array<Tensor<CPU>, 1> inputs = {
		Tensor<CPU>({ 1.0, -2.0, 0.5, 3.0, 0.25, -1.0, 0.75, 1.5 },
		            { 1, 2, 2, 2 }, DataType::Float32),
	};

	ExpectCompiledMatchesInterpreter(graph, std::span<const Tensor<CPU>>(inputs), 1e-5f);
}

TEST(CompiledModuleTest, CPUTimestepEmbeddingArtifactMatchesInterpreter)
{
	Graph graph;
	Subgraph sg;
	const auto timesteps = sg.AddParam(DataType::Float32, { 3 });
	const auto embedding = Layer::AddTimestepEmbedding(sg, { timesteps, 0 }, 5, 1000);
	sg.SetResults({ embedding });
	graph.SetForward(graph.AddSubgraph(std::move(sg)));
	graph.SetInputNames({ "timesteps" });
	graph.SetOutputNames({ "embedding" });

	std::array<Tensor<CPU>, 1> inputs = {
		Tensor<CPU>({ 0.0, 1.0, 999.0 }, { 3 }, DataType::Float32),
	};

	ExpectCompiledMatchesInterpreter(graph, std::span<const Tensor<CPU>>(inputs), 1e-5f);
}

TEST(CompiledModuleTest, CPUGatherPadArtifactMatchesInterpreter)
{
	Graph graph;
	Subgraph sg;
	const auto data = sg.AddParam(DataType::Float32, { 2, 3 });
	const auto indices = sg.AddParam(DataType::Int32, { 2 });
	const auto gathered =
	    sg.AddNode(GatherNode{ { data, 0 }, { indices, 0 }, 1 }, { OutputInfo{ DataType::Float32, { 2, 2 } } });
	const auto constantPad =
	    sg.AddNode(PadNode{ { gathered, 0 }, { 1, 1 }, { 0, 1 }, PadMode::Constant, -5.0 },
	               { OutputInfo{ DataType::Float32, { 3, 4 } } });
	const auto reflectPad =
	    sg.AddNode(PadNode{ { data, 0 }, { 1, 2 }, { 1, 1 }, PadMode::Reflect, 0.0 },
	               { OutputInfo{ DataType::Float32, { 4, 6 } } });
	const auto replicatePad =
	    sg.AddNode(PadNode{ { data, 0 }, { 1, 1 }, { 1, 2 }, PadMode::Replicate, 0.0 },
	               { OutputInfo{ DataType::Float32, { 4, 6 } } });
	sg.SetResults({ { constantPad, 0 }, { reflectPad, 0 }, { replicatePad, 0 } });
	graph.SetForward(graph.AddSubgraph(std::move(sg)));

	std::array<Tensor<CPU>, 2> inputs = {
		Tensor<CPU>({ 1.0, 2.0, 3.0, 4.0, 5.0, 6.0 }, { 2, 3 }, DataType::Float32),
		Tensor<CPU>({ 2, 0 }, { 2 }, DataType::Int32),
	};

	ExpectCompiledMatchesInterpreter(graph, std::span<const Tensor<CPU>>(inputs), 1e-5f);
}

TEST(CompiledModuleTest, CPUCrossEntropyArtifactMatchesInterpreter)
{
	Graph graph;
	Subgraph sg;
	const auto logits = sg.AddParam(DataType::Float32, { 2, 3 });
	const auto labels = sg.AddParam(DataType::Float32, { 2, 3 });
	const auto loss = sg.AddNode(CrossEntropyLossNode{ { logits, 0 }, { labels, 0 } },
	                             { OutputInfo{ DataType::Float32, { 1 } } });
	sg.SetResults({ { loss, 0 } });
	graph.SetForward(graph.AddSubgraph(std::move(sg)));

	std::array<Tensor<CPU>, 2> inputs = {
		Tensor<CPU>({ 1.25, -0.5, 0.25, -1.0, 2.0, 0.5 }, { 2, 3 }, DataType::Float32),
		Tensor<CPU>({ 0.7, 0.2, 0.1, 0.0, 1.0, 0.0 }, { 2, 3 }, DataType::Float32),
	};

	ExpectCompiledMatchesInterpreter(graph, std::span<const Tensor<CPU>>(inputs), 1e-5f);
}

TEST(CompiledModuleTest, CPURankOneSoftmaxCrossEntropyArtifactMatchesInterpreter)
{
	Graph graph;
	Subgraph sg;
	const auto logits = sg.AddParam(DataType::Float32, { 3 });
	const auto labels = sg.AddParam(DataType::Float32, { 3 });
	const auto probabilities =
	    sg.AddNode(SoftmaxNode{ { logits, 0 }, 0 }, { OutputInfo{ DataType::Float32, { 3 } } });
	const auto loss = sg.AddNode(CrossEntropyLossNode{ { logits, 0 }, { labels, 0 } },
	                             { OutputInfo{ DataType::Float32, { 1 } } });
	sg.SetResults({ { probabilities, 0 }, { loss, 0 } });
	graph.SetForward(graph.AddSubgraph(std::move(sg)));

	std::array<Tensor<CPU>, 2> inputs = {
		Tensor<CPU>({ 1.25, -0.5, 0.25 }, { 3 }, DataType::Float32),
		Tensor<CPU>({ 0.7, 0.2, 0.1 }, { 3 }, DataType::Float32),
	};

	ExpectCompiledMatchesInterpreter(graph, std::span<const Tensor<CPU>>(inputs), 1e-5f);
}

TEST(CompiledModuleTest, CPUCrossEntropyBackwardArtifactMatchesInterpreter)
{
	Graph graph;
	Subgraph sg;
	const auto grad = sg.AddParam(DataType::Float32, { 1 });
	const auto logits = sg.AddParam(DataType::Float32, { 2, 3 });
	const auto labels = sg.AddParam(DataType::Float32, { 2, 3 });
	const auto dx = sg.AddNode(CrossEntropyLossBackwardNode{ { grad, 0 }, { logits, 0 }, { labels, 0 } },
	                           { OutputInfo{ DataType::Float32, { 2, 3 } } });
	sg.SetResults({ { dx, 0 } });
	graph.SetForward(graph.AddSubgraph(std::move(sg)));

	std::array<Tensor<CPU>, 3> inputs = {
		Tensor<CPU>({ 1.5 }, { 1 }, DataType::Float32),
		Tensor<CPU>({ 1.25, -0.5, 0.25, -1.0, 2.0, 0.5 }, { 2, 3 }, DataType::Float32),
		Tensor<CPU>({ 0.7, 0.2, 0.1, 0.0, 1.0, 0.0 }, { 2, 3 }, DataType::Float32),
	};

	ExpectCompiledMatchesInterpreter(graph, std::span<const Tensor<CPU>>(inputs), 1e-5f);
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

TEST(CompiledModuleTest, PreservesQuantizationMetadataInCompiledSignatures)
{
	auto graph = BuildQuantizedConstantOutputGraph();
	auto artifact = Compiler<CPU>::CompileArtifact(graph);

	ASSERT_EQ(artifact.OutputSpecs().size(), 1u);
	const auto& spec = artifact.OutputSpecs()[0];
	EXPECT_EQ(spec.name, "quantized_weight");
	EXPECT_EQ(spec.dtype, DataType::Int8);
	EXPECT_EQ(spec.shape, (std::vector<std::size_t>{ 2, 2 }));
	ASSERT_TRUE(spec.quantization.has_value());
	EXPECT_EQ(spec.quantization->scheme, QuantizationScheme::Affine);
	EXPECT_EQ(spec.quantization->storageType, DataType::Int8);
	EXPECT_EQ(spec.quantization->expressedType, DataType::Float32);
	ASSERT_EQ(spec.quantization->scales.size(), 1u);
	EXPECT_FLOAT_EQ(spec.quantization->scales[0], 0.25F);
	ASSERT_EQ(spec.quantization->zeroPoints.size(), 1u);
	EXPECT_EQ(spec.quantization->zeroPoints[0], -3);

	auto copied = CompiledModuleArtifact::CopyFromImage(artifact.Image());
	ASSERT_EQ(copied.OutputSpecs().size(), 1u);
	ASSERT_TRUE(copied.OutputSpecs()[0].quantization.has_value());
	EXPECT_EQ(copied.OutputSpecs()[0].quantization->zeroPoints[0], -3);

	auto loaded = CompiledModule<CPU>::Load(artifact.Image());
	ASSERT_EQ(loaded.OutputSpecs().size(), 1u);
	ASSERT_TRUE(loaded.OutputSpecs()[0].quantization.has_value());
	EXPECT_EQ(loaded.OutputSpecs()[0].quantization->storageType, DataType::Int8);

	const auto dump = Debug::DumpCompiledModuleMetadata(artifact);
	EXPECT_NE(dump.find("quant=Affine"), std::string::npos);
	EXPECT_NE(dump.find("expressed=Float32[2, 2]"), std::string::npos);
}

TEST(CompiledModuleTest, CUDANativeInstructionPayloadRoundTripsLaunchMetadata)
{
	const auto binary = std::vector<std::byte>{ std::byte{ 'p' }, std::byte{ 't' }, std::byte{ 'x' } };
	const auto payload = CUDANativeInstructionPayload{
	    .binaryKind = CUDANativeBinaryKind::PTX,
	    .featureSet = CUDANativeFeatureSet(CUDANativeFeature::StaticShape, CUDANativeFeature::SingleSubgraph),
	    .target = "compute_75",
	    .binary = binary,
	    .scalarData = { std::byte{ 4 }, std::byte{ 0 }, std::byte{ 0 }, std::byte{ 0 } },
	    .workspaceBytes = 256,
	    .kernels = {
	        CUDANativeKernelSpec{
	            .name = "litenn_kernel_0",
	            .grid = { .x = 8, .y = 2, .z = 1 },
	            .block = { .x = 128, .y = 1, .z = 1 },
	            .sharedMemoryBytes = 64,
	            .workspaceBytes = 128,
	            .arguments = {
	                CUDANativeArgumentSpec{
	                    .kind = CUDANativeArgumentKind::InputTensor,
	                    .index = 0,
	                    .byteOffset = 0,
	                    .byteSize = 8,
	                },
	                CUDANativeArgumentSpec{
	                    .kind = CUDANativeArgumentKind::OutputTensor,
	                    .index = 0,
	                    .byteOffset = 8,
	                    .byteSize = 8,
	                },
	                CUDANativeArgumentSpec{
	                    .kind = CUDANativeArgumentKind::Workspace,
	                    .index = 0,
	                    .byteOffset = 16,
	                    .byteSize = 128,
	                },
	                CUDANativeArgumentSpec{
	                    .kind = CUDANativeArgumentKind::Scalar,
	                    .index = 0,
	                    .byteOffset = 0,
	                    .byteSize = 4,
	                },
	            },
	        },
	    },
	};

	auto bytes = SerializeCUDANativeInstructionPayload(payload);
	auto decoded = DeserializeCUDANativeInstructionPayload(bytes);

	EXPECT_EQ(decoded.binaryKind, CUDANativeBinaryKind::PTX);
	EXPECT_EQ(decoded.featureSet,
	          CUDANativeFeatureSet(CUDANativeFeature::StaticShape, CUDANativeFeature::SingleSubgraph));
	EXPECT_EQ(decoded.target, "compute_75");
	EXPECT_EQ(decoded.binary, binary);
	ASSERT_EQ(decoded.scalarData.size(), 4u);
	EXPECT_EQ(decoded.scalarData[0], std::byte{ 4 });
	ASSERT_EQ(decoded.kernels.size(), 1u);
	EXPECT_EQ(decoded.kernels[0].name, "litenn_kernel_0");
	EXPECT_EQ(decoded.kernels[0].grid.x, 8u);
	EXPECT_EQ(decoded.kernels[0].grid.y, 2u);
	EXPECT_EQ(decoded.kernels[0].block.x, 128u);
	EXPECT_EQ(decoded.kernels[0].sharedMemoryBytes, 64u);
	ASSERT_EQ(decoded.kernels[0].arguments.size(), 4u);
	EXPECT_EQ(decoded.kernels[0].arguments[1].kind, CUDANativeArgumentKind::OutputTensor);
	EXPECT_EQ(decoded.kernels[0].arguments[2].byteSize, 128u);
	EXPECT_EQ(decoded.kernels[0].arguments[3].kind, CUDANativeArgumentKind::Scalar);
}

#ifdef LITENN_ENABLE_CUDA
TEST(CompiledModuleTest, CUDANativeCodegenBuildsStablePTXPayloadBytes)
{
	const std::array<std::size_t, 2> outputShape{ 2, 3 };
	const std::array<std::size_t, 2> lhsShape{ 2, 1 };
	const std::array<std::size_t, 2> rhsShape{ 1, 3 };

	EXPECT_EQ(CUDANativeUnaryF32KernelName(UnaryOp::Sqrt), "litenn_sqrt_f32");
	EXPECT_EQ(CUDANativeBinaryF32KernelName(BinaryOp::Subtract), "litenn_subtract_f32");
	EXPECT_EQ(CUDANativeBinaryF32KernelName(BinaryOp::Subtract, true), "litenn_subtract_broadcast_f32");

	const auto ptx = CUDANativeBinaryBroadcastF32PTXFromMLIRNVPTX(CUDANativeBroadcastBinaryF32CodegenSpec{
	    .op = BinaryOp::Subtract,
	    .outputShape = std::span<const std::size_t>{ outputShape },
	    .lhsShape = std::span<const std::size_t>{ lhsShape },
	    .rhsShape = std::span<const std::size_t>{ rhsShape },
	});
	EXPECT_NE(ptx.find(".visible .entry litenn_subtract_broadcast_f32"), std::string::npos);
	EXPECT_NE(ptx.find("sub.rn.f32"), std::string::npos);
	EXPECT_NE(ptx.find("st.global"), std::string::npos);

	const auto bytes = CUDANativeTextBytes(ptx);
	ASSERT_EQ(bytes.size(), ptx.size() + 1);
	EXPECT_EQ(bytes.back(), std::byte{ 0 });
	for (std::size_t i = 0; i < ptx.size(); ++i)
	{
		EXPECT_EQ(bytes[i], static_cast<std::byte>(static_cast<unsigned char>(ptx[i])));
	}

	const auto payload = CUDANativeInstructionPayload{
	    .binaryKind = CUDANativeBinaryKind::PTX,
	    .featureSet = CUDANativeFeatureSet(CUDANativeFeature::StaticShape, CUDANativeFeature::SingleSubgraph, CUDANativeFeature::ElementwiseSubtractF32, CUDANativeFeature::ElementwiseBroadcastF32),
	    .target = CUDANativeNVPTXTargetChip(),
	    .binary = bytes,
	    .scalarData = { std::byte{ 6 }, std::byte{ 0 }, std::byte{ 0 }, std::byte{ 0 } },
	    .kernels = {
	        CUDANativeKernelSpec{
	            .name = std::string(CUDANativeBinaryF32KernelName(BinaryOp::Subtract, true)),
	            .grid = { .x = 1, .y = 1, .z = 1 },
	            .block = { .x = 6, .y = 1, .z = 1 },
	            .arguments = {
	                CUDANativeArgumentSpec{
	                    .kind = CUDANativeArgumentKind::OutputTensor,
	                    .index = 0,
	                    .byteOffset = 0,
	                    .byteSize = 24,
	                },
	                CUDANativeArgumentSpec{
	                    .kind = CUDANativeArgumentKind::InputTensor,
	                    .index = 0,
	                    .byteOffset = 0,
	                    .byteSize = 8,
	                },
	                CUDANativeArgumentSpec{
	                    .kind = CUDANativeArgumentKind::InputTensor,
	                    .index = 1,
	                    .byteOffset = 0,
	                    .byteSize = 12,
	                },
	                CUDANativeArgumentSpec{
	                    .kind = CUDANativeArgumentKind::Scalar,
	                    .index = 0,
	                    .byteOffset = 0,
	                    .byteSize = sizeof(std::uint32_t),
	                },
	            },
	        },
	    },
	};

	const auto decoded = DeserializeCUDANativeInstructionPayload(SerializeCUDANativeInstructionPayload(payload));
	EXPECT_EQ(decoded.binaryKind, CUDANativeBinaryKind::PTX);
	EXPECT_EQ(decoded.target, CUDANativeNVPTXTargetChip());
	ASSERT_FALSE(decoded.binary.empty());
	EXPECT_EQ(decoded.binary.back(), std::byte{ 0 });
	EXPECT_TRUE(decoded.featureSet.HasFeature(CUDANativeFeature::ElementwiseBroadcastF32));
	ASSERT_EQ(decoded.kernels.size(), 1u);
	EXPECT_EQ(decoded.kernels[0].name, "litenn_subtract_broadcast_f32");
	ASSERT_EQ(decoded.kernels[0].arguments.size(), 4u);
	EXPECT_EQ(decoded.kernels[0].arguments[3].kind, CUDANativeArgumentKind::Scalar);
}

TEST(CompiledModuleTest, CUDANativeDefaultTargetUsesModernBaseline)
{
	ScopedEnvVar target("LITENN_CUDA_AOT_TARGET", "");
	EXPECT_EQ(CUDANativeNVPTXTargetChip(), "sm_75");
}

TEST(CompiledModuleTest, CUDANativeMLIRNVPTXGeneratesUnaryPTX)
{
	struct Case
	{
		UnaryOp op;
		const char* kernelName;
		const char* ptxNeedle;
		CUDANativeFeature featureFlag;
	};
	const std::array cases = {
		Case{ UnaryOp::Negate, "litenn_negate_f32", "neg.f32", CUDANativeFeature::ElementwiseNegateF32 },
		Case{ UnaryOp::Abs, "litenn_abs_f32", "abs.ftz.f32", CUDANativeFeature::ElementwiseAbsF32 },
		Case{ UnaryOp::Sqrt, "litenn_sqrt_f32", "sqrt.rn.ftz.f32", CUDANativeFeature::ElementwiseSqrtF32 },
		Case{ UnaryOp::Exp, "litenn_exp_f32", "ex2.approx.ftz.f32", CUDANativeFeature::ElementwiseExpF32 },
		Case{ UnaryOp::Log, "litenn_log_f32", "lg2.approx.ftz.f32", CUDANativeFeature::ElementwiseLogF32 },
		Case{ UnaryOp::Sin, "litenn_sin_f32", "sin.approx.ftz.f32", CUDANativeFeature::ElementwiseSinF32 },
		Case{ UnaryOp::Cos, "litenn_cos_f32", "cos.approx.ftz.f32", CUDANativeFeature::ElementwiseCosF32 },
	};

	for (const auto& testCase : cases)
	{
		SCOPED_TRACE(testCase.kernelName);
		const auto ptx = CUDANativeUnaryF32PTXFromMLIRNVPTX(testCase.op);

		EXPECT_NE(ptx.find("Generated by LLVM NVPTX Back-End"), std::string::npos);
		EXPECT_NE(ptx.find(std::format(".visible .entry {}", testCase.kernelName)), std::string::npos);
		EXPECT_NE(ptx.find("mov.u32"), std::string::npos);
		EXPECT_NE(ptx.find(testCase.ptxNeedle), std::string::npos);
		EXPECT_NE(ptx.find("st.global"), std::string::npos);

		const auto payload = CUDANativeInstructionPayload{
		    .binaryKind = CUDANativeBinaryKind::PTX,
		    .featureSet = CUDANativeFeatureSet(CUDANativeFeature::StaticShape, CUDANativeFeature::SingleSubgraph, testCase.featureFlag),
		    .target = "sm_30",
		    .binary = CUDANativeTextBytes(ptx),
		    .scalarData = { std::byte{ 4 }, std::byte{ 0 }, std::byte{ 0 }, std::byte{ 0 } },
		    .kernels = {
		        CUDANativeKernelSpec{
		            .name = std::string(CUDANativeUnaryF32KernelName(testCase.op)),
		            .grid = { .x = 1, .y = 1, .z = 1 },
		            .block = { .x = 4, .y = 1, .z = 1 },
		            .arguments = {
		                CUDANativeArgumentSpec{
		                    .kind = CUDANativeArgumentKind::OutputTensor,
		                    .index = 0,
		                    .byteOffset = 0,
		                    .byteSize = 16,
		                },
		                CUDANativeArgumentSpec{
		                    .kind = CUDANativeArgumentKind::InputTensor,
		                    .index = 0,
		                    .byteOffset = 0,
		                    .byteSize = 16,
		                },
		                CUDANativeArgumentSpec{
		                    .kind = CUDANativeArgumentKind::Scalar,
		                    .index = 0,
		                    .byteOffset = 0,
		                    .byteSize = sizeof(std::uint32_t),
		                },
		            },
		        },
		    },
		};

		const auto decoded = DeserializeCUDANativeInstructionPayload(SerializeCUDANativeInstructionPayload(payload));
		EXPECT_EQ(decoded.binaryKind, CUDANativeBinaryKind::PTX);
		EXPECT_EQ(decoded.binary.back(), std::byte{ 0 });
		EXPECT_TRUE(decoded.featureSet.HasFeature(testCase.featureFlag));
		ASSERT_EQ(decoded.kernels.size(), 1u);
		EXPECT_EQ(decoded.kernels[0].name, testCase.kernelName);
	}
}

TEST(CompiledModuleTest, CUDANativeMLIRNVPTXGeneratesSameShapeBinaryPTX)
{
	struct Case
	{
		BinaryOp op;
		const char* kernelName;
		const char* ptxNeedle;
		CUDANativeFeature featureFlag;
	};
	const std::array cases = {
		Case{ BinaryOp::Add, "litenn_add_f32", "add.rn.f32", CUDANativeFeature::ElementwiseAddF32 },
		Case{ BinaryOp::Subtract, "litenn_subtract_f32", "sub.rn.f32", CUDANativeFeature::ElementwiseSubtractF32 },
		Case{ BinaryOp::Multiply, "litenn_multiply_f32", "mul.rn.f32", CUDANativeFeature::ElementwiseMultiplyF32 },
		Case{ BinaryOp::Divide, "litenn_divide_f32", "div.rn.f32", CUDANativeFeature::ElementwiseDivideF32 },
		Case{ BinaryOp::Max, "litenn_max_f32", "max.ftz.f32", CUDANativeFeature::ElementwiseMaxF32 },
		Case{ BinaryOp::Min, "litenn_min_f32", "min.ftz.f32", CUDANativeFeature::ElementwiseMinF32 },
	};

	for (const auto& testCase : cases)
	{
		SCOPED_TRACE(testCase.kernelName);
		const auto ptx = CUDANativeBinaryF32PTXFromMLIRNVPTX(testCase.op);

		EXPECT_NE(ptx.find("Generated by LLVM NVPTX Back-End"), std::string::npos);
		EXPECT_NE(ptx.find(std::format(".visible .entry {}", testCase.kernelName)), std::string::npos);
		EXPECT_NE(ptx.find("mov.u32"), std::string::npos);
		EXPECT_NE(ptx.find(testCase.ptxNeedle), std::string::npos);
		EXPECT_NE(ptx.find("st.global"), std::string::npos);

		const auto payload = CUDANativeInstructionPayload{
		    .binaryKind = CUDANativeBinaryKind::PTX,
		    .featureSet = CUDANativeFeatureSet(CUDANativeFeature::StaticShape, CUDANativeFeature::SingleSubgraph, testCase.featureFlag),
		    .target = "sm_30",
		    .binary = CUDANativeTextBytes(ptx),
		    .scalarData = { std::byte{ 4 }, std::byte{ 0 }, std::byte{ 0 }, std::byte{ 0 } },
		    .kernels = {
		        CUDANativeKernelSpec{
		            .name = std::string(CUDANativeBinaryF32KernelName(testCase.op)),
		            .grid = { .x = 1, .y = 1, .z = 1 },
		            .block = { .x = 4, .y = 1, .z = 1 },
		            .arguments = {
		                CUDANativeArgumentSpec{
		                    .kind = CUDANativeArgumentKind::OutputTensor,
		                    .index = 0,
		                    .byteOffset = 0,
		                    .byteSize = 16,
		                },
		                CUDANativeArgumentSpec{
		                    .kind = CUDANativeArgumentKind::InputTensor,
		                    .index = 0,
		                    .byteOffset = 0,
		                    .byteSize = 16,
		                },
		                CUDANativeArgumentSpec{
		                    .kind = CUDANativeArgumentKind::InputTensor,
		                    .index = 1,
		                    .byteOffset = 0,
		                    .byteSize = 16,
		                },
		                CUDANativeArgumentSpec{
		                    .kind = CUDANativeArgumentKind::Scalar,
		                    .index = 0,
		                    .byteOffset = 0,
		                    .byteSize = sizeof(std::uint32_t),
		                },
		            },
		        },
		    },
		};

		const auto decoded = DeserializeCUDANativeInstructionPayload(SerializeCUDANativeInstructionPayload(payload));
		EXPECT_EQ(decoded.binaryKind, CUDANativeBinaryKind::PTX);
		EXPECT_EQ(decoded.binary.back(), std::byte{ 0 });
		EXPECT_TRUE(decoded.featureSet.HasFeature(testCase.featureFlag));
		ASSERT_EQ(decoded.kernels.size(), 1u);
		EXPECT_EQ(decoded.kernels[0].name, testCase.kernelName);
	}
}

TEST(CompiledModuleTest, CUDANativeMLIRNVPTXGeneratesBroadcastBinaryPTX)
{
	const std::array<std::size_t, 2> outputShape{ 2, 3 };
	const std::array<std::size_t, 2> lhsShape{ 2, 1 };
	const std::array<std::size_t, 2> rhsShape{ 1, 3 };
	const auto ptx = CUDANativeBinaryBroadcastF32PTXFromMLIRNVPTX(CUDANativeBroadcastBinaryF32CodegenSpec{
	    .op = BinaryOp::Subtract,
	    .outputShape = std::span<const std::size_t>{ outputShape },
	    .lhsShape = std::span<const std::size_t>{ lhsShape },
	    .rhsShape = std::span<const std::size_t>{ rhsShape },
	});

	EXPECT_NE(ptx.find("Generated by LLVM NVPTX Back-End"), std::string::npos);
	EXPECT_NE(ptx.find(".visible .entry litenn_subtract_broadcast_f32"), std::string::npos);
	EXPECT_NE(ptx.find("sub.rn.f32"), std::string::npos);
	EXPECT_NE(ptx.find("st.global"), std::string::npos);

	const auto maxPtx = CUDANativeBinaryBroadcastF32PTXFromMLIRNVPTX(CUDANativeBroadcastBinaryF32CodegenSpec{
	    .op = BinaryOp::Max,
	    .outputShape = std::span<const std::size_t>{ outputShape },
	    .lhsShape = std::span<const std::size_t>{ lhsShape },
	    .rhsShape = std::span<const std::size_t>{ rhsShape },
	});

	EXPECT_NE(maxPtx.find("Generated by LLVM NVPTX Back-End"), std::string::npos);
	EXPECT_NE(maxPtx.find(".visible .entry litenn_max_broadcast_f32"), std::string::npos);
	EXPECT_NE(maxPtx.find("max.ftz.f32"), std::string::npos);
	EXPECT_NE(maxPtx.find("st.global"), std::string::npos);
}
#endif

TEST(CompiledModuleTest, CUDANativeInstructionPayloadAllowsLibraryCallWithoutBinaryImage)
{
	const auto payload = CUDANativeInstructionPayload{
	    .binaryKind = CUDANativeBinaryKind::LibraryCall,
	    .featureSet = CUDANativeFeatureSet(CUDANativeFeature::StaticShape, CUDANativeFeature::SingleSubgraph, CUDANativeFeature::MatMulCUBLASF32),
	    .target = "cublas",
	    .binary = {},
	    .scalarData = { std::byte{ 2 }, std::byte{ 0 }, std::byte{ 0 }, std::byte{ 0 } },
	    .kernels = {
	        CUDANativeKernelSpec{
	            .name = "litenn_cublas_matmul_f32",
	            .grid = { .x = 1, .y = 1, .z = 1 },
	            .block = { .x = 1, .y = 1, .z = 1 },
	            .arguments = {
	                CUDANativeArgumentSpec{
	                    .kind = CUDANativeArgumentKind::OutputTensor,
	                    .index = 0,
	                    .byteOffset = 0,
	                    .byteSize = 16,
	                },
	                CUDANativeArgumentSpec{
	                    .kind = CUDANativeArgumentKind::InputTensor,
	                    .index = 0,
	                    .byteOffset = 0,
	                    .byteSize = 16,
	                },
	                CUDANativeArgumentSpec{
	                    .kind = CUDANativeArgumentKind::InputTensor,
	                    .index = 1,
	                    .byteOffset = 0,
	                    .byteSize = 16,
	                },
	            },
	        },
	    },
	};

	auto bytes = SerializeCUDANativeInstructionPayload(payload);
	auto decoded = DeserializeCUDANativeInstructionPayload(bytes);

	EXPECT_EQ(decoded.binaryKind, CUDANativeBinaryKind::LibraryCall);
	EXPECT_TRUE(decoded.binary.empty());
	EXPECT_EQ(decoded.target, "cublas");
	EXPECT_TRUE(decoded.featureSet.HasFeature(CUDANativeFeature::MatMulCUBLASF32));
	ASSERT_EQ(decoded.kernels.size(), 1u);
	EXPECT_EQ(decoded.kernels[0].name, "litenn_cublas_matmul_f32");
}

TEST(CompiledModuleTest, CUDANativeInstructionPayloadRejectsInvalidMagic)
{
	std::vector<std::byte> bytes = { std::byte{ 'b' }, std::byte{ 'a' }, std::byte{ 'd' }, std::byte{ 0 } };

	try
	{
		(void) DeserializeCUDANativeInstructionPayload(bytes);
		FAIL() << "expected CUDA native payload validation to throw";
	}
	catch (const std::runtime_error& ex)
	{
		const std::string message = ex.what();
		EXPECT_NE(message.find("magic"), std::string::npos);
	}
}

TEST(CompiledModuleTest, CUDANativeInstructionPayloadRejectsUnknownFeatureFlags)
{
	CUDANativeInstructionPayload payload;
	payload.target = "sm_30";
	payload.binary = { std::byte{ 'p' }, std::byte{ 't' }, std::byte{ 'x' } };
	payload.featureSet.flags = 1ull << 63;

	EXPECT_THROW((void) SerializeCUDANativeInstructionPayload(payload), std::runtime_error);
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
		(void) compiled.Run(inputs);
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
	std::array<Tensor<CPU>, 1> outputs = { Tensor<CPU>(Uninitialized, { 2, 2 }, DataType::Float32) };

	compiled.RunInto(inputs, outputs);

	EXPECT_FLOAT_EQ(ReadFloat(outputs[0], 0), 11.0f);
	EXPECT_FLOAT_EQ(ReadFloat(outputs[0], 1), 22.0f);
	EXPECT_FLOAT_EQ(ReadFloat(outputs[0], 2), 33.0f);
	EXPECT_FLOAT_EQ(ReadFloat(outputs[0], 3), 44.0f);
}

TEST(CompiledModuleTest, NarrowMatMulRowTileMatchesReference)
{
	Graph graph;
	const auto weightIndex = graph.AddVariable(Variable::Create(
	    Tensor<CPU>({ 1.0, -2.0, 0.5, 3.0, -1.0, 0.25, 4.0, -1.5, 2.0, 0.75, -3.0, 1.0, 2.5, -0.5, 1.25 }, { 3, 5 },
	                DataType::Float32)));

	Subgraph sg;
	const auto input = sg.AddParam(DataType::Float32, { 16, 3 });
	const auto weight = sg.AddNode(VariableRefNode{ weightIndex }, { OutputInfo{ DataType::Float32, { 3, 5 } } });
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
	std::array<Tensor<CPU>, 1> outputs = { Tensor<CPU>(Uninitialized, { 16, 5 }, DataType::Float32) };

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
	const auto weightIndex = graph.AddVariable(
	    Variable::Create(Tensor<CPU>(std::span<const double>(weightData), { 3, 256 }, DataType::Float32)));

	Subgraph sg;
	const auto input = sg.AddParam(DataType::Float32, { 8, 3 });
	const auto weight = sg.AddNode(VariableRefNode{ weightIndex }, { OutputInfo{ DataType::Float32, { 3, 256 } } });
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
	std::array<Tensor<CPU>, 1> outputs = { Tensor<CPU>(Uninitialized, { 8, 256 }, DataType::Float32) };

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

TEST(CompiledModuleTest, KPanelPackedWideMatMulMatchesReference)
{
	Graph graph;
	constexpr std::size_t batch = 8;
	constexpr std::size_t kSize = 128;
	constexpr std::size_t nSize = 128;
	std::vector<double> weightData(kSize * nSize);
	for (std::size_t k = 0; k < kSize; ++k)
	{
		for (std::size_t col = 0; col < nSize; ++col)
		{
			weightData[k * nSize + col] = static_cast<double>(static_cast<int>(k % 11) - 5) * 0.03125 +
			                              static_cast<double>(static_cast<int>(col % 7) - 3) * 0.015625;
		}
	}
	const auto weightIndex = graph.AddVariable(
	    Variable::Create(Tensor<CPU>(std::span<const double>(weightData), { kSize, nSize }, DataType::Float32)));

	Subgraph sg;
	const auto input = sg.AddParam(DataType::Float32, { batch, kSize });
	const auto weight =
	    sg.AddNode(VariableRefNode{ weightIndex }, { OutputInfo{ DataType::Float32, { kSize, nSize } } });
	const auto output = sg.AddNode(BinaryOpNode{ BinaryOp::MatMul, { input, 0 }, { weight, 0 } },
	                               { OutputInfo{ DataType::Float32, { batch, nSize } } });
	sg.SetResults({ { output, 0 } });
	graph.AddSubgraph(std::move(sg));
	graph.SetForward(0);

	std::vector<double> inputData(batch * kSize);
	for (std::size_t row = 0; row < batch; ++row)
	{
		for (std::size_t k = 0; k < kSize; ++k)
		{
			inputData[row * kSize + k] = static_cast<double>((row + 1) * ((k % 13) + 1)) * 0.00390625;
		}
	}
	Tensor<CPU> x(std::span<const double>(inputData), { batch, kSize }, DataType::Float32);
	std::array<Tensor<CPU>, 1> inputs = { std::move(x) };
	std::array<Tensor<CPU>, 1> outputs = { Tensor<CPU>(Uninitialized, { batch, nSize }, DataType::Float32) };

	auto compiled = Compiler<CPU>::Compile(graph);
	compiled.RunInto(inputs, outputs);

	for (std::size_t row = 0; row < batch; ++row)
	{
		for (std::size_t col = 0; col < nSize; ++col)
		{
			double expected = 0.0;
			for (std::size_t k = 0; k < kSize; ++k)
			{
				expected += inputData[row * kSize + k] * weightData[k * nSize + col];
			}
			EXPECT_NEAR(ReadFloat(outputs[0], row * nSize + col), expected, 1e-3f);
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
		(void) CompiledModule<CPU>::Load(image);
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
		(void) CompiledModule<CPU>::Load(image);
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
		futures.push_back(
		    std::async(std::launch::async, [&loaded, workerId] { return RunCompiledModuleWorker(loaded, workerId); }));
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
			const auto expected = base + static_cast<float>(element + 1) + static_cast<float>((element + 1) * 10);
			EXPECT_FLOAT_EQ(ReadFloat(outputs[i][0], element), expected);
		}
	}
}

TEST(CompiledModuleTest, CPUParallelLinearChainMatchesInterpreter)
{
	ScopedEnvVar threads("LITENN_CPU_AOT_THREADS", "4");
	ScopedEnvVar minFlops("LITENN_CPU_AOT_PARALLEL_MIN_FLOPS", "1");
	constexpr std::size_t kBatch = 128;
	constexpr std::size_t kInput = 64;
	constexpr std::size_t kOutput = 32;
	auto graph = BuildWideLinearChainGraph(kBatch);
	auto inputData = MakePatternValues(kBatch * kInput, 0.01f);
	std::array<Tensor<CPU>, 1> inputs = { Tensor<CPU>(std::span<const double>(inputData), { kBatch, kInput },
		                                              DataType::Float32) };

	Runtime::Interpreter<CPU> interpreter;
	const auto expected = interpreter.RunForward(graph, std::span<const Tensor<CPU>>(inputs));

	auto optimized = graph;
	FusionPass{}.Run(optimized);
	auto artifact = Compiler<CPU>::CompileArtifact(optimized);
	const std::string instructions(reinterpret_cast<const char*>(artifact.Instructions().data()),
	                               artifact.Instructions().size());
	ASSERT_NE(instructions.find("litenn_cpu_matmul_bias_relu_parallel_f32"), std::string::npos);

	auto module = artifact.Load();
	std::array<Tensor<CPU>, 1> outputs = { Tensor<CPU>(Uninitialized, { kBatch, kOutput }, DataType::Float32) };
	module.RunInto(std::span<const Tensor<CPU>>(inputs), std::span<Tensor<CPU>>(outputs));

	ASSERT_EQ(expected.size(), 1u);
	ASSERT_EQ(outputs[0].NumElements(), expected[0].NumElements());
	for (std::size_t i = 0; i < outputs[0].NumElements(); ++i)
	{
		EXPECT_NEAR(ReadFloat(outputs[0], i), ReadFloat(expected[0], i), 1e-4f);
	}
}

TEST(CompiledModuleTest, CPUParallelLinearChainLoadsExternalConstantsRegion)
{
	ScopedEnvVar threads("LITENN_CPU_AOT_THREADS", "4");
	ScopedEnvVar minFlops("LITENN_CPU_AOT_PARALLEL_MIN_FLOPS", "1");
	ScopedEnvVar externalConstants("LITENN_CPU_AOT_EXTERNAL_CONSTANTS", "1");
	constexpr std::size_t kBatch = 128;
	constexpr std::size_t kInput = 64;
	constexpr std::size_t kOutput = 32;
	auto graph = BuildWideLinearChainGraph(kBatch);
	auto inputData = MakePatternValues(kBatch * kInput, 0.01f);
	std::array<Tensor<CPU>, 1> inputs = { Tensor<CPU>(std::span<const double>(inputData), { kBatch, kInput },
	                                                  DataType::Float32) };

	Runtime::Interpreter<CPU> interpreter;
	const auto expected = interpreter.RunForward(graph, std::span<const Tensor<CPU>>(inputs));

	auto optimized = graph;
	FusionPass{}.Run(optimized);
	auto artifact = Compiler<CPU>::CompileArtifact(optimized);
	auto separated = artifact.SeparateRodata();

	ASSERT_GT(separated.Constants().size(), 0u);
	EXPECT_EQ(separated.Weights().size(), 0u);
	ASSERT_GT(separated.Instructions().size(), 0u);
	const auto* constantsInfo = FindRegionInfo(separated.RegionInfos(), "constants");
	ASSERT_NE(constantsInfo, nullptr);
	EXPECT_EQ(constantsInfo->size, separated.Constants().size());
	const auto externalInfos = separated.ExternalTensorInfos();
	ASSERT_GE(externalInfos.size(), 4u);
	std::vector<std::string> externalNames;
	externalNames.reserve(externalInfos.size());
	for (const auto& info : externalInfos)
	{
		externalNames.push_back(info.name);
		EXPECT_EQ(info.region, "constants");
		EXPECT_EQ(info.dtype, DataType::Float32);
		EXPECT_GT(info.shape.size(), 0u);
		EXPECT_GT(info.byteSize, 0u);
		EXPECT_EQ(info.byteOffset % info.alignment, 0u);
		EXPECT_LE(info.byteOffset + info.byteSize, separated.Constants().size());
		EXPECT_EQ(info.rebindPolicy, CompiledModuleExternalTensorRebindPolicy::ExactChecksum);
	}
	for (std::size_t i = 0; i < 4; ++i)
	{
		EXPECT_NE(std::find(externalNames.begin(), externalNames.end(), std::format("variable{}", i)),
		          externalNames.end());
	}

	auto runAndCheck = [&](CompiledModule<CPU>& module)
	{
		std::array<Tensor<CPU>, 1> outputs = { Tensor<CPU>(Uninitialized, { kBatch, kOutput }, DataType::Float32) };
		module.RunInto(std::span<const Tensor<CPU>>(inputs), std::span<Tensor<CPU>>(outputs));

		ASSERT_EQ(expected.size(), 1u);
		ASSERT_EQ(outputs[0].NumElements(), expected[0].NumElements());
		for (std::size_t i = 0; i < outputs[0].NumElements(); ++i)
		{
			EXPECT_NEAR(ReadFloat(outputs[0], i), ReadFloat(expected[0], i), 1e-4f);
		}
	};

	auto direct = artifact.Load();
	runAndCheck(direct);

	auto imageLoaded = CompiledModule<CPU>::Load(separated.Image());
	runAndCheck(imageLoaded);

	std::vector<std::byte> reboundConstants(separated.Constants().begin(), separated.Constants().end());
	auto rebound = separated.WithReboundConstants({
	    .data = reboundConstants.data(),
	    .size = reboundConstants.size(),
	});
	auto reboundLoaded = rebound.Load();
	runAndCheck(reboundLoaded);

	std::vector<std::byte> wrongSize(separated.Constants().size() + 1);
	EXPECT_THROW((void) separated.WithReboundConstants({ .data = wrongSize.data(), .size = wrongSize.size() }),
	             std::runtime_error);
}
