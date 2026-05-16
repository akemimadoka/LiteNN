#include <gtest/gtest.h>

#include <LiteNN.h>
#include <LiteNN/Compiler/CompiledModule.h>
#include <LiteNN/Compiler/CUDANativeCodegen.h>
#include <LiteNN/Compiler/CUDANativePayload.h>
#include <LiteNN/Pass/FusionPass.h>
#include <LiteNN/Runtime/Interpreter.h>

#include <array>
#include <cstddef>
#include <cstdint>
#include <cstdlib>
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

	Graph BuildTinyLinearChainGraph(std::size_t batch)
	{
		Graph graph;
		const auto h1 = Layer::CreateLinear(graph,
		    Tensor<CPU>({ 0.5, -0.25, 0.75, 0.125, -0.5, 0.25, 1.0, -1.0, 0.375, 0.625, -0.75, 0.5 },
		                { 3, 4 }, DataType::Float32),
		    Tensor<CPU>({ 0.1, -0.2, 0.3, -0.4 }, { 1, 4 }, DataType::Float32));
		const auto h2 = Layer::CreateLinear(graph,
		    Tensor<CPU>({ 0.25, -0.5, 0.75, 0.5, 0.125, -0.25, -0.375, 0.625 },
		                { 4, 2 }, DataType::Float32),
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

TEST(CompiledModuleTest, CUDANativeInstructionPayloadRoundTripsLaunchMetadata)
{
	const auto binary = std::vector<std::byte>{
		std::byte{ 'p' }, std::byte{ 't' }, std::byte{ 'x' }
	};
	const auto payload = CUDANativeInstructionPayload{
	    .binaryKind = CUDANativeBinaryKind::PTX,
	    .featureFlags = kCUDANativeFeatureStaticShape | kCUDANativeFeatureSingleSubgraph,
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
	EXPECT_EQ(decoded.featureFlags, kCUDANativeFeatureStaticShape | kCUDANativeFeatureSingleSubgraph);
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
	    .featureFlags = kCUDANativeFeatureStaticShape | kCUDANativeFeatureSingleSubgraph |
	                    kCUDANativeFeatureElementwiseSubtractF32 | kCUDANativeFeatureElementwiseBroadcastF32,
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
	EXPECT_EQ(decoded.featureFlags & kCUDANativeFeatureElementwiseBroadcastF32,
	          kCUDANativeFeatureElementwiseBroadcastF32);
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
		std::uint64_t featureFlag;
	};
	const std::array cases = {
		Case{ UnaryOp::Negate, "litenn_negate_f32", "neg.f32", kCUDANativeFeatureElementwiseNegateF32 },
		Case{ UnaryOp::Abs, "litenn_abs_f32", "abs.ftz.f32", kCUDANativeFeatureElementwiseAbsF32 },
		Case{ UnaryOp::Sqrt, "litenn_sqrt_f32", "sqrt.rn.ftz.f32", kCUDANativeFeatureElementwiseSqrtF32 },
		Case{ UnaryOp::Exp, "litenn_exp_f32", "ex2.approx.ftz.f32", kCUDANativeFeatureElementwiseExpF32 },
		Case{ UnaryOp::Log, "litenn_log_f32", "lg2.approx.ftz.f32", kCUDANativeFeatureElementwiseLogF32 },
		Case{ UnaryOp::Sin, "litenn_sin_f32", "sin.approx.ftz.f32", kCUDANativeFeatureElementwiseSinF32 },
		Case{ UnaryOp::Cos, "litenn_cos_f32", "cos.approx.ftz.f32", kCUDANativeFeatureElementwiseCosF32 },
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
		    .featureFlags = kCUDANativeFeatureStaticShape | kCUDANativeFeatureSingleSubgraph | testCase.featureFlag,
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
		EXPECT_EQ(decoded.featureFlags & testCase.featureFlag, testCase.featureFlag);
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
		std::uint64_t featureFlag;
	};
	const std::array cases = {
		Case{ BinaryOp::Add, "litenn_add_f32", "add.rn.f32", kCUDANativeFeatureElementwiseAddF32 },
		Case{ BinaryOp::Subtract, "litenn_subtract_f32", "sub.rn.f32", kCUDANativeFeatureElementwiseSubtractF32 },
		Case{ BinaryOp::Multiply, "litenn_multiply_f32", "mul.rn.f32", kCUDANativeFeatureElementwiseMultiplyF32 },
		Case{ BinaryOp::Divide, "litenn_divide_f32", "div.rn.f32", kCUDANativeFeatureElementwiseDivideF32 },
		Case{ BinaryOp::Max, "litenn_max_f32", "max.ftz.f32", kCUDANativeFeatureElementwiseMaxF32 },
		Case{ BinaryOp::Min, "litenn_min_f32", "min.ftz.f32", kCUDANativeFeatureElementwiseMinF32 },
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
		    .featureFlags = kCUDANativeFeatureStaticShape | kCUDANativeFeatureSingleSubgraph | testCase.featureFlag,
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
		EXPECT_EQ(decoded.featureFlags & testCase.featureFlag, testCase.featureFlag);
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

TEST(CompiledModuleTest, CUDANativeInstructionPayloadAllowsLibraryCallWithoutBinaryImage)
{
	const auto payload = CUDANativeInstructionPayload{
	    .binaryKind = CUDANativeBinaryKind::LibraryCall,
	    .featureFlags = kCUDANativeFeatureStaticShape | kCUDANativeFeatureSingleSubgraph |
	                    kCUDANativeFeatureMatMulCUBLASF32,
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
	EXPECT_EQ(decoded.featureFlags & kCUDANativeFeatureMatMulCUBLASF32, kCUDANativeFeatureMatMulCUBLASF32);
	ASSERT_EQ(decoded.kernels.size(), 1u);
	EXPECT_EQ(decoded.kernels[0].name, "litenn_cublas_matmul_f32");
}

TEST(CompiledModuleTest, CUDANativeInstructionPayloadRejectsInvalidMagic)
{
	std::vector<std::byte> bytes = {
		std::byte{ 'b' }, std::byte{ 'a' }, std::byte{ 'd' }, std::byte{ 0 }
	};

	try
	{
		(void)DeserializeCUDANativeInstructionPayload(bytes);
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
	payload.featureFlags = 1ull << 63;

	EXPECT_THROW((void)SerializeCUDANativeInstructionPayload(payload), std::runtime_error);
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
			weightData[k * nSize + col] =
			    static_cast<double>(static_cast<int>(k % 11) - 5) * 0.03125 +
			    static_cast<double>(static_cast<int>(col % 7) - 3) * 0.015625;
		}
	}
	const auto weightIndex = graph.AddVariable(Variable::Create(
	    Tensor<CPU>(std::span<const double>(weightData), { kSize, nSize }, DataType::Float32)));

	Subgraph sg;
	const auto input = sg.AddParam(DataType::Float32, { batch, kSize });
	const auto weight = sg.AddNode(VariableRefNode{ weightIndex },
	                               { OutputInfo{ DataType::Float32, { kSize, nSize } } });
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
			inputData[row * kSize + k] =
			    static_cast<double>((row + 1) * ((k % 13) + 1)) * 0.00390625;
		}
	}
	Tensor<CPU> x(std::span<const double>(inputData), { batch, kSize }, DataType::Float32);
	std::array<Tensor<CPU>, 1> inputs = { std::move(x) };
	std::array<Tensor<CPU>, 1> outputs = {
	    Tensor<CPU>(Uninitialized, { batch, nSize }, DataType::Float32)
	};

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

TEST(CompiledModuleTest, CPULinearChainFastPathMatchesInterpreter)
{
	ScopedEnvVar enableFastPath("LITENN_CPU_AOT_LINEAR_CHAIN_FASTPATH", "1");
	auto graph = BuildTinyLinearChainGraph(4);
	std::array<Tensor<CPU>, 1> inputs = {
		Tensor<CPU>({ 1.0, -2.0, 0.5, -1.0, 0.25, 2.0, 0.5, 0.5, -0.5, 3.0, -1.0, 1.0 },
		            { 4, 3 }, DataType::Float32)
	};
	Runtime::Interpreter<CPU> interpreter;
	const auto expected = interpreter.RunForward(graph, std::span<const Tensor<CPU>>(inputs));

	auto optimized = graph;
	FusionPass{}.Run(optimized);
	auto module = Compiler<CPU>::Compile(optimized);
	std::array<Tensor<CPU>, 1> outputs = {
		Tensor<CPU>(Uninitialized, { 4, 2 }, DataType::Float32)
	};
	module.RunInto(std::span<const Tensor<CPU>>(inputs), std::span<Tensor<CPU>>(outputs));

	ASSERT_EQ(expected.size(), 1u);
	ASSERT_EQ(outputs[0].NumElements(), expected[0].NumElements());
	for (std::size_t i = 0; i < outputs[0].NumElements(); ++i)
	{
		EXPECT_NEAR(ReadFloat(outputs[0], i), ReadFloat(expected[0], i), 1e-5f);
	}
}
