#include <gtest/gtest.h>

#include <LiteNN.h>
#include <LiteNN/Compiler/CompiledModule.h>
#include <LiteNN/Compiler/CUDANativePayload.h>
#include <LiteNN/Runtime/Interpreter.h>

#include <array>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <span>
#include <string>
#include <string_view>
#include <utility>
#include <vector>

using namespace LiteNN;

namespace
{
	float ReadFloat(const Tensor<CPU>& t, std::size_t i)
	{
		return static_cast<const float*>(t.RawData())[i];
	}

	std::uint32_t ReadU32LE(std::span<const std::byte> bytes, std::size_t offset)
	{
		std::uint32_t value = 0;
		for (int i = 0; i < 4; ++i)
		{
			value |= std::to_integer<std::uint32_t>(bytes[offset + i]) << (i * 8);
		}
		return value;
	}

	void ExpectTensorNear(const Tensor<CPU>& tensor, std::span<const float> expected, float tolerance = 1e-6f)
	{
		ASSERT_EQ(tensor.NumElements(), expected.size());
		for (std::size_t i = 0; i < expected.size(); ++i)
		{
			EXPECT_NEAR(ReadFloat(tensor, i), expected[i], tolerance);
		}
	}

	void ExpectTensorNear(const Tensor<CPU>& tensor, const Tensor<CPU>& expected, float tolerance = 1e-6f)
	{
		EXPECT_EQ(tensor.DType(), expected.DType());
		EXPECT_EQ(tensor.Shape().ToOwned(), expected.Shape().ToOwned());
		ASSERT_EQ(tensor.NumElements(), expected.NumElements());
		for (std::size_t i = 0; i < expected.NumElements(); ++i)
		{
			EXPECT_NEAR(ReadFloat(tensor, i), ReadFloat(expected, i), tolerance);
		}
	}

	void ExpectOutputsNear(std::span<const Tensor<CPU>> outputs, std::span<const Tensor<CPU>> expected,
	                       float tolerance = 1e-6f)
	{
		ASSERT_EQ(outputs.size(), expected.size());
		for (std::size_t i = 0; i < outputs.size(); ++i)
		{
			SCOPED_TRACE(i);
			ExpectTensorNear(outputs[i], expected[i], tolerance);
		}
	}

	std::string BytesToString(std::span<const std::byte> bytes)
	{
		return std::string(reinterpret_cast<const char*>(bytes.data()), bytes.size());
	}

	Graph BuildBinaryGraph(BinaryOp op, std::span<const std::size_t> lhsShape,
	                       std::span<const std::size_t> rhsShape, std::span<const std::size_t> outputShape,
	                       std::string outputName)
	{
		Graph graph;
		Subgraph sg;
		const auto a = sg.AddParam(DataType::Float32, std::vector<std::size_t>(lhsShape.begin(), lhsShape.end()));
		const auto b = sg.AddParam(DataType::Float32, std::vector<std::size_t>(rhsShape.begin(), rhsShape.end()));
		const auto y = sg.AddNode(BinaryOpNode{ op, { a, 0 }, { b, 0 } },
		                          { OutputInfo{ DataType::Float32,
		                                        std::vector<std::size_t>(outputShape.begin(), outputShape.end()) } });
		sg.SetResults(std::vector<NodeOutput>{ { y, 0 } });
		graph.AddSubgraph(std::move(sg));
		graph.SetForward(0);
		graph.SetInputNames({ "lhs", "rhs" });
		graph.SetOutputNames({ std::move(outputName) });
		return graph;
	}

	Graph BuildSimpleBinaryGraph(BinaryOp op, std::string outputName)
	{
		return BuildBinaryGraph(op, std::array{ 2uz, 2uz }, std::array{ 2uz, 2uz }, std::array{ 2uz, 2uz },
		                        std::move(outputName));
	}

	Graph BuildSimpleUnaryGraph(UnaryOp op, std::string outputName)
	{
		Graph graph;
		Subgraph sg;
		const auto input = sg.AddParam(DataType::Float32, { 2, 2 });
		const auto y = sg.AddNode(UnaryOpNode{ op, { input, 0 } },
		                          { OutputInfo{ DataType::Float32, { 2, 2 } } });
		sg.SetResults({ { y, 0 } });
		graph.AddSubgraph(std::move(sg));
		graph.SetForward(0);
		graph.SetInputNames({ "input" });
		graph.SetOutputNames({ std::move(outputName) });
		return graph;
	}

	Graph BuildSimpleMatMulGraph()
	{
		Graph graph;
		Subgraph sg;
		const auto a = sg.AddParam(DataType::Float32, { 2, 2 });
		const auto b = sg.AddParam(DataType::Float32, { 2, 2 });
		const auto y = sg.AddNode(BinaryOpNode{ BinaryOp::MatMul, { a, 0 }, { b, 0 } },
		                          { OutputInfo{ DataType::Float32, { 2, 2 } } });
		sg.SetResults({ { y, 0 } });
		graph.AddSubgraph(std::move(sg));
		graph.SetForward(0);
		graph.SetInputNames({ "lhs", "rhs" });
		graph.SetOutputNames({ "matmul" });
		return graph;
	}

	Graph BuildSimplePowGraph()
	{
		return BuildSimpleBinaryGraph(BinaryOp::Pow, "pow");
	}

	struct TensorInputSpec
	{
		std::vector<double> values;
		std::vector<std::size_t> shape;
	};

	std::vector<Tensor<CPU>> MakeCPUInputs(std::span<const TensorInputSpec> specs)
	{
		std::vector<Tensor<CPU>> inputs;
		inputs.reserve(specs.size());
		for (const auto& spec : specs)
		{
			inputs.emplace_back(std::span<const double>(spec.values.data(), spec.values.size()), ShapeView{ spec.shape },
			                    DataType::Float32);
		}
		return inputs;
	}

	std::vector<Tensor<CUDA>> MakeCUDAInputs(std::span<const TensorInputSpec> specs)
	{
		std::vector<Tensor<CUDA>> inputs;
		inputs.reserve(specs.size());
		for (const auto& spec : specs)
		{
			auto cpuInput = Tensor<CPU>(std::span<const double>(spec.values.data(), spec.values.size()),
			                            ShapeView{ spec.shape }, DataType::Float32);
			inputs.push_back(cpuInput.CopyToDevice(CUDA{}));
		}
		return inputs;
	}
} // namespace

TEST(CompiledModuleCUDATest, CompilerArtifactsExposeStableCUDANativeABI)
{
	{
		auto artifact = Compiler<CUDA>::CompileArtifact(BuildSimpleMatMulGraph());
		EXPECT_EQ(artifact.Backend(), CompiledModuleBackend::CUDANative);
		EXPECT_EQ(CompiledModuleArtifact::CopyFromImage(artifact.Image()).Backend(), CompiledModuleBackend::CUDANative);
		ASSERT_EQ(artifact.InputSpecs().size(), 2u);
		ASSERT_EQ(artifact.OutputSpecs().size(), 1u);
		EXPECT_EQ(artifact.InputSpecs()[0].name, "lhs");
		EXPECT_EQ(artifact.InputSpecs()[1].name, "rhs");
		EXPECT_EQ(artifact.OutputSpecs()[0].name, "matmul");

		const auto payload = DeserializeCUDANativeInstructionPayload(artifact.Instructions());
		EXPECT_EQ(payload.binaryKind, CUDANativeBinaryKind::LibraryCall);
		EXPECT_EQ(payload.featureFlags, kCUDANativeFeatureStaticShape | kCUDANativeFeatureSingleSubgraph |
		                                    kCUDANativeFeatureMatMulCUBLASF32);
		EXPECT_EQ(payload.target, "cublas");
		EXPECT_TRUE(payload.binary.empty());
		ASSERT_EQ(payload.scalarData.size(), 3u * sizeof(std::uint32_t));
		EXPECT_EQ(ReadU32LE(payload.scalarData, 0), 2u);
		EXPECT_EQ(ReadU32LE(payload.scalarData, 4), 2u);
		EXPECT_EQ(ReadU32LE(payload.scalarData, 8), 2u);
		ASSERT_EQ(payload.kernels.size(), 1u);
		EXPECT_EQ(payload.kernels[0].name, "litenn_cublas_matmul_f32");
		ASSERT_EQ(payload.kernels[0].arguments.size(), 3u);
		EXPECT_EQ(payload.kernels[0].arguments[0].kind, CUDANativeArgumentKind::OutputTensor);
		EXPECT_EQ(payload.kernels[0].arguments[0].byteSize, 16u);
		EXPECT_EQ(payload.kernels[0].arguments[1].kind, CUDANativeArgumentKind::InputTensor);
		EXPECT_EQ(payload.kernels[0].arguments[1].byteSize, 16u);
		EXPECT_EQ(payload.kernels[0].arguments[2].kind, CUDANativeArgumentKind::InputTensor);
		EXPECT_EQ(payload.kernels[0].arguments[2].byteSize, 16u);
	}

#ifdef LITENN_ENABLE_CUDA_DRIVER
	{
		const std::array outputShape{ 2uz, 3uz };
		auto artifact = Compiler<CUDA>::CompileArtifact(BuildBinaryGraph(BinaryOp::Divide, std::array{ 2uz, 3uz },
		                                                               std::array{ 1uz, 3uz }, outputShape,
		                                                               "broadcast_divide"));
		EXPECT_EQ(artifact.Backend(), CompiledModuleBackend::CUDANative);
		EXPECT_EQ(CompiledModuleArtifact::CopyFromImage(artifact.Image()).Backend(), CompiledModuleBackend::CUDANative);
		ASSERT_EQ(artifact.InputSpecs().size(), 2u);
		ASSERT_EQ(artifact.OutputSpecs().size(), 1u);
		EXPECT_EQ(artifact.InputSpecs()[0].name, "lhs");
		EXPECT_EQ(artifact.InputSpecs()[0].shape, (std::vector<std::size_t>{ 2, 3 }));
		EXPECT_EQ(artifact.InputSpecs()[1].name, "rhs");
		EXPECT_EQ(artifact.InputSpecs()[1].shape, (std::vector<std::size_t>{ 1, 3 }));
		EXPECT_EQ(artifact.OutputSpecs()[0].name, "broadcast_divide");
		EXPECT_EQ(artifact.OutputSpecs()[0].shape, (std::vector<std::size_t>{ 2, 3 }));

		const auto payload = DeserializeCUDANativeInstructionPayload(artifact.Instructions());
		EXPECT_EQ(payload.binaryKind, CUDANativeBinaryKind::PTX);
		EXPECT_EQ(payload.featureFlags, kCUDANativeFeatureStaticShape | kCUDANativeFeatureSingleSubgraph |
		                                    kCUDANativeFeatureElementwiseDivideF32 |
		                                    kCUDANativeFeatureElementwiseBroadcastF32);
		EXPECT_EQ(payload.target, "sm_30");
		ASSERT_FALSE(payload.binary.empty());
		EXPECT_EQ(payload.binary.back(), std::byte{ 0 });
		const auto ptx = BytesToString(payload.binary);
		EXPECT_NE(ptx.find("Generated by LLVM NVPTX Back-End"), std::string::npos);
		EXPECT_NE(ptx.find(".visible .entry litenn_divide_broadcast_f32"), std::string::npos);
		ASSERT_EQ(payload.scalarData.size(), sizeof(std::uint32_t));
		EXPECT_EQ(ReadU32LE(payload.scalarData, 0), 6u);
		ASSERT_EQ(payload.kernels.size(), 1u);
		const auto& kernel = payload.kernels[0];
		EXPECT_EQ(kernel.name, "litenn_divide_broadcast_f32");
		EXPECT_EQ(kernel.grid.x, 1u);
		EXPECT_EQ(kernel.block.x, 6u);
		ASSERT_EQ(kernel.arguments.size(), 4u);
		EXPECT_EQ(kernel.arguments[0].kind, CUDANativeArgumentKind::OutputTensor);
		EXPECT_EQ(kernel.arguments[0].byteSize, 24u);
		EXPECT_EQ(kernel.arguments[1].kind, CUDANativeArgumentKind::InputTensor);
		EXPECT_EQ(kernel.arguments[1].index, 0u);
		EXPECT_EQ(kernel.arguments[1].byteSize, 24u);
		EXPECT_EQ(kernel.arguments[2].kind, CUDANativeArgumentKind::InputTensor);
		EXPECT_EQ(kernel.arguments[2].index, 1u);
		EXPECT_EQ(kernel.arguments[2].byteSize, 12u);
		EXPECT_EQ(kernel.arguments[3].kind, CUDANativeArgumentKind::Scalar);
		EXPECT_EQ(kernel.arguments[3].byteSize, sizeof(std::uint32_t));
	}
#endif
}

TEST(CompiledModuleCUDATest, RunsCPUAOTBridgeWithCUDATensors)
{
	if (!IsCUDADeviceAvailable())
	{
		GTEST_SKIP() << "CUDA device is not available";
	}

	auto graph = BuildSimplePowGraph();
	auto compiled = Compiler<CUDA>::Compile(graph, CUDA{});

	EXPECT_EQ(compiled.Backend(), CompiledModuleBackend::CPUNative);
	ASSERT_GT(compiled.Rodata().size(), 0u);
	ASSERT_GT(compiled.Instructions().size(), 0u);
	EXPECT_EQ(compiled.FindInput("lhs"), 0u);
	EXPECT_EQ(compiled.FindOutput("pow"), 0u);

	auto lhs = Tensor<CPU>({ 2, 3, 4, 5 }, { 2, 2 }, DataType::Float32).CopyToDevice(CUDA{});
	auto rhs = Tensor<CPU>({ 1, 2, 3, 0 }, { 2, 2 }, DataType::Float32).CopyToDevice(CUDA{});
	std::array<Tensor<CUDA>, 2> inputs = { std::move(lhs), std::move(rhs) };

	auto outputs = compiled.Run(inputs);
	ASSERT_EQ(outputs.size(), 1u);
	auto cpuOutput = outputs[0].CopyToDevice(CPU{});
	ExpectTensorNear(cpuOutput, std::array{ 2.0f, 9.0f, 64.0f, 1.0f });

	auto loaded = CompiledModule<CUDA>::Load(compiled.Image(), CUDA{});
	std::array<Tensor<CUDA>, 1> out = {
		Tensor<CUDA>(Uninitialized, { 2, 2 }, DataType::Float32, CUDA{})
	};
	loaded.RunInto(inputs, out);
	auto cpuOutInto = out[0].CopyToDevice(CPU{});
	ExpectTensorNear(cpuOutInto, std::array{ 2.0f, 9.0f, 64.0f, 1.0f });
}

TEST(CompiledModuleCUDATest, ArtifactLoadsAsCUDABridge)
{
	if (!IsCUDADeviceAvailable())
	{
		GTEST_SKIP() << "CUDA device is not available";
	}

	auto graph = BuildSimplePowGraph();
	auto artifact = Compiler<CUDA>::CompileArtifact(graph);
	auto module = artifact.Load(CUDA{});

	EXPECT_EQ(artifact.Backend(), CompiledModuleBackend::CPUNative);
	EXPECT_EQ(module.Backend(), CompiledModuleBackend::CPUNative);

	auto lhs = Tensor<CPU>({ 2, 3, 4, 5 }, { 2, 2 }, DataType::Float32).CopyToDevice(CUDA{});
	auto rhs = Tensor<CPU>({ 3, 2, 1, 0 }, { 2, 2 }, DataType::Float32).CopyToDevice(CUDA{});
	std::array<Tensor<CUDA>, 2> inputs = { std::move(lhs), std::move(rhs) };

	auto outputs = module.Run(inputs);
	auto cpuOutput = outputs[0].CopyToDevice(CPU{});
	ExpectTensorNear(cpuOutput, std::array{ 8.0f, 9.0f, 4.0f, 1.0f });
}

TEST(CompiledModuleCUDATest, RunsNativeMatMulWithCUBLAS)
{
	if (!IsCUDADeviceAvailable())
	{
		GTEST_SKIP() << "CUDA device is not available";
	}

	auto graph = BuildSimpleMatMulGraph();
	auto artifact = Compiler<CUDA>::CompileArtifact(graph);
	ASSERT_EQ(artifact.Backend(), CompiledModuleBackend::CUDANative);
	auto module = artifact.Load(CUDA{});
	EXPECT_EQ(module.Backend(), CompiledModuleBackend::CUDANative);
	EXPECT_EQ(module.FindInput("lhs"), 0u);
	EXPECT_EQ(module.FindInput("rhs"), 1u);
	EXPECT_EQ(module.FindOutput("matmul"), 0u);

	auto lhs = Tensor<CPU>({ 1, 2, 3, 4 }, { 2, 2 }, DataType::Float32).CopyToDevice(CUDA{});
	auto rhs = Tensor<CPU>({ 10, 20, 30, 40 }, { 2, 2 }, DataType::Float32).CopyToDevice(CUDA{});
	std::array<Tensor<CUDA>, 2> inputs = { std::move(lhs), std::move(rhs) };

	auto outputs = module.Run(inputs);
	ASSERT_EQ(outputs.size(), 1u);
	ExpectTensorNear(outputs[0].CopyToDevice(CPU{}), std::array{ 70.0f, 100.0f, 150.0f, 220.0f });

	std::array<Tensor<CUDA>, 1> out = {
		Tensor<CUDA>(Uninitialized, { 2, 2 }, DataType::Float32, CUDA{})
	};
	module.RunInto(inputs, out);
	ExpectTensorNear(out[0].CopyToDevice(CPU{}), std::array{ 70.0f, 100.0f, 150.0f, 220.0f });
}

TEST(CompiledModuleCUDATest, RunsNativeElementwiseBinaryOpsWithCUDATensors)
{
	if (!IsCUDADeviceAvailable())
	{
		GTEST_SKIP() << "CUDA device is not available";
	}
	if (!IsCUDADriverAvailable())
	{
		GTEST_SKIP() << "CUDA driver is not available";
	}

	struct Case
	{
		BinaryOp op;
		std::string_view outputName;
		std::array<float, 4> expected;
	};
	const std::array cases = {
		Case{ BinaryOp::Add, "sum", { 11.0f, 22.0f, 33.0f, 44.0f } },
		Case{ BinaryOp::Subtract, "difference", { -9.0f, -18.0f, -27.0f, -36.0f } },
		Case{ BinaryOp::Multiply, "product", { 10.0f, 40.0f, 90.0f, 160.0f } },
		Case{ BinaryOp::Divide, "quotient", { 0.1f, 0.1f, 0.1f, 0.1f } },
		Case{ BinaryOp::Max, "maximum", { 10.0f, 20.0f, 30.0f, 40.0f } },
		Case{ BinaryOp::Min, "minimum", { 1.0f, 2.0f, 3.0f, 4.0f } },
	};

	for (const auto& testCase : cases)
	{
		SCOPED_TRACE(testCase.outputName);
		auto graph = BuildSimpleBinaryGraph(testCase.op, std::string(testCase.outputName));
		auto artifact = Compiler<CUDA>::CompileArtifact(graph);
		ASSERT_EQ(artifact.Backend(), CompiledModuleBackend::CUDANative);
		const auto payload = DeserializeCUDANativeInstructionPayload(artifact.Instructions());
		ASSERT_EQ(payload.binaryKind, CUDANativeBinaryKind::PTX);
		EXPECT_NE(BytesToString(payload.binary).find("Generated by LLVM NVPTX Back-End"), std::string::npos);
		auto module = artifact.Load(CUDA{});
		EXPECT_EQ(module.Backend(), CompiledModuleBackend::CUDANative);
		EXPECT_EQ(module.FindOutput(testCase.outputName), 0u);

		auto lhs = Tensor<CPU>({ 1, 2, 3, 4 }, { 2, 2 }, DataType::Float32).CopyToDevice(CUDA{});
		auto rhs = Tensor<CPU>({ 10, 20, 30, 40 }, { 2, 2 }, DataType::Float32).CopyToDevice(CUDA{});
		std::array<Tensor<CUDA>, 2> inputs = { std::move(lhs), std::move(rhs) };

		auto outputs = module.Run(inputs);
		ASSERT_EQ(outputs.size(), 1u);
		auto cpuOutput = outputs[0].CopyToDevice(CPU{});
		ExpectTensorNear(cpuOutput, testCase.expected);

		std::array<Tensor<CUDA>, 1> out = {
			Tensor<CUDA>(Uninitialized, { 2, 2 }, DataType::Float32, CUDA{})
		};
		module.RunInto(inputs, out);
		auto cpuOutInto = out[0].CopyToDevice(CPU{});
		ExpectTensorNear(cpuOutInto, testCase.expected);
	}
}

TEST(CompiledModuleCUDATest, RunsNativeElementwiseBroadcastBinaryOpsWithCUDATensors)
{
	if (!IsCUDADeviceAvailable())
	{
		GTEST_SKIP() << "CUDA device is not available";
	}
	if (!IsCUDADriverAvailable())
	{
		GTEST_SKIP() << "CUDA driver is not available";
	}

	struct Case
	{
		BinaryOp op;
		std::string_view outputName;
		std::vector<std::size_t> lhsShape;
		std::vector<std::size_t> rhsShape;
		std::vector<double> lhs;
		std::vector<double> rhs;
		std::vector<float> expected;
	};
	const std::vector cases = {
		Case{
		    .op = BinaryOp::Add,
		    .outputName = "bias_added",
		    .lhsShape = { 2, 3 },
		    .rhsShape = { 1, 3 },
		    .lhs = { 1, 2, 3, 4, 5, 6 },
		    .rhs = { 10, 20, 30 },
		    .expected = { 11.0f, 22.0f, 33.0f, 14.0f, 25.0f, 36.0f },
		},
		Case{
		    .op = BinaryOp::Subtract,
		    .outputName = "row_shifted",
		    .lhsShape = { 2, 3 },
		    .rhsShape = { 2, 1 },
		    .lhs = { 1, 2, 3, 4, 5, 6 },
		    .rhs = { 10, 20 },
		    .expected = { -9.0f, -8.0f, -7.0f, -16.0f, -15.0f, -14.0f },
		},
		Case{
		    .op = BinaryOp::Multiply,
		    .outputName = "scaled",
		    .lhsShape = { 1, 3 },
		    .rhsShape = { 2, 3 },
		    .lhs = { 2, 3, 4 },
		    .rhs = { 1, 2, 3, 4, 5, 6 },
		    .expected = { 2.0f, 6.0f, 12.0f, 8.0f, 15.0f, 24.0f },
		},
		Case{
		    .op = BinaryOp::Max,
		    .outputName = "broadcast_max",
		    .lhsShape = { 2, 3 },
		    .rhsShape = { 1, 3 },
		    .lhs = { 1, 25, 3, 40, 5, 60 },
		    .rhs = { 10, 20, 30 },
		    .expected = { 10.0f, 25.0f, 30.0f, 40.0f, 20.0f, 60.0f },
		},
		Case{
		    .op = BinaryOp::Min,
		    .outputName = "broadcast_min",
		    .lhsShape = { 2, 1 },
		    .rhsShape = { 2, 3 },
		    .lhs = { 7, 18 },
		    .rhs = { 1, 9, 8, 20, 2, 30 },
		    .expected = { 1.0f, 7.0f, 7.0f, 18.0f, 2.0f, 18.0f },
		},
	};

	for (const auto& testCase : cases)
	{
		SCOPED_TRACE(testCase.outputName);
		const std::array outputShape{ 2uz, 3uz };
		auto graph = BuildBinaryGraph(testCase.op, testCase.lhsShape, testCase.rhsShape, outputShape,
		                              std::string(testCase.outputName));
		auto artifact = Compiler<CUDA>::CompileArtifact(graph);
		ASSERT_EQ(artifact.Backend(), CompiledModuleBackend::CUDANative);
		const auto payload = DeserializeCUDANativeInstructionPayload(artifact.Instructions());
		ASSERT_EQ(payload.binaryKind, CUDANativeBinaryKind::PTX);
		EXPECT_NE(BytesToString(payload.binary).find("Generated by LLVM NVPTX Back-End"), std::string::npos);
		auto module = artifact.Load(CUDA{});
		EXPECT_EQ(module.Backend(), CompiledModuleBackend::CUDANative);
		EXPECT_EQ(module.FindOutput(testCase.outputName), 0u);

		auto lhs =
		    Tensor<CPU>(std::span<const double>(testCase.lhs.data(), testCase.lhs.size()),
		                ShapeView{ testCase.lhsShape }, DataType::Float32)
		        .CopyToDevice(CUDA{});
		auto rhs =
		    Tensor<CPU>(std::span<const double>(testCase.rhs.data(), testCase.rhs.size()),
		                ShapeView{ testCase.rhsShape }, DataType::Float32)
		        .CopyToDevice(CUDA{});
		std::array<Tensor<CUDA>, 2> inputs = { std::move(lhs), std::move(rhs) };

		auto outputs = module.Run(inputs);
		ASSERT_EQ(outputs.size(), 1u);
		ExpectTensorNear(outputs[0].CopyToDevice(CPU{}), testCase.expected);

		std::array<Tensor<CUDA>, 1> out = {
			Tensor<CUDA>(Uninitialized, { 2, 3 }, DataType::Float32, CUDA{})
		};
		module.RunInto(inputs, out);
		ExpectTensorNear(out[0].CopyToDevice(CPU{}), testCase.expected);
	}
}

TEST(CompiledModuleCUDATest, RunsNativeElementwiseUnaryOpsWithCUDATensors)
{
	if (!IsCUDADeviceAvailable())
	{
		GTEST_SKIP() << "CUDA device is not available";
	}
	if (!IsCUDADriverAvailable())
	{
		GTEST_SKIP() << "CUDA driver is not available";
	}

	struct Case
	{
		UnaryOp op;
		std::string_view outputName;
		std::array<double, 4> input;
		std::array<float, 4> expected;
		float tolerance{ 1e-6f };
	};
	const std::array cases = {
		Case{ UnaryOp::Negate, "negated", { -4.0, -1.0, 0.0, 9.0 }, { 4.0f, 1.0f, 0.0f, -9.0f } },
		Case{ UnaryOp::Abs, "absolute", { -4.0, -1.0, 0.0, 9.0 }, { 4.0f, 1.0f, 0.0f, 9.0f } },
		Case{ UnaryOp::Sqrt, "sqrt", { 4.0, 1.0, 0.0, 9.0 }, { 2.0f, 1.0f, 0.0f, 3.0f } },
		Case{
		    UnaryOp::Exp,
		    "exp",
		    { 0.0, 1.0, -1.0, 2.0 },
		    { static_cast<float>(std::exp(0.0)), static_cast<float>(std::exp(1.0)),
		      static_cast<float>(std::exp(-1.0)), static_cast<float>(std::exp(2.0)) },
		    2e-3f,
		},
		Case{
		    UnaryOp::Log,
		    "log",
		    { 1.0, 2.0, 4.0, 0.5 },
		    { static_cast<float>(std::log(1.0)), static_cast<float>(std::log(2.0)),
		      static_cast<float>(std::log(4.0)), static_cast<float>(std::log(0.5)) },
		    2e-3f,
		},
		Case{
		    UnaryOp::Sin,
		    "sin",
		    { 0.0, 0.5, -0.5, 1.0 },
		    { static_cast<float>(std::sin(0.0)), static_cast<float>(std::sin(0.5)),
		      static_cast<float>(std::sin(-0.5)), static_cast<float>(std::sin(1.0)) },
		    2e-3f,
		},
		Case{
		    UnaryOp::Cos,
		    "cos",
		    { 0.0, 0.5, -0.5, 1.0 },
		    { static_cast<float>(std::cos(0.0)), static_cast<float>(std::cos(0.5)),
		      static_cast<float>(std::cos(-0.5)), static_cast<float>(std::cos(1.0)) },
		    2e-3f,
		},
	};

	for (const auto& testCase : cases)
	{
		SCOPED_TRACE(testCase.outputName);
		auto graph = BuildSimpleUnaryGraph(testCase.op, std::string(testCase.outputName));
		auto artifact = Compiler<CUDA>::CompileArtifact(graph);
		ASSERT_EQ(artifact.Backend(), CompiledModuleBackend::CUDANative);
		const auto payload = DeserializeCUDANativeInstructionPayload(artifact.Instructions());
		ASSERT_EQ(payload.binaryKind, CUDANativeBinaryKind::PTX);
		EXPECT_NE(BytesToString(payload.binary).find("Generated by LLVM NVPTX Back-End"), std::string::npos);
		auto module = artifact.Load(CUDA{});
		EXPECT_EQ(module.Backend(), CompiledModuleBackend::CUDANative);
		EXPECT_EQ(module.FindInput("input"), 0u);
		EXPECT_EQ(module.FindOutput(testCase.outputName), 0u);

		auto input = Tensor<CPU>({ testCase.input[0], testCase.input[1], testCase.input[2], testCase.input[3] },
		                         { 2, 2 }, DataType::Float32)
		                 .CopyToDevice(CUDA{});
		std::array<Tensor<CUDA>, 1> inputs = { std::move(input) };

		auto outputs = module.Run(inputs);
		ASSERT_EQ(outputs.size(), 1u);
		auto cpuOutput = outputs[0].CopyToDevice(CPU{});
		ExpectTensorNear(cpuOutput, testCase.expected, testCase.tolerance);

		std::array<Tensor<CUDA>, 1> out = {
			Tensor<CUDA>(Uninitialized, { 2, 2 }, DataType::Float32, CUDA{})
		};
		module.RunInto(inputs, out);
		auto cpuOutInto = out[0].CopyToDevice(CPU{});
		ExpectTensorNear(cpuOutInto, testCase.expected, testCase.tolerance);
	}
}

TEST(CompiledModuleCUDATest, MatchesCPUInterpreterAndAOTAcrossNumericalMatrix)
{
	if (!IsCUDADeviceAvailable())
	{
		GTEST_SKIP() << "CUDA device is not available";
	}
	if (!IsCUDADriverAvailable())
	{
		GTEST_SKIP() << "CUDA driver is not available";
	}

	struct Case
	{
		std::string_view name;
		Graph graph;
		std::vector<TensorInputSpec> inputs;
		CompiledModuleBackend expectedCUDABackend{ CompiledModuleBackend::CUDANative };
		float tolerance{ 1e-6f };
	};

	std::vector<Case> cases;
	cases.reserve(22);

	const std::array unaryCases = {
		std::pair{ UnaryOp::Negate, std::string_view{ "unary_negate" } },
		std::pair{ UnaryOp::Abs, std::string_view{ "unary_abs" } },
		std::pair{ UnaryOp::Sqrt, std::string_view{ "unary_sqrt" } },
		std::pair{ UnaryOp::Exp, std::string_view{ "unary_exp" } },
		std::pair{ UnaryOp::Log, std::string_view{ "unary_log" } },
		std::pair{ UnaryOp::Sin, std::string_view{ "unary_sin" } },
		std::pair{ UnaryOp::Cos, std::string_view{ "unary_cos" } },
	};
	for (const auto& [op, name] : unaryCases)
	{
		cases.push_back(Case{
		    .name = name,
		    .graph = BuildSimpleUnaryGraph(op, std::string(name)),
		    .inputs = { TensorInputSpec{ .values = { 0.5, 1.0, 2.0, 4.0 }, .shape = { 2, 2 } } },
		    .tolerance = 2e-3f,
		});
	}

	const std::array binaryCases = {
		std::pair{ BinaryOp::Add, std::string_view{ "binary_add" } },
		std::pair{ BinaryOp::Subtract, std::string_view{ "binary_subtract" } },
		std::pair{ BinaryOp::Multiply, std::string_view{ "binary_multiply" } },
		std::pair{ BinaryOp::Divide, std::string_view{ "binary_divide" } },
		std::pair{ BinaryOp::Max, std::string_view{ "binary_max" } },
		std::pair{ BinaryOp::Min, std::string_view{ "binary_min" } },
	};
	for (const auto& [op, name] : binaryCases)
	{
		cases.push_back(Case{
		    .name = name,
		    .graph = BuildSimpleBinaryGraph(op, std::string(name)),
		    .inputs = { TensorInputSpec{ .values = { 1.0, 2.0, 3.0, 4.0 }, .shape = { 2, 2 } },
		                TensorInputSpec{ .values = { 4.0, 3.0, 2.0, 1.0 }, .shape = { 2, 2 } } },
		});
	}

	const std::array outputShape{ 2uz, 3uz };
	cases.push_back(Case{
	    .name = "broadcast_add",
	    .graph = BuildBinaryGraph(BinaryOp::Add, std::array{ 2uz, 3uz }, std::array{ 1uz, 3uz }, outputShape,
	                              "broadcast_add"),
	    .inputs = { TensorInputSpec{ .values = { 1.0, 2.0, 3.0, 4.0, 5.0, 6.0 }, .shape = { 2, 3 } },
	                TensorInputSpec{ .values = { 10.0, 20.0, 30.0 }, .shape = { 1, 3 } } },
	});
	cases.push_back(Case{
	    .name = "broadcast_subtract",
	    .graph = BuildBinaryGraph(BinaryOp::Subtract, std::array{ 2uz, 3uz }, std::array{ 2uz, 1uz }, outputShape,
	                              "broadcast_subtract"),
	    .inputs = { TensorInputSpec{ .values = { 1.0, 2.0, 3.0, 4.0, 5.0, 6.0 }, .shape = { 2, 3 } },
	                TensorInputSpec{ .values = { 10.0, 20.0 }, .shape = { 2, 1 } } },
	});
	cases.push_back(Case{
	    .name = "broadcast_multiply",
	    .graph = BuildBinaryGraph(BinaryOp::Multiply, std::array{ 1uz, 3uz }, std::array{ 2uz, 3uz }, outputShape,
	                              "broadcast_multiply"),
	    .inputs = { TensorInputSpec{ .values = { 2.0, 3.0, 4.0 }, .shape = { 1, 3 } },
	                TensorInputSpec{ .values = { 1.0, 2.0, 3.0, 4.0, 5.0, 6.0 }, .shape = { 2, 3 } } },
	});
	cases.push_back(Case{
	    .name = "broadcast_divide",
	    .graph = BuildBinaryGraph(BinaryOp::Divide, std::array{ 2uz, 3uz }, std::array{ 1uz, 3uz }, outputShape,
	                              "broadcast_divide"),
	    .inputs = { TensorInputSpec{ .values = { 8.0, 18.0, 32.0, 20.0, 45.0, 80.0 }, .shape = { 2, 3 } },
	                TensorInputSpec{ .values = { 2.0, 3.0, 4.0 }, .shape = { 1, 3 } } },
	});
	cases.push_back(Case{
	    .name = "broadcast_max",
	    .graph = BuildBinaryGraph(BinaryOp::Max, std::array{ 2uz, 3uz }, std::array{ 1uz, 3uz }, outputShape,
	                              "broadcast_max"),
	    .inputs = { TensorInputSpec{ .values = { 1.0, 25.0, 3.0, 40.0, 5.0, 60.0 }, .shape = { 2, 3 } },
	                TensorInputSpec{ .values = { 10.0, 20.0, 30.0 }, .shape = { 1, 3 } } },
	});
	cases.push_back(Case{
	    .name = "broadcast_min",
	    .graph = BuildBinaryGraph(BinaryOp::Min, std::array{ 2uz, 1uz }, std::array{ 2uz, 3uz }, outputShape,
	                              "broadcast_min"),
	    .inputs = { TensorInputSpec{ .values = { 7.0, 18.0 }, .shape = { 2, 1 } },
	                TensorInputSpec{ .values = { 1.0, 9.0, 8.0, 20.0, 2.0, 30.0 }, .shape = { 2, 3 } } },
	});
	cases.push_back(Case{
	    .name = "matmul_cublas",
	    .graph = BuildSimpleMatMulGraph(),
	    .inputs = { TensorInputSpec{ .values = { 1.0, 2.0, 3.0, 4.0 }, .shape = { 2, 2 } },
	                TensorInputSpec{ .values = { 10.0, 20.0, 30.0, 40.0 }, .shape = { 2, 2 } } },
	});
	cases.push_back(Case{
	    .name = "pow_bridge_fallback",
	    .graph = BuildSimplePowGraph(),
	    .inputs = { TensorInputSpec{ .values = { 2.0, 3.0, 4.0, 5.0 }, .shape = { 2, 2 } },
	                TensorInputSpec{ .values = { 3.0, 2.0, 1.0, 0.0 }, .shape = { 2, 2 } } },
	    .expectedCUDABackend = CompiledModuleBackend::CPUNative,
	});

	for (const auto& testCase : cases)
	{
		SCOPED_TRACE(testCase.name);

		auto interpreterInputs = MakeCPUInputs(testCase.inputs);
		Runtime::Interpreter<CPU> interpreter;
		const auto expected = interpreter.RunForward(testCase.graph, interpreterInputs);

		auto cpuAOTInputs = MakeCPUInputs(testCase.inputs);
		auto cpuAOTModule = Compiler<CPU>::CompileArtifact(testCase.graph).Load();
		auto cpuAOTOutputs = cpuAOTModule.Run(cpuAOTInputs);
		ExpectOutputsNear(cpuAOTOutputs, expected, testCase.tolerance);

		auto cudaArtifact = Compiler<CUDA>::CompileArtifact(testCase.graph);
		EXPECT_EQ(cudaArtifact.Backend(), testCase.expectedCUDABackend);
		auto cudaModule = cudaArtifact.Load(CUDA{});
		EXPECT_EQ(cudaModule.Backend(), testCase.expectedCUDABackend);
		auto cudaInputs = MakeCUDAInputs(testCase.inputs);
		auto cudaOutputs = cudaModule.Run(cudaInputs);

		std::vector<Tensor<CPU>> cudaCPUOutputs;
		cudaCPUOutputs.reserve(cudaOutputs.size());
		for (const auto& output : cudaOutputs)
		{
			cudaCPUOutputs.push_back(output.CopyToDevice(CPU{}));
		}
		ExpectOutputsNear(cudaCPUOutputs, expected, testCase.tolerance);
	}
}
