#include <gtest/gtest.h>

#include <LiteNN.h>
#include <LiteNN/Compiler/CompiledModule.h>
#include <LiteNN/Compiler/CUDANativePayload.h>

#include <array>
#include <cmath>
#include <cstddef>
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

	void ExpectTensorNear(const Tensor<CPU>& tensor, std::span<const float> expected, float tolerance = 1e-6f)
	{
		ASSERT_EQ(tensor.NumElements(), expected.size());
		for (std::size_t i = 0; i < expected.size(); ++i)
		{
			EXPECT_NEAR(ReadFloat(tensor, i), expected[i], tolerance);
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
} // namespace

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
