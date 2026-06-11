#include <gtest/gtest.h>

#include <LiteNN.h>
#include <LiteNN/Compiler/CompiledModule.h>
#include <LiteNN/Compiler/VulkanNativeCodegen.h>
#include <LiteNN/Compiler/VulkanNativePayload.h>

#include <array>
#include <cmath>
#include <stdexcept>
#include <string>

using namespace LiteNN;

namespace
{
	Graph BuildSimpleBinaryGraph(BinaryOp op)
	{
		Graph graph;
		Subgraph sg;
		const auto lhs = sg.AddParam(DataType::Float32, { 4 });
		const auto rhs = sg.AddParam(DataType::Float32, { 4 });
		const auto out = sg.AddNode(BinaryOpNode{ op, { lhs, 0 }, { rhs, 0 } },
		                            { OutputInfo{ DataType::Float32, { 4 } } });
		sg.SetResults({ { out, 0 } });
		graph.AddSubgraph(std::move(sg));
		graph.SetForward(0);
		graph.SetInputNames({ "lhs", "rhs" });
		graph.SetOutputNames({ "sum" });
		return graph;
	}

	Graph BuildSimpleUnaryGraph(UnaryOp op)
	{
		Graph graph;
		Subgraph sg;
		const auto input = sg.AddParam(DataType::Float32, { 4 });
		const auto out =
		    sg.AddNode(UnaryOpNode{ op, { input, 0 } }, { OutputInfo{ DataType::Float32, { 4 } } });
		sg.SetResults({ { out, 0 } });
		graph.AddSubgraph(std::move(sg));
		graph.SetForward(0);
		graph.SetInputNames({ "input" });
		graph.SetOutputNames({ "output" });
		return graph;
	}

	Graph BuildSimpleCastGraph(DataType srcType, DataType dstType)
	{
		Graph graph;
		Subgraph sg;
		const auto input = sg.AddParam(srcType, { 4 });
		const auto out = sg.AddNode(CastNode{ { input, 0 }, dstType }, { OutputInfo{ dstType, { 4 } } });
		sg.SetResults({ { out, 0 } });
		graph.AddSubgraph(std::move(sg));
		graph.SetForward(0);
		graph.SetInputNames({ "input" });
		graph.SetOutputNames({ "output" });
		return graph;
	}

	std::array<float, 4> CopyToHost(const Tensor<Vulkan>& tensor)
	{
		Tensor<CPU> host(Uninitialized, tensor.Shape(), tensor.DType(), CPU{});
		auto device = tensor.CurDevice();
		DeviceTraits<Vulkan>::CopyToCPU(device, tensor.DType(), tensor.UnsafeRawData(), tensor.NumElements(),
		                                host.DType(), host.UnsafeRawData());
		const auto* values = static_cast<const float*>(host.UnsafeRawData());
		return { values[0], values[1], values[2], values[3] };
	}

	std::array<float, 4> CopyToHostAsFloat32(const Tensor<Vulkan>& tensor)
	{
		Tensor<CPU> host(Uninitialized, tensor.Shape(), DataType::Float32, CPU{});
		auto device = tensor.CurDevice();
		DeviceTraits<Vulkan>::CopyToCPU(device, tensor.DType(), tensor.UnsafeRawData(), tensor.NumElements(),
		                                host.DType(), host.UnsafeRawData());
		const auto* values = static_cast<const float*>(host.UnsafeRawData());
		return { values[0], values[1], values[2], values[3] };
	}

	struct BinaryCase
	{
		BinaryOp op;
		std::string_view mlirOp;
		std::array<float, 4> expected;
	};

	struct UnaryCase
	{
		UnaryOp op;
		std::string_view mlirOp;
		std::array<double, 4> input;
		float tolerance;
	};

	struct CastCase
	{
		DataType srcType;
		DataType dstType;
		std::string_view mlirOp;
		std::array<double, 4> input;
		std::array<float, 4> expected;
	};

	constexpr std::array kBinaryCases{
		BinaryCase{ BinaryOp::Add, "spirv.FAdd", { 11.0f, 22.0f, 33.0f, 44.0f } },
		BinaryCase{ BinaryOp::Subtract, "spirv.FSub", { -9.0f, -18.0f, -27.0f, -36.0f } },
		BinaryCase{ BinaryOp::Multiply, "spirv.FMul", { 10.0f, 40.0f, 90.0f, 160.0f } },
		BinaryCase{ BinaryOp::Divide, "spirv.FDiv", { 0.1f, 0.1f, 0.1f, 0.1f } },
		BinaryCase{ BinaryOp::Max, "spirv.GL.FMax", { 10.0f, 20.0f, 30.0f, 40.0f } },
		BinaryCase{ BinaryOp::Min, "spirv.GL.FMin", { 1.0f, 2.0f, 3.0f, 4.0f } },
	};

	constexpr std::array kUnaryCases{
		UnaryCase{ UnaryOp::Negate, "spirv.FNegate", { -4.0f, -1.0f, 0.0f, 9.0f }, 0.0f },
		UnaryCase{ UnaryOp::Abs, "spirv.GL.FAbs", { -4.0f, -1.0f, 0.0f, 9.0f }, 0.0f },
		UnaryCase{ UnaryOp::Sqrt, "spirv.GL.Sqrt", { 4.0f, 1.0f, 0.25f, 9.0f }, 1e-5f },
		UnaryCase{ UnaryOp::Exp, "spirv.GL.Exp", { -1.0f, 0.0f, 1.0f, 2.0f }, 1e-4f },
		UnaryCase{ UnaryOp::Log, "spirv.GL.Log", { 0.25f, 1.0f, 2.0f, 4.0f }, 1e-4f },
		UnaryCase{ UnaryOp::Sin, "spirv.GL.Sin", { -1.0f, 0.0f, 1.0f, 2.0f }, 1e-4f },
		UnaryCase{ UnaryOp::Cos, "spirv.GL.Cos", { -1.0f, 0.0f, 1.0f, 2.0f }, 1e-4f },
	};

	constexpr std::array kCastCases{
		CastCase{ DataType::Float32,
		          DataType::Int32,
		          "spirv.ConvertFToS",
		          { -3.5, -1.0, 0.75, 4.0 },
		          { -3.0f, -1.0f, 0.0f, 4.0f } },
		CastCase{ DataType::Int32,
		          DataType::Float32,
		          "spirv.ConvertSToF",
		          { -3.0, -1.0, 0.0, 4.0 },
		          { -3.0f, -1.0f, 0.0f, 4.0f } },
		CastCase{ DataType::Float32,
		          DataType::Float16,
		          "spirv.FConvert",
		          { -3.5, -1.0, 0.75, 4.0 },
		          { -3.5f, -1.0f, 0.75f, 4.0f } },
		CastCase{ DataType::Float16,
		          DataType::Float32,
		          "spirv.FConvert",
		          { -3.5, -1.0, 0.75, 4.0 },
		          { -3.5f, -1.0f, 0.75f, 4.0f } },
		CastCase{ DataType::Float32,
		          DataType::Int8,
		          "spirv.ConvertFToS",
		          { -3.5, -1.0, 0.75, 4.0 },
		          { -3.0f, -1.0f, 0.0f, 4.0f } },
		CastCase{ DataType::Int8,
		          DataType::Float32,
		          "spirv.ConvertSToF",
		          { -3.0, -1.0, 0.0, 4.0 },
		          { -3.0f, -1.0f, 0.0f, 4.0f } },
		CastCase{ DataType::Float32,
		          DataType::UInt8,
		          "spirv.ConvertFToU",
		          { 0.0, 1.0, 2.75, 4.0 },
		          { 0.0f, 1.0f, 2.0f, 4.0f } },
		CastCase{ DataType::UInt8,
		          DataType::Float32,
		          "spirv.ConvertUToF",
		          { 0.0, 1.0, 2.0, 4.0 },
		          { 0.0f, 1.0f, 2.0f, 4.0f } },
		CastCase{ DataType::Int32,
		          DataType::Int8,
		          "spirv.SConvert",
		          { -3.0, -1.0, 0.0, 4.0 },
		          { -3.0f, -1.0f, 0.0f, 4.0f } },
		CastCase{ DataType::UInt8,
		          DataType::Int32,
		          "spirv.UConvert",
		          { 0.0, 1.0, 2.0, 4.0 },
		          { 0.0f, 1.0f, 2.0f, 4.0f } },
	};

	constexpr std::array kRuntimeCastCases{
		kCastCases[0],
		kCastCases[1],
	};

	float ExpectedUnaryValue(UnaryOp op, double value)
	{
		switch (op)
		{
		case UnaryOp::Negate:
			return -value;
		case UnaryOp::Abs:
			return std::fabs(value);
		case UnaryOp::Sqrt:
			return std::sqrt(value);
		case UnaryOp::Exp:
			return std::exp(value);
		case UnaryOp::Log:
			return std::log(value);
		case UnaryOp::Sin:
			return std::sin(value);
		case UnaryOp::Cos:
			return std::cos(value);
		default:
			throw std::runtime_error("Unexpected unary test op");
		}
	}
}

TEST(CompiledModuleVulkanTest, GeneratesSimpleAddSPIRVFromMLIR)
{
	for (const auto& item : kBinaryCases)
	{
		const auto generated = VulkanNativeSameShapeBinaryF32SPIRV(item.op);
		EXPECT_FALSE(generated.words.empty());
		EXPECT_NE(generated.mlir.find("spirv.module"), std::string::npos);
		EXPECT_NE(generated.mlir.find(item.mlirOp), std::string::npos);
		EXPECT_NE(generated.mlir.find("spirv.EntryPoint"), std::string::npos);
	}
}

TEST(CompiledModuleVulkanTest, GeneratesSimpleUnarySPIRVFromMLIR)
{
	for (const auto& item : kUnaryCases)
	{
		const auto generated = VulkanNativeSameShapeUnaryF32SPIRV(item.op);
		EXPECT_FALSE(generated.words.empty());
		EXPECT_NE(generated.mlir.find("spirv.module"), std::string::npos);
		EXPECT_NE(generated.mlir.find(item.mlirOp), std::string::npos);
		EXPECT_NE(generated.mlir.find("spirv.EntryPoint"), std::string::npos);
	}
}

TEST(CompiledModuleVulkanTest, GeneratesSimpleCastSPIRVFromMLIR)
{
	for (const auto& item : kCastCases)
	{
		const auto generated = VulkanNativeSameShapeCastSPIRV(item.srcType, item.dstType);
		EXPECT_FALSE(generated.words.empty());
		EXPECT_NE(generated.mlir.find("spirv.module"), std::string::npos);
		EXPECT_NE(generated.mlir.find(item.mlirOp), std::string::npos);
		EXPECT_NE(generated.mlir.find("spirv.EntryPoint"), std::string::npos);
	}
}

TEST(CompiledModuleVulkanTest, WritesVulkanNativePayloadForSimpleAdd)
{
	const auto graph = BuildSimpleBinaryGraph(BinaryOp::Add);
	const auto artifact = Compiler<Vulkan>::CompileArtifact(Detail::BuildExecutablePlanFromGraph(graph));
	EXPECT_EQ(artifact.Backend(), CompiledModuleBackend::VulkanNative);
	EXPECT_FALSE(artifact.Instructions().empty());

	const auto payload = DeserializeVulkanNativeInstructionPayload(artifact.Instructions());
	const auto generated = VulkanNativeSameShapeBinaryF32SPIRV(BinaryOp::Add);
	EXPECT_EQ(payload.spirv, generated.words);
}

TEST(CompiledModuleVulkanTest, WritesVulkanNativePayloadForSimpleUnary)
{
	const auto graph = BuildSimpleUnaryGraph(UnaryOp::Sqrt);
	const auto artifact = Compiler<Vulkan>::CompileArtifact(Detail::BuildExecutablePlanFromGraph(graph));
	EXPECT_EQ(artifact.Backend(), CompiledModuleBackend::VulkanNative);
	EXPECT_FALSE(artifact.Instructions().empty());

	const auto payload = DeserializeVulkanNativeInstructionPayload(artifact.Instructions());
	const auto generated = VulkanNativeSameShapeUnaryF32SPIRV(UnaryOp::Sqrt);
	EXPECT_EQ(payload.spirv, generated.words);
}

TEST(CompiledModuleVulkanTest, WritesVulkanNativePayloadForSimpleCast)
{
	const auto graph = BuildSimpleCastGraph(DataType::Float32, DataType::Int32);
	const auto artifact = Compiler<Vulkan>::CompileArtifact(Detail::BuildExecutablePlanFromGraph(graph));
	EXPECT_EQ(artifact.Backend(), CompiledModuleBackend::VulkanNative);
	EXPECT_FALSE(artifact.Instructions().empty());

	const auto payload = DeserializeVulkanNativeInstructionPayload(artifact.Instructions());
	const auto generated = VulkanNativeSameShapeCastSPIRV(DataType::Float32, DataType::Int32);
	EXPECT_EQ(payload.spirv, generated.words);
}

TEST(CompiledModuleVulkanTest, WritesVulkanNativePayloadForLowPrecisionCast)
{
	const auto graph = BuildSimpleCastGraph(DataType::Float32, DataType::Float16);
	const auto artifact = Compiler<Vulkan>::CompileArtifact(Detail::BuildExecutablePlanFromGraph(graph));
	EXPECT_EQ(artifact.Backend(), CompiledModuleBackend::VulkanNative);
	EXPECT_FALSE(artifact.Instructions().empty());

	const auto payload = DeserializeVulkanNativeInstructionPayload(artifact.Instructions());
	const auto generated = VulkanNativeSameShapeCastSPIRV(DataType::Float32, DataType::Float16);
	EXPECT_EQ(payload.spirv, generated.words);
	EXPECT_NE(payload.featureSet.flags & (1ull << static_cast<std::uint32_t>(VulkanNativeFeature::SameShapeCastLowPrecision)),
	          0ull);
}

TEST(CompiledModuleVulkanTest, LoadsSeparatedArtifactForSimpleAdd)
{
	if (!IsVulkanDeviceAvailable())
	{
		GTEST_SKIP() << "No Vulkan compute device is available";
	}

	const auto graph = BuildSimpleBinaryGraph(BinaryOp::Add);
	const auto artifact = Compiler<Vulkan>::CompileArtifact(Detail::BuildExecutablePlanFromGraph(graph));
	auto separated = artifact.SeparateRodata();
	EXPECT_EQ(separated.Backend(), CompiledModuleBackend::VulkanNative);
	EXPECT_GT(separated.Metadata().size(), 0u);
	EXPECT_GT(separated.Instructions().size(), 0u);

	auto module = separated.LoadBorrowedExternalRegions(Vulkan{});
	ASSERT_EQ(module.Backend(), CompiledModuleBackend::VulkanNative);

	Vulkan device;
	std::array inputs{
		Tensor<Vulkan>({ 1.0, 2.0, 3.0, 4.0 }, { 4 }, DataType::Float32, device),
		Tensor<Vulkan>({ 10.0, 20.0, 30.0, 40.0 }, { 4 }, DataType::Float32, device),
	};
	auto outputs = module.RunTensors(std::span<const Tensor<Vulkan>>(inputs));
	ASSERT_EQ(outputs.size(), 1);

	const auto actual = CopyToHost(outputs[0]);
	const std::array expected{ 11.0f, 22.0f, 33.0f, 44.0f };
	for (std::size_t i = 0; i < expected.size(); ++i)
	{
		EXPECT_FLOAT_EQ(actual[i], expected[i]);
	}
}

TEST(CompiledModuleVulkanTest, RunsSimpleCastArithmetic)
{
	if (!IsVulkanDeviceAvailable())
	{
		GTEST_SKIP() << "No Vulkan compute device is available";
	}

	for (const auto& item : kRuntimeCastCases)
	{
		const auto graph = BuildSimpleCastGraph(item.srcType, item.dstType);
		auto module = Compiler<Vulkan>::Compile(Detail::BuildExecutablePlanFromGraph(graph), Vulkan{});
		ASSERT_EQ(module.Backend(), CompiledModuleBackend::VulkanNative);

		Vulkan device;
		std::array inputs{
			Tensor<Vulkan>(item.input, { 4 }, item.srcType, device),
		};
		auto outputs = module.RunTensors(std::span<const Tensor<Vulkan>>(inputs));
		ASSERT_EQ(outputs.size(), 1);
		EXPECT_EQ(outputs[0].DType(), item.dstType);

		const auto actual = CopyToHostAsFloat32(outputs[0]);
		for (std::size_t i = 0; i < item.expected.size(); ++i)
		{
			EXPECT_FLOAT_EQ(actual[i], item.expected[i]);
		}
	}
}

TEST(CompiledModuleVulkanTest, RunsSimpleUnaryArithmetic)
{
	if (!IsVulkanDeviceAvailable())
	{
		GTEST_SKIP() << "No Vulkan compute device is available";
	}

	for (const auto& item : kUnaryCases)
	{
		const auto graph = BuildSimpleUnaryGraph(item.op);
		auto module = Compiler<Vulkan>::Compile(Detail::BuildExecutablePlanFromGraph(graph), Vulkan{});
		ASSERT_EQ(module.Backend(), CompiledModuleBackend::VulkanNative);

		Vulkan device;
		std::array inputs{
			Tensor<Vulkan>(item.input, { 4 }, DataType::Float32, device),
		};
		auto outputs = module.RunTensors(std::span<const Tensor<Vulkan>>(inputs));
		ASSERT_EQ(outputs.size(), 1);

		const auto actual = CopyToHost(outputs[0]);
		for (std::size_t i = 0; i < item.input.size(); ++i)
		{
			EXPECT_NEAR(actual[i], ExpectedUnaryValue(item.op, item.input[i]), item.tolerance);
		}
	}
}

TEST(CompiledModuleVulkanTest, RunsSimpleBinaryArithmetic)
{
	if (!IsVulkanDeviceAvailable())
	{
		GTEST_SKIP() << "No Vulkan compute device is available";
	}

	for (const auto& item : kBinaryCases)
	{
		const auto graph = BuildSimpleBinaryGraph(item.op);
		auto module = Compiler<Vulkan>::Compile(Detail::BuildExecutablePlanFromGraph(graph), Vulkan{});
		ASSERT_EQ(module.Backend(), CompiledModuleBackend::VulkanNative);

		Vulkan device;
		std::array inputs{
			Tensor<Vulkan>({ 1.0, 2.0, 3.0, 4.0 }, { 4 }, DataType::Float32, device),
			Tensor<Vulkan>({ 10.0, 20.0, 30.0, 40.0 }, { 4 }, DataType::Float32, device),
		};
		auto outputs = module.RunTensors(std::span<const Tensor<Vulkan>>(inputs));
		ASSERT_EQ(outputs.size(), 1);

		const auto actual = CopyToHost(outputs[0]);
		for (std::size_t i = 0; i < item.expected.size(); ++i)
		{
			EXPECT_FLOAT_EQ(actual[i], item.expected[i]);
		}
	}
}
