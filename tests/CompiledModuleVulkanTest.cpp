#include <gtest/gtest.h>

#include <LiteNN.h>
#include <LiteNN/Compiler/CompiledModule.h>
#include <LiteNN/Compiler/VulkanNativeCodegen.h>
#include <LiteNN/Compiler/VulkanNativePayload.h>
#include <LiteNN/Pass/FusionPass.h>

#include <array>
#include <cmath>
#include <stdexcept>
#include <string>
#include <vector>

using namespace LiteNN;

namespace
{
	constexpr std::uint32_t kElementCount = 4;

	Graph BuildSimpleBinaryGraph(BinaryOp op, std::size_t elementCount = kElementCount)
	{
		Graph graph;
		Subgraph sg;
		const auto lhs = sg.AddParam(DataType::Float32, { elementCount });
		const auto rhs = sg.AddParam(DataType::Float32, { elementCount });
		const auto out = sg.AddNode(BinaryOpNode{ op, { lhs, 0 }, { rhs, 0 } },
		                            { OutputInfo{ DataType::Float32, { elementCount } } });
		sg.SetResults({ { out, 0 } });
		graph.AddSubgraph(std::move(sg));
		graph.SetForward(0);
		graph.SetInputNames({ "lhs", "rhs" });
		graph.SetOutputNames({ "sum" });
		return graph;
	}

	Graph BuildSimpleMatMulGraph()
	{
		Graph graph;
		Subgraph sg;
		const auto lhs = sg.AddParam(DataType::Float32, { 2, 3 });
		const auto rhs = sg.AddParam(DataType::Float32, { 3, 4 });
		const auto out = sg.AddNode(BinaryOpNode{ BinaryOp::MatMul, { lhs, 0 }, { rhs, 0 } },
		                            { OutputInfo{ DataType::Float32, { 2, 4 } } });
		sg.SetResults({ { out, 0 } });
		graph.AddSubgraph(std::move(sg));
		graph.SetForward(0);
		graph.SetInputNames({ "lhs", "rhs" });
		graph.SetOutputNames({ "out" });
		return graph;
	}

	Graph BuildSimpleMatMulBiasGraph(bool relu)
	{
		Graph graph;
		Subgraph sg;
		const auto lhs = sg.AddParam(DataType::Float32, { 2, 3 });
		const auto rhs = sg.AddParam(DataType::Float32, { 3, 4 });
		const auto bias = sg.AddParam(DataType::Float32, { 1, 4 });
		const auto matmul = sg.AddNode(BinaryOpNode{ BinaryOp::MatMul, { lhs, 0 }, { rhs, 0 } },
		                               { OutputInfo{ DataType::Float32, { 2, 4 } } });
		const auto shifted = sg.AddNode(BinaryOpNode{ BinaryOp::Add, { matmul, 0 }, { bias, 0 } },
		                                { OutputInfo{ DataType::Float32, { 2, 4 } } });
		NodeOutput result{ shifted, 0 };
		if (relu)
		{
			Tensor<CPU> zero({ 0.0f }, { 1, 1 }, DataType::Float32);
			const auto zeroNode = sg.AddNode(ConstantNode{ zero.CopyToDevice(PolymorphicDevice{ CPU{} }) },
			                                 { OutputInfo{ DataType::Float32, { 1, 1 } } });
			const auto reluOut = sg.AddNode(BinaryOpNode{ BinaryOp::Max, { shifted, 0 }, { zeroNode, 0 } },
			                                { OutputInfo{ DataType::Float32, { 2, 4 } } });
			result = { reluOut, 0 };
		}
		sg.SetResults({ result });
		graph.AddSubgraph(std::move(sg));
		graph.SetForward(0);
		graph.SetInputNames({ "lhs", "rhs", "bias" });
		graph.SetOutputNames({ relu ? "relu" : "out" });
		FusionPass{}.Run(graph);
		return graph;
	}

	Graph BuildSimpleMatMulBiasVariableGraph(bool relu)
	{
		Graph graph;
		const auto weightIndex = graph.AddVariable(Variable::Create(Tensor<CPU>(
		    { 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0, 12.0 }, { 3, 4 }, DataType::Float32)));
		const auto biasIndex = graph.AddVariable(
		    Variable::Create(Tensor<CPU>({ 1.0, -100.0, 3.0, -200.0 }, { 1, 4 }, DataType::Float32)));
		graph.SetVariableName(weightIndex, "weight");
		graph.SetVariableName(biasIndex, "bias");

		Subgraph sg;
		const auto lhs = sg.AddParam(DataType::Float32, { 2, 3 });
		const auto weight = sg.AddNode(VariableRefNode{ weightIndex }, { OutputInfo{ DataType::Float32, { 3, 4 } } });
		const auto matmul = sg.AddNode(BinaryOpNode{ BinaryOp::MatMul, { lhs, 0 }, { weight, 0 } },
		                               { OutputInfo{ DataType::Float32, { 2, 4 } } });
		const auto bias = sg.AddNode(VariableRefNode{ biasIndex }, { OutputInfo{ DataType::Float32, { 1, 4 } } });
		const auto shifted = sg.AddNode(BinaryOpNode{ BinaryOp::Add, { matmul, 0 }, { bias, 0 } },
		                                { OutputInfo{ DataType::Float32, { 2, 4 } } });
		NodeOutput result{ shifted, 0 };
		if (relu)
		{
			Tensor<CPU> zero({ 0.0f }, { 1, 1 }, DataType::Float32);
			const auto zeroNode = sg.AddNode(ConstantNode{ zero.CopyToDevice(PolymorphicDevice{ CPU{} }) },
			                                 { OutputInfo{ DataType::Float32, { 1, 1 } } });
			const auto reluOut = sg.AddNode(BinaryOpNode{ BinaryOp::Max, { shifted, 0 }, { zeroNode, 0 } },
			                                { OutputInfo{ DataType::Float32, { 2, 4 } } });
			result = { reluOut, 0 };
		}
		sg.SetResults({ result });
		graph.AddSubgraph(std::move(sg));
		graph.SetForward(0);
		graph.SetInputNames({ "lhs" });
		graph.SetOutputNames({ relu ? "relu" : "out" });
		FusionPass{}.Run(graph);
		return graph;
	}

	Graph BuildSimpleUnaryGraph(UnaryOp op)
	{
		Graph graph;
		Subgraph sg;
		const auto input = sg.AddParam(DataType::Float32, { 4 });
		const auto out = sg.AddNode(UnaryOpNode{ op, { input, 0 } }, { OutputInfo{ DataType::Float32, { 4 } } });
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

	std::vector<float> CopyToHostVector(const Tensor<Vulkan>& tensor)
	{
		Tensor<CPU> host(Uninitialized, tensor.Shape(), DataType::Float32, CPU{});
		auto device = tensor.CurDevice();
		DeviceTraits<Vulkan>::CopyToCPU(device, tensor.DType(), tensor.UnsafeRawData(), tensor.NumElements(),
		                                host.DType(), host.UnsafeRawData());
		const auto* values = static_cast<const float*>(host.UnsafeRawData());
		return std::vector<float>(values, values + host.NumElements());
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
		CastCase{
		    DataType::Int32, DataType::Int8, "spirv.SConvert", { -3.0, -1.0, 0.0, 4.0 }, { -3.0f, -1.0f, 0.0f, 4.0f } },
		CastCase{
		    DataType::UInt8, DataType::Int32, "spirv.UConvert", { 0.0, 1.0, 2.0, 4.0 }, { 0.0f, 1.0f, 2.0f, 4.0f } },
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
} // namespace

TEST(CompiledModuleVulkanTest, GeneratesSimpleAddSPIRVFromMLIR)
{
	for (const auto& item : kBinaryCases)
	{
		const auto generated = VulkanNativeSameShapeBinaryF32SPIRV(item.op, kElementCount);
		EXPECT_FALSE(generated.words.empty());
		EXPECT_NE(generated.mlir.find("spirv.module"), std::string::npos);
		EXPECT_NE(generated.mlir.find(item.mlirOp), std::string::npos);
		EXPECT_NE(generated.mlir.find("spirv.ULessThan"), std::string::npos);
		EXPECT_NE(generated.mlir.find("spirv.mlir.selection"), std::string::npos);
		EXPECT_NE(generated.mlir.find("spirv.EntryPoint"), std::string::npos);
		EXPECT_NE(generated.mlir.find("LocalSize"), std::string::npos);
		EXPECT_NE(generated.mlir.find("64, 1, 1"), std::string::npos);
	}
}

TEST(CompiledModuleVulkanTest, GeneratesSimpleUnarySPIRVFromMLIR)
{
	for (const auto& item : kUnaryCases)
	{
		const auto generated = VulkanNativeSameShapeUnaryF32SPIRV(item.op, kElementCount);
		EXPECT_FALSE(generated.words.empty());
		EXPECT_NE(generated.mlir.find("spirv.module"), std::string::npos);
		EXPECT_NE(generated.mlir.find(item.mlirOp), std::string::npos);
		EXPECT_NE(generated.mlir.find("spirv.ULessThan"), std::string::npos);
		EXPECT_NE(generated.mlir.find("spirv.mlir.selection"), std::string::npos);
		EXPECT_NE(generated.mlir.find("spirv.EntryPoint"), std::string::npos);
		EXPECT_NE(generated.mlir.find("LocalSize"), std::string::npos);
		EXPECT_NE(generated.mlir.find("64, 1, 1"), std::string::npos);
	}
}

TEST(CompiledModuleVulkanTest, GeneratesSimpleCastSPIRVFromMLIR)
{
	for (const auto& item : kCastCases)
	{
		const auto generated = VulkanNativeSameShapeCastSPIRV(item.srcType, item.dstType, kElementCount);
		EXPECT_FALSE(generated.words.empty());
		EXPECT_NE(generated.mlir.find("spirv.module"), std::string::npos);
		EXPECT_NE(generated.mlir.find(item.mlirOp), std::string::npos);
		EXPECT_NE(generated.mlir.find("spirv.ULessThan"), std::string::npos);
		EXPECT_NE(generated.mlir.find("spirv.mlir.selection"), std::string::npos);
		EXPECT_NE(generated.mlir.find("spirv.EntryPoint"), std::string::npos);
		EXPECT_NE(generated.mlir.find("LocalSize"), std::string::npos);
		EXPECT_NE(generated.mlir.find("64, 1, 1"), std::string::npos);
	}
}

TEST(CompiledModuleVulkanTest, GeneratesSimpleMatMulSPIRVFromMLIR)
{
	const auto generated = VulkanNativeMatMulF32SPIRV(2, 3, 4);
	EXPECT_FALSE(generated.words.empty());
	EXPECT_NE(generated.mlir.find("spirv.module"), std::string::npos);
	EXPECT_NE(generated.mlir.find("spirv.FMul"), std::string::npos);
	EXPECT_NE(generated.mlir.find("spirv.FAdd"), std::string::npos);
	EXPECT_NE(generated.mlir.find("spirv.UDiv"), std::string::npos);
	EXPECT_NE(generated.mlir.find("spirv.UMod"), std::string::npos);
	EXPECT_NE(generated.mlir.find("spirv.EntryPoint"), std::string::npos);
	EXPECT_NE(generated.mlir.find("LocalSize"), std::string::npos);
	EXPECT_NE(generated.mlir.find("64, 1, 1"), std::string::npos);
}

TEST(CompiledModuleVulkanTest, GeneratesSimpleMatMulBiasReLUSPIRVFromMLIR)
{
	const auto generated = VulkanNativeMatMulBiasF32SPIRV(2, 3, 4, 1, true);
	EXPECT_FALSE(generated.words.empty());
	EXPECT_NE(generated.mlir.find("spirv.module"), std::string::npos);
	EXPECT_NE(generated.mlir.find("spirv.FMul"), std::string::npos);
	EXPECT_NE(generated.mlir.find("spirv.FAdd"), std::string::npos);
	EXPECT_NE(generated.mlir.find("spirv.GL.FMax"), std::string::npos);
	EXPECT_NE(generated.mlir.find("spirv.EntryPoint"), std::string::npos);
	EXPECT_NE(generated.mlir.find("LocalSize"), std::string::npos);
	EXPECT_NE(generated.mlir.find("64, 1, 1"), std::string::npos);
}

TEST(CompiledModuleVulkanTest, WritesVulkanNativePayloadForSimpleAdd)
{
	const auto graph = BuildSimpleBinaryGraph(BinaryOp::Add);
	const auto artifact = Compiler<Vulkan>::CompileArtifact(Detail::BuildExecutablePlanFromGraph(graph));
	EXPECT_EQ(artifact.Backend(), CompiledModuleBackend::VulkanNative);
	EXPECT_FALSE(artifact.Instructions().empty());

	const auto payload = DeserializeVulkanNativeInstructionPayload(artifact.Instructions());
	const auto generated = VulkanNativeSameShapeBinaryF32SPIRV(BinaryOp::Add, kElementCount);
	EXPECT_EQ(payload.spirv, generated.words);
	ASSERT_EQ(payload.kernels.size(), 1u);
	EXPECT_EQ(payload.kernels[0].groups.x, 1u);
}

TEST(CompiledModuleVulkanTest, ReportsNativeSupportForSimpleAdd)
{
	const auto graph = BuildSimpleBinaryGraph(BinaryOp::Add);
	const auto report = Compiler<Vulkan>::QueryNativeSupport(Detail::BuildExecutablePlanFromGraph(graph));

	EXPECT_TRUE(report.supported);
	EXPECT_NE(report.capability.find("same-shape f32 binary"), std::string::npos);
	EXPECT_NE(report.capability.find("Add"), std::string::npos);
	EXPECT_TRUE(report.reason.empty());
}

TEST(CompiledModuleVulkanTest, ReportsNativeSupportForMatMul)
{
	const auto graph = BuildSimpleMatMulGraph();
	const auto report = Compiler<Vulkan>::QueryNativeSupport(Detail::BuildExecutablePlanFromGraph(graph));

	EXPECT_TRUE(report.supported);
	EXPECT_NE(report.capability.find("f32 matmul"), std::string::npos);
	EXPECT_TRUE(report.reason.empty());
}

TEST(CompiledModuleVulkanTest, ReportsNativeSupportForMatMulBiasReLU)
{
	const auto graph = BuildSimpleMatMulBiasGraph(true);
	const auto report = Compiler<Vulkan>::QueryNativeSupport(Detail::BuildExecutablePlanFromGraph(graph));

	EXPECT_TRUE(report.supported);
	EXPECT_NE(report.capability.find("f32 matmul bias relu"), std::string::npos);
	EXPECT_TRUE(report.reason.empty());
}

TEST(CompiledModuleVulkanTest, ReportsNativeSupportForMatMulBiasExternalWeights)
{
	const auto graph = BuildSimpleMatMulBiasVariableGraph(true);
	const auto report = Compiler<Vulkan>::QueryNativeSupport(Detail::BuildExecutablePlanFromGraph(graph));

	EXPECT_TRUE(report.supported);
	EXPECT_NE(report.capability.find("f32 matmul bias relu"), std::string::npos);
	EXPECT_TRUE(report.reason.empty());
}

TEST(CompiledModuleVulkanTest, UsesTunedWorkgroupDispatchForElementwisePayload)
{
	const auto graph = BuildSimpleBinaryGraph(BinaryOp::Add, kVulkanNativeElementwiseWorkgroupSize + 1);
	const auto artifact = Compiler<Vulkan>::CompileArtifact(Detail::BuildExecutablePlanFromGraph(graph));
	EXPECT_EQ(artifact.Backend(), CompiledModuleBackend::VulkanNative);

	const auto payload = DeserializeVulkanNativeInstructionPayload(artifact.Instructions());
	ASSERT_EQ(payload.kernels.size(), 1u);
	EXPECT_EQ(payload.kernels[0].groups.x, 2u);
	const auto generated =
	    VulkanNativeSameShapeBinaryF32SPIRV(BinaryOp::Add, kVulkanNativeElementwiseWorkgroupSize + 1);
	EXPECT_EQ(payload.spirv, generated.words);
}

TEST(CompiledModuleVulkanTest, WritesVulkanNativePayloadForSimpleUnary)
{
	const auto graph = BuildSimpleUnaryGraph(UnaryOp::Sqrt);
	const auto artifact = Compiler<Vulkan>::CompileArtifact(Detail::BuildExecutablePlanFromGraph(graph));
	EXPECT_EQ(artifact.Backend(), CompiledModuleBackend::VulkanNative);
	EXPECT_FALSE(artifact.Instructions().empty());

	const auto payload = DeserializeVulkanNativeInstructionPayload(artifact.Instructions());
	const auto generated = VulkanNativeSameShapeUnaryF32SPIRV(UnaryOp::Sqrt, kElementCount);
	EXPECT_EQ(payload.spirv, generated.words);
	ASSERT_EQ(payload.kernels.size(), 1u);
	EXPECT_EQ(payload.kernels[0].groups.x, 1u);
}

TEST(CompiledModuleVulkanTest, WritesVulkanNativePayloadForSimpleCast)
{
	const auto graph = BuildSimpleCastGraph(DataType::Float32, DataType::Int32);
	const auto artifact = Compiler<Vulkan>::CompileArtifact(Detail::BuildExecutablePlanFromGraph(graph));
	EXPECT_EQ(artifact.Backend(), CompiledModuleBackend::VulkanNative);
	EXPECT_FALSE(artifact.Instructions().empty());

	const auto payload = DeserializeVulkanNativeInstructionPayload(artifact.Instructions());
	const auto generated = VulkanNativeSameShapeCastSPIRV(DataType::Float32, DataType::Int32, kElementCount);
	EXPECT_EQ(payload.spirv, generated.words);
	ASSERT_EQ(payload.kernels.size(), 1u);
	EXPECT_EQ(payload.kernels[0].groups.x, 1u);
}

TEST(CompiledModuleVulkanTest, WritesVulkanNativePayloadForSimpleMatMul)
{
	const auto graph = BuildSimpleMatMulGraph();
	const auto artifact = Compiler<Vulkan>::CompileArtifact(Detail::BuildExecutablePlanFromGraph(graph));
	EXPECT_EQ(artifact.Backend(), CompiledModuleBackend::VulkanNative);
	EXPECT_FALSE(artifact.Instructions().empty());

	const auto payload = DeserializeVulkanNativeInstructionPayload(artifact.Instructions());
	const auto generated = VulkanNativeMatMulF32SPIRV(2, 3, 4);
	EXPECT_EQ(payload.spirv, generated.words);
	EXPECT_NE(payload.featureSet.flags & (1ull << static_cast<std::uint32_t>(VulkanNativeFeature::MatMulF32)), 0ull);
	ASSERT_EQ(payload.kernels.size(), 1u);
	EXPECT_EQ(payload.kernels[0].groups.x, 1u);
	ASSERT_EQ(payload.kernels[0].arguments.size(), 3u);
	EXPECT_EQ(payload.kernels[0].arguments[0].byteSize, 2u * 3u * sizeof(float));
	EXPECT_EQ(payload.kernels[0].arguments[1].byteSize, 3u * 4u * sizeof(float));
	EXPECT_EQ(payload.kernels[0].arguments[2].byteSize, 2u * 4u * sizeof(float));
}

TEST(CompiledModuleVulkanTest, WritesVulkanNativePayloadForSimpleMatMulBiasReLU)
{
	const auto graph = BuildSimpleMatMulBiasGraph(true);
	const auto artifact = Compiler<Vulkan>::CompileArtifact(Detail::BuildExecutablePlanFromGraph(graph));
	EXPECT_EQ(artifact.Backend(), CompiledModuleBackend::VulkanNative);
	EXPECT_FALSE(artifact.Instructions().empty());

	const auto payload = DeserializeVulkanNativeInstructionPayload(artifact.Instructions());
	const auto generated = VulkanNativeMatMulBiasF32SPIRV(2, 3, 4, 1, true);
	EXPECT_EQ(payload.spirv, generated.words);
	EXPECT_NE(payload.featureSet.flags &
	              (1ull << static_cast<std::uint32_t>(VulkanNativeFeature::MatMulBiasAddReLUF32)),
	          0ull);
	ASSERT_EQ(payload.kernels.size(), 1u);
	EXPECT_EQ(payload.kernels[0].groups.x, 1u);
	ASSERT_EQ(payload.kernels[0].arguments.size(), 4u);
	EXPECT_EQ(payload.kernels[0].arguments[0].byteSize, 2u * 3u * sizeof(float));
	EXPECT_EQ(payload.kernels[0].arguments[1].byteSize, 3u * 4u * sizeof(float));
	EXPECT_EQ(payload.kernels[0].arguments[2].byteSize, 1u * 4u * sizeof(float));
	EXPECT_EQ(payload.kernels[0].arguments[3].byteSize, 2u * 4u * sizeof(float));
}

TEST(CompiledModuleVulkanTest, WritesVulkanNativePayloadForMatMulBiasExternalWeights)
{
	const auto graph = BuildSimpleMatMulBiasVariableGraph(true);
	const auto artifact = Compiler<Vulkan>::CompileArtifact(Detail::BuildExecutablePlanFromGraph(graph));
	EXPECT_EQ(artifact.Backend(), CompiledModuleBackend::VulkanNative);
	EXPECT_EQ(artifact.InputSpecs().size(), 1u);
	EXPECT_EQ(artifact.ExternalTensorInfos().size(), 2u);
	EXPECT_EQ(artifact.Constants().size(), 0u);
	EXPECT_GT(artifact.Weights().size(), 0u);

	const auto payload = DeserializeVulkanNativeInstructionPayload(artifact.Instructions());
	ASSERT_EQ(payload.kernels.size(), 1u);
	ASSERT_EQ(payload.kernels[0].arguments.size(), 4u);
	EXPECT_EQ(payload.kernels[0].arguments[0].kind, VulkanNativeArgumentKind::InputTensor);
	EXPECT_EQ(payload.kernels[0].arguments[1].kind, VulkanNativeArgumentKind::ExternalTensor);
	EXPECT_EQ(payload.kernels[0].arguments[2].kind, VulkanNativeArgumentKind::ExternalTensor);
	EXPECT_EQ(payload.kernels[0].arguments[1].index, 0u);
	EXPECT_EQ(payload.kernels[0].arguments[2].index, 1u);
}

TEST(CompiledModuleVulkanTest, WritesVulkanNativePayloadForLowPrecisionCast)
{
	const auto graph = BuildSimpleCastGraph(DataType::Float32, DataType::Float16);
	const auto artifact = Compiler<Vulkan>::CompileArtifact(Detail::BuildExecutablePlanFromGraph(graph));
	EXPECT_EQ(artifact.Backend(), CompiledModuleBackend::VulkanNative);
	EXPECT_FALSE(artifact.Instructions().empty());

	const auto payload = DeserializeVulkanNativeInstructionPayload(artifact.Instructions());
	const auto generated = VulkanNativeSameShapeCastSPIRV(DataType::Float32, DataType::Float16, kElementCount);
	EXPECT_EQ(payload.spirv, generated.words);
	EXPECT_NE(payload.featureSet.flags &
	              (1ull << static_cast<std::uint32_t>(VulkanNativeFeature::SameShapeCastLowPrecision)),
	          0ull);
}

TEST(CompiledModuleVulkanTest, RejectsLowPrecisionCastWhenDeviceFeaturesAreNotEnabled)
{
	if (!IsVulkanDeviceAvailable())
	{
		GTEST_SKIP() << "No Vulkan compute device is available";
	}

	Vulkan device;
	const auto capabilities = QueryVulkanDeviceCapabilities(device);
	if (capabilities.shaderFloat16Enabled && capabilities.storageBuffer16BitAccessEnabled)
	{
		GTEST_SKIP() << "Vulkan Float16 storage features are enabled by the runtime";
	}

	const auto graph = BuildSimpleCastGraph(DataType::Float32, DataType::Float16);
	const auto artifact = Compiler<Vulkan>::CompileArtifact(Detail::BuildExecutablePlanFromGraph(graph));
	ASSERT_EQ(artifact.Backend(), CompiledModuleBackend::VulkanNative);

	try
	{
		(void)artifact.Load(device);
		FAIL() << "Expected low-precision Vulkan artifact loading to require enabled device features";
	}
	catch (const std::runtime_error& ex)
	{
		const std::string message = ex.what();
		EXPECT_TRUE(message.find("shaderFloat16") != std::string::npos ||
		            message.find("storageBuffer16BitAccess") != std::string::npos)
		    << message;
		EXPECT_NE(message.find("enabled=false"), std::string::npos);
	}
}

TEST(CompiledModuleVulkanTest, RunsLowPrecisionCastArithmeticWhenDeviceFeaturesAreEnabled)
{
	if (!IsVulkanDeviceAvailable())
	{
		GTEST_SKIP() << "No Vulkan compute device is available";
	}

	Vulkan device;
	const auto capabilities = QueryVulkanDeviceCapabilities(device);
	if (!capabilities.shaderFloat16Enabled || !capabilities.storageBuffer16BitAccessEnabled)
	{
		GTEST_SKIP() << "Vulkan Float16 storage features are not enabled by the runtime";
	}

	const auto graph = BuildSimpleCastGraph(DataType::Float32, DataType::Float16);
	auto module = Compiler<Vulkan>::Compile(Detail::BuildExecutablePlanFromGraph(graph), device);
	ASSERT_EQ(module.Backend(), CompiledModuleBackend::VulkanNative);

	std::array inputs{
		Tensor<Vulkan>({ -3.5, -1.0, 0.75, 4.0 }, { 4 }, DataType::Float32, device),
	};
	auto outputs = module.RunTensors(std::span<const Tensor<Vulkan>>(inputs));
	ASSERT_EQ(outputs.size(), 1);
	EXPECT_EQ(outputs[0].DType(), DataType::Float16);

	const auto actual = CopyToHostAsFloat32(outputs[0]);
	const std::array expected{ -3.5f, -1.0f, 0.75f, 4.0f };
	for (std::size_t i = 0; i < expected.size(); ++i)
	{
		EXPECT_FLOAT_EQ(actual[i], expected[i]);
	}
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

TEST(CompiledModuleVulkanTest, RunsSimpleMatMulArithmetic)
{
	if (!IsVulkanDeviceAvailable())
	{
		GTEST_SKIP() << "No Vulkan compute device is available";
	}

	const auto graph = BuildSimpleMatMulGraph();
	auto module = Compiler<Vulkan>::Compile(Detail::BuildExecutablePlanFromGraph(graph), Vulkan{});
	ASSERT_EQ(module.Backend(), CompiledModuleBackend::VulkanNative);

	Vulkan device;
	std::array inputs{
		Tensor<Vulkan>({ 1.0, 2.0, 3.0, 4.0, 5.0, 6.0 }, { 2, 3 }, DataType::Float32, device),
		Tensor<Vulkan>({ 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0, 12.0 }, { 3, 4 }, DataType::Float32,
		               device),
	};
	auto outputs = module.RunTensors(std::span<const Tensor<Vulkan>>(inputs));
	ASSERT_EQ(outputs.size(), 1);

	const auto actual = CopyToHostVector(outputs[0]);
	const std::array expected{ 38.0f, 44.0f, 50.0f, 56.0f, 83.0f, 98.0f, 113.0f, 128.0f };
	ASSERT_EQ(actual.size(), expected.size());
	for (std::size_t i = 0; i < expected.size(); ++i)
	{
		EXPECT_FLOAT_EQ(actual[i], expected[i]);
	}
}

TEST(CompiledModuleVulkanTest, RunsMatMulBiasExternalWeightsArithmetic)
{
	if (!IsVulkanDeviceAvailable())
	{
		GTEST_SKIP() << "No Vulkan compute device is available";
	}

	const auto graph = BuildSimpleMatMulBiasVariableGraph(true);
	const auto artifact = Compiler<Vulkan>::CompileArtifact(Detail::BuildExecutablePlanFromGraph(graph));
	ASSERT_EQ(artifact.Backend(), CompiledModuleBackend::VulkanNative);
	ASSERT_GT(artifact.Weights().size(), 0u);
	auto module = artifact.Load(Vulkan{});
	ASSERT_EQ(module.Backend(), CompiledModuleBackend::VulkanNative);

	Vulkan device;
	std::array inputs{
		Tensor<Vulkan>({ 1.0, 2.0, 3.0, 4.0, 5.0, 6.0 }, { 2, 3 }, DataType::Float32, device),
	};
	auto outputs = module.RunTensors(std::span<const Tensor<Vulkan>>(inputs));
	ASSERT_EQ(outputs.size(), 1);

	const auto actual = CopyToHostVector(outputs[0]);
	const std::array expected{ 39.0f, 0.0f, 53.0f, 0.0f, 84.0f, 0.0f, 116.0f, 0.0f };
	ASSERT_EQ(actual.size(), expected.size());
	for (std::size_t i = 0; i < expected.size(); ++i)
	{
		EXPECT_FLOAT_EQ(actual[i], expected[i]);
	}
}

TEST(CompiledModuleVulkanTest, RunsSimpleMatMulBiasReLUArithmetic)
{
	if (!IsVulkanDeviceAvailable())
	{
		GTEST_SKIP() << "No Vulkan compute device is available";
	}

	const auto graph = BuildSimpleMatMulBiasGraph(true);
	auto module = Compiler<Vulkan>::Compile(Detail::BuildExecutablePlanFromGraph(graph), Vulkan{});
	ASSERT_EQ(module.Backend(), CompiledModuleBackend::VulkanNative);

	Vulkan device;
	std::array inputs{
		Tensor<Vulkan>({ 1.0, 2.0, 3.0, 4.0, 5.0, 6.0 }, { 2, 3 }, DataType::Float32, device),
		Tensor<Vulkan>({ 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0, 12.0 }, { 3, 4 }, DataType::Float32,
		               device),
		Tensor<Vulkan>({ 1.0, -100.0, 3.0, -200.0 }, { 1, 4 }, DataType::Float32, device),
	};
	auto outputs = module.RunTensors(std::span<const Tensor<Vulkan>>(inputs));
	ASSERT_EQ(outputs.size(), 1);

	const auto actual = CopyToHostVector(outputs[0]);
	const std::array expected{ 39.0f, 0.0f, 53.0f, 0.0f, 84.0f, 0.0f, 116.0f, 0.0f };
	ASSERT_EQ(actual.size(), expected.size());
	for (std::size_t i = 0; i < expected.size(); ++i)
	{
		EXPECT_FLOAT_EQ(actual[i], expected[i]);
	}
}
