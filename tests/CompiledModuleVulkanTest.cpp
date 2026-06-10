#include <gtest/gtest.h>

#include <LiteNN.h>
#include <LiteNN/Compiler/CompiledModule.h>
#include <LiteNN/Compiler/VulkanNativeCodegen.h>
#include <LiteNN/Compiler/VulkanNativePayload.h>

#include <array>
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

	std::array<float, 4> CopyToHost(const Tensor<Vulkan>& tensor)
	{
		Tensor<CPU> host(Uninitialized, tensor.Shape(), tensor.DType(), CPU{});
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

	constexpr std::array kBinaryCases{
		BinaryCase{ BinaryOp::Add, "spirv.FAdd", { 11.0f, 22.0f, 33.0f, 44.0f } },
		BinaryCase{ BinaryOp::Subtract, "spirv.FSub", { -9.0f, -18.0f, -27.0f, -36.0f } },
		BinaryCase{ BinaryOp::Multiply, "spirv.FMul", { 10.0f, 40.0f, 90.0f, 160.0f } },
		BinaryCase{ BinaryOp::Divide, "spirv.FDiv", { 0.1f, 0.1f, 0.1f, 0.1f } },
	};
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
