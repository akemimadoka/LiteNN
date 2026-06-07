#include <gtest/gtest.h>

#include <LiteNN.h>
#include <LiteNN/Compiler/CompiledModule.h>

#include <array>

using namespace LiteNN;

namespace
{
	Graph BuildSimpleAddGraph()
	{
		Graph graph;
		Subgraph sg;
		const auto lhs = sg.AddParam(DataType::Float32, { 4 });
		const auto rhs = sg.AddParam(DataType::Float32, { 4 });
		const auto out = sg.AddNode(BinaryOpNode{ BinaryOp::Add, { lhs, 0 }, { rhs, 0 } },
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
}

TEST(CompiledModuleVulkanTest, WritesVulkanNativePayloadForSimpleAdd)
{
	const auto graph = BuildSimpleAddGraph();
	const auto artifact = Compiler<Vulkan>::CompileArtifact(Detail::BuildExecutablePlanFromGraph(graph));
	EXPECT_EQ(artifact.Backend(), CompiledModuleBackend::VulkanNative);
	EXPECT_FALSE(artifact.Instructions().empty());
}

TEST(CompiledModuleVulkanTest, RunsSimpleAdd)
{
	if (!IsVulkanDeviceAvailable())
	{
		GTEST_SKIP() << "No Vulkan compute device is available";
	}

	const auto graph = BuildSimpleAddGraph();
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
	const std::array expected{ 11.0f, 22.0f, 33.0f, 44.0f };
	for (std::size_t i = 0; i < expected.size(); ++i)
	{
		EXPECT_FLOAT_EQ(actual[i], expected[i]);
	}
}
