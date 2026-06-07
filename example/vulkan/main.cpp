#include <LiteNN.h>
#include <LiteNN/Compiler/CompiledModule.h>

#include <array>
#include <iostream>

using namespace LiteNN;

namespace
{
	Graph BuildAddGraph()
	{
		Graph graph;
		Subgraph sg;
		const auto lhs = sg.AddParam(DataType::Float32, { 4 });
		const auto rhs = sg.AddParam(DataType::Float32, { 4 });
		const auto sum = sg.AddNode(BinaryOpNode{ BinaryOp::Add, { lhs, 0 }, { rhs, 0 } },
		                            { OutputInfo{ DataType::Float32, { 4 } } });
		sg.SetResults({ { sum, 0 } });
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

int main()
{
	if (!IsVulkanDeviceAvailable())
	{
		std::cout << "No Vulkan compute device is available; example skipped.\n";
		return 0;
	}

	auto graph = BuildAddGraph();
	auto module = Compiler<Vulkan>::Compile(Detail::BuildExecutablePlanFromGraph(graph), Vulkan{});

	Vulkan device;
	std::array inputs{
		Tensor<Vulkan>({ 1.0, 2.0, 3.0, 4.0 }, { 4 }, DataType::Float32, device),
		Tensor<Vulkan>({ 10.0, 20.0, 30.0, 40.0 }, { 4 }, DataType::Float32, device),
	};
	auto outputs = module.RunTensors(std::span<const Tensor<Vulkan>>(inputs));
	const auto result = CopyToHost(outputs[0]);

	std::cout << "Vulkan Add result:";
	for (const auto value : result)
	{
		std::cout << ' ' << value;
	}
	std::cout << '\n';
	return 0;
}
