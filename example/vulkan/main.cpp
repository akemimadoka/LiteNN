#include <LiteNN.h>
#include <LiteNN/Compiler/CompiledModule.h>

#include <array>
#include <exception>
#include <iostream>
#include <stdexcept>
#include <string_view>
#include <vector>

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

	Graph BuildTwoAddGraph()
	{
		Graph graph;
		Subgraph sg;
		const auto lhs = sg.AddParam(DataType::Float32, { 4 });
		const auto rhs = sg.AddParam(DataType::Float32, { 4 });
		const auto tail = sg.AddParam(DataType::Float32, { 4 });
		const auto first = sg.AddNode(BinaryOpNode{ BinaryOp::Add, { lhs, 0 }, { rhs, 0 } },
		                              { OutputInfo{ DataType::Float32, { 4 } } });
		const auto second = sg.AddNode(BinaryOpNode{ BinaryOp::Add, { first, 0 }, { tail, 0 } },
		                               { OutputInfo{ DataType::Float32, { 4 } } });
		sg.SetResults({ { second, 0 } });
		graph.AddSubgraph(std::move(sg));
		graph.SetForward(0);
		graph.SetInputNames({ "lhs", "rhs", "tail" });
		graph.SetOutputNames({ "sum" });
		return graph;
	}

	std::string_view BackendName(CompiledModuleBackend backend)
	{
		switch (backend)
		{
		case CompiledModuleBackend::CPUNative:
			return "cpu_native";
		case CompiledModuleBackend::CUDANative:
			return "cuda_native";
		case CompiledModuleBackend::VulkanNative:
			return "vulkan_native";
		}
		return "unknown";
	}

	void PrintNativeSupport(std::string_view label, const Graph& graph)
	{
		const auto report = Compiler<Vulkan>::QueryNativeSupport(Detail::BuildExecutablePlanFromGraph(graph));
		std::cout << label << " native support: " << (report.supported ? "yes" : "no");
		if (report.supported)
		{
			std::cout << " (" << report.capability << ')';
		}
		else
		{
			std::cout << " (" << report.reason << ')';
		}
		std::cout << '\n';
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

	std::array<float, 4> RunAdd(const CompiledModule<Vulkan>& module, Vulkan device,
	                            std::vector<CompiledModuleVulkanProfileEvent>* profileEvents = nullptr)
	{
		std::array inputs{
			Tensor<Vulkan>({ 1.0, 2.0, 3.0, 4.0 }, { 4 }, DataType::Float32, device),
			Tensor<Vulkan>({ 10.0, 20.0, 30.0, 40.0 }, { 4 }, DataType::Float32, device),
		};
		auto outputs = module.RunTensors(std::span<const Tensor<Vulkan>>(inputs),
		                                 { .synchronize = true, .profileEvents = profileEvents });
		return CopyToHost(outputs[0]);
	}

	std::array<float, 4> RunThreeInput(const CompiledModule<Vulkan>& module, Vulkan device)
	{
		std::array inputs{
			Tensor<Vulkan>({ 1.0, 2.0, 3.0, 4.0 }, { 4 }, DataType::Float32, device),
			Tensor<Vulkan>({ 10.0, 20.0, 30.0, 40.0 }, { 4 }, DataType::Float32, device),
			Tensor<Vulkan>({ 100.0, 200.0, 300.0, 400.0 }, { 4 }, DataType::Float32, device),
		};
		auto outputs = module.RunTensors(std::span<const Tensor<Vulkan>>(inputs));
		return CopyToHost(outputs[0]);
	}

	void PrintResult(std::string_view label, const std::array<float, 4>& result)
	{
		std::cout << label;
		for (const auto value : result)
		{
			std::cout << ' ' << value;
		}
		std::cout << '\n';
	}

	void PrintProfileEvents(const std::vector<CompiledModuleVulkanProfileEvent>& events)
	{
		for (const auto& event : events)
		{
			std::cout << "Profile kernel[" << event.kernelIndex << "] entry=" << event.entryPoint
			          << " groups=(" << event.groups.x << ',' << event.groups.y << ',' << event.groups.z
			          << ") local=(" << event.localSize.x << ',' << event.localSize.y << ',' << event.localSize.z
			          << ") descriptors=" << event.descriptorCount
			          << " module_ms=" << event.moduleCreationWallMs
			          << " dispatch_ms=" << event.dispatchWallMs
			          << " gpu_ms=";
			if (event.gpuTimestampAvailable)
			{
				std::cout << event.gpuElapsedMs;
			}
			else
			{
				std::cout << "n/a";
			}
			std::cout << '\n';
		}
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
	Vulkan device;
	PrintNativeSupport("Add", graph);
	auto artifact = Compiler<Vulkan>::CompileArtifact(Detail::BuildExecutablePlanFromGraph(graph));
	std::cout << "Add artifact backend: " << BackendName(artifact.Backend()) << '\n';
	auto module = artifact.Load(device);
	auto separated = artifact.SeparateRodata();
	auto separatedModule = separated.LoadBorrowedExternalRegions(device);

	std::vector<CompiledModuleVulkanProfileEvent> profileEvents;
	PrintResult("Vulkan Add result:", RunAdd(module, device, &profileEvents));
	PrintResult("Vulkan Add separated result:", RunAdd(separatedModule, device));
	PrintProfileEvents(profileEvents);
	std::cout << "Separated regions: metadata=" << separated.Metadata().size()
	          << " constants=" << separated.Constants().size()
	          << " weights=" << separated.Weights().size()
	          << " instructions=" << separated.Instructions().size() << '\n';

	auto fallbackGraph = BuildTwoAddGraph();
	PrintNativeSupport("TwoAdd", fallbackGraph);
	auto fallbackArtifact = Compiler<Vulkan>::CompileArtifact(Detail::BuildExecutablePlanFromGraph(fallbackGraph));
	std::cout << "TwoAdd artifact backend: " << BackendName(fallbackArtifact.Backend()) << '\n';

	if (fallbackArtifact.Backend() == CompiledModuleBackend::CPUNative)
	{
		bool strictLoadRejected = false;
		try
		{
			(void)fallbackArtifact.Load(device);
		}
		catch (const std::exception& error)
		{
			strictLoadRejected = true;
			std::cout << "Strict Vulkan load rejected CPU bridge: " << error.what() << '\n';
		}
		if (!strictLoadRejected)
		{
			throw std::runtime_error("strict Vulkan load unexpectedly accepted a CPU bridge artifact");
		}

		Vulkan bridgeDevice;
		bridgeDevice.hostFallbackPolicy = VulkanHostFallbackPolicy::Allow;
		auto fallbackModule = fallbackArtifact.Load(bridgeDevice);
		PrintResult("Explicit CPU bridge TwoAdd result:", RunThreeInput(fallbackModule, bridgeDevice));
	}
	else
	{
		auto nativeFallbackModule = fallbackArtifact.Load(device);
		PrintResult("TwoAdd native result:", RunThreeInput(nativeFallbackModule, device));
	}
	return 0;
}
