#ifndef LITENN_VNEXT_PACKAGE_H
#define LITENN_VNEXT_PACKAGE_H

#include <LiteNN/ExecutablePlan.h>
#include <LiteNN/MemoryPlan.h>
#include <LiteNN/OpSchema.h>
#include <LiteNN/Runtime/Scheduler.h>
#include <cstdint>
#include <format>
#include <optional>
#include <stdexcept>
#include <string>
#include <string_view>
#include <vector>

namespace LiteNN
{
	struct VNextVersionSet
	{
		std::uint32_t manifest{ 1 };
		std::uint32_t opSet{ 1 };
		std::uint32_t dtypeSet{ 1 };
		std::uint32_t layoutSet{ 1 };
		std::uint32_t quantizationSet{ 1 };
		std::uint32_t artifactABI{ 1 };
	};

	struct VNextExternalTensorRef
	{
		std::string name;
		TensorType type;
		std::optional<QuantizationParams> quantization;
		ExternalBufferKind kind{ ExternalBufferKind::None };
		std::string relativePath;
		std::size_t byteOffset{};
		std::size_t byteSize{};
		std::size_t alignment{ 1 };
		std::uint64_t checksum{};
		BufferMutability mutability{ BufferMutability::Immutable };
		BufferRebindPolicy rebindPolicy{ BufferRebindPolicy::ExactMetadataAndChecksum };
	};

	struct VNextArtifactRegionRef
	{
		std::string name;
		ExternalBufferKind kind{ ExternalBufferKind::ObjectFile };
		std::string relativePath;
		std::size_t byteOffset{};
		std::size_t byteSize{};
		std::uint64_t checksum{};
	};

	struct VNextArtifactRef
	{
		std::string name;
		std::string backend;
		FunctionId entryFunction{};
		std::vector<VNextArtifactRegionRef> regions;
		std::vector<VNextExternalTensorRef> externalTensors;
	};

	struct VNextPackageLayout
	{
		std::string mode{ "standalone-archive" };
		std::string manifestPath{ "model.ltnn.json" };
		std::string tensorDirectory{ "tensors" };
		std::string artifactDirectory{ "artifacts" };
	};

	struct VNextPackageManifest
	{
		VNextVersionSet versions;
		VNextPackageLayout layout;
		std::vector<ExecutableFunction> functions;
		std::vector<ExecutableRegion> regions;
		std::vector<ExecutablePartition> partitions;
		MemoryPlan memory;
		std::vector<Runtime::RuntimeStateBinding> runtimeStates;
		std::vector<Runtime::RuntimeScheduleStep> runtimeSteps;
		std::vector<VNextExternalTensorRef> tensors;
		std::vector<VNextArtifactRef> artifacts;
		std::vector<OpCoverageRow> opCoverage;
	};

	inline VNextExternalTensorRef ToVNextExternalTensorRef(std::string name, const TensorStorageRef& storage)
	{
		return { .name = std::move(name),
			     .type = storage.type,
			     .quantization = storage.quantization,
			     .kind = storage.region.externalKind,
			     .relativePath = storage.region.name,
			     .byteOffset = storage.region.byteOffset + storage.storageOffsetBytes,
			     .byteSize = storage.LogicalByteSize().value_or(storage.region.byteSize),
			     .alignment = storage.region.alignment,
			     .checksum = storage.region.checksum,
			     .mutability = storage.region.mutability,
			     .rebindPolicy = storage.region.rebindPolicy };
	}

	inline VNextPackageManifest BuildVNextPackageManifest(
	    const ExecutableModule& module, std::vector<VNextArtifactRef> artifacts = {},
	    VNextPackageLayout layout = {}, const OpSchemaRegistry& registry = DefaultOpSchemaRegistry())
	{
		ValidateExecutablePlan(module.plan, registry);
		VNextPackageManifest manifest;
		manifest.layout = std::move(layout);
		manifest.functions = module.functions;
		manifest.regions = module.regions;
		manifest.partitions = module.partitions;
		auto schedule = Runtime::BuildRuntimeSchedule(module);
		Runtime::ValidateRuntimeSchedule(schedule);
		manifest.memory = std::move(schedule.memory);
		manifest.runtimeStates = std::move(schedule.states);
		manifest.runtimeSteps = std::move(schedule.steps);
		manifest.artifacts = std::move(artifacts);
		manifest.opCoverage = registry.CoverageReport();
		manifest.tensors.reserve(module.plan.variables.size());
		for (std::size_t i = 0; i < module.plan.variables.size(); ++i)
		{
			const auto& storage = module.plan.variables[i];
			manifest.tensors.push_back(ToVNextExternalTensorRef(
			    storage.region.name.empty() ? std::format("variable{}", i) : storage.region.name, storage));
		}
		return manifest;
	}

	inline VNextPackageManifest BuildVNextPackageManifest(
	    const Graph& graph, std::vector<VNextArtifactRef> artifacts = {},
	    VNextPackageLayout layout = {}, const OpSchemaRegistry& registry = DefaultOpSchemaRegistry())
	{
		return BuildVNextPackageManifest(BuildExecutableModule(graph, registry), std::move(artifacts), std::move(layout),
		                                 registry);
	}

	inline void ValidateVNextPackageManifest(const VNextPackageManifest& manifest)
	{
		if (manifest.versions.manifest == 0 || manifest.versions.opSet == 0 || manifest.versions.dtypeSet == 0 ||
		    manifest.versions.layoutSet == 0 || manifest.versions.quantizationSet == 0 ||
		    manifest.versions.artifactABI == 0)
		{
			throw std::runtime_error("VNext package manifest versions must be non-zero");
		}
		if (manifest.layout.mode != "static-library" && manifest.layout.mode != "shared-library" &&
		    manifest.layout.mode != "standalone-archive" && manifest.layout.mode != "mobile")
		{
			throw std::runtime_error("vNext package manifest has an unknown package layout mode: " +
			                         manifest.layout.mode);
		}
		if (manifest.layout.manifestPath.empty())
		{
			throw std::runtime_error("vNext package manifest path cannot be empty");
		}
		if (manifest.functions.empty())
		{
			throw std::runtime_error("vNext package manifest contains no executable functions");
		}
		for (std::size_t i = 0; i < manifest.functions.size(); ++i)
		{
			const auto& function = manifest.functions[i];
			if (function.id != i)
			{
				throw std::runtime_error(std::format("vNext function {} has mismatched id {}", i, function.id));
			}
			if (function.name.empty())
			{
				throw std::runtime_error(std::format("vNext function {} has empty name", i));
			}
		}
		for (std::size_t i = 0; i < manifest.regions.size(); ++i)
		{
			const auto& region = manifest.regions[i];
			if (region.id != i)
			{
				throw std::runtime_error(std::format("vNext region {} has mismatched id {}", i, region.id));
			}
			if (region.function >= manifest.functions.size())
			{
				throw std::runtime_error(std::format("vNext region {} references unknown function", i));
			}
		}
		for (std::size_t i = 0; i < manifest.partitions.size(); ++i)
		{
			const auto& partition = manifest.partitions[i];
			if (partition.id != i)
			{
				throw std::runtime_error(std::format("vNext partition {} has mismatched id {}", i, partition.id));
			}
			if (partition.backend.empty())
			{
				throw std::runtime_error(std::format("vNext partition {} has empty backend", i));
			}
			for (const auto region : partition.regions)
			{
				if (region >= manifest.regions.size())
				{
					throw std::runtime_error(std::format("vNext partition {} references unknown region {}", i, region));
				}
			}
		}
		for (std::size_t i = 0; i < manifest.memory.buffers.size(); ++i)
		{
			if (manifest.memory.buffers[i].id != i)
			{
				throw std::runtime_error(std::format("vNext memory buffer {} has mismatched id", i));
			}
		}
		for (std::size_t i = 0; i < manifest.runtimeSteps.size(); ++i)
		{
			const auto& step = manifest.runtimeSteps[i];
			if (step.id != i)
			{
				throw std::runtime_error(std::format("vNext runtime step {} has mismatched id {}", i, step.id));
			}
			if (step.kind == Runtime::RuntimeScheduleStepKind::DispatchRegion)
			{
				if (step.function >= manifest.functions.size() || step.region >= manifest.regions.size())
				{
					throw std::runtime_error(std::format("vNext runtime step {} references unknown dispatch target", i));
				}
			}
			for (const auto buffer : step.inputBuffers)
			{
				if (buffer >= manifest.memory.buffers.size())
				{
					throw std::runtime_error(std::format("vNext runtime step {} references invalid input buffer", i));
				}
			}
			for (const auto buffer : step.outputBuffers)
			{
				if (buffer >= manifest.memory.buffers.size())
				{
					throw std::runtime_error(std::format("vNext runtime step {} references invalid output buffer", i));
				}
			}
		}
		for (std::size_t i = 0; i < manifest.tensors.size(); ++i)
		{
			const auto& tensor = manifest.tensors[i];
			if (tensor.name.empty())
			{
				throw std::runtime_error(std::format("vNext tensor {} has empty name", i));
			}
			ValidateExecutableTensorType(tensor.type, std::format("vNext tensor {}", i));
			if (tensor.alignment == 0)
			{
				throw std::runtime_error(std::format("vNext tensor '{}' has zero alignment", tensor.name));
			}
			if (tensor.byteSize == 0 && tensor.type.ByteSize().value_or(0) != 0)
			{
				throw std::runtime_error(std::format("vNext tensor '{}' has zero byte size", tensor.name));
			}
			if (tensor.kind != ExternalBufferKind::None && tensor.relativePath.empty())
			{
				throw std::runtime_error(std::format("vNext tensor '{}' has empty external path", tensor.name));
			}
		}
		for (const auto& artifact : manifest.artifacts)
		{
			if (artifact.name.empty())
			{
				throw std::runtime_error("vNext artifact has empty name");
			}
			if (artifact.backend.empty())
			{
				throw std::runtime_error("vNext artifact '" + artifact.name + "' has empty backend");
			}
			if (artifact.entryFunction >= manifest.functions.size())
			{
				throw std::runtime_error("vNext artifact '" + artifact.name + "' references an unknown function");
			}
			if (artifact.regions.empty())
			{
				throw std::runtime_error("vNext artifact '" + artifact.name + "' has no regions");
			}
			for (const auto& region : artifact.regions)
			{
				if (region.name.empty())
				{
					throw std::runtime_error("vNext artifact '" + artifact.name + "' has a region with empty name");
				}
				if (region.relativePath.empty())
				{
					throw std::runtime_error("vNext artifact '" + artifact.name + "' has a region with empty path");
				}
				if (region.byteSize == 0)
				{
					throw std::runtime_error("vNext artifact '" + artifact.name + "' has a zero-sized region");
				}
			}
		}
	}
} // namespace LiteNN

#endif
