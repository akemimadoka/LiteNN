#ifndef LITENN_VNEXT_PACKAGE_H
#define LITENN_VNEXT_PACKAGE_H

#include <LiteNN/ExecutablePlan.h>
#include <LiteNN/MemoryPlan.h>
#include <LiteNN/Misc.h>
#include <LiteNN/OpSchema.h>
#include <LiteNN/Runtime/Scheduler.h>
#include <algorithm>
#include <cstdint>
#include <format>
#include <optional>
#include <span>
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

	enum class VNextVersionComponent
	{
		Manifest,
		OpSet,
		DTypeSet,
		LayoutSet,
		QuantizationSet,
		ArtifactABI
	};

	inline std::string_view VNextVersionComponentName(VNextVersionComponent component) noexcept
	{
		return EnumToString<EnumToStringStyle::Unqualified>(component);
	}

	enum class VNextABIChangeArea
	{
		ManifestShape,
		OpSemantics,
		DTypeSemantics,
		LayoutSemantics,
		QuantizationSemantics,
		TensorBinding,
		ExternalRegion,
		BackendRequirement,
		RuntimeState,
		RuntimeSchedule,
		ArtifactEntry
	};

	inline std::string_view VNextABIChangeAreaName(VNextABIChangeArea area) noexcept
	{
		return EnumToString<EnumToStringStyle::Unqualified>(area);
	}

	struct VNextABIVersionBumpRule
	{
		VNextABIChangeArea area{ VNextABIChangeArea::ManifestShape };
		VNextVersionComponent component{ VNextVersionComponent::Manifest };
		std::string_view reason;
	};

	inline VNextABIVersionBumpRule VNextABIVersionBumpRuleFor(VNextABIChangeArea area)
	{
		switch (area)
		{
		case VNextABIChangeArea::ManifestShape:
			return { area, VNextVersionComponent::Manifest,
				     "JSON manifest keys, required sections, or package layout shape changed" };
		case VNextABIChangeArea::OpSemantics:
			return { area, VNextVersionComponent::OpSet, "Executable op semantics or required op attributes changed" };
		case VNextABIChangeArea::DTypeSemantics:
			return { area, VNextVersionComponent::DTypeSet,
				     "Data type encoding, precision behavior, or dtype availability changed" };
		case VNextABIChangeArea::LayoutSemantics:
			return { area, VNextVersionComponent::LayoutSet,
				     "Tensor layout interpretation, strides, or memory-space layout contracts changed" };
		case VNextABIChangeArea::QuantizationSemantics:
			return { area, VNextVersionComponent::QuantizationSet,
				     "Quantization parameter, block format, scale, or zero-point semantics changed" };
		case VNextABIChangeArea::TensorBinding:
			return { area, VNextVersionComponent::ArtifactABI,
				     "Runtime-visible tensor binding names, mutability, rebind policy, or checksum contracts changed" };
		case VNextABIChangeArea::ExternalRegion:
			return { area, VNextVersionComponent::ArtifactABI,
				     "External rodata, weight, instruction, or object-region ownership/alignment contracts changed" };
		case VNextABIChangeArea::BackendRequirement:
			return { area, VNextVersionComponent::ArtifactABI,
				     "Backend selection, capability, fallback, or native artifact requirement contracts changed" };
		case VNextABIChangeArea::RuntimeState:
			return { area, VNextVersionComponent::ArtifactABI,
				     "KV cache, diffusion, training, optimizer, or adapter runtime-state binding contracts changed" };
		case VNextABIChangeArea::RuntimeSchedule:
			return { area, VNextVersionComponent::ArtifactABI,
				     "Runtime schedule step, transfer, fallback, or profile-record contracts changed" };
		case VNextABIChangeArea::ArtifactEntry:
			return { area, VNextVersionComponent::ArtifactABI,
				     "Named artifact entry kinds, entry functions, or required binding contracts changed" };
		}
		return { area, VNextVersionComponent::Manifest, "Unknown vNext ABI change area" };
	}

	inline std::vector<VNextABIVersionBumpRule> DescribeVNextABIVersionBumpRules()
	{
		return {
			VNextABIVersionBumpRuleFor(VNextABIChangeArea::ManifestShape),
			VNextABIVersionBumpRuleFor(VNextABIChangeArea::OpSemantics),
			VNextABIVersionBumpRuleFor(VNextABIChangeArea::DTypeSemantics),
			VNextABIVersionBumpRuleFor(VNextABIChangeArea::LayoutSemantics),
			VNextABIVersionBumpRuleFor(VNextABIChangeArea::QuantizationSemantics),
			VNextABIVersionBumpRuleFor(VNextABIChangeArea::TensorBinding),
			VNextABIVersionBumpRuleFor(VNextABIChangeArea::ExternalRegion),
			VNextABIVersionBumpRuleFor(VNextABIChangeArea::BackendRequirement),
			VNextABIVersionBumpRuleFor(VNextABIChangeArea::RuntimeState),
			VNextABIVersionBumpRuleFor(VNextABIChangeArea::RuntimeSchedule),
			VNextABIVersionBumpRuleFor(VNextABIChangeArea::ArtifactEntry),
		};
	}

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

	enum class VNextArtifactEntryKind
	{
		Forward,
		Loss,
		Backward,
		OptimizerStep,
		BackendSpecific
	};

	inline std::string_view VNextArtifactEntryKindName(VNextArtifactEntryKind kind) noexcept
	{
		return EnumToString<EnumToStringStyle::Unqualified>(kind);
	}

	inline bool IsKnownVNextArtifactEntryKind(VNextArtifactEntryKind kind) noexcept
	{
		switch (kind)
		{
		case VNextArtifactEntryKind::Forward:
		case VNextArtifactEntryKind::Loss:
		case VNextArtifactEntryKind::Backward:
		case VNextArtifactEntryKind::OptimizerStep:
		case VNextArtifactEntryKind::BackendSpecific:
			return true;
		}
		return false;
	}

	struct VNextArtifactEntryRef
	{
		std::string name;
		VNextArtifactEntryKind kind{ VNextArtifactEntryKind::BackendSpecific };
		std::optional<FunctionId> function;
		std::optional<SubgraphId> sourceSubgraph;
		std::vector<std::string> requiredStateBindings;
		std::vector<std::string> requiredBufferBindings;
	};

	struct VNextBackendRequirementRef
	{
		std::optional<std::size_t> segment;
		std::string backend;
		std::vector<std::string> requiredCapabilities;
		std::string transferABI{ "none" };
		bool allowsFallback{};
	};

	struct VNextAvailableBackendRef
	{
		std::string backend;
		std::vector<std::string> capabilities;
		std::vector<std::string> transferABIs{ "none" };
		bool allowFallback{};
	};

	struct VNextBackendRequirementValidationOptions
	{
		bool allowArtifactFallback{};
	};

	struct VNextArtifactRef
	{
		std::string name;
		std::string backend;
		std::vector<VNextArtifactEntryRef> entries;
		std::vector<VNextArtifactRegionRef> regions;
		std::vector<VNextExternalTensorRef> externalTensors;
		std::vector<VNextBackendRequirementRef> backendRequirements;
	};

	struct VNextAdapterRef
	{
		std::string targetName;
		std::string adapterName{ "default" };
		std::string kind{ "linear-lora" };
		std::size_t aTensor{};
		std::size_t bTensor{};
		std::size_t rank{};
		float alpha{ 1.0f };
		float dropout{ 0.0f };
		DataType dtype{ DataType::Float32 };
		std::string mergeMode{ "unmerged" };
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
		std::vector<Runtime::RuntimeStateValueBinding> stateValueBindings;
		std::vector<RuntimeBufferBinding> bufferBindings;
		std::vector<Runtime::RuntimeExecutionSegment> runtimeSegments;
		std::vector<Runtime::RuntimeScheduleStep> runtimeSteps;
		std::vector<VNextExternalTensorRef> tensors;
		std::vector<VNextArtifactRef> artifacts;
		std::vector<VNextAdapterRef> adapters;
		std::vector<OpCoverageRow> opCoverage;
	};

	inline void ValidateVNextPackageManifest(const VNextPackageManifest& manifest);

	struct VNextABIFamilySummary
	{
		VNextVersionSet versions;
		std::vector<std::string> functions;
		std::vector<std::string> runtimeStates;
		std::vector<std::string> stateValueBindings;
		std::vector<std::string> runtimeSegments;
		std::vector<std::string> runtimeStepRecords;
		std::vector<std::string> bufferBindings;
		std::vector<std::string> tensorBindings;
		std::vector<std::string> artifactEntries;
		std::vector<std::string> artifactEntryKinds;
		std::vector<std::string> artifactRegions;
		std::vector<std::string> backendRequirements;
		bool hasRuntimeSchedule{};
		bool hasRuntimeSegments{};
		bool hasExternalTensorBindings{};
		bool hasArtifactMetadata{};
		bool hasBackendRequirements{};
		bool hasFallbackRecords{};
		bool hasTransferRecords{};
		bool hasProfileRecords{};
	};

	inline VNextABIFamilySummary DescribeVNextABIFamily(const VNextPackageManifest& manifest)
	{
		VNextABIFamilySummary summary;
		summary.versions = manifest.versions;
		summary.hasRuntimeSchedule = !manifest.runtimeSteps.empty();
		summary.hasRuntimeSegments = !manifest.runtimeSegments.empty();
		summary.hasExternalTensorBindings = !manifest.tensors.empty();
		summary.hasArtifactMetadata = !manifest.artifacts.empty();
		summary.hasBackendRequirements = std::ranges::any_of(
		    manifest.artifacts, [](const auto& artifact) { return !artifact.backendRequirements.empty(); });
		summary.hasFallbackRecords = std::ranges::any_of(manifest.runtimeSteps, [](const auto& step) {
			return step.kind == Runtime::RuntimeScheduleStepKind::Fallback;
		});
		summary.hasTransferRecords = std::ranges::any_of(manifest.runtimeSteps, [](const auto& step) {
			return step.kind == Runtime::RuntimeScheduleStepKind::Transfer;
		});
		summary.hasProfileRecords = summary.hasRuntimeSchedule;

		summary.functions.reserve(manifest.functions.size());
		for (const auto& function : manifest.functions)
		{
			summary.functions.push_back(function.name);
		}

		summary.runtimeStates.reserve(manifest.runtimeStates.size());
		for (const auto& state : manifest.runtimeStates)
		{
			summary.runtimeStates.push_back(state.name);
		}

		summary.stateValueBindings.reserve(manifest.stateValueBindings.size());
		for (const auto& binding : manifest.stateValueBindings)
		{
			summary.stateValueBindings.push_back(std::format("{}:{}:{}:{}", binding.stateName, binding.function,
			                                                 Runtime::RuntimeStateValueKindName(binding.kind),
			                                                 binding.valueIndex));
		}

		summary.runtimeSegments.reserve(manifest.runtimeSegments.size());
		for (const auto& segment : manifest.runtimeSegments)
		{
			summary.runtimeSegments.push_back(std::format("{}:{}:nodes={}:inputs={}:outputs={}", segment.id,
			                                              segment.backend, segment.nodes.size(),
			                                              segment.inputBuffers.size(), segment.outputBuffers.size()));
		}

		summary.runtimeStepRecords.reserve(manifest.runtimeSteps.size());
		for (const auto& step : manifest.runtimeSteps)
		{
			auto record =
			    std::format("{}:{}:{}", step.id, Runtime::RuntimeScheduleStepKindName(step.kind), step.backend);
			if (!step.fallbackBackend.empty())
			{
				record += "->" + step.fallbackBackend;
			}
			summary.runtimeStepRecords.push_back(std::move(record));
		}

		summary.bufferBindings.reserve(manifest.bufferBindings.size());
		for (std::size_t i = 0; i < manifest.stateValueBindings.size(); ++i)
		{
			const auto& binding = manifest.stateValueBindings[i];
			const auto state = std::ranges::find_if(
			    manifest.runtimeStates, [&](const auto& candidate) { return candidate.name == binding.stateName; });
			if (state == manifest.runtimeStates.end())
			{
				throw std::runtime_error("vNext state value binding references an unknown state: " + binding.stateName);
			}
			if (binding.function >= manifest.functions.size())
			{
				throw std::runtime_error("vNext state value binding references an unknown function");
			}
			const auto& function = manifest.functions[binding.function];
			const auto& values =
			    binding.kind == Runtime::RuntimeStateValueKind::FunctionInput ? function.inputs : function.outputs;
			if (binding.valueIndex >= values.size())
			{
				throw std::runtime_error("vNext state value binding references an unknown function value");
			}
			const auto& valueType = values[binding.valueIndex];
			if (valueType.dtype != state->type.dtype || valueType.memorySpace != state->type.memorySpace)
			{
				throw std::runtime_error("vNext state value binding type is incompatible with state: " +
				                         binding.stateName);
			}
			const auto valueBytes = valueType.ByteSize();
			const auto stateBytes = state->type.ByteSize();
			if (!valueBytes || !stateBytes || binding.stateByteOffset > *stateBytes ||
			    *valueBytes > *stateBytes - binding.stateByteOffset)
			{
				throw std::runtime_error("vNext state value binding exceeds state capacity: " + binding.stateName);
			}
			for (std::size_t j = 0; j < i; ++j)
			{
				const auto& previous = manifest.stateValueBindings[j];
				if (previous.function == binding.function && previous.kind == binding.kind &&
				    previous.valueIndex == binding.valueIndex)
				{
					throw std::runtime_error("vNext state value binding duplicates a function endpoint");
				}
			}
		}
		for (const auto& binding : manifest.bufferBindings)
		{
			summary.bufferBindings.push_back(binding.name);
		}

		summary.tensorBindings.reserve(manifest.tensors.size());
		for (const auto& tensor : manifest.tensors)
		{
			summary.tensorBindings.push_back(tensor.name);
		}

		for (const auto& artifact : manifest.artifacts)
		{
			for (const auto& entry : artifact.entries)
			{
				summary.artifactEntries.push_back(artifact.name + ":" + entry.name);
				summary.artifactEntryKinds.push_back(
				    std::format("{}:{}:{}", artifact.name, entry.name, VNextArtifactEntryKindName(entry.kind)));
			}
			for (const auto& region : artifact.regions)
			{
				summary.artifactRegions.push_back(artifact.name + ":" + region.name);
			}
			for (const auto& requirement : artifact.backendRequirements)
			{
				summary.backendRequirements.push_back(
				    std::format("{}:{}:segment={}:caps={}:transfer={}", artifact.name, requirement.backend,
				                requirement.segment ? std::to_string(*requirement.segment) : std::string("none"),
				                requirement.requiredCapabilities.size(), requirement.transferABI));
			}
		}

		return summary;
	}

	inline std::vector<VNextBackendRequirementRef>
	BuildVNextBackendRequirementsFromSchedule(const Runtime::RuntimeSchedule& schedule)
	{
		const auto hasTransferABI = std::ranges::any_of(schedule.steps, [](const auto& step) {
			return step.kind == Runtime::RuntimeScheduleStepKind::Transfer ||
			       step.kind == Runtime::RuntimeScheduleStepKind::Sync;
		});
		std::vector<VNextBackendRequirementRef> requirements;
		requirements.reserve(schedule.segments.empty() ? schedule.module.partitions.size() : schedule.segments.size());
		for (const auto& segment : schedule.segments)
		{
			requirements.push_back(
			    { .segment = segment.id,
			      .backend = segment.backend,
			      .requiredCapabilities = { "runtime-schedule:dispatch-segment", "backend:" + segment.backend },
			      .transferABI = hasTransferABI ? "runtime-buffer-transfer-v1" : "none" });
		}
		if (!requirements.empty())
		{
			return requirements;
		}
		for (const auto& partition : schedule.module.partitions)
		{
			requirements.push_back(
			    { .backend = partition.backend,
			      .requiredCapabilities = { "runtime-schedule:dispatch-region", "backend:" + partition.backend },
			      .transferABI = "none" });
		}
		return requirements;
	}

	inline const VNextAvailableBackendRef*
	FindVNextAvailableBackend(std::span<const VNextAvailableBackendRef> availableBackends, std::string_view backend)
	{
		const auto it = std::ranges::find_if(
		    availableBackends, [&](const VNextAvailableBackendRef& available) { return available.backend == backend; });
		return it == availableBackends.end() ? nullptr : &*it;
	}

	inline bool VNextBackendHasCapability(const VNextAvailableBackendRef& available, std::string_view capability)
	{
		return std::ranges::find(available.capabilities, capability) != available.capabilities.end();
	}

	inline bool VNextBackendSupportsTransferABI(const VNextAvailableBackendRef& available, std::string_view transferABI)
	{
		if (transferABI.empty())
		{
			return false;
		}
		if (transferABI == "none")
		{
			return true;
		}
		return std::ranges::find(available.transferABIs, transferABI) != available.transferABIs.end();
	}

	inline void ValidateVNextArtifactBackendRequirements(const VNextPackageManifest& manifest,
	                                                     std::span<const VNextAvailableBackendRef> availableBackends,
	                                                     VNextBackendRequirementValidationOptions options = {})
	{
		ValidateVNextPackageManifest(manifest);
		for (const auto& artifact : manifest.artifacts)
		{
			for (const auto& requirement : artifact.backendRequirements)
			{
				const auto* available = FindVNextAvailableBackend(availableBackends, requirement.backend);
				if (available == nullptr)
				{
					throw std::runtime_error("vNext artifact '" + artifact.name +
					                         "' requires unavailable backend: " + requirement.backend);
				}
				if (requirement.allowsFallback && !options.allowArtifactFallback)
				{
					throw std::runtime_error("vNext artifact '" + artifact.name + "' backend '" + requirement.backend +
					                         "' allows fallback, but fallback is disabled");
				}
				if (requirement.allowsFallback && !available->allowFallback)
				{
					throw std::runtime_error("vNext artifact '" + artifact.name + "' backend '" + requirement.backend +
					                         "' allows fallback, but the selected backend does not");
				}
				for (const auto& capability : requirement.requiredCapabilities)
				{
					if (!VNextBackendHasCapability(*available, capability))
					{
						throw std::runtime_error("vNext artifact '" + artifact.name + "' backend '" +
						                         requirement.backend +
						                         "' is missing required capability: " + capability);
					}
				}
				if (!VNextBackendSupportsTransferABI(*available, requirement.transferABI))
				{
					throw std::runtime_error("vNext artifact '" + artifact.name + "' backend '" + requirement.backend +
					                         "' does not support transfer ABI: " + requirement.transferABI);
				}
			}
		}
	}

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

	inline VNextPackageManifest BuildVNextPackageManifest(Runtime::RuntimeSchedule schedule,
	                                                      std::vector<VNextArtifactRef> artifacts = {},
	                                                      VNextPackageLayout layout = {},
	                                                      std::vector<VNextAdapterRef> adapters = {},
	                                                      const OpSchemaRegistry& registry = DefaultOpSchemaRegistry())
	{
		ValidateExecutablePlan(schedule.module.plan, registry);
		Runtime::ValidateRuntimeSchedule(schedule);
		VNextPackageManifest manifest;
		manifest.layout = std::move(layout);
		manifest.functions = schedule.module.functions;
		manifest.regions = schedule.module.regions;
		manifest.partitions = schedule.module.partitions;
		manifest.memory = std::move(schedule.memory);
		manifest.runtimeStates = std::move(schedule.states);
		manifest.stateValueBindings = std::move(schedule.stateValueBindings);
		manifest.bufferBindings = std::move(schedule.bufferBindings);
		manifest.runtimeSegments = std::move(schedule.segments);
		manifest.runtimeSteps = std::move(schedule.steps);
		manifest.artifacts = std::move(artifacts);
		manifest.adapters = std::move(adapters);
		manifest.opCoverage = registry.CoverageReport();
		manifest.tensors.reserve(schedule.module.plan.variables.size());
		for (std::size_t i = 0; i < schedule.module.plan.variables.size(); ++i)
		{
			const auto& storage = schedule.module.plan.variables[i];
			manifest.tensors.push_back(ToVNextExternalTensorRef(
			    i < schedule.module.plan.variableNames.size() && !schedule.module.plan.variableNames[i].empty()
			        ? schedule.module.plan.variableNames[i]
			        : (storage.region.name.empty() ? std::format("variable{}", i) : storage.region.name),
			    storage));
		}
		return manifest;
	}

	inline VNextPackageManifest BuildVNextPackageManifest(const ExecutableModule& module,
	                                                      std::vector<VNextArtifactRef> artifacts = {},
	                                                      VNextPackageLayout layout = {},
	                                                      std::vector<VNextAdapterRef> adapters = {},
	                                                      std::vector<Runtime::RuntimeStateBinding> runtimeStates = {},
	                                                      const OpSchemaRegistry& registry = DefaultOpSchemaRegistry())
	{
		auto schedule = Runtime::BuildRuntimeSchedule(module, std::move(runtimeStates));
		return BuildVNextPackageManifest(std::move(schedule), std::move(artifacts), std::move(layout),
		                                 std::move(adapters), registry);
	}

	inline VNextPackageManifest BuildVNextPackageManifest(const ExecutableModule& module,
	                                                      std::vector<VNextArtifactRef> artifacts,
	                                                      VNextPackageLayout layout,
	                                                      std::vector<VNextAdapterRef> adapters,
	                                                      const OpSchemaRegistry& registry)
	{
		return BuildVNextPackageManifest(module, std::move(artifacts), std::move(layout), std::move(adapters), {},
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
		for (std::size_t i = 0; i < manifest.runtimeSegments.size(); ++i)
		{
			const auto& segment = manifest.runtimeSegments[i];
			if (segment.id != i)
			{
				throw std::runtime_error(std::format("vNext runtime segment {} has mismatched id {}", i, segment.id));
			}
			if (segment.backend.empty())
			{
				throw std::runtime_error(std::format("vNext runtime segment {} has empty backend", i));
			}
			if (segment.nodes.empty())
			{
				throw std::runtime_error(std::format("vNext runtime segment {} has no nodes", i));
			}
			for (const auto buffer : segment.inputBuffers)
			{
				if (buffer >= manifest.memory.buffers.size())
				{
					throw std::runtime_error(
					    std::format("vNext runtime segment {} references invalid input buffer", i));
				}
			}
			for (const auto buffer : segment.outputBuffers)
			{
				if (buffer >= manifest.memory.buffers.size())
				{
					throw std::runtime_error(
					    std::format("vNext runtime segment {} references invalid output buffer", i));
				}
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
					throw std::runtime_error(
					    std::format("vNext runtime step {} references unknown dispatch target", i));
				}
				if (step.backend.empty())
				{
					throw std::runtime_error(std::format("vNext runtime dispatch step {} has empty backend", i));
				}
			}
			if (step.kind == Runtime::RuntimeScheduleStepKind::DispatchSegment)
			{
				if (!step.segment || *step.segment >= manifest.runtimeSegments.size())
				{
					throw std::runtime_error(std::format("vNext runtime step {} references unknown segment", i));
				}
				const auto& segment = manifest.runtimeSegments[*step.segment];
				if (step.backend != segment.backend || step.inputBuffers != segment.inputBuffers ||
				    step.outputBuffers != segment.outputBuffers)
				{
					throw std::runtime_error(std::format("vNext runtime step {} does not match segment metadata", i));
				}
			}
			if (step.kind == Runtime::RuntimeScheduleStepKind::Fallback)
			{
				if (step.backend.empty() || step.fallbackBackend.empty())
				{
					throw std::runtime_error(std::format("vNext runtime fallback step {} must name both backends", i));
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
		for (const auto& state : manifest.runtimeStates)
		{
			if (state.name.empty())
			{
				throw std::runtime_error("vNext runtime state binding name cannot be empty");
			}
			if (state.role.empty())
			{
				throw std::runtime_error("vNext runtime state binding role cannot be empty: " + state.name);
			}
			ValidateExecutableTensorType(state.type, "vNext runtime state " + state.name);
			if (!state.memoryBuffer || *state.memoryBuffer >= manifest.memory.buffers.size())
			{
				throw std::runtime_error("vNext runtime state binding references an invalid memory buffer: " +
				                         state.name);
			}
			const auto& buffer = manifest.memory.buffers[*state.memoryBuffer];
			if (buffer.kind != MemoryBufferKind::Persistent)
			{
				throw std::runtime_error("vNext runtime state binding must use a persistent buffer: " + state.name);
			}
			if (buffer.memorySpace != state.type.memorySpace)
			{
				throw std::runtime_error("vNext runtime state binding memory space mismatch: " + state.name);
			}
			if (const auto stateBytes = state.type.ByteSize(); stateBytes && *stateBytes > buffer.byteSize)
			{
				throw std::runtime_error("vNext runtime state binding is larger than its memory buffer: " + state.name);
			}
		}
		for (const auto& binding : manifest.bufferBindings)
		{
			ValidateRuntimeBufferBinding(binding);
			if (binding.memoryBuffer >= manifest.memory.buffers.size())
			{
				throw std::runtime_error("vNext runtime buffer binding references an invalid memory buffer: " +
				                         binding.name);
			}
			const auto& buffer = manifest.memory.buffers[binding.memoryBuffer];
			if (buffer.memorySpace != binding.type.memorySpace)
			{
				throw std::runtime_error("vNext runtime buffer binding memory space mismatch: " + binding.name);
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
		for (const auto& adapter : manifest.adapters)
		{
			if (adapter.targetName.empty())
			{
				throw std::runtime_error("vNext adapter has empty target name");
			}
			if (adapter.adapterName.empty())
			{
				throw std::runtime_error("vNext adapter '" + adapter.targetName + "' has empty adapter name");
			}
			if (adapter.kind != "linear-lora")
			{
				throw std::runtime_error("vNext adapter '" + adapter.targetName +
				                         "' has unsupported kind: " + adapter.kind);
			}
			if (adapter.aTensor >= manifest.tensors.size() || adapter.bTensor >= manifest.tensors.size())
			{
				throw std::runtime_error("vNext adapter '" + adapter.targetName + "' references an unknown tensor");
			}
			if (adapter.rank == 0)
			{
				throw std::runtime_error("vNext adapter '" + adapter.targetName + "' has zero rank");
			}
			if (adapter.alpha == 0.0f)
			{
				throw std::runtime_error("vNext adapter '" + adapter.targetName + "' has zero alpha");
			}
			if (adapter.dropout < 0.0f || adapter.dropout >= 1.0f)
			{
				throw std::runtime_error("vNext adapter '" + adapter.targetName + "' has invalid dropout");
			}
			if (adapter.mergeMode != "unmerged" && adapter.mergeMode != "merged")
			{
				throw std::runtime_error("vNext adapter '" + adapter.targetName +
				                         "' has unsupported merge mode: " + adapter.mergeMode);
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
			if (artifact.entries.empty())
			{
				throw std::runtime_error("vNext artifact '" + artifact.name + "' has no named entries");
			}
			for (const auto& entry : artifact.entries)
			{
				if (entry.name.empty())
				{
					throw std::runtime_error("vNext artifact '" + artifact.name + "' has an entry with empty name");
				}
				if (!IsKnownVNextArtifactEntryKind(entry.kind))
				{
					throw std::runtime_error("vNext artifact '" + artifact.name + "' entry '" + entry.name +
					                         "' has an unknown entry kind");
				}
				if (!entry.function && !entry.sourceSubgraph)
				{
					throw std::runtime_error("vNext artifact '" + artifact.name + "' entry '" + entry.name +
					                         "' must reference a function or source subgraph");
				}
				if (entry.function && *entry.function >= manifest.functions.size())
				{
					throw std::runtime_error("vNext artifact '" + artifact.name + "' entry '" + entry.name +
					                         "' references an unknown function");
				}
				if (entry.sourceSubgraph)
				{
					const auto found = std::ranges::any_of(
					    manifest.regions, [&](const auto& region) { return region.subgraph == *entry.sourceSubgraph; });
					if (!found)
					{
						throw std::runtime_error("vNext artifact '" + artifact.name + "' entry '" + entry.name +
						                         "' references an unknown source subgraph");
					}
				}
				for (const auto& stateName : entry.requiredStateBindings)
				{
					const auto found = std::ranges::any_of(manifest.runtimeStates,
					                                       [&](const auto& state) { return state.name == stateName; });
					if (!found)
					{
						throw std::runtime_error("vNext artifact '" + artifact.name + "' entry '" + entry.name +
						                         "' requires missing runtime state binding: " + stateName);
					}
				}
				for (const auto& bufferName : entry.requiredBufferBindings)
				{
					const auto found = std::ranges::any_of(
					    manifest.bufferBindings, [&](const auto& binding) { return binding.name == bufferName; });
					if (!found)
					{
						throw std::runtime_error("vNext artifact '" + artifact.name + "' entry '" + entry.name +
						                         "' requires missing buffer binding: " + bufferName);
					}
				}
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
			for (const auto& requirement : artifact.backendRequirements)
			{
				if (requirement.backend.empty())
				{
					throw std::runtime_error("vNext artifact '" + artifact.name + "' has an empty backend requirement");
				}
				if (requirement.requiredCapabilities.empty())
				{
					throw std::runtime_error("vNext artifact '" + artifact.name + "' backend requirement for '" +
					                         requirement.backend + "' has no required capabilities");
				}
				if (requirement.transferABI.empty())
				{
					throw std::runtime_error("vNext artifact '" + artifact.name + "' backend requirement for '" +
					                         requirement.backend + "' has empty transfer ABI");
				}
				if (requirement.segment)
				{
					if (*requirement.segment >= manifest.runtimeSegments.size())
					{
						throw std::runtime_error("vNext artifact '" + artifact.name +
						                         "' backend requirement references an unknown runtime segment");
					}
					const auto& segment = manifest.runtimeSegments[*requirement.segment];
					if (segment.backend != requirement.backend)
					{
						throw std::runtime_error("vNext artifact '" + artifact.name +
						                         "' backend requirement backend does not match runtime segment");
					}
				}
			}
		}
	}

	inline void ValidateVNextABIFamily(const VNextPackageManifest& manifest)
	{
		ValidateVNextPackageManifest(manifest);
		const auto summary = DescribeVNextABIFamily(manifest);
		if (!std::ranges::any_of(summary.functions, [](std::string_view name) { return name == "forward"; }))
		{
			throw std::runtime_error("vNext ABI family must expose a named forward entry point");
		}
		if (!summary.hasRuntimeSchedule)
		{
			throw std::runtime_error("vNext ABI family must include runtime schedule metadata");
		}
	}
} // namespace LiteNN

#endif
