#ifndef LITENN_PRODUCTION_SUPPORT_H
#define LITENN_PRODUCTION_SUPPORT_H

#include <string>
#include <string_view>
#include <vector>

namespace LiteNN
{
	enum class ProductionSupportArea
	{
		CPURuntime,
		CPUAOT,
		CUDARuntime,
		VulkanRuntime,
		Importers,
		VNextPackaging,
		SeparatedArtifacts,
		Benchmarks,
		MobileRuntime,
		TrainingAOT,
		SDXLGeneration,
	};

	enum class ProductionPath
	{
		CPUInterpreter,
		CPUAOTSeparatedArtifact,
		CUDANativeGraphReplay,
		CUDACPUBridgeFallback,
		VulkanNativeSeparatedArtifact,
		VNextModelPackage,
		ImporterManifest,
		MobileSeparatedRuntime,
	};

	enum class ProductionSupportLevel
	{
		Production,
		Supported,
		Experimental,
		Deferred,
		Unavailable,
	};

	enum class ProductionBackendProfile
	{
		CPUReferenceInterpreter,
		CPUAOTSeparatedArtifact,
		CUDANativeGraphReplay,
		CUDACPUBridgeFallback,
		VulkanDesktopNative,
		VulkanMobileConstrained,
	};

	struct ProductionSupportStatus
	{
		ProductionSupportArea area;
		std::string_view name;
		ProductionSupportLevel level;
		bool availableInBuild;
		std::string_view policy;
	};

	struct ProductionPathABIDescriptor
	{
		ProductionPath path;
		std::string_view name;
		ProductionSupportArea area;
		ProductionSupportLevel level;
		bool availableInBuild;
		std::string_view inputs;
		std::string_view outputs;
		std::string_view mutableState;
		std::string_view externalTensors;
		std::string_view ownership;
		std::string_view alignment;
		std::string_view checksum;
		std::string_view fallbackPolicy;
		bool usesMutableState;
		bool supportsExternalTensors;
		bool requiresCallerOwnedBuffers;
		bool usesSeparatedRegions;
		bool usesAlignmentMetadata;
		bool usesChecksums;
		bool allowsHostFallback;
		bool requiresStableDevicePointers;
	};

	struct ProductionBackendProfileDescriptor
	{
		ProductionBackendProfile profile;
		std::string_view name;
		ProductionPath path;
		ProductionSupportArea area;
		ProductionSupportLevel level;
		bool availableInBuild;
		bool referenceCorrectnessPath;
		bool desktopProfile;
		bool mobileProfile;
		bool nativeDeviceProfile;
		bool requiresDeviceCapabilityProbe;
		bool requiresScheduleProfileVisibility;
		bool allowsHostFallback;
		std::string_view verifiedScope;
		std::string_view missingBeforeProduction;
		std::string_view skipOrFailurePolicy;
	};

	struct ProductionCPUKernelStrategy
	{
		std::string_view referencePath;
		std::string_view throughputPolicy;
		bool preferExternalLibraryBackend;
		bool allowSmallNativeKernelSet;
		bool allowUnplannedHandwrittenGemmOrConv;
		std::string_view handwrittenKernelGate;
	};

	inline constexpr std::string_view ProductionSupportAreaName(ProductionSupportArea area)
	{
		switch (area)
		{
		case ProductionSupportArea::CPURuntime:
			return "cpu-runtime";
		case ProductionSupportArea::CPUAOT:
			return "cpu-aot";
		case ProductionSupportArea::CUDARuntime:
			return "cuda-runtime";
		case ProductionSupportArea::VulkanRuntime:
			return "vulkan-runtime";
		case ProductionSupportArea::Importers:
			return "importers";
		case ProductionSupportArea::VNextPackaging:
			return "vnext-packaging";
		case ProductionSupportArea::SeparatedArtifacts:
			return "separated-artifacts";
		case ProductionSupportArea::Benchmarks:
			return "benchmarks";
		case ProductionSupportArea::MobileRuntime:
			return "mobile-runtime";
		case ProductionSupportArea::TrainingAOT:
			return "training-aot";
		case ProductionSupportArea::SDXLGeneration:
			return "sdxl-generation";
		}
		return "unknown";
	}

	inline constexpr std::string_view ProductionPathName(ProductionPath path)
	{
		switch (path)
		{
		case ProductionPath::CPUInterpreter:
			return "cpu-interpreter";
		case ProductionPath::CPUAOTSeparatedArtifact:
			return "cpu-aot-separated-artifact";
		case ProductionPath::CUDANativeGraphReplay:
			return "cuda-native-graph-replay";
		case ProductionPath::CUDACPUBridgeFallback:
			return "cuda-cpu-bridge-fallback";
		case ProductionPath::VulkanNativeSeparatedArtifact:
			return "vulkan-native-separated-artifact";
		case ProductionPath::VNextModelPackage:
			return "vnext-model-package";
		case ProductionPath::ImporterManifest:
			return "importer-manifest";
		case ProductionPath::MobileSeparatedRuntime:
			return "mobile-separated-runtime";
		}
		return "unknown";
	}

	inline constexpr std::string_view ProductionSupportLevelName(ProductionSupportLevel level)
	{
		switch (level)
		{
		case ProductionSupportLevel::Production:
			return "production";
		case ProductionSupportLevel::Supported:
			return "supported";
		case ProductionSupportLevel::Experimental:
			return "experimental";
		case ProductionSupportLevel::Deferred:
			return "deferred";
		case ProductionSupportLevel::Unavailable:
			return "unavailable";
		}
		return "unknown";
	}

	inline constexpr std::string_view ProductionBackendProfileName(ProductionBackendProfile profile)
	{
		switch (profile)
		{
		case ProductionBackendProfile::CPUReferenceInterpreter:
			return "cpu-reference-interpreter";
		case ProductionBackendProfile::CPUAOTSeparatedArtifact:
			return "cpu-aot-separated-artifact";
		case ProductionBackendProfile::CUDANativeGraphReplay:
			return "cuda-native-graph-replay";
		case ProductionBackendProfile::CUDACPUBridgeFallback:
			return "cuda-cpu-bridge-fallback";
		case ProductionBackendProfile::VulkanDesktopNative:
			return "vulkan-desktop-native";
		case ProductionBackendProfile::VulkanMobileConstrained:
			return "vulkan-mobile-constrained";
		}
		return "unknown";
	}

	inline constexpr bool ProductionBuildHasCUDA()
	{
#ifdef LITENN_ENABLE_CUDA
		return true;
#else
		return false;
#endif
	}

	inline constexpr bool ProductionBuildHasVulkan()
	{
#ifdef LITENN_ENABLE_VULKAN
		return true;
#else
		return false;
#endif
	}

	inline constexpr bool ProductionBuildHasMLIR()
	{
#ifdef LITENN_ENABLE_MLIR
		return true;
#else
		return false;
#endif
	}

	inline constexpr ProductionSupportStatus QueryProductionSupportStatus(ProductionSupportArea area)
	{
		switch (area)
		{
		case ProductionSupportArea::CPURuntime:
			return { area, ProductionSupportAreaName(area), ProductionSupportLevel::Production, true,
				     "CPU interpreter is the reference correctness and diagnostics runtime." };
		case ProductionSupportArea::CPUAOT:
			return { area, ProductionSupportAreaName(area),
				     ProductionBuildHasMLIR() ? ProductionSupportLevel::Production
				                              : ProductionSupportLevel::Unavailable,
				     ProductionBuildHasMLIR(),
				     "CPU AOT with vNext/separated artifacts is the reference production deployment path when the "
				     "compiler stack is enabled." };
		case ProductionSupportArea::CUDARuntime:
			return { area, ProductionSupportAreaName(area),
				     ProductionBuildHasCUDA() ? ProductionSupportLevel::Supported : ProductionSupportLevel::Unavailable,
				     ProductionBuildHasCUDA(),
				     "CUDA native runtime is capability-gated; production loops should use explicit CUDA Graph replay "
				     "for pointer-stable static-shape inference." };
		case ProductionSupportArea::VulkanRuntime:
			return { area, ProductionSupportAreaName(area),
				     ProductionBuildHasVulkan() ? ProductionSupportLevel::Experimental
				                                : ProductionSupportLevel::Unavailable,
				     ProductionBuildHasVulkan(),
				     "Vulkan native runtime is available for selected static-shape workloads; broad mobile GPU support "
				     "still requires graph partitioning and device-local memory planning." };
		case ProductionSupportArea::Importers:
			return { area, ProductionSupportAreaName(area), ProductionSupportLevel::Supported, true,
				     "Importers are manifest/package producers; safetensors is tensor storage, not architecture "
				     "discovery." };
		case ProductionSupportArea::VNextPackaging:
			return { area, ProductionSupportAreaName(area), ProductionSupportLevel::Production, true,
				     "vNext package metadata is the production model/package format; old graph archive compatibility is "
				     "not a production path." };
		case ProductionSupportArea::SeparatedArtifacts:
			return { area, ProductionSupportAreaName(area), ProductionSupportLevel::Production, true,
				     "Separated metadata, constants, weights, and instructions are the production artifact ABI for "
				     "host-compiled deployment." };
		case ProductionSupportArea::Benchmarks:
			return { area, ProductionSupportAreaName(area), ProductionSupportLevel::Supported, true,
				     "Benchmark/profile output is required evidence for production backend claims." };
		case ProductionSupportArea::MobileRuntime:
			return { area, ProductionSupportAreaName(area), ProductionSupportLevel::Experimental, true,
				     "Mobile support is constrained to minimal runtime loading and Vulkan-oriented profiles; desktop "
				     "CUDA, MLIR, object JIT, and carrier loading are excluded." };
		case ProductionSupportArea::TrainingAOT:
			return { area, ProductionSupportAreaName(area), ProductionSupportLevel::Deferred, false,
				     "Compiled training remains a future production path after explicit train-step ABI and saved-state "
				     "binding are stabilized." };
		case ProductionSupportArea::SDXLGeneration:
			return { area, ProductionSupportAreaName(area), ProductionSupportLevel::Deferred, false,
				     "SDXL remains a large-model importer and memory-policy stress target, not a production image "
				     "generation promise." };
		}
		return { area, ProductionSupportAreaName(area), ProductionSupportLevel::Unavailable, false,
			     "Unknown production support area." };
	}

	inline constexpr ProductionPathABIDescriptor QueryProductionPathABI(ProductionPath path)
	{
		switch (path)
		{
		case ProductionPath::CPUInterpreter:
			return { path,
				     ProductionPathName(path),
				     ProductionSupportArea::CPURuntime,
				     ProductionSupportLevel::Production,
				     true,
				     "Host tensors are passed through the plan/native runtime entry point.",
				     "Host tensors are returned or written through explicit output tensors.",
				     "No hidden mutable state; callers model state as explicit tensors.",
				     "Not an artifact boundary, so no external rodata/weight tensor binding.",
				     "Caller owns Tensor storage; interpreter keeps no cross-call external pointers.",
				     "Tensor allocator alignment only.",
				     "No package checksum boundary.",
				     "Reference path; no lower fallback.",
				     false,
				     false,
				     true,
				     false,
				     false,
				     false,
				     false,
				     false };
		case ProductionPath::CPUAOTSeparatedArtifact:
			return { path,
				     ProductionPathName(path),
				     ProductionSupportArea::CPUAOT,
				     ProductionBuildHasMLIR() ? ProductionSupportLevel::Production
				                              : ProductionSupportLevel::Unavailable,
				     ProductionBuildHasMLIR(),
				     "Entry-point tensors are bound by vNext metadata name/order.",
				     "Outputs are explicit entry-point tensors; no implicit graph archive side channel.",
				     "Runtime state is explicit in the artifact ABI or rejected.",
				     "Constants, weights, rodata, and instruction regions are separate external bindings.",
				     "Caller owns external regions for the module lifetime; runtime owns only decoded handles.",
				     "Region alignment is described by separated artifact metadata.",
				     "Metadata and external regions carry package checksums.",
				     "No implicit fallback; unsupported nodes must fail during compile or load.",
				     true,
				     true,
				     true,
				     true,
				     true,
				     true,
				     false,
				     false };
		case ProductionPath::CUDANativeGraphReplay:
			return { path,
				     ProductionPathName(path),
				     ProductionSupportArea::CUDARuntime,
				     ProductionBuildHasCUDA() ? ProductionSupportLevel::Supported
				                              : ProductionSupportLevel::Unavailable,
				     ProductionBuildHasCUDA(),
				     "Device tensors are bound to a static-shape native CUDA schedule.",
				     "Device output tensors must be pre-bound and pointer-stable across replay.",
				     "Mutable state is limited to explicitly bound device tensors.",
				     "External device buffers are supported when addresses remain stable.",
				     "Caller owns device buffers and CUDA stream lifetime when an external stream is used.",
				     "CUDA allocation alignment and kernel vectorization requirements are backend-owned.",
				     "No package checksum boundary for raw native runtime replay.",
				     "Reject unsupported graph replay constraints; do not silently drop to per-op launch.",
				     true,
				     true,
				     true,
				     false,
				     true,
				     false,
				     false,
				     true };
		case ProductionPath::CUDACPUBridgeFallback:
			return { path,
				     ProductionPathName(path),
				     ProductionSupportArea::CUDARuntime,
				     ProductionBuildHasCUDA() ? ProductionSupportLevel::Experimental
				                              : ProductionSupportLevel::Unavailable,
				     ProductionBuildHasCUDA(),
				     "CUDA-capable subgraphs use device tensors; unsupported segments bind through host tensors.",
				     "Outputs may cross device/host boundaries through explicit transfer records.",
				     "Mutable state must be surfaced in the schedule before fallback is accepted.",
				     "External tensor support is split by device/host segment ownership.",
				     "Caller owns original tensors; bridge owns only explicit transfer temporaries.",
				     "Device and host alignment requirements are recorded per segment.",
				     "No unified checksum boundary until schedules and package ABI converge.",
				     "Host fallback is allowed only when the schedule/profile records it visibly.",
				     true,
				     true,
				     true,
				     false,
				     true,
				     false,
				     true,
				     false };
		case ProductionPath::VulkanNativeSeparatedArtifact:
			return { path,
				     ProductionPathName(path),
				     ProductionSupportArea::VulkanRuntime,
				     ProductionBuildHasVulkan() ? ProductionSupportLevel::Experimental
				                                : ProductionSupportLevel::Unavailable,
				     ProductionBuildHasVulkan(),
				     "Static-shape tensors are bound through Vulkan native payload metadata.",
				     "Outputs are explicit storage buffers with device capability checks.",
				     "Mutable state is not production-stable beyond explicit buffers.",
				     "Separated weights/rodata are supported for selected native payloads.",
				     "Caller owns package regions; runtime owns Vulkan resources created from them.",
				     "Storage buffer alignment is validated against device limits.",
				     "Separated package regions carry checksums where vNext metadata is present.",
				     "Unsupported device capabilities fail load/run instead of falling back invisibly.",
				     true,
				     true,
				     true,
				     true,
				     true,
				     true,
				     false,
				     false };
		case ProductionPath::VNextModelPackage:
			return { path,
				     ProductionPathName(path),
				     ProductionSupportArea::VNextPackaging,
				     ProductionSupportLevel::Production,
				     true,
				     "Named graph/module inputs are part of the package manifest.",
				     "Named graph/module outputs are part of the package manifest.",
				     "State tensors must be declared as ABI-visible state slots.",
				     "External constants, weights, rodata, and instructions are first-class regions.",
				     "Package metadata names the owner and lifetime for every region.",
				     "Alignment is declared per external region.",
				     "Package metadata carries checksums for manifest and external regions.",
				     "Fallback policy must be declared by the consuming runtime, not inferred by the package.",
				     true,
				     true,
				     true,
				     true,
				     true,
				     true,
				     false,
				     false };
		case ProductionPath::ImporterManifest:
			return { path,
				     ProductionPathName(path),
				     ProductionSupportArea::Importers,
				     ProductionSupportLevel::Supported,
				     true,
				     "Importer manifests declare required model/config inputs before graph construction.",
				     "Importer reports declare produced graph/package outputs.",
				     "Importer-specific state must map to explicit package state slots.",
				     "External tensors from safetensors/GGUF stay external until packaging chooses storage.",
				     "Source tensor files remain caller-owned; importer owns decoded metadata and diagnostics.",
				     "Source and target alignment are reported when they affect zero-copy or packing.",
				     "Source tensor checksums are preserved when the source format provides them.",
				     "Unsupported ops, layouts, dtypes, and quantization schemes are rejected with diagnostics.",
				     true,
				     true,
				     true,
				     false,
				     true,
				     true,
				     false,
				     false };
		case ProductionPath::MobileSeparatedRuntime:
			return { path,
				     ProductionPathName(path),
				     ProductionSupportArea::MobileRuntime,
				     ProductionSupportLevel::Experimental,
				     true,
				     "Mobile entry points use package-declared tensors and a constrained runtime profile.",
				     "Outputs are explicit tensors; background transfers must be profile-visible.",
				     "Mutable state is allowed only as package-declared state tensors.",
				     "External regions are required to avoid monolithic mobile package loading.",
				     "Application owns memory-mapped regions; runtime owns decoded lightweight handles.",
				     "Region alignment must satisfy mobile filesystem, mmap, and backend limits.",
				     "Manifest and external regions carry checksums before execution.",
				     "Fallback is profile-defined; unavailable GPU features must fail or use an explicit CPU path.",
				     true,
				     true,
				     true,
				     true,
				     true,
				     true,
				     true,
				     false };
		}

		return { path,
			     ProductionPathName(path),
			     ProductionSupportArea::VNextPackaging,
			     ProductionSupportLevel::Unavailable,
			     false,
			     "Unknown path.",
			     "Unknown path.",
			     "Unknown path.",
			     "Unknown path.",
			     "Unknown path.",
			     "Unknown path.",
			     "Unknown path.",
			     "Unknown path.",
			     false,
			     false,
			     false,
			     false,
			     false,
			     false,
			     false,
			     false };
	}

	inline constexpr ProductionBackendProfileDescriptor QueryProductionBackendProfile(
	    ProductionBackendProfile profile)
	{
		switch (profile)
		{
		case ProductionBackendProfile::CPUReferenceInterpreter:
			return { profile,
				     ProductionBackendProfileName(profile),
				     ProductionPath::CPUInterpreter,
				     ProductionSupportArea::CPURuntime,
				     ProductionSupportLevel::Production,
				     true,
				     true,
				     true,
				     false,
				     false,
				     false,
				     true,
				     false,
				     "Reference correctness, diagnostics, constant evaluation, and fallback-free host execution.",
				     "Not intended to be the peak-throughput production kernel strategy.",
				     "Always available; failures are graph/runtime validation errors, not device skips." };
		case ProductionBackendProfile::CPUAOTSeparatedArtifact:
			return { profile,
				     ProductionBackendProfileName(profile),
				     ProductionPath::CPUAOTSeparatedArtifact,
				     ProductionSupportArea::CPUAOT,
				     ProductionBuildHasMLIR() ? ProductionSupportLevel::Production
				                              : ProductionSupportLevel::Unavailable,
				     ProductionBuildHasMLIR(),
				     false,
				     true,
				     false,
				     false,
				     false,
				     true,
				     false,
				     "Reference deployment profile for CPU AOT packages with separated rodata/weights/instructions.",
				     "External CPU kernel-library strategy is still undecided.",
				     "Unavailable MLIR/compiler support must fail configure/build or route to interpreter explicitly." };
		case ProductionBackendProfile::CUDANativeGraphReplay:
			return { profile,
				     ProductionBackendProfileName(profile),
				     ProductionPath::CUDANativeGraphReplay,
				     ProductionSupportArea::CUDARuntime,
				     ProductionBuildHasCUDA() ? ProductionSupportLevel::Supported
				                              : ProductionSupportLevel::Unavailable,
				     ProductionBuildHasCUDA(),
				     false,
				     true,
				     false,
				     true,
				     true,
				     true,
				     false,
				     "Static-shape native CUDA execution with pointer-stable graph replay.",
				     "Kernel coverage still needs high-value Linear/MatMul, normalization, attention, and quantized paths.",
				     "Unsupported replay constraints fail loudly instead of silently switching execution mode." };
		case ProductionBackendProfile::CUDACPUBridgeFallback:
			return { profile,
				     ProductionBackendProfileName(profile),
				     ProductionPath::CUDACPUBridgeFallback,
				     ProductionSupportArea::CUDARuntime,
				     ProductionBuildHasCUDA() ? ProductionSupportLevel::Experimental
				                              : ProductionSupportLevel::Unavailable,
				     ProductionBuildHasCUDA(),
				     false,
				     true,
				     false,
				     true,
				     true,
				     true,
				     true,
				     "Diagnostic/profile-visible bridge for unsupported CUDA segments.",
				     "Not a production-performance claim until transfer/fallback rows are separated in benchmarks.",
				     "Host fallback is allowed only when schedule/profile records expose the fallback step." };
		case ProductionBackendProfile::VulkanDesktopNative:
			return { profile,
				     ProductionBackendProfileName(profile),
				     ProductionPath::VulkanNativeSeparatedArtifact,
				     ProductionSupportArea::VulkanRuntime,
				     ProductionBuildHasVulkan() ? ProductionSupportLevel::Experimental
				                                : ProductionSupportLevel::Unavailable,
				     ProductionBuildHasVulkan(),
				     false,
				     true,
				     false,
				     true,
				     true,
				     true,
				     false,
				     "Desktop Vulkan native payloads for selected static-shape workloads.",
				     "Graph partitioning and a clearer device-local memory planner remain before broader production claims.",
				     "Missing storage, subgroup, timestamp, or alignment capabilities must skip or fail explicitly." };
		case ProductionBackendProfile::VulkanMobileConstrained:
			return { profile,
				     ProductionBackendProfileName(profile),
				     ProductionPath::MobileSeparatedRuntime,
				     ProductionSupportArea::MobileRuntime,
				     ProductionSupportLevel::Experimental,
				     ProductionBuildHasVulkan(),
				     false,
				     false,
				     true,
				     true,
				     true,
				     true,
				     true,
				     "Constrained mobile runtime profile using vNext packages and Vulkan-oriented explicit capabilities.",
				     "Needs mobile device matrix, memory-mapping policy, and device-local allocation planning.",
				     "Unavailable mobile GPU features must skip/fail explicitly or use a declared CPU path." };
		}
		return { profile,
			     ProductionBackendProfileName(profile),
			     ProductionPath::VNextModelPackage,
			     ProductionSupportArea::VNextPackaging,
			     ProductionSupportLevel::Unavailable,
			     false,
			     false,
			     false,
			     false,
			     false,
			     false,
			     false,
			     false,
			     "Unknown backend profile.",
			     "Unknown backend profile.",
			     "Unknown backend profile." };
	}

	inline constexpr ProductionCPUKernelStrategy QueryProductionCPUKernelStrategy()
	{
		return { "cpu-reference-interpreter",
			     "CPU production throughput should come from an explicit external-library backend or a deliberately "
			     "small maintained native kernel set.",
			     true,
			     true,
			     false,
			     "New handwritten CPU GEMM/Conv kernels require a measured gap, a workload owner, and benchmark "
			     "evidence before they enter the production profile." };
	}

	inline std::vector<ProductionSupportStatus> QueryProductionSupportStatuses()
	{
		return {
			QueryProductionSupportStatus(ProductionSupportArea::CPURuntime),
			QueryProductionSupportStatus(ProductionSupportArea::CPUAOT),
			QueryProductionSupportStatus(ProductionSupportArea::CUDARuntime),
			QueryProductionSupportStatus(ProductionSupportArea::VulkanRuntime),
			QueryProductionSupportStatus(ProductionSupportArea::Importers),
			QueryProductionSupportStatus(ProductionSupportArea::VNextPackaging),
			QueryProductionSupportStatus(ProductionSupportArea::SeparatedArtifacts),
			QueryProductionSupportStatus(ProductionSupportArea::Benchmarks),
			QueryProductionSupportStatus(ProductionSupportArea::MobileRuntime),
			QueryProductionSupportStatus(ProductionSupportArea::TrainingAOT),
			QueryProductionSupportStatus(ProductionSupportArea::SDXLGeneration),
		};
	}

	inline std::vector<ProductionBackendProfileDescriptor> QueryProductionBackendProfiles()
	{
		return {
			QueryProductionBackendProfile(ProductionBackendProfile::CPUReferenceInterpreter),
			QueryProductionBackendProfile(ProductionBackendProfile::CPUAOTSeparatedArtifact),
			QueryProductionBackendProfile(ProductionBackendProfile::CUDANativeGraphReplay),
			QueryProductionBackendProfile(ProductionBackendProfile::CUDACPUBridgeFallback),
			QueryProductionBackendProfile(ProductionBackendProfile::VulkanDesktopNative),
			QueryProductionBackendProfile(ProductionBackendProfile::VulkanMobileConstrained),
		};
	}

	inline std::vector<ProductionPathABIDescriptor> QueryProductionPathABIs()
	{
		return {
			QueryProductionPathABI(ProductionPath::CPUInterpreter),
			QueryProductionPathABI(ProductionPath::CPUAOTSeparatedArtifact),
			QueryProductionPathABI(ProductionPath::CUDANativeGraphReplay),
			QueryProductionPathABI(ProductionPath::CUDACPUBridgeFallback),
			QueryProductionPathABI(ProductionPath::VulkanNativeSeparatedArtifact),
			QueryProductionPathABI(ProductionPath::VNextModelPackage),
			QueryProductionPathABI(ProductionPath::ImporterManifest),
			QueryProductionPathABI(ProductionPath::MobileSeparatedRuntime),
		};
	}

	inline std::vector<std::string> CollectProductionSupportDiagnostics()
	{
		std::vector<std::string> diagnostics;
		for (const auto& status : QueryProductionSupportStatuses())
		{
			if (status.level != ProductionSupportLevel::Production)
			{
				diagnostics.push_back(std::string(status.name) + " [" +
				                      std::string(ProductionSupportLevelName(status.level)) + "]: " +
				                      std::string(status.policy));
			}
		}
		return diagnostics;
	}

	inline std::vector<std::string> CollectProductionPathABIDiagnostics()
	{
		std::vector<std::string> diagnostics;
		for (const auto& descriptor : QueryProductionPathABIs())
		{
			if (descriptor.level != ProductionSupportLevel::Production || descriptor.allowsHostFallback)
			{
				diagnostics.push_back(std::string(descriptor.name) + " [" +
				                      std::string(ProductionSupportLevelName(descriptor.level)) + "]: " +
				                      std::string(descriptor.fallbackPolicy));
			}
		}
		return diagnostics;
	}
} // namespace LiteNN

#endif
