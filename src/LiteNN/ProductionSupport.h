#ifndef LITENN_PRODUCTION_SUPPORT_H
#define LITENN_PRODUCTION_SUPPORT_H

#include <LiteNN/Misc.h>

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

	enum class ProductionCUDANativeCapability
	{
		StaticShapeDeviceTensorABI,
		GraphReplay,
		ElementwiseF32,
		MatMulF32,
		LinearChainF32,
		ReductionF32,
		Normalization,
		ConcatSliceF32,
		LowPrecisionCast,
		LowPrecisionMatMul,
		Attention,
		QuantizedProjection,
	};

	enum class ProductionQuantizationCapability
	{
		ScalarLowPrecisionDataTypes,
		AffineQuantizedTensors,
		BlockQuantizedStorage,
		PackedFourBitStorage,
		CPUReferencePackUnpackDequantize,
		VNextQuantizationMetadata,
		NativeQuantizedLinearMatMul,
	};

	enum class ProductionSDXLCapability
	{
		TorchManifestDiffusionOps,
		ExternalWeightPackaging,
		CompiledDenoiserSmoke,
		EulerSamplerHarness,
		VAEDecodeStress,
		NativePromptConditioning,
		ReferenceImageParity1024,
		ProductionPromptToImage,
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

	struct ProductionCUDANativeCapabilityDescriptor
	{
		ProductionCUDANativeCapability capability;
		std::string_view name;
		ProductionSupportLevel level;
		bool availableInBuild;
		bool requiresCUDADevice;
		bool requiresRuntimeDeviceProbe;
		bool requiresStablePointers;
		bool highValueKernelPriority;
		bool allowsHostFallback;
		std::string_view verifiedScope;
		std::string_view capabilityGate;
		std::string_view fallbackPolicy;
	};

	struct ProductionQuantizationCapabilityDescriptor
	{
		ProductionQuantizationCapability capability;
		std::string_view name;
		ProductionSupportLevel level;
		bool availableInBuild;
		bool semanticFoundation;
		bool nativeKernel;
		bool requiresExternalMetadata;
		std::string_view verifiedScope;
		std::string_view productionGate;
		std::string_view fallbackPolicy;
	};

	struct ProductionSDXLCapabilityDescriptor
	{
		ProductionSDXLCapability capability;
		std::string_view name;
		ProductionSupportLevel level;
		bool availableInBuild;
		bool importerOrStressTarget;
		bool productionGenerationGate;
		bool blocksVNextProductionProfile;
		std::string_view verifiedScope;
		std::string_view missingBeforeProduction;
		std::string_view fallbackPolicy;
	};

	inline constexpr std::string_view ProductionSupportAreaName(ProductionSupportArea area)
	{
		return EnumToString<EnumToStringStyle::Unqualified>(area);
	}

	inline constexpr std::string_view ProductionPathName(ProductionPath path)
	{
		return EnumToString<EnumToStringStyle::Unqualified>(path);
	}

	inline constexpr std::string_view ProductionSupportLevelName(ProductionSupportLevel level)
	{
		return EnumToString<EnumToStringStyle::Unqualified>(level);
	}

	inline constexpr std::string_view ProductionBackendProfileName(ProductionBackendProfile profile)
	{
		return EnumToString<EnumToStringStyle::Unqualified>(profile);
	}

	inline constexpr std::string_view ProductionCUDANativeCapabilityName(ProductionCUDANativeCapability capability)
	{
		return EnumToString<EnumToStringStyle::Unqualified>(capability);
	}

	inline constexpr std::string_view ProductionQuantizationCapabilityName(ProductionQuantizationCapability capability)
	{
		return EnumToString<EnumToStringStyle::Unqualified>(capability);
	}

	inline constexpr std::string_view ProductionSDXLCapabilityName(ProductionSDXLCapability capability)
	{
		return EnumToString<EnumToStringStyle::Unqualified>(capability);
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
			return {
				area, ProductionSupportAreaName(area), ProductionSupportLevel::Production, true,
				"vNext package metadata is the production model/package format; old graph archive compatibility is "
				"not a production path."
			};
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
				     ProductionBuildHasCUDA() ? ProductionSupportLevel::Supported : ProductionSupportLevel::Unavailable,
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

	inline constexpr ProductionBackendProfileDescriptor QueryProductionBackendProfile(ProductionBackendProfile profile)
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
			return {
				profile,
				ProductionBackendProfileName(profile),
				ProductionPath::CPUAOTSeparatedArtifact,
				ProductionSupportArea::CPUAOT,
				ProductionBuildHasMLIR() ? ProductionSupportLevel::Production : ProductionSupportLevel::Unavailable,
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
				"Unavailable MLIR/compiler support must fail configure/build or route to interpreter explicitly."
			};
		case ProductionBackendProfile::CUDANativeGraphReplay:
			return {
				profile,
				ProductionBackendProfileName(profile),
				ProductionPath::CUDANativeGraphReplay,
				ProductionSupportArea::CUDARuntime,
				ProductionBuildHasCUDA() ? ProductionSupportLevel::Supported : ProductionSupportLevel::Unavailable,
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
				"Unsupported replay constraints fail loudly instead of silently switching execution mode."
			};
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
			return {
				profile,
				ProductionBackendProfileName(profile),
				ProductionPath::VulkanNativeSeparatedArtifact,
				ProductionSupportArea::VulkanRuntime,
				ProductionBuildHasVulkan() ? ProductionSupportLevel::Experimental : ProductionSupportLevel::Unavailable,
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
				"Missing storage, subgroup, timestamp, or alignment capabilities must skip or fail explicitly."
			};
		case ProductionBackendProfile::VulkanMobileConstrained:
			return {
				profile,
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
				"Unavailable mobile GPU features must skip/fail explicitly or use a declared CPU path."
			};
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

	inline constexpr ProductionCUDANativeCapabilityDescriptor
	QueryProductionCUDANativeCapability(ProductionCUDANativeCapability capability)
	{
		const auto unavailableLevel =
		    ProductionBuildHasCUDA() ? ProductionSupportLevel::Deferred : ProductionSupportLevel::Unavailable;
		switch (capability)
		{
		case ProductionCUDANativeCapability::StaticShapeDeviceTensorABI:
			return { capability,
				     ProductionCUDANativeCapabilityName(capability),
				     ProductionBuildHasCUDA() ? ProductionSupportLevel::Supported : ProductionSupportLevel::Unavailable,
				     ProductionBuildHasCUDA(),
				     true,
				     true,
				     false,
				     false,
				     false,
				     "Static-shape CUDA Tensor input/output ABI with explicit backend tag and native payload metadata.",
				     "Requires CUDA build support plus a load-time CUDA device/driver probe.",
				     "Unsupported native ABI must fail or use a separately declared CPU bridge path." };
		case ProductionCUDANativeCapability::GraphReplay:
			return { capability,
				     ProductionCUDANativeCapabilityName(capability),
				     ProductionBuildHasCUDA() ? ProductionSupportLevel::Supported : ProductionSupportLevel::Unavailable,
				     ProductionBuildHasCUDA(),
				     true,
				     true,
				     true,
				     true,
				     false,
				     "Pointer-stable synchronized native CUDA replay for static-shape inference payloads.",
				     "Requires CUDA build support, CUDA runtime/driver availability, and module-owned default stream.",
				     "Unsupported replay constraints fail loudly rather than dropping to non-graph launch." };
		case ProductionCUDANativeCapability::ElementwiseF32:
			return { capability,
				     ProductionCUDANativeCapabilityName(capability),
				     ProductionBuildHasCUDA() ? ProductionSupportLevel::Supported : ProductionSupportLevel::Unavailable,
				     ProductionBuildHasCUDA(),
				     true,
				     true,
				     false,
				     false,
				     false,
				     "Float32 unary/binary elementwise and same-rank broadcast native payloads.",
				     "Requires CUDA build support and successful native payload load for the target device.",
				     "Unsupported shapes or dtypes route only through explicit CPU bridge fallback." };
		case ProductionCUDANativeCapability::MatMulF32:
		case ProductionCUDANativeCapability::LinearChainF32:
			return { capability,
				     ProductionCUDANativeCapabilityName(capability),
				     ProductionBuildHasCUDA() ? ProductionSupportLevel::Supported : ProductionSupportLevel::Unavailable,
				     ProductionBuildHasCUDA(),
				     true,
				     true,
				     false,
				     true,
				     false,
				     capability == ProductionCUDANativeCapability::MatMulF32
				         ? "Float32 MatMul native payloads backed by CUDA library calls."
				         : "Float32 fused Linear/MLP chains with native CUDA launch scheduling.",
				     "Requires CUDA build support, CUDA device availability, and supported static tensor shapes.",
				     "Unsupported graphs must expose CPU bridge fallback as a separate profile row." };
		case ProductionCUDANativeCapability::ReductionF32:
		case ProductionCUDANativeCapability::ConcatSliceF32:
			return { capability,
				     ProductionCUDANativeCapabilityName(capability),
				     ProductionBuildHasCUDA() ? ProductionSupportLevel::Supported : ProductionSupportLevel::Unavailable,
				     ProductionBuildHasCUDA(),
				     true,
				     true,
				     false,
				     false,
				     false,
				     capability == ProductionCUDANativeCapability::ReductionF32
				         ? "Selected static-axis Float32 reductions."
				         : "Selected static-shape Float32 concat/slice payloads.",
				     "Requires CUDA build support and supported static shape/layout metadata.",
				     "Unsupported variants are rejected by native lowering and must be visible as fallback." };
		case ProductionCUDANativeCapability::Normalization:
			return { capability,
				     ProductionCUDANativeCapabilityName(capability),
				     unavailableLevel,
				     ProductionBuildHasCUDA(),
				     true,
				     true,
				     false,
				     true,
				     false,
				     "Production normalization kernels are not yet part of the verified CUDA native profile.",
				     "Requires explicit LayerNorm/RMSNorm kernel implementation, device capability checks, and parity "
				     "evidence.",
				     "Until implemented, normalization workloads must not be advertised as CUDA native production "
				     "support." };
		case ProductionCUDANativeCapability::LowPrecisionCast:
			return { capability,
				     ProductionCUDANativeCapabilityName(capability),
				     ProductionBuildHasCUDA() ? ProductionSupportLevel::Experimental
				                              : ProductionSupportLevel::Unavailable,
				     ProductionBuildHasCUDA(),
				     true,
				     true,
				     false,
				     false,
				     false,
				     "Native low-precision cast payloads where the CUDA device and target ISA support them.",
				     "Requires dtype-specific device capability checks such as FP8 target support.",
				     "Unsupported dtype/device pairs must fail native execution or use explicit bridge conversion." };
		case ProductionCUDANativeCapability::LowPrecisionMatMul:
			return { capability,
				     ProductionCUDANativeCapabilityName(capability),
				     ProductionBuildHasCUDA() ? ProductionSupportLevel::Experimental
				                              : ProductionSupportLevel::Unavailable,
				     ProductionBuildHasCUDA(),
				     true,
				     true,
				     false,
				     true,
				     false,
				     "cuBLAS/cuBLASLt-backed low-precision MatMul where device and build support are present.",
				     "Requires CUDA build support, library availability, and dtype-specific native MatMul capability.",
				     "Unsupported dtype/device pairs must remain explicit fallback or rejection cases." };
		case ProductionCUDANativeCapability::Attention:
		case ProductionCUDANativeCapability::QuantizedProjection:
			return {
				capability,
				ProductionCUDANativeCapabilityName(capability),
				unavailableLevel,
				ProductionBuildHasCUDA(),
				true,
				true,
				false,
				true,
				false,
				capability == ProductionCUDANativeCapability::Attention
				    ? "Production attention kernels are not yet part of the verified CUDA native profile."
				    : "Native quantized projection kernels are not yet part of the verified CUDA native profile.",
				"Requires explicit kernel implementation, device capability checks, and benchmark/parity evidence.",
				"Until implemented, these workloads must not be advertised as CUDA native production support."
			};
		}
		return { capability,
			     ProductionCUDANativeCapabilityName(capability),
			     ProductionSupportLevel::Unavailable,
			     false,
			     true,
			     true,
			     false,
			     false,
			     false,
			     "Unknown CUDA native capability.",
			     "Unknown CUDA native capability.",
			     "Unknown CUDA native capability." };
	}

	inline constexpr ProductionQuantizationCapabilityDescriptor
	QueryProductionQuantizationCapability(ProductionQuantizationCapability capability)
	{
		switch (capability)
		{
		case ProductionQuantizationCapability::ScalarLowPrecisionDataTypes:
			return {
				capability,
				ProductionQuantizationCapabilityName(capability),
				ProductionSupportLevel::Supported,
				true,
				true,
				false,
				false,
				"Scalar FP16, BF16, FP8, signed integer, unsigned integer, and bool dtypes remain byte-addressable "
				"DataType values.",
				"Keep scalar dtypes separate from quantized tensor storage and package metadata.",
				"Unsupported scalar dtype execution must reject or route through an explicit cast/dequantize path."
			};
		case ProductionQuantizationCapability::AffineQuantizedTensors:
			return { capability,
				     ProductionQuantizationCapabilityName(capability),
				     ProductionSupportLevel::Supported,
				     true,
				     true,
				     false,
				     true,
				     "Per-tensor, per-axis, and grouped affine quantization metadata plus CPU quantize/dequantize "
				     "reference helpers.",
				     "Preserve scale, zero-point, axis, group size, and storage dtype through graph/package metadata.",
				     "Optimized kernels may consume the metadata later; correctness falls back to reference "
				     "dequantize." };
		case ProductionQuantizationCapability::BlockQuantizedStorage:
			return { capability,
				     ProductionQuantizationCapabilityName(capability),
				     ProductionSupportLevel::Supported,
				     true,
				     true,
				     false,
				     true,
				     "Block quantization metadata models storage formats such as scalar blocks and ggml-style block "
				     "families without pretending they are scalar DataType values.",
				     "Block storage must stay described by QuantizationParams and external storage metadata.",
				     "Unsupported block formats must fail import/package validation or dequantize through a reference "
				     "path." };
		case ProductionQuantizationCapability::PackedFourBitStorage:
			return { capability,
				     ProductionQuantizationCapabilityName(capability),
				     ProductionSupportLevel::Supported,
				     true,
				     true,
				     false,
				     true,
				     "Packed nibble metadata covers Int4, UInt4, FP4E2M1, FP4E3M0, nibble order, scale layout, and "
				     "logical element count.",
				     "Do not add fake byte-addressable DataType values for int4/fp4; package them as storage metadata.",
				     "Reference unpack/dequantize is the correctness path until native kernels opt in." };
		case ProductionQuantizationCapability::CPUReferencePackUnpackDequantize:
			return { capability,
				     ProductionQuantizationCapabilityName(capability),
				     ProductionSupportLevel::Supported,
				     true,
				     true,
				     false,
				     true,
				     "CPU reference helpers pack integer/float 4-bit values, unpack integer 4-bit values, and "
				     "dequantize packed nibbles deterministically.",
				     "Keep reference behavior covered before adding optimized native kernels.",
				     "Native kernels must match the reference helpers within dtype-specific tolerances." };
		case ProductionQuantizationCapability::VNextQuantizationMetadata:
			return { capability,
				     ProductionQuantizationCapabilityName(capability),
				     ProductionSupportLevel::Supported,
				     true,
				     true,
				     false,
				     true,
				     "vNext package metadata preserves quantization scheme, granularity, packed nibble format, scale "
				     "layout, and logical element count.",
				     "Quantized packages must be inspectable and rebound before execution.",
				     "Loaders must reject incomplete or inconsistent quantization metadata instead of guessing." };
		case ProductionQuantizationCapability::NativeQuantizedLinearMatMul:
			return { capability,
				     ProductionQuantizationCapabilityName(capability),
				     ProductionSupportLevel::Experimental,
				     true,
				     false,
				     true,
				     true,
				     "CPU direct quantized MatMul/Linear covers affine and packed-nibble weight storage with parity "
				     "tests against dequantize-plus-float execution.",
				     "CUDA/Vulkan/AOT lowering, broader GGML block formats, packed-weight benchmarks, and "
				     "workload-specific tolerances are still required before production throughput claims.",
				     "Unsupported quantized formats still fail explicitly or use reference dequantize plus existing "
				     "float execution when the caller chooses that path." };
		}
		return { capability,
			     ProductionQuantizationCapabilityName(capability),
			     ProductionSupportLevel::Unavailable,
			     false,
			     false,
			     false,
			     false,
			     "Unknown quantization capability.",
			     "Unknown quantization capability.",
			     "Unknown quantization capability." };
	}

	inline constexpr ProductionSDXLCapabilityDescriptor
	QueryProductionSDXLCapability(ProductionSDXLCapability capability)
	{
		switch (capability)
		{
		case ProductionSDXLCapability::TorchManifestDiffusionOps:
			return { capability,
				     ProductionSDXLCapabilityName(capability),
				     ProductionSupportLevel::Supported,
				     true,
				     true,
				     false,
				     false,
				     "Torch manifest import covers fixed-shape SDXL diffusion foundation ops and tiny parity fixtures.",
				     "Broader PyTorch/diffusers graph coverage must remain manifest-driven and fixture-backed.",
				     "Unsupported manifest ops must fail with importer diagnostics instead of guessing architecture." };
		case ProductionSDXLCapability::ExternalWeightPackaging:
			return { capability,
				     ProductionSDXLCapabilityName(capability),
				     ProductionSupportLevel::Supported,
				     true,
				     true,
				     false,
				     false,
				     "SDXL import/compile flows can use external weight regions and separated image-region artifacts.",
				     "Large-model package consumers still need workload-specific memory and artifact budget checks.",
				     "Oversized manifests should preflight and fail before expensive import/compile work." };
		case ProductionSDXLCapability::CompiledDenoiserSmoke:
			return { capability,
				     ProductionSDXLCapabilityName(capability),
				     ProductionSupportLevel::Experimental,
				     ProductionBuildHasMLIR(),
				     true,
				     false,
				     false,
				     "Fixed-shape UNet smoke denoiser paths exercise LiteNN import, CPU AOT compile/load, and finite "
				     "output checks.",
				     "Full checkpoint coverage, compile budget, and native CUDA/Vulkan lowering remain outside the "
				     "production promise.",
				     "Use smoke output as pipeline validation, not semantic image-quality acceptance." };
		case ProductionSDXLCapability::EulerSamplerHarness:
			return { capability,
				     ProductionSDXLCapabilityName(capability),
				     ProductionSupportLevel::Experimental,
				     true,
				     true,
				     false,
				     false,
				     "Example-owned Euler sampler orchestration binds latent, timestep, conditioning, CFG, and VAE "
				     "handoff outside the graph.",
				     "Schedulers remain an example/runtime contract until model-level diffusion ABI and parity tests "
				     "are production-gated.",
				     "Keep scheduler ownership explicit; do not hide denoise-loop state inside importer output." };
		case ProductionSDXLCapability::VAEDecodeStress:
			return { capability,
				     ProductionSDXLCapabilityName(capability),
				     ProductionSupportLevel::Experimental,
				     ProductionBuildHasMLIR(),
				     true,
				     false,
				     false,
				     "VAE decode is used as a memory-policy stress target with tiling/fallback diagnostics and PNG "
				     "handoff helpers.",
				     "1024x1024 quality parity still requires reference comparison artifacts and stable native "
				     "coverage.",
				     "Use finite/statistical diagnostics before treating decoded images as valid generation output." };
		case ProductionSDXLCapability::NativePromptConditioning:
			return { capability,
				     ProductionSDXLCapabilityName(capability),
				     ProductionSupportLevel::Deferred,
				     false,
				     false,
				     true,
				     false,
				     "Native tokenizer, text encoder execution, pooled embeddings, and SDXL conditioning graph "
				     "ownership are not production-supported yet.",
				     "Requires tokenizer assets, CLIP/OpenCLIP text encoders, prompt/negative prompt batching, and "
				     "golden conditioning parity.",
				     "Use external conditioning export bridges until native prompt conditioning is implemented." };
		case ProductionSDXLCapability::ReferenceImageParity1024:
			return { capability,
				     ProductionSDXLCapabilityName(capability),
				     ProductionSupportLevel::Deferred,
				     false,
				     false,
				     true,
				     false,
				     "Fixed-seed 1024x1024 reference image parity is not part of the current production profile.",
				     "Requires archived reference runtime artifacts, tensor/image stats comparison, and acceptable "
				     "semantic image validation.",
				     "Keep 1024x1024 SDXL quality parity in the long-term validation queue." };
		case ProductionSDXLCapability::ProductionPromptToImage:
			return { capability,
				     ProductionSDXLCapabilityName(capability),
				     ProductionSupportLevel::Deferred,
				     false,
				     false,
				     true,
				     false,
				     "End-to-end prompt-to-image generation remains a demonstration/stress path rather than a "
				     "production feature.",
				     "Requires native conditioning, full UNet/VAE parity, scheduler parity, memory policy, and native "
				     "backend performance evidence.",
				     "Do not block vNext CPU AOT/package production profile on full SDXL image generation." };
		}
		return { capability,
			     ProductionSDXLCapabilityName(capability),
			     ProductionSupportLevel::Unavailable,
			     false,
			     false,
			     true,
			     false,
			     "Unknown SDXL capability.",
			     "Unknown SDXL capability.",
			     "Unknown SDXL capability." };
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

	inline std::vector<ProductionCUDANativeCapabilityDescriptor> QueryProductionCUDANativeCapabilities()
	{
		return {
			QueryProductionCUDANativeCapability(ProductionCUDANativeCapability::StaticShapeDeviceTensorABI),
			QueryProductionCUDANativeCapability(ProductionCUDANativeCapability::GraphReplay),
			QueryProductionCUDANativeCapability(ProductionCUDANativeCapability::ElementwiseF32),
			QueryProductionCUDANativeCapability(ProductionCUDANativeCapability::MatMulF32),
			QueryProductionCUDANativeCapability(ProductionCUDANativeCapability::LinearChainF32),
			QueryProductionCUDANativeCapability(ProductionCUDANativeCapability::ReductionF32),
			QueryProductionCUDANativeCapability(ProductionCUDANativeCapability::Normalization),
			QueryProductionCUDANativeCapability(ProductionCUDANativeCapability::ConcatSliceF32),
			QueryProductionCUDANativeCapability(ProductionCUDANativeCapability::LowPrecisionCast),
			QueryProductionCUDANativeCapability(ProductionCUDANativeCapability::LowPrecisionMatMul),
			QueryProductionCUDANativeCapability(ProductionCUDANativeCapability::Attention),
			QueryProductionCUDANativeCapability(ProductionCUDANativeCapability::QuantizedProjection),
		};
	}

	inline std::vector<ProductionQuantizationCapabilityDescriptor> QueryProductionQuantizationCapabilities()
	{
		return {
			QueryProductionQuantizationCapability(ProductionQuantizationCapability::ScalarLowPrecisionDataTypes),
			QueryProductionQuantizationCapability(ProductionQuantizationCapability::AffineQuantizedTensors),
			QueryProductionQuantizationCapability(ProductionQuantizationCapability::BlockQuantizedStorage),
			QueryProductionQuantizationCapability(ProductionQuantizationCapability::PackedFourBitStorage),
			QueryProductionQuantizationCapability(ProductionQuantizationCapability::CPUReferencePackUnpackDequantize),
			QueryProductionQuantizationCapability(ProductionQuantizationCapability::VNextQuantizationMetadata),
			QueryProductionQuantizationCapability(ProductionQuantizationCapability::NativeQuantizedLinearMatMul),
		};
	}

	inline std::vector<ProductionSDXLCapabilityDescriptor> QueryProductionSDXLCapabilities()
	{
		return {
			QueryProductionSDXLCapability(ProductionSDXLCapability::TorchManifestDiffusionOps),
			QueryProductionSDXLCapability(ProductionSDXLCapability::ExternalWeightPackaging),
			QueryProductionSDXLCapability(ProductionSDXLCapability::CompiledDenoiserSmoke),
			QueryProductionSDXLCapability(ProductionSDXLCapability::EulerSamplerHarness),
			QueryProductionSDXLCapability(ProductionSDXLCapability::VAEDecodeStress),
			QueryProductionSDXLCapability(ProductionSDXLCapability::NativePromptConditioning),
			QueryProductionSDXLCapability(ProductionSDXLCapability::ReferenceImageParity1024),
			QueryProductionSDXLCapability(ProductionSDXLCapability::ProductionPromptToImage),
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
				                      std::string(ProductionSupportLevelName(status.level)) +
				                      "]: " + std::string(status.policy));
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
				                      std::string(ProductionSupportLevelName(descriptor.level)) +
				                      "]: " + std::string(descriptor.fallbackPolicy));
			}
		}
		return diagnostics;
	}

	inline std::vector<std::string> CollectProductionCUDANativeCapabilityDiagnostics()
	{
		std::vector<std::string> diagnostics;
		for (const auto& capability : QueryProductionCUDANativeCapabilities())
		{
			if (capability.level != ProductionSupportLevel::Supported || capability.allowsHostFallback)
			{
				diagnostics.push_back(std::string(capability.name) + " [" +
				                      std::string(ProductionSupportLevelName(capability.level)) +
				                      "]: " + std::string(capability.capabilityGate));
			}
		}
		return diagnostics;
	}

	inline std::vector<std::string> CollectProductionQuantizationCapabilityDiagnostics()
	{
		std::vector<std::string> diagnostics;
		for (const auto& capability : QueryProductionQuantizationCapabilities())
		{
			if (capability.level != ProductionSupportLevel::Supported || capability.nativeKernel)
			{
				diagnostics.push_back(std::string(capability.name) + " [" +
				                      std::string(ProductionSupportLevelName(capability.level)) +
				                      "]: " + std::string(capability.productionGate));
			}
		}
		return diagnostics;
	}

	inline std::vector<std::string> CollectProductionSDXLCapabilityDiagnostics()
	{
		std::vector<std::string> diagnostics;
		for (const auto& capability : QueryProductionSDXLCapabilities())
		{
			if (capability.level != ProductionSupportLevel::Supported || capability.productionGenerationGate)
			{
				diagnostics.push_back(std::string(capability.name) + " [" +
				                      std::string(ProductionSupportLevelName(capability.level)) +
				                      "]: " + std::string(capability.missingBeforeProduction));
			}
		}
		return diagnostics;
	}
} // namespace LiteNN

#endif
