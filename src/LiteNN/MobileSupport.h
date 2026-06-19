#ifndef LITENN_MOBILE_SUPPORT_H
#define LITENN_MOBILE_SUPPORT_H

#include <LiteNN/Misc.h>

#include <string>
#include <string_view>
#include <vector>

namespace LiteNN
{
	enum class MobileFeature
	{
		CPUInterpreter,
		SeparatedArtifactLoading,
		VulkanNativeRuntime,
		CPUObjectJIT,
		CUDARuntime,
		MLIRCompiler,
		DynamicLibraryCarrierLoading,
		OnDeviceGraphCompilation,
	};

	enum class MobileConstraint
	{
		CXXStandardLibrary,
		Filesystem,
		Reflection,
		DynamicLoading,
		Threading,
	};

	enum class MobileVulkanProductionGate
	{
		GraphPartitioning,
		ExplicitFallbackVisibility,
		SeparatedArtifactRegions,
		DeviceLocalMemoryPlanning,
		MobileDeviceMatrix,
	};

	enum class MobileConstraintLevel
	{
		Supported,
		Constrained,
		Unsupported,
	};

	struct MobileFeatureStatus
	{
		MobileFeature feature;
		std::string_view name;
		bool supported;
		std::string_view reason;
	};

	struct MobileConstraintStatus
	{
		MobileConstraint constraint;
		std::string_view name;
		MobileConstraintLevel level;
		std::string_view policy;
	};

	struct MobileVulkanProductionGateStatus
	{
		MobileVulkanProductionGate gate;
		std::string_view name;
		MobileConstraintLevel level;
		bool requiredForBroadMobileGPU;
		std::string_view evidence;
		std::string_view missing;
	};

	inline constexpr std::string_view MobileFeatureName(MobileFeature feature)
	{
		return EnumToString<EnumToStringStyle::Unqualified>(feature);
	}

	inline constexpr MobileFeatureStatus QueryMobileFeatureStatus(MobileFeature feature)
	{
		switch (feature)
		{
		case MobileFeature::CPUInterpreter:
			return { feature, MobileFeatureName(feature), true,
				     "Reference execution is available when LiteNNCore is linked." };
		case MobileFeature::SeparatedArtifactLoading:
			return { feature, MobileFeatureName(feature), true,
				     "Separated metadata/constants/weights/instructions are the preferred mobile package ABI." };
		case MobileFeature::VulkanNativeRuntime:
			return { feature, MobileFeatureName(feature), true,
				     "Supported when LiteNNVulkanRuntime is linked and a compute-capable Vulkan device is present." };
		case MobileFeature::CPUObjectJIT:
			return { feature, MobileFeatureName(feature), false,
				     "Desktop CPU AOT object JIT loading is not part of the mobile runtime ABI." };
		case MobileFeature::CUDARuntime:
			return { feature, MobileFeatureName(feature), false,
				     "CUDA is a desktop/server backend and must not be required by mobile packages." };
		case MobileFeature::MLIRCompiler:
			return { feature, MobileFeatureName(feature), false,
				     "Mobile apps should load host-compiled artifacts instead of linking the compiler stack." };
		case MobileFeature::DynamicLibraryCarrierLoading:
			return { feature, MobileFeatureName(feature), false,
				     "Mobile packages should use separated regions, not desktop shared-library carrier loading." };
		case MobileFeature::OnDeviceGraphCompilation:
			return { feature, MobileFeatureName(feature), false,
				     "On-device graph compilation is intentionally excluded from the production mobile profile." };
		}
		return { feature, MobileFeatureName(feature), false, "Unknown mobile feature." };
	}

	inline constexpr std::string_view MobileConstraintName(MobileConstraint constraint)
	{
		return EnumToString<EnumToStringStyle::Unqualified>(constraint);
	}

	inline constexpr std::string_view MobileVulkanProductionGateName(MobileVulkanProductionGate gate)
	{
		return EnumToString<EnumToStringStyle::Unqualified>(gate);
	}

	inline constexpr MobileConstraintStatus QueryMobileConstraintStatus(MobileConstraint constraint)
	{
		switch (constraint)
		{
		case MobileConstraint::CXXStandardLibrary:
			return {
				constraint, MobileConstraintName(constraint), MobileConstraintLevel::Supported,
				"LiteNNCore uses the C++ standard library directly; mobile builds must use the platform NDK/libc++ "
				"profile and keep exception/RTTI policy consistent across the app."
			};
		case MobileConstraint::Filesystem:
			return {
				constraint, MobileConstraintName(constraint), MobileConstraintLevel::Constrained,
				"Filesystem APIs are allowed in host tools and desktop loaders; production mobile inference should "
				"prefer borrowed separated regions from app assets or memory-mapped packages."
			};
		case MobileConstraint::Reflection:
			return {
				constraint, MobileConstraintName(constraint), MobileConstraintLevel::Constrained,
				"The public headers currently use C++ reflection-enabled builds; mobile runtime presets must pin a "
				"toolchain that supports the same language mode or compile a reduced runtime surface."
			};
		case MobileConstraint::DynamicLoading:
			return {
				constraint, MobileConstraintName(constraint), MobileConstraintLevel::Unsupported,
				"Desktop shared-library carrier loading is outside the mobile ABI; use separated artifact regions."
			};
		case MobileConstraint::Threading:
			return { constraint, MobileConstraintName(constraint), MobileConstraintLevel::Constrained,
				     "Threading is permitted only through caller-owned policy; mobile runtime code should avoid hidden "
				     "background threads and expose synchronization cost in profiles." };
		}
		return { constraint, MobileConstraintName(constraint), MobileConstraintLevel::Unsupported,
			     "Unknown mobile constraint." };
	}

	inline constexpr MobileVulkanProductionGateStatus
	QueryMobileVulkanProductionGateStatus(MobileVulkanProductionGate gate)
	{
		switch (gate)
		{
		case MobileVulkanProductionGate::GraphPartitioning:
			return {
				gate,
				MobileVulkanProductionGateName(gate),
				MobileConstraintLevel::Supported,
				true,
				"Runtime placement builds backend partitions from executable plans and records explicit fallback "
				"steps.",
				"Coverage is still workload-dependent; unsupported ops must remain visible in placement diagnostics."
			};
		case MobileVulkanProductionGate::ExplicitFallbackVisibility:
			return {
				gate,
				MobileVulkanProductionGateName(gate),
				MobileConstraintLevel::Supported,
				true,
				"Runtime schedule profile records expose transfer/fallback rows instead of hiding CPU fallback inside "
				"native backend labels.",
				"Benchmark consumers must keep native and fallback rows separate."
			};
		case MobileVulkanProductionGate::SeparatedArtifactRegions:
			return {
				gate,
				MobileVulkanProductionGateName(gate),
				MobileConstraintLevel::Supported,
				true,
				"vNext packages and mobile feature policy prefer separated metadata/constants/weights/instructions "
				"regions.",
				"Mobile loaders still need platform-specific mmap/asset binding examples."
			};
		case MobileVulkanProductionGate::DeviceLocalMemoryPlanning:
			return { gate,
				     MobileVulkanProductionGateName(gate),
				     MobileConstraintLevel::Supported,
				     true,
				     "MemoryPlan exposes a DeviceLocalMemoryPlan with device-local allocation classes, upload/download "
				     "staging steps, and validation against the canonical buffer plan.",
				     "Backend allocators still need to consume the plan with per-device limit checks before broad "
				     "mobile GPU "
				     "production support." };
		case MobileVulkanProductionGate::MobileDeviceMatrix:
			return { gate,
				     MobileVulkanProductionGateName(gate),
				     MobileConstraintLevel::Constrained,
				     true,
				     "Desktop Vulkan tests cover selected native payload behavior.",
				     "Need Android/iOS-adjacent Vulkan device matrix, storage/subgroup/timestamp/alignment probes, and "
				     "skip/fail policy evidence." };
		}
		return { gate, MobileVulkanProductionGateName(gate),     MobileConstraintLevel::Unsupported,
			     true, "Unknown mobile Vulkan production gate.", "Unknown mobile Vulkan production gate." };
	}

	inline std::vector<MobileFeatureStatus> QueryMobileFeatureStatuses()
	{
		return {
			QueryMobileFeatureStatus(MobileFeature::CPUInterpreter),
			QueryMobileFeatureStatus(MobileFeature::SeparatedArtifactLoading),
			QueryMobileFeatureStatus(MobileFeature::VulkanNativeRuntime),
			QueryMobileFeatureStatus(MobileFeature::CPUObjectJIT),
			QueryMobileFeatureStatus(MobileFeature::CUDARuntime),
			QueryMobileFeatureStatus(MobileFeature::MLIRCompiler),
			QueryMobileFeatureStatus(MobileFeature::DynamicLibraryCarrierLoading),
			QueryMobileFeatureStatus(MobileFeature::OnDeviceGraphCompilation),
		};
	}

	inline std::vector<MobileConstraintStatus> QueryMobileConstraintStatuses()
	{
		return {
			QueryMobileConstraintStatus(MobileConstraint::CXXStandardLibrary),
			QueryMobileConstraintStatus(MobileConstraint::Filesystem),
			QueryMobileConstraintStatus(MobileConstraint::Reflection),
			QueryMobileConstraintStatus(MobileConstraint::DynamicLoading),
			QueryMobileConstraintStatus(MobileConstraint::Threading),
		};
	}

	inline std::vector<MobileVulkanProductionGateStatus> QueryMobileVulkanProductionGateStatuses()
	{
		return {
			QueryMobileVulkanProductionGateStatus(MobileVulkanProductionGate::GraphPartitioning),
			QueryMobileVulkanProductionGateStatus(MobileVulkanProductionGate::ExplicitFallbackVisibility),
			QueryMobileVulkanProductionGateStatus(MobileVulkanProductionGate::SeparatedArtifactRegions),
			QueryMobileVulkanProductionGateStatus(MobileVulkanProductionGate::DeviceLocalMemoryPlanning),
			QueryMobileVulkanProductionGateStatus(MobileVulkanProductionGate::MobileDeviceMatrix),
		};
	}

	inline std::vector<std::string> CollectUnsupportedMobileFeatureDiagnostics()
	{
		std::vector<std::string> diagnostics;
		for (const auto& status : QueryMobileFeatureStatuses())
		{
			if (!status.supported)
			{
				diagnostics.push_back(std::string(status.name) + ": " + std::string(status.reason));
			}
		}
		return diagnostics;
	}

	inline std::vector<std::string> CollectMobileConstraintDiagnostics()
	{
		std::vector<std::string> diagnostics;
		for (const auto& status : QueryMobileConstraintStatuses())
		{
			if (status.level != MobileConstraintLevel::Supported)
			{
				diagnostics.push_back(std::string(status.name) + ": " + std::string(status.policy));
			}
		}
		return diagnostics;
	}

	inline std::vector<std::string> CollectMobileVulkanProductionGateDiagnostics()
	{
		std::vector<std::string> diagnostics;
		for (const auto& status : QueryMobileVulkanProductionGateStatuses())
		{
			if (status.level != MobileConstraintLevel::Supported)
			{
				diagnostics.push_back(std::string(status.name) + ": " + std::string(status.missing));
			}
		}
		return diagnostics;
	}
} // namespace LiteNN

#endif
