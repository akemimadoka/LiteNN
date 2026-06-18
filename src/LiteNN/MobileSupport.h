#ifndef LITENN_MOBILE_SUPPORT_H
#define LITENN_MOBILE_SUPPORT_H

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

	struct MobileFeatureStatus
	{
		MobileFeature feature;
		std::string_view name;
		bool supported;
		std::string_view reason;
	};

	inline constexpr std::string_view MobileFeatureName(MobileFeature feature)
	{
		switch (feature)
		{
		case MobileFeature::CPUInterpreter:
			return "cpu-interpreter";
		case MobileFeature::SeparatedArtifactLoading:
			return "separated-artifact-loading";
		case MobileFeature::VulkanNativeRuntime:
			return "vulkan-native-runtime";
		case MobileFeature::CPUObjectJIT:
			return "cpu-object-jit";
		case MobileFeature::CUDARuntime:
			return "cuda-runtime";
		case MobileFeature::MLIRCompiler:
			return "mlir-compiler";
		case MobileFeature::DynamicLibraryCarrierLoading:
			return "dynamic-library-carrier-loading";
		case MobileFeature::OnDeviceGraphCompilation:
			return "on-device-graph-compilation";
		}
		return "unknown";
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
} // namespace LiteNN

#endif
