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

	enum class ProductionSupportLevel
	{
		Production,
		Supported,
		Experimental,
		Deferred,
		Unavailable,
	};

	struct ProductionSupportStatus
	{
		ProductionSupportArea area;
		std::string_view name;
		ProductionSupportLevel level;
		bool availableInBuild;
		std::string_view policy;
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
} // namespace LiteNN

#endif
