#include <gtest/gtest.h>

#include <LiteNN.h>

#include <algorithm>
#include <string>

using namespace LiteNN;

namespace
{
	bool Contains(std::string_view haystack, std::string_view needle)
	{
		return std::string(haystack).find(std::string(needle)) != std::string::npos;
	}

	bool HasProductionDiagnostic(std::string_view needle)
	{
		const auto diagnostics = CollectProductionSupportDiagnostics();
		return std::ranges::any_of(diagnostics, [&](const std::string& diagnostic) {
			return diagnostic.find(needle) != std::string::npos;
		});
	}

	bool HasProductionABIDiagnostic(std::string_view needle)
	{
		const auto diagnostics = CollectProductionPathABIDiagnostics();
		return std::ranges::any_of(diagnostics, [&](const std::string& diagnostic) {
			return diagnostic.find(needle) != std::string::npos;
		});
	}
} // namespace

TEST(ProductionSupportTest, ReportsProductionDeploymentCore)
{
	EXPECT_EQ(QueryProductionSupportStatus(ProductionSupportArea::CPURuntime).level,
	          ProductionSupportLevel::Production);
	EXPECT_EQ(QueryProductionSupportStatus(ProductionSupportArea::VNextPackaging).level,
	          ProductionSupportLevel::Production);
	EXPECT_EQ(QueryProductionSupportStatus(ProductionSupportArea::SeparatedArtifacts).level,
	          ProductionSupportLevel::Production);
}

TEST(ProductionSupportTest, ReportsBuildDependentBackends)
{
	const auto cpuAOT = QueryProductionSupportStatus(ProductionSupportArea::CPUAOT);
	EXPECT_EQ(cpuAOT.availableInBuild, ProductionBuildHasMLIR());
	EXPECT_EQ(cpuAOT.level, ProductionBuildHasMLIR() ? ProductionSupportLevel::Production
	                                                 : ProductionSupportLevel::Unavailable);

	const auto cuda = QueryProductionSupportStatus(ProductionSupportArea::CUDARuntime);
	EXPECT_EQ(cuda.availableInBuild, ProductionBuildHasCUDA());
	EXPECT_EQ(cuda.level, ProductionBuildHasCUDA() ? ProductionSupportLevel::Supported
	                                               : ProductionSupportLevel::Unavailable);

	const auto vulkan = QueryProductionSupportStatus(ProductionSupportArea::VulkanRuntime);
	EXPECT_EQ(vulkan.availableInBuild, ProductionBuildHasVulkan());
	EXPECT_EQ(vulkan.level, ProductionBuildHasVulkan() ? ProductionSupportLevel::Experimental
	                                                   : ProductionSupportLevel::Unavailable);
}

TEST(ProductionSupportTest, ReportsDeferredLongTailWork)
{
	EXPECT_EQ(QueryProductionSupportStatus(ProductionSupportArea::TrainingAOT).level,
	          ProductionSupportLevel::Deferred);
	EXPECT_EQ(QueryProductionSupportStatus(ProductionSupportArea::SDXLGeneration).level,
	          ProductionSupportLevel::Deferred);

	EXPECT_TRUE(HasProductionDiagnostic("training-aot"));
	EXPECT_TRUE(HasProductionDiagnostic("sdxl-generation"));
}

TEST(ProductionSupportTest, ReportsProductionPathABIContracts)
{
	const auto all = QueryProductionPathABIs();
	EXPECT_GE(all.size(), 8u);

	const auto cpuAOT = QueryProductionPathABI(ProductionPath::CPUAOTSeparatedArtifact);
	EXPECT_EQ(cpuAOT.availableInBuild, ProductionBuildHasMLIR());
	EXPECT_TRUE(cpuAOT.supportsExternalTensors);
	EXPECT_TRUE(cpuAOT.usesSeparatedRegions);
	EXPECT_TRUE(cpuAOT.usesAlignmentMetadata);
	EXPECT_TRUE(cpuAOT.usesChecksums);
	EXPECT_FALSE(cpuAOT.allowsHostFallback);
	EXPECT_TRUE(Contains(cpuAOT.externalTensors, "rodata"));
	EXPECT_TRUE(Contains(cpuAOT.fallbackPolicy, "No implicit fallback"));

	const auto cudaGraph = QueryProductionPathABI(ProductionPath::CUDANativeGraphReplay);
	EXPECT_EQ(cudaGraph.availableInBuild, ProductionBuildHasCUDA());
	EXPECT_TRUE(cudaGraph.supportsExternalTensors);
	EXPECT_TRUE(cudaGraph.requiresStableDevicePointers);
	EXPECT_FALSE(cudaGraph.allowsHostFallback);
	EXPECT_TRUE(Contains(cudaGraph.outputs, "pointer-stable"));
	EXPECT_TRUE(Contains(cudaGraph.fallbackPolicy, "Reject"));

	const auto cudaBridge = QueryProductionPathABI(ProductionPath::CUDACPUBridgeFallback);
	EXPECT_TRUE(cudaBridge.allowsHostFallback);
	EXPECT_TRUE(Contains(cudaBridge.fallbackPolicy, "Host fallback"));

	const auto package = QueryProductionPathABI(ProductionPath::VNextModelPackage);
	EXPECT_EQ(package.level, ProductionSupportLevel::Production);
	EXPECT_TRUE(package.supportsExternalTensors);
	EXPECT_TRUE(package.usesSeparatedRegions);
	EXPECT_TRUE(package.usesChecksums);
	EXPECT_TRUE(Contains(package.checksum, "checksums"));
}

TEST(ProductionSupportTest, ReportsProductionPathABIDiagnostics)
{
	EXPECT_TRUE(HasProductionABIDiagnostic("cuda-cpu-bridge-fallback"));
	EXPECT_TRUE(HasProductionABIDiagnostic("vulkan-native-separated-artifact"));
	EXPECT_TRUE(HasProductionABIDiagnostic("mobile-separated-runtime"));
	EXPECT_FALSE(HasProductionABIDiagnostic("vnext-model-package"));
}
