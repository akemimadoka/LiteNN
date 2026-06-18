#include <gtest/gtest.h>

#include <LiteNN.h>

#include <algorithm>
#include <string>

using namespace LiteNN;

namespace
{
	bool HasProductionDiagnostic(std::string_view needle)
	{
		const auto diagnostics = CollectProductionSupportDiagnostics();
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
