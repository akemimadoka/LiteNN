#include <gtest/gtest.h>

#include <LiteNN/MobileSupport.h>

#include <algorithm>
#include <string>

using namespace LiteNN;

namespace
{
	bool ContainsDiagnostic(std::string_view needle)
	{
		const auto diagnostics = CollectUnsupportedMobileFeatureDiagnostics();
		return std::ranges::any_of(diagnostics, [&](const std::string& diagnostic) {
			return diagnostic.find(needle) != std::string::npos;
		});
	}
} // namespace

TEST(MobileSupportTest, ReportsSupportedRuntimeFeatures)
{
	EXPECT_TRUE(QueryMobileFeatureStatus(MobileFeature::CPUInterpreter).supported);
	EXPECT_TRUE(QueryMobileFeatureStatus(MobileFeature::SeparatedArtifactLoading).supported);
	EXPECT_TRUE(QueryMobileFeatureStatus(MobileFeature::VulkanNativeRuntime).supported);
}

TEST(MobileSupportTest, RejectsDesktopOnlyMobileFeatures)
{
	EXPECT_FALSE(QueryMobileFeatureStatus(MobileFeature::CPUObjectJIT).supported);
	EXPECT_FALSE(QueryMobileFeatureStatus(MobileFeature::CUDARuntime).supported);
	EXPECT_FALSE(QueryMobileFeatureStatus(MobileFeature::MLIRCompiler).supported);
	EXPECT_FALSE(QueryMobileFeatureStatus(MobileFeature::DynamicLibraryCarrierLoading).supported);
	EXPECT_FALSE(QueryMobileFeatureStatus(MobileFeature::OnDeviceGraphCompilation).supported);

	EXPECT_TRUE(ContainsDiagnostic("cpu-object-jit"));
	EXPECT_TRUE(ContainsDiagnostic("cuda-runtime"));
	EXPECT_TRUE(ContainsDiagnostic("mlir-compiler"));
	EXPECT_TRUE(ContainsDiagnostic("dynamic-library-carrier-loading"));
	EXPECT_TRUE(ContainsDiagnostic("on-device-graph-compilation"));
}

TEST(MobileSupportTest, ReportsMobileConstraintPolicies)
{
	EXPECT_EQ(QueryMobileConstraintStatus(MobileConstraint::CXXStandardLibrary).level,
	          MobileConstraintLevel::Supported);
	EXPECT_EQ(QueryMobileConstraintStatus(MobileConstraint::Filesystem).level, MobileConstraintLevel::Constrained);
	EXPECT_EQ(QueryMobileConstraintStatus(MobileConstraint::Reflection).level, MobileConstraintLevel::Constrained);
	EXPECT_EQ(QueryMobileConstraintStatus(MobileConstraint::DynamicLoading).level, MobileConstraintLevel::Unsupported);
	EXPECT_EQ(QueryMobileConstraintStatus(MobileConstraint::Threading).level, MobileConstraintLevel::Constrained);

	const auto diagnostics = CollectMobileConstraintDiagnostics();
	EXPECT_TRUE(std::ranges::any_of(diagnostics, [](const std::string& diagnostic) {
		return diagnostic.find("filesystem") != std::string::npos;
	}));
	EXPECT_TRUE(std::ranges::any_of(diagnostics, [](const std::string& diagnostic) {
		return diagnostic.find("dynamic-loading") != std::string::npos;
	}));
}
