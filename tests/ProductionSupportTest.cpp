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
		return std::ranges::any_of(
		    diagnostics, [&](const std::string& diagnostic) { return diagnostic.find(needle) != std::string::npos; });
	}

	bool HasProductionABIDiagnostic(std::string_view needle)
	{
		const auto diagnostics = CollectProductionPathABIDiagnostics();
		return std::ranges::any_of(
		    diagnostics, [&](const std::string& diagnostic) { return diagnostic.find(needle) != std::string::npos; });
	}

	bool HasCUDANativeCapabilityDiagnostic(std::string_view needle)
	{
		const auto diagnostics = CollectProductionCUDANativeCapabilityDiagnostics();
		return std::ranges::any_of(
		    diagnostics, [&](const std::string& diagnostic) { return diagnostic.find(needle) != std::string::npos; });
	}

	bool HasQuantizationCapabilityDiagnostic(std::string_view needle)
	{
		const auto diagnostics = CollectProductionQuantizationCapabilityDiagnostics();
		return std::ranges::any_of(
		    diagnostics, [&](const std::string& diagnostic) { return diagnostic.find(needle) != std::string::npos; });
	}

	bool HasSDXLCapabilityDiagnostic(std::string_view needle)
	{
		const auto diagnostics = CollectProductionSDXLCapabilityDiagnostics();
		return std::ranges::any_of(
		    diagnostics, [&](const std::string& diagnostic) { return diagnostic.find(needle) != std::string::npos; });
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
	EXPECT_EQ(cpuAOT.level,
	          ProductionBuildHasMLIR() ? ProductionSupportLevel::Production : ProductionSupportLevel::Unavailable);

	const auto cuda = QueryProductionSupportStatus(ProductionSupportArea::CUDARuntime);
	EXPECT_EQ(cuda.availableInBuild, ProductionBuildHasCUDA());
	EXPECT_EQ(cuda.level,
	          ProductionBuildHasCUDA() ? ProductionSupportLevel::Supported : ProductionSupportLevel::Unavailable);

	const auto vulkan = QueryProductionSupportStatus(ProductionSupportArea::VulkanRuntime);
	EXPECT_EQ(vulkan.availableInBuild, ProductionBuildHasVulkan());
	EXPECT_EQ(vulkan.level,
	          ProductionBuildHasVulkan() ? ProductionSupportLevel::Experimental : ProductionSupportLevel::Unavailable);
}

TEST(ProductionSupportTest, ReportsDeferredLongTailWork)
{
	EXPECT_EQ(QueryProductionSupportStatus(ProductionSupportArea::TrainingAOT).level, ProductionSupportLevel::Deferred);
	EXPECT_EQ(QueryProductionSupportStatus(ProductionSupportArea::SDXLGeneration).level,
	          ProductionSupportLevel::Deferred);

	EXPECT_TRUE(HasProductionDiagnostic("TrainingAOT"));
	EXPECT_TRUE(HasProductionDiagnostic("SDXLGeneration"));
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
	EXPECT_TRUE(HasProductionABIDiagnostic("CUDACPUBridgeFallback"));
	EXPECT_TRUE(HasProductionABIDiagnostic("VulkanNativeSeparatedArtifact"));
	EXPECT_TRUE(HasProductionABIDiagnostic("MobileSeparatedRuntime"));
	EXPECT_FALSE(HasProductionABIDiagnostic("VNextModelPackage"));
}

TEST(ProductionSupportTest, ReportsBackendProfilesWithoutCollapsingMobileAndDesktop)
{
	const auto profiles = QueryProductionBackendProfiles();
	EXPECT_GE(profiles.size(), 6u);

	const auto cpu = QueryProductionBackendProfile(ProductionBackendProfile::CPUReferenceInterpreter);
	EXPECT_EQ(cpu.path, ProductionPath::CPUInterpreter);
	EXPECT_EQ(cpu.level, ProductionSupportLevel::Production);
	EXPECT_TRUE(cpu.referenceCorrectnessPath);
	EXPECT_FALSE(cpu.nativeDeviceProfile);
	EXPECT_TRUE(Contains(cpu.verifiedScope, "Reference correctness"));

	const auto desktopVulkan = QueryProductionBackendProfile(ProductionBackendProfile::VulkanDesktopNative);
	const auto mobileVulkan = QueryProductionBackendProfile(ProductionBackendProfile::VulkanMobileConstrained);
	EXPECT_NE(desktopVulkan.name, mobileVulkan.name);
	EXPECT_TRUE(desktopVulkan.desktopProfile);
	EXPECT_FALSE(desktopVulkan.mobileProfile);
	EXPECT_FALSE(desktopVulkan.allowsHostFallback);
	EXPECT_FALSE(mobileVulkan.desktopProfile);
	EXPECT_TRUE(mobileVulkan.mobileProfile);
	EXPECT_TRUE(mobileVulkan.allowsHostFallback);
	EXPECT_TRUE(desktopVulkan.requiresDeviceCapabilityProbe);
	EXPECT_TRUE(mobileVulkan.requiresDeviceCapabilityProbe);
	EXPECT_TRUE(Contains(desktopVulkan.skipOrFailurePolicy, "skip or fail explicitly"));
	EXPECT_TRUE(Contains(mobileVulkan.skipOrFailurePolicy, "skip/fail explicitly"));
}

TEST(ProductionSupportTest, ReportsCPUKernelStrategy)
{
	const auto strategy = QueryProductionCPUKernelStrategy();
	EXPECT_TRUE(Contains(strategy.referencePath, "cpu-reference-interpreter"));
	EXPECT_TRUE(strategy.preferExternalLibraryBackend);
	EXPECT_TRUE(strategy.allowSmallNativeKernelSet);
	EXPECT_FALSE(strategy.allowUnplannedHandwrittenGemmOrConv);
	EXPECT_TRUE(Contains(strategy.throughputPolicy, "external-library"));
	EXPECT_TRUE(Contains(strategy.handwrittenKernelGate, "benchmark"));
}

TEST(ProductionSupportTest, ReportsCUDANativeCapabilitiesAsGatedProfiles)
{
	const auto capabilities = QueryProductionCUDANativeCapabilities();
	EXPECT_GE(capabilities.size(), 10u);

	const auto graphReplay = QueryProductionCUDANativeCapability(ProductionCUDANativeCapability::GraphReplay);
	EXPECT_EQ(graphReplay.availableInBuild, ProductionBuildHasCUDA());
	EXPECT_EQ(graphReplay.level,
	          ProductionBuildHasCUDA() ? ProductionSupportLevel::Supported : ProductionSupportLevel::Unavailable);
	EXPECT_TRUE(graphReplay.requiresCUDADevice);
	EXPECT_TRUE(graphReplay.requiresRuntimeDeviceProbe);
	EXPECT_TRUE(graphReplay.requiresStablePointers);
	EXPECT_TRUE(graphReplay.highValueKernelPriority);
	EXPECT_FALSE(graphReplay.allowsHostFallback);
	EXPECT_TRUE(Contains(graphReplay.fallbackPolicy, "fail loudly"));

	const auto matmul = QueryProductionCUDANativeCapability(ProductionCUDANativeCapability::MatMulF32);
	EXPECT_TRUE(matmul.highValueKernelPriority);
	EXPECT_TRUE(Contains(matmul.verifiedScope, "MatMul"));
	EXPECT_TRUE(Contains(matmul.capabilityGate, "device"));
	EXPECT_FALSE(matmul.allowsHostFallback);

	const auto attention = QueryProductionCUDANativeCapability(ProductionCUDANativeCapability::Attention);
	EXPECT_TRUE(attention.highValueKernelPriority);
	EXPECT_NE(attention.level, ProductionSupportLevel::Supported);
	EXPECT_TRUE(Contains(attention.capabilityGate, "kernel implementation"));
	EXPECT_TRUE(HasCUDANativeCapabilityDiagnostic("Attention"));

	const auto normalization = QueryProductionCUDANativeCapability(ProductionCUDANativeCapability::Normalization);
	EXPECT_TRUE(normalization.highValueKernelPriority);
	EXPECT_NE(normalization.level, ProductionSupportLevel::Supported);
	EXPECT_TRUE(HasCUDANativeCapabilityDiagnostic("Normalization"));

	const auto quantizedProjection =
	    QueryProductionCUDANativeCapability(ProductionCUDANativeCapability::QuantizedProjection);
	EXPECT_TRUE(quantizedProjection.highValueKernelPriority);
	EXPECT_NE(quantizedProjection.level, ProductionSupportLevel::Supported);
	EXPECT_TRUE(HasCUDANativeCapabilityDiagnostic("QuantizedProjection"));
}

TEST(ProductionSupportTest, ReportsQuantizationCapabilitiesBeforeNativeKernels)
{
	const auto capabilities = QueryProductionQuantizationCapabilities();
	EXPECT_GE(capabilities.size(), 7u);

	const auto scalar =
	    QueryProductionQuantizationCapability(ProductionQuantizationCapability::ScalarLowPrecisionDataTypes);
	EXPECT_EQ(scalar.level, ProductionSupportLevel::Supported);
	EXPECT_TRUE(scalar.semanticFoundation);
	EXPECT_FALSE(scalar.nativeKernel);
	EXPECT_FALSE(scalar.requiresExternalMetadata);
	EXPECT_TRUE(Contains(scalar.verifiedScope, "DataType"));
	EXPECT_TRUE(Contains(scalar.productionGate, "separate"));

	const auto packed = QueryProductionQuantizationCapability(ProductionQuantizationCapability::PackedFourBitStorage);
	EXPECT_EQ(packed.level, ProductionSupportLevel::Supported);
	EXPECT_TRUE(packed.semanticFoundation);
	EXPECT_FALSE(packed.nativeKernel);
	EXPECT_TRUE(packed.requiresExternalMetadata);
	EXPECT_TRUE(Contains(packed.verifiedScope, "Int4"));
	EXPECT_TRUE(Contains(packed.verifiedScope, "FP4"));
	EXPECT_TRUE(Contains(packed.productionGate, "fake"));

	const auto reference =
	    QueryProductionQuantizationCapability(ProductionQuantizationCapability::CPUReferencePackUnpackDequantize);
	EXPECT_EQ(reference.level, ProductionSupportLevel::Supported);
	EXPECT_TRUE(reference.semanticFoundation);
	EXPECT_FALSE(reference.nativeKernel);
	EXPECT_TRUE(Contains(reference.verifiedScope, "pack"));
	EXPECT_TRUE(Contains(reference.verifiedScope, "unpack"));
	EXPECT_TRUE(Contains(reference.verifiedScope, "dequantize"));

	const auto native =
	    QueryProductionQuantizationCapability(ProductionQuantizationCapability::NativeQuantizedLinearMatMul);
	EXPECT_EQ(native.level, ProductionSupportLevel::Experimental);
	EXPECT_TRUE(native.availableInBuild);
	EXPECT_FALSE(native.semanticFoundation);
	EXPECT_TRUE(native.nativeKernel);
	EXPECT_TRUE(native.requiresExternalMetadata);
	EXPECT_TRUE(Contains(native.verifiedScope, "CPU direct"));
	EXPECT_TRUE(Contains(native.verifiedScope, "affine"));
	EXPECT_TRUE(Contains(native.verifiedScope, "packed-nibble"));
	EXPECT_TRUE(Contains(native.productionGate, "CUDA"));
	EXPECT_TRUE(Contains(native.productionGate, "Vulkan"));
	EXPECT_TRUE(Contains(native.productionGate, "benchmarks"));
	EXPECT_TRUE(HasQuantizationCapabilityDiagnostic("NativeQuantizedLinearMatMul"));
	EXPECT_FALSE(HasQuantizationCapabilityDiagnostic("PackedFourBitStorage"));
}

TEST(ProductionSupportTest, ReportsSDXLAsImporterAndStressTarget)
{
	const auto capabilities = QueryProductionSDXLCapabilities();
	EXPECT_GE(capabilities.size(), 8u);

	const auto manifest = QueryProductionSDXLCapability(ProductionSDXLCapability::TorchManifestDiffusionOps);
	EXPECT_EQ(manifest.level, ProductionSupportLevel::Supported);
	EXPECT_TRUE(manifest.importerOrStressTarget);
	EXPECT_FALSE(manifest.productionGenerationGate);
	EXPECT_FALSE(manifest.blocksVNextProductionProfile);
	EXPECT_TRUE(Contains(manifest.verifiedScope, "fixed-shape SDXL"));

	const auto denoiser = QueryProductionSDXLCapability(ProductionSDXLCapability::CompiledDenoiserSmoke);
	EXPECT_EQ(denoiser.availableInBuild, ProductionBuildHasMLIR());
	EXPECT_EQ(denoiser.level, ProductionSupportLevel::Experimental);
	EXPECT_TRUE(denoiser.importerOrStressTarget);
	EXPECT_FALSE(denoiser.productionGenerationGate);
	EXPECT_TRUE(Contains(denoiser.fallbackPolicy, "pipeline validation"));
	EXPECT_TRUE(HasSDXLCapabilityDiagnostic("CompiledDenoiserSmoke"));

	const auto prompt = QueryProductionSDXLCapability(ProductionSDXLCapability::NativePromptConditioning);
	EXPECT_EQ(prompt.level, ProductionSupportLevel::Deferred);
	EXPECT_TRUE(prompt.productionGenerationGate);
	EXPECT_FALSE(prompt.blocksVNextProductionProfile);
	EXPECT_TRUE(Contains(prompt.missingBeforeProduction, "tokenizer"));
	EXPECT_TRUE(HasSDXLCapabilityDiagnostic("NativePromptConditioning"));

	const auto fullImage = QueryProductionSDXLCapability(ProductionSDXLCapability::ProductionPromptToImage);
	EXPECT_EQ(fullImage.level, ProductionSupportLevel::Deferred);
	EXPECT_TRUE(fullImage.productionGenerationGate);
	EXPECT_FALSE(fullImage.blocksVNextProductionProfile);
	EXPECT_TRUE(Contains(fullImage.fallbackPolicy, "Do not block vNext"));
	EXPECT_TRUE(HasSDXLCapabilityDiagnostic("ProductionPromptToImage"));
}
