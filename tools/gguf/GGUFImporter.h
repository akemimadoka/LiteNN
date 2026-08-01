#include <LiteNN.h>
#include <LiteNN/Serialization/ExternalWeights.h>

#include <cstddef>
#include <filesystem>
#include <optional>
#include <string>
#include <string_view>
#include <vector>

#ifndef LITENN_GGUFIMPORTER_H
#define LITENN_GGUFIMPORTER_H

namespace LiteNN::GGUF
{
	struct LLaMAHyperparameters
	{
		std::string architecture;
		std::size_t contextLength{};
		std::size_t embeddingLength{};
		std::size_t blockCount{};
		std::size_t feedForwardLength{};
		std::size_t attentionHeadCount{};
		std::size_t attentionHeadCountKV{};
		double rmsNormEpsilon{};
		double ropeFrequencyBase{ 10000.0 };
		double ropeFrequencyScale{ 1.0 };
		std::size_t ropeDimensionCount{};
		std::string ropeScalingType{ "none" };
		std::optional<double> ropeScalingFactor;
		std::optional<double> ropeScalingAlpha;
		std::optional<double> ropeScalingAttentionFactor;
		std::optional<std::size_t> ropeScalingOriginalContextLength;
		std::optional<bool> ropeScalingFinetuned;
		std::optional<double> ropeScalingYarnLogMultiplier;
		std::optional<double> ropeScalingYarnExtFactor;
		std::optional<double> ropeScalingYarnAttentionFactor;
		std::optional<double> ropeScalingYarnBetaFast;
		std::optional<double> ropeScalingYarnBetaSlow;

		std::size_t HeadDimension() const;
		std::size_t QueryGroupsPerKVHead() const;
	};

	struct ImportSummary
	{
		std::size_t tensorCount{};
		std::size_t metadataCount{};
	};

	struct ImportResult
	{
		ModelGraph model;
		ImportSummary summary;
	};

	enum class LLaMACompatibilityProfileKind
	{
		TinyFixture,
		LLaMA2LikeCausalLM,
		LLaMA3LikeCausalLM,
		Qwen2LikeCausalLM,
	};

	struct LLaMACompatibilityProfileDescriptor
	{
		LLaMACompatibilityProfileKind kind;
		std::string_view name;
		std::string_view architecture;
		bool selectedProductionProfile;
		bool supportsPrefill;
		bool supportsDecode;
		bool supportsLinearRoPE;
		bool supportsYaRNOrLongRoPE;
		bool importsQuantizedWeightsByDequantizing;
		bool requiresExternalLLaMACppGolden;
		std::string_view supportedSignature;
		std::string_view unsupportedPolicy;
		std::string_view acceptancePolicy;
	};

	struct LLaMACompatibilityDiagnostic
	{
		std::string subject;
		std::string message;
		bool blocking{};
	};

	struct LLaMACompatibilityReport
	{
		LLaMACompatibilityProfileDescriptor profile;
		bool lowerable{};
		bool externalGoldenRequired{};
		std::vector<LLaMACompatibilityDiagnostic> diagnostics;
	};

	struct LLaMAContextValidationReport
	{
		std::size_t requestedTokenCount{};
		std::size_t maxCacheLength{};
		std::size_t modelContextLength{};
		std::size_t trainedContextLength{};
		std::string ropeScalingType;
		bool accepted{ true };
		bool usesContextExtension{};
		std::vector<LLaMACompatibilityDiagnostic> diagnostics;
	};

	struct LLMTokenizerMetadataSummary
	{
		std::optional<std::string> model;
		std::size_t tokenCount{};
		std::size_t tokenTypeCount{};
		bool hasChatTemplate{};
		std::size_t chatTemplateBytes{};
		bool hasBosTokenId{};
		bool hasEosTokenId{};
		bool hasUnknownTokenId{};
		std::optional<std::int64_t> bosTokenId;
		std::optional<std::int64_t> eosTokenId;
		std::optional<std::int64_t> unknownTokenId;
	};

	enum class LLaMAQuantizedExecutionPolicy
	{
		None,
		Reject,
		CPUNativeQuantized,
		CPUReferenceDequantize,
		CUDADequantizeThenGEMM,
		CUDANativeQuantized
	};

	struct LLaMAQuantizedFormatDecision
	{
		QuantizedBlockFormat format{ QuantizedBlockFormat::Scalar };
		std::size_t tensorCount{};
		std::size_t storedBytes{};
		std::size_t dequantizedBytes{};
		LLaMAQuantizedExecutionPolicy selectedPolicy{ LLaMAQuantizedExecutionPolicy::None };
		bool blocking{};
		std::string reason;
	};

	struct LLaMAQuantizedExecutionPlan
	{
		std::size_t tensorCount{};
		std::size_t storedBytes{};
		std::size_t dequantizedBytes{};
		std::size_t dequantizedMemoryBudgetBytes{};
		bool lowerable{ true };
		std::vector<LLaMAQuantizedFormatDecision> decisions;
	};

	std::string_view LLaMACompatibilityProfileName(LLaMACompatibilityProfileKind kind);
	LLaMACompatibilityProfileDescriptor QueryLLaMACompatibilityProfile(LLaMACompatibilityProfileKind kind);
	std::vector<LLaMACompatibilityProfileDescriptor> QueryLLaMACompatibilityProfiles();
	std::optional<LLaMACompatibilityProfileKind> TryInferLLaMACompatibilityProfile(std::string_view architecture);
	LLaMACompatibilityReport AnalyzeLLaMACompatibility(const Graph& archive, LLaMACompatibilityProfileKind kind,
	                                                   std::size_t dequantizedMemoryBudgetBytes = 0);
	LLMTokenizerMetadataSummary SummarizeLLMTokenizerMetadata(const Graph& graph);
	std::string_view LLaMAQuantizedExecutionPolicyName(LLaMAQuantizedExecutionPolicy policy);
	LLaMAQuantizedExecutionPlan PlanLLaMAQuantizedWeightExecution(const Graph& archive,
	                                                              std::size_t dequantizedMemoryBudgetBytes = 0);
	LLaMAHyperparameters ParseLLaMAHyperparameters(const Graph& graph);
	LLaMAContextValidationReport ValidateLLaMAContextRequest(const LLaMAHyperparameters& hyperparameters,
	                                                         std::size_t requestedTokenCount,
	                                                         std::size_t maxCacheLength);
	ImportResult ImportGGUFMetadata(const std::filesystem::path& inputPath);
	ImportResult ImportGGUFArchive(const std::filesystem::path& inputPath);
	ImportSummary ConvertGGUFArchive(const std::filesystem::path& inputPath, const std::filesystem::path& outputPath);
	ImportSummary ConvertGGUFArchiveExternalWeights(const std::filesystem::path& inputPath,
	                                                const std::filesystem::path& outputPath,
	                                                const std::filesystem::path& weightsPath,
	                                                const Serialization::ExternalWeightSaveOptions& options = {});
} // namespace LiteNN::GGUF

#endif
