#include <LiteNN.h>

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
	};

	std::string_view LLaMACompatibilityProfileName(LLaMACompatibilityProfileKind kind);
	LLaMACompatibilityProfileDescriptor QueryLLaMACompatibilityProfile(LLaMACompatibilityProfileKind kind);
	std::vector<LLaMACompatibilityProfileDescriptor> QueryLLaMACompatibilityProfiles();
	std::optional<LLaMACompatibilityProfileKind> TryInferLLaMACompatibilityProfile(std::string_view architecture);
	LLaMACompatibilityReport AnalyzeLLaMACompatibility(const Graph& archive, LLaMACompatibilityProfileKind kind);
	LLMTokenizerMetadataSummary SummarizeLLMTokenizerMetadata(const Graph& graph);
	LLaMAHyperparameters ParseLLaMAHyperparameters(const Graph& graph);
	ImportResult ImportGGUFArchive(const std::filesystem::path& inputPath);
	ImportSummary ConvertGGUFArchive(const std::filesystem::path& inputPath, const std::filesystem::path& outputPath);
} // namespace LiteNN::GGUF

#endif
