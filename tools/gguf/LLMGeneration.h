#include "GGUFImporter.h"

#include <cstdint>
#include <optional>
#include <span>
#include <string_view>
#include <vector>

#ifndef LITENN_LLMGENERATION_H
#define LITENN_LLMGENERATION_H

namespace LiteNN::GGUF
{
	enum class LLMSamplingMode
	{
		Greedy,
		Random
	};

	struct LLMSamplingConfig
	{
		LLMSamplingMode mode{ LLMSamplingMode::Greedy };
		float temperature{ 1.0f };
		std::size_t topK{};
		float topP{ 1.0f };
		float repeatPenalty{ 1.0f };
		std::uint64_t seed{};
	};

	struct LLMSamplerState
	{
		LLMSamplingConfig config;
		std::uint64_t drawCount{};
	};

	struct LLMPromptTokens
	{
		std::vector<std::int32_t> tokenIds;
		bool callerProvided{};
	};

	struct LLMGenerationState
	{
		std::vector<std::int32_t> tokens;
		std::optional<std::int32_t> eosTokenId;
		bool finished{};
		std::size_t generatedTokenCount{};
	};

	LLMPromptTokens MakeCallerProvidedPromptTokens(std::span<const std::int32_t> tokenIds,
	                                               const LLMTokenizerMetadataSummary& tokenizer);
	LLMPromptTokens MakeExactVocabularyPromptTokens(std::string_view text, const Graph& archive, bool addBos = true);
	LLMGenerationState BeginGeneration(LLMPromptTokens prompt, std::optional<std::int32_t> eosTokenId = std::nullopt);
	std::vector<float> ExtractLastTokenLogits(const Tensor<CPU>& logits);
	std::int32_t SelectNextToken(std::span<const float> logits, LLMSamplerState& sampler,
	                             std::span<const std::int32_t> history = {},
	                             std::optional<std::int32_t> suppressedTokenId = std::nullopt);
	std::int32_t SelectNextToken(const Tensor<CPU>& logits, LLMSamplerState& sampler,
	                             std::span<const std::int32_t> history = {},
	                             std::optional<std::int32_t> suppressedTokenId = std::nullopt);
	std::int32_t StepGeneration(LLMGenerationState& generation, std::span<const float> logits,
	                            LLMSamplerState& sampler);
	std::int32_t StepGeneration(LLMGenerationState& generation, const Tensor<CPU>& logits, LLMSamplerState& sampler);
} // namespace LiteNN::GGUF

#endif
