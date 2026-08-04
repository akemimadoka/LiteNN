#include "LLMGeneration.h"

#include <algorithm>
#include <cmath>
#include <cstring>
#include <limits>
#include <numeric>
#include <random>
#include <stdexcept>
#include <string_view>
#include <unordered_set>

namespace LiteNN::GGUF
{
	namespace
	{
		struct Candidate
		{
			std::int32_t tokenId{};
			float logit{};
			double weight{};
		};

		float ApplyRepeatPenalty(float logit, float penalty)
		{
			if (penalty <= 1.0f)
			{
				return logit;
			}
			return logit >= 0.0f ? logit / penalty : logit * penalty;
		}

		void ValidateSamplingConfig(const LLMSamplingConfig& config)
		{
			if (!std::isfinite(config.temperature) || config.temperature < 0.0f)
			{
				throw std::runtime_error("LLM sampling temperature must be finite and non-negative");
			}
			if (!std::isfinite(config.topP) || config.topP <= 0.0f || config.topP > 1.0f)
			{
				throw std::runtime_error("LLM sampling topP must be in (0, 1]");
			}
			if (!std::isfinite(config.repeatPenalty) || config.repeatPenalty < 1.0f)
			{
				throw std::runtime_error("LLM sampling repeatPenalty must be finite and >= 1");
			}
		}

		std::vector<Candidate> BuildCandidates(std::span<const float> logits, const LLMSamplingConfig& config,
		                                       std::span<const std::int32_t> history,
		                                       std::optional<std::int32_t> suppressedTokenId)
		{
			if (logits.empty() || logits.size() > static_cast<std::size_t>(std::numeric_limits<std::int32_t>::max()))
			{
				throw std::runtime_error("LLM sampling requires a non-empty logits vector representable as int32 ids");
			}

			std::unordered_set<std::int32_t> seenTokens;
			if (config.repeatPenalty > 1.0f)
			{
				seenTokens.insert(history.begin(), history.end());
			}
			std::vector<Candidate> candidates;
			candidates.reserve(logits.size());
			for (std::size_t i = 0; i < logits.size(); ++i)
			{
				if (!std::isfinite(logits[i]))
				{
					throw std::runtime_error("LLM sampling logits must be finite");
				}
				const auto tokenId = static_cast<std::int32_t>(i);
				if (suppressedTokenId == tokenId)
				{
					continue;
				}
				const auto adjusted = config.repeatPenalty > 1.0f && seenTokens.contains(tokenId)
				                          ? ApplyRepeatPenalty(logits[i], config.repeatPenalty)
				                          : logits[i];
				candidates.push_back({ .tokenId = tokenId, .logit = adjusted, .weight = 0.0 });
			}
			if (candidates.empty())
			{
				throw std::runtime_error("LLM sampling suppressed every candidate token");
			}
			std::ranges::sort(candidates, [](const Candidate& lhs, const Candidate& rhs) {
				if (lhs.logit == rhs.logit)
				{
					return lhs.tokenId < rhs.tokenId;
				}
				return lhs.logit > rhs.logit;
			});
			if (config.topK > 0 && candidates.size() > config.topK)
			{
				candidates.resize(config.topK);
			}
			return candidates;
		}

		std::int32_t SelectGreedyToken(std::span<const float> logits, const LLMSamplingConfig& config,
		                               std::span<const std::int32_t> history,
		                               std::optional<std::int32_t> suppressedTokenId)
		{
			if (logits.empty() || logits.size() > static_cast<std::size_t>(std::numeric_limits<std::int32_t>::max()))
			{
				throw std::runtime_error("LLM sampling requires a non-empty logits vector representable as int32 ids");
			}
			std::unordered_set<std::int32_t> seenTokens;
			if (config.repeatPenalty > 1.0f)
			{
				seenTokens.insert(history.begin(), history.end());
			}
			std::int32_t bestToken = 0;
			float bestLogit = -std::numeric_limits<float>::infinity();
			bool foundCandidate = false;
			for (std::size_t i = 0; i < logits.size(); ++i)
			{
				if (!std::isfinite(logits[i]))
				{
					throw std::runtime_error("LLM sampling logits must be finite");
				}
				const auto tokenId = static_cast<std::int32_t>(i);
				if (suppressedTokenId == tokenId)
				{
					continue;
				}
				const auto adjusted = config.repeatPenalty > 1.0f && seenTokens.contains(tokenId)
				                          ? ApplyRepeatPenalty(logits[i], config.repeatPenalty)
				                          : logits[i];
				if (adjusted > bestLogit)
				{
					bestLogit = adjusted;
					bestToken = tokenId;
					foundCandidate = true;
				}
			}
			if (!foundCandidate)
			{
				throw std::runtime_error("LLM sampling suppressed every candidate token");
			}
			return bestToken;
		}

		std::span<const float> LastTokenLogitsView(const Tensor<CPU>& logits)
		{
			if (logits.DType() != DataType::Float32)
			{
				throw std::runtime_error("LLM logits post-processing currently requires Float32 logits");
			}
			const auto shape = logits.Shape();
			const auto* data = static_cast<const float*>(logits.UnsafeRawData());
			if (shape.NumDim() == 1 && shape[0] > 0)
			{
				return { data, shape[0] };
			}
			if (shape.NumDim() != 2 || shape[0] == 0 || shape[1] == 0)
			{
				throw std::runtime_error(
				    "LLM logits post-processing expects rank-1 [vocab] or rank-2 [sequence, vocab]");
			}
			return { data + (shape[0] - 1) * shape[1], shape[1] };
		}
	} // namespace

	LLMPromptTokens MakeCallerProvidedPromptTokens(std::span<const std::int32_t> tokenIds,
	                                               const LLMTokenizerMetadataSummary& tokenizer)
	{
		if (tokenIds.empty())
		{
			throw std::runtime_error("LLM prompt token bridge requires at least one caller-provided token id");
		}
		for (const auto tokenId : tokenIds)
		{
			if (tokenId < 0)
			{
				throw std::runtime_error("LLM prompt token ids must be non-negative");
			}
			if (tokenizer.tokenCount > 0 && static_cast<std::size_t>(tokenId) >= tokenizer.tokenCount)
			{
				throw std::runtime_error("LLM prompt token id exceeds tokenizer vocabulary size");
			}
		}
		return { .tokenIds = std::vector<std::int32_t>(tokenIds.begin(), tokenIds.end()), .callerProvided = true };
	}

	LLMPromptTokens MakeExactVocabularyPromptTokens(std::string_view text, const Graph& archive, bool addBos)
	{
		if (text.empty())
		{
			throw std::runtime_error("LLM exact tokenizer bridge requires a non-empty prompt");
		}
		const auto* tokensEntry = archive.FindMetadata("tokenizer.ggml.tokens");
		if (tokensEntry == nullptr)
		{
			throw std::runtime_error("LLM exact tokenizer bridge requires tokenizer.ggml.tokens metadata");
		}
		const auto* tokens = std::get_if<std::vector<std::string>>(&tokensEntry->value);
		if (tokens == nullptr || tokens->empty())
		{
			throw std::runtime_error("LLM exact tokenizer bridge requires a non-empty string token vocabulary");
		}

		std::vector<std::int32_t> tokenIds;
		const auto tokenizer = SummarizeLLMTokenizerMetadata(archive);
		if (addBos && tokenizer.bosTokenId)
		{
			if (*tokenizer.bosTokenId < 0 ||
			    *tokenizer.bosTokenId > static_cast<std::int64_t>(std::numeric_limits<std::int32_t>::max()) ||
			    static_cast<std::size_t>(*tokenizer.bosTokenId) >= tokens->size())
			{
				throw std::runtime_error("LLM exact tokenizer bridge found an out-of-range BOS token id");
			}
			tokenIds.push_back(static_cast<std::int32_t>(*tokenizer.bosTokenId));
		}

		std::size_t offset = 0;
		while (offset < text.size())
		{
			std::size_t bestIndex = tokens->size();
			std::size_t bestLength = 0;
			for (std::size_t i = 0; i < tokens->size(); ++i)
			{
				const std::string_view token = (*tokens)[i];
				if (!token.empty() && token.size() > bestLength && text.substr(offset).starts_with(token))
				{
					bestIndex = i;
					bestLength = token.size();
				}
			}
			if (bestIndex == tokens->size())
			{
				throw std::runtime_error("LLM exact tokenizer bridge cannot match prompt at byte offset " +
				                         std::to_string(offset));
			}
			if (bestIndex > static_cast<std::size_t>(std::numeric_limits<std::int32_t>::max()))
			{
				throw std::runtime_error("LLM exact tokenizer bridge vocabulary exceeds int32 token-id range");
			}
			tokenIds.push_back(static_cast<std::int32_t>(bestIndex));
			offset += bestLength;
		}
		return { .tokenIds = std::move(tokenIds), .callerProvided = false };
	}

	LLMGenerationState BeginGeneration(LLMPromptTokens prompt, std::optional<std::int32_t> eosTokenId)
	{
		if (prompt.tokenIds.empty())
		{
			throw std::runtime_error("LLM generation requires a non-empty prompt token sequence");
		}
		return {
			.tokens = std::move(prompt.tokenIds),
			.eosTokenId = eosTokenId,
			.finished = false,
			.generatedTokenCount = 0,
		};
	}

	std::vector<float> ExtractLastTokenLogits(const Tensor<CPU>& logits)
	{
		if (logits.DType() != DataType::Float32)
		{
			throw std::runtime_error("LLM logits post-processing currently requires Float32 logits");
		}
		const auto shape = logits.Shape();
		if (shape.NumDim() == 1)
		{
			std::vector<float> result(logits.NumElements());
			std::memcpy(result.data(), logits.UnsafeRawData(), result.size() * sizeof(float));
			return result;
		}
		if (shape.NumDim() != 2 || shape[0] == 0 || shape[1] == 0)
		{
			throw std::runtime_error("LLM logits post-processing expects rank-1 [vocab] or rank-2 [sequence, vocab]");
		}

		const auto vocabSize = shape[1];
		std::vector<float> result(vocabSize);
		const auto* data = static_cast<const float*>(logits.UnsafeRawData());
		std::copy_n(data + (shape[0] - 1) * vocabSize, vocabSize, result.begin());
		return result;
	}

	std::int32_t SelectNextToken(std::span<const float> logits, LLMSamplerState& sampler,
	                             std::span<const std::int32_t> history, std::optional<std::int32_t> suppressedTokenId)
	{
		ValidateSamplingConfig(sampler.config);
		if (suppressedTokenId &&
		    (*suppressedTokenId < 0 || static_cast<std::size_t>(*suppressedTokenId) >= logits.size()))
		{
			throw std::runtime_error("LLM sampling suppressed token id is outside the logits vocabulary");
		}
		if (sampler.config.mode == LLMSamplingMode::Greedy || sampler.config.temperature == 0.0f)
		{
			return SelectGreedyToken(logits, sampler.config, history, suppressedTokenId);
		}
		auto candidates = BuildCandidates(logits, sampler.config, history, suppressedTokenId);

		const auto maxLogit = candidates.front().logit;
		double totalWeight = 0.0;
		for (auto& candidate : candidates)
		{
			candidate.weight = std::exp((static_cast<double>(candidate.logit) - maxLogit) /
			                            static_cast<double>(sampler.config.temperature));
			totalWeight += candidate.weight;
		}
		if (!(totalWeight > 0.0) || !std::isfinite(totalWeight))
		{
			throw std::runtime_error("LLM sampling produced invalid probability mass");
		}

		if (sampler.config.topP < 1.0f)
		{
			double cumulative = 0.0;
			std::size_t keep = 0;
			for (; keep < candidates.size(); ++keep)
			{
				cumulative += candidates[keep].weight;
				if ((cumulative / totalWeight) >= sampler.config.topP)
				{
					++keep;
					break;
				}
			}
			candidates.resize(std::max<std::size_t>(keep, 1));
			totalWeight =
			    std::accumulate(candidates.begin(), candidates.end(), 0.0,
			                    [](double sum, const Candidate& candidate) { return sum + candidate.weight; });
		}

		std::mt19937_64 rng(sampler.config.seed + sampler.drawCount++);
		std::uniform_real_distribution<double> distribution(0.0, totalWeight);
		auto draw = distribution(rng);
		for (const auto& candidate : candidates)
		{
			if (draw <= candidate.weight)
			{
				return candidate.tokenId;
			}
			draw -= candidate.weight;
		}
		return candidates.back().tokenId;
	}

	std::int32_t SelectNextToken(const Tensor<CPU>& logits, LLMSamplerState& sampler,
	                             std::span<const std::int32_t> history, std::optional<std::int32_t> suppressedTokenId)
	{
		return SelectNextToken(LastTokenLogitsView(logits), sampler, history, suppressedTokenId);
	}

	std::int32_t StepGeneration(LLMGenerationState& generation, std::span<const float> logits, LLMSamplerState& sampler)
	{
		if (generation.finished)
		{
			throw std::runtime_error("LLM generation cannot step after EOS");
		}
		const auto nextToken = SelectNextToken(logits, sampler, generation.tokens);
		generation.tokens.push_back(nextToken);
		++generation.generatedTokenCount;
		if (generation.eosTokenId && nextToken == *generation.eosTokenId)
		{
			generation.finished = true;
		}
		return nextToken;
	}

	std::int32_t StepGeneration(LLMGenerationState& generation, const Tensor<CPU>& logits, LLMSamplerState& sampler)
	{
		return StepGeneration(generation, LastTokenLogitsView(logits), sampler);
	}
} // namespace LiteNN::GGUF
