#ifndef LITENN_TOOLS_LLAMACPP_TOKENIZER_ADAPTER_H
#define LITENN_TOOLS_LLAMACPP_TOKENIZER_ADAPTER_H

#include <cstdint>
#include <filesystem>
#include <memory>
#include <span>
#include <string>
#include <string_view>
#include <vector>

namespace LiteNN::LlamaCppAdapter
{
	struct TokenizationResult
	{
		std::vector<std::int32_t> tokenIds;
		bool addBos{};
		bool parseSpecial{ true };
	};

	struct NaturalGenerationResult
	{
		std::vector<std::int32_t> generatedTokenIds;
		std::size_t requestedTokenCount{};
		bool stoppedOnEos{};
	};

	class Model
	{
	public:
		explicit Model(const std::filesystem::path& path);
		~Model();

		Model(Model&&) noexcept;
		Model& operator=(Model&&) noexcept;

		Model(const Model&) = delete;
		Model& operator=(const Model&) = delete;

		TokenizationResult Tokenize(std::string_view text) const;
		std::string Detokenize(std::span<const std::int32_t> tokenIds) const;
		std::string ApplyChatTemplate(std::string_view userText) const;
		NaturalGenerationResult CaptureGreedyGeneration(std::span<const std::int32_t> promptTokenIds,
		                                                std::size_t maximumGeneratedTokens,
		                                                const std::filesystem::path& logitsOutputDirectory) const;
		void CaptureTeacherForcedLogits(std::span<const std::int32_t> promptTokenIds,
		                                std::span<const std::int32_t> targetTokenIds,
		                                const std::filesystem::path& logitsOutputDirectory) const;
		void CaptureDecodeLogits(std::span<const std::int32_t> promptTokenIds,
		                         std::span<const std::int32_t> generatedTokenIds,
		                         const std::filesystem::path& outputDirectory) const;
		void CaptureDecodeLayerCheckpoints(std::span<const std::int32_t> promptTokenIds,
		                                   std::span<const std::int32_t> generatedTokenIds,
		                                   std::span<const std::size_t> generatedIndices,
		                                   const std::filesystem::path& outputDirectory) const;
		void CaptureDecodeSubLayerCheckpoints(std::span<const std::int32_t> promptTokenIds,
		                                      std::span<const std::int32_t> generatedTokenIds,
		                                      std::span<const std::size_t> generatedIndices,
		                                      std::span<const std::size_t> blockIndices,
		                                      const std::filesystem::path& outputDirectory,
		                                      const std::filesystem::path& logitsOutputDirectory = {}) const;

	private:
		struct Impl;
		std::unique_ptr<Impl> impl_;
	};

	void WriteTokensJson(const TokenizationResult& result, const std::filesystem::path& path);
	void WriteNaturalGenerationManifest(std::span<const std::int32_t> promptTokenIds,
	                                    const NaturalGenerationResult& result,
	                                    const std::filesystem::path& outputDirectory);
	void WriteTeacherForcedManifest(std::span<const std::int32_t> promptTokenIds,
	                                std::span<const std::int32_t> targetTokenIds,
	                                const std::filesystem::path& outputDirectory);
	std::vector<std::int32_t> ParseCommaTokenIds(std::string_view text, std::string_view label);
} // namespace LiteNN::LlamaCppAdapter

#endif
