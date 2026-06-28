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
		void CaptureDecodeLogits(std::span<const std::int32_t> promptTokenIds,
		                         std::span<const std::int32_t> generatedTokenIds,
		                         const std::filesystem::path& outputDirectory) const;

	private:
		struct Impl;
		std::unique_ptr<Impl> impl_;
	};

	void WriteTokensJson(const TokenizationResult& result, const std::filesystem::path& path);
	std::vector<std::int32_t> ParseCommaTokenIds(std::string_view text, std::string_view label);
} // namespace LiteNN::LlamaCppAdapter

#endif
