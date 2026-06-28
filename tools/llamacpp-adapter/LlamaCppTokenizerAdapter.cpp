#include "LlamaCppTokenizerAdapter.h"

#include <llama.h>

#include <charconv>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <limits>
#include <memory>
#include <stdexcept>
#include <string>
#include <utility>

namespace LiteNN::LlamaCppAdapter
{
	namespace
	{
		void EnsureBackendInitialized()
		{
			static const bool initialized = [] {
				llama_backend_init();
				return true;
			}();
			(void) initialized;
		}

		std::vector<llama_token> ToLlamaTokens(std::span<const std::int32_t> tokenIds)
		{
			std::vector<llama_token> tokens;
			tokens.reserve(tokenIds.size());
			for (const auto token : tokenIds)
			{
				if (token < 0)
				{
					throw std::runtime_error("token ids must be non-negative");
				}
				tokens.push_back(static_cast<llama_token>(token));
			}
			return tokens;
		}

		std::vector<std::int32_t> FromLlamaTokens(std::span<const llama_token> tokens)
		{
			std::vector<std::int32_t> result;
			result.reserve(tokens.size());
			for (const auto token : tokens)
			{
				if (token < 0 || token > std::numeric_limits<std::int32_t>::max())
				{
					throw std::runtime_error("llama.cpp returned an out-of-range token id");
				}
				result.push_back(static_cast<std::int32_t>(token));
			}
			return result;
		}

		void WriteLogits(const float* logits, std::int32_t vocabularySize, const std::filesystem::path& path)
		{
			if (logits == nullptr)
			{
				throw std::runtime_error("llama.cpp returned null logits");
			}
			std::ofstream output(path);
			if (!output)
			{
				throw std::runtime_error("failed to open logits output: " + path.string());
			}
			output << std::setprecision(9);
			for (std::int32_t i = 0; i < vocabularySize; ++i)
			{
				output << i << ": " << logits[i] << '\n';
			}
		}
	} // namespace

	struct Model::Impl
	{
		explicit Impl(const std::filesystem::path& path)
		{
			EnsureBackendInitialized();
			auto params = llama_model_default_params();
			model = llama_model_load_from_file(path.string().c_str(), params);
			if (model == nullptr)
			{
				throw std::runtime_error("failed to load llama.cpp model");
			}
		}

		~Impl()
		{
			if (model != nullptr)
			{
				llama_model_free(model);
			}
		}

		llama_model* model{};
	};

	Model::Model(const std::filesystem::path& path) : impl_(std::make_unique<Impl>(path))
	{
	}

	Model::~Model() = default;
	Model::Model(Model&&) noexcept = default;
	Model& Model::operator=(Model&&) noexcept = default;

	TokenizationResult Model::Tokenize(std::string_view text) const
	{
		if (text.size() > static_cast<std::size_t>(std::numeric_limits<std::int32_t>::max()))
		{
			throw std::runtime_error("text is too large for llama.cpp tokenization");
		}
		const auto* vocabulary = llama_model_get_vocab(impl_->model);
		const auto textLength = static_cast<std::int32_t>(text.size());
		const bool addSpecial = llama_vocab_get_add_bos(vocabulary);
		const auto required = llama_tokenize(vocabulary, text.data(), textLength, nullptr, 0, addSpecial, true);
		if (required == std::numeric_limits<std::int32_t>::min())
		{
			throw std::runtime_error("llama.cpp tokenization size overflow");
		}
		std::vector<llama_token> tokens(static_cast<std::size_t>(required < 0 ? -required : required));
		const auto count = llama_tokenize(vocabulary, text.data(), textLength, tokens.data(),
		                                  static_cast<std::int32_t>(tokens.size()), addSpecial, true);
		if (count < 0)
		{
			throw std::runtime_error("llama.cpp tokenization failed");
		}
		tokens.resize(static_cast<std::size_t>(count));
		return { .tokenIds = FromLlamaTokens(tokens), .addBos = addSpecial, .parseSpecial = true };
	}

	std::string Model::Detokenize(std::span<const std::int32_t> tokenIds) const
	{
		const auto tokens = ToLlamaTokens(tokenIds);
		const auto tokenCount = static_cast<std::int32_t>(tokens.size());
		const auto* vocabulary = llama_model_get_vocab(impl_->model);
		const auto required = llama_detokenize(vocabulary, tokens.data(), tokenCount, nullptr, 0, false, true);
		if (required == std::numeric_limits<std::int32_t>::min())
		{
			throw std::runtime_error("llama.cpp detokenization size overflow");
		}
		std::string text(static_cast<std::size_t>(required < 0 ? -required : required), '\0');
		const auto count = llama_detokenize(vocabulary, tokens.data(), tokenCount, text.data(),
		                                    static_cast<std::int32_t>(text.size()), false, true);
		if (count < 0)
		{
			throw std::runtime_error("llama.cpp detokenization failed");
		}
		text.resize(static_cast<std::size_t>(count));
		return text;
	}

	std::string Model::ApplyChatTemplate(std::string_view userText) const
	{
		const auto* chatTemplate = llama_model_chat_template(impl_->model, nullptr);
		if (chatTemplate == nullptr)
		{
			throw std::runtime_error("GGUF model does not define a supported chat template");
		}
		const std::string content(userText);
		const llama_chat_message message{ "user", content.c_str() };
		const auto required = llama_chat_apply_template(chatTemplate, &message, 1, true, nullptr, 0);
		if (required < 0)
		{
			throw std::runtime_error("llama.cpp failed to size the chat template output");
		}
		std::string prompt(static_cast<std::size_t>(required), '\0');
		const auto count = llama_chat_apply_template(chatTemplate, &message, 1, true, prompt.data(), required);
		if (count < 0 || count > required)
		{
			throw std::runtime_error("llama.cpp failed to apply the chat template");
		}
		prompt.resize(static_cast<std::size_t>(count));
		return prompt;
	}

	void Model::CaptureDecodeLogits(std::span<const std::int32_t> promptTokenIds,
	                                std::span<const std::int32_t> generatedTokenIds,
	                                const std::filesystem::path& outputDirectory) const
	{
		if (promptTokenIds.empty() || generatedTokenIds.empty())
		{
			throw std::runtime_error("decode-logits requires non-empty prompt and generated token ids");
		}
		std::filesystem::create_directories(outputDirectory);
		auto contextParams = llama_context_default_params();
		contextParams.n_ctx = static_cast<std::uint32_t>(promptTokenIds.size() + generatedTokenIds.size());
		contextParams.n_batch = static_cast<std::uint32_t>(promptTokenIds.size());
		contextParams.no_perf = true;
		auto* rawContext = llama_init_from_model(impl_->model, contextParams);
		if (rawContext == nullptr)
		{
			throw std::runtime_error("failed to create llama.cpp context");
		}
		const std::unique_ptr<llama_context, decltype(&llama_free)> context(rawContext, llama_free);
		auto prompt = ToLlamaTokens(promptTokenIds);
		if (llama_decode(context.get(), llama_batch_get_one(prompt.data(), static_cast<std::int32_t>(prompt.size()))) !=
		    0)
		{
			throw std::runtime_error("llama.cpp prompt decode failed");
		}
		const auto vocabularySize = llama_vocab_n_tokens(llama_model_get_vocab(impl_->model));
		const auto generated = ToLlamaTokens(generatedTokenIds);
		for (std::size_t step = 0; step < generated.size(); ++step)
		{
			auto token = generated[step];
			if (llama_decode(context.get(), llama_batch_get_one(&token, 1)) != 0)
			{
				throw std::runtime_error("llama.cpp token decode failed at step " + std::to_string(step + 1));
			}
			WriteLogits(llama_get_logits_ith(context.get(), -1), vocabularySize,
			            outputDirectory / ("decode-step-" + std::to_string(step + 1) + ".txt"));
		}
	}

	void WriteTokensJson(const TokenizationResult& result, const std::filesystem::path& path)
	{
		std::ofstream output(path);
		if (!output)
		{
			throw std::runtime_error("failed to open token output: " + path.string());
		}
		output << "{\n  \"schema\": \"litenn.llamacpp_tokens.v1\",\n"
		       << "  \"addBos\": " << (result.addBos ? "true" : "false") << ",\n"
		       << "  \"parseSpecial\": " << (result.parseSpecial ? "true" : "false") << ",\n  \"tokenIds\": [";
		for (std::size_t i = 0; i < result.tokenIds.size(); ++i)
		{
			if (i != 0)
			{
				output << ", ";
			}
			output << result.tokenIds[i];
		}
		output << "]\n}\n";
	}

	std::vector<std::int32_t> ParseCommaTokenIds(std::string_view text, std::string_view label)
	{
		std::vector<std::int32_t> result;
		while (!text.empty())
		{
			const auto separator = text.find(',');
			const auto part = text.substr(0, separator);
			std::int32_t value{};
			const auto parsed = std::from_chars(part.data(), part.data() + part.size(), value);
			if (part.empty() || parsed.ec != std::errc{} || parsed.ptr != part.data() + part.size() || value < 0)
			{
				throw std::runtime_error(std::string(label) + " must contain comma-separated non-negative integers");
			}
			result.push_back(value);
			if (separator == std::string_view::npos)
			{
				break;
			}
			text.remove_prefix(separator + 1);
		}
		if (result.empty())
		{
			throw std::runtime_error(std::string(label) + " must not be empty");
		}
		return result;
	}
} // namespace LiteNN::LlamaCppAdapter
