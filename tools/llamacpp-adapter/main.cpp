#include <llama.h>

#include <charconv>
#include <cstdint>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <limits>
#include <memory>
#include <stdexcept>
#include <string>
#include <string_view>
#include <vector>

namespace
{
	using ModelPtr = std::unique_ptr<llama_model, decltype(&llama_model_free)>;
	using ContextPtr = std::unique_ptr<llama_context, decltype(&llama_free)>;

	struct BackendGuard
	{
		BackendGuard()
		{
			llama_backend_init();
		}

		~BackendGuard()
		{
			llama_backend_free();
		}
	};

	void PrintUsage(std::string_view executable)
	{
		std::cerr << "Usage:\n"
		          << "  " << executable << " tokenize <model.gguf> <text> <tokens.json>\n"
		          << "  " << executable << " detokenize <model.gguf> <comma-token-ids> <text.bin>\n"
		          << "  " << executable << " chat-template <model.gguf> <user-text> <prompt.bin>\n"
		          << "  " << executable
		          << " decode-logits <model.gguf> <comma-prompt-token-ids> <comma-generated-token-ids> <output-dir>\n";
	}

	ModelPtr LoadModel(std::string_view path)
	{
		auto params = llama_model_default_params();
		auto* model = llama_model_load_from_file(std::string(path).c_str(), params);
		if (model == nullptr)
		{
			throw std::runtime_error("failed to load llama.cpp model");
		}
		return ModelPtr(model, llama_model_free);
	}

	std::vector<llama_token> ParseTokenIds(std::string_view text, std::string_view label)
	{
		std::vector<llama_token> result;
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

	std::vector<llama_token> Tokenize(const llama_vocab* vocabulary, std::string_view text)
	{
		if (text.size() > static_cast<std::size_t>(std::numeric_limits<std::int32_t>::max()))
		{
			throw std::runtime_error("text is too large for llama.cpp tokenization");
		}
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
		return tokens;
	}

	void WriteTokens(const std::vector<llama_token>& tokens, bool addBos, const std::filesystem::path& path)
	{
		std::ofstream output(path);
		if (!output)
		{
			throw std::runtime_error("failed to open token output: " + path.string());
		}
		output << "{\n  \"schema\": \"litenn.llamacpp_tokens.v1\",\n"
		       << "  \"addBos\": " << (addBos ? "true" : "false") << ",\n"
		       << "  \"parseSpecial\": true,\n  \"tokenIds\": [";
		for (std::size_t i = 0; i < tokens.size(); ++i)
		{
			if (i != 0)
			{
				output << ", ";
			}
			output << tokens[i];
		}
		output << "]\n}\n";
	}

	std::string Detokenize(const llama_vocab* vocabulary, const std::vector<llama_token>& tokens)
	{
		const auto tokenCount = static_cast<std::int32_t>(tokens.size());
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

	void WriteBinary(std::string_view text, const std::filesystem::path& path)
	{
		std::ofstream output(path, std::ios::binary);
		if (!output)
		{
			throw std::runtime_error("failed to open text output: " + path.string());
		}
		output.write(text.data(), static_cast<std::streamsize>(text.size()));
	}

	std::string ApplyChatTemplate(const llama_model* model, std::string_view userText)
	{
		const auto* chatTemplate = llama_model_chat_template(model, nullptr);
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

	void RunDecodeLogits(llama_model* model, std::string_view promptText, std::string_view generatedText,
	                     const std::filesystem::path& outputDirectory)
	{
		const auto promptTokenIds = ParseTokenIds(promptText, "prompt token ids");
		const auto generatedTokenIds = ParseTokenIds(generatedText, "generated token ids");
		std::filesystem::create_directories(outputDirectory);
		auto contextParams = llama_context_default_params();
		contextParams.n_ctx = static_cast<std::uint32_t>(promptTokenIds.size() + generatedTokenIds.size());
		contextParams.n_batch = static_cast<std::uint32_t>(promptTokenIds.size());
		contextParams.no_perf = true;
		auto* rawContext = llama_init_from_model(model, contextParams);
		if (rawContext == nullptr)
		{
			throw std::runtime_error("failed to create llama.cpp context");
		}
		const ContextPtr context(rawContext, llama_free);
		auto prompt = promptTokenIds;
		if (llama_decode(context.get(), llama_batch_get_one(prompt.data(), static_cast<std::int32_t>(prompt.size()))) !=
		    0)
		{
			throw std::runtime_error("llama.cpp prompt decode failed");
		}
		const auto vocabularySize = llama_vocab_n_tokens(llama_model_get_vocab(model));
		for (std::size_t step = 0; step < generatedTokenIds.size(); ++step)
		{
			auto token = generatedTokenIds[step];
			if (llama_decode(context.get(), llama_batch_get_one(&token, 1)) != 0)
			{
				throw std::runtime_error("llama.cpp token decode failed at step " + std::to_string(step + 1));
			}
			WriteLogits(llama_get_logits_ith(context.get(), -1), vocabularySize,
			            outputDirectory / ("decode-step-" + std::to_string(step + 1) + ".txt"));
		}
		std::cout << "Captured " << generatedTokenIds.size() << " llama.cpp decode-logits steps in " << outputDirectory
		          << '\n';
	}
} // namespace

int main(int argc, char** argv)
try
{
	if (argc < 2)
	{
		PrintUsage(argv[0]);
		return 2;
	}
	const BackendGuard backend;
	const std::string_view command = argv[1];
	if (command == "tokenize" && argc == 5)
	{
		const auto model = LoadModel(argv[2]);
		const auto* vocabulary = llama_model_get_vocab(model.get());
		const auto tokens = Tokenize(vocabulary, argv[3]);
		WriteTokens(tokens, llama_vocab_get_add_bos(vocabulary), argv[4]);
		return 0;
	}
	if (command == "detokenize" && argc == 5)
	{
		const auto model = LoadModel(argv[2]);
		const auto tokens = ParseTokenIds(argv[3], "token ids");
		WriteBinary(Detokenize(llama_model_get_vocab(model.get()), tokens), argv[4]);
		return 0;
	}
	if (command == "chat-template" && argc == 5)
	{
		const auto model = LoadModel(argv[2]);
		WriteBinary(ApplyChatTemplate(model.get(), argv[3]), argv[4]);
		return 0;
	}
	if (command == "decode-logits" && argc == 6)
	{
		const auto model = LoadModel(argv[2]);
		RunDecodeLogits(model.get(), argv[3], argv[4], argv[5]);
		return 0;
	}
	PrintUsage(argv[0]);
	return 2;
}
catch (const std::exception& error)
{
	std::cerr << "error: " << error.what() << '\n';
	return 1;
}
