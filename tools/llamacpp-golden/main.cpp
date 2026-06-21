#include <llama.h>

#include <charconv>
#include <cstdint>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <memory>
#include <stdexcept>
#include <string>
#include <string_view>
#include <vector>

namespace
{
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

int main(int argc, char** argv)
try
{
	if (argc != 5)
	{
		std::cerr << "Usage: " << argv[0]
		          << " <model.gguf> <comma-prompt-token-ids> <comma-generated-token-ids> <output-dir>\n";
		return 2;
	}
	const auto promptTokenIds = ParseTokenIds(argv[2], "prompt token ids");
	const auto generatedTokenIds = ParseTokenIds(argv[3], "generated token ids");
	const std::filesystem::path outputDirectory = argv[4];
	std::filesystem::create_directories(outputDirectory);

	llama_backend_init();
	const auto backendGuard =
	    std::unique_ptr<void, void (*)(void*)>(reinterpret_cast<void*>(1), [](void*) { llama_backend_free(); });
	auto modelParams = llama_model_default_params();
	auto* rawModel = llama_model_load_from_file(argv[1], modelParams);
	if (rawModel == nullptr)
	{
		throw std::runtime_error("failed to load llama.cpp model");
	}
	const auto model = std::unique_ptr<llama_model, decltype(&llama_model_free)>(rawModel, llama_model_free);

	auto contextParams = llama_context_default_params();
	contextParams.n_ctx = static_cast<std::uint32_t>(promptTokenIds.size() + generatedTokenIds.size());
	contextParams.n_batch = static_cast<std::uint32_t>(promptTokenIds.size());
	contextParams.no_perf = true;
	auto* rawContext = llama_init_from_model(model.get(), contextParams);
	if (rawContext == nullptr)
	{
		throw std::runtime_error("failed to create llama.cpp context");
	}
	const auto context = std::unique_ptr<llama_context, decltype(&llama_free)>(rawContext, llama_free);

	auto prompt = promptTokenIds;
	if (llama_decode(context.get(), llama_batch_get_one(prompt.data(), static_cast<std::int32_t>(prompt.size()))) != 0)
	{
		throw std::runtime_error("llama.cpp prompt decode failed");
	}
	const auto vocabularySize = llama_vocab_n_tokens(llama_model_get_vocab(model.get()));
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
	return 0;
}
catch (const std::exception& error)
{
	std::cerr << "error: " << error.what() << '\n';
	return 1;
}
