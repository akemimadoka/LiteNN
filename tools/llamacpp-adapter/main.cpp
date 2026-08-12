#include "LlamaCppTokenizerAdapter.h"

#include <charconv>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <iterator>
#include <stdexcept>
#include <string>
#include <string_view>

namespace
{
	std::size_t ParsePositiveSize(std::string_view text, std::string_view label)
	{
		std::size_t value{};
		const auto parsed = std::from_chars(text.data(), text.data() + text.size(), value);
		if (text.empty() || parsed.ec != std::errc{} || parsed.ptr != text.data() + text.size() || value == 0)
		{
			throw std::runtime_error(std::string(label) + " must be a positive integer");
		}
		return value;
	}

	std::vector<std::int32_t> ParseGeneratedPrefix(std::string_view text)
	{
		if (text == "-")
		{
			return {};
		}
		return LiteNN::LlamaCppAdapter::ParseCommaTokenIds(text, "generated token ids");
	}

	void PrintUsage(std::string_view executable)
	{
		std::cerr
		    << "Usage:\n"
		    << "  " << executable << " tokenize <model.gguf> <text> <tokens.json>\n"
		    << "  " << executable << " tokenize-file <model.gguf> <text.bin> <tokens.json>\n"
		    << "  " << executable << " detokenize <model.gguf> <comma-token-ids> <text.bin>\n"
		    << "  " << executable << " chat-template <model.gguf> <user-text> <prompt.bin>\n"
		    << "  " << executable << " chat-template-file <model.gguf> <user-text.bin> <prompt.bin>\n"
		    << "  " << executable
		    << " decode-logits <model.gguf> <comma-prompt-token-ids> <comma-generated-token-ids> <output-dir>\n"
		    << "  " << executable
		    << " generate-greedy-logits <model.gguf> <comma-prompt-token-ids> <max-generated-tokens> <output-dir>\n"
		    << "  " << executable
		    << " decode-layer-checkpoints <model.gguf> <comma-prompt-token-ids> <comma-generated-token-ids-or-dash> "
		       "<comma-generated-indices> <output-dir>\n"
		    << "  " << executable
		    << " decode-sub-layer-checkpoints <model.gguf> <comma-prompt-token-ids> "
		       "<comma-generated-token-ids-or-dash> <comma-generated-indices> <comma-block-indices> <output-dir> "
		       "[logits-output-dir]\n";
	}

	std::string ReadBinary(const std::filesystem::path& path)
	{
		std::ifstream input(path, std::ios::binary);
		if (!input)
		{
			throw std::runtime_error("failed to open text input: " + path.string());
		}
		return { std::istreambuf_iterator<char>{ input }, std::istreambuf_iterator<char>{} };
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
} // namespace

int main(int argc, char** argv)
try
{
	if (argc < 2)
	{
		PrintUsage(argv[0]);
		return 2;
	}
	const std::string_view command = argv[1];
	if (command == "tokenize" && argc == 5)
	{
		const LiteNN::LlamaCppAdapter::Model model(argv[2]);
		LiteNN::LlamaCppAdapter::WriteTokensJson(model.Tokenize(argv[3]), argv[4]);
		return 0;
	}
	if (command == "tokenize-file" && argc == 5)
	{
		const LiteNN::LlamaCppAdapter::Model model(argv[2]);
		LiteNN::LlamaCppAdapter::WriteTokensJson(model.Tokenize(ReadBinary(argv[3])), argv[4]);
		return 0;
	}
	if (command == "detokenize" && argc == 5)
	{
		const LiteNN::LlamaCppAdapter::Model model(argv[2]);
		const auto tokens = LiteNN::LlamaCppAdapter::ParseCommaTokenIds(argv[3], "token ids");
		WriteBinary(model.Detokenize(tokens), argv[4]);
		return 0;
	}
	if (command == "chat-template" && argc == 5)
	{
		const LiteNN::LlamaCppAdapter::Model model(argv[2]);
		WriteBinary(model.ApplyChatTemplate(argv[3]), argv[4]);
		return 0;
	}
	if (command == "chat-template-file" && argc == 5)
	{
		const LiteNN::LlamaCppAdapter::Model model(argv[2]);
		WriteBinary(model.ApplyChatTemplate(ReadBinary(argv[3])), argv[4]);
		return 0;
	}
	if (command == "decode-logits" && argc == 6)
	{
		const LiteNN::LlamaCppAdapter::Model model(argv[2]);
		const auto promptTokens = LiteNN::LlamaCppAdapter::ParseCommaTokenIds(argv[3], "prompt token ids");
		const auto generatedTokens = LiteNN::LlamaCppAdapter::ParseCommaTokenIds(argv[4], "generated token ids");
		model.CaptureDecodeLogits(promptTokens, generatedTokens, argv[5]);
		std::cout << "Captured " << generatedTokens.size() << " llama.cpp decode-logits steps in " << argv[5] << '\n';
		return 0;
	}
	if (command == "generate-greedy-logits" && argc == 6)
	{
		const LiteNN::LlamaCppAdapter::Model model(argv[2]);
		const auto promptTokens = LiteNN::LlamaCppAdapter::ParseCommaTokenIds(argv[3], "prompt token ids");
		const auto maximumTokens = ParsePositiveSize(argv[4], "max-generated-tokens");
		const std::filesystem::path outputDirectory = argv[5];
		const auto result = model.CaptureGreedyGeneration(promptTokens, maximumTokens, outputDirectory / "logits");
		LiteNN::LlamaCppAdapter::WriteNaturalGenerationManifest(promptTokens, result, outputDirectory);
		std::cout << "Generated " << result.generatedTokenIds.size() << " greedy llama.cpp tokens with logits in "
		          << outputDirectory << '\n';
		return 0;
	}
	if (command == "decode-layer-checkpoints" && argc == 7)
	{
		const LiteNN::LlamaCppAdapter::Model model(argv[2]);
		const auto promptTokens = LiteNN::LlamaCppAdapter::ParseCommaTokenIds(argv[3], "prompt token ids");
		const auto generatedTokens = ParseGeneratedPrefix(argv[4]);
		const auto parsedIndices = LiteNN::LlamaCppAdapter::ParseCommaTokenIds(argv[5], "generated indices");
		std::vector<std::size_t> generatedIndices;
		generatedIndices.reserve(parsedIndices.size());
		for (const auto index : parsedIndices)
		{
			generatedIndices.push_back(static_cast<std::size_t>(index));
		}
		model.CaptureDecodeLayerCheckpoints(promptTokens, generatedTokens, generatedIndices, argv[6]);
		std::cout << "Captured " << generatedIndices.size() << " llama.cpp layer-checkpoint steps in " << argv[6]
		          << '\n';
		return 0;
	}
	if (command == "decode-sub-layer-checkpoints" && (argc == 8 || argc == 9))
	{
		const LiteNN::LlamaCppAdapter::Model model(argv[2]);
		const auto promptTokens = LiteNN::LlamaCppAdapter::ParseCommaTokenIds(argv[3], "prompt token ids");
		const auto generatedTokens = ParseGeneratedPrefix(argv[4]);
		const auto parsedIndices = LiteNN::LlamaCppAdapter::ParseCommaTokenIds(argv[5], "generated indices");
		const auto parsedBlocks = LiteNN::LlamaCppAdapter::ParseCommaTokenIds(argv[6], "block indices");
		std::vector<std::size_t> generatedIndices;
		std::vector<std::size_t> blockIndices;
		generatedIndices.reserve(parsedIndices.size());
		blockIndices.reserve(parsedBlocks.size());
		for (const auto index : parsedIndices)
		{
			generatedIndices.push_back(static_cast<std::size_t>(index));
		}
		for (const auto block : parsedBlocks)
		{
			blockIndices.push_back(static_cast<std::size_t>(block));
		}
		model.CaptureDecodeSubLayerCheckpoints(promptTokens, generatedTokens, generatedIndices, blockIndices, argv[7],
		                                       argc == 9 ? std::filesystem::path(argv[8]) : std::filesystem::path{});
		std::cout << "Captured " << generatedIndices.size() << " llama.cpp sub-layer checkpoint steps for "
		          << blockIndices.size() << " blocks in " << argv[7] << '\n';
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
