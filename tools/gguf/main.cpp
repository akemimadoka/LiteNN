#include "GGMLQuantizedKernels.h"
#include "GGUFImporter.h"
#include "LLMGeneration.h"
#include "LLaMABuilder.h"

#ifdef LITENN_GGUF_CONVERT_ENABLE_AOT
#include <LiteNN/Compiler/CompiledModule.h>
#endif
#ifdef LITENN_GGUF_CONVERT_ENABLE_LLAMA_CPP_TOKENIZER
#include <LlamaCppTokenizerAdapter.h>
#endif
#include <LiteNN/Serialization/ModelPackageIO.h>

#include <algorithm>
#include <array>
#include <cctype>
#include <charconv>
#include <chrono>
#include <cstdint>
#include <cstdlib>
#include <filesystem>
#include <format>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <limits>
#include <optional>
#include <span>
#include <stdexcept>
#include <string>
#include <string_view>
#include <type_traits>
#include <vector>

namespace
{
	void PrintUsage(std::string_view executable)
	{
		std::cerr << "Usage:\n"
		          << "  " << executable << " --import <input.gguf> <output.ltnn>\n"
		          << "  " << executable << " --import-external <input.gguf> <output.ltnn> <weights.bin>\n"
		          << "  " << executable
		          << " --analyze-llm <input.gguf> [profile] [--dequantized-budget-bytes N|--dequantized-budget-mib N]\n"
		          << "  " << executable << " --plan-llm <input.gguf> <prefill-sequence-length> <decode-past-length> "
		          << "[max-cache-length]\n"
		          << "  " << executable
		          << " --lower-llama <input.gguf> <output.ltnn> <sequence-length> [position-offset]\n"
		          << "  " << executable
		          << " --lower-llama-quantized <input.gguf> <output.ltnn> <weights.bin> <sequence-length> "
		             "[position-offset]\n"
		          << "  " << executable
		          << " --lower-llama-decode <input.gguf> <output.ltnn> <sequence-length> <past-length>\n"
		          << "  " << executable
		          << " --lower-llama-decode-stateful <input.gguf> <output.ltnn> <weights.bin> <past-length> "
		             "<max-cache-length>\n"
		          << "  " << executable << " --run-llama-token-ids <input.gguf> <comma-token-ids> [position-offset]\n"
		          << "  " << executable
		          << " --dump-llama-token-id-logits <input.gguf> <comma-token-ids> <output.txt> [position-offset]\n"
		          << "  " << executable
		          << " --tokenize-llama-prompt <input.gguf> <prompt> <tokens.json> [--chat-template]\n"
		          << "  " << executable << " --run-llama-prompt <input.gguf> <prompt> [position-offset]\n"
		          << "  " << executable << " --run-llama-package-token-ids <input.ltnn> <comma-token-ids>\n"
		          << "  " << executable
		          << " --run-llama-decode-loop-token-id <input.gguf> <initial-token-id> <steps> [output.txt] "
		             "[--sample greedy|random] [--temperature T] [--top-k K] [--top-p P] [--repeat-penalty R] "
		             "[--seed N] [--logits-output output.txt] [--logits-output-dir dir] [--ignore-eos] "
		             "[--stateful|--functional] [--stream-tokens] [--stream-stats] [--compile-only] "
		             "[--max-cache-length N] "
		             "[--cpu-aot-threads N] [--cpu-aot-affinity none|compact] [--cpu-aot-llvm-opt-level 0|1|2|3] "
		             "[--cpu-aot-parallel-min-flops N] [--compile-diagnostics|--no-compile-diagnostics] "
		             "[--cpu-aot-q8k-staged-matmul]\n"
		          << "  " << executable
		          << " --run-llama-decode-loop-token-ids <input.gguf> <comma-token-ids> <steps> [output.txt] "
		             "[--sample greedy|random] [--temperature T] [--top-k K] [--top-p P] [--repeat-penalty R] "
		             "[--seed N] [--logits-output output.txt] [--logits-output-dir dir] [--ignore-eos] "
		             "[--stateful|--functional] [--stream-tokens] [--stream-stats] [--compile-only] "
		             "[--max-cache-length N] "
		             "[--cpu-aot-threads N] [--cpu-aot-affinity none|compact] [--cpu-aot-llvm-opt-level 0|1|2|3] "
		             "[--cpu-aot-parallel-min-flops N] [--compile-diagnostics|--no-compile-diagnostics] "
		             "[--cpu-aot-q8k-staged-matmul]\n"
		          << "  " << executable
		          << " --run-llama-prompt-decode-loop <input.gguf> <prompt> <steps> [output.txt] "
		             "[--sample greedy|random] [--temperature T] [--top-k K] [--top-p P] [--repeat-penalty R] "
		             "[--seed N] [--logits-output output.txt] [--logits-output-dir dir] [--ignore-eos] "
		             "[--stateful|--functional] [--stream-tokens] [--stream-stats] [--compile-only] "
		             "[--max-cache-length N] "
		             "[--cpu-aot-threads N] [--cpu-aot-affinity none|compact] [--cpu-aot-llvm-opt-level 0|1|2|3] "
		             "[--cpu-aot-parallel-min-flops N] [--compile-diagnostics|--no-compile-diagnostics] "
		             "[--cpu-aot-q8k-staged-matmul]\n"
		          << "  " << executable << " --compile-cpu <input.ltnn> <output.o> [symbol-prefix]\n"
		          << "  " << executable << " --compile-cuda <input.ltnn> <output.o> [symbol-prefix]\n"
		          << "  " << executable << " --compile-cpu-separated <input.ltnn> <output-dir> [symbol-prefix]\n"
		          << "  " << executable << " --compile-cuda-separated <input.ltnn> <output-dir> [symbol-prefix]\n"
		          << "  " << executable << " <input.gguf> <output.ltnn>  (alias for --import)\n";
	}

	std::size_t ParseSize(std::string_view text, std::string_view label, bool allowZero = false)
	{
		std::size_t value{};
		const auto* first = text.data();
		const auto* last = text.data() + text.size();
		const auto result = std::from_chars(first, last, value);
		if (result.ec != std::errc{} || result.ptr != last || (!allowZero && value == 0))
		{
			throw std::runtime_error(std::string(label) +
			                         (allowZero ? " must be a non-negative integer" : " must be a positive integer"));
		}
		return value;
	}

	std::uint64_t ParseU64(std::string_view text, std::string_view label)
	{
		std::uint64_t value{};
		const auto* first = text.data();
		const auto* last = text.data() + text.size();
		const auto result = std::from_chars(first, last, value);
		if (result.ec != std::errc{} || result.ptr != last)
		{
			throw std::runtime_error(std::string(label) + " must be a non-negative integer");
		}
		return value;
	}

	float ParseFloat(std::string_view text, std::string_view label)
	{
		float value{};
		const auto* first = text.data();
		const auto* last = text.data() + text.size();
		const auto result = std::from_chars(first, last, value);
		if (result.ec != std::errc{} || result.ptr != last)
		{
			throw std::runtime_error(std::string(label) + " must be a floating-point value");
		}
		return value;
	}

	std::vector<std::int32_t> ParseTokenIds(std::string_view text)
	{
		if (text.empty())
		{
			throw std::runtime_error("comma-token-ids must not be empty");
		}
		std::vector<std::int32_t> ids;
		std::size_t offset = 0;
		while (offset <= text.size())
		{
			const auto comma = text.find(',', offset);
			const auto end = comma == std::string_view::npos ? text.size() : comma;
			const auto token = text.substr(offset, end - offset);
			if (token.empty())
			{
				throw std::runtime_error("comma-token-ids contains an empty item");
			}
			std::int32_t value{};
			const auto* first = token.data();
			const auto* last = token.data() + token.size();
			const auto result = std::from_chars(first, last, value);
			if (result.ec != std::errc{} || result.ptr != last || value < 0)
			{
				throw std::runtime_error("comma-token-ids must contain non-negative int32 values");
			}
			ids.push_back(value);
			if (comma == std::string_view::npos)
			{
				break;
			}
			offset = comma + 1;
		}
		return ids;
	}

	std::int32_t ParseTokenId(std::string_view text, std::string_view label)
	{
		const auto ids = ParseTokenIds(text);
		if (ids.size() != 1)
		{
			throw std::runtime_error(std::string(label) + " must contain exactly one token id");
		}
		return ids.front();
	}

	LiteNN::GGUF::LLaMACompatibilityProfileKind ParseLLMProfile(std::string_view text)
	{
		for (const auto& profile : LiteNN::GGUF::QueryLLaMACompatibilityProfiles())
		{
			if (text == profile.name)
			{
				return profile.kind;
			}
		}
		throw std::runtime_error(std::format("Unknown LLM compatibility profile '{}'", text));
	}

	LiteNN::GGUF::LLMSamplingMode ParseSamplingMode(std::string_view text)
	{
		if (text == "greedy")
		{
			return LiteNN::GGUF::LLMSamplingMode::Greedy;
		}
		if (text == "random")
		{
			return LiteNN::GGUF::LLMSamplingMode::Random;
		}
		throw std::runtime_error(std::format("Unknown sampling mode '{}'", text));
	}

	struct DecodeLoopCommandOptions
	{
		std::string inputPath;
		std::vector<std::int32_t> initialTokenIds;
		std::optional<std::string> exactPrompt;
		bool applyChatTemplate{};
		std::size_t steps{};
		std::optional<std::string> outputPath;
		std::optional<std::string> logitsOutputPath;
		std::optional<std::string> logitsOutputDirectory;
		LiteNN::GGUF::LLMSamplingConfig sampling;
		bool stopAtEos{ true };
		bool statefulDecode{ true };
		bool streamTokens{};
		bool streamStats{};
		bool compileOnly{};
		bool enableCPUAOTQ8KStagedMatMul{};
		std::optional<std::size_t> cpuAOTThreadCount;
		std::optional<std::uint64_t> cpuAOTParallelMinFlops;
		std::optional<std::uint8_t> cpuAOTLLVMOptLevel;
		std::optional<std::string> cpuAOTAffinityPolicy;
		std::optional<bool> enableCompileDiagnostics;
		std::optional<std::size_t> maxCacheLength;
	};

	void ParseDecodeLoopTrailingOptions(int argc, char** argv, int firstOptionIndex, DecodeLoopCommandOptions& options)
	{
		for (int i = firstOptionIndex; i < argc; ++i)
		{
			const std::string_view arg = argv[i];
			const auto requireValue = [&](std::string_view name) -> std::string_view {
				if (i + 1 >= argc)
				{
					throw std::runtime_error(std::string(name) + " requires a value");
				}
				++i;
				return argv[i];
			};
			if (arg == "--output")
			{
				options.outputPath = std::string(requireValue(arg));
			}
			else if (arg == "--logits-output")
			{
				options.logitsOutputPath = std::string(requireValue(arg));
			}
			else if (arg == "--logits-output-dir")
			{
				options.logitsOutputDirectory = std::string(requireValue(arg));
			}
			else if (arg == "--sample")
			{
				options.sampling.mode = ParseSamplingMode(requireValue(arg));
			}
			else if (arg == "--temperature")
			{
				options.sampling.temperature = ParseFloat(requireValue(arg), "temperature");
			}
			else if (arg == "--top-k")
			{
				options.sampling.topK = ParseSize(requireValue(arg), "top-k", true);
			}
			else if (arg == "--top-p")
			{
				options.sampling.topP = ParseFloat(requireValue(arg), "top-p");
			}
			else if (arg == "--repeat-penalty")
			{
				options.sampling.repeatPenalty = ParseFloat(requireValue(arg), "repeat-penalty");
			}
			else if (arg == "--seed")
			{
				options.sampling.seed = ParseU64(requireValue(arg), "seed");
			}
			else if (arg == "--ignore-eos")
			{
				options.stopAtEos = false;
			}
			else if (arg == "--stateful")
			{
				options.statefulDecode = true;
			}
			else if (arg == "--functional")
			{
				options.statefulDecode = false;
			}
			else if (arg == "--stream-tokens")
			{
				options.streamTokens = true;
			}
			else if (arg == "--stream-stats")
			{
				options.streamStats = true;
			}
			else if (arg == "--compile-only")
			{
				options.compileOnly = true;
			}
			else if (arg == "--cpu-aot-q8k-staged-matmul")
			{
				options.enableCPUAOTQ8KStagedMatMul = true;
			}
			else if (arg == "--cpu-aot-threads")
			{
				options.cpuAOTThreadCount = ParseSize(requireValue(arg), "cpu-aot-threads", true);
			}
			else if (arg == "--cpu-aot-parallel-min-flops")
			{
				options.cpuAOTParallelMinFlops = ParseU64(requireValue(arg), "cpu-aot-parallel-min-flops");
			}
			else if (arg == "--cpu-aot-llvm-opt-level")
			{
				const auto optLevel = ParseU64(requireValue(arg), "cpu-aot-llvm-opt-level");
				if (optLevel > 3)
				{
					throw std::runtime_error("cpu-aot-llvm-opt-level must be between 0 and 3");
				}
				options.cpuAOTLLVMOptLevel = static_cast<std::uint8_t>(optLevel);
			}
			else if (arg == "--cpu-aot-affinity")
			{
				const auto value = requireValue(arg);
				if (value == "none")
				{
					options.cpuAOTAffinityPolicy = "none";
				}
				else if (value == "compact")
				{
					options.cpuAOTAffinityPolicy = "compact";
				}
				else
				{
					throw std::runtime_error("cpu-aot-affinity must be 'none' or 'compact'");
				}
			}
			else if (arg == "--compile-diagnostics")
			{
				options.enableCompileDiagnostics = true;
			}
			else if (arg == "--no-compile-diagnostics")
			{
				options.enableCompileDiagnostics = false;
			}
			else if (arg == "--max-cache-length")
			{
				options.maxCacheLength = ParseSize(requireValue(arg), "max-cache-length");
			}
			else if (arg == "--chat-template")
			{
				options.applyChatTemplate = true;
			}
			else if (!arg.starts_with("--") && !options.outputPath)
			{
				options.outputPath = std::string(arg);
			}
			else
			{
				throw std::runtime_error(std::format("Unknown decode-loop option '{}'", arg));
			}
		}
	}

	DecodeLoopCommandOptions ParseDecodeLoopOptions(int argc, char** argv)
	{
		if (argc < 5)
		{
			throw std::runtime_error("--run-llama-decode-loop-token-id requires input, initial-token-id, and steps");
		}
		DecodeLoopCommandOptions options{
			.inputPath = argv[2],
			.initialTokenIds = { ParseTokenId(argv[3], "initial-token-id") },
			.steps = ParseSize(argv[4], "steps"),
		};
		ParseDecodeLoopTrailingOptions(argc, argv, 5, options);
		return options;
	}

	DecodeLoopCommandOptions ParseTokenIdsDecodeLoopOptions(int argc, char** argv)
	{
		if (argc < 5)
		{
			throw std::runtime_error("--run-llama-decode-loop-token-ids requires input, comma-token-ids, and steps");
		}
		DecodeLoopCommandOptions options{
			.inputPath = argv[2],
			.initialTokenIds = ParseTokenIds(argv[3]),
			.steps = ParseSize(argv[4], "steps"),
		};
		ParseDecodeLoopTrailingOptions(argc, argv, 5, options);
		return options;
	}

	DecodeLoopCommandOptions ParsePromptDecodeLoopOptions(int argc, char** argv)
	{
		if (argc < 5)
		{
			throw std::runtime_error("--run-llama-prompt-decode-loop requires input, prompt, and steps");
		}
		DecodeLoopCommandOptions options{
			.inputPath = argv[2],
			.exactPrompt = argv[3],
			.steps = ParseSize(argv[4], "steps"),
		};
		ParseDecodeLoopTrailingOptions(argc, argv, 5, options);
		return options;
	}

	LiteNN::GGUF::LLaMACompatibilityProfileKind InferLLMProfile(const LiteNN::Graph& archive)
	{
		const auto hyperparameters = LiteNN::GGUF::ParseLLaMAHyperparameters(archive);
		if (const auto profile = LiteNN::GGUF::TryInferLLaMACompatibilityProfile(hyperparameters.architecture))
		{
			return *profile;
		}
		throw std::runtime_error(std::format(
		    "No default LLM compatibility profile for GGUF architecture '{}'; supported automatic profiles "
		    "currently include 'llama' and 'qwen2'. Pass an explicit profile if this architecture shares one "
		    "of the supported contracts.",
		    hyperparameters.architecture));
	}

	struct AnalyzeLLMCommandOptions
	{
		std::string inputPath;
		std::optional<LiteNN::GGUF::LLaMACompatibilityProfileKind> profile;
		std::size_t dequantizedBudgetBytes{};
	};

	std::size_t ParseMibAsBytes(std::string_view text)
	{
		const auto mib = ParseSize(text, "dequantized-budget-mib", true);
		constexpr std::size_t bytesPerMib = 1024 * 1024;
		if (mib > std::numeric_limits<std::size_t>::max() / bytesPerMib)
		{
			throw std::runtime_error("dequantized-budget-mib is too large");
		}
		return mib * bytesPerMib;
	}

	AnalyzeLLMCommandOptions ParseAnalyzeLLMOptions(int argc, char** argv)
	{
		if (argc < 3)
		{
			throw std::runtime_error("--analyze-llm requires an input GGUF path");
		}
		AnalyzeLLMCommandOptions options{
			.inputPath = argv[2],
		};
		for (int i = 3; i < argc; ++i)
		{
			const std::string_view arg = argv[i];
			if (arg == "--dequantized-budget-bytes" || arg == "--dequantized-budget-mib")
			{
				if (i + 1 >= argc)
				{
					throw std::runtime_error(std::string(arg) + " requires a value");
				}
				++i;
				options.dequantizedBudgetBytes = arg == "--dequantized-budget-bytes"
				                                     ? ParseSize(argv[i], "dequantized-budget-bytes", true)
				                                     : ParseMibAsBytes(argv[i]);
				continue;
			}
			if (arg.starts_with("--"))
			{
				throw std::runtime_error(std::format("Unknown --analyze-llm option '{}'", arg));
			}
			if (options.profile)
			{
				throw std::runtime_error("--analyze-llm accepts at most one profile argument");
			}
			options.profile = ParseLLMProfile(arg);
		}
		return options;
	}

	void PrintLLMCompatibilityReport(const LiteNN::GGUF::LLaMACompatibilityReport& report)
	{
		std::cout << "LLM compatibility profile=" << report.profile.name
		          << " architecture=" << report.profile.architecture
		          << " lowerable=" << (report.lowerable ? "true" : "false")
		          << " external_golden_required=" << (report.externalGoldenRequired ? "true" : "false") << '\n';
		for (const auto& diagnostic : report.diagnostics)
		{
			std::cout << (diagnostic.blocking ? "blocking" : "note") << " subject=" << diagnostic.subject << ": "
			          << diagnostic.message << '\n';
		}
	}

	void PrintStringList(std::span<const std::string> values, std::size_t maxItems = 12)
	{
		std::cout << '[';
		const auto shown = std::min(values.size(), maxItems);
		for (std::size_t i = 0; i < shown; ++i)
		{
			if (i != 0)
			{
				std::cout << ',';
			}
			std::cout << values[i];
		}
		if (shown < values.size())
		{
			if (shown != 0)
			{
				std::cout << ',';
			}
			std::cout << "...+" << (values.size() - shown) << " more";
		}
		std::cout << ']';
	}

	void PrintSizeList(std::span<const std::size_t> values)
	{
		std::cout << '[';
		for (std::size_t i = 0; i < values.size(); ++i)
		{
			if (i != 0)
			{
				std::cout << ',';
			}
			std::cout << values[i];
		}
		std::cout << ']';
	}

	std::string TokenListText(std::span<const std::int32_t> values)
	{
		std::string text;
		text += '[';
		for (std::size_t i = 0; i < values.size(); ++i)
		{
			if (i != 0)
			{
				text += ',';
			}
			text += std::to_string(values[i]);
		}
		text += ']';
		return text;
	}

	void PrintTokenList(std::span<const std::int32_t> values)
	{
		std::cout << TokenListText(values);
	}

	std::string EscapeTokenPiece(std::string_view value)
	{
		std::string escaped;
		escaped.reserve(value.size() + 2);
		escaped += '"';
		for (const char c : value)
		{
			switch (c)
			{
			case '\\':
				escaped += "\\\\";
				break;
			case '"':
				escaped += "\\\"";
				break;
			case '\n':
				escaped += "\\n";
				break;
			case '\r':
				escaped += "\\r";
				break;
			case '\t':
				escaped += "\\t";
				break;
			default:
				escaped += c;
				break;
			}
		}
		escaped += '"';
		return escaped;
	}

	std::string TokenPieceText(const LiteNN::Graph& archive, std::int32_t tokenId)
	{
		const auto* tokens = archive.FindMetadata("tokenizer.ggml.tokens");
		const auto* tokenList = tokens == nullptr ? nullptr : std::get_if<std::vector<std::string>>(&tokens->value);
		if (tokenList == nullptr)
		{
			return "<missing-tokenizer.ggml.tokens>";
		}
		if (tokenId < 0 || static_cast<std::size_t>(tokenId) >= tokenList->size())
		{
			return "<out-of-range>";
		}
		return (*tokenList)[static_cast<std::size_t>(tokenId)];
	}

	std::string TokenPieceText(std::span<const std::string> tokenPieces, std::int32_t tokenId)
	{
		if (tokenPieces.empty())
		{
			return "<missing-tokenizer.ggml.tokens>";
		}
		if (tokenId < 0 || static_cast<std::size_t>(tokenId) >= tokenPieces.size())
		{
			return "<out-of-range>";
		}
		return tokenPieces[static_cast<std::size_t>(tokenId)];
	}

	std::string TokenPiecesText(const LiteNN::Graph& archive, std::span<const std::int32_t> tokenIds)
	{
		std::string text = "[";
		for (std::size_t i = 0; i < tokenIds.size(); ++i)
		{
			if (i != 0)
			{
				text += ',';
			}
			text += EscapeTokenPiece(TokenPieceText(archive, tokenIds[i]));
		}
		text += "]";
		return text;
	}

	std::string TokenPiecesText(std::span<const std::string> tokenPieces, std::span<const std::int32_t> tokenIds)
	{
		std::string text = "[";
		for (std::size_t i = 0; i < tokenIds.size(); ++i)
		{
			if (i != 0)
			{
				text += ',';
			}
			text += EscapeTokenPiece(TokenPieceText(tokenPieces, tokenIds[i]));
		}
		text += "]";
		return text;
	}

	std::vector<std::string> CopyTokenPieces(const LiteNN::Graph& archive)
	{
		const auto* tokens = archive.FindMetadata("tokenizer.ggml.tokens");
		const auto* tokenList = tokens == nullptr ? nullptr : std::get_if<std::vector<std::string>>(&tokens->value);
		if (tokenList == nullptr)
		{
			return {};
		}
		return *tokenList;
	}

	std::vector<std::int32_t> TokenizePrompt(const std::filesystem::path& modelPath, std::string_view prompt,
	                                         const LiteNN::Graph& archive, bool applyChatTemplate)
	{
#ifdef LITENN_GGUF_CONVERT_ENABLE_LLAMA_CPP_TOKENIZER
		const LiteNN::LlamaCppAdapter::Model model(modelPath);
		const auto promptText = applyChatTemplate ? model.ApplyChatTemplate(prompt) : std::string(prompt);
		return model.Tokenize(promptText).tokenIds;
#else
		(void) modelPath;
		if (applyChatTemplate)
		{
			throw std::runtime_error(
			    "llama.cpp chat-template tokenization requires LITENN_ENABLE_LLAMA_CPP_TOKENIZER=ON");
		}
		return LiteNN::GGUF::MakeExactVocabularyPromptTokens(prompt, archive).tokenIds;
#endif
	}

	void WritePromptTokens(const std::filesystem::path& modelPath, std::string_view prompt,
	                       const std::filesystem::path& outputPath, bool applyChatTemplate)
	{
#ifdef LITENN_GGUF_CONVERT_ENABLE_LLAMA_CPP_TOKENIZER
		const LiteNN::LlamaCppAdapter::Model model(modelPath);
		const auto promptText = applyChatTemplate ? model.ApplyChatTemplate(prompt) : std::string(prompt);
		LiteNN::LlamaCppAdapter::WriteTokensJson(model.Tokenize(promptText), outputPath);
#else
		(void) modelPath;
		(void) prompt;
		(void) outputPath;
		(void) applyChatTemplate;
		throw std::runtime_error("llama.cpp prompt tokenization requires LITENN_ENABLE_LLAMA_CPP_TOKENIZER=ON");
#endif
	}

	void PrintTensorShape(LiteNN::ShapeView shape)
	{
		std::cout << '[';
		for (std::size_t i = 0; i < shape.NumDim(); ++i)
		{
			if (i != 0)
			{
				std::cout << ',';
			}
			std::cout << shape[i];
		}
		std::cout << ']';
	}

	void WriteLastTokenLogitsText(const LiteNN::Tensor<LiteNN::CPU>& logits, std::string_view outputPath)
	{
		const auto lastTokenLogits = LiteNN::GGUF::ExtractLastTokenLogits(logits);
		std::ofstream output(std::string(outputPath), std::ios::binary);
		if (!output)
		{
			throw std::runtime_error("Failed to open logits output file: " + std::string(outputPath));
		}
		output << std::setprecision(9);
		for (std::size_t i = 0; i < lastTokenLogits.size(); ++i)
		{
			output << i << ": " << lastTokenLogits[i] << '\n';
		}
		if (!output)
		{
			throw std::runtime_error("Failed to write logits output file: " + std::string(outputPath));
		}
	}

	void PrintLLMArtifactPlan(const LiteNN::GGUF::LLaMAArtifactPlan& plan)
	{
		std::cout << "LLM artifact plan architecture=" << plan.hyperparameters.architecture
		          << " dtype=" << LiteNN::DataTypeName(plan.dtype) << " vocab=" << plan.vocabSize
		          << " blocks=" << plan.hyperparameters.blockCount << '\n';
		const auto printEntry = [](const LiteNN::GGUF::LLaMAArtifactEntry& entry) {
			std::cout << "entry name=" << entry.name << " sequence_length=" << entry.sequenceLength
			          << " past_length=" << entry.pastLength << " max_cache_length=" << entry.maxCacheLength
			          << " dynamic_position=" << (entry.dynamicPosition ? "true" : "false") << " inputs=";
			PrintStringList(entry.inputNames);
			std::cout << " outputs=";
			PrintStringList(entry.outputNames);
			std::cout << " kv_caches=" << entry.kvCaches.size() << '\n';
		};
		printEntry(plan.prefill);
		printEntry(plan.decodeStep);
		if (!plan.decodeStep.kvCaches.empty())
		{
			const auto& firstCache = plan.decodeStep.kvCaches.front();
			std::cout << "kv_cache sample name=" << firstCache.stateBinding.name
			          << " dtype=" << LiteNN::DataTypeName(firstCache.stateType.dtype) << " state_shape=";
			const auto stateShape = firstCache.stateType.StaticShape();
			PrintSizeList(stateShape);
			std::cout << " key_offset=" << firstCache.keyByteOffset << " value_offset=" << firstCache.valueByteOffset
			          << " layer_stride=" << firstCache.layerByteStride
			          << " token_stride=" << firstCache.tokenByteStride << '\n';
		}
		for (const auto& layout : plan.tensorLayouts)
		{
			std::cout << "layout name=" << layout.name << " domain=" << layout.domain << " axes=";
			PrintStringList(layout.axes);
			std::cout << " layout=" << layout.layout << '\n';
		}
	}

	LiteNN::Tensor<LiteNN::CPU> MakeTokenIdTensor(std::span<const std::int32_t> tokenIds,
	                                              const LiteNN::ExecutablePlan& plan)
	{
		if (plan.inputs.empty())
		{
			throw std::runtime_error("LLM package has no inputs");
		}
		const auto& input = plan.inputs.front();
		if (input.type.dtype != LiteNN::DataType::Int32)
		{
			throw std::runtime_error(std::format("LLM package first input must be Int32 token ids, got {}",
			                                     LiteNN::DataTypeName(input.type.dtype)));
		}
		if (!input.type.IsFullyStatic())
		{
			throw std::runtime_error("LLM package first input must have a static shape");
		}
		const auto shape = input.type.StaticShape();
		const auto expected = LiteNN::Detail::Product(shape);
		if (expected != tokenIds.size())
		{
			throw std::runtime_error(
			    std::format("LLM token id count mismatch: package expects {}, got {}", expected, tokenIds.size()));
		}
		LiteNN::CPU cpu;
		LiteNN::Tensor<LiteNN::CPU> tensor(LiteNN::Uninitialized, shape, LiteNN::DataType::Int32, cpu);
		LiteNN::DeviceTraits<LiteNN::CPU>::CopyFromCPU(cpu, LiteNN::DataType::Int32, tensor.UnsafeRawData(),
		                                               LiteNN::DataType::Int32, tokenIds.data(), tokenIds.size());
		return tensor;
	}

	std::vector<LiteNN::Tensor<LiteNN::CPU>> MakeZeroStateInputs(const LiteNN::ExecutablePlan& plan,
	                                                             LiteNN::Tensor<LiteNN::CPU> tokenIds)
	{
		std::vector<LiteNN::Tensor<LiteNN::CPU>> inputs;
		inputs.push_back(std::move(tokenIds));
		LiteNN::CPU cpu;
		for (std::size_t i = 1; i < plan.inputs.size(); ++i)
		{
			const auto& input = plan.inputs[i];
			if (!input.type.IsFullyStatic())
			{
				throw std::runtime_error(std::format("LLM package input {} must have a static shape", i));
			}
			inputs.emplace_back(input.type.StaticShape(), input.type.dtype, cpu);
		}
		return inputs;
	}

	LiteNN::Tensor<LiteNN::CPU> MakeTokenIdTensorForPlan(std::int32_t tokenId, const LiteNN::ExecutablePlan& plan)
	{
		const std::array<std::int32_t, 1> ids{ tokenId };
		return MakeTokenIdTensor(ids, plan);
	}

	LiteNN::Tensor<LiteNN::CPU> MakeDecodePositionTensor(std::int64_t position, const LiteNN::ExecutablePlan& plan)
	{
		if (plan.inputs.size() < 2 || plan.inputs[1].type.dtype != LiteNN::DataType::Int64 ||
		    plan.inputs[1].type.StaticShape() != std::vector<std::size_t>{ 1 })
		{
			throw std::runtime_error("Capacity decode plan requires Int64 current_position input with shape [1]");
		}
		LiteNN::CPU cpu;
		LiteNN::Tensor<LiteNN::CPU> tensor(LiteNN::Uninitialized, { 1 }, LiteNN::DataType::Int64, cpu);
		LiteNN::DeviceTraits<LiteNN::CPU>::CopyFromCPU(cpu, LiteNN::DataType::Int64, tensor.UnsafeRawData(),
		                                               LiteNN::DataType::Int64, &position, 1);
		return tensor;
	}

	void StoreScalarTokenId(LiteNN::Tensor<LiteNN::CPU>& tensor, std::int32_t tokenId)
	{
		if (tensor.DType() != LiteNN::DataType::Int32 || tensor.NumElements() != 1)
		{
			throw std::runtime_error("decode-loop stateful token input must be Int32 with one element");
		}
		LiteNN::CPU cpu;
		LiteNN::DeviceTraits<LiteNN::CPU>::CopyFromCPU(cpu, LiteNN::DataType::Int32, tensor.UnsafeRawData(),
		                                               LiteNN::DataType::Int32, &tokenId, 1);
	}

#ifdef LITENN_GGUF_CONVERT_ENABLE_AOT
	LiteNN::CompilerOptions CompilerOptionsFromEnvironment();
	void ApplyDecodeLoopCompilerOptions(const DecodeLoopCommandOptions& decodeOptions,
	                                    LiteNN::CompilerOptions& compilerOptions);

	void LogGGUFDiagnostic(bool enabled, std::string_view message)
	{
		if (enabled)
		{
			std::cerr << "[LiteNN gguf] " << message << '\n' << std::flush;
		}
	}

	template <class F>
	decltype(auto) TimedGGUFDiagnostic(bool enabled, std::string_view label, F&& f)
	{
		LogGGUFDiagnostic(enabled, std::format("{}...", label));
		const auto start = std::chrono::steady_clock::now();
		if constexpr (std::is_void_v<std::invoke_result_t<F>>)
		{
			std::forward<F>(f)();
			const auto end = std::chrono::steady_clock::now();
			LogGGUFDiagnostic(enabled, std::format("{}: ok {:.3f} ms", label,
			                                       std::chrono::duration<double, std::milli>(end - start).count()));
		}
		else
		{
			auto result = std::forward<F>(f)();
			const auto end = std::chrono::steady_clock::now();
			LogGGUFDiagnostic(enabled, std::format("{}: ok {:.3f} ms", label,
			                                       std::chrono::duration<double, std::milli>(end - start).count()));
			return result;
		}
	}

	void LogGGUFHelperProfile(bool enabled, std::size_t step,
	                          std::span<const LiteNN::CompiledModuleCPUHelperProfileEvent> events)
	{
		if (!enabled || events.empty())
		{
			return;
		}
		double totalMs = 0.0;
		std::uint64_t totalCalls = 0;
		for (const auto& event : events)
		{
			totalMs += event.totalMilliseconds;
			totalCalls += event.calls;
		}
		LogGGUFDiagnostic(enabled, std::format("decode step {} helper_profile total_ms={:.3f} calls={} helpers={}",
		                                       step, totalMs, totalCalls, events.size()));
		for (const auto& event : events)
		{
			const auto averageMs = event.calls == 0 ? 0.0 : event.totalMilliseconds / static_cast<double>(event.calls);
			const auto detail = event.detail.empty() ? std::string{} : std::format(" detail=\"{}\"", event.detail);
			LogGGUFDiagnostic(enabled,
			                  std::format("decode step {} helper {}{} calls={} total_ms={:.3f} avg_ms={:.6f}", step,
			                              event.helper, detail, event.calls, event.totalMilliseconds, averageMs));
		}
	}

	std::vector<std::byte> ReadBinaryFile(const std::filesystem::path& path)
	{
		std::ifstream input(path, std::ios::binary);
		if (!input)
		{
			throw std::runtime_error("failed to open cached artifact file: " + path.string());
		}
		input.seekg(0, std::ios::end);
		const auto size = input.tellg();
		if (size < 0)
		{
			throw std::runtime_error("failed to size cached artifact file: " + path.string());
		}
		input.seekg(0, std::ios::beg);
		std::vector<std::byte> bytes(static_cast<std::size_t>(size));
		constexpr std::size_t kChunkBytes = 64ull * 1024ull * 1024ull;
		std::size_t offset = 0;
		while (offset < bytes.size())
		{
			const auto remaining = bytes.size() - offset;
			const auto chunk = std::min(remaining, kChunkBytes);
			input.read(reinterpret_cast<char*>(bytes.data() + offset), static_cast<std::streamsize>(chunk));
			if (input.gcount() != static_cast<std::streamsize>(chunk))
			{
				throw std::runtime_error("failed to read cached artifact file: " + path.string());
			}
			offset += chunk;
		}
		if (!input && offset != bytes.size())
		{
			throw std::runtime_error("failed to read cached artifact file: " + path.string());
		}
		return bytes;
	}

	std::string ReadTextFile(const std::filesystem::path& path)
	{
		std::ifstream input(path, std::ios::binary);
		if (!input)
		{
			throw std::runtime_error("failed to open cached artifact text file: " + path.string());
		}
		std::string text((std::istreambuf_iterator<char>(input)), std::istreambuf_iterator<char>());
		while (!text.empty() &&
		       (text.back() == '\n' || text.back() == '\r' || text.back() == ' ' || text.back() == '\t'))
		{
			text.pop_back();
		}
		return text;
	}

	void WriteBinaryFileTimed(const std::filesystem::path& path, std::span<const std::byte> bytes, bool diagnostics,
	                          std::string_view label)
	{
		LogGGUFDiagnostic(diagnostics, std::format("{} bytes={} path={}", label, bytes.size(), path.generic_string()));
		TimedGGUFDiagnostic(diagnostics, label, [&] {
			std::ofstream output(path, std::ios::binary);
			if (!output)
			{
				throw std::runtime_error("failed to open cached artifact file for write: " + path.string());
			}

			constexpr std::size_t kChunkBytes = 64ull * 1024ull * 1024ull;
			constexpr std::size_t kProgressBytes = 512ull * 1024ull * 1024ull;
			std::size_t offset = 0;
			std::size_t nextProgress = kProgressBytes;
			while (offset < bytes.size())
			{
				const auto remaining = bytes.size() - offset;
				const auto chunk = std::min(remaining, kChunkBytes);
				output.write(reinterpret_cast<const char*>(bytes.data() + offset), static_cast<std::streamsize>(chunk));
				if (!output)
				{
					throw std::runtime_error("failed to write cached artifact file: " + path.string());
				}
				offset += chunk;
				if (diagnostics && (offset >= nextProgress || offset == bytes.size()))
				{
					LogGGUFDiagnostic(diagnostics, std::format("{} progress {}/{} bytes", label, offset, bytes.size()));
					while (nextProgress <= offset)
					{
						nextProgress += kProgressBytes;
					}
				}
			}
		});
	}

	void WriteTextFileTimed(const std::filesystem::path& path, std::string_view text, bool diagnostics,
	                        std::string_view label)
	{
		const auto bytes = std::as_bytes(std::span<const char>(text.data(), text.size()));
		WriteBinaryFileTimed(path, bytes, diagnostics, label);
	}

	std::uint64_t FNV1a(std::string_view text)
	{
		std::uint64_t hash = 14695981039346656037ull;
		for (const unsigned char ch : text)
		{
			hash ^= ch;
			hash *= 1099511628211ull;
		}
		return hash;
	}

	std::optional<std::filesystem::path> DecodeAOTCachePath(std::string_view modelPath, std::size_t requestedTokenCount,
	                                                        const LiteNN::CompilerOptions& options,
	                                                        std::string_view decodeMode)
	{
		const char* root = std::getenv("LITENN_GGUF_AOT_CACHE_DIR");
		if (root == nullptr || std::string_view(root).empty())
		{
			return std::nullopt;
		}
		const std::filesystem::path model(modelPath);
		std::error_code ec;
		const auto modelSize = std::filesystem::file_size(model, ec);
		const auto lastWrite = std::filesystem::last_write_time(model, ec).time_since_epoch().count();
		const auto keyText =
		    std::format("gguf-decode-{}-v4|{}|{}|{}|tokens={}|opt={}|external={}|threads={}|affinity={}|min_flops={}|"
		                "q8k_staged={}",
		                decodeMode, std::filesystem::absolute(model, ec).string(), modelSize, lastWrite,
		                requestedTokenCount, options.cpuAOTLLVMOptLevel, options.enableCPUAOTExternalRegions ? 1 : 0,
		                options.cpuAOTThreadCount, static_cast<std::uint32_t>(options.cpuAOTAffinityPolicy),
		                options.cpuAOTParallelMinFlops, options.enableCPUAOTGGMLQ8KStagedMatMul ? 1 : 0);
		return std::filesystem::path(root) / std::format("{:016x}", FNV1a(keyText));
	}

	std::optional<std::filesystem::path> DecodeAOTSharedWeightsPath(std::string_view modelPath)
	{
		const char* root = std::getenv("LITENN_GGUF_AOT_CACHE_DIR");
		if (root == nullptr || std::string_view(root).empty())
		{
			return std::nullopt;
		}
		const std::filesystem::path model(modelPath);
		std::error_code ec;
		const auto modelSize = std::filesystem::file_size(model, ec);
		const auto lastWrite = std::filesystem::last_write_time(model, ec).time_since_epoch().count();
		const auto keyText = std::format("gguf-shared-weights-v1|{}|{}|{}",
		                                 std::filesystem::absolute(model, ec).string(), modelSize, lastWrite);
		return std::filesystem::path(root) / "_weights" / std::format("{:016x}", FNV1a(keyText)) / "weights.bin";
	}

	struct DecodeAOTCacheFiles
	{
		std::filesystem::path metadata;
		std::filesystem::path constants;
		std::filesystem::path weights;
		std::filesystem::path weightReference;
		std::filesystem::path instructions;
		std::filesystem::path complete;
	};

	DecodeAOTCacheFiles DecodeAOTCacheFilesFor(const std::filesystem::path& cachePath)
	{
		return {
			.metadata = cachePath / "metadata.bin",
			.constants = cachePath / "constants.bin",
			.weights = cachePath / "weights.bin",
			.weightReference = cachePath / "weights.path.txt",
			.instructions = cachePath / "instructions.bin",
			.complete = cachePath / "complete",
		};
	}

	std::optional<std::filesystem::path> DecodeAOTReferencedWeightsPath(const DecodeAOTCacheFiles& files)
	{
		if (!std::filesystem::exists(files.weightReference))
		{
			return std::nullopt;
		}
		const auto text = ReadTextFile(files.weightReference);
		if (text.empty())
		{
			throw std::runtime_error("gguf decode aot cache weights reference is empty");
		}
		return std::filesystem::path(text);
	}

	bool DecodeAOTWeightsAvailable(const DecodeAOTCacheFiles& files)
	{
		if (std::filesystem::exists(files.weightReference))
		{
			const auto referenced = DecodeAOTReferencedWeightsPath(files);
			if (!referenced)
			{
				return false;
			}
			return std::filesystem::exists(*referenced) &&
			       std::filesystem::exists(referenced->parent_path() / "complete");
		}
		return std::filesystem::exists(files.weights);
	}

	bool DecodeAOTCacheComplete(const std::filesystem::path& cachePath)
	{
		const auto files = DecodeAOTCacheFilesFor(cachePath);
		return std::filesystem::exists(files.metadata) && std::filesystem::exists(files.constants) &&
		       std::filesystem::exists(files.instructions) && std::filesystem::exists(files.complete) &&
		       DecodeAOTWeightsAvailable(files);
	}

	bool RequireDecodeAOTCacheHit()
	{
		const char* value = std::getenv("LITENN_GGUF_AOT_CACHE_REQUIRE_HIT");
		if (value == nullptr)
		{
			return false;
		}
		const std::string_view text{ value };
		return text == "1" || text == "true" || text == "TRUE" || text == "on" || text == "ON";
	}

	bool DecodeAOTCacheWriteEnabled()
	{
		const char* value = std::getenv("LITENN_GGUF_AOT_CACHE_WRITE");
		if (value == nullptr)
		{
			return true;
		}
		std::string text{ value };
		std::ranges::transform(text, text.begin(),
		                       [](unsigned char ch) { return static_cast<char>(std::tolower(ch)); });
		return !(text == "0" || text == "false" || text == "off" || text == "no");
	}

	void ThrowDecodeAOTCacheMiss(const std::optional<std::filesystem::path>& cachePath)
	{
		if (RequireDecodeAOTCacheHit())
		{
			if (!cachePath)
			{
				throw std::runtime_error(
				    "gguf decode aot cache hit is required but LITENN_GGUF_AOT_CACHE_DIR is not set");
			}
			throw std::runtime_error("gguf decode aot cache hit is required but cache is missing or invalid: " +
			                         cachePath->string());
		}
	}

	LiteNN::CompiledModule<LiteNN::CPU> LoadDecodeAOTCache(const std::filesystem::path& cachePath, bool diagnostics)
	{
		auto artifact = TimedGGUFDiagnostic(diagnostics, "gguf decode aot cache read separated artifact", [&] {
			const auto files = DecodeAOTCacheFilesFor(cachePath);
			std::vector<std::byte> weights;
			if (auto referencedWeights = DecodeAOTReferencedWeightsPath(files))
			{
				LogGGUFDiagnostic(diagnostics,
				                  "gguf decode aot cache: using shared weights " + referencedWeights->generic_string());
				weights = ReadBinaryFile(*referencedWeights);
			}
			else
			{
				weights = ReadBinaryFile(files.weights);
			}
			return LiteNN::CompiledModuleSeparatedArtifact::FromOwnedRegions(
			    ReadBinaryFile(files.metadata), ReadBinaryFile(files.constants), std::move(weights),
			    ReadBinaryFile(files.instructions));
		});
		LogGGUFDiagnostic(diagnostics, "gguf decode aot cache: hit");
		return TimedGGUFDiagnostic(diagnostics, "gguf decode aot cache load module",
		                           [&] { return std::move(artifact).LoadBorrowedExternalRegions(); });
	}

	void WriteDecodeAOTCache(const std::filesystem::path& cachePath, const LiteNN::CompiledModuleArtifact& artifact,
	                         const std::optional<std::filesystem::path>& sharedWeightsPath, bool diagnostics)
	{
		if (!DecodeAOTCacheWriteEnabled())
		{
			LogGGUFDiagnostic(diagnostics, "gguf decode aot cache: write skipped by LITENN_GGUF_AOT_CACHE_WRITE=0");
			return;
		}

		std::filesystem::create_directories(cachePath);
		const auto files = DecodeAOTCacheFilesFor(cachePath);
		auto metadata = TimedGGUFDiagnostic(diagnostics, "gguf decode aot cache build metadata",
		                                    [&] { return artifact.BuildSeparatedMetadata(); });
		LogGGUFDiagnostic(diagnostics, std::format("gguf decode aot cache regions: metadata={} constants={} weights={} "
		                                           "instructions={}",
		                                           metadata.size(), artifact.Constants().size(),
		                                           artifact.Weights().size(), artifact.Instructions().size()));
		WriteBinaryFileTimed(files.metadata, metadata, diagnostics, "gguf decode aot cache write metadata");
		WriteBinaryFileTimed(files.constants, artifact.Constants(), diagnostics,
		                     "gguf decode aot cache write constants");
		if (sharedWeightsPath && !artifact.Weights().empty())
		{
			const auto sharedComplete = sharedWeightsPath->parent_path() / "complete";
			const auto sharedSizeMatches = std::filesystem::exists(*sharedWeightsPath) &&
			                               std::filesystem::file_size(*sharedWeightsPath) == artifact.Weights().size();
			if (!sharedSizeMatches || !std::filesystem::exists(sharedComplete))
			{
				std::filesystem::create_directories(sharedWeightsPath->parent_path());
				WriteBinaryFileTimed(*sharedWeightsPath, artifact.Weights(), diagnostics,
				                     "gguf decode aot shared weight store write weights");
				WriteBinaryFileTimed(sharedComplete, std::span<const std::byte>{}, diagnostics,
				                     "gguf decode aot shared weight store write complete marker");
			}
			else
			{
				LogGGUFDiagnostic(diagnostics,
				                  "gguf decode aot shared weight store: reused " + sharedWeightsPath->generic_string());
			}
			WriteTextFileTimed(files.weightReference, std::filesystem::absolute(*sharedWeightsPath).string(),
			                   diagnostics, "gguf decode aot cache write weights reference");
		}
		else
		{
			WriteBinaryFileTimed(files.weights, artifact.Weights(), diagnostics, "gguf decode aot cache write weights");
		}
		WriteBinaryFileTimed(files.instructions, artifact.Instructions(), diagnostics,
		                     "gguf decode aot cache write instructions");
		WriteBinaryFileTimed(files.complete, std::span<const std::byte>{}, diagnostics,
		                     "gguf decode aot cache write complete marker");
		LogGGUFDiagnostic(diagnostics, std::format("gguf decode aot cache: wrote {}", cachePath.string()));
	}

	LiteNN::CompiledModule<LiteNN::CPU>
	LoadOrCompileDecodeModule(const LiteNN::ExecutablePlan& plan, const LiteNN::CompilerOptions& options,
	                          const std::optional<std::filesystem::path>& cachePath,
	                          const std::optional<std::filesystem::path>& sharedWeightsPath, bool diagnostics)
	{
		if (cachePath)
		{
			if (DecodeAOTCacheComplete(*cachePath))
			{
				try
				{
					return LoadDecodeAOTCache(*cachePath, diagnostics);
				}
				catch (const std::exception& ex)
				{
					LogGGUFDiagnostic(diagnostics,
					                  std::format("gguf decode aot cache: ignored invalid cache ({})", ex.what()));
				}
			}
			else
			{
				LogGGUFDiagnostic(diagnostics, "gguf decode aot cache: miss");
			}
		}
		ThrowDecodeAOTCacheMiss(cachePath);

		auto artifact = TimedGGUFDiagnostic(diagnostics, "gguf compile cpu aot decode artifact", [&] {
			return LiteNN::Compiler<LiteNN::CPU>::CompileArtifact(plan, options);
		});
		if (cachePath)
		{
			try
			{
				WriteDecodeAOTCache(*cachePath, artifact, sharedWeightsPath, diagnostics);
			}
			catch (const std::exception& ex)
			{
				LogGGUFDiagnostic(diagnostics, std::format("gguf decode aot cache: write failed ({})", ex.what()));
			}
		}
		return TimedGGUFDiagnostic(diagnostics, "gguf load freshly compiled cpu aot decode module",
		                           [&] { return std::move(artifact).Load(); });
	}

	LiteNN::CompiledModule<LiteNN::CPU>
	LoadOrCompileDecodeModule(const LiteNN::Runtime::RuntimeSchedule& schedule, const LiteNN::CompilerOptions& options,
	                          const std::optional<std::filesystem::path>& cachePath,
	                          const std::optional<std::filesystem::path>& sharedWeightsPath, bool diagnostics)
	{
		if (cachePath)
		{
			if (DecodeAOTCacheComplete(*cachePath))
			{
				try
				{
					return LoadDecodeAOTCache(*cachePath, diagnostics);
				}
				catch (const std::exception& ex)
				{
					LogGGUFDiagnostic(diagnostics,
					                  std::format("gguf decode aot cache: ignored invalid cache ({})", ex.what()));
				}
			}
			else
			{
				LogGGUFDiagnostic(diagnostics, "gguf decode aot cache: miss");
			}
		}
		ThrowDecodeAOTCacheMiss(cachePath);

		auto artifact = TimedGGUFDiagnostic(diagnostics, "gguf compile cpu aot stateful decode artifact", [&] {
			return LiteNN::Compiler<LiteNN::CPU>::CompileArtifact(schedule, options);
		});
		if (cachePath)
		{
			try
			{
				WriteDecodeAOTCache(*cachePath, artifact, sharedWeightsPath, diagnostics);
			}
			catch (const std::exception& ex)
			{
				LogGGUFDiagnostic(diagnostics, std::format("gguf decode aot cache: write failed ({})", ex.what()));
			}
		}
		return TimedGGUFDiagnostic(diagnostics, "gguf load freshly compiled cpu aot stateful decode module",
		                           [&] { return std::move(artifact).Load(); });
	}
#endif

	void RunDecodeLoopFromGGUF(const DecodeLoopCommandOptions& options)
	{
#ifndef LITENN_GGUF_CONVERT_ENABLE_AOT
		(void) options;
		throw std::runtime_error(
		    "LLaMA decode execution requires the AOT compiler; configure with LITENN_ENABLE_MLIR=ON");
#else
		if (options.steps == 0)
		{
			throw std::runtime_error("decode-loop steps must be positive");
		}
		auto compilerOptions = CompilerOptionsFromEnvironment();
		compilerOptions.enableCPUAOTExternalRegions = true;
		ApplyDecodeLoopCompilerOptions(options, compilerOptions);
		const auto diagnostics = compilerOptions.enableCompileDiagnostics;
		LogGGUFDiagnostic(diagnostics, std::format("decode-loop start input={} requested_steps={}", options.inputPath,
		                                           options.steps));
		auto imported = TimedGGUFDiagnostic(diagnostics, "gguf import archive",
		                                    [&] { return LiteNN::GGUF::ImportGGUFArchive(options.inputPath); });
		const auto importSummary = imported.summary;
		auto initialTokenIds = options.initialTokenIds;
		if (options.exactPrompt)
		{
			initialTokenIds = TimedGGUFDiagnostic(diagnostics, "gguf tokenize exact prompt", [&] {
				return TokenizePrompt(options.inputPath, *options.exactPrompt, imported.model.UnsafeGraphView(),
				                      options.applyChatTemplate);
			});
		}
		if (initialTokenIds.empty())
		{
			throw std::runtime_error("decode-loop requires at least one initial token");
		}
		const auto hyperparameters = LiteNN::GGUF::ParseLLaMAHyperparameters(imported.model.UnsafeGraphView());
		const auto tokenizer = LiteNN::GGUF::SummarizeLLMTokenizerMetadata(imported.model.UnsafeGraphView());
		auto tokenPieces = CopyTokenPieces(imported.model.UnsafeGraphView());
		const auto requestedTokenCount = initialTokenIds.size() + options.steps;
		const auto maxCacheLength = options.maxCacheLength.value_or(requestedTokenCount);
		if (maxCacheLength < requestedTokenCount)
		{
			throw std::runtime_error(
			    std::format("decode-loop max-cache-length {} is smaller than requested token count {}", maxCacheLength,
			                requestedTokenCount));
		}
		if (hyperparameters.contextLength > 0 && requestedTokenCount > hyperparameters.contextLength)
		{
			throw std::runtime_error(std::format("decode-loop requested {} total tokens but model context length is {}",
			                                     requestedTokenCount, hyperparameters.contextLength));
		}
		if (hyperparameters.contextLength > 0 && maxCacheLength > hyperparameters.contextLength)
		{
			throw std::runtime_error(std::format("decode-loop max-cache-length {} exceeds model context length {}",
			                                     maxCacheLength, hyperparameters.contextLength));
		}
		LogGGUFDiagnostic(diagnostics,
		                  std::format("decode-loop tokens prompt={} generated_request={} requested_token_count={} "
		                              "max_cache_length={}",
		                              initialTokenIds.size(), options.steps, requestedTokenCount, maxCacheLength));
		const auto maxRunCount = requestedTokenCount - 1;
		const auto buildStart = std::chrono::steady_clock::now();
		const std::string_view decodeMode = options.statefulDecode ? "stateful" : "functional";
		LiteNN::ExecutablePlan decodePlan;
		LiteNN::CompiledModule<LiteNN::CPU> decodeModule = [&] {
			if (options.statefulDecode)
			{
				auto schedule = TimedGGUFDiagnostic(diagnostics, "gguf build stateful decode runtime schedule", [&] {
					return LiteNN::GGUF::BuildLLaMADecodeRuntimeSchedule(imported.model.UnsafeGraphView(),
					                                                     { .prefillSequenceLength = 1,
					                                                       .decodePastLength = 0,
					                                                       .maxCacheLength = maxCacheLength,
					                                                       .preserveQuantizedWeights = true,
					                                                       .dynamicDecodePosition = true });
				});
				decodePlan = schedule.module.plan;
				const auto projection =
				    LiteNN::Runtime::RuntimeScheduleOutputProjectionForFunction(schedule, schedule.module.plan.forward);
				LogGGUFDiagnostic(diagnostics,
				                  std::format("decode-schedule states={} bindings={} inputs={} functional_outputs={} "
				                              "public_outputs={} state_aliases={}",
				                              schedule.states.size(), schedule.stateValueBindings.size(),
				                              decodePlan.inputs.size(), projection.functionalOutputCount,
				                              projection.publicOutputIndices.size(), projection.stateAliases.size()));
				const auto cachePath =
				    DecodeAOTCachePath(options.inputPath, maxCacheLength, compilerOptions, decodeMode);
				const auto sharedWeightsPath = DecodeAOTSharedWeightsPath(options.inputPath);
				return TimedGGUFDiagnostic(diagnostics, "gguf load-or-compile cpu aot stateful decode module", [&] {
					return LoadOrCompileDecodeModule(schedule, compilerOptions, cachePath, sharedWeightsPath,
					                                 diagnostics);
				});
			}

			auto graph = TimedGGUFDiagnostic(diagnostics, "gguf lower decode-capacity graph", [&] {
				return LiteNN::GGUF::LowerLLaMACausalLMDecodeCapacity(imported.model.UnsafeGraphView(), maxCacheLength,
				                                                      { .preserveQuantizedWeights = true });
			});
			decodePlan = TimedGGUFDiagnostic(diagnostics, "gguf build executable plan",
			                                 [&] { return LiteNN::Detail::BuildExecutablePlanFromGraph(graph); });
			const auto cachePath = DecodeAOTCachePath(options.inputPath, maxCacheLength, compilerOptions, decodeMode);
			const auto sharedWeightsPath = DecodeAOTSharedWeightsPath(options.inputPath);
			return TimedGGUFDiagnostic(diagnostics, "gguf load-or-compile cpu aot decode module", [&] {
				return LoadOrCompileDecodeModule(decodePlan, compilerOptions, cachePath, sharedWeightsPath,
				                                 diagnostics);
			});
		}();
		LogGGUFDiagnostic(diagnostics,
		                  std::format("decode-plan inputs={} outputs={} variables={}", decodePlan.inputs.size(),
		                              decodePlan.outputs.size(), decodePlan.variables.size()));
		const auto buildEnd = std::chrono::steady_clock::now();
		if (options.compileOnly)
		{
			const auto buildMs = std::chrono::duration<double, std::milli>(buildEnd - buildStart).count();
			std::cout << "Compiled LLaMA decode loop tensors=" << importSummary.tensorCount
			          << " metadata=" << importSummary.metadataCount << " steps=" << options.steps
			          << " prompt_tokens=" << initialTokenIds.size() << " backend=cpu_aot decode_mode=" << decodeMode
			          << " fallback_count=0 fallback=false cached_modules=1 build_ms=" << buildMs
			          << " compile_only=true requested_token_count=" << requestedTokenCount
			          << " max_cache_length=" << maxCacheLength << " max_run_steps=" << maxRunCount
			          << " inputs=" << decodePlan.inputs.size() << " outputs=" << decodePlan.outputs.size()
			          << " variables=" << decodePlan.variables.size() << '\n';
			return;
		}
		imported = {};
		LogGGUFDiagnostic(diagnostics, "decode-loop released imported GGUF archive before token execution");

		LiteNN::GGUF::LLMSamplerState sampler{ .config = options.sampling };
		std::vector<std::int32_t> history = initialTokenIds;
		std::vector<LiteNN::Tensor<LiteNN::CPU>> caches;
		auto currentPosition = MakeDecodePositionTensor(0, decodePlan);
		std::int32_t currentToken = initialTokenIds.front();
		std::size_t lastOutputCount = 0;
		std::vector<std::size_t> lastLogitsShape;
		std::size_t generatedTokenCount = 0;
		bool stoppedOnEos = false;
		std::vector<double> stepTimesMs;
		stepTimesMs.reserve(maxRunCount);
		std::size_t promptReplayStepCount = 0;
		std::size_t generationStepCount = 0;
		double promptReplayMs = 0.0;
		double generationMs = 0.0;
		std::vector<LiteNN::Tensor<LiteNN::CPU>> statefulInputs;
		if (options.statefulDecode)
		{
			statefulInputs = MakeZeroStateInputs(decodePlan, MakeTokenIdTensorForPlan(currentToken, decodePlan));
		}
		if (options.logitsOutputDirectory)
		{
			std::filesystem::create_directories(*options.logitsOutputDirectory);
		}

		const auto runStart = std::chrono::steady_clock::now();
		LogGGUFDiagnostic(diagnostics, std::format("decode-loop run max_steps={}", maxRunCount));
		for (std::size_t step = 0; step < maxRunCount; ++step)
		{
			LogGGUFDiagnostic(diagnostics, std::format("decode step {} begin position={}", step + 1, step));
			const auto stepStart = std::chrono::steady_clock::now();
			const bool isPromptReplayStep = step + 1 < initialTokenIds.size();
			std::vector<LiteNN::Tensor<LiteNN::CPU>> outputs;
			std::optional<LiteNN::CompiledModuleCPUHelperProfiler> helperProfiler;
			if (diagnostics)
			{
				helperProfiler.emplace();
			}
			if (options.statefulDecode)
			{
				StoreScalarTokenId(statefulInputs.front(), currentToken);
				outputs = decodeModule.RunTensors(statefulInputs);
			}
			else
			{
				std::vector<LiteNN::Tensor<LiteNN::CPU>> inputs;
				inputs.push_back(MakeTokenIdTensorForPlan(currentToken, decodePlan));
				inputs.push_back(currentPosition);
				if (caches.empty())
				{
					LiteNN::CPU cpu;
					for (std::size_t i = 2; i < decodePlan.inputs.size(); ++i)
					{
						const auto& input = decodePlan.inputs[i];
						if (!input.type.IsFullyStatic())
						{
							throw std::runtime_error("decode-loop cache inputs must have static shapes");
						}
						inputs.emplace_back(input.type.StaticShape(), input.type.dtype, cpu);
					}
				}
				else
				{
					if (caches.size() + 2 != decodePlan.inputs.size())
					{
						throw std::runtime_error("decode-loop cache count does not match decode graph inputs");
					}
					for (auto& cache : caches)
					{
						inputs.push_back(std::move(cache));
					}
					caches.clear();
				}
				outputs = decodeModule.RunTensors(inputs);
			}
			if (helperProfiler)
			{
				LogGGUFHelperProfile(diagnostics, step + 1, helperProfiler->Snapshot());
			}
			if (outputs.empty() || (!options.statefulDecode && outputs.size() < 2))
			{
				throw std::runtime_error("decode-loop produced no outputs");
			}
			lastLogitsShape = outputs.front().Shape().ToOwned();
			if (options.logitsOutputDirectory && step + 1 >= initialTokenIds.size())
			{
				const auto path =
				    std::filesystem::path(*options.logitsOutputDirectory) / std::format("position-{:06}.txt", step + 1);
				WriteLastTokenLogitsText(outputs.front(), path.string());
			}
			if (options.logitsOutputPath)
			{
				WriteLastTokenLogitsText(outputs.front(), *options.logitsOutputPath);
			}
			if (isPromptReplayStep)
			{
				currentToken = initialTokenIds[step + 1];
			}
			else
			{
				currentToken = LiteNN::GGUF::SelectNextToken(outputs.front(), sampler, history);
				history.push_back(currentToken);
				++generatedTokenCount;
				if (options.stopAtEos && tokenizer.eosTokenId &&
				    currentToken == static_cast<std::int32_t>(*tokenizer.eosTokenId))
				{
					stoppedOnEos = true;
				}
			}
			lastOutputCount = outputs.size();
			if (!options.statefulDecode)
			{
				currentPosition = std::move(outputs[1]);
				caches.reserve(outputs.size() - 2);
				for (std::size_t i = 2; i < outputs.size(); ++i)
				{
					caches.push_back(std::move(outputs[i]));
				}
			}
			const auto stepEnd = std::chrono::steady_clock::now();
			const auto stepMs = std::chrono::duration<double, std::milli>(stepEnd - stepStart).count();
			stepTimesMs.push_back(stepMs);
			if (isPromptReplayStep)
			{
				++promptReplayStepCount;
				promptReplayMs += stepMs;
			}
			else
			{
				++generationStepCount;
				generationMs += stepMs;
			}
			if (options.streamTokens && !isPromptReplayStep)
			{
				std::cout << "stream token step=" << (step + 1) << " position=" << step << " token_id=" << currentToken
				          << " piece=" << EscapeTokenPiece(TokenPieceText(tokenPieces, currentToken))
				          << " generated_tokens=" << generatedTokenCount << " eos=" << (stoppedOnEos ? "true" : "false")
				          << '\n';
				std::cout.flush();
			}
			if (options.streamStats)
			{
				const auto liveTokensPerSecond =
				    generationMs == 0.0 ? 0.0 : static_cast<double>(generatedTokenCount) * 1000.0 / generationMs;
				std::cout << "stream stats step=" << (step + 1) << " position=" << step
				          << " phase=" << (isPromptReplayStep ? "prompt_replay" : "generation") << " step_ms=" << stepMs
				          << " prompt_replay_steps=" << promptReplayStepCount << " prompt_replay_ms=" << promptReplayMs
				          << " generation_steps=" << generationStepCount << " generation_ms=" << generationMs
				          << " generated_tokens=" << generatedTokenCount
				          << " generated_tokens_per_second=" << liveTokensPerSecond
				          << " eos=" << (stoppedOnEos ? "true" : "false") << '\n';
				std::cout.flush();
			}
			LogGGUFDiagnostic(diagnostics, std::format("decode step {} ok {:.3f} ms", step + 1, stepTimesMs.back()));
			if (stoppedOnEos)
			{
				break;
			}
		}
		const auto runEnd = std::chrono::steady_clock::now();

		const auto buildMs = std::chrono::duration<double, std::milli>(buildEnd - buildStart).count();
		const auto runMs = std::chrono::duration<double, std::milli>(runEnd - runStart).count();
		const auto executedSteps = stepTimesMs.size();
		double stepMsSum = 0.0;
		for (const auto stepMs : stepTimesMs)
		{
			stepMsSum += stepMs;
		}
		const auto stepMsAvg = executedSteps == 0 ? 0.0 : stepMsSum / static_cast<double>(executedSteps);
		const auto [stepMsMinIt, stepMsMaxIt] = std::ranges::minmax_element(stepTimesMs);
		const auto stepMsMin = stepTimesMs.empty() ? 0.0 : *stepMsMinIt;
		const auto stepMsMax = stepTimesMs.empty() ? 0.0 : *stepMsMaxIt;
		const auto msPerToken =
		    generatedTokenCount == 0 ? 0.0 : generationMs / static_cast<double>(generatedTokenCount);
		const auto tokensPerSecond =
		    generationMs == 0.0 ? 0.0 : static_cast<double>(generatedTokenCount) * 1000.0 / generationMs;

		std::cout << "Ran LLaMA decode loop tensors=" << importSummary.tensorCount
		          << " metadata=" << importSummary.metadataCount << " steps=" << options.steps
		          << " prompt_tokens=" << initialTokenIds.size() << " generated_tokens=" << generatedTokenCount
		          << " stopped_on_eos=" << (stoppedOnEos ? "true" : "false")
		          << " backend=cpu_aot decode_mode=" << decodeMode
		          << " fallback_count=0 fallback=false cached_modules=1 executed_steps=" << executedSteps
		          << " requested_token_count=" << requestedTokenCount << " max_cache_length=" << maxCacheLength
		          << " build_ms=" << buildMs << " run_ms=" << runMs << " step_ms_avg=" << stepMsAvg
		          << " step_ms_min=" << stepMsMin << " step_ms_max=" << stepMsMax
		          << " prompt_replay_steps=" << promptReplayStepCount << " prompt_replay_ms=" << promptReplayMs
		          << " generation_steps=" << generationStepCount << " generation_ms=" << generationMs
		          << " ms_per_generated_token=" << msPerToken << " generated_tokens_per_second=" << tokensPerSecond
		          << " outputs_per_step=" << lastOutputCount << " last_logits_shape=";
		PrintTensorShape(lastLogitsShape);
		if (options.logitsOutputPath)
		{
			std::cout << " logits_output=" << *options.logitsOutputPath;
		}
		if (options.logitsOutputDirectory)
		{
			std::cout << " logits_output_dir=" << *options.logitsOutputDirectory;
		}
		std::cout << " generated=";
		PrintTokenList(history);
		std::cout << " pieces=" << TokenPiecesText(tokenPieces, history);
		std::cout << '\n';
		if (options.outputPath)
		{
			std::ofstream output(*options.outputPath, std::ios::binary);
			if (!output)
			{
				throw std::runtime_error("Failed to open decode-loop output file: " + *options.outputPath);
			}
			output << TokenListText(history) << '\n'
			       << TokenPiecesText(tokenPieces, history) << '\n'
			       << "generated_tokens=" << generatedTokenCount
			       << " stopped_on_eos=" << (stoppedOnEos ? "true" : "false")
			       << " backend=cpu_aot decode_mode=" << decodeMode
			       << " fallback_count=0 executed_steps=" << executedSteps
			       << " requested_token_count=" << requestedTokenCount << " max_cache_length=" << maxCacheLength
			       << " run_ms=" << runMs << " step_ms_avg=" << stepMsAvg
			       << " prompt_replay_steps=" << promptReplayStepCount << " prompt_replay_ms=" << promptReplayMs
			       << " generation_steps=" << generationStepCount << " generation_ms=" << generationMs
			       << " ms_per_generated_token=" << msPerToken << " generated_tokens_per_second=" << tokensPerSecond
			       << '\n';
			if (!output)
			{
				throw std::runtime_error("Failed to write decode-loop output file: " + *options.outputPath);
			}
		}
#endif
	}

#ifdef LITENN_GGUF_CONVERT_ENABLE_AOT
	bool TruthyEnvValue(const char* value)
	{
		if (value == nullptr)
		{
			return false;
		}
		const std::string_view text{ value };
		return text == "1" || text == "true" || text == "TRUE" || text == "on" || text == "ON";
	}

	std::optional<std::uint64_t> ParseU64Env(const char* name)
	{
		if (const char* value = std::getenv(name))
		{
			std::uint64_t parsed{};
			const std::string_view text{ value };
			const auto* begin = text.data();
			const auto* end = begin + text.size();
			if (const auto result = std::from_chars(begin, end, parsed); result.ec == std::errc{} && result.ptr == end)
			{
				return parsed;
			}
		}
		return std::nullopt;
	}

	LiteNN::CompilerOptions CompilerOptionsFromEnvironment()
	{
		auto options = LiteNN::CompilerOptions::Defaults();
		if (const auto threadCount = ParseU64Env("LITENN_CPU_AOT_THREADS"); threadCount && *threadCount > 0)
		{
			options.cpuAOTThreadCount = static_cast<std::size_t>(*threadCount);
		}
		if (const auto minFlops = ParseU64Env("LITENN_CPU_AOT_PARALLEL_MIN_FLOPS"))
		{
			options.cpuAOTParallelMinFlops = *minFlops;
		}
		if (const auto minConstantBytes = ParseU64Env("LITENN_CPU_AOT_EXTERNAL_CONSTANT_MIN_BYTES"))
		{
			options.cpuAOTExternalConstantMinBytes = *minConstantBytes;
		}
		if (const auto optLevel = ParseU64Env("LITENN_CPU_AOT_LLVM_OPT_LEVEL"))
		{
			options.cpuAOTLLVMOptLevel = static_cast<std::uint8_t>(std::min<std::uint64_t>(*optLevel, 3));
		}
		if (const char* affinity = std::getenv("LITENN_CPU_AOT_AFFINITY"))
		{
			std::string value{ affinity };
			std::ranges::transform(value, value.begin(),
			                       [](unsigned char ch) { return static_cast<char>(std::tolower(ch)); });
			if (value == "compact" || value == "1" || value == "true" || value == "on")
			{
				options.cpuAOTAffinityPolicy = LiteNN::CPUAOTAffinityPolicy::Compact;
			}
		}
		options.enableCPUAOTExternalRegions = TruthyEnvValue(std::getenv("LITENN_CPU_AOT_EXTERNAL_REGIONS")) ||
		                                      TruthyEnvValue(std::getenv("LITENN_CPU_AOT_EXTERNAL_CONSTANTS"));
		if (const char* value = std::getenv("LITENN_CPU_AOT_EXTERNAL_REGION_FUSION"))
		{
			options.enableCPUAOTExternalRegionFusion = TruthyEnvValue(value);
		}
		if (TruthyEnvValue(std::getenv("LITENN_CUDA_DISABLE_NATIVE_AOT")))
		{
			options.enableCUDANativeAOT = false;
		}
		options.enableCPUAOTGGMLQ8KStagedMatMul = TruthyEnvValue(std::getenv("LITENN_CPU_AOT_Q8K_STAGED_MATMUL"));
		options.enableCompileDiagnostics = TruthyEnvValue(std::getenv("LITENN_COMPILE_DIAGNOSTICS"));
		return options;
	}

	void ApplyDecodeLoopCompilerOptions(const DecodeLoopCommandOptions& decodeOptions,
	                                    LiteNN::CompilerOptions& compilerOptions)
	{
		if (decodeOptions.cpuAOTThreadCount)
		{
			compilerOptions.cpuAOTThreadCount = *decodeOptions.cpuAOTThreadCount;
		}
		if (decodeOptions.cpuAOTParallelMinFlops)
		{
			compilerOptions.cpuAOTParallelMinFlops = *decodeOptions.cpuAOTParallelMinFlops;
		}
		if (decodeOptions.cpuAOTLLVMOptLevel)
		{
			compilerOptions.cpuAOTLLVMOptLevel = *decodeOptions.cpuAOTLLVMOptLevel;
		}
		if (decodeOptions.cpuAOTAffinityPolicy)
		{
			compilerOptions.cpuAOTAffinityPolicy = *decodeOptions.cpuAOTAffinityPolicy == "compact"
			                                           ? LiteNN::CPUAOTAffinityPolicy::Compact
			                                           : LiteNN::CPUAOTAffinityPolicy::None;
		}
		if (decodeOptions.enableCompileDiagnostics)
		{
			compilerOptions.enableCompileDiagnostics = *decodeOptions.enableCompileDiagnostics;
		}
		if (decodeOptions.enableCPUAOTQ8KStagedMatMul)
		{
			compilerOptions.enableCPUAOTGGMLQ8KStagedMatMul = true;
		}
	}

	std::string_view BackendName(LiteNN::CompiledModuleBackend backend)
	{
		switch (backend)
		{
		case LiteNN::CompiledModuleBackend::CPUNative:
			return "cpu_native";
		case LiteNN::CompiledModuleBackend::CUDANative:
			return "cuda_native";
		}
		return "unknown";
	}

	void PrintArtifactSummary(const LiteNN::CompiledModuleArtifact& artifact, std::string_view outputPath)
	{
		std::cout << "Wrote AOT carrier object " << outputPath << " backend=" << BackendName(artifact.Backend())
		          << " rodata=" << artifact.Rodata().size() << " bytes instructions=" << artifact.Instructions().size()
		          << " bytes\n";
	}

	void PrintSeparatedArtifactSummary(const LiteNN::CompiledModuleSeparatedArtifact& artifact,
	                                   std::string_view outputDir)
	{
		std::cout << "Wrote separated AOT carrier objects " << outputDir
		          << " backend=" << BackendName(artifact.Backend()) << " metadata=" << artifact.Metadata().size()
		          << " bytes constants=" << artifact.Constants().size()
		          << " bytes weights=" << artifact.Weights().size()
		          << " bytes instructions=" << artifact.Instructions().size() << " bytes\n";
	}
#endif

	std::vector<LiteNN::Tensor<LiteNN::CPU>> RunCPUModelAOT(const LiteNN::ExecutablePlan& plan,
	                                                        std::span<const LiteNN::Tensor<LiteNN::CPU>> inputs)
	{
#ifdef LITENN_GGUF_CONVERT_ENABLE_AOT
		auto options = CompilerOptionsFromEnvironment();
		options.enableCPUAOTExternalRegions = true;
		auto module = LiteNN::Compiler<LiteNN::CPU>::Compile(plan, options);
		return module.RunTensors(inputs);
#else
		(void) plan;
		(void) inputs;
		throw std::runtime_error("GGUF model execution requires LITENN_ENABLE_MLIR=ON");
#endif
	}
} // namespace

int main(int argc, char** argv)
{
	try
	{
		if (argc == 2)
		{
			const std::string_view arg = argv[1];
			if (arg == "-h" || arg == "--help")
			{
				PrintUsage(argv[0]);
				return 0;
			}
		}

		if (argc >= 2 && std::string_view(argv[1]) == "--import")
		{
			if (argc != 4)
			{
				PrintUsage(argv[0]);
				return 1;
			}
			const auto summary = LiteNN::GGUF::ConvertGGUFArchive(argv[2], argv[3]);
			std::cout << "Imported archive with " << summary.tensorCount << " tensors and " << summary.metadataCount
			          << " metadata entries\n";
			return 0;
		}

		if (argc >= 2 && std::string_view(argv[1]) == "--import-external")
		{
			if (argc != 5)
			{
				PrintUsage(argv[0]);
				return 1;
			}
			const auto summary = LiteNN::GGUF::ConvertGGUFArchiveExternalWeights(argv[2], argv[3], argv[4]);
			std::cout << "Imported archive with " << summary.tensorCount << " tensors and " << summary.metadataCount
			          << " metadata entries using external quantized weights\n";
			return 0;
		}

		if (argc >= 2 && std::string_view(argv[1]) == "--analyze-llm")
		{
			const auto options = ParseAnalyzeLLMOptions(argc, argv);
			const auto imported = LiteNN::GGUF::ImportGGUFArchive(options.inputPath);
			const auto profile = options.profile ? *options.profile : InferLLMProfile(imported.model.UnsafeGraphView());
			const auto report = LiteNN::GGUF::AnalyzeLLaMACompatibility(imported.model.UnsafeGraphView(), profile,
			                                                            options.dequantizedBudgetBytes);
			PrintLLMCompatibilityReport(report);
			return report.lowerable ? 0 : 2;
		}

		if (argc == 3 && !std::string_view(argv[1]).starts_with("--"))
		{
			const auto summary = LiteNN::GGUF::ConvertGGUFArchive(argv[1], argv[2]);
			std::cout << "Imported archive with " << summary.tensorCount << " tensors and " << summary.metadataCount
			          << " metadata entries\n";
			return 0;
		}

		if (argc >= 2 && std::string_view(argv[1]) == "--plan-llm")
		{
			if (argc != 5 && argc != 6)
			{
				PrintUsage(argv[0]);
				return 1;
			}
			const auto imported = LiteNN::GGUF::ImportGGUFArchive(argv[2]);
			const auto prefillSequenceLength = ParseSize(argv[3], "prefill-sequence-length");
			const auto decodePastLength = ParseSize(argv[4], "decode-past-length", true);
			const auto maxCacheLength = argc == 6 ? ParseSize(argv[5], "max-cache-length", true) : 0uz;
			PrintLLMArtifactPlan(LiteNN::GGUF::PlanLLaMAArtifacts(imported.model.UnsafeGraphView(),
			                                                      LiteNN::GGUF::LLaMAArtifactPlanningOptions{
			                                                          .prefillSequenceLength = prefillSequenceLength,
			                                                          .decodePastLength = decodePastLength,
			                                                          .maxCacheLength = maxCacheLength,
			                                                      }));
			return 0;
		}

		if (argc >= 2 && std::string_view(argv[1]) == "--compile-cpu")
		{
			if (argc != 4 && argc != 5)
			{
				PrintUsage(argv[0]);
				return 1;
			}
#ifdef LITENN_GGUF_CONVERT_ENABLE_AOT
			auto artifact = LiteNN::Compiler<LiteNN::CPU>::CompileArtifact(
			    LiteNN::Serialization::LoadVNextModelPackage(argv[2]).plan, CompilerOptionsFromEnvironment());
			const std::string_view symbolPrefix = argc == 5 ? std::string_view(argv[4]) : "litenn_gguf_module";
			artifact.WriteObjectFile(argv[3], symbolPrefix);
			PrintArtifactSummary(artifact, argv[3]);
			return 0;
#else
			throw std::runtime_error("AOT compiler support is not enabled; configure with LITENN_ENABLE_MLIR=ON");
#endif
		}

		if (argc >= 2 && std::string_view(argv[1]) == "--compile-cuda")
		{
			if (argc != 4 && argc != 5)
			{
				PrintUsage(argv[0]);
				return 1;
			}
#if defined(LITENN_GGUF_CONVERT_ENABLE_AOT) && defined(LITENN_ENABLE_CUDA)
			auto artifact = LiteNN::Compiler<LiteNN::CUDA>::CompileArtifact(
			    LiteNN::Serialization::LoadVNextModelPackage(argv[2]).plan, CompilerOptionsFromEnvironment());
			const std::string_view symbolPrefix = argc == 5 ? std::string_view(argv[4]) : "litenn_gguf_module";
			artifact.WriteObjectFile(argv[3], symbolPrefix);
			PrintArtifactSummary(artifact, argv[3]);
			return 0;
#elif defined(LITENN_GGUF_CONVERT_ENABLE_AOT)
			throw std::runtime_error(
			    "CUDA AOT support is not enabled in this build; configure with LITENN_ENABLE_CUDA=ON");
#else
			throw std::runtime_error("AOT compiler support is not enabled; configure with LITENN_ENABLE_MLIR=ON");
#endif
		}

		if (argc >= 2 && std::string_view(argv[1]) == "--compile-cpu-separated")
		{
			if (argc != 4 && argc != 5)
			{
				PrintUsage(argv[0]);
				return 1;
			}
#ifdef LITENN_GGUF_CONVERT_ENABLE_AOT
			auto artifact =
			    LiteNN::Compiler<LiteNN::CPU>::CompileArtifact(
			        LiteNN::Serialization::LoadVNextModelPackage(argv[2]).plan, CompilerOptionsFromEnvironment())
			        .SeparateRodata();
			const std::string_view symbolPrefix = argc == 5 ? std::string_view(argv[4]) : "litenn_gguf_module";
			artifact.WriteObjectFiles(argv[3], symbolPrefix);
			PrintSeparatedArtifactSummary(artifact, argv[3]);
			return 0;
#else
			throw std::runtime_error("AOT compiler support is not enabled; configure with LITENN_ENABLE_MLIR=ON");
#endif
		}

		if (argc >= 2 && std::string_view(argv[1]) == "--compile-cuda-separated")
		{
			if (argc != 4 && argc != 5)
			{
				PrintUsage(argv[0]);
				return 1;
			}
#if defined(LITENN_GGUF_CONVERT_ENABLE_AOT) && defined(LITENN_ENABLE_CUDA)
			auto artifact =
			    LiteNN::Compiler<LiteNN::CUDA>::CompileArtifact(
			        LiteNN::Serialization::LoadVNextModelPackage(argv[2]).plan, CompilerOptionsFromEnvironment())
			        .SeparateRodata();
			const std::string_view symbolPrefix = argc == 5 ? std::string_view(argv[4]) : "litenn_gguf_module";
			artifact.WriteObjectFiles(argv[3], symbolPrefix);
			PrintSeparatedArtifactSummary(artifact, argv[3]);
			return 0;
#elif defined(LITENN_GGUF_CONVERT_ENABLE_AOT)
			throw std::runtime_error(
			    "CUDA AOT support is not enabled in this build; configure with LITENN_ENABLE_CUDA=ON");
#else
			throw std::runtime_error("AOT compiler support is not enabled; configure with LITENN_ENABLE_MLIR=ON");
#endif
		}

		if (argc >= 2 && std::string_view(argv[1]) == "--lower-llama")
		{
			if (argc != 5 && argc != 6)
			{
				PrintUsage(argv[0]);
				return 1;
			}
			const auto imported = LiteNN::GGUF::ImportGGUFArchive(argv[2]);
			const auto sequenceLength = ParseSize(argv[4], "sequence-length");
			const auto positionOffset = argc == 6 ? ParseSize(argv[5], "position-offset", true) : 0uz;
			auto lowered =
			    LiteNN::GGUF::LowerLLaMACausalLM(imported.model.UnsafeGraphView(), sequenceLength, positionOffset);
			LiteNN::Serialization::SaveVNextModelPackage(LiteNN::Detail::BuildExecutableModuleFromGraph(lowered),
			                                             argv[3]);
			std::cout << "Lowered LLaMA graph from " << imported.summary.tensorCount << " tensors and "
			          << imported.summary.metadataCount << " metadata entries\n";
			return 0;
		}

		if (argc >= 2 && std::string_view(argv[1]) == "--lower-llama-quantized")
		{
			if (argc != 6 && argc != 7)
			{
				PrintUsage(argv[0]);
				return 1;
			}
			const auto imported = LiteNN::GGUF::ImportGGUFArchive(argv[2]);
			const auto sequenceLength = ParseSize(argv[5], "sequence-length");
			const auto positionOffset = argc == 7 ? ParseSize(argv[6], "position-offset", true) : 0uz;
			auto lowered = LiteNN::GGUF::LowerLLaMACausalLM(imported.model.UnsafeGraphView(), sequenceLength,
			                                                positionOffset, { .preserveQuantizedWeights = true });
			LiteNN::Serialization::SaveVNextModelPackageExternalWeights(lowered, argv[3], argv[4]);
			std::cout << "Lowered quantized LLaMA graph from " << imported.summary.tensorCount << " tensors and "
			          << imported.summary.metadataCount << " metadata entries using external quantized weights\n";
			return 0;
		}

		if (argc >= 2 && std::string_view(argv[1]) == "--run-llama-token-ids")
		{
			if (argc != 4 && argc != 5)
			{
				PrintUsage(argv[0]);
				return 1;
			}
			const auto tokenIds = ParseTokenIds(argv[3]);
			const auto positionOffset = argc == 5 ? ParseSize(argv[4], "position-offset", true) : 0uz;
			const auto imported = LiteNN::GGUF::ImportGGUFArchive(argv[2]);
			auto lowered = LiteNN::GGUF::LowerLLaMACausalLM(imported.model.UnsafeGraphView(), tokenIds.size(),
			                                                positionOffset, { .preserveQuantizedWeights = true });
			const auto plan = LiteNN::Detail::BuildExecutablePlanFromGraph(lowered);
			auto inputs = MakeZeroStateInputs(plan, MakeTokenIdTensor(tokenIds, plan));
			const auto outputs = RunCPUModelAOT(plan, inputs);
			if (outputs.empty())
			{
				throw std::runtime_error("LLM package produced no outputs");
			}
			const auto& logits = outputs.front();
			LiteNN::GGUF::LLMSamplerState sampler;
			const auto nextToken = LiteNN::GGUF::SelectNextToken(logits, sampler, tokenIds);
			std::cout << "Ran LLaMA GGUF token-id smoke tensors=" << imported.summary.tensorCount
			          << " metadata=" << imported.summary.metadataCount << " inputs=" << plan.inputs.size()
			          << " outputs=" << outputs.size() << " logits_dtype=" << LiteNN::DataTypeName(logits.DType())
			          << " logits_shape=";
			PrintTensorShape(logits.Shape());
			std::cout << " next_token=" << nextToken << '\n';
			return 0;
		}

		if (argc >= 2 && std::string_view(argv[1]) == "--dump-llama-token-id-logits")
		{
			if (argc != 5 && argc != 6)
			{
				PrintUsage(argv[0]);
				return 1;
			}
			const auto tokenIds = ParseTokenIds(argv[3]);
			const auto positionOffset = argc == 6 ? ParseSize(argv[5], "position-offset", true) : 0uz;
			const auto imported = LiteNN::GGUF::ImportGGUFArchive(argv[2]);
			auto lowered = LiteNN::GGUF::LowerLLaMACausalLM(imported.model.UnsafeGraphView(), tokenIds.size(),
			                                                positionOffset, { .preserveQuantizedWeights = true });
			const auto plan = LiteNN::Detail::BuildExecutablePlanFromGraph(lowered);
			auto inputs = MakeZeroStateInputs(plan, MakeTokenIdTensor(tokenIds, plan));
			const auto outputs = RunCPUModelAOT(plan, inputs);
			if (outputs.empty())
			{
				throw std::runtime_error("LLM package produced no outputs");
			}
			WriteLastTokenLogitsText(outputs.front(), argv[4]);
			std::cout << "Dumped LLaMA GGUF last-token logits tensors=" << imported.summary.tensorCount
			          << " metadata=" << imported.summary.metadataCount << " token_ids=";
			PrintTokenList(tokenIds);
			std::cout << " logits_shape=";
			PrintTensorShape(outputs.front().Shape());
			std::cout << " output=" << argv[4] << '\n';
			return 0;
		}

		if (argc >= 2 && std::string_view(argv[1]) == "--tokenize-llama-prompt")
		{
			if (argc != 5 && argc != 6)
			{
				PrintUsage(argv[0]);
				return 1;
			}
			const bool applyChatTemplate = argc == 6 && std::string_view(argv[5]) == "--chat-template";
			if (argc == 6 && !applyChatTemplate)
			{
				throw std::runtime_error("--tokenize-llama-prompt only accepts --chat-template as an optional flag");
			}
			WritePromptTokens(argv[2], argv[3], argv[4], applyChatTemplate);
			std::cout << "Tokenized LLaMA prompt with llama.cpp backend output=" << argv[4]
			          << " chat_template=" << (applyChatTemplate ? "true" : "false") << '\n';
			return 0;
		}

		if (argc >= 2 && std::string_view(argv[1]) == "--run-llama-prompt")
		{
			if (argc < 4 || argc > 6)
			{
				PrintUsage(argv[0]);
				return 1;
			}
			std::size_t positionOffset = 0;
			bool applyChatTemplate = false;
			for (int i = 4; i < argc; ++i)
			{
				const std::string_view arg = argv[i];
				if (arg == "--chat-template")
				{
					applyChatTemplate = true;
				}
				else
				{
					positionOffset = ParseSize(arg, "position-offset", true);
				}
			}
			const auto imported = LiteNN::GGUF::ImportGGUFArchive(argv[2]);
			const auto promptTokenIds =
			    TokenizePrompt(argv[2], argv[3], imported.model.UnsafeGraphView(), applyChatTemplate);
			auto lowered = LiteNN::GGUF::LowerLLaMACausalLM(imported.model.UnsafeGraphView(), promptTokenIds.size(),
			                                                positionOffset, { .preserveQuantizedWeights = true });
			const auto plan = LiteNN::Detail::BuildExecutablePlanFromGraph(lowered);
			auto inputs = MakeZeroStateInputs(plan, MakeTokenIdTensor(promptTokenIds, plan));
			const auto outputs = RunCPUModelAOT(plan, inputs);
			if (outputs.empty())
			{
				throw std::runtime_error("LLM package produced no outputs");
			}
			const auto& logits = outputs.front();
			LiteNN::GGUF::LLMSamplerState sampler;
			const auto nextToken = LiteNN::GGUF::SelectNextToken(logits, sampler, promptTokenIds);
			std::cout << "Ran LLaMA GGUF exact-prompt smoke tensors=" << imported.summary.tensorCount
			          << " metadata=" << imported.summary.metadataCount << " token_ids=";
			PrintTokenList(promptTokenIds);
			std::cout << " tokenizer_backend="
#ifdef LITENN_GGUF_CONVERT_ENABLE_LLAMA_CPP_TOKENIZER
			          << "llama.cpp"
#else
			          << "exact-vocabulary"
#endif
			          << " chat_template=" << (applyChatTemplate ? "true" : "false")
			          << " pieces=" << TokenPiecesText(imported.model.UnsafeGraphView(), promptTokenIds)
			          << " inputs=" << plan.inputs.size() << " outputs=" << outputs.size()
			          << " logits_dtype=" << LiteNN::DataTypeName(logits.DType()) << " logits_shape=";
			PrintTensorShape(logits.Shape());
			std::cout << " next_token=" << nextToken << '\n';
			return 0;
		}

		if (argc >= 2 && std::string_view(argv[1]) == "--run-llama-package-token-ids")
		{
			if (argc != 4)
			{
				PrintUsage(argv[0]);
				return 1;
			}
			const auto package = LiteNN::Serialization::LoadVNextModelPackage(argv[2]);
			const auto tokenIds = ParseTokenIds(argv[3]);
			auto inputs = MakeZeroStateInputs(package.plan, MakeTokenIdTensor(tokenIds, package.plan));
			const auto outputs = RunCPUModelAOT(package.plan, inputs);
			if (outputs.empty())
			{
				throw std::runtime_error("LLM package produced no outputs");
			}
			const auto& logits = outputs.front();
			LiteNN::GGUF::LLMSamplerState sampler;
			const auto nextToken = LiteNN::GGUF::SelectNextToken(logits, sampler, tokenIds);
			std::cout << "Ran LLaMA package token-id smoke inputs=" << package.plan.inputs.size()
			          << " outputs=" << outputs.size() << " logits_dtype=" << LiteNN::DataTypeName(logits.DType())
			          << " logits_shape=";
			PrintTensorShape(logits.Shape());
			std::cout << " next_token=" << nextToken << '\n';
			return 0;
		}

		if (argc >= 2 && std::string_view(argv[1]) == "--run-llama-decode-loop-token-id")
		{
			RunDecodeLoopFromGGUF(ParseDecodeLoopOptions(argc, argv));
			return 0;
		}

		if (argc >= 2 && std::string_view(argv[1]) == "--run-llama-decode-loop-token-ids")
		{
			RunDecodeLoopFromGGUF(ParseTokenIdsDecodeLoopOptions(argc, argv));
			return 0;
		}

		if (argc >= 2 && std::string_view(argv[1]) == "--run-llama-prompt-decode-loop")
		{
			RunDecodeLoopFromGGUF(ParsePromptDecodeLoopOptions(argc, argv));
			return 0;
		}

		if (argc >= 2 && std::string_view(argv[1]) == "--lower-llama-decode")
		{
			if (argc != 6)
			{
				PrintUsage(argv[0]);
				return 1;
			}
			const auto imported = LiteNN::GGUF::ImportGGUFArchive(argv[2]);
			const auto sequenceLength = ParseSize(argv[4], "sequence-length");
			const auto pastLength = ParseSize(argv[5], "past-length", true);
			auto lowered = LiteNN::GGUF::LowerLLaMACausalLMDecode(imported.model.UnsafeGraphView(), sequenceLength,
			                                                      pastLength, pastLength);
			LiteNN::Serialization::SaveVNextModelPackage(LiteNN::Detail::BuildExecutableModuleFromGraph(lowered),
			                                             argv[3]);
			std::cout << "Lowered LLaMA decode graph from " << imported.summary.tensorCount << " tensors and "
			          << imported.summary.metadataCount << " metadata entries\n";
			return 0;
		}

		if (argc >= 2 && std::string_view(argv[1]) == "--lower-llama-decode-stateful")
		{
			if (argc != 7)
			{
				PrintUsage(argv[0]);
				return 1;
			}
			const auto imported = LiteNN::GGUF::ImportGGUFArchive(argv[2]);
			const auto pastLength = ParseSize(argv[5], "past-length", true);
			const auto maxCacheLength = ParseSize(argv[6], "max-cache-length");
			auto schedule = LiteNN::GGUF::BuildLLaMADecodeRuntimeSchedule(imported.model.UnsafeGraphView(),
			                                                              { .prefillSequenceLength = 1,
			                                                                .decodePastLength = pastLength,
			                                                                .maxCacheLength = maxCacheLength,
			                                                                .preserveQuantizedWeights = true,
			                                                                .dynamicDecodePosition = true });
			const auto forward = schedule.module.plan.forward;
			const auto projection = LiteNN::Runtime::RuntimeScheduleOutputProjectionForFunction(schedule, forward);
			LiteNN::Serialization::SaveVNextModelPackageExternalWeights(schedule, argv[3], argv[4]);
			std::cout << "Lowered stateful LLaMA decode package with " << schedule.states.size()
			          << " runtime states and " << schedule.stateValueBindings.size() << " value bindings"
			          << " functional_outputs=" << projection.functionalOutputCount
			          << " state_aliases=" << projection.stateAliases.size()
			          << " public_outputs=" << projection.publicOutputIndices.size() << " public_output_shape=";
			if (projection.publicOutputTypes.empty())
			{
				PrintSizeList({});
			}
			else
			{
				PrintSizeList(projection.publicOutputTypes.front().StaticShape());
			}
			std::cout << '\n';
			return 0;
		}

		PrintUsage(argv[0]);
		return 1;
	}
	catch (const std::exception& ex)
	{
		std::cerr << "litenn_gguf_convert: " << ex.what() << '\n';
		return 1;
	}
}
