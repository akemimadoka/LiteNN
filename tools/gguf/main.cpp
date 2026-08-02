#include "GGMLQuantizedKernels.h"
#include "GGUFImporter.h"
#include "LLMGeneration.h"
#include "LLaMABuilder.h"

#ifdef LITENN_GGUF_CONVERT_ENABLE_AOT
#include "DecodeAOTCache.h"

#include <LiteNN/Compiler/CompiledModule.h>
#endif
#ifdef LITENN_GGUF_CONVERT_ENABLE_LLAMA_CPP_TOKENIZER
#include <LlamaCppTokenizerAdapter.h>
#endif
#include <LiteNN/Runtime/Scheduler.h>
#include <LiteNN/Serialization/ModelPackageIO.h>

#include <algorithm>
#include <array>
#include <cctype>
#include <cerrno>
#include <charconv>
#include <chrono>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <format>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <limits>
#include <memory>
#include <optional>
#include <span>
#include <stdexcept>
#include <string>
#include <string_view>
#include <type_traits>
#include <vector>

#ifdef _WIN32
#ifndef NOMINMAX
#define NOMINMAX
#endif
#include <windows.h>
#else
#include <fcntl.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <unistd.h>
#endif

namespace
{
#ifdef LITENN_GGUF_CONVERT_ENABLE_AOT
	using LiteNN::GGUF::Tooling::DecodeAOTSharedWeightsIdentity;
	using LiteNN::GGUF::Tooling::FNV1a;
	using LiteNN::GGUF::Tooling::PublishDecodeAOTSharedWeightsAtomically;
	using LiteNN::GGUF::Tooling::SharedWeightsPublishResult;
#endif

	void PrintUsage(std::string_view executable)
	{
		std::cerr
		    << "Usage:\n"
		    << "  " << executable << " --import <input.gguf> <output.ltnn>\n"
		    << "  " << executable << " --import-external <input.gguf> <output.ltnn> <weights.bin>\n"
		    << "  " << executable
		    << " --analyze-llm <input.gguf> [profile] [--dequantized-budget-bytes N|--dequantized-budget-mib N]\n"
		    << "  " << executable << " --plan-llm <input.gguf> <prefill-sequence-length> <decode-past-length> "
		    << "[max-cache-length]\n"
		    << "  " << executable << " --lower-llama <input.gguf> <output.ltnn> <sequence-length> [position-offset]\n"
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
		    << "  " << executable << " --tokenize-llama-prompt <input.gguf> <prompt> <tokens.json> [--chat-template]\n"
		    << "  " << executable << " --run-llama-prompt <input.gguf> <prompt> [position-offset]\n"
		    << "  " << executable << " --run-llama-package-token-ids <input.ltnn> <comma-token-ids>\n"
		    << "  " << executable
		    << " --run-llama-decode-loop-token-id <input.gguf> <initial-token-id> <steps> [output.txt] "
		       "[--sample greedy|random] [--temperature T] [--top-k K] [--top-p P] [--repeat-penalty R] "
		       "[--seed N] [--logits-output output.txt] [--logits-output-dir dir] [--ignore-eos] "
		       "[--stateful|--functional] [--stream-tokens] [--stream-stats] [--profile-helpers] [--profile-nodes] "
		       "[--compile-only] "
		       "[--max-cache-length N] [--paged-reference-decode] [--paged-resident-pages N] "
		       "[--cpu-aot-threads N] [--cpu-aot-affinity none|compact|spread] "
		       "[--cpu-aot-worker-wait adaptive|low-power|latency] [--cpu-aot-llvm-opt-level 0|1|2|3] "
		       "[--cpu-aot-parallel-min-flops N] [--compile-diagnostics|--no-compile-diagnostics] "
		       "[--cpu-aot-q8k-staged-matmul] [--cpu-aot-ggml-prepacked-weights] "
		       "[--cpu-aot-ggml-prepacked-weight-policy disabled|profitable|all] "
		       "[--cpu-aot-ggml-prepacked-weight-layout expanded-v1|compact-v3|field-interleaved-v4]\n"
		    << "  " << executable
		    << " --run-llama-decode-loop-token-ids <input.gguf> <comma-token-ids> <steps> [output.txt] "
		       "[--sample greedy|random] [--temperature T] [--top-k K] [--top-p P] [--repeat-penalty R] "
		       "[--seed N] [--logits-output output.txt] [--logits-output-dir dir] [--ignore-eos] "
		       "[--stateful|--functional] [--stream-tokens] [--stream-stats] [--profile-helpers] [--profile-nodes] "
		       "[--compile-only] "
		       "[--max-cache-length N] [--paged-reference-decode] [--paged-resident-pages N] "
		       "[--cpu-aot-threads N] [--cpu-aot-affinity none|compact|spread] "
		       "[--cpu-aot-worker-wait adaptive|low-power|latency] [--cpu-aot-llvm-opt-level 0|1|2|3] "
		       "[--cpu-aot-parallel-min-flops N] [--compile-diagnostics|--no-compile-diagnostics] "
		       "[--cpu-aot-q8k-staged-matmul] [--cpu-aot-ggml-prepacked-weights] "
		       "[--cpu-aot-ggml-prepacked-weight-policy disabled|profitable|all] "
		       "[--cpu-aot-ggml-prepacked-weight-layout expanded-v1|compact-v3|field-interleaved-v4]\n"
		    << "  " << executable
		    << " --run-llama-prompt-decode-loop <input.gguf> <prompt> <steps> [output.txt] "
		       "[--sample greedy|random] [--temperature T] [--top-k K] [--top-p P] [--repeat-penalty R] "
		       "[--seed N] [--logits-output output.txt] [--logits-output-dir dir] [--ignore-eos] "
		       "[--stateful|--functional] [--stream-tokens] [--stream-stats] [--profile-helpers] [--profile-nodes] "
		       "[--compile-only] "
		       "[--max-cache-length N] [--paged-reference-decode] [--paged-resident-pages N] "
		       "[--cpu-aot-threads N] [--cpu-aot-affinity none|compact|spread] "
		       "[--cpu-aot-worker-wait adaptive|low-power|latency] [--cpu-aot-llvm-opt-level 0|1|2|3] "
		       "[--cpu-aot-parallel-min-flops N] [--compile-diagnostics|--no-compile-diagnostics] "
		       "[--cpu-aot-q8k-staged-matmul] [--cpu-aot-ggml-prepacked-weights] "
		       "[--cpu-aot-ggml-prepacked-weight-policy disabled|profitable|all] "
		       "[--cpu-aot-ggml-prepacked-weight-layout expanded-v1|compact-v3|field-interleaved-v4]\n"
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

	std::string ParseGGMLPrepackedWeightPolicy(std::string_view text)
	{
		std::string value{ text };
		std::ranges::transform(value, value.begin(),
		                       [](unsigned char ch) { return static_cast<char>(std::tolower(ch)); });
		if (value == "off" || value == "none")
		{
			value = "disabled";
		}
		else if (value == "on")
		{
			value = "all";
		}
		if (value != "disabled" && value != "profitable" && value != "all")
		{
			throw std::runtime_error("cpu-aot-ggml-prepacked-weight-policy must be disabled, profitable, or all");
		}
		return value;
	}

	std::string ParseGGMLPrepackedWeightLayout(std::string_view text)
	{
		std::string value{ text };
		std::ranges::transform(value, value.begin(),
		                       [](unsigned char ch) { return static_cast<char>(std::tolower(ch)); });
		if (value == "expanded" || value == "expanded-v1")
		{
			return "expanded_f32_scales_v1";
		}
		if (value == "compact" || value == "compact-v3")
		{
			return "compact_block_grouped_v3";
		}
		if (value == "field-interleaved" || value == "field-interleaved-v4")
		{
			return "field_interleaved_v4";
		}
		if (value != "expanded_f32_scales_v1" && value != "compact_block_grouped_v3" && value != "field_interleaved_v4")
		{
			throw std::runtime_error(
			    "cpu-aot-ggml-prepacked-weight-layout must be expanded-v1, compact-v3, or field-interleaved-v4");
		}
		return value;
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
		bool profileHelpers{};
		bool profileNodes{};
		bool compileOnly{};
		bool pagedReferenceDecode{};
		std::optional<std::size_t> pagedResidentPageCount;
		bool enableCPUAOTQ8KStagedMatMul{};
		bool enableCPUAOTGGMLPrepackedWeights{};
		std::optional<std::string> cpuAOTGGMLPrepackedWeightPolicy;
		std::optional<std::string> cpuAOTGGMLPrepackedWeightLayout;
		std::optional<std::size_t> cpuAOTThreadCount;
		std::optional<std::uint64_t> cpuAOTParallelMinFlops;
		std::optional<std::uint8_t> cpuAOTLLVMOptLevel;
		std::optional<std::string> cpuAOTAffinityPolicy;
		std::optional<std::string> cpuAOTWorkerWaitPolicy;
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
			else if (arg == "--profile-helpers")
			{
				options.profileHelpers = true;
			}
			else if (arg == "--profile-nodes")
			{
				options.profileNodes = true;
			}
			else if (arg == "--compile-only")
			{
				options.compileOnly = true;
			}
			else if (arg == "--paged-reference-decode")
			{
				options.pagedReferenceDecode = true;
				options.statefulDecode = true;
			}
			else if (arg == "--paged-resident-pages")
			{
				options.pagedResidentPageCount = ParseSize(requireValue(arg), "paged-resident-pages");
			}
			else if (arg == "--cpu-aot-q8k-staged-matmul")
			{
				options.enableCPUAOTQ8KStagedMatMul = true;
			}
			else if (arg == "--cpu-aot-ggml-prepacked-weights")
			{
				options.enableCPUAOTGGMLPrepackedWeights = true;
			}
			else if (arg == "--cpu-aot-ggml-prepacked-weight-policy")
			{
				options.cpuAOTGGMLPrepackedWeightPolicy = ParseGGMLPrepackedWeightPolicy(requireValue(arg));
			}
			else if (arg == "--cpu-aot-ggml-prepacked-weight-layout")
			{
				options.cpuAOTGGMLPrepackedWeightLayout = ParseGGMLPrepackedWeightLayout(requireValue(arg));
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
				else if (value == "spread")
				{
					options.cpuAOTAffinityPolicy = "spread";
				}
				else
				{
					throw std::runtime_error("cpu-aot-affinity must be 'none', 'compact', or 'spread'");
				}
			}
			else if (arg == "--cpu-aot-worker-wait")
			{
				const auto value = requireValue(arg);
				if (value != "adaptive" && value != "low-power" && value != "latency")
				{
					throw std::runtime_error("cpu-aot-worker-wait must be 'adaptive', 'low-power', or 'latency'");
				}
				options.cpuAOTWorkerWaitPolicy = value;
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

	void EnsureParentDirectory(const std::filesystem::path& path)
	{
		const auto parent = path.parent_path();
		if (!parent.empty())
		{
			std::filesystem::create_directories(parent);
		}
	}

	void WriteLastTokenLogitsText(const LiteNN::Tensor<LiteNN::CPU>& logits, std::string_view outputPath)
	{
		const auto lastTokenLogits = LiteNN::GGUF::ExtractLastTokenLogits(logits);
		EnsureParentDirectory(std::filesystem::path(outputPath));
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

	std::string_view AttentionExecutionModeName(LiteNN::GGUF::LLaMAAttentionExecutionMode mode)
	{
		switch (mode)
		{
		case LiteNN::GGUF::LLaMAAttentionExecutionMode::ActivePrefix:
			return "active-prefix";
		case LiteNN::GGUF::LLaMAAttentionExecutionMode::PagedAttention:
			return "paged-attention";
		}
		return "unknown";
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
		for (const auto& attentionPlan : plan.attentionExecutionPlans)
		{
			std::cout << "attention_plan name=" << attentionPlan.name
			          << " mode=" << AttentionExecutionModeName(attentionPlan.mode)
			          << " backend=" << attentionPlan.backend << " status=" << attentionPlan.status
			          << " max_context_length=" << attentionPlan.maxContextLength
			          << " page_size_tokens=" << attentionPlan.pageSizeTokens
			          << " uses_paged_kv=" << (attentionPlan.usesPagedKV ? "true" : "false")
			          << " requires_page_table=" << (attentionPlan.requiresPageTable ? "true" : "false")
			          << " materializes_full_mask=" << (attentionPlan.materializesFullMask ? "true" : "false")
			          << " streaming_decode=" << (attentionPlan.streamingDecode ? "true" : "false")
			          << " required_states=";
			PrintStringList(attentionPlan.requiredRuntimeStates);
			std::cout << '\n';
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

#ifdef LITENN_GGUF_CONVERT_ENABLE_AOT
	void InitializePagedKVMetadataInput(const LiteNN::CompiledTensorSpec& input, LiteNN::Tensor<LiteNN::CPU>& tensor)
	{
		if (input.type.dtype != LiteNN::DataType::Int64 || !input.type.IsFullyStatic())
		{
			return;
		}
		const auto shape = input.type.StaticShape();
		auto* values = static_cast<std::int64_t*>(tensor.UnsafeRawData());
		if (input.name.starts_with("page_table_") && shape.size() == 1)
		{
			std::fill_n(values, LiteNN::Detail::Product(shape), LiteNN::Runtime::PagedKVInvalidPage);
			return;
		}
		if (!input.name.starts_with("page_descriptor_") || shape.size() != 2)
		{
			return;
		}
		const auto columnCount = static_cast<std::size_t>(LiteNN::Runtime::PagedKVPageDescriptorColumn::Count);
		if (shape[1] < columnCount)
		{
			throw std::runtime_error(
			    std::format("Paged KV descriptor input {} has too few columns: {}", input.name, shape[1]));
		}
		const auto logicalPageColumn =
		    static_cast<std::size_t>(LiteNN::Runtime::PagedKVPageDescriptorColumn::LogicalPage);
		for (std::size_t page = 0; page < shape[0]; ++page)
		{
			values[page * shape[1] + logicalPageColumn] = LiteNN::Runtime::PagedKVInvalidPage;
		}
	}

	std::vector<LiteNN::Tensor<LiteNN::CPU>> MakeZeroStateInputs(std::span<const LiteNN::CompiledTensorSpec> inputSpecs,
	                                                             LiteNN::Tensor<LiteNN::CPU> tokenIds)
	{
		if (inputSpecs.empty())
		{
			throw std::runtime_error("Compiled LLM module has no inputs");
		}
		if (inputSpecs.front().type.dtype != LiteNN::DataType::Int32 ||
		    inputSpecs.front().type.StaticShape() != tokenIds.Shape())
		{
			throw std::runtime_error("Compiled LLM module first input does not match token-id tensor");
		}

		std::vector<LiteNN::Tensor<LiteNN::CPU>> inputs;
		inputs.push_back(std::move(tokenIds));
		LiteNN::CPU cpu;
		for (std::size_t i = 1; i < inputSpecs.size(); ++i)
		{
			const auto& input = inputSpecs[i];
			if (!input.type.IsFullyStatic())
			{
				throw std::runtime_error(
				    std::format("Compiled LLM module input {} ({}) must have a static shape", i, input.name));
			}
			auto& tensor = inputs.emplace_back(input.type.StaticShape(), input.type.dtype, cpu);
			InitializePagedKVMetadataInput(input, tensor);
		}
		return inputs;
	}
#endif

	LiteNN::Tensor<LiteNN::CPU> MakeTokenIdTensorForPlan(std::int32_t tokenId, const LiteNN::ExecutablePlan& plan)
	{
		const std::array<std::int32_t, 1> ids{ tokenId };
		return MakeTokenIdTensor(ids, plan);
	}

#ifdef LITENN_GGUF_CONVERT_ENABLE_AOT
	LiteNN::Tensor<LiteNN::CPU> MakeTokenIdTensorForModule(std::int32_t tokenId,
	                                                       std::span<const LiteNN::CompiledTensorSpec> inputSpecs)
	{
		if (inputSpecs.empty() || inputSpecs.front().type.dtype != LiteNN::DataType::Int32 ||
		    !inputSpecs.front().type.IsFullyStatic() ||
		    LiteNN::Detail::Product(inputSpecs.front().type.StaticShape()) != 1)
		{
			throw std::runtime_error("Compiled LLM module first input must be a scalar Int32 token id");
		}
		LiteNN::CPU cpu;
		LiteNN::Tensor<LiteNN::CPU> tensor(LiteNN::Uninitialized, inputSpecs.front().type.StaticShape(),
		                                   LiteNN::DataType::Int32, cpu);
		LiteNN::DeviceTraits<LiteNN::CPU>::CopyFromCPU(cpu, LiteNN::DataType::Int32, tensor.UnsafeRawData(),
		                                               LiteNN::DataType::Int32, &tokenId, 1);
		return tensor;
	}
#endif

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

	void StoreScalarBool(LiteNN::Tensor<LiteNN::CPU>& tensor, bool value)
	{
		if (tensor.DType() != LiteNN::DataType::Bool || tensor.NumElements() != 1)
		{
			throw std::runtime_error("decode-loop emit_logits input must be Bool with one element");
		}
		LiteNN::CPU cpu;
		LiteNN::DeviceTraits<LiteNN::CPU>::CopyFromCPU(cpu, LiteNN::DataType::Bool, tensor.UnsafeRawData(),
		                                               LiteNN::DataType::Bool, &value, 1);
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

	void LogGGUFNodeProfile(bool enabled, std::size_t step,
	                        std::span<const LiteNN::CompiledModuleCPUNodeProfileEvent> events)
	{
		if (!enabled || events.empty())
		{
			return;
		}
		double selfMs = 0.0;
		double helperMs = 0.0;
		std::uint64_t calls = 0;
		std::size_t emittedNodes = 0;
		for (const auto& event : events)
		{
			selfMs += event.selfMilliseconds;
			helperMs += event.helperMilliseconds;
			calls += event.calls;
			if (std::max({ event.inclusiveMilliseconds, event.selfMilliseconds, event.helperMilliseconds }) >= 0.001)
			{
				++emittedNodes;
			}
		}
		LogGGUFDiagnostic(enabled, std::format("decode step {} node_profile self_ms={:.3f} helper_ms={:.3f} "
		                                       "calls={} nodes={} emitted_nodes={}",
		                                       step, selfMs, helperMs, calls, events.size(), emittedNodes));
		for (const auto& event : events)
		{
			if (std::max({ event.inclusiveMilliseconds, event.selfMilliseconds, event.helperMilliseconds }) < 0.001)
			{
				continue;
			}
			LogGGUFDiagnostic(
			    enabled,
			    std::format("decode step {} node subgraph={} node={} op={} schema={} calls={} inclusive_ms={:.3f} "
			                "self_ms={:.3f} helper_ms={:.3f}",
			                step, event.subgraphId, event.nodeId, event.opKind, event.schemaId, event.calls,
			                event.inclusiveMilliseconds, event.selfMilliseconds, event.helperMilliseconds));
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

	class MappedReadOnlyFile
	{
	public:
		explicit MappedReadOnlyFile(const std::filesystem::path& path)
		{
#ifdef _WIN32
			file_ = CreateFileW(path.wstring().c_str(), GENERIC_READ,
			                    FILE_SHARE_READ | FILE_SHARE_WRITE | FILE_SHARE_DELETE, nullptr, OPEN_EXISTING,
			                    FILE_ATTRIBUTE_NORMAL, nullptr);
			if (file_ == INVALID_HANDLE_VALUE)
			{
				throw std::runtime_error("failed to open cached artifact file for mapping: " + path.string());
			}
			LARGE_INTEGER size;
			if (!GetFileSizeEx(file_, &size) || size.QuadPart < 0)
			{
				throw std::runtime_error("failed to size cached artifact file for mapping: " + path.string());
			}
			size_ = static_cast<std::size_t>(size.QuadPart);
			if (size_ == 0)
			{
				return;
			}
			mapping_ = CreateFileMappingW(file_, nullptr, PAGE_READONLY, 0, 0, nullptr);
			if (mapping_ == nullptr)
			{
				throw std::runtime_error("failed to create cached artifact file mapping: " + path.string());
			}
			view_ = MapViewOfFile(mapping_, FILE_MAP_READ, 0, 0, 0);
			if (view_ == nullptr)
			{
				throw std::runtime_error("failed to map cached artifact file: " + path.string());
			}
			data_ = view_;
#else
			fd_ = open(path.string().c_str(), O_RDONLY);
			if (fd_ < 0)
			{
				throw std::runtime_error("failed to open cached artifact file for mapping: " + path.string() + ": " +
				                         std::strerror(errno));
			}
			struct stat st;
			if (fstat(fd_, &st) != 0 || st.st_size < 0)
			{
				throw std::runtime_error("failed to size cached artifact file for mapping: " + path.string() + ": " +
				                         std::strerror(errno));
			}
			size_ = static_cast<std::size_t>(st.st_size);
			if (size_ == 0)
			{
				return;
			}
			view_ = mmap(nullptr, size_, PROT_READ, MAP_PRIVATE, fd_, 0);
			if (view_ == MAP_FAILED)
			{
				view_ = nullptr;
				throw std::runtime_error("failed to map cached artifact file: " + path.string() + ": " +
				                         std::strerror(errno));
			}
			data_ = view_;
#endif
		}

		MappedReadOnlyFile(const MappedReadOnlyFile&) = delete;
		MappedReadOnlyFile& operator=(const MappedReadOnlyFile&) = delete;

		~MappedReadOnlyFile()
		{
#ifdef _WIN32
			if (view_ != nullptr)
			{
				UnmapViewOfFile(view_);
			}
			if (mapping_ != nullptr)
			{
				CloseHandle(mapping_);
			}
			if (file_ != nullptr && file_ != INVALID_HANDLE_VALUE)
			{
				CloseHandle(file_);
			}
#else
			if (view_ != nullptr)
			{
				munmap(view_, size_);
			}
			if (fd_ >= 0)
			{
				close(fd_);
			}
#endif
		}

		LiteNN::CompiledModuleRegion Region() const
		{
			return { .data = data_, .size = size_ };
		}

	private:
		const void* data_{};
		std::size_t size_{};
#ifdef _WIN32
		HANDLE file_{ INVALID_HANDLE_VALUE };
		HANDLE mapping_{};
		void* view_{};
#else
		int fd_{ -1 };
		void* view_{};
#endif
	};

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

	std::string_view CPUAOTGGMLPrepackedWeightLayoutName(LiteNN::CPUAOTGGMLPrepackedWeightLayout layout)
	{
		switch (layout)
		{
		case LiteNN::CPUAOTGGMLPrepackedWeightLayout::ExpandedF32ScalesV1:
			return "expanded_f32_scales_v1";
		case LiteNN::CPUAOTGGMLPrepackedWeightLayout::CompactBlockGroupedV3:
			return "compact_block_grouped_v3";
		case LiteNN::CPUAOTGGMLPrepackedWeightLayout::FieldInterleavedV4:
			return "field_interleaved_v4";
		}
		return "unknown";
	}

	std::optional<std::filesystem::path>
	DecodeAOTCachePath(std::string_view modelPath, std::size_t requestedTokenCount,
	                   const LiteNN::CompilerOptions& options, std::string_view decodeMode,
	                   std::optional<std::size_t> pagedResidentPageCount = std::nullopt)
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
		const auto residentPagesText =
		    pagedResidentPageCount ? std::to_string(*pagedResidentPageCount) : std::string("auto");
		constexpr std::uint32_t decodePlanCacheVersion = 8;
		const auto keyText = std::format(
		    "gguf-decode-{}-v{}|cpu_aot_compilation_v{}|{}|{}|{}|tokens={}|opt={}|external={}|threads={}|"
		    "affinity={}|worker_wait={}|min_flops={}|node_profile={}|"
		    "q8k_staged={}|ggml_prepacked_weights={}|ggml_prepacked_weight_policy={}|ggml_prepacked_layout={}|"
		    "paged_resident_pages={}",
		    decodeMode, decodePlanCacheVersion, LiteNN::CPUAOTCompilationCacheVersion,
		    std::filesystem::absolute(model, ec).string(), modelSize, lastWrite, requestedTokenCount,
		    options.cpuAOTLLVMOptLevel, options.enableCPUAOTExternalRegions ? 1 : 0, options.cpuAOTThreadCount,
		    static_cast<std::uint32_t>(options.cpuAOTAffinityPolicy),
		    static_cast<std::uint32_t>(options.cpuAOTWorkerWaitPolicy), options.cpuAOTParallelMinFlops,
		    options.enableCPUAOTNodeProfiling ? 1 : 0, options.enableCPUAOTGGMLQ8KStagedMatMul ? 1 : 0,
		    options.enableCPUAOTGGMLPrepackedWeights ? 1 : 0,
		    static_cast<std::uint32_t>(options.cpuAOTGGMLPrepackedWeightPolicy),
		    CPUAOTGGMLPrepackedWeightLayoutName(options.cpuAOTGGMLPrepackedWeightLayout), residentPagesText);
		return std::filesystem::path(root) / std::format("{:016x}", FNV1a(keyText));
	}

	std::optional<std::filesystem::path> DecodeAOTSharedWeightsPath(std::string_view modelPath,
	                                                                const LiteNN::CompilerOptions& options)
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
		const auto keyText = std::format("gguf-shared-weights-v3|{}|{}|{}|ggml_prepacked_weights={}|"
		                                 "ggml_prepacked_weight_policy={}|ggml_prepacked_layout={}",
		                                 std::filesystem::absolute(model, ec).string(), modelSize, lastWrite,
		                                 options.enableCPUAOTGGMLPrepackedWeights ? 1 : 0,
		                                 static_cast<std::uint32_t>(options.cpuAOTGGMLPrepackedWeightPolicy),
		                                 CPUAOTGGMLPrepackedWeightLayoutName(options.cpuAOTGGMLPrepackedWeightLayout));
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
			if (auto referencedWeights = DecodeAOTReferencedWeightsPath(files))
			{
				LogGGUFDiagnostic(diagnostics, "gguf decode aot cache: mapping shared weights " +
				                                   referencedWeights->generic_string());
				auto mappedWeights = std::make_shared<MappedReadOnlyFile>(*referencedWeights);
				std::shared_ptr<const void> mappedOwner = mappedWeights;
				auto artifact = LiteNN::CompiledModuleSeparatedArtifact::FromOwnedRegionsWithTrustedBorrowedWeights(
				    ReadBinaryFile(files.metadata), ReadBinaryFile(files.constants), mappedWeights->Region(),
				    mappedOwner, ReadBinaryFile(files.instructions));
				const auto expectedIdentity =
				    DecodeAOTSharedWeightsIdentity(mappedWeights->Region().size, artifact.ExternalTensorInfos());
				if (referencedWeights->parent_path().filename() != expectedIdentity)
				{
					throw std::runtime_error("gguf decode aot cache shared weight layout identity mismatch");
				}
				return artifact;
			}
			return LiteNN::CompiledModuleSeparatedArtifact::FromOwnedRegions(
			    ReadBinaryFile(files.metadata), ReadBinaryFile(files.constants), ReadBinaryFile(files.weights),
			    ReadBinaryFile(files.instructions));
		});
		LogGGUFDiagnostic(diagnostics, "gguf decode aot cache: hit");
		return TimedGGUFDiagnostic(diagnostics, "gguf decode aot cache load module",
		                           [&] { return std::move(artifact).LoadBorrowedExternalRegions(); });
	}

	std::optional<LiteNN::CompiledModule<LiteNN::CPU>>
	TryLoadDecodeAOTCache(const std::optional<std::filesystem::path>& cachePath, bool diagnostics)
	{
		if (!cachePath)
		{
			return std::nullopt;
		}
		if (!DecodeAOTCacheComplete(*cachePath))
		{
			LogGGUFDiagnostic(diagnostics, "gguf decode aot cache: miss");
			return std::nullopt;
		}
		try
		{
			return LoadDecodeAOTCache(*cachePath, diagnostics);
		}
		catch (const std::exception& ex)
		{
			LogGGUFDiagnostic(diagnostics, std::format("gguf decode aot cache: ignored invalid cache ({})", ex.what()));
			return std::nullopt;
		}
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
		std::error_code removeError;
		std::filesystem::remove(files.complete, removeError);
		if (removeError)
		{
			throw std::runtime_error("Failed to invalidate gguf decode aot cache complete marker: " +
			                         removeError.message());
		}
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
			const auto resolvedSharedWeightsPath =
			    sharedWeightsPath->parent_path() /
			    DecodeAOTSharedWeightsIdentity(artifact.Weights().size(), artifact.ExternalTensorInfos()) /
			    "weights.bin";
			const auto publishResult = PublishDecodeAOTSharedWeightsAtomically(
			    resolvedSharedWeightsPath, artifact.Weights().size(),
			    [&](const std::filesystem::path& stagingWeights, const std::filesystem::path& stagingComplete) {
				    WriteBinaryFileTimed(stagingWeights, artifact.Weights(), diagnostics,
				                         "gguf decode aot shared weight store write weights");
				    WriteBinaryFileTimed(stagingComplete, std::span<const std::byte>{}, diagnostics,
				                         "gguf decode aot shared weight store write complete marker");
			    });
			if (publishResult == SharedWeightsPublishResult::Reused)
			{
				LogGGUFDiagnostic(diagnostics, "gguf decode aot shared weight store: reused " +
				                                   resolvedSharedWeightsPath.generic_string());
			}
			WriteTextFileTimed(files.weightReference, std::filesystem::absolute(resolvedSharedWeightsPath).string(),
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
	                          const std::optional<std::filesystem::path>& sharedWeightsPath, bool diagnostics,
	                          bool cacheAlreadyChecked = false)
	{
		if (!cacheAlreadyChecked)
		{
			if (auto cached = TryLoadDecodeAOTCache(cachePath, diagnostics))
			{
				return std::move(*cached);
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
	                          const std::optional<std::filesystem::path>& sharedWeightsPath, bool diagnostics,
	                          bool cacheAlreadyChecked = false)
	{
		if (!cacheAlreadyChecked)
		{
			if (auto cached = TryLoadDecodeAOTCache(cachePath, diagnostics))
			{
				return std::move(*cached);
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
		auto metadataImport = TimedGGUFDiagnostic(diagnostics, "gguf import metadata",
		                                          [&] { return LiteNN::GGUF::ImportGGUFMetadata(options.inputPath); });
		const auto importSummary = metadataImport.summary;
		auto initialTokenIds = options.initialTokenIds;
		if (options.exactPrompt)
		{
			initialTokenIds = TimedGGUFDiagnostic(diagnostics, "gguf tokenize exact prompt", [&] {
				return TokenizePrompt(options.inputPath, *options.exactPrompt, metadataImport.model.UnsafeGraphView(),
				                      options.applyChatTemplate);
			});
		}
		if (initialTokenIds.empty())
		{
			throw std::runtime_error("decode-loop requires at least one initial token");
		}
		const auto hyperparameters = LiteNN::GGUF::ParseLLaMAHyperparameters(metadataImport.model.UnsafeGraphView());
		const auto tokenizer = LiteNN::GGUF::SummarizeLLMTokenizerMetadata(metadataImport.model.UnsafeGraphView());
		auto tokenPieces = CopyTokenPieces(metadataImport.model.UnsafeGraphView());
		metadataImport = {};
		const auto requestedTokenCount = initialTokenIds.size() + options.steps;
		const auto maxCacheLength = options.maxCacheLength.value_or(requestedTokenCount);
		const auto contextValidation =
		    LiteNN::GGUF::ValidateLLaMAContextRequest(hyperparameters, requestedTokenCount, maxCacheLength);
		LogGGUFDiagnostic(diagnostics,
		                  std::format("decode-loop context model={} trained={} requested={} max_cache={} "
		                              "rope_scaling={} extension={}",
		                              contextValidation.modelContextLength, contextValidation.trainedContextLength,
		                              requestedTokenCount, maxCacheLength, contextValidation.ropeScalingType,
		                              contextValidation.usesContextExtension ? "true" : "false"));
		for (const auto& diagnostic : contextValidation.diagnostics)
		{
			LogGGUFDiagnostic(diagnostics,
			                  std::format("decode-loop context diagnostic blocking={} subject={} message={}",
			                              diagnostic.blocking ? "true" : "false", diagnostic.subject,
			                              diagnostic.message));
		}
		if (!contextValidation.accepted)
		{
			const auto blocking = std::ranges::find_if(
			    contextValidation.diagnostics,
			    [](const LiteNN::GGUF::LLaMACompatibilityDiagnostic& diagnostic) { return diagnostic.blocking; });
			throw std::runtime_error(blocking == contextValidation.diagnostics.end()
			                             ? "decode-loop context validation failed"
			                             : "decode-loop context validation failed: " + blocking->message);
		}
		if (options.pagedResidentPageCount && !options.pagedReferenceDecode)
		{
			throw std::runtime_error("--paged-resident-pages requires --paged-reference-decode");
		}
		LogGGUFDiagnostic(diagnostics,
		                  std::format("decode-loop tokens prompt={} generated_request={} requested_token_count={} "
		                              "max_cache_length={} paged_resident_pages={}",
		                              initialTokenIds.size(), options.steps, requestedTokenCount, maxCacheLength,
		                              options.pagedResidentPageCount ? std::to_string(*options.pagedResidentPageCount)
		                                                             : std::string("auto")));
		const auto maxRunCount = requestedTokenCount - 1;
		const auto buildStart = std::chrono::steady_clock::now();
		const std::string_view decodeMode = options.pagedReferenceDecode ? "paged_reference"
		                                    : options.statefulDecode     ? "stateful"
		                                                                 : "functional";
		LiteNN::ExecutablePlan decodePlan;
		std::optional<LiteNN::GGUF::ImportResult> imported;
		const auto ensureTensorPayloadsImported = [&]() -> LiteNN::GGUF::ImportResult& {
			if (!imported)
			{
				imported.emplace(TimedGGUFDiagnostic(diagnostics, "gguf import tensor payloads", [&] {
					return LiteNN::GGUF::ImportGGUFArchive(options.inputPath);
				}));
			}
			return *imported;
		};
		LiteNN::CompiledModule<LiteNN::CPU> decodeModule = [&] {
			if (options.statefulDecode)
			{
				const auto cachePath = DecodeAOTCachePath(options.inputPath, maxCacheLength, compilerOptions,
				                                          decodeMode, options.pagedResidentPageCount);
				const auto sharedWeightsPath = DecodeAOTSharedWeightsPath(options.inputPath, compilerOptions);
				if (auto cached = TryLoadDecodeAOTCache(cachePath, diagnostics))
				{
					LogGGUFDiagnostic(diagnostics,
					                  "gguf decode aot cache hit skipped graph and tensor payload materialization");
					return std::move(*cached);
				}
				ThrowDecodeAOTCacheMiss(cachePath);
				auto& archive = ensureTensorPayloadsImported();
				auto schedule = TimedGGUFDiagnostic(diagnostics, "gguf build stateful decode runtime schedule", [&] {
					return LiteNN::GGUF::BuildLLaMADecodeRuntimeSchedule(
					    archive.model.UnsafeGraphView(), { .prefillSequenceLength = 1,
					                                       .decodePastLength = 0,
					                                       .maxCacheLength = maxCacheLength,
					                                       .preserveQuantizedWeights = true,
					                                       .dynamicDecodePosition = true,
					                                       .conditionalLogits = true,
					                                       .usePagedReferenceDecode = options.pagedReferenceDecode,
					                                       .pagedResidentPageCount = options.pagedResidentPageCount });
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
				return TimedGGUFDiagnostic(diagnostics, "gguf load-or-compile cpu aot stateful decode module", [&] {
					return LoadOrCompileDecodeModule(schedule, compilerOptions, cachePath, sharedWeightsPath,
					                                 diagnostics, true);
				});
			}

			auto& archive = ensureTensorPayloadsImported();
			auto graph = TimedGGUFDiagnostic(diagnostics, "gguf lower decode-capacity graph", [&] {
				return LiteNN::GGUF::LowerLLaMACausalLMDecodeCapacity(archive.model.UnsafeGraphView(), maxCacheLength,
				                                                      { .preserveQuantizedWeights = true });
			});
			decodePlan = TimedGGUFDiagnostic(diagnostics, "gguf build executable plan",
			                                 [&] { return LiteNN::Detail::BuildExecutablePlanFromGraph(graph); });
			const auto cachePath = DecodeAOTCachePath(options.inputPath, maxCacheLength, compilerOptions, decodeMode);
			const auto sharedWeightsPath = DecodeAOTSharedWeightsPath(options.inputPath, compilerOptions);
			return TimedGGUFDiagnostic(diagnostics, "gguf load-or-compile cpu aot decode module", [&] {
				return LoadOrCompileDecodeModule(decodePlan, compilerOptions, cachePath, sharedWeightsPath,
				                                 diagnostics);
			});
		}();
		const auto planInputCount =
		    decodePlan.inputs.empty() ? decodeModule.InputSpecs().size() : decodePlan.inputs.size();
		const auto planOutputCount =
		    decodePlan.outputs.empty() ? decodeModule.OutputSpecs().size() : decodePlan.outputs.size();
		const auto planVariableCount =
		    decodePlan.variables.empty() ? importSummary.tensorCount : decodePlan.variables.size();
		LogGGUFDiagnostic(diagnostics, std::format("decode-runtime inputs={} outputs={} variables={}", planInputCount,
		                                           planOutputCount, planVariableCount));
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
			          << " inputs=" << planInputCount << " outputs=" << planOutputCount
			          << " variables=" << planVariableCount << '\n';
			return;
		}
		if (imported)
		{
			imported.reset();
			LogGGUFDiagnostic(diagnostics, "decode-loop released imported GGUF tensor payloads before token execution");
		}

		LiteNN::GGUF::LLMSamplerState sampler{ .config = options.sampling };
		std::vector<std::int32_t> history = initialTokenIds;
		std::vector<LiteNN::Tensor<LiteNN::CPU>> caches;
		std::optional<LiteNN::Tensor<LiteNN::CPU>> currentPosition;
		if (!options.statefulDecode)
		{
			currentPosition.emplace(MakeDecodePositionTensor(0, decodePlan));
		}
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
		std::vector<LiteNN::Tensor<LiteNN::CPU>> statefulOutputs;
		std::optional<std::size_t> emitLogitsInputIndex;
		if (options.statefulDecode)
		{
			statefulInputs = MakeZeroStateInputs(decodeModule.InputSpecs(),
			                                     MakeTokenIdTensorForModule(currentToken, decodeModule.InputSpecs()));
			const auto inputSpecs = decodeModule.InputSpecs();
			const auto emitLogits =
			    std::ranges::find_if(inputSpecs, [](const auto& input) { return input.name == "emit_logits"; });
			if (emitLogits == inputSpecs.end())
			{
				throw std::runtime_error("stateful decode module is missing the emit_logits control input");
			}
			emitLogitsInputIndex = static_cast<std::size_t>(emitLogits - inputSpecs.begin());
			statefulOutputs.reserve(decodeModule.OutputSpecs().size());
			for (const auto& spec : decodeModule.OutputSpecs())
			{
				if (!spec.type.IsFullyStatic())
				{
					throw std::runtime_error("stateful decode module output must have a static shape");
				}
				statefulOutputs.emplace_back(LiteNN::Uninitialized, LiteNN::ShapeView{ spec.type.StaticShape() },
				                             spec.type.dtype, LiteNN::CPU{});
			}
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
			double inputPrepMs = 0.0;
			double moduleRunMs = 0.0;
			double helperTotalMs = 0.0;
			double moduleNonHelperMs = 0.0;
			double nodeSelfTotalMs = 0.0;
			double nodeInstrumentationMs = 0.0;
			double moduleUnattributedMs = 0.0;
			double helperProfileEmitMs = 0.0;
			double logitsOutputMs = 0.0;
			double samplingMs = 0.0;
			double stateUpdateMs = 0.0;
			std::vector<LiteNN::Tensor<LiteNN::CPU>> allocatedOutputs;
			std::span<LiteNN::Tensor<LiteNN::CPU>> outputs;
			std::optional<LiteNN::CompiledModuleCPUHelperProfiler> helperProfiler;
			if (options.profileHelpers || options.profileNodes)
			{
				helperProfiler.emplace();
			}
			if (options.statefulDecode)
			{
				const auto inputPrepStart = std::chrono::steady_clock::now();
				StoreScalarTokenId(statefulInputs.front(), currentToken);
				StoreScalarBool(statefulInputs[*emitLogitsInputIndex], !isPromptReplayStep);
				const auto inputPrepEnd = std::chrono::steady_clock::now();
				inputPrepMs = std::chrono::duration<double, std::milli>(inputPrepEnd - inputPrepStart).count();
				const auto moduleRunStart = std::chrono::steady_clock::now();
				decodeModule.RunTensorsInto(statefulInputs, statefulOutputs);
				const auto moduleRunEnd = std::chrono::steady_clock::now();
				moduleRunMs = std::chrono::duration<double, std::milli>(moduleRunEnd - moduleRunStart).count();
				outputs = statefulOutputs;
			}
			else
			{
				const auto inputPrepStart = std::chrono::steady_clock::now();
				std::vector<LiteNN::Tensor<LiteNN::CPU>> inputs;
				inputs.push_back(MakeTokenIdTensorForPlan(currentToken, decodePlan));
				inputs.push_back(std::move(*currentPosition));
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
				const auto inputPrepEnd = std::chrono::steady_clock::now();
				inputPrepMs = std::chrono::duration<double, std::milli>(inputPrepEnd - inputPrepStart).count();
				const auto moduleRunStart = std::chrono::steady_clock::now();
				allocatedOutputs = decodeModule.RunTensors(inputs);
				const auto moduleRunEnd = std::chrono::steady_clock::now();
				moduleRunMs = std::chrono::duration<double, std::milli>(moduleRunEnd - moduleRunStart).count();
				outputs = allocatedOutputs;
			}
			if (helperProfiler)
			{
				const auto helperProfileEmitStart = std::chrono::steady_clock::now();
				const auto helperEvents = helperProfiler->Snapshot();
				const auto nodeEvents = helperProfiler->SnapshotNodes();
				nodeInstrumentationMs = helperProfiler->NodeInstrumentationMilliseconds();
				for (const auto& event : helperEvents)
				{
					helperTotalMs += event.totalMilliseconds;
				}
				for (const auto& event : nodeEvents)
				{
					nodeSelfTotalMs += event.selfMilliseconds;
				}
				moduleNonHelperMs = moduleRunMs >= helperTotalMs ? moduleRunMs - helperTotalMs : 0.0;
				const auto attributedModuleMs = helperTotalMs + nodeSelfTotalMs + nodeInstrumentationMs;
				moduleUnattributedMs = moduleRunMs >= attributedModuleMs ? moduleRunMs - attributedModuleMs : 0.0;
				LogGGUFHelperProfile(options.profileHelpers || options.profileNodes, step + 1, helperEvents);
				LogGGUFNodeProfile(options.profileNodes, step + 1, nodeEvents);
				const auto helperProfileEmitEnd = std::chrono::steady_clock::now();
				helperProfileEmitMs =
				    std::chrono::duration<double, std::milli>(helperProfileEmitEnd - helperProfileEmitStart).count();
			}
			const auto logitsOutputStart = std::chrono::steady_clock::now();
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
			const auto logitsOutputEnd = std::chrono::steady_clock::now();
			logitsOutputMs = std::chrono::duration<double, std::milli>(logitsOutputEnd - logitsOutputStart).count();
			const auto samplingStart = std::chrono::steady_clock::now();
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
			const auto samplingEnd = std::chrono::steady_clock::now();
			samplingMs = std::chrono::duration<double, std::milli>(samplingEnd - samplingStart).count();
			const auto stateUpdateStart = std::chrono::steady_clock::now();
			lastOutputCount = outputs.size();
			if (!options.statefulDecode)
			{
				currentPosition.emplace(std::move(outputs[1]));
				caches.reserve(outputs.size() - 2);
				for (std::size_t i = 2; i < outputs.size(); ++i)
				{
					caches.push_back(std::move(outputs[i]));
				}
			}
			const auto stateUpdateEnd = std::chrono::steady_clock::now();
			stateUpdateMs = std::chrono::duration<double, std::milli>(stateUpdateEnd - stateUpdateStart).count();
			const auto stepEnd = std::chrono::steady_clock::now();
			const auto stepMs = std::chrono::duration<double, std::milli>(stepEnd - stepStart).count();
			const auto accountedStepMs =
			    inputPrepMs + moduleRunMs + helperProfileEmitMs + logitsOutputMs + samplingMs + stateUpdateMs;
			const auto hostOverheadMs = accountedStepMs >= stepMs ? 0.0 : stepMs - accountedStepMs;
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
				          << " input_prep_ms=" << inputPrepMs << " module_run_ms=" << moduleRunMs
				          << " helper_profile_enabled=" << (helperProfiler ? "true" : "false")
				          << " node_profile_enabled=" << (options.profileNodes ? "true" : "false");
				if (helperProfiler)
				{
					std::cout << " helper_total_ms=" << helperTotalMs << " module_non_helper_ms=" << moduleNonHelperMs
					          << " helper_profile_emit_ms=" << helperProfileEmitMs;
					if (options.profileNodes)
					{
						std::cout << " node_self_total_ms=" << nodeSelfTotalMs
						          << " node_instrumentation_ms=" << nodeInstrumentationMs
						          << " module_unattributed_ms=" << moduleUnattributedMs;
					}
				}
				std::cout << " logits_output_ms=" << logitsOutputMs << " sampling_ms=" << samplingMs
				          << " state_update_ms=" << stateUpdateMs << " host_overhead_ms=" << hostOverheadMs
				          << " prompt_replay_steps=" << promptReplayStepCount << " prompt_replay_ms=" << promptReplayMs
				          << " generation_steps=" << generationStepCount << " generation_ms=" << generationMs
				          << " generated_tokens=" << generatedTokenCount
				          << " generated_tokens_per_second=" << liveTokensPerSecond
				          << " eos=" << (stoppedOnEos ? "true" : "false") << '\n';
				std::cout.flush();
			}
			if (helperProfiler)
			{
				LogGGUFDiagnostic(diagnostics,
				                  std::format("decode step {} buckets input_prep_ms={:.3f} module_run_ms={:.3f} "
				                              "helper_profile_enabled=true helper_total_ms={:.3f} "
				                              "module_non_helper_ms={:.3f} helper_profile_emit_ms={:.3f} "
				                              "logits_output_ms={:.3f} sampling_ms={:.3f} state_update_ms={:.3f} "
				                              "host_overhead_ms={:.3f}",
				                              step + 1, inputPrepMs, moduleRunMs, helperTotalMs, moduleNonHelperMs,
				                              helperProfileEmitMs, logitsOutputMs, samplingMs, stateUpdateMs,
				                              hostOverheadMs));
				if (options.profileNodes)
				{
					LogGGUFDiagnostic(
					    diagnostics,
					    std::format("decode step {} node buckets self_ms={:.3f} instrumentation_ms={:.3f} "
					                "module_unattributed_ms={:.3f}",
					                step + 1, nodeSelfTotalMs, nodeInstrumentationMs, moduleUnattributedMs));
				}
			}
			else
			{
				LogGGUFDiagnostic(diagnostics,
				                  std::format("decode step {} buckets input_prep_ms={:.3f} module_run_ms={:.3f} "
				                              "helper_profile_enabled=false logits_output_ms={:.3f} sampling_ms={:.3f} "
				                              "state_update_ms={:.3f} host_overhead_ms={:.3f}",
				                              step + 1, inputPrepMs, moduleRunMs, logitsOutputMs, samplingMs,
				                              stateUpdateMs, hostOverheadMs));
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
			EnsureParentDirectory(std::filesystem::path(*options.outputPath));
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

	LiteNN::CPUAOTGGMLPrepackedWeightPolicy ToCompilerPrepackedWeightPolicy(std::string_view policy)
	{
		if (policy == "disabled")
		{
			return LiteNN::CPUAOTGGMLPrepackedWeightPolicy::Disabled;
		}
		if (policy == "profitable")
		{
			return LiteNN::CPUAOTGGMLPrepackedWeightPolicy::Profitable;
		}
		if (policy == "all")
		{
			return LiteNN::CPUAOTGGMLPrepackedWeightPolicy::All;
		}
		throw std::runtime_error("unsupported CPU AOT GGML prepacked weight policy");
	}

	LiteNN::CPUAOTGGMLPrepackedWeightLayout ToCompilerPrepackedWeightLayout(std::string_view layout)
	{
		if (layout == "expanded_f32_scales_v1")
		{
			return LiteNN::CPUAOTGGMLPrepackedWeightLayout::ExpandedF32ScalesV1;
		}
		if (layout == "compact_block_grouped_v3")
		{
			return LiteNN::CPUAOTGGMLPrepackedWeightLayout::CompactBlockGroupedV3;
		}
		if (layout == "field_interleaved_v4")
		{
			return LiteNN::CPUAOTGGMLPrepackedWeightLayout::FieldInterleavedV4;
		}
		throw std::runtime_error("unsupported CPU AOT GGML prepacked weight layout");
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
		if (const char* workerWait = std::getenv("LITENN_CPU_AOT_WORKER_WAIT"))
		{
			const std::string_view value{ workerWait };
			if (value == "low-power")
			{
				options.cpuAOTWorkerWaitPolicy = LiteNN::CPUAOTWorkerWaitPolicy::LowPower;
			}
			else if (value == "latency")
			{
				options.cpuAOTWorkerWaitPolicy = LiteNN::CPUAOTWorkerWaitPolicy::Latency;
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
		options.enableCPUAOTGGMLPrepackedWeights = TruthyEnvValue(std::getenv("LITENN_CPU_AOT_GGML_PREPACKED_WEIGHTS"));
		if (const char* policy = std::getenv("LITENN_CPU_AOT_GGML_PREPACKED_WEIGHT_POLICY"))
		{
			options.cpuAOTGGMLPrepackedWeightPolicy =
			    ToCompilerPrepackedWeightPolicy(ParseGGMLPrepackedWeightPolicy(policy));
			if (options.cpuAOTGGMLPrepackedWeightPolicy != LiteNN::CPUAOTGGMLPrepackedWeightPolicy::Disabled)
			{
				options.enableCPUAOTExternalRegions = true;
			}
		}
		if (const char* layout = std::getenv("LITENN_CPU_AOT_GGML_PREPACKED_WEIGHT_LAYOUT"))
		{
			options.cpuAOTGGMLPrepackedWeightLayout =
			    ToCompilerPrepackedWeightLayout(ParseGGMLPrepackedWeightLayout(layout));
		}
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
			compilerOptions.cpuAOTAffinityPolicy =
			    *decodeOptions.cpuAOTAffinityPolicy == "compact"
			        ? LiteNN::CPUAOTAffinityPolicy::Compact
			        : (*decodeOptions.cpuAOTAffinityPolicy == "spread" ? LiteNN::CPUAOTAffinityPolicy::Spread
			                                                           : LiteNN::CPUAOTAffinityPolicy::None);
		}
		if (decodeOptions.cpuAOTWorkerWaitPolicy)
		{
			compilerOptions.cpuAOTWorkerWaitPolicy =
			    *decodeOptions.cpuAOTWorkerWaitPolicy == "low-power"
			        ? LiteNN::CPUAOTWorkerWaitPolicy::LowPower
			        : (*decodeOptions.cpuAOTWorkerWaitPolicy == "latency" ? LiteNN::CPUAOTWorkerWaitPolicy::Latency
			                                                              : LiteNN::CPUAOTWorkerWaitPolicy::Adaptive);
		}
		if (decodeOptions.enableCompileDiagnostics)
		{
			compilerOptions.enableCompileDiagnostics = *decodeOptions.enableCompileDiagnostics;
		}
		compilerOptions.enableCPUAOTNodeProfiling = decodeOptions.profileNodes;
		if (decodeOptions.enableCPUAOTQ8KStagedMatMul)
		{
			compilerOptions.enableCPUAOTGGMLQ8KStagedMatMul = true;
		}
		if (decodeOptions.enableCPUAOTGGMLPrepackedWeights)
		{
			compilerOptions.enableCPUAOTGGMLPrepackedWeights = true;
			compilerOptions.enableCPUAOTExternalRegions = true;
		}
		if (decodeOptions.cpuAOTGGMLPrepackedWeightPolicy)
		{
			compilerOptions.cpuAOTGGMLPrepackedWeightPolicy =
			    ToCompilerPrepackedWeightPolicy(*decodeOptions.cpuAOTGGMLPrepackedWeightPolicy);
			if (compilerOptions.cpuAOTGGMLPrepackedWeightPolicy != LiteNN::CPUAOTGGMLPrepackedWeightPolicy::Disabled)
			{
				compilerOptions.enableCPUAOTExternalRegions = true;
			}
		}
		if (decodeOptions.cpuAOTGGMLPrepackedWeightLayout)
		{
			compilerOptions.cpuAOTGGMLPrepackedWeightLayout =
			    ToCompilerPrepackedWeightLayout(*decodeOptions.cpuAOTGGMLPrepackedWeightLayout);
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
