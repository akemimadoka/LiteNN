#include "GGMLQuantizedKernels.h"
#include "GGUFImporter.h"
#include "LLMGeneration.h"
#include "LLaMABuilder.h"

#ifdef LITENN_GGUF_CONVERT_ENABLE_AOT
#include <LiteNN/Compiler/CompiledModule.h>
#endif
#include <LiteNN/Runtime/Interpreter.h>
#include <LiteNN/Serialization/ModelPackageIO.h>

#include <algorithm>
#include <array>
#include <charconv>
#include <chrono>
#include <cstdint>
#include <cstdlib>
#include <format>
#include <fstream>
#include <iostream>
#include <limits>
#include <optional>
#include <span>
#include <stdexcept>
#include <string>
#include <string_view>
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
		          << "  " << executable << " --run-llama-prompt <input.gguf> <prompt> [position-offset]\n"
		          << "  " << executable << " --run-llama-package-token-ids <input.ltnn> <comma-token-ids>\n"
		          << "  " << executable
		          << " --run-llama-decode-loop-token-id <input.gguf> <initial-token-id> <steps> [output.txt] "
		             "[--sample greedy|random] [--temperature T] [--top-k K] [--top-p P] [--repeat-penalty R] "
		             "[--seed N] [--ignore-eos]\n"
		          << "  " << executable
		          << " --run-llama-prompt-decode-loop <input.gguf> <prompt> <steps> [output.txt] "
		             "[--sample greedy|random] [--temperature T] [--top-k K] [--top-p P] [--repeat-penalty R] "
		             "[--seed N] [--ignore-eos]\n"
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
		std::size_t steps{};
		std::optional<std::string> outputPath;
		LiteNN::GGUF::LLMSamplingConfig sampling;
		bool stopAtEos{ true };
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

	std::string TokenPiecesText(const LiteNN::Graph& archive, std::span<const std::int32_t> tokenIds)
	{
		const auto* tokens = archive.FindMetadata("tokenizer.ggml.tokens");
		const auto* tokenList = tokens == nullptr ? nullptr : std::get_if<std::vector<std::string>>(&tokens->value);
		std::string text = "[";
		for (std::size_t i = 0; i < tokenIds.size(); ++i)
		{
			if (i != 0)
			{
				text += ',';
			}
			if (tokenList == nullptr)
			{
				text += "\"<missing-tokenizer.ggml.tokens>\"";
			}
			else if (tokenIds[i] < 0 || static_cast<std::size_t>(tokenIds[i]) >= tokenList->size())
			{
				text += "\"<out-of-range>\"";
			}
			else
			{
				text += EscapeTokenPiece((*tokenList)[static_cast<std::size_t>(tokenIds[i])]);
			}
		}
		text += "]";
		return text;
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

	void PrintLLMArtifactPlan(const LiteNN::GGUF::LLaMAArtifactPlan& plan)
	{
		std::cout << "LLM artifact plan architecture=" << plan.hyperparameters.architecture
		          << " dtype=" << LiteNN::DataTypeName(plan.dtype) << " vocab=" << plan.vocabSize
		          << " blocks=" << plan.hyperparameters.blockCount << '\n';
		const auto printEntry = [](const LiteNN::GGUF::LLaMAArtifactEntry& entry) {
			std::cout << "entry name=" << entry.name << " sequence_length=" << entry.sequenceLength
			          << " past_length=" << entry.pastLength << " max_cache_length=" << entry.maxCacheLength
			          << " inputs=";
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

	void RunDecodeLoopFromGGUF(const DecodeLoopCommandOptions& options)
	{
		if (options.steps == 0)
		{
			throw std::runtime_error("decode-loop steps must be positive");
		}
		const auto imported = LiteNN::GGUF::ImportGGUFArchive(options.inputPath);
		auto initialTokenIds = options.initialTokenIds;
		if (options.exactPrompt)
		{
			initialTokenIds =
			    LiteNN::GGUF::MakeExactVocabularyPromptTokens(*options.exactPrompt, imported.model.UnsafeGraphView())
			        .tokenIds;
		}
		if (initialTokenIds.empty())
		{
			throw std::runtime_error("decode-loop requires at least one initial token");
		}
		const auto hyperparameters = LiteNN::GGUF::ParseLLaMAHyperparameters(imported.model.UnsafeGraphView());
		const auto tokenizer = LiteNN::GGUF::SummarizeLLMTokenizerMetadata(imported.model.UnsafeGraphView());
		const auto requestedTokenCount = initialTokenIds.size() + options.steps;
		if (hyperparameters.contextLength > 0 && requestedTokenCount > hyperparameters.contextLength)
		{
			throw std::runtime_error(std::format("decode-loop requested {} total tokens but model context length is {}",
			                                     requestedTokenCount, hyperparameters.contextLength));
		}
		const auto maxRunCount = requestedTokenCount - 1;
		std::vector<LiteNN::ExecutablePlan> decodePlans;
		decodePlans.reserve(maxRunCount);
		const auto buildStart = std::chrono::steady_clock::now();
		for (std::size_t step = 0; step < maxRunCount; ++step)
		{
			const auto pastLength = step + 1;
			auto graph = LiteNN::GGUF::LowerLLaMACausalLMDecode(imported.model.UnsafeGraphView(), 1, pastLength,
			                                                    pastLength, { .preserveQuantizedWeights = true });
			decodePlans.push_back(LiteNN::Detail::BuildExecutablePlanFromGraph(graph));
		}
		const auto buildEnd = std::chrono::steady_clock::now();

		LiteNN::Runtime::Interpreter<LiteNN::CPU> interpreter(LiteNN::GGUF::TryEvalGGMLQuantizedMatMul);
		LiteNN::GGUF::LLMSamplerState sampler{ .config = options.sampling };
		std::vector<std::int32_t> history = initialTokenIds;
		std::vector<LiteNN::Tensor<LiteNN::CPU>> caches;
		std::int32_t currentToken = initialTokenIds.front();
		std::size_t lastOutputCount = 0;
		std::vector<std::size_t> lastLogitsShape;
		std::size_t generatedTokenCount = 0;
		bool stoppedOnEos = false;

		const auto runStart = std::chrono::steady_clock::now();
		for (std::size_t step = 0; step < maxRunCount; ++step)
		{
			const auto& plan = decodePlans[step];
			std::vector<LiteNN::Tensor<LiteNN::CPU>> inputs;
			inputs.push_back(MakeTokenIdTensorForPlan(currentToken, plan));
			if (caches.empty())
			{
				LiteNN::CPU cpu;
				for (std::size_t i = 1; i < plan.inputs.size(); ++i)
				{
					const auto& input = plan.inputs[i];
					if (!input.type.IsFullyStatic())
					{
						throw std::runtime_error("decode-loop cache inputs must have static shapes");
					}
					inputs.emplace_back(input.type.StaticShape(), input.type.dtype, cpu);
				}
			}
			else
			{
				if (caches.size() + 1 != plan.inputs.size())
				{
					throw std::runtime_error("decode-loop cache count does not match decode graph inputs");
				}
				for (auto& cache : caches)
				{
					inputs.push_back(std::move(cache));
				}
				caches.clear();
			}

			auto outputs = interpreter.RunForward(plan, inputs);
			if (outputs.empty())
			{
				throw std::runtime_error("decode-loop produced no outputs");
			}
			lastLogitsShape = outputs.front().Shape().ToOwned();
			if (step + 1 < initialTokenIds.size())
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
			caches.reserve(outputs.size() - 1);
			for (std::size_t i = 1; i < outputs.size(); ++i)
			{
				caches.push_back(std::move(outputs[i]));
			}
			if (stoppedOnEos)
			{
				break;
			}
		}
		const auto runEnd = std::chrono::steady_clock::now();

		const auto buildMs = std::chrono::duration<double, std::milli>(buildEnd - buildStart).count();
		const auto runMs = std::chrono::duration<double, std::milli>(runEnd - runStart).count();

		std::cout << "Ran LLaMA decode loop tensors=" << imported.summary.tensorCount
		          << " metadata=" << imported.summary.metadataCount << " steps=" << options.steps
		          << " prompt_tokens=" << initialTokenIds.size() << " generated_tokens=" << generatedTokenCount
		          << " stopped_on_eos=" << (stoppedOnEos ? "true" : "false") << " cached_plans=" << decodePlans.size()
		          << " build_ms=" << buildMs << " run_ms=" << runMs << " outputs_per_step=" << lastOutputCount
		          << " last_logits_shape=";
		PrintTensorShape(lastLogitsShape);
		std::cout << " generated=";
		PrintTokenList(history);
		std::cout << " pieces=" << TokenPiecesText(imported.model.UnsafeGraphView(), history);
		std::cout << '\n';
		if (options.outputPath)
		{
			std::ofstream output(*options.outputPath, std::ios::binary);
			if (!output)
			{
				throw std::runtime_error("Failed to open decode-loop output file: " + *options.outputPath);
			}
			output << TokenListText(history) << '\n'
			       << TokenPiecesText(imported.model.UnsafeGraphView(), history) << '\n'
			       << "generated_tokens=" << generatedTokenCount
			       << " stopped_on_eos=" << (stoppedOnEos ? "true" : "false") << '\n';
			if (!output)
			{
				throw std::runtime_error("Failed to write decode-loop output file: " + *options.outputPath);
			}
		}
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
		return options;
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
			LiteNN::Runtime::Interpreter<LiteNN::CPU> interpreter(LiteNN::GGUF::TryEvalGGMLQuantizedMatMul);
			const auto outputs = interpreter.RunForward(plan, inputs);
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

		if (argc >= 2 && std::string_view(argv[1]) == "--run-llama-prompt")
		{
			if (argc != 4 && argc != 5)
			{
				PrintUsage(argv[0]);
				return 1;
			}
			const auto positionOffset = argc == 5 ? ParseSize(argv[4], "position-offset", true) : 0uz;
			const auto imported = LiteNN::GGUF::ImportGGUFArchive(argv[2]);
			const auto prompt =
			    LiteNN::GGUF::MakeExactVocabularyPromptTokens(argv[3], imported.model.UnsafeGraphView());
			auto lowered = LiteNN::GGUF::LowerLLaMACausalLM(imported.model.UnsafeGraphView(), prompt.tokenIds.size(),
			                                                positionOffset, { .preserveQuantizedWeights = true });
			const auto plan = LiteNN::Detail::BuildExecutablePlanFromGraph(lowered);
			auto inputs = MakeZeroStateInputs(plan, MakeTokenIdTensor(prompt.tokenIds, plan));
			LiteNN::Runtime::Interpreter<LiteNN::CPU> interpreter(LiteNN::GGUF::TryEvalGGMLQuantizedMatMul);
			const auto outputs = interpreter.RunForward(plan, inputs);
			if (outputs.empty())
			{
				throw std::runtime_error("LLM package produced no outputs");
			}
			const auto& logits = outputs.front();
			LiteNN::GGUF::LLMSamplerState sampler;
			const auto nextToken = LiteNN::GGUF::SelectNextToken(logits, sampler, prompt.tokenIds);
			std::cout << "Ran LLaMA GGUF exact-prompt smoke tensors=" << imported.summary.tensorCount
			          << " metadata=" << imported.summary.metadataCount << " token_ids=";
			PrintTokenList(prompt.tokenIds);
			std::cout << " pieces=" << TokenPiecesText(imported.model.UnsafeGraphView(), prompt.tokenIds)
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
			LiteNN::Runtime::Interpreter<LiteNN::CPU> interpreter(LiteNN::GGUF::TryEvalGGMLQuantizedMatMul);
			const auto outputs = interpreter.RunForward(package.plan, inputs);
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
			                                                                .preserveQuantizedWeights = true });
			LiteNN::Serialization::SaveVNextModelPackageExternalWeights(schedule, argv[3], argv[4]);
			std::cout << "Lowered stateful LLaMA decode package with " << schedule.states.size()
			          << " runtime states and " << schedule.stateValueBindings.size() << " value bindings\n";
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
