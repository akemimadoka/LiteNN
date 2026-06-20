#include "GGUFImporter.h"
#include "LLaMABuilder.h"

#ifdef LITENN_GGUF_CONVERT_ENABLE_AOT
#include <LiteNN/Compiler/CompiledModule.h>
#endif
#include <LiteNN/Serialization/ModelPackageIO.h>

#include <algorithm>
#include <charconv>
#include <cstdint>
#include <cstdlib>
#include <format>
#include <iostream>
#include <limits>
#include <optional>
#include <span>
#include <stdexcept>
#include <string>
#include <string_view>

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
		          << " --lower-llama-decode <input.gguf> <output.ltnn> <sequence-length> <past-length>\n"
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
			const auto maxCacheLength = argc == 6 ? ParseSize(argv[5], "max-cache-length", true) : decodePastLength;
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

		PrintUsage(argv[0]);
		return 1;
	}
	catch (const std::exception& ex)
	{
		std::cerr << "litenn_gguf_convert: " << ex.what() << '\n';
		return 1;
	}
}
