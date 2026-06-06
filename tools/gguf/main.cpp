#include "GGUFImporter.h"
#include "LLaMABuilder.h"

#ifdef LITENN_GGUF_CONVERT_ENABLE_AOT
#include <LiteNN/Compiler/CompiledModule.h>
#endif
#include <LiteNN/Serialization/ModelPackageIO.h>

#include <charconv>
#include <algorithm>
#include <cstdlib>
#include <iostream>
#include <optional>
#include <stdexcept>
#include <cstdint>
#include <string>
#include <string_view>

namespace
{
	void PrintUsage(std::string_view executable)
	{
		std::cerr << "Usage:\n"
		          << "  " << executable << " --import <input.gguf> <output.ltnn>\n"
		          << "  " << executable << " --lower-llama <input.gguf> <output.ltnn> <sequence-length> [position-offset]\n"
		          << "  " << executable << " --lower-llama-decode <input.gguf> <output.ltnn> <sequence-length> <past-length>\n"
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
			throw std::runtime_error(std::string(label) + (allowZero ? " must be a non-negative integer"
			                                                        : " must be a positive integer"));
		}
		return value;
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
		options.enableCPUAOTExternalRegions =
		    TruthyEnvValue(std::getenv("LITENN_CPU_AOT_EXTERNAL_REGIONS")) ||
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
		std::cout << "Wrote AOT carrier object " << outputPath
		          << " backend=" << BackendName(artifact.Backend())
		          << " rodata=" << artifact.Rodata().size()
		          << " bytes instructions=" << artifact.Instructions().size() << " bytes\n";
	}

	void PrintSeparatedArtifactSummary(const LiteNN::CompiledModuleSeparatedArtifact& artifact,
	                                   std::string_view outputDir)
	{
		std::cout << "Wrote separated AOT carrier objects " << outputDir
		          << " backend=" << BackendName(artifact.Backend())
		          << " metadata=" << artifact.Metadata().size()
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

		if (argc == 3)
		{
			const auto summary = LiteNN::GGUF::ConvertGGUFArchive(argv[1], argv[2]);
			std::cout << "Imported archive with " << summary.tensorCount << " tensors and " << summary.metadataCount
			          << " metadata entries\n";
			return 0;
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
			throw std::runtime_error("CUDA AOT support is not enabled in this build; configure with LITENN_ENABLE_CUDA=ON");
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
			auto artifact = LiteNN::Compiler<LiteNN::CPU>::CompileArtifact(
			    LiteNN::Serialization::LoadVNextModelPackage(argv[2]).plan, CompilerOptionsFromEnvironment()).SeparateRodata();
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
			auto artifact = LiteNN::Compiler<LiteNN::CUDA>::CompileArtifact(
			    LiteNN::Serialization::LoadVNextModelPackage(argv[2]).plan, CompilerOptionsFromEnvironment()).SeparateRodata();
			const std::string_view symbolPrefix = argc == 5 ? std::string_view(argv[4]) : "litenn_gguf_module";
			artifact.WriteObjectFiles(argv[3], symbolPrefix);
			PrintSeparatedArtifactSummary(artifact, argv[3]);
			return 0;
#elif defined(LITENN_GGUF_CONVERT_ENABLE_AOT)
			throw std::runtime_error("CUDA AOT support is not enabled in this build; configure with LITENN_ENABLE_CUDA=ON");
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
			auto lowered = LiteNN::GGUF::LowerLLaMACausalLM(imported.model.UnsafeGraphView(), sequenceLength, positionOffset);
			LiteNN::Serialization::SaveVNextModelPackage(LiteNN::Detail::BuildExecutableModuleFromGraph(lowered), argv[3]);
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
			auto lowered = LiteNN::GGUF::LowerLLaMACausalLMDecode(imported.model.UnsafeGraphView(), sequenceLength, pastLength, pastLength);
			LiteNN::Serialization::SaveVNextModelPackage(LiteNN::Detail::BuildExecutableModuleFromGraph(lowered), argv[3]);
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
