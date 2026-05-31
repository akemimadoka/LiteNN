#include <LiteNN.h>

#ifdef LITENN_ENABLE_MLIR
#include <LiteNN/Compiler/CompiledModule.h>
#include <LiteNN/Runtime/Interpreter.h>
#endif

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <exception>
#include <filesystem>
#include <format>
#include <fstream>
#include <iostream>
#include <limits>
#include <map>
#include <optional>
#include <random>
#include <sstream>
#include <span>
#include <stdexcept>
#include <string>
#include <string_view>
#include <type_traits>
#include <vector>

#if defined(_WIN32)
#include <Windows.h>
#else
#include <dlfcn.h>
#endif

namespace
{
	void PrintUsage(std::string_view executable)
	{
		std::cout << std::format(
		    "Usage:\n"
		    "  {} --inspect <sdxl.safetensors>\n"
		    "  {} --import <manifest.json> <sdxl.safetensors> <output.ltnn>"
		    " [--allow-extra-tensors] [--external-weights weights.bin] [--external-weight-min-bytes N]\n"
		    "  {} --compile-budget <input.ltnn>\n"
		    "  {} --run-model <input.ltnn>\n"
	    "  {} --run-model-with-inputs <input.ltnn> <inputs.safetensors>"
	    " [--output outputs.safetensors] [--allow-nonfinite]\n"
	    "  {} --diagnose-model-with-inputs <input.ltnn> <inputs.safetensors>"
	    " [--verbose] [--max-nodes N] [--allow-nonfinite]\n"
	    "  {} --benchmark-model-with-inputs <input.ltnn> <inputs.safetensors>"
		    " [--device cpu|cuda] [--warmup N] [--iterations N] [--json result.json]\n"
		    "  {} --compile-raw-object <input.ltnn> <output.o|obj>\n"
		    "  {} --compile-image-regions <input.ltnn> <output-dir> [file-prefix]"
		    " [--cpu-aot-llvm-opt-level 0|1|2|3]\n"
		    "  {} --run-image-with-inputs <rodata.bin> <instructions.o|obj> <inputs.safetensors>"
		    " [--output outputs.safetensors] [--allow-nonfinite]\n"
		    "  {} --compile-object <input.ltnn> <output.o|obj> [symbol-prefix]\n"
		    "  {} --load-dll <module.dll|so|dylib> [symbol-prefix]\n"
		    "  {} --load-dll-with-inputs <module.dll|so|dylib> <inputs.safetensors> [symbol-prefix]"
		    " [--output outputs.safetensors] [--allow-nonfinite]\n"
		    "  {} --sample-euler <module.dll|so|dylib> [symbol-prefix] [--steps N] [--seed N]"
		    " [--sigma-max X] [--sigma-min X] [--scheduler linear|edm] [--rho X]"
		    " [--denoiser-contract epsilon|denoised|sgm-edm|sgm-eps|sgm-v]"
		    " [--timestep-mode auto|legacy|sigma|edm-log|zero] [--cfg-mode auto|none|dual]"
		    " [--cfg-scale X] [--sigma-data X] [--latent-init random|zero|inputs]"
		    " [--inputs inputs.safetensors] [--output-latent latent.safetensors] [--allow-nonfinite]\n\n"
		    "  {} --denoise-latent <module.dll|so|dylib> <inputs.safetensors> <output-latent.safetensors>"
		    " [symbol-prefix] [same sampler options except --inputs/--output-latent]\n\n"
		    "  {} --denoise-latent-image <rodata.bin> <instructions.o|obj> <inputs.safetensors>"
		    " <output-latent.safetensors> [same sampler options except --inputs/--output-latent]\n\n"
		    "This example intentionally requires a LiteNN Torch manifest. A raw SDXL safetensors file contains\n"
		    "weights only; it does not define the UNet/text-encoder/VAE graph, scheduler, or fixed input shapes.\n",
	    executable, executable, executable, executable, executable, executable, executable, executable,
	    executable, executable, executable, executable, executable, executable, executable, executable);
	}

	void PrintReport(const LiteNN::Serialization::TorchManifestReport& report)
	{
		const auto printGroup = [](std::string_view title, const std::vector<std::string>& entries) {
			if (entries.empty())
			{
				return;
			}
			std::cout << title << ":\n";
			for (const auto& entry : entries)
			{
				std::cout << "  - " << entry << '\n';
			}
		};
		printGroup("Imported tensors", report.importedTensors);
		printGroup("Lowered ops", report.loweredOps);
		printGroup("Folded constants", report.foldedConstants);
		printGroup("Unsupported ops", report.unsupportedOps);
		printGroup("Fallbacks", report.fallbacks);
		printGroup("Diagnostics", report.diagnostics);
	}

	std::string ComponentForTensor(std::string_view name)
	{
		if (name.starts_with("model.diffusion_model.") || name.starts_with("unet."))
		{
			return "unet";
		}
		if (name.starts_with("first_stage_model.") || name.starts_with("vae."))
		{
			return "vae";
		}
		if (name.starts_with("conditioner.embedders.0.") || name.starts_with("text_encoder."))
		{
			return "text_encoder";
		}
		if (name.starts_with("conditioner.embedders.1.") || name.starts_with("text_encoder_2."))
		{
			return "text_encoder_2";
		}
		return "unknown";
	}

	std::string ShapeToString(LiteNN::ShapeView shape)
	{
		std::string result = "[";
		for (std::size_t i = 0; i < shape.NumDim(); ++i)
		{
			if (i != 0)
			{
				result += ", ";
			}
			result += std::to_string(shape[i]);
		}
		result += "]";
		return result;
	}

	void InspectSafetensors(const std::filesystem::path& path)
	{
		const auto archive = LiteNN::Serialization::SafetensorsArchive::LoadFile(path);
		std::map<std::string, std::pair<std::size_t, std::size_t>, std::less<>> components;
		std::size_t totalBytes = 0;
		for (const auto& tensor : archive.Tensors())
		{
			totalBytes += tensor.ByteSize();
			auto& summary = components[ComponentForTensor(tensor.name)];
			++summary.first;
			summary.second += tensor.ByteSize();
		}

		std::cout << std::format("Loaded {} tensors from safetensors file\n", archive.Tensors().size());
		std::cout << std::format("Total payload bytes: {}\n", totalBytes);
		for (const auto& [component, summary] : components)
		{
			std::cout << std::format("  {}: {} tensor(s), {} byte(s)\n",
			                         component, summary.first, summary.second);
		}

		const auto preview = std::min<std::size_t>(archive.Tensors().size(), 16uz);
		if (preview != 0)
		{
			std::cout << "First tensors:\n";
			for (std::size_t i = 0; i < preview; ++i)
			{
				const auto& tensor = archive.Tensors()[i];
				std::cout << std::format("  {} {} {}\n", tensor.name, LiteNN::DataTypeName(tensor.dtype),
				                         ShapeToString(tensor.shape));
			}
		}
	}

	void ImportManifest(const std::filesystem::path& manifestPath,
	                    const std::filesystem::path& safetensorsPath,
	                    const std::filesystem::path& outputPath,
	                    bool allowExtraTensors,
	                    const std::optional<std::filesystem::path>& externalWeightsPath,
	                    std::uint64_t externalWeightMinBytes)
	{
		LiteNN::Serialization::TorchManifestImportOptions options;
		options.failOnUnusedWeights = !allowExtraTensors;
		auto imported = LiteNN::Serialization::LoadTorchManifest(manifestPath, safetensorsPath, options);
		if (externalWeightsPath)
		{
			LiteNN::Serialization::ExternalWeightSaveOptions externalOptions;
			externalOptions.minVariableBytes = externalWeightMinBytes;
			LiteNN::Serialization::SaveModelExternalWeights(imported.graph, outputPath,
			                                                *externalWeightsPath, externalOptions);
		}
		else
		{
			LiteNN::Serialization::SaveModel(imported.graph, outputPath);
		}
		std::cout << std::format("Wrote LiteNN graph {} with {} variable(s), {} input(s), {} output(s)\n",
		                         outputPath.string(), imported.graph.VariableCount(),
		                         imported.graph.InputSignature().size(), imported.graph.OutputSignature().size());
		if (externalWeightsPath)
		{
			std::cout << std::format("Wrote external LiteNN weights {}\n", externalWeightsPath->string());
		}
		PrintReport(imported.report);
	}

#ifdef LITENN_ENABLE_MLIR
	struct EulerSamplerOptions
	{
		std::size_t steps{ 4 };
		std::uint32_t seed{ 5489 };
		double sigmaMax{ 1.0 };
		double sigmaMin{ 0.0 };
		double rho{ 3.0 };
		double sigmaData{ 0.5 };
		double cfgScale{ 1.0 };
		std::string scheduler{ "linear" };
		std::string denoiserContract{ "epsilon" };
		std::string timestepMode{ "auto" };
		std::string cfgMode{ "auto" };
		std::string latentInit{ "random" };
		std::optional<std::filesystem::path> inputBindings;
		std::optional<std::filesystem::path> outputLatent;
		bool failOnNonFinite{ true };
	};

	struct DenoiserStepCoefficients
	{
		double cSkip{ 0.0 };
		double cOut{ 1.0 };
		double cIn{ 1.0 };
		double cNoise{ 0.0 };
		bool rawOutputIsDerivative{ true };
		bool rawOutputIsDenoised{ false };
	};

	enum class InputBindingFlavor
	{
		Default,
		Negative,
	};

	struct TensorStats
	{
		double mean{};
		double rms{};
		double min{ std::numeric_limits<double>::infinity() };
		double max{ -std::numeric_limits<double>::infinity() };
		std::size_t finiteCount{};
		std::size_t nonFiniteCount{};

		[[nodiscard]] std::size_t TotalCount() const
		{
			return finiteCount + nonFiniteCount;
		}

		[[nodiscard]] bool AllFinite() const
		{
			return nonFiniteCount == 0;
		}
	};

	struct OutputRunOptions
	{
		std::optional<std::filesystem::path> outputPath;
		bool failOnNonFinite{ true };
	};

	struct FiniteDiagnosticOptions
	{
		bool verbose{ false };
		bool failOnNonFinite{ true };
		std::optional<std::size_t> maxNodes;
	};

	struct ExampleCompilerSettings
	{
		std::uint8_t cpuAOTLLVMOptLevel{ 0 };
		bool diagnostics{ true };
	};

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

	std::string FormatByteSize(std::uint64_t bytes)
	{
		const auto mib = static_cast<double>(bytes) / (1024.0 * 1024.0);
		return std::format("{} bytes ({:.2f} MiB)", bytes, mib);
	}

	LiteNN::CompilerOptions MakeExampleCompilerOptions(const ExampleCompilerSettings& settings = {})
	{
		auto options = LiteNN::CompilerOptions::Defaults();
		options.enableCPUAOTExternalRegions = true;
		options.cpuAOTLLVMOptLevel = settings.cpuAOTLLVMOptLevel;
		options.enableCompileDiagnostics = settings.diagnostics;
		return options;
	}

	void PrintCompileBudget(const LiteNN::Graph& graph,
	                        const LiteNN::CompilerOptions& options,
	                        std::string_view label)
	{
		const auto budget = LiteNN::EstimateCompileBudget(graph, options);
		std::cout << std::format(
		    "{} compile budget: subgraphs={} nodes={} variables={} variable_refs={} constants={} qconstants={}\n"
		    "  variable_payload={} constant_payload={} qconstant_payload={}\n"
		    "  cpu_external_regions={} cpu_llvm_opt=O{} projected_inline_mlir_payload={} projected_external_constants={}"
		    " projected_external_weights={}\n",
		    label, budget.subgraphCount, budget.nodeCount, budget.variableCount, budget.variableRefNodeCount,
		    budget.constantNodeCount, budget.quantizedConstantNodeCount, FormatByteSize(budget.variablePayloadBytes),
		    FormatByteSize(budget.constantPayloadBytes), FormatByteSize(budget.quantizedConstantPayloadBytes),
		    budget.cpuAOTExternalRegionsEnabled ? "on" : "off", options.cpuAOTLLVMOptLevel,
		    FormatByteSize(budget.projectedInlineMLIRPayloadBytes),
		    FormatByteSize(budget.projectedExternalConstantBytes),
		    FormatByteSize(budget.projectedExternalWeightBytes))
		          << std::flush;
	}

	template <typename F>
	auto TimedStep(std::string_view label, F&& f)
	{
		std::cout << std::format("{}...\n", label) << std::flush;
		const auto start = std::chrono::steady_clock::now();
		if constexpr (std::is_void_v<std::invoke_result_t<F>>)
		{
			std::forward<F>(f)();
			const auto elapsed = std::chrono::duration<double, std::milli>(
			    std::chrono::steady_clock::now() - start);
			std::cout << std::format("{}: ok {:.3f} ms\n", label, elapsed.count()) << std::flush;
		}
		else
		{
			auto result = std::forward<F>(f)();
			const auto elapsed = std::chrono::duration<double, std::milli>(
			    std::chrono::steady_clock::now() - start);
			std::cout << std::format("{}: ok {:.3f} ms\n", label, elapsed.count()) << std::flush;
			return result;
		}
	}

	std::size_t ParseSize(std::string_view text, std::string_view label)
	{
		std::size_t consumed = 0;
		const auto result = std::stoull(std::string(text), &consumed);
		if (consumed != text.size())
		{
			throw std::runtime_error(std::format("Invalid {}: {}", label, text));
		}
		return static_cast<std::size_t>(result);
	}

	std::uint32_t ParseU32(std::string_view text, std::string_view label)
	{
		const auto result = ParseSize(text, label);
		if (result > std::numeric_limits<std::uint32_t>::max())
		{
			throw std::runtime_error(std::format("{} is too large for uint32", label));
		}
		return static_cast<std::uint32_t>(result);
	}

	std::uint8_t ParseOptLevel(std::string_view text, std::string_view label)
	{
		const auto result = ParseSize(text, label);
		if (result > 3)
		{
			throw std::runtime_error(std::format("{} must be between 0 and 3", label));
		}
		return static_cast<std::uint8_t>(result);
	}

	double ParseDouble(std::string_view text, std::string_view label)
	{
		std::size_t consumed = 0;
		const auto result = std::stod(std::string(text), &consumed);
		if (consumed != text.size() || !std::isfinite(result))
		{
			throw std::runtime_error(std::format("Invalid {}: {}", label, text));
		}
		return result;
	}

	OutputRunOptions ParseOutputRunOptions(int argc, char** argv, int firstOption)
	{
		OutputRunOptions options;
		for (int i = firstOption; i < argc; ++i)
		{
			const std::string_view option = argv[i];
			if (option == "--allow-nonfinite")
			{
				options.failOnNonFinite = false;
				continue;
			}
			if (option == "--fail-on-nonfinite")
			{
				options.failOnNonFinite = true;
				continue;
			}
			if (option != "--output")
			{
				throw std::runtime_error("Unknown output option: " + std::string(option));
			}
			if (options.outputPath)
			{
				throw std::runtime_error("--output was specified more than once");
			}
			if (i + 1 >= argc)
			{
				throw std::runtime_error("--output requires a value");
			}
			++i;
			options.outputPath = std::filesystem::path(argv[i]);
		}
		return options;
	}

	FiniteDiagnosticOptions ParseFiniteDiagnosticOptions(int argc, char** argv, int firstOption)
	{
		FiniteDiagnosticOptions options;
		for (int i = firstOption; i < argc; ++i)
		{
			const std::string_view option = argv[i];
			const auto requireValue = [&](std::string_view label) -> std::string_view {
				if (i + 1 >= argc)
				{
					throw std::runtime_error(std::format("{} requires a value", label));
				}
				++i;
				return argv[i];
			};
			if (option == "--verbose")
			{
				options.verbose = true;
			}
			else if (option == "--allow-nonfinite")
			{
				options.failOnNonFinite = false;
			}
			else if (option == "--fail-on-nonfinite")
			{
				options.failOnNonFinite = true;
			}
		else if (option == "--max-nodes")
		{
			options.maxNodes = ParseSize(requireValue(option), "max-nodes");
			if (*options.maxNodes == 0)
			{
				throw std::runtime_error("--max-nodes must be positive");
			}
		}
			else
			{
				throw std::runtime_error("Unknown --diagnose-model-with-inputs option: " + std::string(option));
			}
		}
		return options;
	}

	std::vector<std::byte> ReadAllBytes(const std::filesystem::path& path)
	{
		std::ifstream in(path, std::ios::binary | std::ios::ate);
		if (!in)
		{
			throw std::runtime_error("Failed to open input file");
		}
		const auto size = in.tellg();
		if (size < 0)
		{
			throw std::runtime_error("Failed to determine input file size");
		}
		std::vector<std::byte> bytes(static_cast<std::size_t>(size));
		in.seekg(0, std::ios::beg);
		if (!bytes.empty())
		{
			constexpr std::size_t kChunkBytes = 64ull * 1024ull * 1024ull;
			std::size_t offset = 0;
			while (offset < bytes.size())
			{
				const auto remaining = bytes.size() - offset;
				const auto chunk = std::min(remaining, kChunkBytes);
				in.read(reinterpret_cast<char*>(bytes.data() + static_cast<std::ptrdiff_t>(offset)),
				        static_cast<std::streamsize>(chunk));
				if (!in)
				{
					throw std::runtime_error("Failed to read input file");
				}
				offset += chunk;
			}
		}
		return bytes;
	}

	void WriteAllBytes(const std::filesystem::path& path, std::span<const std::byte> bytes)
	{
		if (path.has_parent_path())
		{
			std::filesystem::create_directories(path.parent_path());
		}
		std::ofstream out(path, std::ios::binary);
		if (!out)
		{
			throw std::runtime_error("Failed to open output file");
		}
		constexpr std::size_t kChunkBytes = 64ull * 1024ull * 1024ull;
		std::size_t offset = 0;
		while (offset < bytes.size())
		{
			const auto remaining = bytes.size() - offset;
			const auto chunk = std::min(remaining, kChunkBytes);
			out.write(reinterpret_cast<const char*>(bytes.data() + static_cast<std::ptrdiff_t>(offset)),
			          static_cast<std::streamsize>(chunk));
			if (!out)
			{
				throw std::runtime_error("Failed to write output file");
			}
			offset += chunk;
		}
	}

	EulerSamplerOptions ParseEulerOptions(int argc, char** argv, int firstOption)
	{
		EulerSamplerOptions options;
		for (int i = firstOption; i < argc; ++i)
		{
			const std::string_view option = argv[i];
			const auto requireValue = [&](std::string_view label) -> std::string_view {
				if (i + 1 >= argc)
				{
					throw std::runtime_error(std::format("{} requires a value", label));
				}
				++i;
				return argv[i];
			};
			if (option == "--steps")
			{
				options.steps = ParseSize(requireValue(option), "steps");
			}
			else if (option == "--seed")
			{
				options.seed = ParseU32(requireValue(option), "seed");
			}
			else if (option == "--sigma-max")
			{
				options.sigmaMax = ParseDouble(requireValue(option), "sigma-max");
			}
			else if (option == "--sigma-min")
			{
				options.sigmaMin = ParseDouble(requireValue(option), "sigma-min");
			}
			else if (option == "--rho")
			{
				options.rho = ParseDouble(requireValue(option), "rho");
			}
			else if (option == "--sigma-data")
			{
				options.sigmaData = ParseDouble(requireValue(option), "sigma-data");
			}
			else if (option == "--cfg-scale")
			{
				options.cfgScale = ParseDouble(requireValue(option), "cfg-scale");
			}
			else if (option == "--scheduler")
			{
				options.scheduler = std::string(requireValue(option));
			}
			else if (option == "--denoiser-contract")
			{
				options.denoiserContract = std::string(requireValue(option));
			}
			else if (option == "--timestep-mode")
			{
				options.timestepMode = std::string(requireValue(option));
			}
			else if (option == "--cfg-mode")
			{
				options.cfgMode = std::string(requireValue(option));
			}
			else if (option == "--latent-init")
			{
				options.latentInit = std::string(requireValue(option));
			}
			else if (option == "--inputs")
			{
				options.inputBindings = std::filesystem::path(std::string(requireValue(option)));
			}
			else if (option == "--output-latent")
			{
				options.outputLatent = std::filesystem::path(std::string(requireValue(option)));
			}
			else if (option == "--allow-nonfinite")
			{
				options.failOnNonFinite = false;
			}
			else if (option == "--fail-on-nonfinite")
			{
				options.failOnNonFinite = true;
			}
			else
			{
				throw std::runtime_error("Unknown --sample-euler option: " + std::string(option));
			}
		}
		if (options.steps == 0)
		{
			throw std::runtime_error("Euler sampler requires steps > 0");
		}
		if (options.sigmaMax < options.sigmaMin)
		{
			throw std::runtime_error("Euler sampler requires sigma-max >= sigma-min");
		}
		if (options.scheduler != "linear" && options.scheduler != "edm")
		{
			throw std::runtime_error("Euler sampler --scheduler must be 'linear' or 'edm'");
		}
		if (options.denoiserContract != "epsilon" && options.denoiserContract != "denoised" &&
		    options.denoiserContract != "sgm-edm" && options.denoiserContract != "sgm-eps" &&
		    options.denoiserContract != "sgm-v")
		{
			throw std::runtime_error(
			    "Euler sampler --denoiser-contract must be one of epsilon, denoised, sgm-edm, sgm-eps, or sgm-v");
		}
		if (options.timestepMode != "auto" && options.timestepMode != "legacy" &&
		    options.timestepMode != "sigma" && options.timestepMode != "edm-log" &&
		    options.timestepMode != "zero")
		{
			throw std::runtime_error("Euler sampler --timestep-mode must be auto, legacy, sigma, edm-log, or zero");
		}
		if (options.cfgMode != "auto" && options.cfgMode != "none" && options.cfgMode != "dual")
		{
			throw std::runtime_error("Euler sampler --cfg-mode must be auto, none, or dual");
		}
		if (options.latentInit != "random" && options.latentInit != "zero" && options.latentInit != "inputs")
		{
			throw std::runtime_error("Euler sampler --latent-init must be random, zero, or inputs");
		}
		if (options.sigmaData <= 0.0)
		{
			throw std::runtime_error("Euler sampler requires sigma-data > 0");
		}
		if (options.scheduler == "edm")
		{
			if (options.sigmaMin <= 0.0 || options.sigmaMax <= 0.0)
			{
				throw std::runtime_error("Euler EDM scheduler requires positive sigma-min and sigma-max");
			}
			if (options.rho <= 0.0)
			{
				throw std::runtime_error("Euler EDM scheduler requires rho > 0");
			}
		}
		if ((options.denoiserContract == "sgm-edm" || options.timestepMode == "edm-log") &&
		    (options.sigmaMin <= 0.0 || options.sigmaMax <= 0.0))
		{
			throw std::runtime_error("EDM denoiser/timestep modes require positive sigma-min and sigma-max");
		}
		return options;
	}

	double SigmaAtStep(const EulerSamplerOptions& options, std::size_t step)
	{
		const auto progress = static_cast<double>(step) / static_cast<double>(options.steps);
		if (options.scheduler == "linear")
		{
			return options.sigmaMax + (options.sigmaMin - options.sigmaMax) * progress;
		}
		const auto invRho = 1.0 / options.rho;
		const auto maxInv = std::pow(options.sigmaMax, invRho);
		const auto minInv = std::pow(options.sigmaMin, invRho);
		return std::pow(maxInv + (minInv - maxInv) * progress, options.rho);
	}

	DenoiserStepCoefficients ComputeDenoiserStepCoefficients(const EulerSamplerOptions& options, double sigma)
	{
		if (options.denoiserContract == "epsilon")
		{
			return {
				.cNoise = sigma,
				.rawOutputIsDerivative = true,
			};
		}
		if (options.denoiserContract == "denoised")
		{
			return {
				.cNoise = sigma,
				.rawOutputIsDerivative = false,
				.rawOutputIsDenoised = true,
			};
		}
		if (options.denoiserContract == "sgm-edm")
		{
			const auto sigmaData2 = options.sigmaData * options.sigmaData;
			const auto sigma2 = sigma * sigma;
			return {
				.cSkip = sigmaData2 / (sigma2 + sigmaData2),
				.cOut = sigma * options.sigmaData / std::sqrt(sigma2 + sigmaData2),
				.cIn = 1.0 / std::sqrt(sigma2 + sigmaData2),
				.cNoise = 0.25 * std::log(sigma),
				.rawOutputIsDerivative = false,
				.rawOutputIsDenoised = false,
			};
		}
		if (options.denoiserContract == "sgm-eps")
		{
			return {
				.cSkip = 1.0,
				.cOut = -sigma,
				.cIn = 1.0 / std::sqrt(sigma * sigma + 1.0),
				.cNoise = sigma,
				.rawOutputIsDerivative = false,
				.rawOutputIsDenoised = false,
			};
		}
		if (options.denoiserContract == "sgm-v")
		{
			const auto denom = std::sqrt(sigma * sigma + 1.0);
			return {
				.cSkip = 1.0 / (sigma * sigma + 1.0),
				.cOut = -sigma / denom,
				.cIn = 1.0 / denom,
				.cNoise = sigma,
				.rawOutputIsDerivative = false,
				.rawOutputIsDenoised = false,
			};
		}
		throw std::runtime_error("Unsupported denoiser contract");
	}

	double TimestepForStep(const EulerSamplerOptions& options,
	                       const DenoiserStepCoefficients& coefficients,
	                       double sigma)
	{
		auto mode = options.timestepMode;
		if (mode == "auto")
		{
			if (options.denoiserContract.starts_with("sgm-"))
			{
				return coefficients.cNoise;
			}
			mode = "legacy";
		}
		if (mode == "legacy")
		{
			return options.sigmaMax == 0.0 ? 0.0 : sigma * 999.0 / options.sigmaMax;
		}
		if (mode == "sigma")
		{
			return sigma;
		}
		if (mode == "edm-log")
		{
			return 0.25 * std::log(sigma);
		}
		if (mode == "zero")
		{
			return 0.0;
		}
		throw std::runtime_error("Unsupported timestep mode");
	}

	std::string ResolveCFGMode(const EulerSamplerOptions& options)
	{
		if (options.cfgMode != "auto")
		{
			return options.cfgMode;
		}
		return options.cfgScale == 1.0 ? "none" : "dual";
	}

	template <typename T>
	void FillRandomNormalTyped(LiteNN::Tensor<LiteNN::CPU>& tensor, std::uint32_t seed, double sigma)
	{
		std::mt19937 rng(seed);
		std::normal_distribution<float> dist(0.0F, static_cast<float>(sigma));
		auto* data = static_cast<T*>(tensor.RawData());
		for (std::size_t i = 0; i < tensor.NumElements(); ++i)
		{
			data[i] = static_cast<T>(dist(rng));
		}
	}

	void FillRandomLatent(LiteNN::Tensor<LiteNN::CPU>& tensor, std::uint32_t seed, double sigma)
	{
		if (!LiteNN::IsFloatingDataType(tensor.DType()))
		{
			throw std::runtime_error("Euler latent input must be floating-point");
		}
		switch (tensor.DType())
		{
		case LiteNN::DataType::Float16:
			FillRandomNormalTyped<LiteNN::Float16>(tensor, seed, sigma);
			break;
		case LiteNN::DataType::BFloat16:
			FillRandomNormalTyped<LiteNN::BFloat16>(tensor, seed, sigma);
			break;
		case LiteNN::DataType::Float32:
			FillRandomNormalTyped<float>(tensor, seed, sigma);
			break;
		case LiteNN::DataType::Float64:
			FillRandomNormalTyped<double>(tensor, seed, sigma);
			break;
		default:
			throw std::runtime_error("Euler latent input dtype is not supported");
		}
	}

	template <typename T>
	void FillScalarTyped(LiteNN::Tensor<LiteNN::CPU>& tensor, double value)
	{
		auto* data = static_cast<T*>(tensor.RawData());
		for (std::size_t i = 0; i < tensor.NumElements(); ++i)
		{
			data[i] = static_cast<T>(value);
		}
	}

	void FillFloatingScalar(LiteNN::Tensor<LiteNN::CPU>& tensor, double value, std::string_view label)
	{
		if (!LiteNN::IsFloatingDataType(tensor.DType()))
		{
			throw std::runtime_error(std::format("{} input must be floating-point", label));
		}
		switch (tensor.DType())
		{
		case LiteNN::DataType::Float16:
			FillScalarTyped<LiteNN::Float16>(tensor, value);
			break;
		case LiteNN::DataType::BFloat16:
			FillScalarTyped<LiteNN::BFloat16>(tensor, value);
			break;
		case LiteNN::DataType::Float32:
			FillScalarTyped<float>(tensor, value);
			break;
		case LiteNN::DataType::Float64:
			FillScalarTyped<double>(tensor, value);
			break;
		default:
			throw std::runtime_error(std::format("{} input dtype is not supported", label));
		}
	}

	template <typename T>
	void CopyScaledTensorTyped(const LiteNN::Tensor<LiteNN::CPU>& source,
	                           LiteNN::Tensor<LiteNN::CPU>& destination,
	                           double scale)
	{
		const auto* src = static_cast<const T*>(source.RawData());
		auto* dst = static_cast<T*>(destination.RawData());
		for (std::size_t i = 0; i < source.NumElements(); ++i)
		{
			dst[i] = static_cast<T>(static_cast<double>(src[i]) * scale);
		}
	}

	void CopyScaledTensor(const LiteNN::Tensor<LiteNN::CPU>& source,
	                      LiteNN::Tensor<LiteNN::CPU>& destination,
	                      double scale,
	                      std::string_view label)
	{
		if (source.DType() != destination.DType() || source.Shape() != destination.Shape())
		{
			throw std::runtime_error(std::format(
			    "{} copy requires matching dtype/shape; source {} {}, destination {} {}", label,
			    LiteNN::DataTypeName(source.DType()), ShapeToString(source.Shape()),
			    LiteNN::DataTypeName(destination.DType()), ShapeToString(destination.Shape())));
		}
		switch (source.DType())
		{
		case LiteNN::DataType::Float16:
			CopyScaledTensorTyped<LiteNN::Float16>(source, destination, scale);
			break;
		case LiteNN::DataType::BFloat16:
			CopyScaledTensorTyped<LiteNN::BFloat16>(source, destination, scale);
			break;
		case LiteNN::DataType::Float32:
			CopyScaledTensorTyped<float>(source, destination, scale);
			break;
		case LiteNN::DataType::Float64:
			CopyScaledTensorTyped<double>(source, destination, scale);
			break;
		default:
			throw std::runtime_error(std::format("{} currently requires floating-point tensors", label));
		}
	}

	template <typename T>
	TensorStats ComputeTensorStatsTyped(const LiteNN::Tensor<LiteNN::CPU>& tensor)
	{
		TensorStats stats;
		const auto* data = static_cast<const T*>(tensor.RawData());
		double sum = 0.0;
		double sumSquares = 0.0;
		for (std::size_t i = 0; i < tensor.NumElements(); ++i)
		{
			const auto value = static_cast<double>(data[i]);
			if (!std::isfinite(value))
			{
				++stats.nonFiniteCount;
				continue;
			}
			++stats.finiteCount;
			sum += value;
			sumSquares += value * value;
			stats.min = std::min(stats.min, value);
			stats.max = std::max(stats.max, value);
		}
		if (stats.finiteCount == 0)
		{
			stats.mean = std::numeric_limits<double>::quiet_NaN();
			stats.rms = std::numeric_limits<double>::quiet_NaN();
			return stats;
		}
		const auto count = static_cast<double>(stats.finiteCount);
		stats.mean = sum / count;
		stats.rms = std::sqrt(sumSquares / count);
		return stats;
	}

	TensorStats ComputeTensorStats(const LiteNN::Tensor<LiteNN::CPU>& tensor)
	{
		switch (tensor.DType())
		{
		case LiteNN::DataType::Float16:
			return ComputeTensorStatsTyped<LiteNN::Float16>(tensor);
		case LiteNN::DataType::BFloat16:
			return ComputeTensorStatsTyped<LiteNN::BFloat16>(tensor);
		case LiteNN::DataType::Float32:
			return ComputeTensorStatsTyped<float>(tensor);
		case LiteNN::DataType::Float64:
			return ComputeTensorStatsTyped<double>(tensor);
		default:
			throw std::runtime_error("Tensor stats currently require floating-point tensors");
		}
	}

	template <typename T>
	void EulerUpdateFromPredictionsTyped(LiteNN::Tensor<LiteNN::CPU>& latent,
	                                     const LiteNN::Tensor<LiteNN::CPU>& condPrediction,
	                                     const LiteNN::Tensor<LiteNN::CPU>* uncondPrediction,
	                                     const DenoiserStepCoefficients& coefficients,
	                                     double cfgScale,
	                                     double sigma,
	                                     double dt)
	{
		auto* latentData = static_cast<T*>(latent.RawData());
		const auto* condData = static_cast<const T*>(condPrediction.RawData());
		const auto* uncondData = uncondPrediction == nullptr ? nullptr : static_cast<const T*>(uncondPrediction->RawData());
		for (std::size_t i = 0; i < latent.NumElements(); ++i)
		{
			const auto state = static_cast<double>(latentData[i]);
			const auto condRaw = static_cast<double>(condData[i]);
			const auto uncondRaw = uncondData == nullptr ? 0.0 : static_cast<double>(uncondData[i]);
			double derivative = 0.0;
			if (coefficients.rawOutputIsDerivative)
			{
				const auto prediction = uncondData == nullptr ? condRaw : uncondRaw + cfgScale * (condRaw - uncondRaw);
				derivative = prediction;
			}
			else
			{
				const auto toDenoised = [&](double raw) {
					return coefficients.rawOutputIsDenoised ? raw : raw * coefficients.cOut + state * coefficients.cSkip;
				};
				const auto condDenoised = toDenoised(condRaw);
				const auto denoised = uncondData == nullptr
				                          ? condDenoised
				                          : toDenoised(uncondRaw) + cfgScale * (condDenoised - toDenoised(uncondRaw));
				derivative = sigma == 0.0 ? 0.0 : (state - denoised) / sigma;
			}
			const auto updated = state + dt * derivative;
			latentData[i] = static_cast<T>(updated);
		}
	}

	void EulerUpdateFromPredictions(LiteNN::Tensor<LiteNN::CPU>& latent,
	                                const LiteNN::Tensor<LiteNN::CPU>& condPrediction,
	                                const LiteNN::Tensor<LiteNN::CPU>* uncondPrediction,
	                                const DenoiserStepCoefficients& coefficients,
	                                double cfgScale,
	                                double sigma,
	                                double dt)
	{
		const auto validatePrediction = [&](const LiteNN::Tensor<LiteNN::CPU>& prediction, std::string_view label) {
			if (latent.DType() != prediction.DType() || latent.Shape() != prediction.Shape())
			{
				throw std::runtime_error(std::format(
				    "Euler update requires {} prediction to match latent dtype/shape; latent {} {}, prediction {} {}",
				    label, LiteNN::DataTypeName(latent.DType()), ShapeToString(latent.Shape()),
				    LiteNN::DataTypeName(prediction.DType()), ShapeToString(prediction.Shape())));
			}
		};
		validatePrediction(condPrediction, "conditional");
		if (uncondPrediction != nullptr)
		{
			validatePrediction(*uncondPrediction, "unconditional");
		}
		switch (latent.DType())
		{
		case LiteNN::DataType::Float16:
			EulerUpdateFromPredictionsTyped<LiteNN::Float16>(
			    latent, condPrediction, uncondPrediction, coefficients, cfgScale, sigma, dt);
			break;
		case LiteNN::DataType::BFloat16:
			EulerUpdateFromPredictionsTyped<LiteNN::BFloat16>(
			    latent, condPrediction, uncondPrediction, coefficients, cfgScale, sigma, dt);
			break;
		case LiteNN::DataType::Float32:
			EulerUpdateFromPredictionsTyped<float>(
			    latent, condPrediction, uncondPrediction, coefficients, cfgScale, sigma, dt);
			break;
		case LiteNN::DataType::Float64:
			EulerUpdateFromPredictionsTyped<double>(
			    latent, condPrediction, uncondPrediction, coefficients, cfgScale, sigma, dt);
			break;
		default:
			throw std::runtime_error("Euler update currently requires floating-point tensors");
		}
	}

	void PrintStats(std::string_view label, const TensorStats& stats)
	{
		std::cout << std::format("{} mean={} rms={} min={} max={} finite={} nonfinite={}\n",
		                         label, stats.mean, stats.rms, stats.min, stats.max,
		                         stats.finiteCount, stats.nonFiniteCount);
	}

	void RequireFinite(std::string label, const TensorStats& stats)
	{
		if (!stats.AllFinite())
		{
			throw std::runtime_error(std::format("{} contains {} non-finite value(s) out of {}",
			                                     label, stats.nonFiniteCount, stats.TotalCount()));
		}
	}

	std::string_view SafetensorsDataTypeName(LiteNN::DataType dtype)
	{
		switch (dtype)
		{
		case LiteNN::DataType::Float64:
			return "F64";
		case LiteNN::DataType::Float32:
			return "F32";
		case LiteNN::DataType::Float16:
			return "F16";
		case LiteNN::DataType::BFloat16:
			return "BF16";
		case LiteNN::DataType::Float8E4M3:
			return "F8_E4M3";
		case LiteNN::DataType::Float8E5M2:
			return "F8_E5M2";
		case LiteNN::DataType::Int64:
			return "I64";
		case LiteNN::DataType::Int32:
			return "I32";
		case LiteNN::DataType::Int8:
			return "I8";
		case LiteNN::DataType::UInt8:
			return "U8";
		case LiteNN::DataType::Bool:
			return "BOOL";
		}
		throw std::runtime_error("Unsupported dtype for safetensors output");
	}

	void WriteU64LE(std::ostream& out, std::uint64_t value)
	{
		for (std::size_t i = 0; i < sizeof(std::uint64_t); ++i)
		{
			const auto byte = static_cast<char>((value >> (8 * i)) & 0xFFu);
			out.write(&byte, 1);
		}
	}

	void WriteTensorSafetensors(const std::filesystem::path& path,
	                            std::string_view name,
	                            const LiteNN::Tensor<LiteNN::CPU>& tensor)
	{
		const auto byteCount = tensor.NumElements() * LiteNN::ElementByteSize(tensor.DType());
		std::string header = std::format("{{\"{}\":{{\"dtype\":\"{}\",\"shape\":[", name,
		                                 SafetensorsDataTypeName(tensor.DType()));
		for (std::size_t i = 0; i < tensor.Shape().NumDim(); ++i)
		{
			if (i != 0)
			{
				header += ",";
			}
			header += std::to_string(tensor.Shape()[i]);
		}
		header += std::format("],\"data_offsets\":[0,{}]}}}}", byteCount);

		if (path.has_parent_path())
		{
			std::filesystem::create_directories(path.parent_path());
		}
		std::ofstream out(path, std::ios::binary);
		if (!out)
		{
			throw std::runtime_error("Failed to open safetensors output file");
		}
		WriteU64LE(out, static_cast<std::uint64_t>(header.size()));
		out.write(header.data(), static_cast<std::streamsize>(header.size()));
		out.write(static_cast<const char*>(tensor.RawData()), static_cast<std::streamsize>(byteCount));
		if (!out)
		{
			throw std::runtime_error("Failed to write safetensors output file");
		}
	}

	void WriteWindowsDefFile(const std::filesystem::path& objectPath, std::string_view symbolPrefix)
	{
#if defined(_WIN32)
		const auto defPath = objectPath.parent_path() / (std::string(symbolPrefix) + "_exports.def");
		std::ofstream out(defPath);
		if (!out)
		{
			throw std::runtime_error("Failed to write Windows export definition file");
		}
		out << "EXPORTS\n"
		    << "    " << symbolPrefix << "_rodata DATA\n"
		    << "    " << symbolPrefix << "_rodata_size DATA\n"
		    << "    " << symbolPrefix << "_instructions DATA\n"
		    << "    " << symbolPrefix << "_instructions_size DATA\n";
		std::cout << std::format("Wrote Windows export definition file {}\n", defPath.string());
#else
		(void)objectPath;
		(void)symbolPrefix;
#endif
	}

	void WriteSeparatedWindowsDefFile(const std::filesystem::path& objectPath, std::string_view symbolPrefix)
	{
#if defined(_WIN32)
		const auto defPath = objectPath.parent_path() / (std::string(symbolPrefix) + "_exports.def");
		std::ofstream out(defPath);
		if (!out)
		{
			throw std::runtime_error("Failed to write Windows export definition file");
		}
		out << "EXPORTS\n"
		    << "    " << symbolPrefix << "_metadata DATA\n"
		    << "    " << symbolPrefix << "_metadata_size DATA\n"
		    << "    " << symbolPrefix << "_constants DATA\n"
		    << "    " << symbolPrefix << "_constants_size DATA\n"
		    << "    " << symbolPrefix << "_weights DATA\n"
		    << "    " << symbolPrefix << "_weights_size DATA\n"
		    << "    " << symbolPrefix << "_instructions DATA\n"
		    << "    " << symbolPrefix << "_instructions_size DATA\n";
		std::cout << std::format("Wrote Windows export definition file {}\n", defPath.string());
#else
		(void)objectPath;
		(void)symbolPrefix;
#endif
	}

	void CompileObject(const std::filesystem::path& graphPath,
	                   const std::filesystem::path& objectPath,
	                   std::string_view symbolPrefix)
	{
		auto graph = LiteNN::Serialization::LoadModel(graphPath);
		auto options = MakeExampleCompilerOptions();
		PrintCompileBudget(graph, options, "compile-object");
		auto artifact = TimedStep("compile-object codegen",
		                          [&] { return LiteNN::Compiler<LiteNN::CPU>::CompileArtifact(graph, options); });
		auto separated = artifact.SeparateRodata();
		if (!separated.Constants().empty() || !separated.Weights().empty() ||
		    !separated.ExternalTensorInfos().empty())
		{
			separated.WriteObjectFile(objectPath, symbolPrefix);
			std::cout << std::format(
			    "Wrote separated carrier object {} backend={} metadata={} bytes constants={} bytes weights={} bytes"
			    " instructions={} bytes external_tensors={}\n",
			    objectPath.string(), BackendName(separated.Backend()), separated.Metadata().size(),
			    separated.Constants().size(), separated.Weights().size(), separated.Instructions().size(),
			    separated.ExternalTensorInfos().size());
			WriteSeparatedWindowsDefFile(objectPath, symbolPrefix);
			return;
		}

		artifact.WriteObjectFile(objectPath, symbolPrefix);
		std::cout << std::format("Wrote carrier object {} backend={} rodata={} bytes instructions={} bytes\n",
		                         objectPath.string(), BackendName(artifact.Backend()), artifact.Rodata().size(),
		                         artifact.Instructions().size());
		WriteWindowsDefFile(objectPath, symbolPrefix);
	}

	void PrintModelCompileBudget(const std::filesystem::path& graphPath)
	{
		auto graph = LiteNN::Serialization::LoadModel(graphPath);
		auto options = MakeExampleCompilerOptions();
		PrintCompileBudget(graph, options, "compile-budget");
	}

	class DynamicLibrary
	{
	public:
		explicit DynamicLibrary(const std::filesystem::path& path)
		{
#if defined(_WIN32)
			handle_ = LoadLibraryW(path.c_str());
			if (!handle_)
			{
				throw std::runtime_error(std::format("Failed to open shared library {}", path.string()));
			}
#else
			handle_ = dlopen(path.string().c_str(), RTLD_NOW | RTLD_LOCAL);
			if (!handle_)
			{
				throw std::runtime_error(std::format("Failed to open shared library {}: {}", path.string(), dlerror()));
			}
#endif
		}

		DynamicLibrary(const DynamicLibrary&) = delete;
		DynamicLibrary& operator=(const DynamicLibrary&) = delete;

		~DynamicLibrary()
		{
#if defined(_WIN32)
			if (handle_)
			{
				FreeLibrary(handle_);
			}
#else
			if (handle_)
			{
				dlclose(handle_);
			}
#endif
		}

		const void* Lookup(std::string_view name) const
		{
#if defined(_WIN32)
			auto* address = reinterpret_cast<const void*>(GetProcAddress(handle_, std::string(name).c_str()));
			if (!address)
			{
				throw std::runtime_error(std::format("Missing exported symbol {}", name));
			}
			return address;
#else
			dlerror();
			auto* address = dlsym(handle_, std::string(name).c_str());
			if (!address)
			{
				throw std::runtime_error(std::format("Missing exported symbol {}: {}", name, dlerror()));
			}
			return address;
#endif
		}

		const void* TryLookup(std::string_view name) const
		{
#if defined(_WIN32)
			return reinterpret_cast<const void*>(GetProcAddress(handle_, std::string(name).c_str()));
#else
			dlerror();
			return dlsym(handle_, std::string(name).c_str());
#endif
		}

	private:
#if defined(_WIN32)
		HMODULE handle_{};
#else
		void* handle_{};
#endif
	};

	std::string SymbolName(std::string_view prefix, std::string_view suffix)
	{
		return std::string(prefix) + std::string(suffix);
	}

	std::string CompiledSpecToString(const LiteNN::CompiledTensorSpec& spec)
	{
		return std::format("{} {}", LiteNN::DataTypeName(spec.dtype), ShapeToString(spec.shape));
	}

	std::vector<LiteNN::CompiledTensorSpec> GraphInputSpecs(const LiteNN::Graph& graph)
	{
		const auto signature = graph.InputSignature();
		std::vector<LiteNN::CompiledTensorSpec> specs;
		specs.reserve(signature.size());
		for (const auto& input : signature)
		{
			specs.push_back(LiteNN::CompiledTensorSpec{
			    .dtype = input.dtype,
			    .shape = input.shape,
			    .name = input.name,
			});
		}
		return specs;
	}

	std::vector<LiteNN::CompiledTensorSpec> GraphOutputSpecs(const LiteNN::Graph& graph)
	{
		const auto signature = graph.OutputSignature();
		std::vector<LiteNN::CompiledTensorSpec> specs;
		specs.reserve(signature.size());
		for (const auto& output : signature)
		{
			specs.push_back(LiteNN::CompiledTensorSpec{
			    .dtype = output.dtype,
			    .shape = output.shape,
			    .name = output.name,
			});
		}
		return specs;
	}

	void ValidateBoundTensor(const LiteNN::CompiledTensorSpec& spec,
	                         const LiteNN::Serialization::SafetensorsTensorInfo& tensor,
	                         const std::filesystem::path& inputPath)
	{
		if (tensor.dtype != spec.dtype)
		{
			throw std::runtime_error(std::format(
			    "Input tensor '{}' from {} has dtype {}, but compiled module expects {}", spec.name,
			    inputPath.string(), LiteNN::DataTypeName(tensor.dtype), LiteNN::DataTypeName(spec.dtype)));
		}
		if (tensor.shape != spec.shape)
		{
			throw std::runtime_error(std::format(
			    "Input tensor '{}' from {} has shape {}, but compiled module expects {}", spec.name,
			    inputPath.string(), ShapeToString(tensor.shape), ShapeToString(spec.shape)));
		}
	}

	std::vector<LiteNN::Tensor<LiteNN::CPU>> MakeZeroInputs(std::span<const LiteNN::CompiledTensorSpec> specs)
	{
		std::vector<LiteNN::Tensor<LiteNN::CPU>> inputs;
		inputs.reserve(specs.size());
		for (const auto& spec : specs)
		{
			inputs.emplace_back(LiteNN::Tensor<LiteNN::CPU>(spec.shape, spec.dtype));
		}
		return inputs;
	}

	std::vector<std::string> BindingCandidates(std::string_view name, InputBindingFlavor flavor)
	{
		std::vector<std::string> candidates;
		const auto text = std::string(name);
		const bool cfgConditioningName = name == "context" || name == "vector_cond" || name == "concat_cond";
		if (flavor == InputBindingFlavor::Negative)
		{
			candidates.push_back("negative_" + text);
			candidates.push_back("uncond." + text);
			if (name == "context")
			{
				candidates.emplace_back("uncond.crossattn");
			}
			else if (name == "vector_cond")
			{
				candidates.emplace_back("uncond.vector");
			}
			else if (name == "concat_cond")
			{
				candidates.emplace_back("uncond.concat");
			}
		}
		if (flavor != InputBindingFlavor::Negative || !cfgConditioningName)
		{
			candidates.push_back(text);
		}
		if (flavor == InputBindingFlavor::Default)
		{
			candidates.push_back("cond." + text);
			if (name == "context")
			{
				candidates.emplace_back("cond.crossattn");
			}
			else if (name == "vector_cond")
			{
				candidates.emplace_back("cond.vector");
			}
			else if (name == "concat_cond")
			{
				candidates.emplace_back("cond.concat");
			}
		}
		return candidates;
	}

	const LiteNN::Serialization::SafetensorsTensorInfo* FindBindingTensor(
	    const LiteNN::Serialization::SafetensorsArchive& archive,
	    std::string_view name,
	    InputBindingFlavor flavor,
	    std::string& matchedName)
	{
		for (const auto& candidate : BindingCandidates(name, flavor))
		{
			if (const auto* tensor = archive.FindTensor(candidate))
			{
				matchedName = candidate;
				return tensor;
			}
		}
		return nullptr;
	}

	std::vector<LiteNN::Tensor<LiteNN::CPU>> MakeInputsFromSafetensors(
	    std::span<const LiteNN::CompiledTensorSpec> specs,
	    const std::filesystem::path& inputPath,
	    bool zeroFillMissing,
	    InputBindingFlavor flavor = InputBindingFlavor::Default)
	{
		auto archive = LiteNN::Serialization::SafetensorsArchive::LoadFile(inputPath);
		std::vector<LiteNN::Tensor<LiteNN::CPU>> inputs;
		inputs.reserve(specs.size());
		for (std::size_t i = 0; i < specs.size(); ++i)
		{
			const auto& spec = specs[i];
			if (spec.name.empty())
			{
				throw std::runtime_error(std::format(
				    "Compiled input {} has no name; safetensors input binding requires named inputs", i));
			}
			std::string matchedName;
			const auto* tensor = FindBindingTensor(archive, spec.name, flavor, matchedName);
			if (tensor == nullptr)
			{
				if (zeroFillMissing)
				{
					inputs.emplace_back(spec.shape, spec.dtype);
					std::cout << std::format("  input {} '{}' zero-filled ({})\n", i, spec.name,
					                         CompiledSpecToString(spec))
					          << std::flush;
					continue;
				}
				throw std::runtime_error(std::format(
				    "Missing compiled input '{}' in safetensors bindings file {}", spec.name, inputPath.string()));
			}
			ValidateBoundTensor(spec, *tensor, inputPath);
			auto& input = inputs.emplace_back(LiteNN::Uninitialized, spec.shape, spec.dtype);
			const auto bytes = archive.TensorData(*tensor);
			std::memcpy(input.RawData(), bytes.data(), bytes.size());
			std::cout << std::format("  input {} '{}' bound from {}:{} ({} byte(s))\n", i, spec.name,
			                         inputPath.string(), matchedName, tensor->ByteSize())
			          << std::flush;
		}
		return inputs;
	}

	void PrintOutputs(std::span<const LiteNN::CompiledTensorSpec> specs,
	                  std::span<const LiteNN::Tensor<LiteNN::CPU>> outputs,
	                  bool failOnNonFinite)
	{
		for (std::size_t i = 0; i < outputs.size(); ++i)
		{
			const auto& spec = specs[i];
			std::cout << std::format("  output {} '{}' dtype={} elements={}\n", i, spec.name,
			                         LiteNN::DataTypeName(outputs[i].DType()), outputs[i].NumElements());
			if (LiteNN::IsFloatingDataType(outputs[i].DType()) && outputs[i].NumElements() != 0)
			{
				const auto stats = ComputeTensorStats(outputs[i]);
				PrintStats("    stats", stats);
				if (failOnNonFinite)
				{
					RequireFinite(std::format("output {} '{}'", i, spec.name), stats);
				}
			}
		}
	}

	void WriteSingleOutputIfRequested(const std::optional<std::filesystem::path>& outputPath,
	                                  std::span<const LiteNN::CompiledTensorSpec> specs,
	                                  std::span<const LiteNN::Tensor<LiteNN::CPU>> outputs)
	{
		if (!outputPath)
		{
			return;
		}
		if (outputs.size() != 1)
		{
			throw std::runtime_error("--output currently requires the compiled module to have exactly one output");
		}
		const auto& spec = specs[0];
		const auto outputName = spec.name.empty() ? std::string("output") : spec.name;
		WriteTensorSafetensors(*outputPath, outputName, outputs[0]);
		std::cout << std::format("  wrote output tensor '{}' {}\n", outputName, outputPath->string());
	}

	using BenchmarkClock = std::chrono::steady_clock;

	struct ModelBenchmarkOptions
	{
		std::string device{ "cpu" };
		std::size_t warmup{ 1 };
		std::size_t iterations{ 1 };
		std::optional<std::filesystem::path> jsonOutput;
	};

	struct ModelBenchmarkResult
	{
		std::string device;
		std::string status{ "ok" };
		std::string message;
		std::string backend;
		std::size_t warmup{};
		std::size_t iterations{};
		double compileMs{};
		double loadMs{};
		double inputBindMs{};
		double inputUploadMs{};
		double runTotalMs{};
		double runMeanMs{};
		std::uint64_t rodataBytes{};
		std::uint64_t instructionBytes{};
		std::uint64_t inputBytes{};
		std::uint64_t outputBytes{};
	};

	double ElapsedMs(BenchmarkClock::time_point begin, BenchmarkClock::time_point end)
	{
		return std::chrono::duration<double, std::milli>(end - begin).count();
	}

	std::uint64_t SpecByteSize(const LiteNN::CompiledTensorSpec& spec)
	{
		return static_cast<std::uint64_t>(LiteNN::ShapeView{ spec.shape }.NumElements()) *
		       LiteNN::ElementByteSize(spec.dtype);
	}

	std::uint64_t SpecBytes(std::span<const LiteNN::CompiledTensorSpec> specs)
	{
		std::uint64_t result = 0;
		for (const auto& spec : specs)
		{
			result += SpecByteSize(spec);
		}
		return result;
	}

	std::vector<LiteNN::Tensor<LiteNN::CPU>> AllocateCPUOutputs(
	    std::span<const LiteNN::CompiledTensorSpec> specs)
	{
		std::vector<LiteNN::Tensor<LiteNN::CPU>> outputs;
		outputs.reserve(specs.size());
		for (const auto& spec : specs)
		{
			outputs.emplace_back(LiteNN::Uninitialized, LiteNN::ShapeView{ spec.shape }, spec.dtype,
			                     LiteNN::CPU{});
		}
		return outputs;
	}

	std::string JsonEscape(std::string_view text)
	{
		std::string result;
		result.reserve(text.size() + 8);
		for (const char ch : text)
		{
			switch (ch)
			{
			case '\\':
				result += "\\\\";
				break;
			case '"':
				result += "\\\"";
				break;
			case '\n':
				result += "\\n";
				break;
			case '\r':
				result += "\\r";
				break;
			case '\t':
				result += "\\t";
				break;
			default:
				if (static_cast<unsigned char>(ch) < 0x20)
				{
					result += std::format("\\u{:04x}", static_cast<unsigned int>(static_cast<unsigned char>(ch)));
				}
				else
				{
					result.push_back(ch);
				}
				break;
			}
		}
		return result;
	}

	void WriteBenchmarkJson(const std::filesystem::path& path, const ModelBenchmarkResult& result)
	{
		std::ofstream out(path, std::ios::binary);
		if (!out)
		{
			throw std::runtime_error("Failed to open benchmark JSON output");
		}
		out << "{\n"
		    << "  \"device\": \"" << JsonEscape(result.device) << "\",\n"
		    << "  \"status\": \"" << JsonEscape(result.status) << "\",\n"
		    << "  \"message\": \"" << JsonEscape(result.message) << "\",\n"
		    << "  \"backend\": \"" << JsonEscape(result.backend) << "\",\n"
		    << "  \"warmup\": " << result.warmup << ",\n"
		    << "  \"iterations\": " << result.iterations << ",\n"
		    << "  \"compile_ms\": " << result.compileMs << ",\n"
		    << "  \"load_ms\": " << result.loadMs << ",\n"
		    << "  \"input_bind_ms\": " << result.inputBindMs << ",\n"
		    << "  \"input_upload_ms\": " << result.inputUploadMs << ",\n"
		    << "  \"run_total_ms\": " << result.runTotalMs << ",\n"
		    << "  \"run_mean_ms\": " << result.runMeanMs << ",\n"
		    << "  \"rodata_bytes\": " << result.rodataBytes << ",\n"
		    << "  \"instruction_bytes\": " << result.instructionBytes << ",\n"
		    << "  \"input_bytes\": " << result.inputBytes << ",\n"
		    << "  \"output_bytes\": " << result.outputBytes << "\n"
		    << "}\n";
	}

	void PrintBenchmarkResult(const ModelBenchmarkResult& result)
	{
		std::cout << std::format(
		    "Benchmark device={} status={} backend={} compile_ms={} load_ms={} input_bind_ms={}"
		    " input_upload_ms={} run_mean_ms={} iterations={} rodata_bytes={} instruction_bytes={}"
		    " input_bytes={} output_bytes={}\n",
		    result.device, result.status, result.backend, result.compileMs, result.loadMs, result.inputBindMs,
		    result.inputUploadMs, result.runMeanMs, result.iterations, result.rodataBytes,
		    result.instructionBytes, result.inputBytes, result.outputBytes);
		if (!result.message.empty())
		{
			std::cout << "  message: " << result.message << '\n';
		}
	}

	ModelBenchmarkOptions ParseModelBenchmarkOptions(int argc, char** argv, int optionStart)
	{
		ModelBenchmarkOptions options;
		for (int i = optionStart; i < argc; ++i)
		{
			const std::string_view option(argv[i]);
			const auto requireValue = [&](std::string_view label) -> std::string_view {
				if (i + 1 >= argc)
				{
					throw std::runtime_error(std::format("{} requires a value", label));
				}
				++i;
				return argv[i];
			};
			if (option == "--device")
			{
				options.device = std::string(requireValue(option));
				if (options.device != "cpu" && options.device != "cuda")
				{
					throw std::runtime_error("--device must be cpu or cuda");
				}
			}
			else if (option == "--warmup")
			{
				options.warmup = ParseSize(requireValue(option), "--warmup");
			}
			else if (option == "--iterations")
			{
				options.iterations = ParseSize(requireValue(option), "--iterations");
				if (options.iterations == 0)
				{
					throw std::runtime_error("--iterations must be positive");
				}
			}
			else if (option == "--json")
			{
				options.jsonOutput = std::filesystem::path(requireValue(option));
			}
			else
			{
				throw std::runtime_error("Unknown --benchmark-model-with-inputs option: " + std::string(option));
			}
		}
		return options;
	}

	ModelBenchmarkResult BenchmarkCPUModelWithInputs(const std::filesystem::path& graphPath,
	                                                 const std::filesystem::path& inputPath,
	                                                 const ModelBenchmarkOptions& options)
	{
		ModelBenchmarkResult result;
		result.device = "cpu";
		result.warmup = options.warmup;
		result.iterations = options.iterations;

		auto begin = BenchmarkClock::now();
		auto graph = LiteNN::Serialization::LoadModel(graphPath);
		auto compilerOptions = MakeExampleCompilerOptions();
		PrintCompileBudget(graph, compilerOptions, "benchmark-cpu");
		auto artifact = LiteNN::Compiler<LiteNN::CPU>::CompileArtifact(graph, compilerOptions);
		auto end = BenchmarkClock::now();
		result.compileMs = ElapsedMs(begin, end);
		result.backend = std::string(BackendName(artifact.Backend()));
		result.rodataBytes = artifact.Rodata().size();
		result.instructionBytes = artifact.Instructions().size();
		result.inputBytes = SpecBytes(artifact.InputSpecs());
		result.outputBytes = SpecBytes(artifact.OutputSpecs());

		begin = BenchmarkClock::now();
		auto module = artifact.Load();
		end = BenchmarkClock::now();
		result.loadMs = ElapsedMs(begin, end);

		begin = BenchmarkClock::now();
		auto inputs = MakeInputsFromSafetensors(module.InputSpecs(), inputPath, false);
		auto outputs = AllocateCPUOutputs(module.OutputSpecs());
		end = BenchmarkClock::now();
		result.inputBindMs = ElapsedMs(begin, end);

		for (std::size_t i = 0; i < options.warmup; ++i)
		{
			module.RunInto(std::span<const LiteNN::Tensor<LiteNN::CPU>>(inputs),
			               std::span<LiteNN::Tensor<LiteNN::CPU>>(outputs));
		}
		begin = BenchmarkClock::now();
		for (std::size_t i = 0; i < options.iterations; ++i)
		{
			module.RunInto(std::span<const LiteNN::Tensor<LiteNN::CPU>>(inputs),
			               std::span<LiteNN::Tensor<LiteNN::CPU>>(outputs));
		}
		end = BenchmarkClock::now();
		result.runTotalMs = ElapsedMs(begin, end);
		result.runMeanMs = result.runTotalMs / static_cast<double>(options.iterations);
		return result;
	}

#ifdef LITENN_ENABLE_CUDA
	std::vector<LiteNN::Tensor<LiteNN::CUDA>> CopyInputsToCUDA(
	    std::span<const LiteNN::Tensor<LiteNN::CPU>> cpuInputs,
	    LiteNN::CUDA device)
	{
		std::vector<LiteNN::Tensor<LiteNN::CUDA>> inputs;
		inputs.reserve(cpuInputs.size());
		for (const auto& input : cpuInputs)
		{
			inputs.emplace_back(input.CopyToDevice(device));
		}
		return inputs;
	}

	std::vector<LiteNN::Tensor<LiteNN::CUDA>> AllocateCUDAOutputs(
	    std::span<const LiteNN::CompiledTensorSpec> specs,
	    LiteNN::CUDA device)
	{
		std::vector<LiteNN::Tensor<LiteNN::CUDA>> outputs;
		outputs.reserve(specs.size());
		for (const auto& spec : specs)
		{
			outputs.emplace_back(LiteNN::Uninitialized, LiteNN::ShapeView{ spec.shape }, spec.dtype, device);
		}
		return outputs;
	}
#endif

	ModelBenchmarkResult BenchmarkCUDAModelWithInputs(const std::filesystem::path& graphPath,
	                                                  const std::filesystem::path& inputPath,
	                                                  const ModelBenchmarkOptions& options)
	{
		ModelBenchmarkResult result;
		result.device = "cuda";
		result.warmup = options.warmup;
		result.iterations = options.iterations;
#ifdef LITENN_ENABLE_CUDA
		if (!LiteNN::IsCUDADeviceAvailable())
		{
			result.status = "skipped";
			result.message = "CUDA device is not available";
			return result;
		}
		LiteNN::CUDA device{};
		auto begin = BenchmarkClock::now();
		auto graph = LiteNN::Serialization::LoadModel(graphPath);
		auto compilerOptions = MakeExampleCompilerOptions();
		PrintCompileBudget(graph, compilerOptions, "benchmark-cuda");
		auto artifact = LiteNN::Compiler<LiteNN::CUDA>::CompileArtifact(graph, compilerOptions);
		auto end = BenchmarkClock::now();
		result.compileMs = ElapsedMs(begin, end);
		result.backend = std::string(BackendName(artifact.Backend()));
		result.rodataBytes = artifact.Rodata().size();
		result.instructionBytes = artifact.Instructions().size();
		result.inputBytes = SpecBytes(artifact.InputSpecs());
		result.outputBytes = SpecBytes(artifact.OutputSpecs());

		begin = BenchmarkClock::now();
		auto module = artifact.Load(device);
		end = BenchmarkClock::now();
		result.loadMs = ElapsedMs(begin, end);

		begin = BenchmarkClock::now();
		auto cpuInputs = MakeInputsFromSafetensors(module.InputSpecs(), inputPath, false);
		end = BenchmarkClock::now();
		result.inputBindMs = ElapsedMs(begin, end);

		begin = BenchmarkClock::now();
		auto inputs = CopyInputsToCUDA(std::span<const LiteNN::Tensor<LiteNN::CPU>>(cpuInputs), device);
		auto outputs = AllocateCUDAOutputs(module.OutputSpecs(), device);
		end = BenchmarkClock::now();
		result.inputUploadMs = ElapsedMs(begin, end);

		for (std::size_t i = 0; i < options.warmup; ++i)
		{
			module.RunInto(std::span<const LiteNN::Tensor<LiteNN::CUDA>>(inputs),
			               std::span<LiteNN::Tensor<LiteNN::CUDA>>(outputs));
		}
		begin = BenchmarkClock::now();
		for (std::size_t i = 0; i < options.iterations; ++i)
		{
			module.RunInto(std::span<const LiteNN::Tensor<LiteNN::CUDA>>(inputs),
			               std::span<LiteNN::Tensor<LiteNN::CUDA>>(outputs));
		}
		end = BenchmarkClock::now();
		result.runTotalMs = ElapsedMs(begin, end);
		result.runMeanMs = result.runTotalMs / static_cast<double>(options.iterations);
#else
		(void)graphPath;
		(void)inputPath;
		result.status = "skipped";
		result.message = "LiteNN was built without CUDA support";
#endif
		return result;
	}

	void BenchmarkModelWithInputs(const std::filesystem::path& graphPath,
	                              const std::filesystem::path& inputPath,
	                              const ModelBenchmarkOptions& options)
	{
		const auto result = options.device == "cuda" ? BenchmarkCUDAModelWithInputs(graphPath, inputPath, options)
		                                             : BenchmarkCPUModelWithInputs(graphPath, inputPath, options);
		PrintBenchmarkResult(result);
		if (options.jsonOutput)
		{
			WriteBenchmarkJson(*options.jsonOutput, result);
		}
	}

	std::vector<LiteNN::Tensor<LiteNN::CPU>> RunInputsAndPrint(
	    const LiteNN::CompiledModule<LiteNN::CPU>& module,
	    std::span<const LiteNN::Tensor<LiteNN::CPU>> inputs,
	    bool failOnNonFinite = true)
	{
		auto outputs = module.Run(inputs);
		PrintOutputs(module.OutputSpecs(), outputs, failOnNonFinite);
		return outputs;
	}

	void RunZeroInputsAndPrint(const LiteNN::CompiledModule<LiteNN::CPU>& module)
	{
		auto inputs = MakeZeroInputs(module.InputSpecs());
		RunInputsAndPrint(module, inputs);
	}

	void CompileAndRunModel(const std::filesystem::path& graphPath)
	{
		LiteNN::CompiledModule<LiteNN::CPU> module;
		{
		auto graph = LiteNN::Serialization::LoadModel(graphPath);
		auto options = MakeExampleCompilerOptions();
		PrintCompileBudget(graph, options, "run-model");
		auto artifact = TimedStep("run-model codegen",
		                          [&] { return LiteNN::Compiler<LiteNN::CPU>::CompileArtifact(graph, options); });
			std::cout << std::format("Compiled {} backend={} rodata={} bytes instructions={} bytes\n",
			                         graphPath.string(), BackendName(artifact.Backend()), artifact.Rodata().size(),
			                         artifact.Instructions().size())
			          << std::flush;
			module = artifact.Load();
		}
		std::cout << std::format("Loaded compiled model backend={} input_count={} output_count={}\n",
		                         BackendName(module.Backend()), module.InputSpecs().size(), module.OutputSpecs().size())
		          << std::flush;
		RunZeroInputsAndPrint(module);
	}

	void CompileAndRunModelWithInputs(const std::filesystem::path& graphPath,
	                                  const std::filesystem::path& inputPath,
	                                  const OutputRunOptions& outputOptions)
	{
		LiteNN::CompiledModule<LiteNN::CPU> module;
		{
		auto graph = LiteNN::Serialization::LoadModel(graphPath);
		auto options = MakeExampleCompilerOptions();
		PrintCompileBudget(graph, options, "run-model-with-inputs");
		auto artifact = TimedStep("run-model-with-inputs codegen",
		                          [&] { return LiteNN::Compiler<LiteNN::CPU>::CompileArtifact(graph, options); });
			std::cout << std::format("Compiled {} backend={} rodata={} bytes instructions={} bytes\n",
			                         graphPath.string(), BackendName(artifact.Backend()), artifact.Rodata().size(),
			                         artifact.Instructions().size())
			          << std::flush;
			module = artifact.Load();
		}
		std::cout << std::format("Loaded compiled model backend={} input_count={} output_count={}\n",
		                         BackendName(module.Backend()), module.InputSpecs().size(), module.OutputSpecs().size())
		          << std::flush;
		auto inputs = MakeInputsFromSafetensors(module.InputSpecs(), inputPath, false);
		auto outputs = RunInputsAndPrint(module, inputs, outputOptions.failOnNonFinite);
		WriteSingleOutputIfRequested(outputOptions.outputPath, module.OutputSpecs(), outputs);
	}

	void DiagnoseModelWithInputs(const std::filesystem::path& graphPath,
	                             const std::filesystem::path& inputPath,
	                             const FiniteDiagnosticOptions& options)
	{
		auto graph = LiteNN::Serialization::LoadModel(graphPath);
		const auto inputSpecs = GraphInputSpecs(graph);
		const auto outputSpecs = GraphOutputSpecs(graph);
		auto inputs = MakeInputsFromSafetensors(inputSpecs, inputPath, false);

		std::size_t visitedNodes = 0;
		std::size_t floatingTensors = 0;
		std::size_t nonFiniteTensors = 0;
		std::size_t nonFiniteValues = 0;
		bool printedFirstNonFinite = false;

		std::cout << std::format("Running finite diagnostics for {} with {} input(s), {} output(s)\n",
		                         graphPath.string(), inputSpecs.size(), outputSpecs.size())
		          << std::flush;

		LiteNN::Runtime::Interpreter<LiteNN::CPU> interpreter;
		auto outputs = interpreter.RunForwardWithTrace(
		    graph, std::span<const LiteNN::Tensor<LiteNN::CPU>>(inputs.data(), inputs.size()),
		    [&](LiteNN::SubgraphId subgraphId, LiteNN::NodeId nodeId, const LiteNN::NodeEntry& entry,
		        std::span<const LiteNN::Tensor<LiteNN::CPU>> nodeOutputs) {
			    ++visitedNodes;
			    if (options.maxNodes && visitedNodes > *options.maxNodes)
			    {
				    throw std::runtime_error(std::format(
				        "Finite diagnostics reached --max-nodes {} without finding a non-finite tensor",
				        *options.maxNodes));
			    }
			    for (std::size_t port = 0; port < nodeOutputs.size(); ++port)
			    {
				    const auto& tensor = nodeOutputs[port];
				    if (!LiteNN::IsFloatingDataType(tensor.DType()))
				    {
					    if (options.verbose)
					    {
						    std::cout << std::format("  subgraph={} node={} kind={} port={} dtype={} shape={} skipped\n",
						                             subgraphId, nodeId, LiteNN::Validation::NodeKindName(entry.node),
						                             port, LiteNN::DataTypeName(tensor.DType()),
						                             ShapeToString(tensor.Shape()));
					    }
					    continue;
				    }
				    ++floatingTensors;
				    const auto stats = ComputeTensorStats(tensor);
				    if (stats.nonFiniteCount != 0)
				    {
					    ++nonFiniteTensors;
					    nonFiniteValues += stats.nonFiniteCount;
					    std::cout << std::format(
					        "  non-finite subgraph={} node={} kind={} port={} dtype={} shape={} nonfinite={}/{}"
					        " mean={} rms={} min={} max={}\n",
					        subgraphId, nodeId, LiteNN::Validation::NodeKindName(entry.node), port,
					        LiteNN::DataTypeName(tensor.DType()), ShapeToString(tensor.Shape()),
					        stats.nonFiniteCount, stats.TotalCount(), stats.mean, stats.rms, stats.min, stats.max);
					    printedFirstNonFinite = true;
					    if (options.failOnNonFinite)
					    {
						    throw std::runtime_error("Finite diagnostics stopped at the first non-finite tensor");
					    }
				    }
				    else if (options.verbose)
				    {
					    std::cout << std::format(
					        "  finite subgraph={} node={} kind={} port={} dtype={} shape={} mean={} rms={} min={} max={}\n",
					        subgraphId, nodeId, LiteNN::Validation::NodeKindName(entry.node), port,
					        LiteNN::DataTypeName(tensor.DType()), ShapeToString(tensor.Shape()), stats.mean,
					        stats.rms, stats.min, stats.max);
				    }
			    }
		    });

		std::cout << std::format(
		    "Finite diagnostics completed: visited_nodes={} floating_tensors={} nonfinite_tensors={}"
		    " nonfinite_values={}\n",
		    visitedNodes, floatingTensors, nonFiniteTensors, nonFiniteValues);
		if (!printedFirstNonFinite)
		{
			std::cout << "  no non-finite floating outputs found\n";
		}
		PrintOutputs(outputSpecs, outputs, options.failOnNonFinite);
	}

	void CompileRawObject(const std::filesystem::path& graphPath, const std::filesystem::path& objectPath)
	{
		auto graph = LiteNN::Serialization::LoadModel(graphPath);
		auto options = MakeExampleCompilerOptions();
		PrintCompileBudget(graph, options, "compile-raw-object");
		auto artifact = TimedStep("compile-raw-object codegen",
		                          [&] { return LiteNN::Compiler<LiteNN::CPU>::CompileArtifact(graph, options); });
		std::ofstream out(objectPath, std::ios::binary);
		if (!out)
		{
			throw std::runtime_error("Failed to open output raw object file");
		}
		const auto instructions = artifact.Instructions();
		out.write(reinterpret_cast<const char*>(instructions.data()), static_cast<std::streamsize>(instructions.size()));
		if (!out)
		{
			throw std::runtime_error("Failed to write output raw object file");
		}
		std::cout << std::format("Wrote raw instruction object {} backend={} rodata={} bytes instructions={} bytes\n",
		                         objectPath.string(), BackendName(artifact.Backend()), artifact.Rodata().size(),
		                         instructions.size());
	}

	void CompileImageRegions(const std::filesystem::path& graphPath,
	                         const std::filesystem::path& outputDir,
	                         std::string_view filePrefix,
	                         const ExampleCompilerSettings& compilerSettings = {})
	{
		auto graph = LiteNN::Serialization::LoadModel(graphPath);
		auto options = MakeExampleCompilerOptions(compilerSettings);
		PrintCompileBudget(graph, options, "compile-image-regions");
		auto artifact = TimedStep("compile-image-regions codegen",
		                          [&] { return LiteNN::Compiler<LiteNN::CPU>::CompileArtifact(graph, options); });
		std::filesystem::create_directories(outputDir);
		const auto prefix = std::string(filePrefix);
		const auto rodataPath = outputDir / (prefix + ".rodata.bin");
		const auto instructionsPath = outputDir / (prefix + ".instructions.obj");
		const auto constants = artifact.Constants();
		const auto weights = artifact.Weights();
		const auto externalTensorInfos = artifact.ExternalTensorInfos();
		if (!constants.empty() || !weights.empty() || !externalTensorInfos.empty())
		{
			const auto constantsPath = outputDir / (prefix + ".constants.bin");
			const auto weightsPath = outputDir / (prefix + ".weights.bin");
			auto metadata = TimedStep("compile-image-regions metadata",
			                          [&] { return artifact.BuildSeparatedMetadata(); });
			TimedStep("compile-image-regions write metadata", [&] { WriteAllBytes(rodataPath, metadata); });
			TimedStep("compile-image-regions write constants", [&] { WriteAllBytes(constantsPath, constants); });
			TimedStep("compile-image-regions write weights", [&] { WriteAllBytes(weightsPath, weights); });
			TimedStep("compile-image-regions write instructions",
			          [&] { WriteAllBytes(instructionsPath, artifact.Instructions()); });
			std::cout << std::format(
			    "Wrote separated image regions metadata={} constants={} weights={} instructions={} backend={}"
			    " metadata={} bytes constants={} bytes weights={} bytes instructions={} bytes external_tensors={}\n",
			    rodataPath.string(), constantsPath.string(), weightsPath.string(), instructionsPath.string(),
			    BackendName(artifact.Backend()), metadata.size(), constants.size(), weights.size(),
			    artifact.Instructions().size(), externalTensorInfos.size());
			return;
		}

		WriteAllBytes(rodataPath, artifact.Rodata());
		WriteAllBytes(instructionsPath, artifact.Instructions());
		std::cout << std::format("Wrote image regions rodata={} instructions={} backend={} rodata={} bytes"
		                         " instructions={} bytes\n",
		                         rodataPath.string(), instructionsPath.string(), BackendName(artifact.Backend()),
		                         artifact.Rodata().size(), artifact.Instructions().size());
	}

	std::optional<std::string> RegionFilePrefixFromRodataPath(const std::filesystem::path& rodataPath)
	{
		const auto filename = rodataPath.filename().string();
		const std::string suffix = ".rodata.bin";
		if (filename.size() <= suffix.size() || !filename.ends_with(suffix))
		{
			return std::nullopt;
		}
		return filename.substr(0, filename.size() - suffix.size());
	}

	std::optional<LiteNN::CompiledModuleSeparatedArtifact> TryLoadSeparatedImageRegions(
	    const std::filesystem::path& rodataPath,
	    const std::filesystem::path& instructionsPath)
	{
		const auto prefix = RegionFilePrefixFromRodataPath(rodataPath);
		if (!prefix)
		{
			return std::nullopt;
		}
		const auto directory = rodataPath.parent_path();
		const auto constantsPath = directory / (*prefix + ".constants.bin");
		const auto weightsPath = directory / (*prefix + ".weights.bin");
		if (!std::filesystem::exists(constantsPath) || !std::filesystem::exists(weightsPath))
		{
			return std::nullopt;
		}

		auto metadata = ReadAllBytes(rodataPath);
		auto constants = ReadAllBytes(constantsPath);
		auto weights = ReadAllBytes(weightsPath);
		auto instructions = ReadAllBytes(instructionsPath);
		try
		{
			return LiteNN::CompiledModuleSeparatedArtifact::FromOwnedRegions(
			    std::move(metadata), std::move(constants), std::move(weights), std::move(instructions));
		}
		catch (const std::exception&)
		{
			return std::nullopt;
		}
	}

	LiteNN::CompiledModuleArtifact LoadArtifactFromLibrary(const DynamicLibrary& library,
	                                                       std::string_view symbolPrefix)
	{
		return LiteNN::CompiledModuleArtifact::FromExportedSymbols({
		    .rodata = library.Lookup(SymbolName(symbolPrefix, "_rodata")),
		    .rodataSize = library.Lookup(SymbolName(symbolPrefix, "_rodata_size")),
		    .instructions = library.Lookup(SymbolName(symbolPrefix, "_instructions")),
		    .instructionSize = library.Lookup(SymbolName(symbolPrefix, "_instructions_size")),
		});
	}

	std::optional<LiteNN::CompiledModuleSeparatedArtifact> TryLoadSeparatedArtifactFromLibrary(
	    const DynamicLibrary& library,
	    std::string_view symbolPrefix)
	{
		const auto metadata = library.TryLookup(SymbolName(symbolPrefix, "_metadata"));
		const auto metadataSize = library.TryLookup(SymbolName(symbolPrefix, "_metadata_size"));
		if (!metadata || !metadataSize)
		{
			return std::nullopt;
		}
		return LiteNN::CompiledModuleSeparatedArtifact::FromExportedSymbols({
		    .metadata = metadata,
		    .metadataSize = metadataSize,
		    .constants = library.Lookup(SymbolName(symbolPrefix, "_constants")),
		    .constantsSize = library.Lookup(SymbolName(symbolPrefix, "_constants_size")),
		    .weights = library.Lookup(SymbolName(symbolPrefix, "_weights")),
		    .weightsSize = library.Lookup(SymbolName(symbolPrefix, "_weights_size")),
		    .instructions = library.Lookup(SymbolName(symbolPrefix, "_instructions")),
		    .instructionsSize = library.Lookup(SymbolName(symbolPrefix, "_instructions_size")),
		});
	}

	void LoadDllAndRun(const std::filesystem::path& libraryPath, std::string_view symbolPrefix)
	{
		DynamicLibrary library(libraryPath);
		if (auto separated = TryLoadSeparatedArtifactFromLibrary(library, symbolPrefix))
		{
			std::cout << std::format(
			    "Loaded separated carrier image {} backend={} metadata={} bytes constants={} bytes weights={} bytes"
			    " instructions={} bytes external_tensors={}\n",
			    libraryPath.string(), BackendName(separated->Backend()), separated->Metadata().size(),
			    separated->Constants().size(), separated->Weights().size(), separated->Instructions().size(),
			    separated->ExternalTensorInfos().size())
			          << std::flush;
			auto module = separated->LoadBorrowedExternalRegions();
			std::cout << std::format("Loaded separated carrier DLL {} backend={} input_count={} output_count={}\n",
			                         libraryPath.string(), BackendName(module.Backend()), module.InputSpecs().size(),
			                         module.OutputSpecs().size())
			          << std::flush;
			RunZeroInputsAndPrint(module);
			return;
		}

		auto artifact = LoadArtifactFromLibrary(library, symbolPrefix);
		std::cout << std::format("Loaded carrier image {} backend={} rodata={} bytes instructions={} bytes\n",
		                         libraryPath.string(), BackendName(artifact.Backend()), artifact.Rodata().size(),
		                         artifact.Instructions().size())
		          << std::flush;
		auto module = artifact.Load();
		std::cout << std::format("Loaded carrier DLL {} backend={} input_count={} output_count={}\n",
		                         libraryPath.string(), BackendName(module.Backend()),
		                         module.InputSpecs().size(), module.OutputSpecs().size())
		          << std::flush;
		RunZeroInputsAndPrint(module);
	}

	void LoadDllAndRunWithInputs(const std::filesystem::path& libraryPath,
	                             const std::filesystem::path& inputPath,
	                             std::string_view symbolPrefix,
	                             const OutputRunOptions& outputOptions)
	{
		DynamicLibrary library(libraryPath);
		if (auto separated = TryLoadSeparatedArtifactFromLibrary(library, symbolPrefix))
		{
			std::cout << std::format(
			    "Loaded separated carrier image {} backend={} metadata={} bytes constants={} bytes weights={} bytes"
			    " instructions={} bytes external_tensors={}\n",
			    libraryPath.string(), BackendName(separated->Backend()), separated->Metadata().size(),
			    separated->Constants().size(), separated->Weights().size(), separated->Instructions().size(),
			    separated->ExternalTensorInfos().size())
			          << std::flush;
			auto module = separated->LoadBorrowedExternalRegions();
			std::cout << std::format("Loaded separated carrier DLL {} backend={} input_count={} output_count={}\n",
			                         libraryPath.string(), BackendName(module.Backend()), module.InputSpecs().size(),
			                         module.OutputSpecs().size())
			          << std::flush;
			auto inputs = MakeInputsFromSafetensors(module.InputSpecs(), inputPath, false);
			auto outputs = RunInputsAndPrint(module, inputs, outputOptions.failOnNonFinite);
			WriteSingleOutputIfRequested(outputOptions.outputPath, module.OutputSpecs(), outputs);
			return;
		}

		auto artifact = LoadArtifactFromLibrary(library, symbolPrefix);
		std::cout << std::format("Loaded carrier image {} backend={} rodata={} bytes instructions={} bytes\n",
		                         libraryPath.string(), BackendName(artifact.Backend()), artifact.Rodata().size(),
		                         artifact.Instructions().size())
		          << std::flush;
		auto module = artifact.Load();
		std::cout << std::format("Loaded carrier DLL {} backend={} input_count={} output_count={}\n",
		                         libraryPath.string(), BackendName(module.Backend()),
		                         module.InputSpecs().size(), module.OutputSpecs().size())
		          << std::flush;
		auto inputs = MakeInputsFromSafetensors(module.InputSpecs(), inputPath, false);
		auto outputs = RunInputsAndPrint(module, inputs, outputOptions.failOnNonFinite);
		WriteSingleOutputIfRequested(outputOptions.outputPath, module.OutputSpecs(), outputs);
	}

	void LoadImageAndRunWithInputs(const std::filesystem::path& rodataPath,
	                               const std::filesystem::path& instructionsPath,
	                               const std::filesystem::path& inputPath,
	                               const OutputRunOptions& outputOptions)
	{
		if (auto separated = TryLoadSeparatedImageRegions(rodataPath, instructionsPath))
		{
			auto module = separated->LoadBorrowedExternalRegions();
			std::cout << std::format(
			    "Loaded separated image regions metadata={} instructions={} backend={} input_count={} output_count={}"
			    " constants={} bytes weights={} bytes external_tensors={}\n",
			    rodataPath.string(), instructionsPath.string(), BackendName(module.Backend()),
			    module.InputSpecs().size(), module.OutputSpecs().size(), separated->Constants().size(),
			    separated->Weights().size(), separated->ExternalTensorInfos().size())
			          << std::flush;
			auto inputs = MakeInputsFromSafetensors(module.InputSpecs(), inputPath, false);
			auto outputs = RunInputsAndPrint(module, inputs, outputOptions.failOnNonFinite);
			WriteSingleOutputIfRequested(outputOptions.outputPath, module.OutputSpecs(), outputs);
			return;
		}

		auto rodata = ReadAllBytes(rodataPath);
		auto instructions = ReadAllBytes(instructionsPath);
		auto module = LiteNN::CompiledModule<LiteNN::CPU>::Load({
		    .rodata = rodata.data(),
		    .rodataSize = rodata.size(),
		    .instructions = instructions.data(),
		    .instructionSize = instructions.size(),
		});
		std::cout << std::format("Loaded image regions backend={} input_count={} output_count={}\n",
		                         BackendName(module.Backend()), module.InputSpecs().size(), module.OutputSpecs().size())
		          << std::flush;
		auto inputs = MakeInputsFromSafetensors(module.InputSpecs(), inputPath, false);
		auto outputs = RunInputsAndPrint(module, inputs, outputOptions.failOnNonFinite);
		WriteSingleOutputIfRequested(outputOptions.outputPath, module.OutputSpecs(), outputs);
	}

	void SampleEulerModule(const LiteNN::CompiledModule<LiteNN::CPU>& module,
	                       std::string_view sourceLabel,
	                       const EulerSamplerOptions& options)
	{
		const auto latentInput = module.FindInput("latent");
		if (!latentInput)
		{
			throw std::runtime_error("Euler sampler requires a compiled input named 'latent'");
		}
		auto timestepInput = module.FindInput("timestep");
		if (!timestepInput)
		{
			timestepInput = module.FindInput("timesteps");
		}
		auto noiseOutput = module.FindOutput("noise_pred");
		if (!noiseOutput)
		{
			if (module.OutputSpecs().size() != 1)
			{
				throw std::runtime_error("Euler sampler requires output 'noise_pred' or exactly one output");
			}
			noiseOutput = 0;
		}

		const auto cfgMode = ResolveCFGMode(options);
		if (cfgMode == "dual" && !options.inputBindings)
		{
			throw std::runtime_error("Euler CFG dual mode requires --inputs with negative conditioning tensors");
		}
		auto inputs = options.inputBindings
		                  ? MakeInputsFromSafetensors(module.InputSpecs(), *options.inputBindings, true,
		                                               InputBindingFlavor::Default)
		                  : MakeZeroInputs(module.InputSpecs());
		auto uncondInputs = cfgMode == "dual"
		                        ? MakeInputsFromSafetensors(module.InputSpecs(), *options.inputBindings, true,
		                                                     InputBindingFlavor::Negative)
		                        : std::vector<LiteNN::Tensor<LiteNN::CPU>>{};
		LiteNN::Tensor<LiteNN::CPU> latentState(LiteNN::Uninitialized, inputs[*latentInput].Shape(),
		                                        inputs[*latentInput].DType());
		if (options.latentInit == "inputs")
		{
			if (!options.inputBindings)
			{
				throw std::runtime_error("Euler --latent-init inputs requires --inputs with a latent tensor");
			}
			CopyScaledTensor(inputs[*latentInput], latentState, 1.0, "Euler latent init");
		}
		else if (options.latentInit == "zero")
		{
			latentState = LiteNN::Tensor<LiteNN::CPU>(latentState.Shape(), latentState.DType());
		}
		else
		{
			FillRandomLatent(latentState, options.seed, options.sigmaMax);
		}
		std::cout << std::format(
		    "Euler sampler loaded {} backend={} steps={} seed={} scheduler={} sigma_max={} sigma_min={}"
		    " denoiser={} timestep_mode={} cfg_mode={} cfg_scale={} latent={}\n",
		    sourceLabel, BackendName(module.Backend()), options.steps, options.seed, options.scheduler,
		    options.sigmaMax, options.sigmaMin, options.denoiserContract, options.timestepMode, cfgMode,
		    options.cfgScale, ShapeToString(latentState.Shape()));
		const auto initialLatentStats = ComputeTensorStats(latentState);
		PrintStats("  initial latent", initialLatentStats);
		if (options.failOnNonFinite)
		{
			RequireFinite("initial latent", initialLatentStats);
		}

		for (std::size_t step = 0; step < options.steps; ++step)
		{
			const auto sigma = SigmaAtStep(options, step);
			const auto nextSigma = SigmaAtStep(options, step + 1);
			const auto dt = nextSigma - sigma;
			const auto coefficients = ComputeDenoiserStepCoefficients(options, sigma);
			const auto timestep = TimestepForStep(options, coefficients, sigma);
			CopyScaledTensor(latentState, inputs[*latentInput], coefficients.cIn, "Euler latent input");
			if (cfgMode == "dual")
			{
				CopyScaledTensor(latentState, uncondInputs[*latentInput], coefficients.cIn, "Euler CFG latent input");
			}
			if (timestepInput)
			{
				FillFloatingScalar(inputs[*timestepInput], timestep, "Euler timestep");
				if (cfgMode == "dual")
				{
					FillFloatingScalar(uncondInputs[*timestepInput], timestep, "Euler CFG timestep");
				}
			}
			std::vector<LiteNN::Tensor<LiteNN::CPU>> uncondOutputs;
			if (cfgMode == "dual")
			{
				uncondOutputs = module.Run(std::span<const LiteNN::Tensor<LiteNN::CPU>>(uncondInputs));
			}
			const auto outputs = module.Run(std::span<const LiteNN::Tensor<LiteNN::CPU>>(inputs));
			const auto* uncondPrediction = cfgMode == "dual" ? &uncondOutputs[*noiseOutput] : nullptr;
			const auto noiseStats = ComputeTensorStats(outputs[*noiseOutput]);
			std::optional<TensorStats> uncondNoiseStats;
			if (uncondPrediction != nullptr)
			{
				uncondNoiseStats = ComputeTensorStats(*uncondPrediction);
			}
			if (options.failOnNonFinite)
			{
				RequireFinite(std::format("step {} prediction", step + 1), noiseStats);
				if (uncondNoiseStats)
				{
					RequireFinite(std::format("step {} unconditional prediction", step + 1),
					              *uncondNoiseStats);
				}
			}
			EulerUpdateFromPredictions(latentState, outputs[*noiseOutput], uncondPrediction, coefficients,
			                           options.cfgScale, sigma, dt);
			const auto latentStats = ComputeTensorStats(latentState);
			if (options.failOnNonFinite)
			{
				RequireFinite(std::format("step {} latent", step + 1), latentStats);
			}
			if (uncondNoiseStats)
			{
				std::cout << std::format(
				    "  step {}/{} sigma={} next_sigma={} timestep={} c_in={} dt={} pred_rms={}"
				    " pred_nonfinite={} uncond_pred_rms={} uncond_pred_nonfinite={} latent_rms={}"
				    " latent_nonfinite={}\n",
				    step + 1, options.steps, sigma, nextSigma, timestep, coefficients.cIn, dt,
				    noiseStats.rms, noiseStats.nonFiniteCount, uncondNoiseStats->rms,
				    uncondNoiseStats->nonFiniteCount, latentStats.rms, latentStats.nonFiniteCount);
			}
			else
			{
				std::cout << std::format(
				    "  step {}/{} sigma={} next_sigma={} timestep={} c_in={} dt={} pred_rms={}"
				    " pred_nonfinite={} latent_rms={} latent_nonfinite={}\n",
				    step + 1, options.steps, sigma, nextSigma, timestep, coefficients.cIn, dt,
				    noiseStats.rms, noiseStats.nonFiniteCount, latentStats.rms,
				    latentStats.nonFiniteCount);
			}
		}

		const auto finalLatentStats = ComputeTensorStats(latentState);
		PrintStats("  final latent", finalLatentStats);
		if (options.failOnNonFinite)
		{
			RequireFinite("final latent", finalLatentStats);
		}
		if (options.outputLatent)
		{
			WriteTensorSafetensors(*options.outputLatent, "latent", latentState);
			std::cout << std::format("  wrote final latent {}\n", options.outputLatent->string());
		}
	}

	void SampleEuler(const std::filesystem::path& libraryPath, std::string_view symbolPrefix,
	                 const EulerSamplerOptions& options)
	{
		DynamicLibrary library(libraryPath);
		if (auto separated = TryLoadSeparatedArtifactFromLibrary(library, symbolPrefix))
		{
			std::cout << std::format(
			    "Loaded separated carrier image {} backend={} metadata={} bytes constants={} bytes weights={} bytes"
			    " instructions={} bytes external_tensors={}\n",
			    libraryPath.string(), BackendName(separated->Backend()), separated->Metadata().size(),
			    separated->Constants().size(), separated->Weights().size(), separated->Instructions().size(),
			    separated->ExternalTensorInfos().size())
			          << std::flush;
			auto module = separated->LoadBorrowedExternalRegions();
			SampleEulerModule(module, libraryPath.string(), options);
			return;
		}

		auto artifact = LoadArtifactFromLibrary(library, symbolPrefix);
		std::cout << std::format("Loaded carrier image {} backend={} rodata={} bytes instructions={} bytes\n",
		                         libraryPath.string(), BackendName(artifact.Backend()), artifact.Rodata().size(),
		                         artifact.Instructions().size())
		          << std::flush;
		auto module = artifact.Load();
		SampleEulerModule(module, libraryPath.string(), options);
	}

	void SampleEulerImage(const std::filesystem::path& rodataPath,
	                      const std::filesystem::path& instructionsPath,
	                      const EulerSamplerOptions& options)
	{
		if (auto separated = TryLoadSeparatedImageRegions(rodataPath, instructionsPath))
		{
			auto module = separated->LoadBorrowedExternalRegions();
			std::cout << std::format(
			    "Loaded separated image regions metadata={} instructions={} backend={} input_count={} output_count={}"
			    " constants={} bytes weights={} bytes external_tensors={}\n",
			    rodataPath.string(), instructionsPath.string(), BackendName(module.Backend()),
			    module.InputSpecs().size(), module.OutputSpecs().size(), separated->Constants().size(),
			    separated->Weights().size(), separated->ExternalTensorInfos().size())
			          << std::flush;
			SampleEulerModule(module, rodataPath.string(), options);
			return;
		}

		auto rodata = ReadAllBytes(rodataPath);
		auto instructions = ReadAllBytes(instructionsPath);
		auto module = LiteNN::CompiledModule<LiteNN::CPU>::Load({
		    .rodata = rodata.data(),
		    .rodataSize = rodata.size(),
		    .instructions = instructions.data(),
		    .instructionSize = instructions.size(),
		});
		std::cout << std::format("Loaded image regions rodata={} instructions={} backend={} input_count={} output_count={}\n",
		                         rodataPath.string(), instructionsPath.string(), BackendName(module.Backend()),
		                         module.InputSpecs().size(), module.OutputSpecs().size())
		          << std::flush;
		SampleEulerModule(module, rodataPath.string(), options);
	}
#endif
} // namespace

int main(int argc, char** argv)
{
	try
	{
		if (argc == 2 && (std::string_view(argv[1]) == "--help" || std::string_view(argv[1]) == "-h"))
		{
			PrintUsage(argv[0]);
			return 0;
		}
		if (argc >= 2 && std::string_view(argv[1]) == "--inspect")
		{
			if (argc != 3)
			{
				PrintUsage(argv[0]);
				return 1;
			}
			InspectSafetensors(argv[2]);
			return 0;
		}
		if (argc >= 2 && std::string_view(argv[1]) == "--import")
		{
			if (argc < 5)
			{
				PrintUsage(argv[0]);
				return 1;
			}
			bool allowExtra = false;
			std::optional<std::filesystem::path> externalWeightsPath;
			std::uint64_t externalWeightMinBytes = 0;
			for (int i = 5; i < argc; ++i)
			{
				const std::string_view option(argv[i]);
				if (option == "--allow-extra-tensors")
				{
					allowExtra = true;
				}
				else if (option == "--external-weights")
				{
					if (++i >= argc)
					{
						throw std::runtime_error("--external-weights requires a path");
					}
					externalWeightsPath = std::filesystem::path(argv[i]);
				}
				else if (option == "--external-weight-min-bytes")
				{
					if (++i >= argc)
					{
						throw std::runtime_error("--external-weight-min-bytes requires a byte count");
					}
					externalWeightMinBytes = std::stoull(argv[i]);
				}
				else
				{
					throw std::runtime_error("Unknown --import option: " + std::string(option));
				}
			}
			ImportManifest(argv[2], argv[3], argv[4], allowExtra, externalWeightsPath, externalWeightMinBytes);
			return 0;
		}
		if (argc >= 2 && std::string_view(argv[1]) == "--compile-object")
		{
			if (argc != 4 && argc != 5)
			{
				PrintUsage(argv[0]);
				return 1;
			}
#ifdef LITENN_ENABLE_MLIR
			const std::string_view prefix = argc == 5 ? std::string_view(argv[4]) : "litenn_sdxl_module";
			CompileObject(argv[2], argv[3], prefix);
			return 0;
#else
			throw std::runtime_error("AOT compiler support is not enabled; configure with LITENN_ENABLE_MLIR=ON");
#endif
		}
		if (argc >= 2 && std::string_view(argv[1]) == "--compile-budget")
		{
			if (argc != 3)
			{
				PrintUsage(argv[0]);
				return 1;
			}
#ifdef LITENN_ENABLE_MLIR
			PrintModelCompileBudget(argv[2]);
			return 0;
#else
			throw std::runtime_error("AOT compiler support is not enabled; configure with LITENN_ENABLE_MLIR=ON");
#endif
		}
		if (argc >= 2 && std::string_view(argv[1]) == "--run-model")
		{
			if (argc != 3)
			{
				PrintUsage(argv[0]);
				return 1;
			}
#ifdef LITENN_ENABLE_MLIR
			CompileAndRunModel(argv[2]);
			return 0;
#else
			throw std::runtime_error("AOT compiler support is not enabled; configure with LITENN_ENABLE_MLIR=ON");
#endif
		}
		if (argc >= 2 && std::string_view(argv[1]) == "--run-model-with-inputs")
		{
			if (argc < 4)
			{
				PrintUsage(argv[0]);
				return 1;
			}
#ifdef LITENN_ENABLE_MLIR
			CompileAndRunModelWithInputs(argv[2], argv[3], ParseOutputRunOptions(argc, argv, 4));
			return 0;
#else
			throw std::runtime_error("AOT compiler support is not enabled; configure with LITENN_ENABLE_MLIR=ON");
#endif
		}
		if (argc >= 2 && std::string_view(argv[1]) == "--diagnose-model-with-inputs")
		{
			if (argc < 4)
			{
				PrintUsage(argv[0]);
				return 1;
			}
#ifdef LITENN_ENABLE_MLIR
			DiagnoseModelWithInputs(argv[2], argv[3], ParseFiniteDiagnosticOptions(argc, argv, 4));
			return 0;
#else
			throw std::runtime_error("Finite diagnostics require LITENN_ENABLE_MLIR=ON in the SDXL example build");
#endif
		}
		if (argc >= 2 && std::string_view(argv[1]) == "--benchmark-model-with-inputs")
		{
			if (argc < 4)
			{
				PrintUsage(argv[0]);
				return 1;
			}
#ifdef LITENN_ENABLE_MLIR
			BenchmarkModelWithInputs(argv[2], argv[3], ParseModelBenchmarkOptions(argc, argv, 4));
			return 0;
#else
			throw std::runtime_error("AOT compiler support is not enabled; configure with LITENN_ENABLE_MLIR=ON");
#endif
		}
		if (argc >= 2 && std::string_view(argv[1]) == "--compile-raw-object")
		{
			if (argc != 4)
			{
				PrintUsage(argv[0]);
				return 1;
			}
#ifdef LITENN_ENABLE_MLIR
			CompileRawObject(argv[2], argv[3]);
			return 0;
#else
			throw std::runtime_error("AOT compiler support is not enabled; configure with LITENN_ENABLE_MLIR=ON");
#endif
		}
		if (argc >= 2 && std::string_view(argv[1]) == "--compile-image-regions")
		{
			if (argc < 4)
			{
				PrintUsage(argv[0]);
				return 1;
			}
#ifdef LITENN_ENABLE_MLIR
			std::string prefix = "litenn_sdxl_module";
			bool prefixSet = false;
			ExampleCompilerSettings compilerSettings;
			for (int i = 4; i < argc; ++i)
			{
				const std::string_view option(argv[i]);
				if (option == "--cpu-aot-llvm-opt-level")
				{
					if (++i >= argc)
					{
						throw std::runtime_error("--cpu-aot-llvm-opt-level requires a value");
					}
					compilerSettings.cpuAOTLLVMOptLevel = ParseOptLevel(argv[i], "--cpu-aot-llvm-opt-level");
				}
				else if (option.starts_with("--"))
				{
					throw std::runtime_error("Unknown --compile-image-regions option: " + std::string(option));
				}
				else if (!prefixSet)
				{
					prefix = std::string(option);
					prefixSet = true;
				}
				else
				{
					PrintUsage(argv[0]);
					return 1;
				}
			}
			CompileImageRegions(argv[2], argv[3], prefix, compilerSettings);
			return 0;
#else
			throw std::runtime_error("AOT compiler support is not enabled; configure with LITENN_ENABLE_MLIR=ON");
#endif
		}
		if (argc >= 2 && std::string_view(argv[1]) == "--run-image-with-inputs")
		{
			if (argc < 5)
			{
				PrintUsage(argv[0]);
				return 1;
			}
#ifdef LITENN_ENABLE_MLIR
			LoadImageAndRunWithInputs(argv[2], argv[3], argv[4], ParseOutputRunOptions(argc, argv, 5));
			return 0;
#else
			throw std::runtime_error("AOT compiler support is not enabled; configure with LITENN_ENABLE_MLIR=ON");
#endif
		}
		if (argc >= 2 && std::string_view(argv[1]) == "--load-dll")
		{
			if (argc != 3 && argc != 4)
			{
				PrintUsage(argv[0]);
				return 1;
			}
#ifdef LITENN_ENABLE_MLIR
			const std::string_view prefix = argc == 4 ? std::string_view(argv[3]) : "litenn_sdxl_module";
			LoadDllAndRun(argv[2], prefix);
			return 0;
#else
			throw std::runtime_error("AOT compiler support is not enabled; configure with LITENN_ENABLE_MLIR=ON");
#endif
		}
		if (argc >= 2 && std::string_view(argv[1]) == "--load-dll-with-inputs")
		{
			if (argc < 4)
			{
				PrintUsage(argv[0]);
				return 1;
			}
#ifdef LITENN_ENABLE_MLIR
			std::string_view prefix = "litenn_sdxl_module";
			int optionStart = 4;
			if (argc >= 5 && !std::string_view(argv[4]).starts_with("--"))
			{
				prefix = argv[4];
				optionStart = 5;
			}
			LoadDllAndRunWithInputs(argv[2], argv[3], prefix, ParseOutputRunOptions(argc, argv, optionStart));
			return 0;
#else
			throw std::runtime_error("AOT compiler support is not enabled; configure with LITENN_ENABLE_MLIR=ON");
#endif
		}
		if (argc >= 2 && std::string_view(argv[1]) == "--sample-euler")
		{
			if (argc < 3)
			{
				PrintUsage(argv[0]);
				return 1;
			}
#ifdef LITENN_ENABLE_MLIR
			std::string_view prefix = "litenn_sdxl_module";
			int optionStart = 3;
			if (argc >= 4 && !std::string_view(argv[3]).starts_with("--"))
			{
				prefix = argv[3];
				optionStart = 4;
			}
			SampleEuler(argv[2], prefix, ParseEulerOptions(argc, argv, optionStart));
			return 0;
#else
			throw std::runtime_error("AOT compiler support is not enabled; configure with LITENN_ENABLE_MLIR=ON");
#endif
		}
		if (argc >= 2 && std::string_view(argv[1]) == "--denoise-latent")
		{
			if (argc < 5)
			{
				PrintUsage(argv[0]);
				return 1;
			}
#ifdef LITENN_ENABLE_MLIR
			std::string_view prefix = "litenn_sdxl_module";
			int optionStart = 5;
			if (argc >= 6 && !std::string_view(argv[5]).starts_with("--"))
			{
				prefix = argv[5];
				optionStart = 6;
			}
			auto options = ParseEulerOptions(argc, argv, optionStart);
			if (options.inputBindings || options.outputLatent)
			{
				throw std::runtime_error(
				    "--denoise-latent takes input/output paths positionally; do not also pass --inputs or --output-latent");
			}
			options.inputBindings = std::filesystem::path(argv[3]);
			options.outputLatent = std::filesystem::path(argv[4]);
			SampleEuler(argv[2], prefix, options);
			return 0;
#else
			throw std::runtime_error("AOT compiler support is not enabled; configure with LITENN_ENABLE_MLIR=ON");
#endif
		}
		if (argc >= 2 && std::string_view(argv[1]) == "--denoise-latent-image")
		{
			if (argc < 6)
			{
				PrintUsage(argv[0]);
				return 1;
			}
#ifdef LITENN_ENABLE_MLIR
			auto options = ParseEulerOptions(argc, argv, 6);
			if (options.inputBindings || options.outputLatent)
			{
				throw std::runtime_error(
				    "--denoise-latent-image takes input/output paths positionally; do not also pass --inputs or --output-latent");
			}
			options.inputBindings = std::filesystem::path(argv[4]);
			options.outputLatent = std::filesystem::path(argv[5]);
			SampleEulerImage(argv[2], argv[3], options);
			return 0;
#else
			throw std::runtime_error("AOT compiler support is not enabled; configure with LITENN_ENABLE_MLIR=ON");
#endif
		}

		PrintUsage(argv[0]);
		return 1;
	}
	catch (const std::exception& ex)
	{
		std::cerr << "litenn_sdxl_example: " << ex.what() << '\n';
		return 1;
	}
}
