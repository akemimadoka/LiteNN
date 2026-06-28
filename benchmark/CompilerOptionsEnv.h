#ifndef LITENN_BENCHMARK_COMPILER_OPTIONS_ENV_H
#define LITENN_BENCHMARK_COMPILER_OPTIONS_ENV_H

#include <LiteNN/Compiler/CompiledModule.h>

#include <algorithm>
#include <cctype>
#include <charconv>
#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <optional>
#include <string>
#include <string_view>
#include <system_error>

inline bool LiteNNBenchTruthyEnvValue(const char* value)
{
	if (value == nullptr)
	{
		return false;
	}
	const std::string_view text{ value };
	return text == "1" || text == "true" || text == "TRUE" || text == "on" || text == "ON";
}

inline std::optional<std::uint64_t> LiteNNBenchParseU64Env(const char* name)
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

inline LiteNN::CompilerOptions LiteNNBenchCompilerOptionsFromEnvironment()
{
	auto options = LiteNN::CompilerOptions::Defaults();
	if (const auto threadCount = LiteNNBenchParseU64Env("LITENN_CPU_AOT_THREADS"); threadCount && *threadCount > 0)
	{
		options.cpuAOTThreadCount = static_cast<std::size_t>(*threadCount);
	}
	if (const auto minFlops = LiteNNBenchParseU64Env("LITENN_CPU_AOT_PARALLEL_MIN_FLOPS"))
	{
		options.cpuAOTParallelMinFlops = *minFlops;
	}
	if (const auto minConstantBytes = LiteNNBenchParseU64Env("LITENN_CPU_AOT_EXTERNAL_CONSTANT_MIN_BYTES"))
	{
		options.cpuAOTExternalConstantMinBytes = *minConstantBytes;
	}
	if (const auto optLevel = LiteNNBenchParseU64Env("LITENN_CPU_AOT_LLVM_OPT_LEVEL"))
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
	options.enableCPUAOTExternalRegions = LiteNNBenchTruthyEnvValue(std::getenv("LITENN_CPU_AOT_EXTERNAL_REGIONS")) ||
	                                      LiteNNBenchTruthyEnvValue(std::getenv("LITENN_CPU_AOT_EXTERNAL_CONSTANTS"));
	if (const char* value = std::getenv("LITENN_CPU_AOT_EXTERNAL_REGION_FUSION"))
	{
		options.enableCPUAOTExternalRegionFusion = LiteNNBenchTruthyEnvValue(value);
	}
	if (LiteNNBenchTruthyEnvValue(std::getenv("LITENN_CUDA_DISABLE_NATIVE_AOT")))
	{
		options.enableCUDANativeAOT = false;
	}
	if (LiteNNBenchTruthyEnvValue(std::getenv("LITENN_COMPILE_DIAGNOSTICS")))
	{
		options.enableCompileDiagnostics = true;
	}
	return options;
}

#endif
