#pragma once

#include <LiteNN/Compiler/CompiledModule.h>

#include <algorithm>
#include <cstdint>
#include <filesystem>
#include <format>
#include <functional>
#include <ranges>
#include <span>
#include <string>
#include <string_view>
#include <tuple>
#include <vector>

namespace LiteNN::GGUF::Tooling
{
	enum class SharedWeightsPublishResult
	{
		Published,
		Reused,
	};

	using SharedWeightsStagingWriter =
	    std::function<void(const std::filesystem::path& weightsPath, const std::filesystem::path& completePath)>;

	inline std::uint64_t FNV1a(std::string_view text)
	{
		std::uint64_t hash = 14695981039346656037ull;
		for (const unsigned char ch : text)
		{
			hash ^= ch;
			hash *= 1099511628211ull;
		}
		return hash;
	}

	inline std::string DecodeAOTSharedWeightsIdentity(std::size_t weightBytes,
	                                                  std::span<const CompiledModuleExternalTensorInfo> externalTensors)
	{
		std::vector<const CompiledModuleExternalTensorInfo*> weightTensors;
		weightTensors.reserve(externalTensors.size());
		for (const auto& tensor : externalTensors)
		{
			if (tensor.region == "weights")
			{
				weightTensors.push_back(&tensor);
			}
		}
		std::ranges::sort(weightTensors, [](const auto* lhs, const auto* rhs) {
			return std::tuple(lhs->byteOffset, lhs->byteSize, lhs->checksum) <
			       std::tuple(rhs->byteOffset, rhs->byteSize, rhs->checksum);
		});

		std::string identity = std::format("bytes={}", weightBytes);
		for (const auto* tensor : weightTensors)
		{
			identity += std::format("|{}:{}:{}", tensor->byteOffset, tensor->byteSize, tensor->checksum);
		}
		return std::format("{:016x}", FNV1a(identity));
	}

	SharedWeightsPublishResult PublishDecodeAOTSharedWeightsAtomically(const std::filesystem::path& weightsPath,
	                                                                   std::size_t expectedBytes,
	                                                                   const SharedWeightsStagingWriter& stagingWriter);
} // namespace LiteNN::GGUF::Tooling
