#pragma once

#include <LiteNN/Compiler/CompiledModule.h>

#include <cstdint>
#include <format>
#include <span>
#include <string>
#include <string_view>

namespace LiteNN::GGUF::Tooling
{
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
		std::string identity = std::format("bytes={}", weightBytes);
		for (const auto& tensor : externalTensors)
		{
			if (tensor.region != "weights")
			{
				continue;
			}
			identity += std::format("|{}:{}:{}:{}:{}", tensor.name, tensor.byteOffset, tensor.byteSize,
			                        tensor.alignment, tensor.checksum);
		}
		return std::format("{:016x}", FNV1a(identity));
	}
} // namespace LiteNN::GGUF::Tooling
