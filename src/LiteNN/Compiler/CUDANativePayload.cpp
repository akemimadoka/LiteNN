#include "CUDANativePayload.h"

#include <algorithm>
#include <array>
#include <limits>
#include <stdexcept>
#include <string_view>
#include <utility>

namespace LiteNN
{
namespace
{
	constexpr std::array<std::byte, 8> kPayloadMagic = {
		std::byte{ 'L' }, std::byte{ 'T' }, std::byte{ 'N' }, std::byte{ 'N' },
		std::byte{ 'C' }, std::byte{ 'U' }, std::byte{ 'D' }, std::byte{ 'A' },
	};
	constexpr std::uint32_t kPayloadVersion = 2;

	void AppendU32(std::vector<std::byte>& bytes, std::uint32_t value)
	{
		for (int i = 0; i < 4; ++i)
		{
			bytes.push_back(static_cast<std::byte>((value >> (i * 8)) & 0xffu));
		}
	}

	void AppendU64(std::vector<std::byte>& bytes, std::uint64_t value)
	{
		for (int i = 0; i < 8; ++i)
		{
			bytes.push_back(static_cast<std::byte>((value >> (i * 8)) & 0xffu));
		}
	}

	void AppendString(std::vector<std::byte>& bytes, std::string_view value)
	{
		AppendU64(bytes, static_cast<std::uint64_t>(value.size()));
		bytes.insert(bytes.end(), reinterpret_cast<const std::byte*>(value.data()),
		             reinterpret_cast<const std::byte*>(value.data() + value.size()));
	}

	void AppendBytes(std::vector<std::byte>& bytes, std::span<const std::byte> value)
	{
		AppendU64(bytes, static_cast<std::uint64_t>(value.size()));
		bytes.insert(bytes.end(), value.begin(), value.end());
	}

	std::uint32_t ReadU32(std::span<const std::byte> bytes, std::size_t& offset)
	{
		if (offset + 4 > bytes.size())
		{
			throw std::runtime_error("CUDA native instruction payload is truncated");
		}
		std::uint32_t value = 0;
		for (int i = 0; i < 4; ++i)
		{
			value |= std::to_integer<std::uint32_t>(bytes[offset + i]) << (i * 8);
		}
		offset += 4;
		return value;
	}

	std::uint64_t ReadU64(std::span<const std::byte> bytes, std::size_t& offset)
	{
		if (offset + 8 > bytes.size())
		{
			throw std::runtime_error("CUDA native instruction payload is truncated");
		}
		std::uint64_t value = 0;
		for (int i = 0; i < 8; ++i)
		{
			value |= std::to_integer<std::uint64_t>(bytes[offset + i]) << (i * 8);
		}
		offset += 8;
		return value;
	}

	std::string ReadString(std::span<const std::byte> bytes, std::size_t& offset)
	{
		const auto size = ReadU64(bytes, offset);
		if (size > std::numeric_limits<std::size_t>::max() || static_cast<std::size_t>(size) > bytes.size() - offset)
		{
			throw std::runtime_error("CUDA native instruction payload string is truncated");
		}
		std::string result(reinterpret_cast<const char*>(bytes.data() + offset), static_cast<std::size_t>(size));
		offset += static_cast<std::size_t>(size);
		return result;
	}

	std::vector<std::byte> ReadBytes(std::span<const std::byte> bytes, std::size_t& offset)
	{
		const auto size = ReadU64(bytes, offset);
		if (size > std::numeric_limits<std::size_t>::max() || static_cast<std::size_t>(size) > bytes.size() - offset)
		{
			throw std::runtime_error("CUDA native instruction payload binary is truncated");
		}
		std::vector<std::byte> result(bytes.begin() + static_cast<std::ptrdiff_t>(offset),
		                              bytes.begin() + static_cast<std::ptrdiff_t>(offset + size));
		offset += static_cast<std::size_t>(size);
		return result;
	}

	CUDANativeBinaryKind DecodeBinaryKind(std::uint32_t value)
	{
		switch (value)
		{
			case static_cast<std::uint32_t>(CUDANativeBinaryKind::PTX):
				return CUDANativeBinaryKind::PTX;
			case static_cast<std::uint32_t>(CUDANativeBinaryKind::Cubin):
				return CUDANativeBinaryKind::Cubin;
			case static_cast<std::uint32_t>(CUDANativeBinaryKind::Fatbin):
				return CUDANativeBinaryKind::Fatbin;
			case static_cast<std::uint32_t>(CUDANativeBinaryKind::LibraryCall):
				return CUDANativeBinaryKind::LibraryCall;
			default:
				throw std::runtime_error("CUDA native instruction payload contains an invalid binary kind");
		}
	}

	CUDANativeArgumentKind DecodeArgumentKind(std::uint32_t value)
	{
		switch (value)
		{
			case static_cast<std::uint32_t>(CUDANativeArgumentKind::InputTensor):
				return CUDANativeArgumentKind::InputTensor;
			case static_cast<std::uint32_t>(CUDANativeArgumentKind::OutputTensor):
				return CUDANativeArgumentKind::OutputTensor;
			case static_cast<std::uint32_t>(CUDANativeArgumentKind::Workspace):
				return CUDANativeArgumentKind::Workspace;
			case static_cast<std::uint32_t>(CUDANativeArgumentKind::Scalar):
				return CUDANativeArgumentKind::Scalar;
			default:
				throw std::runtime_error("CUDA native instruction payload contains an invalid argument kind");
		}
	}

	void ValidateDim(CUDANativeLaunchDim dim, std::string_view label)
	{
		if (dim.x == 0 || dim.y == 0 || dim.z == 0)
		{
			throw std::runtime_error(std::string("CUDA native instruction payload has an invalid ") +
			                         std::string(label) + " dimension");
		}
	}

	void ValidatePayload(const CUDANativeInstructionPayload& payload)
	{
		if ((payload.featureFlags & ~kCUDANativeKnownFeatureMask) != 0)
		{
			throw std::runtime_error("CUDA native instruction payload contains unknown feature flags");
		}
		if (payload.target.empty())
		{
			throw std::runtime_error("CUDA native instruction payload target must not be empty");
		}
		if (payload.binaryKind != CUDANativeBinaryKind::LibraryCall && payload.binary.empty())
		{
			throw std::runtime_error("CUDA native instruction payload binary must not be empty");
		}
		for (const auto& kernel : payload.kernels)
		{
			if (kernel.name.empty())
			{
				throw std::runtime_error("CUDA native instruction payload kernel name must not be empty");
			}
			ValidateDim(kernel.grid, "grid");
			ValidateDim(kernel.block, "block");
		}
	}

	void AppendDim(std::vector<std::byte>& bytes, CUDANativeLaunchDim dim)
	{
		AppendU32(bytes, dim.x);
		AppendU32(bytes, dim.y);
		AppendU32(bytes, dim.z);
	}

	CUDANativeLaunchDim ReadDim(std::span<const std::byte> bytes, std::size_t& offset)
	{
		return {
			.x = ReadU32(bytes, offset),
			.y = ReadU32(bytes, offset),
			.z = ReadU32(bytes, offset),
		};
	}
} // namespace

std::vector<std::byte> SerializeCUDANativeInstructionPayload(const CUDANativeInstructionPayload& payload)
{
	ValidatePayload(payload);

	std::vector<std::byte> bytes;
	bytes.insert(bytes.end(), kPayloadMagic.begin(), kPayloadMagic.end());
	AppendU32(bytes, kPayloadVersion);
	AppendU32(bytes, static_cast<std::uint32_t>(payload.binaryKind));
	AppendU64(bytes, payload.featureFlags);
	AppendString(bytes, payload.target);
	AppendBytes(bytes, payload.binary);
	AppendBytes(bytes, payload.scalarData);
	AppendU64(bytes, payload.workspaceBytes);
	AppendU32(bytes, static_cast<std::uint32_t>(payload.kernels.size()));

	for (const auto& kernel : payload.kernels)
	{
		AppendString(bytes, kernel.name);
		AppendDim(bytes, kernel.grid);
		AppendDim(bytes, kernel.block);
		AppendU32(bytes, kernel.sharedMemoryBytes);
		AppendU64(bytes, kernel.workspaceBytes);
		AppendU32(bytes, static_cast<std::uint32_t>(kernel.arguments.size()));
		for (const auto& argument : kernel.arguments)
		{
			AppendU32(bytes, static_cast<std::uint32_t>(argument.kind));
			AppendU32(bytes, argument.index);
			AppendU64(bytes, argument.byteOffset);
			AppendU64(bytes, argument.byteSize);
		}
	}

	return bytes;
}

CUDANativeInstructionPayload DeserializeCUDANativeInstructionPayload(std::span<const std::byte> bytes)
{
	if (bytes.size() < kPayloadMagic.size() ||
	    !std::equal(kPayloadMagic.begin(), kPayloadMagic.end(), bytes.begin()))
	{
		throw std::runtime_error("CUDA native instruction payload has an invalid magic header");
	}

	std::size_t offset = kPayloadMagic.size();
	const auto version = ReadU32(bytes, offset);
	if (version == 0 || version > kPayloadVersion)
	{
		throw std::runtime_error("Unsupported CUDA native instruction payload version");
	}

	CUDANativeInstructionPayload payload;
	payload.binaryKind = DecodeBinaryKind(ReadU32(bytes, offset));
	if (version >= 2)
	{
		payload.featureFlags = ReadU64(bytes, offset);
	}
	payload.target = ReadString(bytes, offset);
	payload.binary = ReadBytes(bytes, offset);
	if (version >= 2)
	{
		payload.scalarData = ReadBytes(bytes, offset);
	}
	payload.workspaceBytes = ReadU64(bytes, offset);

	const auto kernelCount = ReadU32(bytes, offset);
	payload.kernels.reserve(kernelCount);
	for (std::uint32_t i = 0; i < kernelCount; ++i)
	{
		CUDANativeKernelSpec kernel;
		kernel.name = ReadString(bytes, offset);
		kernel.grid = ReadDim(bytes, offset);
		kernel.block = ReadDim(bytes, offset);
		kernel.sharedMemoryBytes = ReadU32(bytes, offset);
		kernel.workspaceBytes = ReadU64(bytes, offset);

		const auto argumentCount = ReadU32(bytes, offset);
		kernel.arguments.reserve(argumentCount);
		for (std::uint32_t arg = 0; arg < argumentCount; ++arg)
		{
			kernel.arguments.push_back({
			    .kind = DecodeArgumentKind(ReadU32(bytes, offset)),
			    .index = ReadU32(bytes, offset),
			    .byteOffset = ReadU64(bytes, offset),
			    .byteSize = ReadU64(bytes, offset),
			});
		}
		payload.kernels.push_back(std::move(kernel));
	}

	if (offset != bytes.size())
	{
		throw std::runtime_error("CUDA native instruction payload contains trailing bytes");
	}

	ValidatePayload(payload);
	return payload;
}
} // namespace LiteNN
