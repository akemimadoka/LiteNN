#include "VulkanNativePayload.h"

#include <algorithm>
#include <array>
#include <limits>
#include <stdexcept>
#include <string_view>

namespace LiteNN
{
	namespace
	{
		constexpr std::array<std::byte, 8> kPayloadMagic = {
			std::byte{ 'L' }, std::byte{ 'T' }, std::byte{ 'N' }, std::byte{ 'N' },
			std::byte{ 'V' }, std::byte{ 'K' }, std::byte{ 'S' }, std::byte{ 'P' },
		};
		constexpr std::uint32_t kPayloadVersion = 3;

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

		std::uint32_t ReadU32(std::span<const std::byte> bytes, std::size_t& offset)
		{
			if (offset + 4 > bytes.size())
			{
				throw std::runtime_error("Vulkan native instruction payload is truncated");
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
				throw std::runtime_error("Vulkan native instruction payload is truncated");
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
			if (size > std::numeric_limits<std::size_t>::max() ||
			    static_cast<std::size_t>(size) > bytes.size() - offset)
			{
				throw std::runtime_error("Vulkan native instruction payload string is truncated");
			}
			std::string result(reinterpret_cast<const char*>(bytes.data() + offset), static_cast<std::size_t>(size));
			offset += static_cast<std::size_t>(size);
			return result;
		}

		VulkanNativeArgumentKind DecodeArgumentKind(std::uint32_t value)
		{
			switch (value)
			{
			case static_cast<std::uint32_t>(VulkanNativeArgumentKind::InputTensor):
				return VulkanNativeArgumentKind::InputTensor;
			case static_cast<std::uint32_t>(VulkanNativeArgumentKind::OutputTensor):
				return VulkanNativeArgumentKind::OutputTensor;
			case static_cast<std::uint32_t>(VulkanNativeArgumentKind::ExternalTensor):
				return VulkanNativeArgumentKind::ExternalTensor;
			case static_cast<std::uint32_t>(VulkanNativeArgumentKind::WorkspaceTensor):
				return VulkanNativeArgumentKind::WorkspaceTensor;
			default:
				throw std::runtime_error("Vulkan native instruction payload contains an invalid argument kind");
			}
		}

		void ValidateDim(VulkanNativeDispatchDim dim)
		{
			if (dim.x == 0 || dim.y == 0 || dim.z == 0)
			{
				throw std::runtime_error("Vulkan native instruction payload has an invalid dispatch dimension");
			}
		}

		bool IsPowerOfTwo(std::uint32_t value)
		{
			return value != 0 && (value & (value - 1)) == 0;
		}

		bool IsPowerOfTwo(std::uint64_t value)
		{
			return value != 0 && (value & (value - 1)) == 0;
		}

		void ValidatePayload(const VulkanNativeInstructionPayload& payload)
		{
			if (!payload.featureSet.CheckIsValid())
			{
				throw std::runtime_error("Vulkan native instruction payload contains unknown feature flags");
			}
			if (payload.target.empty())
			{
				throw std::runtime_error("Vulkan native instruction payload target must not be empty");
			}
			if (payload.spirv.empty())
			{
				throw std::runtime_error("Vulkan native instruction payload SPIR-V must not be empty");
			}
			if (payload.kernels.empty())
			{
				throw std::runtime_error("Vulkan native instruction payload must contain at least one kernel");
			}
			for (const auto& workspace : payload.workspaceTensors)
			{
				if (workspace.byteSize == 0)
				{
					throw std::runtime_error("Vulkan native workspace tensor byte size must not be zero");
				}
				if (!IsPowerOfTwo(workspace.alignment))
				{
					throw std::runtime_error("Vulkan native workspace tensor alignment must be a power of two");
				}
			}
			for (const auto& kernel : payload.kernels)
			{
				if (kernel.entryPoint.empty())
				{
					throw std::runtime_error("Vulkan native kernel entry point must not be empty");
				}
				ValidateDim(kernel.groups);
				if (kernel.requirements.descriptorAbiVersion == 0)
				{
					throw std::runtime_error("Vulkan native kernel descriptor ABI version must not be zero");
				}
				ValidateDim(kernel.requirements.localSize);
				if (!kernel.requirements.deviceRequirements.CheckIsValid())
				{
					throw std::runtime_error("Vulkan native kernel contains unknown device requirement flags");
				}
				if (kernel.requirements.requiredStorageBufferOffsetAlignment != 0 &&
				    !IsPowerOfTwo(kernel.requirements.requiredStorageBufferOffsetAlignment))
				{
					throw std::runtime_error(
					    "Vulkan native kernel storage-buffer offset alignment requirement must be a power of two");
				}
				for (const auto& argument : kernel.arguments)
				{
					if (argument.byteSize == 0)
					{
						throw std::runtime_error("Vulkan native argument byte size must not be zero");
					}
					if (argument.kind == VulkanNativeArgumentKind::WorkspaceTensor &&
					    argument.index >= payload.workspaceTensors.size())
					{
						throw std::runtime_error("Vulkan native workspace argument index is out of bounds");
					}
				}
			}
		}

		void AppendDim(std::vector<std::byte>& bytes, VulkanNativeDispatchDim dim)
		{
			AppendU32(bytes, dim.x);
			AppendU32(bytes, dim.y);
			AppendU32(bytes, dim.z);
		}

		VulkanNativeDispatchDim ReadDim(std::span<const std::byte> bytes, std::size_t& offset)
		{
			return {
				.x = ReadU32(bytes, offset),
				.y = ReadU32(bytes, offset),
				.z = ReadU32(bytes, offset),
			};
		}
	} // namespace

	std::vector<std::byte> SerializeVulkanNativeInstructionPayload(const VulkanNativeInstructionPayload& payload)
	{
		ValidatePayload(payload);

		std::vector<std::byte> bytes;
		bytes.insert(bytes.end(), kPayloadMagic.begin(), kPayloadMagic.end());
		AppendU32(bytes, kPayloadVersion);
		AppendU64(bytes, payload.featureSet.flags);
		AppendString(bytes, payload.target);
		AppendU64(bytes, static_cast<std::uint64_t>(payload.spirv.size()));
		for (const auto word : payload.spirv)
		{
			AppendU32(bytes, word);
		}
		AppendU32(bytes, static_cast<std::uint32_t>(payload.workspaceTensors.size()));
		for (const auto& workspace : payload.workspaceTensors)
		{
			AppendU64(bytes, workspace.byteSize);
			AppendU64(bytes, workspace.alignment);
		}
		AppendU32(bytes, static_cast<std::uint32_t>(payload.kernels.size()));
		for (const auto& kernel : payload.kernels)
		{
			AppendString(bytes, kernel.entryPoint);
			AppendDim(bytes, kernel.groups);
			AppendU32(bytes, kernel.requirements.descriptorAbiVersion);
			AppendDim(bytes, kernel.requirements.localSize);
			AppendU64(bytes, kernel.requirements.deviceRequirements.flags);
			AppendU32(bytes, kernel.requirements.requiredSubgroupSize);
			AppendU32(bytes, kernel.requirements.requiredStorageBufferOffsetAlignment);
			AppendU32(bytes, static_cast<std::uint32_t>(kernel.arguments.size()));
			for (const auto& argument : kernel.arguments)
			{
				AppendU32(bytes, static_cast<std::uint32_t>(argument.kind));
				AppendU32(bytes, argument.index);
				AppendU32(bytes, argument.binding);
				AppendU64(bytes, argument.byteOffset);
				AppendU64(bytes, argument.byteSize);
			}
		}
		return bytes;
	}

	VulkanNativeInstructionPayload DeserializeVulkanNativeInstructionPayload(std::span<const std::byte> bytes)
	{
		if (bytes.size() < kPayloadMagic.size() ||
		    !std::equal(kPayloadMagic.begin(), kPayloadMagic.end(), bytes.begin()))
		{
			throw std::runtime_error("Vulkan native instruction payload has an invalid magic header");
		}

		std::size_t offset = kPayloadMagic.size();
		const auto version = ReadU32(bytes, offset);
		if (version == 0 || version > kPayloadVersion)
		{
			throw std::runtime_error("Unsupported Vulkan native instruction payload version");
		}

		VulkanNativeInstructionPayload payload;
		payload.featureSet.flags = ReadU64(bytes, offset);
		payload.target = ReadString(bytes, offset);
		const auto wordCount = ReadU64(bytes, offset);
		if (wordCount > std::numeric_limits<std::size_t>::max())
		{
			throw std::runtime_error("Vulkan native instruction payload SPIR-V is too large");
		}
		payload.spirv.reserve(static_cast<std::size_t>(wordCount));
		for (std::uint64_t i = 0; i < wordCount; ++i)
		{
			payload.spirv.push_back(ReadU32(bytes, offset));
		}

		if (version >= 3)
		{
			const auto workspaceCount = ReadU32(bytes, offset);
			payload.workspaceTensors.reserve(workspaceCount);
			for (std::uint32_t i = 0; i < workspaceCount; ++i)
			{
				payload.workspaceTensors.push_back({
				    .byteSize = ReadU64(bytes, offset),
				    .alignment = ReadU64(bytes, offset),
				});
			}
		}

		const auto kernelCount = ReadU32(bytes, offset);
		payload.kernels.reserve(kernelCount);
		for (std::uint32_t i = 0; i < kernelCount; ++i)
		{
			VulkanNativeKernelSpec kernel;
			kernel.entryPoint = ReadString(bytes, offset);
			kernel.groups = ReadDim(bytes, offset);
			if (version >= 2)
			{
				kernel.requirements.descriptorAbiVersion = ReadU32(bytes, offset);
				kernel.requirements.localSize = ReadDim(bytes, offset);
				kernel.requirements.deviceRequirements.flags = ReadU64(bytes, offset);
				kernel.requirements.requiredSubgroupSize = ReadU32(bytes, offset);
				kernel.requirements.requiredStorageBufferOffsetAlignment = ReadU32(bytes, offset);
			}
			const auto argumentCount = ReadU32(bytes, offset);
			kernel.arguments.reserve(argumentCount);
			for (std::uint32_t arg = 0; arg < argumentCount; ++arg)
			{
				kernel.arguments.push_back({
				    .kind = DecodeArgumentKind(ReadU32(bytes, offset)),
				    .index = ReadU32(bytes, offset),
				    .binding = ReadU32(bytes, offset),
				    .byteOffset = ReadU64(bytes, offset),
				    .byteSize = ReadU64(bytes, offset),
				});
			}
			payload.kernels.push_back(std::move(kernel));
		}

		if (offset != bytes.size())
		{
			throw std::runtime_error("Vulkan native instruction payload contains trailing bytes");
		}
		ValidatePayload(payload);
		return payload;
	}
} // namespace LiteNN
