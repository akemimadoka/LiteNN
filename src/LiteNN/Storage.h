#ifndef LITENN_STORAGE_H
#define LITENN_STORAGE_H

#include <LiteNN/Quantization.h>
#include <LiteNN/TensorType.h>
#include <cstddef>
#include <cstdint>
#include <format>
#include <memory>
#include <optional>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

namespace LiteNN
{
	enum class BufferOwnership
	{
		Owned,
		Borrowed,
		External,
		Mapped,
		DeviceOwned
	};

	enum class ExternalBufferKind
	{
		None,
		Rodata,
		Safetensors,
		GGUF,
		User,
		ObjectFile
	};

	enum class BufferMutability
	{
		Immutable,
		Mutable
	};

	enum class BufferRebindPolicy
	{
		ExactMetadataAndChecksum,
		CompatibleMetadata,
		AnyCompatibleBuffer
	};

	struct BufferRegion
	{
		BufferOwnership ownership{ BufferOwnership::Owned };
		ExternalBufferKind externalKind{ ExternalBufferKind::None };
		TensorMemorySpace memorySpace{ TensorMemorySpace::Host };
		std::string name;
		const void* data{};
		std::size_t byteOffset{};
		std::size_t byteSize{};
		std::size_t alignment{ 1 };
		std::uint64_t checksum{};
		BufferMutability mutability{ BufferMutability::Immutable };
		BufferRebindPolicy rebindPolicy{ BufferRebindPolicy::ExactMetadataAndChecksum };
		std::shared_ptr<const void> owner;

		bool IsExternal() const noexcept
		{
			return externalKind != ExternalBufferKind::None || ownership == BufferOwnership::External ||
			       ownership == BufferOwnership::Mapped;
		}
	};

	struct TensorView
	{
		TensorType type;
		std::optional<QuantizationParams> quantization;
		std::size_t storageOffsetBytes{};
		std::vector<std::size_t> strides;
		std::string layoutTag;
		std::size_t aliasSet{};
		BufferMutability mutability{ BufferMutability::Immutable };

		bool HasExplicitStrides() const noexcept
		{
			return !strides.empty();
		}
	};

	struct TensorStorageRef
	{
		TensorType type;
		std::optional<QuantizationParams> quantization;
		BufferRegion region;
		std::size_t storageOffsetBytes{};
		std::vector<std::size_t> viewStrides;
		std::string layoutTag;
		std::size_t aliasSet{};
		BufferMutability viewMutability{ BufferMutability::Immutable };

		bool IsExternal() const noexcept
		{
			return region.IsExternal();
		}

		std::optional<std::size_t> LogicalByteSize() const
		{
			return type.ByteSize();
		}

		TensorView View() const
		{
			return { .type = type,
				     .quantization = quantization,
				     .storageOffsetBytes = storageOffsetBytes,
				     .strides = viewStrides,
				     .layoutTag = layoutTag,
				     .aliasSet = aliasSet,
				     .mutability = viewMutability };
		}
	};

	struct RuntimeBufferBinding
	{
		std::string name;
		TensorType type;
		std::optional<QuantizationParams> quantization;
		BufferOwnership ownership{ BufferOwnership::Owned };
		ExternalBufferKind externalKind{ ExternalBufferKind::None };
		TensorMemorySpace memorySpace{ TensorMemorySpace::Host };
		std::size_t memoryBuffer{};
		std::size_t byteOffset{};
		std::size_t byteSize{};
		std::size_t alignment{ 1 };
		std::uint64_t checksum{};
		BufferMutability mutability{ BufferMutability::Immutable };
		BufferRebindPolicy rebindPolicy{ BufferRebindPolicy::ExactMetadataAndChecksum };
		std::vector<std::size_t> strides;
		std::string layoutTag;
		std::size_t aliasSet{};
	};

	inline RuntimeBufferBinding ToRuntimeBufferBinding(std::string name, const TensorStorageRef& storage,
	                                                   std::size_t memoryBuffer)
	{
		return { .name = std::move(name),
			     .type = storage.type,
			     .quantization = storage.quantization,
			     .ownership = storage.region.ownership,
			     .externalKind = storage.region.externalKind,
			     .memorySpace = storage.region.memorySpace,
			     .memoryBuffer = memoryBuffer,
			     .byteOffset = storage.region.byteOffset + storage.storageOffsetBytes,
			     .byteSize = storage.LogicalByteSize().value_or(storage.region.byteSize),
			     .alignment = storage.region.alignment,
			     .checksum = storage.region.checksum,
			     .mutability = storage.region.mutability,
			     .rebindPolicy = storage.region.rebindPolicy,
			     .strides = storage.viewStrides,
			     .layoutTag = storage.layoutTag,
			     .aliasSet = storage.aliasSet };
	}

	inline void ValidateRuntimeBufferBinding(const RuntimeBufferBinding& binding)
	{
		if (binding.name.empty())
		{
			throw std::runtime_error("Runtime buffer binding name cannot be empty");
		}
		if (!IsValidDataTypeValue(binding.type.dtype))
		{
			throw std::runtime_error("Runtime buffer binding has invalid dtype: " + binding.name);
		}
		if (binding.alignment == 0)
		{
			throw std::runtime_error("Runtime buffer binding has zero alignment: " + binding.name);
		}
		if (!binding.strides.empty() && binding.strides.size() != binding.type.Rank())
		{
			throw std::runtime_error(std::format("Runtime buffer binding '{}' has {} strides for rank {}", binding.name,
			                                     binding.strides.size(), binding.type.Rank()));
		}
		if (const auto logicalBytes = binding.type.ByteSize();
		    logicalBytes && binding.byteSize != 0 && *logicalBytes > binding.byteSize)
		{
			throw std::runtime_error("Runtime buffer binding byte size is smaller than its tensor type: " +
			                         binding.name);
		}
		if (binding.externalKind != ExternalBufferKind::None && binding.ownership == BufferOwnership::Owned)
		{
			throw std::runtime_error("Runtime external buffer binding cannot use owned storage: " + binding.name);
		}
	}

	inline BufferRegion MakeBorrowedBufferRegion(const void* data, std::size_t byteSize,
	                                             TensorMemorySpace memorySpace = TensorMemorySpace::Host)
	{
		return { .ownership = BufferOwnership::Borrowed,
			     .externalKind = ExternalBufferKind::None,
			     .memorySpace = memorySpace,
			     .data = data,
			     .byteOffset = 0,
			     .byteSize = byteSize };
	}

	inline BufferRegion MakeExternalBufferRegion(std::string name, const void* data, std::size_t byteSize,
	                                             ExternalBufferKind kind,
	                                             TensorMemorySpace memorySpace = TensorMemorySpace::External)
	{
		return { .ownership = BufferOwnership::External,
			     .externalKind = kind,
			     .memorySpace = memorySpace,
			     .name = std::move(name),
			     .data = data,
			     .byteOffset = 0,
			     .byteSize = byteSize };
	}
} // namespace LiteNN

#endif
