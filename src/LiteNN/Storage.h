#ifndef LITENN_STORAGE_H
#define LITENN_STORAGE_H

#include <LiteNN/Quantization.h>
#include <LiteNN/TensorType.h>
#include <cstddef>
#include <cstdint>
#include <memory>
#include <optional>
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
