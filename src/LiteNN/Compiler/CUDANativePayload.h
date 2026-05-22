#ifndef LITENN_COMPILER_CUDA_NATIVE_PAYLOAD_H
#define LITENN_COMPILER_CUDA_NATIVE_PAYLOAD_H

#include <compare>
#include <concepts>
#include <cstddef>
#include <cstdint>
#include <meta>
#include <span>
#include <string>
#include <vector>

#include <LiteNN/Misc.h>

namespace LiteNN
{
	enum class CUDANativeBinaryKind : std::uint32_t
	{
		PTX = 1,
		Cubin = 2,
		Fatbin = 3,
		LibraryCall = 4,
	};

	enum class CUDANativeArgumentKind : std::uint32_t
	{
		InputTensor = 1,
		OutputTensor = 2,
		Workspace = 3,
		Scalar = 4,
		ConstantTensor = 5,
	};

	enum class CUDANativeFeature
	{
		StaticShape,
		SingleSubgraph,
		ElementwiseAddF32,
		ElementwiseSubtractF32,
		ElementwiseMultiplyF32,
		ElementwiseDivideF32,
		ElementwiseNegateF32,
		ElementwiseAbsF32,
		ElementwiseSqrtF32,
		ElementwiseBroadcastF32,
		MatMulCUBLASF32,
		ElementwiseExpF32,
		ElementwiseLogF32,
		ElementwiseSinF32,
		ElementwiseCosF32,
		ElementwiseMaxF32,
		ElementwiseMinF32,
		ReduceF32,
		ConcatF32,
		SliceF32,
		MatMulBiasAddF32,
		MatMulBiasAddReLUF32,
		MultiKernelLaunch,
		Workspace,
		ConstantTensor,
		Cast,
		MatMulCUBLASLowPrecision,
		MatMulBiasAddLowPrecision,
		MatMulBiasAddReLULowPrecision,
	};

	constexpr std::uint64_t FeatureToFlag(std::same_as<CUDANativeFeature> auto... feature)
	{
		return (0 | ... | (1ull << static_cast<std::uint32_t>(feature)));
	}

	struct CUDANativeFeatureSet
	{
		static constexpr std::uint64_t KnownFeatureMask = [] consteval {
			static_assert(IsZeroStartedContinuousEnum<CUDANativeFeature>());
			return (UINT64_C(1) << std::meta::enumerators_of(^^CUDANativeFeature).size()) - 1;
		}();

		std::uint64_t flags{};

		constexpr CUDANativeFeatureSet(std::same_as<CUDANativeFeature> auto... feature) noexcept
		    : flags(FeatureToFlag(feature...))
		{
		}

		constexpr bool HasFeature(std::same_as<CUDANativeFeature> auto... feature) const
		{
			return (flags & FeatureToFlag(feature...)) != 0;
		}

		constexpr void AddFeature(std::same_as<CUDANativeFeature> auto... feature)
		{
			flags |= FeatureToFlag(feature...);
		}

		constexpr void RemoveFeature(std::same_as<CUDANativeFeature> auto... feature)
		{
			flags &= ~FeatureToFlag(feature...);
		}

		constexpr bool CheckIsValid() const
		{
			return (flags & ~KnownFeatureMask) == 0;
		}

		constexpr bool operator==(const CUDANativeFeatureSet& other) const = default;
	};

	struct CUDANativeLaunchDim
	{
		std::uint32_t x{ 1 };
		std::uint32_t y{ 1 };
		std::uint32_t z{ 1 };
	};

	struct CUDANativeArgumentSpec
	{
		CUDANativeArgumentKind kind{ CUDANativeArgumentKind::InputTensor };
		std::uint32_t index{};
		std::uint64_t byteOffset{};
		std::uint64_t byteSize{};
	};

	struct CUDANativeKernelSpec
	{
		std::string name;
		CUDANativeLaunchDim grid;
		CUDANativeLaunchDim block;
		std::uint32_t sharedMemoryBytes{};
		std::uint64_t workspaceBytes{};
		std::vector<CUDANativeArgumentSpec> arguments;
	};

	struct CUDANativeInstructionPayload
	{
		CUDANativeBinaryKind binaryKind{ CUDANativeBinaryKind::PTX };
		CUDANativeFeatureSet featureSet;
		std::string target;
		std::vector<std::byte> binary;
		std::vector<std::byte> scalarData;
		std::vector<std::byte> constantData;
		std::uint64_t workspaceBytes{};
		std::vector<CUDANativeKernelSpec> kernels;
	};

	std::vector<std::byte> SerializeCUDANativeInstructionPayload(const CUDANativeInstructionPayload& payload);
	CUDANativeInstructionPayload DeserializeCUDANativeInstructionPayload(std::span<const std::byte> bytes);
} // namespace LiteNN

#endif
