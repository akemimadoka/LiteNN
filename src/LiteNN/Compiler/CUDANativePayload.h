#ifndef LITENN_COMPILER_CUDA_NATIVE_PAYLOAD_H
#define LITENN_COMPILER_CUDA_NATIVE_PAYLOAD_H

#include <cstddef>
#include <cstdint>
#include <span>
#include <string>
#include <vector>

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
	};

	inline constexpr std::uint64_t kCUDANativeFeatureStaticShape = 1ull << 0;
	inline constexpr std::uint64_t kCUDANativeFeatureSingleSubgraph = 1ull << 1;
	inline constexpr std::uint64_t kCUDANativeFeatureElementwiseAddF32 = 1ull << 2;
	inline constexpr std::uint64_t kCUDANativeFeatureElementwiseSubtractF32 = 1ull << 3;
	inline constexpr std::uint64_t kCUDANativeFeatureElementwiseMultiplyF32 = 1ull << 4;
	inline constexpr std::uint64_t kCUDANativeFeatureElementwiseDivideF32 = 1ull << 5;
	inline constexpr std::uint64_t kCUDANativeFeatureElementwiseNegateF32 = 1ull << 6;
	inline constexpr std::uint64_t kCUDANativeFeatureElementwiseAbsF32 = 1ull << 7;
	inline constexpr std::uint64_t kCUDANativeFeatureElementwiseSqrtF32 = 1ull << 8;
	inline constexpr std::uint64_t kCUDANativeFeatureElementwiseBroadcastF32 = 1ull << 9;
	inline constexpr std::uint64_t kCUDANativeFeatureMatMulCUBLASF32 = 1ull << 10;
	inline constexpr std::uint64_t kCUDANativeKnownFeatureMask = kCUDANativeFeatureStaticShape |
	                                                             kCUDANativeFeatureSingleSubgraph |
	                                                             kCUDANativeFeatureElementwiseAddF32 |
	                                                             kCUDANativeFeatureElementwiseSubtractF32 |
	                                                             kCUDANativeFeatureElementwiseMultiplyF32 |
	                                                             kCUDANativeFeatureElementwiseDivideF32 |
	                                                             kCUDANativeFeatureElementwiseNegateF32 |
	                                                             kCUDANativeFeatureElementwiseAbsF32 |
	                                                             kCUDANativeFeatureElementwiseSqrtF32 |
	                                                             kCUDANativeFeatureElementwiseBroadcastF32 |
	                                                             kCUDANativeFeatureMatMulCUBLASF32;

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
		std::uint64_t featureFlags{};
		std::string target;
		std::vector<std::byte> binary;
		std::vector<std::byte> scalarData;
		std::uint64_t workspaceBytes{};
		std::vector<CUDANativeKernelSpec> kernels;
	};

	std::vector<std::byte> SerializeCUDANativeInstructionPayload(const CUDANativeInstructionPayload& payload);
	CUDANativeInstructionPayload DeserializeCUDANativeInstructionPayload(std::span<const std::byte> bytes);
} // namespace LiteNN

#endif
