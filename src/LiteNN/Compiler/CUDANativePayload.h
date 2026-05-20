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
		ConstantTensor = 5,
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
	inline constexpr std::uint64_t kCUDANativeFeatureElementwiseExpF32 = 1ull << 11;
	inline constexpr std::uint64_t kCUDANativeFeatureElementwiseLogF32 = 1ull << 12;
	inline constexpr std::uint64_t kCUDANativeFeatureElementwiseSinF32 = 1ull << 13;
	inline constexpr std::uint64_t kCUDANativeFeatureElementwiseCosF32 = 1ull << 14;
	inline constexpr std::uint64_t kCUDANativeFeatureElementwiseMaxF32 = 1ull << 15;
	inline constexpr std::uint64_t kCUDANativeFeatureElementwiseMinF32 = 1ull << 16;
	inline constexpr std::uint64_t kCUDANativeFeatureReduceF32 = 1ull << 17;
	inline constexpr std::uint64_t kCUDANativeFeatureConcatF32 = 1ull << 18;
	inline constexpr std::uint64_t kCUDANativeFeatureSliceF32 = 1ull << 19;
	inline constexpr std::uint64_t kCUDANativeFeatureMatMulBiasAddF32 = 1ull << 20;
	inline constexpr std::uint64_t kCUDANativeFeatureMatMulBiasAddReLUF32 = 1ull << 21;
	inline constexpr std::uint64_t kCUDANativeFeatureMultiKernelLaunch = 1ull << 22;
	inline constexpr std::uint64_t kCUDANativeFeatureWorkspace = 1ull << 23;
	inline constexpr std::uint64_t kCUDANativeFeatureConstantTensor = 1ull << 24;
	inline constexpr std::uint64_t kCUDANativeFeatureCast = 1ull << 25;
	inline constexpr std::uint64_t kCUDANativeFeatureMatMulCUBLASLowPrecision = 1ull << 26;
	inline constexpr std::uint64_t kCUDANativeFeatureMatMulBiasAddLowPrecision = 1ull << 27;
	inline constexpr std::uint64_t kCUDANativeFeatureMatMulBiasAddReLULowPrecision = 1ull << 28;
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
	                                                             kCUDANativeFeatureMatMulCUBLASF32 |
	                                                             kCUDANativeFeatureElementwiseExpF32 |
	                                                             kCUDANativeFeatureElementwiseLogF32 |
	                                                             kCUDANativeFeatureElementwiseSinF32 |
	                                                             kCUDANativeFeatureElementwiseCosF32 |
	                                                             kCUDANativeFeatureElementwiseMaxF32 |
	                                                             kCUDANativeFeatureElementwiseMinF32 |
	                                                             kCUDANativeFeatureReduceF32 |
	                                                             kCUDANativeFeatureConcatF32 |
	                                                             kCUDANativeFeatureSliceF32 |
	                                                             kCUDANativeFeatureMatMulBiasAddF32 |
	                                                             kCUDANativeFeatureMatMulBiasAddReLUF32 |
	                                                             kCUDANativeFeatureMultiKernelLaunch |
	                                                             kCUDANativeFeatureWorkspace |
	                                                             kCUDANativeFeatureConstantTensor |
	                                                             kCUDANativeFeatureCast |
	                                                             kCUDANativeFeatureMatMulCUBLASLowPrecision |
	                                                             kCUDANativeFeatureMatMulBiasAddLowPrecision |
	                                                             kCUDANativeFeatureMatMulBiasAddReLULowPrecision;

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
		std::vector<std::byte> constantData;
		std::uint64_t workspaceBytes{};
		std::vector<CUDANativeKernelSpec> kernels;
	};

	std::vector<std::byte> SerializeCUDANativeInstructionPayload(const CUDANativeInstructionPayload& payload);
	CUDANativeInstructionPayload DeserializeCUDANativeInstructionPayload(std::span<const std::byte> bytes);
} // namespace LiteNN

#endif
