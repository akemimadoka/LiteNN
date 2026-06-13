#ifndef LITENN_COMPILER_VULKAN_NATIVE_PAYLOAD_H
#define LITENN_COMPILER_VULKAN_NATIVE_PAYLOAD_H

#include <cstddef>
#include <cstdint>
#include <span>
#include <string>
#include <vector>

namespace LiteNN
{
	enum class VulkanNativeArgumentKind : std::uint32_t
	{
		InputTensor = 1,
		OutputTensor = 2,
		ExternalTensor = 3,
	};

	enum class VulkanNativeFeature : std::uint32_t
	{
		StaticShape = 0,
		SingleSubgraph = 1,
		SameShapeElementwiseAddF32 = 2,
		SameShapeElementwiseSubtractF32 = 3,
		SameShapeElementwiseMultiplyF32 = 4,
		SameShapeElementwiseDivideF32 = 5,
		SameShapeElementwiseMaxF32 = 6,
		SameShapeElementwiseMinF32 = 7,
		SameShapeElementwiseNegateF32 = 8,
		SameShapeElementwiseAbsF32 = 9,
		SameShapeElementwiseSqrtF32 = 10,
		SameShapeElementwiseExpF32 = 11,
		SameShapeElementwiseLogF32 = 12,
		SameShapeElementwiseSinF32 = 13,
		SameShapeElementwiseCosF32 = 14,
		SameShapeCastFloat32ToInt32 = 15,
		SameShapeCastInt32ToFloat32 = 16,
		SameShapeCastLowPrecision = 17,
		MatMulF32 = 18,
		MatMulBiasAddF32 = 19,
		MatMulBiasAddReLUF32 = 20,
		ReduceF32 = 21,
		SoftmaxF32 = 22,
		NormalizationF32 = 23,
		Pool2DF32 = 24,
	};

	struct VulkanNativeFeatureSet
	{
		static constexpr std::uint64_t KnownFeatureMask = (1ull << 25) - 1;

		std::uint64_t flags{};

		constexpr void AddFeature(VulkanNativeFeature feature)
		{
			flags |= (1ull << static_cast<std::uint32_t>(feature));
		}

		constexpr bool CheckIsValid() const
		{
			return (flags & ~KnownFeatureMask) == 0;
		}
	};

	enum class VulkanNativeDeviceRequirement : std::uint64_t
	{
		ShaderFloat16 = 1ull << 0,
		ShaderInt8 = 1ull << 1,
		StorageBuffer16BitAccess = 1ull << 2,
		StorageBuffer8BitAccess = 1ull << 3,
		SubgroupArithmetic = 1ull << 4,
		SubgroupBallot = 1ull << 5,
		SubgroupShuffle = 1ull << 6,
		ShaderStorageBufferArrayNonUniformIndexing = 1ull << 7,
		DescriptorBindingStorageBufferUpdateAfterBind = 1ull << 8,
		DescriptorBindingPartiallyBound = 1ull << 9,
		DescriptorBindingVariableDescriptorCount = 1ull << 10,
		RuntimeDescriptorArray = 1ull << 11,
	};

	struct VulkanNativeDeviceRequirementSet
	{
		static constexpr std::uint64_t KnownRequirementMask = (1ull << 12) - 1;

		std::uint64_t flags{};

		constexpr void AddRequirement(VulkanNativeDeviceRequirement requirement)
		{
			flags |= static_cast<std::uint64_t>(requirement);
		}

		constexpr bool HasRequirement(VulkanNativeDeviceRequirement requirement) const
		{
			return (flags & static_cast<std::uint64_t>(requirement)) != 0;
		}

		constexpr bool CheckIsValid() const
		{
			return (flags & ~KnownRequirementMask) == 0;
		}
	};

	struct VulkanNativeDispatchDim
	{
		std::uint32_t x{ 1 };
		std::uint32_t y{ 1 };
		std::uint32_t z{ 1 };
	};

	struct VulkanNativeKernelRequirements
	{
		std::uint32_t descriptorAbiVersion{ 1 };
		VulkanNativeDispatchDim localSize{ 1, 1, 1 };
		VulkanNativeDeviceRequirementSet deviceRequirements;
		std::uint32_t requiredSubgroupSize{};
		std::uint32_t requiredStorageBufferOffsetAlignment{};
	};

	struct VulkanNativeArgumentSpec
	{
		VulkanNativeArgumentKind kind{ VulkanNativeArgumentKind::InputTensor };
		std::uint32_t index{};
		std::uint32_t binding{};
		std::uint64_t byteOffset{};
		std::uint64_t byteSize{};
	};

	struct VulkanNativeKernelSpec
	{
		std::string entryPoint{ "main" };
		VulkanNativeDispatchDim groups;
		VulkanNativeKernelRequirements requirements;
		std::vector<VulkanNativeArgumentSpec> arguments;
	};

	struct VulkanNativeInstructionPayload
	{
		VulkanNativeFeatureSet featureSet;
		std::string target{ "vulkan1.1" };
		std::vector<std::uint32_t> spirv;
		std::vector<VulkanNativeKernelSpec> kernels;
	};

	std::vector<std::byte> SerializeVulkanNativeInstructionPayload(const VulkanNativeInstructionPayload& payload);
	VulkanNativeInstructionPayload DeserializeVulkanNativeInstructionPayload(std::span<const std::byte> bytes);
} // namespace LiteNN

#endif
