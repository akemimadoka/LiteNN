#ifndef LITENN_DEVICE_VULKAN_H
#define LITENN_DEVICE_VULKAN_H

#include <LiteNN/Device.h>

#include <array>
#include <cstdint>
#include <memory>
#include <span>
#include <string>
#include <string_view>

#ifdef LITENN_ENABLE_VULKAN

namespace LiteNN
{
	class VulkanContext;

	/// Host fallback is an explicit execution policy, not part of Vulkan device identity.
	enum class VulkanHostFallbackPolicy
	{
		Reject,
		Allow
	};

	struct Vulkan
	{
		std::uint32_t deviceIndex = 0;
		VulkanHostFallbackPolicy hostFallbackPolicy{ VulkanHostFallbackPolicy::Reject };
		mutable std::string infoCache;
		mutable std::shared_ptr<VulkanContext> context;

		bool operator==(const Vulkan& other) const
		{
			return deviceIndex == other.deviceIndex;
		}
	};

	struct VulkanDispatchDim
	{
		std::uint32_t x{ 1 };
		std::uint32_t y{ 1 };
		std::uint32_t z{ 1 };
	};

	struct VulkanExecutionOptions
	{
		bool synchronize{ true };
	};

	struct VulkanDeviceCapabilities
	{
		std::uint32_t apiVersionMajor{};
		std::uint32_t apiVersionMinor{};
		std::uint32_t apiVersionPatch{};
		std::uint32_t maxComputeWorkGroupInvocations{};
		std::array<std::uint32_t, 3> maxComputeWorkGroupSize{};
		std::array<std::uint32_t, 3> maxComputeWorkGroupCount{};
		std::uint64_t maxStorageBufferRange{};
		std::uint64_t minStorageBufferOffsetAlignment{};
		std::uint32_t maxPerStageDescriptorStorageBuffers{};
		std::uint32_t maxDescriptorSetStorageBuffers{};
		std::uint32_t maxBoundDescriptorSets{};
		std::uint32_t subgroupSize{};
		bool subgroupComputeAvailable{};
		bool subgroupBasicAvailable{};
		bool subgroupArithmeticAvailable{};
		bool shaderFloat16Available{};
		bool shaderInt8Available{};
		bool storageBuffer16BitAccessAvailable{};
		bool storageBuffer8BitAccessAvailable{};
		bool shaderFloat16Enabled{};
		bool shaderInt8Enabled{};
		bool storageBuffer16BitAccessEnabled{};
		bool storageBuffer8BitAccessEnabled{};
		std::string deviceName;
	};

	std::uint32_t VulkanDeviceCount() noexcept;
	bool IsVulkanDeviceAvailable(std::uint32_t deviceIndex = 0) noexcept;
	VulkanDeviceCapabilities QueryVulkanDeviceCapabilities(const Vulkan& device);

	class VulkanComputeModule
	{
	public:
		VulkanComputeModule();
		VulkanComputeModule(Vulkan device, std::span<const std::uint32_t> spirv, std::string_view entryPoint,
		                    std::uint32_t descriptorCount);
		VulkanComputeModule(const VulkanComputeModule&) = delete;
		VulkanComputeModule& operator=(const VulkanComputeModule&) = delete;
		VulkanComputeModule(VulkanComputeModule&&) noexcept;
		VulkanComputeModule& operator=(VulkanComputeModule&&) noexcept;
		~VulkanComputeModule();

		bool Empty() const noexcept;
		double CreationWallTimeMs() const noexcept;
		void Dispatch(std::span<const void*> descriptorBuffers, VulkanDispatchDim groups,
		              VulkanExecutionOptions options = {}) const;

	private:
		struct Impl;

		std::unique_ptr<Impl> impl_;
	};

	template <>
	struct DeviceTraits<Vulkan>
	{
		static consteval std::meta::info DataTypeMappingFunc(DataType dataType)
		{
			switch (dataType)
			{
			case DataType::Float32:
				return ^^float;
			case DataType::Float64:
				return ^^double;
			case DataType::Float16:
				return ^^Float16;
			case DataType::BFloat16:
				return ^^BFloat16;
			case DataType::Float8E4M3:
				return ^^Float8E4M3;
			case DataType::Float8E5M2:
				return ^^Float8E5M2;
			case DataType::Int32:
				return ^^int32_t;
			case DataType::Int64:
				return ^^int64_t;
			case DataType::Int8:
				return ^^int8_t;
			case DataType::UInt8:
				return ^^uint8_t;
			case DataType::Bool:
				return ^^bool;
			}
		}

		template <DataType DT>
		using DataTypeMapping = [:DataTypeMappingFunc(DT):];

		static std::string_view Name();
		static std::string_view Info(const Vulkan& device);
		static void* Allocate(Vulkan& device, DataType type, std::size_t size);
		static void Deallocate(Vulkan& device, void* ptr, DataType type, std::size_t size);
		static void ZeroFill(Vulkan& device, void* ptr, DataType type, std::size_t size);
		static void CopyToCPU(Vulkan& device, DataType srcType, const void* src, std::size_t size, DataType dstType,
		                      void* dst);
		static void CopyFromCPU(Vulkan& device, DataType dstType, void* dst, DataType srcType, const void* src,
		                        std::size_t size);
		static void ConvertTo(Vulkan& device, DataType srcType, const void* src, std::size_t size, DataType dstType,
		                      void* dst);
		static void DoUnaryOp(Vulkan& device, UnaryOp unaryOp, void* dst, DataType type, ShapeView shape,
		                      const void* src);
		static void DoBinaryOp(Vulkan& device, BinaryOp binaryOp, void* dst, DataType type1, ShapeView shape1,
		                       const void* src1, DataType type2, ShapeView shape2, const void* src2);
		static void DoReduceOp(Vulkan& device, ReduceOp reduceOp, void* dst, DataType type, ShapeView shape,
		                       const void* src, std::size_t axis);
		static void DoConcatOp(Vulkan& device, void* dst, DataType type, const void* const* srcPtrs,
		                       const ShapeView* srcShapes, std::size_t inputCount, std::size_t axis);
		static void DoSliceOp(Vulkan& device, void* dst, DataType type, ShapeView srcShape, const void* src,
		                      std::size_t axis, std::size_t start, std::size_t length);
		static void DoGetRowsOp(Vulkan& device, void* dst, DataType dataType, ShapeView dataShape, const void* data,
		                       DataType indexType, ShapeView indexShape, const void* indices);
		static void DoPermuteOp(Vulkan& device, void* dst, DataType type, ShapeView srcShape, const void* src,
		                       ShapeView permutation);
	};
} // namespace LiteNN

#endif

#endif
