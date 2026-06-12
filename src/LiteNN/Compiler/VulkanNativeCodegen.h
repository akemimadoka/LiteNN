#ifndef LITENN_COMPILER_VULKAN_NATIVE_CODEGEN_H
#define LITENN_COMPILER_VULKAN_NATIVE_CODEGEN_H

#include <LiteNN/Operators.h>

#include <cstdint>
#include <string>
#include <vector>

namespace LiteNN
{
	struct VulkanNativeGeneratedSPIRV
	{
		std::vector<std::uint32_t> words;
		std::string mlir;
	};

	constexpr std::uint32_t kVulkanNativeElementwiseWorkgroupSize = 64;
	constexpr std::uint32_t kVulkanNativeMatMulWorkgroupSize = 64;

	bool VulkanNativeSupportsSameShapeUnaryF32(UnaryOp op);
	VulkanNativeGeneratedSPIRV VulkanNativeSameShapeUnaryF32SPIRV(UnaryOp op, std::uint32_t elementCount);
	bool VulkanNativeSupportsSameShapeBinaryF32(BinaryOp op);
	VulkanNativeGeneratedSPIRV VulkanNativeSameShapeBinaryF32SPIRV(BinaryOp op, std::uint32_t elementCount);
	bool VulkanNativeSupportsSameShapeCast(DataType srcType, DataType dstType);
	VulkanNativeGeneratedSPIRV VulkanNativeSameShapeCastSPIRV(DataType srcType, DataType dstType,
	                                                          std::uint32_t elementCount);
	bool VulkanNativeSupportsMatMulF32(std::uint32_t m, std::uint32_t k, std::uint32_t n);
	VulkanNativeGeneratedSPIRV VulkanNativeMatMulF32SPIRV(std::uint32_t m, std::uint32_t k, std::uint32_t n);
} // namespace LiteNN

#endif
