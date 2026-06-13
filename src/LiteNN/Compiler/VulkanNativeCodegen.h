#ifndef LITENN_COMPILER_VULKAN_NATIVE_CODEGEN_H
#define LITENN_COMPILER_VULKAN_NATIVE_CODEGEN_H

#include <LiteNN/Operators.h>

#include <cstddef>
#include <cstdint>
#include <span>
#include <string>
#include <string_view>
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
	std::string VulkanNativeSameShapeBinaryF32KernelName(BinaryOp op);
	VulkanNativeGeneratedSPIRV VulkanNativeSameShapeBinaryF32ChainSPIRV(std::span<const BinaryOp> ops,
	                                                                    std::uint32_t elementCount);
	bool VulkanNativeSupportsSameShapeCast(DataType srcType, DataType dstType);
	VulkanNativeGeneratedSPIRV VulkanNativeSameShapeCastSPIRV(DataType srcType, DataType dstType,
	                                                          std::uint32_t elementCount);
	bool VulkanNativeSupportsMatMulF32(std::uint32_t m, std::uint32_t k, std::uint32_t n);
	VulkanNativeGeneratedSPIRV VulkanNativeMatMulF32SPIRV(std::uint32_t m, std::uint32_t k, std::uint32_t n);
	bool VulkanNativeSupportsMatMulBiasF32(std::uint32_t m, std::uint32_t k, std::uint32_t n, std::uint32_t biasRows);
	VulkanNativeGeneratedSPIRV VulkanNativeMatMulBiasF32SPIRV(std::uint32_t m, std::uint32_t k, std::uint32_t n,
	                                                          std::uint32_t biasRows, bool relu);
	std::string_view VulkanNativeReduceF32KernelName(ReduceOp op);
	bool VulkanNativeSupportsReduceF32(ReduceOp op, std::span<const std::size_t> inputShape, std::size_t axis);
	VulkanNativeGeneratedSPIRV VulkanNativeReduceF32SPIRV(ReduceOp op, std::span<const std::size_t> inputShape,
	                                                      std::size_t axis);
	bool VulkanNativeSupportsSoftmaxF32(std::span<const std::size_t> inputShape, std::size_t axis);
	VulkanNativeGeneratedSPIRV VulkanNativeSoftmaxF32SPIRV(std::span<const std::size_t> inputShape, std::size_t axis);
	std::string_view VulkanNativeNormalizationF32KernelName(NormalizationMode mode);
	bool VulkanNativeSupportsNormalizationF32(NormalizationMode mode, std::span<const std::size_t> inputShape,
	                                          std::size_t axis);
	VulkanNativeGeneratedSPIRV VulkanNativeNormalizationF32SPIRV(NormalizationMode mode,
	                                                             std::span<const std::size_t> inputShape,
	                                                             std::size_t axis, double epsilon);
} // namespace LiteNN

#endif
