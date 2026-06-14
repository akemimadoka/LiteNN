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
	                                          std::size_t axis, std::size_t groupCount = 1);
	VulkanNativeGeneratedSPIRV VulkanNativeNormalizationF32SPIRV(NormalizationMode mode,
	                                                             std::span<const std::size_t> inputShape,
	                                                             std::size_t axis, double epsilon,
	                                                             bool hasScale = false, bool hasBias = false,
	                                                             std::size_t groupCount = 1);
	std::string_view VulkanNativePool2DF32KernelName(PoolMode mode);
	bool VulkanNativeSupportsPool2DF32(PoolMode mode, std::span<const std::size_t> inputShape,
	                                   std::span<const std::size_t> outputShape,
	                                   std::span<const std::size_t> kernelShape,
	                                   std::span<const std::size_t> strides,
	                                   std::span<const std::size_t> lowPads,
	                                   std::span<const std::size_t> highPads,
	                                   bool countIncludePad);
	VulkanNativeGeneratedSPIRV VulkanNativePool2DF32SPIRV(PoolMode mode,
	                                                      std::span<const std::size_t> inputShape,
	                                                      std::span<const std::size_t> outputShape,
	                                                      std::span<const std::size_t> kernelShape,
	                                                      std::span<const std::size_t> strides,
	                                                      std::span<const std::size_t> lowPads = {},
	                                                      std::span<const std::size_t> highPads = {},
	                                                      bool countIncludePad = false);
	std::string_view VulkanNativeConv2DF32KernelName();
	bool VulkanNativeSupportsConv2DF32(std::span<const std::size_t> inputShape,
	                                   std::span<const std::size_t> weightShape,
	                                   std::span<const std::size_t> outputShape,
	                                   std::span<const std::size_t> strides,
	                                   std::span<const std::size_t> dilations,
	                                   std::span<const std::size_t> lowPads,
	                                   std::span<const std::size_t> highPads,
	                                   std::size_t groupCount);
	VulkanNativeGeneratedSPIRV VulkanNativeConv2DF32SPIRV(std::span<const std::size_t> inputShape,
	                                                      std::span<const std::size_t> weightShape,
	                                                      std::span<const std::size_t> outputShape,
	                                                      std::span<const std::size_t> strides,
	                                                      std::span<const std::size_t> dilations,
	                                                      std::span<const std::size_t> lowPads,
	                                                      std::span<const std::size_t> highPads,
	                                                      std::size_t groupCount,
	                                                      bool hasBias);
	std::string_view VulkanNativeUpsampleNearestF32KernelName();
	bool VulkanNativeSupportsUpsampleNearestF32(std::span<const std::size_t> inputShape,
	                                           std::span<const std::size_t> outputShape,
	                                           bool alignCorners);
	VulkanNativeGeneratedSPIRV VulkanNativeUpsampleNearestF32SPIRV(std::span<const std::size_t> inputShape,
	                                                               std::span<const std::size_t> outputShape,
	                                                               bool alignCorners);
} // namespace LiteNN

#endif
