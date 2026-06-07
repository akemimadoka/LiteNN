#ifndef LITENN_COMPILER_VULKAN_NATIVE_CODEGEN_H
#define LITENN_COMPILER_VULKAN_NATIVE_CODEGEN_H

#include <LiteNN/Operators.h>

#include <cstdint>
#include <span>

namespace LiteNN
{
	bool VulkanNativeSupportsSameShapeBinaryF32(BinaryOp op);
	std::span<const std::uint32_t> VulkanNativeSameShapeBinaryF32SPIRV(BinaryOp op);
} // namespace LiteNN

#endif
