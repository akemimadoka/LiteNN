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

	bool VulkanNativeSupportsSameShapeBinaryF32(BinaryOp op);
	VulkanNativeGeneratedSPIRV VulkanNativeSameShapeBinaryF32SPIRV(BinaryOp op);
} // namespace LiteNN

#endif
