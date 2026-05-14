#ifndef LITENN_COMPILER_CUDA_NATIVE_CODEGEN_H
#define LITENN_COMPILER_CUDA_NATIVE_CODEGEN_H

#include <LiteNN/Operators.h>

#include <cstddef>
#include <optional>
#include <span>
#include <string>
#include <string_view>
#include <vector>

namespace LiteNN
{
	struct CUDANativeBroadcastBinaryF32CodegenSpec
	{
		BinaryOp op{ BinaryOp::Add };
		std::span<const std::size_t> outputShape;
		std::span<const std::size_t> lhsShape;
		std::span<const std::size_t> rhsShape;
	};

	std::string_view CUDANativeBinaryF32KernelName(BinaryOp op, bool broadcast = false);
	std::string_view CUDANativeUnaryF32KernelName(UnaryOp op);

	std::string CUDANativeBinaryF32PTX(BinaryOp op);
	std::string CUDANativeBinaryBroadcastF32PTX(const CUDANativeBroadcastBinaryF32CodegenSpec& spec);
	std::string CUDANativeUnaryF32PTX(UnaryOp op);

	/**
	 * Generates a minimal same-shape CUDA binary f32 kernel by lowering MLIR GPU/NVVM dialects to NVPTX PTX.
	 *
	 * Broadcast binary kernels use the overload below, which emits static shape index lowering in MLIR.
	 */
	std::string CUDANativeBinaryF32PTXFromMLIRNVPTX(BinaryOp op);
	std::optional<std::string> TryCUDANativeBinaryF32PTXFromMLIRNVPTX(BinaryOp op);
	std::string CUDANativeBinaryBroadcastF32PTXFromMLIRNVPTX(const CUDANativeBroadcastBinaryF32CodegenSpec& spec);
	std::optional<std::string> TryCUDANativeBinaryBroadcastF32PTXFromMLIRNVPTX(
	    const CUDANativeBroadcastBinaryF32CodegenSpec& spec);

	/**
	 * Generates a minimal CUDA unary f32 kernel by lowering MLIR GPU/NVVM dialects to NVPTX PTX.
	 *
	 * This path currently covers `UnaryOp::Negate`, `UnaryOp::Abs`, and `UnaryOp::Sqrt`.
	 * Callers should keep the template PTX path as fallback until the MLIR/NVPTX route covers the
	 * rest of CUDA native codegen.
	 */
	std::string CUDANativeUnaryF32PTXFromMLIRNVPTX(UnaryOp op);
	std::optional<std::string> TryCUDANativeUnaryF32PTXFromMLIRNVPTX(UnaryOp op);

	std::vector<std::byte> CUDANativeTextBytes(std::string_view text);
} // namespace LiteNN

#endif
