#ifndef LITENN_COMPILER_CUDA_NATIVE_CODEGEN_H
#define LITENN_COMPILER_CUDA_NATIVE_CODEGEN_H

#include <LiteNN/Operators.h>

#include <cstddef>
#include <cstdint>
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

	struct CUDANativeReduceF32CodegenSpec
	{
		ReduceOp op{ ReduceOp::Sum };
		std::span<const std::size_t> inputShape;
		std::size_t axis{};
	};

	struct CUDANativeConcatF32CodegenSpec
	{
		std::span<const std::size_t> outputShape;
		std::span<const std::vector<std::size_t>> inputShapes;
		std::size_t axis{};
	};

	struct CUDANativeSliceF32CodegenSpec
	{
		std::span<const std::size_t> inputShape;
		std::span<const std::size_t> outputShape;
		std::size_t axis{};
		std::size_t start{};
	};

	struct CUDANativeMatMulBiasEpilogueF32CodegenSpec
	{
		std::string kernelName;
		std::vector<std::size_t> outputShape;
		std::vector<std::size_t> biasShape;
		bool relu{};
	};

	std::string_view CUDANativeBinaryF32KernelName(BinaryOp op, bool broadcast = false);
	std::string_view CUDANativeUnaryF32KernelName(UnaryOp op);
	std::string_view CUDANativeReduceF32KernelName(ReduceOp op);
	std::string CUDANativeConcatF32KernelName(std::size_t inputIndex);
	std::string_view CUDANativeSliceF32KernelName();
	std::string_view CUDANativeMatMulBiasEpilogueF32KernelName(bool relu);
	std::string CUDANativeNVPTXTargetChip();

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
	std::string CUDANativeReduceF32PTXFromMLIRNVPTX(const CUDANativeReduceF32CodegenSpec& spec);
	std::optional<std::string> TryCUDANativeReduceF32PTXFromMLIRNVPTX(
	    const CUDANativeReduceF32CodegenSpec& spec);
	std::string CUDANativeConcatF32PTXFromMLIRNVPTX(const CUDANativeConcatF32CodegenSpec& spec);
	std::optional<std::string> TryCUDANativeConcatF32PTXFromMLIRNVPTX(
	    const CUDANativeConcatF32CodegenSpec& spec);
	std::string CUDANativeSliceF32PTXFromMLIRNVPTX(const CUDANativeSliceF32CodegenSpec& spec);
	std::optional<std::string> TryCUDANativeSliceF32PTXFromMLIRNVPTX(
	    const CUDANativeSliceF32CodegenSpec& spec);
	std::string CUDANativeMatMulBiasEpilogueF32PTXFromMLIRNVPTX(
	    const CUDANativeMatMulBiasEpilogueF32CodegenSpec& spec);
	std::optional<std::string> TryCUDANativeMatMulBiasEpilogueF32PTXFromMLIRNVPTX(
	    const CUDANativeMatMulBiasEpilogueF32CodegenSpec& spec);
	std::string CUDANativeMatMulBiasEpiloguesF32PTXFromMLIRNVPTX(
	    std::span<const CUDANativeMatMulBiasEpilogueF32CodegenSpec> specs);
	std::optional<std::string> TryCUDANativeMatMulBiasEpiloguesF32PTXFromMLIRNVPTX(
	    std::span<const CUDANativeMatMulBiasEpilogueF32CodegenSpec> specs);

	/**
	 * Generates a minimal CUDA unary f32 kernel by lowering MLIR GPU/NVVM dialects to NVPTX PTX.
	 *
	 * This path currently covers `UnaryOp::Negate`, `UnaryOp::Abs`, `UnaryOp::Sqrt`, `UnaryOp::Exp`,
	 * `UnaryOp::Log`, `UnaryOp::Sin`, and `UnaryOp::Cos`.
	 * Callers should keep the template PTX path as fallback until the MLIR/NVPTX route covers the
	 * rest of CUDA native codegen.
	 */
	std::string CUDANativeUnaryF32PTXFromMLIRNVPTX(UnaryOp op);
	std::optional<std::string> TryCUDANativeUnaryF32PTXFromMLIRNVPTX(UnaryOp op);

	std::vector<std::byte> CUDANativeTextBytes(std::string_view text);
} // namespace LiteNN

#endif
