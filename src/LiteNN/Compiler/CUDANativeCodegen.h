#ifndef LITENN_COMPILER_CUDA_NATIVE_CODEGEN_H
#define LITENN_COMPILER_CUDA_NATIVE_CODEGEN_H

#include <LiteNN/Operators.h>
#include <LiteNN/Quantization.h>

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

	struct CUDANativeSoftmaxF32CodegenSpec
	{
		std::span<const std::size_t> inputShape;
		std::size_t axis{};
	};

	struct CUDANativeGetRowsF32CodegenSpec
	{
		DataType indexType{ DataType::Int32 };
		std::uint32_t rowSize{};
	};

	struct CUDANativeRMSNormF32CodegenSpec
	{
		std::uint32_t rowSize{};
		float epsilon{ 1.0e-5F };
		bool hasScale{};
	};

	struct CUDANativeRoPEF32CodegenSpec
	{
		std::uint32_t featureSize{};
		std::uint32_t positionOffset{};
		std::optional<DataType> positionType;
	};

	struct CUDANativeMatMulBiasEpilogueF32CodegenSpec
	{
		std::string kernelName;
		std::vector<std::size_t> outputShape;
		std::vector<std::size_t> biasShape;
		bool relu{};
	};

	struct CUDANativeMatMulBiasEpilogueCodegenSpec
	{
		std::string kernelName;
		DataType dtype{ DataType::Float32 };
		std::vector<std::size_t> outputShape;
		std::vector<std::size_t> biasShape;
		bool relu{};
	};

	struct CUDANativeCastCodegenSpec
	{
		DataType srcType{ DataType::Float32 };
		DataType dstType{ DataType::Float32 };
	};

	struct CUDANativeGGMLBlockMatMulF32CodegenSpec
	{
		QuantizedBlockFormat format{ QuantizedBlockFormat::GGML_Q8_0 };
		std::uint32_t m{};
		std::uint32_t k{};
		std::uint32_t n{};
	};

	std::string_view CUDANativeBinaryF32KernelName(BinaryOp op, bool broadcast = false);
	std::string_view CUDANativeUnaryF32KernelName(UnaryOp op);
	std::string_view CUDANativeReduceF32KernelName(ReduceOp op);
	std::string CUDANativeConcatF32KernelName(std::size_t inputIndex);
	std::string_view CUDANativeSliceF32KernelName();
	std::string_view CUDANativeSoftmaxF32KernelName();
	std::string CUDANativeGetRowsF32KernelName(DataType indexType);
	std::string_view CUDANativeRMSNormF32KernelName(bool hasScale);
	std::string CUDANativeRoPEF32KernelName(std::optional<DataType> positionType);
	std::string_view CUDANativeMatMulBiasEpilogueF32KernelName(bool relu);
	std::string CUDANativeMatMulBiasEpilogueKernelName(DataType dtype, bool relu);
	bool CUDANativeSupportsCast(DataType srcType, DataType dstType);
	std::string CUDANativeCastKernelName(DataType srcType, DataType dstType);
	std::string_view CUDANativeGGMLBlockMatMulF32KernelName(QuantizedBlockFormat format);
	std::string CUDANativeNVPTXTargetChip();
	std::string CUDANativeNVPTXTargetChip(std::string_view requestedTarget);

	/**
	 * Generates a minimal same-shape CUDA binary f32 kernel by lowering MLIR GPU/NVVM dialects to NVPTX PTX.
	 *
	 * Broadcast binary kernels use the overload below, which emits static shape index lowering in MLIR.
	 */
	std::string CUDANativeBinaryF32PTXFromMLIRNVPTX(BinaryOp op);
	std::optional<std::string> TryCUDANativeBinaryF32PTXFromMLIRNVPTX(BinaryOp op);
	std::string CUDANativeBinaryBroadcastF32PTXFromMLIRNVPTX(const CUDANativeBroadcastBinaryF32CodegenSpec& spec);
	std::optional<std::string>
	TryCUDANativeBinaryBroadcastF32PTXFromMLIRNVPTX(const CUDANativeBroadcastBinaryF32CodegenSpec& spec);
	std::string CUDANativeReduceF32PTXFromMLIRNVPTX(const CUDANativeReduceF32CodegenSpec& spec);
	std::optional<std::string> TryCUDANativeReduceF32PTXFromMLIRNVPTX(const CUDANativeReduceF32CodegenSpec& spec);
	std::string CUDANativeConcatF32PTXFromMLIRNVPTX(const CUDANativeConcatF32CodegenSpec& spec);
	std::optional<std::string> TryCUDANativeConcatF32PTXFromMLIRNVPTX(const CUDANativeConcatF32CodegenSpec& spec);
	std::string CUDANativeSliceF32PTXFromMLIRNVPTX(const CUDANativeSliceF32CodegenSpec& spec);
	std::optional<std::string> TryCUDANativeSliceF32PTXFromMLIRNVPTX(const CUDANativeSliceF32CodegenSpec& spec);
	std::string CUDANativeSoftmaxF32PTXFromMLIRNVPTX(const CUDANativeSoftmaxF32CodegenSpec& spec);
	std::optional<std::string> TryCUDANativeSoftmaxF32PTXFromMLIRNVPTX(const CUDANativeSoftmaxF32CodegenSpec& spec);
	std::string CUDANativeGetRowsF32PTXFromMLIRNVPTX(const CUDANativeGetRowsF32CodegenSpec& spec);
	std::optional<std::string> TryCUDANativeGetRowsF32PTXFromMLIRNVPTX(const CUDANativeGetRowsF32CodegenSpec& spec);
	std::string CUDANativeRMSNormF32PTXFromMLIRNVPTX(const CUDANativeRMSNormF32CodegenSpec& spec);
	std::optional<std::string> TryCUDANativeRMSNormF32PTXFromMLIRNVPTX(const CUDANativeRMSNormF32CodegenSpec& spec);
	std::string CUDANativeRoPEF32PTXFromMLIRNVPTX(const CUDANativeRoPEF32CodegenSpec& spec);
	std::optional<std::string> TryCUDANativeRoPEF32PTXFromMLIRNVPTX(const CUDANativeRoPEF32CodegenSpec& spec);
	std::string CUDANativeCastPTXFromMLIRNVPTX(const CUDANativeCastCodegenSpec& spec);
	std::optional<std::string> TryCUDANativeCastPTXFromMLIRNVPTX(const CUDANativeCastCodegenSpec& spec);
	std::string CUDANativeMatMulBiasEpilogueF32PTXFromMLIRNVPTX(const CUDANativeMatMulBiasEpilogueF32CodegenSpec& spec);
	std::optional<std::string>
	TryCUDANativeMatMulBiasEpilogueF32PTXFromMLIRNVPTX(const CUDANativeMatMulBiasEpilogueF32CodegenSpec& spec);
	std::string CUDANativeMatMulBiasEpiloguePTXFromMLIRNVPTX(const CUDANativeMatMulBiasEpilogueCodegenSpec& spec);
	std::optional<std::string>
	TryCUDANativeMatMulBiasEpiloguePTXFromMLIRNVPTX(const CUDANativeMatMulBiasEpilogueCodegenSpec& spec);
	std::string
	CUDANativeMatMulBiasEpiloguesF32PTXFromMLIRNVPTX(std::span<const CUDANativeMatMulBiasEpilogueF32CodegenSpec> specs);
	std::optional<std::string> TryCUDANativeMatMulBiasEpiloguesF32PTXFromMLIRNVPTX(
	    std::span<const CUDANativeMatMulBiasEpilogueF32CodegenSpec> specs);
	std::string
	CUDANativeMatMulBiasEpiloguesPTXFromMLIRNVPTX(std::span<const CUDANativeMatMulBiasEpilogueCodegenSpec> specs);
	std::optional<std::string>
	TryCUDANativeMatMulBiasEpiloguesPTXFromMLIRNVPTX(std::span<const CUDANativeMatMulBiasEpilogueCodegenSpec> specs);
	std::string CUDANativeGGMLBlockMatMulF32PTXFromMLIRNVPTX(const CUDANativeGGMLBlockMatMulF32CodegenSpec& spec);
	std::optional<std::string>
	TryCUDANativeGGMLBlockMatMulF32PTXFromMLIRNVPTX(const CUDANativeGGMLBlockMatMulF32CodegenSpec& spec);

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
