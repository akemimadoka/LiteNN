#ifndef LITENN_GGML_QUANTIZED_KERNELS_H
#define LITENN_GGML_QUANTIZED_KERNELS_H

#include <LiteNN/Graph.h>

#include <optional>
#include <string_view>

namespace LiteNN::GGUF
{
	Tensor<CPU> DequantizeGGMLBlockVariable(const Variable& variable, std::string_view name);
	Tensor<CPU> EvalGGMLExactDequantizedMatMul(const Tensor<CPU>& input, const Variable& weight, bool transposeWeight);
	Tensor<CPU> EvalGGMLExactDequantizedMatMul(const Tensor<CPU>& input, const Tensor<CPU>& weightStorage,
	                                           const QuantizationParams& params, bool transposeWeight);

	Tensor<CPU> EvalGGMLQuantizedMatMul(const Tensor<CPU>& input, const Variable& weight, bool transposeWeight);
	Tensor<CPU> EvalGGMLQuantizedMatMul(const Tensor<CPU>& input, const Tensor<CPU>& weightStorage,
	                                    const QuantizationParams& params, bool transposeWeight);
	std::optional<Tensor<CPU>> TryEvalGGMLQuantizedMatMul(const Tensor<CPU>& input, const Tensor<CPU>& weightStorage,
	                                                      const QuantizationParams& params, bool transposeWeight);
} // namespace LiteNN::GGUF

#endif
