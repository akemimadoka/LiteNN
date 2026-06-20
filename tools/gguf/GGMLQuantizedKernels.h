#ifndef LITENN_GGML_QUANTIZED_KERNELS_H
#define LITENN_GGML_QUANTIZED_KERNELS_H

#include <LiteNN/Graph.h>

#include <string_view>

namespace LiteNN::GGUF
{
	Tensor<CPU> DequantizeGGMLBlockVariable(const Variable& variable, std::string_view name);

	Tensor<CPU> EvalGGMLQuantizedMatMul(const Tensor<CPU>& input, const Variable& weight, bool transposeWeight);
} // namespace LiteNN::GGUF

#endif
