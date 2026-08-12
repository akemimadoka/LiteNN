#include "GGMLQuantizedKernels.h"

#include <cstddef>
#include <cstdint>
#include <format>
#include <mutex>
#include <optional>
#include <stdexcept>
#include <vector>

#include <ggml-cpu.h>
#include <ggml.h>

namespace LiteNN::GGUF
{
	namespace
	{
		std::optional<ggml_type> TryMapGGMLQuantizedType(QuantizedBlockFormat format)
		{
			switch (format)
			{
			case QuantizedBlockFormat::GGML_Q4_0:
				return GGML_TYPE_Q4_0;
			case QuantizedBlockFormat::GGML_Q4_1:
				return GGML_TYPE_Q4_1;
			case QuantizedBlockFormat::GGML_Q5_0:
				return GGML_TYPE_Q5_0;
			case QuantizedBlockFormat::GGML_Q5_1:
				return GGML_TYPE_Q5_1;
			case QuantizedBlockFormat::GGML_Q8_0:
				return GGML_TYPE_Q8_0;
			case QuantizedBlockFormat::GGML_Q8_1:
				return GGML_TYPE_Q8_1;
			case QuantizedBlockFormat::GGML_Q2_K:
				return GGML_TYPE_Q2_K;
			case QuantizedBlockFormat::GGML_Q3_K:
				return GGML_TYPE_Q3_K;
			case QuantizedBlockFormat::GGML_Q4_K:
				return GGML_TYPE_Q4_K;
			case QuantizedBlockFormat::GGML_Q5_K:
				return GGML_TYPE_Q5_K;
			case QuantizedBlockFormat::GGML_Q6_K:
				return GGML_TYPE_Q6_K;
			case QuantizedBlockFormat::GGML_Q8_K:
				return GGML_TYPE_Q8_K;
			case QuantizedBlockFormat::GGML_IQ2_XXS:
				return GGML_TYPE_IQ2_XXS;
			case QuantizedBlockFormat::GGML_IQ2_XS:
				return GGML_TYPE_IQ2_XS;
			case QuantizedBlockFormat::GGML_IQ3_XXS:
				return GGML_TYPE_IQ3_XXS;
			case QuantizedBlockFormat::GGML_IQ1_S:
				return GGML_TYPE_IQ1_S;
			case QuantizedBlockFormat::GGML_IQ4_NL:
				return GGML_TYPE_IQ4_NL;
			case QuantizedBlockFormat::GGML_IQ3_S:
				return GGML_TYPE_IQ3_S;
			case QuantizedBlockFormat::GGML_IQ2_S:
				return GGML_TYPE_IQ2_S;
			case QuantizedBlockFormat::GGML_IQ4_XS:
				return GGML_TYPE_IQ4_XS;
			default:
				return std::nullopt;
			}
		}

		struct GGMLBlockLayout
		{
			ggml_type type{};
			const ggml_type_traits* traits{};
			std::size_t rowCount{};
			std::size_t rowSize{};
			std::size_t rowBytes{};
		};

		GGMLBlockLayout ValidateGGMLBlockStorage(const Tensor<CPU>& storage, const QuantizationParams& params,
		                                         std::string_view name)
		{
			if (params.scheme != QuantizationScheme::Block || !IsFloatingDataType(params.expressedType))
			{
				throw std::runtime_error(std::format(
				    "GGUF tensor '{}' must use block quantization with a floating-point expressed type", name));
			}
			const auto ggmlType = TryMapGGMLQuantizedType(params.blockFormat);
			if (!ggmlType)
			{
				throw std::runtime_error(std::format("GGUF tensor '{}' uses unsupported block quantization format {}",
				                                     name, QuantizedBlockFormatName(params.blockFormat)));
			}
			const auto* traits = ggml_get_type_traits(*ggmlType);
			if (!traits || !traits->to_float || params.expressedShape.empty())
			{
				throw std::runtime_error(
				    std::format("GGUF tensor '{}' does not expose a usable row-wise ggml block codec", name));
			}

			const auto rowSize = params.expressedShape.back();
			const auto totalElements = ShapeView{ params.expressedShape }.NumElements();
			if (rowSize == 0 || totalElements % rowSize != 0 ||
			    rowSize % static_cast<std::size_t>(traits->blck_size) != 0)
			{
				throw std::runtime_error(
				    std::format("GGUF tensor '{}' expressed shape is incompatible with the {} block size", name,
				                QuantizedBlockFormatName(params.blockFormat)));
			}
			const auto rowCount = totalElements / rowSize;
			const auto rowBytes = ggml_row_size(*ggmlType, static_cast<std::int64_t>(rowSize));
			if (storage.DType() != DataType::UInt8 || storage.NumElements() != rowCount * rowBytes)
			{
				throw std::runtime_error(std::format(
				    "GGUF tensor '{}' payload byte count or storage dtype does not match its block layout", name));
			}
			return { *ggmlType, traits, rowCount, rowSize, rowBytes };
		}

		GGMLBlockLayout ValidateGGMLBlockVariable(const Variable& variable, std::string_view name)
		{
			if (!variable.IsQuantized())
			{
				throw std::runtime_error(std::format("GGUF tensor '{}' is not quantized", name));
			}
			return ValidateGGMLBlockStorage(variable.Data().CopyToDevice(CPU{}), *variable.Quantization(), name);
		}
	} // namespace

	Tensor<CPU> DequantizeGGMLBlockVariable(const Variable& variable, std::string_view name)
	{
		const auto layout = ValidateGGMLBlockVariable(variable, name);
		const auto& params = *variable.Quantization();
		const auto storage = variable.Data().CopyToDevice(CPU{});
		Tensor<CPU> dequantizedF32(Uninitialized, params.expressedShape, DataType::Float32);
		const auto* src = static_cast<const std::uint8_t*>(storage.UnsafeRawData());
		auto* dst = static_cast<float*>(dequantizedF32.UnsafeRawData());
		for (std::size_t row = 0; row < layout.rowCount; ++row)
		{
			layout.traits->to_float(src + row * layout.rowBytes, dst + row * layout.rowSize,
			                        static_cast<std::int64_t>(layout.rowSize));
		}

		if (params.expressedType == DataType::Float32)
		{
			return dequantizedF32;
		}
		CPU cpu;
		Tensor<CPU> converted(Uninitialized, params.expressedShape, params.expressedType, cpu);
		DeviceTraits<CPU>::ConvertTo(cpu, DataType::Float32, dequantizedF32.UnsafeRawData(),
		                             dequantizedF32.NumElements(), params.expressedType, converted.UnsafeRawData());
		return converted;
	}

	Tensor<CPU> EvalGGMLExactDequantizedMatMul(const Tensor<CPU>& input, const Variable& weight, bool transposeWeight)
	{
		const auto& params = *weight.Quantization();
		const auto storage = weight.Data().CopyToDevice(CPU{});
		return EvalGGMLExactDequantizedMatMul(input, storage, params, transposeWeight);
	}

	Tensor<CPU> EvalGGMLExactDequantizedMatMul(const Tensor<CPU>& input, const Tensor<CPU>& weightStorage,
	                                           const QuantizationParams& params, bool transposeWeight)
	{
		const auto layout = ValidateGGMLBlockStorage(weightStorage, params, "exact-dequantized MatMul weight");
		if (input.DType() != DataType::Float32 || input.Shape().NumDim() != 2 ||
		    params.expressedType != DataType::Float32 || params.expressedShape.size() != 2)
		{
			throw std::runtime_error(
			    "GGML exact-dequantized MatMul currently requires 2D Float32 input and weight expressed types");
		}
		if (!transposeWeight)
		{
			throw std::runtime_error(
			    "GGML exact-dequantized MatMul requires output-major [outFeatures, inFeatures] weight storage");
		}

		const auto batch = input.Shape()[0];
		const auto inFeatures = input.Shape()[1];
		const auto outFeatures = params.expressedShape[0];
		if (params.expressedShape[1] != inFeatures || layout.rowCount != outFeatures || layout.rowSize != inFeatures)
		{
			throw std::runtime_error("GGML exact-dequantized MatMul input and weight shapes are incompatible");
		}

		const auto* weightBytes = static_cast<const std::uint8_t*>(weightStorage.UnsafeRawData());
		const auto* inputValues = static_cast<const float*>(input.UnsafeRawData());
		Tensor<CPU> result(Uninitialized, { batch, outFeatures }, DataType::Float32);
		auto* output = static_cast<float*>(result.UnsafeRawData());
		std::vector<float> dequantizedRow(inFeatures);
		for (std::size_t column = 0; column < outFeatures; ++column)
		{
			layout.traits->to_float(weightBytes + column * layout.rowBytes, dequantizedRow.data(),
			                        static_cast<std::int64_t>(inFeatures));
			for (std::size_t row = 0; row < batch; ++row)
			{
				double sum = 0.0;
				for (std::size_t feature = 0; feature < inFeatures; ++feature)
				{
					sum += static_cast<double>(inputValues[row * inFeatures + feature]) *
					       static_cast<double>(dequantizedRow[feature]);
				}
				output[row * outFeatures + column] = static_cast<float>(sum);
			}
		}
		return result;
	}

	Tensor<CPU> EvalGGMLQuantizedMatMul(const Tensor<CPU>& input, const Variable& weight, bool transposeWeight)
	{
		const auto& params = *weight.Quantization();
		const auto storage = weight.Data().CopyToDevice(CPU{});
		return EvalGGMLQuantizedMatMul(input, storage, params, transposeWeight);
	}

	Tensor<CPU> EvalGGMLQuantizedMatMul(const Tensor<CPU>& input, const Tensor<CPU>& weightStorage,
	                                    const QuantizationParams& params, bool transposeWeight)
	{
		const auto layout = ValidateGGMLBlockStorage(weightStorage, params, "quantized MatMul weight");
		if (input.DType() != DataType::Float32 || input.Shape().NumDim() != 2 ||
		    params.expressedType != DataType::Float32 || params.expressedShape.size() != 2)
		{
			throw std::runtime_error(
			    "GGML quantized MatMul currently requires 2D Float32 input and weight expressed types");
		}
		if (!transposeWeight)
		{
			throw std::runtime_error(
			    "GGML direct quantized MatMul requires output-major [outFeatures, inFeatures] weight storage");
		}

		const auto batch = input.Shape()[0];
		const auto inFeatures = input.Shape()[1];
		const auto outFeatures = params.expressedShape[0];
		if (params.expressedShape[1] != inFeatures || layout.rowCount != outFeatures || layout.rowSize != inFeatures)
		{
			throw std::runtime_error("GGML quantized MatMul input and weight shapes are incompatible");
		}

		static std::once_flag cpuInitialization;
		std::call_once(cpuInitialization, [] { ggml_cpu_init(); });
		const auto* cpuTraits = ggml_get_type_traits_cpu(layout.type);
		if (!cpuTraits || !cpuTraits->vec_dot)
		{
			throw std::runtime_error(std::format("GGML format {} does not expose a CPU vec_dot kernel",
			                                     QuantizedBlockFormatName(params.blockFormat)));
		}
		const auto* inputTraits = ggml_get_type_traits_cpu(cpuTraits->vec_dot_type);
		if (!inputTraits || !inputTraits->from_float)
		{
			throw std::runtime_error("GGML quantized MatMul input quantizer is unavailable");
		}

		const auto inputRowBytes = ggml_row_size(cpuTraits->vec_dot_type, static_cast<std::int64_t>(inFeatures));
		std::vector<std::byte> quantizedInput(inputRowBytes);
		const auto* weightBytes = static_cast<const std::byte*>(weightStorage.UnsafeRawData());
		const auto* inputValues = static_cast<const float*>(input.UnsafeRawData());
		Tensor<CPU> result(Uninitialized, { batch, outFeatures }, DataType::Float32);
		auto* output = static_cast<float*>(result.UnsafeRawData());

		for (std::size_t row = 0; row < batch; ++row)
		{
			inputTraits->from_float(inputValues + row * inFeatures, quantizedInput.data(),
			                        static_cast<std::int64_t>(inFeatures));
			for (std::size_t column = 0; column < outFeatures; ++column)
			{
				cpuTraits->vec_dot(static_cast<int>(inFeatures), output + row * outFeatures + column, 0,
				                   weightBytes + column * layout.rowBytes, 0, quantizedInput.data(), 0, 1);
			}
		}
		return result;
	}

	std::optional<Tensor<CPU>> TryEvalGGMLQuantizedMatMul(const Tensor<CPU>& input, const Tensor<CPU>& weightStorage,
	                                                      const QuantizationParams& params, bool transposeWeight)
	{
		if (params.scheme != QuantizationScheme::Block || !IsGGMLQuantizedBlockFormat(params.blockFormat))
		{
			return std::nullopt;
		}
		return EvalGGMLQuantizedMatMul(input, weightStorage, params, transposeWeight);
	}
} // namespace LiteNN::GGUF
