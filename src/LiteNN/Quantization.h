#include <LiteNN/Tensor.h>

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <limits>
#include <optional>
#include <stdexcept>
#include <string_view>
#include <utility>
#include <vector>

#ifndef LITENN_QUANTIZATION_H
#define LITENN_QUANTIZATION_H

namespace LiteNN
{
	enum class QuantizationScheme : std::uint32_t
	{
		Affine,
		Block,
	};

	enum class QuantizationGranularity : std::uint32_t
	{
		PerTensor,
		PerAxis,
		Grouped,
	};

	enum class QuantizedBlockFormat : std::uint32_t
	{
		Scalar,
		GGML_Q4_0,
		GGML_Q4_1,
		GGML_Q5_0,
		GGML_Q5_1,
		GGML_Q8_0,
		GGML_Q8_1,
		GGML_Q2_K,
		GGML_Q3_K,
		GGML_Q4_K,
		GGML_Q5_K,
		GGML_Q6_K,
		GGML_Q8_K,
		GGML_IQ2_XXS,
		GGML_IQ2_XS,
		GGML_IQ3_XXS,
		GGML_IQ1_S,
		GGML_IQ4_NL,
		GGML_IQ3_S,
		GGML_IQ2_S,
		GGML_IQ4_XS,
		GGML_I8,
		GGML_I16,
		GGML_I32,
		GGML_I64,
		GGML_F16,
		GGML_BF16,
		GGML_F32,
		GGML_F64,
	};

	struct QuantizationParams
	{
		QuantizationScheme scheme{ QuantizationScheme::Affine };
		QuantizationGranularity granularity{ QuantizationGranularity::PerTensor };
		QuantizedBlockFormat blockFormat{ QuantizedBlockFormat::Scalar };
		DataType storageType{ DataType::Int8 };
		DataType expressedType{ DataType::Float32 };
		std::int64_t axis{ -1 };
		std::size_t groupSize{};
		std::vector<float> scales;
		std::vector<std::int32_t> zeroPoints;
		std::vector<std::size_t> expressedShape;
	};

	struct QuantizedBlockLayout
	{
		std::size_t elementsPerBlock;
		std::size_t bytesPerBlock;
	};

	inline std::string_view QuantizationSchemeName(QuantizationScheme scheme)
	{
		switch (scheme)
		{
		case QuantizationScheme::Affine:
			return "Affine";
		case QuantizationScheme::Block:
			return "Block";
		}
		throw std::runtime_error("Invalid quantization scheme");
	}

	inline std::string_view QuantizationGranularityName(QuantizationGranularity granularity)
	{
		switch (granularity)
		{
		case QuantizationGranularity::PerTensor:
			return "PerTensor";
		case QuantizationGranularity::PerAxis:
			return "PerAxis";
		case QuantizationGranularity::Grouped:
			return "Grouped";
		}
		throw std::runtime_error("Invalid quantization granularity");
	}

	inline std::string_view QuantizedBlockFormatName(QuantizedBlockFormat format)
	{
		switch (format)
		{
		case QuantizedBlockFormat::Scalar:
			return "Scalar";
		case QuantizedBlockFormat::GGML_Q4_0:
			return "GGML_Q4_0";
		case QuantizedBlockFormat::GGML_Q4_1:
			return "GGML_Q4_1";
		case QuantizedBlockFormat::GGML_Q5_0:
			return "GGML_Q5_0";
		case QuantizedBlockFormat::GGML_Q5_1:
			return "GGML_Q5_1";
		case QuantizedBlockFormat::GGML_Q8_0:
			return "GGML_Q8_0";
		case QuantizedBlockFormat::GGML_Q8_1:
			return "GGML_Q8_1";
		case QuantizedBlockFormat::GGML_Q2_K:
			return "GGML_Q2_K";
		case QuantizedBlockFormat::GGML_Q3_K:
			return "GGML_Q3_K";
		case QuantizedBlockFormat::GGML_Q4_K:
			return "GGML_Q4_K";
		case QuantizedBlockFormat::GGML_Q5_K:
			return "GGML_Q5_K";
		case QuantizedBlockFormat::GGML_Q6_K:
			return "GGML_Q6_K";
		case QuantizedBlockFormat::GGML_Q8_K:
			return "GGML_Q8_K";
		case QuantizedBlockFormat::GGML_IQ2_XXS:
			return "GGML_IQ2_XXS";
		case QuantizedBlockFormat::GGML_IQ2_XS:
			return "GGML_IQ2_XS";
		case QuantizedBlockFormat::GGML_IQ3_XXS:
			return "GGML_IQ3_XXS";
		case QuantizedBlockFormat::GGML_IQ1_S:
			return "GGML_IQ1_S";
		case QuantizedBlockFormat::GGML_IQ4_NL:
			return "GGML_IQ4_NL";
		case QuantizedBlockFormat::GGML_IQ3_S:
			return "GGML_IQ3_S";
		case QuantizedBlockFormat::GGML_IQ2_S:
			return "GGML_IQ2_S";
		case QuantizedBlockFormat::GGML_IQ4_XS:
			return "GGML_IQ4_XS";
		case QuantizedBlockFormat::GGML_I8:
			return "GGML_I8";
		case QuantizedBlockFormat::GGML_I16:
			return "GGML_I16";
		case QuantizedBlockFormat::GGML_I32:
			return "GGML_I32";
		case QuantizedBlockFormat::GGML_I64:
			return "GGML_I64";
		case QuantizedBlockFormat::GGML_F16:
			return "GGML_F16";
		case QuantizedBlockFormat::GGML_BF16:
			return "GGML_BF16";
		case QuantizedBlockFormat::GGML_F32:
			return "GGML_F32";
		case QuantizedBlockFormat::GGML_F64:
			return "GGML_F64";
		}
		throw std::runtime_error("Invalid quantized block format");
	}

	inline bool IsScalarQuantizedBlockFormat(QuantizedBlockFormat format)
	{
		return format == QuantizedBlockFormat::Scalar;
	}

	inline bool IsGGMLQuantizedBlockFormat(QuantizedBlockFormat format)
	{
		return !IsScalarQuantizedBlockFormat(format);
	}

	inline std::optional<QuantizedBlockLayout> GetQuantizedBlockLayout(QuantizedBlockFormat format)
	{
		switch (format)
		{
		case QuantizedBlockFormat::GGML_Q4_0:
			return QuantizedBlockLayout{ 32, 18 };
		case QuantizedBlockFormat::GGML_Q4_1:
			return QuantizedBlockLayout{ 32, 20 };
		case QuantizedBlockFormat::GGML_Q5_0:
			return QuantizedBlockLayout{ 32, 22 };
		case QuantizedBlockFormat::GGML_Q5_1:
			return QuantizedBlockLayout{ 32, 24 };
		case QuantizedBlockFormat::GGML_Q8_0:
			return QuantizedBlockLayout{ 32, 34 };
		case QuantizedBlockFormat::GGML_Q8_1:
			return QuantizedBlockLayout{ 32, 40 };
		case QuantizedBlockFormat::GGML_Q2_K:
			return QuantizedBlockLayout{ 256, 84 };
		case QuantizedBlockFormat::GGML_Q3_K:
			return QuantizedBlockLayout{ 256, 110 };
		case QuantizedBlockFormat::GGML_Q4_K:
			return QuantizedBlockLayout{ 256, 144 };
		case QuantizedBlockFormat::GGML_Q5_K:
			return QuantizedBlockLayout{ 256, 176 };
		case QuantizedBlockFormat::GGML_Q6_K:
			return QuantizedBlockLayout{ 256, 210 };
		case QuantizedBlockFormat::GGML_Q8_K:
			return QuantizedBlockLayout{ 256, 292 };
		default:
			return std::nullopt;
		}
	}

	inline bool IsAffineQuantizedStorageType(DataType dtype)
	{
		return dtype == DataType::Int8 || dtype == DataType::UInt8;
	}

	namespace QuantizationDetail
	{
		inline std::size_t CeilDiv(std::size_t lhs, std::size_t rhs)
		{
			return (lhs + rhs - 1) / rhs;
		}

		inline std::size_t NormalizeAxis(std::int64_t axis, ShapeView shape)
		{
			const auto rank = static_cast<std::int64_t>(shape.NumDim());
			const auto normalized = axis < 0 ? axis + rank : axis;
			if (normalized < 0 || normalized >= rank)
			{
				throw std::runtime_error("Quantization axis is out of range");
			}
			return static_cast<std::size_t>(normalized);
		}

		inline std::size_t AxisStride(ShapeView shape, std::size_t axis)
		{
			std::size_t stride = 1;
			for (auto i = axis + 1; i < shape.NumDim(); ++i)
			{
				stride *= shape[i];
			}
			return stride;
		}

		inline std::size_t ExpectedScaleCount(const QuantizationParams& params, ShapeView shape)
		{
			switch (params.granularity)
			{
			case QuantizationGranularity::PerTensor:
				return 1;
			case QuantizationGranularity::PerAxis: {
				const auto axis = NormalizeAxis(params.axis, shape);
				return shape[axis];
			}
			case QuantizationGranularity::Grouped: {
				const auto axis = NormalizeAxis(params.axis, shape);
				if (params.groupSize == 0)
				{
					throw std::runtime_error("Grouped quantization requires groupSize > 0");
				}
				const auto groupsPerLine = CeilDiv(shape[axis], params.groupSize);
				const auto lineCount = shape.NumElements() / shape[axis];
				return lineCount * groupsPerLine;
			}
			}
			throw std::runtime_error("Invalid quantization granularity");
		}

		inline std::size_t ScaleIndexForElement(const QuantizationParams& params, ShapeView shape,
		                                        std::size_t elementIndex)
		{
			switch (params.granularity)
			{
			case QuantizationGranularity::PerTensor:
				return 0;
			case QuantizationGranularity::PerAxis: {
				const auto axis = NormalizeAxis(params.axis, shape);
				const auto stride = AxisStride(shape, axis);
				return (elementIndex / stride) % shape[axis];
			}
			case QuantizationGranularity::Grouped: {
				const auto axis = NormalizeAxis(params.axis, shape);
				const auto stride = AxisStride(shape, axis);
				const auto axisDim = shape[axis];
				const auto groupsPerLine = CeilDiv(axisDim, params.groupSize);
				const auto axisCoord = (elementIndex / stride) % axisDim;
				const auto outer = elementIndex / (axisDim * stride);
				const auto inner = elementIndex % stride;
				const auto line = outer * stride + inner;
				return line * groupsPerLine + axisCoord / params.groupSize;
			}
			}
			throw std::runtime_error("Invalid quantization granularity");
		}

		inline std::int32_t ZeroPointAt(const QuantizationParams& params, std::size_t scaleIndex)
		{
			if (params.zeroPoints.empty())
			{
				return 0;
			}
			return params.zeroPoints[scaleIndex];
		}

		template <typename T>
		constexpr std::int32_t StorageMin()
		{
			return static_cast<std::int32_t>(std::numeric_limits<T>::min());
		}

		template <typename T>
		constexpr std::int32_t StorageMax()
		{
			return static_cast<std::int32_t>(std::numeric_limits<T>::max());
		}

		inline Tensor<CPU> CopyToFloat32(const Tensor<CPU>& tensor)
		{
			if (tensor.DType() == DataType::Float32)
			{
				return tensor;
			}
			Tensor<CPU> converted(Uninitialized, tensor.Shape(), DataType::Float32);
			CPU cpu;
			DeviceTraits<CPU>::ConvertTo(cpu, tensor.DType(), tensor.RawData(), tensor.NumElements(), DataType::Float32,
			                             converted.RawData());
			return converted;
		}
	} // namespace QuantizationDetail

	inline void ValidateQuantizationParams(const QuantizationParams& params, ShapeView storageShape,
	                                       DataType actualStorageType)
	{
		if (!IsFloatingDataType(params.expressedType))
		{
			throw std::runtime_error("Quantization expressed type must be floating-point");
		}
		for (const auto dim : params.expressedShape)
		{
			if (dim == 0)
			{
				throw std::runtime_error("Quantization expressed shape contains a zero dimension");
			}
		}

		switch (params.scheme)
		{
		case QuantizationScheme::Affine: {
			if (params.blockFormat != QuantizedBlockFormat::Scalar)
			{
				throw std::runtime_error("Affine quantization requires scalar storage format");
			}
			if (params.storageType != actualStorageType)
			{
				throw std::runtime_error("Quantization storage type does not match tensor dtype");
			}
			if (!IsAffineQuantizedStorageType(params.storageType))
			{
				throw std::runtime_error("Affine quantization currently requires Int8 or UInt8 storage");
			}
			if (!params.expressedShape.empty())
			{
				if (params.expressedShape.size() != storageShape.NumDim())
				{
					throw std::runtime_error("Affine quantization expressed shape must match storage rank");
				}
				for (std::size_t i = 0; i < params.expressedShape.size(); ++i)
				{
					if (params.expressedShape[i] != storageShape[i])
					{
						throw std::runtime_error("Affine quantization expressed shape must match storage shape");
					}
				}
			}
			if (storageShape.NumDim() == 0 && params.granularity != QuantizationGranularity::PerTensor)
			{
				throw std::runtime_error("Scalar quantized tensors only support per-tensor parameters");
			}

			const auto expectedScales = QuantizationDetail::ExpectedScaleCount(params, storageShape);
			if (params.scales.size() != expectedScales)
			{
				throw std::runtime_error("Quantization scale count does not match tensor shape/granularity");
			}
			if (!params.zeroPoints.empty() && params.zeroPoints.size() != params.scales.size())
			{
				throw std::runtime_error("Quantization zero-point count must be zero or equal to scale count");
			}
			for (const auto scale : params.scales)
			{
				if (!(std::isfinite(scale) && scale > 0.0F))
				{
					throw std::runtime_error("Quantization scales must be finite and greater than zero");
				}
			}
			break;
		}
		case QuantizationScheme::Block:
			if (!IsGGMLQuantizedBlockFormat(params.blockFormat))
			{
				throw std::runtime_error("Block quantization requires a non-scalar block format");
			}
			if (params.storageType != DataType::UInt8 || actualStorageType != DataType::UInt8)
			{
				throw std::runtime_error("Block quantization stores raw payload bytes as UInt8");
			}
			if (params.granularity != QuantizationGranularity::PerTensor)
			{
				throw std::runtime_error("Block quantization metadata currently uses per-tensor granularity");
			}
			if (params.expressedShape.empty())
			{
				throw std::runtime_error("Block quantization requires expressedShape");
			}
			if (!params.scales.empty() || !params.zeroPoints.empty())
			{
				throw std::runtime_error("Block quantization does not use affine scales or zero-points");
			}
			break;
		}
	}

	inline QuantizationParams PerTensorAffineQuantization(DataType storageType, float scale,
	                                                      std::int32_t zeroPoint = 0,
	                                                      DataType expressedType = DataType::Float32)
	{
		return {
			.storageType = storageType,
			.expressedType = expressedType,
			.scales = { scale },
			.zeroPoints = { zeroPoint },
		};
	}

	inline QuantizationParams BlockQuantization(QuantizedBlockFormat format, std::vector<std::size_t> expressedShape,
	                                            DataType expressedType = DataType::Float32)
	{
		return {
			.scheme = QuantizationScheme::Block,
			.blockFormat = format,
			.storageType = DataType::UInt8,
			.expressedType = expressedType,
			.expressedShape = std::move(expressedShape),
		};
	}

	inline QuantizationParams PerAxisAffineQuantization(DataType storageType, std::int64_t axis,
	                                                    std::vector<float> scales,
	                                                    std::vector<std::int32_t> zeroPoints = {},
	                                                    DataType expressedType = DataType::Float32)
	{
		return {
			.granularity = QuantizationGranularity::PerAxis,
			.storageType = storageType,
			.expressedType = expressedType,
			.axis = axis,
			.scales = std::move(scales),
			.zeroPoints = std::move(zeroPoints),
		};
	}

	inline QuantizationParams GroupedAffineQuantization(DataType storageType, std::int64_t axis,
	                                                    std::size_t groupSize, std::vector<float> scales,
	                                                    std::vector<std::int32_t> zeroPoints = {},
	                                                    DataType expressedType = DataType::Float32)
	{
		return {
			.granularity = QuantizationGranularity::Grouped,
			.storageType = storageType,
			.expressedType = expressedType,
			.axis = axis,
			.groupSize = groupSize,
			.scales = std::move(scales),
			.zeroPoints = std::move(zeroPoints),
		};
	}

	template <Device D>
	class QuantizedTensor
	{
	public:
		QuantizedTensor(Tensor<D> storage, QuantizationParams params)
		    : storage_(std::move(storage)), params_(std::move(params))
		{
			ValidateQuantizationParams(params_, storage_.Shape(), storage_.DType());
		}

		auto& Storage(this auto&& self)
		{
			return self.storage_;
		}

		const QuantizationParams& Params() const
		{
			return params_;
		}

	private:
		Tensor<D> storage_;
		QuantizationParams params_;
	};

	inline QuantizedTensor<CPU> QuantizeAffine(const Tensor<CPU>& source, QuantizationParams params)
	{
		if (!IsFloatingDataType(source.DType()))
		{
			throw std::runtime_error("QuantizeAffine requires floating-point source tensor");
		}
		Tensor<CPU> storage(Uninitialized, source.Shape(), params.storageType);
		ValidateQuantizationParams(params, storage.Shape(), storage.DType());
		const auto sourceF32 = QuantizationDetail::CopyToFloat32(source);
		const auto* src = static_cast<const float*>(sourceF32.RawData());

		EnumDispatch(params.storageType, [&]<DataType StorageTypeValue> {
			if constexpr (StorageTypeValue == DataType::Int8 || StorageTypeValue == DataType::UInt8)
			{
				using StorageT = typename DeviceTraits<CPU>::template DataTypeMapping<StorageTypeValue>;
				auto* dst = static_cast<StorageT*>(storage.RawData());
				const auto minValue = QuantizationDetail::StorageMin<StorageT>();
				const auto maxValue = QuantizationDetail::StorageMax<StorageT>();
				for (std::size_t i = 0; i < source.NumElements(); ++i)
				{
					const auto scaleIndex = QuantizationDetail::ScaleIndexForElement(params, source.Shape(), i);
					const auto q = static_cast<std::int32_t>(
					                   std::lround(src[i] / params.scales[scaleIndex])) +
					               QuantizationDetail::ZeroPointAt(params, scaleIndex);
					dst[i] = static_cast<StorageT>(std::clamp(q, minValue, maxValue));
				}
			}
		});

		return QuantizedTensor<CPU>(std::move(storage), std::move(params));
	}

	inline Tensor<CPU> DequantizeAffine(const Tensor<CPU>& storage, const QuantizationParams& params,
	                                    DataType targetType = DataType::Float32)
	{
		if (!IsFloatingDataType(targetType))
		{
			throw std::runtime_error("DequantizeAffine target type must be floating-point");
		}
		ValidateQuantizationParams(params, storage.Shape(), storage.DType());

		Tensor<CPU> result(Uninitialized, storage.Shape(), targetType);
		EnumDispatch(storage.DType(), [&]<DataType StorageTypeValue> {
			if constexpr (StorageTypeValue == DataType::Int8 || StorageTypeValue == DataType::UInt8)
			{
				using StorageT = typename DeviceTraits<CPU>::template DataTypeMapping<StorageTypeValue>;
				const auto* src = static_cast<const StorageT*>(storage.RawData());
				EnumDispatch(targetType, [&]<DataType TargetTypeValue> {
					if constexpr (IsFloatingDataType(TargetTypeValue))
					{
						using TargetT = typename DeviceTraits<CPU>::template DataTypeMapping<TargetTypeValue>;
						auto* dst = static_cast<TargetT*>(result.RawData());
						for (std::size_t i = 0; i < storage.NumElements(); ++i)
						{
							const auto scaleIndex = QuantizationDetail::ScaleIndexForElement(params, storage.Shape(), i);
							const auto value =
							    (static_cast<std::int32_t>(src[i]) -
							     QuantizationDetail::ZeroPointAt(params, scaleIndex)) *
							    params.scales[scaleIndex];
							dst[i] = static_cast<TargetT>(value);
						}
					}
				});
			}
		});
		return result;
	}

	inline Tensor<CPU> DequantizeAffine(const QuantizedTensor<CPU>& tensor,
	                                    DataType targetType = DataType::Float32)
	{
		return DequantizeAffine(tensor.Storage(), tensor.Params(), targetType);
	}
} // namespace LiteNN

#endif
