#include <LiteNN/Tensor.h>

#include <algorithm>
#include <array>
#include <bit>
#include <cmath>
#include <cstdint>
#include <limits>
#include <optional>
#include <span>
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
		PackedNibble,
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

	enum class PackedNibbleFormat : std::uint32_t
	{
		None,
		Int4,
		UInt4,
		FP4E2M1,
		FP4E3M0,
	};

	enum class PackedNibbleOrder : std::uint32_t
	{
		LowThenHigh,
		HighThenLow,
	};

	enum class BlockScaleLayout : std::uint32_t
	{
		None,
		PerBlockFloat16,
		PerBlockBFloat16,
		PerBlockFloat32,
	};

	struct QuantizationParams
	{
		QuantizationScheme scheme{ QuantizationScheme::Affine };
		QuantizationGranularity granularity{ QuantizationGranularity::PerTensor };
		QuantizedBlockFormat blockFormat{ QuantizedBlockFormat::Scalar };
		PackedNibbleFormat packedFormat{ PackedNibbleFormat::None };
		PackedNibbleOrder packedOrder{ PackedNibbleOrder::LowThenHigh };
		BlockScaleLayout blockScaleLayout{ BlockScaleLayout::None };
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

	struct PackedNibbleLayout
	{
		PackedNibbleFormat format{ PackedNibbleFormat::None };
		PackedNibbleOrder order{ PackedNibbleOrder::LowThenHigh };
		std::size_t valuesPerByte{ 2 };
	};

	inline std::string_view QuantizationSchemeName(QuantizationScheme scheme)
	{
		return EnumToString<EnumToStringStyle::Unqualified>(scheme);
	}

	inline std::string_view QuantizationGranularityName(QuantizationGranularity granularity)
	{
		return EnumToString<EnumToStringStyle::Unqualified>(granularity);
	}

	inline std::string_view QuantizedBlockFormatName(QuantizedBlockFormat format)
	{
		return EnumToString<EnumToStringStyle::Unqualified>(format);
	}

	inline std::string_view PackedNibbleFormatName(PackedNibbleFormat format)
	{
		return EnumToString<EnumToStringStyle::Unqualified>(format);
	}

	inline std::string_view PackedNibbleOrderName(PackedNibbleOrder order)
	{
		return EnumToString<EnumToStringStyle::Unqualified>(order);
	}

	inline std::string_view BlockScaleLayoutName(BlockScaleLayout layout)
	{
		return EnumToString<EnumToStringStyle::Unqualified>(layout);
	}

	inline bool IsScalarQuantizedBlockFormat(QuantizedBlockFormat format)
	{
		return format == QuantizedBlockFormat::Scalar;
	}

	inline bool IsGGMLQuantizedBlockFormat(QuantizedBlockFormat format)
	{
		return format != QuantizedBlockFormat::Scalar && format != QuantizedBlockFormat::PackedNibble;
	}

	inline bool IsPackedNibbleQuantizedBlockFormat(QuantizedBlockFormat format)
	{
		return format == QuantizedBlockFormat::PackedNibble;
	}

	inline std::optional<QuantizedBlockLayout> GetQuantizedBlockLayout(QuantizedBlockFormat format)
	{
		switch (format)
		{
		case QuantizedBlockFormat::PackedNibble:
			return QuantizedBlockLayout{ 2, 1 };
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

	inline bool IsPackedNibbleFormat(PackedNibbleFormat format)
	{
		return format != PackedNibbleFormat::None;
	}

	inline bool IsIntegerPackedNibbleFormat(PackedNibbleFormat format)
	{
		return format == PackedNibbleFormat::Int4 || format == PackedNibbleFormat::UInt4;
	}

	inline bool IsFloatPackedNibbleFormat(PackedNibbleFormat format)
	{
		return format == PackedNibbleFormat::FP4E2M1 || format == PackedNibbleFormat::FP4E3M0;
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
			DeviceTraits<CPU>::ConvertTo(cpu, tensor.DType(), tensor.UnsafeRawData(), tensor.NumElements(),
			                             DataType::Float32, converted.UnsafeRawData());
			return converted;
		}

		inline std::span<const float> FP4PositiveValues(PackedNibbleFormat format)
		{
			static constexpr std::array kE2M1 = { 0.0F, 0.5F, 1.0F, 1.5F, 2.0F, 3.0F, 4.0F, 6.0F };
			static constexpr std::array kE3M0 = { 0.0F, 0.25F, 0.5F, 1.0F, 2.0F, 4.0F, 8.0F, 16.0F };
			switch (format)
			{
			case PackedNibbleFormat::FP4E2M1:
				return kE2M1;
			case PackedNibbleFormat::FP4E3M0:
				return kE3M0;
			default:
				throw std::runtime_error("Unsupported FP4 packed format");
			}
		}

		inline std::uint8_t Float32ToFP4Bits(float value, PackedNibbleFormat format)
		{
			const auto sign = std::signbit(value) ? 0x08U : 0U;
			auto magnitude = std::fabs(value);
			const auto values = FP4PositiveValues(format);
			if (!std::isfinite(magnitude))
			{
				return static_cast<std::uint8_t>(sign | 0x07U);
			}
			std::size_t best = 0;
			auto bestDistance = std::numeric_limits<float>::infinity();
			for (std::size_t i = 0; i < values.size(); ++i)
			{
				const auto distance = std::fabs(magnitude - values[i]);
				if (distance < bestDistance || (distance == bestDistance && values[i] > values[best]))
				{
					best = i;
					bestDistance = distance;
				}
			}
			return static_cast<std::uint8_t>(sign | best);
		}

		inline float FP4BitsToFloat32(std::uint8_t bits, PackedNibbleFormat format)
		{
			const auto sign = (bits & 0x08U) != 0 ? -1.0F : 1.0F;
			const auto values = FP4PositiveValues(format);
			return sign * values[bits & 0x07U];
		}

		inline float DecodePackedNibble(std::uint8_t nibble, const QuantizationParams& params)
		{
			const auto scale = params.scales.empty() ? 1.0F : params.scales[0];
			const auto zeroPoint = params.zeroPoints.empty() ? 0 : params.zeroPoints[0];
			if (IsIntegerPackedNibbleFormat(params.packedFormat))
			{
				std::int32_t value = nibble;
				if (params.packedFormat == PackedNibbleFormat::Int4 && (nibble & 0x08U) != 0)
				{
					value -= 16;
				}
				return (static_cast<float>(value) - static_cast<float>(zeroPoint)) * scale;
			}
			if (IsFloatPackedNibbleFormat(params.packedFormat))
			{
				return FP4BitsToFloat32(nibble, params.packedFormat) * scale;
			}
			throw std::runtime_error("Unsupported packed nibble format");
		}

		inline float DecodePackedNibbleElement(const Tensor<CPU>& storage, const QuantizationParams& params,
		                                       std::size_t elementIndex)
		{
			const auto byteIndex = elementIndex / 2;
			if (byteIndex >= storage.NumElements())
			{
				throw std::runtime_error("Packed nibble element index is out of storage range");
			}
			const auto byte = static_cast<const std::uint8_t*>(storage.UnsafeRawData())[byteIndex];
			const auto low = static_cast<std::uint8_t>(byte & 0x0f);
			const auto high = static_cast<std::uint8_t>((byte >> 4) & 0x0f);
			const auto even = (elementIndex % 2) == 0;
			const auto nibble =
			    params.packedOrder == PackedNibbleOrder::LowThenHigh ? (even ? low : high) : (even ? high : low);
			return DecodePackedNibble(nibble, params);
		}

		inline float Float16BitsToFloat32(std::uint16_t bits)
		{
			const auto sign = (bits & 0x8000U) != 0 ? -1.0F : 1.0F;
			const auto exponent = static_cast<int>((bits >> 10U) & 0x1fU);
			const auto mantissa = static_cast<int>(bits & 0x03ffU);
			if (exponent == 0)
			{
				return mantissa == 0 ? sign * 0.0F : sign * std::ldexp(static_cast<float>(mantissa), -24);
			}
			if (exponent == 31)
			{
				return mantissa == 0 ? sign * std::numeric_limits<float>::infinity()
				                     : std::numeric_limits<float>::quiet_NaN();
			}
			return sign * std::ldexp(1.0F + static_cast<float>(mantissa) / 1024.0F, exponent - 15);
		}

		inline float ReadGGMLF16(const std::uint8_t* bytes)
		{
			const auto bits = static_cast<std::uint16_t>(bytes[0]) |
			                  static_cast<std::uint16_t>(static_cast<std::uint16_t>(bytes[1]) << 8U);
			return Float16BitsToFloat32(bits);
		}

		inline float DecodeGGMLQ4Or5KElement(const std::uint8_t* block, QuantizedBlockFormat format, std::size_t lane)
		{
			const auto d = ReadGGMLF16(block);
			const auto dmin = ReadGGMLF16(block + 2);
			const auto subblock = lane / 32;
			const auto belowFour = subblock < 4;
			const auto* scales = block + 4;
			const auto scaleLowOffset = belowFour ? subblock : subblock + 4;
			const auto scaleLow = static_cast<std::uint32_t>(scales[scaleLowOffset]);
			const auto minLow = static_cast<std::uint32_t>(scales[subblock + 4]);
			std::uint32_t scale = scaleLow & 63U;
			std::uint32_t minimum = minLow & 63U;
			if (!belowFour)
			{
				const auto highOffset = subblock - 4;
				const auto highSource = static_cast<std::uint32_t>(scales[highOffset]);
				const auto minHighSource = static_cast<std::uint32_t>(scales[subblock]);
				scale = (scaleLow & 15U) | ((highSource >> 6U) << 4U);
				minimum = (minLow >> 4U) | ((minHighSource >> 6U) << 4U);
			}

			const auto quantOffset = (lane / 64) * 32 + (lane % 32);
			const auto quantBaseOffset = format == QuantizedBlockFormat::GGML_Q5_K ? 48 : 16;
			const auto quantByte = static_cast<std::uint32_t>(block[quantBaseOffset + quantOffset]);
			auto nibble = (lane % 64) >= 32 ? (quantByte >> 4U) & 15U : quantByte & 15U;
			if (format == QuantizedBlockFormat::GGML_Q5_K)
			{
				const auto highBits = static_cast<std::uint32_t>(block[16 + (lane % 32)]);
				const auto highBit = (highBits >> subblock) & 1U;
				nibble |= highBit << 4U;
			}
			return d * static_cast<float>(scale) * static_cast<float>(nibble) - dmin * static_cast<float>(minimum);
		}

		inline float DecodeGGMLQ6KElement(const std::uint8_t* block, std::size_t lane)
		{
			const auto d = ReadGGMLF16(block + 208);
			const auto halfBlock = lane / 128;
			const auto local = lane % 128;
			const auto segment = local / 32;
			const auto laneInSegment = local % 32;
			const auto oddSegment = segment % 2;
			const auto qlOffset = halfBlock * 64 + laneInSegment + oddSegment * 32;
			const auto ql = static_cast<std::uint32_t>(block[qlOffset]);
			const auto lowFour = segment >= 2 ? (ql >> 4U) & 15U : ql & 15U;
			const auto qhOffset = halfBlock * 32 + laneInSegment;
			const auto qh = static_cast<std::uint32_t>(block[128 + qhOffset]);
			const auto highTwo = (qh >> (segment * 2)) & 3U;
			const auto quant = static_cast<std::int32_t>(lowFour | (highTwo << 4U)) - 32;
			const auto scaleOffset = halfBlock * 8 + (laneInSegment / 16) + segment * 2;
			const auto scale = static_cast<std::int8_t>(block[192 + scaleOffset]);
			return d * static_cast<float>(scale) * static_cast<float>(quant);
		}

		inline float DecodeGGMLBlockElement(const std::uint8_t* block, QuantizedBlockFormat format, std::size_t lane)
		{
			switch (format)
			{
			case QuantizedBlockFormat::GGML_Q8_0:
				return ReadGGMLF16(block) * static_cast<float>(static_cast<std::int8_t>(block[2 + lane]));
			case QuantizedBlockFormat::GGML_Q4_K:
			case QuantizedBlockFormat::GGML_Q5_K:
				return DecodeGGMLQ4Or5KElement(block, format, lane);
			case QuantizedBlockFormat::GGML_Q6_K:
				return DecodeGGMLQ6KElement(block, lane);
			default:
				throw std::runtime_error("Unsupported GGML block format for reference dequantization");
			}
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
			if (!IsGGMLQuantizedBlockFormat(params.blockFormat) &&
			    !IsPackedNibbleQuantizedBlockFormat(params.blockFormat))
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
				if (!IsPackedNibbleQuantizedBlockFormat(params.blockFormat))
				{
					throw std::runtime_error("GGML block quantization does not use affine scales or zero-points");
				}
				if (params.scales.size() != 1)
				{
					throw std::runtime_error("Packed nibble quantization currently requires one scale");
				}
				if (!params.zeroPoints.empty() && params.zeroPoints.size() != params.scales.size())
				{
					throw std::runtime_error("Packed nibble zero-point count must be zero or equal to scale count");
				}
			}
			if (IsPackedNibbleQuantizedBlockFormat(params.blockFormat))
			{
				if (!IsPackedNibbleFormat(params.packedFormat))
				{
					throw std::runtime_error("Packed nibble quantization requires a packedFormat");
				}
				if (params.blockScaleLayout != BlockScaleLayout::None && !params.scales.empty())
				{
					throw std::runtime_error(
					    "Packed nibble quantization cannot use both inline scales and blockScaleLayout");
				}
				const auto expectedBytes =
				    QuantizationDetail::CeilDiv(ShapeView{ params.expressedShape }.NumElements(), std::size_t{ 2 });
				if (storageShape.NumElements() != expectedBytes)
				{
					throw std::runtime_error("Packed nibble storage byte count does not match expressed shape");
				}
			}
			else if (params.packedFormat != PackedNibbleFormat::None ||
			         params.blockScaleLayout != BlockScaleLayout::None)
			{
				throw std::runtime_error("Only packed nibble quantization may set packedFormat or blockScaleLayout");
			}
			break;
		}
	}

	inline QuantizationParams PerTensorAffineQuantization(DataType storageType, float scale, std::int32_t zeroPoint = 0,
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

	inline QuantizationParams PackedNibbleQuantization(PackedNibbleFormat format,
	                                                   std::vector<std::size_t> expressedShape, float scale = 1.0F,
	                                                   std::int32_t zeroPoint = 0,
	                                                   PackedNibbleOrder order = PackedNibbleOrder::LowThenHigh,
	                                                   DataType expressedType = DataType::Float32)
	{
		QuantizationParams params{
			.scheme = QuantizationScheme::Block,
			.blockFormat = QuantizedBlockFormat::PackedNibble,
			.packedFormat = format,
			.packedOrder = order,
			.storageType = DataType::UInt8,
			.expressedType = expressedType,
			.scales = { scale },
			.zeroPoints = { zeroPoint },
			.expressedShape = std::move(expressedShape),
		};
		return params;
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

	inline QuantizationParams GroupedAffineQuantization(DataType storageType, std::int64_t axis, std::size_t groupSize,
	                                                    std::vector<float> scales,
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
		const auto* src = static_cast<const float*>(sourceF32.UnsafeRawData());

		EnumDispatch(params.storageType, [&]<DataType StorageTypeValue> {
			if constexpr (StorageTypeValue == DataType::Int8 || StorageTypeValue == DataType::UInt8)
			{
				using StorageT = typename DeviceTraits<CPU>::template DataTypeMapping<StorageTypeValue>;
				auto* dst = static_cast<StorageT*>(storage.UnsafeRawData());
				const auto minValue = QuantizationDetail::StorageMin<StorageT>();
				const auto maxValue = QuantizationDetail::StorageMax<StorageT>();
				for (std::size_t i = 0; i < source.NumElements(); ++i)
				{
					const auto scaleIndex = QuantizationDetail::ScaleIndexForElement(params, source.Shape(), i);
					const auto q = static_cast<std::int32_t>(std::lround(src[i] / params.scales[scaleIndex])) +
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
				const auto* src = static_cast<const StorageT*>(storage.UnsafeRawData());
				EnumDispatch(targetType, [&]<DataType TargetTypeValue> {
					if constexpr (IsFloatingDataType(TargetTypeValue))
					{
						using TargetT = typename DeviceTraits<CPU>::template DataTypeMapping<TargetTypeValue>;
						auto* dst = static_cast<TargetT*>(result.UnsafeRawData());
						for (std::size_t i = 0; i < storage.NumElements(); ++i)
						{
							const auto scaleIndex =
							    QuantizationDetail::ScaleIndexForElement(params, storage.Shape(), i);
							const auto value = (static_cast<std::int32_t>(src[i]) -
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

	inline Tensor<CPU> DequantizeAffine(const QuantizedTensor<CPU>& tensor, DataType targetType = DataType::Float32)
	{
		return DequantizeAffine(tensor.Storage(), tensor.Params(), targetType);
	}

	inline Tensor<CPU> PackInteger4(const Tensor<CPU>& source, QuantizationParams params)
	{
		if (!IsIntegerPackedNibbleFormat(params.packedFormat))
		{
			throw std::runtime_error("PackInteger4 requires Int4 or UInt4 packed format");
		}
		if (source.DType() != DataType::Int8 && source.DType() != DataType::UInt8)
		{
			throw std::runtime_error("PackInteger4 requires Int8 or UInt8 source tensor");
		}
		params.scheme = QuantizationScheme::Block;
		params.blockFormat = QuantizedBlockFormat::PackedNibble;
		params.storageType = DataType::UInt8;
		if (params.expressedShape.empty())
		{
			params.expressedShape = source.Shape().ToOwned();
		}
		const std::vector<std::size_t> packedShape{ QuantizationDetail::CeilDiv(source.NumElements(),
			                                                                    std::size_t{ 2 }) };
		Tensor<CPU> storage(Uninitialized, packedShape, DataType::UInt8);
		ValidateQuantizationParams(params, storage.Shape(), storage.DType());

		const auto readNibble = [&](std::size_t index) -> std::uint8_t {
			if (source.DType() == DataType::Int8)
			{
				const auto value = static_cast<const std::int8_t*>(source.UnsafeRawData())[index];
				if (params.packedFormat == PackedNibbleFormat::Int4)
				{
					if (value < -8 || value > 7)
					{
						throw std::runtime_error("Int4 value is outside [-8, 7]");
					}
					return static_cast<std::uint8_t>(static_cast<std::int32_t>(value) & 0x0f);
				}
				if (value < 0 || value > 15)
				{
					throw std::runtime_error("UInt4 value is outside [0, 15]");
				}
				return static_cast<std::uint8_t>(value);
			}
			const auto value = static_cast<const std::uint8_t*>(source.UnsafeRawData())[index];
			if (params.packedFormat == PackedNibbleFormat::Int4)
			{
				if (value > 15)
				{
					throw std::runtime_error("Packed Int4 source byte must contain a sign-extended nibble");
				}
				return value;
			}
			if (value > 15)
			{
				throw std::runtime_error("UInt4 value is outside [0, 15]");
			}
			return value;
		};

		auto* dst = static_cast<std::uint8_t*>(storage.UnsafeRawData());
		for (std::size_t byte = 0; byte < storage.NumElements(); ++byte)
		{
			const auto first = readNibble(byte * 2) & 0x0f;
			const auto second = byte * 2 + 1 < source.NumElements() ? (readNibble(byte * 2 + 1) & 0x0f) : 0;
			dst[byte] = params.packedOrder == PackedNibbleOrder::LowThenHigh
			                ? static_cast<std::uint8_t>(first | (second << 4))
			                : static_cast<std::uint8_t>((first << 4) | second);
		}
		return storage;
	}

	inline Tensor<CPU> PackFloat4(const Tensor<CPU>& source, QuantizationParams params)
	{
		if (!IsFloatPackedNibbleFormat(params.packedFormat))
		{
			throw std::runtime_error("PackFloat4 requires FP4 packed format");
		}
		if (!IsFloatingDataType(source.DType()))
		{
			throw std::runtime_error("PackFloat4 requires floating-point source tensor");
		}
		params.scheme = QuantizationScheme::Block;
		params.blockFormat = QuantizedBlockFormat::PackedNibble;
		params.storageType = DataType::UInt8;
		if (params.expressedShape.empty())
		{
			params.expressedShape = source.Shape().ToOwned();
		}
		const std::vector<std::size_t> packedShape{ QuantizationDetail::CeilDiv(source.NumElements(),
			                                                                    std::size_t{ 2 }) };
		Tensor<CPU> storage(Uninitialized, packedShape, DataType::UInt8);
		ValidateQuantizationParams(params, storage.Shape(), storage.DType());

		const auto sourceF32 = QuantizationDetail::CopyToFloat32(source);
		const auto* src = static_cast<const float*>(sourceF32.UnsafeRawData());
		auto* dst = static_cast<std::uint8_t*>(storage.UnsafeRawData());
		const auto scale = params.scales.empty() ? 1.0F : params.scales[0];
		if (!(std::isfinite(scale) && scale > 0.0F))
		{
			throw std::runtime_error("PackFloat4 requires a finite positive scale");
		}
		for (std::size_t byte = 0; byte < storage.NumElements(); ++byte)
		{
			const auto first = QuantizationDetail::Float32ToFP4Bits(src[byte * 2] / scale, params.packedFormat);
			const auto second =
			    byte * 2 + 1 < source.NumElements()
			        ? QuantizationDetail::Float32ToFP4Bits(src[byte * 2 + 1] / scale, params.packedFormat)
			        : std::uint8_t{ 0 };
			dst[byte] = params.packedOrder == PackedNibbleOrder::LowThenHigh
			                ? static_cast<std::uint8_t>(first | (second << 4))
			                : static_cast<std::uint8_t>((first << 4) | second);
		}
		return storage;
	}

	inline Tensor<CPU> UnpackInteger4(const Tensor<CPU>& storage, const QuantizationParams& params)
	{
		ValidateQuantizationParams(params, storage.Shape(), storage.DType());
		if (!IsIntegerPackedNibbleFormat(params.packedFormat))
		{
			throw std::runtime_error("UnpackInteger4 requires Int4 or UInt4 packed format");
		}
		const std::vector<std::size_t> outputShape{ params.expressedShape };
		Tensor<CPU> output(Uninitialized, outputShape,
		                   params.packedFormat == PackedNibbleFormat::Int4 ? DataType::Int8 : DataType::UInt8);
		const auto* src = static_cast<const std::uint8_t*>(storage.UnsafeRawData());
		for (std::size_t byte = 0; byte < storage.NumElements(); ++byte)
		{
			const auto low = static_cast<std::uint8_t>(src[byte] & 0x0f);
			const auto high = static_cast<std::uint8_t>((src[byte] >> 4) & 0x0f);
			const auto first = params.packedOrder == PackedNibbleOrder::LowThenHigh ? low : high;
			const auto second = params.packedOrder == PackedNibbleOrder::LowThenHigh ? high : low;
			const auto write = [&](std::size_t index, std::uint8_t nibble) {
				if (index >= output.NumElements())
				{
					return;
				}
				if (params.packedFormat == PackedNibbleFormat::Int4)
				{
					const auto signedValue = (nibble & 0x08U) != 0
					                             ? static_cast<std::int8_t>(static_cast<int>(nibble) - 16)
					                             : static_cast<std::int8_t>(nibble);
					static_cast<std::int8_t*>(output.UnsafeRawData())[index] = signedValue;
				}
				else
				{
					static_cast<std::uint8_t*>(output.UnsafeRawData())[index] = nibble;
				}
			};
			write(byte * 2, first);
			write(byte * 2 + 1, second);
		}
		return output;
	}

	inline Tensor<CPU> DequantizePackedNibble(const Tensor<CPU>& storage, const QuantizationParams& params,
	                                          DataType targetType = DataType::Float32)
	{
		ValidateQuantizationParams(params, storage.Shape(), storage.DType());
		if (!IsFloatingDataType(targetType))
		{
			throw std::runtime_error("DequantizePackedNibble target type must be floating-point");
		}
		if (!IsPackedNibbleQuantizedBlockFormat(params.blockFormat))
		{
			throw std::runtime_error("DequantizePackedNibble requires packed nibble quantization");
		}
		const std::vector<std::size_t> outputShape{ params.expressedShape };
		Tensor<CPU> output(Uninitialized, outputShape, targetType);

		EnumDispatch(targetType, [&]<DataType TargetTypeValue> {
			if constexpr (IsFloatingDataType(TargetTypeValue))
			{
				using TargetT = typename DeviceTraits<CPU>::template DataTypeMapping<TargetTypeValue>;
				auto* dst = static_cast<TargetT*>(output.UnsafeRawData());
				for (std::size_t i = 0; i < output.NumElements(); ++i)
				{
					dst[i] = static_cast<TargetT>(QuantizationDetail::DecodePackedNibbleElement(storage, params, i));
				}
			}
		});
		return output;
	}

	inline Tensor<CPU> DequantizeGGMLBlock(const Tensor<CPU>& storage, const QuantizationParams& params,
	                                       DataType targetType = DataType::Float32)
	{
		ValidateQuantizationParams(params, storage.Shape(), storage.DType());
		if (!IsFloatingDataType(targetType))
		{
			throw std::runtime_error("DequantizeGGMLBlock target type must be floating-point");
		}
		if (!IsGGMLQuantizedBlockFormat(params.blockFormat))
		{
			throw std::runtime_error("DequantizeGGMLBlock requires GGML block quantization");
		}
		if (storage.DType() != DataType::UInt8 || params.storageType != DataType::UInt8)
		{
			throw std::runtime_error("DequantizeGGMLBlock requires UInt8 storage");
		}
		const auto layout = GetQuantizedBlockLayout(params.blockFormat);
		if (!layout || (params.blockFormat != QuantizedBlockFormat::GGML_Q8_0 &&
		                params.blockFormat != QuantizedBlockFormat::GGML_Q4_K &&
		                params.blockFormat != QuantizedBlockFormat::GGML_Q5_K &&
		                params.blockFormat != QuantizedBlockFormat::GGML_Q6_K))
		{
			throw std::runtime_error("DequantizeGGMLBlock currently supports GGML_Q4_K/Q5_K/Q6_K/Q8_0 only");
		}
		const auto total = ShapeView{ params.expressedShape }.NumElements();
		if (total == 0 || total % layout->elementsPerBlock != 0)
		{
			throw std::runtime_error("DequantizeGGMLBlock expressed shape is not aligned to the block size");
		}
		const auto blockCount = total / layout->elementsPerBlock;
		if (storage.NumElements() != blockCount * layout->bytesPerBlock)
		{
			throw std::runtime_error("DequantizeGGMLBlock storage byte count does not match expressed shape");
		}

		Tensor<CPU> output(Uninitialized, params.expressedShape, targetType);
		EnumDispatch(targetType, [&]<DataType TargetTypeValue> {
			if constexpr (IsFloatingDataType(TargetTypeValue))
			{
				using TargetT = typename DeviceTraits<CPU>::template DataTypeMapping<TargetTypeValue>;
				const auto* src = static_cast<const std::uint8_t*>(storage.UnsafeRawData());
				auto* dst = static_cast<TargetT*>(output.UnsafeRawData());
				for (std::size_t blockIndex = 0; blockIndex < blockCount; ++blockIndex)
				{
					const auto* block = src + blockIndex * layout->bytesPerBlock;
					for (std::size_t lane = 0; lane < layout->elementsPerBlock; ++lane)
					{
						dst[blockIndex * layout->elementsPerBlock + lane] = static_cast<TargetT>(
						    QuantizationDetail::DecodeGGMLBlockElement(block, params.blockFormat, lane));
					}
				}
			}
		});
		return output;
	}

	inline void DequantizeGGMLBlockRowToFloat32(const Tensor<CPU>& storage, const QuantizationParams& params,
	                                            std::size_t row, float* output)
	{
		ValidateQuantizationParams(params, storage.Shape(), storage.DType());
		if (params.scheme != QuantizationScheme::Block || !IsGGMLQuantizedBlockFormat(params.blockFormat) ||
		    storage.DType() != DataType::UInt8 || params.storageType != DataType::UInt8 ||
		    params.expressedShape.size() != 2)
		{
			throw std::runtime_error("DequantizeGGMLBlockRowToFloat32 requires 2D UInt8 GGML block storage");
		}
		const auto layout = GetQuantizedBlockLayout(params.blockFormat);
		if (!layout || (params.blockFormat != QuantizedBlockFormat::GGML_Q8_0 &&
		                params.blockFormat != QuantizedBlockFormat::GGML_Q4_K &&
		                params.blockFormat != QuantizedBlockFormat::GGML_Q5_K &&
		                params.blockFormat != QuantizedBlockFormat::GGML_Q6_K))
		{
			throw std::runtime_error(
			    "DequantizeGGMLBlockRowToFloat32 currently supports GGML_Q4_K/Q5_K/Q6_K/Q8_0 only");
		}
		const auto rowCount = params.expressedShape[0];
		const auto rowWidth = params.expressedShape[1];
		if (row >= rowCount || rowWidth == 0 || rowWidth % layout->elementsPerBlock != 0)
		{
			throw std::runtime_error("DequantizeGGMLBlockRowToFloat32 row or width is out of range");
		}
		const auto blocksPerRow = rowWidth / layout->elementsPerBlock;
		if (storage.NumElements() != rowCount * blocksPerRow * layout->bytesPerBlock)
		{
			throw std::runtime_error("DequantizeGGMLBlockRowToFloat32 storage byte count does not match row layout");
		}
		const auto* src = static_cast<const std::uint8_t*>(storage.UnsafeRawData());
		const auto rowBase = row * blocksPerRow * layout->bytesPerBlock;
		for (std::size_t blockIndex = 0; blockIndex < blocksPerRow; ++blockIndex)
		{
			const auto* block = src + rowBase + blockIndex * layout->bytesPerBlock;
			for (std::size_t lane = 0; lane < layout->elementsPerBlock; ++lane)
			{
				output[blockIndex * layout->elementsPerBlock + lane] =
				    QuantizationDetail::DecodeGGMLBlockElement(block, params.blockFormat, lane);
			}
		}
	}

	inline std::vector<std::size_t> QuantizedMatMulOutputShape(ShapeView lhsShape, const QuantizationParams& rhsParams,
	                                                           ShapeView rhsStorageShape)
	{
		if (lhsShape.NumDim() != 2)
		{
			throw std::runtime_error("Quantized MatMul currently requires rank-2 lhs");
		}
		const auto rhsShape = rhsParams.expressedShape.empty()
		                          ? rhsStorageShape.Dims
		                          : std::span<const std::size_t>{ rhsParams.expressedShape };
		if (rhsShape.size() != 2)
		{
			throw std::runtime_error("Quantized MatMul currently requires rank-2 quantized rhs expressed shape");
		}
		if (lhsShape[1] != rhsShape[0])
		{
			throw std::runtime_error("Quantized MatMul inner dimensions do not match");
		}
		return { lhsShape[0], rhsShape[1] };
	}

	inline float ReadAffineQuantizedElementAsFloat(const Tensor<CPU>& storage, const QuantizationParams& params,
	                                               std::size_t elementIndex)
	{
		if (elementIndex >= storage.NumElements())
		{
			throw std::runtime_error("Affine quantized element index is out of range");
		}
		const auto scaleIndex = QuantizationDetail::ScaleIndexForElement(params, storage.Shape(), elementIndex);
		const auto zeroPoint = QuantizationDetail::ZeroPointAt(params, scaleIndex);
		if (storage.DType() == DataType::Int8)
		{
			const auto value = static_cast<const std::int8_t*>(storage.UnsafeRawData())[elementIndex];
			return (static_cast<float>(value) - static_cast<float>(zeroPoint)) * params.scales[scaleIndex];
		}
		if (storage.DType() == DataType::UInt8)
		{
			const auto value = static_cast<const std::uint8_t*>(storage.UnsafeRawData())[elementIndex];
			return (static_cast<float>(value) - static_cast<float>(zeroPoint)) * params.scales[scaleIndex];
		}
		throw std::runtime_error("Affine quantized MatMul currently supports Int8 and UInt8 storage");
	}

	inline float ReadQuantizedElementAsFloat(const Tensor<CPU>& storage, const QuantizationParams& params,
	                                         std::size_t elementIndex)
	{
		if (params.scheme == QuantizationScheme::Affine)
		{
			return ReadAffineQuantizedElementAsFloat(storage, params, elementIndex);
		}
		if (params.scheme == QuantizationScheme::Block && IsPackedNibbleQuantizedBlockFormat(params.blockFormat))
		{
			const auto expressedElements = ShapeView{ params.expressedShape }.NumElements();
			if (elementIndex >= expressedElements)
			{
				throw std::runtime_error("Packed quantized element index is out of range");
			}
			return QuantizationDetail::DecodePackedNibbleElement(storage, params, elementIndex);
		}
		throw std::runtime_error("Native quantized MatMul currently supports affine and packed-nibble weights only");
	}

	struct PreparedQuantizedLinearWeight
	{
		Tensor<CPU> dequantizedWeight;
		std::size_t inputWidth{};
		std::size_t outputWidth{};
		QuantizationParams sourceParams;
		DataType sourceStorageType{ DataType::Float32 };
	};

	inline PreparedQuantizedLinearWeight PrepareQuantizedLinearWeight(const Tensor<CPU>& weightStorage,
	                                                                  const QuantizationParams& weightParams)
	{
		ValidateQuantizationParams(weightParams, weightStorage.Shape(), weightStorage.DType());
		const auto storageShape = weightStorage.Shape();
		const auto expressedShape = weightParams.expressedShape.empty()
		                                ? std::span<const std::size_t>{ storageShape.Dims }
		                                : std::span<const std::size_t>{ weightParams.expressedShape };
		if (expressedShape.size() != 2)
		{
			throw std::runtime_error("Prepared quantized Linear weight requires rank-2 expressed shape");
		}

		if (weightParams.scheme == QuantizationScheme::Affine)
		{
			return {
				.dequantizedWeight = DequantizeAffine(weightStorage, weightParams, DataType::Float32),
				.inputWidth = expressedShape[0],
				.outputWidth = expressedShape[1],
				.sourceParams = weightParams,
				.sourceStorageType = weightStorage.DType(),
			};
		}
		if (weightParams.scheme == QuantizationScheme::Block &&
		    IsPackedNibbleQuantizedBlockFormat(weightParams.blockFormat))
		{
			return {
				.dequantizedWeight = DequantizePackedNibble(weightStorage, weightParams, DataType::Float32),
				.inputWidth = expressedShape[0],
				.outputWidth = expressedShape[1],
				.sourceParams = weightParams,
				.sourceStorageType = weightStorage.DType(),
			};
		}
		throw std::runtime_error("Prepared quantized Linear currently supports affine and packed-nibble weights only");
	}

	inline Tensor<CPU> EvalPreparedQuantizedMatMul(const Tensor<CPU>& lhs, const PreparedQuantizedLinearWeight& weight,
	                                               DataType outputType = DataType::Float32)
	{
		if (!IsFloatingDataType(lhs.DType()))
		{
			throw std::runtime_error("Prepared quantized MatMul lhs must be floating-point");
		}
		if (!IsFloatingDataType(outputType))
		{
			throw std::runtime_error("Prepared quantized MatMul output type must be floating-point");
		}
		const auto lhsShape = lhs.Shape();
		if (lhsShape.NumDim() != 2)
		{
			throw std::runtime_error("Prepared quantized MatMul currently requires rank-2 lhs");
		}
		if (lhsShape[1] != weight.inputWidth)
		{
			throw std::runtime_error("Prepared quantized MatMul inner dimensions do not match");
		}
		if (weight.dequantizedWeight.DType() != DataType::Float32 ||
		    weight.dequantizedWeight.Shape() != ShapeView{ { weight.inputWidth, weight.outputWidth } })
		{
			throw std::runtime_error("Prepared quantized MatMul weight payload is invalid");
		}

		Tensor<CPU> result(Uninitialized, { lhsShape[0], weight.outputWidth }, outputType);
		const auto lhsF32 = QuantizationDetail::CopyToFloat32(lhs);
		const auto* lhsPtr = static_cast<const float*>(lhsF32.UnsafeRawData());
		const auto* rhsPtr = static_cast<const float*>(weight.dequantizedWeight.UnsafeRawData());
		const auto m = lhsShape[0];
		const auto k = weight.inputWidth;
		const auto n = weight.outputWidth;

		EnumDispatch(outputType, [&]<DataType OutputTypeValue> {
			if constexpr (IsFloatingDataType(OutputTypeValue))
			{
				using OutputT = typename DeviceTraits<CPU>::template DataTypeMapping<OutputTypeValue>;
				auto* dst = static_cast<OutputT*>(result.UnsafeRawData());
				for (std::size_t row = 0; row < m; ++row)
				{
					for (std::size_t col = 0; col < n; ++col)
					{
						double acc = 0.0;
						for (std::size_t kk = 0; kk < k; ++kk)
						{
							acc +=
							    static_cast<double>(lhsPtr[row * k + kk]) * static_cast<double>(rhsPtr[kk * n + col]);
						}
						dst[row * n + col] = static_cast<OutputT>(acc);
					}
				}
			}
		});
		return result;
	}

	inline Tensor<CPU> EvalPreparedQuantizedLinear(const Tensor<CPU>& input,
	                                               const PreparedQuantizedLinearWeight& weight,
	                                               const Tensor<CPU>* bias = nullptr,
	                                               DataType outputType = DataType::Float32)
	{
		auto result = EvalPreparedQuantizedMatMul(input, weight, outputType);
		if (bias == nullptr)
		{
			return result;
		}
		if (bias->DType() != outputType)
		{
			throw std::runtime_error("Prepared quantized Linear bias dtype must match output dtype");
		}
		const auto n = result.Shape()[1];
		const auto biasShape = bias->Shape();
		const auto biasIsVector = biasShape.NumDim() == 1 && biasShape[0] == n;
		const auto biasIsRow = biasShape.NumDim() == 2 && biasShape[0] == 1 && biasShape[1] == n;
		if (!biasIsVector && !biasIsRow)
		{
			throw std::runtime_error("Prepared quantized Linear bias shape must be {N} or {1, N}");
		}

		EnumDispatch(outputType, [&]<DataType OutputTypeValue> {
			if constexpr (IsFloatingDataType(OutputTypeValue))
			{
				using OutputT = typename DeviceTraits<CPU>::template DataTypeMapping<OutputTypeValue>;
				auto* dst = static_cast<OutputT*>(result.UnsafeRawData());
				const auto* biasPtr = static_cast<const OutputT*>(bias->UnsafeRawData());
				for (std::size_t row = 0; row < result.Shape()[0]; ++row)
				{
					for (std::size_t col = 0; col < n; ++col)
					{
						dst[row * n + col] = static_cast<OutputT>(dst[row * n + col] + biasPtr[col]);
					}
				}
			}
		});
		return result;
	}

	inline Tensor<CPU> EvalQuantizedMatMul(const Tensor<CPU>& lhs, const Tensor<CPU>& rhsStorage,
	                                       const QuantizationParams& rhsParams, DataType outputType = DataType::Float32)
	{
		if (!IsFloatingDataType(lhs.DType()))
		{
			throw std::runtime_error("Quantized MatMul lhs must be floating-point");
		}
		if (!IsFloatingDataType(outputType))
		{
			throw std::runtime_error("Quantized MatMul output type must be floating-point");
		}
		ValidateQuantizationParams(rhsParams, rhsStorage.Shape(), rhsStorage.DType());

		const auto outputShape = QuantizedMatMulOutputShape(lhs.Shape(), rhsParams, rhsStorage.Shape());
		Tensor<CPU> result(Uninitialized, outputShape, outputType);
		const auto lhsF32 = QuantizationDetail::CopyToFloat32(lhs);
		const auto* lhsPtr = static_cast<const float*>(lhsF32.UnsafeRawData());
		const auto m = outputShape[0];
		const auto k = lhs.Shape()[1];
		const auto n = outputShape[1];

		EnumDispatch(outputType, [&]<DataType OutputTypeValue> {
			if constexpr (IsFloatingDataType(OutputTypeValue))
			{
				using OutputT = typename DeviceTraits<CPU>::template DataTypeMapping<OutputTypeValue>;
				auto* dst = static_cast<OutputT*>(result.UnsafeRawData());
				for (std::size_t row = 0; row < m; ++row)
				{
					for (std::size_t col = 0; col < n; ++col)
					{
						double acc = 0.0;
						for (std::size_t kk = 0; kk < k; ++kk)
						{
							acc +=
							    static_cast<double>(lhsPtr[row * k + kk]) *
							    static_cast<double>(ReadQuantizedElementAsFloat(rhsStorage, rhsParams, kk * n + col));
						}
						dst[row * n + col] = static_cast<OutputT>(acc);
					}
				}
			}
		});
		return result;
	}

	inline Tensor<CPU> EvalQuantizedLinear(const Tensor<CPU>& input, const Tensor<CPU>& weightStorage,
	                                       const QuantizationParams& weightParams, const Tensor<CPU>* bias = nullptr,
	                                       DataType outputType = DataType::Float32)
	{
		auto result = EvalQuantizedMatMul(input, weightStorage, weightParams, outputType);
		if (bias == nullptr)
		{
			return result;
		}
		if (bias->DType() != outputType)
		{
			throw std::runtime_error("Quantized Linear bias dtype must match output dtype");
		}
		const auto n = result.Shape()[1];
		const auto biasShape = bias->Shape();
		const auto biasIsVector = biasShape.NumDim() == 1 && biasShape[0] == n;
		const auto biasIsRow = biasShape.NumDim() == 2 && biasShape[0] == 1 && biasShape[1] == n;
		if (!biasIsVector && !biasIsRow)
		{
			throw std::runtime_error("Quantized Linear bias shape must be {N} or {1, N}");
		}

		EnumDispatch(outputType, [&]<DataType OutputTypeValue> {
			if constexpr (IsFloatingDataType(OutputTypeValue))
			{
				using OutputT = typename DeviceTraits<CPU>::template DataTypeMapping<OutputTypeValue>;
				auto* dst = static_cast<OutputT*>(result.UnsafeRawData());
				const auto* biasPtr = static_cast<const OutputT*>(bias->UnsafeRawData());
				for (std::size_t row = 0; row < result.Shape()[0]; ++row)
				{
					for (std::size_t col = 0; col < n; ++col)
					{
						dst[row * n + col] = static_cast<OutputT>(dst[row * n + col] + biasPtr[col]);
					}
				}
			}
		});
		return result;
	}
} // namespace LiteNN

#endif
