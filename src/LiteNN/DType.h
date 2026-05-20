#include <algorithm>
#include <bit>
#include <cmath>
#include <concepts>
#include <cstdint>
#include <format>
#include <limits>
#include <stdexcept>
#include <string_view>
#include <type_traits>

#ifndef LITENN_DTYPE_H
#define LITENN_DTYPE_H

namespace LiteNN
{
	enum class DataType
	{
		Float32,
		Float64,
		Int32,
		Int64,
		Bool,
		Float16,
		BFloat16,
		Float8E4M3,
		Float8E5M2,
		Int8,
		UInt8,
	};

	namespace Detail
	{
		constexpr float Ldexp(float value, int exponent)
		{
			if (value == 0.0F)
			{
				return 0.0F;
			}
			while (exponent > 0)
			{
				value *= 2.0F;
				--exponent;
			}
			while (exponent < 0)
			{
				value *= 0.5F;
				++exponent;
			}
			return value;
		}

		constexpr std::uint32_t RoundToUInt(float value)
		{
			return static_cast<std::uint32_t>(value + 0.5F);
		}

		constexpr std::uint16_t Float32ToFloat16Bits(float value)
		{
			const auto bits = std::bit_cast<std::uint32_t>(value);
			const auto sign = static_cast<std::uint16_t>((bits >> 16) & 0x8000U);
			auto exponent = static_cast<int>((bits >> 23) & 0xffU) - 127;
			auto mantissa = bits & 0x7fffffU;

			if ((bits & 0x7fffffffU) == 0)
			{
				return sign;
			}
			if (((bits >> 23) & 0xffU) == 0xffU)
			{
				return static_cast<std::uint16_t>(sign | 0x7c00U | (mantissa ? 0x0200U : 0U));
			}
			if (exponent > 15)
			{
				return static_cast<std::uint16_t>(sign | 0x7c00U);
			}
			if (exponent < -14)
			{
				if (exponent < -24)
				{
					return sign;
				}
				mantissa |= 0x800000U;
				const auto shift = static_cast<std::uint32_t>(-exponent - 14 + 13);
				auto halfMantissa = static_cast<std::uint16_t>(mantissa >> shift);
				const auto roundBit = 1U << (shift - 1);
				if ((mantissa & roundBit) != 0)
				{
					++halfMantissa;
				}
				return static_cast<std::uint16_t>(sign | halfMantissa);
			}

			auto halfExponent = static_cast<std::uint16_t>((exponent + 15) << 10);
			auto halfMantissa = static_cast<std::uint16_t>(mantissa >> 13);
			if ((mantissa & 0x1000U) != 0)
			{
				++halfMantissa;
				if ((halfMantissa & 0x0400U) != 0)
				{
					halfMantissa = 0;
					halfExponent = static_cast<std::uint16_t>(halfExponent + 0x0400U);
					if (halfExponent >= 0x7c00U)
					{
						halfExponent = 0x7c00U;
					}
				}
			}
			return static_cast<std::uint16_t>(sign | halfExponent | halfMantissa);
		}

		constexpr float Float16BitsToFloat32(std::uint16_t bits)
		{
			const auto sign = (bits & 0x8000U) ? -1.0F : 1.0F;
			const auto exponent = static_cast<int>((bits >> 10) & 0x1fU);
			const auto mantissa = static_cast<std::uint32_t>(bits & 0x03ffU);
			if (exponent == 0)
			{
				if (mantissa == 0)
				{
					return sign * 0.0F;
				}
				return sign * Ldexp(static_cast<float>(mantissa) / 1024.0F, -14);
			}
			if (exponent == 0x1f)
			{
				if (mantissa == 0)
				{
					return sign * std::numeric_limits<float>::infinity();
				}
				return std::numeric_limits<float>::quiet_NaN();
			}
			return sign * Ldexp(1.0F + static_cast<float>(mantissa) / 1024.0F, exponent - 15);
		}

		constexpr std::uint16_t Float32ToBFloat16Bits(float value)
		{
			const auto bits = std::bit_cast<std::uint32_t>(value);
			const auto lsb = (bits >> 16) & 1U;
			return static_cast<std::uint16_t>((bits + 0x7fffU + lsb) >> 16);
		}

		constexpr float BFloat16BitsToFloat32(std::uint16_t bits)
		{
			return std::bit_cast<float>(static_cast<std::uint32_t>(bits) << 16);
		}

		constexpr std::uint8_t Float32ToFloat8Bits(float value, int exponentBits, int mantissaBits, int exponentBias)
		{
			const auto signBit = std::signbit(value) ? 0x80U : 0U;
			float magnitude = value < 0.0F ? -value : value;
			if (magnitude == 0.0F)
			{
				return static_cast<std::uint8_t>(signBit);
			}
			if (!std::isfinite(magnitude))
			{
				const auto maxExponent = (1U << exponentBits) - 1U;
				return static_cast<std::uint8_t>(signBit | (maxExponent << mantissaBits));
			}

			int exponent = 0;
			while (magnitude >= 2.0F)
			{
				magnitude *= 0.5F;
				++exponent;
			}
			while (magnitude < 1.0F)
			{
				magnitude *= 2.0F;
				--exponent;
			}

			const auto maxExponent = (1U << exponentBits) - 1U;
			const auto storedExponent = exponent + exponentBias;
			const auto mantissaScale = static_cast<float>(1U << mantissaBits);
			if (storedExponent >= static_cast<int>(maxExponent))
			{
				return static_cast<std::uint8_t>(signBit | (maxExponent << mantissaBits));
			}
			if (storedExponent <= 0)
			{
				const auto subnormal = Ldexp(magnitude, exponent + exponentBias - 1);
				const auto mantissa = std::min(RoundToUInt(subnormal * mantissaScale),
				                               (1U << mantissaBits) - 1U);
				return static_cast<std::uint8_t>(signBit | mantissa);
			}

			auto mantissa = RoundToUInt((magnitude - 1.0F) * mantissaScale);
			auto exponentField = static_cast<std::uint32_t>(storedExponent);
			if (mantissa == (1U << mantissaBits))
			{
				mantissa = 0;
				++exponentField;
				if (exponentField >= maxExponent)
				{
					exponentField = maxExponent;
				}
			}
			return static_cast<std::uint8_t>(signBit | (exponentField << mantissaBits) | mantissa);
		}

		constexpr float Float8BitsToFloat32(std::uint8_t bits, int exponentBits, int mantissaBits, int exponentBias)
		{
			const auto sign = (bits & 0x80U) ? -1.0F : 1.0F;
			const auto exponentMask = (1U << exponentBits) - 1U;
			const auto exponentField = (bits >> mantissaBits) & exponentMask;
			const auto mantissa = bits & ((1U << mantissaBits) - 1U);
			const auto mantissaScale = static_cast<float>(1U << mantissaBits);
			if (exponentField == 0)
			{
				if (mantissa == 0)
				{
					return sign * 0.0F;
				}
				return sign * Ldexp(static_cast<float>(mantissa) / mantissaScale, 1 - exponentBias);
			}
			if (exponentField == exponentMask)
			{
				if (mantissa == 0)
				{
					return sign * std::numeric_limits<float>::infinity();
				}
				return std::numeric_limits<float>::quiet_NaN();
			}
			return sign * Ldexp(1.0F + static_cast<float>(mantissa) / mantissaScale,
			                    static_cast<int>(exponentField) - exponentBias);
		}

		template <typename T>
		concept FloatConvertible = requires(T value) {
			static_cast<float>(value);
		};
	} // namespace Detail

	struct Float16
	{
		std::uint16_t bits{};

		constexpr Float16() = default;
		constexpr Float16(Detail::FloatConvertible auto value)
		    : bits(Detail::Float32ToFloat16Bits(static_cast<float>(value)))
		{
		}
		static constexpr Float16 FromBits(std::uint16_t rawBits)
		{
			Float16 value;
			value.bits = rawBits;
			return value;
		}

		constexpr operator float() const
		{
			return Detail::Float16BitsToFloat32(bits);
		}
		constexpr Float16& operator+=(Detail::FloatConvertible auto rhs)
		{
			*this = Float16(static_cast<float>(*this) + static_cast<float>(rhs));
			return *this;
		}
	};

	struct BFloat16
	{
		std::uint16_t bits{};

		constexpr BFloat16() = default;
		constexpr BFloat16(Detail::FloatConvertible auto value)
		    : bits(Detail::Float32ToBFloat16Bits(static_cast<float>(value)))
		{
		}
		static constexpr BFloat16 FromBits(std::uint16_t rawBits)
		{
			BFloat16 value;
			value.bits = rawBits;
			return value;
		}

		constexpr operator float() const
		{
			return Detail::BFloat16BitsToFloat32(bits);
		}
		constexpr BFloat16& operator+=(Detail::FloatConvertible auto rhs)
		{
			*this = BFloat16(static_cast<float>(*this) + static_cast<float>(rhs));
			return *this;
		}
	};

	struct Float8E4M3
	{
		std::uint8_t bits{};

		constexpr Float8E4M3() = default;
		constexpr Float8E4M3(Detail::FloatConvertible auto value)
		    : bits(Detail::Float32ToFloat8Bits(static_cast<float>(value), 4, 3, 7))
		{
		}
		static constexpr Float8E4M3 FromBits(std::uint8_t rawBits)
		{
			Float8E4M3 value;
			value.bits = rawBits;
			return value;
		}

		constexpr operator float() const
		{
			return Detail::Float8BitsToFloat32(bits, 4, 3, 7);
		}
		constexpr Float8E4M3& operator+=(Detail::FloatConvertible auto rhs)
		{
			*this = Float8E4M3(static_cast<float>(*this) + static_cast<float>(rhs));
			return *this;
		}
	};

	struct Float8E5M2
	{
		std::uint8_t bits{};

		constexpr Float8E5M2() = default;
		constexpr Float8E5M2(Detail::FloatConvertible auto value)
		    : bits(Detail::Float32ToFloat8Bits(static_cast<float>(value), 5, 2, 15))
		{
		}
		static constexpr Float8E5M2 FromBits(std::uint8_t rawBits)
		{
			Float8E5M2 value;
			value.bits = rawBits;
			return value;
		}

		constexpr operator float() const
		{
			return Detail::Float8BitsToFloat32(bits, 5, 2, 15);
		}
		constexpr Float8E5M2& operator+=(Detail::FloatConvertible auto rhs)
		{
			*this = Float8E5M2(static_cast<float>(*this) + static_cast<float>(rhs));
			return *this;
		}
	};

	constexpr bool operator==(Float16 lhs, Float16 rhs)
	{
		return static_cast<float>(lhs) == static_cast<float>(rhs);
	}
	constexpr bool operator==(BFloat16 lhs, BFloat16 rhs)
	{
		return static_cast<float>(lhs) == static_cast<float>(rhs);
	}
	constexpr bool operator==(Float8E4M3 lhs, Float8E4M3 rhs)
	{
		return static_cast<float>(lhs) == static_cast<float>(rhs);
	}
	constexpr bool operator==(Float8E5M2 lhs, Float8E5M2 rhs)
	{
		return static_cast<float>(lhs) == static_cast<float>(rhs);
	}

	constexpr bool operator<(Float16 lhs, Float16 rhs)
	{
		return static_cast<float>(lhs) < static_cast<float>(rhs);
	}
	constexpr bool operator<(BFloat16 lhs, BFloat16 rhs)
	{
		return static_cast<float>(lhs) < static_cast<float>(rhs);
	}
	constexpr bool operator<(Float8E4M3 lhs, Float8E4M3 rhs)
	{
		return static_cast<float>(lhs) < static_cast<float>(rhs);
	}
	constexpr bool operator<(Float8E5M2 lhs, Float8E5M2 rhs)
	{
		return static_cast<float>(lhs) < static_cast<float>(rhs);
	}

	constexpr bool operator>(auto lhs, auto rhs)
	    requires(std::same_as<std::remove_cvref_t<decltype(lhs)>, Float16> ||
	             std::same_as<std::remove_cvref_t<decltype(lhs)>, BFloat16> ||
	             std::same_as<std::remove_cvref_t<decltype(lhs)>, Float8E4M3> ||
	             std::same_as<std::remove_cvref_t<decltype(lhs)>, Float8E5M2>)
	{
		return rhs < lhs;
	}

	inline constexpr DataType LastDataType = DataType::UInt8;

	constexpr bool IsValidDataTypeValue(DataType dtype)
	{
		switch (dtype)
		{
		case DataType::Float32:
		case DataType::Float64:
		case DataType::Int32:
		case DataType::Int64:
		case DataType::Bool:
		case DataType::Float16:
		case DataType::BFloat16:
		case DataType::Float8E4M3:
		case DataType::Float8E5M2:
		case DataType::Int8:
		case DataType::UInt8:
			return true;
		}
		return false;
	}

	constexpr bool IsFloatingDataType(DataType dtype)
	{
		switch (dtype)
		{
		case DataType::Float32:
		case DataType::Float64:
		case DataType::Float16:
		case DataType::BFloat16:
		case DataType::Float8E4M3:
		case DataType::Float8E5M2:
			return true;
		case DataType::Int32:
		case DataType::Int64:
		case DataType::Bool:
		case DataType::Int8:
		case DataType::UInt8:
			return false;
		}
		return false;
	}

	constexpr std::size_t ElementByteSize(DataType dtype)
	{
		switch (dtype)
		{
		case DataType::Float32:
			return sizeof(float);
		case DataType::Float64:
			return sizeof(double);
		case DataType::Int32:
			return sizeof(std::int32_t);
		case DataType::Int64:
			return sizeof(std::int64_t);
		case DataType::Bool:
			return sizeof(bool);
		case DataType::Float16:
			return sizeof(Float16);
		case DataType::BFloat16:
			return sizeof(BFloat16);
		case DataType::Float8E4M3:
			return sizeof(Float8E4M3);
		case DataType::Float8E5M2:
			return sizeof(Float8E5M2);
		case DataType::Int8:
			return sizeof(std::int8_t);
		case DataType::UInt8:
			return sizeof(std::uint8_t);
		}
		throw std::runtime_error("Invalid data type");
	}

	constexpr std::string_view DataTypeName(DataType dtype)
	{
		switch (dtype)
		{
		case DataType::Float32:
			return "Float32";
		case DataType::Float64:
			return "Float64";
		case DataType::Int32:
			return "Int32";
		case DataType::Int64:
			return "Int64";
		case DataType::Bool:
			return "Bool";
		case DataType::Float16:
			return "Float16";
		case DataType::BFloat16:
			return "BFloat16";
		case DataType::Float8E4M3:
			return "Float8E4M3";
		case DataType::Float8E5M2:
			return "Float8E5M2";
		case DataType::Int8:
			return "Int8";
		case DataType::UInt8:
			return "UInt8";
		}
		throw std::runtime_error("Invalid data type");
	}
} // namespace LiteNN

template <>
struct std::formatter<LiteNN::Float16> : std::formatter<float>
{
	template <class FormatContext>
	auto format(LiteNN::Float16 value, FormatContext& ctx) const
	{
		return std::formatter<float>::format(static_cast<float>(value), ctx);
	}
};

template <>
struct std::formatter<LiteNN::BFloat16> : std::formatter<float>
{
	template <class FormatContext>
	auto format(LiteNN::BFloat16 value, FormatContext& ctx) const
	{
		return std::formatter<float>::format(static_cast<float>(value), ctx);
	}
};

template <>
struct std::formatter<LiteNN::Float8E4M3> : std::formatter<float>
{
	template <class FormatContext>
	auto format(LiteNN::Float8E4M3 value, FormatContext& ctx) const
	{
		return std::formatter<float>::format(static_cast<float>(value), ctx);
	}
};

template <>
struct std::formatter<LiteNN::Float8E5M2> : std::formatter<float>
{
	template <class FormatContext>
	auto format(LiteNN::Float8E5M2 value, FormatContext& ctx) const
	{
		return std::formatter<float>::format(static_cast<float>(value), ctx);
	}
};

#endif
