#ifndef LITENN_MISC_H
#define LITENN_MISC_H

#include <format>
#include <meta>
#include <numeric>
#include <ranges>
#include <type_traits>

namespace LiteNN
{
	template <typename T>
	    requires(std::is_enum_v<T>)
	consteval bool IsZeroStartedContinuousEnum()
	{
		constexpr auto enumType = ^^T;
		constexpr auto enumerators = std::define_static_array(std::meta::enumerators_of(enumType));
		if constexpr (enumerators.size() > std::numeric_limits<std::underlying_type_t<T>>::max())
		{
			// 枚举值数量超过底层类型的表示范围，不可能满足连续且从 0 开始的条件
			return false;
		}
		else
		{
			template for (constexpr auto i : std::views::iota(0uz, enumerators.size()))
			{
				if ([:enumerators[i]:] != static_cast<T>(i))
				{
					return false;
				}
			}
			return true;
		}
	}

	// 假定 receiver 是具有模板 operator() 的可调用对象，且 operator()
	// 的模板参数仅有一个接受对应的枚举类型的非类型模板参数
	template <typename Enum, typename Receiver>
	    requires(std::is_enum_v<Enum>)
	constexpr auto EnumDispatch(Enum enumValue, Receiver&& receiver)
	{
		// if constexpr (IsZeroStartedContinuousEnum<Enum>())
		// {
		// 	constexpr auto enumerators =
		// 	    std::define_static_array(std::meta::enumerators_of(^^Enum));
		// 	static constexpr auto dispatchers = [&]<std::size_t...
		// I>(std::index_sequence<I...>) { 		return std::array{
		// +[](Receiver&& receiver) { 			return
		// std::forward<Receiver>(receiver) 			    .template
		// operator()<([:enumerators[I]:])>();
		// 		}... };
		// 	}(std::make_index_sequence<enumerators.size()>{});
		// 	return dispatchers[static_cast<std::size_t>(enumValue)](
		// 	    std::forward<Receiver>(receiver));
		// }
		template for (constexpr auto value :
		              std::define_static_array(std::meta::enumerators_of(^^typename std::decay_t<Enum>)))
		{
			if ([:value:] == enumValue)
			{
				return std::forward<Receiver>(receiver).template operator()<([:value:])>();
			}
		}

		throw std::runtime_error("Invalid enum value");
	}

	enum class EnumToStringStyle
	{
		// EnumType::Value
		Qualified,
		// Value
		Unqualified,
	};

	template <EnumToStringStyle Style, typename Enum>
	    requires(std::is_enum_v<Enum>)
	constexpr std::string_view EnumToString(Enum enumValue)
	{
		template for (constexpr auto enumerator : std::define_static_array(std::meta::enumerators_of(^^Enum)))
		{
			if ([:enumerator:] == enumValue)
			{
				if constexpr (Style == EnumToStringStyle::Qualified)
				{
					return std::define_static_string(
					    std::format("{}::{}", std::meta::identifier_of(^^Enum), std::meta::identifier_of(enumerator)));
				}
				else
				{
					return std::define_static_string(std::meta::identifier_of(enumerator));
				}
			}
		}

		throw std::runtime_error("Invalid enum value");
	}

	struct ShapeView
	{
		std::span<const std::size_t> Dims;

		constexpr ShapeView(std::initializer_list<std::size_t> dims) : Dims(dims)
		{
		}

		constexpr ShapeView(std::span<const std::size_t> dims) : Dims(dims)
		{
		}

		constexpr ShapeView(const std::vector<std::size_t>& dims) : Dims(dims)
		{
		}

		constexpr std::size_t NumElements() const
		{
			return std::accumulate(Dims.begin(), Dims.end(), 1uz, std::multiplies{});
		}

		constexpr std::size_t NumDim() const
		{
			return Dims.size();
		}

		constexpr std::size_t operator[](std::size_t index) const
		{
			return Dims[index];
		}

		constexpr ShapeView SubShape(std::size_t startDim) const
		{
			if (startDim > NumDim())
			{
				throw std::runtime_error("SubShape out of range");
			}
			return { { Dims.data() + startDim, NumDim() - startDim } };
		}

		constexpr bool IsScalar() const
		{
			return NumDim() == 0;
		}

		constexpr std::vector<std::size_t> ToOwned() const
		{
			return std::vector(Dims.begin(), Dims.end());
		}
	};
} // namespace LiteNN

#endif
