#ifndef LITENN_TENSOR_TYPE_H
#define LITENN_TENSOR_TYPE_H

#include <LiteNN/DType.h>
#include <LiteNN/Misc.h>
#include <cstddef>
#include <limits>
#include <optional>
#include <span>
#include <stdexcept>
#include <string>
#include <string_view>
#include <vector>

namespace LiteNN
{
	enum class TensorDimKind
	{
		Static,
		Dynamic,
		Symbolic
	};

	struct TensorDim
	{
		static constexpr std::size_t DynamicExtent = std::numeric_limits<std::size_t>::max();

		TensorDimKind kind{ TensorDimKind::Dynamic };
		std::size_t extent{ DynamicExtent };
		std::string symbol;

		static TensorDim Static(std::size_t extent)
		{
			if (extent == 0)
			{
				throw std::runtime_error("Tensor dimensions must be greater than 0");
			}
			return { TensorDimKind::Static, extent, {} };
		}

		static TensorDim Dynamic()
		{
			return { TensorDimKind::Dynamic, DynamicExtent, {} };
		}

		static TensorDim Symbolic(std::string symbol)
		{
			if (symbol.empty())
			{
				throw std::runtime_error("Symbolic tensor dimensions require a non-empty symbol");
			}
			return { TensorDimKind::Symbolic, DynamicExtent, std::move(symbol) };
		}

		bool IsStatic() const noexcept
		{
			return kind == TensorDimKind::Static;
		}

		friend bool operator==(const TensorDim&, const TensorDim&) = default;
	};

	struct TensorShape
	{
		std::vector<TensorDim> dims;

		static TensorShape FromStatic(ShapeView shape)
		{
			TensorShape result;
			result.dims.reserve(shape.NumDim());
			for (const auto dim : shape.Dims)
			{
				result.dims.push_back(TensorDim::Static(dim));
			}
			return result;
		}

		static TensorShape FromStatic(std::span<const std::size_t> shape)
		{
			return FromStatic(ShapeView{ shape });
		}

		std::size_t Rank() const noexcept
		{
			return dims.size();
		}

		bool IsScalar() const noexcept
		{
			return dims.empty();
		}

		bool IsFullyStatic() const noexcept
		{
			for (const auto& dim : dims)
			{
				if (!dim.IsStatic())
				{
					return false;
				}
			}
			return true;
		}

		std::vector<std::size_t> ToStaticShape() const
		{
			std::vector<std::size_t> result;
			result.reserve(dims.size());
			for (const auto& dim : dims)
			{
				if (!dim.IsStatic())
				{
					throw std::runtime_error("Tensor shape is not fully static");
				}
				result.push_back(dim.extent);
			}
			return result;
		}

		std::optional<std::size_t> NumElements() const
		{
			if (!IsFullyStatic())
			{
				return std::nullopt;
			}
			std::size_t result = 1;
			for (const auto& dim : dims)
			{
				result *= dim.extent;
			}
			return result;
		}

		friend bool operator==(const TensorShape&, const TensorShape&) = default;
	};

	enum class TensorLayoutKind
	{
		RowMajor,
		ColumnMajor,
		ChannelsFirst,
		ChannelsLast,
		GGML,
		Torch,
		Blocked,
		Opaque
	};

	struct TensorLayout
	{
		TensorLayoutKind kind{ TensorLayoutKind::RowMajor };
		std::vector<std::size_t> strides;
		std::string tag;

		static TensorLayout RowMajor()
		{
			return {};
		}

		static TensorLayout WithStrides(TensorLayoutKind kind, std::vector<std::size_t> strides, std::string tag = {})
		{
			return { kind, std::move(strides), std::move(tag) };
		}

		bool HasExplicitStrides() const noexcept
		{
			return !strides.empty();
		}

		friend bool operator==(const TensorLayout&, const TensorLayout&) = default;
	};

	enum class TensorMemorySpace
	{
		Host,
		Device,
		Unified,
		Constant,
		External
	};

	struct TensorType
	{
		DataType dtype{ DataType::Float32 };
		TensorShape shape;
		TensorLayout layout;
		TensorMemorySpace memorySpace{ TensorMemorySpace::Host };

		static TensorType Dense(DataType dtype, ShapeView shape,
		                        TensorMemorySpace memorySpace = TensorMemorySpace::Host)
		{
			return { dtype, TensorShape::FromStatic(shape), TensorLayout::RowMajor(), memorySpace };
		}

		static TensorType Ranked(DataType dtype, TensorShape shape, TensorLayout layout = TensorLayout::RowMajor(),
		                         TensorMemorySpace memorySpace = TensorMemorySpace::Host)
		{
			return { dtype, std::move(shape), std::move(layout), memorySpace };
		}

		std::size_t Rank() const noexcept
		{
			return shape.Rank();
		}

		bool IsFullyStatic() const noexcept
		{
			return shape.IsFullyStatic();
		}

		std::vector<std::size_t> StaticShape() const
		{
			return shape.ToStaticShape();
		}

		std::optional<std::size_t> NumElements() const
		{
			return shape.NumElements();
		}

		std::optional<std::size_t> ByteSize() const
		{
			const auto elements = NumElements();
			if (!elements)
			{
				return std::nullopt;
			}
			return *elements * ElementByteSize(dtype);
		}

		friend bool operator==(const TensorType&, const TensorType&) = default;
	};

	inline TensorType MakeTensorType(DataType dtype, std::span<const std::size_t> shape,
	                                 TensorMemorySpace memorySpace = TensorMemorySpace::Host)
	{
		return TensorType::Dense(dtype, ShapeView{ shape }, memorySpace);
	}
} // namespace LiteNN

#endif
