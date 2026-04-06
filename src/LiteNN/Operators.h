#ifndef LITENN_OPERATORS_H
#define LITENN_OPERATORS_H

#include <expected>
#include <vector>

#include <LiteNN/Misc.h>

namespace LiteNN
{
	enum class UnaryOp
	{
		Negate,
		Abs,
		Sqrt,
		Transpose,
	};

	// TODO: 应实现 IR 来表示操作
	enum class BinaryOp
	{
		Add,
		Subtract,
		// hadamard product
		Multiply,
		Divide,

		MatMul,
	};

	enum class DataType
	{
		Float32,
		Float64,
		Int32,
		Int64,
	};

	// 默认保持相同类型，相同 shape
	template <UnaryOp Op>
	struct UnaryOpTraits
	{
		static constexpr std::expected<DataType, const char*> ResultType(DataType inputType)
		{
			return inputType;
		}

		static std::expected<std::vector<std::size_t>, const char*>
		ResultShape(ShapeView inputShape)
		{
			return std::vector<std::size_t>(inputShape.Dims.begin(), inputShape.Dims.end());
		}
	};

	template <>
	struct UnaryOpTraits<UnaryOp::Transpose>
	{
		static constexpr std::expected<DataType, const char*> ResultType(DataType inputType)
		{
			return inputType;
		}

		static constexpr std::expected<std::vector<std::size_t>, const char*>
		ResultShape(ShapeView inputShape)
		{
			if (inputShape.NumDim() != 2)
			{
				return std::unexpected("Transpose only supports 2D tensors");
			}
			return std::vector{ inputShape[1], inputShape[0] };
		}
	};

	// 默认保持相同类型，shape 由输入的两个 operand 决定，默认支持广播
	template <BinaryOp Op>
	struct BinaryOpTraits
	{
		static constexpr std::expected<DataType, const char*> ResultType(DataType inputType1,
		                                                                 DataType inputType2)
		{
			if (inputType1 != inputType2)
			{
				return std::unexpected(
				    "Default BinaryOp requires both inputs to have the same data type");
			}
			return inputType1;
		}

		static constexpr std::expected<std::vector<std::size_t>, const char*>
		ResultShape(ShapeView inputShape1, ShapeView inputShape2)
		{
			if (inputShape1.NumDim() != inputShape2.NumDim())
			{
				return std::unexpected(
				    "Default BinaryOp requires both inputs to have the same number of dimensions");
			}
			// 广播机制：其中一个维度为 1 时，可以与另一个维度匹配，否则要求两个维度相等
			// 广播产生的值将会如同重复一般使用
			std::vector<std::size_t> resultShape(inputShape1.Dims.size());
			for (auto i = 0uz; i < inputShape1.NumDim(); ++i)
			{
				if (inputShape1[i] != inputShape2[i] && inputShape1[i] != 1 && inputShape2[i] != 1)
				{
					return std::unexpected(
					    "BinaryOp requires input shapes to be compatible for broadcasting");
				}
				resultShape[i] = std::max(inputShape1[i], inputShape2[i]);
			}
			return std::move(resultShape);
		}
	};

	template <>
	struct BinaryOpTraits<BinaryOp::MatMul>
	{
		static constexpr std::expected<DataType, const char*> ResultType(DataType inputType1,
		                                                                 DataType inputType2)
		{
			if (inputType1 != inputType2)
			{
				return std::unexpected("MatMul requires both inputs to have the same data type");
			}
			return inputType1;
		}

		static constexpr std::expected<std::vector<std::size_t>, const char*>
		ResultShape(ShapeView inputShape1, ShapeView inputShape2)
		{
			if (inputShape1.NumDim() != 2 || inputShape2.NumDim() != 2)
			{
				return std::unexpected("MatMul only supports 2D tensors");
			}
			if (inputShape1[1] != inputShape2[0])
			{
				return std::unexpected(
				    "MatMul requires the inner dimensions of the two inputs to match");
			}
			return std::vector{ inputShape1[0], inputShape2[1] };
		}
	};
} // namespace LiteNN

#endif
