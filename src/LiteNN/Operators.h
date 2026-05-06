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
		Exp,
		Log,
		Sin,
		Cos,
		Tan,
		Arcsin,
		Arccos,
		Arctan,

		Transpose,

		LogicalNegation,
	};

	enum class BinaryOp
	{
		Add,
		Subtract,
		// hadamard product
		Multiply,
		Divide,

		MatMul,

		Pow,

		Max,
		Min,

		Less,
		Greater,
		Equal,
	};

	enum class DataType
	{
		Float32,
		Float64,
		Int32,
		Int64,
		Bool,
	};

	template <UnaryOp Op>
	struct DefaultUnaryOpTraits
	{
		static constexpr std::expected<DataType, const char*> ResultType(DataType inputType)
		{
			if (inputType == DataType::Bool)
			{
				return std::unexpected("Default UnaryOp does not support boolean tensors");
			}
			return inputType;
		}

		static std::expected<std::vector<std::size_t>, const char*> ResultShape(ShapeView inputShape)
		{
			return std::vector<std::size_t>(inputShape.Dims.begin(), inputShape.Dims.end());
		}
	};

	// 默认保持相同类型，相同 shape
	template <UnaryOp Op>
	struct UnaryOpTraits : DefaultUnaryOpTraits<Op>
	{
	};

	template <>
	struct UnaryOpTraits<UnaryOp::Transpose>
	{
		static constexpr std::expected<DataType, const char*> ResultType(DataType inputType)
		{
			return inputType;
		}

		static constexpr std::expected<std::vector<std::size_t>, const char*> ResultShape(ShapeView inputShape)
		{
			if (inputShape.NumDim() != 2)
			{
				return std::unexpected("Transpose only supports 2D tensors");
			}
			return std::vector{ inputShape[1], inputShape[0] };
		}
	};

	template <>
	struct UnaryOpTraits<UnaryOp::LogicalNegation> : DefaultUnaryOpTraits<UnaryOp::LogicalNegation>
	{
		static constexpr std::expected<DataType, const char*> ResultType(DataType inputType)
		{
			if (inputType != DataType::Bool)
			{
				return std::unexpected("LogicalNegation only supports boolean tensors");
			}
			return DataType::Bool;
		}
	};

	template <BinaryOp Op>
	struct DefaultBinaryOpTraits
	{
		static constexpr std::expected<DataType, const char*> ResultType(DataType inputType1, DataType inputType2)
		{
			if (inputType1 != inputType2)
			{
				return std::unexpected("Default BinaryOp requires both inputs to have the same data type");
			}
			if (inputType1 == DataType::Bool)
			{
				return std::unexpected("Default BinaryOp does not support boolean tensors");
			}
			return inputType1;
		}

		// 广播机制：其中一个维度为 1 时，或其中一个输入不存在该维度时，可以与另一个维度匹配，否则要求两个维度相等
		// 广播将会产生重复值
		static constexpr std::expected<std::vector<std::size_t>, const char*> ResultShape(ShapeView inputShape1,
		                                                                                  ShapeView inputShape2)
		{
			std::vector<std::size_t> resultShape(std::max(inputShape1.Dims.size(), inputShape2.Dims.size()));
			for (auto i = 0uz; i < resultShape.size(); ++i)
			{
				const auto d1 = inputShape1.NumDim() > i ? inputShape1[i] : 1;
				const auto d2 = inputShape2.NumDim() > i ? inputShape2[i] : 1;
				if (d1 != d2 && d1 != 1 && d2 != 1)
				{
					return std::unexpected("BinaryOp requires input shapes to be compatible for broadcasting");
				}
				resultShape[i] = std::max(d1, d2);
			}
			return std::move(resultShape);
		}
	};

	// 默认保持相同类型，shape 由输入的两个 operand 决定，默认支持广播
	template <BinaryOp Op>
	struct BinaryOpTraits : DefaultBinaryOpTraits<Op>
	{
	};

	template <>
	struct BinaryOpTraits<BinaryOp::MatMul>
	{
		static constexpr std::expected<DataType, const char*> ResultType(DataType inputType1, DataType inputType2)
		{
			if (inputType1 != inputType2)
			{
				return std::unexpected("MatMul requires both inputs to have the same data type");
			}
			return inputType1;
		}

		static constexpr std::expected<std::vector<std::size_t>, const char*> ResultShape(ShapeView inputShape1,
		                                                                                  ShapeView inputShape2)
		{
			if (inputShape1.NumDim() != 2 || inputShape2.NumDim() != 2)
			{
				return std::unexpected("MatMul only supports 2D tensors");
			}
			if (inputShape1[1] != inputShape2[0])
			{
				return std::unexpected("MatMul requires the inner dimensions of the two inputs to match");
			}
			return std::vector{ inputShape1[0], inputShape2[1] };
		}
	};

	template <BinaryOp Op>
	    requires(Op == BinaryOp::Less || Op == BinaryOp::Greater || Op == BinaryOp::Equal)
	struct BinaryOpTraits<Op> : DefaultBinaryOpTraits<Op>
	{
		static constexpr std::expected<DataType, const char*> ResultType(DataType inputType1, DataType inputType2)
		{
			if (inputType1 != inputType2)
			{
				return std::unexpected("Comparison ops require both inputs to have the same data type");
			}
			return DataType::Bool;
		}
	};
	enum class ReduceOp
	{
		Sum,
		Mean,
		Max,
	};

	template <ReduceOp Op>
	struct DefaultReduceOpTraits
	{
		static constexpr std::expected<DataType, const char*> ResultType(DataType inputType)
		{
			if (inputType == DataType::Bool)
			{
				return std::unexpected("ReduceOp does not support boolean tensors");
			}
			return inputType;
		}

		static constexpr std::expected<std::vector<std::size_t>, const char*> ResultShape(ShapeView inputShape,
		                                                                                  std::size_t axis)
		{
			if (axis >= inputShape.NumDim())
			{
				return std::unexpected("ReduceOp axis out of range");
			}
			std::vector<std::size_t> result;
			for (auto i = 0uz; i < inputShape.NumDim(); ++i)
			{
				if (i != axis)
				{
					result.push_back(inputShape[i]);
				}
			}
			if (result.empty())
			{
				result.push_back(1);
			}
			return result;
		}
	};

	template <ReduceOp Op>
	struct ReduceOpTraits : DefaultReduceOpTraits<Op>
	{
	};

	enum class FusionPattern
	{
		// y = MatMul(a, b) + c
		MatMulBiasAdd,
		// 2+ 逐元素操作链，每个中间结果仅一个消费者
		ElementWiseChain,
	};

} // namespace LiteNN

#endif
