#include <LiteNN/Graph.h>
#include <LiteNN/Layer/Activation.h>
#include <LiteNN/Layer/CausalMask.h>
#include <LiteNN/Layer/LayerUtils.h>

#include <cmath>
#include <numeric>
#include <optional>
#include <stdexcept>
#include <vector>

#ifndef LITENN_LAYER_FLASHATTNEXT_H
#define LITENN_LAYER_FLASHATTNEXT_H

namespace LiteNN::Layer
{
	struct FlashAttnExtOptions
	{
		std::optional<NodeOutput> mask;
		std::optional<NodeOutput> sinks;
		double scale = 1.0;
		double maxBias = 0.0;
		double logitSoftcap = 0.0;
		std::size_t headIndex = 0;
		std::size_t headCount = 1;
		bool causal = false;
		std::size_t keyPositionOffset = 0;
		std::size_t queryPositionOffset = 0;
	};

	namespace Detail
	{
		inline NodeOutput CastIfNeeded(Subgraph& subgraph, NodeOutput input, DataType targetType)
		{
			const auto info = subgraph.GetOutputInfo(input);
			if (info.dtype == targetType)
			{
				return input;
			}
			const auto cast = subgraph.AddNode(CastNode{ input, targetType }, { OutputInfo{ targetType, info.shape } });
			return { cast, 0 };
		}

		inline NodeOutput AddTranspose2D(Subgraph& subgraph, NodeOutput input)
		{
			const auto info = subgraph.GetOutputInfo(input);
			if (info.shape.size() != 2)
			{
				throw std::runtime_error("FlashAttnExt transpose expects a 2D tensor");
			}
			const auto transpose = subgraph.AddNode(
			    UnaryOpNode{ UnaryOp::Transpose, input },
			    { OutputInfo{ info.dtype, { info.shape[1], info.shape[0] } } });
			return { transpose, 0 };
		}

		inline NodeOutput AddScalarMultiply(Subgraph& subgraph, NodeOutput input, double scale)
		{
			if (scale == 1.0)
			{
				return input;
			}
			const auto info = subgraph.GetOutputInfo(input);
			const auto scalar = AddConstant(subgraph, MakeScalarTensor(info.dtype, scale));
			const auto scaled = subgraph.AddNode(BinaryOpNode{ BinaryOp::Multiply, input, { scalar, 0 } }, { info });
			return { scaled, 0 };
		}

		inline double ComputeALiBiSlope(std::size_t headIndex, std::size_t headCount, double maxBias)
		{
			if (maxBias <= 0.0)
			{
				return 1.0;
			}
			if (headCount == 0)
			{
				throw std::runtime_error("FlashAttnExt requires headCount > 0 when maxBias is enabled");
			}
			if (headIndex >= headCount)
			{
				throw std::runtime_error("FlashAttnExt headIndex must be smaller than headCount");
			}

			auto headLog2 = 1uz;
			while ((headLog2 << 1uz) <= headCount)
			{
				headLog2 <<= 1uz;
			}

			const auto m0 = std::pow(2.0, -maxBias / static_cast<double>(headLog2));
			const auto m1 = std::pow(2.0, -(maxBias / 2.0) / static_cast<double>(headLog2));
			if (headIndex < headLog2)
			{
				return std::pow(m0, static_cast<double>(headIndex + 1));
			}
			return std::pow(m1, static_cast<double>(2 * (headIndex - headLog2) + 1));
		}
	} // namespace Detail

	inline NodeOutput AddFlashAttnExt(Subgraph& subgraph, NodeOutput queries, NodeOutput keys, NodeOutput values,
	                                 const FlashAttnExtOptions& options = {})
	{
		const auto queryInfo = subgraph.GetOutputInfo(queries);
		if (queryInfo.shape.size() != 2)
		{
			throw std::runtime_error("FlashAttnExt queries must be 2D with shape [queryLength, headDim]");
		}
		if (!IsFloatingDataType(queryInfo.dtype))
		{
			throw std::runtime_error("FlashAttnExt queries must use a floating-point dtype");
		}

		const auto typedKeys = Detail::CastIfNeeded(subgraph, keys, queryInfo.dtype);
		const auto typedValues = Detail::CastIfNeeded(subgraph, values, queryInfo.dtype);
		const auto keyInfo = subgraph.GetOutputInfo(typedKeys);
		const auto valueInfo = subgraph.GetOutputInfo(typedValues);
		if (keyInfo.shape.size() != 2 || keyInfo.shape[1] != queryInfo.shape[1])
		{
			throw std::runtime_error("FlashAttnExt keys must be 2D with shape [keyLength, headDim]");
		}
		if (valueInfo.shape.size() != 2 || valueInfo.shape[0] != keyInfo.shape[0])
		{
			throw std::runtime_error("FlashAttnExt values must be 2D with shape [keyLength, valueDim]");
		}

		const auto transposedKeys = Detail::AddTranspose2D(subgraph, typedKeys);
		const auto scores = subgraph.AddNode(
		    BinaryOpNode{ BinaryOp::MatMul, queries, transposedKeys },
		    { OutputInfo{ queryInfo.dtype, { queryInfo.shape[0], keyInfo.shape[0] } } });
		NodeOutput scoreOutput{ scores, 0 };

		if (options.scale != 1.0 || options.logitSoftcap != 0.0)
		{
			const auto scoreScale = options.logitSoftcap != 0.0 ? options.scale / options.logitSoftcap : options.scale;
			scoreOutput = Detail::AddScalarMultiply(subgraph, scoreOutput, scoreScale);
		}
		if (options.logitSoftcap != 0.0)
		{
			scoreOutput = AddTanh(subgraph, scoreOutput);
			scoreOutput = Detail::AddScalarMultiply(subgraph, scoreOutput, options.logitSoftcap);
		}

		if (options.mask)
		{
			auto mask = Detail::CastIfNeeded(subgraph, *options.mask, queryInfo.dtype);
			const auto maskInfo = subgraph.GetOutputInfo(mask);
			if (maskInfo.shape.size() != 2 || maskInfo.shape[0] != queryInfo.shape[0] || maskInfo.shape[1] != keyInfo.shape[0])
			{
				throw std::runtime_error("FlashAttnExt mask must be 2D with shape [queryLength, keyLength]");
			}
			if (options.maxBias > 0.0)
			{
				mask = Detail::AddScalarMultiply(
				    subgraph, mask, Detail::ComputeALiBiSlope(options.headIndex, options.headCount, options.maxBias));
			}
			const auto scoreInfo = subgraph.GetOutputInfo(scoreOutput);
			const auto maskedScores =
			    subgraph.AddNode(BinaryOpNode{ BinaryOp::Add, scoreOutput, mask }, { scoreInfo });
			scoreOutput = { maskedScores, 0 };
		}
		else if (options.maxBias > 0.0)
		{
			throw std::runtime_error("FlashAttnExt maxBias requires an explicit additive mask");
		}

		if (options.causal)
		{
			scoreOutput = AddCausalMask(subgraph, scoreOutput, -1.0e9, options.keyPositionOffset,
			                            options.queryPositionOffset);
		}

		std::optional<NodeOutput> typedSinks;
		if (options.sinks)
		{
			typedSinks = Detail::CastIfNeeded(subgraph, *options.sinks, queryInfo.dtype);
			const auto sinkInfo = subgraph.GetOutputInfo(*typedSinks);
			const auto elementCount = std::accumulate(
			    sinkInfo.shape.begin(), sinkInfo.shape.end(), 1uz, [](std::size_t lhs, std::size_t rhs) { return lhs * rhs; });
			if (elementCount != 1)
			{
				throw std::runtime_error("FlashAttnExt sinks must be a scalar tensor");
			}
		}

		const auto scoreInfo = subgraph.GetOutputInfo(scoreOutput);
		NodeOutput maxScores{ subgraph.AddNode(ReduceOpNode{ ReduceOp::Max, scoreOutput, 1 },
		                                     { OutputInfo{ scoreInfo.dtype, { scoreInfo.shape[0] } } }),
		                     0 };
		if (typedSinks)
		{
			const auto maxWithSink = subgraph.AddNode(
			    BinaryOpNode{ BinaryOp::Max, maxScores, *typedSinks },
			    { OutputInfo{ scoreInfo.dtype, { scoreInfo.shape[0] } } });
			maxScores = { maxWithSink, 0 };
		}

		const std::vector<std::size_t> broadcastRowShape{ scoreInfo.shape[0], 1 };
		const auto maxScores2D = subgraph.AddNode(
		    ReshapeNode{ maxScores, broadcastRowShape }, { OutputInfo{ scoreInfo.dtype, broadcastRowShape } });
		const auto shiftedScores = subgraph.AddNode(
		    BinaryOpNode{ BinaryOp::Subtract, scoreOutput, { maxScores2D, 0 } }, { scoreInfo });
		const auto expScores = subgraph.AddNode(UnaryOpNode{ UnaryOp::Exp, { shiftedScores, 0 } }, { scoreInfo });

		NodeOutput denominator{
		    subgraph.AddNode(ReduceOpNode{ ReduceOp::Sum, { expScores, 0 }, 1 },
		                     { OutputInfo{ scoreInfo.dtype, { scoreInfo.shape[0] } } }),
		    0 };
		if (typedSinks)
		{
			const auto sinkShift = subgraph.AddNode(
			    BinaryOpNode{ BinaryOp::Subtract, *typedSinks, maxScores },
			    { OutputInfo{ scoreInfo.dtype, { scoreInfo.shape[0] } } });
			const auto sinkExp = subgraph.AddNode(UnaryOpNode{ UnaryOp::Exp, { sinkShift, 0 } },
			                                    { OutputInfo{ scoreInfo.dtype, { scoreInfo.shape[0] } } });
			const auto correctedDenominator = subgraph.AddNode(
			    BinaryOpNode{ BinaryOp::Add, denominator, { sinkExp, 0 } },
			    { OutputInfo{ scoreInfo.dtype, { scoreInfo.shape[0] } } });
			denominator = { correctedDenominator, 0 };
		}

		const auto denominator2D = subgraph.AddNode(
		    ReshapeNode{ denominator, broadcastRowShape }, { OutputInfo{ scoreInfo.dtype, broadcastRowShape } });
		const auto probabilities = subgraph.AddNode(
		    BinaryOpNode{ BinaryOp::Divide, { expScores, 0 }, { denominator2D, 0 } }, { scoreInfo });
		const auto attended = subgraph.AddNode(
		    BinaryOpNode{ BinaryOp::MatMul, { probabilities, 0 }, typedValues },
		    { OutputInfo{ queryInfo.dtype, { queryInfo.shape[0], valueInfo.shape[1] } } });
		return { attended, 0 };
	}
} // namespace LiteNN::Layer

#endif
