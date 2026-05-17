#include <LiteNN/Graph.h>
#include <LiteNN/Layer/LayerUtils.h>

#include <cmath>
#include <span>
#include <stdexcept>
#include <vector>

#ifndef LITENN_LAYER_ROPE_H
#define LITENN_LAYER_ROPE_H

namespace LiteNN::Layer
{
	namespace Detail
	{
		enum class RoPETrig
		{
			Cos,
			Sin,
		};

		inline Tensor<CPU> MakeRoPEAngleTable(std::size_t sequenceLength, std::size_t halfDim, DataType dtype,
		                                     double base, double frequencyScale, std::size_t positionOffset,
		                                     RoPETrig trig)
		{
			std::vector<double> values;
			values.reserve(sequenceLength * halfDim);
			for (std::size_t pos = 0; pos < sequenceLength; ++pos)
			{
				for (std::size_t dim = 0; dim < halfDim; ++dim)
				{
					const auto exponent = -2.0 * static_cast<double>(dim) / static_cast<double>(halfDim * 2);
					const auto invFreq = std::pow(base, exponent);
					const auto angle = static_cast<double>(positionOffset + pos) * invFreq * frequencyScale;
					values.push_back(trig == RoPETrig::Cos ? std::cos(angle) : std::sin(angle));
				}
			}
			return Tensor<CPU>(std::span<const double>(values), { sequenceLength, halfDim, 1 }, dtype);
		}
	} // namespace Detail

	// Rotary Position Embedding helper.
	// 当前实现支持 2D 输入 [sequenceLength, featureSize]，并在最后一维上按 pair 做旋转。
	inline NodeOutput AddRoPE(Subgraph& subgraph, NodeOutput input, double base = 10000.0,
	                         std::size_t positionOffset = 0, double frequencyScale = 1.0)
	{
		const auto info = subgraph.GetOutputInfo(input); // copy
		if (info.shape.size() != 2)
		{
			throw std::runtime_error("RoPE input must be 2D with shape [sequenceLength, featureSize]");
		}
		if (!IsFloatingDataType(info.dtype))
		{
			throw std::runtime_error("RoPE input dtype must be floating-point");
		}
		if ((info.shape[1] % 2) != 0)
		{
			throw std::runtime_error("RoPE featureSize must be even");
		}
		if (!(std::isfinite(base) && base > 0.0))
		{
			throw std::runtime_error("RoPE base must be finite and greater than zero");
		}
		if (!(std::isfinite(frequencyScale) && frequencyScale > 0.0))
		{
			throw std::runtime_error("RoPE frequencyScale must be finite and greater than zero");
		}

		const auto sequenceLength = info.shape[0];
		const auto featureSize = info.shape[1];
		const auto halfDim = featureSize / 2;
		const std::vector<std::size_t> pairShape{ sequenceLength, halfDim, 2 };
		const std::vector<std::size_t> laneShape{ sequenceLength, halfDim, 1 };

		const auto reshaped = subgraph.AddNode(ReshapeNode{ input, pairShape }, { OutputInfo{ info.dtype, pairShape } });

		const auto first =
		    subgraph.AddNode(SliceNode{ { reshaped, 0 }, 2, 0, 1 }, { OutputInfo{ info.dtype, laneShape } });
		const auto second =
		    subgraph.AddNode(SliceNode{ { reshaped, 0 }, 2, 1, 1 }, { OutputInfo{ info.dtype, laneShape } });

		const auto cosTable = Detail::AddConstant(
		    subgraph, Detail::MakeRoPEAngleTable(sequenceLength, halfDim, info.dtype, base, frequencyScale,
		                                          positionOffset,
		                                          Detail::RoPETrig::Cos));
		const auto sinTable = Detail::AddConstant(
		    subgraph, Detail::MakeRoPEAngleTable(sequenceLength, halfDim, info.dtype, base, frequencyScale,
		                                          positionOffset,
		                                          Detail::RoPETrig::Sin));

		const auto firstCos =
		    subgraph.AddNode(BinaryOpNode{ BinaryOp::Multiply, { first, 0 }, { cosTable, 0 } },
		                     { OutputInfo{ info.dtype, laneShape } });
		const auto secondSin =
		    subgraph.AddNode(BinaryOpNode{ BinaryOp::Multiply, { second, 0 }, { sinTable, 0 } },
		                     { OutputInfo{ info.dtype, laneShape } });
		const auto rotatedFirst =
		    subgraph.AddNode(BinaryOpNode{ BinaryOp::Subtract, { firstCos, 0 }, { secondSin, 0 } },
		                     { OutputInfo{ info.dtype, laneShape } });

		const auto firstSin =
		    subgraph.AddNode(BinaryOpNode{ BinaryOp::Multiply, { first, 0 }, { sinTable, 0 } },
		                     { OutputInfo{ info.dtype, laneShape } });
		const auto secondCos =
		    subgraph.AddNode(BinaryOpNode{ BinaryOp::Multiply, { second, 0 }, { cosTable, 0 } },
		                     { OutputInfo{ info.dtype, laneShape } });
		const auto rotatedSecond =
		    subgraph.AddNode(BinaryOpNode{ BinaryOp::Add, { firstSin, 0 }, { secondCos, 0 } },
		                     { OutputInfo{ info.dtype, laneShape } });

		const auto concatenated = subgraph.AddNode(
		    ConcatNode{ { { rotatedFirst, 0 }, { rotatedSecond, 0 } }, 2 }, { OutputInfo{ info.dtype, pairShape } });
		const auto restored =
		    subgraph.AddNode(ReshapeNode{ { concatenated, 0 }, info.shape }, { OutputInfo{ info.dtype, info.shape } });

		return { restored, 0 };
	}

	inline SubgraphId BuildRoPE(Graph& graph, DataType dtype, ShapeView shape, double base = 10000.0,
	                           std::size_t positionOffset = 0, double frequencyScale = 1.0)
	{
		Subgraph subgraph;
		const auto input = subgraph.AddParam(dtype, shape.ToOwned());
		const auto result = AddRoPE(subgraph, { input, 0 }, base, positionOffset, frequencyScale);
		subgraph.SetResults({ result });
		return graph.AddSubgraph(std::move(subgraph));
	}
} // namespace LiteNN::Layer

#endif
