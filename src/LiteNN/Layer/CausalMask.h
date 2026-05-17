#include <LiteNN/Graph.h>
#include <LiteNN/Layer/LayerUtils.h>

#include <span>
#include <stdexcept>
#include <vector>

#ifndef LITENN_LAYER_CAUSALMASK_H
#define LITENN_LAYER_CAUSALMASK_H

namespace LiteNN::Layer
{
	namespace Detail
	{
		inline Tensor<CPU> MakeCausalMaskTensor(std::size_t queryLength, std::size_t keyLength, DataType dtype,
		                                        double maskedValue, std::size_t keyPositionOffset = 0,
		                                        std::size_t queryPositionOffset = 0)
		{
			std::vector<double> values(queryLength * keyLength, 0.0);
			for (std::size_t row = 0; row < queryLength; ++row)
			{
				const auto queryPosition = queryPositionOffset + row;
				for (std::size_t col = 0; col < keyLength; ++col)
				{
					const auto keyPosition = keyPositionOffset + col;
					if (keyPosition > queryPosition)
					{
						values[row * keyLength + col] = maskedValue;
					}
				}
			}
			return Tensor<CPU>(std::span<const double>(values), { queryLength, keyLength }, dtype);
		}
	} // namespace Detail

	inline NodeOutput AddCausalMask(Subgraph& subgraph, NodeOutput input, double maskedValue = -1.0e9,
	                                std::size_t keyPositionOffset = 0, std::size_t queryPositionOffset = 0)
	{
		const auto info = subgraph.GetOutputInfo(input);
		if (info.shape.size() != 2)
		{
			throw std::runtime_error("Causal mask input must be 2D with shape [queryLength, keyLength]");
		}
		if (!IsFloatingDataType(info.dtype))
		{
			throw std::runtime_error("Causal mask input dtype must be floating-point");
		}

		const auto mask =
		    Detail::AddConstant(subgraph, Detail::MakeCausalMaskTensor(info.shape[0], info.shape[1], info.dtype,
		                                                               maskedValue, keyPositionOffset,
		                                                               queryPositionOffset));
		const auto result = subgraph.AddNode(BinaryOpNode{ BinaryOp::Add, input, mask }, { info });
		return { result, 0 };
	}

	inline SubgraphId BuildCausalMask(Graph& graph, DataType dtype, std::size_t sequenceLength,
	                                double maskedValue = -1.0e9)
	{
		Subgraph subgraph;
		const auto input = subgraph.AddParam(dtype, { sequenceLength, sequenceLength });
		const auto result = AddCausalMask(subgraph, { input, 0 }, maskedValue);
		subgraph.SetResults({ result });
		return graph.AddSubgraph(std::move(subgraph));
	}
} // namespace LiteNN::Layer

#endif
