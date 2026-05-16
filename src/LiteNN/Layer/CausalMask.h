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
		inline Tensor<CPU> MakeCausalMaskTensor(std::size_t sequenceLength, DataType dtype, double maskedValue)
		{
			std::vector<double> values(sequenceLength * sequenceLength, 0.0);
			for (std::size_t row = 0; row < sequenceLength; ++row)
			{
				for (std::size_t col = row + 1; col < sequenceLength; ++col)
				{
					values[row * sequenceLength + col] = maskedValue;
				}
			}
			return Tensor<CPU>(std::span<const double>(values), { sequenceLength, sequenceLength }, dtype);
		}
	} // namespace Detail

	inline NodeOutput AddCausalMask(Subgraph& subgraph, NodeOutput input, double maskedValue = -1.0e9)
	{
		const auto info = subgraph.GetOutputInfo(input);
		if (info.shape.size() != 2)
		{
			throw std::runtime_error("Causal mask input must be 2D with shape [sequenceLength, sequenceLength]");
		}
		if (!IsFloatingDataType(info.dtype))
		{
			throw std::runtime_error("Causal mask input dtype must be floating-point");
		}
		if (info.shape[0] != info.shape[1])
		{
			throw std::runtime_error("Causal mask input must be square");
		}

		const auto mask = Detail::AddConstant(subgraph,
		                                     Detail::MakeCausalMaskTensor(info.shape[0], info.dtype, maskedValue));
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