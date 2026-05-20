#ifndef LITENN_LAYER_REPEAT_H
#define LITENN_LAYER_REPEAT_H

#include <LiteNN/Layer/BroadcastTo.h>
#include <LiteNN/Layer/Reshape.h>

#include <format>
#include <span>
#include <stdexcept>
#include <vector>

namespace LiteNN::Layer
{
	inline NodeOutput AddRepeat(Subgraph& subgraph, NodeOutput input,
	                            std::span<const std::size_t> targetShape)
	{
		const auto info = subgraph.GetOutputInfo(input);
		::LiteNN::Detail::ValidatePositiveShape(targetShape, "Repeat target shape");
		if (targetShape.size() < info.shape.size())
		{
			throw std::runtime_error("Repeat target rank must be >= input rank");
		}

		std::vector<std::size_t> paddedInputShape(targetShape.size() - info.shape.size(), 1uz);
		paddedInputShape.insert(paddedInputShape.end(), info.shape.begin(), info.shape.end());

		std::vector<std::size_t> reshapedShape;
		std::vector<std::size_t> broadcastShape;
		reshapedShape.reserve(targetShape.size() * 2);
		broadcastShape.reserve(targetShape.size() * 2);
		for (auto dim = 0uz; dim < targetShape.size(); ++dim)
		{
			const auto inputDim = paddedInputShape[dim];
			const auto targetDim = targetShape[dim];
			if (targetDim % inputDim != 0)
			{
				throw std::runtime_error(std::format("Repeat target dim {} ({}) must be a multiple of input dim {}",
				                               dim, targetDim, inputDim));
			}
			reshapedShape.push_back(1uz);
			reshapedShape.push_back(inputDim);
			broadcastShape.push_back(targetDim / inputDim);
			broadcastShape.push_back(inputDim);
		}

		const auto expanded = AddReshape(subgraph, input, reshapedShape);
		const auto tiled = AddBroadcastTo(subgraph, expanded, broadcastShape);
		return AddReshape(subgraph, tiled, targetShape);
	}

	inline NodeOutput AddRepeat(Subgraph& subgraph, NodeOutput input,
	                            std::initializer_list<std::size_t> targetShape)
	{
		return AddRepeat(subgraph, input,
		                 std::span<const std::size_t>{ targetShape.begin(), targetShape.size() });
	}

	inline SubgraphId BuildRepeat(Graph& graph, DataType dtype, ShapeView inputShape,
	                              std::span<const std::size_t> targetShape)
	{
		Subgraph subgraph;
		const auto input = subgraph.AddParam(dtype, inputShape.ToOwned());
		const auto result = AddRepeat(subgraph, { input, 0 }, targetShape);
		subgraph.SetResults({ result });
		return graph.AddSubgraph(std::move(subgraph));
	}
} // namespace LiteNN::Layer

#endif