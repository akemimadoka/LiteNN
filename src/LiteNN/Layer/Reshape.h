#ifndef LITENN_LAYER_RESHAPE_H
#define LITENN_LAYER_RESHAPE_H

#include <LiteNN/DataMovement.h>
#include <LiteNN/Graph.h>

#include <span>
#include <stdexcept>
#include <utility>
#include <vector>

namespace LiteNN::Layer
{
	inline NodeOutput AddReshape(Subgraph& subgraph, NodeOutput input,
	                             std::span<const std::size_t> targetShape)
	{
		const auto info = subgraph.GetOutputInfo(input);
		::LiteNN::Detail::ValidatePositiveShape(targetShape, "Reshape target shape");
		if (ShapeView{ info.shape }.NumElements() != ShapeView{ targetShape }.NumElements())
		{
			throw std::runtime_error("Reshape target element count must match input element count");
		}

		auto outputShape = std::vector<std::size_t>(targetShape.begin(), targetShape.end());
		const auto result = subgraph.AddNode(ReshapeNode{ input, outputShape },
		                                     { OutputInfo{ info.dtype, outputShape } });
		return { result, 0 };
	}

	inline NodeOutput AddReshape(Subgraph& subgraph, NodeOutput input,
	                             std::initializer_list<std::size_t> targetShape)
	{
		return AddReshape(subgraph, input,
		                  std::span<const std::size_t>{ targetShape.begin(), targetShape.size() });
	}

	inline SubgraphId BuildReshape(Graph& graph, DataType dtype, ShapeView inputShape,
	                               std::span<const std::size_t> targetShape)
	{
		Subgraph subgraph;
		const auto input = subgraph.AddParam(dtype, inputShape.ToOwned());
		const auto result = AddReshape(subgraph, { input, 0 }, targetShape);
		subgraph.SetResults({ result });
		return graph.AddSubgraph(std::move(subgraph));
	}
} // namespace LiteNN::Layer

#endif