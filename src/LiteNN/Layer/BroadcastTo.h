#ifndef LITENN_LAYER_BROADCASTTO_H
#define LITENN_LAYER_BROADCASTTO_H

#include <LiteNN/DataMovement.h>
#include <LiteNN/Graph.h>

#include <span>
#include <vector>

namespace LiteNN::Layer
{
	inline NodeOutput AddBroadcastTo(Subgraph& subgraph, NodeOutput input,
	                                 std::span<const std::size_t> targetShape)
	{
		const auto info = subgraph.GetOutputInfo(input);
		auto target = std::vector<std::size_t>(targetShape.begin(), targetShape.end());
		auto outputShape = ::LiteNN::Detail::BroadcastToShape(info.shape, target);
		const auto result = subgraph.AddNode(BroadcastToNode{ input, std::move(target) },
		                                     { OutputInfo{ info.dtype, std::move(outputShape) } });
		return { result, 0 };
	}

	inline SubgraphId BuildBroadcastTo(Graph& graph, DataType dtype, ShapeView shape,
	                                   std::span<const std::size_t> targetShape)
	{
		Subgraph subgraph;
		const auto input = subgraph.AddParam(dtype, shape.ToOwned());
		const auto result = AddBroadcastTo(subgraph, { input, 0 }, targetShape);
		subgraph.SetResults({ result });
		return graph.AddSubgraph(std::move(subgraph));
	}
} // namespace LiteNN::Layer

#endif
