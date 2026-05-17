#include <LiteNN/Graph.h>
#include <LiteNN/Layer/Argsort.h>

#include <stdexcept>

#ifndef LITENN_LAYER_TOPK_H
#define LITENN_LAYER_TOPK_H

namespace LiteNN::Layer
{
	inline NodeOutput AddTopK(Subgraph& subgraph, NodeOutput input, std::size_t k)
	{
		const auto info = subgraph.GetOutputInfo(input);
		if (info.shape.empty())
		{
			throw std::runtime_error("TopK input must have rank >= 1");
		}
		if (k == 0 || k > info.shape[0])
		{
			throw std::runtime_error("TopK requires 1 <= k <= input.shape[0]");
		}

		const auto sorted = AddArgsort(subgraph, input, SortOrder::Descending);
		if (k == info.shape[0])
		{
			return sorted;
		}

		auto outputShape = info.shape;
		outputShape[0] = k;
		const auto nodeId = subgraph.AddNode(SliceNode{ sorted, 0, 0, k }, { OutputInfo{ DataType::Int32, outputShape } });
		return { nodeId, 0 };
	}

	inline SubgraphId BuildTopK(Graph& graph, DataType dtype, ShapeView shape, std::size_t k)
	{
		Subgraph subgraph;
		const auto input = subgraph.AddParam(dtype, shape.ToOwned());
		const auto result = AddTopK(subgraph, { input, 0 }, k);
		subgraph.SetResults({ result });
		return graph.AddSubgraph(std::move(subgraph));
	}
} // namespace LiteNN::Layer

#endif