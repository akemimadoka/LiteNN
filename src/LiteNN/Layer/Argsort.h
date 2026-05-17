#include <LiteNN/Graph.h>

#include <stdexcept>

#ifndef LITENN_LAYER_ARGSORT_H
#define LITENN_LAYER_ARGSORT_H

namespace LiteNN::Layer
{
	inline NodeOutput AddArgsort(Subgraph& subgraph, NodeOutput input, SortOrder order = SortOrder::Descending)
	{
		const auto info = subgraph.GetOutputInfo(input);
		if (info.shape.empty())
		{
			throw std::runtime_error("Argsort input must have rank >= 1");
		}

		const auto nodeId = subgraph.AddNode(ArgsortNode{ input, order }, { OutputInfo{ DataType::Int32, info.shape } });
		return { nodeId, 0 };
	}

	inline SubgraphId BuildArgsort(Graph& graph, DataType dtype, ShapeView shape,
	                              SortOrder order = SortOrder::Descending)
	{
		Subgraph subgraph;
		const auto input = subgraph.AddParam(dtype, shape.ToOwned());
		const auto result = AddArgsort(subgraph, { input, 0 }, order);
		subgraph.SetResults({ result });
		return graph.AddSubgraph(std::move(subgraph));
	}
} // namespace LiteNN::Layer

#endif