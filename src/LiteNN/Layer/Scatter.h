#ifndef LITENN_LAYER_SCATTER_H
#define LITENN_LAYER_SCATTER_H

#include <LiteNN/DataMovement.h>
#include <LiteNN/Graph.h>

#include <ranges>
#include <stdexcept>

namespace LiteNN::Layer
{
	inline NodeOutput AddScatter(Subgraph& subgraph, NodeOutput data, NodeOutput indices, NodeOutput updates,
	                             std::size_t axis, ScatterMode mode = ScatterMode::Update)
	{
		const auto dataInfo = subgraph.GetOutputInfo(data);
		const auto indicesInfo = subgraph.GetOutputInfo(indices);
		const auto updatesInfo = subgraph.GetOutputInfo(updates);
		auto expectedUpdatesShape = ::LiteNN::Detail::ScatterUpdatesShape(dataInfo.shape, indicesInfo.shape, axis);
		if (updatesInfo.dtype != dataInfo.dtype || !std::ranges::equal(updatesInfo.shape, expectedUpdatesShape))
		{
			throw std::runtime_error("Scatter updates shape or dtype does not match data/indices/axis");
		}
		const auto result = subgraph.AddNode(ScatterNode{ data, indices, updates, axis, mode },
		                                     { OutputInfo{ dataInfo.dtype, dataInfo.shape } });
		return { result, 0 };
	}
} // namespace LiteNN::Layer

#endif
