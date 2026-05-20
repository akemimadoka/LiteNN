#ifndef LITENN_LAYER_GATHER_H
#define LITENN_LAYER_GATHER_H

#include <LiteNN/DataMovement.h>
#include <LiteNN/Graph.h>

namespace LiteNN::Layer
{
	inline NodeOutput AddGather(Subgraph& subgraph, NodeOutput data, NodeOutput indices, std::size_t axis)
	{
		const auto dataInfo = subgraph.GetOutputInfo(data);
		const auto indicesInfo = subgraph.GetOutputInfo(indices);
		auto outputShape = ::LiteNN::Detail::GatherOutputShape(dataInfo.shape, indicesInfo.shape, axis);
		const auto result = subgraph.AddNode(GatherNode{ data, indices, axis },
		                                     { OutputInfo{ dataInfo.dtype, std::move(outputShape) } });
		return { result, 0 };
	}
} // namespace LiteNN::Layer

#endif
