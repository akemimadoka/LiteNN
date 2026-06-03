#ifndef LITENN_LAYER_TIMESTEPEMBEDDING_H
#define LITENN_LAYER_TIMESTEPEMBEDDING_H

#include <LiteNN/ComputePrimitives.h>
#include <LiteNN/Graph.h>
#include <LiteNN/ModelBuilder.h>

#include <stdexcept>
#include <utility>

namespace LiteNN::Layer
{
	inline NodeOutput AddTimestepEmbedding(Subgraph& subgraph, NodeOutput timesteps, std::size_t dim,
	                                       std::size_t maxPeriod = 10000)
	{
		const auto info = subgraph.GetOutputInfo(timesteps);
		if (!IsFloatingDataType(info.dtype))
		{
			throw std::runtime_error("TimestepEmbedding requires floating-point timesteps");
		}
		const auto outputShape = ::LiteNN::Detail::TimestepEmbeddingOutputShape(info.shape, dim, maxPeriod);
		const auto result = subgraph.AddNode(TimestepEmbeddingNode{ timesteps, dim, maxPeriod },
		                                     { OutputInfo{ DataType::Float32, outputShape } });
		return { result, 0 };
	}

	inline SubgraphId BuildTimestepEmbedding(ModelBuilder& builder, DataType dtype, ShapeView timestepShape,
	                                         std::size_t dim, std::size_t maxPeriod = 10000)
	{
		Subgraph subgraph;
		const auto timesteps = subgraph.AddParam(dtype, timestepShape.ToOwned());
		const auto result = AddTimestepEmbedding(subgraph, { timesteps, 0 }, dim, maxPeriod);
		subgraph.SetResults({ result });
		return builder.AddSubgraph(std::move(subgraph));
	}
} // namespace LiteNN::Layer

#endif
