#ifndef LITENN_LAYER_UPSAMPLE_H
#define LITENN_LAYER_UPSAMPLE_H

#include <LiteNN/ComputePrimitives.h>
#include <LiteNN/Graph.h>

#include <stdexcept>
#include <vector>

namespace LiteNN::Layer
{
	inline NodeOutput AddUpsample(Subgraph& subgraph, NodeOutput input, UpsampleMode mode,
	                              std::vector<std::size_t> outputSpatialShape,
	                              bool alignCorners = false)
	{
		const auto inputInfo = subgraph.GetOutputInfo(input);
		const auto outputShape = ::LiteNN::Detail::UpsampleOutputShape(inputInfo.shape, outputSpatialShape);
		if (inputInfo.dtype == DataType::Bool && mode != UpsampleMode::Nearest)
		{
			throw std::runtime_error("Upsample only supports Bool tensors in nearest mode");
		}
		const auto result = subgraph.AddNode(
		    UpsampleNode{ input, mode, std::move(outputSpatialShape), alignCorners },
		    { OutputInfo{ inputInfo.dtype, outputShape } });
		return { result, 0 };
	}

	inline NodeOutput AddNearestUpsample2D(Subgraph& subgraph, NodeOutput input,
	                                       std::vector<std::size_t> outputSpatialShape)
	{
		return AddUpsample(subgraph, input, UpsampleMode::Nearest, std::move(outputSpatialShape));
	}

	inline NodeOutput AddBilinearUpsample2D(Subgraph& subgraph, NodeOutput input,
	                                        std::vector<std::size_t> outputSpatialShape,
	                                        bool alignCorners = false)
	{
		return AddUpsample(subgraph, input, UpsampleMode::Bilinear, std::move(outputSpatialShape), alignCorners);
	}

	inline NodeOutput AddBicubicUpsample2D(Subgraph& subgraph, NodeOutput input,
	                                       std::vector<std::size_t> outputSpatialShape,
	                                       bool alignCorners = false)
	{
		return AddUpsample(subgraph, input, UpsampleMode::Bicubic, std::move(outputSpatialShape), alignCorners);
	}

	inline SubgraphId BuildUpsample(Graph& graph, DataType dtype, ShapeView inputShape, UpsampleMode mode,
	                                std::vector<std::size_t> outputSpatialShape,
	                                bool alignCorners = false)
	{
		Subgraph subgraph;
		const auto input = subgraph.AddParam(dtype, inputShape.ToOwned());
		const auto result = AddUpsample(subgraph, { input, 0 }, mode, std::move(outputSpatialShape), alignCorners);
		subgraph.SetResults({ result });
		return graph.AddSubgraph(std::move(subgraph));
	}
} // namespace LiteNN::Layer

#endif
