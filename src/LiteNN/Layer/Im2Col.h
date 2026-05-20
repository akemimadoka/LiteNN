#ifndef LITENN_LAYER_IM2COL_H
#define LITENN_LAYER_IM2COL_H

#include <LiteNN/ComputePrimitives.h>
#include <LiteNN/Graph.h>

#include <vector>

namespace LiteNN::Layer
{
	inline NodeOutput AddIm2Col(Subgraph& subgraph, NodeOutput input,
	                            std::vector<std::size_t> kernelShape,
	                            std::vector<std::size_t> strides,
	                            std::vector<std::size_t> dilations,
	                            std::vector<std::size_t> lowPads,
	                            std::vector<std::size_t> highPads)
	{
		const auto info = subgraph.GetOutputInfo(input);
		const auto outputShape = ::LiteNN::Detail::Im2ColOutputShape(info.shape, kernelShape, strides,
		                                                             dilations, lowPads, highPads);
		const auto result = subgraph.AddNode(
		    Im2ColNode{ input, std::move(kernelShape), std::move(strides), std::move(dilations),
		                std::move(lowPads), std::move(highPads) },
		    { OutputInfo{ info.dtype, outputShape } });
		return { result, 0 };
	}

	inline SubgraphId BuildIm2Col(Graph& graph, DataType dtype, ShapeView inputShape,
	                              std::vector<std::size_t> kernelShape,
	                              std::vector<std::size_t> strides,
	                              std::vector<std::size_t> dilations,
	                              std::vector<std::size_t> lowPads,
	                              std::vector<std::size_t> highPads)
	{
		Subgraph subgraph;
		const auto input = subgraph.AddParam(dtype, inputShape.ToOwned());
		const auto result = AddIm2Col(subgraph, { input, 0 }, std::move(kernelShape), std::move(strides),
		                              std::move(dilations), std::move(lowPads), std::move(highPads));
		subgraph.SetResults({ result });
		return graph.AddSubgraph(std::move(subgraph));
	}
} // namespace LiteNN::Layer

#endif
