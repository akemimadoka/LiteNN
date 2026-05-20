#ifndef LITENN_LAYER_OUTPROD_H
#define LITENN_LAYER_OUTPROD_H

#include <LiteNN/ComputePrimitives.h>
#include <LiteNN/Graph.h>

#include <stdexcept>
#include <utility>

namespace LiteNN::Layer
{
	inline NodeOutput AddOutProd(Subgraph& subgraph, NodeOutput lhs, NodeOutput rhs)
	{
		const auto lhsInfo = subgraph.GetOutputInfo(lhs);
		const auto rhsInfo = subgraph.GetOutputInfo(rhs);
		if (lhsInfo.dtype != rhsInfo.dtype)
		{
			throw std::runtime_error("OutProd inputs must have the same dtype");
		}
		if (!IsFloatingDataType(lhsInfo.dtype))
		{
			throw std::runtime_error("OutProd requires floating-point tensors");
		}
		const auto outputShape = ::LiteNN::Detail::OutProdOutputShape(lhsInfo.shape, rhsInfo.shape);
		const auto result = subgraph.AddNode(OutProdNode{ lhs, rhs }, { OutputInfo{ lhsInfo.dtype, outputShape } });
		return { result, 0 };
	}

	inline SubgraphId BuildOutProd(Graph& graph, DataType dtype, ShapeView lhsShape, ShapeView rhsShape)
	{
		Subgraph subgraph;
		const auto lhs = subgraph.AddParam(dtype, lhsShape.ToOwned());
		const auto rhs = subgraph.AddParam(dtype, rhsShape.ToOwned());
		const auto result = AddOutProd(subgraph, { lhs, 0 }, { rhs, 0 });
		subgraph.SetResults({ result });
		return graph.AddSubgraph(std::move(subgraph));
	}
} // namespace LiteNN::Layer

#endif
