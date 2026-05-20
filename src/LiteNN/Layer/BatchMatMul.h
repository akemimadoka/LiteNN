#ifndef LITENN_LAYER_BATCHMATMUL_H
#define LITENN_LAYER_BATCHMATMUL_H

#include <LiteNN/ComputePrimitives.h>
#include <LiteNN/Graph.h>

#include <stdexcept>

namespace LiteNN::Layer
{
	inline NodeOutput AddBatchMatMul(Subgraph& subgraph, NodeOutput lhs, NodeOutput rhs)
	{
		const auto lhsInfo = subgraph.GetOutputInfo(lhs);
		const auto rhsInfo = subgraph.GetOutputInfo(rhs);
		if (lhsInfo.dtype != rhsInfo.dtype)
		{
			throw std::runtime_error("BatchMatMul inputs must have the same dtype");
		}
		if (lhsInfo.dtype == DataType::Bool)
		{
			throw std::runtime_error("BatchMatMul does not support Bool tensors");
		}
		const auto outputShape = ::LiteNN::Detail::BatchMatMulOutputShape(lhsInfo.shape, rhsInfo.shape);
		const auto result =
		    subgraph.AddNode(BatchMatMulNode{ lhs, rhs }, { OutputInfo{ lhsInfo.dtype, outputShape } });
		return { result, 0 };
	}

	inline SubgraphId BuildBatchMatMul(Graph& graph, DataType dtype, ShapeView lhsShape, ShapeView rhsShape)
	{
		Subgraph subgraph;
		const auto lhs = subgraph.AddParam(dtype, lhsShape.ToOwned());
		const auto rhs = subgraph.AddParam(dtype, rhsShape.ToOwned());
		const auto result = AddBatchMatMul(subgraph, { lhs, 0 }, { rhs, 0 });
		subgraph.SetResults({ result });
		return graph.AddSubgraph(std::move(subgraph));
	}
} // namespace LiteNN::Layer

#endif
