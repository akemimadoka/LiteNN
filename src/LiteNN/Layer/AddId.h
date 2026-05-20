#include <LiteNN/Graph.h>

#include <stdexcept>

#ifndef LITENN_LAYER_ADDID_H
#define LITENN_LAYER_ADDID_H

namespace LiteNN::Layer
{
	// 匹配 ggml_add_id：对 a[:, usedExpert, token] 加上 b[:, ids[usedExpert, token]]。
	inline NodeOutput AddId(Subgraph& subgraph, NodeOutput a, NodeOutput b, NodeOutput ids)
	{
		const auto aInfo = subgraph.GetOutputInfo(a);
		const auto bInfo = subgraph.GetOutputInfo(b);
		const auto idsInfo = subgraph.GetOutputInfo(ids);

		if (aInfo.shape.size() != 3 || bInfo.shape.size() != 2 || idsInfo.shape.size() != 2)
		{
			throw std::runtime_error("AddId expects a rank-3 tensor, a rank-2 tensor, and a rank-2 id tensor");
		}
		if (aInfo.dtype != bInfo.dtype)
		{
			throw std::runtime_error("AddId requires a and b to have the same dtype");
		}
		if (idsInfo.dtype != DataType::Int32 && idsInfo.dtype != DataType::Int64)
		{
			throw std::runtime_error("AddId ids must have dtype Int32 or Int64");
		}
		if (aInfo.shape[0] != bInfo.shape[0])
		{
			throw std::runtime_error("AddId requires a.shape[0] == b.shape[0]");
		}
		if (aInfo.shape[1] != idsInfo.shape[0] || aInfo.shape[2] != idsInfo.shape[1])
		{
			throw std::runtime_error("AddId requires ids shape to match the last two dimensions of a");
		}

		const auto embd = aInfo.shape[0];
		const auto usedExperts = aInfo.shape[1];
		const auto tokenCount = aInfo.shape[2];
		const auto flatCount = usedExperts * tokenCount;

		const auto bTranspose = subgraph.AddNode(UnaryOpNode{ UnaryOp::Transpose, b },
		                                        { OutputInfo{ bInfo.dtype, { bInfo.shape[1], bInfo.shape[0] } } });
		const auto flatIds = subgraph.AddNode(ReshapeNode{ ids, { flatCount } },
		                                     { OutputInfo{ idsInfo.dtype, { flatCount } } });
		const auto gathered = subgraph.AddNode(GetRowsNode{ { bTranspose, 0 }, { flatIds, 0 } },
		                                      { OutputInfo{ bInfo.dtype, { flatCount, embd } } });
		const auto gatheredTranspose = subgraph.AddNode(UnaryOpNode{ UnaryOp::Transpose, { gathered, 0 } },
		                                               { OutputInfo{ bInfo.dtype, { embd, flatCount } } });
		const auto gatheredReshaped = subgraph.AddNode(ReshapeNode{ { gatheredTranspose, 0 }, aInfo.shape },
		                                              { OutputInfo{ aInfo.dtype, aInfo.shape } });
		const auto result = subgraph.AddNode(BinaryOpNode{ BinaryOp::Add, a, { gatheredReshaped, 0 } },
		                                    { OutputInfo{ aInfo.dtype, aInfo.shape } });

		return { result, 0 };
	}

	inline SubgraphId BuildAddId(Graph& graph, DataType dtype, ShapeView aShape, ShapeView bShape, DataType idsType,
	                            ShapeView idsShape)
	{
		Subgraph subgraph;
		const auto a = subgraph.AddParam(dtype, aShape.ToOwned());
		const auto b = subgraph.AddParam(dtype, bShape.ToOwned());
		const auto ids = subgraph.AddParam(idsType, idsShape.ToOwned());
		const auto result = AddId(subgraph, { a, 0 }, { b, 0 }, { ids, 0 });
		subgraph.SetResults({ result });
		return graph.AddSubgraph(std::move(subgraph));
	}
} // namespace LiteNN::Layer

#endif