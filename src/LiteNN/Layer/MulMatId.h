#include <LiteNN/Graph.h>

#include <stdexcept>

#ifndef LITENN_LAYER_MULMATID_H
#define LITENN_LAYER_MULMATID_H

namespace LiteNN::Layer
{
	// Add ggml-compatible MUL_MAT_ID lowering. Shapes follow ggml order:
	// as=[k, m, expertCount], b=[k, usedExpertSlots, tokenCount], ids=[usedExperts, tokenCount].
	// The result is always Float32 [m, usedExperts, tokenCount] to match the interpreter accumulator.
	inline NodeOutput AddMulMatId(Subgraph& subgraph, NodeOutput as, NodeOutput b, NodeOutput ids)
	{
		const auto asInfo = subgraph.GetOutputInfo(as);
		const auto bInfo = subgraph.GetOutputInfo(b);
		const auto idsInfo = subgraph.GetOutputInfo(ids);

		if (asInfo.shape.size() != 3 || bInfo.shape.size() != 3 || idsInfo.shape.size() != 2)
		{
			throw std::runtime_error("MulMatId expects rank-3 expert and input tensors plus a rank-2 id tensor");
		}
		if (!IsFloatingDataType(asInfo.dtype) || !IsFloatingDataType(bInfo.dtype))
		{
			throw std::runtime_error("MulMatId requires floating-point expert and input tensors");
		}
		if (idsInfo.dtype != DataType::Int32 && idsInfo.dtype != DataType::Int64)
		{
			throw std::runtime_error("MulMatId ids must have dtype Int32 or Int64");
		}
		if (asInfo.shape[0] != bInfo.shape[0])
		{
			throw std::runtime_error("MulMatId requires as.shape[0] == b.shape[0]");
		}
		if (idsInfo.shape[1] != bInfo.shape[2])
		{
			throw std::runtime_error("MulMatId requires ids.shape[1] == b.shape[2]");
		}
		if (idsInfo.shape[0] % bInfo.shape[1] != 0)
		{
			throw std::runtime_error("MulMatId requires ids.shape[0] to be divisible by b.shape[1]");
		}

		const auto nodeId = subgraph.AddNode(MulMatIdNode{ as, b, ids },
		                                    { OutputInfo{ DataType::Float32, { asInfo.shape[1], idsInfo.shape[0], bInfo.shape[2] } } });
		return { nodeId, 0 };
	}

	inline SubgraphId BuildMulMatId(Graph& graph, DataType asType, ShapeView asShape, DataType bType, ShapeView bShape,
	                               DataType idsType, ShapeView idsShape)
	{
		Subgraph subgraph;
		const auto as = subgraph.AddParam(asType, asShape.ToOwned());
		const auto b = subgraph.AddParam(bType, bShape.ToOwned());
		const auto ids = subgraph.AddParam(idsType, idsShape.ToOwned());
		const auto result = AddMulMatId(subgraph, { as, 0 }, { b, 0 }, { ids, 0 });
		subgraph.SetResults({ result });
		return graph.AddSubgraph(std::move(subgraph));
	}
} // namespace LiteNN::Layer

#endif
