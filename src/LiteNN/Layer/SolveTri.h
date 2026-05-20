#ifndef LITENN_LAYER_SOLVETRI_H
#define LITENN_LAYER_SOLVETRI_H

#include <LiteNN/ComputePrimitives.h>
#include <LiteNN/Graph.h>

#include <stdexcept>
#include <utility>

namespace LiteNN::Layer
{
	inline NodeOutput AddSolveTri(Subgraph& subgraph, NodeOutput a, NodeOutput b,
	                              bool lower = true, bool unitDiagonal = false)
	{
		const auto aInfo = subgraph.GetOutputInfo(a);
		const auto bInfo = subgraph.GetOutputInfo(b);
		if (aInfo.dtype != DataType::Float32 || bInfo.dtype != DataType::Float32)
		{
			throw std::runtime_error("SolveTri currently supports Float32 tensors only");
		}
		const auto outputShape = ::LiteNN::Detail::SolveTriOutputShape(aInfo.shape, bInfo.shape, lower, unitDiagonal);
		const auto result = subgraph.AddNode(SolveTriNode{ a, b, lower, unitDiagonal },
		                                     { OutputInfo{ DataType::Float32, outputShape } });
		return { result, 0 };
	}

	inline SubgraphId BuildSolveTri(Graph& graph, ShapeView aShape, ShapeView bShape,
	                                bool lower = true, bool unitDiagonal = false)
	{
		Subgraph subgraph;
		const auto a = subgraph.AddParam(DataType::Float32, aShape.ToOwned());
		const auto b = subgraph.AddParam(DataType::Float32, bShape.ToOwned());
		const auto result = AddSolveTri(subgraph, { a, 0 }, { b, 0 }, lower, unitDiagonal);
		subgraph.SetResults({ result });
		return graph.AddSubgraph(std::move(subgraph));
	}
} // namespace LiteNN::Layer

#endif
