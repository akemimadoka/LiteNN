#include <LiteNN/Graph.h>

#include <numeric>
#include <stdexcept>
#include <utility>
#include <vector>

#ifndef LITENN_LAYER_PERMUTE_H
#define LITENN_LAYER_PERMUTE_H

namespace LiteNN::Layer
{
	inline std::vector<std::size_t> PermutedShape(ShapeView shape, const std::vector<std::size_t>& permutation)
	{
		if (shape.NumDim() != permutation.size())
		{
			throw std::runtime_error("Permute permutation rank does not match input rank");
		}
		std::vector<std::size_t> result(permutation.size());
		std::vector<bool> seen(permutation.size(), false);
		for (auto d = 0uz; d < permutation.size(); ++d)
		{
			const auto p = permutation[d];
			if (p >= permutation.size() || seen[p])
			{
				throw std::runtime_error("Permute permutation must be a valid permutation of [0, rank)");
			}
			seen[p] = true;
			result[d] = shape.Dims[p];
		}
		return result;
	}

	inline NodeOutput AddPermute(Subgraph& subgraph, NodeOutput input, std::vector<std::size_t> permutation)
	{
		const auto info = subgraph.GetOutputInfo(input);
		auto outputShape = PermutedShape(info.shape, permutation);
		const auto result = subgraph.AddNode(PermuteNode{ input, std::move(permutation) },
		                                     { OutputInfo{ info.dtype, outputShape } });
		return { result, 0 };
	}

	inline SubgraphId BuildPermute(Graph& graph, DataType dtype, ShapeView shape,
	                               std::vector<std::size_t> permutation)
	{
		Subgraph subgraph;
		const auto input = subgraph.AddParam(dtype, shape.ToOwned());
		const auto result = AddPermute(subgraph, { input, 0 }, std::move(permutation));
		subgraph.SetResults({ result });
		return graph.AddSubgraph(std::move(subgraph));
	}

	// 便捷的二维转置：交换最后两个维度
	inline NodeOutput AddTranspose(Subgraph& subgraph, NodeOutput input)
	{
		const auto info = subgraph.GetOutputInfo(input);
		const auto rank = info.shape.size();
		if (rank < 2)
		{
			throw std::runtime_error("Transpose requires rank >= 2");
		}
		std::vector<std::size_t> permutation(rank);
		std::iota(permutation.begin(), permutation.end(), 0uz);
		std::swap(permutation[rank - 1], permutation[rank - 2]);
		return AddPermute(subgraph, input, std::move(permutation));
	}
} // namespace LiteNN::Layer

#endif
