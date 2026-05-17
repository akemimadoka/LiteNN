#include <LiteNN/Graph.h>
#include <LiteNN/Layer/LayerUtils.h>
#include <LiteNN/Layer/Normalization.h>

#include <stdexcept>
#include <vector>

#ifndef LITENN_LAYER_GROUPNORM_H
#define LITENN_LAYER_GROUPNORM_H

namespace LiteNN::Layer
{
	// 匹配 ggml_group_norm 语义：rank == 4 时最后一维视为 batch，
	// 其余维度展平后按连续 chunk 划分为 numGroups 个 group。
	inline NodeOutput AddGroupNorm(Subgraph& subgraph, NodeOutput input, std::size_t numGroups, double eps = 1e-5)
	{
		const auto info = subgraph.GetOutputInfo(input);
		if (info.shape.empty() || info.shape.size() > 4)
		{
			throw std::runtime_error("GroupNorm input rank must be between 1 and 4");
		}
		if (numGroups == 0)
		{
			throw std::runtime_error("GroupNorm requires numGroups > 0");
		}
		if (!IsFloatingDataType(info.dtype))
		{
			throw std::runtime_error("GroupNorm requires a floating-point input tensor");
		}

		const auto batch = info.shape.size() == 4 ? info.shape[3] : 1uz;
		auto groupedVolume = 1uz;
		for (std::size_t axis = 0; axis < std::min<std::size_t>(info.shape.size(), 3); ++axis)
		{
			groupedVolume *= info.shape[axis];
		}
		if (groupedVolume % numGroups != 0)
		{
			throw std::runtime_error("GroupNorm requires grouped element count to be divisible by numGroups");
		}

		(void)batch;
		return AddNormalization(subgraph, input, NormalizationMode::GroupNorm, 0, eps, std::nullopt, std::nullopt,
		                        numGroups);
	}

	inline SubgraphId BuildGroupNorm(Graph& graph, DataType dtype, ShapeView shape, std::size_t numGroups,
	                                double eps = 1e-5)
	{
		Subgraph subgraph;
		const auto input = subgraph.AddParam(dtype, shape.ToOwned());
		const auto result = AddGroupNorm(subgraph, { input, 0 }, numGroups, eps);
		subgraph.SetResults({ result });
		return graph.AddSubgraph(std::move(subgraph));
	}
} // namespace LiteNN::Layer

#endif
