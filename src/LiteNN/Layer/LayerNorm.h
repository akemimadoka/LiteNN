#include <LiteNN/Graph.h>
#include <LiteNN/Layer/LayerUtils.h>
#include <LiteNN/Layer/Normalization.h>

#include <stdexcept>
#include <utility>
#include <vector>

#ifndef LITENN_LAYER_LAYERNORM_H
#define LITENN_LAYER_LAYERNORM_H

namespace LiteNN::Layer
{
	// LayerNorm 层描述符，持有 gamma/beta 变量索引
	// 输入形状为 [*, featureSize]，在最后一维（axis = rank-1）归一化
	// gamma 和 bias 形状均为 [1, featureSize]，与最后两个维度广播兼容
	struct LayerNormLayer
	{
		std::size_t gammaVariable{};
		std::size_t betaVariable{};
		std::size_t featureSize{};
		DataType dtype{ DataType::Float32 };
		double eps{ 1e-5 };
	};

	// 创建 LayerNorm 层，gamma 初始化为 1，beta 初始化为 0
	inline LayerNormLayer CreateLayerNorm(Graph& graph, std::size_t featureSize,
	                                      DataType dtype = DataType::Float32, double eps = 1e-5)
	{
		LayerNormLayer layer;
		layer.featureSize = featureSize;
		layer.dtype = dtype;
		layer.eps = eps;
		layer.gammaVariable =
		    graph.AddVariable(Variable::Create(Detail::MakeFilledTensor({ 1, featureSize }, dtype, 1.0)));
		layer.betaVariable =
		    graph.AddVariable(Variable::Create(Detail::MakeFilledTensor({ 1, featureSize }, dtype, 0.0)));
		return layer;
	}

	// 在已有子图中追加 LayerNorm 节点（在最后一个轴上归一化）
	// input 的形状必须是 2D：[batch, featureSize]
	inline NodeOutput AddLayerNorm(Subgraph& subgraph, const LayerNormLayer& layer, NodeOutput input)
	{
		const auto info = subgraph.GetOutputInfo(input); // copy
		if (info.shape.size() != 2 || info.shape[1] != layer.featureSize || info.dtype != layer.dtype)
		{
			throw std::runtime_error(
			    "LayerNorm input must be 2D with shape [batch, featureSize] and matching dtype");
		}

		const std::size_t features = info.shape[1];
		// 归一化参数形状：[1, features]
		const std::vector<std::size_t> paramShape{ 1, features };
		const auto gamma = subgraph.AddNode(VariableRefNode{ layer.gammaVariable },
		                                    { OutputInfo{ layer.dtype, paramShape } });
		const auto beta = subgraph.AddNode(VariableRefNode{ layer.betaVariable },
		                                   { OutputInfo{ layer.dtype, paramShape } });
		return AddNormalization(subgraph, input, NormalizationMode::LayerNorm, 1, layer.eps,
		                        NodeOutput{ gamma, 0 }, NodeOutput{ beta, 0 });
	}
} // namespace LiteNN::Layer

#endif
