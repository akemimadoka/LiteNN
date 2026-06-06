#include <LiteNN/Graph.h>
#include <LiteNN/Layer/LayerUtils.h>
#include <LiteNN/Layer/Normalization.h>
#include <LiteNN/ModelBuilder.h>

#include <stdexcept>
#include <utility>
#include <vector>

#ifndef LITENN_LAYER_RMSNORM_H
#define LITENN_LAYER_RMSNORM_H

namespace LiteNN::Layer
{
	// RMSNorm 层描述符，持有缩放权重变量索引
	// 输入形状为 [*, featureSize]，在最后一维（axis = rank-1）做 root-mean-square 归一化
	// weight 形状为 [1, featureSize]，与最后两个维度广播兼容
	struct RMSNormLayer
	{
		std::size_t weightVariable{};
		std::size_t featureSize{};
		DataType dtype{ DataType::Float32 };
		double eps{ 1e-6 };
	};

	namespace Detail
	{
		inline RMSNormLayer CreateRMSNormImpl(Graph& graph, std::size_t featureSize,
		                                      DataType dtype, double eps)
		{
			RMSNormLayer layer;
			layer.featureSize = featureSize;
			layer.dtype = dtype;
			layer.eps = eps;
			layer.weightVariable =
			    graph.AddVariable(Variable::Create(Detail::MakeFilledTensor({ 1, featureSize }, dtype, 1.0)));
			return layer;
		}
	} // namespace Detail

	inline RMSNormLayer CreateRMSNorm(ModelBuilder& builder, std::size_t featureSize,
	                                  DataType dtype = DataType::Float32, double eps = 1e-6)
	{
		return Detail::CreateRMSNormImpl(builder.UnsafeMutableGraph(), featureSize, dtype, eps);
	}

	// 在已有子图中追加 RMSNorm 节点（在最后一个轴上归一化）
	// input 的形状必须是 2D：[batch, featureSize]
	inline NodeOutput AddRMSNorm(Subgraph& subgraph, const RMSNormLayer& layer, NodeOutput input)
	{
		const auto info = subgraph.GetOutputInfo(input); // copy
		if (info.shape.size() != 2 || info.shape[1] != layer.featureSize || info.dtype != layer.dtype)
		{
			throw std::runtime_error(
			    "RMSNorm input must be 2D with shape [batch, featureSize] and matching dtype");
		}

		const std::size_t features = info.shape[1];
		const std::vector<std::size_t> paramShape{ 1, features };
		const auto weight = subgraph.AddNode(VariableRefNode{ layer.weightVariable },
		                                    { OutputInfo{ layer.dtype, paramShape } });
		return AddNormalization(subgraph, input, NormalizationMode::RMSNorm, 1, layer.eps,
		                        NodeOutput{ weight, 0 });
	}

	namespace Detail
	{
		inline SubgraphId BuildRMSNormImpl(Graph& graph, const RMSNormLayer& layer, std::size_t batchSize)
		{
			Subgraph subgraph;
			const auto input = subgraph.AddParam(layer.dtype, { batchSize, layer.featureSize });
			const auto result = AddRMSNorm(subgraph, layer, { input, 0 });
			subgraph.SetResults({ result });
			return graph.AddSubgraph(std::move(subgraph));
		}
	} // namespace Detail

	inline SubgraphId BuildRMSNorm(ModelBuilder& builder, const RMSNormLayer& layer, std::size_t batchSize = 1)
	{
		return Detail::BuildRMSNormImpl(builder.UnsafeMutableGraph(), layer, batchSize);
	}
} // namespace LiteNN::Layer

#endif
