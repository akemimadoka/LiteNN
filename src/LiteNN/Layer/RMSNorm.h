#include <LiteNN/Graph.h>
#include <LiteNN/Layer/LayerUtils.h>

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

	// 创建 RMSNorm 层，weight 初始化为 1
	inline RMSNormLayer CreateRMSNorm(Graph& graph, std::size_t featureSize,
	                                 DataType dtype = DataType::Float32, double eps = 1e-6)
	{
		RMSNormLayer layer;
		layer.featureSize = featureSize;
		layer.dtype = dtype;
		layer.eps = eps;
		layer.weightVariable =
		    graph.AddVariable(Variable::Create(Detail::MakeFilledTensor({ 1, featureSize }, dtype, 1.0)));
		return layer;
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

		const std::size_t batch = info.shape[0];
		const std::size_t features = info.shape[1];
		constexpr std::size_t kNormAxis = 1;

		const std::vector<std::size_t> reducedShape{ batch };
		const std::vector<std::size_t> broadcastShape{ batch, 1 };
		const std::vector<std::size_t> paramShape{ 1, features };

		const auto xSquared =
		    subgraph.AddNode(BinaryOpNode{ BinaryOp::Multiply, input, input }, { OutputInfo{ info.dtype, info.shape } });

		const auto meanSquare = subgraph.AddNode(ReduceOpNode{ ReduceOp::Mean, { xSquared, 0 }, kNormAxis },
		                                     { OutputInfo{ info.dtype, reducedShape } });
		const auto meanSquareBc = subgraph.AddNode(ReshapeNode{ { meanSquare, 0 }, broadcastShape },
		                                      { OutputInfo{ info.dtype, broadcastShape } });

		const auto epsC =
		    Detail::AddConstant(subgraph, Detail::MakeFilledTensor(broadcastShape, info.dtype, layer.eps));
		const auto meanSquarePlusEps =
		    subgraph.AddNode(BinaryOpNode{ BinaryOp::Add, { meanSquareBc, 0 }, { epsC, 0 } },
		                     { OutputInfo{ info.dtype, broadcastShape } });
		const auto rms = subgraph.AddNode(UnaryOpNode{ UnaryOp::Sqrt, { meanSquarePlusEps, 0 } },
		                                 { OutputInfo{ info.dtype, broadcastShape } });

		const auto normalized =
		    subgraph.AddNode(BinaryOpNode{ BinaryOp::Divide, input, { rms, 0 } }, { OutputInfo{ info.dtype, info.shape } });

		const auto weight = subgraph.AddNode(VariableRefNode{ layer.weightVariable },
		                                    { OutputInfo{ layer.dtype, paramShape } });
		const auto result =
		    subgraph.AddNode(BinaryOpNode{ BinaryOp::Multiply, { weight, 0 }, { normalized, 0 } },
		                     { OutputInfo{ info.dtype, info.shape } });

		return { result, 0 };
	}

	inline SubgraphId BuildRMSNorm(Graph& graph, const RMSNormLayer& layer, std::size_t batchSize = 1)
	{
		Subgraph subgraph;
		const auto input = subgraph.AddParam(layer.dtype, { batchSize, layer.featureSize });
		const auto result = AddRMSNorm(subgraph, layer, { input, 0 });
		subgraph.SetResults({ result });
		return graph.AddSubgraph(std::move(subgraph));
	}
} // namespace LiteNN::Layer

#endif