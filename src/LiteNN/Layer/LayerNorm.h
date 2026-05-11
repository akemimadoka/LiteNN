#include <LiteNN/Graph.h>
#include <LiteNN/Layer/LayerUtils.h>

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

		const std::size_t batch = info.shape[0];
		const std::size_t features = info.shape[1];
		constexpr std::size_t kNormAxis = 1;

		// 归约后的形状：[batch]（ReduceOp 移除 axis 维度）
		const std::vector<std::size_t> reducedShape{ batch };
		// 广播用形状：[batch, 1]
		const std::vector<std::size_t> broadcastShape{ batch, 1 };
		// 归一化参数形状：[1, features]
		const std::vector<std::size_t> paramShape{ 1, features };

		// --- 1. 均值 ---
		const auto mean = subgraph.AddNode(ReduceOpNode{ ReduceOp::Mean, input, kNormAxis },
		                                   { OutputInfo{ info.dtype, reducedShape } });
		const auto meanBc = subgraph.AddNode(ReshapeNode{ { mean, 0 }, broadcastShape },
		                                     { OutputInfo{ info.dtype, broadcastShape } });

		// --- 2. 中心化：x - mean ---
		const auto xCentered = subgraph.AddNode(BinaryOpNode{ BinaryOp::Subtract, input, { meanBc, 0 } },
		                                        { OutputInfo{ info.dtype, info.shape } });

		// --- 3. 方差：mean((x - mean)²) ---
		const auto xCenteredSq =
		    subgraph.AddNode(BinaryOpNode{ BinaryOp::Multiply, { xCentered, 0 }, { xCentered, 0 } },
		                     { OutputInfo{ info.dtype, info.shape } });
		const auto variance = subgraph.AddNode(ReduceOpNode{ ReduceOp::Mean, { xCenteredSq, 0 }, kNormAxis },
		                                       { OutputInfo{ info.dtype, reducedShape } });
		const auto varianceBc = subgraph.AddNode(ReshapeNode{ { variance, 0 }, broadcastShape },
		                                         { OutputInfo{ info.dtype, broadcastShape } });

		// --- 4. 标准差：sqrt(variance + eps) ---
		const auto epsC =
		    Detail::AddConstant(subgraph, Detail::MakeFilledTensor(broadcastShape, info.dtype, layer.eps));
		const auto varPlusEps =
		    subgraph.AddNode(BinaryOpNode{ BinaryOp::Add, { varianceBc, 0 }, { epsC, 0 } },
		                     { OutputInfo{ info.dtype, broadcastShape } });
		const auto stdDev =
		    subgraph.AddNode(UnaryOpNode{ UnaryOp::Sqrt, { varPlusEps, 0 } }, { OutputInfo{ info.dtype, broadcastShape } });

		// --- 5. 归一化 ---
		const auto xNorm = subgraph.AddNode(BinaryOpNode{ BinaryOp::Divide, { xCentered, 0 }, { stdDev, 0 } },
		                                    { OutputInfo{ info.dtype, info.shape } });

		// --- 6. 仿射变换：gamma * xNorm + beta ---
		const auto gamma = subgraph.AddNode(VariableRefNode{ layer.gammaVariable },
		                                    { OutputInfo{ layer.dtype, paramShape } });
		const auto beta = subgraph.AddNode(VariableRefNode{ layer.betaVariable },
		                                   { OutputInfo{ layer.dtype, paramShape } });

		const auto scaled = subgraph.AddNode(BinaryOpNode{ BinaryOp::Multiply, { gamma, 0 }, { xNorm, 0 } },
		                                     { OutputInfo{ info.dtype, info.shape } });
		const auto result = subgraph.AddNode(BinaryOpNode{ BinaryOp::Add, { scaled, 0 }, { beta, 0 } },
		                                     { OutputInfo{ info.dtype, info.shape } });

		return { result, 0 };
	}
} // namespace LiteNN::Layer

#endif
