#ifndef LITENN_OPTIMIZER_GRAPHOPS_H
#define LITENN_OPTIMIZER_GRAPHOPS_H

#include <LiteNN/ComputePrimitives.h>
#include <LiteNN/Graph.h>

#include <optional>
#include <stdexcept>
#include <utility>
#include <vector>

namespace LiteNN::Optimizer
{
	inline std::vector<NodeOutput> AddSGDStep(Subgraph& subgraph, NodeOutput parameter, NodeOutput gradient,
	                                          std::optional<NodeOutput> velocity, double learningRate,
	                                          double momentum = 0.0, double weightDecay = 0.0,
	                                          bool nesterov = false)
	{
		const auto parameterInfo = subgraph.GetOutputInfo(parameter);
		const auto gradientInfo = subgraph.GetOutputInfo(gradient);
		if (parameterInfo.dtype != DataType::Float32 || gradientInfo.dtype != DataType::Float32)
		{
			throw std::runtime_error("SGDStep currently supports Float32 tensors only");
		}
		::LiteNN::Detail::ValidateOptimizerStepShape(parameterInfo.shape, gradientInfo.shape, "SGDStep gradient");
		if (velocity)
		{
			const auto velocityInfo = subgraph.GetOutputInfo(*velocity);
			if (velocityInfo.dtype != DataType::Float32)
			{
				throw std::runtime_error("SGDStep velocity must be Float32");
			}
			::LiteNN::Detail::ValidateOptimizerStepShape(parameterInfo.shape, velocityInfo.shape, "SGDStep velocity");
		}
		const auto outputCount = ::LiteNN::Detail::SGDStepOutputCount(velocity.has_value(), momentum);
		std::vector<OutputInfo> outputs;
		outputs.reserve(outputCount);
		outputs.push_back({ DataType::Float32, parameterInfo.shape });
		if (outputCount == 2)
		{
			outputs.push_back({ DataType::Float32, parameterInfo.shape });
		}
		const auto node = subgraph.AddNode(
		    SGDStepNode{ parameter, gradient, velocity, learningRate, momentum, weightDecay, nesterov },
		    std::move(outputs));
		std::vector<NodeOutput> result;
		result.reserve(outputCount);
		for (auto port = 0uz; port < outputCount; ++port)
		{
			result.push_back({ node, port });
		}
		return result;
	}

	inline std::vector<NodeOutput> AddAdamWStep(Subgraph& subgraph, NodeOutput parameter, NodeOutput gradient,
	                                           NodeOutput firstMoment, NodeOutput secondMoment,
	                                           double learningRate = 1e-3, double beta1 = 0.9,
	                                           double beta2 = 0.999, double epsilon = 1e-8,
	                                           double weightDecay = 0.01, std::size_t step = 1)
	{
		const auto parameterInfo = subgraph.GetOutputInfo(parameter);
		const auto gradientInfo = subgraph.GetOutputInfo(gradient);
		const auto firstMomentInfo = subgraph.GetOutputInfo(firstMoment);
		const auto secondMomentInfo = subgraph.GetOutputInfo(secondMoment);
		if (parameterInfo.dtype != DataType::Float32 || gradientInfo.dtype != DataType::Float32 ||
		    firstMomentInfo.dtype != DataType::Float32 || secondMomentInfo.dtype != DataType::Float32)
		{
			throw std::runtime_error("AdamWStep currently supports Float32 tensors only");
		}
		::LiteNN::Detail::ValidateOptimizerStepShape(parameterInfo.shape, gradientInfo.shape, "AdamWStep gradient");
		::LiteNN::Detail::ValidateOptimizerStepShape(parameterInfo.shape, firstMomentInfo.shape, "AdamWStep firstMoment");
		::LiteNN::Detail::ValidateOptimizerStepShape(parameterInfo.shape, secondMomentInfo.shape, "AdamWStep secondMoment");
		const auto node = subgraph.AddNode(
		    AdamWStepNode{ parameter, gradient, firstMoment, secondMoment, learningRate, beta1, beta2, epsilon,
		                   weightDecay, step },
		    { OutputInfo{ DataType::Float32, parameterInfo.shape },
		      OutputInfo{ DataType::Float32, parameterInfo.shape },
		      OutputInfo{ DataType::Float32, parameterInfo.shape } });
		return { NodeOutput{ node, 0 }, NodeOutput{ node, 1 }, NodeOutput{ node, 2 } };
	}
} // namespace LiteNN::Optimizer

#endif
