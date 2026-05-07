#ifndef LITENN_OPTIMIZER_OPTIMIZERUTILS_H
#define LITENN_OPTIMIZER_OPTIMIZERUTILS_H

#include <LiteNN/Graph.h>

#include <format>
#include <span>
#include <stdexcept>

namespace LiteNN::Optimizer::Detail
{
	inline std::size_t InferInputGradientCount(const Graph& graph)
	{
		const auto backwardId = graph.Backward();
		if (!backwardId)
		{
			throw std::runtime_error("Graph has no backward subgraph");
		}

		const auto outputGradientCount = graph.GetSubgraph(graph.Forward()).Results().size();
		const auto backwardParamCount = graph.GetSubgraph(*backwardId).Params().size();
		if (backwardParamCount < outputGradientCount)
		{
			throw std::runtime_error("Backward subgraph parameter metadata is inconsistent");
		}
		return backwardParamCount - outputGradientCount;
	}

	inline void ValidateBackwardResults(const Graph& graph, std::span<const Tensor<CPU>> backwardResults,
	                                    std::size_t inputGradientCount)
	{
		if (backwardResults.size() < inputGradientCount + graph.VariableCount())
		{
			throw std::runtime_error("Backward result count does not include all variable gradients");
		}
	}

	inline void ValidateVariableGradient(const Tensor<PolymorphicDevice>& variable, const Tensor<CPU>& gradient,
	                                     std::size_t variableIndex)
	{
		if (!variable.CurDevice().template Is<CPU>())
		{
			throw std::runtime_error("CPU optimizers currently require CPU-backed variables");
		}
		if (variable.DType() != DataType::Float32 || gradient.DType() != DataType::Float32 ||
		    variable.NumElements() != gradient.NumElements())
		{
			throw std::runtime_error(std::format(
			    "Unexpected gradient for variable {}: variable dtype={}, elements={}; gradient dtype={}, elements={}",
			    variableIndex, static_cast<int>(variable.DType()), variable.NumElements(), static_cast<int>(gradient.DType()),
			    gradient.NumElements()));
		}
	}

	inline const Tensor<CPU>& VariableGradient(std::span<const Tensor<CPU>> backwardResults, std::size_t inputGradientCount,
	                                          std::size_t variableIndex)
	{
		return backwardResults[inputGradientCount + variableIndex];
	}
} // namespace LiteNN::Optimizer::Detail

namespace LiteNN::Optimizer
{
	inline std::size_t InferInputGradientCount(const Graph& graph)
	{
		return Detail::InferInputGradientCount(graph);
	}
} // namespace LiteNN::Optimizer

#endif
