#include <LiteNN/Graph.h>

#include <cstring>
#include <cstdint>
#include <format>
#include <span>
#include <stdexcept>

#ifndef LITENN_OPTIMIZER_OPTIMIZERUTILS_H
#define LITENN_OPTIMIZER_OPTIMIZERUTILS_H

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

	inline std::size_t ElementByteSize(DataType dtype)
	{
		switch (dtype)
		{
		case DataType::Float32:
			return sizeof(float);
		case DataType::Float64:
			return sizeof(double);
		case DataType::Int32:
			return sizeof(std::int32_t);
		case DataType::Int64:
			return sizeof(std::int64_t);
		case DataType::Bool:
			return sizeof(bool);
		}
		throw std::runtime_error("Invalid data type");
	}

	inline void ZeroTensor(Tensor<PolymorphicDevice>& tensor)
	{
		if (!tensor.CurDevice().template Is<CPU>())
		{
			throw std::runtime_error("CPU training utilities currently require CPU-backed tensors");
		}
		std::memset(tensor.RawData(), 0, tensor.AllocatedNumElements() * ElementByteSize(tensor.DType()));
	}
} // namespace LiteNN::Optimizer::Detail

namespace LiteNN::Optimizer
{
	inline std::size_t InferInputGradientCount(const Graph& graph)
	{
		return Detail::InferInputGradientCount(graph);
	}

	inline void ZeroGradients(Graph& graph)
	{
		for (std::size_t variableIndex = 0; variableIndex < graph.VariableCount(); ++variableIndex)
		{
			Detail::ZeroTensor(graph.GetVariable(variableIndex)->Grad());
		}
	}

	inline void StoreVariableGradients(Graph& graph, std::span<const Tensor<CPU>> backwardResults,
	                                   std::size_t inputGradientCount)
	{
		Detail::ValidateBackwardResults(graph, backwardResults, inputGradientCount);
		for (std::size_t variableIndex = 0; variableIndex < graph.VariableCount(); ++variableIndex)
		{
			auto& variable = graph.GetVariable(variableIndex)->Data();
			auto& targetGrad = graph.GetVariable(variableIndex)->Grad();
			const auto& gradient = Detail::VariableGradient(backwardResults, inputGradientCount, variableIndex);
			Detail::ValidateVariableGradient(variable, gradient, variableIndex);
			DeviceTraits<PolymorphicDevice>::CopyFromCPU(targetGrad.CurDevice(), targetGrad.DType(), targetGrad.RawData(),
			                                             gradient.DType(), gradient.RawData(), gradient.NumElements());
		}
	}

	inline void StoreVariableGradients(Graph& graph, std::span<const Tensor<CPU>> backwardResults)
	{
		StoreVariableGradients(graph, backwardResults, InferInputGradientCount(graph));
	}
} // namespace LiteNN::Optimizer

#endif
