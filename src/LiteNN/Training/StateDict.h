#ifndef LITENN_TRAINING_STATE_DICT_H
#define LITENN_TRAINING_STATE_DICT_H

#include <LiteNN/Graph.h>

#include <format>
#include <span>
#include <stdexcept>
#include <string>
#include <string_view>
#include <vector>

namespace LiteNN::Training
{
	inline TensorMemorySpace ParameterMemorySpace(const PolymorphicDevice& device)
	{
		return device.Is<CPU>() ? TensorMemorySpace::Host : TensorMemorySpace::Device;
	}

	struct ParameterBinding
	{
		std::string name;
		TensorType type;
		Tensor<PolymorphicDevice>* parameter{};
		Tensor<PolymorphicDevice>* gradient{};

		Tensor<PolymorphicDevice>& Parameter() const
		{
			if (!parameter)
			{
				throw std::runtime_error("ParameterBinding has no parameter tensor: " + name);
			}
			return *parameter;
		}

		Tensor<PolymorphicDevice>& Gradient() const
		{
			if (!gradient)
			{
				throw std::runtime_error("ParameterBinding has no gradient tensor: " + name);
			}
			return *gradient;
		}
	};

	class ParameterSet
	{
	public:
		ParameterSet() = default;

		static ParameterSet BindGraph(Graph& graph)
		{
			ParameterSet set;
			set.entries_.reserve(graph.VariableCount());
			for (std::size_t i = 0; i < graph.VariableCount(); ++i)
			{
				auto variable = graph.GetVariable(i);
				auto& parameter = variable->Data();
				set.entries_.push_back({
				    .name = graph.VariableName(i).empty() ? std::format("parameter.{}", i) : graph.VariableName(i),
				    .type = TensorType::Dense(parameter.DType(), parameter.Shape(),
				                              ParameterMemorySpace(parameter.CurDevice())),
				    .parameter = &parameter,
				    .gradient = &variable->Grad(),
				});
			}
			return set;
		}

		std::size_t Size() const noexcept
		{
			return entries_.size();
		}

		bool Empty() const noexcept
		{
			return entries_.empty();
		}

		ParameterBinding& operator[](std::size_t index)
		{
			return entries_.at(index);
		}

		const ParameterBinding& operator[](std::size_t index) const
		{
			return entries_.at(index);
		}

		std::span<ParameterBinding> Entries()
		{
			return entries_;
		}

		std::span<const ParameterBinding> Entries() const
		{
			return entries_;
		}

	private:
		std::vector<ParameterBinding> entries_;
	};

	struct StateDictEntry
	{
		std::string name;
		TensorType type;
		Tensor<CPU> value;
	};

	struct StateDict
	{
		std::vector<StateDictEntry> parameters;

		const StateDictEntry* Find(std::string_view name) const
		{
			for (const auto& entry : parameters)
			{
				if (entry.name == name)
				{
					return &entry;
				}
			}
			return nullptr;
		}
	};

	inline StateDict SaveStateDict(const ParameterSet& parameters)
	{
		StateDict state;
		state.parameters.reserve(parameters.Size());
		for (const auto& entry : parameters.Entries())
		{
			state.parameters.push_back(
			    { .name = entry.name, .type = entry.type, .value = entry.Parameter().CopyToDevice(CPU{}) });
		}
		return state;
	}

	inline void LoadStateDict(ParameterSet& parameters, const StateDict& state)
	{
		for (const auto& binding : parameters.Entries())
		{
			const auto* entry = state.Find(binding.name);
			if (!entry)
			{
				throw std::runtime_error("StateDict is missing parameter: " + binding.name);
			}
			if (entry->type != binding.type)
			{
				throw std::runtime_error("StateDict parameter type mismatch: " + binding.name);
			}
			auto& parameter = binding.Parameter();
			DeviceTraits<PolymorphicDevice>::CopyFromCPU(parameter.CurDevice(), parameter.DType(),
			                                             parameter.UnsafeRawData(), entry->value.DType(),
			                                             entry->value.UnsafeRawData(), entry->value.NumElements());
		}
	}
} // namespace LiteNN::Training

#endif
