#ifndef LITENN_TRAINING_TRAIN_STEP_PLAN_H
#define LITENN_TRAINING_TRAIN_STEP_PLAN_H

#include <LiteNN/ExecutablePlan.h>
#include <LiteNN/Misc.h>
#include <LiteNN/Runtime/Scheduler.h>

#include <cstddef>
#include <format>
#include <optional>
#include <ranges>
#include <span>
#include <stdexcept>
#include <string>
#include <string_view>
#include <variant>
#include <vector>

namespace LiteNN::Training
{
	enum class TrainExecutionPolicy
	{
		Interpreter,
		AOT,
		Auto
	};

	struct OptimizerUpdateSpec
	{
		std::string name;
		std::string opKind;
		SubgraphId subgraph{};
		std::vector<NodeOutput> parameters;
		std::vector<NodeOutput> gradients;
		std::vector<NodeOutput> optimizerStateInputs;
		std::vector<NodeOutput> updatedParameters;
		std::vector<NodeOutput> updatedOptimizerStates;
	};

	enum class TrainStepABIRole
	{
		SavedActivation,
		MutableParameter,
		Gradient,
		OptimizerState,
		LossInput,
		UpdatedParameter,
		UpdatedOptimizerState
	};

	inline std::string_view TrainStepABIRoleName(TrainStepABIRole role) noexcept
	{
		return EnumToString<EnumToStringStyle::Unqualified>(role);
	}

	struct TrainStepABIBinding
	{
		std::string name;
		TrainStepABIRole role{ TrainStepABIRole::MutableParameter };
		TensorType type;
		BufferMutability mutability{ BufferMutability::Mutable };
		std::vector<NodeOutput> values;
		std::optional<std::size_t> runtimeState;
	};

	enum class TrainStepArtifactEntryKind
	{
		Forward,
		Backward,
		Loss,
		OptimizerUpdate
	};

	inline std::string_view TrainStepArtifactEntryKindName(TrainStepArtifactEntryKind kind) noexcept
	{
		return EnumToString<EnumToStringStyle::Unqualified>(kind);
	}

	struct TrainStepArtifactEntry
	{
		std::string name;
		TrainStepArtifactEntryKind kind{ TrainStepArtifactEntryKind::Forward };
		std::optional<FunctionId> function;
		std::optional<std::size_t> update;
		std::vector<std::size_t> inputBindings;
		std::vector<std::size_t> outputBindings;
	};

	struct TrainStepPlan
	{
		ExecutableModule module;
		FunctionId forwardFunction{};
		std::optional<FunctionId> backwardFunction;
		std::vector<OptimizerUpdateSpec> updates;
		std::vector<TrainStepABIBinding> abiBindings;
		std::vector<TrainStepArtifactEntry> artifactEntries;
		std::vector<Runtime::RuntimeStateBinding> runtimeStates;
		Runtime::RuntimeSchedule schedule;
		TrainExecutionPolicy policy{ TrainExecutionPolicy::Auto };
	};

	inline TrainExecutionPolicy ResolveTrainExecutionPolicy(TrainExecutionPolicy requested, bool hasAOTBackend)
	{
		if (requested != TrainExecutionPolicy::Auto)
		{
			return requested;
		}
		return hasAOTBackend ? TrainExecutionPolicy::AOT : TrainExecutionPolicy::Interpreter;
	}

	inline std::vector<OptimizerUpdateSpec> CollectOptimizerUpdates(const ExecutablePlan& plan)
	{
		std::vector<OptimizerUpdateSpec> updates;
		for (const auto& subgraph : plan.subgraphs)
		{
			for (const auto& node : subgraph.nodes)
			{
				if (!std::holds_alternative<SGDStepNode>(node.node) &&
				    !std::holds_alternative<AdamWStepNode>(node.node))
				{
					continue;
				}
				OptimizerUpdateSpec update;
				update.name = std::format("update.{}.{}", subgraph.sourceSubgraph, node.sourceNode);
				update.opKind = node.opKind;
				update.subgraph = subgraph.sourceSubgraph;
				if (!node.inputs.empty())
				{
					update.parameters.push_back(node.inputs[0]);
				}
				if (node.inputs.size() > 1)
				{
					update.gradients.push_back(node.inputs[1]);
				}
				for (std::size_t i = 2; i < node.inputs.size(); ++i)
				{
					update.optimizerStateInputs.push_back(node.inputs[i]);
				}
				for (std::size_t i = 0; i < node.outputs.size(); ++i)
				{
					const NodeOutput output{ node.sourceNode, i };
					if (i == 0)
					{
						update.updatedParameters.push_back(output);
					}
					else
					{
						update.updatedOptimizerStates.push_back(output);
					}
				}
				updates.push_back(std::move(update));
			}
		}
		return updates;
	}

	inline const TensorType& RequirePlanValueType(const ExecutablePlan& plan, SubgraphId sourceSubgraph,
	                                              NodeOutput value)
	{
		const auto subgraphIt = std::ranges::find_if(plan.subgraphs, [&](const ExecutablePlanSubgraph& subgraph) {
			return subgraph.sourceSubgraph == sourceSubgraph;
		});
		if (subgraphIt == plan.subgraphs.end())
		{
			throw std::runtime_error(std::format("TrainStep ABI references unknown subgraph {}", sourceSubgraph));
		}
		const auto nodeIt = std::ranges::find_if(subgraphIt->nodes, [&](const ExecutablePlanNode& node) {
			return node.sourceNode == value.node;
		});
		if (nodeIt == subgraphIt->nodes.end() || value.port >= nodeIt->outputs.size())
		{
			throw std::runtime_error(std::format(
			    "TrainStep ABI references unknown value {}:{} in subgraph {}", value.node, value.port,
			    sourceSubgraph));
		}
		return nodeIt->outputs[value.port];
	}

	inline std::vector<TrainStepABIBinding> BuildTrainStepABIBindings(
	    const ExecutableModule& module,
	    std::span<const Runtime::RuntimeStateBinding> runtimeStates,
	    std::span<const OptimizerUpdateSpec> updates)
	{
		const auto& plan = module.plan;
		std::vector<TrainStepABIBinding> bindings;
		bindings.reserve(plan.activationSlots.size() + plan.variables.size() * 2 + plan.outputs.size() +
		                 updates.size() * 4);

		for (std::size_t i = 0; i < plan.activationSlots.size(); ++i)
		{
			bindings.push_back({ .name = std::format("activation.{}", i),
			                     .role = TrainStepABIRole::SavedActivation,
			                     .type = plan.activationSlots[i],
			                     .mutability = BufferMutability::Mutable,
			                     .runtimeState = i });
		}
		const auto parameterStateOffset = plan.activationSlots.size();
		for (std::size_t i = 0; i < plan.variables.size(); ++i)
		{
			const auto name = plan.variables[i].region.name.empty() ? std::format("parameter.{}", i)
			                                                        : plan.variables[i].region.name;
			bindings.push_back({ .name = name,
			                     .role = TrainStepABIRole::MutableParameter,
			                     .type = plan.variables[i].type,
			                     .mutability = BufferMutability::Mutable,
			                     .runtimeState = parameterStateOffset + i });
			bindings.push_back({ .name = name + ".grad",
			                     .role = TrainStepABIRole::Gradient,
			                     .type = plan.variables[i].type,
			                     .mutability = BufferMutability::Mutable });
		}
		for (const auto& output : plan.outputs)
		{
			bindings.push_back({ .name = output.name.empty() ? "loss.input" : std::format("loss.input.{}", output.name),
			                     .role = TrainStepABIRole::LossInput,
			                     .type = output.type,
			                     .mutability = BufferMutability::Immutable,
			                     .values = { output.source } });
		}
		for (const auto& update : updates)
		{
			for (std::size_t i = 0; i < update.optimizerStateInputs.size(); ++i)
			{
				const auto value = update.optimizerStateInputs[i];
				bindings.push_back({ .name = std::format("{}.optimizer_state.{}", update.name, i),
				                     .role = TrainStepABIRole::OptimizerState,
				                     .type = RequirePlanValueType(plan, update.subgraph, value),
				                     .mutability = BufferMutability::Mutable,
				                     .values = { value } });
			}
			for (std::size_t i = 0; i < update.updatedParameters.size(); ++i)
			{
				const auto value = update.updatedParameters[i];
				bindings.push_back({ .name = std::format("{}.updated_parameter.{}", update.name, i),
				                     .role = TrainStepABIRole::UpdatedParameter,
				                     .type = RequirePlanValueType(plan, update.subgraph, value),
				                     .mutability = BufferMutability::Mutable,
				                     .values = { value } });
			}
			for (std::size_t i = 0; i < update.updatedOptimizerStates.size(); ++i)
			{
				const auto value = update.updatedOptimizerStates[i];
				bindings.push_back({ .name = std::format("{}.updated_optimizer_state.{}", update.name, i),
				                     .role = TrainStepABIRole::UpdatedOptimizerState,
				                     .type = RequirePlanValueType(plan, update.subgraph, value),
				                     .mutability = BufferMutability::Mutable,
				                     .values = { value } });
			}
		}

		for (const auto& binding : bindings)
		{
			if (binding.runtimeState && *binding.runtimeState >= runtimeStates.size())
			{
				throw std::runtime_error("TrainStep ABI binding references an invalid runtime state: " +
				                         binding.name);
			}
		}
		return bindings;
	}

	inline std::vector<std::size_t> FindTrainStepBindingsByRole(
	    std::span<const TrainStepABIBinding> bindings, TrainStepABIRole role)
	{
		std::vector<std::size_t> indices;
		for (std::size_t i = 0; i < bindings.size(); ++i)
		{
			if (bindings[i].role == role)
			{
				indices.push_back(i);
			}
		}
		return indices;
	}

	inline std::vector<std::size_t> FindTrainStepBindingsByRoleAndPrefix(
	    std::span<const TrainStepABIBinding> bindings, TrainStepABIRole role, std::string_view prefix)
	{
		std::vector<std::size_t> indices;
		for (std::size_t i = 0; i < bindings.size(); ++i)
		{
			if (bindings[i].role == role && bindings[i].name.starts_with(prefix))
			{
				indices.push_back(i);
			}
		}
		return indices;
	}

	inline void AppendTrainStepBindings(std::vector<std::size_t>& destination, std::vector<std::size_t> source)
	{
		destination.insert(destination.end(), source.begin(), source.end());
	}

	inline std::vector<TrainStepArtifactEntry> BuildTrainStepArtifactEntries(
	    const ExecutableModule& module, const TrainStepPlan& plan)
	{
		std::vector<TrainStepArtifactEntry> entries;
		entries.reserve(2 + module.plan.outputs.size() + plan.updates.size());

		TrainStepArtifactEntry forward;
		forward.name = "forward";
		forward.kind = TrainStepArtifactEntryKind::Forward;
		forward.function = plan.forwardFunction;
		AppendTrainStepBindings(forward.outputBindings,
		                         FindTrainStepBindingsByRole(plan.abiBindings, TrainStepABIRole::LossInput));
		entries.push_back(std::move(forward));

		if (plan.backwardFunction)
		{
			TrainStepArtifactEntry backward;
			backward.name = "backward";
			backward.kind = TrainStepArtifactEntryKind::Backward;
			backward.function = plan.backwardFunction;
			AppendTrainStepBindings(backward.inputBindings,
			                         FindTrainStepBindingsByRole(plan.abiBindings, TrainStepABIRole::LossInput));
			AppendTrainStepBindings(backward.inputBindings,
			                         FindTrainStepBindingsByRole(plan.abiBindings, TrainStepABIRole::Gradient));
			AppendTrainStepBindings(backward.inputBindings,
			                         FindTrainStepBindingsByRole(plan.abiBindings, TrainStepABIRole::MutableParameter));
			AppendTrainStepBindings(backward.outputBindings,
			                         FindTrainStepBindingsByRole(plan.abiBindings, TrainStepABIRole::Gradient));
			entries.push_back(std::move(backward));
		}

		for (std::size_t outputIndex = 0; outputIndex < module.plan.outputs.size(); ++outputIndex)
		{
			TrainStepArtifactEntry loss;
			loss.name = outputIndex == 0 ? "loss" : std::format("loss.{}", outputIndex);
			loss.kind = TrainStepArtifactEntryKind::Loss;
			AppendTrainStepBindings(loss.inputBindings,
			                         FindTrainStepBindingsByRole(plan.abiBindings, TrainStepABIRole::LossInput));
			AppendTrainStepBindings(loss.outputBindings,
			                         FindTrainStepBindingsByRole(plan.abiBindings, TrainStepABIRole::Gradient));
			entries.push_back(std::move(loss));
		}

		for (std::size_t updateIndex = 0; updateIndex < plan.updates.size(); ++updateIndex)
		{
			const auto& update = plan.updates[updateIndex];
			TrainStepArtifactEntry updateEntry;
			updateEntry.name = update.name;
			updateEntry.kind = TrainStepArtifactEntryKind::OptimizerUpdate;
			updateEntry.update = updateIndex;
			AppendTrainStepBindings(updateEntry.inputBindings,
			                         FindTrainStepBindingsByRole(plan.abiBindings, TrainStepABIRole::MutableParameter));
			AppendTrainStepBindings(updateEntry.inputBindings,
			                         FindTrainStepBindingsByRole(plan.abiBindings, TrainStepABIRole::Gradient));
			AppendTrainStepBindings(updateEntry.inputBindings,
			                         FindTrainStepBindingsByRoleAndPrefix(
			                             plan.abiBindings, TrainStepABIRole::OptimizerState, update.name));
			AppendTrainStepBindings(updateEntry.outputBindings,
			                         FindTrainStepBindingsByRoleAndPrefix(
			                             plan.abiBindings, TrainStepABIRole::UpdatedParameter, update.name));
			AppendTrainStepBindings(updateEntry.outputBindings,
			                         FindTrainStepBindingsByRoleAndPrefix(
			                             plan.abiBindings, TrainStepABIRole::UpdatedOptimizerState, update.name));
			entries.push_back(std::move(updateEntry));
		}

		return entries;
	}

	inline TrainStepPlan BuildTrainStepPlan(ExecutableModule module,
	                                        TrainExecutionPolicy policy = TrainExecutionPolicy::Auto,
	                                        bool hasAOTBackend = false)
	{
		ValidateExecutablePlan(module.plan);
		TrainStepPlan train;
		train.forwardFunction = module.plan.forward;
		train.backwardFunction = module.plan.backward;
		train.updates = CollectOptimizerUpdates(module.plan);
		train.policy = ResolveTrainExecutionPolicy(policy, hasAOTBackend);

		for (std::size_t i = 0; i < module.plan.activationSlots.size(); ++i)
		{
			train.runtimeStates.push_back(Runtime::MakeTrainingState(
			    std::format("activation.{}", i), "saved-activation", module.plan.activationSlots[i]));
		}
		for (std::size_t i = 0; i < module.plan.variables.size(); ++i)
		{
			train.runtimeStates.push_back(Runtime::MakeTrainingState(
			    module.plan.variables[i].region.name.empty() ? std::format("parameter.{}", i)
			                                                 : module.plan.variables[i].region.name,
			    "mutable-parameter", module.plan.variables[i].type));
		}
		train.abiBindings = BuildTrainStepABIBindings(module, train.runtimeStates, train.updates);
		train.artifactEntries = BuildTrainStepArtifactEntries(module, train);
		train.schedule = Runtime::BuildRuntimeSchedule(module, train.runtimeStates);
		Runtime::ValidateRuntimeSchedule(train.schedule);
		train.module = std::move(module);
		return train;
	}

	inline void ValidateTrainStepPlan(const TrainStepPlan& plan)
	{
		ValidateExecutablePlan(plan.module.plan);
		Runtime::ValidateRuntimeSchedule(plan.schedule);
		if (plan.forwardFunction >= plan.module.functions.size())
		{
			throw std::runtime_error("TrainStepPlan forward function is out of range");
		}
		if (plan.backwardFunction && *plan.backwardFunction >= plan.module.functions.size())
		{
			throw std::runtime_error("TrainStepPlan backward function is out of range");
		}
		for (const auto& update : plan.updates)
		{
			if (update.opKind != "SGDStepNode" && update.opKind != "AdamWStepNode")
			{
				throw std::runtime_error("TrainStepPlan contains an unsupported optimizer update op: " +
				                         update.opKind);
			}
			if (update.parameters.empty() || update.gradients.empty() || update.updatedParameters.empty())
			{
				throw std::runtime_error("TrainStepPlan optimizer update is missing parameter/gradient/output");
			}
		}
		for (const auto& binding : plan.abiBindings)
		{
			if (binding.name.empty())
			{
				throw std::runtime_error("TrainStepPlan ABI binding name cannot be empty");
			}
			if (!binding.type.IsFullyStatic())
			{
				throw std::runtime_error("TrainStepPlan ABI binding must have a static tensor type: " +
				                         binding.name);
			}
			if (binding.runtimeState && *binding.runtimeState >= plan.runtimeStates.size())
			{
				throw std::runtime_error("TrainStepPlan ABI binding references an invalid runtime state: " +
				                         binding.name);
			}
		}
		for (std::size_t entryIndex = 0; entryIndex < plan.artifactEntries.size(); ++entryIndex)
		{
			const auto& entry = plan.artifactEntries[entryIndex];
			if (entry.name.empty())
			{
				throw std::runtime_error("TrainStepPlan artifact entry name cannot be empty");
			}
			if (entry.function && *entry.function >= plan.module.functions.size())
			{
				throw std::runtime_error("TrainStepPlan artifact entry references an invalid function: " +
				                         entry.name);
			}
			if (entry.update && *entry.update >= plan.updates.size())
			{
				throw std::runtime_error("TrainStepPlan artifact entry references an invalid optimizer update: " +
				                         entry.name);
			}
			for (const auto binding : entry.inputBindings)
			{
				if (binding >= plan.abiBindings.size())
				{
					throw std::runtime_error(std::format(
					    "TrainStepPlan artifact entry {} has invalid input binding {}", entry.name, binding));
				}
			}
			for (const auto binding : entry.outputBindings)
			{
				if (binding >= plan.abiBindings.size())
				{
					throw std::runtime_error(std::format(
					    "TrainStepPlan artifact entry {} has invalid output binding {}", entry.name, binding));
				}
			}
			for (std::size_t previous = 0; previous < entryIndex; ++previous)
			{
				if (plan.artifactEntries[previous].name == entry.name)
				{
					throw std::runtime_error("TrainStepPlan artifact entry name is duplicated: " + entry.name);
				}
			}
		}
	}
} // namespace LiteNN::Training

#endif
