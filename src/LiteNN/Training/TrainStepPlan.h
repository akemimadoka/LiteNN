#ifndef LITENN_TRAINING_TRAIN_STEP_PLAN_H
#define LITENN_TRAINING_TRAIN_STEP_PLAN_H

#include <LiteNN/ExecutablePlan.h>
#include <LiteNN/Runtime/Scheduler.h>

#include <cstddef>
#include <format>
#include <optional>
#include <stdexcept>
#include <string>
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
		std::vector<NodeOutput> parameters;
		std::vector<NodeOutput> gradients;
		std::vector<NodeOutput> optimizerStateInputs;
		std::vector<NodeOutput> updatedParameters;
		std::vector<NodeOutput> updatedOptimizerStates;
	};

	struct TrainStepPlan
	{
		ExecutableModule module;
		FunctionId forwardFunction{};
		std::optional<FunctionId> backwardFunction;
		std::vector<OptimizerUpdateSpec> updates;
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
	}
} // namespace LiteNN::Training

#endif
