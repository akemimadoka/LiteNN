#ifndef LITENN_RUNTIME_SCHEDULER_H
#define LITENN_RUNTIME_SCHEDULER_H

#include <LiteNN/ExecutablePlan.h>
#include <LiteNN/MemoryPlan.h>
#include <LiteNN/Misc.h>
#include <LiteNN/Runtime/Placement.h>
#include <algorithm>
#include <cstddef>
#include <format>
#include <optional>
#include <span>
#include <stdexcept>
#include <string>
#include <string_view>
#include <utility>
#include <vector>

namespace LiteNN::Runtime
{
	enum class RuntimeStateKind
	{
		Generic,
		KVCache,
		Diffusion,
		Training,
		LoRAAdapter
	};

	struct RuntimeStateBinding
	{
		std::string name;
		RuntimeStateKind kind{ RuntimeStateKind::Generic };
		std::string role;
		TensorType type;
		BufferMutability mutability{ BufferMutability::Mutable };
		std::vector<std::string> effects;
		std::optional<std::size_t> memoryBuffer;
	};

	enum class RuntimeStateValueKind
	{
		FunctionInput,
		FunctionOutput
	};

	struct RuntimeStateValueBinding
	{
		std::string stateName;
		FunctionId function{};
		RuntimeStateValueKind kind{ RuntimeStateValueKind::FunctionInput };
		std::size_t valueIndex{};
		std::size_t stateByteOffset{};
	};

	struct RuntimeStateOutputAlias
	{
		std::size_t outputIndex{};
		std::size_t inputIndex{};
		std::string stateName;
		std::size_t stateByteOffset{};
	};

	struct RuntimeScheduleOutputProjection
	{
		std::size_t functionalOutputCount{};
		std::vector<std::size_t> publicOutputIndices;
		std::vector<TensorType> publicOutputTypes;
		std::vector<RuntimeStateOutputAlias> stateAliases;
	};

	struct LLMDecodeStateABI
	{
		std::vector<RuntimeStateBinding> kvCaches;
		std::optional<RuntimeStateBinding> currentPosition;
		std::optional<RuntimeStateBinding> batchMetadata;
		std::optional<RuntimeStateBinding> sequenceMetadata;
	};

	struct DiffusionExecutionABI
	{
		RuntimeStateBinding latent;
		std::optional<RuntimeStateBinding> timestepSchedule;
		std::optional<RuntimeStateBinding> conditioning;
		std::optional<RuntimeStateBinding> guidanceScale;
		std::optional<RuntimeStateBinding> vaeDecodeInput;
	};

	struct TrainingExecutionABI
	{
		std::vector<RuntimeStateBinding> savedActivations;
		std::vector<RuntimeStateBinding> mutableParameters;
		std::vector<RuntimeStateBinding> optimizerStates;
		std::optional<RuntimeStateBinding> lossInputs;
		std::string recomputationStrategy{ "none" };
	};

	struct LoRAAdapterExecutionABI
	{
		std::vector<RuntimeStateBinding> adapterWeights;
		std::optional<RuntimeStateBinding> activeAdapter;
		std::optional<RuntimeStateBinding> mergeState;
	};

	enum class RuntimeScheduleStepKind
	{
		DispatchRegion,
		Transfer,
		Sync,
		Fallback,
		StateRead,
		StateWrite,
		DispatchSegment
	};

	struct RuntimeExecutionSegment
	{
		std::size_t id{};
		SubgraphId subgraph{};
		std::string backend;
		std::vector<NodeId> nodes;
		std::vector<std::size_t> inputBuffers;
		std::vector<std::size_t> outputBuffers;
	};

	struct RuntimeScheduleStep
	{
		std::size_t id{};
		RuntimeScheduleStepKind kind{ RuntimeScheduleStepKind::DispatchRegion };
		FunctionId function{};
		RegionId region{};
		std::optional<std::size_t> segment;
		std::string backend;
		std::string fallbackBackend;
		std::string streamOwner;
		std::string eventOwner;
		std::string syncScope;
		std::vector<std::size_t> inputBuffers;
		std::vector<std::size_t> outputBuffers;
	};

	struct RuntimeTraceEvent
	{
		std::size_t step{};
		RuntimeScheduleStepKind kind{ RuntimeScheduleStepKind::DispatchRegion };
		std::string backend;
		std::string fallbackBackend;
		std::string streamOwner;
		std::string eventOwner;
		std::string syncScope;
		std::string message;
	};

	struct RuntimeScheduleProfileRecord
	{
		std::size_t step{};
		RuntimeScheduleStepKind kind{ RuntimeScheduleStepKind::DispatchRegion };
		std::string backend;
		std::string fallbackBackend;
		std::string streamOwner;
		std::string eventOwner;
		std::string syncScope;
		std::vector<std::size_t> inputBuffers;
		std::vector<std::size_t> outputBuffers;
		std::string label;
		std::optional<double> wallTimeMs;
		std::optional<double> deviceTimeMs;
	};

	struct RuntimeScheduleProfileBucket
	{
		RuntimeScheduleStepKind kind{ RuntimeScheduleStepKind::DispatchRegion };
		std::string label;
		std::string backend;
		std::size_t steps{};
		double wallTimeMs{};
		double deviceTimeMs{};
		bool hasWallTime{};
		bool hasDeviceTime{};
	};

	struct RuntimeScheduleDeviceProfile
	{
		std::string backend;
		std::size_t dispatchSteps{};
		std::size_t transferSteps{};
		std::size_t syncSteps{};
		std::size_t fallbackSteps{};
		double dispatchWallTimeMs{};
		double dispatchDeviceTimeMs{};
		double transferWallTimeMs{};
		double transferDeviceTimeMs{};
		double syncWallTimeMs{};
		double syncDeviceTimeMs{};
		double fallbackWallTimeMs{};
		double fallbackDeviceTimeMs{};
		bool hasMeasuredTimings{};
	};

	struct RuntimeScheduleProfileSummary
	{
		std::vector<RuntimeScheduleProfileBucket> buckets;
		std::vector<RuntimeScheduleDeviceProfile> devices;
		std::size_t dispatchSteps{};
		std::size_t transferSteps{};
		std::size_t syncSteps{};
		std::size_t fallbackSteps{};
		bool hasMeasuredTimings{};
	};

	struct RuntimeSchedule
	{
		ExecutableModule module;
		MemoryPlan memory;
		std::vector<RuntimeStateBinding> states;
		std::vector<RuntimeStateValueBinding> stateValueBindings;
		std::vector<RuntimeBufferBinding> bufferBindings;
		std::vector<RuntimeExecutionSegment> segments;
		std::vector<RuntimeScheduleStep> steps;
	};

	inline std::string_view RuntimeStateValueKindName(RuntimeStateValueKind kind) noexcept
	{
		return EnumToString<EnumToStringStyle::Unqualified>(kind);
	}

	inline std::string_view RuntimeScheduleStepKindName(RuntimeScheduleStepKind kind) noexcept
	{
		return EnumToString<EnumToStringStyle::Unqualified>(kind);
	}

	inline RuntimeStateBinding MakeRuntimeStateBinding(std::string name, RuntimeStateKind kind, std::string role,
	                                                   TensorType type,
	                                                   BufferMutability mutability = BufferMutability::Mutable,
	                                                   std::vector<std::string> effects = {})
	{
		return { .name = std::move(name),
			     .kind = kind,
			     .role = std::move(role),
			     .type = std::move(type),
			     .mutability = mutability,
			     .effects = std::move(effects) };
	}

	inline RuntimeStateBinding MakeKVCacheState(std::string name, TensorType type)
	{
		return MakeRuntimeStateBinding(std::move(name), RuntimeStateKind::KVCache, "kv-cache", std::move(type),
		                               BufferMutability::Mutable, { "read", "append", "view" });
	}

	inline RuntimeStateBinding MakeDiffusionState(std::string name, std::string role, TensorType type,
	                                              BufferMutability mutability = BufferMutability::Mutable)
	{
		return MakeRuntimeStateBinding(std::move(name), RuntimeStateKind::Diffusion, std::move(role), std::move(type),
		                               mutability, { "read", "write" });
	}

	inline RuntimeStateBinding MakeTrainingState(std::string name, std::string role, TensorType type,
	                                             BufferMutability mutability = BufferMutability::Mutable)
	{
		return MakeRuntimeStateBinding(std::move(name), RuntimeStateKind::Training, std::move(role), std::move(type),
		                               mutability, { "read", "write" });
	}

	inline RuntimeStateBinding MakeLoRAAdapterState(std::string name, std::string role, TensorType type,
	                                                BufferMutability mutability = BufferMutability::Mutable)
	{
		return MakeRuntimeStateBinding(std::move(name), RuntimeStateKind::LoRAAdapter, std::move(role), std::move(type),
		                               mutability, { "read", "rebind", "merge" });
	}

	inline RuntimeSchedule BuildRuntimeSchedule(ExecutableModule module, std::vector<RuntimeStateBinding> states = {},
	                                            std::vector<RuntimeStateValueBinding> stateValueBindings = {})
	{
		ValidateExecutablePlan(module.plan);
		OwnExecutableModuleVariableStorage(module);
		RuntimeSchedule schedule;
		schedule.memory = BuildMemoryPlan(module.plan);
		ValidateMemoryPlan(module.plan, schedule.memory);
		schedule.module = std::move(module);
		schedule.states = std::move(states);
		schedule.stateValueBindings = std::move(stateValueBindings);

		for (auto& state : schedule.states)
		{
			if (state.memoryBuffer)
			{
				if (*state.memoryBuffer >= schedule.memory.buffers.size())
				{
					throw std::runtime_error(
					    std::format("Runtime state '{}' references memory buffer {}, but bufferCount={}", state.name,
					                *state.memoryBuffer, schedule.memory.buffers.size()));
				}
				continue;
			}
			const auto byteSize = state.type.ByteSize();
			if (!byteSize)
			{
				throw std::runtime_error("Runtime state requires a static tensor type: " + state.name);
			}
			const auto bufferId = schedule.memory.buffers.size();
			schedule.memory.buffers.push_back({ .id = bufferId,
			                                    .kind = MemoryBufferKind::Persistent,
			                                    .memorySpace = state.type.memorySpace,
			                                    .byteSize = *byteSize,
			                                    .alignment = 1,
			                                    .aliasSet = bufferId });
			schedule.memory.persistentBytes += *byteSize;
			state.memoryBuffer = bufferId;
		}

		schedule.bufferBindings.reserve(schedule.module.plan.variables.size() + schedule.states.size());
		for (std::size_t i = 0; i < schedule.module.plan.variables.size(); ++i)
		{
			const auto& variable = schedule.module.plan.variables[i];
			schedule.bufferBindings.push_back(ToRuntimeBufferBinding(
			    i < schedule.module.plan.variableNames.size() && !schedule.module.plan.variableNames[i].empty()
			        ? schedule.module.plan.variableNames[i]
			        : (variable.region.name.empty() ? std::format("variable.{}", i) : variable.region.name),
			    variable, i));
		}
		for (const auto& state : schedule.states)
		{
			RuntimeBufferBinding binding;
			binding.name = state.name;
			binding.type = state.type;
			binding.ownership = BufferOwnership::Owned;
			binding.externalKind = ExternalBufferKind::None;
			binding.memorySpace = state.type.memorySpace;
			binding.memoryBuffer = *state.memoryBuffer;
			binding.byteSize = state.type.ByteSize().value_or(0);
			binding.alignment = 1;
			binding.mutability = state.mutability;
			binding.rebindPolicy = BufferRebindPolicy::CompatibleMetadata;
			schedule.bufferBindings.push_back(std::move(binding));
		}

		for (const auto& valueBinding : schedule.stateValueBindings)
		{
			const auto stateIt = std::ranges::find_if(schedule.states, [&](const RuntimeStateBinding& state) {
				return state.name == valueBinding.stateName;
			});
			if (stateIt == schedule.states.end())
			{
				throw std::runtime_error("Runtime state value binding references an unknown state: " +
				                         valueBinding.stateName);
			}
			if (valueBinding.function >= schedule.module.functions.size())
			{
				throw std::runtime_error("Runtime state value binding references an unknown function");
			}
			const auto& function = schedule.module.functions[valueBinding.function];
			const auto subgraphIt = std::ranges::find_if(schedule.module.plan.subgraphs, [&](const auto& subgraph) {
				return subgraph.sourceSubgraph == function.body;
			});
			if (subgraphIt == schedule.module.plan.subgraphs.end())
			{
				throw std::runtime_error("Runtime state value binding function has no executable subgraph");
			}
			NodeOutput value;
			TensorType valueType;
			if (valueBinding.kind == RuntimeStateValueKind::FunctionInput)
			{
				if (valueBinding.valueIndex >= function.inputs.size())
				{
					throw std::runtime_error("Runtime state value binding references an unknown function input");
				}
				const auto nodeIt = std::ranges::find_if(subgraphIt->nodes, [&](const auto& node) {
					const auto* param = std::get_if<ParamRefNode>(&node.node);
					return param != nullptr && param->paramIndex == valueBinding.valueIndex;
				});
				if (nodeIt == subgraphIt->nodes.end())
				{
					throw std::runtime_error("Runtime state value binding input has no ParamRefNode");
				}
				value = { nodeIt->sourceNode, 0 };
				valueType = function.inputs[valueBinding.valueIndex];
			}
			else
			{
				if (valueBinding.valueIndex >= function.outputs.size() ||
				    valueBinding.valueIndex >= subgraphIt->results.size())
				{
					throw std::runtime_error("Runtime state value binding references an unknown function output");
				}
				value = subgraphIt->results[valueBinding.valueIndex];
				valueType = function.outputs[valueBinding.valueIndex];
			}
			if (valueType.dtype != stateIt->type.dtype || valueType.memorySpace != stateIt->type.memorySpace)
			{
				throw std::runtime_error("Runtime state value binding type is incompatible with state: " +
				                         valueBinding.stateName);
			}
			const auto valueBytes = valueType.ByteSize();
			const auto stateBytes = stateIt->type.ByteSize();
			if (!valueBytes || !stateBytes || valueBinding.stateByteOffset > *stateBytes ||
			    *valueBytes > *stateBytes - valueBinding.stateByteOffset)
			{
				throw std::runtime_error("Runtime state value binding exceeds state capacity: " +
				                         valueBinding.stateName);
			}
			auto assignment = std::ranges::find_if(schedule.memory.assignments, [&](const MemoryAssignment& candidate) {
				return candidate.subgraph == subgraphIt->sourceSubgraph && candidate.value == value;
			});
			if (assignment == schedule.memory.assignments.end())
			{
				throw std::runtime_error("Runtime state value binding has no memory assignment");
			}
			assignment->buffer = *stateIt->memoryBuffer;
			assignment->offset = valueBinding.stateByteOffset;
		}

		for (const auto& state : schedule.states)
		{
			RuntimeScheduleStep step;
			step.id = schedule.steps.size();
			step.kind = RuntimeScheduleStepKind::StateRead;
			step.backend = "runtime";
			step.outputBuffers.push_back(*state.memoryBuffer);
			schedule.steps.push_back(std::move(step));
		}

		for (const auto& partition : schedule.module.partitions)
		{
			for (const auto regionId : partition.regions)
			{
				if (regionId >= schedule.module.regions.size())
				{
					throw std::runtime_error(
					    std::format("Runtime schedule partition {} references region {}, but regionCount={}",
					                partition.id, regionId, schedule.module.regions.size()));
				}
				const auto& region = schedule.module.regions[regionId];
				RuntimeScheduleStep step;
				step.id = schedule.steps.size();
				step.kind = RuntimeScheduleStepKind::DispatchRegion;
				step.function = region.function;
				step.region = region.id;
				step.backend = partition.backend;

				const auto& subgraph = schedule.module.plan.subgraphs[region.subgraph];
				for (const auto nodeId : region.nodes)
				{
					const auto& node = subgraph.nodes[nodeId];
					for (const auto input : node.inputs)
					{
						if (const auto* assignment =
						        FindMemoryAssignment(schedule.memory, subgraph.sourceSubgraph, input))
						{
							step.inputBuffers.push_back(assignment->buffer);
						}
					}
					for (std::size_t outputIndex = 0; outputIndex < node.outputs.size(); ++outputIndex)
					{
						if (const auto* assignment = FindMemoryAssignment(schedule.memory, subgraph.sourceSubgraph,
						                                                  { node.sourceNode, outputIndex }))
						{
							step.outputBuffers.push_back(assignment->buffer);
						}
					}
				}
				schedule.steps.push_back(std::move(step));
			}
		}
		for (const auto& state : schedule.states)
		{
			if (state.mutability != BufferMutability::Mutable)
			{
				continue;
			}
			RuntimeScheduleStep step;
			step.id = schedule.steps.size();
			step.kind = RuntimeScheduleStepKind::StateWrite;
			step.backend = "runtime";
			step.inputBuffers.push_back(*state.memoryBuffer);
			schedule.steps.push_back(std::move(step));
		}
		return schedule;
	}

	inline std::vector<std::size_t> RuntimeScheduleStateOutputIndices(const RuntimeSchedule& schedule,
	                                                                  FunctionId function)
	{
		if (function >= schedule.module.functions.size())
		{
			throw std::runtime_error("Runtime schedule state-output query references an unknown function");
		}
		std::vector<std::size_t> indices;
		for (const auto& binding : schedule.stateValueBindings)
		{
			if (binding.function != function || binding.kind != RuntimeStateValueKind::FunctionOutput)
			{
				continue;
			}
			if (binding.valueIndex >= schedule.module.functions[function].outputs.size())
			{
				throw std::runtime_error("Runtime schedule state-output binding references an unknown function output");
			}
			if (!std::ranges::contains(indices, binding.valueIndex))
			{
				indices.push_back(binding.valueIndex);
			}
		}
		std::ranges::sort(indices);
		return indices;
	}

	inline std::vector<std::size_t> RuntimeSchedulePublicOutputIndices(const RuntimeSchedule& schedule,
	                                                                   FunctionId function)
	{
		if (function >= schedule.module.functions.size())
		{
			throw std::runtime_error("Runtime schedule public-output query references an unknown function");
		}
		const auto stateOutputs = RuntimeScheduleStateOutputIndices(schedule, function);
		std::vector<std::size_t> publicOutputs;
		const auto outputCount = schedule.module.functions[function].outputs.size();
		publicOutputs.reserve(outputCount);
		for (std::size_t outputIndex = 0; outputIndex < outputCount; ++outputIndex)
		{
			if (!std::ranges::binary_search(stateOutputs, outputIndex))
			{
				publicOutputs.push_back(outputIndex);
			}
		}
		return publicOutputs;
	}

	inline std::vector<TensorType> RuntimeSchedulePublicOutputTypes(const RuntimeSchedule& schedule,
	                                                                FunctionId function)
	{
		const auto publicOutputIndices = RuntimeSchedulePublicOutputIndices(schedule, function);
		std::vector<TensorType> types;
		types.reserve(publicOutputIndices.size());
		for (const auto outputIndex : publicOutputIndices)
		{
			types.push_back(schedule.module.functions[function].outputs[outputIndex]);
		}
		return types;
	}

	inline std::vector<RuntimeStateOutputAlias> RuntimeScheduleStateOutputAliases(const RuntimeSchedule& schedule,
	                                                                              FunctionId function)
	{
		if (function >= schedule.module.functions.size())
		{
			throw std::runtime_error("Runtime schedule state-output alias query references an unknown function");
		}
		std::vector<RuntimeStateOutputAlias> aliases;
		for (const auto& outputBinding : schedule.stateValueBindings)
		{
			if (outputBinding.function != function || outputBinding.kind != RuntimeStateValueKind::FunctionOutput)
			{
				continue;
			}
			const auto inputIt = std::ranges::find_if(schedule.stateValueBindings, [&](const auto& inputBinding) {
				return inputBinding.function == function && inputBinding.kind == RuntimeStateValueKind::FunctionInput &&
				       inputBinding.stateName == outputBinding.stateName &&
				       inputBinding.stateByteOffset == outputBinding.stateByteOffset;
			});
			if (inputIt == schedule.stateValueBindings.end())
			{
				throw std::runtime_error("Runtime schedule state output has no matching input alias: " +
				                         outputBinding.stateName);
			}
			if (outputBinding.valueIndex >= schedule.module.functions[function].outputs.size() ||
			    inputIt->valueIndex >= schedule.module.functions[function].inputs.size())
			{
				throw std::runtime_error("Runtime schedule state-output alias references an unknown function value");
			}
			aliases.push_back({ .outputIndex = outputBinding.valueIndex,
			                    .inputIndex = inputIt->valueIndex,
			                    .stateName = outputBinding.stateName,
			                    .stateByteOffset = outputBinding.stateByteOffset });
		}
		std::ranges::sort(aliases, {}, &RuntimeStateOutputAlias::outputIndex);
		return aliases;
	}

	inline RuntimeScheduleOutputProjection RuntimeScheduleOutputProjectionForFunction(const RuntimeSchedule& schedule,
	                                                                                  FunctionId function)
	{
		if (function >= schedule.module.functions.size())
		{
			throw std::runtime_error("Runtime schedule output projection references an unknown function");
		}
		auto publicOutputs = RuntimeSchedulePublicOutputIndices(schedule, function);
		auto publicTypes = RuntimeSchedulePublicOutputTypes(schedule, function);
		auto aliases = RuntimeScheduleStateOutputAliases(schedule, function);
		const auto functionalOutputCount = schedule.module.functions[function].outputs.size();
		if (publicOutputs.size() + aliases.size() != functionalOutputCount)
		{
			throw std::runtime_error("Runtime schedule output projection does not cover every functional output");
		}
		return RuntimeScheduleOutputProjection{ .functionalOutputCount = functionalOutputCount,
			                                    .publicOutputIndices = std::move(publicOutputs),
			                                    .publicOutputTypes = std::move(publicTypes),
			                                    .stateAliases = std::move(aliases) };
	}

	inline void AppendUniqueBuffer(std::vector<std::size_t>& buffers, std::size_t buffer)
	{
		if (std::ranges::find(buffers, buffer) == buffers.end())
		{
			buffers.push_back(buffer);
		}
	}

	inline const PlacementDecision* FindPlacementDecision(const PlacementPlan& placement, SubgraphId subgraph,
	                                                      NodeId node)
	{
		const auto it = std::ranges::find_if(placement.decisions, [&](const PlacementDecision& decision) {
			return decision.subgraph == subgraph && decision.node == node;
		});
		return it == placement.decisions.end() ? nullptr : &*it;
	}

	inline bool SegmentContainsNode(const RuntimeExecutionSegment& segment, NodeId node)
	{
		return std::ranges::find(segment.nodes, node) != segment.nodes.end();
	}

	inline bool ValueHasConsumerOutsideSegment(const ExecutablePlanSubgraph& subgraph,
	                                           const RuntimeExecutionSegment& segment, NodeOutput value)
	{
		if (std::ranges::find(subgraph.results, value) != subgraph.results.end())
		{
			return true;
		}
		for (const auto& node : subgraph.nodes)
		{
			for (const auto input : node.inputs)
			{
				if (input == value && !SegmentContainsNode(segment, node.sourceNode))
				{
					return true;
				}
			}
		}
		return false;
	}

	inline void FinalizePlacementSegment(RuntimeExecutionSegment& segment, const PlacementPlan& placement,
	                                     const ExecutablePlanSubgraph& subgraph)
	{
		for (const auto nodeId : segment.nodes)
		{
			const auto* node = FindPlanNode(subgraph, nodeId);
			if (node == nullptr)
			{
				throw std::runtime_error("Runtime placement segment references an unknown node");
			}
			for (const auto input : node->inputs)
			{
				if (SegmentContainsNode(segment, input.node))
				{
					continue;
				}
				if (const auto* assignment = FindMemoryAssignment(placement.memory, subgraph.sourceSubgraph, input))
				{
					AppendUniqueBuffer(segment.inputBuffers, assignment->buffer);
				}
			}
			for (std::size_t outputIndex = 0; outputIndex < node->outputs.size(); ++outputIndex)
			{
				const auto output = NodeOutput{ node->sourceNode, outputIndex };
				if (!ValueHasConsumerOutsideSegment(subgraph, segment, output))
				{
					continue;
				}
				if (const auto* assignment = FindMemoryAssignment(placement.memory, subgraph.sourceSubgraph, output))
				{
					AppendUniqueBuffer(segment.outputBuffers, assignment->buffer);
				}
			}
		}
	}

	inline std::vector<RuntimeExecutionSegment> BuildPlacementSegments(const PlacementPlan& placement)
	{
		std::vector<RuntimeExecutionSegment> segments;
		for (const auto& subgraph : placement.plan.subgraphs)
		{
			std::optional<RuntimeExecutionSegment> current;
			for (const auto& node : subgraph.nodes)
			{
				const auto* decision = FindPlacementDecision(placement, subgraph.sourceSubgraph, node.sourceNode);
				if (decision == nullptr)
				{
					throw std::runtime_error("Runtime placement segment builder found a node without placement");
				}
				if (current && current->backend != decision->backend)
				{
					FinalizePlacementSegment(*current, placement, subgraph);
					segments.push_back(std::move(*current));
					current.reset();
				}
				if (!current)
				{
					current = RuntimeExecutionSegment{ .id = segments.size(),
						                               .subgraph = subgraph.sourceSubgraph,
						                               .backend = decision->backend };
				}
				current->nodes.push_back(node.sourceNode);
			}
			if (current)
			{
				FinalizePlacementSegment(*current, placement, subgraph);
				segments.push_back(std::move(*current));
			}
		}
		for (std::size_t i = 0; i < segments.size(); ++i)
		{
			segments[i].id = i;
		}
		return segments;
	}

	inline void AppendPlacementSegmentSteps(RuntimeSchedule& schedule, const PlacementPlan& placement)
	{
		const auto baseSegmentId = schedule.segments.size();
		auto segments = BuildPlacementSegments(placement);
		for (auto& segment : segments)
		{
			segment.id += baseSegmentId;
			RuntimeScheduleStep step;
			step.id = schedule.steps.size();
			step.kind = RuntimeScheduleStepKind::DispatchSegment;
			step.segment = segment.id;
			step.backend = segment.backend;
			step.inputBuffers = segment.inputBuffers;
			step.outputBuffers = segment.outputBuffers;
			schedule.segments.push_back(std::move(segment));
			schedule.steps.push_back(std::move(step));
		}
	}

	inline void AppendPlacementFallbackSteps(RuntimeSchedule& schedule, const PlacementPlan& placement)
	{
		for (const auto& fallback : placement.fallbackSteps)
		{
			RuntimeScheduleStep step;
			step.id = schedule.steps.size();
			step.kind = RuntimeScheduleStepKind::Fallback;
			step.backend = fallback.requestedBackend;
			step.fallbackBackend = fallback.fallbackBackend;
			step.inputBuffers = fallback.inputBuffers;
			step.outputBuffers = fallback.outputBuffers;
			schedule.steps.push_back(std::move(step));
		}
	}

	inline void AppendPlacementTransferSteps(RuntimeSchedule& schedule, const PlacementPlan& placement)
	{
		for (const auto& transfer : placement.transferSteps)
		{
			RuntimeScheduleStep step;
			step.id = schedule.steps.size();
			step.kind = RuntimeScheduleStepKind::Transfer;
			step.backend = transfer.sourceBackend;
			step.fallbackBackend = transfer.targetBackend;
			step.inputBuffers.push_back(transfer.buffer);
			step.outputBuffers.push_back(transfer.buffer);
			schedule.steps.push_back(std::move(step));
		}
	}

	inline bool BackendNeedsRuntimeSync(std::string_view backend) noexcept
	{
		return backend == BackendCUDANative || backend == BackendCUDABridge || backend == BackendVulkanNative ||
		       backend == BackendVulkanBridge;
	}

	inline std::string RuntimeStreamOwnerForBackend(std::string_view backend)
	{
		if (backend == BackendCUDANative || backend == BackendCUDABridge)
		{
			return "cuda-default-stream";
		}
		if (backend == BackendVulkanNative || backend == BackendVulkanBridge)
		{
			return "vulkan-command-queue";
		}
		return {};
	}

	inline std::string RuntimeEventOwnerForBackend(std::string_view backend)
	{
		if (backend == BackendCUDANative || backend == BackendCUDABridge)
		{
			return "cuda-runtime-event";
		}
		if (backend == BackendVulkanNative || backend == BackendVulkanBridge)
		{
			return "vulkan-runtime-fence";
		}
		return {};
	}

	inline void AppendPlacementSyncSteps(RuntimeSchedule& schedule, const PlacementPlan& placement)
	{
		for (const auto& transfer : placement.transferSteps)
		{
			const auto sourceNeedsSync = BackendNeedsRuntimeSync(transfer.sourceBackend);
			const auto targetNeedsSync = BackendNeedsRuntimeSync(transfer.targetBackend);
			if (!sourceNeedsSync && !targetNeedsSync)
			{
				continue;
			}
			RuntimeScheduleStep step;
			step.id = schedule.steps.size();
			step.kind = RuntimeScheduleStepKind::Sync;
			step.backend = targetNeedsSync ? transfer.targetBackend : transfer.sourceBackend;
			step.fallbackBackend = targetNeedsSync ? transfer.sourceBackend : transfer.targetBackend;
			step.streamOwner = RuntimeStreamOwnerForBackend(step.backend);
			step.eventOwner = RuntimeEventOwnerForBackend(step.backend);
			step.syncScope = "transfer-boundary";
			step.inputBuffers.push_back(transfer.buffer);
			step.outputBuffers.push_back(transfer.buffer);
			schedule.steps.push_back(std::move(step));
		}
	}

	inline std::vector<RuntimeTraceEvent> TraceRuntimeSchedule(const RuntimeSchedule& schedule)
	{
		std::vector<RuntimeTraceEvent> events;
		events.reserve(schedule.steps.size());
		for (const auto& step : schedule.steps)
		{
			std::string message;
			if (step.kind == RuntimeScheduleStepKind::DispatchRegion)
			{
				message = std::format("dispatch region {} function {} on {}", step.region, step.function, step.backend);
			}
			else if (step.kind == RuntimeScheduleStepKind::DispatchSegment)
			{
				message =
				    std::format("dispatch segment {} on {} inputBuffers={} outputBuffers={}", step.segment.value_or(0),
				                step.backend, step.inputBuffers.size(), step.outputBuffers.size());
			}
			else if (step.kind == RuntimeScheduleStepKind::Fallback)
			{
				message = std::format("fallback from {} to {} inputBuffers={} outputBuffers={}", step.backend,
				                      step.fallbackBackend, step.inputBuffers.size(), step.outputBuffers.size());
			}
			else if (step.kind == RuntimeScheduleStepKind::Transfer)
			{
				message = std::format("transfer from {} to {} buffers={}", step.backend, step.fallbackBackend,
				                      step.inputBuffers.size());
			}
			else if (step.kind == RuntimeScheduleStepKind::Sync)
			{
				message = std::format("sync {} with {} buffers={} stream={} event={} scope={}", step.backend,
				                      step.fallbackBackend, step.inputBuffers.size(), step.streamOwner, step.eventOwner,
				                      step.syncScope);
			}
			else
			{
				message =
				    std::format("{} on {} inputBuffers={} outputBuffers={}", RuntimeScheduleStepKindName(step.kind),
				                step.backend, step.inputBuffers.size(), step.outputBuffers.size());
			}
			events.push_back({ .step = step.id,
			                   .kind = step.kind,
			                   .backend = step.backend,
			                   .fallbackBackend = step.fallbackBackend,
			                   .streamOwner = step.streamOwner,
			                   .eventOwner = step.eventOwner,
			                   .syncScope = step.syncScope,
			                   .message = std::move(message) });
		}
		return events;
	}

	inline RuntimeScheduleProfileRecord MakeRuntimeScheduleProfileRecord(const RuntimeScheduleStep& step)
	{
		std::string label;
		if (step.kind == RuntimeScheduleStepKind::Fallback)
		{
			label = std::format("fallback:{}->{}", step.backend, step.fallbackBackend);
		}
		else if (step.kind == RuntimeScheduleStepKind::DispatchSegment)
		{
			label = std::format("segment:{}:{}", step.segment.value_or(0), step.backend);
		}
		else if (step.kind == RuntimeScheduleStepKind::Transfer)
		{
			label = std::format("transfer:{}->{}", step.backend, step.fallbackBackend);
		}
		else if (step.kind == RuntimeScheduleStepKind::Sync)
		{
			label = std::format("sync:{}<->{}", step.backend, step.fallbackBackend);
		}
		else
		{
			label = std::format("{}:{}", RuntimeScheduleStepKindName(step.kind), step.backend);
		}
		return { .step = step.id,
			     .kind = step.kind,
			     .backend = step.backend,
			     .fallbackBackend = step.fallbackBackend,
			     .streamOwner = step.streamOwner,
			     .eventOwner = step.eventOwner,
			     .syncScope = step.syncScope,
			     .inputBuffers = step.inputBuffers,
			     .outputBuffers = step.outputBuffers,
			     .label = std::move(label) };
	}

	inline std::vector<RuntimeScheduleProfileRecord> BuildRuntimeScheduleProfileRecords(const RuntimeSchedule& schedule)
	{
		std::vector<RuntimeScheduleProfileRecord> records;
		records.reserve(schedule.steps.size());
		for (const auto& step : schedule.steps)
		{
			records.push_back(MakeRuntimeScheduleProfileRecord(step));
		}
		return records;
	}

	inline void AccumulateRuntimeScheduleProfileBucket(RuntimeScheduleProfileSummary& summary,
	                                                   const RuntimeScheduleProfileRecord& record)
	{
		const auto it = std::ranges::find_if(summary.buckets, [&](const RuntimeScheduleProfileBucket& bucket) {
			return bucket.kind == record.kind && bucket.label == record.label && bucket.backend == record.backend;
		});
		auto& bucket = it == summary.buckets.end()
		                   ? summary.buckets.emplace_back(RuntimeScheduleProfileBucket{
		                         .kind = record.kind, .label = record.label, .backend = record.backend })
		                   : *it;
		++bucket.steps;
		if (record.wallTimeMs)
		{
			bucket.wallTimeMs += *record.wallTimeMs;
			bucket.hasWallTime = true;
			summary.hasMeasuredTimings = true;
		}
		if (record.deviceTimeMs)
		{
			bucket.deviceTimeMs += *record.deviceTimeMs;
			bucket.hasDeviceTime = true;
			summary.hasMeasuredTimings = true;
		}
	}

	inline RuntimeScheduleDeviceProfile& EnsureRuntimeScheduleDeviceProfile(RuntimeScheduleProfileSummary& summary,
	                                                                        std::string_view backend)
	{
		const auto it = std::ranges::find_if(
		    summary.devices, [&](const RuntimeScheduleDeviceProfile& device) { return device.backend == backend; });
		if (it != summary.devices.end())
		{
			return *it;
		}
		return summary.devices.emplace_back(RuntimeScheduleDeviceProfile{ .backend = std::string(backend) });
	}

	inline void AddRuntimeScheduleDeviceTiming(RuntimeScheduleDeviceProfile& device, RuntimeScheduleStepKind kind,
	                                           const RuntimeScheduleProfileRecord& record)
	{
		const auto wallTime = record.wallTimeMs.value_or(0.0);
		const auto deviceTime = record.deviceTimeMs.value_or(0.0);
		if (record.wallTimeMs || record.deviceTimeMs)
		{
			device.hasMeasuredTimings = true;
		}
		switch (kind)
		{
		case RuntimeScheduleStepKind::DispatchRegion:
		case RuntimeScheduleStepKind::DispatchSegment:
			++device.dispatchSteps;
			device.dispatchWallTimeMs += wallTime;
			device.dispatchDeviceTimeMs += deviceTime;
			break;
		case RuntimeScheduleStepKind::Transfer:
			++device.transferSteps;
			device.transferWallTimeMs += wallTime;
			device.transferDeviceTimeMs += deviceTime;
			break;
		case RuntimeScheduleStepKind::Sync:
			++device.syncSteps;
			device.syncWallTimeMs += wallTime;
			device.syncDeviceTimeMs += deviceTime;
			break;
		case RuntimeScheduleStepKind::Fallback:
			++device.fallbackSteps;
			device.fallbackWallTimeMs += wallTime;
			device.fallbackDeviceTimeMs += deviceTime;
			break;
		case RuntimeScheduleStepKind::StateRead:
		case RuntimeScheduleStepKind::StateWrite:
			break;
		}
	}

	inline void AccumulateRuntimeScheduleDeviceProfile(RuntimeScheduleProfileSummary& summary,
	                                                   const RuntimeScheduleProfileRecord& record)
	{
		if (!record.backend.empty())
		{
			AddRuntimeScheduleDeviceTiming(EnsureRuntimeScheduleDeviceProfile(summary, record.backend), record.kind,
			                               record);
		}
		if ((record.kind == RuntimeScheduleStepKind::Transfer || record.kind == RuntimeScheduleStepKind::Sync ||
		     record.kind == RuntimeScheduleStepKind::Fallback) &&
		    !record.fallbackBackend.empty() && record.fallbackBackend != record.backend)
		{
			AddRuntimeScheduleDeviceTiming(EnsureRuntimeScheduleDeviceProfile(summary, record.fallbackBackend),
			                               record.kind, record);
		}
	}

	inline RuntimeScheduleProfileSummary
	BuildRuntimeScheduleProfileSummary(std::span<const RuntimeScheduleProfileRecord> records)
	{
		RuntimeScheduleProfileSummary summary;
		for (const auto& record : records)
		{
			switch (record.kind)
			{
			case RuntimeScheduleStepKind::DispatchRegion:
			case RuntimeScheduleStepKind::DispatchSegment:
				++summary.dispatchSteps;
				break;
			case RuntimeScheduleStepKind::Transfer:
				++summary.transferSteps;
				break;
			case RuntimeScheduleStepKind::Sync:
				++summary.syncSteps;
				break;
			case RuntimeScheduleStepKind::Fallback:
				++summary.fallbackSteps;
				break;
			case RuntimeScheduleStepKind::StateRead:
			case RuntimeScheduleStepKind::StateWrite:
				break;
			}
			AccumulateRuntimeScheduleProfileBucket(summary, record);
			AccumulateRuntimeScheduleDeviceProfile(summary, record);
		}
		return summary;
	}

	inline void ValidateRuntimeSchedule(const RuntimeSchedule& schedule)
	{
		ValidateExecutablePlan(schedule.module.plan);
		ValidateMemoryPlan(schedule.module.plan, schedule.memory);
		for (std::size_t i = 0; i < schedule.steps.size(); ++i)
		{
			const auto& step = schedule.steps[i];
			if (step.id != i)
			{
				throw std::runtime_error(std::format("Runtime schedule step {} has mismatched id {}", i, step.id));
			}
			if (step.kind == RuntimeScheduleStepKind::DispatchRegion)
			{
				if (step.function >= schedule.module.functions.size())
				{
					throw std::runtime_error("Runtime dispatch step references an unknown function");
				}
				if (step.region >= schedule.module.regions.size())
				{
					throw std::runtime_error("Runtime dispatch step references an unknown region");
				}
				if (step.backend.empty())
				{
					throw std::runtime_error("Runtime dispatch step has empty backend");
				}
			}
			if (step.kind == RuntimeScheduleStepKind::DispatchSegment)
			{
				if (!step.segment || *step.segment >= schedule.segments.size())
				{
					throw std::runtime_error("Runtime segment dispatch step references an unknown segment");
				}
				if (step.backend.empty())
				{
					throw std::runtime_error("Runtime segment dispatch step has empty backend");
				}
				const auto& segment = schedule.segments[*step.segment];
				if (segment.backend != step.backend || segment.inputBuffers != step.inputBuffers ||
				    segment.outputBuffers != step.outputBuffers)
				{
					throw std::runtime_error("Runtime segment dispatch step does not match segment metadata");
				}
				if (segment.nodes.empty())
				{
					throw std::runtime_error("Runtime segment dispatch step has no nodes");
				}
			}
			if (step.kind == RuntimeScheduleStepKind::Fallback)
			{
				if (step.backend.empty() || step.fallbackBackend.empty())
				{
					throw std::runtime_error("Runtime fallback step must name requested and fallback backends");
				}
			}
			if (step.kind == RuntimeScheduleStepKind::Transfer)
			{
				if (step.backend.empty() || step.fallbackBackend.empty())
				{
					throw std::runtime_error("Runtime transfer step must name source and target backends");
				}
				if (step.inputBuffers.empty() || step.outputBuffers.empty())
				{
					throw std::runtime_error("Runtime transfer step must name transferred buffers");
				}
			}
			if (step.kind == RuntimeScheduleStepKind::Sync)
			{
				if (step.backend.empty())
				{
					throw std::runtime_error("Runtime sync step must name a synchronizing backend");
				}
				if (BackendNeedsRuntimeSync(step.backend) &&
				    (step.streamOwner.empty() || step.eventOwner.empty() || step.syncScope.empty()))
				{
					throw std::runtime_error("Runtime sync step must record stream/event ownership");
				}
				if (step.inputBuffers.empty() || step.outputBuffers.empty())
				{
					throw std::runtime_error("Runtime sync step must name synchronized buffers");
				}
			}
			for (const auto buffer : step.inputBuffers)
			{
				if (buffer >= schedule.memory.buffers.size())
				{
					throw std::runtime_error("Runtime schedule step references an invalid input buffer");
				}
			}
			for (const auto buffer : step.outputBuffers)
			{
				if (buffer >= schedule.memory.buffers.size())
				{
					throw std::runtime_error("Runtime schedule step references an invalid output buffer");
				}
			}
		}
		for (const auto& state : schedule.states)
		{
			if (state.name.empty())
			{
				throw std::runtime_error("Runtime state binding name cannot be empty");
			}
			if (state.role.empty())
			{
				throw std::runtime_error("Runtime state binding role cannot be empty");
			}
			if (!state.memoryBuffer || *state.memoryBuffer >= schedule.memory.buffers.size())
			{
				throw std::runtime_error("Runtime state binding references an invalid memory buffer");
			}
			if (schedule.memory.buffers[*state.memoryBuffer].kind != MemoryBufferKind::Persistent)
			{
				throw std::runtime_error("Runtime state binding must use a persistent buffer");
			}
		}
		for (const auto& binding : schedule.bufferBindings)
		{
			ValidateRuntimeBufferBinding(binding);
			if (binding.memoryBuffer >= schedule.memory.buffers.size())
			{
				throw std::runtime_error("Runtime buffer binding references an invalid memory buffer: " + binding.name);
			}
			const auto& buffer = schedule.memory.buffers[binding.memoryBuffer];
			if (buffer.memorySpace != binding.type.memorySpace)
			{
				throw std::runtime_error("Runtime buffer binding memory space does not match its memory buffer: " +
				                         binding.name);
			}
			if (binding.byteSize != 0 && binding.byteSize > buffer.byteSize)
			{
				throw std::runtime_error("Runtime buffer binding is larger than its memory buffer: " + binding.name);
			}
			if (buffer.alignment == 0 || binding.alignment == 0)
			{
				throw std::runtime_error("Runtime buffer binding has invalid alignment: " + binding.name);
			}
		}
	}
} // namespace LiteNN::Runtime

#endif
