#ifndef LITENN_RUNTIME_SCHEDULER_H
#define LITENN_RUNTIME_SCHEDULER_H

#include <LiteNN/ExecutablePlan.h>
#include <LiteNN/MemoryPlan.h>
#include <LiteNN/Misc.h>
#include <LiteNN/Runtime/Placement.h>
#include <cstddef>
#include <format>
#include <optional>
#include <span>
#include <stdexcept>
#include <string>
#include <string_view>
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
		StateWrite
	};

	struct RuntimeScheduleStep
	{
		std::size_t id{};
		RuntimeScheduleStepKind kind{ RuntimeScheduleStepKind::DispatchRegion };
		FunctionId function{};
		RegionId region{};
		std::string backend;
		std::string fallbackBackend;
		std::vector<std::size_t> inputBuffers;
		std::vector<std::size_t> outputBuffers;
	};

	struct RuntimeTraceEvent
	{
		std::size_t step{};
		RuntimeScheduleStepKind kind{ RuntimeScheduleStepKind::DispatchRegion };
		std::string backend;
		std::string fallbackBackend;
		std::string message;
	};

	struct RuntimeScheduleProfileRecord
	{
		std::size_t step{};
		RuntimeScheduleStepKind kind{ RuntimeScheduleStepKind::DispatchRegion };
		std::string backend;
		std::string fallbackBackend;
		std::vector<std::size_t> inputBuffers;
		std::vector<std::size_t> outputBuffers;
		std::string label;
		std::optional<double> wallTimeMs;
		std::optional<double> deviceTimeMs;
	};

	struct RuntimeSchedule
	{
		ExecutableModule module;
		MemoryPlan memory;
		std::vector<RuntimeStateBinding> states;
		std::vector<RuntimeBufferBinding> bufferBindings;
		std::vector<RuntimeScheduleStep> steps;
	};

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

	inline RuntimeSchedule BuildRuntimeSchedule(ExecutableModule module, std::vector<RuntimeStateBinding> states = {})
	{
		ValidateExecutablePlan(module.plan);
		RuntimeSchedule schedule;
		schedule.memory = BuildMemoryPlan(module.plan);
		ValidateMemoryPlan(module.plan, schedule.memory);
		schedule.module = std::move(module);
		schedule.states = std::move(states);

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
			    variable.region.name.empty() ? std::format("variable.{}", i) : variable.region.name, variable, i));
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
			else if (step.kind == RuntimeScheduleStepKind::Fallback)
			{
				message = std::format("fallback from {} to {} inputBuffers={} outputBuffers={}", step.backend,
				                      step.fallbackBackend, step.inputBuffers.size(), step.outputBuffers.size());
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
		else
		{
			label = std::format("{}:{}", RuntimeScheduleStepKindName(step.kind), step.backend);
		}
		return { .step = step.id,
			     .kind = step.kind,
			     .backend = step.backend,
			     .fallbackBackend = step.fallbackBackend,
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
			if (step.kind == RuntimeScheduleStepKind::Fallback)
			{
				if (step.backend.empty() || step.fallbackBackend.empty())
				{
					throw std::runtime_error("Runtime fallback step must name requested and fallback backends");
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
