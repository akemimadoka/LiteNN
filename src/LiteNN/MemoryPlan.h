#ifndef LITENN_MEMORY_PLAN_H
#define LITENN_MEMORY_PLAN_H

#include <LiteNN/ExecutablePlan.h>
#include <algorithm>
#include <cstddef>
#include <format>
#include <optional>
#include <span>
#include <stdexcept>
#include <string>
#include <variant>
#include <vector>

namespace LiteNN
{
	enum class MemoryBufferKind
	{
		Workspace,
		Persistent,
		External,
		Constant
	};

	enum class DeviceMemoryAllocationKind
	{
		HostVisible,
		DeviceLocal,
		Unified,
		ExternalDevice,
		ExternalHost,
		ConstantDeviceLocal
	};

	enum class DeviceMemoryStagingDirection
	{
		Upload,
		Download
	};

	struct MemoryValueLifetime
	{
		SubgraphId subgraph{};
		NodeOutput value{};
		TensorType type;
		std::size_t firstUse{};
		std::size_t lastUse{};
		bool publicOutput{ false };
	};

	struct MemoryBuffer
	{
		std::size_t id{};
		MemoryBufferKind kind{ MemoryBufferKind::Workspace };
		TensorMemorySpace memorySpace{ TensorMemorySpace::Host };
		std::size_t byteSize{};
		std::size_t alignment{ 1 };
		std::size_t aliasSet{};
	};

	struct MemoryAssignment
	{
		SubgraphId subgraph{};
		NodeOutput value{};
		std::size_t buffer{};
		std::size_t offset{};
		MemoryValueLifetime lifetime;
	};

	struct MemoryPlan
	{
		std::vector<MemoryBuffer> buffers;
		std::vector<MemoryAssignment> assignments;
		std::vector<TensorStorageRef> externalVariables;
		std::size_t workspaceBytes{};
		std::size_t persistentBytes{};
		std::size_t externalBytes{};
		std::size_t constantBytes{};
	};

	struct DeviceMemoryStagingStep
	{
		std::size_t buffer{};
		DeviceMemoryStagingDirection direction{ DeviceMemoryStagingDirection::Upload };
		std::size_t byteSize{};
		std::string reason;
	};

	struct DeviceMemoryBufferPlan
	{
		std::size_t buffer{};
		DeviceMemoryAllocationKind allocation{ DeviceMemoryAllocationKind::HostVisible };
		TensorMemorySpace memorySpace{ TensorMemorySpace::Host };
		MemoryBufferKind kind{ MemoryBufferKind::Workspace };
		std::size_t byteSize{};
		std::size_t alignment{ 1 };
		bool requiresDeviceLocal{};
		bool requiresHostStaging{};
	};

	struct DeviceLocalMemoryPlan
	{
		std::vector<DeviceMemoryBufferPlan> buffers;
		std::vector<DeviceMemoryStagingStep> stagingSteps;
		std::size_t deviceLocalBytes{};
		std::size_t hostVisibleBytes{};
		std::size_t stagingBytes{};
		bool requiresDeviceLocalAllocator{};
		bool requiresStagingAllocator{};
	};

	inline bool MemoryLifetimesOverlap(const MemoryValueLifetime& lhs, const MemoryValueLifetime& rhs) noexcept
	{
		return lhs.subgraph == rhs.subgraph && lhs.firstUse <= rhs.lastUse && rhs.firstUse <= lhs.lastUse;
	}

	inline MemoryPlan BuildMemoryPlan(const ExecutablePlan& plan)
	{
		ValidateExecutablePlan(plan);
		MemoryPlan memoryPlan;
		memoryPlan.externalVariables = plan.variables;

		const auto addBuffer = [&](MemoryBufferKind kind, TensorMemorySpace memorySpace, std::size_t byteSize,
		                           std::size_t alignment, std::size_t aliasSet) {
			const auto id = memoryPlan.buffers.size();
			memoryPlan.buffers.push_back({ .id = id,
			                               .kind = kind,
			                               .memorySpace = memorySpace,
			                               .byteSize = byteSize,
			                               .alignment = alignment == 0 ? 1 : alignment,
			                               .aliasSet = aliasSet });
			switch (kind)
			{
			case MemoryBufferKind::Workspace:
				memoryPlan.workspaceBytes += byteSize;
				break;
			case MemoryBufferKind::Persistent:
				memoryPlan.persistentBytes += byteSize;
				break;
			case MemoryBufferKind::External:
				memoryPlan.externalBytes += byteSize;
				break;
			case MemoryBufferKind::Constant:
				memoryPlan.constantBytes += byteSize;
				break;
			}
			return id;
		};

		for (std::size_t i = 0; i < plan.variables.size(); ++i)
		{
			const auto& variable = plan.variables[i];
			const auto byteSize = variable.LogicalByteSize().value_or(variable.region.byteSize);
			const auto kind = variable.IsExternal() ? MemoryBufferKind::External : MemoryBufferKind::Persistent;
			(void) addBuffer(kind, variable.type.memorySpace, byteSize, variable.region.alignment, variable.aliasSet);
		}

		for (std::size_t subgraphIndex = 0; subgraphIndex < plan.subgraphs.size(); ++subgraphIndex)
		{
			const auto& subgraph = plan.subgraphs[subgraphIndex];
			std::vector<std::size_t> paramBuffers;
			paramBuffers.reserve(subgraph.params.size());
			for (std::size_t paramIndex = 0; paramIndex < subgraph.params.size(); ++paramIndex)
			{
				const auto byteSize = subgraph.params[paramIndex].ByteSize();
				if (!byteSize)
				{
					throw std::runtime_error(
					    std::format("Memory planner requires static byte size for subgraph {} param {}",
					                subgraph.sourceSubgraph, paramIndex));
				}
				paramBuffers.push_back(addBuffer(MemoryBufferKind::External, subgraph.params[paramIndex].memorySpace,
				                                 *byteSize, 1, memoryPlan.buffers.size()));
			}

			std::vector<std::vector<MemoryValueLifetime>> lifetimes(subgraph.nodes.size());
			for (std::size_t nodeIndex = 0; nodeIndex < subgraph.nodes.size(); ++nodeIndex)
			{
				const auto& node = subgraph.nodes[nodeIndex];
				lifetimes[nodeIndex].reserve(node.outputs.size());
				for (std::size_t outputIndex = 0; outputIndex < node.outputs.size(); ++outputIndex)
				{
					lifetimes[nodeIndex].push_back({ .subgraph = subgraph.sourceSubgraph,
					                                 .value = { node.sourceNode, outputIndex },
					                                 .type = node.outputs[outputIndex],
					                                 .firstUse = node.sourceNode,
					                                 .lastUse = node.sourceNode });
				}
			}

			for (const auto& node : subgraph.nodes)
			{
				for (const auto input : node.inputs)
				{
					lifetimes[input.node][input.port].lastUse =
					    std::max(lifetimes[input.node][input.port].lastUse, node.sourceNode);
				}
			}
			for (const auto result : subgraph.results)
			{
				auto& lifetime = lifetimes[result.node][result.port];
				lifetime.publicOutput = true;
				lifetime.lastUse = std::max(lifetime.lastUse, subgraph.nodes.size());
			}

			std::vector<MemoryAssignment> activeAssignments;
			for (std::size_t nodeIndex = 0; nodeIndex < lifetimes.size(); ++nodeIndex)
			{
				for (const auto& lifetime : lifetimes[nodeIndex])
				{
					const auto byteSize = lifetime.type.ByteSize();
					if (!byteSize)
					{
						throw std::runtime_error(
						    std::format("Memory planner requires static byte size for subgraph {} node {} port {}",
						                lifetime.subgraph, lifetime.value.node, lifetime.value.port));
					}

					const auto& node = subgraph.nodes[lifetime.value.node];
					std::optional<std::size_t> fixedBuffer;
					if (const auto* param = std::get_if<ParamRefNode>(&node.node))
					{
						if (param->paramIndex >= paramBuffers.size())
						{
							throw std::runtime_error("Memory planner found ParamRefNode with invalid parameter index");
						}
						fixedBuffer = paramBuffers[param->paramIndex];
					}
					else if (const auto* variable = std::get_if<VariableRefNode>(&node.node))
					{
						if (variable->variableIndex >= plan.variables.size())
						{
							throw std::runtime_error(
							    "Memory planner found VariableRefNode with invalid variable index");
						}
						fixedBuffer = variable->variableIndex;
					}
					else if (std::holds_alternative<ConstantNode>(node.node) ||
					         std::holds_alternative<QuantizedConstantNode>(node.node))
					{
						fixedBuffer = addBuffer(MemoryBufferKind::Constant, lifetime.type.memorySpace, *byteSize, 1,
						                        memoryPlan.buffers.size());
					}

					std::optional<std::size_t> reusable;
					if (!fixedBuffer)
					{
						for (const auto& assignment : memoryPlan.assignments)
						{
							const auto& buffer = memoryPlan.buffers[assignment.buffer];
							if (buffer.kind != MemoryBufferKind::Workspace ||
							    buffer.memorySpace != lifetime.type.memorySpace || buffer.byteSize < *byteSize)
							{
								continue;
							}
							bool overlaps = false;
							for (const auto& active : activeAssignments)
							{
								if (active.buffer == assignment.buffer &&
								    MemoryLifetimesOverlap(active.lifetime, lifetime))
								{
									overlaps = true;
									break;
								}
							}
							if (!overlaps)
							{
								reusable = assignment.buffer;
								break;
							}
						}
					}

					const auto bufferId = fixedBuffer.value_or(reusable.value_or(memoryPlan.buffers.size()));
					if (!fixedBuffer && !reusable)
					{
						(void) addBuffer(MemoryBufferKind::Workspace, lifetime.type.memorySpace, *byteSize, 1,
						                 bufferId);
					}
					MemoryAssignment assignment{ .subgraph = lifetime.subgraph,
						                         .value = lifetime.value,
						                         .buffer = bufferId,
						                         .offset = 0,
						                         .lifetime = lifetime };
					activeAssignments.push_back(assignment);
					memoryPlan.assignments.push_back(std::move(assignment));
				}
			}
		}

		return memoryPlan;
	}

	inline const MemoryAssignment* FindMemoryAssignment(const MemoryPlan& plan, SubgraphId subgraph, NodeOutput value)
	{
		for (const auto& assignment : plan.assignments)
		{
			if (assignment.subgraph == subgraph && assignment.value == value)
			{
				return &assignment;
			}
		}
		return nullptr;
	}

	inline bool MemoryBufferIsPublicOutput(const MemoryPlan& plan, std::size_t buffer)
	{
		for (const auto& assignment : plan.assignments)
		{
			if (assignment.buffer == buffer && assignment.lifetime.publicOutput)
			{
				return true;
			}
		}
		return false;
	}

	inline bool MemoryBufferIsGraphInput(const MemoryPlan& plan, std::size_t buffer)
	{
		for (const auto& assignment : plan.assignments)
		{
			if (assignment.buffer != buffer)
			{
				continue;
			}
			if (assignment.lifetime.firstUse == assignment.lifetime.value.node)
			{
				return true;
			}
		}
		return false;
	}

	inline DeviceMemoryAllocationKind DeviceAllocationForBuffer(const MemoryBuffer& buffer)
	{
		switch (buffer.memorySpace)
		{
		case TensorMemorySpace::Host:
		case TensorMemorySpace::External:
			return buffer.kind == MemoryBufferKind::External ? DeviceMemoryAllocationKind::ExternalHost
			                                                 : DeviceMemoryAllocationKind::HostVisible;
		case TensorMemorySpace::Unified:
			return DeviceMemoryAllocationKind::Unified;
		case TensorMemorySpace::Constant:
			return DeviceMemoryAllocationKind::ConstantDeviceLocal;
		case TensorMemorySpace::Device:
			if (buffer.kind == MemoryBufferKind::External)
			{
				return DeviceMemoryAllocationKind::ExternalDevice;
			}
			if (buffer.kind == MemoryBufferKind::Constant)
			{
				return DeviceMemoryAllocationKind::ConstantDeviceLocal;
			}
			return DeviceMemoryAllocationKind::DeviceLocal;
		}
		return DeviceMemoryAllocationKind::HostVisible;
	}

	inline DeviceLocalMemoryPlan BuildDeviceLocalMemoryPlan(const MemoryPlan& memoryPlan)
	{
		DeviceLocalMemoryPlan devicePlan;
		devicePlan.buffers.reserve(memoryPlan.buffers.size());
		for (const auto& buffer : memoryPlan.buffers)
		{
			const auto allocation = DeviceAllocationForBuffer(buffer);
			const auto requiresDeviceLocal = allocation == DeviceMemoryAllocationKind::DeviceLocal ||
			                                 allocation == DeviceMemoryAllocationKind::ConstantDeviceLocal;
			const auto needsUpload =
			    requiresDeviceLocal &&
			    (buffer.kind == MemoryBufferKind::Constant || buffer.kind == MemoryBufferKind::Persistent ||
			     (buffer.kind == MemoryBufferKind::External && MemoryBufferIsGraphInput(memoryPlan, buffer.id)));
			const auto needsDownload = requiresDeviceLocal && MemoryBufferIsPublicOutput(memoryPlan, buffer.id);
			const auto requiresHostStaging = needsUpload || needsDownload;

			devicePlan.buffers.push_back({
			    .buffer = buffer.id,
			    .allocation = allocation,
			    .memorySpace = buffer.memorySpace,
			    .kind = buffer.kind,
			    .byteSize = buffer.byteSize,
			    .alignment = buffer.alignment,
			    .requiresDeviceLocal = requiresDeviceLocal,
			    .requiresHostStaging = requiresHostStaging,
			});

			if (requiresDeviceLocal)
			{
				devicePlan.deviceLocalBytes += buffer.byteSize;
				devicePlan.requiresDeviceLocalAllocator = true;
			}
			else
			{
				devicePlan.hostVisibleBytes += buffer.byteSize;
			}
			if (needsUpload)
			{
				devicePlan.stagingSteps.push_back({ .buffer = buffer.id,
				                                    .direction = DeviceMemoryStagingDirection::Upload,
				                                    .byteSize = buffer.byteSize,
				                                    .reason = "device-local buffer requires host-to-device staging" });
				devicePlan.stagingBytes += buffer.byteSize;
				devicePlan.requiresStagingAllocator = true;
			}
			if (needsDownload)
			{
				devicePlan.stagingSteps.push_back(
				    { .buffer = buffer.id,
				      .direction = DeviceMemoryStagingDirection::Download,
				      .byteSize = buffer.byteSize,
				      .reason = "public device-local output requires device-to-host staging" });
				devicePlan.stagingBytes += buffer.byteSize;
				devicePlan.requiresStagingAllocator = true;
			}
		}
		return devicePlan;
	}

	inline void ValidateDeviceLocalMemoryPlan(const MemoryPlan& memoryPlan, const DeviceLocalMemoryPlan& devicePlan)
	{
		if (devicePlan.buffers.size() != memoryPlan.buffers.size())
		{
			throw std::runtime_error("DeviceLocalMemoryPlan buffer count does not match MemoryPlan");
		}
		for (std::size_t i = 0; i < devicePlan.buffers.size(); ++i)
		{
			const auto& planned = devicePlan.buffers[i];
			const auto& source = memoryPlan.buffers[i];
			if (planned.buffer != source.id)
			{
				throw std::runtime_error(
				    std::format("DeviceLocalMemoryPlan buffer {} has mismatched id {}", i, planned.buffer));
			}
			if (planned.byteSize != source.byteSize || planned.alignment != source.alignment ||
			    planned.memorySpace != source.memorySpace || planned.kind != source.kind)
			{
				throw std::runtime_error("DeviceLocalMemoryPlan buffer metadata does not match MemoryPlan");
			}
		}
		for (const auto& step : devicePlan.stagingSteps)
		{
			if (step.buffer >= memoryPlan.buffers.size())
			{
				throw std::runtime_error("DeviceLocalMemoryPlan staging step references an out-of-range buffer");
			}
			if (step.byteSize != memoryPlan.buffers[step.buffer].byteSize)
			{
				throw std::runtime_error("DeviceLocalMemoryPlan staging byte size does not match buffer size");
			}
			if (step.reason.empty())
			{
				throw std::runtime_error("DeviceLocalMemoryPlan staging step requires a reason");
			}
		}
	}

	inline void ValidateMemoryPlan(const ExecutablePlan& executablePlan, const MemoryPlan& memoryPlan)
	{
		ValidateExecutablePlan(executablePlan);
		for (std::size_t i = 0; i < memoryPlan.buffers.size(); ++i)
		{
			const auto& buffer = memoryPlan.buffers[i];
			if (buffer.id != i)
			{
				throw std::runtime_error(std::format("Memory buffer {} has mismatched id {}", i, buffer.id));
			}
			if (buffer.alignment == 0)
			{
				throw std::runtime_error(std::format("Memory buffer {} has zero alignment", i));
			}
		}

		for (const auto& subgraph : executablePlan.subgraphs)
		{
			for (const auto& node : subgraph.nodes)
			{
				for (const auto input : node.inputs)
				{
					const auto& inputType = subgraph.nodes[input.node].outputs[input.port];
					for (const auto& outputType : node.outputs)
					{
						if (inputType.memorySpace != outputType.memorySpace)
						{
							throw std::runtime_error(
							    std::format("Hidden memory-space copy is not allowed in subgraph {} node {} ({})",
							                subgraph.sourceSubgraph, node.sourceNode, node.opKind));
						}
					}
				}
				for (std::size_t outputIndex = 0; outputIndex < node.outputs.size(); ++outputIndex)
				{
					const auto output = NodeOutput{ node.sourceNode, outputIndex };
					const auto* assignment = FindMemoryAssignment(memoryPlan, subgraph.sourceSubgraph, output);
					if (!assignment)
					{
						throw std::runtime_error(
						    std::format("Missing memory assignment for subgraph {} node {} port {}",
						                subgraph.sourceSubgraph, output.node, output.port));
					}
					if (assignment->buffer >= memoryPlan.buffers.size())
					{
						throw std::runtime_error("Memory assignment references an out-of-range buffer");
					}
					if (memoryPlan.buffers[assignment->buffer].memorySpace != node.outputs[outputIndex].memorySpace)
					{
						throw std::runtime_error("Memory assignment buffer memory space does not match output type");
					}
					if (const auto byteSize = node.outputs[outputIndex].ByteSize())
					{
						if (assignment->offset + *byteSize > memoryPlan.buffers[assignment->buffer].byteSize)
						{
							throw std::runtime_error("Memory assignment exceeds buffer size");
						}
					}
				}
			}
		}
	}
} // namespace LiteNN

#endif
