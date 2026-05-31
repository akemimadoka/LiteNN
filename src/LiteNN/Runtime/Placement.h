#ifndef LITENN_RUNTIME_PLACEMENT_H
#define LITENN_RUNTIME_PLACEMENT_H

#include <LiteNN/ExecutablePlan.h>
#include <LiteNN/MemoryPlan.h>

#include <algorithm>
#include <cstddef>
#include <format>
#include <limits>
#include <span>
#include <stdexcept>
#include <string>
#include <string_view>
#include <vector>

namespace LiteNN::Runtime
{
	struct CostModelWeights
	{
		double opCost{ 1.0 };
		double transferCost{ 8.0 };
		double layoutConversionCost{ 4.0 };
		double compileCacheCost{ 2.0 };
		double precisionPenalty{ 3.0 };
		double workspacePressureCost{ 1.0 / (1024.0 * 1024.0) };
	};

	struct PlacementDecision
	{
		SubgraphId subgraph{};
		NodeId node{};
		std::string opKind;
		std::string backend;
		double cost{};
		BackendSupportLevel support{ BackendSupportLevel::Unsupported };
		std::string fallback;
	};

	struct PlacementPlan
	{
		ExecutablePlan plan;
		MemoryPlan memory;
		std::vector<PlacementDecision> decisions;
		std::vector<ExecutablePartition> partitions;
		std::vector<OpCoverageRow> coverage;
	};

	inline bool CapabilitySupportsType(const BackendCapability& capability, const TensorType& type)
	{
		const auto dtypeSupported = capability.dtypes.empty() ||
		                            std::ranges::find(capability.dtypes, type.dtype) != capability.dtypes.end();
		const auto layoutSupported = capability.layouts.empty() ||
		                             std::ranges::find(capability.layouts, type.layout.kind) != capability.layouts.end();
		const auto memorySupported =
		    capability.memorySpaces.empty() ||
		    std::ranges::find(capability.memorySpaces, type.memorySpace) != capability.memorySpaces.end();
		return dtypeSupported && layoutSupported && memorySupported;
	}

	inline double EstimateNodeCost(const ExecutablePlanNode& node, const BackendCapability& capability,
	                               const MemoryPlan& memory, const CostModelWeights& weights)
	{
		double elements = 1.0;
		if (!node.outputs.empty())
		{
			elements = static_cast<double>(std::max<std::size_t>(1, node.outputs[0].NumElements().value_or(1)));
		}
		double cost = elements * weights.opCost * capability.relativeCost;
		if (capability.support == BackendSupportLevel::Fallback)
		{
			cost += weights.transferCost;
		}
		for (const auto& output : node.outputs)
		{
			if (output.layout.kind != TensorLayoutKind::RowMajor)
			{
				cost += weights.layoutConversionCost;
			}
			if (output.dtype != DataType::Float32)
			{
				cost += weights.precisionPenalty;
			}
		}
		cost += static_cast<double>(memory.workspaceBytes) * weights.workspacePressureCost;
		cost += weights.compileCacheCost;
		return cost;
	}

	inline PlacementDecision ChoosePlacementForNode(const ExecutablePlanNode& node, const MemoryPlan& memory,
	                                               std::span<const std::string_view> candidateBackends,
	                                               const OpSchemaRegistry& registry,
	                                               const CostModelWeights& weights)
	{
		const auto& schema = registry.Require(node.opKind);
		PlacementDecision best{ .node = node.sourceNode,
			                    .opKind = node.opKind,
			                    .cost = std::numeric_limits<double>::infinity() };
		for (const auto backend : candidateBackends)
		{
			const auto* capability = schema.FindCapability(backend);
			if (!capability || capability->support == BackendSupportLevel::Unsupported)
			{
				continue;
			}
			bool legal = true;
			for (const auto& output : node.outputs)
			{
				if (!CapabilitySupportsType(*capability, output))
				{
					legal = false;
					break;
				}
			}
			if (!legal)
			{
				continue;
			}
			const auto cost = EstimateNodeCost(node, *capability, memory, weights);
			if (cost < best.cost)
			{
				best.backend = std::string(backend);
				best.cost = cost;
				best.support = capability->support;
				best.fallback = capability->fallback;
			}
		}
		if (best.backend.empty())
		{
			throw std::runtime_error("No legal backend placement for op: " + node.opKind);
		}
		return best;
	}

	inline PlacementPlan BuildPlacementPlan(ExecutablePlan plan,
	                                        std::span<const std::string_view> candidateBackends =
	                                            std::span<const std::string_view>{ DefaultBackendNames },
	                                        const OpSchemaRegistry& registry = DefaultOpSchemaRegistry(),
	                                        CostModelWeights weights = {})
	{
		ValidateExecutablePlan(plan, registry);
		PlacementPlan placement;
		placement.memory = BuildMemoryPlan(plan);
		ValidateMemoryPlan(plan, placement.memory);
		placement.coverage = registry.CoverageReport(candidateBackends);

		for (const auto& subgraph : plan.subgraphs)
		{
			for (const auto& node : subgraph.nodes)
			{
				auto decision = ChoosePlacementForNode(node, placement.memory, candidateBackends, registry, weights);
				decision.subgraph = subgraph.sourceSubgraph;
				placement.decisions.push_back(std::move(decision));
			}
		}

		for (const auto backend : candidateBackends)
		{
			ExecutablePartition partition;
			partition.id = placement.partitions.size();
			partition.backend = std::string(backend);
			for (const auto& decision : placement.decisions)
			{
				if (decision.backend == backend)
				{
					partition.regions.push_back(decision.subgraph);
				}
			}
			std::ranges::sort(partition.regions);
			partition.regions.erase(std::ranges::unique(partition.regions).begin(), partition.regions.end());
			if (!partition.regions.empty())
			{
				placement.partitions.push_back(std::move(partition));
			}
		}

		placement.plan = std::move(plan);
		return placement;
	}

	inline PlacementPlan BuildPlacementPlan(const Graph& graph,
	                                        std::span<const std::string_view> candidateBackends =
	                                            std::span<const std::string_view>{ DefaultBackendNames },
	                                        const OpSchemaRegistry& registry = DefaultOpSchemaRegistry(),
	                                        CostModelWeights weights = {})
	{
		return BuildPlacementPlan(BuildExecutablePlan(graph, registry), candidateBackends, registry, weights);
	}

	inline void ValidatePlacementPlan(const PlacementPlan& placement)
	{
		ValidateExecutablePlan(placement.plan);
		ValidateMemoryPlan(placement.plan, placement.memory);
		if (placement.decisions.empty())
		{
			throw std::runtime_error("PlacementPlan contains no node decisions");
		}
		for (const auto& decision : placement.decisions)
		{
			if (decision.backend.empty())
			{
				throw std::runtime_error("PlacementPlan decision has empty backend");
			}
			if (decision.support == BackendSupportLevel::Unsupported)
			{
				throw std::runtime_error("PlacementPlan decision uses unsupported backend");
			}
		}
	}
} // namespace LiteNN::Runtime

#endif
