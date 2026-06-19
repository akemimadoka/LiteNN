#ifndef LITENN_PASS_H
#define LITENN_PASS_H

#include <LiteNN/ExecutablePlan.h>
#include <LiteNN/Misc.h>
#include <LiteNN/Validation/GraphValidator.h>

#include <cstddef>
#include <functional>
#include <span>
#include <stdexcept>
#include <string>
#include <string_view>
#include <utility>
#include <vector>

namespace LiteNN
{
	enum class TransformStageKind
	{
		ModelGraphToModelGraph,
		ModelGraphToExecutablePlan,
		ExecutablePlanToExecutablePlan,
		ExecutablePlanToBackendPlan
	};

	inline std::string_view TransformStageKindName(TransformStageKind stage) noexcept
	{
		return EnumToString<EnumToStringStyle::Unqualified>(stage);
	}

	enum class TransformInvalidation
	{
		None,
		GraphTopology,
		TypeFacts,
		SchemaFacts,
		ExecutablePlan,
		MemoryPlan,
		BackendPlacement,
		CodegenCache,
		StateBindings
	};

	struct TransformStats
	{
		std::size_t subgraphs{};
		std::size_t nodes{};
		std::size_t variables{};
		std::size_t functions{};
		std::size_t regions{};
		std::size_t partitions{};
	};

	struct TransformStepMetadata
	{
		TransformStageKind stage{ TransformStageKind::ModelGraphToModelGraph };
		std::string passName;
		std::vector<TransformInvalidation> invalidates;
		TransformStats before;
		TransformStats after;
	};

	struct TransformPipelineOptions
	{
		bool validateAfterEachStep{ true };
		std::function<void(const TransformStepMetadata&)> debugDump;
	};

	template <typename T>
	struct TransformResult
	{
		T value;
		std::vector<TransformStepMetadata> steps;
	};

	struct BackendPlan
	{
		ExecutableModule module;
		std::vector<std::string> candidateBackends;
	};

	namespace Detail
	{
		/// Internal construction-time Graph -> Graph mutation pass.
		///
		/// Production pipelines should prefer the typed transform stages below. This adapter remains for
		/// construction-time graph rewrites such as autograd, inlining, constant folding, fusion, and e-graph rewrites.
		struct GraphMutationPass
		{
			virtual ~GraphMutationPass() = default;
			virtual std::string_view Name() const noexcept
			{
				return "GraphMutationPass";
			}
			virtual std::vector<TransformInvalidation> Invalidates() const
			{
				return { TransformInvalidation::GraphTopology, TransformInvalidation::TypeFacts,
					     TransformInvalidation::ExecutablePlan };
			}
			virtual void Run(Graph& graph) = 0;
		};
	} // namespace Detail

	struct ModelGraphTransformStage
	{
	};
	struct ModelToExecutablePlanTransformStage
	{
	};
	struct ExecutablePlanTransformStage
	{
	};
	struct BackendPlanTransformStage
	{
	};

	template <typename Stage>
	struct TransformStageTraits;

	template <>
	struct TransformStageTraits<ModelGraphTransformStage>
	{
		using Input = ModelGraph;
		using Output = ModelGraph;
		static constexpr TransformStageKind kind = TransformStageKind::ModelGraphToModelGraph;
	};

	template <>
	struct TransformStageTraits<ModelToExecutablePlanTransformStage>
	{
		using Input = ModelGraph;
		using Output = ExecutablePlan;
		static constexpr TransformStageKind kind = TransformStageKind::ModelGraphToExecutablePlan;
	};

	template <>
	struct TransformStageTraits<ExecutablePlanTransformStage>
	{
		using Input = ExecutablePlan;
		using Output = ExecutablePlan;
		static constexpr TransformStageKind kind = TransformStageKind::ExecutablePlanToExecutablePlan;
	};

	template <>
	struct TransformStageTraits<BackendPlanTransformStage>
	{
		using Input = ExecutablePlan;
		using Output = BackendPlan;
		static constexpr TransformStageKind kind = TransformStageKind::ExecutablePlanToBackendPlan;
	};

	struct NamedExecutablePlanTransform
	{
		std::string name;
		std::function<void(ExecutablePlan&)> run;
		std::vector<TransformInvalidation> invalidates{ TransformInvalidation::ExecutablePlan,
			                                            TransformInvalidation::MemoryPlan,
			                                            TransformInvalidation::BackendPlacement,
			                                            TransformInvalidation::CodegenCache };
	};

	inline TransformStats CollectTransformStats(const Graph& graph)
	{
		TransformStats stats;
		stats.subgraphs = graph.SubgraphCount();
		stats.variables = graph.VariableCount();
		for (std::size_t i = 0; i < graph.SubgraphCount(); ++i)
		{
			stats.nodes += graph.GetSubgraph(i).NodeCount();
		}
		return stats;
	}

	inline TransformStats CollectTransformStats(const ModelGraph& model)
	{
		return CollectTransformStats(model.UnsafeGraphView());
	}

	inline TransformStats CollectTransformStats(const ExecutablePlan& plan)
	{
		TransformStats stats;
		stats.subgraphs = plan.subgraphs.size();
		stats.variables = plan.variables.size();
		for (const auto& subgraph : plan.subgraphs)
		{
			stats.nodes += subgraph.nodes.size();
		}
		return stats;
	}

	inline TransformStats CollectTransformStats(const ExecutableModule& module)
	{
		auto stats = CollectTransformStats(module.plan);
		stats.functions = module.functions.size();
		stats.regions = module.regions.size();
		stats.partitions = module.partitions.size();
		return stats;
	}

	inline TransformStats CollectTransformStats(const BackendPlan& plan)
	{
		return CollectTransformStats(plan.module);
	}

	inline void EmitTransformStep(std::vector<TransformStepMetadata>& steps, TransformStepMetadata step,
	                              const TransformPipelineOptions& options)
	{
		if (options.debugDump)
		{
			options.debugDump(step);
		}
		steps.push_back(std::move(step));
	}

	inline TransformResult<ModelGraph> RunModelGraphPassPipeline(ModelGraph model,
	                                                             std::span<Detail::GraphMutationPass* const> passes,
	                                                             const TransformPipelineOptions& options = {})
	{
		std::vector<TransformStepMetadata> steps;
		for (Detail::GraphMutationPass* pass : passes)
		{
			if (pass == nullptr)
			{
				throw std::runtime_error("ModelGraph transform pipeline contains a null pass");
			}
			const auto before = CollectTransformStats(model);
			pass->Run(model.UnsafeMutableGraph());
			if (options.validateAfterEachStep)
			{
				Validation::ValidateGraph(model.UnsafeGraphView());
			}
			EmitTransformStep(steps,
			                  { .stage = TransformStageKind::ModelGraphToModelGraph,
			                    .passName = std::string(pass->Name()),
			                    .invalidates = pass->Invalidates(),
			                    .before = before,
			                    .after = CollectTransformStats(model) },
			                  options);
		}
		return { .value = std::move(model), .steps = std::move(steps) };
	}

	inline TransformResult<ExecutablePlan>
	RunModelToExecutablePlanPipeline(ModelGraph model, const OpSchemaRegistry& registry = DefaultOpSchemaRegistry(),
	                                 const TransformPipelineOptions& options = {})
	{
		if (options.validateAfterEachStep)
		{
			Validation::ValidateGraph(model.UnsafeGraphView());
		}
		const auto before = CollectTransformStats(model);
		auto plan = BuildExecutablePlan(model, registry);
		if (options.validateAfterEachStep)
		{
			ValidateExecutablePlan(plan, registry);
		}
		std::vector<TransformStepMetadata> steps;
		EmitTransformStep(
		    steps,
		    { .stage = TransformStageKind::ModelGraphToExecutablePlan,
		      .passName = "BuildExecutablePlan",
		      .invalidates = { TransformInvalidation::ExecutablePlan, TransformInvalidation::MemoryPlan,
		                       TransformInvalidation::BackendPlacement, TransformInvalidation::CodegenCache },
		      .before = before,
		      .after = CollectTransformStats(plan) },
		    options);
		return { .value = std::move(plan), .steps = std::move(steps) };
	}

	inline TransformResult<ExecutablePlan>
	RunExecutablePlanPipeline(ExecutablePlan plan, std::span<const NamedExecutablePlanTransform> transforms,
	                          const TransformPipelineOptions& options = {},
	                          const OpSchemaRegistry& registry = DefaultOpSchemaRegistry())
	{
		std::vector<TransformStepMetadata> steps;
		for (const auto& transform : transforms)
		{
			if (!transform.run)
			{
				throw std::runtime_error("ExecutablePlan transform pipeline contains an empty transform");
			}
			const auto before = CollectTransformStats(plan);
			transform.run(plan);
			if (options.validateAfterEachStep)
			{
				ValidateExecutablePlan(plan, registry);
			}
			EmitTransformStep(steps,
			                  { .stage = TransformStageKind::ExecutablePlanToExecutablePlan,
			                    .passName = transform.name.empty() ? "ExecutablePlanTransform" : transform.name,
			                    .invalidates = transform.invalidates,
			                    .before = before,
			                    .after = CollectTransformStats(plan) },
			                  options);
		}
		return { .value = std::move(plan), .steps = std::move(steps) };
	}

	inline TransformResult<BackendPlan> RunExecutablePlanToBackendPlanPipeline(
	    ExecutablePlan plan,
	    std::span<const std::string_view> candidateBackends = std::span<const std::string_view>{ DefaultBackendNames },
	    const TransformPipelineOptions& options = {}, const OpSchemaRegistry& registry = DefaultOpSchemaRegistry())
	{
		const auto before = CollectTransformStats(plan);
		if (options.validateAfterEachStep)
		{
			ValidateExecutablePlan(plan, registry);
		}
		BackendPlan backendPlan;
		backendPlan.module = BuildExecutableModule(std::move(plan));
		backendPlan.candidateBackends.reserve(candidateBackends.size());
		for (const auto backend : candidateBackends)
		{
			backendPlan.candidateBackends.emplace_back(backend);
		}
		std::vector<TransformStepMetadata> steps;
		EmitTransformStep(
		    steps,
		    { .stage = TransformStageKind::ExecutablePlanToBackendPlan,
		      .passName = "BuildBackendPlan",
		      .invalidates = { TransformInvalidation::BackendPlacement, TransformInvalidation::CodegenCache },
		      .before = before,
		      .after = CollectTransformStats(backendPlan) },
		    options);
		return { .value = std::move(backendPlan), .steps = std::move(steps) };
	}
} // namespace LiteNN

#endif
