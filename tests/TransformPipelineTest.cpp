#include <gtest/gtest.h>

#include <LiteNN.h>
#include <LiteNN/Pass.h>

#include <array>
#include <span>
#include <vector>

using namespace LiteNN;

namespace
{
	Graph BuildSmallGraph()
	{
		Graph graph;
		Subgraph subgraph;
		const auto input = subgraph.AddParam(DataType::Float32, { 2, 2 });
		const auto neg = subgraph.AddNode(UnaryOpNode{ UnaryOp::Negate, { input, 0 } },
		                                  { OutputInfo{ DataType::Float32, { 2, 2 } } });
		subgraph.SetResults({ { neg, 0 } });
		graph.SetForward(graph.AddSubgraph(std::move(subgraph)));
		graph.SetInputNames({ "x" });
		graph.SetOutputNames({ "y" });
		return graph;
	}

	class RecordingPass : public Detail::GraphMutationPass
	{
	public:
		std::string_view Name() const noexcept override
		{
			return "RecordingPass";
		}

		std::vector<TransformInvalidation> Invalidates() const override
		{
			return { TransformInvalidation::GraphTopology, TransformInvalidation::TypeFacts };
		}

		void Run(Graph& graph) override
		{
			++runs;
			Validation::ValidateGraph(graph);
		}

		int runs{};
	};
} // namespace

TEST(TransformPipeline, RunsModelGraphPassesWithoutCallerOwnedGraphMutation)
{
	RecordingPass pass;
	std::array<Detail::GraphMutationPass*, 1> passes{ &pass };
	std::vector<TransformStepMetadata> dumps;
	TransformPipelineOptions options;
	options.debugDump = [&](const TransformStepMetadata& step) { dumps.push_back(step); };

	auto result = RunModelGraphPassPipeline(ModelGraph{ BuildSmallGraph() },
	                                        std::span<Detail::GraphMutationPass* const>{ passes }, options);

	EXPECT_EQ(pass.runs, 1);
	ASSERT_EQ(result.steps.size(), 1u);
	EXPECT_EQ(result.steps[0].stage, TransformStageKind::ModelGraphToModelGraph);
	EXPECT_EQ(result.steps[0].passName, "RecordingPass");
	EXPECT_EQ(result.steps[0].before.nodes, result.steps[0].after.nodes);
	ASSERT_EQ(result.steps[0].invalidates.size(), 2u);
	ASSERT_EQ(dumps.size(), 1u);
	EXPECT_EQ(dumps[0].passName, "RecordingPass");
	EXPECT_NO_THROW(Validation::ValidateGraph(result.value.UnsafeGraphView()));
}

TEST(TransformPipeline, BuildsExecutablePlanAsTypedStage)
{
	auto result = RunModelToExecutablePlanPipeline(ModelGraph{ BuildSmallGraph() });

	ASSERT_EQ(result.steps.size(), 1u);
	EXPECT_EQ(result.steps[0].stage, TransformStageKind::ModelGraphToExecutablePlan);
	EXPECT_EQ(result.steps[0].passName, "BuildExecutablePlan");
	EXPECT_EQ(result.steps[0].before.subgraphs, 1u);
	EXPECT_EQ(result.steps[0].after.subgraphs, 1u);
	EXPECT_EQ(result.value.inputs[0].name, "x");
	EXPECT_NO_THROW(ValidateExecutablePlan(result.value));
}

TEST(TransformPipeline, RunsExecutablePlanTransformsWithInvalidationMetadata)
{
	auto plan = Detail::BuildExecutablePlanFromGraph(BuildSmallGraph());
	NamedExecutablePlanTransform transform{
		.name = "NoOpPlanTransform",
		.run = [](ExecutablePlan& target) { ValidateExecutablePlan(target); },
		.invalidates = { TransformInvalidation::MemoryPlan, TransformInvalidation::BackendPlacement },
	};

	auto result =
	    RunExecutablePlanPipeline(std::move(plan), std::span<const NamedExecutablePlanTransform>{ &transform, 1 });

	ASSERT_EQ(result.steps.size(), 1u);
	EXPECT_EQ(result.steps[0].stage, TransformStageKind::ExecutablePlanToExecutablePlan);
	EXPECT_EQ(result.steps[0].passName, "NoOpPlanTransform");
	EXPECT_EQ(result.steps[0].before.nodes, result.steps[0].after.nodes);
	ASSERT_EQ(result.steps[0].invalidates.size(), 2u);
	EXPECT_NO_THROW(ValidateExecutablePlan(result.value));
}

TEST(TransformPipeline, BuildsBackendPlanAsTypedStage)
{
	auto plan = Detail::BuildExecutablePlanFromGraph(BuildSmallGraph());
	std::array<std::string_view, 1> backends{ BackendCPUInterpreter };

	auto result =
	    RunExecutablePlanToBackendPlanPipeline(std::move(plan), std::span<const std::string_view>{ backends });

	ASSERT_EQ(result.steps.size(), 1u);
	EXPECT_EQ(result.steps[0].stage, TransformStageKind::ExecutablePlanToBackendPlan);
	ASSERT_EQ(result.value.candidateBackends.size(), 1u);
	EXPECT_EQ(result.value.candidateBackends[0], BackendCPUInterpreter);
	EXPECT_EQ(result.value.module.functions[0].name, "forward");
	EXPECT_EQ(result.value.module.partitions[0].backend, BackendCPUInterpreter);
}
