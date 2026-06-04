#include <gtest/gtest.h>

#include <LiteNN.h>

#include <array>

using namespace LiteNN;

namespace
{
	Graph BuildTrainableGraph()
	{
		Graph graph;
		const auto parameterIndex =
		    graph.AddVariable(Variable::Create(Tensor<CPU>({ 1.0F, 2.0F }, { 2 }, DataType::Float32)));
		graph.SetVariableName(parameterIndex, "linear.weight");

		Subgraph forward;
		const auto x = forward.AddParam(DataType::Float32, { 2 });
		const auto parameter =
		    forward.AddNode(VariableRefNode{ parameterIndex }, { OutputInfo{ DataType::Float32, { 2 } } });
		const auto y = forward.AddNode(BinaryOpNode{ BinaryOp::Multiply, { x, 0 }, { parameter, 0 } },
		                               { OutputInfo{ DataType::Float32, { 2 } } });
		forward.SetResults({ { y, 0 } });
		graph.SetForward(graph.AddSubgraph(std::move(forward)));
		graph.SetInputNames({ "x" });
		graph.SetOutputNames({ "y" });

		Subgraph backwardAndUpdate;
		const auto backwardInput = backwardAndUpdate.AddParam(DataType::Float32, { 2 });
		const auto outputGradient = backwardAndUpdate.AddParam(DataType::Float32, { 2 });
		const auto backwardParameter = backwardAndUpdate.AddNode(
		    VariableRefNode{ parameterIndex }, { OutputInfo{ DataType::Float32, { 2 } } });
		const auto inputGradient = backwardAndUpdate.AddNode(
		    BinaryOpNode{ BinaryOp::Multiply, { outputGradient, 0 }, { backwardParameter, 0 } },
		    { OutputInfo{ DataType::Float32, { 2 } } });
		const auto parameterGradient = backwardAndUpdate.AddNode(
		    BinaryOpNode{ BinaryOp::Multiply, { outputGradient, 0 }, { backwardInput, 0 } },
		    { OutputInfo{ DataType::Float32, { 2 } } });
		const auto update = backwardAndUpdate.AddNode(
		    SGDStepNode{ { backwardParameter, 0 }, { parameterGradient, 0 }, std::nullopt, 0.1, 0.0, 0.0, false },
		    { OutputInfo{ DataType::Float32, { 2 } } });
		backwardAndUpdate.SetResults({ { inputGradient, 0 }, { update, 0 } });
		graph.SetBackward(graph.AddSubgraph(std::move(backwardAndUpdate)));
		return graph;
	}
} // namespace

TEST(G14Remaining, BuildsAndValidatesTrainStepPlan)
{
	const auto graph = BuildTrainableGraph();
	const auto train = Training::BuildTrainStepPlan(BuildExecutableModule(graph), Training::TrainExecutionPolicy::Auto,
	                                                true);

	EXPECT_EQ(train.policy, Training::TrainExecutionPolicy::AOT);
	EXPECT_TRUE(train.backwardFunction.has_value());
	ASSERT_EQ(train.updates.size(), 1u);
	EXPECT_EQ(train.updates[0].opKind, "SGDStepNode");
	EXPECT_FALSE(train.runtimeStates.empty());
	EXPECT_NO_THROW(Training::ValidateTrainStepPlan(train));
}

TEST(G14Remaining, BuildsCostBasedPlacementPlanAndCoverage)
{
	const auto graph = BuildTrainableGraph();
	constexpr std::array<std::string_view, 1> backends{ BackendCPUInterpreter };
	const auto placement = Runtime::BuildPlacementPlan(BuildExecutablePlan(graph), backends);

	EXPECT_FALSE(placement.decisions.empty());
	EXPECT_FALSE(placement.coverage.empty());
	for (const auto& decision : placement.decisions)
	{
		EXPECT_EQ(decision.backend, BackendCPUInterpreter);
		EXPECT_EQ(decision.support, BackendSupportLevel::Native);
		EXPECT_GT(decision.cost, 0.0);
	}
	EXPECT_NO_THROW(Runtime::ValidatePlacementPlan(placement));
}

TEST(G14Remaining, PlacementFallbacksAreExplicitAndCanBeRejected)
{
	Graph graph;
	Subgraph subgraph;
	const auto input = subgraph.AddParam(DataType::Float32, { 1 });
	subgraph.SetResults({ { input, 0 } });
	graph.SetForward(graph.AddSubgraph(std::move(subgraph)));

	auto registry = BuildDefaultOpSchemaRegistry();
	registry.RegisterCapability("ParamRefNode", {
	                                                .backend = std::string(BackendCUDANative),
	                                                .support = BackendSupportLevel::Fallback,
	                                                .fallback = std::string(BackendCPUInterpreter),
	                                                .relativeCost = 1.0,
	                                            });
	constexpr std::array<std::string_view, 1> backends{ BackendCUDANative };
	const auto placement = Runtime::BuildPlacementPlan(BuildExecutablePlan(graph), backends, registry);

	ASSERT_EQ(placement.decisions.size(), 1u);
	EXPECT_EQ(placement.decisions[0].support, BackendSupportLevel::Fallback);
	ASSERT_EQ(placement.fallbackSteps.size(), 1u);
	EXPECT_EQ(placement.fallbackSteps[0].fallbackBackend, BackendCPUInterpreter);
	EXPECT_NO_THROW(Runtime::ValidatePlacementPlan(placement));

	auto schedule = Runtime::BuildRuntimeSchedule(BuildExecutableModule(BuildExecutablePlan(graph)));
	Runtime::AppendPlacementFallbackSteps(schedule, placement);
	ASSERT_FALSE(schedule.steps.empty());
	EXPECT_EQ(schedule.steps.back().kind, Runtime::RuntimeScheduleStepKind::Fallback);
	EXPECT_EQ(schedule.steps.back().backend, BackendCUDANative);
	EXPECT_EQ(schedule.steps.back().fallbackBackend, BackendCPUInterpreter);
	EXPECT_NO_THROW(Runtime::ValidateRuntimeSchedule(schedule));
	const auto trace = Runtime::TraceRuntimeSchedule(schedule);
	ASSERT_FALSE(trace.empty());
	EXPECT_EQ(trace.back().kind, Runtime::RuntimeScheduleStepKind::Fallback);
	EXPECT_NE(trace.back().message.find("fallback from"), std::string::npos);

	EXPECT_THROW((void)Runtime::BuildPlacementPlan(BuildExecutablePlan(graph), backends, registry, {},
	                                              Runtime::PlacementFallbackPolicy::RejectFallback),
	             std::runtime_error);
}

TEST(G14Remaining, ImportManifestTargetsModelGraphAndReportsDiagnostics)
{
	auto manifest = Serialization::BuildImporterOwnedManifest("torch+safetensors", BuildTrainableGraph());
	manifest.weights.push_back({
	    .sourceName = "linear.weight",
	    .graphName = "linear.weight",
	    .sourceType = TensorType::Dense(DataType::Float32, ShapeView{ 2 }),
	    .graphType = TensorType::Dense(DataType::Float32, ShapeView{ 2 }),
	    .layoutConversion = "identity",
	    .quantizationMapping = "none",
	    .loraBinding = "none",
	});
	constexpr std::array<std::string_view, 1> backends{ BackendMobile };
	Serialization::AddImportBackendDiagnostics(manifest, backends);

	EXPECT_EQ(manifest.sourceFormat, "torch+safetensors");
	EXPECT_FALSE(manifest.diagnostics.empty());
	EXPECT_EQ(manifest.diagnostics[0].kind, Serialization::ImportDiagnosticKind::UnsupportedBackendCapability);
	EXPECT_NO_THROW(Serialization::ValidateImporterOwnedManifest(manifest));
}

TEST(G14Remaining, MigrationRulesAreExecutableInvariants)
{
	const auto graph = BuildTrainableGraph();
	const auto plan = BuildExecutablePlan(graph);
	const auto manifest = BuildVNextPackageManifest(BuildExecutableModule(plan));
	const auto rules = VNextMigrationRules();

	EXPECT_GE(rules.size(), 7u);
	EXPECT_NO_THROW(ValidateVNextMigrationInvariants(plan, &manifest));
}
