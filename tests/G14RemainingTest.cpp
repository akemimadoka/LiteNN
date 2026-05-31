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
	const auto train = Training::BuildTrainStepPlan(graph, Training::TrainExecutionPolicy::Auto, true);

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
	const auto placement = Runtime::BuildPlacementPlan(graph, backends);

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
	const auto manifest = BuildVNextPackageManifest(graph);
	const auto rules = VNextMigrationRules();

	EXPECT_GE(rules.size(), 7u);
	EXPECT_NO_THROW(ValidateVNextMigrationInvariants(plan, &manifest));
}
