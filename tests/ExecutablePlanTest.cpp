#include <gtest/gtest.h>

#include <LiteNN.h>

#include <variant>

using namespace LiteNN;

namespace
{
	Graph BuildSmallGraph()
	{
		Graph graph;
		const auto variable = graph.AddVariable(Variable::Create(Tensor<CPU>(ShapeView{ 2, 2 }, DataType::Float32)));

		Subgraph subgraph;
		const auto input = subgraph.AddParam(DataType::Float32, { 2, 2 });
		const auto bias =
		    subgraph.AddNode(VariableRefNode{ variable }, { OutputInfo{ DataType::Float32, { 2, 2 } } });
		const auto add = subgraph.AddNode(BinaryOpNode{ BinaryOp::Add, { input, 0 }, { bias, 0 } },
		                                  { OutputInfo{ DataType::Float32, { 2, 2 } } });
		subgraph.SetResults({ { add, 0 } });

		graph.SetForward(graph.AddSubgraph(std::move(subgraph)));
		graph.SetInputNames({ "x" });
		graph.SetOutputNames({ "y" });
		return graph;
	}
} // namespace

TEST(ExecutablePlanTest, BuildsPlanFromGraphSnapshot)
{
	const auto graph = BuildSmallGraph();
	const auto plan = Detail::BuildExecutablePlanFromGraph(graph);

	ASSERT_EQ(plan.subgraphs.size(), 1);
	EXPECT_EQ(plan.forward, graph.Forward());
	EXPECT_FALSE(plan.backward.has_value());

	const auto& subgraph = plan.subgraphs[0];
	ASSERT_EQ(subgraph.params.size(), 1);
	EXPECT_EQ(subgraph.params[0].StaticShape(), (std::vector<std::size_t>{ 2, 2 }));
	ASSERT_EQ(subgraph.nodes.size(), 3);
	EXPECT_EQ(subgraph.nodes[0].opKind, "ParamRefNode");
	EXPECT_EQ(subgraph.nodes[1].opKind, "VariableRefNode");
	EXPECT_EQ(subgraph.nodes[2].opKind, "BinaryOpNode");
	EXPECT_EQ(subgraph.nodes[2].op.kind, "BinaryOpNode");
	EXPECT_EQ(subgraph.nodes[2].op.category, OpCategory::Elementwise);
	EXPECT_FALSE(subgraph.nodes[2].op.attributes.empty());
	EXPECT_EQ(subgraph.nodes[2].op.attributes[0].name, "op");
	EXPECT_TRUE(std::holds_alternative<BinaryOpNode>(subgraph.nodes[2].node));
	EXPECT_EQ(subgraph.nodes[2].category, OpCategory::Elementwise);
	ASSERT_EQ(subgraph.nodes[2].inputs.size(), 2);
	EXPECT_EQ(subgraph.nodes[2].outputs[0].dtype, DataType::Float32);

	ASSERT_EQ(plan.variables.size(), 1);
	EXPECT_FALSE(plan.variables[0].IsExternal());
	EXPECT_EQ(plan.variables[0].type.memorySpace, TensorMemorySpace::Host);
	EXPECT_EQ(plan.variables[0].region.memorySpace, TensorMemorySpace::Host);
	EXPECT_EQ(plan.variables[0].region.name, "variable0");
	ASSERT_EQ(plan.inputs.size(), 1);
	EXPECT_EQ(plan.inputs[0].name, "x");
	ASSERT_EQ(plan.outputs.size(), 1);
	EXPECT_EQ(plan.outputs[0].name, "y");
	EXPECT_NO_THROW(ValidateExecutablePlan(plan));
}

TEST(ExecutablePlanTest, BuildsPlanFromModelGraphWrapper)
{
	ModelGraph model(BuildSmallGraph());
	const auto plan = BuildExecutablePlan(model);

	ASSERT_EQ(plan.subgraphs.size(), 1);
	ASSERT_EQ(plan.outputs.size(), 1);
	EXPECT_EQ(plan.outputs[0].type.StaticShape(), (std::vector<std::size_t>{ 2, 2 }));
}

TEST(ExecutablePlanTest, PreservesQuantizedVariableStorageMetadata)
{
	const Tensor<CPU> source({ 1.0F, 2.0F, 3.0F, 4.0F }, { 2, 2 }, DataType::Float32);
	const auto quantized = QuantizeAffine(source, PerTensorAffineQuantization(DataType::Int8, 0.5F, -1));

	Graph graph;
	const auto variable = graph.AddVariable(Variable::CreateQuantized(quantized.Storage(), quantized.Params()));
	Subgraph subgraph;
	const auto weight = subgraph.AddNode(VariableRefNode{ variable }, { OutputInfo{ DataType::Int8, { 2, 2 } } });
	subgraph.SetResults({ { weight, 0 } });
	graph.SetForward(graph.AddSubgraph(std::move(subgraph)));
	graph.SetOutputNames({ "weight" });

	const auto plan = Detail::BuildExecutablePlanFromGraph(graph);
	ASSERT_EQ(plan.variables.size(), 1);
	ASSERT_TRUE(plan.variables[0].quantization.has_value());
	EXPECT_EQ(plan.variables[0].quantization->storageType, DataType::Int8);
	const auto view = plan.variables[0].View();
	ASSERT_TRUE(view.quantization.has_value());
	EXPECT_EQ(view.quantization->zeroPoints[0], -1);
}

TEST(ExecutablePlanTest, BuildsExecutableModuleWithFunctionsRegionsAndPartitions)
{
	const auto module = Detail::BuildExecutableModuleFromGraph(BuildSmallGraph());

	ASSERT_EQ(module.functions.size(), 1);
	EXPECT_EQ(module.functions[0].name, "forward");
	EXPECT_EQ(module.functions[0].body, module.plan.forward);
	ASSERT_EQ(module.functions[0].inputs.size(), 1);
	ASSERT_EQ(module.functions[0].outputs.size(), 1);

	ASSERT_EQ(module.regions.size(), 1);
	EXPECT_EQ(module.regions[0].function, module.functions[0].id);
	EXPECT_EQ(module.regions[0].subgraph, module.plan.forward);
	EXPECT_EQ(module.regions[0].nodes.size(), module.plan.subgraphs[0].nodes.size());

	ASSERT_EQ(module.partitions.size(), 1);
	EXPECT_EQ(module.partitions[0].backend, BackendCPUInterpreter);
	ASSERT_EQ(module.partitions[0].regions.size(), 1);
	EXPECT_EQ(module.partitions[0].regions[0], module.regions[0].id);
}

TEST(ExecutablePlanTest, ValidationRejectsSchemaAndReferenceErrors)
{
	auto plan = Detail::BuildExecutablePlanFromGraph(BuildSmallGraph());
	auto badKind = plan;
	badKind.subgraphs[0].nodes[2].op.kind = "MissingNode";
	EXPECT_THROW(ValidateExecutablePlan(badKind), std::runtime_error);

	auto badSchema = plan;
	badSchema.subgraphs[0].nodes[2].op.schemaId = 999;
	EXPECT_THROW(ValidateExecutablePlan(badSchema), std::runtime_error);

	auto badInput = plan;
	badInput.subgraphs[0].nodes[2].inputs[0] = { 99, 0 };
	EXPECT_THROW(ValidateExecutablePlan(badInput), std::runtime_error);

	auto badOutput = plan;
	badOutput.subgraphs[0].nodes[2].outputs.clear();
	EXPECT_THROW(ValidateExecutablePlan(badOutput), std::runtime_error);

	auto badInputSignature = plan;
	badInputSignature.inputs.clear();
	EXPECT_THROW(ValidateExecutablePlan(badInputSignature), std::runtime_error);
}

TEST(ExecutablePlanTest, ReportsBackendUnsupportedOpsBeforeLowering)
{
	const auto plan = Detail::BuildExecutablePlanFromGraph(BuildSmallGraph());

	EXPECT_NO_THROW(RequireExecutablePlanBackendSupport(plan, BackendCPUInterpreter));

	const auto issues = CollectExecutablePlanBackendIssues(plan, BackendCPUAOT);
	ASSERT_EQ(issues.size(), plan.subgraphs[0].nodes.size());
	EXPECT_EQ(issues[0].subgraph, 0);
	EXPECT_EQ(issues[0].node, 0);
	EXPECT_EQ(issues[0].support, BackendSupportLevel::Unsupported);
	EXPECT_EQ(issues[0].fallback, BackendCPUInterpreter);
	EXPECT_THROW(RequireExecutablePlanBackendSupport(plan, BackendCPUAOT), std::runtime_error);
}
