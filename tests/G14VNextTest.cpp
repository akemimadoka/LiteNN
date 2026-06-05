#include <gtest/gtest.h>

#include <LiteNN.h>

#include <filesystem>
#include <fstream>
#include <iterator>

using namespace LiteNN;

namespace
{
	Graph BuildLinearAddGraph()
	{
		Graph graph;
		Tensor<CPU> bias({ 10.0F, 20.0F, 30.0F, 40.0F }, { 2, 2 }, DataType::Float32);
		const auto variable = graph.AddVariable(Variable::CreateFrozen(std::move(bias)));
		graph.SetVariableName(variable, "linear.bias");

		Subgraph subgraph;
		const auto input = subgraph.AddParam(DataType::Float32, { 2, 2 });
		const auto biasNode =
		    subgraph.AddNode(VariableRefNode{ variable }, { OutputInfo{ DataType::Float32, { 2, 2 } } });
		const auto output = subgraph.AddNode(BinaryOpNode{ BinaryOp::Add, { input, 0 }, { biasNode, 0 } },
		                                     { OutputInfo{ DataType::Float32, { 2, 2 } } });
		subgraph.SetResults({ { output, 0 } });

		graph.SetForward(graph.AddSubgraph(std::move(subgraph)));
		graph.SetInputNames({ "x" });
		graph.SetOutputNames({ "y" });
		return graph;
	}
} // namespace

TEST(G14VNext, BuildsManifestWithTensorArtifactAndCoverageTables)
{
	const auto graph = BuildLinearAddGraph();
	auto module = BuildExecutableModule(graph);
	VNextArtifactRef artifact;
	artifact.name = "cpu_forward";
	artifact.backend = std::string(BackendCPUAOT);
	artifact.entries.push_back({ .name = "forward",
		                         .function = 0,
		                         .requiredBufferBindings = { "linear.bias" } });
	artifact.regions.push_back({ .name = "instructions",
	                             .kind = ExternalBufferKind::ObjectFile,
	                             .relativePath = "artifacts/cpu_forward.o",
	                             .byteOffset = 0,
	                             .byteSize = 128,
	                             .checksum = 123 });

	const auto manifest = BuildVNextPackageManifest(module, { artifact });

	EXPECT_EQ(manifest.versions.manifest, 1u);
	ASSERT_EQ(manifest.functions.size(), 1u);
	EXPECT_EQ(manifest.functions[0].name, "forward");
	ASSERT_EQ(manifest.tensors.size(), 1u);
	EXPECT_EQ(manifest.tensors[0].name, "linear.bias");
	EXPECT_EQ(manifest.tensors[0].type.StaticShape(), (std::vector<std::size_t>{ 2, 2 }));
	ASSERT_EQ(manifest.artifacts.size(), 1u);
	EXPECT_EQ(manifest.artifacts[0].backend, BackendCPUAOT);
	ASSERT_EQ(manifest.artifacts[0].entries.size(), 1u);
	EXPECT_EQ(manifest.artifacts[0].entries[0].name, "forward");
	EXPECT_FALSE(manifest.runtimeSteps.empty());
	EXPECT_GT(manifest.memory.buffers.size(), 0u);
	ASSERT_FALSE(manifest.bufferBindings.empty());
	EXPECT_EQ(manifest.bufferBindings[0].name, "linear.bias");
	EXPECT_FALSE(manifest.opCoverage.empty());
	EXPECT_NO_THROW(ValidateVNextPackageManifest(manifest));
}

TEST(G14VNext, VNextModelPackageRoundTripsManifestAndExecutablePlan)
{
	const auto graph = BuildLinearAddGraph();
	auto module = BuildExecutableModule(graph);
	VNextArtifactRef artifact;
	artifact.name = "cpu_forward";
	artifact.backend = std::string(BackendCPUAOT);
	artifact.entries.push_back({ .name = "forward",
		                         .function = 0,
		                         .requiredBufferBindings = { "linear.bias" } });
	artifact.regions.push_back({ .name = "instructions",
	                             .kind = ExternalBufferKind::ObjectFile,
	                             .relativePath = "artifacts/cpu_forward.o",
	                             .byteOffset = 0,
	                             .byteSize = 128,
	                             .checksum = 123 });

	const auto path = std::filesystem::temp_directory_path() / "litenn_vnext_package_roundtrip.json";
	Serialization::SaveVNextModelPackage(module, path, { artifact });
	{
		std::ifstream input(path, std::ios::binary);
		const std::string json((std::istreambuf_iterator<char>(input)), std::istreambuf_iterator<char>());
		EXPECT_NE(json.find("\"op\":"), std::string::npos);
		EXPECT_EQ(json.find("\"opKind\""), std::string::npos);
		EXPECT_NE(json.find("\"bufferBindings\""), std::string::npos);
		EXPECT_NE(json.find("\"entries\""), std::string::npos);
		EXPECT_EQ(json.find("\"entryFunction\""), std::string::npos);
	}
	const auto package = Serialization::LoadVNextModelPackage(path);
	std::filesystem::remove(path);

	EXPECT_NO_THROW(ValidateVNextPackageManifest(package.manifest));
	EXPECT_NO_THROW(ValidateExecutablePlan(package.plan));
	ASSERT_EQ(package.manifest.functions.size(), 1u);
	EXPECT_EQ(package.manifest.functions[0].name, "forward");
	ASSERT_EQ(package.manifest.tensors.size(), 1u);
	EXPECT_EQ(package.manifest.tensors[0].name, "linear.bias");
	ASSERT_FALSE(package.manifest.bufferBindings.empty());
	EXPECT_EQ(package.manifest.bufferBindings[0].name, "linear.bias");
	ASSERT_EQ(package.manifest.artifacts.size(), 1u);
	EXPECT_EQ(package.manifest.artifacts[0].backend, BackendCPUAOT);
	ASSERT_EQ(package.manifest.artifacts[0].entries.size(), 1u);
	EXPECT_EQ(package.manifest.artifacts[0].entries[0].requiredBufferBindings[0], "linear.bias");
	ASSERT_EQ(package.plan.subgraphs.size(), module.plan.subgraphs.size());
	EXPECT_EQ(package.plan.subgraphs[package.plan.forward].nodes[2].op.kind, "BinaryOpNode");
	EXPECT_FALSE(package.plan.subgraphs[package.plan.forward].nodes[2].op.attributes.empty());
	EXPECT_EQ(package.plan.outputs[0].name, "y");
}

TEST(G14VNext, VNextModelPackageExternalWeightsBindLoadedPlanStorage)
{
	const auto graph = BuildLinearAddGraph();
	const auto root = std::filesystem::temp_directory_path();
	const auto path = root / "litenn_vnext_external_weights.json";
	const auto weightsPath = root / "litenn_vnext_external_weights.bin";

	Serialization::ExternalWeightSaveOptions options;
	options.minVariableBytes = 0;
	Serialization::SaveVNextModelPackageExternalWeights(graph, path, weightsPath, options);

	const auto package = Serialization::LoadVNextModelPackage(path);
	ASSERT_EQ(package.plan.variables.size(), 1u);
	EXPECT_TRUE(package.plan.variables[0].IsExternal());
	ASSERT_NE(package.plan.variables[0].region.data, nullptr);
	EXPECT_EQ(package.plan.variables[0].region.owner.use_count(), 1);

	const auto& region = package.plan.variables[0].region;
	ASSERT_GE(region.byteSize, 4u * sizeof(float));
	const auto* bytes = static_cast<const std::byte*>(region.data);
	ASSERT_NE(bytes, nullptr);
	const auto* values = reinterpret_cast<const float*>(bytes + region.byteOffset);
	EXPECT_FLOAT_EQ(values[0], 10.0F);
	EXPECT_FLOAT_EQ(values[1], 20.0F);
	EXPECT_FLOAT_EQ(values[2], 30.0F);
	EXPECT_FLOAT_EQ(values[3], 40.0F);

	Runtime::Interpreter<CPU> interpreter;
	std::vector<Tensor<CPU>> inputs;
	inputs.emplace_back(Tensor<CPU>({ 1.0F, 2.0F, 3.0F, 4.0F }, { 2, 2 }, DataType::Float32));
	const auto outputs = interpreter.RunForward(package.plan, inputs);
	ASSERT_EQ(outputs.size(), 1u);
	const auto* outputValues = static_cast<const float*>(outputs[0].RawData());
	EXPECT_FLOAT_EQ(outputValues[0], 11.0F);
	EXPECT_FLOAT_EQ(outputValues[1], 22.0F);
	EXPECT_FLOAT_EQ(outputValues[2], 33.0F);
	EXPECT_FLOAT_EQ(outputValues[3], 44.0F);

	std::filesystem::remove(path);
	std::filesystem::remove(weightsPath);
}

TEST(G14VNext, VNextModelPackageRejectsLegacyFormat)
{
	const auto path = std::filesystem::temp_directory_path() / "litenn_vnext_package_legacy.json";
	{
		std::ofstream out(path, std::ios::binary);
		out << "{\"format\":\"litenn.legacy.graph\"}";
	}
	EXPECT_THROW((void)Serialization::LoadVNextModelPackage(path), std::runtime_error);
	std::filesystem::remove(path);
}

TEST(G14VNext, BuildsRuntimeScheduleWithStateBindingsAndTrace)
{
	const auto graph = BuildLinearAddGraph();
	auto kvCache = Runtime::MakeKVCacheState("kv.cache.0",
	                                         TensorType::Dense(DataType::Float32, ShapeView{ 1, 2, 4 }));

	const auto schedule = Runtime::BuildRuntimeSchedule(BuildExecutableModule(graph), { kvCache });

	ASSERT_EQ(schedule.steps.size(), 3u);
	EXPECT_EQ(schedule.steps[0].kind, Runtime::RuntimeScheduleStepKind::StateRead);
	EXPECT_EQ(schedule.steps[1].backend, BackendCPUInterpreter);
	EXPECT_EQ(schedule.steps[1].kind, Runtime::RuntimeScheduleStepKind::DispatchRegion);
	EXPECT_EQ(schedule.steps[2].kind, Runtime::RuntimeScheduleStepKind::StateWrite);
	ASSERT_EQ(schedule.states.size(), 1u);
	ASSERT_TRUE(schedule.states[0].memoryBuffer.has_value());
	EXPECT_LT(*schedule.states[0].memoryBuffer, schedule.memory.buffers.size());
	ASSERT_GE(schedule.bufferBindings.size(), 2u);
	EXPECT_EQ(schedule.bufferBindings.back().name, "kv.cache.0");
	EXPECT_NO_THROW(Runtime::ValidateRuntimeSchedule(schedule));
	const auto trace = Runtime::TraceRuntimeSchedule(schedule);
	ASSERT_EQ(trace.size(), 3u);
	EXPECT_EQ(trace[1].step, 1u);
	EXPECT_NE(trace[1].message.find("dispatch region"), std::string::npos);
}

TEST(G14VNext, MemoryPlanAssignsStaticValuesAndReusesWorkspace)
{
	const auto graph = BuildLinearAddGraph();
	const auto plan = BuildExecutablePlan(graph);
	const auto memory = BuildMemoryPlan(plan);

	EXPECT_EQ(memory.externalVariables.size(), 1u);
	EXPECT_GE(memory.buffers.size(), memory.externalVariables.size());
	EXPECT_GT(memory.workspaceBytes, 0u);
	EXPECT_GT(memory.persistentBytes, 0u);
	EXPECT_GT(memory.externalBytes, 0u);
	EXPECT_NO_THROW(ValidateMemoryPlan(plan, memory));

	const auto& subgraph = plan.subgraphs[plan.forward];
	for (const auto& node : subgraph.nodes)
	{
		for (std::size_t output = 0; output < node.outputs.size(); ++output)
		{
			EXPECT_NE(FindMemoryAssignment(memory, subgraph.sourceSubgraph, { node.sourceNode, output }), nullptr);
		}
	}

	const auto* inputAssignment = FindMemoryAssignment(memory, subgraph.sourceSubgraph, { 0, 0 });
	ASSERT_NE(inputAssignment, nullptr);
	EXPECT_EQ(memory.buffers[inputAssignment->buffer].kind, MemoryBufferKind::External);
	const auto* variableAssignment = FindMemoryAssignment(memory, subgraph.sourceSubgraph, { 1, 0 });
	ASSERT_NE(variableAssignment, nullptr);
	EXPECT_EQ(memory.buffers[variableAssignment->buffer].kind, MemoryBufferKind::Persistent);
}

TEST(G14VNext, ManifestValidationRejectsInvalidVersionsAndArtifacts)
{
	auto manifest = BuildVNextPackageManifest(BuildExecutableModule(BuildLinearAddGraph()));
	manifest.versions.artifactABI = 0;
	EXPECT_THROW(ValidateVNextPackageManifest(manifest), std::runtime_error);

	manifest = BuildVNextPackageManifest(BuildExecutableModule(BuildLinearAddGraph()));
	manifest.artifacts.push_back({ .name = "broken", .backend = "", .entries = { { .name = "forward" } } });
	EXPECT_THROW(ValidateVNextPackageManifest(manifest), std::runtime_error);

	manifest.artifacts[0].backend = std::string(BackendCPUAOT);
	manifest.artifacts[0].entries[0].function = 99;
	EXPECT_THROW(ValidateVNextPackageManifest(manifest), std::runtime_error);

	manifest = BuildVNextPackageManifest(BuildExecutableModule(BuildLinearAddGraph()));
	manifest.artifacts.push_back({ .name = "missing-binding",
		                           .backend = std::string(BackendCPUAOT),
		                           .entries = { { .name = "forward",
		                                          .function = 0,
		                                          .requiredBufferBindings = { "missing.weight" } } },
		                           .regions = { { .name = "instructions",
		                                          .relativePath = "artifacts/missing.o",
		                                          .byteSize = 1 } } });
	EXPECT_THROW(ValidateVNextPackageManifest(manifest), std::runtime_error);

	manifest = BuildVNextPackageManifest(BuildExecutableModule(BuildLinearAddGraph()));
	manifest.layout.mode = "legacy";
	EXPECT_THROW(ValidateVNextPackageManifest(manifest), std::runtime_error);

	manifest = BuildVNextPackageManifest(BuildExecutableModule(BuildLinearAddGraph()));
	manifest.runtimeSteps[0].id = 99;
	EXPECT_THROW(ValidateVNextPackageManifest(manifest), std::runtime_error);

	manifest = BuildVNextPackageManifest(BuildExecutableModule(BuildLinearAddGraph()));
	manifest.bufferBindings[0].memoryBuffer = manifest.memory.buffers.size();
	EXPECT_THROW(ValidateVNextPackageManifest(manifest), std::runtime_error);
}

TEST(G14VNext, MemoryPlanRejectsHiddenMemorySpaceCopies)
{
	auto plan = BuildExecutablePlan(BuildLinearAddGraph());
	auto& subgraph = plan.subgraphs[plan.forward];
	subgraph.nodes[2].outputs[0].memorySpace = TensorMemorySpace::Device;
	plan.outputs[0].type.memorySpace = TensorMemorySpace::Device;
	const auto memory = BuildMemoryPlan(plan);
	EXPECT_THROW(ValidateMemoryPlan(plan, memory), std::runtime_error);
}

TEST(G14VNext, RuntimeStateABICoversLLMDiffusionAndTraining)
{
	Runtime::LLMDecodeStateABI llm;
	llm.kvCaches.push_back(Runtime::MakeKVCacheState(
	    "kv.layer0", TensorType::Dense(DataType::Float16, ShapeView{ 1, 8, 16, 64 })));
	llm.currentPosition = Runtime::MakeRuntimeStateBinding(
	    "position", Runtime::RuntimeStateKind::KVCache, "current-position",
	    TensorType::Dense(DataType::Int64, ShapeView{ 1 }), BufferMutability::Mutable, { "read", "increment" });

	Runtime::DiffusionExecutionABI diffusion{
		.latent = Runtime::MakeDiffusionState(
		    "latent", "latent-state", TensorType::Dense(DataType::Float16, ShapeView{ 1, 4, 128, 128 })),
		.timestepSchedule = Runtime::MakeDiffusionState(
		    "timesteps", "timestep-schedule", TensorType::Dense(DataType::Float32, ShapeView{ 32 }),
		    BufferMutability::Immutable)
	};

	Runtime::TrainingExecutionABI training;
	training.savedActivations.push_back(Runtime::MakeTrainingState(
	    "act.0", "saved-activation", TensorType::Dense(DataType::Float32, ShapeView{ 2, 4 })));
	training.optimizerStates.push_back(Runtime::MakeTrainingState(
	    "adam.m.0", "optimizer-state", TensorType::Dense(DataType::Float32, ShapeView{ 2, 4 })));
	training.recomputationStrategy = "none";

	EXPECT_EQ(llm.kvCaches[0].role, "kv-cache");
	ASSERT_TRUE(llm.currentPosition.has_value());
	EXPECT_EQ(diffusion.latent.kind, Runtime::RuntimeStateKind::Diffusion);
	ASSERT_TRUE(diffusion.timestepSchedule.has_value());
	EXPECT_EQ(training.optimizerStates[0].role, "optimizer-state");
}
