#include <gtest/gtest.h>

#include <LiteNN.h>
#include <LiteNNImporters.h>

#include <algorithm>
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

	Graph BuildLinearAddGraphWithLoRAWeights()
	{
		auto graph = BuildLinearAddGraph();
		const auto a = graph.AddVariable(
		    Variable::CreateFrozen(Tensor<CPU>({ 0.5F, -0.25F }, { 2, 1 }, DataType::Float32)));
		const auto b = graph.AddVariable(
		    Variable::CreateFrozen(Tensor<CPU>({ 1.5F, 2.0F }, { 1, 2 }, DataType::Float32)));
		graph.SetVariableName(a, "linear.lora_A.default.weight");
		graph.SetVariableName(b, "linear.lora_B.default.weight");
		return graph;
	}
} // namespace

TEST(G14VNext, BuildsManifestWithTensorArtifactAndCoverageTables)
{
	const auto graph = BuildLinearAddGraph();
	auto module = Detail::BuildExecutableModuleFromGraph(graph);
	VNextArtifactRef artifact;
	artifact.name = "cpu_forward";
	artifact.backend = std::string(BackendCPUAOT);
	artifact.entries.push_back({ .name = "forward",
		                         .kind = VNextArtifactEntryKind::Forward,
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
	EXPECT_NO_THROW(ValidateVNextABIFamily(manifest));

	const auto abi = DescribeVNextABIFamily(manifest);
	EXPECT_EQ(abi.versions.artifactABI, 1u);
	EXPECT_TRUE(abi.hasRuntimeSchedule);
	EXPECT_TRUE(abi.hasExternalTensorBindings);
	EXPECT_TRUE(abi.hasArtifactMetadata);
	EXPECT_TRUE(std::ranges::contains(abi.functions, std::string("forward")));
	EXPECT_TRUE(std::ranges::contains(abi.bufferBindings, std::string("linear.bias")));
	EXPECT_TRUE(std::ranges::contains(abi.tensorBindings, std::string("linear.bias")));
	EXPECT_TRUE(std::ranges::contains(abi.artifactEntries, std::string("cpu_forward:forward")));
	EXPECT_TRUE(std::ranges::contains(abi.artifactEntryKinds, std::string("cpu_forward:forward:forward")));
	EXPECT_TRUE(std::ranges::contains(abi.artifactRegions, std::string("cpu_forward:instructions")));
}

TEST(G14VNext, VNextModelPackageRoundTripsManifestAndExecutablePlan)
{
	const auto graph = BuildLinearAddGraph();
	auto module = Detail::BuildExecutableModuleFromGraph(graph);
	auto kvCache = Runtime::MakeKVCacheState("kv.cache.0",
	                                         TensorType::Dense(DataType::Float16, ShapeView{ 1, 2, 4, 8 }));
	VNextArtifactRef artifact;
	artifact.name = "cpu_forward";
	artifact.backend = std::string(BackendCPUAOT);
	artifact.entries.push_back({ .name = "forward",
		                         .kind = VNextArtifactEntryKind::Forward,
		                         .function = 0,
		                         .requiredStateBindings = { "kv.cache.0" },
		                         .requiredBufferBindings = { "linear.bias" } });
	artifact.regions.push_back({ .name = "instructions",
	                             .kind = ExternalBufferKind::ObjectFile,
	                             .relativePath = "artifacts/cpu_forward.o",
	                             .byteOffset = 0,
	                             .byteSize = 128,
	                             .checksum = 123 });

	const auto path = std::filesystem::temp_directory_path() / "litenn_vnext_package_roundtrip.json";
	Serialization::SaveVNextModelPackage(module, path, { artifact }, {}, {},
	                                     std::vector<Runtime::RuntimeStateBinding>{ kvCache });
	{
		std::ifstream input(path, std::ios::binary);
		const std::string json((std::istreambuf_iterator<char>(input)), std::istreambuf_iterator<char>());
		EXPECT_NE(json.find("\"op\":"), std::string::npos);
		EXPECT_EQ(json.find("\"opKind\""), std::string::npos);
	EXPECT_NE(json.find("\"runtimeStates\""), std::string::npos);
	EXPECT_NE(json.find("\"kv.cache.0\""), std::string::npos);
	EXPECT_NE(json.find("\"kind\":"), std::string::npos);
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
	ASSERT_EQ(package.manifest.runtimeStates.size(), 1u);
	EXPECT_EQ(package.manifest.runtimeStates[0].name, "kv.cache.0");
	EXPECT_EQ(package.manifest.runtimeStates[0].kind, Runtime::RuntimeStateKind::KVCache);
	EXPECT_EQ(package.manifest.runtimeStates[0].role, "kv-cache");
	EXPECT_TRUE(package.manifest.runtimeStates[0].memoryBuffer.has_value());
	EXPECT_TRUE(std::ranges::contains(package.manifest.runtimeStates[0].effects, std::string("append")));
	ASSERT_EQ(package.manifest.artifacts.size(), 1u);
	EXPECT_EQ(package.manifest.artifacts[0].backend, BackendCPUAOT);
	ASSERT_EQ(package.manifest.artifacts[0].entries.size(), 1u);
	EXPECT_EQ(package.manifest.artifacts[0].entries[0].kind, VNextArtifactEntryKind::Forward);
	ASSERT_EQ(package.manifest.artifacts[0].entries[0].requiredStateBindings.size(), 1u);
	EXPECT_EQ(package.manifest.artifacts[0].entries[0].requiredStateBindings[0], "kv.cache.0");
	EXPECT_EQ(package.manifest.artifacts[0].entries[0].requiredBufferBindings[0], "linear.bias");
	ASSERT_EQ(package.plan.subgraphs.size(), module.plan.subgraphs.size());
	EXPECT_EQ(package.plan.subgraphs[package.plan.forward].nodes[2].op.kind, "BinaryOpNode");
	EXPECT_FALSE(package.plan.subgraphs[package.plan.forward].nodes[2].op.attributes.empty());
	EXPECT_EQ(package.plan.outputs[0].name, "y");
}

TEST(G14VNext, VNextModelPackageRoundTripsRuntimeScheduleFallbackRecords)
{
	const auto graph = BuildLinearAddGraph();
	auto schedule = Runtime::BuildRuntimeSchedule(Detail::BuildExecutableModuleFromGraph(graph));
	ASSERT_FALSE(schedule.memory.buffers.empty());
	Runtime::RuntimeScheduleStep fallback;
	fallback.id = schedule.steps.size();
	fallback.kind = Runtime::RuntimeScheduleStepKind::Fallback;
	fallback.backend = std::string(BackendCUDANative);
	fallback.fallbackBackend = std::string(BackendCPUInterpreter);
	fallback.inputBuffers.push_back(0);
	fallback.outputBuffers.push_back(0);
	schedule.steps.push_back(std::move(fallback));
	ASSERT_NO_THROW(Runtime::ValidateRuntimeSchedule(schedule));
	const auto profileRecords = Runtime::BuildRuntimeScheduleProfileRecords(schedule);
	ASSERT_EQ(profileRecords.size(), schedule.steps.size());
	EXPECT_EQ(profileRecords.back().fallbackBackend, BackendCPUInterpreter);

	const auto path = std::filesystem::temp_directory_path() / "litenn_vnext_schedule_fallback_roundtrip.json";
	Serialization::SaveVNextModelPackage(schedule, path);
	{
		std::ifstream input(path, std::ios::binary);
		const std::string json((std::istreambuf_iterator<char>(input)), std::istreambuf_iterator<char>());
		EXPECT_NE(json.find("\"fallbackBackend\""), std::string::npos);
		EXPECT_NE(json.find(BackendCPUInterpreter), std::string::npos);
	}
	const auto package = Serialization::LoadVNextModelPackage(path);
	std::filesystem::remove(path);

	ASSERT_FALSE(package.manifest.runtimeSteps.empty());
	const auto& loadedFallback = package.manifest.runtimeSteps.back();
	EXPECT_EQ(loadedFallback.kind, Runtime::RuntimeScheduleStepKind::Fallback);
	EXPECT_EQ(loadedFallback.backend, BackendCUDANative);
	EXPECT_EQ(loadedFallback.fallbackBackend, BackendCPUInterpreter);
	const auto abi = DescribeVNextABIFamily(package.manifest);
	EXPECT_TRUE(abi.hasFallbackRecords);
	EXPECT_TRUE(abi.hasProfileRecords);
	EXPECT_TRUE(std::ranges::contains(abi.runtimeStepRecords,
	                                  std::format("{}:fallback:{}->{}",
	                                              loadedFallback.id, BackendCUDANative, BackendCPUInterpreter)));
	EXPECT_NO_THROW(ValidateVNextPackageManifest(package.manifest));
}

TEST(G14VNext, VNextModelPackageRoundTripsLoRAAdapterManifest)
{
	const auto graph = BuildLinearAddGraphWithLoRAWeights();
	auto module = Detail::BuildExecutableModuleFromGraph(graph);
	std::vector<VNextAdapterRef> adapters;
	adapters.push_back({ .targetName = "linear",
	                     .adapterName = "default",
	                     .aTensor = 1,
	                     .bTensor = 2,
	                     .rank = 1,
	                     .alpha = 2.0F,
	                     .dtype = DataType::Float32 });

	const auto path = std::filesystem::temp_directory_path() / "litenn_vnext_lora_adapter_roundtrip.json";
	Serialization::SaveVNextModelPackage(module, path, {}, {}, std::move(adapters));
	{
		std::ifstream input(path, std::ios::binary);
		const std::string json((std::istreambuf_iterator<char>(input)), std::istreambuf_iterator<char>());
		EXPECT_NE(json.find("\"adapters\""), std::string::npos);
		EXPECT_NE(json.find("\"linear-lora\""), std::string::npos);
	}
	const auto package = Serialization::LoadVNextModelPackage(path);
	std::filesystem::remove(path);

	EXPECT_NO_THROW(ValidateVNextPackageManifest(package.manifest));
	ASSERT_EQ(package.manifest.adapters.size(), 1u);
	const auto& adapter = package.manifest.adapters[0];
	EXPECT_EQ(adapter.targetName, "linear");
	EXPECT_EQ(adapter.adapterName, "default");
	EXPECT_EQ(adapter.kind, "linear-lora");
	EXPECT_EQ(adapter.aTensor, 1u);
	EXPECT_EQ(adapter.bTensor, 2u);
	EXPECT_EQ(adapter.rank, 1u);
	EXPECT_FLOAT_EQ(adapter.alpha, 2.0F);
	EXPECT_EQ(adapter.dtype, DataType::Float32);
	ASSERT_EQ(package.manifest.tensors.size(), 3u);
	EXPECT_EQ(package.manifest.tensors[adapter.aTensor].name, "linear.lora_A.default.weight");
	EXPECT_EQ(package.manifest.tensors[adapter.bTensor].name, "linear.lora_B.default.weight");
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
	const auto* outputValues = static_cast<const float*>(outputs[0].UnsafeRawData());
	EXPECT_FLOAT_EQ(outputValues[0], 11.0F);
	EXPECT_FLOAT_EQ(outputValues[1], 22.0F);
	EXPECT_FLOAT_EQ(outputValues[2], 33.0F);
	EXPECT_FLOAT_EQ(outputValues[3], 44.0F);

	std::filesystem::remove(path);
	std::filesystem::remove(weightsPath);
}

TEST(G14VNext, VNextModelPackageRoundTripsPackedNibbleQuantizationMetadata)
{
	Graph graph;
	auto packed = PackInteger4(Tensor<CPU>({ 1.0, 15.0, 2.0 }, { 3 }, DataType::UInt8),
	                           PackedNibbleQuantization(PackedNibbleFormat::UInt4, { 3 }));
	auto params = PackedNibbleQuantization(PackedNibbleFormat::UInt4, { 3 }, 0.25F, 8,
	                                       PackedNibbleOrder::HighThenLow);
	const auto packedVariable = graph.AddVariable(Variable::CreateFrozenQuantized(std::move(packed), params));
	graph.SetVariableName(packedVariable, "linear.weight.uint4");
	Subgraph subgraph;
	const auto ref = subgraph.AddNode(VariableRefNode{ packedVariable }, { OutputInfo{ DataType::UInt8, { 2 } } });
	subgraph.SetResults({ { ref, 0 } });
	graph.SetForward(graph.AddSubgraph(std::move(subgraph)));
	graph.SetOutputNames({ "packed" });

	const auto root = std::filesystem::temp_directory_path();
	const auto path = root / "litenn_vnext_packed_nibble_quantization.json";
	const auto weightsPath = root / "litenn_vnext_packed_nibble_quantization.bin";

	Serialization::ExternalWeightSaveOptions options;
	options.minVariableBytes = 0;
	Serialization::SaveVNextModelPackageExternalWeights(graph, path, weightsPath, options);

	const auto package = Serialization::LoadVNextModelPackage(path);
	const auto found = std::ranges::find_if(package.manifest.tensors, [](const auto& tensor) {
		return tensor.relativePath == "litenn_vnext_packed_nibble_quantization.bin";
	});
	ASSERT_NE(found, package.manifest.tensors.end());
	ASSERT_TRUE(found->quantization.has_value());
	const auto& loaded = *found->quantization;
	EXPECT_EQ(loaded.scheme, QuantizationScheme::Block);
	EXPECT_EQ(loaded.blockFormat, QuantizedBlockFormat::PackedNibble);
	EXPECT_EQ(loaded.packedFormat, PackedNibbleFormat::UInt4);
	EXPECT_EQ(loaded.packedOrder, PackedNibbleOrder::HighThenLow);
	EXPECT_EQ(loaded.blockScaleLayout, BlockScaleLayout::None);
	EXPECT_EQ(loaded.storageType, DataType::UInt8);
	EXPECT_EQ(loaded.expressedShape, (std::vector<std::size_t>{ 3 }));
	ASSERT_EQ(loaded.scales.size(), 1u);
	EXPECT_FLOAT_EQ(loaded.scales[0], 0.25F);
	ASSERT_EQ(loaded.zeroPoints.size(), 1u);
	EXPECT_EQ(loaded.zeroPoints[0], 8);

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

	const auto schedule = Runtime::BuildRuntimeSchedule(Detail::BuildExecutableModuleFromGraph(graph), { kvCache });

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
	const auto plan = Detail::BuildExecutablePlanFromGraph(graph);
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
	auto manifest = BuildVNextPackageManifest(Detail::BuildExecutableModuleFromGraph(BuildLinearAddGraph()));
	manifest.versions.artifactABI = 0;
	EXPECT_THROW(ValidateVNextPackageManifest(manifest), std::runtime_error);

	manifest = BuildVNextPackageManifest(Detail::BuildExecutableModuleFromGraph(BuildLinearAddGraph()));
	manifest.artifacts.push_back({ .name = "broken", .backend = "", .entries = { { .name = "forward" } } });
	EXPECT_THROW(ValidateVNextPackageManifest(manifest), std::runtime_error);

	manifest = BuildVNextPackageManifest(Detail::BuildExecutableModuleFromGraph(BuildLinearAddGraph()));
	manifest.artifacts.push_back({ .name = "broken-kind",
		                           .backend = std::string(BackendCPUAOT),
		                           .entries = { { .name = "forward",
		                                          .kind = static_cast<VNextArtifactEntryKind>(99),
		                                          .function = 0 } },
		                           .regions = { { .name = "instructions",
		                                          .relativePath = "artifacts/broken.o",
		                                          .byteSize = 1 } } });
	EXPECT_THROW(ValidateVNextPackageManifest(manifest), std::runtime_error);

	manifest = BuildVNextPackageManifest(Detail::BuildExecutableModuleFromGraph(BuildLinearAddGraph()));
	manifest.artifacts.push_back({ .name = "broken", .backend = "", .entries = { { .name = "forward" } } });
	manifest.artifacts[0].backend = std::string(BackendCPUAOT);
	manifest.artifacts[0].entries[0].function = 99;
	EXPECT_THROW(ValidateVNextPackageManifest(manifest), std::runtime_error);

	manifest = BuildVNextPackageManifest(Detail::BuildExecutableModuleFromGraph(BuildLinearAddGraph()));
	manifest.artifacts.push_back({ .name = "missing-binding",
		                           .backend = std::string(BackendCPUAOT),
		                           .entries = { { .name = "forward",
		                                          .function = 0,
		                                          .requiredBufferBindings = { "missing.weight" } } },
		                           .regions = { { .name = "instructions",
		                                          .relativePath = "artifacts/missing.o",
		                                          .byteSize = 1 } } });
	EXPECT_THROW(ValidateVNextPackageManifest(manifest), std::runtime_error);

	manifest = BuildVNextPackageManifest(Detail::BuildExecutableModuleFromGraph(BuildLinearAddGraph()));
	manifest.layout.mode = "legacy";
	EXPECT_THROW(ValidateVNextPackageManifest(manifest), std::runtime_error);

	manifest = BuildVNextPackageManifest(Detail::BuildExecutableModuleFromGraph(BuildLinearAddGraph()));
	manifest.runtimeSteps[0].id = 99;
	EXPECT_THROW(ValidateVNextPackageManifest(manifest), std::runtime_error);

	manifest = BuildVNextPackageManifest(
	    Detail::BuildExecutableModuleFromGraph(BuildLinearAddGraph()), {}, {}, {},
	    { Runtime::MakeKVCacheState("kv.cache.0",
	                                TensorType::Dense(DataType::Float16, ShapeView{ 1, 2, 4, 8 })) });
	manifest.runtimeStates[0].memoryBuffer = manifest.memory.buffers.size();
	EXPECT_THROW(ValidateVNextPackageManifest(manifest), std::runtime_error);

	manifest = BuildVNextPackageManifest(Detail::BuildExecutableModuleFromGraph(BuildLinearAddGraph()));
	manifest.bufferBindings[0].memoryBuffer = manifest.memory.buffers.size();
	EXPECT_THROW(ValidateVNextPackageManifest(manifest), std::runtime_error);
}

TEST(G14VNext, VNextABIVersionBumpRulesCoverProductionContracts)
{
	const auto tensorBinding = VNextABIVersionBumpRuleFor(VNextABIChangeArea::TensorBinding);
	const auto externalRegion = VNextABIVersionBumpRuleFor(VNextABIChangeArea::ExternalRegion);
	const auto backendRequirement = VNextABIVersionBumpRuleFor(VNextABIChangeArea::BackendRequirement);
	const auto runtimeState = VNextABIVersionBumpRuleFor(VNextABIChangeArea::RuntimeState);
	const auto runtimeSchedule = VNextABIVersionBumpRuleFor(VNextABIChangeArea::RuntimeSchedule);
	const auto artifactEntry = VNextABIVersionBumpRuleFor(VNextABIChangeArea::ArtifactEntry);

	EXPECT_EQ(tensorBinding.component, VNextVersionComponent::ArtifactABI);
	EXPECT_EQ(externalRegion.component, VNextVersionComponent::ArtifactABI);
	EXPECT_EQ(backendRequirement.component, VNextVersionComponent::ArtifactABI);
	EXPECT_EQ(runtimeState.component, VNextVersionComponent::ArtifactABI);
	EXPECT_EQ(runtimeSchedule.component, VNextVersionComponent::ArtifactABI);
	EXPECT_EQ(artifactEntry.component, VNextVersionComponent::ArtifactABI);
	EXPECT_EQ(VNextVersionComponentName(tensorBinding.component), "ArtifactABI");
	EXPECT_EQ(VNextABIChangeAreaName(VNextABIChangeArea::TensorBinding), "TensorBinding");
	EXPECT_NE(runtimeState.reason.find("runtime-state"), std::string_view::npos);

	const auto rules = DescribeVNextABIVersionBumpRules();
	EXPECT_GE(rules.size(), 11u);
	const auto covers = [&](VNextABIChangeArea area) {
		return std::ranges::any_of(rules, [&](const VNextABIVersionBumpRule& rule) {
			return rule.area == area;
		});
	};
	EXPECT_TRUE(covers(VNextABIChangeArea::TensorBinding));
	EXPECT_TRUE(covers(VNextABIChangeArea::ExternalRegion));
	EXPECT_TRUE(covers(VNextABIChangeArea::BackendRequirement));
	EXPECT_TRUE(covers(VNextABIChangeArea::RuntimeState));
	EXPECT_TRUE(covers(VNextABIChangeArea::RuntimeSchedule));
	EXPECT_TRUE(covers(VNextABIChangeArea::ArtifactEntry));
}

TEST(G14VNext, MemoryPlanRejectsHiddenMemorySpaceCopies)
{
	auto plan = Detail::BuildExecutablePlanFromGraph(BuildLinearAddGraph());
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

	Runtime::LoRAAdapterExecutionABI lora;
	lora.adapterWeights.push_back(Runtime::MakeLoRAAdapterState(
	    "lora.linear.default.A", "adapter-weight-a",
	    TensorType::Dense(DataType::Float16, ShapeView{ 8, 4 }), BufferMutability::Mutable));
	lora.mergeState = Runtime::MakeLoRAAdapterState(
	    "lora.linear.default.merge", "adapter-merge-state",
	    TensorType::Dense(DataType::Int32, ShapeView{ 1 }), BufferMutability::Mutable);

	EXPECT_EQ(llm.kvCaches[0].role, "kv-cache");
	ASSERT_TRUE(llm.currentPosition.has_value());
	EXPECT_EQ(diffusion.latent.kind, Runtime::RuntimeStateKind::Diffusion);
	ASSERT_TRUE(diffusion.timestepSchedule.has_value());
	EXPECT_EQ(training.optimizerStates[0].role, "optimizer-state");
	ASSERT_FALSE(lora.adapterWeights.empty());
	EXPECT_EQ(lora.adapterWeights[0].kind, Runtime::RuntimeStateKind::LoRAAdapter);
	ASSERT_TRUE(lora.mergeState.has_value());
	EXPECT_EQ(lora.mergeState->role, "adapter-merge-state");
}
