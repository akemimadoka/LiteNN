#include <gtest/gtest.h>

#include <LiteNN.h>
#include <LiteNNImporters.h>

#include <algorithm>
#include <array>
#include <cstdint>
#include <filesystem>
#include <fstream>
#include <iterator>
#include <span>

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
		const auto a =
		    graph.AddVariable(Variable::CreateFrozen(Tensor<CPU>({ 0.5F, -0.25F }, { 2, 1 }, DataType::Float32)));
		const auto b =
		    graph.AddVariable(Variable::CreateFrozen(Tensor<CPU>({ 1.5F, 2.0F }, { 1, 2 }, DataType::Float32)));
		graph.SetVariableName(a, "linear.lora_A.default.weight");
		graph.SetVariableName(b, "linear.lora_B.default.weight");
		return graph;
	}

	std::uint64_t ChecksumBytesForTest(std::span<const std::byte> bytes)
	{
		std::uint64_t hash = 1469598103934665603ull;
		for (const auto byte : bytes)
		{
			hash ^= std::to_integer<std::uint8_t>(byte);
			hash *= 1099511628211ull;
		}
		return hash;
	}

	void WriteBytesForTest(const std::filesystem::path& path, std::span<const std::byte> bytes)
	{
		std::filesystem::create_directories(path.parent_path());
		std::ofstream out(path, std::ios::binary);
		ASSERT_TRUE(out) << path.string();
		out.write(reinterpret_cast<const char*>(bytes.data()), static_cast<std::streamsize>(bytes.size()));
		ASSERT_TRUE(out) << path.string();
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
	EXPECT_TRUE(std::ranges::contains(
	    abi.artifactEntryKinds,
	    std::format("cpu_forward:forward:{}", VNextArtifactEntryKindName(VNextArtifactEntryKind::Forward))));
	EXPECT_TRUE(std::ranges::contains(abi.artifactRegions, std::string("cpu_forward:instructions")));
}

TEST(G14VNext, VNextModelPackageRoundTripsManifestAndExecutablePlan)
{
	const auto graph = BuildLinearAddGraph();
	auto module = Detail::BuildExecutableModuleFromGraph(graph);
	auto kvCache =
	    Runtime::MakeKVCacheState("kv.cache.0", TensorType::Dense(DataType::Float16, ShapeView{ 1, 2, 4, 8 }));
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

TEST(G14VNext, VNextModelPackageLoadsSeparatedArtifactRegions)
{
	const auto graph = BuildLinearAddGraph();
	auto module = Detail::BuildExecutableModuleFromGraph(graph);
	const auto base = std::filesystem::temp_directory_path() / "litenn_vnext_region_package";
	std::filesystem::remove_all(base);
	std::filesystem::create_directories(base / "artifacts");

	const std::vector<std::byte> rodata{ std::byte{ 0x4c }, std::byte{ 0x54 }, std::byte{ 0x4e }, std::byte{ 0x4e } };
	const std::vector<std::byte> instructions{ std::byte{ 0x01 }, std::byte{ 0x02 }, std::byte{ 0x03 },
		                                       std::byte{ 0x04 }, std::byte{ 0x05 } };
	WriteBytesForTest(base / "artifacts" / "cpu_forward.rodata.bin", rodata);
	WriteBytesForTest(base / "artifacts" / "cpu_forward.instructions.bin", instructions);

	VNextArtifactRef artifact;
	artifact.name = "cpu_forward";
	artifact.backend = std::string(BackendCPUAOT);
	artifact.entries.push_back({ .name = "forward", .kind = VNextArtifactEntryKind::Forward, .function = 0 });
	artifact.regions.push_back({ .name = "rodata",
	                             .kind = ExternalBufferKind::Rodata,
	                             .relativePath = "artifacts/cpu_forward.rodata.bin",
	                             .byteSize = rodata.size(),
	                             .checksum = ChecksumBytesForTest(rodata) });
	artifact.regions.push_back({ .name = "instructions",
	                             .kind = ExternalBufferKind::ObjectFile,
	                             .relativePath = "artifacts/cpu_forward.instructions.bin",
	                             .byteSize = instructions.size(),
	                             .checksum = ChecksumBytesForTest(instructions) });

	const auto packagePath = base / "model.ltnn.json";
	Serialization::SaveVNextModelPackage(module, packagePath, { artifact });
	const auto package = Serialization::LoadVNextModelPackage(packagePath);
	EXPECT_EQ(package.sourcePath, packagePath);

	const auto loaded = Serialization::LoadVNextArtifactRegions(package, "cpu_forward");
	ASSERT_EQ(loaded.regions.size(), 2u);
	ASSERT_NE(loaded.FindRegion("rodata"), nullptr);
	ASSERT_NE(loaded.FindRegion("instructions"), nullptr);
	EXPECT_EQ(loaded.FindRegion("rodata")->bytes, rodata);
	EXPECT_EQ(loaded.FindRegion("instructions")->bytes, instructions);

	auto memoryPackage = package;
	memoryPackage.sourcePath.clear();
	EXPECT_THROW((void) Serialization::LoadVNextArtifactRegions(memoryPackage, "cpu_forward"), std::runtime_error);
	EXPECT_NO_THROW((void) Serialization::LoadVNextArtifactRegions(memoryPackage, base, "cpu_forward"));

	WriteBytesForTest(base / "artifacts" / "cpu_forward.instructions.bin",
	                  std::span<const std::byte>{ rodata.data(), rodata.size() });
	EXPECT_THROW((void) Serialization::LoadVNextArtifactRegions(package, "cpu_forward"), std::runtime_error);

	std::filesystem::remove_all(base);
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
	                                  std::format("{}:{}:{}->{}", loadedFallback.id,
	                                              Runtime::RuntimeScheduleStepKindName(loadedFallback.kind),
	                                              BackendCUDANative, BackendCPUInterpreter)));
	EXPECT_NO_THROW(ValidateVNextPackageManifest(package.manifest));
}

TEST(G14VNext, VNextModelPackageRoundTripsRuntimeScheduleSegments)
{
	Graph graph;
	Subgraph subgraph;
	const auto input = subgraph.AddParam(DataType::Float32, { 2 });
	const auto cast = subgraph.AddNode(CastNode{ { input, 0 } }, { OutputInfo{ DataType::Float32, { 2 } } });
	const auto negated =
	    subgraph.AddNode(UnaryOpNode{ UnaryOp::Negate, { cast, 0 } }, { OutputInfo{ DataType::Float32, { 2 } } });
	subgraph.SetResults({ { negated, 0 } });
	graph.SetForward(graph.AddSubgraph(std::move(subgraph)));

	auto registry = BuildDefaultOpSchemaRegistry();
	registry.RegisterCapability("CastNode", {
	                                            .backend = std::string(BackendCUDANative),
	                                            .support = BackendSupportLevel::Native,
	                                            .layouts = { TensorLayoutKind::RowMajor },
	                                            .memorySpaces = { TensorMemorySpace::Host },
	                                            .relativeCost = 0.01,
	                                        });
	constexpr std::array<std::string_view, 2> backends{ BackendCPUInterpreter, BackendCUDANative };
	Runtime::PlacementOptions options;
	options.defaultBackend = std::string(BackendCPUInterpreter);
	options.valueConstraints.push_back({ .subgraph = graph.Forward(),
	                                     .value = { cast, 0 },
	                                     .backend = std::string(BackendCUDANative),
	                                     .reason = "package a heterogeneous schedule segment" });
	const auto plan = Detail::BuildExecutablePlanFromGraph(graph);
	const auto placement = Runtime::BuildPlacementPlan(plan, backends, registry, options);
	auto schedule = Runtime::BuildRuntimeSchedule(BuildExecutableModule(plan));
	Runtime::AppendPlacementSegmentSteps(schedule, placement);
	Runtime::AppendPlacementTransferSteps(schedule, placement);
	Runtime::AppendPlacementSyncSteps(schedule, placement);
	ASSERT_FALSE(schedule.segments.empty());
	ASSERT_EQ(schedule.segments.size(), 3u);
	EXPECT_NO_THROW(Runtime::ValidateRuntimeSchedule(schedule));

	VNextArtifactRef artifact;
	artifact.name = "heterogeneous_forward";
	artifact.backend = "heterogeneous";
	artifact.entries.push_back({ .name = "forward", .kind = VNextArtifactEntryKind::Forward, .function = 0 });
	artifact.regions.push_back({ .name = "instructions",
	                             .kind = ExternalBufferKind::ObjectFile,
	                             .relativePath = "artifacts/heterogeneous_forward.bin",
	                             .byteSize = 1,
	                             .checksum = 7 });
	artifact.backendRequirements = BuildVNextBackendRequirementsFromSchedule(schedule);
	ASSERT_EQ(artifact.backendRequirements.size(), schedule.segments.size());

	const auto path = std::filesystem::temp_directory_path() / "litenn_vnext_schedule_segments_roundtrip.json";
	Serialization::SaveVNextModelPackage(schedule, path, { artifact });
	{
		std::ifstream inputFile(path, std::ios::binary);
		const std::string json((std::istreambuf_iterator<char>(inputFile)), std::istreambuf_iterator<char>());
		EXPECT_NE(json.find("\"runtimeSegments\""), std::string::npos);
		EXPECT_NE(json.find("\"segment\""), std::string::npos);
		EXPECT_NE(json.find("\"backendRequirements\""), std::string::npos);
		EXPECT_NE(json.find("\"runtime-buffer-transfer-v1\""), std::string::npos);
		EXPECT_NE(json.find("\"streamOwner\""), std::string::npos);
		EXPECT_NE(json.find("\"cuda-default-stream\""), std::string::npos);
		EXPECT_NE(json.find("\"eventOwner\""), std::string::npos);
		EXPECT_NE(json.find("\"cuda-runtime-event\""), std::string::npos);
		EXPECT_NE(json.find(BackendCUDANative), std::string::npos);
	}
	const auto package = Serialization::LoadVNextModelPackage(path);
	std::filesystem::remove(path);

	ASSERT_EQ(package.manifest.runtimeSegments.size(), 3u);
	const auto& cudaSegment = package.manifest.runtimeSegments[1];
	EXPECT_EQ(cudaSegment.backend, BackendCUDANative);
	EXPECT_EQ(cudaSegment.nodes, (std::vector<NodeId>{ cast }));
	EXPECT_FALSE(cudaSegment.inputBuffers.empty());
	EXPECT_FALSE(cudaSegment.outputBuffers.empty());
	ASSERT_FALSE(package.manifest.runtimeSteps.empty());
	const auto segmentStepIt = std::ranges::find_if(package.manifest.runtimeSteps, [](const auto& step) {
		return step.kind == Runtime::RuntimeScheduleStepKind::DispatchSegment;
	});
	ASSERT_NE(segmentStepIt, package.manifest.runtimeSteps.end());
	ASSERT_TRUE(segmentStepIt->segment.has_value());
	EXPECT_LT(*segmentStepIt->segment, package.manifest.runtimeSegments.size());
	const auto syncStepIt = std::ranges::find_if(package.manifest.runtimeSteps, [](const auto& step) {
		return step.kind == Runtime::RuntimeScheduleStepKind::Sync;
	});
	ASSERT_NE(syncStepIt, package.manifest.runtimeSteps.end());
	EXPECT_EQ(syncStepIt->streamOwner, "cuda-default-stream");
	EXPECT_EQ(syncStepIt->eventOwner, "cuda-runtime-event");
	EXPECT_EQ(syncStepIt->syncScope, "transfer-boundary");
	EXPECT_NO_THROW(ValidateVNextPackageManifest(package.manifest));
	ASSERT_EQ(package.manifest.artifacts.size(), 1u);
	ASSERT_EQ(package.manifest.artifacts[0].backendRequirements.size(), 3u);
	EXPECT_EQ(package.manifest.artifacts[0].backendRequirements[1].backend, BackendCUDANative);
	ASSERT_TRUE(package.manifest.artifacts[0].backendRequirements[1].segment.has_value());
	EXPECT_EQ(*package.manifest.artifacts[0].backendRequirements[1].segment, cudaSegment.id);
	EXPECT_EQ(package.manifest.artifacts[0].backendRequirements[1].transferABI, "runtime-buffer-transfer-v1");
	EXPECT_TRUE(std::ranges::contains(package.manifest.artifacts[0].backendRequirements[1].requiredCapabilities,
	                                  std::string("backend:") + std::string(BackendCUDANative)));
	const auto abi = DescribeVNextABIFamily(package.manifest);
	EXPECT_TRUE(abi.hasRuntimeSegments);
	EXPECT_TRUE(abi.hasBackendRequirements);
	EXPECT_FALSE(abi.runtimeSegments.empty());
	EXPECT_FALSE(abi.backendRequirements.empty());

	auto invalid = package.manifest;
	invalid.artifacts[0].backendRequirements[1].segment = invalid.runtimeSegments.size();
	EXPECT_THROW(ValidateVNextPackageManifest(invalid), std::runtime_error);
}

TEST(G14VNext, VNextArtifactBackendRequirementsRejectUnavailableCapabilities)
{
	Graph graph;
	Subgraph subgraph;
	const auto input = subgraph.AddParam(DataType::Float32, { 2 });
	const auto output =
	    subgraph.AddNode(UnaryOpNode{ UnaryOp::Negate, { input, 0 } }, { OutputInfo{ DataType::Float32, { 2 } } });
	subgraph.SetResults({ { output, 0 } });
	graph.SetForward(graph.AddSubgraph(std::move(subgraph)));

	auto schedule = Runtime::BuildRuntimeSchedule(Detail::BuildExecutableModuleFromGraph(graph));
	ASSERT_FALSE(schedule.module.partitions.empty());
	const auto scheduleBackend = schedule.module.partitions[0].backend;

	VNextArtifactRef artifact;
	artifact.name = "cpu_forward";
	artifact.backend = scheduleBackend;
	artifact.entries.push_back({ .name = "forward", .kind = VNextArtifactEntryKind::Forward, .function = 0 });
	artifact.regions.push_back({ .name = "object",
	                             .kind = ExternalBufferKind::ObjectFile,
	                             .relativePath = "artifacts/cpu_forward.o",
	                             .byteSize = 16,
	                             .checksum = 11 });
	artifact.backendRequirements = BuildVNextBackendRequirementsFromSchedule(schedule);
	ASSERT_EQ(artifact.backendRequirements.size(), 1u);
	artifact.backendRequirements[0].requiredCapabilities.push_back("op:UnaryOpNode");
	artifact.backendRequirements[0].transferABI = "runtime-buffer-transfer-v1";

	auto manifest = BuildVNextPackageManifest(std::move(schedule), { artifact });
	const std::vector<VNextAvailableBackendRef> available{
		{ .backend = scheduleBackend,
		  .capabilities = { "runtime-schedule:dispatch-region", std::string("backend:") + scheduleBackend,
		                    "op:UnaryOpNode" },
		  .transferABIs = { "none", "runtime-buffer-transfer-v1" } },
	};
	EXPECT_NO_THROW(ValidateVNextArtifactBackendRequirements(manifest, available));

	EXPECT_THROW(ValidateVNextArtifactBackendRequirements(manifest, std::span<const VNextAvailableBackendRef>{}),
	             std::runtime_error);

	auto missingCapability = available;
	missingCapability[0].capabilities.pop_back();
	EXPECT_THROW(ValidateVNextArtifactBackendRequirements(manifest, missingCapability), std::runtime_error);

	auto missingTransferABI = available;
	missingTransferABI[0].transferABIs = { "none" };
	EXPECT_THROW(ValidateVNextArtifactBackendRequirements(manifest, missingTransferABI), std::runtime_error);

	manifest.artifacts[0].backendRequirements[0].allowsFallback = true;
	EXPECT_THROW(ValidateVNextArtifactBackendRequirements(manifest, available), std::runtime_error);
	EXPECT_THROW(ValidateVNextArtifactBackendRequirements(
	                 manifest, available, VNextBackendRequirementValidationOptions{ .allowArtifactFallback = true }),
	             std::runtime_error);

	auto fallbackAvailable = available;
	fallbackAvailable[0].allowFallback = true;
	EXPECT_NO_THROW(ValidateVNextArtifactBackendRequirements(
	    manifest, fallbackAvailable, VNextBackendRequirementValidationOptions{ .allowArtifactFallback = true }));
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
	auto params = PackedNibbleQuantization(PackedNibbleFormat::UInt4, { 3 }, 0.25F, 8, PackedNibbleOrder::HighThenLow);
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

TEST(G14VNext, VNextModelPackageRoundTripsDequantizeNodeParameters)
{
	auto params =
	    PackedNibbleQuantization(PackedNibbleFormat::Int4, { 2, 3 }, 0.25F, -2, PackedNibbleOrder::HighThenLow);
	Graph graph;
	Subgraph forward;
	const auto input = forward.AddParam(DataType::UInt8, { 3 });
	const auto output = forward.AddNode(DequantizeNode{ { input, 0 }, params, DataType::Float32 },
	                                    { OutputInfo{ DataType::Float32, { 2, 3 } } });
	forward.SetResults({ { output, 0 } });
	graph.SetForward(graph.AddSubgraph(std::move(forward)));
	graph.SetInputNames({ "packed" });
	graph.SetOutputNames({ "dequantized" });

	const auto path = std::filesystem::temp_directory_path() / "litenn_vnext_dequantize_node.json";
	Serialization::SaveVNextModelPackage(Detail::BuildExecutableModuleFromGraph(graph), path);
	const auto package = Serialization::LoadVNextModelPackage(path);
	std::filesystem::remove(path);

	const auto& node = package.plan.subgraphs[package.plan.forward].nodes[1];
	ASSERT_TRUE(std::holds_alternative<DequantizeNode>(node.node));
	const auto& loaded = std::get<DequantizeNode>(node.node);
	EXPECT_EQ(loaded.targetType, DataType::Float32);
	EXPECT_EQ(loaded.params.scheme, QuantizationScheme::Block);
	EXPECT_EQ(loaded.params.blockFormat, QuantizedBlockFormat::PackedNibble);
	EXPECT_EQ(loaded.params.packedFormat, PackedNibbleFormat::Int4);
	EXPECT_EQ(loaded.params.packedOrder, PackedNibbleOrder::HighThenLow);
	EXPECT_EQ(loaded.params.expressedShape, std::vector<std::size_t>({ 2, 3 }));
	ASSERT_EQ(loaded.params.scales.size(), 1u);
	EXPECT_FLOAT_EQ(loaded.params.scales[0], 0.25F);
	ASSERT_EQ(loaded.params.zeroPoints.size(), 1u);
	EXPECT_EQ(loaded.params.zeroPoints[0], -2);
	EXPECT_NO_THROW(ValidateExecutablePlan(package.plan));
}

TEST(G14VNext, VNextModelPackageRoundTripsConstantPayloadsForInterpreter)
{
	Graph graph;
	Subgraph forward;
	const auto input = forward.AddParam(DataType::Float32, { 2, 2 });
	auto constant =
	    Tensor<CPU>({ 10.0, 20.0, 30.0, 40.0 }, { 2, 2 }, DataType::Float32).CopyToDevice(PolymorphicDevice{ CPU{} });
	const auto constantNode =
	    forward.AddNode(ConstantNode{ std::move(constant) }, { OutputInfo{ DataType::Float32, { 2, 2 } } });
	const auto output = forward.AddNode(BinaryOpNode{ BinaryOp::Add, { input, 0 }, { constantNode, 0 } },
	                                    { OutputInfo{ DataType::Float32, { 2, 2 } } });
	forward.SetResults({ { output, 0 } });
	graph.SetForward(graph.AddSubgraph(std::move(forward)));
	graph.SetInputNames({ "x" });
	graph.SetOutputNames({ "y" });

	const auto path = std::filesystem::temp_directory_path() / "litenn_vnext_constant_payload.json";
	Serialization::SaveVNextModelPackage(Detail::BuildExecutableModuleFromGraph(graph), path);
	const auto package = Serialization::LoadVNextModelPackage(path);
	std::filesystem::remove(path);

	Runtime::Interpreter<CPU> interpreter;
	std::array<Tensor<CPU>, 1> inputs{ Tensor<CPU>({ 1.0, 2.0, 3.0, 4.0 }, { 2, 2 }, DataType::Float32) };
	const auto outputs = interpreter.RunForward(package.plan, inputs);
	ASSERT_EQ(outputs.size(), 1u);
	const auto* values = static_cast<const float*>(outputs[0].UnsafeRawData());
	EXPECT_FLOAT_EQ(values[0], 11.0F);
	EXPECT_FLOAT_EQ(values[1], 22.0F);
	EXPECT_FLOAT_EQ(values[2], 33.0F);
	EXPECT_FLOAT_EQ(values[3], 44.0F);
}

TEST(G14VNext, VNextModelPackageRoundTripsRoPEWithRuntimePositions)
{
	Graph graph;
	Subgraph forward;
	const auto input = forward.AddParam(DataType::Float32, { 2, 2 });
	const auto positions = forward.AddParam(DataType::Int64, { 2 });
	const auto output = Layer::AddRoPEAtPositions(forward, { input, 0 }, { positions, 0 }, 100.0, 0.5);
	forward.SetResults({ output });
	graph.SetForward(graph.AddSubgraph(std::move(forward)));
	graph.SetInputNames({ "input", "positions" });
	graph.SetOutputNames({ "rotated" });

	const auto path = std::filesystem::temp_directory_path() / "litenn_vnext_rope.json";
	Serialization::SaveVNextModelPackage(Detail::BuildExecutableModuleFromGraph(graph), path);
	const auto package = Serialization::LoadVNextModelPackage(path);
	std::filesystem::remove(path);

	const auto& node = package.plan.subgraphs[package.plan.forward].nodes.back();
	ASSERT_TRUE(std::holds_alternative<RoPENode>(node.node));
	const auto& rope = std::get<RoPENode>(node.node);
	ASSERT_TRUE(rope.positions.has_value());
	EXPECT_DOUBLE_EQ(rope.base, 100.0);
	EXPECT_DOUBLE_EQ(rope.frequencyScale, 0.5);

	Runtime::Interpreter<CPU> interpreter;
	std::array inputs{ Tensor<CPU>({ 1.0, 0.0, 1.0, 0.0 }, { 2, 2 }, DataType::Float32),
		               Tensor<CPU>({ 2.0, 6.0 }, { 2 }, DataType::Int64) };
	const auto outputs = interpreter.RunForward(package.plan, inputs);
	ASSERT_EQ(outputs.size(), 1u);
	const auto* values = static_cast<const float*>(outputs[0].UnsafeRawData());
	EXPECT_NEAR(values[0], std::cos(1.0), 1.0e-5);
	EXPECT_NEAR(values[1], std::sin(1.0), 1.0e-5);
	EXPECT_NEAR(values[2], std::cos(3.0), 1.0e-5);
	EXPECT_NEAR(values[3], std::sin(3.0), 1.0e-5);
}

TEST(G14VNext, VNextModelPackageRoundTripsRuntimeStateValueBindings)
{
	Graph graph;
	Subgraph forward;
	const auto input = forward.AddParam(DataType::Float32, { 2 });
	forward.SetResults({ { input, 0 } });
	graph.AddSubgraph(std::move(forward));
	graph.SetForward(0);

	auto state = Runtime::MakeKVCacheState("kv.cache.0", TensorType::Dense(DataType::Float32, ShapeView{ 4 }));
	const std::vector<Runtime::RuntimeStateValueBinding> aliases{
		{ "kv.cache.0", 0, Runtime::RuntimeStateValueKind::FunctionInput, 0, sizeof(float) * 2 },
	};
	const auto schedule =
	    Runtime::BuildRuntimeSchedule(Detail::BuildExecutableModuleFromGraph(graph), { std::move(state) }, aliases);
	const auto path = std::filesystem::temp_directory_path() / "litenn_vnext_state_value_binding.json";
	Serialization::SaveVNextModelPackage(schedule, path);
	const auto package = Serialization::LoadVNextModelPackage(path);
	std::filesystem::remove(path);

	ASSERT_EQ(package.manifest.stateValueBindings.size(), 1u);
	EXPECT_EQ(package.manifest.stateValueBindings[0].stateName, "kv.cache.0");
	EXPECT_EQ(package.manifest.stateValueBindings[0].kind, Runtime::RuntimeStateValueKind::FunctionInput);
	EXPECT_EQ(package.manifest.stateValueBindings[0].stateByteOffset, sizeof(float) * 2);
	EXPECT_NO_THROW(ValidateVNextPackageManifest(package.manifest));
}

TEST(G14VNext, VNextModelPackageRejectsLegacyFormat)
{
	const auto path = std::filesystem::temp_directory_path() / "litenn_vnext_package_legacy.json";
	{
		std::ofstream out(path, std::ios::binary);
		out << "{\"format\":\"litenn.legacy.graph\"}";
	}
	EXPECT_THROW((void) Serialization::LoadVNextModelPackage(path), std::runtime_error);
	std::filesystem::remove(path);
}

TEST(G14VNext, BuildsRuntimeScheduleWithStateBindingsAndTrace)
{
	const auto graph = BuildLinearAddGraph();
	auto kvCache = Runtime::MakeKVCacheState("kv.cache.0", TensorType::Dense(DataType::Float32, ShapeView{ 1, 2, 4 }));

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

TEST(G14VNext, RebindsFunctionValuesToPersistentRuntimeState)
{
	Graph graph;
	Subgraph forward;
	const auto input = forward.AddParam(DataType::Float32, { 2 });
	const auto output =
	    forward.AddNode(UnaryOpNode{ UnaryOp::Negate, { input, 0 } }, { OutputInfo{ DataType::Float32, { 2 } } });
	forward.SetResults({ { output, 0 } });
	graph.AddSubgraph(std::move(forward));
	graph.SetForward(0);

	auto state = Runtime::MakeKVCacheState("kv.cache.0", TensorType::Dense(DataType::Float32, ShapeView{ 2, 2 }));
	const std::vector<Runtime::RuntimeStateValueBinding> aliases{
		{ "kv.cache.0", 0, Runtime::RuntimeStateValueKind::FunctionInput, 0, 0 },
		{ "kv.cache.0", 0, Runtime::RuntimeStateValueKind::FunctionOutput, 0, sizeof(float) * 2 },
	};
	const auto schedule =
	    Runtime::BuildRuntimeSchedule(Detail::BuildExecutableModuleFromGraph(graph), { std::move(state) }, aliases);

	ASSERT_EQ(schedule.stateValueBindings.size(), 2u);
	const auto stateBuffer = *schedule.states[0].memoryBuffer;
	const auto& subgraph = schedule.module.plan.subgraphs[schedule.module.plan.forward];
	const auto* inputAssignment = FindMemoryAssignment(schedule.memory, subgraph.sourceSubgraph, { input, 0 });
	const auto* outputAssignment = FindMemoryAssignment(schedule.memory, subgraph.sourceSubgraph, subgraph.results[0]);
	ASSERT_NE(inputAssignment, nullptr);
	ASSERT_NE(outputAssignment, nullptr);
	EXPECT_EQ(inputAssignment->buffer, stateBuffer);
	EXPECT_EQ(inputAssignment->offset, 0u);
	EXPECT_EQ(outputAssignment->buffer, stateBuffer);
	EXPECT_EQ(outputAssignment->offset, sizeof(float) * 2);
	EXPECT_NO_THROW(Runtime::ValidateRuntimeSchedule(schedule));
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
	manifest.artifacts.push_back(
	    { .name = "broken-kind",
	      .backend = std::string(BackendCPUAOT),
	      .entries = { { .name = "forward", .kind = static_cast<VNextArtifactEntryKind>(99), .function = 0 } },
	      .regions = { { .name = "instructions", .relativePath = "artifacts/broken.o", .byteSize = 1 } } });
	EXPECT_THROW(ValidateVNextPackageManifest(manifest), std::runtime_error);

	manifest = BuildVNextPackageManifest(Detail::BuildExecutableModuleFromGraph(BuildLinearAddGraph()));
	manifest.artifacts.push_back({ .name = "broken", .backend = "", .entries = { { .name = "forward" } } });
	manifest.artifacts[0].backend = std::string(BackendCPUAOT);
	manifest.artifacts[0].entries[0].function = 99;
	EXPECT_THROW(ValidateVNextPackageManifest(manifest), std::runtime_error);

	manifest = BuildVNextPackageManifest(Detail::BuildExecutableModuleFromGraph(BuildLinearAddGraph()));
	manifest.artifacts.push_back(
	    { .name = "missing-binding",
	      .backend = std::string(BackendCPUAOT),
	      .entries = { { .name = "forward", .function = 0, .requiredBufferBindings = { "missing.weight" } } },
	      .regions = { { .name = "instructions", .relativePath = "artifacts/missing.o", .byteSize = 1 } } });
	EXPECT_THROW(ValidateVNextPackageManifest(manifest), std::runtime_error);

	manifest = BuildVNextPackageManifest(Detail::BuildExecutableModuleFromGraph(BuildLinearAddGraph()));
	manifest.layout.mode = "legacy";
	EXPECT_THROW(ValidateVNextPackageManifest(manifest), std::runtime_error);

	manifest = BuildVNextPackageManifest(Detail::BuildExecutableModuleFromGraph(BuildLinearAddGraph()));
	manifest.runtimeSteps[0].id = 99;
	EXPECT_THROW(ValidateVNextPackageManifest(manifest), std::runtime_error);

	manifest = BuildVNextPackageManifest(
	    Detail::BuildExecutableModuleFromGraph(BuildLinearAddGraph()), {}, {}, {},
	    { Runtime::MakeKVCacheState("kv.cache.0", TensorType::Dense(DataType::Float16, ShapeView{ 1, 2, 4, 8 })) });
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
		return std::ranges::any_of(rules, [&](const VNextABIVersionBumpRule& rule) { return rule.area == area; });
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
	llm.kvCaches.push_back(
	    Runtime::MakeKVCacheState("kv.layer0", TensorType::Dense(DataType::Float16, ShapeView{ 1, 8, 16, 64 })));
	llm.currentPosition = Runtime::MakeRuntimeStateBinding(
	    "position", Runtime::RuntimeStateKind::KVCache, "current-position",
	    TensorType::Dense(DataType::Int64, ShapeView{ 1 }), BufferMutability::Mutable, { "read", "increment" });

	Runtime::DiffusionExecutionABI diffusion{
		.latent = Runtime::MakeDiffusionState("latent", "latent-state",
		                                      TensorType::Dense(DataType::Float16, ShapeView{ 1, 4, 128, 128 })),
		.timestepSchedule = Runtime::MakeDiffusionState("timesteps", "timestep-schedule",
		                                                TensorType::Dense(DataType::Float32, ShapeView{ 32 }),
		                                                BufferMutability::Immutable)
	};

	Runtime::TrainingExecutionABI training;
	training.savedActivations.push_back(Runtime::MakeTrainingState(
	    "act.0", "saved-activation", TensorType::Dense(DataType::Float32, ShapeView{ 2, 4 })));
	training.optimizerStates.push_back(Runtime::MakeTrainingState(
	    "adam.m.0", "optimizer-state", TensorType::Dense(DataType::Float32, ShapeView{ 2, 4 })));
	training.recomputationStrategy = "none";

	Runtime::LoRAAdapterExecutionABI lora;
	lora.adapterWeights.push_back(Runtime::MakeLoRAAdapterState("lora.linear.default.A", "adapter-weight-a",
	                                                            TensorType::Dense(DataType::Float16, ShapeView{ 8, 4 }),
	                                                            BufferMutability::Mutable));
	lora.mergeState =
	    Runtime::MakeLoRAAdapterState("lora.linear.default.merge", "adapter-merge-state",
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
