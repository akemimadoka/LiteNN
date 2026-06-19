#include <gtest/gtest.h>

#include <LiteNN.h>
#include <LiteNNImporters.h>

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
		const auto backwardParameter =
		    backwardAndUpdate.AddNode(VariableRefNode{ parameterIndex }, { OutputInfo{ DataType::Float32, { 2 } } });
		const auto inputGradient = backwardAndUpdate.AddNode(
		    BinaryOpNode{ BinaryOp::Multiply, { outputGradient, 0 }, { backwardParameter, 0 } },
		    { OutputInfo{ DataType::Float32, { 2 } } });
		const auto parameterGradient =
		    backwardAndUpdate.AddNode(BinaryOpNode{ BinaryOp::Multiply, { outputGradient, 0 }, { backwardInput, 0 } },
		                              { OutputInfo{ DataType::Float32, { 2 } } });
		const auto update = backwardAndUpdate.AddNode(
		    SGDStepNode{ { backwardParameter, 0 }, { parameterGradient, 0 }, std::nullopt, 0.1, 0.0, 0.0, false },
		    { OutputInfo{ DataType::Float32, { 2 } } });
		backwardAndUpdate.SetResults({ { inputGradient, 0 }, { update, 0 } });
		graph.SetBackward(graph.AddSubgraph(std::move(backwardAndUpdate)));
		return graph;
	}

	Graph BuildTrainStepGraphWithInterpreterLocalBackwardState()
	{
		Graph graph;
		const auto slot = graph.AddActivationSlot(TensorType::Dense(DataType::Float32, ShapeView{ 2 }));

		Subgraph forward;
		const auto input = forward.AddParam(DataType::Float32, { 2 });
		forward.SetResults({ { input, 0 } });
		graph.SetForward(graph.AddSubgraph(std::move(forward)));
		graph.SetInputNames({ "x" });
		graph.SetOutputNames({ "y" });

		Subgraph backward;
		const auto backwardInput = backward.AddParam(DataType::Float32, { 2 });
		const auto saved = backward.AddNode(SaveActivationNode{ { backwardInput, 0 }, slot },
		                                    { OutputInfo{ DataType::Float32, { 2 } } });
		backward.SetResults({ { saved, 0 } });
		graph.SetBackward(graph.AddSubgraph(std::move(backward)));
		return graph;
	}
} // namespace

TEST(G14Remaining, BuildsAndValidatesTrainStepPlan)
{
	const auto graph = BuildTrainableGraph();
	const auto train = Training::BuildTrainStepPlan(Detail::BuildExecutableModuleFromGraph(graph),
	                                                Training::TrainExecutionPolicy::Auto, true);

	EXPECT_EQ(train.policy, Training::TrainExecutionPolicy::AOT);
	EXPECT_TRUE(train.backwardFunction.has_value());
	ASSERT_EQ(train.updates.size(), 1u);
	EXPECT_EQ(train.updates[0].opKind, "SGDStepNode");
	EXPECT_FALSE(train.runtimeStates.empty());
	EXPECT_TRUE(Training::CollectTrainStepAOTReadinessDiagnostics(train).empty());
	EXPECT_NO_THROW(Training::RequireTrainStepAOTReady(train));
	EXPECT_NO_THROW(Training::ValidateTrainStepPlan(train));
}

TEST(G14Remaining, TrainStepAOTReadinessRejectsInterpreterLocalActivationState)
{
	const auto graph = BuildTrainStepGraphWithInterpreterLocalBackwardState();
	const auto train = Training::BuildTrainStepPlan(Detail::BuildExecutableModuleFromGraph(graph),
	                                                Training::TrainExecutionPolicy::AOT, true);

	const auto diagnostics = Training::CollectTrainStepAOTReadinessDiagnostics(train);
	ASSERT_EQ(diagnostics.size(), 1u);
	EXPECT_EQ(diagnostics[0].entryName, "backward");
	EXPECT_EQ(diagnostics[0].opKind, "SaveActivationNode");
	EXPECT_NE(diagnostics[0].message.find("interpreter-local"), std::string::npos);
	EXPECT_THROW(Training::RequireTrainStepAOTReady(train), std::runtime_error);
	EXPECT_NO_THROW(Training::ValidateTrainStepPlan(train));
}

TEST(G14Remaining, TrainStepPlanExposesNamedArtifactEntries)
{
	const auto graph = BuildTrainableGraph();
	const auto train = Training::BuildTrainStepPlan(Detail::BuildExecutableModuleFromGraph(graph),
	                                                Training::TrainExecutionPolicy::AOT, true);

	const auto findEntry = [&](std::string_view name) -> const Training::TrainStepArtifactEntry* {
		for (const auto& entry : train.artifactEntries)
		{
			if (entry.name == name)
			{
				return &entry;
			}
		}
		return nullptr;
	};

	const auto* forward = findEntry("forward");
	ASSERT_NE(forward, nullptr);
	EXPECT_EQ(forward->kind, Training::TrainStepArtifactEntryKind::Forward);
	ASSERT_TRUE(forward->function.has_value());
	EXPECT_EQ(*forward->function, train.forwardFunction);
	EXPECT_FALSE(forward->outputBindings.empty());

	const auto* backward = findEntry("backward");
	ASSERT_NE(backward, nullptr);
	EXPECT_EQ(backward->kind, Training::TrainStepArtifactEntryKind::Backward);
	ASSERT_TRUE(backward->function.has_value());
	ASSERT_TRUE(train.backwardFunction.has_value());
	EXPECT_EQ(*backward->function, *train.backwardFunction);
	EXPECT_FALSE(backward->inputBindings.empty());
	EXPECT_FALSE(backward->outputBindings.empty());

	const auto* loss = findEntry("loss");
	ASSERT_NE(loss, nullptr);
	EXPECT_EQ(loss->kind, Training::TrainStepArtifactEntryKind::Loss);
	EXPECT_FALSE(loss->function.has_value());
	EXPECT_FALSE(loss->inputBindings.empty());
	EXPECT_FALSE(loss->outputBindings.empty());

	ASSERT_EQ(train.updates.size(), 1u);
	const auto* update = findEntry(train.updates[0].name);
	ASSERT_NE(update, nullptr);
	EXPECT_EQ(update->kind, Training::TrainStepArtifactEntryKind::OptimizerUpdate);
	ASSERT_TRUE(update->update.has_value());
	EXPECT_EQ(*update->update, 0u);
	EXPECT_FALSE(update->inputBindings.empty());
	EXPECT_FALSE(update->outputBindings.empty());

	const auto hasBindingRole = [&](const Training::TrainStepArtifactEntry& entry,
	                                std::span<const std::size_t> bindings, Training::TrainStepABIRole role) {
		for (const auto binding : bindings)
		{
			if (train.abiBindings[binding].role == role)
			{
				return true;
			}
		}
		return false;
	};
	EXPECT_TRUE(hasBindingRole(*update, update->inputBindings, Training::TrainStepABIRole::MutableParameter));
	EXPECT_TRUE(hasBindingRole(*update, update->inputBindings, Training::TrainStepABIRole::Gradient));
	EXPECT_TRUE(hasBindingRole(*update, update->outputBindings, Training::TrainStepABIRole::UpdatedParameter));
	EXPECT_NO_THROW(Training::ValidateTrainStepPlan(train));
}

TEST(G14Remaining, BuildsCostBasedPlacementPlanAndCoverage)
{
	const auto graph = BuildTrainableGraph();
	constexpr std::array<std::string_view, 1> backends{ BackendCPUInterpreter };
	const auto placement = Runtime::BuildPlacementPlan(Detail::BuildExecutablePlanFromGraph(graph), backends);

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

TEST(G14Remaining, CostModelRanksCPUAOTCUDANativeAndInterpreterFallback)
{
	Graph graph;
	Subgraph subgraph;
	const auto input = subgraph.AddParam(DataType::Float32, { 8 });
	const auto negated =
	    subgraph.AddNode(UnaryOpNode{ UnaryOp::Negate, { input, 0 } }, { OutputInfo{ DataType::Float32, { 8 } } });
	subgraph.SetResults({ { negated, 0 } });
	graph.SetForward(graph.AddSubgraph(std::move(subgraph)));

	auto registry = BuildDefaultOpSchemaRegistry();
	for (std::string_view kind : { "ParamRefNode", "UnaryOpNode" })
	{
		registry.RegisterCapability(kind, {
		                                      .backend = std::string(BackendCPUAOT),
		                                      .support = BackendSupportLevel::Native,
		                                      .layouts = { TensorLayoutKind::RowMajor },
		                                      .memorySpaces = { TensorMemorySpace::Host },
		                                      .relativeCost = 0.20,
		                                  });
		registry.RegisterCapability(kind, {
		                                      .backend = std::string(BackendCUDABridge),
		                                      .support = BackendSupportLevel::Fallback,
		                                      .layouts = { TensorLayoutKind::RowMajor },
		                                      .memorySpaces = { TensorMemorySpace::Host },
		                                      .fallback = std::string(BackendCPUInterpreter),
		                                      .relativeCost = 0.05,
		                                  });
	}
	registry.RegisterCapability("UnaryOpNode", {
	                                               .backend = std::string(BackendCUDANative),
	                                               .support = BackendSupportLevel::Native,
	                                               .layouts = { TensorLayoutKind::RowMajor },
	                                               .memorySpaces = { TensorMemorySpace::Host },
	                                               .relativeCost = 0.01,
	                                           });
	constexpr std::array<std::string_view, 4> backends{ BackendCPUInterpreter, BackendCPUAOT, BackendCUDANative,
		                                                BackendCUDABridge };
	const auto plan = Detail::BuildExecutablePlanFromGraph(graph);
	const auto placement = Runtime::BuildPlacementPlan(plan, backends, registry);
	ASSERT_EQ(placement.decisions.size(), 2u);
	EXPECT_EQ(placement.decisions[0].backend, BackendCPUAOT);
	EXPECT_EQ(placement.decisions[0].support, BackendSupportLevel::Native);
	EXPECT_EQ(placement.decisions[1].backend, BackendCUDANative);
	EXPECT_EQ(placement.decisions[1].support, BackendSupportLevel::Native);
	EXPECT_TRUE(placement.fallbackSteps.empty());
	EXPECT_NO_THROW(Runtime::ValidatePlacementPlan(placement));

	constexpr std::array<std::string_view, 1> fallbackOnly{ BackendCUDABridge };
	const auto fallbackPlacement = Runtime::BuildPlacementPlan(plan, fallbackOnly, registry);
	ASSERT_EQ(fallbackPlacement.decisions.size(), 2u);
	EXPECT_EQ(fallbackPlacement.decisions[0].support, BackendSupportLevel::Fallback);
	EXPECT_EQ(fallbackPlacement.decisions[0].fallback, BackendCPUInterpreter);
	ASSERT_EQ(fallbackPlacement.fallbackSteps.size(), 2u);

	Runtime::PlacementOptions rejectFallback;
	rejectFallback.fallbackPolicy = Runtime::PlacementFallbackPolicy::RejectFallback;
	EXPECT_THROW((void) Runtime::BuildPlacementPlan(plan, fallbackOnly, registry, rejectFallback), std::runtime_error);
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
	const auto placement = Runtime::BuildPlacementPlan(Detail::BuildExecutablePlanFromGraph(graph), backends, registry);

	ASSERT_EQ(placement.decisions.size(), 1u);
	EXPECT_EQ(placement.decisions[0].support, BackendSupportLevel::Fallback);
	ASSERT_EQ(placement.fallbackSteps.size(), 1u);
	EXPECT_EQ(placement.fallbackSteps[0].fallbackBackend, BackendCPUInterpreter);
	EXPECT_NO_THROW(Runtime::ValidatePlacementPlan(placement));

	auto schedule = Runtime::BuildRuntimeSchedule(BuildExecutableModule(Detail::BuildExecutablePlanFromGraph(graph)));
	Runtime::AppendPlacementFallbackSteps(schedule, placement);
	ASSERT_FALSE(schedule.steps.empty());
	EXPECT_EQ(schedule.steps.back().kind, Runtime::RuntimeScheduleStepKind::Fallback);
	EXPECT_EQ(schedule.steps.back().backend, BackendCUDANative);
	EXPECT_EQ(schedule.steps.back().fallbackBackend, BackendCPUInterpreter);
	EXPECT_NO_THROW(Runtime::ValidateRuntimeSchedule(schedule));
	const auto trace = Runtime::TraceRuntimeSchedule(schedule);
	ASSERT_FALSE(trace.empty());
	EXPECT_EQ(trace.back().kind, Runtime::RuntimeScheduleStepKind::Fallback);
	EXPECT_EQ(trace.back().fallbackBackend, BackendCPUInterpreter);
	EXPECT_NE(trace.back().message.find("fallback from"), std::string::npos);
	const auto profileRecords = Runtime::BuildRuntimeScheduleProfileRecords(schedule);
	ASSERT_EQ(profileRecords.size(), schedule.steps.size());
	EXPECT_EQ(profileRecords.back().kind, Runtime::RuntimeScheduleStepKind::Fallback);
	EXPECT_EQ(profileRecords.back().backend, BackendCUDANative);
	EXPECT_EQ(profileRecords.back().fallbackBackend, BackendCPUInterpreter);
	EXPECT_NE(profileRecords.back().label.find("fallback"), std::string::npos);

	EXPECT_THROW((void) Runtime::BuildPlacementPlan(Detail::BuildExecutablePlanFromGraph(graph), backends, registry, {},
	                                                Runtime::PlacementFallbackPolicy::RejectFallback),
	             std::runtime_error);
}

TEST(G14Remaining, MalformedPlacementFallbackStepsAreRejected)
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
	const auto valid = Runtime::BuildPlacementPlan(Detail::BuildExecutablePlanFromGraph(graph), backends, registry);
	ASSERT_NO_THROW(Runtime::ValidatePlacementPlan(valid));
	ASSERT_FALSE(valid.fallbackSteps.empty());

	auto missingBackend = valid;
	missingBackend.fallbackSteps[0].fallbackBackend.clear();
	EXPECT_THROW(Runtime::ValidatePlacementPlan(missingBackend), std::runtime_error);

	auto unmatchedBackend = valid;
	unmatchedBackend.fallbackSteps[0].requestedBackend = std::string(BackendVulkanNative);
	EXPECT_THROW(Runtime::ValidatePlacementPlan(unmatchedBackend), std::runtime_error);

	auto invalidBuffer = valid;
	invalidBuffer.fallbackSteps[0].outputBuffers.push_back(invalidBuffer.memory.buffers.size());
	EXPECT_THROW(Runtime::ValidatePlacementPlan(invalidBuffer), std::runtime_error);
}

TEST(G14Remaining, BackendPlacementTransfersAreExplicitInScheduleProfile)
{
	Graph graph;
	Subgraph subgraph;
	const auto input = subgraph.AddParam(DataType::Float32, { 2 });
	const auto cast = subgraph.AddNode(CastNode{ { input, 0 } }, { OutputInfo{ DataType::Float32, { 2 } } });
	subgraph.SetResults({ { cast, 0 } });
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
	const auto plan = Detail::BuildExecutablePlanFromGraph(graph);
	const auto placement = Runtime::BuildPlacementPlan(plan, backends, registry);

	ASSERT_EQ(placement.transferSteps.size(), 1u);
	EXPECT_EQ(placement.transferSteps[0].sourceBackend, BackendCPUInterpreter);
	EXPECT_EQ(placement.transferSteps[0].targetBackend, BackendCUDANative);
	EXPECT_NO_THROW(Runtime::ValidatePlacementPlan(placement));

	auto schedule = Runtime::BuildRuntimeSchedule(BuildExecutableModule(plan));
	Runtime::AppendPlacementTransferSteps(schedule, placement);
	ASSERT_FALSE(schedule.steps.empty());
	EXPECT_EQ(schedule.steps.back().kind, Runtime::RuntimeScheduleStepKind::Transfer);
	EXPECT_EQ(schedule.steps.back().backend, BackendCPUInterpreter);
	EXPECT_EQ(schedule.steps.back().fallbackBackend, BackendCUDANative);
	EXPECT_NO_THROW(Runtime::ValidateRuntimeSchedule(schedule));

	const auto trace = Runtime::TraceRuntimeSchedule(schedule);
	ASSERT_FALSE(trace.empty());
	EXPECT_EQ(trace.back().kind, Runtime::RuntimeScheduleStepKind::Transfer);
	EXPECT_NE(trace.back().message.find("transfer from"), std::string::npos);
	const auto profileRecords = Runtime::BuildRuntimeScheduleProfileRecords(schedule);
	ASSERT_EQ(profileRecords.size(), schedule.steps.size());
	EXPECT_EQ(profileRecords.back().kind, Runtime::RuntimeScheduleStepKind::Transfer);
	EXPECT_NE(profileRecords.back().label.find("transfer"), std::string::npos);

	auto invalidTransfer = placement;
	invalidTransfer.transferSteps[0].buffer = invalidTransfer.memory.buffers.size();
	EXPECT_THROW(Runtime::ValidatePlacementPlan(invalidTransfer), std::runtime_error);
}

TEST(G14Remaining, PlacementTransfersCreateSyncStepsAndProfileSummary)
{
	Graph graph;
	Subgraph subgraph;
	const auto input = subgraph.AddParam(DataType::Float32, { 2 });
	const auto cast = subgraph.AddNode(CastNode{ { input, 0 } }, { OutputInfo{ DataType::Float32, { 2 } } });
	subgraph.SetResults({ { cast, 0 } });
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
	const auto plan = Detail::BuildExecutablePlanFromGraph(graph);
	const auto placement = Runtime::BuildPlacementPlan(plan, backends, registry);
	ASSERT_EQ(placement.transferSteps.size(), 1u);

	auto schedule = Runtime::BuildRuntimeSchedule(BuildExecutableModule(plan));
	Runtime::AppendPlacementTransferSteps(schedule, placement);
	Runtime::AppendPlacementSyncSteps(schedule, placement);
	ASSERT_FALSE(schedule.steps.empty());
	EXPECT_EQ(schedule.steps.back().kind, Runtime::RuntimeScheduleStepKind::Sync);
	EXPECT_EQ(schedule.steps.back().backend, BackendCUDANative);
	EXPECT_EQ(schedule.steps.back().fallbackBackend, BackendCPUInterpreter);
	EXPECT_EQ(schedule.steps.back().streamOwner, "cuda-default-stream");
	EXPECT_EQ(schedule.steps.back().eventOwner, "cuda-runtime-event");
	EXPECT_EQ(schedule.steps.back().syncScope, "transfer-boundary");
	EXPECT_NO_THROW(Runtime::ValidateRuntimeSchedule(schedule));

	auto profileRecords = Runtime::BuildRuntimeScheduleProfileRecords(schedule);
	for (auto& record : profileRecords)
	{
		if (record.kind == Runtime::RuntimeScheduleStepKind::Transfer)
		{
			record.wallTimeMs = 0.25;
		}
		if (record.kind == Runtime::RuntimeScheduleStepKind::Sync)
		{
			record.wallTimeMs = 0.05;
			record.deviceTimeMs = 0.04;
		}
	}
	const auto summary = Runtime::BuildRuntimeScheduleProfileSummary(profileRecords);
	EXPECT_EQ(summary.transferSteps, 1u);
	EXPECT_EQ(summary.syncSteps, 1u);
	ASSERT_EQ(summary.devices.size(), 2u);
	const auto cudaDevice =
	    std::ranges::find_if(summary.devices, [](const auto& device) { return device.backend == BackendCUDANative; });
	ASSERT_NE(cudaDevice, summary.devices.end());
	EXPECT_EQ(cudaDevice->transferSteps, 1u);
	EXPECT_EQ(cudaDevice->syncSteps, 1u);
	EXPECT_TRUE(summary.hasMeasuredTimings);
	EXPECT_TRUE(std::ranges::any_of(summary.buckets, [](const Runtime::RuntimeScheduleProfileBucket& bucket) {
		return bucket.kind == Runtime::RuntimeScheduleStepKind::Sync && bucket.hasWallTime && bucket.hasDeviceTime &&
		       bucket.wallTimeMs > 0.0 && bucket.deviceTimeMs > 0.0;
	}));

	auto invalid = schedule;
	ASSERT_FALSE(invalid.steps.empty());
	invalid.steps.back().inputBuffers.clear();
	EXPECT_THROW(Runtime::ValidateRuntimeSchedule(invalid), std::runtime_error);

	auto missingOwnership = schedule;
	ASSERT_FALSE(missingOwnership.steps.empty());
	missingOwnership.steps.back().streamOwner.clear();
	EXPECT_THROW(Runtime::ValidateRuntimeSchedule(missingOwnership), std::runtime_error);
}

TEST(G14Remaining, PlacementConstraintsSelectBackendsAndValidateDefaults)
{
	Graph graph;
	Subgraph subgraph;
	const auto input = subgraph.AddParam(DataType::Float32, { 2 });
	const auto cast = subgraph.AddNode(CastNode{ { input, 0 } }, { OutputInfo{ DataType::Float32, { 2 } } });
	subgraph.SetResults({ { cast, 0 } });
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
	const auto plan = Detail::BuildExecutablePlanFromGraph(graph);

	Runtime::PlacementOptions cpuDefault;
	cpuDefault.defaultBackend = std::string(BackendCPUInterpreter);
	const auto cpuPlacement = Runtime::BuildPlacementPlan(plan, backends, registry, cpuDefault);
	ASSERT_EQ(cpuPlacement.decisions.size(), 2u);
	for (const auto& decision : cpuPlacement.decisions)
	{
		EXPECT_EQ(decision.backend, BackendCPUInterpreter);
	}
	EXPECT_TRUE(cpuPlacement.transferSteps.empty());
	EXPECT_NO_THROW(Runtime::ValidatePlacementPlan(cpuPlacement));

	Runtime::PlacementOptions constrained;
	constrained.defaultBackend = std::string(BackendCPUInterpreter);
	constrained.valueConstraints.push_back({ .subgraph = graph.Forward(),
	                                         .value = { cast, 0 },
	                                         .backend = std::string(BackendCUDANative),
	                                         .reason = "force result on CUDA for heterogeneous smoke" });
	const auto placement = Runtime::BuildPlacementPlan(plan, backends, registry, constrained);
	const auto castDecision = std::ranges::find_if(
	    placement.decisions, [&](const Runtime::PlacementDecision& decision) { return decision.node == cast; });
	ASSERT_NE(castDecision, placement.decisions.end());
	EXPECT_EQ(castDecision->backend, BackendCUDANative);
	ASSERT_EQ(placement.transferSteps.size(), 1u);
	EXPECT_EQ(placement.transferSteps[0].targetBackend, BackendCUDANative);
	EXPECT_NO_THROW(Runtime::ValidatePlacementPlan(placement));

	auto tampered = placement;
	ASSERT_FALSE(tampered.valueConstraints.empty());
	tampered.valueConstraints[0].backend = std::string(BackendCPUInterpreter);
	EXPECT_THROW(Runtime::ValidatePlacementPlan(tampered), std::runtime_error);

	Runtime::PlacementOptions missingBackend;
	missingBackend.nodeConstraints.push_back({ .subgraph = graph.Forward(),
	                                           .node = cast,
	                                           .backend = std::string(BackendVulkanNative),
	                                           .reason = "not in candidate list" });
	EXPECT_THROW((void) Runtime::BuildPlacementPlan(plan, backends, registry, missingBackend), std::runtime_error);
}

TEST(G14Remaining, PlacementSegmentsExposePerBackendBufferBoundaries)
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
	                                     .reason = "isolate a CUDA middle segment" });
	const auto plan = Detail::BuildExecutablePlanFromGraph(graph);
	const auto placement = Runtime::BuildPlacementPlan(plan, backends, registry, options);
	const auto segments = Runtime::BuildPlacementSegments(placement);

	ASSERT_EQ(segments.size(), 3u);
	EXPECT_EQ(segments[0].backend, BackendCPUInterpreter);
	EXPECT_EQ(segments[1].backend, BackendCUDANative);
	EXPECT_EQ(segments[2].backend, BackendCPUInterpreter);
	EXPECT_EQ(segments[1].nodes, (std::vector<NodeId>{ cast }));
	EXPECT_FALSE(segments[1].inputBuffers.empty());
	EXPECT_FALSE(segments[1].outputBuffers.empty());

	auto schedule = Runtime::BuildRuntimeSchedule(BuildExecutableModule(plan));
	Runtime::AppendPlacementSegmentSteps(schedule, placement);
	Runtime::AppendPlacementTransferSteps(schedule, placement);
	Runtime::AppendPlacementSyncSteps(schedule, placement);
	ASSERT_GE(schedule.steps.size(), segments.size());
	EXPECT_TRUE(std::ranges::any_of(schedule.steps, [](const auto& step) {
		return step.kind == Runtime::RuntimeScheduleStepKind::DispatchSegment;
	}));
	EXPECT_NO_THROW(Runtime::ValidateRuntimeSchedule(schedule));

	const auto trace = Runtime::TraceRuntimeSchedule(schedule);
	ASSERT_FALSE(trace.empty());
	EXPECT_TRUE(std::ranges::any_of(
	    trace, [](const auto& event) { return event.message.find("dispatch segment") != std::string::npos; }));
	EXPECT_TRUE(std::ranges::any_of(trace, [](const auto& event) {
		return event.kind == Runtime::RuntimeScheduleStepKind::Sync && event.streamOwner == "cuda-default-stream" &&
		       event.eventOwner == "cuda-runtime-event";
	}));
	const auto profileRecords = Runtime::BuildRuntimeScheduleProfileRecords(schedule);
	ASSERT_EQ(profileRecords.size(), schedule.steps.size());
	EXPECT_TRUE(std::ranges::any_of(
	    profileRecords, [](const auto& record) { return record.label.find("segment") != std::string::npos; }));
	EXPECT_TRUE(std::ranges::any_of(profileRecords, [](const auto& record) {
		return record.kind == Runtime::RuntimeScheduleStepKind::Sync && record.streamOwner == "cuda-default-stream" &&
		       record.syncScope == "transfer-boundary";
	}));
	const auto summary = Runtime::BuildRuntimeScheduleProfileSummary(profileRecords);
	EXPECT_EQ(summary.dispatchSteps, 4u);
	EXPECT_EQ(summary.transferSteps, 2u);
	EXPECT_EQ(summary.syncSteps, 2u);
	const auto cpuDevice = std::ranges::find_if(
	    summary.devices, [](const auto& device) { return device.backend == BackendCPUInterpreter; });
	const auto cudaDevice =
	    std::ranges::find_if(summary.devices, [](const auto& device) { return device.backend == BackendCUDANative; });
	ASSERT_NE(cpuDevice, summary.devices.end());
	ASSERT_NE(cudaDevice, summary.devices.end());
	EXPECT_EQ(cpuDevice->dispatchSteps, 3u);
	EXPECT_EQ(cudaDevice->dispatchSteps, 1u);
	EXPECT_EQ(cpuDevice->transferSteps, 2u);
	EXPECT_EQ(cudaDevice->transferSteps, 2u);
	EXPECT_EQ(cpuDevice->syncSteps, 2u);
	EXPECT_EQ(cudaDevice->syncSteps, 2u);

	auto invalid = schedule;
	const auto segmentStep = std::ranges::find_if(
	    invalid.steps, [](const auto& step) { return step.kind == Runtime::RuntimeScheduleStepKind::DispatchSegment; });
	ASSERT_NE(segmentStep, invalid.steps.end());
	segmentStep->segment = invalid.segments.size();
	EXPECT_THROW(Runtime::ValidateRuntimeSchedule(invalid), std::runtime_error);
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

TEST(G14Remaining, CompatibilityOpsAreReportedByImporterDiagnostics)
{
	Graph graph;
	Subgraph subgraph;
	const auto experts = subgraph.AddParam(DataType::Float32, { 2, 3, 1 });
	const auto input = subgraph.AddParam(DataType::Float32, { 2, 1, 4 });
	const auto ids = subgraph.AddParam(DataType::Int32, { 1, 4 });
	const auto routed = subgraph.AddNode(MulMatIdNode{ { experts, 0 }, { input, 0 }, { ids, 0 } },
	                                     { OutputInfo{ DataType::Float32, { 3, 1, 4 } } });
	subgraph.SetResults({ { routed, 0 } });
	graph.SetForward(graph.AddSubgraph(std::move(subgraph)));

	const auto plan = Detail::BuildExecutablePlanFromGraph(graph);
	const auto diagnostics = CollectExecutablePlanCompatibilityDiagnostics(plan);
	ASSERT_EQ(diagnostics.size(), 1u);
	EXPECT_EQ(diagnostics[0].opKind, "MulMatIdNode");
	EXPECT_EQ(diagnostics[0].domain, OpDomain::GGMLCompatibility);
	EXPECT_NE(diagnostics[0].message.find("compatibility domain"), std::string::npos);

	auto manifest = Serialization::BuildImporterOwnedManifest("gguf", std::move(graph));
	Serialization::AddImportCompatibilityDiagnostics(manifest);
	ASSERT_EQ(manifest.diagnostics.size(), 1u);
	EXPECT_EQ(manifest.diagnostics[0].kind, Serialization::ImportDiagnosticKind::CompatibilityOp);
	EXPECT_NE(manifest.diagnostics[0].message.find("tagged compatibility partition"), std::string::npos);
	EXPECT_NO_THROW(Serialization::ValidateImporterOwnedManifest(manifest));
}

TEST(G14Remaining, VNextRulesAreExecutableInvariants)
{
	const auto graph = BuildTrainableGraph();
	const auto plan = Detail::BuildExecutablePlanFromGraph(graph);
	const auto manifest = BuildVNextPackageManifest(BuildExecutableModule(plan));
	const auto rules = VNextRules();

	EXPECT_GE(rules.size(), 7u);
	EXPECT_NO_THROW(ValidateVNextInvariants(plan, &manifest));
}
