#include <gtest/gtest.h>

#include <LiteNN.h>
#include <LiteNN/Compiler/CompiledModule.h>

#include <array>
#include <cstddef>
#include <utility>

using namespace LiteNN;

namespace
{
	float ReadFloat(const Tensor<CPU>& t, std::size_t i)
	{
		return static_cast<const float*>(t.RawData())[i];
	}

	Graph BuildSimpleAddGraph()
	{
		Graph graph;
		Subgraph sg;
		const auto a = sg.AddParam(DataType::Float32, { 2, 2 });
		const auto b = sg.AddParam(DataType::Float32, { 2, 2 });
		const auto y = sg.AddNode(BinaryOpNode{ BinaryOp::Add, { a, 0 }, { b, 0 } },
		                          { OutputInfo{ DataType::Float32, { 2, 2 } } });
		sg.SetResults({ { y, 0 } });
		graph.AddSubgraph(std::move(sg));
		graph.SetForward(0);
		graph.SetInputNames({ "lhs", "rhs" });
		graph.SetOutputNames({ "sum" });
		return graph;
	}

	Graph BuildSimpleMultiplyGraph()
	{
		Graph graph;
		Subgraph sg;
		const auto a = sg.AddParam(DataType::Float32, { 2, 2 });
		const auto b = sg.AddParam(DataType::Float32, { 2, 2 });
		const auto y = sg.AddNode(BinaryOpNode{ BinaryOp::Multiply, { a, 0 }, { b, 0 } },
		                          { OutputInfo{ DataType::Float32, { 2, 2 } } });
		sg.SetResults({ { y, 0 } });
		graph.AddSubgraph(std::move(sg));
		graph.SetForward(0);
		graph.SetInputNames({ "lhs", "rhs" });
		graph.SetOutputNames({ "product" });
		return graph;
	}
} // namespace

TEST(CompiledModuleCUDATest, RunsCPUAOTBridgeWithCUDATensors)
{
	if (!IsCUDADeviceAvailable())
	{
		GTEST_SKIP() << "CUDA device is not available";
	}

	auto graph = BuildSimpleMultiplyGraph();
	auto compiled = Compiler<CUDA>::Compile(graph, CUDA{});

	EXPECT_EQ(compiled.Backend(), CompiledModuleBackend::CPUNative);
	ASSERT_GT(compiled.Rodata().size(), 0u);
	ASSERT_GT(compiled.Instructions().size(), 0u);
	EXPECT_EQ(compiled.FindInput("lhs"), 0u);
	EXPECT_EQ(compiled.FindOutput("product"), 0u);

	auto lhs = Tensor<CPU>({ 1, 2, 3, 4 }, { 2, 2 }, DataType::Float32).CopyToDevice(CUDA{});
	auto rhs = Tensor<CPU>({ 10, 20, 30, 40 }, { 2, 2 }, DataType::Float32).CopyToDevice(CUDA{});
	std::array<Tensor<CUDA>, 2> inputs = { std::move(lhs), std::move(rhs) };

	auto outputs = compiled.Run(inputs);
	ASSERT_EQ(outputs.size(), 1u);
	auto cpuOutput = outputs[0].CopyToDevice(CPU{});
	EXPECT_FLOAT_EQ(ReadFloat(cpuOutput, 0), 10.0f);
	EXPECT_FLOAT_EQ(ReadFloat(cpuOutput, 1), 40.0f);
	EXPECT_FLOAT_EQ(ReadFloat(cpuOutput, 2), 90.0f);
	EXPECT_FLOAT_EQ(ReadFloat(cpuOutput, 3), 160.0f);

	auto loaded = CompiledModule<CUDA>::Load(compiled.Image(), CUDA{});
	std::array<Tensor<CUDA>, 1> out = {
		Tensor<CUDA>(Uninitialized, { 2, 2 }, DataType::Float32, CUDA{})
	};
	loaded.RunInto(inputs, out);
	auto cpuOutInto = out[0].CopyToDevice(CPU{});
	EXPECT_FLOAT_EQ(ReadFloat(cpuOutInto, 0), 10.0f);
	EXPECT_FLOAT_EQ(ReadFloat(cpuOutInto, 1), 40.0f);
	EXPECT_FLOAT_EQ(ReadFloat(cpuOutInto, 2), 90.0f);
	EXPECT_FLOAT_EQ(ReadFloat(cpuOutInto, 3), 160.0f);
}

TEST(CompiledModuleCUDATest, ArtifactLoadsAsCUDABridge)
{
	if (!IsCUDADeviceAvailable())
	{
		GTEST_SKIP() << "CUDA device is not available";
	}

	auto graph = BuildSimpleMultiplyGraph();
	auto artifact = Compiler<CUDA>::CompileArtifact(graph);
	auto module = artifact.Load(CUDA{});

	EXPECT_EQ(artifact.Backend(), CompiledModuleBackend::CPUNative);
	EXPECT_EQ(module.Backend(), CompiledModuleBackend::CPUNative);

	auto lhs = Tensor<CPU>({ 5, 6, 7, 8 }, { 2, 2 }, DataType::Float32).CopyToDevice(CUDA{});
	auto rhs = Tensor<CPU>({ 1, 2, 3, 4 }, { 2, 2 }, DataType::Float32).CopyToDevice(CUDA{});
	std::array<Tensor<CUDA>, 2> inputs = { std::move(lhs), std::move(rhs) };

	auto outputs = module.Run(inputs);
	auto cpuOutput = outputs[0].CopyToDevice(CPU{});
	EXPECT_FLOAT_EQ(ReadFloat(cpuOutput, 0), 5.0f);
	EXPECT_FLOAT_EQ(ReadFloat(cpuOutput, 1), 12.0f);
	EXPECT_FLOAT_EQ(ReadFloat(cpuOutput, 2), 21.0f);
	EXPECT_FLOAT_EQ(ReadFloat(cpuOutput, 3), 32.0f);
}

TEST(CompiledModuleCUDATest, RunsNativeAddWithCUDATensors)
{
	if (!IsCUDADeviceAvailable())
	{
		GTEST_SKIP() << "CUDA device is not available";
	}
	if (!IsCUDADriverAvailable())
	{
		GTEST_SKIP() << "CUDA driver is not available";
	}

	auto graph = BuildSimpleAddGraph();
	auto artifact = Compiler<CUDA>::CompileArtifact(graph);
	ASSERT_EQ(artifact.Backend(), CompiledModuleBackend::CUDANative);
	auto module = artifact.Load(CUDA{});
	EXPECT_EQ(module.Backend(), CompiledModuleBackend::CUDANative);
	EXPECT_EQ(module.FindOutput("sum"), 0u);

	auto lhs = Tensor<CPU>({ 1, 2, 3, 4 }, { 2, 2 }, DataType::Float32).CopyToDevice(CUDA{});
	auto rhs = Tensor<CPU>({ 10, 20, 30, 40 }, { 2, 2 }, DataType::Float32).CopyToDevice(CUDA{});
	std::array<Tensor<CUDA>, 2> inputs = { std::move(lhs), std::move(rhs) };

	auto outputs = module.Run(inputs);
	ASSERT_EQ(outputs.size(), 1u);
	auto cpuOutput = outputs[0].CopyToDevice(CPU{});
	EXPECT_FLOAT_EQ(ReadFloat(cpuOutput, 0), 11.0f);
	EXPECT_FLOAT_EQ(ReadFloat(cpuOutput, 1), 22.0f);
	EXPECT_FLOAT_EQ(ReadFloat(cpuOutput, 2), 33.0f);
	EXPECT_FLOAT_EQ(ReadFloat(cpuOutput, 3), 44.0f);

	std::array<Tensor<CUDA>, 1> out = {
		Tensor<CUDA>(Uninitialized, { 2, 2 }, DataType::Float32, CUDA{})
	};
	module.RunInto(inputs, out);
	auto cpuOutInto = out[0].CopyToDevice(CPU{});
	EXPECT_FLOAT_EQ(ReadFloat(cpuOutInto, 0), 11.0f);
	EXPECT_FLOAT_EQ(ReadFloat(cpuOutInto, 1), 22.0f);
	EXPECT_FLOAT_EQ(ReadFloat(cpuOutInto, 2), 33.0f);
	EXPECT_FLOAT_EQ(ReadFloat(cpuOutInto, 3), 44.0f);
}
