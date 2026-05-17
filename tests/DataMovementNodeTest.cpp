#include <gtest/gtest.h>

#include <LiteNN.h>
#include <LiteNN/Layer/BroadcastTo.h>
#include <LiteNN/Layer/Gather.h>
#include <LiteNN/Layer/Pad.h>
#include <LiteNN/Layer/Scatter.h>
#include <LiteNN/Pass/ConstFoldPass.h>
#include <LiteNN/Runtime/Interpreter.h>

#include <array>
#include <cstdint>
#include <filesystem>
#include <ranges>
#include <span>
#include <vector>

using namespace LiteNN;

namespace
{
	float ReadFloat(const Tensor<CPU>& tensor, std::size_t index)
	{
		return static_cast<const float*>(tensor.RawData())[index];
	}

	Tensor<CPU> MakeInt32Tensor(std::initializer_list<std::int32_t> values,
	                            std::initializer_list<std::size_t> shape)
	{
		CPU device;
		Tensor<CPU> tensor(Uninitialized, shape, DataType::Int32, device);
		DeviceTraits<CPU>::CopyFromCPU(device, DataType::Int32, tensor.RawData(), DataType::Int32, values.begin(),
		                               values.size());
		return tensor;
	}

	std::vector<Tensor<CPU>> RunGraph(Graph& graph, std::vector<Tensor<CPU>> inputs)
	{
		Runtime::Interpreter<CPU> interpreter;
		return interpreter.RunForward(graph, inputs);
	}

	void ExpectShape(ShapeView shape, std::initializer_list<std::size_t> expected)
	{
		EXPECT_TRUE(std::ranges::equal(shape.Dims, expected));
	}
} // namespace

TEST(DataMovementNode, BroadcastToLeadingAndSingletonDims)
{
	Graph graph;
	Subgraph sg;
	const auto x = sg.AddParam(DataType::Float32, { 1, 3 });
	const auto y = Layer::AddBroadcastTo(sg, { x, 0 }, std::array<std::size_t, 3>{ 2, 2, 3 });
	sg.SetResults({ y });
	graph.SetForward(graph.AddSubgraph(std::move(sg)));

	std::vector<Tensor<CPU>> inputs;
	inputs.emplace_back(Tensor<CPU>({ 10, 20, 30 }, { 1, 3 }));
	const auto outputs = RunGraph(graph, std::move(inputs));

	ASSERT_EQ(outputs.size(), 1);
	ExpectShape(outputs[0].Shape(), { 2, 2, 3 });
	const float expected[] = { 10, 20, 30, 10, 20, 30, 10, 20, 30, 10, 20, 30 };
	for (std::size_t i = 0; i < outputs[0].NumElements(); ++i)
	{
		EXPECT_FLOAT_EQ(ReadFloat(outputs[0], i), expected[i]);
	}
}

TEST(DataMovementNode, PadConstantReflectAndReplicate)
{
	{
		Graph graph;
		Subgraph sg;
		const auto x = sg.AddParam(DataType::Float32, { 2, 2 });
		const auto y = Layer::AddPad(sg, { x, 0 }, std::array<std::size_t, 2>{ 1, 0 },
		                             std::array<std::size_t, 2>{ 0, 1 }, PadMode::Constant, -1.0);
		sg.SetResults({ y });
		graph.SetForward(graph.AddSubgraph(std::move(sg)));

		std::vector<Tensor<CPU>> inputs;
		inputs.emplace_back(Tensor<CPU>({ 1, 2, 3, 4 }, { 2, 2 }));
		const auto outputs = RunGraph(graph, std::move(inputs));
		ExpectShape(outputs[0].Shape(), { 3, 3 });
		const float expected[] = { -1, -1, -1, 1, 2, -1, 3, 4, -1 };
		for (std::size_t i = 0; i < outputs[0].NumElements(); ++i)
		{
			EXPECT_FLOAT_EQ(ReadFloat(outputs[0], i), expected[i]);
		}
	}

	{
		Graph graph;
		Subgraph sg;
		const auto x = sg.AddParam(DataType::Float32, { 3 });
		const auto reflect = Layer::AddPad(sg, { x, 0 }, std::array<std::size_t, 1>{ 2 },
		                                   std::array<std::size_t, 1>{ 2 }, PadMode::Reflect);
		const auto replicate = Layer::AddPad(sg, { x, 0 }, std::array<std::size_t, 1>{ 2 },
		                                     std::array<std::size_t, 1>{ 2 }, PadMode::Replicate);
		sg.SetResults({ reflect, replicate });
		graph.SetForward(graph.AddSubgraph(std::move(sg)));

		std::vector<Tensor<CPU>> inputs;
		inputs.emplace_back(Tensor<CPU>({ 1, 2, 3 }, { 3 }));
		const auto outputs = RunGraph(graph, std::move(inputs));
		const float reflectExpected[] = { 3, 2, 1, 2, 3, 2, 1 };
		const float replicateExpected[] = { 1, 1, 1, 2, 3, 3, 3 };
		for (std::size_t i = 0; i < outputs[0].NumElements(); ++i)
		{
			EXPECT_FLOAT_EQ(ReadFloat(outputs[0], i), reflectExpected[i]);
			EXPECT_FLOAT_EQ(ReadFloat(outputs[1], i), replicateExpected[i]);
		}
	}
}

TEST(DataMovementNode, GatherArbitraryAxisAndIndexRank)
{
	Graph graph;
	Subgraph sg;
	const auto data = sg.AddParam(DataType::Float32, { 2, 3 });
	const auto indices = sg.AddParam(DataType::Int32, { 2 });
	const auto y = Layer::AddGather(sg, { data, 0 }, { indices, 0 }, 1);
	sg.SetResults({ y });
	graph.SetForward(graph.AddSubgraph(std::move(sg)));

	std::vector<Tensor<CPU>> inputs;
	inputs.emplace_back(Tensor<CPU>({ 1, 2, 3, 4, 5, 6 }, { 2, 3 }));
	inputs.emplace_back(MakeInt32Tensor({ 2, 0 }, { 2 }));
	const auto outputs = RunGraph(graph, std::move(inputs));

	ExpectShape(outputs[0].Shape(), { 2, 2 });
	const float expected[] = { 3, 1, 6, 4 };
	for (std::size_t i = 0; i < outputs[0].NumElements(); ++i)
	{
		EXPECT_FLOAT_EQ(ReadFloat(outputs[0], i), expected[i]);
	}
}

TEST(DataMovementNode, ScatterUpdateAndAdd)
{
	Graph graph;
	Subgraph sg;
	const auto data = sg.AddParam(DataType::Float32, { 2, 3 });
	const auto indices = sg.AddParam(DataType::Int32, { 2 });
	const auto updates = sg.AddParam(DataType::Float32, { 2, 2 });
	const auto updated = Layer::AddScatter(sg, { data, 0 }, { indices, 0 }, { updates, 0 }, 1, ScatterMode::Update);
	const auto added = Layer::AddScatter(sg, { data, 0 }, { indices, 0 }, { updates, 0 }, 1, ScatterMode::Add);
	sg.SetResults({ updated, added });
	graph.SetForward(graph.AddSubgraph(std::move(sg)));

	std::vector<Tensor<CPU>> inputs;
	inputs.emplace_back(Tensor<CPU>({ 1, 1, 1, 1, 1, 1 }, { 2, 3 }));
	inputs.emplace_back(MakeInt32Tensor({ 1, 1 }, { 2 }));
	inputs.emplace_back(Tensor<CPU>({ 2, 3, 4, 5 }, { 2, 2 }));
	const auto outputs = RunGraph(graph, std::move(inputs));

	const float updateExpected[] = { 1, 3, 1, 1, 5, 1 };
	const float addExpected[] = { 1, 6, 1, 1, 10, 1 };
	for (std::size_t i = 0; i < outputs[0].NumElements(); ++i)
	{
		EXPECT_FLOAT_EQ(ReadFloat(outputs[0], i), updateExpected[i]);
		EXPECT_FLOAT_EQ(ReadFloat(outputs[1], i), addExpected[i]);
	}
}

TEST(DataMovementNode, ConstFoldBroadcastAndGather)
{
	Graph graph;
	Subgraph sg;
	const auto data = sg.AddNode(
	    ConstantNode{ Tensor<CPU>({ 1, 2, 3, 4, 5, 6 }, { 2, 3 }).CopyToDevice(PolymorphicDevice{ CPU{} }) },
	    { OutputInfo{ DataType::Float32, { 2, 3 } } });
	const auto indices = sg.AddNode(
	    ConstantNode{ MakeInt32Tensor({ 2, 0 }, { 2 }).CopyToDevice(PolymorphicDevice{ CPU{} }) },
	    { OutputInfo{ DataType::Int32, { 2 } } });
	const auto gathered = sg.AddNode(GatherNode{ { data, 0 }, { indices, 0 }, 1 },
	                                 { OutputInfo{ DataType::Float32, { 2, 2 } } });
	sg.SetResults({ { gathered, 0 } });
	graph.SetForward(graph.AddSubgraph(std::move(sg)));

	ConstFoldPass pass;
	pass.Run(graph);

	const auto outputs = RunGraph(graph, {});
	const float expected[] = { 3, 1, 6, 4 };
	for (std::size_t i = 0; i < outputs[0].NumElements(); ++i)
	{
		EXPECT_FLOAT_EQ(ReadFloat(outputs[0], i), expected[i]);
	}
}

TEST(DataMovementNode, SerializationRoundTripPreservesG51Nodes)
{
	Graph graph;
	Subgraph sg;
	const auto x = sg.AddParam(DataType::Float32, { 1, 3 });
	const auto data = sg.AddParam(DataType::Float32, { 2, 3 });
	const auto indices = sg.AddParam(DataType::Int32, { 2 });
	const auto updates = sg.AddParam(DataType::Float32, { 2, 2 });
	const auto broadcast = Layer::AddBroadcastTo(sg, { x, 0 }, std::array<std::size_t, 3>{ 2, 1, 3 });
	const auto pad = Layer::AddPad(sg, { data, 0 }, std::array<std::size_t, 2>{ 0, 1 },
	                               std::array<std::size_t, 2>{ 0, 0 }, PadMode::Constant, -5.0);
	const auto gather = Layer::AddGather(sg, { data, 0 }, { indices, 0 }, 1);
	const auto scatter = Layer::AddScatter(sg, { data, 0 }, { indices, 0 }, { updates, 0 }, 1, ScatterMode::Add);
	sg.SetResults({ broadcast, pad, gather, scatter });
	graph.SetForward(graph.AddSubgraph(std::move(sg)));

	const auto path = std::filesystem::path("litenn_g51_nodes_roundtrip_test.ltnn");
	std::filesystem::remove(path);
	Serialization::SaveModel(graph, path);
	auto loaded = Serialization::LoadModel(path);
	std::filesystem::remove(path);

	std::vector<Tensor<CPU>> inputs;
	inputs.emplace_back(Tensor<CPU>({ 7, 8, 9 }, { 1, 3 }));
	inputs.emplace_back(Tensor<CPU>({ 1, 2, 3, 4, 5, 6 }, { 2, 3 }));
	inputs.emplace_back(MakeInt32Tensor({ 2, 0 }, { 2 }));
	inputs.emplace_back(Tensor<CPU>({ 10, 20, 30, 40 }, { 2, 2 }));
	const auto outputs = RunGraph(loaded, std::move(inputs));

	ASSERT_EQ(outputs.size(), 4);
	ExpectShape(outputs[0].Shape(), { 2, 1, 3 });
	EXPECT_FLOAT_EQ(ReadFloat(outputs[0], 3), 7);
	ExpectShape(outputs[1].Shape(), { 2, 4 });
	EXPECT_FLOAT_EQ(ReadFloat(outputs[1], 0), -5);
	EXPECT_FLOAT_EQ(ReadFloat(outputs[2], 0), 3);
	EXPECT_FLOAT_EQ(ReadFloat(outputs[3], 0), 21);
	EXPECT_FLOAT_EQ(ReadFloat(outputs[3], 2), 13);
}
