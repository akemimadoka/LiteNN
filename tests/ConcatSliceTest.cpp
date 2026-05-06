#include <gtest/gtest.h>

#include <LiteNN.h>
#include <LiteNN/Pass/AutogradPass.h>
#include <LiteNN/Pass/ConstFoldPass.h>
#include <LiteNN/Pass/FusionPass.h>
#include <LiteNN/Pass/InlinePass.h>
#include <LiteNN/Runtime/Interpreter.h>

using namespace LiteNN;

// 辅助函数: 读取 CPU Tensor 的第 i 个 float 元素
static float ReadFloat(const Tensor<CPU>& t, std::size_t i)
{
	return static_cast<const float*>(t.RawData())[i];
}

// ============================================================
// 前向测试
// ============================================================

// 测试 1: Concat axis=0
TEST(ConcatSlice, ConcatForward_Axis0)
{
	Graph graph;

	// y = Concat([a, b], axis=0)
	// a: [2,3], b: [2,3] → y: [4,3]
	Subgraph sg;
	const auto a = sg.AddParam(DataType::Float32, { 2, 3 });
	const auto b = sg.AddParam(DataType::Float32, { 2, 3 });
	const auto y = sg.AddNode(ConcatNode{ { { a, 0 }, { b, 0 } }, 0 }, { OutputInfo{ DataType::Float32, { 4, 3 } } });
	sg.SetResults({ { y, 0 } });
	graph.SetForward(graph.AddSubgraph(std::move(sg)));

	Tensor<CPU> tensorA({ 1, 2, 3, 4, 5, 6 }, { 2, 3 });
	Tensor<CPU> tensorB({ 7, 8, 9, 10, 11, 12 }, { 2, 3 });
	std::array<Tensor<CPU>, 2> inputs = { std::move(tensorA), std::move(tensorB) };

	Runtime::Interpreter<CPU> interp;
	auto results = interp.RunForward(graph, inputs);

	ASSERT_EQ(results.size(), 1);
	ASSERT_EQ(results[0].NumElements(), 12);
	// [1,2,3,4,5,6,7,8,9,10,11,12]
	for (std::size_t i = 0; i < 12; ++i)
	{
		EXPECT_FLOAT_EQ(ReadFloat(results[0], i), static_cast<float>(i + 1));
	}
}

// 测试 2: Concat axis=1
TEST(ConcatSlice, ConcatForward_Axis1)
{
	Graph graph;

	// a: [2,3], b: [2,2] → y: [2,5]
	Subgraph sg;
	const auto a = sg.AddParam(DataType::Float32, { 2, 3 });
	const auto b = sg.AddParam(DataType::Float32, { 2, 2 });
	const auto y = sg.AddNode(ConcatNode{ { { a, 0 }, { b, 0 } }, 1 }, { OutputInfo{ DataType::Float32, { 2, 5 } } });
	sg.SetResults({ { y, 0 } });
	graph.SetForward(graph.AddSubgraph(std::move(sg)));

	Tensor<CPU> tensorA({ 1, 2, 3, 4, 5, 6 }, { 2, 3 });
	Tensor<CPU> tensorB({ 10, 20, 30, 40 }, { 2, 2 });
	std::array<Tensor<CPU>, 2> inputs = { std::move(tensorA), std::move(tensorB) };

	Runtime::Interpreter<CPU> interp;
	auto results = interp.RunForward(graph, inputs);

	ASSERT_EQ(results.size(), 1);
	ASSERT_EQ(results[0].NumElements(), 10);
	// row 0: [1,2,3,10,20], row 1: [4,5,6,30,40]
	float expected[] = { 1, 2, 3, 10, 20, 4, 5, 6, 30, 40 };
	for (std::size_t i = 0; i < 10; ++i)
	{
		EXPECT_FLOAT_EQ(ReadFloat(results[0], i), expected[i]);
	}
}

// 测试 3: Concat 三个输入
TEST(ConcatSlice, ConcatForward_ThreeInputs)
{
	Graph graph;

	// a: [1,2], b: [1,2], c: [1,2] → y: [3,2]
	Subgraph sg;
	const auto a = sg.AddParam(DataType::Float32, { 1, 2 });
	const auto b = sg.AddParam(DataType::Float32, { 1, 2 });
	const auto c = sg.AddParam(DataType::Float32, { 1, 2 });
	const auto y = sg.AddNode(ConcatNode{ { { a, 0 }, { b, 0 }, { c, 0 } }, 0 },
	                           { OutputInfo{ DataType::Float32, { 3, 2 } } });
	sg.SetResults({ { y, 0 } });
	graph.SetForward(graph.AddSubgraph(std::move(sg)));

	Tensor<CPU> tensorA({ 1, 2 }, { 1, 2 });
	Tensor<CPU> tensorB({ 3, 4 }, { 1, 2 });
	Tensor<CPU> tensorC({ 5, 6 }, { 1, 2 });
	std::array<Tensor<CPU>, 3> inputs = { std::move(tensorA), std::move(tensorB), std::move(tensorC) };

	Runtime::Interpreter<CPU> interp;
	auto results = interp.RunForward(graph, inputs);

	ASSERT_EQ(results.size(), 1);
	for (std::size_t i = 0; i < 6; ++i)
	{
		EXPECT_FLOAT_EQ(ReadFloat(results[0], i), static_cast<float>(i + 1));
	}
}

// 测试 4: Slice 基础
TEST(ConcatSlice, SliceForward_Basic)
{
	Graph graph;

	// x: [4,3] → Slice(axis=0, start=1, length=2) → [2,3]
	Subgraph sg;
	const auto x = sg.AddParam(DataType::Float32, { 4, 3 });
	const auto y = sg.AddNode(SliceNode{ { x, 0 }, 0, 1, 2 }, { OutputInfo{ DataType::Float32, { 2, 3 } } });
	sg.SetResults({ { y, 0 } });
	graph.SetForward(graph.AddSubgraph(std::move(sg)));

	Tensor<CPU> tensorX({ 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12 }, { 4, 3 });
	std::array<Tensor<CPU>, 1> inputs = { std::move(tensorX) };

	Runtime::Interpreter<CPU> interp;
	auto results = interp.RunForward(graph, inputs);

	ASSERT_EQ(results.size(), 1);
	ASSERT_EQ(results[0].NumElements(), 6);
	// rows 1-2: [4,5,6,7,8,9]
	float expected[] = { 4, 5, 6, 7, 8, 9 };
	for (std::size_t i = 0; i < 6; ++i)
	{
		EXPECT_FLOAT_EQ(ReadFloat(results[0], i), expected[i]);
	}
}

// 测试 5: Slice axis=1
TEST(ConcatSlice, SliceForward_Axis1)
{
	Graph graph;

	// x: [2,5] → Slice(axis=1, start=1, length=3) → [2,3]
	Subgraph sg;
	const auto x = sg.AddParam(DataType::Float32, { 2, 5 });
	const auto y = sg.AddNode(SliceNode{ { x, 0 }, 1, 1, 3 }, { OutputInfo{ DataType::Float32, { 2, 3 } } });
	sg.SetResults({ { y, 0 } });
	graph.SetForward(graph.AddSubgraph(std::move(sg)));

	Tensor<CPU> tensorX({ 1, 2, 3, 4, 5, 6, 7, 8, 9, 10 }, { 2, 5 });
	std::array<Tensor<CPU>, 1> inputs = { std::move(tensorX) };

	Runtime::Interpreter<CPU> interp;
	auto results = interp.RunForward(graph, inputs);

	ASSERT_EQ(results.size(), 1);
	ASSERT_EQ(results[0].NumElements(), 6);
	// row 0: [2,3,4], row 1: [7,8,9]
	float expected[] = { 2, 3, 4, 7, 8, 9 };
	for (std::size_t i = 0; i < 6; ++i)
	{
		EXPECT_FLOAT_EQ(ReadFloat(results[0], i), expected[i]);
	}
}

// 测试 6: Concat 再 Slice 回来（round trip）
TEST(ConcatSlice, ConcatSlice_RoundTrip)
{
	Graph graph;

	// a: [2,3], b: [3,3]
	// concat = Concat([a, b], axis=0) → [5,3]
	// y = Slice(concat, axis=0, start=0, length=2) → [2,3] (should == a)
	Subgraph sg;
	const auto a = sg.AddParam(DataType::Float32, { 2, 3 });
	const auto b = sg.AddParam(DataType::Float32, { 3, 3 });
	const auto cat =
	    sg.AddNode(ConcatNode{ { { a, 0 }, { b, 0 } }, 0 }, { OutputInfo{ DataType::Float32, { 5, 3 } } });
	const auto y = sg.AddNode(SliceNode{ { cat, 0 }, 0, 0, 2 }, { OutputInfo{ DataType::Float32, { 2, 3 } } });
	sg.SetResults({ { y, 0 } });
	graph.SetForward(graph.AddSubgraph(std::move(sg)));

	Tensor<CPU> tensorA({ 10, 20, 30, 40, 50, 60 }, { 2, 3 });
	Tensor<CPU> tensorB({ 1, 2, 3, 4, 5, 6, 7, 8, 9 }, { 3, 3 });
	std::array<Tensor<CPU>, 2> inputs = { Tensor<CPU>(tensorA), std::move(tensorB) };

	Runtime::Interpreter<CPU> interp;
	auto results = interp.RunForward(graph, inputs);

	ASSERT_EQ(results.size(), 1);
	ASSERT_EQ(results[0].NumElements(), 6);
	for (std::size_t i = 0; i < 6; ++i)
	{
		EXPECT_FLOAT_EQ(ReadFloat(results[0], i), ReadFloat(tensorA, i));
	}
}

// ============================================================
// 梯度测试
// ============================================================

// 测试 7: Concat 梯度 — dy 被 Slice 切分
TEST(ConcatSlice, ConcatGrad)
{
	Graph graph;

	// y = Concat([a, b], axis=0), a: [2,2], b: [1,2] → y: [3,2]
	Subgraph sg;
	const auto a = sg.AddParam(DataType::Float32, { 2, 2 });
	const auto b = sg.AddParam(DataType::Float32, { 1, 2 });
	const auto y = sg.AddNode(ConcatNode{ { { a, 0 }, { b, 0 } }, 0 }, { OutputInfo{ DataType::Float32, { 3, 2 } } });
	sg.SetResults({ { y, 0 } });
	graph.SetForward(graph.AddSubgraph(std::move(sg)));

	AutogradPass autograd;
	autograd.Run(graph);

	Tensor<CPU> tensorA({ 1, 2, 3, 4 }, { 2, 2 });
	Tensor<CPU> tensorB({ 5, 6 }, { 1, 2 });
	Tensor<CPU> gradOut({ 10, 20, 30, 40, 50, 60 }, { 3, 2 });

	Runtime::Interpreter<CPU> interp;
	std::array<Tensor<CPU>, 2> fwdInputs = { Tensor<CPU>(tensorA), Tensor<CPU>(tensorB) };
	auto fwdResults = interp.RunForward(graph, fwdInputs);

	std::array<Tensor<CPU>, 3> bwdInputs = { std::move(tensorA), std::move(tensorB), std::move(gradOut) };
	auto bwdResults = interp.RunBackward(graph, bwdInputs);

	// grad_a = dy[0:2, :] = [[10,20],[30,40]]
	ASSERT_EQ(bwdResults.size(), 2);
	ASSERT_EQ(bwdResults[0].NumElements(), 4);
	EXPECT_FLOAT_EQ(ReadFloat(bwdResults[0], 0), 10);
	EXPECT_FLOAT_EQ(ReadFloat(bwdResults[0], 1), 20);
	EXPECT_FLOAT_EQ(ReadFloat(bwdResults[0], 2), 30);
	EXPECT_FLOAT_EQ(ReadFloat(bwdResults[0], 3), 40);

	// grad_b = dy[2:3, :] = [[50,60]]
	ASSERT_EQ(bwdResults[1].NumElements(), 2);
	EXPECT_FLOAT_EQ(ReadFloat(bwdResults[1], 0), 50);
	EXPECT_FLOAT_EQ(ReadFloat(bwdResults[1], 1), 60);
}

// 测试 8: Slice 梯度 — zero-pad + Concat
TEST(ConcatSlice, SliceGrad)
{
	Graph graph;

	// x: [4,2] → Slice(axis=0, start=1, length=2) → [2,2]
	Subgraph sg;
	const auto x = sg.AddParam(DataType::Float32, { 4, 2 });
	const auto y = sg.AddNode(SliceNode{ { x, 0 }, 0, 1, 2 }, { OutputInfo{ DataType::Float32, { 2, 2 } } });
	sg.SetResults({ { y, 0 } });
	graph.SetForward(graph.AddSubgraph(std::move(sg)));

	AutogradPass autograd;
	autograd.Run(graph);

	Tensor<CPU> tensorX({ 1, 2, 3, 4, 5, 6, 7, 8 }, { 4, 2 });
	Tensor<CPU> gradOut({ 10, 20, 30, 40 }, { 2, 2 });

	Runtime::Interpreter<CPU> interp;
	std::array<Tensor<CPU>, 1> fwdInputs = { Tensor<CPU>(tensorX) };
	auto fwdResults = interp.RunForward(graph, fwdInputs);

	std::array<Tensor<CPU>, 2> bwdInputs = { std::move(tensorX), std::move(gradOut) };
	auto bwdResults = interp.RunBackward(graph, bwdInputs);

	// grad_x = Concat([zeros[1,2], dy, zeros[1,2]], axis=0) → [4,2]
	// = [[0,0], [10,20], [30,40], [0,0]]
	ASSERT_EQ(bwdResults.size(), 1);
	ASSERT_EQ(bwdResults[0].NumElements(), 8);
	float expected[] = { 0, 0, 10, 20, 30, 40, 0, 0 };
	for (std::size_t i = 0; i < 8; ++i)
	{
		EXPECT_FLOAT_EQ(ReadFloat(bwdResults[0], i), expected[i]);
	}
}

// 测试 9: Concat 三输入梯度
TEST(ConcatSlice, ConcatGrad_ThreeInputs)
{
	Graph graph;

	// a: [1,3], b: [2,3], c: [1,3] → y: [4,3]
	Subgraph sg;
	const auto a = sg.AddParam(DataType::Float32, { 1, 3 });
	const auto b = sg.AddParam(DataType::Float32, { 2, 3 });
	const auto c = sg.AddParam(DataType::Float32, { 1, 3 });
	const auto y = sg.AddNode(ConcatNode{ { { a, 0 }, { b, 0 }, { c, 0 } }, 0 },
	                           { OutputInfo{ DataType::Float32, { 4, 3 } } });
	sg.SetResults({ { y, 0 } });
	graph.SetForward(graph.AddSubgraph(std::move(sg)));

	AutogradPass autograd;
	autograd.Run(graph);

	Tensor<CPU> tensorA({ 1, 2, 3 }, { 1, 3 });
	Tensor<CPU> tensorB({ 4, 5, 6, 7, 8, 9 }, { 2, 3 });
	Tensor<CPU> tensorC({ 10, 11, 12 }, { 1, 3 });
	Tensor<CPU> gradOut({ 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12 }, { 4, 3 });

	Runtime::Interpreter<CPU> interp;
	std::array<Tensor<CPU>, 3> fwdInputs = { Tensor<CPU>(tensorA), Tensor<CPU>(tensorB), Tensor<CPU>(tensorC) };
	interp.RunForward(graph, fwdInputs);

	std::array<Tensor<CPU>, 4> bwdInputs = { std::move(tensorA), std::move(tensorB), std::move(tensorC),
		                                      std::move(gradOut) };
	auto bwdResults = interp.RunBackward(graph, bwdInputs);

	ASSERT_EQ(bwdResults.size(), 3);
	// grad_a = dy[0:1] = [1,2,3]
	EXPECT_FLOAT_EQ(ReadFloat(bwdResults[0], 0), 1);
	EXPECT_FLOAT_EQ(ReadFloat(bwdResults[0], 1), 2);
	EXPECT_FLOAT_EQ(ReadFloat(bwdResults[0], 2), 3);
	// grad_b = dy[1:3] = [4,5,6,7,8,9]
	for (std::size_t i = 0; i < 6; ++i)
	{
		EXPECT_FLOAT_EQ(ReadFloat(bwdResults[1], i), static_cast<float>(i + 4));
	}
	// grad_c = dy[3:4] = [10,11,12]
	EXPECT_FLOAT_EQ(ReadFloat(bwdResults[2], 0), 10);
	EXPECT_FLOAT_EQ(ReadFloat(bwdResults[2], 1), 11);
	EXPECT_FLOAT_EQ(ReadFloat(bwdResults[2], 2), 12);
}

// 测试 10: Slice start=0（无前置零填充）
TEST(ConcatSlice, SliceGrad_EdgeCase_Start0)
{
	Graph graph;

	// x: [3,2] → Slice(axis=0, start=0, length=2) → [2,2]
	Subgraph sg;
	const auto x = sg.AddParam(DataType::Float32, { 3, 2 });
	const auto y = sg.AddNode(SliceNode{ { x, 0 }, 0, 0, 2 }, { OutputInfo{ DataType::Float32, { 2, 2 } } });
	sg.SetResults({ { y, 0 } });
	graph.SetForward(graph.AddSubgraph(std::move(sg)));

	AutogradPass autograd;
	autograd.Run(graph);

	Tensor<CPU> tensorX({ 1, 2, 3, 4, 5, 6 }, { 3, 2 });
	Tensor<CPU> gradOut({ 10, 20, 30, 40 }, { 2, 2 });

	Runtime::Interpreter<CPU> interp;
	std::array<Tensor<CPU>, 1> fwdInputs = { Tensor<CPU>(tensorX) };
	interp.RunForward(graph, fwdInputs);

	std::array<Tensor<CPU>, 2> bwdInputs = { std::move(tensorX), std::move(gradOut) };
	auto bwdResults = interp.RunBackward(graph, bwdInputs);

	// grad_x = Concat([dy, zeros[1,2]], axis=0) → [[10,20],[30,40],[0,0]]
	ASSERT_EQ(bwdResults.size(), 1);
	ASSERT_EQ(bwdResults[0].NumElements(), 6);
	float expected[] = { 10, 20, 30, 40, 0, 0 };
	for (std::size_t i = 0; i < 6; ++i)
	{
		EXPECT_FLOAT_EQ(ReadFloat(bwdResults[0], i), expected[i]);
	}
}

// 测试 11: Slice start+length=totalDim（无后置零填充）
TEST(ConcatSlice, SliceGrad_EdgeCase_End)
{
	Graph graph;

	// x: [3,2] → Slice(axis=0, start=1, length=2) → [2,2]
	Subgraph sg;
	const auto x = sg.AddParam(DataType::Float32, { 3, 2 });
	const auto y = sg.AddNode(SliceNode{ { x, 0 }, 0, 1, 2 }, { OutputInfo{ DataType::Float32, { 2, 2 } } });
	sg.SetResults({ { y, 0 } });
	graph.SetForward(graph.AddSubgraph(std::move(sg)));

	AutogradPass autograd;
	autograd.Run(graph);

	Tensor<CPU> tensorX({ 1, 2, 3, 4, 5, 6 }, { 3, 2 });
	Tensor<CPU> gradOut({ 10, 20, 30, 40 }, { 2, 2 });

	Runtime::Interpreter<CPU> interp;
	std::array<Tensor<CPU>, 1> fwdInputs = { Tensor<CPU>(tensorX) };
	interp.RunForward(graph, fwdInputs);

	std::array<Tensor<CPU>, 2> bwdInputs = { std::move(tensorX), std::move(gradOut) };
	auto bwdResults = interp.RunBackward(graph, bwdInputs);

	// grad_x = Concat([zeros[1,2], dy], axis=0) → [[0,0],[10,20],[30,40]]
	ASSERT_EQ(bwdResults.size(), 1);
	ASSERT_EQ(bwdResults[0].NumElements(), 6);
	float expected[] = { 0, 0, 10, 20, 30, 40 };
	for (std::size_t i = 0; i < 6; ++i)
	{
		EXPECT_FLOAT_EQ(ReadFloat(bwdResults[0], i), expected[i]);
	}
}

// ============================================================
// Pass 集成测试
// ============================================================

// 测试 12: 全常量 Concat 折叠
TEST(ConcatSlice, ConstFold_Concat)
{
	Graph graph;

	Subgraph sg;
	const auto a =
	    sg.AddNode(ConstantNode{ Tensor<CPU>({ 1, 2, 3 }, { 1, 3 }).CopyToDevice(PolymorphicDevice{ CPU{} }) },
	               { OutputInfo{ DataType::Float32, { 1, 3 } } });
	const auto b =
	    sg.AddNode(ConstantNode{ Tensor<CPU>({ 4, 5, 6 }, { 1, 3 }).CopyToDevice(PolymorphicDevice{ CPU{} }) },
	               { OutputInfo{ DataType::Float32, { 1, 3 } } });
	const auto y = sg.AddNode(ConcatNode{ { { a, 0 }, { b, 0 } }, 0 }, { OutputInfo{ DataType::Float32, { 2, 3 } } });
	sg.SetResults({ { y, 0 } });
	graph.SetForward(graph.AddSubgraph(std::move(sg)));

	ConstFoldPass constFold;
	constFold.Run(graph);

	// 应该折叠为单个 ConstantNode
	const auto& fwdSg = graph.GetSubgraph(graph.Forward());
	ASSERT_EQ(fwdSg.NodeCount(), 1);
	EXPECT_TRUE(std::holds_alternative<ConstantNode>(fwdSg.GetNodeEntry(0).node));

	// 验证数值
	Runtime::Interpreter<CPU> interp;
	std::array<Tensor<CPU>, 0> inputs;
	auto results = interp.RunForward(graph, inputs);
	ASSERT_EQ(results.size(), 1);
	for (std::size_t i = 0; i < 6; ++i)
	{
		EXPECT_FLOAT_EQ(ReadFloat(results[0], i), static_cast<float>(i + 1));
	}
}

// 测试 13: 常量 Slice 折叠
TEST(ConcatSlice, ConstFold_Slice)
{
	Graph graph;

	Subgraph sg;
	const auto x = sg.AddNode(
	    ConstantNode{ Tensor<CPU>({ 1, 2, 3, 4, 5, 6 }, { 2, 3 }).CopyToDevice(PolymorphicDevice{ CPU{} }) },
	    { OutputInfo{ DataType::Float32, { 2, 3 } } });
	const auto y = sg.AddNode(SliceNode{ { x, 0 }, 0, 1, 1 }, { OutputInfo{ DataType::Float32, { 1, 3 } } });
	sg.SetResults({ { y, 0 } });
	graph.SetForward(graph.AddSubgraph(std::move(sg)));

	ConstFoldPass constFold;
	constFold.Run(graph);

	const auto& fwdSg = graph.GetSubgraph(graph.Forward());
	ASSERT_EQ(fwdSg.NodeCount(), 1);
	EXPECT_TRUE(std::holds_alternative<ConstantNode>(fwdSg.GetNodeEntry(0).node));

	Runtime::Interpreter<CPU> interp;
	std::array<Tensor<CPU>, 0> inputs;
	auto results = interp.RunForward(graph, inputs);
	ASSERT_EQ(results.size(), 1);
	ASSERT_EQ(results[0].NumElements(), 3);
	EXPECT_FLOAT_EQ(ReadFloat(results[0], 0), 4);
	EXPECT_FLOAT_EQ(ReadFloat(results[0], 1), 5);
	EXPECT_FLOAT_EQ(ReadFloat(results[0], 2), 6);
}

// 测试 14: InlinePass — callee 中包含 ConcatNode
TEST(ConcatSlice, InlinePass_ConcatInCallee)
{
	Graph graph;

	// callee: f(a, b) = Concat([a, b], axis=0)
	Subgraph calleeSg;
	const auto ca = calleeSg.AddParam(DataType::Float32, { 2 });
	const auto cb = calleeSg.AddParam(DataType::Float32, { 2 });
	const auto cConcat = calleeSg.AddNode(ConcatNode{ { { ca, 0 }, { cb, 0 } }, 0 },
	                                       { OutputInfo{ DataType::Float32, { 4 } } });
	calleeSg.SetResults({ { cConcat, 0 } });
	const auto calleeId = graph.AddSubgraph(std::move(calleeSg));

	// 前向: y = Call(callee, [x1, x2])
	Subgraph sg;
	const auto x1 = sg.AddParam(DataType::Float32, { 2 });
	const auto x2 = sg.AddParam(DataType::Float32, { 2 });
	const auto y =
	    sg.AddNode(CallNode{ calleeId, { { x1, 0 }, { x2, 0 } } }, { OutputInfo{ DataType::Float32, { 4 } } });
	sg.SetResults({ { y, 0 } });
	graph.SetForward(graph.AddSubgraph(std::move(sg)));

	Tensor<CPU> tensorX1({ 10, 20 }, { 2 });
	Tensor<CPU> tensorX2({ 30, 40 }, { 2 });

	Runtime::Interpreter<CPU> interp;
	std::array<Tensor<CPU>, 2> inputs = { Tensor<CPU>(tensorX1), Tensor<CPU>(tensorX2) };
	auto expected = interp.RunForward(graph, inputs);

	InlinePass inlinePass;
	inlinePass.Run(graph);

	// 验证无 CallNode
	const auto& fwdSg = graph.GetSubgraph(graph.Forward());
	bool hasCall = false;
	for (NodeId id = 0; id < fwdSg.NodeCount(); ++id)
	{
		if (std::holds_alternative<CallNode>(fwdSg.GetNodeEntry(id).node))
		{
			hasCall = true;
		}
	}
	EXPECT_FALSE(hasCall);

	std::array<Tensor<CPU>, 2> inputs2 = { std::move(tensorX1), std::move(tensorX2) };
	auto actual = interp.RunForward(graph, inputs2);

	ASSERT_EQ(actual.size(), 1);
	for (std::size_t i = 0; i < 4; ++i)
	{
		EXPECT_FLOAT_EQ(ReadFloat(actual[0], i), ReadFloat(expected[0], i));
	}
}

// 测试 15: 全流水线 — AutogradPass → InlinePass → ConstFoldPass → FusionPass
TEST(ConcatSlice, FullPipeline)
{
	Graph graph;

	// y = Concat([x, Negate(x)], axis=0), x: [2,2] → y: [4,2]
	Subgraph sg;
	const auto x = sg.AddParam(DataType::Float32, { 2, 2 });
	const auto negX =
	    sg.AddNode(UnaryOpNode{ UnaryOp::Negate, { x, 0 } }, { OutputInfo{ DataType::Float32, { 2, 2 } } });
	const auto y =
	    sg.AddNode(ConcatNode{ { { x, 0 }, { negX, 0 } }, 0 }, { OutputInfo{ DataType::Float32, { 4, 2 } } });
	sg.SetResults({ { y, 0 } });
	graph.SetForward(graph.AddSubgraph(std::move(sg)));

	// 参考图（相同结构，只 autograd）
	Graph refGraph;
	{
		Subgraph rSg;
		const auto rx = rSg.AddParam(DataType::Float32, { 2, 2 });
		const auto rNeg =
		    rSg.AddNode(UnaryOpNode{ UnaryOp::Negate, { rx, 0 } }, { OutputInfo{ DataType::Float32, { 2, 2 } } });
		const auto ry =
		    rSg.AddNode(ConcatNode{ { { rx, 0 }, { rNeg, 0 } }, 0 }, { OutputInfo{ DataType::Float32, { 4, 2 } } });
		rSg.SetResults({ { ry, 0 } });
		refGraph.SetForward(refGraph.AddSubgraph(std::move(rSg)));
	}

	AutogradPass autograd;
	autograd.Run(refGraph);

	// 全流水线
	autograd.Run(graph);
	InlinePass().Run(graph);
	ConstFoldPass().Run(graph);
	FusionPass().Run(graph);

	Tensor<CPU> tensorX({ 1, 2, 3, 4 }, { 2, 2 });
	Tensor<CPU> gradOut({ 10, 20, 30, 40, 50, 60, 70, 80 }, { 4, 2 });

	Runtime::Interpreter<CPU> interpRef;
	std::array<Tensor<CPU>, 1> refFwd = { Tensor<CPU>(tensorX) };
	auto expectedFwd = interpRef.RunForward(refGraph, refFwd);

	std::array<Tensor<CPU>, 2> refBwd = { Tensor<CPU>(tensorX), Tensor<CPU>(gradOut) };
	auto expectedBwd = interpRef.RunBackward(refGraph, refBwd);

	Runtime::Interpreter<CPU> interp;
	std::array<Tensor<CPU>, 1> fwdInputs = { Tensor<CPU>(tensorX) };
	auto actualFwd = interp.RunForward(graph, fwdInputs);

	ASSERT_EQ(actualFwd.size(), expectedFwd.size());
	for (std::size_t i = 0; i < actualFwd[0].NumElements(); ++i)
	{
		EXPECT_FLOAT_EQ(ReadFloat(actualFwd[0], i), ReadFloat(expectedFwd[0], i));
	}

	std::array<Tensor<CPU>, 2> bwdInputs = { std::move(tensorX), std::move(gradOut) };
	auto actualBwd = interp.RunBackward(graph, bwdInputs);

	ASSERT_EQ(actualBwd.size(), expectedBwd.size());
	for (std::size_t i = 0; i < actualBwd.size(); ++i)
	{
		for (std::size_t j = 0; j < actualBwd[i].NumElements(); ++j)
		{
			EXPECT_FLOAT_EQ(ReadFloat(actualBwd[i], j), ReadFloat(expectedBwd[i], j));
		}
	}
}
