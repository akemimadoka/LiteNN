#include <gtest/gtest.h>

#include <LiteNN.h>
#include <LiteNN/Layer/ReLU.h>
#include <LiteNN/Pass/AutogradPass.h>
#include <LiteNN/Pass/FusionPass.h>
#include <LiteNN/Pass/InlinePass.h>
#include <LiteNN/Runtime/Interpreter.h>

using namespace LiteNN;

// 辅助函数: 读取 CPU Tensor 的第 i 个 float 元素
static float ReadFloat(const Tensor<CPU>& t, std::size_t i)
{
	return static_cast<const float*>(t.RawData())[i];
}

// 辅助函数: 检查子图中是否存在 CallNode
static bool HasCallNode(const Subgraph& sg)
{
	for (NodeId id = 0; id < sg.NodeCount(); ++id)
	{
		if (std::holds_alternative<CallNode>(sg.GetNodeEntry(id).node))
		{
			return true;
		}
	}
	return false;
}

// 辅助函数: 计算子图中 CallNode 的数量
static std::size_t CountCallNodes(const Subgraph& sg)
{
	std::size_t count = 0;
	for (NodeId id = 0; id < sg.NodeCount(); ++id)
	{
		if (std::holds_alternative<CallNode>(sg.GetNodeEntry(id).node))
		{
			++count;
		}
	}
	return count;
}

// 测试 1: 基础内联 f(x) = Negate(x)
TEST(InlinePass, BasicInline)
{
	Graph graph;

	// 构建 callee: f(x) = Negate(x)
	Subgraph calleeSg;
	const auto cx = calleeSg.AddParam(DataType::Float32, { 2, 2 });
	const auto cNeg =
	    calleeSg.AddNode(UnaryOpNode{ UnaryOp::Negate, { cx, 0 } }, { OutputInfo{ DataType::Float32, { 2, 2 } } });
	calleeSg.SetResults({ { cNeg, 0 } });
	const auto calleeId = graph.AddSubgraph(std::move(calleeSg));

	// 构建前向子图: y = Call(callee, [x])
	Subgraph sg;
	const auto x = sg.AddParam(DataType::Float32, { 2, 2 });
	const auto y = sg.AddNode(CallNode{ calleeId, { { x, 0 } } }, { OutputInfo{ DataType::Float32, { 2, 2 } } });
	sg.SetResults({ { y, 0 } });
	const auto fwdId = graph.AddSubgraph(std::move(sg));
	graph.SetForward(fwdId);

	// 计算未内联的结果
	Tensor<CPU> tensorX({ 1, -2, 3, -4 }, { 2, 2 });
	std::array<Tensor<CPU>, 1> inputs = { Tensor<CPU>(tensorX) };

	Runtime::Interpreter<CPU> interp;
	auto expected = interp.RunForward(graph, inputs);

	// 运行 InlinePass
	InlinePass inlinePass;
	inlinePass.Run(graph);

	// 验证无 CallNode
	const auto& inlinedSg = graph.GetSubgraph(graph.Forward());
	EXPECT_FALSE(HasCallNode(inlinedSg));

	// 验证数值正确
	std::array<Tensor<CPU>, 1> inputs2 = { std::move(tensorX) };
	auto actual = interp.RunForward(graph, inputs2);

	ASSERT_EQ(actual.size(), 1);
	for (std::size_t i = 0; i < 4; ++i)
	{
		EXPECT_FLOAT_EQ(ReadFloat(actual[0], i), ReadFloat(expected[0], i));
	}
}

// 测试 2: 多参数内联 f(a, b) = a + b
TEST(InlinePass, MultiParamInline)
{
	Graph graph;

	// callee: f(a, b) = a + b
	Subgraph calleeSg;
	const auto ca = calleeSg.AddParam(DataType::Float32, { 2 });
	const auto cb = calleeSg.AddParam(DataType::Float32, { 2 });
	const auto cAdd = calleeSg.AddNode(BinaryOpNode{ BinaryOp::Add, { ca, 0 }, { cb, 0 } },
	                                    { OutputInfo{ DataType::Float32, { 2 } } });
	calleeSg.SetResults({ { cAdd, 0 } });
	const auto calleeId = graph.AddSubgraph(std::move(calleeSg));

	// 前向: y = Call(callee, [x1, x2])
	Subgraph sg;
	const auto x1 = sg.AddParam(DataType::Float32, { 2 });
	const auto x2 = sg.AddParam(DataType::Float32, { 2 });
	const auto y =
	    sg.AddNode(CallNode{ calleeId, { { x1, 0 }, { x2, 0 } } }, { OutputInfo{ DataType::Float32, { 2 } } });
	sg.SetResults({ { y, 0 } });
	const auto fwdId = graph.AddSubgraph(std::move(sg));
	graph.SetForward(fwdId);

	Tensor<CPU> tensorX1({ 10, 20 }, { 2 });
	Tensor<CPU> tensorX2({ 3, 7 }, { 2 });
	std::array<Tensor<CPU>, 2> inputs = { Tensor<CPU>(tensorX1), Tensor<CPU>(tensorX2) };

	Runtime::Interpreter<CPU> interp;
	auto expected = interp.RunForward(graph, inputs);

	InlinePass inlinePass;
	inlinePass.Run(graph);

	EXPECT_FALSE(HasCallNode(graph.GetSubgraph(graph.Forward())));

	std::array<Tensor<CPU>, 2> inputs2 = { std::move(tensorX1), std::move(tensorX2) };
	auto actual = interp.RunForward(graph, inputs2);

	ASSERT_EQ(actual.size(), 1);
	EXPECT_FLOAT_EQ(ReadFloat(actual[0], 0), 13);
	EXPECT_FLOAT_EQ(ReadFloat(actual[0], 1), 27);
}

// 测试 3: 嵌套 CallNode g(x) = f(f(x)), f(x) = Negate(x)
TEST(InlinePass, NestedCallNode)
{
	Graph graph;

	// f(x) = Negate(x)
	Subgraph fSg;
	const auto fx = fSg.AddParam(DataType::Float32, { 2 });
	const auto fNeg =
	    fSg.AddNode(UnaryOpNode{ UnaryOp::Negate, { fx, 0 } }, { OutputInfo{ DataType::Float32, { 2 } } });
	fSg.SetResults({ { fNeg, 0 } });
	const auto fId = graph.AddSubgraph(std::move(fSg));

	// g(x) = f(f(x))
	Subgraph gSg;
	const auto gx = gSg.AddParam(DataType::Float32, { 2 });
	const auto inner =
	    gSg.AddNode(CallNode{ fId, { { gx, 0 } } }, { OutputInfo{ DataType::Float32, { 2 } } });
	const auto outer =
	    gSg.AddNode(CallNode{ fId, { { inner, 0 } } }, { OutputInfo{ DataType::Float32, { 2 } } });
	gSg.SetResults({ { outer, 0 } });
	const auto gId = graph.AddSubgraph(std::move(gSg));

	// 前向: y = Call(g, [x])
	Subgraph sg;
	const auto x = sg.AddParam(DataType::Float32, { 2 });
	const auto y = sg.AddNode(CallNode{ gId, { { x, 0 } } }, { OutputInfo{ DataType::Float32, { 2 } } });
	sg.SetResults({ { y, 0 } });
	const auto fwdId = graph.AddSubgraph(std::move(sg));
	graph.SetForward(fwdId);

	Tensor<CPU> tensorX({ 5, -3 }, { 2 });
	std::array<Tensor<CPU>, 1> inputs = { Tensor<CPU>(tensorX) };

	Runtime::Interpreter<CPU> interp;
	auto expected = interp.RunForward(graph, inputs);

	InlinePass inlinePass;
	inlinePass.Run(graph);

	// fixpoint 应完全展开所有层：前向子图无 CallNode
	EXPECT_FALSE(HasCallNode(graph.GetSubgraph(graph.Forward())));

	std::array<Tensor<CPU>, 1> inputs2 = { std::move(tensorX) };
	auto actual = interp.RunForward(graph, inputs2);

	ASSERT_EQ(actual.size(), 1);
	// Negate(Negate(x)) = x
	EXPECT_FLOAT_EQ(ReadFloat(actual[0], 0), 5);
	EXPECT_FLOAT_EQ(ReadFloat(actual[0], 1), -3);
}

// 测试 4: 多输出 callee
TEST(InlinePass, MultiOutputCallee)
{
	Graph graph;

	// callee: returns (Negate(x), Abs(x))
	Subgraph calleeSg;
	const auto cx = calleeSg.AddParam(DataType::Float32, { 2 });
	const auto cNeg =
	    calleeSg.AddNode(UnaryOpNode{ UnaryOp::Negate, { cx, 0 } }, { OutputInfo{ DataType::Float32, { 2 } } });
	const auto cAbs =
	    calleeSg.AddNode(UnaryOpNode{ UnaryOp::Abs, { cx, 0 } }, { OutputInfo{ DataType::Float32, { 2 } } });
	calleeSg.SetResults({ { cNeg, 0 }, { cAbs, 0 } });
	const auto calleeId = graph.AddSubgraph(std::move(calleeSg));

	// 前向: (negResult, absResult) = Call(callee, [x]); y = negResult + absResult
	Subgraph sg;
	const auto x = sg.AddParam(DataType::Float32, { 2 });
	const auto call = sg.AddNode(
	    CallNode{ calleeId, { { x, 0 } } },
	    { OutputInfo{ DataType::Float32, { 2 } }, OutputInfo{ DataType::Float32, { 2 } } });
	// 使用 port 0 (Negate result) 和 port 1 (Abs result)
	const auto y = sg.AddNode(BinaryOpNode{ BinaryOp::Add, { call, 0 }, { call, 1 } },
	                           { OutputInfo{ DataType::Float32, { 2 } } });
	sg.SetResults({ { y, 0 } });
	const auto fwdId = graph.AddSubgraph(std::move(sg));
	graph.SetForward(fwdId);

	Tensor<CPU> tensorX({ -3, 5 }, { 2 });
	std::array<Tensor<CPU>, 1> inputs = { Tensor<CPU>(tensorX) };

	Runtime::Interpreter<CPU> interp;
	auto expected = interp.RunForward(graph, inputs);

	InlinePass inlinePass;
	inlinePass.Run(graph);

	EXPECT_FALSE(HasCallNode(graph.GetSubgraph(graph.Forward())));

	std::array<Tensor<CPU>, 1> inputs2 = { std::move(tensorX) };
	auto actual = interp.RunForward(graph, inputs2);

	ASSERT_EQ(actual.size(), 1);
	// For x=-3: Negate(-3)=3, Abs(-3)=3, sum=6
	// For x=5:  Negate(5)=-5, Abs(5)=5, sum=0
	EXPECT_FLOAT_EQ(ReadFloat(actual[0], 0), ReadFloat(expected[0], 0));
	EXPECT_FLOAT_EQ(ReadFloat(actual[0], 1), ReadFloat(expected[0], 1));
}

// 测试 5: CondNode 不内联分支
TEST(InlinePass, CondNodeBranchesNotInlined)
{
	Graph graph;

	// then 分支: f(x) = Negate(x)
	Subgraph thenSg;
	const auto tx = thenSg.AddParam(DataType::Float32, { 2 });
	const auto tNeg =
	    thenSg.AddNode(UnaryOpNode{ UnaryOp::Negate, { tx, 0 } }, { OutputInfo{ DataType::Float32, { 2 } } });
	thenSg.SetResults({ { tNeg, 0 } });
	const auto thenId = graph.AddSubgraph(std::move(thenSg));

	// else 分支: g(x) = Abs(x)
	Subgraph elseSg;
	const auto ex = elseSg.AddParam(DataType::Float32, { 2 });
	const auto eAbs =
	    elseSg.AddNode(UnaryOpNode{ UnaryOp::Abs, { ex, 0 } }, { OutputInfo{ DataType::Float32, { 2 } } });
	elseSg.SetResults({ { eAbs, 0 } });
	const auto elseId = graph.AddSubgraph(std::move(elseSg));

	// 前向: y = cond ? Negate(x) : Abs(x)
	Subgraph sg;
	const auto cond = sg.AddParam(DataType::Bool, { 1 });
	const auto x = sg.AddParam(DataType::Float32, { 2 });
	const auto y = sg.AddNode(CondNode{ { cond, 0 }, thenId, elseId, { { x, 0 } } },
	                           { OutputInfo{ DataType::Float32, { 2 } } });
	sg.SetResults({ { y, 0 } });
	const auto fwdId = graph.AddSubgraph(std::move(sg));
	graph.SetForward(fwdId);

	// InlinePass 不应该改变 CondNode 的分支
	InlinePass inlinePass;
	inlinePass.Run(graph);

	// 验证 CondNode 仍然存在且数值正确
	Tensor<CPU> tensorCond({ 1.0 }, { 1 }, DataType::Bool);
	Tensor<CPU> tensorX({ -3, 5 }, { 2 });
	std::array<Tensor<CPU>, 2> inputs = { std::move(tensorCond), std::move(tensorX) };

	Runtime::Interpreter<CPU> interp;
	auto results = interp.RunForward(graph, inputs);

	ASSERT_EQ(results.size(), 1);
	// cond=true → Negate: Negate(-3)=3, Negate(5)=-5
	EXPECT_FLOAT_EQ(ReadFloat(results[0], 0), 3);
	EXPECT_FLOAT_EQ(ReadFloat(results[0], 1), -5);
}

// 测试 6: FusedOpNode body 不被内联
TEST(InlinePass, FusedOpNodeNotInlined)
{
	Graph graph;

	// 构建一个有 FusedOpNode 的图
	Subgraph sg;
	const auto x = sg.AddParam(DataType::Float32, { 2, 2 });

	const auto neg =
	    sg.AddNode(UnaryOpNode{ UnaryOp::Negate, { x, 0 } }, { OutputInfo{ DataType::Float32, { 2, 2 } } });
	const auto expNode =
	    sg.AddNode(UnaryOpNode{ UnaryOp::Exp, { neg, 0 } }, { OutputInfo{ DataType::Float32, { 2, 2 } } });

	sg.SetResults({ { expNode, 0 } });
	const auto fwdId = graph.AddSubgraph(std::move(sg));
	graph.SetForward(fwdId);

	// 先运行 FusionPass 创建 FusedOpNode
	FusionPass fusionPass;
	fusionPass.Run(graph);

	// 验证 FusedOpNode 存在
	const auto& fusedSg = graph.GetSubgraph(graph.Forward());
	bool hasFused = false;
	for (NodeId id = 0; id < fusedSg.NodeCount(); ++id)
	{
		if (std::holds_alternative<FusedOpNode>(fusedSg.GetNodeEntry(id).node))
		{
			hasFused = true;
		}
	}
	EXPECT_TRUE(hasFused);

	// 运行 InlinePass（不应该改变 FusedOpNode）
	InlinePass inlinePass;
	inlinePass.Run(graph);

	// FusedOpNode 应该仍然存在
	const auto& afterSg = graph.GetSubgraph(graph.Forward());
	bool stillHasFused = false;
	for (NodeId id = 0; id < afterSg.NodeCount(); ++id)
	{
		if (std::holds_alternative<FusedOpNode>(afterSg.GetNodeEntry(id).node))
		{
			stillHasFused = true;
		}
	}
	EXPECT_TRUE(stillHasFused);
}

// 测试 7: callee 中包含 VariableRefNode 和 ConstantNode
TEST(InlinePass, VariableRefAndConstant)
{
	Graph graph;

	auto wVar = Variable::Create(Tensor<CPU>({ 2, 3 }, { 2 }));
	const auto wIdx = graph.AddVariable(wVar);

	// callee: f(x) = x * W + Constant(10, 20)
	Subgraph calleeSg;
	const auto cx = calleeSg.AddParam(DataType::Float32, { 2 });
	const auto cw = calleeSg.AddNode(VariableRefNode{ wIdx }, { OutputInfo{ DataType::Float32, { 2 } } });
	const auto cMul = calleeSg.AddNode(BinaryOpNode{ BinaryOp::Multiply, { cx, 0 }, { cw, 0 } },
	                                    { OutputInfo{ DataType::Float32, { 2 } } });
	const auto cConst = calleeSg.AddNode(
	    ConstantNode{ Tensor<CPU>({ 10, 20 }, { 2 }).CopyToDevice(PolymorphicDevice{ CPU{} }) },
	    { OutputInfo{ DataType::Float32, { 2 } } });
	const auto cAdd = calleeSg.AddNode(BinaryOpNode{ BinaryOp::Add, { cMul, 0 }, { cConst, 0 } },
	                                    { OutputInfo{ DataType::Float32, { 2 } } });
	calleeSg.SetResults({ { cAdd, 0 } });
	const auto calleeId = graph.AddSubgraph(std::move(calleeSg));

	// 前向: y = Call(callee, [x])
	Subgraph sg;
	const auto x = sg.AddParam(DataType::Float32, { 2 });
	const auto y = sg.AddNode(CallNode{ calleeId, { { x, 0 } } }, { OutputInfo{ DataType::Float32, { 2 } } });
	sg.SetResults({ { y, 0 } });
	const auto fwdId = graph.AddSubgraph(std::move(sg));
	graph.SetForward(fwdId);

	Tensor<CPU> tensorX({ 5, 7 }, { 2 });
	std::array<Tensor<CPU>, 1> inputs = { Tensor<CPU>(tensorX) };

	Runtime::Interpreter<CPU> interp;
	auto expected = interp.RunForward(graph, inputs);

	InlinePass inlinePass;
	inlinePass.Run(graph);

	EXPECT_FALSE(HasCallNode(graph.GetSubgraph(graph.Forward())));

	std::array<Tensor<CPU>, 1> inputs2 = { std::move(tensorX) };
	auto actual = interp.RunForward(graph, inputs2);

	ASSERT_EQ(actual.size(), 1);
	// x * W + C = [5*2+10, 7*3+20] = [20, 41]
	EXPECT_FLOAT_EQ(ReadFloat(actual[0], 0), ReadFloat(expected[0], 0));
	EXPECT_FLOAT_EQ(ReadFloat(actual[0], 1), ReadFloat(expected[0], 1));
}

// 测试 8: InlinePass + AutogradPass 联合
TEST(InlinePass, AfterAutogradPass)
{
	Graph graph;

	// callee: f(x) = Negate(x)
	Subgraph calleeSg;
	const auto cx = calleeSg.AddParam(DataType::Float32, { 2 });
	const auto cNeg =
	    calleeSg.AddNode(UnaryOpNode{ UnaryOp::Negate, { cx, 0 } }, { OutputInfo{ DataType::Float32, { 2 } } });
	calleeSg.SetResults({ { cNeg, 0 } });
	const auto calleeId = graph.AddSubgraph(std::move(calleeSg));

	// 前向: y = Call(callee, [x])
	Subgraph sg;
	const auto x = sg.AddParam(DataType::Float32, { 2 });
	const auto y = sg.AddNode(CallNode{ calleeId, { { x, 0 } } }, { OutputInfo{ DataType::Float32, { 2 } } });
	sg.SetResults({ { y, 0 } });
	const auto fwdId = graph.AddSubgraph(std::move(sg));
	graph.SetForward(fwdId);

	// 参考图（相同结构）
	Graph refGraph;
	{
		Subgraph rCalleeSg;
		const auto rcx = rCalleeSg.AddParam(DataType::Float32, { 2 });
		const auto rcNeg = rCalleeSg.AddNode(UnaryOpNode{ UnaryOp::Negate, { rcx, 0 } },
		                                      { OutputInfo{ DataType::Float32, { 2 } } });
		rCalleeSg.SetResults({ { rcNeg, 0 } });
		const auto rCalleeId = refGraph.AddSubgraph(std::move(rCalleeSg));

		Subgraph rSg;
		const auto rx = rSg.AddParam(DataType::Float32, { 2 });
		const auto ry =
		    rSg.AddNode(CallNode{ rCalleeId, { { rx, 0 } } }, { OutputInfo{ DataType::Float32, { 2 } } });
		rSg.SetResults({ { ry, 0 } });
		const auto rFwdId = refGraph.AddSubgraph(std::move(rSg));
		refGraph.SetForward(rFwdId);
	}

	// 参考：只 autograd
	AutogradPass autograd;
	autograd.Run(refGraph);

	Tensor<CPU> tensorX({ 3, -7 }, { 2 });
	Tensor<CPU> gradOut({ 1, 1 }, { 2 });

	Runtime::Interpreter<CPU> interpRef;
	std::array<Tensor<CPU>, 1> refInputs = { Tensor<CPU>(tensorX) };
	auto expectedFwd = interpRef.RunForward(refGraph, refInputs);

	std::array<Tensor<CPU>, 2> refBwdInputs = { Tensor<CPU>(tensorX), Tensor<CPU>(gradOut) };
	auto expectedBwd = interpRef.RunBackward(refGraph, refBwdInputs);

	// 测试图：autograd → inline
	autograd.Run(graph);
	InlinePass inlinePass;
	inlinePass.Run(graph);

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

// 测试 9: 空 CallNode（callee 只透传参数）
TEST(InlinePass, PassthroughCallee)
{
	Graph graph;

	// callee: f(x) = x（直接返回参数）
	Subgraph calleeSg;
	const auto cx = calleeSg.AddParam(DataType::Float32, { 2 });
	calleeSg.SetResults({ { cx, 0 } });
	const auto calleeId = graph.AddSubgraph(std::move(calleeSg));

	// 前向: y = Negate(Call(callee, [x]))
	Subgraph sg;
	const auto x = sg.AddParam(DataType::Float32, { 2 });
	const auto call = sg.AddNode(CallNode{ calleeId, { { x, 0 } } }, { OutputInfo{ DataType::Float32, { 2 } } });
	const auto y =
	    sg.AddNode(UnaryOpNode{ UnaryOp::Negate, { call, 0 } }, { OutputInfo{ DataType::Float32, { 2 } } });
	sg.SetResults({ { y, 0 } });
	const auto fwdId = graph.AddSubgraph(std::move(sg));
	graph.SetForward(fwdId);

	Tensor<CPU> tensorX({ 10, -20 }, { 2 });
	std::array<Tensor<CPU>, 1> inputs = { Tensor<CPU>(tensorX) };

	Runtime::Interpreter<CPU> interp;
	auto expected = interp.RunForward(graph, inputs);

	InlinePass inlinePass;
	inlinePass.Run(graph);

	EXPECT_FALSE(HasCallNode(graph.GetSubgraph(graph.Forward())));

	std::array<Tensor<CPU>, 1> inputs2 = { std::move(tensorX) };
	auto actual = interp.RunForward(graph, inputs2);

	ASSERT_EQ(actual.size(), 1);
	EXPECT_FLOAT_EQ(ReadFloat(actual[0], 0), -10);
	EXPECT_FLOAT_EQ(ReadFloat(actual[0], 1), 20);
}

// 测试 10: 使用 ReLU Layer（实际 CallNode 场景）
TEST(InlinePass, ReLULayerInline)
{
	Graph graph;

	const auto reluId = Layer::BuildReLU(graph, DataType::Float32, { 2, 3 });

	// 前向: y = ReLU(x)
	Subgraph sg;
	const auto x = sg.AddParam(DataType::Float32, { 2, 3 });
	const auto y = sg.AddNode(CallNode{ reluId, { { x, 0 } } }, { OutputInfo{ DataType::Float32, { 2, 3 } } });
	sg.SetResults({ { y, 0 } });
	const auto fwdId = graph.AddSubgraph(std::move(sg));
	graph.SetForward(fwdId);

	Tensor<CPU> tensorX({ -1, 2, -3, 4, -5, 6 }, { 2, 3 });
	std::array<Tensor<CPU>, 1> inputs = { Tensor<CPU>(tensorX) };

	Runtime::Interpreter<CPU> interp;
	auto expected = interp.RunForward(graph, inputs);

	InlinePass inlinePass;
	inlinePass.Run(graph);

	// ReLU 通过 CallNode 调用，内联后应该无 CallNode
	EXPECT_FALSE(HasCallNode(graph.GetSubgraph(graph.Forward())));

	std::array<Tensor<CPU>, 1> inputs2 = { std::move(tensorX) };
	auto actual = interp.RunForward(graph, inputs2);

	ASSERT_EQ(actual.size(), 1);
	for (std::size_t i = 0; i < 6; ++i)
	{
		EXPECT_FLOAT_EQ(ReadFloat(actual[0], i), ReadFloat(expected[0], i));
	}
}
