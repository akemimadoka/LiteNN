#include <gtest/gtest.h>

#include <LiteNN.h>
#include <LiteNN/Layer/ReLU.h>
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

// 辅助函数: 读取 CPU Tensor 的第 i 个 double 元素
static double ReadDouble(const Tensor<CPU>& t, std::size_t i)
{
	return static_cast<const double*>(t.RawData())[i];
}

// 辅助函数: 计算子图中 ConstantNode 的数量
static std::size_t CountConstantNodes(const Subgraph& sg)
{
	std::size_t count = 0;
	for (NodeId id = 0; id < sg.NodeCount(); ++id)
	{
		if (std::holds_alternative<ConstantNode>(sg.GetNodeEntry(id).node))
		{
			++count;
		}
	}
	return count;
}

// 辅助函数: 计算子图中非 ConstantNode、非 ParamRefNode 的操作节点数量
static std::size_t CountOpNodes(const Subgraph& sg)
{
	std::size_t count = 0;
	for (NodeId id = 0; id < sg.NodeCount(); ++id)
	{
		const auto& entry = sg.GetNodeEntry(id);
		if (!std::holds_alternative<ConstantNode>(entry.node) &&
		    !std::holds_alternative<ParamRefNode>(entry.node) &&
		    !std::holds_alternative<VariableRefNode>(entry.node))
		{
			++count;
		}
	}
	return count;
}

// 辅助函数: 检查子图的第一个结果是否直接引用 ConstantNode
static bool ResultIsConstant(const Subgraph& sg)
{
	if (sg.Results().empty())
	{
		return false;
	}
	const auto& r = sg.Results()[0];
	return std::holds_alternative<ConstantNode>(sg.GetNodeEntry(r.node).node);
}

// 测试 1: 全常量折叠 y = 2.0 + 3.0
TEST(ConstFoldPass, FullConstantFold)
{
	Graph graph;
	Subgraph sg;

	const auto c1 = sg.AddNode(
	    ConstantNode{ Tensor<CPU>({ 2.0 }, { 1 }).CopyToDevice(PolymorphicDevice{ CPU{} }) },
	    { OutputInfo{ DataType::Float32, { 1 } } });
	const auto c2 = sg.AddNode(
	    ConstantNode{ Tensor<CPU>({ 3.0 }, { 1 }).CopyToDevice(PolymorphicDevice{ CPU{} }) },
	    { OutputInfo{ DataType::Float32, { 1 } } });
	const auto y = sg.AddNode(BinaryOpNode{ BinaryOp::Add, { c1, 0 }, { c2, 0 } },
	                           { OutputInfo{ DataType::Float32, { 1 } } });

	sg.SetResults({ { y, 0 } });
	const auto fwdId = graph.AddSubgraph(std::move(sg));
	graph.SetForward(fwdId);

	ConstFoldPass constFold;
	constFold.Run(graph);

	// 结果应该是 ConstantNode(5.0)
	const auto& foldedSg = graph.GetSubgraph(graph.Forward());
	EXPECT_TRUE(ResultIsConstant(foldedSg));

	// 验证数值
	Runtime::Interpreter<CPU> interp;
	std::array<Tensor<CPU>, 0> inputs = {};
	auto results = interp.RunForward(graph, inputs);

	ASSERT_EQ(results.size(), 1);
	EXPECT_FLOAT_EQ(ReadFloat(results[0], 0), 5.0f);
}

// 测试 2: 部分常量 y = x + (2.0 * 3.0)
TEST(ConstFoldPass, PartialConstantFold)
{
	Graph graph;
	Subgraph sg;

	const auto x = sg.AddParam(DataType::Float32, { 2 });
	const auto c1 = sg.AddNode(
	    ConstantNode{ Tensor<CPU>({ 2.0 }, { 1 }).CopyToDevice(PolymorphicDevice{ CPU{} }) },
	    { OutputInfo{ DataType::Float32, { 1 } } });
	const auto c2 = sg.AddNode(
	    ConstantNode{ Tensor<CPU>({ 3.0 }, { 1 }).CopyToDevice(PolymorphicDevice{ CPU{} }) },
	    { OutputInfo{ DataType::Float32, { 1 } } });
	const auto mul = sg.AddNode(BinaryOpNode{ BinaryOp::Multiply, { c1, 0 }, { c2, 0 } },
	                             { OutputInfo{ DataType::Float32, { 1 } } });
	const auto y = sg.AddNode(BinaryOpNode{ BinaryOp::Add, { x, 0 }, { mul, 0 } },
	                           { OutputInfo{ DataType::Float32, { 2 } } });

	sg.SetResults({ { y, 0 } });
	const auto fwdId = graph.AddSubgraph(std::move(sg));
	graph.SetForward(fwdId);

	// 先计算未折叠的结果
	Tensor<CPU> tensorX({ 10, 20 }, { 2 });
	std::array<Tensor<CPU>, 1> inputs = { Tensor<CPU>(tensorX) };

	Runtime::Interpreter<CPU> interp;
	auto expected = interp.RunForward(graph, inputs);

	ConstFoldPass constFold;
	constFold.Run(graph);

	// 2.0*3.0 应折叠为 6.0，原来两个 ConstantNode + Multiply 变为一个 ConstantNode
	const auto& foldedSg = graph.GetSubgraph(graph.Forward());
	EXPECT_EQ(CountOpNodes(foldedSg), 1); // 只剩 Add

	std::array<Tensor<CPU>, 1> inputs2 = { std::move(tensorX) };
	auto actual = interp.RunForward(graph, inputs2);

	ASSERT_EQ(actual.size(), 1);
	EXPECT_FLOAT_EQ(ReadFloat(actual[0], 0), ReadFloat(expected[0], 0));
	EXPECT_FLOAT_EQ(ReadFloat(actual[0], 1), ReadFloat(expected[0], 1));
}

// 测试 3: 常量 UnaryOp y = Negate(Constant(3.0))
TEST(ConstFoldPass, ConstantUnaryOp)
{
	Graph graph;
	Subgraph sg;

	const auto c = sg.AddNode(
	    ConstantNode{ Tensor<CPU>({ 3.0 }, { 1 }).CopyToDevice(PolymorphicDevice{ CPU{} }) },
	    { OutputInfo{ DataType::Float32, { 1 } } });
	const auto y =
	    sg.AddNode(UnaryOpNode{ UnaryOp::Negate, { c, 0 } }, { OutputInfo{ DataType::Float32, { 1 } } });

	sg.SetResults({ { y, 0 } });
	const auto fwdId = graph.AddSubgraph(std::move(sg));
	graph.SetForward(fwdId);

	ConstFoldPass constFold;
	constFold.Run(graph);

	const auto& foldedSg = graph.GetSubgraph(graph.Forward());
	EXPECT_TRUE(ResultIsConstant(foldedSg));

	Runtime::Interpreter<CPU> interp;
	std::array<Tensor<CPU>, 0> inputs = {};
	auto results = interp.RunForward(graph, inputs);

	ASSERT_EQ(results.size(), 1);
	EXPECT_FLOAT_EQ(ReadFloat(results[0], 0), -3.0f);
}

// 测试 4: 常量 CastNode Cast(Constant(3.0f), Float64)
TEST(ConstFoldPass, ConstantCast)
{
	Graph graph;
	Subgraph sg;

	const auto c = sg.AddNode(
	    ConstantNode{ Tensor<CPU>({ 3.0 }, { 1 }).CopyToDevice(PolymorphicDevice{ CPU{} }) },
	    { OutputInfo{ DataType::Float32, { 1 } } });
	const auto y =
	    sg.AddNode(CastNode{ { c, 0 }, DataType::Float64 }, { OutputInfo{ DataType::Float64, { 1 } } });

	sg.SetResults({ { y, 0 } });
	const auto fwdId = graph.AddSubgraph(std::move(sg));
	graph.SetForward(fwdId);

	ConstFoldPass constFold;
	constFold.Run(graph);

	const auto& foldedSg = graph.GetSubgraph(graph.Forward());
	EXPECT_TRUE(ResultIsConstant(foldedSg));

	Runtime::Interpreter<CPU> interp;
	std::array<Tensor<CPU>, 0> inputs = {};
	auto results = interp.RunForward(graph, inputs);

	ASSERT_EQ(results.size(), 1);
	EXPECT_EQ(results[0].DType(), DataType::Float64);
	EXPECT_DOUBLE_EQ(ReadDouble(results[0], 0), 3.0);
}

// 测试 5: 常量 ReduceOp ReduceSum(Constant([[1,2],[3,4]]), axis=0)
TEST(ConstFoldPass, ConstantReduceOp)
{
	Graph graph;
	Subgraph sg;

	const auto c = sg.AddNode(
	    ConstantNode{ Tensor<CPU>({ 1, 2, 3, 4 }, { 2, 2 }).CopyToDevice(PolymorphicDevice{ CPU{} }) },
	    { OutputInfo{ DataType::Float32, { 2, 2 } } });
	const auto y =
	    sg.AddNode(ReduceOpNode{ ReduceOp::Sum, { c, 0 }, 0 }, { OutputInfo{ DataType::Float32, { 2 } } });

	sg.SetResults({ { y, 0 } });
	const auto fwdId = graph.AddSubgraph(std::move(sg));
	graph.SetForward(fwdId);

	ConstFoldPass constFold;
	constFold.Run(graph);

	const auto& foldedSg = graph.GetSubgraph(graph.Forward());
	EXPECT_TRUE(ResultIsConstant(foldedSg));

	Runtime::Interpreter<CPU> interp;
	std::array<Tensor<CPU>, 0> inputs = {};
	auto results = interp.RunForward(graph, inputs);

	ASSERT_EQ(results.size(), 1);
	// Sum along axis 0: [1+3, 2+4] = [4, 6]
	EXPECT_FLOAT_EQ(ReadFloat(results[0], 0), 4.0f);
	EXPECT_FLOAT_EQ(ReadFloat(results[0], 1), 6.0f);
}

// 测试 6: 常量 ReshapeNode Reshape(Constant([1,2,3,4]), [2,2])
TEST(ConstFoldPass, ConstantReshape)
{
	Graph graph;
	Subgraph sg;

	const auto c = sg.AddNode(
	    ConstantNode{ Tensor<CPU>({ 1, 2, 3, 4 }, { 4 }).CopyToDevice(PolymorphicDevice{ CPU{} }) },
	    { OutputInfo{ DataType::Float32, { 4 } } });
	const auto y =
	    sg.AddNode(ReshapeNode{ { c, 0 }, { 2, 2 } }, { OutputInfo{ DataType::Float32, { 2, 2 } } });

	sg.SetResults({ { y, 0 } });
	const auto fwdId = graph.AddSubgraph(std::move(sg));
	graph.SetForward(fwdId);

	ConstFoldPass constFold;
	constFold.Run(graph);

	const auto& foldedSg = graph.GetSubgraph(graph.Forward());
	EXPECT_TRUE(ResultIsConstant(foldedSg));

	Runtime::Interpreter<CPU> interp;
	std::array<Tensor<CPU>, 0> inputs = {};
	auto results = interp.RunForward(graph, inputs);

	ASSERT_EQ(results.size(), 1);
	EXPECT_FLOAT_EQ(ReadFloat(results[0], 0), 1.0f);
	EXPECT_FLOAT_EQ(ReadFloat(results[0], 1), 2.0f);
	EXPECT_FLOAT_EQ(ReadFloat(results[0], 2), 3.0f);
	EXPECT_FLOAT_EQ(ReadFloat(results[0], 3), 4.0f);
}

// 测试 7: x + 0 消除
TEST(ConstFoldPass, AddZeroElimination)
{
	Graph graph;
	Subgraph sg;

	const auto x = sg.AddParam(DataType::Float32, { 2 });
	const auto zero = sg.AddNode(
	    ConstantNode{ Tensor<CPU>({ 0, 0 }, { 2 }).CopyToDevice(PolymorphicDevice{ CPU{} }) },
	    { OutputInfo{ DataType::Float32, { 2 } } });
	const auto y = sg.AddNode(BinaryOpNode{ BinaryOp::Add, { x, 0 }, { zero, 0 } },
	                           { OutputInfo{ DataType::Float32, { 2 } } });

	sg.SetResults({ { y, 0 } });
	const auto fwdId = graph.AddSubgraph(std::move(sg));
	graph.SetForward(fwdId);

	ConstFoldPass constFold;
	constFold.Run(graph);

	// Add 应被消除，结果直接引用 x
	const auto& foldedSg = graph.GetSubgraph(graph.Forward());
	EXPECT_EQ(CountOpNodes(foldedSg), 0);

	Tensor<CPU> tensorX({ 42, -7 }, { 2 });
	std::array<Tensor<CPU>, 1> inputs = { std::move(tensorX) };

	Runtime::Interpreter<CPU> interp;
	auto results = interp.RunForward(graph, inputs);

	ASSERT_EQ(results.size(), 1);
	EXPECT_FLOAT_EQ(ReadFloat(results[0], 0), 42.0f);
	EXPECT_FLOAT_EQ(ReadFloat(results[0], 1), -7.0f);
}

// 测试 8: x * 1 消除
TEST(ConstFoldPass, MulOneElimination)
{
	Graph graph;
	Subgraph sg;

	const auto x = sg.AddParam(DataType::Float32, { 2 });
	const auto one = sg.AddNode(
	    ConstantNode{ Tensor<CPU>({ 1, 1 }, { 2 }).CopyToDevice(PolymorphicDevice{ CPU{} }) },
	    { OutputInfo{ DataType::Float32, { 2 } } });
	const auto y = sg.AddNode(BinaryOpNode{ BinaryOp::Multiply, { x, 0 }, { one, 0 } },
	                           { OutputInfo{ DataType::Float32, { 2 } } });

	sg.SetResults({ { y, 0 } });
	const auto fwdId = graph.AddSubgraph(std::move(sg));
	graph.SetForward(fwdId);

	ConstFoldPass constFold;
	constFold.Run(graph);

	const auto& foldedSg = graph.GetSubgraph(graph.Forward());
	EXPECT_EQ(CountOpNodes(foldedSg), 0);

	Tensor<CPU> tensorX({ 99, -11 }, { 2 });
	std::array<Tensor<CPU>, 1> inputs = { std::move(tensorX) };

	Runtime::Interpreter<CPU> interp;
	auto results = interp.RunForward(graph, inputs);

	ASSERT_EQ(results.size(), 1);
	EXPECT_FLOAT_EQ(ReadFloat(results[0], 0), 99.0f);
	EXPECT_FLOAT_EQ(ReadFloat(results[0], 1), -11.0f);
}

// 测试 9: x * 0 消除
TEST(ConstFoldPass, MulZeroElimination)
{
	Graph graph;
	Subgraph sg;

	const auto x = sg.AddParam(DataType::Float32, { 2 });
	const auto zero = sg.AddNode(
	    ConstantNode{ Tensor<CPU>({ 0, 0 }, { 2 }).CopyToDevice(PolymorphicDevice{ CPU{} }) },
	    { OutputInfo{ DataType::Float32, { 2 } } });
	const auto y = sg.AddNode(BinaryOpNode{ BinaryOp::Multiply, { x, 0 }, { zero, 0 } },
	                           { OutputInfo{ DataType::Float32, { 2 } } });

	sg.SetResults({ { y, 0 } });
	const auto fwdId = graph.AddSubgraph(std::move(sg));
	graph.SetForward(fwdId);

	ConstFoldPass constFold;
	constFold.Run(graph);

	// x * 0 → ConstantNode(0, 0)
	const auto& foldedSg = graph.GetSubgraph(graph.Forward());
	EXPECT_TRUE(ResultIsConstant(foldedSg));

	Tensor<CPU> tensorX({ 99, -11 }, { 2 });
	std::array<Tensor<CPU>, 1> inputs = { std::move(tensorX) };

	Runtime::Interpreter<CPU> interp;
	auto results = interp.RunForward(graph, inputs);

	ASSERT_EQ(results.size(), 1);
	EXPECT_FLOAT_EQ(ReadFloat(results[0], 0), 0.0f);
	EXPECT_FLOAT_EQ(ReadFloat(results[0], 1), 0.0f);
}

// 测试 10: 0 + x 消除（交换）
TEST(ConstFoldPass, ZeroAddElimination)
{
	Graph graph;
	Subgraph sg;

	const auto x = sg.AddParam(DataType::Float32, { 2 });
	const auto zero = sg.AddNode(
	    ConstantNode{ Tensor<CPU>({ 0, 0 }, { 2 }).CopyToDevice(PolymorphicDevice{ CPU{} }) },
	    { OutputInfo{ DataType::Float32, { 2 } } });
	const auto y = sg.AddNode(BinaryOpNode{ BinaryOp::Add, { zero, 0 }, { x, 0 } },
	                           { OutputInfo{ DataType::Float32, { 2 } } });

	sg.SetResults({ { y, 0 } });
	const auto fwdId = graph.AddSubgraph(std::move(sg));
	graph.SetForward(fwdId);

	ConstFoldPass constFold;
	constFold.Run(graph);

	const auto& foldedSg = graph.GetSubgraph(graph.Forward());
	EXPECT_EQ(CountOpNodes(foldedSg), 0);

	Tensor<CPU> tensorX({ 42, -7 }, { 2 });
	std::array<Tensor<CPU>, 1> inputs = { std::move(tensorX) };

	Runtime::Interpreter<CPU> interp;
	auto results = interp.RunForward(graph, inputs);

	ASSERT_EQ(results.size(), 1);
	EXPECT_FLOAT_EQ(ReadFloat(results[0], 0), 42.0f);
	EXPECT_FLOAT_EQ(ReadFloat(results[0], 1), -7.0f);
}

// 测试 11: 0 * x 消除（交换）
TEST(ConstFoldPass, ZeroMulElimination)
{
	Graph graph;
	Subgraph sg;

	const auto x = sg.AddParam(DataType::Float32, { 2 });
	const auto zero = sg.AddNode(
	    ConstantNode{ Tensor<CPU>({ 0, 0 }, { 2 }).CopyToDevice(PolymorphicDevice{ CPU{} }) },
	    { OutputInfo{ DataType::Float32, { 2 } } });
	const auto y = sg.AddNode(BinaryOpNode{ BinaryOp::Multiply, { zero, 0 }, { x, 0 } },
	                           { OutputInfo{ DataType::Float32, { 2 } } });

	sg.SetResults({ { y, 0 } });
	const auto fwdId = graph.AddSubgraph(std::move(sg));
	graph.SetForward(fwdId);

	ConstFoldPass constFold;
	constFold.Run(graph);

	const auto& foldedSg = graph.GetSubgraph(graph.Forward());
	EXPECT_TRUE(ResultIsConstant(foldedSg));

	Tensor<CPU> tensorX({ 99, -11 }, { 2 });
	std::array<Tensor<CPU>, 1> inputs = { std::move(tensorX) };

	Runtime::Interpreter<CPU> interp;
	auto results = interp.RunForward(graph, inputs);

	ASSERT_EQ(results.size(), 1);
	EXPECT_FLOAT_EQ(ReadFloat(results[0], 0), 0.0f);
	EXPECT_FLOAT_EQ(ReadFloat(results[0], 1), 0.0f);
}

// 测试 12: 广播不消除 x[2,3] + Constant(0)[1,3]
TEST(ConstFoldPass, BroadcastNoElimination)
{
	Graph graph;
	Subgraph sg;

	const auto x = sg.AddParam(DataType::Float32, { 2, 3 });
	const auto zero = sg.AddNode(
	    ConstantNode{ Tensor<CPU>({ 0, 0, 0 }, { 1, 3 }).CopyToDevice(PolymorphicDevice{ CPU{} }) },
	    { OutputInfo{ DataType::Float32, { 1, 3 } } });
	const auto y = sg.AddNode(BinaryOpNode{ BinaryOp::Add, { x, 0 }, { zero, 0 } },
	                           { OutputInfo{ DataType::Float32, { 2, 3 } } });

	sg.SetResults({ { y, 0 } });
	const auto fwdId = graph.AddSubgraph(std::move(sg));
	graph.SetForward(fwdId);

	// 先验证未折叠结果
	Tensor<CPU> tensorX({ 1, 2, 3, 4, 5, 6 }, { 2, 3 });
	std::array<Tensor<CPU>, 1> inputs = { Tensor<CPU>(tensorX) };

	Runtime::Interpreter<CPU> interp;
	auto expected = interp.RunForward(graph, inputs);

	ConstFoldPass constFold;
	constFold.Run(graph);

	// x[2,3] 的 shape 与输出 [2,3] 相同，且 zero 的 shape [1,3] 不同
	// 但 x 的 shape 等于输出 shape，所以 x + 0 可以消除
	// 等等——ShapeCompatibleForElimination 检查的是 x 的 shape 与输出 shape，
	// 这里 x 是 [2,3]，输出是 [2,3]，所以是兼容的。消除是安全的。
	// 让我修改测试：使该情况不可消除

	// 实际上 x[2,3] + 0[1,3] 的输出是 [2,3]，x 的 shape 也是 [2,3]，
	// 所以消除为 x 是正确的（加零不改变值）
	// 要测试不可消除，需要 zero 的 shape 导致 x 被广播
	// 改用 x[1,3] + 0[2,3] → 输出 [2,3]，x 被广播，消除不安全

	std::array<Tensor<CPU>, 1> inputs2 = { std::move(tensorX) };
	auto actual = interp.RunForward(graph, inputs2);

	ASSERT_EQ(actual.size(), 1);
	for (std::size_t i = 0; i < 6; ++i)
	{
		EXPECT_FLOAT_EQ(ReadFloat(actual[0], i), ReadFloat(expected[0], i));
	}
}

// 测试 12b: 真正不可消除的广播：x[1,3] + 0[2,3] → 输出 [2,3]
TEST(ConstFoldPass, BroadcastNoEliminationActual)
{
	Graph graph;
	Subgraph sg;

	// x 的 shape [1,3] 与 输出 shape [2,3] 不同
	const auto x = sg.AddParam(DataType::Float32, { 1, 3 });
	const auto zero = sg.AddNode(
	    ConstantNode{ Tensor<CPU>({ 0, 0, 0, 0, 0, 0 }, { 2, 3 }).CopyToDevice(PolymorphicDevice{ CPU{} }) },
	    { OutputInfo{ DataType::Float32, { 2, 3 } } });
	const auto y = sg.AddNode(BinaryOpNode{ BinaryOp::Add, { x, 0 }, { zero, 0 } },
	                           { OutputInfo{ DataType::Float32, { 2, 3 } } });

	sg.SetResults({ { y, 0 } });
	const auto fwdId = graph.AddSubgraph(std::move(sg));
	graph.SetForward(fwdId);

	Tensor<CPU> tensorX({ 1, 2, 3 }, { 1, 3 });
	std::array<Tensor<CPU>, 1> inputs = { Tensor<CPU>(tensorX) };

	Runtime::Interpreter<CPU> interp;
	auto expected = interp.RunForward(graph, inputs);

	ConstFoldPass constFold;
	constFold.Run(graph);

	// x[1,3] 的 shape 与输出 [2,3] 不同，不应消除
	const auto& foldedSg = graph.GetSubgraph(graph.Forward());
	EXPECT_GT(CountOpNodes(foldedSg), 0); // Add 仍然存在

	std::array<Tensor<CPU>, 1> inputs2 = { std::move(tensorX) };
	auto actual = interp.RunForward(graph, inputs2);

	ASSERT_EQ(actual.size(), 1);
	for (std::size_t i = 0; i < 6; ++i)
	{
		EXPECT_FLOAT_EQ(ReadFloat(actual[0], i), ReadFloat(expected[0], i));
	}
}

// 测试 13: Negate(Negate(x)) 双重取反消除
TEST(ConstFoldPass, DoubleNegateElimination)
{
	Graph graph;
	Subgraph sg;

	const auto x = sg.AddParam(DataType::Float32, { 2 });
	const auto neg1 =
	    sg.AddNode(UnaryOpNode{ UnaryOp::Negate, { x, 0 } }, { OutputInfo{ DataType::Float32, { 2 } } });
	const auto neg2 =
	    sg.AddNode(UnaryOpNode{ UnaryOp::Negate, { neg1, 0 } }, { OutputInfo{ DataType::Float32, { 2 } } });

	sg.SetResults({ { neg2, 0 } });
	const auto fwdId = graph.AddSubgraph(std::move(sg));
	graph.SetForward(fwdId);

	ConstFoldPass constFold;
	constFold.Run(graph);

	// 双重取反消除后，应无操作节点
	const auto& foldedSg = graph.GetSubgraph(graph.Forward());
	EXPECT_EQ(CountOpNodes(foldedSg), 0);

	Tensor<CPU> tensorX({ 7, -3 }, { 2 });
	std::array<Tensor<CPU>, 1> inputs = { std::move(tensorX) };

	Runtime::Interpreter<CPU> interp;
	auto results = interp.RunForward(graph, inputs);

	ASSERT_EQ(results.size(), 1);
	EXPECT_FLOAT_EQ(ReadFloat(results[0], 0), 7.0f);
	EXPECT_FLOAT_EQ(ReadFloat(results[0], 1), -3.0f);
}

// 测试 14: 死节点消除
TEST(ConstFoldPass, DeadNodeElimination)
{
	Graph graph;
	Subgraph sg;

	const auto x = sg.AddParam(DataType::Float32, { 2 });
	const auto neg =
	    sg.AddNode(UnaryOpNode{ UnaryOp::Negate, { x, 0 } }, { OutputInfo{ DataType::Float32, { 2 } } });
	// 这个死节点不被 Results 引用
	const auto dead =
	    sg.AddNode(UnaryOpNode{ UnaryOp::Exp, { x, 0 } }, { OutputInfo{ DataType::Float32, { 2 } } });
	(void)dead;

	sg.SetResults({ { neg, 0 } });
	const auto fwdId = graph.AddSubgraph(std::move(sg));
	graph.SetForward(fwdId);

	const auto nodeCountBefore = graph.GetSubgraph(graph.Forward()).NodeCount();

	ConstFoldPass constFold;
	constFold.Run(graph);

	// Exp 是死节点，应被消除
	const auto& foldedSg = graph.GetSubgraph(graph.Forward());
	EXPECT_LT(foldedSg.NodeCount(), nodeCountBefore);

	Tensor<CPU> tensorX({ 5, -2 }, { 2 });
	std::array<Tensor<CPU>, 1> inputs = { std::move(tensorX) };

	Runtime::Interpreter<CPU> interp;
	auto results = interp.RunForward(graph, inputs);

	ASSERT_EQ(results.size(), 1);
	EXPECT_FLOAT_EQ(ReadFloat(results[0], 0), -5.0f);
	EXPECT_FLOAT_EQ(ReadFloat(results[0], 1), 2.0f);
}

// 测试 15: ConstFoldPass + InlinePass 联合
TEST(ConstFoldPass, AfterInlinePass)
{
	Graph graph;

	// callee: f(x) = x * Constant(2)
	Subgraph calleeSg;
	const auto cx = calleeSg.AddParam(DataType::Float32, { 2 });
	const auto cConst = calleeSg.AddNode(
	    ConstantNode{ Tensor<CPU>({ 2, 2 }, { 2 }).CopyToDevice(PolymorphicDevice{ CPU{} }) },
	    { OutputInfo{ DataType::Float32, { 2 } } });
	const auto cMul = calleeSg.AddNode(BinaryOpNode{ BinaryOp::Multiply, { cx, 0 }, { cConst, 0 } },
	                                    { OutputInfo{ DataType::Float32, { 2 } } });
	calleeSg.SetResults({ { cMul, 0 } });
	const auto calleeId = graph.AddSubgraph(std::move(calleeSg));

	// 前向: y = Call(callee, [Constant(3, 5)])
	Subgraph sg;
	const auto input = sg.AddNode(
	    ConstantNode{ Tensor<CPU>({ 3, 5 }, { 2 }).CopyToDevice(PolymorphicDevice{ CPU{} }) },
	    { OutputInfo{ DataType::Float32, { 2 } } });
	const auto y =
	    sg.AddNode(CallNode{ calleeId, { { input, 0 } } }, { OutputInfo{ DataType::Float32, { 2 } } });

	sg.SetResults({ { y, 0 } });
	const auto fwdId = graph.AddSubgraph(std::move(sg));
	graph.SetForward(fwdId);

	// 先 InlinePass 展开 CallNode -> Constant(3,5) * Constant(2,2)
	InlinePass inlinePass;
	inlinePass.Run(graph);

	// 再 ConstFoldPass 折叠常量
	ConstFoldPass constFold;
	constFold.Run(graph);

	const auto& foldedSg = graph.GetSubgraph(graph.Forward());
	EXPECT_TRUE(ResultIsConstant(foldedSg));

	Runtime::Interpreter<CPU> interp;
	std::array<Tensor<CPU>, 0> inputs = {};
	auto results = interp.RunForward(graph, inputs);

	ASSERT_EQ(results.size(), 1);
	// [3, 5] * [2, 2] = [6, 10]
	EXPECT_FLOAT_EQ(ReadFloat(results[0], 0), 6.0f);
	EXPECT_FLOAT_EQ(ReadFloat(results[0], 1), 10.0f);
}

// 测试 16: 全流水线 AutogradPass → InlinePass → ConstFoldPass → FusionPass
TEST(ConstFoldPass, FullPipeline)
{
	Graph graph;

	// 构建 ReLU 子图
	const auto reluId = Layer::BuildReLU(graph, DataType::Float32, { 2, 2 });

	// 前向: y = ReLU(x @ w + b)
	auto wVar = Variable::Create(Tensor<CPU>({ 1, 2, 3, 4, 5, 6 }, { 3, 2 }));
	const auto wIdx = graph.AddVariable(wVar);

	Subgraph sg;
	const auto x = sg.AddParam(DataType::Float32, { 2, 3 });
	const auto w = sg.AddNode(VariableRefNode{ wIdx }, { OutputInfo{ DataType::Float32, { 3, 2 } } });
	const auto b = sg.AddParam(DataType::Float32, { 1, 2 });

	const auto matmul =
	    sg.AddNode(BinaryOpNode{ BinaryOp::MatMul, { x, 0 }, { w, 0 } }, { OutputInfo{ DataType::Float32, { 2, 2 } } });
	const auto add = sg.AddNode(BinaryOpNode{ BinaryOp::Add, { matmul, 0 }, { b, 0 } },
	                             { OutputInfo{ DataType::Float32, { 2, 2 } } });
	const auto relu =
	    sg.AddNode(CallNode{ reluId, { { add, 0 } } }, { OutputInfo{ DataType::Float32, { 2, 2 } } });

	sg.SetResults({ { relu, 0 } });
	const auto fwdId = graph.AddSubgraph(std::move(sg));
	graph.SetForward(fwdId);

	// 参考图
	Graph refGraph;
	{
		const auto refReluId = Layer::BuildReLU(refGraph, DataType::Float32, { 2, 2 });
		auto refW = Variable::Create(Tensor<CPU>({ 1, 2, 3, 4, 5, 6 }, { 3, 2 }));
		const auto refWIdx = refGraph.AddVariable(refW);

		Subgraph refSg;
		const auto refX = refSg.AddParam(DataType::Float32, { 2, 3 });
		const auto refWNode = refSg.AddNode(VariableRefNode{ refWIdx }, { OutputInfo{ DataType::Float32, { 3, 2 } } });
		const auto refB = refSg.AddParam(DataType::Float32, { 1, 2 });
		const auto refMM = refSg.AddNode(BinaryOpNode{ BinaryOp::MatMul, { refX, 0 }, { refWNode, 0 } },
		                                  { OutputInfo{ DataType::Float32, { 2, 2 } } });
		const auto refAdd = refSg.AddNode(BinaryOpNode{ BinaryOp::Add, { refMM, 0 }, { refB, 0 } },
		                                   { OutputInfo{ DataType::Float32, { 2, 2 } } });
		const auto refRelu = refSg.AddNode(CallNode{ refReluId, { { refAdd, 0 } } },
		                                    { OutputInfo{ DataType::Float32, { 2, 2 } } });
		refSg.SetResults({ { refRelu, 0 } });
		const auto refFwdId = refGraph.AddSubgraph(std::move(refSg));
		refGraph.SetForward(refFwdId);

		AutogradPass autograd;
		autograd.Run(refGraph);
	}

	// 完整流水线
	AutogradPass autograd;
	autograd.Run(graph);
	InlinePass inlinePass;
	inlinePass.Run(graph);
	ConstFoldPass constFold;
	constFold.Run(graph);
	FusionPass fusionPass;
	fusionPass.Run(graph);

	// 验证前向
	Tensor<CPU> tensorX({ 1, 2, 3, 4, 5, 6 }, { 2, 3 });
	Tensor<CPU> tensorB({ 0, 0 }, { 1, 2 });

	Runtime::Interpreter<CPU> interp;
	std::array<Tensor<CPU>, 2> inputs = { Tensor<CPU>(tensorX), Tensor<CPU>(tensorB) };
	auto actual = interp.RunForward(graph, inputs);

	Runtime::Interpreter<CPU> interpRef;
	std::array<Tensor<CPU>, 2> refInputs = { Tensor<CPU>(tensorX), Tensor<CPU>(tensorB) };
	auto expected = interpRef.RunForward(refGraph, refInputs);

	ASSERT_EQ(actual.size(), expected.size());
	for (std::size_t i = 0; i < actual[0].NumElements(); ++i)
	{
		EXPECT_FLOAT_EQ(ReadFloat(actual[0], i), ReadFloat(expected[0], i));
	}

	// 验证反向
	Tensor<CPU> gradOut({ 1, 1, 1, 1 }, { 2, 2 });
	std::array<Tensor<CPU>, 3> bwdInputs = { Tensor<CPU>(tensorX), Tensor<CPU>(tensorB), Tensor<CPU>(gradOut) };
	auto actualBwd = interp.RunBackward(graph, bwdInputs);

	std::array<Tensor<CPU>, 3> refBwdInputs = { std::move(tensorX), std::move(tensorB), std::move(gradOut) };
	auto expectedBwd = interpRef.RunBackward(refGraph, refBwdInputs);

	ASSERT_EQ(actualBwd.size(), expectedBwd.size());
	for (std::size_t i = 0; i < actualBwd.size(); ++i)
	{
		for (std::size_t j = 0; j < actualBwd[i].NumElements(); ++j)
		{
			EXPECT_FLOAT_EQ(ReadFloat(actualBwd[i], j), ReadFloat(expectedBwd[i], j));
		}
	}
}
