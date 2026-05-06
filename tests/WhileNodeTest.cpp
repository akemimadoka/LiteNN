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

// 辅助函数: 读取 CPU Tensor 的标量 Int64
static std::int64_t ReadInt64(const Tensor<CPU>& t, std::size_t i = 0)
{
	return static_cast<const std::int64_t*>(t.RawData())[i];
}

// ============================================================
// 前向测试
// ============================================================

// 测试 1: 基础 WhileNode — x *= 2 while x < 100
TEST(WhileNode, Forward_Basic)
{
	Graph graph;

	// condBranch: (x) → x < 100
	Subgraph condSg;
	const auto cx = condSg.AddParam(DataType::Float32, { 1 });
	const auto threshold =
	    condSg.AddNode(ConstantNode{ Tensor<CPU>({ 100.0 }, { 1 }).CopyToDevice(PolymorphicDevice{ CPU{} }) },
	                   { OutputInfo{ DataType::Float32, { 1 } } });
	const auto cmp = condSg.AddNode(BinaryOpNode{ BinaryOp::Less, { cx, 0 }, { threshold, 0 } },
	                                { OutputInfo{ DataType::Bool, { 1 } } });
	condSg.SetResults({ { cmp, 0 } });
	const auto condId = graph.AddSubgraph(std::move(condSg));

	// bodyBranch: (x) → x * 2
	Subgraph bodySg;
	const auto bx = bodySg.AddParam(DataType::Float32, { 1 });
	const auto two =
	    bodySg.AddNode(ConstantNode{ Tensor<CPU>({ 2.0 }, { 1 }).CopyToDevice(PolymorphicDevice{ CPU{} }) },
	                   { OutputInfo{ DataType::Float32, { 1 } } });
	const auto doubled = bodySg.AddNode(BinaryOpNode{ BinaryOp::Multiply, { bx, 0 }, { two, 0 } },
	                                    { OutputInfo{ DataType::Float32, { 1 } } });
	bodySg.SetResults({ { doubled, 0 } });
	const auto bodyId = graph.AddSubgraph(std::move(bodySg));

	// forward: y = While(cond, body, [x])
	Subgraph fwdSg;
	const auto x = fwdSg.AddParam(DataType::Float32, { 1 });
	const auto whileNode =
	    fwdSg.AddNode(WhileNode{ condId, bodyId, { { x, 0 } } }, { OutputInfo{ DataType::Float32, { 1 } } });
	fwdSg.SetResults({ { whileNode, 0 } });
	graph.SetForward(graph.AddSubgraph(std::move(fwdSg)));

	// x=1 → 2 → 4 → 8 → 16 → 32 → 64 → 128 (stop, 128 >= 100)
	Tensor<CPU> tensorX({ 1.0 }, { 1 });
	std::array<Tensor<CPU>, 1> inputs = { std::move(tensorX) };

	Runtime::Interpreter<CPU> interp;
	auto results = interp.RunForward(graph, inputs);

	ASSERT_EQ(results.size(), 1);
	EXPECT_FLOAT_EQ(ReadFloat(results[0], 0), 128.0f);
}

// 测试 2: 多 carry 值 — 斐波那契 (a, b) → (b, a+b) while a < 50
TEST(WhileNode, Forward_MultiCarry)
{
	Graph graph;

	// condBranch: (a, b) → a < 50
	Subgraph condSg;
	const auto ca = condSg.AddParam(DataType::Float32, { 1 });
	condSg.AddParam(DataType::Float32, { 1 }); // b（未使用但需要参数匹配）
	const auto limit =
	    condSg.AddNode(ConstantNode{ Tensor<CPU>({ 50.0 }, { 1 }).CopyToDevice(PolymorphicDevice{ CPU{} }) },
	                   { OutputInfo{ DataType::Float32, { 1 } } });
	const auto cmp = condSg.AddNode(BinaryOpNode{ BinaryOp::Less, { ca, 0 }, { limit, 0 } },
	                                { OutputInfo{ DataType::Bool, { 1 } } });
	condSg.SetResults({ { cmp, 0 } });
	const auto condId = graph.AddSubgraph(std::move(condSg));

	// bodyBranch: (a, b) → (b, a + b)
	Subgraph bodySg;
	const auto ba = bodySg.AddParam(DataType::Float32, { 1 });
	const auto bb = bodySg.AddParam(DataType::Float32, { 1 });
	const auto sum = bodySg.AddNode(BinaryOpNode{ BinaryOp::Add, { ba, 0 }, { bb, 0 } },
	                                { OutputInfo{ DataType::Float32, { 1 } } });
	bodySg.SetResults({ { bb, 0 }, { sum, 0 } }); // (b, a+b)
	const auto bodyId = graph.AddSubgraph(std::move(bodySg));

	// forward
	Subgraph fwdSg;
	const auto a = fwdSg.AddParam(DataType::Float32, { 1 });
	const auto b = fwdSg.AddParam(DataType::Float32, { 1 });
	const auto whileNode = fwdSg.AddNode(WhileNode{ condId, bodyId, { { a, 0 }, { b, 0 } } },
	                                     { OutputInfo{ DataType::Float32, { 1 } }, OutputInfo{ DataType::Float32, { 1 } } });
	fwdSg.SetResults({ { whileNode, 0 }, { whileNode, 1 } });
	graph.SetForward(graph.AddSubgraph(std::move(fwdSg)));

	// fib: (1,1) → (1,2) → (2,3) → (3,5) → (5,8) → (8,13) → (13,21) → (21,34) → (34,55) → (55,89)
	// stop when a >= 50, so last iteration has a=34 → body produces (55, 89), then cond(55) → false
	// result: a=55, b=89
	Tensor<CPU> tensorA({ 1.0 }, { 1 });
	Tensor<CPU> tensorB({ 1.0 }, { 1 });
	std::array<Tensor<CPU>, 2> inputs = { std::move(tensorA), std::move(tensorB) };

	Runtime::Interpreter<CPU> interp;
	auto results = interp.RunForward(graph, inputs);

	ASSERT_EQ(results.size(), 2);
	EXPECT_FLOAT_EQ(ReadFloat(results[0], 0), 55.0f);
	EXPECT_FLOAT_EQ(ReadFloat(results[1], 0), 89.0f);
}

// 测试 3: 零迭代 — 初始条件为 false
TEST(WhileNode, Forward_ZeroIteration)
{
	Graph graph;

	// condBranch: (x) → x < 0（永远 false，因为输入 > 0）
	Subgraph condSg;
	const auto cx = condSg.AddParam(DataType::Float32, { 1 });
	const auto zero =
	    condSg.AddNode(ConstantNode{ Tensor<CPU>({ 0.0 }, { 1 }).CopyToDevice(PolymorphicDevice{ CPU{} }) },
	                   { OutputInfo{ DataType::Float32, { 1 } } });
	const auto cmp = condSg.AddNode(BinaryOpNode{ BinaryOp::Less, { cx, 0 }, { zero, 0 } },
	                                { OutputInfo{ DataType::Bool, { 1 } } });
	condSg.SetResults({ { cmp, 0 } });
	const auto condId = graph.AddSubgraph(std::move(condSg));

	// bodyBranch: (x) → x * 2（不该执行）
	Subgraph bodySg;
	const auto bx = bodySg.AddParam(DataType::Float32, { 1 });
	const auto two =
	    bodySg.AddNode(ConstantNode{ Tensor<CPU>({ 2.0 }, { 1 }).CopyToDevice(PolymorphicDevice{ CPU{} }) },
	                   { OutputInfo{ DataType::Float32, { 1 } } });
	const auto doubled = bodySg.AddNode(BinaryOpNode{ BinaryOp::Multiply, { bx, 0 }, { two, 0 } },
	                                    { OutputInfo{ DataType::Float32, { 1 } } });
	bodySg.SetResults({ { doubled, 0 } });
	const auto bodyId = graph.AddSubgraph(std::move(bodySg));

	// forward
	Subgraph fwdSg;
	const auto x = fwdSg.AddParam(DataType::Float32, { 1 });
	const auto whileNode =
	    fwdSg.AddNode(WhileNode{ condId, bodyId, { { x, 0 } } }, { OutputInfo{ DataType::Float32, { 1 } } });
	fwdSg.SetResults({ { whileNode, 0 } });
	graph.SetForward(graph.AddSubgraph(std::move(fwdSg)));

	Tensor<CPU> tensorX({ 42.0 }, { 1 });
	std::array<Tensor<CPU>, 1> inputs = { std::move(tensorX) };

	Runtime::Interpreter<CPU> interp;
	auto results = interp.RunForward(graph, inputs);

	ASSERT_EQ(results.size(), 1);
	EXPECT_FLOAT_EQ(ReadFloat(results[0], 0), 42.0f); // 未修改
}

// ============================================================
// 自动微分测试
// ============================================================

// 辅助：构建 x * 2 while x < 100 图（bodyBranch 可微分）
static void BuildDoubleWhileGraph(Graph& graph, SubgraphId& condId, SubgraphId& bodyId)
{
	// condBranch: (x) → x < 100
	Subgraph condSg;
	const auto cx = condSg.AddParam(DataType::Float32, { 1 });
	const auto threshold =
	    condSg.AddNode(ConstantNode{ Tensor<CPU>({ 100.0f }, { 1 }).CopyToDevice(PolymorphicDevice{ CPU{} }) },
	                   { OutputInfo{ DataType::Float32, { 1 } } });
	const auto cmp = condSg.AddNode(BinaryOpNode{ BinaryOp::Less, { cx, 0 }, { threshold, 0 } },
	                                { OutputInfo{ DataType::Bool, { 1 } } });
	condSg.SetResults({ { cmp, 0 } });
	condId = graph.AddSubgraph(std::move(condSg));

	// bodyBranch: (x) → x * 2
	Subgraph bodySg;
	const auto bx = bodySg.AddParam(DataType::Float32, { 1 });
	const auto two =
	    bodySg.AddNode(ConstantNode{ Tensor<CPU>({ 2.0f }, { 1 }).CopyToDevice(PolymorphicDevice{ CPU{} }) },
	                   { OutputInfo{ DataType::Float32, { 1 } } });
	const auto doubled = bodySg.AddNode(BinaryOpNode{ BinaryOp::Multiply, { bx, 0 }, { two, 0 } },
	                                    { OutputInfo{ DataType::Float32, { 1 } } });
	bodySg.SetResults({ { doubled, 0 } });
	bodyId = graph.AddSubgraph(std::move(bodySg));
}

// 测试 4: WhileNode 梯度 — x *= 2 while x < 100，初始 x=1 → 7 次迭代，grad = 2^7 = 128
TEST(WhileNode, Autograd_Simple)
{
	Graph graph;
	SubgraphId condId, bodyId;
	BuildDoubleWhileGraph(graph, condId, bodyId);

	Subgraph fwdSg;
	const auto x = fwdSg.AddParam(DataType::Float32, { 1 });
	const auto whileNode =
	    fwdSg.AddNode(WhileNode{ condId, bodyId, { { x, 0 } } }, { OutputInfo{ DataType::Float32, { 1 } } });
	fwdSg.SetResults({ { whileNode, 0 } });
	graph.SetForward(graph.AddSubgraph(std::move(fwdSg)));

	AutogradPass autograd;
	autograd.Run(graph);

	Tensor<CPU> tensorX({ 1.0f }, { 1 });
	Tensor<CPU> gradOut({ 1.0f }, { 1 });

	Runtime::Interpreter<CPU> interp;
	std::array<Tensor<CPU>, 1> fwdInputs = { Tensor<CPU>(tensorX) };
	interp.RunForward(graph, fwdInputs);

	std::array<Tensor<CPU>, 2> bwdInputs = { std::move(tensorX), std::move(gradOut) };
	auto bwdResults = interp.RunBackward(graph, bwdInputs);

	// d_x = 2^7 = 128（7 次迭代，每次梯度乘以 2）
	ASSERT_EQ(bwdResults.size(), 1);
	EXPECT_FLOAT_EQ(ReadFloat(bwdResults[0], 0), 128.0f);
}

// 测试 5: WhileNode 零迭代梯度 — 条件初始为 false，输出 = 输入，梯度 = 1
TEST(WhileNode, Autograd_ZeroIteration)
{
	Graph graph;

	// condBranch: (x) → x < 0（对正数永远 false）
	Subgraph condSg;
	const auto cx = condSg.AddParam(DataType::Float32, { 1 });
	const auto zero =
	    condSg.AddNode(ConstantNode{ Tensor<CPU>({ 0.0f }, { 1 }).CopyToDevice(PolymorphicDevice{ CPU{} }) },
	                   { OutputInfo{ DataType::Float32, { 1 } } });
	const auto cmp = condSg.AddNode(BinaryOpNode{ BinaryOp::Less, { cx, 0 }, { zero, 0 } },
	                                { OutputInfo{ DataType::Bool, { 1 } } });
	condSg.SetResults({ { cmp, 0 } });
	const auto condId = graph.AddSubgraph(std::move(condSg));

	// bodyBranch: (x) → x * 2（不会执行）
	Subgraph bodySg;
	const auto bx = bodySg.AddParam(DataType::Float32, { 1 });
	const auto two =
	    bodySg.AddNode(ConstantNode{ Tensor<CPU>({ 2.0f }, { 1 }).CopyToDevice(PolymorphicDevice{ CPU{} }) },
	                   { OutputInfo{ DataType::Float32, { 1 } } });
	const auto doubled = bodySg.AddNode(BinaryOpNode{ BinaryOp::Multiply, { bx, 0 }, { two, 0 } },
	                                    { OutputInfo{ DataType::Float32, { 1 } } });
	bodySg.SetResults({ { doubled, 0 } });
	const auto bodyId = graph.AddSubgraph(std::move(bodySg));

	Subgraph fwdSg;
	const auto x = fwdSg.AddParam(DataType::Float32, { 1 });
	const auto whileNode =
	    fwdSg.AddNode(WhileNode{ condId, bodyId, { { x, 0 } } }, { OutputInfo{ DataType::Float32, { 1 } } });
	fwdSg.SetResults({ { whileNode, 0 } });
	graph.SetForward(graph.AddSubgraph(std::move(fwdSg)));

	AutogradPass autograd;
	autograd.Run(graph);

	Tensor<CPU> tensorX({ 42.0f }, { 1 });
	Tensor<CPU> gradOut({ 1.0f }, { 1 });

	Runtime::Interpreter<CPU> interp;
	std::array<Tensor<CPU>, 1> fwdInputs = { Tensor<CPU>(tensorX) };
	interp.RunForward(graph, fwdInputs);

	std::array<Tensor<CPU>, 2> bwdInputs = { std::move(tensorX), std::move(gradOut) };
	auto bwdResults = interp.RunBackward(graph, bwdInputs);

	// 零迭代 → 恒等映射，梯度 = 1
	ASSERT_EQ(bwdResults.size(), 1);
	EXPECT_FLOAT_EQ(ReadFloat(bwdResults[0], 0), 1.0f);
}

// 测试 6: WhileNode 梯度 — 使用加法（x = x + c while x < 100），grad = 1（加法不累积）
TEST(WhileNode, Autograd_Addition)
{
	Graph graph;

	// condBranch: (x) → x < 100
	Subgraph condSg;
	const auto cx = condSg.AddParam(DataType::Float32, { 1 });
	const auto threshold =
	    condSg.AddNode(ConstantNode{ Tensor<CPU>({ 100.0f }, { 1 }).CopyToDevice(PolymorphicDevice{ CPU{} }) },
	                   { OutputInfo{ DataType::Float32, { 1 } } });
	const auto cmp = condSg.AddNode(BinaryOpNode{ BinaryOp::Less, { cx, 0 }, { threshold, 0 } },
	                                { OutputInfo{ DataType::Bool, { 1 } } });
	condSg.SetResults({ { cmp, 0 } });
	const auto condId = graph.AddSubgraph(std::move(condSg));

	// bodyBranch: (x) → x + 10
	Subgraph bodySg;
	const auto bx = bodySg.AddParam(DataType::Float32, { 1 });
	const auto ten =
	    bodySg.AddNode(ConstantNode{ Tensor<CPU>({ 10.0f }, { 1 }).CopyToDevice(PolymorphicDevice{ CPU{} }) },
	                   { OutputInfo{ DataType::Float32, { 1 } } });
	const auto added = bodySg.AddNode(BinaryOpNode{ BinaryOp::Add, { bx, 0 }, { ten, 0 } },
	                                  { OutputInfo{ DataType::Float32, { 1 } } });
	bodySg.SetResults({ { added, 0 } });
	const auto bodyId = graph.AddSubgraph(std::move(bodySg));

	Subgraph fwdSg;
	const auto x = fwdSg.AddParam(DataType::Float32, { 1 });
	const auto whileNode =
	    fwdSg.AddNode(WhileNode{ condId, bodyId, { { x, 0 } } }, { OutputInfo{ DataType::Float32, { 1 } } });
	fwdSg.SetResults({ { whileNode, 0 } });
	graph.SetForward(graph.AddSubgraph(std::move(fwdSg)));

	AutogradPass autograd;
	autograd.Run(graph);

	// x=5 → 15 → 25 → 35 → 45 → 55 → 65 → 75 → 85 → 95 → 105（10 次迭代）
	Tensor<CPU> tensorX({ 5.0f }, { 1 });
	Tensor<CPU> gradOut({ 1.0f }, { 1 });

	Runtime::Interpreter<CPU> interp;
	std::array<Tensor<CPU>, 1> fwdInputs = { Tensor<CPU>(tensorX) };
	auto fwdResults = interp.RunForward(graph, fwdInputs);
	EXPECT_FLOAT_EQ(ReadFloat(fwdResults[0], 0), 105.0f);

	std::array<Tensor<CPU>, 2> bwdInputs = { std::move(tensorX), std::move(gradOut) };
	auto bwdResults = interp.RunBackward(graph, bwdInputs);

	// 加法的梯度传播：d_x = 1（加法不改变梯度）
	ASSERT_EQ(bwdResults.size(), 1);
	EXPECT_FLOAT_EQ(ReadFloat(bwdResults[0], 0), 1.0f);
}
