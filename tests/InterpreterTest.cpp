#include <gtest/gtest.h>

#include <LiteNN.h>
#include <LiteNN/Layer/ReLU.h>
#include <LiteNN/Pass/AutogradPass.h>
#include <LiteNN/Runtime/Interpreter.h>

using namespace LiteNN;

// 辅助函数: 读取 CPU Tensor 的第 i 个 float 元素
float ReadFloat(const Tensor<CPU>& t, std::size_t i)
{
	return static_cast<const float*>(t.RawData())[i];
}

// 测试 1: 简单二元操作 y = a + b
TEST(Interpreter, Add)
{
	Graph graph;
	Subgraph sg;

	// 两个输入参数: a[2,2], b[2,2]
	const auto a = sg.AddParam(DataType::Float32, { 2, 2 });
	const auto b = sg.AddParam(DataType::Float32, { 2, 2 });

	// y = a + b
	const auto y =
	    sg.AddNode(BinaryOpNode{ BinaryOp::Add, { a, 0 }, { b, 0 } }, { OutputInfo{ DataType::Float32, { 2, 2 } } });
	sg.SetResults({ { y, 0 } });

	const auto fwdId = graph.AddSubgraph(std::move(sg));
	graph.SetForward(fwdId);

	// 执行
	Tensor<CPU> tensorA({ 1, 2, 3, 4 }, { 2, 2 });
	Tensor<CPU> tensorB({ 10, 20, 30, 40 }, { 2, 2 });
	std::array<Tensor<CPU>, 2> inputs = { std::move(tensorA), std::move(tensorB) };

	Runtime::Interpreter<CPU> interp;
	auto results = interp.RunForward(graph, inputs);

	ASSERT_EQ(results.size(), 1);
	EXPECT_FLOAT_EQ(ReadFloat(results[0], 0), 11);
	EXPECT_FLOAT_EQ(ReadFloat(results[0], 1), 22);
	EXPECT_FLOAT_EQ(ReadFloat(results[0], 2), 33);
	EXPECT_FLOAT_EQ(ReadFloat(results[0], 3), 44);
}

// 测试 2: MatMul y = x @ w + bias (使用 Variable 作为权重)
TEST(Interpreter, MatMulWithVariable)
{
	Graph graph;

	// Variable: w[3,2], bias[1,2]
	auto w = Variable::Create(Tensor<CPU>({ 1, 2, 3, 4, 5, 6 }, { 3, 2 }));
	auto bias = Variable::Create(Tensor<CPU>({ 100, 200 }, { 1, 2 }));
	const auto wIdx = graph.AddVariable(std::move(w));
	const auto biasIdx = graph.AddVariable(std::move(bias));

	Subgraph sg;
	// 输入: x[2,3]
	const auto x = sg.AddParam(DataType::Float32, { 2, 3 });

	// wRef, biasRef
	const auto wRef = sg.AddNode(VariableRefNode{ wIdx }, { OutputInfo{ DataType::Float32, { 3, 2 } } });
	const auto biasRef = sg.AddNode(VariableRefNode{ biasIdx }, { OutputInfo{ DataType::Float32, { 1, 2 } } });

	// matmul = x @ w -> [2,2]
	const auto matmul = sg.AddNode(BinaryOpNode{ BinaryOp::MatMul, { x, 0 }, { wRef, 0 } },
	                               { OutputInfo{ DataType::Float32, { 2, 2 } } });

	// y = matmul + bias (广播 [2,2] + [1,2] -> [2,2])
	const auto y = sg.AddNode(BinaryOpNode{ BinaryOp::Add, { matmul, 0 }, { biasRef, 0 } },
	                          { OutputInfo{ DataType::Float32, { 2, 2 } } });

	sg.SetResults({ { y, 0 } });

	const auto fwdId = graph.AddSubgraph(std::move(sg));
	graph.SetForward(fwdId);

	// x = [[1,2,3],[4,5,6]], w = [[1,2],[3,4],[5,6]]
	// x @ w = [[22,28],[49,64]]
	// + bias[100,200] = [[122,228],[149,264]]
	Tensor<CPU> tensorX({ 1, 2, 3, 4, 5, 6 }, { 2, 3 });
	std::array<Tensor<CPU>, 1> inputs = { std::move(tensorX) };

	Runtime::Interpreter<CPU> interp;
	auto results = interp.RunForward(graph, inputs);

	ASSERT_EQ(results.size(), 1);
	EXPECT_FLOAT_EQ(ReadFloat(results[0], 0), 122);
	EXPECT_FLOAT_EQ(ReadFloat(results[0], 1), 228);
	EXPECT_FLOAT_EQ(ReadFloat(results[0], 2), 149);
	EXPECT_FLOAT_EQ(ReadFloat(results[0], 3), 264);
}

// 测试 3: CallNode 调用 ReLU 子图
TEST(Interpreter, ReLU)
{
	Graph graph;

	// 构建 ReLU 子图
	const auto reluId = Layer::BuildReLU(graph, DataType::Float32, { 2, 3 });

	// 构建前向子图: y = ReLU(x)
	Subgraph sg;
	const auto x = sg.AddParam(DataType::Float32, { 2, 3 });
	const auto y = sg.AddNode(CallNode{ reluId, { { x, 0 } } }, { OutputInfo{ DataType::Float32, { 2, 3 } } });
	sg.SetResults({ { y, 0 } });

	const auto fwdId = graph.AddSubgraph(std::move(sg));
	graph.SetForward(fwdId);

	// x = [[-1, 2, -3], [4, -5, 6]]
	// ReLU(x) = x * cast(x > 0, Float32) = [[0, 2, 0], [4, 0, 6]]
	Tensor<CPU> tensorX({ -1, 2, -3, 4, -5, 6 }, { 2, 3 });
	std::array<Tensor<CPU>, 1> inputs = { std::move(tensorX) };

	Runtime::Interpreter<CPU> interp;
	auto results = interp.RunForward(graph, inputs);

	ASSERT_EQ(results.size(), 1);
	EXPECT_FLOAT_EQ(ReadFloat(results[0], 0), 0);
	EXPECT_FLOAT_EQ(ReadFloat(results[0], 1), 2);
	EXPECT_FLOAT_EQ(ReadFloat(results[0], 2), 0);
	EXPECT_FLOAT_EQ(ReadFloat(results[0], 3), 4);
	EXPECT_FLOAT_EQ(ReadFloat(results[0], 4), 0);
	EXPECT_FLOAT_EQ(ReadFloat(results[0], 5), 6);
}

// 测试 4: 链式操作 y = (x + x) * x
TEST(Interpreter, ChainedOps)
{
	Graph graph;
	Subgraph sg;

	const auto x = sg.AddParam(DataType::Float32, { 3 });

	// t = x + x
	const auto t =
	    sg.AddNode(BinaryOpNode{ BinaryOp::Add, { x, 0 }, { x, 0 } }, { OutputInfo{ DataType::Float32, { 3 } } });

	// y = t * x = 2x * x = 2x^2
	const auto y =
	    sg.AddNode(BinaryOpNode{ BinaryOp::Multiply, { t, 0 }, { x, 0 } }, { OutputInfo{ DataType::Float32, { 3 } } });

	sg.SetResults({ { y, 0 } });

	const auto fwdId = graph.AddSubgraph(std::move(sg));
	graph.SetForward(fwdId);

	// x = [2, 3, 4], y = 2x^2 = [8, 18, 32]
	Tensor<CPU> tensorX({ 2, 3, 4 }, { 3 });
	std::array<Tensor<CPU>, 1> inputs = { std::move(tensorX) };

	Runtime::Interpreter<CPU> interp;
	auto results = interp.RunForward(graph, inputs);

	ASSERT_EQ(results.size(), 1);
	EXPECT_FLOAT_EQ(ReadFloat(results[0], 0), 8);
	EXPECT_FLOAT_EQ(ReadFloat(results[0], 1), 18);
	EXPECT_FLOAT_EQ(ReadFloat(results[0], 2), 32);
}

// 测试 5: 反向传播 y = x @ w + b
TEST(Interpreter, AutogradLinear)
{
	Graph graph;

	// Variable: w[3,2], b[1,2]
	auto w = Variable::Create(Tensor<CPU>({ 1, 2, 3, 4, 5, 6 }, { 3, 2 }));
	auto b = Variable::Create(Tensor<CPU>({ 10, 20 }, { 1, 2 }));
	const auto wIdx = graph.AddVariable(std::move(w));
	const auto bIdx = graph.AddVariable(std::move(b));

	Subgraph sg;
	// 输入: x[1,3]
	const auto x = sg.AddParam(DataType::Float32, { 1, 3 });
	const auto wRef = sg.AddNode(VariableRefNode{ wIdx }, { OutputInfo{ DataType::Float32, { 3, 2 } } });
	const auto bRef = sg.AddNode(VariableRefNode{ bIdx }, { OutputInfo{ DataType::Float32, { 1, 2 } } });

	// matmul = x @ w -> [1,2]
	const auto matmul = sg.AddNode(BinaryOpNode{ BinaryOp::MatMul, { x, 0 }, { wRef, 0 } },
	                               { OutputInfo{ DataType::Float32, { 1, 2 } } });

	// y = matmul + b -> [1,2]
	const auto y = sg.AddNode(BinaryOpNode{ BinaryOp::Add, { matmul, 0 }, { bRef, 0 } },
	                          { OutputInfo{ DataType::Float32, { 1, 2 } } });

	sg.SetResults({ { y, 0 } });
	const auto fwdId = graph.AddSubgraph(std::move(sg));
	graph.SetForward(fwdId);

	// 运行 autograd pass
	AutogradPass autograd;
	autograd.Run(graph);

	ASSERT_TRUE(graph.Backward().has_value());

	// 执行前向
	Tensor<CPU> tensorX({ 1, 2, 3 }, { 1, 3 });
	std::array<Tensor<CPU>, 1> fwdInputs = { std::move(tensorX) };

	Runtime::Interpreter<CPU> interp;
	auto fwdResults = interp.RunForward(graph, fwdInputs);

	ASSERT_EQ(fwdResults.size(), 1);
	// x @ w = [1,2,3] @ [[1,2],[3,4],[5,6]] = [22, 28]
	// + b = [32, 48]
	EXPECT_FLOAT_EQ(ReadFloat(fwdResults[0], 0), 32);
	EXPECT_FLOAT_EQ(ReadFloat(fwdResults[0], 1), 48);

	// 执行反向: dy = ones[1,2]
	Tensor<CPU> gradY({ 1, 1 }, { 1, 2 });

	// backward params: [forward_inputs..., grad_outputs...] = [x, dy]
	Tensor<CPU> tensorX2({ 1, 2, 3 }, { 1, 3 });
	std::vector<Tensor<CPU>> bwdInputs;
	bwdInputs.push_back(std::move(tensorX2));
	bwdInputs.push_back(std::move(gradY));

	auto bwdResults = interp.RunBackward(graph, bwdInputs);

	// backward results: [grad_x, grad_w, grad_b]
	ASSERT_EQ(bwdResults.size(), 3);

	// grad_x = dy @ w^T = [3, 7, 11]
	EXPECT_FLOAT_EQ(ReadFloat(bwdResults[0], 0), 3);
	EXPECT_FLOAT_EQ(ReadFloat(bwdResults[0], 1), 7);
	EXPECT_FLOAT_EQ(ReadFloat(bwdResults[0], 2), 11);

	// grad_w = x^T @ dy = [[1,1],[2,2],[3,3]]
	EXPECT_FLOAT_EQ(ReadFloat(bwdResults[1], 0), 1);
	EXPECT_FLOAT_EQ(ReadFloat(bwdResults[1], 1), 1);
	EXPECT_FLOAT_EQ(ReadFloat(bwdResults[1], 2), 2);
	EXPECT_FLOAT_EQ(ReadFloat(bwdResults[1], 3), 2);
	EXPECT_FLOAT_EQ(ReadFloat(bwdResults[1], 4), 3);
	EXPECT_FLOAT_EQ(ReadFloat(bwdResults[1], 5), 3);

	// grad_b = [1, 1]
	EXPECT_FLOAT_EQ(ReadFloat(bwdResults[2], 0), 1);
	EXPECT_FLOAT_EQ(ReadFloat(bwdResults[2], 1), 1);
}

// 测试 6: Pow 标量广播 y = x ^ 2
TEST(Interpreter, PowScalarBroadcast)
{
	Graph graph;
	Subgraph sg;

	// 输入: x[2,3]
	const auto x = sg.AddParam(DataType::Float32, { 2, 3 });

	// 常量: 标量 2.0 (shape {1})
	auto expTensor = Tensor<CPU>({ 2.0f }, { 1 });
	const auto exp = sg.AddNode(ConstantNode{ expTensor.CopyToDevice(PolymorphicDevice{ CPU{} }) },
	                            { OutputInfo{ DataType::Float32, { 1 } } });

	// y = x ^ 2 (广播: {2,3} ^ {1} -> {2,3})
	const auto y =
	    sg.AddNode(BinaryOpNode{ BinaryOp::Pow, { x, 0 }, { exp, 0 } }, { OutputInfo{ DataType::Float32, { 2, 3 } } });
	sg.SetResults({ { y, 0 } });

	const auto fwdId = graph.AddSubgraph(std::move(sg));
	graph.SetForward(fwdId);

	// x = [[1, 2, 3], [4, 5, 6]], y = x^2 = [[1, 4, 9], [16, 25, 36]]
	Tensor<CPU> tensorX({ 1, 2, 3, 4, 5, 6 }, { 2, 3 });
	std::array<Tensor<CPU>, 1> inputs = { std::move(tensorX) };

	Runtime::Interpreter<CPU> interp;
	auto results = interp.RunForward(graph, inputs);

	ASSERT_EQ(results.size(), 1);
	EXPECT_FLOAT_EQ(ReadFloat(results[0], 0), 1);
	EXPECT_FLOAT_EQ(ReadFloat(results[0], 1), 4);
	EXPECT_FLOAT_EQ(ReadFloat(results[0], 2), 9);
	EXPECT_FLOAT_EQ(ReadFloat(results[0], 3), 16);
	EXPECT_FLOAT_EQ(ReadFloat(results[0], 4), 25);
	EXPECT_FLOAT_EQ(ReadFloat(results[0], 5), 36);
}

// 测试 7: Pow 同 shape element-wise
TEST(Interpreter, PowElementWise)
{
	Graph graph;
	Subgraph sg;

	const auto base = sg.AddParam(DataType::Float32, { 4 });
	const auto exp = sg.AddParam(DataType::Float32, { 4 });

	const auto y =
	    sg.AddNode(BinaryOpNode{ BinaryOp::Pow, { base, 0 }, { exp, 0 } }, { OutputInfo{ DataType::Float32, { 4 } } });
	sg.SetResults({ { y, 0 } });

	const auto fwdId = graph.AddSubgraph(std::move(sg));
	graph.SetForward(fwdId);

	// base = [2, 3, 4, 5], exp = [3, 2, 1, 0]
	// y = [8, 9, 4, 1]
	Tensor<CPU> tensorBase({ 2, 3, 4, 5 }, { 4 });
	Tensor<CPU> tensorExp({ 3, 2, 1, 0 }, { 4 });
	std::array<Tensor<CPU>, 2> inputs = { std::move(tensorBase), std::move(tensorExp) };

	Runtime::Interpreter<CPU> interp;
	auto results = interp.RunForward(graph, inputs);

	ASSERT_EQ(results.size(), 1);
	EXPECT_FLOAT_EQ(ReadFloat(results[0], 0), 8);
	EXPECT_FLOAT_EQ(ReadFloat(results[0], 1), 9);
	EXPECT_FLOAT_EQ(ReadFloat(results[0], 2), 4);
	EXPECT_FLOAT_EQ(ReadFloat(results[0], 3), 1);
}

// 测试 8: Max/Min 同 shape 操作
TEST(Interpreter, MaxMin)
{
	Graph graph;
	Subgraph sg;

	const auto a = sg.AddParam(DataType::Float32, { 2, 3 });
	const auto b = sg.AddParam(DataType::Float32, { 2, 3 });

	const auto maxResult =
	    sg.AddNode(BinaryOpNode{ BinaryOp::Max, { a, 0 }, { b, 0 } }, { OutputInfo{ DataType::Float32, { 2, 3 } } });
	const auto minResult =
	    sg.AddNode(BinaryOpNode{ BinaryOp::Min, { a, 0 }, { b, 0 } }, { OutputInfo{ DataType::Float32, { 2, 3 } } });
	sg.SetResults({ { maxResult, 0 }, { minResult, 0 } });

	const auto fwdId = graph.AddSubgraph(std::move(sg));
	graph.SetForward(fwdId);

	// a = [[1, 5, 3], [7, 2, 6]], b = [[4, 2, 8], [1, 9, 3]]
	// max = [[4, 5, 8], [7, 9, 6]], min = [[1, 2, 3], [1, 2, 3]]
	Tensor<CPU> tensorA({ 1, 5, 3, 7, 2, 6 }, { 2, 3 });
	Tensor<CPU> tensorB({ 4, 2, 8, 1, 9, 3 }, { 2, 3 });
	std::array<Tensor<CPU>, 2> inputs = { std::move(tensorA), std::move(tensorB) };

	Runtime::Interpreter<CPU> interp;
	auto results = interp.RunForward(graph, inputs);

	ASSERT_EQ(results.size(), 2);

	// max
	EXPECT_FLOAT_EQ(ReadFloat(results[0], 0), 4);
	EXPECT_FLOAT_EQ(ReadFloat(results[0], 1), 5);
	EXPECT_FLOAT_EQ(ReadFloat(results[0], 2), 8);
	EXPECT_FLOAT_EQ(ReadFloat(results[0], 3), 7);
	EXPECT_FLOAT_EQ(ReadFloat(results[0], 4), 9);
	EXPECT_FLOAT_EQ(ReadFloat(results[0], 5), 6);

	// min
	EXPECT_FLOAT_EQ(ReadFloat(results[1], 0), 1);
	EXPECT_FLOAT_EQ(ReadFloat(results[1], 1), 2);
	EXPECT_FLOAT_EQ(ReadFloat(results[1], 2), 3);
	EXPECT_FLOAT_EQ(ReadFloat(results[1], 3), 1);
	EXPECT_FLOAT_EQ(ReadFloat(results[1], 4), 2);
	EXPECT_FLOAT_EQ(ReadFloat(results[1], 5), 3);
}

// 测试 9: ReduceSum
TEST(Interpreter, ReduceSum)
{
	Graph graph;
	Subgraph sg;

	// 输入: x[2,3]
	const auto x = sg.AddParam(DataType::Float32, { 2, 3 });

	// sum(x, axis=0) -> [3]
	const auto sumAxis0 =
	    sg.AddNode(ReduceOpNode{ ReduceOp::Sum, { x, 0 }, 0 }, { OutputInfo{ DataType::Float32, { 3 } } });

	// sum(x, axis=1) -> [2]
	const auto sumAxis1 =
	    sg.AddNode(ReduceOpNode{ ReduceOp::Sum, { x, 0 }, 1 }, { OutputInfo{ DataType::Float32, { 2 } } });

	sg.SetResults({ { sumAxis0, 0 }, { sumAxis1, 0 } });

	const auto fwdId = graph.AddSubgraph(std::move(sg));
	graph.SetForward(fwdId);

	// x = [[1, 2, 3], [4, 5, 6]]
	// sum(axis=0) = [5, 7, 9]
	// sum(axis=1) = [6, 15]
	Tensor<CPU> tensorX({ 1, 2, 3, 4, 5, 6 }, { 2, 3 });
	std::array<Tensor<CPU>, 1> inputs = { std::move(tensorX) };

	Runtime::Interpreter<CPU> interp;
	auto results = interp.RunForward(graph, inputs);

	ASSERT_EQ(results.size(), 2);

	// axis=0
	EXPECT_FLOAT_EQ(ReadFloat(results[0], 0), 5);
	EXPECT_FLOAT_EQ(ReadFloat(results[0], 1), 7);
	EXPECT_FLOAT_EQ(ReadFloat(results[0], 2), 9);

	// axis=1
	EXPECT_FLOAT_EQ(ReadFloat(results[1], 0), 6);
	EXPECT_FLOAT_EQ(ReadFloat(results[1], 1), 15);
}

// 测试 10: ReduceMean
TEST(Interpreter, ReduceMean)
{
	Graph graph;
	Subgraph sg;

	const auto x = sg.AddParam(DataType::Float32, { 2, 3 });

	// mean(x, axis=1) -> [2]
	const auto meanAxis1 =
	    sg.AddNode(ReduceOpNode{ ReduceOp::Mean, { x, 0 }, 1 }, { OutputInfo{ DataType::Float32, { 2 } } });

	sg.SetResults({ { meanAxis1, 0 } });

	const auto fwdId = graph.AddSubgraph(std::move(sg));
	graph.SetForward(fwdId);

	// x = [[1, 2, 3], [4, 5, 6]]
	// mean(axis=1) = [2, 5]
	Tensor<CPU> tensorX({ 1, 2, 3, 4, 5, 6 }, { 2, 3 });
	std::array<Tensor<CPU>, 1> inputs = { std::move(tensorX) };

	Runtime::Interpreter<CPU> interp;
	auto results = interp.RunForward(graph, inputs);

	ASSERT_EQ(results.size(), 1);
	EXPECT_FLOAT_EQ(ReadFloat(results[0], 0), 2);
	EXPECT_FLOAT_EQ(ReadFloat(results[0], 1), 5);
}

// 测试 11: ReduceMax
TEST(Interpreter, ReduceMax)
{
	Graph graph;
	Subgraph sg;

	const auto x = sg.AddParam(DataType::Float32, { 2, 3 });

	// max(x, axis=0) -> [3]
	const auto maxAxis0 =
	    sg.AddNode(ReduceOpNode{ ReduceOp::Max, { x, 0 }, 0 }, { OutputInfo{ DataType::Float32, { 3 } } });

	sg.SetResults({ { maxAxis0, 0 } });

	const auto fwdId = graph.AddSubgraph(std::move(sg));
	graph.SetForward(fwdId);

	// x = [[1, 5, 3], [4, 2, 6]]
	// max(axis=0) = [4, 5, 6]
	Tensor<CPU> tensorX({ 1, 5, 3, 4, 2, 6 }, { 2, 3 });
	std::array<Tensor<CPU>, 1> inputs = { std::move(tensorX) };

	Runtime::Interpreter<CPU> interp;
	auto results = interp.RunForward(graph, inputs);

	ASSERT_EQ(results.size(), 1);
	EXPECT_FLOAT_EQ(ReadFloat(results[0], 0), 4);
	EXPECT_FLOAT_EQ(ReadFloat(results[0], 1), 5);
	EXPECT_FLOAT_EQ(ReadFloat(results[0], 2), 6);
}

// 测试 12: Reshape
TEST(Interpreter, Reshape)
{
	Graph graph;
	Subgraph sg;

	const auto x = sg.AddParam(DataType::Float32, { 2, 3 });

	// reshape [2,3] -> [3,2]
	const auto r1 = sg.AddNode(ReshapeNode{ { x, 0 }, { 3, 2 } }, { OutputInfo{ DataType::Float32, { 3, 2 } } });

	// reshape [3,2] -> [6]
	const auto r2 = sg.AddNode(ReshapeNode{ { r1, 0 }, { 6 } }, { OutputInfo{ DataType::Float32, { 6 } } });

	sg.SetResults({ { r1, 0 }, { r2, 0 } });

	const auto fwdId = graph.AddSubgraph(std::move(sg));
	graph.SetForward(fwdId);

	// x = [[1, 2, 3], [4, 5, 6]]
	// reshape [3,2] -> [[1,2],[3,4],[5,6]] (row-major)
	// reshape [6] -> [1,2,3,4,5,6]
	Tensor<CPU> tensorX({ 1, 2, 3, 4, 5, 6 }, { 2, 3 });
	std::array<Tensor<CPU>, 1> inputs = { std::move(tensorX) };

	Runtime::Interpreter<CPU> interp;
	auto results = interp.RunForward(graph, inputs);

	ASSERT_EQ(results.size(), 2);

	// [3,2]
	for (int i = 0; i < 6; ++i)
	{
		EXPECT_FLOAT_EQ(ReadFloat(results[0], i), static_cast<float>(i + 1));
	}

	// [6]
	for (int i = 0; i < 6; ++i)
	{
		EXPECT_FLOAT_EQ(ReadFloat(results[1], i), static_cast<float>(i + 1));
	}
}

// 测试 13: ReduceSum 反向传播
TEST(Interpreter, AutogradReduceSum)
{
	Graph graph;
	Subgraph sg;

	// 输入: x[2,3]
	const auto x = sg.AddParam(DataType::Float32, { 2, 3 });

	// y = sum(x, axis=1) -> [2]
	const auto y = sg.AddNode(ReduceOpNode{ ReduceOp::Sum, { x, 0 }, 1 }, { OutputInfo{ DataType::Float32, { 2 } } });

	sg.SetResults({ { y, 0 } });
	const auto fwdId = graph.AddSubgraph(std::move(sg));
	graph.SetForward(fwdId);

	AutogradPass autograd;
	autograd.Run(graph);

	ASSERT_TRUE(graph.Backward().has_value());

	// 执行前向
	Tensor<CPU> tensorX({ 1, 2, 3, 4, 5, 6 }, { 2, 3 });
	std::array<Tensor<CPU>, 1> fwdInputs = { std::move(tensorX) };

	Runtime::Interpreter<CPU> interp;
	auto fwdResults = interp.RunForward(graph, fwdInputs);

	ASSERT_EQ(fwdResults.size(), 1);
	EXPECT_FLOAT_EQ(ReadFloat(fwdResults[0], 0), 6);  // 1+2+3
	EXPECT_FLOAT_EQ(ReadFloat(fwdResults[0], 1), 15); // 4+5+6

	// 执行反向: dy = [1, 1]
	Tensor<CPU> gradY({ 1, 1 }, { 2 });
	Tensor<CPU> tensorX2({ 1, 2, 3, 4, 5, 6 }, { 2, 3 });
	std::vector<Tensor<CPU>> bwdInputs;
	bwdInputs.push_back(std::move(tensorX2));
	bwdInputs.push_back(std::move(gradY));

	auto bwdResults = interp.RunBackward(graph, bwdInputs);

	// d(sum)/dx = all ones [2,3]
	ASSERT_EQ(bwdResults.size(), 1);
	for (int i = 0; i < 6; ++i)
	{
		EXPECT_FLOAT_EQ(ReadFloat(bwdResults[0], i), 1.0f);
	}
}

// 测试 14: Max 广播操作，标量与矩阵
TEST(Interpreter, MaxMinBroadcast)
{
	Graph graph;
	Subgraph sg;

	// 输入: x[2,3]
	const auto x = sg.AddParam(DataType::Float32, { 2, 3 });

	// 常量: 标量 3.0 (shape {1})
	auto threshTensor = Tensor<CPU>({ 3.0f }, { 1 });
	const auto thresh = sg.AddNode(ConstantNode{ threshTensor.CopyToDevice(PolymorphicDevice{ CPU{} }) },
	                               { OutputInfo{ DataType::Float32, { 1 } } });

	// clampMin = max(x, 3) -> 将小于3的值替换为3
	const auto clampMin = sg.AddNode(BinaryOpNode{ BinaryOp::Max, { x, 0 }, { thresh, 0 } },
	                                 { OutputInfo{ DataType::Float32, { 2, 3 } } });
	sg.SetResults({ { clampMin, 0 } });

	const auto fwdId = graph.AddSubgraph(std::move(sg));
	graph.SetForward(fwdId);

	// x = [[1, 5, 2], [4, 0, 6]]
	// max(x, 3) = [[3, 5, 3], [4, 3, 6]]
	Tensor<CPU> tensorX({ 1, 5, 2, 4, 0, 6 }, { 2, 3 });
	std::array<Tensor<CPU>, 1> inputs = { std::move(tensorX) };

	Runtime::Interpreter<CPU> interp;
	auto results = interp.RunForward(graph, inputs);

	ASSERT_EQ(results.size(), 1);
	EXPECT_FLOAT_EQ(ReadFloat(results[0], 0), 3);
	EXPECT_FLOAT_EQ(ReadFloat(results[0], 1), 5);
	EXPECT_FLOAT_EQ(ReadFloat(results[0], 2), 3);
	EXPECT_FLOAT_EQ(ReadFloat(results[0], 3), 4);
	EXPECT_FLOAT_EQ(ReadFloat(results[0], 4), 3);
	EXPECT_FLOAT_EQ(ReadFloat(results[0], 5), 6);
}

// 测试 15: CondNode 前向 f(x) = x*x if x>0 else -x
TEST(Interpreter, CondNode)
{
	Graph graph;

	// thenBranch: y = x * x (标量)
	Subgraph thenSg;
	const auto thenX = thenSg.AddParam(DataType::Float32, { 1 });
	const auto thenY = thenSg.AddNode(BinaryOpNode{ BinaryOp::Multiply, { thenX, 0 }, { thenX, 0 } },
	                                  { OutputInfo{ DataType::Float32, { 1 } } });
	thenSg.SetResults({ { thenY, 0 } });
	const auto thenId = graph.AddSubgraph(std::move(thenSg));

	// elseBranch: y = -x (标量)
	Subgraph elseSg;
	const auto elseX = elseSg.AddParam(DataType::Float32, { 1 });
	const auto elseY =
	    elseSg.AddNode(UnaryOpNode{ UnaryOp::Negate, { elseX, 0 } }, { OutputInfo{ DataType::Float32, { 1 } } });
	elseSg.SetResults({ { elseY, 0 } });
	const auto elseId = graph.AddSubgraph(std::move(elseSg));

	// forward: cond = (x > 0), y = CondNode(cond, then, else, [x])
	Subgraph sg;
	const auto x = sg.AddParam(DataType::Float32, { 1 });

	// 常量: 标量 0
	auto zeroTensor = Tensor<CPU>({ 0.0 }, { 1 });
	const auto zero = sg.AddNode(ConstantNode{ zeroTensor.CopyToDevice(PolymorphicDevice{ CPU{} }) },
	                             { OutputInfo{ DataType::Float32, { 1 } } });

	const auto cond =
	    sg.AddNode(BinaryOpNode{ BinaryOp::Greater, { x, 0 }, { zero, 0 } }, { OutputInfo{ DataType::Bool, { 1 } } });

	const auto y =
	    sg.AddNode(CondNode{ { cond, 0 }, thenId, elseId, { { x, 0 } } }, { OutputInfo{ DataType::Float32, { 1 } } });
	sg.SetResults({ { y, 0 } });

	const auto fwdId = graph.AddSubgraph(std::move(sg));
	graph.SetForward(fwdId);

	Runtime::Interpreter<CPU> interp;

	// x=3: 3>0 => then => 3*3 = 9
	{
		Tensor<CPU> tensorX({ 3.0 }, { 1 });
		std::array<Tensor<CPU>, 1> inputs = { std::move(tensorX) };
		auto results = interp.RunForward(graph, inputs);
		ASSERT_EQ(results.size(), 1);
		EXPECT_FLOAT_EQ(ReadFloat(results[0], 0), 9);
	}

	// x=-2: -2>0 false => else => -(-2) = 2
	{
		Tensor<CPU> tensorX({ -2.0 }, { 1 });
		std::array<Tensor<CPU>, 1> inputs = { std::move(tensorX) };
		auto results = interp.RunForward(graph, inputs);
		ASSERT_EQ(results.size(), 1);
		EXPECT_FLOAT_EQ(ReadFloat(results[0], 0), 2);
	}
}

// 测试 16: CondNode 反向传播
TEST(Interpreter, AutogradCondNode)
{
	Graph graph;

	// thenBranch: y = x * x
	Subgraph thenSg;
	const auto thenX = thenSg.AddParam(DataType::Float32, { 1 });
	const auto thenY = thenSg.AddNode(BinaryOpNode{ BinaryOp::Multiply, { thenX, 0 }, { thenX, 0 } },
	                                  { OutputInfo{ DataType::Float32, { 1 } } });
	thenSg.SetResults({ { thenY, 0 } });
	const auto thenId = graph.AddSubgraph(std::move(thenSg));

	// elseBranch: y = -x
	Subgraph elseSg;
	const auto elseX = elseSg.AddParam(DataType::Float32, { 1 });
	const auto elseY =
	    elseSg.AddNode(UnaryOpNode{ UnaryOp::Negate, { elseX, 0 } }, { OutputInfo{ DataType::Float32, { 1 } } });
	elseSg.SetResults({ { elseY, 0 } });
	const auto elseId = graph.AddSubgraph(std::move(elseSg));

	// forward
	Subgraph sg;
	const auto x = sg.AddParam(DataType::Float32, { 1 });

	auto zeroTensor = Tensor<CPU>({ 0.0 }, { 1 });
	const auto zero = sg.AddNode(ConstantNode{ zeroTensor.CopyToDevice(PolymorphicDevice{ CPU{} }) },
	                             { OutputInfo{ DataType::Float32, { 1 } } });

	const auto cond =
	    sg.AddNode(BinaryOpNode{ BinaryOp::Greater, { x, 0 }, { zero, 0 } }, { OutputInfo{ DataType::Bool, { 1 } } });

	const auto y =
	    sg.AddNode(CondNode{ { cond, 0 }, thenId, elseId, { { x, 0 } } }, { OutputInfo{ DataType::Float32, { 1 } } });
	sg.SetResults({ { y, 0 } });

	const auto fwdId = graph.AddSubgraph(std::move(sg));
	graph.SetForward(fwdId);

	// AutogradPass
	AutogradPass autograd;
	autograd.Run(graph);
	ASSERT_TRUE(graph.Backward().has_value());

	Runtime::Interpreter<CPU> interp;

	// x=3: thenBranch (x^2), dy=1 => grad_x = 2x = 6
	{
		Tensor<CPU> tensorX({ 3.0 }, { 1 });
		std::array<Tensor<CPU>, 1> fwdInputs = { std::move(tensorX) };
		auto fwdResults = interp.RunForward(graph, fwdInputs);
		ASSERT_EQ(fwdResults.size(), 1);
		EXPECT_FLOAT_EQ(ReadFloat(fwdResults[0], 0), 9);

		Tensor<CPU> gradY({ 1.0 }, { 1 });
		Tensor<CPU> tensorX2({ 3.0 }, { 1 });
		std::vector<Tensor<CPU>> bwdInputs;
		bwdInputs.push_back(std::move(tensorX2));
		bwdInputs.push_back(std::move(gradY));
		auto bwdResults = interp.RunBackward(graph, bwdInputs);

		ASSERT_EQ(bwdResults.size(), 1);
		EXPECT_FLOAT_EQ(ReadFloat(bwdResults[0], 0), 6);
	}

	// x=-2: elseBranch (-x), dy=1 => grad_x = -1
	{
		Tensor<CPU> tensorX({ -2.0 }, { 1 });
		std::array<Tensor<CPU>, 1> fwdInputs = { std::move(tensorX) };
		auto fwdResults = interp.RunForward(graph, fwdInputs);
		ASSERT_EQ(fwdResults.size(), 1);
		EXPECT_FLOAT_EQ(ReadFloat(fwdResults[0], 0), 2);

		Tensor<CPU> gradY({ 1.0 }, { 1 });
		Tensor<CPU> tensorX2({ -2.0 }, { 1 });
		std::vector<Tensor<CPU>> bwdInputs;
		bwdInputs.push_back(std::move(tensorX2));
		bwdInputs.push_back(std::move(gradY));
		auto bwdResults = interp.RunBackward(graph, bwdInputs);

		ASSERT_EQ(bwdResults.size(), 1);
		EXPECT_FLOAT_EQ(ReadFloat(bwdResults[0], 0), -1);
	}
}

// 测试 17: CallNode 内 VariableRef 的梯度传播
// callee 子图: y = x @ W + B（W、B 为 VariableRef）
// parent: out = Call(callee, [input])
TEST(Interpreter, AutogradCallNodeWithVariable)
{
	Graph graph;

	// Variable: W[3,2], B[1,2]
	auto w = Variable::Create(Tensor<CPU>({ 1, 2, 3, 4, 5, 6 }, { 3, 2 }));
	auto b = Variable::Create(Tensor<CPU>({ 10, 20 }, { 1, 2 }));
	const auto wIdx = graph.AddVariable(std::move(w));
	const auto bIdx = graph.AddVariable(std::move(b));

	// 构建 callee 子图: y = x @ W + B
	Subgraph calleeSg;
	const auto cx = calleeSg.AddParam(DataType::Float32, { 1, 3 });
	const auto cwRef = calleeSg.AddNode(VariableRefNode{ wIdx }, { OutputInfo{ DataType::Float32, { 3, 2 } } });
	const auto cbRef = calleeSg.AddNode(VariableRefNode{ bIdx }, { OutputInfo{ DataType::Float32, { 1, 2 } } });
	const auto cmatmul = calleeSg.AddNode(BinaryOpNode{ BinaryOp::MatMul, { cx, 0 }, { cwRef, 0 } },
	                                      { OutputInfo{ DataType::Float32, { 1, 2 } } });
	const auto cy = calleeSg.AddNode(BinaryOpNode{ BinaryOp::Add, { cmatmul, 0 }, { cbRef, 0 } },
	                                 { OutputInfo{ DataType::Float32, { 1, 2 } } });
	calleeSg.SetResults({ { cy, 0 } });
	const auto calleeId = graph.AddSubgraph(std::move(calleeSg));

	// 构建 parent 子图: out = Call(callee, [x])
	Subgraph sg;
	const auto x = sg.AddParam(DataType::Float32, { 1, 3 });
	const auto out = sg.AddNode(CallNode{ calleeId, { { x, 0 } } }, { OutputInfo{ DataType::Float32, { 1, 2 } } });
	sg.SetResults({ { out, 0 } });
	const auto fwdId = graph.AddSubgraph(std::move(sg));
	graph.SetForward(fwdId);

	// Autograd
	AutogradPass autograd;
	autograd.Run(graph);
	ASSERT_TRUE(graph.Backward().has_value());

	// Forward: x = [1,2,3], W = [[1,2],[3,4],[5,6]], B = [10,20]
	// x@W = [22,28], + B = [32,48]
	Tensor<CPU> tensorX({ 1, 2, 3 }, { 1, 3 });
	std::array<Tensor<CPU>, 1> fwdInputs = { std::move(tensorX) };

	Runtime::Interpreter<CPU> interp;
	auto fwdResults = interp.RunForward(graph, fwdInputs);

	ASSERT_EQ(fwdResults.size(), 1);
	EXPECT_FLOAT_EQ(ReadFloat(fwdResults[0], 0), 32);
	EXPECT_FLOAT_EQ(ReadFloat(fwdResults[0], 1), 48);

	// Backward: dy = [1, 1]
	Tensor<CPU> gradY({ 1, 1 }, { 1, 2 });
	Tensor<CPU> tensorX2({ 1, 2, 3 }, { 1, 3 });
	std::vector<Tensor<CPU>> bwdInputs;
	bwdInputs.push_back(std::move(tensorX2));
	bwdInputs.push_back(std::move(gradY));

	auto bwdResults = interp.RunBackward(graph, bwdInputs);

	// backward results: [grad_x, grad_W, grad_B]
	ASSERT_EQ(bwdResults.size(), 3);

	// grad_x = dy @ W^T = [1,1] @ [[1,3,5],[2,4,6]] = [3, 7, 11]
	EXPECT_FLOAT_EQ(ReadFloat(bwdResults[0], 0), 3);
	EXPECT_FLOAT_EQ(ReadFloat(bwdResults[0], 1), 7);
	EXPECT_FLOAT_EQ(ReadFloat(bwdResults[0], 2), 11);

	// grad_W = x^T @ dy = [[1],[2],[3]] @ [[1,1]] = [[1,1],[2,2],[3,3]]
	EXPECT_FLOAT_EQ(ReadFloat(bwdResults[1], 0), 1);
	EXPECT_FLOAT_EQ(ReadFloat(bwdResults[1], 1), 1);
	EXPECT_FLOAT_EQ(ReadFloat(bwdResults[1], 2), 2);
	EXPECT_FLOAT_EQ(ReadFloat(bwdResults[1], 3), 2);
	EXPECT_FLOAT_EQ(ReadFloat(bwdResults[1], 4), 3);
	EXPECT_FLOAT_EQ(ReadFloat(bwdResults[1], 5), 3);

	// grad_B = dy = [1, 1]
	EXPECT_FLOAT_EQ(ReadFloat(bwdResults[2], 0), 1);
	EXPECT_FLOAT_EQ(ReadFloat(bwdResults[2], 1), 1);
}

// 测试 18: 两次 CallNode 调用同一个 callee（含 Variable），梯度累加
// callee: y = W * x（element-wise），W 是 Variable
// parent: out = call1(x1) + call2(x2)
TEST(Interpreter, AutogradCallNodeSharedCallee)
{
	Graph graph;

	// Variable: W[2]
	auto w = Variable::Create(Tensor<CPU>({ 3, 4 }, { 2 }));
	const auto wIdx = graph.AddVariable(std::move(w));

	// callee 子图: y = W * x
	Subgraph calleeSg;
	const auto cx = calleeSg.AddParam(DataType::Float32, { 2 });
	const auto cwRef = calleeSg.AddNode(VariableRefNode{ wIdx }, { OutputInfo{ DataType::Float32, { 2 } } });
	const auto cy = calleeSg.AddNode(BinaryOpNode{ BinaryOp::Multiply, { cwRef, 0 }, { cx, 0 } },
	                                 { OutputInfo{ DataType::Float32, { 2 } } });
	calleeSg.SetResults({ { cy, 0 } });
	const auto calleeId = graph.AddSubgraph(std::move(calleeSg));

	// parent: out = call1(x1) + call2(x2)
	Subgraph sg;
	const auto x1 = sg.AddParam(DataType::Float32, { 2 });
	const auto x2 = sg.AddParam(DataType::Float32, { 2 });
	const auto call1 = sg.AddNode(CallNode{ calleeId, { { x1, 0 } } }, { OutputInfo{ DataType::Float32, { 2 } } });
	const auto call2 = sg.AddNode(CallNode{ calleeId, { { x2, 0 } } }, { OutputInfo{ DataType::Float32, { 2 } } });
	const auto out = sg.AddNode(BinaryOpNode{ BinaryOp::Add, { call1, 0 }, { call2, 0 } },
	                            { OutputInfo{ DataType::Float32, { 2 } } });
	sg.SetResults({ { out, 0 } });
	const auto fwdId = graph.AddSubgraph(std::move(sg));
	graph.SetForward(fwdId);

	AutogradPass autograd;
	autograd.Run(graph);
	ASSERT_TRUE(graph.Backward().has_value());

	// x1 = [1, 2], x2 = [5, 6], W = [3, 4]
	// call1 = W*x1 = [3, 8], call2 = W*x2 = [15, 24]
	// out = [18, 32]
	Tensor<CPU> t1({ 1, 2 }, { 2 });
	Tensor<CPU> t2({ 5, 6 }, { 2 });
	std::array<Tensor<CPU>, 2> fwdInputs = { std::move(t1), std::move(t2) };

	Runtime::Interpreter<CPU> interp;
	auto fwdResults = interp.RunForward(graph, fwdInputs);

	ASSERT_EQ(fwdResults.size(), 1);
	EXPECT_FLOAT_EQ(ReadFloat(fwdResults[0], 0), 18);
	EXPECT_FLOAT_EQ(ReadFloat(fwdResults[0], 1), 32);

	// dy = [1, 1]
	// d(W*x)/dx = W, d(W*x)/dW = x
	// grad_x1 = W = [3, 4], grad_x2 = W = [3, 4]
	// grad_W = x1 + x2 = [1+5, 2+6] = [6, 8]  (两次调用的梯度累加)
	Tensor<CPU> gradY({ 1, 1 }, { 2 });
	Tensor<CPU> t1b({ 1, 2 }, { 2 });
	Tensor<CPU> t2b({ 5, 6 }, { 2 });
	std::vector<Tensor<CPU>> bwdInputs;
	bwdInputs.push_back(std::move(t1b));
	bwdInputs.push_back(std::move(t2b));
	bwdInputs.push_back(std::move(gradY));

	auto bwdResults = interp.RunBackward(graph, bwdInputs);

	// [grad_x1, grad_x2, grad_W]
	ASSERT_EQ(bwdResults.size(), 3);

	// grad_x1 = W * dy = [3, 4]
	EXPECT_FLOAT_EQ(ReadFloat(bwdResults[0], 0), 3);
	EXPECT_FLOAT_EQ(ReadFloat(bwdResults[0], 1), 4);

	// grad_x2 = W * dy = [3, 4]
	EXPECT_FLOAT_EQ(ReadFloat(bwdResults[1], 0), 3);
	EXPECT_FLOAT_EQ(ReadFloat(bwdResults[1], 1), 4);

	// grad_W = x1 + x2 = [6, 8]
	EXPECT_FLOAT_EQ(ReadFloat(bwdResults[2], 0), 6);
	EXPECT_FLOAT_EQ(ReadFloat(bwdResults[2], 1), 8);
}

// 测试 19: CondNode 分支中含 Variable
// then: y = W * x, else: y = V * x
TEST(Interpreter, AutogradCondNodeWithVariable)
{
	Graph graph;

	// Variables
	auto w = Variable::Create(Tensor<CPU>({ 2.0 }, { 1 }));
	auto v = Variable::Create(Tensor<CPU>({ 5.0 }, { 1 }));
	const auto wIdx = graph.AddVariable(std::move(w));
	const auto vIdx = graph.AddVariable(std::move(v));

	// thenBranch: y = W * x
	Subgraph thenSg;
	const auto thenX = thenSg.AddParam(DataType::Float32, { 1 });
	const auto thenW = thenSg.AddNode(VariableRefNode{ wIdx }, { OutputInfo{ DataType::Float32, { 1 } } });
	const auto thenY = thenSg.AddNode(BinaryOpNode{ BinaryOp::Multiply, { thenW, 0 }, { thenX, 0 } },
	                                  { OutputInfo{ DataType::Float32, { 1 } } });
	thenSg.SetResults({ { thenY, 0 } });
	const auto thenId = graph.AddSubgraph(std::move(thenSg));

	// elseBranch: y = V * x
	Subgraph elseSg;
	const auto elseX = elseSg.AddParam(DataType::Float32, { 1 });
	const auto elseV = elseSg.AddNode(VariableRefNode{ vIdx }, { OutputInfo{ DataType::Float32, { 1 } } });
	const auto elseY = elseSg.AddNode(BinaryOpNode{ BinaryOp::Multiply, { elseV, 0 }, { elseX, 0 } },
	                                  { OutputInfo{ DataType::Float32, { 1 } } });
	elseSg.SetResults({ { elseY, 0 } });
	const auto elseId = graph.AddSubgraph(std::move(elseSg));

	// forward: cond = (x > 0), y = CondNode(cond, then, else, [x])
	Subgraph sg;
	const auto x = sg.AddParam(DataType::Float32, { 1 });
	auto zeroTensor = Tensor<CPU>({ 0.0 }, { 1 });
	const auto zero = sg.AddNode(ConstantNode{ zeroTensor.CopyToDevice(PolymorphicDevice{ CPU{} }) },
	                             { OutputInfo{ DataType::Float32, { 1 } } });
	const auto cond =
	    sg.AddNode(BinaryOpNode{ BinaryOp::Greater, { x, 0 }, { zero, 0 } }, { OutputInfo{ DataType::Bool, { 1 } } });
	const auto y =
	    sg.AddNode(CondNode{ { cond, 0 }, thenId, elseId, { { x, 0 } } }, { OutputInfo{ DataType::Float32, { 1 } } });
	sg.SetResults({ { y, 0 } });

	const auto fwdId = graph.AddSubgraph(std::move(sg));
	graph.SetForward(fwdId);

	AutogradPass autograd;
	autograd.Run(graph);
	ASSERT_TRUE(graph.Backward().has_value());

	Runtime::Interpreter<CPU> interp;

	// x=3 (>0): thenBranch, y = W*x = 2*3 = 6
	// grad_x = W = 2, grad_W = x = 3, grad_V = 0
	{
		Tensor<CPU> tensorX({ 3.0 }, { 1 });
		std::array<Tensor<CPU>, 1> fwdInputs = { std::move(tensorX) };
		auto fwdResults = interp.RunForward(graph, fwdInputs);
		ASSERT_EQ(fwdResults.size(), 1);
		EXPECT_FLOAT_EQ(ReadFloat(fwdResults[0], 0), 6);

		Tensor<CPU> gradY({ 1.0 }, { 1 });
		Tensor<CPU> tensorX2({ 3.0 }, { 1 });
		std::vector<Tensor<CPU>> bwdInputs;
		bwdInputs.push_back(std::move(tensorX2));
		bwdInputs.push_back(std::move(gradY));
		auto bwdResults = interp.RunBackward(graph, bwdInputs);

		// [grad_x, grad_V, grad_W] — sorted by variableIndex: wIdx=0, vIdx=1
		ASSERT_EQ(bwdResults.size(), 3);
		EXPECT_FLOAT_EQ(ReadFloat(bwdResults[0], 0), 2);  // grad_x = W = 2
		EXPECT_FLOAT_EQ(ReadFloat(bwdResults[1], 0), 3);  // grad_W = x = 3
		EXPECT_FLOAT_EQ(ReadFloat(bwdResults[2], 0), 0);  // grad_V = 0 (else 分支未执行)
	}

	// x=-4 (<= 0): elseBranch, y = V*x = 5*(-4) = -20
	// grad_x = V = 5, grad_W = 0, grad_V = x = -4
	{
		Tensor<CPU> tensorX({ -4.0 }, { 1 });
		std::array<Tensor<CPU>, 1> fwdInputs = { std::move(tensorX) };
		auto fwdResults = interp.RunForward(graph, fwdInputs);
		ASSERT_EQ(fwdResults.size(), 1);
		EXPECT_FLOAT_EQ(ReadFloat(fwdResults[0], 0), -20);

		Tensor<CPU> gradY({ 1.0 }, { 1 });
		Tensor<CPU> tensorX2({ -4.0 }, { 1 });
		std::vector<Tensor<CPU>> bwdInputs;
		bwdInputs.push_back(std::move(tensorX2));
		bwdInputs.push_back(std::move(gradY));
		auto bwdResults = interp.RunBackward(graph, bwdInputs);

		ASSERT_EQ(bwdResults.size(), 3);
		EXPECT_FLOAT_EQ(ReadFloat(bwdResults[0], 0), 5);   // grad_x = V = 5
		EXPECT_FLOAT_EQ(ReadFloat(bwdResults[1], 0), 0);   // grad_W = 0 (then 分支未执行)
		EXPECT_FLOAT_EQ(ReadFloat(bwdResults[2], 0), -4);   // grad_V = x = -4
	}
}

// 测试 20: 嵌套 CallNode 的 Variable 梯度传播
// sub-callee: y = W * x（W 是 Variable）
// callee: y = Call(sub-callee, [x])（透传）
// parent: out = Call(callee, [input])
TEST(Interpreter, AutogradNestedCallWithVariable)
{
	Graph graph;

	// Variable: W[2]
	auto w = Variable::Create(Tensor<CPU>({ 3, 4 }, { 2 }));
	const auto wIdx = graph.AddVariable(std::move(w));

	// sub-callee: y = W * x
	Subgraph subCalleeSg;
	const auto scx = subCalleeSg.AddParam(DataType::Float32, { 2 });
	const auto scw = subCalleeSg.AddNode(VariableRefNode{ wIdx }, { OutputInfo{ DataType::Float32, { 2 } } });
	const auto scy = subCalleeSg.AddNode(BinaryOpNode{ BinaryOp::Multiply, { scw, 0 }, { scx, 0 } },
	                                     { OutputInfo{ DataType::Float32, { 2 } } });
	subCalleeSg.SetResults({ { scy, 0 } });
	const auto subCalleeId = graph.AddSubgraph(std::move(subCalleeSg));

	// callee: y = Call(sub-callee, [x])
	Subgraph calleeSg;
	const auto cx = calleeSg.AddParam(DataType::Float32, { 2 });
	const auto ccall = calleeSg.AddNode(CallNode{ subCalleeId, { { cx, 0 } } }, { OutputInfo{ DataType::Float32, { 2 } } });
	calleeSg.SetResults({ { ccall, 0 } });
	const auto calleeId = graph.AddSubgraph(std::move(calleeSg));

	// parent: out = Call(callee, [x])
	Subgraph sg;
	const auto x = sg.AddParam(DataType::Float32, { 2 });
	const auto out = sg.AddNode(CallNode{ calleeId, { { x, 0 } } }, { OutputInfo{ DataType::Float32, { 2 } } });
	sg.SetResults({ { out, 0 } });
	const auto fwdId = graph.AddSubgraph(std::move(sg));
	graph.SetForward(fwdId);

	AutogradPass autograd;
	autograd.Run(graph);
	ASSERT_TRUE(graph.Backward().has_value());

	// x = [2, 5], W = [3, 4]
	// out = W * x = [6, 20]
	Tensor<CPU> tensorX({ 2, 5 }, { 2 });
	std::array<Tensor<CPU>, 1> fwdInputs = { std::move(tensorX) };

	Runtime::Interpreter<CPU> interp;
	auto fwdResults = interp.RunForward(graph, fwdInputs);

	ASSERT_EQ(fwdResults.size(), 1);
	EXPECT_FLOAT_EQ(ReadFloat(fwdResults[0], 0), 6);
	EXPECT_FLOAT_EQ(ReadFloat(fwdResults[0], 1), 20);

	// dy = [1, 1]
	// grad_x = W = [3, 4]
	// grad_W = x = [2, 5]
	Tensor<CPU> gradY({ 1, 1 }, { 2 });
	Tensor<CPU> tensorX2({ 2, 5 }, { 2 });
	std::vector<Tensor<CPU>> bwdInputs;
	bwdInputs.push_back(std::move(tensorX2));
	bwdInputs.push_back(std::move(gradY));

	auto bwdResults = interp.RunBackward(graph, bwdInputs);

	// [grad_x, grad_W]
	ASSERT_EQ(bwdResults.size(), 2);

	// grad_x = W = [3, 4]
	EXPECT_FLOAT_EQ(ReadFloat(bwdResults[0], 0), 3);
	EXPECT_FLOAT_EQ(ReadFloat(bwdResults[0], 1), 4);

	// grad_W = x = [2, 5]
	EXPECT_FLOAT_EQ(ReadFloat(bwdResults[1], 0), 2);
	EXPECT_FLOAT_EQ(ReadFloat(bwdResults[1], 1), 5);
}
