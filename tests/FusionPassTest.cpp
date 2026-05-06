#include <gtest/gtest.h>

#include <LiteNN.h>
#include <LiteNN/Pass/AutogradPass.h>
#include <LiteNN/Pass/FusionPass.h>
#include <LiteNN/Runtime/Interpreter.h>

using namespace LiteNN;

// 辅助函数: 读取 CPU Tensor 的第 i 个 float 元素
static float ReadFloat(const Tensor<CPU>& t, std::size_t i)
{
	return static_cast<const float*>(t.RawData())[i];
}

// 辅助函数: 检查子图中是否存在 FusedOpNode
static bool HasFusedOpNode(const Subgraph& sg, FusionPattern expectedPattern)
{
	for (NodeId id = 0; id < sg.NodeCount(); ++id)
	{
		if (auto* fused = std::get_if<FusedOpNode>(&sg.GetNodeEntry(id).node))
		{
			if (fused->pattern == expectedPattern)
			{
				return true;
			}
		}
	}
	return false;
}

// 辅助函数: 计算子图中 FusedOpNode 的数量
static std::size_t CountFusedOpNodes(const Subgraph& sg)
{
	std::size_t count = 0;
	for (NodeId id = 0; id < sg.NodeCount(); ++id)
	{
		if (std::holds_alternative<FusedOpNode>(sg.GetNodeEntry(id).node))
		{
			++count;
		}
	}
	return count;
}

// 测试 1: MatMulBiasAdd 融合与正确性
// y = MatMul(x, w) + b
TEST(FusionPass, MatMulBiasAdd)
{
	Graph graph;
	Subgraph sg;

	// 参数: x[2,3], w[3,2], b[1,2]
	const auto x = sg.AddParam(DataType::Float32, { 2, 3 });
	const auto w = sg.AddParam(DataType::Float32, { 3, 2 });
	const auto b = sg.AddParam(DataType::Float32, { 1, 2 });

	// matmul = x @ w
	const auto matmul =
	    sg.AddNode(BinaryOpNode{ BinaryOp::MatMul, { x, 0 }, { w, 0 } }, { OutputInfo{ DataType::Float32, { 2, 2 } } });

	// y = matmul + b
	const auto y = sg.AddNode(BinaryOpNode{ BinaryOp::Add, { matmul, 0 }, { b, 0 } },
	                           { OutputInfo{ DataType::Float32, { 2, 2 } } });

	sg.SetResults({ { y, 0 } });
	const auto fwdId = graph.AddSubgraph(std::move(sg));
	graph.SetForward(fwdId);

	// 计算未融合的结果
	Tensor<CPU> tensorX({ 1, 2, 3, 4, 5, 6 }, { 2, 3 });
	Tensor<CPU> tensorW({ 1, 2, 3, 4, 5, 6 }, { 3, 2 });
	Tensor<CPU> tensorB({ 100, 200 }, { 1, 2 });
	std::array<Tensor<CPU>, 3> inputs = { Tensor<CPU>(tensorX), Tensor<CPU>(tensorW), Tensor<CPU>(tensorB) };

	Runtime::Interpreter<CPU> interp;
	auto expected = interp.RunForward(graph, inputs);

	// 运行 FusionPass
	FusionPass fusionPass;
	fusionPass.Run(graph);

	// 验证融合节点存在
	const auto& fusedSg = graph.GetSubgraph(graph.Forward());
	EXPECT_TRUE(HasFusedOpNode(fusedSg, FusionPattern::MatMulBiasAdd));

	// 验证数值正确
	std::array<Tensor<CPU>, 3> inputs2 = { std::move(tensorX), std::move(tensorW), std::move(tensorB) };
	auto actual = interp.RunForward(graph, inputs2);

	ASSERT_EQ(actual.size(), 1);
	for (std::size_t i = 0; i < 4; ++i)
	{
		EXPECT_FLOAT_EQ(ReadFloat(actual[0], i), ReadFloat(expected[0], i));
	}
}

// 测试 2: MatMulBiasAdd 交换 Add 操作数
// y = b + MatMul(x, w)
TEST(FusionPass, MatMulBiasAddSwapped)
{
	Graph graph;
	Subgraph sg;

	const auto x = sg.AddParam(DataType::Float32, { 2, 3 });
	const auto w = sg.AddParam(DataType::Float32, { 3, 2 });
	const auto b = sg.AddParam(DataType::Float32, { 1, 2 });

	const auto matmul =
	    sg.AddNode(BinaryOpNode{ BinaryOp::MatMul, { x, 0 }, { w, 0 } }, { OutputInfo{ DataType::Float32, { 2, 2 } } });

	// 注意: b + matmul（交换顺序）
	const auto y = sg.AddNode(BinaryOpNode{ BinaryOp::Add, { b, 0 }, { matmul, 0 } },
	                           { OutputInfo{ DataType::Float32, { 2, 2 } } });

	sg.SetResults({ { y, 0 } });
	const auto fwdId = graph.AddSubgraph(std::move(sg));
	graph.SetForward(fwdId);

	// 计算未融合的结果
	Tensor<CPU> tensorX({ 1, 2, 3, 4, 5, 6 }, { 2, 3 });
	Tensor<CPU> tensorW({ 1, 2, 3, 4, 5, 6 }, { 3, 2 });
	Tensor<CPU> tensorB({ 100, 200 }, { 1, 2 });
	std::array<Tensor<CPU>, 3> inputs = { Tensor<CPU>(tensorX), Tensor<CPU>(tensorW), Tensor<CPU>(tensorB) };

	Runtime::Interpreter<CPU> interp;
	auto expected = interp.RunForward(graph, inputs);

	FusionPass fusionPass;
	fusionPass.Run(graph);

	const auto& fusedSg = graph.GetSubgraph(graph.Forward());
	EXPECT_TRUE(HasFusedOpNode(fusedSg, FusionPattern::MatMulBiasAdd));

	std::array<Tensor<CPU>, 3> inputs2 = { std::move(tensorX), std::move(tensorW), std::move(tensorB) };
	auto actual = interp.RunForward(graph, inputs2);

	ASSERT_EQ(actual.size(), 1);
	for (std::size_t i = 0; i < 4; ++i)
	{
		EXPECT_FLOAT_EQ(ReadFloat(actual[0], i), ReadFloat(expected[0], i));
	}
}

// 测试 3: MatMul 多消费者不融合
TEST(FusionPass, MatMulMultiConsumerNoFusion)
{
	Graph graph;
	Subgraph sg;

	const auto x = sg.AddParam(DataType::Float32, { 2, 3 });
	const auto w = sg.AddParam(DataType::Float32, { 3, 2 });
	const auto b = sg.AddParam(DataType::Float32, { 1, 2 });

	const auto matmul =
	    sg.AddNode(BinaryOpNode{ BinaryOp::MatMul, { x, 0 }, { w, 0 } }, { OutputInfo{ DataType::Float32, { 2, 2 } } });

	// matmul 被两个节点消费
	const auto add = sg.AddNode(BinaryOpNode{ BinaryOp::Add, { matmul, 0 }, { b, 0 } },
	                             { OutputInfo{ DataType::Float32, { 2, 2 } } });
	const auto mul = sg.AddNode(BinaryOpNode{ BinaryOp::Multiply, { matmul, 0 }, { b, 0 } },
	                             { OutputInfo{ DataType::Float32, { 2, 2 } } });

	sg.SetResults({ { add, 0 }, { mul, 0 } });
	const auto fwdId = graph.AddSubgraph(std::move(sg));
	graph.SetForward(fwdId);

	FusionPass fusionPass;
	fusionPass.Run(graph);

	const auto& fusedSg = graph.GetSubgraph(graph.Forward());
	EXPECT_FALSE(HasFusedOpNode(fusedSg, FusionPattern::MatMulBiasAdd));
}

// 测试 4: ElementWiseChain 基础
// y = Exp(Negate(x))
TEST(FusionPass, ElementWiseChainBasic)
{
	Graph graph;
	Subgraph sg;

	const auto x = sg.AddParam(DataType::Float32, { 2, 2 });

	const auto neg =
	    sg.AddNode(UnaryOpNode{ UnaryOp::Negate, { x, 0 } }, { OutputInfo{ DataType::Float32, { 2, 2 } } });
	const auto expNode =
	    sg.AddNode(UnaryOpNode{ UnaryOp::Exp, { neg, 0 } }, { OutputInfo{ DataType::Float32, { 2, 2 } } });

	sg.SetResults({ { expNode, 0 } });
	const auto fwdId = graph.AddSubgraph(std::move(sg));
	graph.SetForward(fwdId);

	// 未融合结果
	Tensor<CPU> tensorX({ 1, 2, 3, 4 }, { 2, 2 });
	std::array<Tensor<CPU>, 1> inputs = { Tensor<CPU>(tensorX) };

	Runtime::Interpreter<CPU> interp;
	auto expected = interp.RunForward(graph, inputs);

	FusionPass fusionPass;
	fusionPass.Run(graph);

	const auto& fusedSg = graph.GetSubgraph(graph.Forward());
	EXPECT_TRUE(HasFusedOpNode(fusedSg, FusionPattern::ElementWiseChain));

	std::array<Tensor<CPU>, 1> inputs2 = { std::move(tensorX) };
	auto actual = interp.RunForward(graph, inputs2);

	ASSERT_EQ(actual.size(), 1);
	for (std::size_t i = 0; i < 4; ++i)
	{
		EXPECT_FLOAT_EQ(ReadFloat(actual[0], i), ReadFloat(expected[0], i));
	}
}

// 测试 5: 较长逐元素链
// y = Abs(x * c + d)
TEST(FusionPass, ElementWiseChainLonger)
{
	Graph graph;
	Subgraph sg;

	const auto x = sg.AddParam(DataType::Float32, { 2, 2 });
	const auto c = sg.AddParam(DataType::Float32, { 2, 2 });
	const auto d = sg.AddParam(DataType::Float32, { 2, 2 });

	const auto mul = sg.AddNode(BinaryOpNode{ BinaryOp::Multiply, { x, 0 }, { c, 0 } },
	                             { OutputInfo{ DataType::Float32, { 2, 2 } } });
	const auto add = sg.AddNode(BinaryOpNode{ BinaryOp::Add, { mul, 0 }, { d, 0 } },
	                             { OutputInfo{ DataType::Float32, { 2, 2 } } });
	const auto absNode =
	    sg.AddNode(UnaryOpNode{ UnaryOp::Abs, { add, 0 } }, { OutputInfo{ DataType::Float32, { 2, 2 } } });

	sg.SetResults({ { absNode, 0 } });
	const auto fwdId = graph.AddSubgraph(std::move(sg));
	graph.SetForward(fwdId);

	Tensor<CPU> tensorX({ -1, 2, -3, 4 }, { 2, 2 });
	Tensor<CPU> tensorC({ 2, 3, 4, 5 }, { 2, 2 });
	Tensor<CPU> tensorD({ 10, -20, 30, -40 }, { 2, 2 });
	std::array<Tensor<CPU>, 3> inputs = { Tensor<CPU>(tensorX), Tensor<CPU>(tensorC), Tensor<CPU>(tensorD) };

	Runtime::Interpreter<CPU> interp;
	auto expected = interp.RunForward(graph, inputs);

	FusionPass fusionPass;
	fusionPass.Run(graph);

	const auto& fusedSg = graph.GetSubgraph(graph.Forward());
	EXPECT_TRUE(HasFusedOpNode(fusedSg, FusionPattern::ElementWiseChain));

	std::array<Tensor<CPU>, 3> inputs2 = { std::move(tensorX), std::move(tensorC), std::move(tensorD) };
	auto actual = interp.RunForward(graph, inputs2);

	ASSERT_EQ(actual.size(), 1);
	for (std::size_t i = 0; i < 4; ++i)
	{
		EXPECT_FLOAT_EQ(ReadFloat(actual[0], i), ReadFloat(expected[0], i));
	}
}

// 测试 6: FusionPass + AutogradPass 联合
// 前向 y = MatMul(x, w) + b, 验证 autograd + fusion 后前向/反向均正确
TEST(FusionPass, AfterAutogradPass)
{
	Graph graph;

	auto wVar = Variable::Create(Tensor<CPU>({ 1, 2, 3, 4, 5, 6 }, { 3, 2 }));
	const auto wIdx = graph.AddVariable(wVar);

	Subgraph sg;
	const auto x = sg.AddParam(DataType::Float32, { 2, 3 });
	const auto w = sg.AddNode(VariableRefNode{ wIdx }, { OutputInfo{ DataType::Float32, { 3, 2 } } });
	const auto b = sg.AddParam(DataType::Float32, { 1, 2 });

	const auto matmul =
	    sg.AddNode(BinaryOpNode{ BinaryOp::MatMul, { x, 0 }, { w, 0 } }, { OutputInfo{ DataType::Float32, { 2, 2 } } });
	const auto y = sg.AddNode(BinaryOpNode{ BinaryOp::Add, { matmul, 0 }, { b, 0 } },
	                           { OutputInfo{ DataType::Float32, { 2, 2 } } });

	sg.SetResults({ { y, 0 } });
	const auto fwdId = graph.AddSubgraph(std::move(sg));
	graph.SetForward(fwdId);

	// 先计算未融合的前向/反向结果
	Graph graphRef;
	auto wVarRef = Variable::Create(Tensor<CPU>({ 1, 2, 3, 4, 5, 6 }, { 3, 2 }));
	graphRef.AddVariable(wVarRef);

	{
		Subgraph sgRef;
		const auto xRef = sgRef.AddParam(DataType::Float32, { 2, 3 });
		const auto wRef = sgRef.AddNode(VariableRefNode{ 0 }, { OutputInfo{ DataType::Float32, { 3, 2 } } });
		const auto bRef = sgRef.AddParam(DataType::Float32, { 1, 2 });
		const auto mmRef = sgRef.AddNode(BinaryOpNode{ BinaryOp::MatMul, { xRef, 0 }, { wRef, 0 } },
		                                  { OutputInfo{ DataType::Float32, { 2, 2 } } });
		const auto yRef = sgRef.AddNode(BinaryOpNode{ BinaryOp::Add, { mmRef, 0 }, { bRef, 0 } },
		                                 { OutputInfo{ DataType::Float32, { 2, 2 } } });
		sgRef.SetResults({ { yRef, 0 } });
		const auto fwdRefId = graphRef.AddSubgraph(std::move(sgRef));
		graphRef.SetForward(fwdRefId);
	}

	AutogradPass autogradRef;
	autogradRef.Run(graphRef);

	Tensor<CPU> tensorX({ 1, 2, 3, 4, 5, 6 }, { 2, 3 });
	Tensor<CPU> tensorB({ 100, 200 }, { 1, 2 });

	Runtime::Interpreter<CPU> interpRef;
	std::array<Tensor<CPU>, 2> inputsRef = { Tensor<CPU>(tensorX), Tensor<CPU>(tensorB) };
	auto expectedFwd = interpRef.RunForward(graphRef, inputsRef);

	Tensor<CPU> gradOut({ 1, 1, 1, 1 }, { 2, 2 });
	std::array<Tensor<CPU>, 3> bwdInputsRef = { Tensor<CPU>(tensorX), Tensor<CPU>(tensorB), Tensor<CPU>(gradOut) };
	auto expectedBwd = interpRef.RunBackward(graphRef, bwdInputsRef);

	// 对原图运行 autograd + fusion
	AutogradPass autogradPass;
	autogradPass.Run(graph);

	FusionPass fusionPass;
	fusionPass.Run(graph);

	// 验证前向
	Runtime::Interpreter<CPU> interp;
	std::array<Tensor<CPU>, 2> inputs = { Tensor<CPU>(tensorX), Tensor<CPU>(tensorB) };
	auto actualFwd = interp.RunForward(graph, inputs);

	ASSERT_EQ(actualFwd.size(), expectedFwd.size());
	for (std::size_t i = 0; i < 4; ++i)
	{
		EXPECT_FLOAT_EQ(ReadFloat(actualFwd[0], i), ReadFloat(expectedFwd[0], i));
	}

	// 验证反向
	std::array<Tensor<CPU>, 3> bwdInputs = { std::move(tensorX), std::move(tensorB), std::move(gradOut) };
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

// 测试 7: 单操作不融合
TEST(FusionPass, SingleOpNoFusion)
{
	Graph graph;
	Subgraph sg;

	const auto x = sg.AddParam(DataType::Float32, { 2, 2 });
	const auto neg =
	    sg.AddNode(UnaryOpNode{ UnaryOp::Negate, { x, 0 } }, { OutputInfo{ DataType::Float32, { 2, 2 } } });

	sg.SetResults({ { neg, 0 } });
	const auto fwdId = graph.AddSubgraph(std::move(sg));
	graph.SetForward(fwdId);

	FusionPass fusionPass;
	fusionPass.Run(graph);

	const auto& fusedSg = graph.GetSubgraph(graph.Forward());
	EXPECT_EQ(CountFusedOpNodes(fusedSg), 0);
}

// 测试 8: 多消费者断链
// t = Negate(x); y1 = Exp(t); y2 = Abs(t)
// t 有 2 个消费者，不应该融合
TEST(FusionPass, MultiConsumerBreaksChain)
{
	Graph graph;
	Subgraph sg;

	const auto x = sg.AddParam(DataType::Float32, { 2, 2 });

	const auto neg =
	    sg.AddNode(UnaryOpNode{ UnaryOp::Negate, { x, 0 } }, { OutputInfo{ DataType::Float32, { 2, 2 } } });
	const auto expNode =
	    sg.AddNode(UnaryOpNode{ UnaryOp::Exp, { neg, 0 } }, { OutputInfo{ DataType::Float32, { 2, 2 } } });
	const auto absNode =
	    sg.AddNode(UnaryOpNode{ UnaryOp::Abs, { neg, 0 } }, { OutputInfo{ DataType::Float32, { 2, 2 } } });

	sg.SetResults({ { expNode, 0 }, { absNode, 0 } });
	const auto fwdId = graph.AddSubgraph(std::move(sg));
	graph.SetForward(fwdId);

	// 计算未融合的结果
	Tensor<CPU> tensorX({ 1, 2, 3, 4 }, { 2, 2 });
	std::array<Tensor<CPU>, 1> inputs = { Tensor<CPU>(tensorX) };

	Runtime::Interpreter<CPU> interp;
	auto expected = interp.RunForward(graph, inputs);

	FusionPass fusionPass;
	fusionPass.Run(graph);

	const auto& fusedSg = graph.GetSubgraph(graph.Forward());
	// neg 有 2 个消费者 (exp 和 abs)，不应该融合
	EXPECT_EQ(CountFusedOpNodes(fusedSg), 0);

	std::array<Tensor<CPU>, 1> inputs2 = { std::move(tensorX) };
	auto actual = interp.RunForward(graph, inputs2);

	ASSERT_EQ(actual.size(), 2);
	for (std::size_t i = 0; i < 4; ++i)
	{
		EXPECT_FLOAT_EQ(ReadFloat(actual[0], i), ReadFloat(expected[0], i));
		EXPECT_FLOAT_EQ(ReadFloat(actual[1], i), ReadFloat(expected[1], i));
	}
}
