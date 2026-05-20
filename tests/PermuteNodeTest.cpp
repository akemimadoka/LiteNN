#include <gtest/gtest.h>

#include <LiteNN.h>
#include <LiteNN/Layer/Permute.h>
#include <LiteNN/Pass/AutogradPass.h>
#include <LiteNN/Pass/ConstFoldPass.h>
#include <LiteNN/Runtime/Interpreter.h>

#include <array>
#include <vector>

using namespace LiteNN;

static float ReadFloat(const Tensor<CPU>& t, std::size_t i)
{
	return static_cast<const float*>(t.RawData())[i];
}

// ============================================================
// 前向
// ============================================================

// (2,3) permutation=[1,0] → (3,2) 等价于矩阵转置
TEST(PermuteNode, Forward_Transpose2D)
{
	Graph graph;
	Subgraph sg;
	const auto x = sg.AddParam(DataType::Float32, { 2, 3 });
	const auto y = sg.AddNode(PermuteNode{ { x, 0 }, { 1, 0 } },
	                          { OutputInfo{ DataType::Float32, { 3, 2 } } });
	sg.SetResults({ { y, 0 } });
	graph.SetForward(graph.AddSubgraph(std::move(sg)));

	Tensor<CPU> input({ 1, 2, 3, 4, 5, 6 }, { 2, 3 });
	std::array<Tensor<CPU>, 1> inputs = { std::move(input) };

	Runtime::Interpreter<CPU> interp;
	auto results = interp.RunForward(graph, inputs);
	ASSERT_EQ(results.size(), 1);
	ASSERT_EQ(results[0].NumElements(), 6);
	// 原矩阵 [[1,2,3],[4,5,6]] 转置后 [[1,4],[2,5],[3,6]]
	const float expected[] = { 1, 4, 2, 5, 3, 6 };
	for (std::size_t i = 0; i < 6; ++i)
	{
		EXPECT_FLOAT_EQ(ReadFloat(results[0], i), expected[i]);
	}
}

// (2,3,4) permutation=[2,0,1]
TEST(PermuteNode, Forward_Permute3D)
{
	Graph graph;
	Subgraph sg;
	const auto x = sg.AddParam(DataType::Float32, { 2, 3, 4 });
	const auto y = sg.AddNode(PermuteNode{ { x, 0 }, { 2, 0, 1 } },
	                          { OutputInfo{ DataType::Float32, { 4, 2, 3 } } });
	sg.SetResults({ { y, 0 } });
	graph.SetForward(graph.AddSubgraph(std::move(sg)));

	std::vector<double> data(24);
	for (std::size_t i = 0; i < 24; ++i) data[i] = static_cast<double>(i);
	Tensor<CPU> input(std::span<const double>(data), { 2, 3, 4 });
	std::array<Tensor<CPU>, 1> inputs = { std::move(input) };

	Runtime::Interpreter<CPU> interp;
	auto results = interp.RunForward(graph, inputs);
	ASSERT_EQ(results.size(), 1);
	ASSERT_EQ(results[0].NumElements(), 24);

	// out[k,n,c] = in[n,c,k]
	// in 的线性偏移 = n*12 + c*4 + k
	for (std::size_t k = 0; k < 4; ++k)
	{
		for (std::size_t n = 0; n < 2; ++n)
		{
			for (std::size_t c = 0; c < 3; ++c)
			{
				const auto outIdx = k * 6 + n * 3 + c;
				const auto inIdx = n * 12 + c * 4 + k;
				EXPECT_FLOAT_EQ(ReadFloat(results[0], outIdx), static_cast<float>(inIdx));
			}
		}
	}
}

// Identity permutation
TEST(PermuteNode, Forward_Identity)
{
	Graph graph;
	Subgraph sg;
	const auto x = sg.AddParam(DataType::Float32, { 2, 3 });
	const auto y = sg.AddNode(PermuteNode{ { x, 0 }, { 0, 1 } },
	                          { OutputInfo{ DataType::Float32, { 2, 3 } } });
	sg.SetResults({ { y, 0 } });
	graph.SetForward(graph.AddSubgraph(std::move(sg)));

	Tensor<CPU> input({ 1, 2, 3, 4, 5, 6 }, { 2, 3 });
	std::array<Tensor<CPU>, 1> inputs = { std::move(input) };

	Runtime::Interpreter<CPU> interp;
	auto results = interp.RunForward(graph, inputs);
	for (std::size_t i = 0; i < 6; ++i)
	{
		EXPECT_FLOAT_EQ(ReadFloat(results[0], i), static_cast<float>(i + 1));
	}
}

// Layer helper
TEST(PermuteNode, Layer_AddTranspose)
{
	Graph graph;
	Subgraph sg;
	const auto x = sg.AddParam(DataType::Float32, { 2, 3 });
	const auto y = Layer::AddTranspose(sg, { x, 0 });
	sg.SetResults({ y });
	graph.SetForward(graph.AddSubgraph(std::move(sg)));

	Tensor<CPU> input({ 1, 2, 3, 4, 5, 6 }, { 2, 3 });
	std::array<Tensor<CPU>, 1> inputs = { std::move(input) };

	Runtime::Interpreter<CPU> interp;
	auto results = interp.RunForward(graph, inputs);
	ASSERT_EQ(results[0].Shape().Dims[0], 3uz);
	ASSERT_EQ(results[0].Shape().Dims[1], 2uz);
	const float expected[] = { 1, 4, 2, 5, 3, 6 };
	for (std::size_t i = 0; i < 6; ++i)
	{
		EXPECT_FLOAT_EQ(ReadFloat(results[0], i), expected[i]);
	}
}

// ============================================================
// 反向：Permute 梯度 = 用逆置换再做一次 Permute
// ============================================================

TEST(PermuteNode, Backward_Transpose2D)
{
	Graph graph;
	Subgraph sg;
	const auto x = sg.AddParam(DataType::Float32, { 2, 3 });
	const auto y = sg.AddNode(PermuteNode{ { x, 0 }, { 1, 0 } },
	                          { OutputInfo{ DataType::Float32, { 3, 2 } } });
	sg.SetResults({ { y, 0 } });
	graph.SetForward(graph.AddSubgraph(std::move(sg)));

	AutogradPass autograd;
	autograd.Run(graph);

	Runtime::Interpreter<CPU> interp;
	std::vector<Tensor<CPU>> fwdInputs;
	fwdInputs.emplace_back(Tensor<CPU>({ 1, 2, 3, 4, 5, 6 }, { 2, 3 }));
	(void)interp.RunForward(graph, fwdInputs);

	std::vector<Tensor<CPU>> bwdInputs;
	bwdInputs.emplace_back(Tensor<CPU>({ 1, 2, 3, 4, 5, 6 }, { 2, 3 }));
	bwdInputs.emplace_back(Tensor<CPU>({ 10, 20, 30, 40, 50, 60 }, { 3, 2 }));
	auto gradients = interp.RunBackward(graph, bwdInputs);

	ASSERT_EQ(gradients.size(), 1);
	ASSERT_EQ(gradients[0].NumElements(), 6);
	// gradY shape (3,2) = [[10,20],[30,40],[50,60]]
	// dx[i,j] = gradY[j,i] (逆置换 [1,0] = [1,0])
	// dx = [[10,30,50],[20,40,60]]
	const float expected[] = { 10, 30, 50, 20, 40, 60 };
	for (std::size_t i = 0; i < 6; ++i)
	{
		EXPECT_FLOAT_EQ(ReadFloat(gradients[0], i), expected[i]);
	}
}

// ============================================================
// 常量折叠：PermuteNode 输入是常量 → 应被折叠
// ============================================================

TEST(PermuteNode, ConstFold_Transpose)
{
	Graph graph;
	Subgraph sg;
	const auto c = sg.AddNode(
	    ConstantNode{ Tensor<CPU>({ 1, 2, 3, 4, 5, 6 }, { 2, 3 }).CopyToDevice(PolymorphicDevice{ CPU{} }) },
	    { OutputInfo{ DataType::Float32, { 2, 3 } } });
	const auto y = sg.AddNode(PermuteNode{ { c, 0 }, { 1, 0 } },
	                          { OutputInfo{ DataType::Float32, { 3, 2 } } });
	sg.SetResults({ { y, 0 } });
	graph.SetForward(graph.AddSubgraph(std::move(sg)));

	ConstFoldPass pass;
	pass.Run(graph);

	std::vector<Tensor<CPU>> inputs;
	Runtime::Interpreter<CPU> interp;
	auto results = interp.RunForward(graph, inputs);
	const float expected[] = { 1, 4, 2, 5, 3, 6 };
	for (std::size_t i = 0; i < 6; ++i)
	{
		EXPECT_FLOAT_EQ(ReadFloat(results[0], i), expected[i]);
	}
}
