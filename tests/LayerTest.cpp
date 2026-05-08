#include <gtest/gtest.h>

#include <LiteNN.h>
#include <LiteNN/Layer/Activation.h>
#include <LiteNN/Layer/LayerNorm.h>
#include <LiteNN/Layer/Softmax.h>
#include <LiteNN/Runtime/Interpreter.h>

#include <cmath>
#include <numeric>

using namespace LiteNN;

namespace
{
	float ReadFloat(const Tensor<CPU>& t, std::size_t i)
	{
		return static_cast<const float*>(t.RawData())[i];
	}

	// 用 Interpreter 运行只有一个参数、一个输出的前向图
	Tensor<CPU> RunSingleIO(Graph& graph, std::vector<float> inputData, std::vector<std::size_t> shape)
	{
		Runtime::Interpreter<CPU> interp;
		Tensor<CPU> inputTensor = Optimizer::MakeFloatTensor(std::span<const float>(inputData), shape);
		std::array<Tensor<CPU>, 1> inputs = { std::move(inputTensor) };
		auto results = interp.RunForward(graph, inputs);
		EXPECT_EQ(results.size(), 1u);
		return std::move(results[0]);
	}
} // namespace

// ─────────────────────────────────────────────
//  Softmax 测试
// ─────────────────────────────────────────────

TEST(LayerSoftmax, OutputSumsToOneAlongAxis1)
{
	// 2×3 输入，沿 axis=1 做 softmax，每行之和应为 1
	Graph graph;
	Subgraph sg;
	const auto input = sg.AddParam(DataType::Float32, { 2, 3 });
	const auto out = Layer::AddSoftmax(sg, { input, 0 }, 1);
	sg.SetResults({ out });
	graph.SetForward(graph.AddSubgraph(std::move(sg)));

	const auto result = RunSingleIO(graph, { 1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f }, { 2, 3 });

	ASSERT_EQ(result.Shape().NumElements(), 6u);
	const float row0 = ReadFloat(result, 0) + ReadFloat(result, 1) + ReadFloat(result, 2);
	const float row1 = ReadFloat(result, 3) + ReadFloat(result, 4) + ReadFloat(result, 5);
	EXPECT_NEAR(row0, 1.0f, 1e-5f);
	EXPECT_NEAR(row1, 1.0f, 1e-5f);
}

TEST(LayerSoftmax, AllElementsPositive)
{
	// softmax 输出所有值应大于 0（使用数值范围不触发 float32 下溢的输入）
	Graph graph;
	Subgraph sg;
	const auto input = sg.AddParam(DataType::Float32, { 1, 4 });
	const auto out = Layer::AddSoftmax(sg, { input, 0 }, 1);
	sg.SetResults({ out });
	graph.SetForward(graph.AddSubgraph(std::move(sg)));

	// 差值在 float32 可表示范围内（< 80）
	const auto result = RunSingleIO(graph, { -3.0f, 0.0f, 3.0f, 10.0f }, { 1, 4 });

	for (std::size_t i = 0; i < 4; ++i)
	{
		EXPECT_GT(ReadFloat(result, i), 0.0f) << "element " << i << " should be positive";
	}
}

TEST(LayerSoftmax, NumericallyStableForLargeInput)
{
	// 含极端值时不应溢出/下溢
	Graph graph;
	Subgraph sg;
	const auto input = sg.AddParam(DataType::Float32, { 1, 3 });
	const auto out = Layer::AddSoftmax(sg, { input, 0 }, 1);
	sg.SetResults({ out });
	graph.SetForward(graph.AddSubgraph(std::move(sg)));

	const auto result = RunSingleIO(graph, { 1e30f, 1e30f, 1e30f }, { 1, 3 });

	// 相等输入 → 均匀分布
	for (std::size_t i = 0; i < 3; ++i)
	{
		EXPECT_TRUE(std::isfinite(ReadFloat(result, i))) << "element " << i << " should be finite";
	}
	const float sum = ReadFloat(result, 0) + ReadFloat(result, 1) + ReadFloat(result, 2);
	EXPECT_NEAR(sum, 1.0f, 1e-5f);
}

// ─────────────────────────────────────────────
//  GELU 测试
// ─────────────────────────────────────────────

TEST(LayerGELU, ZeroInputGivesZero)
{
	Graph graph;
	Subgraph sg;
	const auto input = sg.AddParam(DataType::Float32, { 1 });
	const auto out = Layer::AddGELU(sg, { input, 0 });
	sg.SetResults({ out });
	graph.SetForward(graph.AddSubgraph(std::move(sg)));

	const auto result = RunSingleIO(graph, { 0.0f }, { 1 });
	EXPECT_NEAR(ReadFloat(result, 0), 0.0f, 1e-5f);
}

TEST(LayerGELU, PositiveInputApproximation)
{
	// GELU(1.0) ≈ 0.8413 (tanh 近似)
	Graph graph;
	Subgraph sg;
	const auto input = sg.AddParam(DataType::Float32, { 1 });
	const auto out = Layer::AddGELU(sg, { input, 0 });
	sg.SetResults({ out });
	graph.SetForward(graph.AddSubgraph(std::move(sg)));

	const auto result = RunSingleIO(graph, { 1.0f }, { 1 });
	EXPECT_NEAR(ReadFloat(result, 0), 0.8413f, 1e-3f);
}

TEST(LayerGELU, NegativeInputDampened)
{
	// GELU 对负值有衰减，结果应在 (-0.17, 0)
	Graph graph;
	Subgraph sg;
	const auto input = sg.AddParam(DataType::Float32, { 1 });
	const auto out = Layer::AddGELU(sg, { input, 0 });
	sg.SetResults({ out });
	graph.SetForward(graph.AddSubgraph(std::move(sg)));

	const auto result = RunSingleIO(graph, { -1.0f }, { 1 });
	const float v = ReadFloat(result, 0);
	EXPECT_LT(v, 0.0f) << "GELU(-1) should be negative";
	EXPECT_GT(v, -0.2f) << "GELU(-1) should be near 0, got " << v;
}

// ─────────────────────────────────────────────
//  ELU 测试
// ─────────────────────────────────────────────

TEST(LayerELU, PositiveInputPassthrough)
{
	// ELU(x>0) = x
	Graph graph;
	Subgraph sg;
	const auto input = sg.AddParam(DataType::Float32, { 3 });
	const auto out = Layer::AddELU(sg, { input, 0 });
	sg.SetResults({ out });
	graph.SetForward(graph.AddSubgraph(std::move(sg)));

	const auto result = RunSingleIO(graph, { 0.5f, 1.0f, 2.0f }, { 3 });
	EXPECT_NEAR(ReadFloat(result, 0), 0.5f, 1e-5f);
	EXPECT_NEAR(ReadFloat(result, 1), 1.0f, 1e-5f);
	EXPECT_NEAR(ReadFloat(result, 2), 2.0f, 1e-5f);
}

TEST(LayerELU, ZeroInputGivesZero)
{
	Graph graph;
	Subgraph sg;
	const auto input = sg.AddParam(DataType::Float32, { 1 });
	const auto out = Layer::AddELU(sg, { input, 0 });
	sg.SetResults({ out });
	graph.SetForward(graph.AddSubgraph(std::move(sg)));

	const auto result = RunSingleIO(graph, { 0.0f }, { 1 });
	EXPECT_NEAR(ReadFloat(result, 0), 0.0f, 1e-5f);
}

TEST(LayerELU, NegativeInputExponential)
{
	// ELU(-1, alpha=1) = exp(-1) - 1 ≈ -0.6321
	Graph graph;
	Subgraph sg;
	const auto input = sg.AddParam(DataType::Float32, { 1 });
	const auto out = Layer::AddELU(sg, { input, 0 }, 1.0);
	sg.SetResults({ out });
	graph.SetForward(graph.AddSubgraph(std::move(sg)));

	const auto result = RunSingleIO(graph, { -1.0f }, { 1 });
	const float expected = std::exp(-1.0f) - 1.0f; // ≈ -0.6321
	EXPECT_NEAR(ReadFloat(result, 0), expected, 1e-4f);
}

TEST(LayerELU, NegativeInputWithCustomAlpha)
{
	// ELU(-2, alpha=2) = 2*(exp(-2) - 1) ≈ -1.7293
	Graph graph;
	Subgraph sg;
	const auto input = sg.AddParam(DataType::Float32, { 1 });
	const auto out = Layer::AddELU(sg, { input, 0 }, 2.0);
	sg.SetResults({ out });
	graph.SetForward(graph.AddSubgraph(std::move(sg)));

	const auto result = RunSingleIO(graph, { -2.0f }, { 1 });
	const float expected = 2.0f * (std::exp(-2.0f) - 1.0f);
	EXPECT_NEAR(ReadFloat(result, 0), expected, 1e-4f);
}

// ─────────────────────────────────────────────
//  LayerNorm 测试
// ─────────────────────────────────────────────

TEST(LayerLayerNorm, OutputMeanNearZero)
{
	// gamma=1, beta=0 → 输出均值应约为 0
	Graph graph;
	const auto norm = Layer::CreateLayerNorm(graph, 4);

	Subgraph sg;
	const auto input = sg.AddParam(DataType::Float32, { 1, 4 });
	const auto out = Layer::AddLayerNorm(sg, norm, { input, 0 });
	sg.SetResults({ out });
	graph.SetForward(graph.AddSubgraph(std::move(sg)));

	const auto result = RunSingleIO(graph, { 1.0f, 2.0f, 3.0f, 4.0f }, { 1, 4 });
	ASSERT_EQ(result.Shape().NumElements(), 4u);

	float mean = 0.0f;
	for (std::size_t i = 0; i < 4; ++i)
	{
		mean += ReadFloat(result, i);
	}
	mean /= 4.0f;
	EXPECT_NEAR(mean, 0.0f, 1e-5f);
}

TEST(LayerLayerNorm, OutputVarianceNearOne)
{
	// gamma=1, beta=0 → 输出方差应约为 1
	Graph graph;
	const auto norm = Layer::CreateLayerNorm(graph, 4);

	Subgraph sg;
	const auto input = sg.AddParam(DataType::Float32, { 1, 4 });
	const auto out = Layer::AddLayerNorm(sg, norm, { input, 0 });
	sg.SetResults({ out });
	graph.SetForward(graph.AddSubgraph(std::move(sg)));

	const auto result = RunSingleIO(graph, { 10.0f, 20.0f, 30.0f, 40.0f }, { 1, 4 });

	float mean = 0.0f;
	for (std::size_t i = 0; i < 4; ++i)
	{
		mean += ReadFloat(result, i);
	}
	mean /= 4.0f;

	float var = 0.0f;
	for (std::size_t i = 0; i < 4; ++i)
	{
		const float d = ReadFloat(result, i) - mean;
		var += d * d;
	}
	var /= 4.0f;
	// 无偏方差 ≈ 1（eps=1e-5 引起极小偏差）
	EXPECT_NEAR(var, 1.0f, 1e-3f);
}

TEST(LayerLayerNorm, BetaShiftsOutput)
{
	// 将 beta 设为 5 → 输出均值应约为 5
	Graph graph;

	// 手动构造 LayerNormLayer，beta 初始化为 5
	Layer::LayerNormLayer norm;
	norm.featureSize = 3;
	norm.dtype = DataType::Float32;
	norm.eps = 1e-5;
	norm.gammaVariable = graph.AddVariable(Variable::Create(Tensor<CPU>({ 1.0f, 1.0f, 1.0f }, { 1, 3 })));
	norm.betaVariable = graph.AddVariable(Variable::Create(Tensor<CPU>({ 5.0f, 5.0f, 5.0f }, { 1, 3 })));

	Subgraph sg;
	const auto input = sg.AddParam(DataType::Float32, { 1, 3 });
	const auto out = Layer::AddLayerNorm(sg, norm, { input, 0 });
	sg.SetResults({ out });
	graph.SetForward(graph.AddSubgraph(std::move(sg)));

	const auto result = RunSingleIO(graph, { 1.0f, 2.0f, 3.0f }, { 1, 3 });

	float mean = 0.0f;
	for (std::size_t i = 0; i < 3; ++i)
	{
		mean += ReadFloat(result, i);
	}
	mean /= 3.0f;
	EXPECT_NEAR(mean, 5.0f, 1e-4f);
}
