#include <gtest/gtest.h>

#include <LiteNN.h>
#include <LiteNN/Layer/Activation.h>
#include <LiteNN/Layer/CausalMask.h>
#include <LiteNN/Layer/KVCache.h>
#include <LiteNN/Layer/LayerNorm.h>
#include <LiteNN/Layer/RMSNorm.h>
#include <LiteNN/Layer/RoPE.h>
#include <LiteNN/Layer/Softmax.h>
#include <LiteNN/Layer/SwiGLU.h>
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

		std::vector<Tensor<CPU>> RunWithInputs(Graph& graph, std::vector<Tensor<CPU>> inputs)
		{
			Runtime::Interpreter<CPU> interp;
			return interp.RunForward(graph, inputs);
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

// ─────────────────────────────────────────────
//  RMSNorm 测试
// ─────────────────────────────────────────────

TEST(LayerRMSNorm, OutputMeanSquareNearOne)
{
	Graph graph;
	const auto norm = Layer::CreateRMSNorm(graph, 4);

	Subgraph sg;
	const auto input = sg.AddParam(DataType::Float32, { 1, 4 });
	const auto out = Layer::AddRMSNorm(sg, norm, { input, 0 });
	sg.SetResults({ out });
	graph.SetForward(graph.AddSubgraph(std::move(sg)));

	const auto result = RunSingleIO(graph, { 1.0f, 2.0f, 3.0f, 4.0f }, { 1, 4 });

	float meanSquare = 0.0f;
	for (std::size_t i = 0; i < 4; ++i)
	{
		const auto value = ReadFloat(result, i);
		meanSquare += value * value;
	}
	meanSquare /= 4.0f;
	EXPECT_NEAR(meanSquare, 1.0f, 1e-4f);
}

TEST(LayerRMSNorm, ZeroInputStaysZero)
{
	Graph graph;
	const auto norm = Layer::CreateRMSNorm(graph, 3);

	Subgraph sg;
	const auto input = sg.AddParam(DataType::Float32, { 1, 3 });
	const auto out = Layer::AddRMSNorm(sg, norm, { input, 0 });
	sg.SetResults({ out });
	graph.SetForward(graph.AddSubgraph(std::move(sg)));

	const auto result = RunSingleIO(graph, { 0.0f, 0.0f, 0.0f }, { 1, 3 });
	EXPECT_NEAR(ReadFloat(result, 0), 0.0f, 1e-6f);
	EXPECT_NEAR(ReadFloat(result, 1), 0.0f, 1e-6f);
	EXPECT_NEAR(ReadFloat(result, 2), 0.0f, 1e-6f);
}

TEST(LayerRMSNorm, WeightScalesNormalizedOutput)
{
	Graph graph;

	Layer::RMSNormLayer norm;
	norm.featureSize = 3;
	norm.dtype = DataType::Float32;
	norm.eps = 1e-6;
	norm.weightVariable = graph.AddVariable(
	    Variable::Create(Tensor<CPU>({ 2.0f, 2.0f, 2.0f }, { 1, 3 })));

	Subgraph sg;
	const auto input = sg.AddParam(DataType::Float32, { 1, 3 });
	const auto out = Layer::AddRMSNorm(sg, norm, { input, 0 });
	sg.SetResults({ out });
	graph.SetForward(graph.AddSubgraph(std::move(sg)));

	const auto result = RunSingleIO(graph, { 1.0f, 2.0f, 3.0f }, { 1, 3 });

	const float denom = std::sqrt((1.0f + 4.0f + 9.0f) / 3.0f + 1e-6f);
	EXPECT_NEAR(ReadFloat(result, 0), 2.0f * 1.0f / denom, 1e-5f);
	EXPECT_NEAR(ReadFloat(result, 1), 2.0f * 2.0f / denom, 1e-5f);
	EXPECT_NEAR(ReadFloat(result, 2), 2.0f * 3.0f / denom, 1e-5f);
}

// ─────────────────────────────────────────────
//  RoPE 测试
// ─────────────────────────────────────────────

TEST(LayerRoPE, PositionZeroIsIdentity)
{
	Graph graph;
	Subgraph sg;
	const auto input = sg.AddParam(DataType::Float32, { 1, 4 });
	const auto out = Layer::AddRoPE(sg, { input, 0 });
	sg.SetResults({ out });
	graph.SetForward(graph.AddSubgraph(std::move(sg)));

	const auto result = RunSingleIO(graph, { 1.0f, 2.0f, 3.0f, 4.0f }, { 1, 4 });
	EXPECT_NEAR(ReadFloat(result, 0), 1.0f, 1e-5f);
	EXPECT_NEAR(ReadFloat(result, 1), 2.0f, 1e-5f);
	EXPECT_NEAR(ReadFloat(result, 2), 3.0f, 1e-5f);
	EXPECT_NEAR(ReadFloat(result, 3), 4.0f, 1e-5f);
}

TEST(LayerRoPE, RotatesPairsAtPositionOne)
{
	Graph graph;
	Subgraph sg;
	const auto input = sg.AddParam(DataType::Float32, { 2, 4 });
	const auto out = Layer::AddRoPE(sg, { input, 0 }, 1.0);
	sg.SetResults({ out });
	graph.SetForward(graph.AddSubgraph(std::move(sg)));

	const auto result = RunSingleIO(graph, { 0.0f, 0.0f, 0.0f, 0.0f, 1.0f, 0.0f, 0.0f, 1.0f }, { 2, 4 });
	const auto cos1 = std::cos(1.0f);
	const auto sin1 = std::sin(1.0f);
	EXPECT_NEAR(ReadFloat(result, 4), cos1, 1e-5f);
	EXPECT_NEAR(ReadFloat(result, 5), sin1, 1e-5f);
	EXPECT_NEAR(ReadFloat(result, 6), -sin1, 1e-5f);
	EXPECT_NEAR(ReadFloat(result, 7), cos1, 1e-5f);
}

TEST(LayerRoPE, RejectsOddFeatureSize)
{
	Graph graph;
	Subgraph sg;
	const auto input = sg.AddParam(DataType::Float32, { 2, 3 });
	EXPECT_THROW(static_cast<void>(Layer::AddRoPE(sg, { input, 0 })), std::runtime_error);
}

TEST(LayerSiLU, MatchesAnalyticValue)
{
	Graph graph;
	Subgraph sg;
	const auto input = sg.AddParam(DataType::Float32, { 1 });
	const auto out = Layer::AddSiLU(sg, { input, 0 });
	sg.SetResults({ out });
	graph.SetForward(graph.AddSubgraph(std::move(sg)));

	const auto result = RunSingleIO(graph, { 2.0f }, { 1 });
	const auto expected = 2.0f / (1.0f + std::exp(-2.0f));
	EXPECT_NEAR(ReadFloat(result, 0), expected, 1e-5f);
}

TEST(LayerSwiGLU, IdentityProjectionsMatchAnalyticResult)
{
	Graph graph;
	const auto layer = Layer::CreateSwiGLUMLP(
	    graph,
	    Tensor<CPU>({ 1.0f, 0.0f, 0.0f, 1.0f }, { 2, 2 }),
	    Tensor<CPU>({ 1.0f, 0.0f, 0.0f, 1.0f }, { 2, 2 }),
	    Tensor<CPU>({ 1.0f, 0.0f, 0.0f, 1.0f }, { 2, 2 }));

	Subgraph sg;
	const auto input = sg.AddParam(DataType::Float32, { 1, 2 });
	const auto out = Layer::AddSwiGLUMLP(sg, layer, { input, 0 });
	sg.SetResults({ out });
	graph.SetForward(graph.AddSubgraph(std::move(sg)));

	const auto result = RunSingleIO(graph, { 1.0f, 2.0f }, { 1, 2 });
	EXPECT_NEAR(ReadFloat(result, 0), 1.0f * (1.0f / (1.0f + std::exp(-1.0f))) * 1.0f, 1e-5f);
	EXPECT_NEAR(ReadFloat(result, 1), 2.0f * (1.0f / (1.0f + std::exp(-2.0f))) * 2.0f, 1e-5f);
}

TEST(LayerSwiGLU, DownProjectionChangesOutputWidth)
{
	Graph graph;
	const auto layer = Layer::CreateSwiGLUMLP(
	    graph,
	    Tensor<CPU>({ 1.0f, 0.0f, 0.0f, 1.0f }, { 2, 2 }),
	    Tensor<CPU>({ 1.0f, 0.0f, 0.0f, 1.0f }, { 2, 2 }),
	    Tensor<CPU>({ 1.0f, 1.0f }, { 2, 1 }));

	Subgraph sg;
	const auto input = sg.AddParam(DataType::Float32, { 1, 2 });
	const auto out = Layer::AddSwiGLUMLP(sg, layer, { input, 0 });
	sg.SetResults({ out });
	graph.SetForward(graph.AddSubgraph(std::move(sg)));

	const auto result = RunSingleIO(graph, { 1.0f, 2.0f }, { 1, 2 });
	ASSERT_EQ(result.Shape().NumElements(), 1u);
	const auto expected0 = 1.0f * (1.0f / (1.0f + std::exp(-1.0f))) * 1.0f;
	const auto expected1 = 2.0f * (1.0f / (1.0f + std::exp(-2.0f))) * 2.0f;
	EXPECT_NEAR(ReadFloat(result, 0), expected0 + expected1, 1e-5f);
}

TEST(LayerSwiGLU, RejectsMismatchedHiddenSizes)
{
	Graph graph;
	EXPECT_THROW(static_cast<void>(Layer::CreateSwiGLUMLP(
	                 graph,
	                 Tensor<CPU>({ 1.0f, 0.0f, 0.0f, 1.0f }, { 2, 2 }),
	                 Tensor<CPU>({ 1.0f, 0.0f, 0.0f, 1.0f, 0.0f, 1.0f }, { 2, 3 }),
	                 Tensor<CPU>({ 1.0f, 1.0f }, { 2, 1 }))),
	             std::runtime_error);
}

TEST(LayerCausalMask, PreservesDiagonalAndLowerTriangle)
{
	Graph graph;
	Subgraph sg;
	const auto input = sg.AddParam(DataType::Float32, { 3, 3 });
	const auto out = Layer::AddCausalMask(sg, { input, 0 });
	sg.SetResults({ out });
	graph.SetForward(graph.AddSubgraph(std::move(sg)));

	const auto result = RunSingleIO(graph,
	                                { 1.0f, 2.0f, 3.0f,
	                                  4.0f, 5.0f, 6.0f,
	                                  7.0f, 8.0f, 9.0f },
	                                { 3, 3 });
	EXPECT_NEAR(ReadFloat(result, 0), 1.0f, 1e-5f);
	EXPECT_NEAR(ReadFloat(result, 3), 4.0f, 1e-5f);
	EXPECT_NEAR(ReadFloat(result, 4), 5.0f, 1e-5f);
	EXPECT_NEAR(ReadFloat(result, 6), 7.0f, 1e-5f);
	EXPECT_NEAR(ReadFloat(result, 7), 8.0f, 1e-5f);
	EXPECT_NEAR(ReadFloat(result, 8), 9.0f, 1e-5f);
}

TEST(LayerCausalMask, MasksStrictUpperTriangle)
{
	Graph graph;
	Subgraph sg;
	const auto input = sg.AddParam(DataType::Float32, { 3, 3 });
	const auto out = Layer::AddCausalMask(sg, { input, 0 });
	sg.SetResults({ out });
	graph.SetForward(graph.AddSubgraph(std::move(sg)));

	const auto result = RunSingleIO(graph,
	                                { 1.0f, 2.0f, 3.0f,
	                                  4.0f, 5.0f, 6.0f,
	                                  7.0f, 8.0f, 9.0f },
	                                { 3, 3 });
	EXPECT_LT(ReadFloat(result, 1), -1.0e8f);
	EXPECT_LT(ReadFloat(result, 2), -1.0e8f);
	EXPECT_LT(ReadFloat(result, 5), -1.0e8f);
}

TEST(LayerCausalMask, RejectsNonSquareMatrix)
{
	Graph graph;
	Subgraph sg;
	const auto input = sg.AddParam(DataType::Float32, { 2, 3 });
	EXPECT_THROW(static_cast<void>(Layer::AddCausalMask(sg, { input, 0 })), std::runtime_error);
}

TEST(LayerKVCache, AppendConcatenatesPastAndPresent)
{
	Graph graph;
	Subgraph sg;
	const auto pastKeys = sg.AddParam(DataType::Float32, { 2, 2 });
	const auto pastValues = sg.AddParam(DataType::Float32, { 2, 3 });
	const auto newKeys = sg.AddParam(DataType::Float32, { 1, 2 });
	const auto newValues = sg.AddParam(DataType::Float32, { 1, 3 });
	const auto updated = Layer::AddKVCacheAppend(sg, { { pastKeys, 0 }, { pastValues, 0 } },
	                                          { { newKeys, 0 }, { newValues, 0 } });
	sg.SetResults({ updated.keys, updated.values });
	graph.SetForward(graph.AddSubgraph(std::move(sg)));

	auto results = RunWithInputs(graph,
	                            {
	                                Tensor<CPU>({ 1.0f, 2.0f, 3.0f, 4.0f }, { 2, 2 }),
	                                Tensor<CPU>({ 10.0f, 11.0f, 12.0f, 13.0f, 14.0f, 15.0f }, { 2, 3 }),
	                                Tensor<CPU>({ 5.0f, 6.0f }, { 1, 2 }),
	                                Tensor<CPU>({ 16.0f, 17.0f, 18.0f }, { 1, 3 }),
	                            });

	ASSERT_EQ(results.size(), 2u);
	EXPECT_EQ(results[0].Shape().NumElements(), 6u);
	EXPECT_EQ(results[1].Shape().NumElements(), 9u);
	EXPECT_NEAR(ReadFloat(results[0], 0), 1.0f, 1e-5f);
	EXPECT_NEAR(ReadFloat(results[0], 4), 5.0f, 1e-5f);
	EXPECT_NEAR(ReadFloat(results[0], 5), 6.0f, 1e-5f);
	EXPECT_NEAR(ReadFloat(results[1], 6), 16.0f, 1e-5f);
	EXPECT_NEAR(ReadFloat(results[1], 7), 17.0f, 1e-5f);
	EXPECT_NEAR(ReadFloat(results[1], 8), 18.0f, 1e-5f);
}

TEST(LayerKVCache, ViewReturnsRequestedWindow)
{
	Graph graph;
	Subgraph sg;
	const auto keys = sg.AddParam(DataType::Float32, { 3, 2 });
	const auto values = sg.AddParam(DataType::Float32, { 3, 3 });
	const auto window = Layer::AddKVCacheView(sg, { { keys, 0 }, { values, 0 } }, 1, 2);
	sg.SetResults({ window.keys, window.values });
	graph.SetForward(graph.AddSubgraph(std::move(sg)));

	auto results = RunWithInputs(graph,
	                            {
	                                Tensor<CPU>({ 1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f }, { 3, 2 }),
	                                Tensor<CPU>({ 10.0f, 11.0f, 12.0f, 13.0f, 14.0f, 15.0f, 16.0f, 17.0f, 18.0f },
	                                            { 3, 3 }),
	                            });

	ASSERT_EQ(results.size(), 2u);
	EXPECT_EQ(results[0].Shape().NumElements(), 4u);
	EXPECT_EQ(results[1].Shape().NumElements(), 6u);
	EXPECT_NEAR(ReadFloat(results[0], 0), 3.0f, 1e-5f);
	EXPECT_NEAR(ReadFloat(results[0], 3), 6.0f, 1e-5f);
	EXPECT_NEAR(ReadFloat(results[1], 0), 13.0f, 1e-5f);
	EXPECT_NEAR(ReadFloat(results[1], 5), 18.0f, 1e-5f);
}

TEST(LayerKVCache, RejectsMismatchedSequenceLengths)
{
	Graph graph;
	Subgraph sg;
	const auto keys = sg.AddParam(DataType::Float32, { 2, 2 });
	const auto values = sg.AddParam(DataType::Float32, { 3, 3 });
	EXPECT_THROW(static_cast<void>(Layer::AddKVCacheView(sg, { { keys, 0 }, { values, 0 } }, 0, 2)),
	             std::runtime_error);
}
