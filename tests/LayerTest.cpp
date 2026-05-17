#include <gtest/gtest.h>

#include <LiteNN.h>
#include <LiteNN/Layer/AddId.h>
#include <LiteNN/Layer/Activation.h>
#include <LiteNN/Layer/Argsort.h>
#include <LiteNN/Layer/Arange.h>
#include <LiteNN/Layer/CausalMask.h>
#include <LiteNN/Layer/Cumsum.h>
#include <LiteNN/Layer/FlashAttnExt.h>
#include <LiteNN/Layer/GroupNorm.h>
#include <LiteNN/Layer/KVCache.h>
#include <LiteNN/Layer/L2Norm.h>
#include <LiteNN/Layer/LayerNorm.h>
#include <LiteNN/Layer/MulMatId.h>
#include <LiteNN/Layer/Pad.h>
#include <LiteNN/Layer/RMSNorm.h>
#include <LiteNN/Layer/Roll.h>
#include <LiteNN/Layer/RoPE.h>
#include <LiteNN/Layer/Softmax.h>
#include <LiteNN/Layer/SumRows.h>
#include <LiteNN/Layer/SwiGLU.h>
#include <LiteNN/Layer/TopK.h>
#include <LiteNN/Runtime/Interpreter.h>

#include <cmath>
#include <cstdint>
#include <limits>
#include <numeric>
#include <optional>

using namespace LiteNN;

namespace
{
	float ReadFloat(const Tensor<CPU>& t, std::size_t i)
	{
		return static_cast<const float*>(t.RawData())[i];
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

		std::int32_t ReadInt32(const Tensor<CPU>& t, std::size_t i)
		{
			return static_cast<const std::int32_t*>(t.RawData())[i];
		}

		struct FlashAttnReferenceOptions
		{
			bool causal = false;
			std::size_t keyPositionOffset = 0;
			std::size_t queryPositionOffset = 0;
			std::vector<float> mask;
			bool hasMask = false;
			float scale = 1.0f;
			float maxBias = 0.0f;
			float logitSoftcap = 0.0f;
			std::optional<float> sink;
			std::size_t headIndex = 0;
			std::size_t headCount = 1;
		};

		double ComputeALiBiSlope(std::size_t headIndex, std::size_t headCount, double maxBias)
		{
			if (maxBias <= 0.0)
			{
				return 1.0;
			}
			auto headLog2 = 1uz;
			while ((headLog2 << 1uz) <= headCount)
			{
				headLog2 <<= 1uz;
			}
			const auto m0 = std::pow(2.0, -maxBias / static_cast<double>(headLog2));
			const auto m1 = std::pow(2.0, -(maxBias / 2.0) / static_cast<double>(headLog2));
			if (headIndex < headLog2)
			{
				return std::pow(m0, static_cast<double>(headIndex + 1));
			}
			return std::pow(m1, static_cast<double>(2 * (headIndex - headLog2) + 1));
		}

		std::vector<float> ComputeFlashAttnExpected(const std::vector<float>& queries, std::size_t queryLength,
		                                          const std::vector<float>& keys, std::size_t keyLength,
		                                          std::size_t headDim, const std::vector<float>& values,
		                                          std::size_t valueDim, const FlashAttnReferenceOptions& options)
		{
			std::vector<float> result(queryLength * valueDim, 0.0f);
			const auto slope = ComputeALiBiSlope(options.headIndex, options.headCount, options.maxBias);

			for (std::size_t row = 0; row < queryLength; ++row)
			{
				std::vector<double> scores(keyLength, 0.0);
				for (std::size_t col = 0; col < keyLength; ++col)
				{
					double score = 0.0;
					for (std::size_t dim = 0; dim < headDim; ++dim)
					{
						score += static_cast<double>(queries[row * headDim + dim]) *
						         static_cast<double>(keys[col * headDim + dim]);
					}
					score *= options.scale;
					if (options.logitSoftcap != 0.0f)
					{
						score = static_cast<double>(options.logitSoftcap) *
						        std::tanh(score / static_cast<double>(options.logitSoftcap));
					}
					if (options.hasMask)
					{
						score += slope * static_cast<double>(options.mask[row * keyLength + col]);
					}
					if (options.causal && options.keyPositionOffset + col > options.queryPositionOffset + row)
					{
						score += -1.0e9;
					}
					scores[col] = score;
				}

				double maxScore = -std::numeric_limits<double>::infinity();
				for (const auto score : scores)
				{
					maxScore = std::max(maxScore, score);
				}
				if (options.sink)
				{
					maxScore = std::max(maxScore, static_cast<double>(*options.sink));
				}

				double denominator = 0.0;
				std::vector<double> probabilities(keyLength, 0.0);
				for (std::size_t col = 0; col < keyLength; ++col)
				{
					probabilities[col] = std::exp(scores[col] - maxScore);
					denominator += probabilities[col];
				}
				if (options.sink)
				{
					denominator += std::exp(static_cast<double>(*options.sink) - maxScore);
				}

				for (std::size_t col = 0; col < keyLength; ++col)
				{
					const auto weight = probabilities[col] / denominator;
					for (std::size_t outCol = 0; outCol < valueDim; ++outCol)
					{
						result[row * valueDim + outCol] +=
						    static_cast<float>(weight * static_cast<double>(values[col * valueDim + outCol]));
					}
				}
			}

			return result;
		}
} // namespace

TEST(LayerGetRows, LooksUpEmbeddingRowsFromTokenIds)
{
	Graph graph;
	Subgraph sg;
	const auto table = sg.AddParam(DataType::Float32, { 4, 2 });
	const auto indices = sg.AddParam(DataType::Int32, { 3 });
	const auto output = sg.AddNode(GetRowsNode{ { table, 0 }, { indices, 0 } },
	                              { OutputInfo{ DataType::Float32, { 3, 2 } } });
	sg.SetResults({ { output, 0 } });
	graph.SetForward(graph.AddSubgraph(std::move(sg)));

	std::vector<Tensor<CPU>> inputs;
	inputs.emplace_back(Tensor<CPU>({ 10.0f, 11.0f,
	                                 20.0f, 21.0f,
	                                 30.0f, 31.0f,
	                                 40.0f, 41.0f },
	                                { 4, 2 }));
	inputs.emplace_back(MakeInt32Tensor({ 2, 0, 3 }, { 3 }));

	const auto outputs = RunWithInputs(graph, std::move(inputs));
	ASSERT_EQ(outputs.size(), 1u);
	ASSERT_EQ(outputs[0].Shape().NumElements(), 6u);
	EXPECT_NEAR(ReadFloat(outputs[0], 0), 30.0f, 1e-5f);
	EXPECT_NEAR(ReadFloat(outputs[0], 1), 31.0f, 1e-5f);
	EXPECT_NEAR(ReadFloat(outputs[0], 2), 10.0f, 1e-5f);
	EXPECT_NEAR(ReadFloat(outputs[0], 3), 11.0f, 1e-5f);
	EXPECT_NEAR(ReadFloat(outputs[0], 4), 40.0f, 1e-5f);
	EXPECT_NEAR(ReadFloat(outputs[0], 5), 41.0f, 1e-5f);
}

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

TEST(LayerFlashAttnExt, AppliesCausalMaskBeforeSoftmax)
{
	Graph graph;
	Subgraph sg;
	const auto queries = sg.AddParam(DataType::Float32, { 3, 2 });
	const auto keys = sg.AddParam(DataType::Float32, { 3, 2 });
	const auto values = sg.AddParam(DataType::Float32, { 3, 1 });

	Layer::FlashAttnExtOptions options;
	options.scale = 1.0 / std::sqrt(2.0);
	options.causal = true;
	const auto output = Layer::AddFlashAttnExt(sg, { queries, 0 }, { keys, 0 }, { values, 0 }, options);
	sg.SetResults({ output });
	graph.SetForward(graph.AddSubgraph(std::move(sg)));

	const std::vector<float> queryData{ 1.0f, 0.0f,
	                                  0.0f, 1.0f,
	                                  1.0f, 1.0f };
	const std::vector<float> keyData{ 1.0f, 0.0f,
	                                0.0f, 1.0f,
	                                1.0f, 1.0f };
	const std::vector<float> valueData{ 2.0f,
	                                  5.0f,
	                                  11.0f };

	std::vector<Tensor<CPU>> inputs;
	inputs.emplace_back(Optimizer::MakeFloatTensor(queryData, { 3, 2 }));
	inputs.emplace_back(Optimizer::MakeFloatTensor(keyData, { 3, 2 }));
	inputs.emplace_back(Optimizer::MakeFloatTensor(valueData, { 3, 1 }));

	const auto outputs = RunWithInputs(graph, std::move(inputs));
	ASSERT_EQ(outputs.size(), 1u);
	FlashAttnReferenceOptions referenceOptions;
	referenceOptions.causal = true;
	referenceOptions.scale = static_cast<float>(1.0 / std::sqrt(2.0));
	const auto expected = ComputeFlashAttnExpected(queryData, 3, keyData, 3, 2, valueData, 1, referenceOptions);
	for (std::size_t i = 0; i < expected.size(); ++i)
	{
		EXPECT_NEAR(ReadFloat(outputs[0], i), expected[i], 1e-5f);
	}
}

TEST(LayerFlashAttnExt, SupportsRectangularCausalDecodeMask)
{
	Graph graph;
	Subgraph sg;
	const auto queries = sg.AddParam(DataType::Float32, { 1, 2 });
	const auto keys = sg.AddParam(DataType::Float32, { 3, 2 });
	const auto values = sg.AddParam(DataType::Float32, { 3, 1 });

	Layer::FlashAttnExtOptions options;
	options.causal = true;
	options.queryPositionOffset = 1;
	const auto output = Layer::AddFlashAttnExt(sg, { queries, 0 }, { keys, 0 }, { values, 0 }, options);
	sg.SetResults({ output });
	graph.SetForward(graph.AddSubgraph(std::move(sg)));

	const std::vector<float> queryData{ 1.0f, 0.0f };
	const std::vector<float> keyData{ 1.0f, 0.0f,
	                                0.0f, 1.0f,
	                                10.0f, 0.0f };
	const std::vector<float> valueData{ 2.0f,
	                                  5.0f,
	                                  1000.0f };

	std::vector<Tensor<CPU>> inputs;
	inputs.emplace_back(Optimizer::MakeFloatTensor(queryData, { 1, 2 }));
	inputs.emplace_back(Optimizer::MakeFloatTensor(keyData, { 3, 2 }));
	inputs.emplace_back(Optimizer::MakeFloatTensor(valueData, { 3, 1 }));

	const auto outputs = RunWithInputs(graph, std::move(inputs));
	ASSERT_EQ(outputs.size(), 1u);
	FlashAttnReferenceOptions referenceOptions;
	referenceOptions.causal = true;
	referenceOptions.queryPositionOffset = 1;
	const auto expected = ComputeFlashAttnExpected(queryData, 1, keyData, 3, 2, valueData, 1, referenceOptions);
	EXPECT_NEAR(ReadFloat(outputs[0], 0), expected[0], 1e-5f);
	EXPECT_LT(ReadFloat(outputs[0], 0), 5.0f);
}

TEST(LayerFlashAttnExt, AppliesMaskSoftcapAndSinks)
{
	Graph graph;
	Subgraph sg;
	const auto queries = sg.AddParam(DataType::Float32, { 2, 2 });
	const auto keys = sg.AddParam(DataType::Float32, { 2, 2 });
	const auto values = sg.AddParam(DataType::Float32, { 2, 1 });
	const auto mask = sg.AddParam(DataType::Float32, { 2, 2 });
	const auto sink = sg.AddParam(DataType::Float32, { 1 });

	Layer::FlashAttnExtOptions options;
	options.mask = NodeOutput{ mask, 0 };
	options.sinks = NodeOutput{ sink, 0 };
	options.scale = 1.25;
	options.maxBias = 4.0;
	options.logitSoftcap = 0.75;
	options.headIndex = 1;
	options.headCount = 4;
	const auto output = Layer::AddFlashAttnExt(sg, { queries, 0 }, { keys, 0 }, { values, 0 }, options);
	sg.SetResults({ output });
	graph.SetForward(graph.AddSubgraph(std::move(sg)));

	const std::vector<float> queryData{ 1.0f, 2.0f,
	                                  0.5f, -1.0f };
	const std::vector<float> keyData{ 2.0f, -1.0f,
	                                1.0f, 1.0f };
	const std::vector<float> valueData{ 1.0f,
	                                  4.0f };
	const std::vector<float> maskData{ 0.0f, 1.0f,
	                                 -2.0f, 0.5f };
	const std::vector<float> sinkData{ 0.2f };

	std::vector<Tensor<CPU>> inputs;
	inputs.emplace_back(Optimizer::MakeFloatTensor(queryData, { 2, 2 }));
	inputs.emplace_back(Optimizer::MakeFloatTensor(keyData, { 2, 2 }));
	inputs.emplace_back(Optimizer::MakeFloatTensor(valueData, { 2, 1 }));
	inputs.emplace_back(Optimizer::MakeFloatTensor(maskData, { 2, 2 }));
	inputs.emplace_back(Optimizer::MakeFloatTensor(sinkData, { 1 }));

	const auto outputs = RunWithInputs(graph, std::move(inputs));
	ASSERT_EQ(outputs.size(), 1u);

	FlashAttnReferenceOptions referenceOptions;
	referenceOptions.hasMask = true;
	referenceOptions.mask = maskData;
	referenceOptions.scale = 1.25f;
	referenceOptions.maxBias = 4.0f;
	referenceOptions.logitSoftcap = 0.75f;
	referenceOptions.sink = 0.2f;
	referenceOptions.headIndex = 1;
	referenceOptions.headCount = 4;
	const auto expected = ComputeFlashAttnExpected(queryData, 2, keyData, 2, 2, valueData, 1, referenceOptions);
	for (std::size_t i = 0; i < expected.size(); ++i)
	{
		EXPECT_NEAR(ReadFloat(outputs[0], i), expected[i], 1e-5f);
	}
}

TEST(LayerSumRows, ReducesFirstAxisAndKeepsLeadingDimension)
{
	Graph graph;
	Subgraph sg;
	const auto input = sg.AddParam(DataType::Float32, { 3, 2 });
	const auto out = Layer::AddSumRows(sg, { input, 0 });
	sg.SetResults({ out });
	graph.SetForward(graph.AddSubgraph(std::move(sg)));

	const auto result = RunSingleIO(graph, { 1.0f, 2.0f,
	                                        3.0f, 4.0f,
	                                        5.0f, 6.0f },
	                                { 3, 2 });

	ASSERT_EQ(result.Shape().NumDim(), 2u);
	EXPECT_EQ(result.Shape()[0], 1u);
	EXPECT_EQ(result.Shape()[1], 2u);
	EXPECT_NEAR(ReadFloat(result, 0), 9.0f, 1e-5f);
	EXPECT_NEAR(ReadFloat(result, 1), 12.0f, 1e-5f);
}

TEST(LayerSumRows, PreservesTrailingDimensions)
{
	Graph graph;
	Subgraph sg;
	const auto input = sg.AddParam(DataType::Float32, { 2, 2, 2 });
	const auto out = Layer::AddSumRows(sg, { input, 0 });
	sg.SetResults({ out });
	graph.SetForward(graph.AddSubgraph(std::move(sg)));

	const auto result = RunSingleIO(graph, { 1.0f, 2.0f,
	                                        3.0f, 4.0f,
	                                        10.0f, 20.0f,
	                                        30.0f, 40.0f },
	                                { 2, 2, 2 });

	ASSERT_EQ(result.Shape().NumDim(), 3u);
	EXPECT_EQ(result.Shape()[0], 1u);
	EXPECT_EQ(result.Shape()[1], 2u);
	EXPECT_EQ(result.Shape()[2], 2u);
	EXPECT_NEAR(ReadFloat(result, 0), 11.0f, 1e-5f);
	EXPECT_NEAR(ReadFloat(result, 1), 22.0f, 1e-5f);
	EXPECT_NEAR(ReadFloat(result, 2), 33.0f, 1e-5f);
	EXPECT_NEAR(ReadFloat(result, 3), 44.0f, 1e-5f);
}

TEST(LayerRoll, PositiveShiftRotatesAxisOne)
{
	Graph graph;
	Subgraph sg;
	const auto input = sg.AddParam(DataType::Float32, { 2, 4 });
	const auto out = Layer::AddRoll(sg, { input, 0 }, 1, 1);
	sg.SetResults({ out });
	graph.SetForward(graph.AddSubgraph(std::move(sg)));

	const auto result = RunSingleIO(graph, { 1.0f, 2.0f, 3.0f, 4.0f,
	                                        5.0f, 6.0f, 7.0f, 8.0f },
	                                { 2, 4 });

	EXPECT_NEAR(ReadFloat(result, 0), 4.0f, 1e-5f);
	EXPECT_NEAR(ReadFloat(result, 1), 1.0f, 1e-5f);
	EXPECT_NEAR(ReadFloat(result, 2), 2.0f, 1e-5f);
	EXPECT_NEAR(ReadFloat(result, 3), 3.0f, 1e-5f);
	EXPECT_NEAR(ReadFloat(result, 4), 8.0f, 1e-5f);
	EXPECT_NEAR(ReadFloat(result, 5), 5.0f, 1e-5f);
	EXPECT_NEAR(ReadFloat(result, 6), 6.0f, 1e-5f);
	EXPECT_NEAR(ReadFloat(result, 7), 7.0f, 1e-5f);
}

TEST(LayerRoll, NegativeShiftRotatesAxisZero)
{
	Graph graph;
	Subgraph sg;
	const auto input = sg.AddParam(DataType::Float32, { 3, 2 });
	const auto out = Layer::AddRoll(sg, { input, 0 }, 0, -1);
	sg.SetResults({ out });
	graph.SetForward(graph.AddSubgraph(std::move(sg)));

	const auto result = RunSingleIO(graph, { 1.0f, 2.0f,
	                                        3.0f, 4.0f,
	                                        5.0f, 6.0f },
	                                { 3, 2 });

	EXPECT_NEAR(ReadFloat(result, 0), 3.0f, 1e-5f);
	EXPECT_NEAR(ReadFloat(result, 1), 4.0f, 1e-5f);
	EXPECT_NEAR(ReadFloat(result, 2), 5.0f, 1e-5f);
	EXPECT_NEAR(ReadFloat(result, 3), 6.0f, 1e-5f);
	EXPECT_NEAR(ReadFloat(result, 4), 1.0f, 1e-5f);
	EXPECT_NEAR(ReadFloat(result, 5), 2.0f, 1e-5f);
}

TEST(LayerArange, BuildsFloatSequence)
{
	Graph graph;
	graph.SetForward(Layer::BuildArange(graph, DataType::Float32, 5, 1.5, 0.5));

	const auto outputs = RunWithInputs(graph, {});
	ASSERT_EQ(outputs.size(), 1u);
	ASSERT_EQ(outputs[0].Shape().NumElements(), 5u);
	EXPECT_NEAR(ReadFloat(outputs[0], 0), 1.5f, 1e-5f);
	EXPECT_NEAR(ReadFloat(outputs[0], 1), 2.0f, 1e-5f);
	EXPECT_NEAR(ReadFloat(outputs[0], 2), 2.5f, 1e-5f);
	EXPECT_NEAR(ReadFloat(outputs[0], 3), 3.0f, 1e-5f);
	EXPECT_NEAR(ReadFloat(outputs[0], 4), 3.5f, 1e-5f);
}

TEST(LayerArgsort, SortsEachTrailingPositionAlongFirstAxis)
{
	Graph graph;
	Subgraph sg;
	const auto input = sg.AddParam(DataType::Float32, { 4, 2 });
	const auto out = Layer::AddArgsort(sg, { input, 0 }, SortOrder::Descending);
	sg.SetResults({ out });
	graph.SetForward(graph.AddSubgraph(std::move(sg)));

	const auto result = RunSingleIO(graph, { 0.5f, 4.0f,
	                                        2.0f, 0.0f,
	                                        -1.0f, 5.0f,
	                                        2.0f, 1.0f },
	                                { 4, 2 });

	ASSERT_EQ(result.DType(), DataType::Int32);
	EXPECT_EQ(ReadInt32(result, 0), 1);
	EXPECT_EQ(ReadInt32(result, 1), 2);
	EXPECT_EQ(ReadInt32(result, 2), 3);
	EXPECT_EQ(ReadInt32(result, 3), 0);
	EXPECT_EQ(ReadInt32(result, 4), 0);
	EXPECT_EQ(ReadInt32(result, 5), 3);
	EXPECT_EQ(ReadInt32(result, 6), 2);
	EXPECT_EQ(ReadInt32(result, 7), 1);
}

TEST(LayerTopK, ReturnsLeadingDescendingIndices)
{
	Graph graph;
	Subgraph sg;
	const auto input = sg.AddParam(DataType::Float32, { 4, 2 });
	const auto out = Layer::AddTopK(sg, { input, 0 }, 2);
	sg.SetResults({ out });
	graph.SetForward(graph.AddSubgraph(std::move(sg)));

	const auto result = RunSingleIO(graph, { 0.5f, 4.0f,
	                                        2.0f, 0.0f,
	                                        -1.0f, 5.0f,
	                                        2.0f, 1.0f },
	                                { 4, 2 });

	ASSERT_EQ(result.DType(), DataType::Int32);
	ASSERT_EQ(result.Shape().NumDim(), 2u);
	EXPECT_EQ(result.Shape()[0], 2u);
	EXPECT_EQ(result.Shape()[1], 2u);
	EXPECT_EQ(ReadInt32(result, 0), 1);
	EXPECT_EQ(ReadInt32(result, 1), 2);
	EXPECT_EQ(ReadInt32(result, 2), 3);
	EXPECT_EQ(ReadInt32(result, 3), 0);
}

TEST(LayerTopK, SupportsExplicitLastAxis)
{
	Graph graph;
	Subgraph sg;
	const auto input = sg.AddParam(DataType::Float32, { 2, 4 });
	const auto out = Layer::AddTopK(sg, { input, 0 }, 2, 1);
	sg.SetResults({ out });
	graph.SetForward(graph.AddSubgraph(std::move(sg)));

	const auto result = RunSingleIO(graph, { 0.5f, 4.0f, -1.0f, 2.0f,
	                                        3.0f, 1.0f, 5.0f, 5.0f },
	                                { 2, 4 });

	ASSERT_EQ(result.DType(), DataType::Int32);
	ASSERT_EQ(result.Shape().NumDim(), 2u);
	EXPECT_EQ(result.Shape()[0], 2u);
	EXPECT_EQ(result.Shape()[1], 2u);
	EXPECT_EQ(ReadInt32(result, 0), 1);
	EXPECT_EQ(ReadInt32(result, 1), 3);
	EXPECT_EQ(ReadInt32(result, 2), 2);
	EXPECT_EQ(ReadInt32(result, 3), 3);
}

TEST(LayerPad, AppendsZerosAlongEachAxis)
{
	Graph graph;
	Subgraph sg;
	const auto input = sg.AddParam(DataType::Float32, { 2, 2 });
	const std::array<std::size_t, 2> paddings{ 1, 2 };
	const auto out = Layer::AddPad(sg, { input, 0 }, paddings);
	sg.SetResults({ out });
	graph.SetForward(graph.AddSubgraph(std::move(sg)));

	const auto result = RunSingleIO(graph, { 1.0f, 2.0f,
	                                        3.0f, 4.0f },
	                                { 2, 2 });

	ASSERT_EQ(result.Shape().NumDim(), 2u);
	EXPECT_EQ(result.Shape()[0], 3u);
	EXPECT_EQ(result.Shape()[1], 4u);
	EXPECT_NEAR(ReadFloat(result, 0), 1.0f, 1e-5f);
	EXPECT_NEAR(ReadFloat(result, 1), 2.0f, 1e-5f);
	EXPECT_NEAR(ReadFloat(result, 2), 0.0f, 1e-5f);
	EXPECT_NEAR(ReadFloat(result, 3), 0.0f, 1e-5f);
	EXPECT_NEAR(ReadFloat(result, 4), 3.0f, 1e-5f);
	EXPECT_NEAR(ReadFloat(result, 5), 4.0f, 1e-5f);
	EXPECT_NEAR(ReadFloat(result, 6), 0.0f, 1e-5f);
	EXPECT_NEAR(ReadFloat(result, 7), 0.0f, 1e-5f);
	EXPECT_NEAR(ReadFloat(result, 8), 0.0f, 1e-5f);
	EXPECT_NEAR(ReadFloat(result, 9), 0.0f, 1e-5f);
	EXPECT_NEAR(ReadFloat(result, 10), 0.0f, 1e-5f);
	EXPECT_NEAR(ReadFloat(result, 11), 0.0f, 1e-5f);
}

TEST(LayerCumsum, AccumulatesAlongAxisZero)
{
	Graph graph;
	Subgraph sg;
	const auto input = sg.AddParam(DataType::Float32, { 3, 2 });
	const auto out = Layer::AddCumsum(sg, { input, 0 }, 0);
	sg.SetResults({ out });
	graph.SetForward(graph.AddSubgraph(std::move(sg)));

	const auto result = RunSingleIO(graph, { 1.0f, 2.0f,
	                                        3.0f, 4.0f,
	                                        5.0f, 6.0f },
	                                { 3, 2 });

	EXPECT_NEAR(ReadFloat(result, 0), 1.0f, 1e-5f);
	EXPECT_NEAR(ReadFloat(result, 1), 2.0f, 1e-5f);
	EXPECT_NEAR(ReadFloat(result, 2), 4.0f, 1e-5f);
	EXPECT_NEAR(ReadFloat(result, 3), 6.0f, 1e-5f);
	EXPECT_NEAR(ReadFloat(result, 4), 9.0f, 1e-5f);
	EXPECT_NEAR(ReadFloat(result, 5), 12.0f, 1e-5f);
}

TEST(LayerCumsum, AccumulatesAlongAxisOne)
{
	Graph graph;
	Subgraph sg;
	const auto input = sg.AddParam(DataType::Float32, { 2, 3 });
	const auto out = Layer::AddCumsum(sg, { input, 0 }, 1);
	sg.SetResults({ out });
	graph.SetForward(graph.AddSubgraph(std::move(sg)));

	const auto result = RunSingleIO(graph, { 1.0f, 2.0f, 3.0f,
	                                        4.0f, 5.0f, 6.0f },
	                                { 2, 3 });

	EXPECT_NEAR(ReadFloat(result, 0), 1.0f, 1e-5f);
	EXPECT_NEAR(ReadFloat(result, 1), 3.0f, 1e-5f);
	EXPECT_NEAR(ReadFloat(result, 2), 6.0f, 1e-5f);
	EXPECT_NEAR(ReadFloat(result, 3), 4.0f, 1e-5f);
	EXPECT_NEAR(ReadFloat(result, 4), 9.0f, 1e-5f);
	EXPECT_NEAR(ReadFloat(result, 5), 15.0f, 1e-5f);
}

TEST(LayerGroupNorm, NormalizesContiguousGroupsInOneDimension)
{
	Graph graph;
	Subgraph sg;
	const auto input = sg.AddParam(DataType::Float32, { 8 });
	const auto out = Layer::AddGroupNorm(sg, { input, 0 }, 4, 1e-6);
	sg.SetResults({ out });
	graph.SetForward(graph.AddSubgraph(std::move(sg)));

	const auto result = RunSingleIO(graph, { 1.0f, 3.0f,
	                                        2.0f, 4.0f,
	                                        5.0f, 7.0f,
	                                        6.0f, 8.0f },
	                                { 8 });

	for (std::size_t group = 0; group < 4; ++group)
	{
		EXPECT_NEAR(ReadFloat(result, group * 2 + 0), -1.0f, 1e-3f);
		EXPECT_NEAR(ReadFloat(result, group * 2 + 1), 1.0f, 1e-3f);
	}
}

TEST(LayerGroupNorm, PreservesBatchSeparationForRankFourInput)
{
	Graph graph;
	Subgraph sg;
	const auto input = sg.AddParam(DataType::Float32, { 4, 1, 1, 2 });
	const auto out = Layer::AddGroupNorm(sg, { input, 0 }, 2, 1e-6);
	sg.SetResults({ out });
	graph.SetForward(graph.AddSubgraph(std::move(sg)));

	const auto result = RunSingleIO(graph, { 1.0f, 10.0f,
	                                        3.0f, 30.0f,
	                                        2.0f, 20.0f,
	                                        4.0f, 40.0f },
	                                { 4, 1, 1, 2 });

	EXPECT_NEAR(ReadFloat(result, 0), -1.0f, 1e-3f);
	EXPECT_NEAR(ReadFloat(result, 1), -1.0f, 1e-3f);
	EXPECT_NEAR(ReadFloat(result, 2), 1.0f, 1e-3f);
	EXPECT_NEAR(ReadFloat(result, 3), 1.0f, 1e-3f);
	EXPECT_NEAR(ReadFloat(result, 4), -1.0f, 1e-3f);
	EXPECT_NEAR(ReadFloat(result, 5), -1.0f, 1e-3f);
	EXPECT_NEAR(ReadFloat(result, 6), 1.0f, 1e-3f);
	EXPECT_NEAR(ReadFloat(result, 7), 1.0f, 1e-3f);
}

TEST(LayerAddId, AddsExpertBiasBySelectedIds)
{
	Graph graph;
	graph.SetForward(Layer::BuildAddId(graph, DataType::Float32, { 2, 2, 3 }, { 2, 4 }, DataType::Int32, { 2, 3 }));

	std::vector<Tensor<CPU>> inputs;
	inputs.emplace_back(Tensor<CPU>({ 0.0f, 0.0f, 0.0f,
	                                 0.0f, 0.0f, 0.0f,
	                                 0.0f, 0.0f, 0.0f,
	                                 0.0f, 0.0f, 0.0f },
	                                { 2, 2, 3 }));
	inputs.emplace_back(Tensor<CPU>({ 10.0f, 20.0f, 30.0f, 40.0f,
	                                 100.0f, 200.0f, 300.0f, 400.0f },
	                                { 2, 4 }));
	inputs.emplace_back(MakeInt32Tensor({ 1, 3, 0,
	                                     2, 1, 3 },
	                                    { 2, 3 }));

	const auto outputs = RunWithInputs(graph, std::move(inputs));
	ASSERT_EQ(outputs.size(), 1u);
	const auto& result = outputs[0];

	EXPECT_NEAR(ReadFloat(result, 0), 20.0f, 1e-5f);
	EXPECT_NEAR(ReadFloat(result, 1), 40.0f, 1e-5f);
	EXPECT_NEAR(ReadFloat(result, 2), 10.0f, 1e-5f);
	EXPECT_NEAR(ReadFloat(result, 3), 30.0f, 1e-5f);
	EXPECT_NEAR(ReadFloat(result, 4), 20.0f, 1e-5f);
	EXPECT_NEAR(ReadFloat(result, 5), 40.0f, 1e-5f);
	EXPECT_NEAR(ReadFloat(result, 6), 200.0f, 1e-5f);
	EXPECT_NEAR(ReadFloat(result, 7), 400.0f, 1e-5f);
	EXPECT_NEAR(ReadFloat(result, 8), 100.0f, 1e-5f);
	EXPECT_NEAR(ReadFloat(result, 9), 300.0f, 1e-5f);
	EXPECT_NEAR(ReadFloat(result, 10), 200.0f, 1e-5f);
	EXPECT_NEAR(ReadFloat(result, 11), 400.0f, 1e-5f);
}

TEST(LayerMulMatId, SelectsPerExpertMatrices)
{
	Graph graph;
	graph.SetForward(Layer::BuildMulMatId(graph, DataType::Float32, { 2, 3, 2 }, DataType::Float32, { 2, 2, 2 },
	                                     DataType::Int32, { 2, 2 }));

	std::vector<Tensor<CPU>> inputs;
	inputs.emplace_back(Tensor<CPU>({
	    1.0f, 4.0f,
	    2.0f, 5.0f,
	    3.0f, 6.0f,
	    10.0f, 40.0f,
	    20.0f, 50.0f,
	    30.0f, 60.0f,
	}, { 2, 3, 2 }));
	inputs.emplace_back(Tensor<CPU>({
	    1.0f, 3.0f,
	    5.0f, 7.0f,
	    2.0f, 4.0f,
	    6.0f, 8.0f,
	}, { 2, 2, 2 }));
	inputs.emplace_back(MakeInt32Tensor({ 0, 1,
	                                     1, 0 },
	                                    { 2, 2 }));

	const auto outputs = RunWithInputs(graph, std::move(inputs));
	ASSERT_EQ(outputs.size(), 1u);
	const auto& result = outputs[0];

	EXPECT_NEAR(ReadFloat(result, 0), 21.0f, 1e-5f);
	EXPECT_NEAR(ReadFloat(result, 1), 172.0f, 1e-5f);
	EXPECT_NEAR(ReadFloat(result, 2), 260.0f, 1e-5f);
	EXPECT_NEAR(ReadFloat(result, 3), 87.0f, 1e-5f);
	EXPECT_NEAR(ReadFloat(result, 4), 42.0f, 1e-5f);
	EXPECT_NEAR(ReadFloat(result, 5), 215.0f, 1e-5f);
	EXPECT_NEAR(ReadFloat(result, 6), 325.0f, 1e-5f);
	EXPECT_NEAR(ReadFloat(result, 7), 174.0f, 1e-5f);
	EXPECT_NEAR(ReadFloat(result, 8), 63.0f, 1e-5f);
	EXPECT_NEAR(ReadFloat(result, 9), 258.0f, 1e-5f);
	EXPECT_NEAR(ReadFloat(result, 10), 390.0f, 1e-5f);
	EXPECT_NEAR(ReadFloat(result, 11), 261.0f, 1e-5f);
}

TEST(LayerMulMatId, BroadcastsInputVectorsAcrossUsedExperts)
{
	Graph graph;
	graph.SetForward(Layer::BuildMulMatId(graph, DataType::Float32, { 2, 2, 2 }, DataType::Float32, { 2, 1, 2 },
	                                     DataType::Int32, { 2, 2 }));

	std::vector<Tensor<CPU>> inputs;
	inputs.emplace_back(Tensor<CPU>({
	    1.0f, 3.0f,
	    2.0f, 4.0f,
	    10.0f, 30.0f,
	    20.0f, 40.0f,
	}, { 2, 2, 2 }));
	inputs.emplace_back(Tensor<CPU>({
	    1.0f, 3.0f,
	    2.0f, 4.0f,
	}, { 2, 1, 2 }));
	inputs.emplace_back(MakeInt32Tensor({ 0, 1,
	                                     1, 0 },
	                                    { 2, 2 }));

	const auto outputs = RunWithInputs(graph, std::move(inputs));
	ASSERT_EQ(outputs.size(), 1u);
	const auto& result = outputs[0];

	EXPECT_NEAR(ReadFloat(result, 0), 21.0f, 1e-5f);
	EXPECT_NEAR(ReadFloat(result, 1), 129.0f, 1e-5f);
	EXPECT_NEAR(ReadFloat(result, 2), 63.0f, 1e-5f);
	EXPECT_NEAR(ReadFloat(result, 3), 43.0f, 1e-5f);
	EXPECT_NEAR(ReadFloat(result, 4), 42.0f, 1e-5f);
	EXPECT_NEAR(ReadFloat(result, 5), 172.0f, 1e-5f);
	EXPECT_NEAR(ReadFloat(result, 6), 84.0f, 1e-5f);
	EXPECT_NEAR(ReadFloat(result, 7), 86.0f, 1e-5f);
}

TEST(LayerArange, BuildsIntegerSequence)
{
	Graph graph;
	graph.SetForward(Layer::BuildArange(graph, DataType::Int32, 4, -2.0, 3.0));

	const auto outputs = RunWithInputs(graph, {});
	ASSERT_EQ(outputs.size(), 1u);
	ASSERT_EQ(outputs[0].Shape().NumElements(), 4u);
	EXPECT_EQ(ReadInt32(outputs[0], 0), -2);
	EXPECT_EQ(ReadInt32(outputs[0], 1), 1);
	EXPECT_EQ(ReadInt32(outputs[0], 2), 4);
	EXPECT_EQ(ReadInt32(outputs[0], 3), 7);
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

TEST(LayerClamp, LimitsValuesToRange)
{
	Graph graph;
	Subgraph sg;
	const auto input = sg.AddParam(DataType::Float32, { 4 });
	const auto out = Layer::AddClamp(sg, { input, 0 }, -1.0, 1.0);
	sg.SetResults({ out });
	graph.SetForward(graph.AddSubgraph(std::move(sg)));

	const auto result = RunSingleIO(graph, { -2.0f, -0.5f, 0.5f, 2.0f }, { 4 });
	EXPECT_NEAR(ReadFloat(result, 0), -1.0f, 1e-5f);
	EXPECT_NEAR(ReadFloat(result, 1), -0.5f, 1e-5f);
	EXPECT_NEAR(ReadFloat(result, 2), 0.5f, 1e-5f);
	EXPECT_NEAR(ReadFloat(result, 3), 1.0f, 1e-5f);
}

TEST(LayerLeakyReLU, NegativeBranchUsesSlope)
{
	Graph graph;
	Subgraph sg;
	const auto input = sg.AddParam(DataType::Float32, { 3 });
	const auto out = Layer::AddLeakyReLU(sg, { input, 0 }, 0.2);
	sg.SetResults({ out });
	graph.SetForward(graph.AddSubgraph(std::move(sg)));

	const auto result = RunSingleIO(graph, { -2.0f, 0.0f, 3.0f }, { 3 });
	EXPECT_NEAR(ReadFloat(result, 0), -0.4f, 1e-5f);
	EXPECT_NEAR(ReadFloat(result, 1), 0.0f, 1e-5f);
	EXPECT_NEAR(ReadFloat(result, 2), 3.0f, 1e-5f);
}

TEST(LayerHardSigmoid, MatchesPiecewiseFormula)
{
	Graph graph;
	Subgraph sg;
	const auto input = sg.AddParam(DataType::Float32, { 3 });
	const auto out = Layer::AddHardSigmoid(sg, { input, 0 });
	sg.SetResults({ out });
	graph.SetForward(graph.AddSubgraph(std::move(sg)));

	const auto result = RunSingleIO(graph, { -4.0f, 0.0f, 3.0f }, { 3 });
	EXPECT_NEAR(ReadFloat(result, 0), 0.0f, 1e-5f);
	EXPECT_NEAR(ReadFloat(result, 1), 0.5f, 1e-5f);
	EXPECT_NEAR(ReadFloat(result, 2), 1.0f, 1e-5f);
}

TEST(LayerHardSwish, MatchesPiecewiseFormula)
{
	Graph graph;
	Subgraph sg;
	const auto input = sg.AddParam(DataType::Float32, { 3 });
	const auto out = Layer::AddHardSwish(sg, { input, 0 });
	sg.SetResults({ out });
	graph.SetForward(graph.AddSubgraph(std::move(sg)));

	const auto result = RunSingleIO(graph, { -4.0f, 1.0f, 3.0f }, { 3 });
	EXPECT_NEAR(ReadFloat(result, 0), 0.0f, 1e-5f);
	EXPECT_NEAR(ReadFloat(result, 1), 2.0f / 3.0f, 1e-5f);
	EXPECT_NEAR(ReadFloat(result, 2), 3.0f, 1e-5f);
}

TEST(LayerGELUQuick, MatchesAnalyticValue)
{
	Graph graph;
	Subgraph sg;
	const auto input = sg.AddParam(DataType::Float32, { 1 });
	const auto out = Layer::AddGELUQuick(sg, { input, 0 });
	sg.SetResults({ out });
	graph.SetForward(graph.AddSubgraph(std::move(sg)));

	const auto result = RunSingleIO(graph, { 1.0f }, { 1 });
	const auto expected = 1.0f / (1.0f + std::exp(-1.702f));
	EXPECT_NEAR(ReadFloat(result, 0), expected, 1e-5f);
}

TEST(LayerGELUErf, MatchesAnalyticValue)
{
	Graph graph;
	Subgraph sg;
	const auto input = sg.AddParam(DataType::Float32, { 3 });
	const auto out = Layer::AddGELUErf(sg, { input, 0 });
	sg.SetResults({ out });
	graph.SetForward(graph.AddSubgraph(std::move(sg)));

	const auto result = RunSingleIO(graph, { -1.0f, 0.0f, 1.0f }, { 3 });
	const auto expected0 = 0.5f * -1.0f * (1.0f + std::erf(-1.0f * 0.7071067811865476f));
	const auto expected1 = 0.0f;
	const auto expected2 = 0.5f * 1.0f * (1.0f + std::erf(1.0f * 0.7071067811865476f));
	EXPECT_NEAR(ReadFloat(result, 0), expected0, 1e-5f);
	EXPECT_NEAR(ReadFloat(result, 1), expected1, 1e-5f);
	EXPECT_NEAR(ReadFloat(result, 2), expected2, 1e-5f);
}

TEST(LayerL2Norm, NormalizesRowsToUnitLength)
{
	Graph graph;
	Subgraph sg;
	const auto input = sg.AddParam(DataType::Float32, { 2, 2 });
	const auto out = Layer::AddL2Norm(sg, { input, 0 }, 1);
	sg.SetResults({ out });
	graph.SetForward(graph.AddSubgraph(std::move(sg)));

	const auto result = RunSingleIO(graph, { 3.0f, 4.0f, 5.0f, 12.0f }, { 2, 2 });
	EXPECT_NEAR(ReadFloat(result, 0), 3.0f / 5.0f, 1e-5f);
	EXPECT_NEAR(ReadFloat(result, 1), 4.0f / 5.0f, 1e-5f);
	EXPECT_NEAR(ReadFloat(result, 2), 5.0f / 13.0f, 1e-5f);
	EXPECT_NEAR(ReadFloat(result, 3), 12.0f / 13.0f, 1e-5f);
}

TEST(LayerL2Norm, SupportsAxisZero)
{
	Graph graph;
	Subgraph sg;
	const auto input = sg.AddParam(DataType::Float32, { 2, 2 });
	const auto out = Layer::AddL2Norm(sg, { input, 0 }, 0);
	sg.SetResults({ out });
	graph.SetForward(graph.AddSubgraph(std::move(sg)));

	const auto result = RunSingleIO(graph, { 3.0f, 4.0f, 4.0f, 3.0f }, { 2, 2 });
	EXPECT_NEAR(ReadFloat(result, 0), 3.0f / 5.0f, 1e-5f);
	EXPECT_NEAR(ReadFloat(result, 1), 4.0f / 5.0f, 1e-5f);
	EXPECT_NEAR(ReadFloat(result, 2), 4.0f / 5.0f, 1e-5f);
	EXPECT_NEAR(ReadFloat(result, 3), 3.0f / 5.0f, 1e-5f);
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

TEST(LayerCausalMask, SupportsRectangularDecodeMask)
{
	Graph graph;
	Subgraph sg;
	const auto input = sg.AddParam(DataType::Float32, { 1, 3 });
	const auto out = Layer::AddCausalMask(sg, { input, 0 }, -1.0e9, 0, 1);
	sg.SetResults({ out });
	graph.SetForward(graph.AddSubgraph(std::move(sg)));

	const auto result = RunSingleIO(graph, { 1.0f, 2.0f, 3.0f }, { 1, 3 });
	EXPECT_NEAR(ReadFloat(result, 0), 1.0f, 1e-5f);
	EXPECT_NEAR(ReadFloat(result, 1), 2.0f, 1e-5f);
	EXPECT_LT(ReadFloat(result, 2), -1.0e8f);
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
