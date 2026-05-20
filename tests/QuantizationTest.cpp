#include <gtest/gtest.h>

#include <LiteNN.h>
#include <LiteNN/Pass/ConstFoldPass.h>
#include <LiteNN/Runtime/Interpreter.h>

#include <cstdint>
#include <filesystem>

using namespace LiteNN;

namespace
{
	float ReadFloat(const Tensor<CPU>& tensor, std::size_t index)
	{
		return static_cast<const float*>(tensor.RawData())[index];
	}
} // namespace

TEST(Quantization, PerTensorAffineQuantizeDequantizeRoundTrip)
{
	const Tensor<CPU> source({ -1.0, 0.0, 1.5 }, { 3 }, DataType::Float32);
	const auto params = PerTensorAffineQuantization(DataType::Int8, 0.5F, 0);

	const auto quantized = QuantizeAffine(source, params);
	ASSERT_EQ(quantized.Storage().DType(), DataType::Int8);
	const auto* storage = static_cast<const std::int8_t*>(quantized.Storage().RawData());
	EXPECT_EQ(storage[0], -2);
	EXPECT_EQ(storage[1], 0);
	EXPECT_EQ(storage[2], 3);

	const auto dequantized = DequantizeAffine(quantized);
	EXPECT_FLOAT_EQ(ReadFloat(dequantized, 0), -1.0F);
	EXPECT_FLOAT_EQ(ReadFloat(dequantized, 1), 0.0F);
	EXPECT_FLOAT_EQ(ReadFloat(dequantized, 2), 1.5F);
}

TEST(Quantization, PerAxisAffineUsesAxisScale)
{
	const Tensor<CPU> source({ 1.0, 2.0, 3.0, 10.0, 20.0, 30.0 }, { 2, 3 }, DataType::Float32);
	const auto params = PerAxisAffineQuantization(DataType::Int8, 0, { 1.0F, 10.0F }, { 0, 0 });

	const auto quantized = QuantizeAffine(source, params);
	const auto* storage = static_cast<const std::int8_t*>(quantized.Storage().RawData());
	EXPECT_EQ(storage[0], 1);
	EXPECT_EQ(storage[1], 2);
	EXPECT_EQ(storage[2], 3);
	EXPECT_EQ(storage[3], 1);
	EXPECT_EQ(storage[4], 2);
	EXPECT_EQ(storage[5], 3);

	const auto dequantized = DequantizeAffine(quantized);
	EXPECT_FLOAT_EQ(ReadFloat(dequantized, 0), 1.0F);
	EXPECT_FLOAT_EQ(ReadFloat(dequantized, 1), 2.0F);
	EXPECT_FLOAT_EQ(ReadFloat(dequantized, 2), 3.0F);
	EXPECT_FLOAT_EQ(ReadFloat(dequantized, 3), 10.0F);
	EXPECT_FLOAT_EQ(ReadFloat(dequantized, 4), 20.0F);
	EXPECT_FLOAT_EQ(ReadFloat(dequantized, 5), 30.0F);
}

TEST(Quantization, GroupedAffineUsesPerLineGroups)
{
	const Tensor<CPU> source({ 1.0, 2.0, 10.0, 20.0, 3.0, 6.0, 40.0, 80.0 }, { 2, 4 }, DataType::Float32);
	const auto params =
	    GroupedAffineQuantization(DataType::Int8, 1, 2, { 1.0F, 10.0F, 3.0F, 40.0F }, { 0, 0, 0, 0 });

	const auto quantized = QuantizeAffine(source, params);
	const auto* storage = static_cast<const std::int8_t*>(quantized.Storage().RawData());
	EXPECT_EQ(storage[0], 1);
	EXPECT_EQ(storage[1], 2);
	EXPECT_EQ(storage[2], 1);
	EXPECT_EQ(storage[3], 2);
	EXPECT_EQ(storage[4], 1);
	EXPECT_EQ(storage[5], 2);
	EXPECT_EQ(storage[6], 1);
	EXPECT_EQ(storage[7], 2);

	const auto dequantized = DequantizeAffine(quantized);
	for (std::size_t i = 0; i < source.NumElements(); ++i)
	{
		EXPECT_FLOAT_EQ(ReadFloat(dequantized, i), ReadFloat(source, i));
	}
}

TEST(Quantization, UInt8ZeroPointIsApplied)
{
	const Tensor<CPU> source({ -1.0, 0.0, 1.0 }, { 3 }, DataType::Float32);
	const auto params = PerTensorAffineQuantization(DataType::UInt8, 0.5F, 128);

	const auto quantized = QuantizeAffine(source, params);
	const auto* storage = static_cast<const std::uint8_t*>(quantized.Storage().RawData());
	EXPECT_EQ(storage[0], 126);
	EXPECT_EQ(storage[1], 128);
	EXPECT_EQ(storage[2], 130);

	const auto dequantized = DequantizeAffine(quantized);
	EXPECT_FLOAT_EQ(ReadFloat(dequantized, 0), -1.0F);
	EXPECT_FLOAT_EQ(ReadFloat(dequantized, 1), 0.0F);
	EXPECT_FLOAT_EQ(ReadFloat(dequantized, 2), 1.0F);
}

TEST(Quantization, RejectsInvalidScaleCount)
{
	const Tensor<CPU> source({ 1.0, 2.0, 3.0, 4.0 }, { 2, 2 }, DataType::Float32);
	const auto params = PerAxisAffineQuantization(DataType::Int8, 0, { 1.0F });
	EXPECT_THROW((void)QuantizeAffine(source, params), std::runtime_error);
}

TEST(Quantization, VariableMetadataSurvivesModelIORoundTrip)
{
	const Tensor<CPU> source({ 1.0, 2.0, 3.0, 4.0 }, { 2, 2 }, DataType::Float32);
	const auto quantized = QuantizeAffine(source, PerTensorAffineQuantization(DataType::Int8, 0.5F, 0));

	Graph graph;
	const auto variableIndex = graph.AddVariable(Variable::CreateQuantized(quantized.Storage(), quantized.Params()));
	Subgraph sg;
	const auto weight =
	    sg.AddNode(VariableRefNode{ variableIndex }, { OutputInfo{ DataType::Int8, { 2, 2 } } });
	sg.SetResults({ { weight, 0 } });
	graph.SetForward(graph.AddSubgraph(std::move(sg)));
	graph.SetOutputNames({ "quantized_weight" });

	const auto path = std::filesystem::path("litenn_quantization_roundtrip_test.ltnn");
	std::filesystem::remove(path);
	Serialization::SaveModel(graph, path);
	auto loaded = Serialization::LoadModel(path);
	std::filesystem::remove(path);

	ASSERT_EQ(loaded.VariableCount(), 1);
	const auto& loadedVariable = loaded.GetVariable(0);
	ASSERT_TRUE(loadedVariable->IsQuantized());
	const auto& params = *loadedVariable->Quantization();
	EXPECT_EQ(params.storageType, DataType::Int8);
	ASSERT_EQ(params.scales.size(), 1);
	EXPECT_FLOAT_EQ(params.scales[0], 0.5F);

	const auto storage = loadedVariable->Data().CopyToDevice(CPU{});
	const auto dequantized = DequantizeAffine(storage, params);
	for (std::size_t i = 0; i < source.NumElements(); ++i)
	{
		EXPECT_FLOAT_EQ(ReadFloat(dequantized, i), ReadFloat(source, i));
	}
}

TEST(Quantization, GraphQuantizeDequantizeRunsInInterpreter)
{
	const auto params = PerTensorAffineQuantization(DataType::UInt8, 0.5F, 128);

	Graph graph;
	Subgraph sg;
	const auto input = sg.AddParam(DataType::Float32, { 3 });
	const auto q = sg.AddNode(QuantizeNode{ { input, 0 }, params }, { OutputInfo{ DataType::UInt8, { 3 } } });
	const auto dq = sg.AddNode(DequantizeNode{ { q, 0 }, params, DataType::Float32 },
	                           { OutputInfo{ DataType::Float32, { 3 } } });
	sg.SetResults({ { dq, 0 } });
	graph.SetForward(graph.AddSubgraph(std::move(sg)));

	Runtime::Interpreter<CPU> interpreter;
	const Tensor<CPU> source({ -1.0, 0.0, 1.0 }, { 3 }, DataType::Float32);
	std::vector<Tensor<CPU>> inputs;
	inputs.push_back(source);
	const auto results = interpreter.RunForward(graph, inputs);

	ASSERT_EQ(results.size(), 1);
	EXPECT_FLOAT_EQ(ReadFloat(results[0], 0), -1.0F);
	EXPECT_FLOAT_EQ(ReadFloat(results[0], 1), 0.0F);
	EXPECT_FLOAT_EQ(ReadFloat(results[0], 2), 1.0F);
}

TEST(Quantization, ConstFoldQuantizeDequantize)
{
	const auto params = PerTensorAffineQuantization(DataType::Int8, 0.5F, 0);

	Graph graph;
	Subgraph sg;
	const auto c =
	    sg.AddNode(ConstantNode{ Tensor<CPU>({ -1.0, 0.0, 1.0 }, { 3 }).CopyToDevice(PolymorphicDevice{ CPU{} }) },
	               { OutputInfo{ DataType::Float32, { 3 } } });
	const auto q = sg.AddNode(QuantizeNode{ { c, 0 }, params }, { OutputInfo{ DataType::Int8, { 3 } } });
	const auto dq = sg.AddNode(DequantizeNode{ { q, 0 }, params, DataType::Float32 },
	                           { OutputInfo{ DataType::Float32, { 3 } } });
	sg.SetResults({ { dq, 0 } });
	graph.SetForward(graph.AddSubgraph(std::move(sg)));

	ConstFoldPass pass;
	pass.Run(graph);

	const auto& fwd = graph.GetSubgraph(graph.Forward());
	ASSERT_EQ(fwd.NodeCount(), 1);
	ASSERT_TRUE(std::holds_alternative<ConstantNode>(fwd.GetNodeEntry(0).node));

	Runtime::Interpreter<CPU> interpreter;
	std::vector<Tensor<CPU>> inputs;
	const auto results = interpreter.RunForward(graph, inputs);
	ASSERT_EQ(results.size(), 1);
	EXPECT_FLOAT_EQ(ReadFloat(results[0], 0), -1.0F);
	EXPECT_FLOAT_EQ(ReadFloat(results[0], 1), 0.0F);
	EXPECT_FLOAT_EQ(ReadFloat(results[0], 2), 1.0F);
}

TEST(Quantization, QuantizedConstantPayloadSurvivesModelIORoundTrip)
{
	const Tensor<CPU> source({ -1.0, 0.0, 1.0 }, { 3 }, DataType::Float32);
	const auto quantized = QuantizeAffine(source, PerTensorAffineQuantization(DataType::Int8, 0.5F, 0));

	Graph graph;
	Subgraph sg;
	const auto payload =
	    sg.AddNode(QuantizedConstantNode{ quantized.Storage().CopyToDevice(PolymorphicDevice{ CPU{} }),
	                                      quantized.Params() },
	               { OutputInfo{ DataType::Int8, { 3 } } });
	const auto dq = sg.AddNode(DequantizeNode{ { payload, 0 }, quantized.Params(), DataType::Float32 },
	                           { OutputInfo{ DataType::Float32, { 3 } } });
	sg.SetResults({ { dq, 0 } });
	graph.SetForward(graph.AddSubgraph(std::move(sg)));

	const auto path = std::filesystem::path("litenn_quantized_constant_roundtrip_test.ltnn");
	std::filesystem::remove(path);
	Serialization::SaveModel(graph, path);
	auto loaded = Serialization::LoadModel(path);
	std::filesystem::remove(path);

	Runtime::Interpreter<CPU> interpreter;
	std::vector<Tensor<CPU>> inputs;
	const auto results = interpreter.RunForward(loaded, inputs);
	ASSERT_EQ(results.size(), 1);
	for (std::size_t i = 0; i < source.NumElements(); ++i)
	{
		EXPECT_FLOAT_EQ(ReadFloat(results[0], i), ReadFloat(source, i));
	}
}

TEST(Quantization, BlockFormatRawPayloadMetadataSurvivesModelIORoundTrip)
{
	const Tensor<CPU> rawPayload({ 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18 },
	                             { 18 }, DataType::UInt8);
	const auto params = BlockQuantization(QuantizedBlockFormat::GGML_Q4_0, { 32 }, DataType::Float32);

	Graph graph;
	Subgraph sg;
	const auto payload =
	    sg.AddNode(QuantizedConstantNode{ rawPayload.CopyToDevice(PolymorphicDevice{ CPU{} }), params },
	               { OutputInfo{ DataType::UInt8, { 18 } } });
	sg.SetResults({ { payload, 0 } });
	graph.SetForward(graph.AddSubgraph(std::move(sg)));

	const auto layout = GetQuantizedBlockLayout(params.blockFormat);
	ASSERT_TRUE(layout.has_value());
	EXPECT_EQ(layout->elementsPerBlock, 32);
	EXPECT_EQ(layout->bytesPerBlock, 18);

	const auto path = std::filesystem::path("litenn_block_payload_roundtrip_test.ltnn");
	std::filesystem::remove(path);
	Serialization::SaveModel(graph, path);
	auto loaded = Serialization::LoadModel(path);
	std::filesystem::remove(path);

	const auto& loadedNode = loaded.GetSubgraph(loaded.Forward()).GetNodeEntry(0).node;
	const auto* qconst = std::get_if<QuantizedConstantNode>(&loadedNode);
	ASSERT_NE(qconst, nullptr);
	EXPECT_EQ(qconst->params.scheme, QuantizationScheme::Block);
	EXPECT_EQ(qconst->params.blockFormat, QuantizedBlockFormat::GGML_Q4_0);
	ASSERT_EQ(qconst->params.expressedShape.size(), 1);
	EXPECT_EQ(qconst->params.expressedShape[0], 32);

	Runtime::Interpreter<CPU> interpreter;
	std::vector<Tensor<CPU>> inputs;
	const auto results = interpreter.RunForward(loaded, inputs);
	ASSERT_EQ(results.size(), 1);
	const auto* data = static_cast<const std::uint8_t*>(results[0].RawData());
	for (std::size_t i = 0; i < rawPayload.NumElements(); ++i)
	{
		EXPECT_EQ(data[i], static_cast<std::uint8_t>(i + 1));
	}
}
