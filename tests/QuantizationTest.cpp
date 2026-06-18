#include <gtest/gtest.h>

#include <LiteNN.h>
#include <LiteNN/Serialization/ModelIO.h>
#include <LiteNN/Pass/ConstFoldPass.h>
#include <LiteNN/Runtime/Interpreter.h>

#include <cstdint>
#include <filesystem>

using namespace LiteNN;

namespace
{
	float ReadFloat(const Tensor<CPU>& tensor, std::size_t index)
	{
		return static_cast<const float*>(tensor.UnsafeRawData())[index];
	}
} // namespace

TEST(Quantization, PerTensorAffineQuantizeDequantizeRoundTrip)
{
	const Tensor<CPU> source({ -1.0, 0.0, 1.5 }, { 3 }, DataType::Float32);
	const auto params = PerTensorAffineQuantization(DataType::Int8, 0.5F, 0);

	const auto quantized = QuantizeAffine(source, params);
	ASSERT_EQ(quantized.Storage().DType(), DataType::Int8);
	const auto* storage = static_cast<const std::int8_t*>(quantized.Storage().UnsafeRawData());
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
	const auto* storage = static_cast<const std::int8_t*>(quantized.Storage().UnsafeRawData());
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
	const auto* storage = static_cast<const std::int8_t*>(quantized.Storage().UnsafeRawData());
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
	const auto* storage = static_cast<const std::uint8_t*>(quantized.Storage().UnsafeRawData());
	EXPECT_EQ(storage[0], 126);
	EXPECT_EQ(storage[1], 128);
	EXPECT_EQ(storage[2], 130);

	const auto dequantized = DequantizeAffine(quantized);
	EXPECT_FLOAT_EQ(ReadFloat(dequantized, 0), -1.0F);
	EXPECT_FLOAT_EQ(ReadFloat(dequantized, 1), 0.0F);
	EXPECT_FLOAT_EQ(ReadFloat(dequantized, 2), 1.0F);
}

TEST(Quantization, PackedUInt4RoundTripsLowThenHigh)
{
	const Tensor<CPU> source({ 1.0, 15.0, 2.0 }, { 3 }, DataType::UInt8);
	auto params = PackedNibbleQuantization(PackedNibbleFormat::UInt4, { 3 });

	const auto packed = PackInteger4(source, params);
	ASSERT_EQ(packed.DType(), DataType::UInt8);
	ASSERT_EQ(packed.NumElements(), 2);
	const auto* packedData = static_cast<const std::uint8_t*>(packed.UnsafeRawData());
	EXPECT_EQ(packedData[0], 0xf1);
	EXPECT_EQ(packedData[1], 0x02);

	const auto unpacked = UnpackInteger4(packed, params);
	ASSERT_EQ(unpacked.DType(), DataType::UInt8);
	const auto* unpackedData = static_cast<const std::uint8_t*>(unpacked.UnsafeRawData());
	EXPECT_EQ(unpackedData[0], 1);
	EXPECT_EQ(unpackedData[1], 15);
	EXPECT_EQ(unpackedData[2], 2);
}

TEST(Quantization, PackedInt4RoundTripsHighThenLow)
{
	const Tensor<CPU> source({ -8.0, -1.0, 0.0, 7.0 }, { 4 }, DataType::Int8);
	auto params =
	    PackedNibbleQuantization(PackedNibbleFormat::Int4, { 4 }, 1.0F, 0, PackedNibbleOrder::HighThenLow);

	const auto packed = PackInteger4(source, params);
	const auto* packedData = static_cast<const std::uint8_t*>(packed.UnsafeRawData());
	EXPECT_EQ(packedData[0], 0x8f);
	EXPECT_EQ(packedData[1], 0x07);

	const auto unpacked = UnpackInteger4(packed, params);
	ASSERT_EQ(unpacked.DType(), DataType::Int8);
	const auto* unpackedData = static_cast<const std::int8_t*>(unpacked.UnsafeRawData());
	EXPECT_EQ(unpackedData[0], -8);
	EXPECT_EQ(unpackedData[1], -1);
	EXPECT_EQ(unpackedData[2], 0);
	EXPECT_EQ(unpackedData[3], 7);
}

TEST(Quantization, PackedInt4RejectsOutOfRangeValues)
{
	const Tensor<CPU> source({ -9.0 }, { 1 }, DataType::Int8);
	auto params = PackedNibbleQuantization(PackedNibbleFormat::Int4, { 1 });

	EXPECT_THROW((void)PackInteger4(source, params), std::runtime_error);
}

TEST(Quantization, PackedInt4DequantizesToFloat32)
{
	const Tensor<CPU> source({ -8.0, -1.0, 0.0, 7.0 }, { 4 }, DataType::Int8);
	auto params = PackedNibbleQuantization(PackedNibbleFormat::Int4, { 4 }, 0.5F);
	const auto packed = PackInteger4(source, params);

	const auto dequantized = DequantizePackedNibble(packed, params);
	EXPECT_FLOAT_EQ(ReadFloat(dequantized, 0), -4.0F);
	EXPECT_FLOAT_EQ(ReadFloat(dequantized, 1), -0.5F);
	EXPECT_FLOAT_EQ(ReadFloat(dequantized, 2), 0.0F);
	EXPECT_FLOAT_EQ(ReadFloat(dequantized, 3), 3.5F);
}

TEST(Quantization, PackedFP4E2M1QuantizesFiniteValues)
{
	const Tensor<CPU> source({ 0.0, 0.49, 0.75, 1.25, 2.5, 5.0, -6.5 }, { 7 }, DataType::Float32);
	auto params = PackedNibbleQuantization(PackedNibbleFormat::FP4E2M1, { 7 });

	const auto packed = PackFloat4(source, params);
	const auto* packedData = static_cast<const std::uint8_t*>(packed.UnsafeRawData());
	EXPECT_EQ(packedData[0], 0x10);
	EXPECT_EQ(packedData[1], 0x32);
	EXPECT_EQ(packedData[2], 0x75);
	EXPECT_EQ(packedData[3], 0x0f);

	const auto dequantized = DequantizePackedNibble(packed, params);
	EXPECT_FLOAT_EQ(ReadFloat(dequantized, 0), 0.0F);
	EXPECT_FLOAT_EQ(ReadFloat(dequantized, 1), 0.5F);
	EXPECT_FLOAT_EQ(ReadFloat(dequantized, 2), 1.0F);
	EXPECT_FLOAT_EQ(ReadFloat(dequantized, 3), 1.5F);
	EXPECT_FLOAT_EQ(ReadFloat(dequantized, 4), 3.0F);
	EXPECT_FLOAT_EQ(ReadFloat(dequantized, 5), 6.0F);
	EXPECT_FLOAT_EQ(ReadFloat(dequantized, 6), -6.0F);
}

TEST(Quantization, PackedFP4E3M0UsesPowerOfTwoGridAndScale)
{
	const Tensor<CPU> source({ 0.1, 0.3, 0.75, 3.0, 20.0, -5.0 }, { 6 }, DataType::Float32);
	auto params = PackedNibbleQuantization(PackedNibbleFormat::FP4E3M0, { 6 }, 2.0F);

	const auto packed = PackFloat4(source, params);
	const auto dequantized = DequantizePackedNibble(packed, params);
	EXPECT_FLOAT_EQ(ReadFloat(dequantized, 0), 0.0F);
	EXPECT_FLOAT_EQ(ReadFloat(dequantized, 1), 0.5F);
	EXPECT_FLOAT_EQ(ReadFloat(dequantized, 2), 1.0F);
	EXPECT_FLOAT_EQ(ReadFloat(dequantized, 3), 4.0F);
	EXPECT_FLOAT_EQ(ReadFloat(dequantized, 4), 16.0F);
	EXPECT_FLOAT_EQ(ReadFloat(dequantized, 5), -4.0F);
}

TEST(Quantization, RejectsInvalidScaleCount)
{
	const Tensor<CPU> source({ 1.0, 2.0, 3.0, 4.0 }, { 2, 2 }, DataType::Float32);
	const auto params = PerAxisAffineQuantization(DataType::Int8, 0, { 1.0F });
	EXPECT_THROW((void)QuantizeAffine(source, params), std::runtime_error);
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
	const auto results = interpreter.RunForward(Detail::BuildExecutablePlanFromGraph(graph), inputs);

	ASSERT_EQ(results.size(), 1);
	EXPECT_FLOAT_EQ(ReadFloat(results[0], 0), -1.0F);
	EXPECT_FLOAT_EQ(ReadFloat(results[0], 1), 0.0F);
	EXPECT_FLOAT_EQ(ReadFloat(results[0], 2), 1.0F);
}

TEST(Quantization, GraphPackedNibbleDequantizeRunsInInterpreter)
{
	auto params = PackedNibbleQuantization(PackedNibbleFormat::FP4E2M1, { 3 });
	const auto packed = PackFloat4(Tensor<CPU>({ 0.5, 1.5, -6.0 }, { 3 }, DataType::Float32), params);

	Graph graph;
	Subgraph sg;
	const auto input = sg.AddParam(DataType::UInt8, { 2 });
	const auto dq = sg.AddNode(DequantizeNode{ { input, 0 }, params, DataType::Float32 },
	                           { OutputInfo{ DataType::Float32, { 3 } } });
	sg.SetResults({ { dq, 0 } });
	graph.SetForward(graph.AddSubgraph(std::move(sg)));

	Runtime::Interpreter<CPU> interpreter;
	std::vector<Tensor<CPU>> inputs;
	inputs.push_back(packed);
	const auto results = interpreter.RunForward(Detail::BuildExecutablePlanFromGraph(graph), inputs);

	ASSERT_EQ(results.size(), 1);
	EXPECT_FLOAT_EQ(ReadFloat(results[0], 0), 0.5F);
	EXPECT_FLOAT_EQ(ReadFloat(results[0], 1), 1.5F);
	EXPECT_FLOAT_EQ(ReadFloat(results[0], 2), -6.0F);
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
	const auto results = interpreter.RunForward(Detail::BuildExecutablePlanFromGraph(graph), inputs);
	ASSERT_EQ(results.size(), 1);
	EXPECT_FLOAT_EQ(ReadFloat(results[0], 0), -1.0F);
	EXPECT_FLOAT_EQ(ReadFloat(results[0], 1), 0.0F);
	EXPECT_FLOAT_EQ(ReadFloat(results[0], 2), 1.0F);
}

TEST(Quantization, ConstFoldPackedNibbleDequantize)
{
	auto params = PackedNibbleQuantization(PackedNibbleFormat::FP4E3M0, { 3 }, 2.0F);
	const auto packed = PackFloat4(Tensor<CPU>({ 0.5, 4.0, -16.0 }, { 3 }, DataType::Float32), params);

	Graph graph;
	Subgraph sg;
	const auto c = sg.AddNode(ConstantNode{ packed.CopyToDevice(PolymorphicDevice{ CPU{} }) },
	                          { OutputInfo{ DataType::UInt8, { 2 } } });
	const auto dq = sg.AddNode(DequantizeNode{ { c, 0 }, params, DataType::Float32 },
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
	const auto results = interpreter.RunForward(Detail::BuildExecutablePlanFromGraph(graph), inputs);
	ASSERT_EQ(results.size(), 1);
	EXPECT_FLOAT_EQ(ReadFloat(results[0], 0), 0.5F);
	EXPECT_FLOAT_EQ(ReadFloat(results[0], 1), 4.0F);
	EXPECT_FLOAT_EQ(ReadFloat(results[0], 2), -16.0F);
}
