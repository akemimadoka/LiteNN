#include <gtest/gtest.h>

#include <LiteNN.h>

#include <array>
#include <span>

using namespace LiteNN;

TEST(TensorTypeTest, RepresentsStaticShapeLayoutAndByteSize)
{
	const auto type = TensorType::Dense(DataType::Float16, ShapeView{ 2, 3, 4 });

	EXPECT_EQ(type.dtype, DataType::Float16);
	EXPECT_EQ(type.Rank(), 3);
	EXPECT_TRUE(type.IsFullyStatic());
	EXPECT_EQ(type.StaticShape(), (std::vector<std::size_t>{ 2, 3, 4 }));
	ASSERT_TRUE(type.NumElements().has_value());
	EXPECT_EQ(*type.NumElements(), 24);
	ASSERT_TRUE(type.ByteSize().has_value());
	EXPECT_EQ(*type.ByteSize(), 48);
	EXPECT_EQ(type.layout.kind, TensorLayoutKind::RowMajor);
	EXPECT_EQ(type.memorySpace, TensorMemorySpace::Host);
}

TEST(TensorTypeTest, RepresentsSymbolicAndDynamicDimensions)
{
	TensorShape shape;
	shape.dims.push_back(TensorDim::Symbolic("batch"));
	shape.dims.push_back(TensorDim::Static(128));
	shape.dims.push_back(TensorDim::Dynamic());

	const auto type = TensorType::Ranked(DataType::Float32, std::move(shape),
	                                     TensorLayout::WithStrides(TensorLayoutKind::Torch, { 128, 1 }, "torch"));

	EXPECT_EQ(type.Rank(), 3);
	EXPECT_FALSE(type.IsFullyStatic());
	EXPECT_FALSE(type.NumElements().has_value());
	EXPECT_EQ(type.layout.kind, TensorLayoutKind::Torch);
	EXPECT_EQ(type.layout.tag, "torch");
	EXPECT_THROW((void) type.StaticShape(), std::runtime_error);
}

TEST(StorageTest, DescribesExternalWeightRegions)
{
	const std::byte bytes[8]{};
	const auto region = MakeExternalBufferRegion("weights.rodata", bytes, sizeof(bytes), ExternalBufferKind::Rodata);

	TensorStorageRef storage;
	storage.type = TensorType::Dense(DataType::Float32, ShapeView{ 2 });
	storage.quantization = PerTensorAffineQuantization(DataType::Int8, 0.25F, 3);
	storage.region = region;

	EXPECT_TRUE(storage.IsExternal());
	EXPECT_EQ(storage.region.externalKind, ExternalBufferKind::Rodata);
	EXPECT_EQ(storage.region.name, "weights.rodata");
	EXPECT_EQ(storage.region.alignment, 1);
	EXPECT_EQ(storage.region.mutability, BufferMutability::Immutable);
	EXPECT_EQ(storage.region.rebindPolicy, BufferRebindPolicy::ExactMetadataAndChecksum);
	ASSERT_TRUE(storage.LogicalByteSize().has_value());
	EXPECT_EQ(*storage.LogicalByteSize(), sizeof(bytes));

	storage.storageOffsetBytes = 4;
	storage.viewStrides = { 1 };
	storage.layoutTag = "packed";
	storage.aliasSet = 7;
	const auto view = storage.View();
	ASSERT_TRUE(view.quantization.has_value());
	EXPECT_EQ(view.quantization->storageType, DataType::Int8);
	EXPECT_EQ(view.storageOffsetBytes, 4);
	EXPECT_TRUE(view.HasExplicitStrides());
	EXPECT_EQ(view.layoutTag, "packed");
	EXPECT_EQ(view.aliasSet, 7);
}

TEST(GraphTypeTest, AcceptsTensorTypeConstructionAndReportsTypeSignatures)
{
	Graph graph;
	Subgraph subgraph;
	const auto dense = TensorType::Dense(DataType::Float32, ShapeView{ 2, 4 });
	const auto input = subgraph.AddParam(dense);
	const std::array<TensorType, 1> outputTypes{ dense };
	const auto neg =
	    subgraph.AddNode(UnaryOpNode{ UnaryOp::Negate, { input, 0 } }, std::span<const TensorType>{ outputTypes });
	subgraph.SetResults({ { neg, 0 } });

	graph.SetForward(graph.AddSubgraph(std::move(subgraph)));
	graph.SetInputNames({ "x" });
	graph.SetOutputNames({ "y" });
	graph.AddActivationSlot(dense);
	graph.AddTapeSlot(dense);

	EXPECT_EQ(graph.InputType(0), dense);
	EXPECT_EQ(graph.OutputType(0), dense);
	ASSERT_EQ(graph.InputTypeSignature().size(), 1);
	EXPECT_EQ(graph.InputTypeSignature()[0].name, "x");
	EXPECT_EQ(graph.InputTypeSignature()[0].type, dense);
	ASSERT_EQ(graph.OutputTypeSignature().size(), 1);
	EXPECT_EQ(graph.OutputTypeSignature()[0].name, "y");
	EXPECT_EQ(graph.OutputTypeSignature()[0].type, dense);
	EXPECT_EQ(graph.GetActivationSlot(0).Type(), dense);
	EXPECT_EQ(graph.GetTapeSlot(0).Type(), dense);
}
