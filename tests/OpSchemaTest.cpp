#include <gtest/gtest.h>

#include <LiteNN.h>

#include <ranges>
#include <string>
#include <variant>
#include <vector>

using namespace LiteNN;

TEST(OpSchemaTest, DefaultRegistryCoversEveryNodeVariantAlternative)
{
	const auto& registry = DefaultOpSchemaRegistry();

	EXPECT_EQ(registry.Schemas().size(), std::variant_size_v<NodeVariant>);
	EXPECT_TRUE(registry.Contains("ParamRefNode"));
	EXPECT_TRUE(registry.Contains("BinaryOpNode"));
	EXPECT_TRUE(registry.Contains("Conv2DNode"));
	EXPECT_TRUE(registry.Contains("FusedOpNode"));
}

TEST(OpSchemaTest, CapturesCoreArityAndCategories)
{
	const auto& registry = DefaultOpSchemaRegistry();

	const auto& binary = registry.Require("BinaryOpNode");
	EXPECT_EQ(binary.category, OpCategory::Elementwise);
	EXPECT_TRUE(binary.AllowsInputCount(2));
	EXPECT_FALSE(binary.AllowsInputCount(1));
	EXPECT_TRUE(binary.AllowsOutputCount(1));

	const auto& call = registry.Require("CallNode");
	EXPECT_EQ(call.category, OpCategory::ControlFlow);
	EXPECT_TRUE(call.AllowsInputCount(0));
	EXPECT_TRUE(call.AllowsInputCount(32));
	EXPECT_TRUE(call.AllowsOutputCount(0));

	const auto& sgd = registry.Require("SGDStepNode");
	EXPECT_EQ(sgd.category, OpCategory::Optimizer);
	EXPECT_TRUE(sgd.AllowsInputCount(2));
	EXPECT_TRUE(sgd.AllowsInputCount(3));
	EXPECT_FALSE(sgd.AllowsInputCount(4));
	EXPECT_TRUE(sgd.AllowsOutputCount(1));
	EXPECT_TRUE(sgd.AllowsOutputCount(2));
}

TEST(OpSchemaTest, ReportsBackendCapabilities)
{
	const auto& registry = DefaultOpSchemaRegistry();
	const auto& binary = registry.Require("BinaryOpNode");
	ASSERT_NE(binary.FindCapability(BackendCPUInterpreter), nullptr);
	EXPECT_TRUE(binary.SupportsBackend(BackendCPUInterpreter));
	EXPECT_EQ(binary.FindCapability(BackendCPUInterpreter)->support, BackendSupportLevel::Native);
	ASSERT_NE(binary.FindCapability(BackendCUDANative), nullptr);
	EXPECT_FALSE(binary.SupportsBackend(BackendCUDANative));

	auto custom = BuildDefaultOpSchemaRegistry();
	custom.RegisterCapability("BinaryOpNode",
	                          { .backend = std::string(BackendCUDANative),
	                            .support = BackendSupportLevel::Fallback,
	                            .dtypes = { DataType::Float32 },
	                            .layouts = { TensorLayoutKind::RowMajor },
	                            .relativeCost = 3.0 });
	const auto& updated = custom.Require("BinaryOpNode");
	ASSERT_NE(updated.FindCapability(BackendCUDANative), nullptr);
	EXPECT_EQ(updated.FindCapability(BackendCUDANative)->support, BackendSupportLevel::Fallback);
	EXPECT_EQ(updated.FindCapability(BackendCUDANative)->dtypes, (std::vector<DataType>{ DataType::Float32 }));
}

TEST(OpSchemaTest, BuildsCoverageReportForDefaultBackends)
{
	const auto& registry = DefaultOpSchemaRegistry();
	const auto report = registry.CoverageReport();

	ASSERT_EQ(report.size(), std::variant_size_v<NodeVariant>);
	const auto binary = std::ranges::find_if(report, [](const OpCoverageRow& row) {
		return row.kind == "BinaryOpNode";
	});
	ASSERT_NE(binary, report.end());
	EXPECT_EQ(binary->capabilities.size(), DefaultBackendNames.size());
	EXPECT_EQ(binary->capabilities[0].backend, BackendCPUInterpreter);
	EXPECT_EQ(binary->capabilities[0].support, BackendSupportLevel::Native);
	EXPECT_EQ(binary->capabilities[1].backend, BackendCPUAOT);
	EXPECT_EQ(binary->capabilities[1].support, BackendSupportLevel::Unsupported);
}

TEST(OpSchemaTest, ExtractsInputsFromNodePayloads)
{
	const NodeVariant add = BinaryOpNode{ BinaryOp::Add, { 1, 0 }, { 2, 0 } };
	const auto addInputs = NodeInputs(add);
	ASSERT_EQ(addInputs.size(), 2);
	EXPECT_EQ(addInputs[0].node, 1);
	EXPECT_EQ(addInputs[1].node, 2);

	const NodeVariant norm = NormalizationNode{ .input = { 3, 0 },
		                                        .scale = NodeOutput{ 4, 0 },
		                                        .bias = std::nullopt,
		                                        .mode = NormalizationMode::RMSNorm,
		                                        .axis = 1 };
	const auto normInputs = NodeInputs(norm);
	ASSERT_EQ(normInputs.size(), 2);
	EXPECT_EQ(normInputs[0].node, 3);
	EXPECT_EQ(normInputs[1].node, 4);
}
