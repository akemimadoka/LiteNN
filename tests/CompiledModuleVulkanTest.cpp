#include <gtest/gtest.h>

#include <LiteNN.h>
#include <LiteNN/Compiler/CompiledModule.h>
#include <LiteNN/Compiler/VulkanNativeCodegen.h>
#include <LiteNN/Compiler/VulkanNativePayload.h>
#include <LiteNN/Pass/FusionPass.h>

#include <array>
#include <cmath>
#include <limits>
#include <optional>
#include <stdexcept>
#include <string>
#include <vector>

using namespace LiteNN;

namespace
{
	constexpr std::uint32_t kElementCount = 4;

	Graph BuildSimpleBinaryGraph(BinaryOp op, std::size_t elementCount = kElementCount,
	                             DataType dtype = DataType::Float32)
	{
		Graph graph;
		Subgraph sg;
		const auto lhs = sg.AddParam(dtype, { elementCount });
		const auto rhs = sg.AddParam(dtype, { elementCount });
		const auto out = sg.AddNode(BinaryOpNode{ op, { lhs, 0 }, { rhs, 0 } },
		                            { OutputInfo{ dtype, { elementCount } } });
		sg.SetResults({ { out, 0 } });
		graph.AddSubgraph(std::move(sg));
		graph.SetForward(0);
		graph.SetInputNames({ "lhs", "rhs" });
		graph.SetOutputNames({ "sum" });
		return graph;
	}

	Graph BuildBinaryChainGraph(BinaryOp firstOp, BinaryOp secondOp)
	{
		Graph graph;
		Subgraph sg;
		const auto lhs = sg.AddParam(DataType::Float32, { 4 });
		const auto rhs = sg.AddParam(DataType::Float32, { 4 });
		const auto tail = sg.AddParam(DataType::Float32, { 4 });
		const auto first =
		    sg.AddNode(BinaryOpNode{ firstOp, { lhs, 0 }, { rhs, 0 } }, { OutputInfo{ DataType::Float32, { 4 } } });
		const auto second =
		    sg.AddNode(BinaryOpNode{ secondOp, { first, 0 }, { tail, 0 } }, { OutputInfo{ DataType::Float32, { 4 } } });
		sg.SetResults({ { second, 0 } });
		graph.AddSubgraph(std::move(sg));
		graph.SetForward(0);
		graph.SetInputNames({ "lhs", "rhs", "tail" });
		graph.SetOutputNames({ "sum" });
		return graph;
	}

	Graph BuildThreeStageBinaryChainGraph()
	{
		Graph graph;
		Subgraph sg;
		const auto a = sg.AddParam(DataType::Float32, { 4 });
		const auto b = sg.AddParam(DataType::Float32, { 4 });
		const auto c = sg.AddParam(DataType::Float32, { 4 });
		const auto d = sg.AddParam(DataType::Float32, { 4 });
		const auto first =
		    sg.AddNode(BinaryOpNode{ BinaryOp::Add, { a, 0 }, { b, 0 } }, { OutputInfo{ DataType::Float32, { 4 } } });
		const auto second = sg.AddNode(BinaryOpNode{ BinaryOp::Multiply, { first, 0 }, { c, 0 } },
		                               { OutputInfo{ DataType::Float32, { 4 } } });
		const auto third = sg.AddNode(BinaryOpNode{ BinaryOp::Subtract, { second, 0 }, { d, 0 } },
		                              { OutputInfo{ DataType::Float32, { 4 } } });
		sg.SetResults({ { third, 0 } });
		graph.AddSubgraph(std::move(sg));
		graph.SetForward(0);
		graph.SetInputNames({ "a", "b", "c", "d" });
		graph.SetOutputNames({ "out" });
		return graph;
	}

	Graph BuildBinaryDiamondGraph()
	{
		Graph graph;
		Subgraph sg;
		const auto lhs = sg.AddParam(DataType::Float32, { 4 });
		const auto rhs = sg.AddParam(DataType::Float32, { 4 });
		const auto tail = sg.AddParam(DataType::Float32, { 4 });
		const auto first = sg.AddNode(BinaryOpNode{ BinaryOp::Add, { lhs, 0 }, { rhs, 0 } },
		                              { OutputInfo{ DataType::Float32, { 4 } } });
		const auto second = sg.AddNode(BinaryOpNode{ BinaryOp::Add, { lhs, 0 }, { tail, 0 } },
		                               { OutputInfo{ DataType::Float32, { 4 } } });
		const auto out = sg.AddNode(BinaryOpNode{ BinaryOp::Add, { first, 0 }, { second, 0 } },
		                            { OutputInfo{ DataType::Float32, { 4 } } });
		sg.SetResults({ { out, 0 } });
		graph.AddSubgraph(std::move(sg));
		graph.SetForward(0);
		graph.SetInputNames({ "lhs", "rhs", "tail" });
		graph.SetOutputNames({ "sum" });
		return graph;
	}

	Graph BuildBranchedBinaryDAGWithTailGraph()
	{
		Graph graph;
		Subgraph sg;
		const auto a = sg.AddParam(DataType::Float32, { 4 });
		const auto b = sg.AddParam(DataType::Float32, { 4 });
		const auto c = sg.AddParam(DataType::Float32, { 4 });
		const auto d = sg.AddParam(DataType::Float32, { 4 });
		const auto e = sg.AddParam(DataType::Float32, { 4 });
		const auto first =
		    sg.AddNode(BinaryOpNode{ BinaryOp::Add, { a, 0 }, { b, 0 } }, { OutputInfo{ DataType::Float32, { 4 } } });
		const auto second =
		    sg.AddNode(BinaryOpNode{ BinaryOp::Add, { c, 0 }, { d, 0 } }, { OutputInfo{ DataType::Float32, { 4 } } });
		const auto merged = sg.AddNode(BinaryOpNode{ BinaryOp::Multiply, { first, 0 }, { second, 0 } },
		                               { OutputInfo{ DataType::Float32, { 4 } } });
		const auto tail = sg.AddNode(BinaryOpNode{ BinaryOp::Subtract, { first, 0 }, { e, 0 } },
		                             { OutputInfo{ DataType::Float32, { 4 } } });
		const auto out = sg.AddNode(BinaryOpNode{ BinaryOp::Add, { merged, 0 }, { tail, 0 } },
		                            { OutputInfo{ DataType::Float32, { 4 } } });
		sg.SetResults({ { out, 0 } });
		graph.AddSubgraph(std::move(sg));
		graph.SetForward(0);
		graph.SetInputNames({ "a", "b", "c", "d", "e" });
		graph.SetOutputNames({ "out" });
		return graph;
	}

	Graph BuildMixedElementwiseDAGGraph()
	{
		Graph graph;
		Subgraph sg;
		const auto lhs = sg.AddParam(DataType::Float32, { 4 });
		const auto rhs = sg.AddParam(DataType::Float32, { 4 });
		const auto tail = sg.AddParam(DataType::Float32, { 4 });
		const auto added = sg.AddNode(BinaryOpNode{ BinaryOp::Add, { lhs, 0 }, { rhs, 0 } },
		                              { OutputInfo{ DataType::Float32, { 4 } } });
		const auto abs =
		    sg.AddNode(UnaryOpNode{ UnaryOp::Abs, { added, 0 } }, { OutputInfo{ DataType::Float32, { 4 } } });
		const auto out = sg.AddNode(BinaryOpNode{ BinaryOp::Multiply, { abs, 0 }, { tail, 0 } },
		                            { OutputInfo{ DataType::Float32, { 4 } } });
		sg.SetResults({ { out, 0 } });
		graph.AddSubgraph(std::move(sg));
		graph.SetForward(0);
		graph.SetInputNames({ "lhs", "rhs", "tail" });
		graph.SetOutputNames({ "out" });
		return graph;
	}

	Graph BuildReduceGraph(ReduceOp op, std::size_t axis, std::vector<std::size_t> outputShape)
	{
		Graph graph;
		Subgraph sg;
		const auto input = sg.AddParam(DataType::Float32, { 2, 3 });
		const auto out = sg.AddNode(ReduceOpNode{ op, { input, 0 }, axis },
		                            { OutputInfo{ DataType::Float32, std::move(outputShape) } });
		sg.SetResults({ { out, 0 } });
		graph.AddSubgraph(std::move(sg));
		graph.SetForward(0);
		graph.SetInputNames({ "input" });
		graph.SetOutputNames({ "out" });
		return graph;
	}

	Graph BuildSoftmaxGraph(std::size_t axis)
	{
		Graph graph;
		Subgraph sg;
		const auto input = sg.AddParam(DataType::Float32, { 2, 3 });
		const auto out = sg.AddNode(SoftmaxNode{ { input, 0 }, axis }, { OutputInfo{ DataType::Float32, { 2, 3 } } });
		sg.SetResults({ { out, 0 } });
		graph.AddSubgraph(std::move(sg));
		graph.SetForward(0);
		graph.SetInputNames({ "input" });
		graph.SetOutputNames({ "out" });
		return graph;
	}

	Graph BuildNormalizationGraph(NormalizationMode mode, std::size_t axis, double epsilon = 1e-5)
	{
		Graph graph;
		Subgraph sg;
		const auto input = sg.AddParam(DataType::Float32, { 2, 3 });
		const auto out = sg.AddNode(NormalizationNode{ .input = { input, 0 },
		                                               .scale = std::nullopt,
		                                               .bias = std::nullopt,
		                                               .mode = mode,
		                                               .axis = axis,
		                                               .groupCount = 1,
		                                               .epsilon = epsilon },
		                            { OutputInfo{ DataType::Float32, { 2, 3 } } });
		sg.SetResults({ { out, 0 } });
		graph.AddSubgraph(std::move(sg));
		graph.SetForward(0);
		graph.SetInputNames({ "input" });
		graph.SetOutputNames({ "out" });
		return graph;
	}

	Graph BuildAffineNormalizationVariableGraph(NormalizationMode mode, std::size_t axis, double epsilon = 1e-5)
	{
		Graph graph;
		const auto scaleIndex =
		    graph.AddVariable(Variable::Create(Tensor<CPU>({ 2.0, 3.0, 4.0 }, { 3 }, DataType::Float32)));
		const auto biasIndex =
		    graph.AddVariable(Variable::Create(Tensor<CPU>({ 0.5, -0.5, 1.0 }, { 3 }, DataType::Float32)));
		graph.SetVariableName(scaleIndex, "norm_scale");
		graph.SetVariableName(biasIndex, "norm_bias");

		Subgraph sg;
		const auto input = sg.AddParam(DataType::Float32, { 2, 3 });
		const auto scale = sg.AddNode(VariableRefNode{ scaleIndex }, { OutputInfo{ DataType::Float32, { 3 } } });
		const auto bias = sg.AddNode(VariableRefNode{ biasIndex }, { OutputInfo{ DataType::Float32, { 3 } } });
		const auto out = sg.AddNode(NormalizationNode{ .input = { input, 0 },
		                                               .scale = NodeOutput{ scale, 0 },
		                                               .bias = NodeOutput{ bias, 0 },
		                                               .mode = mode,
		                                               .axis = axis,
		                                               .groupCount = 1,
		                                               .epsilon = epsilon },
		                            { OutputInfo{ DataType::Float32, { 2, 3 } } });
		sg.SetResults({ { out, 0 } });
		graph.AddSubgraph(std::move(sg));
		graph.SetForward(0);
		graph.SetInputNames({ "input" });
		graph.SetOutputNames({ "out" });
		return graph;
	}

	Graph BuildGroupNormGraph(std::vector<std::size_t> shape, std::size_t groupCount, double epsilon = 1e-6)
	{
		Graph graph;
		Subgraph sg;
		const auto input = sg.AddParam(DataType::Float32, shape);
		const auto out = sg.AddNode(NormalizationNode{ .input = { input, 0 },
		                                               .scale = std::nullopt,
		                                               .bias = std::nullopt,
		                                               .mode = NormalizationMode::GroupNorm,
		                                               .axis = 0,
		                                               .groupCount = groupCount,
		                                               .epsilon = epsilon },
		                            { OutputInfo{ DataType::Float32, std::move(shape) } });
		sg.SetResults({ { out, 0 } });
		graph.AddSubgraph(std::move(sg));
		graph.SetForward(0);
		graph.SetInputNames({ "input" });
		graph.SetOutputNames({ "out" });
		return graph;
	}

	Graph BuildPool2DGraph(PoolMode mode)
	{
		Graph graph;
		Subgraph sg;
		const auto input = sg.AddParam(DataType::Float32, { 1, 1, 3, 3 });
		const auto out = sg.AddNode(Pool2DNode{ .input = { input, 0 },
		                                        .mode = mode,
		                                        .kernelShape = { 2, 2 },
		                                        .strides = { 1, 1 },
		                                        .lowPads = { 0, 0 },
		                                        .highPads = { 0, 0 },
		                                        .countIncludePad = false },
		                            { OutputInfo{ DataType::Float32, { 1, 1, 2, 2 } } });
		sg.SetResults({ { out, 0 } });
		graph.AddSubgraph(std::move(sg));
		graph.SetForward(0);
		graph.SetInputNames({ "input" });
		graph.SetOutputNames({ "out" });
		return graph;
	}

	Graph BuildPaddedPool2DGraph(PoolMode mode, bool countIncludePad)
	{
		Graph graph;
		Subgraph sg;
		const auto input = sg.AddParam(DataType::Float32, { 1, 1, 2, 2 });
		const auto out = sg.AddNode(Pool2DNode{ .input = { input, 0 },
		                                        .mode = mode,
		                                        .kernelShape = { 2, 2 },
		                                        .strides = { 1, 1 },
		                                        .lowPads = { 1, 1 },
		                                        .highPads = { 1, 1 },
		                                        .countIncludePad = countIncludePad },
		                            { OutputInfo{ DataType::Float32, { 1, 1, 3, 3 } } });
		sg.SetResults({ { out, 0 } });
		graph.AddSubgraph(std::move(sg));
		graph.SetForward(0);
		graph.SetInputNames({ "input" });
		graph.SetOutputNames({ "out" });
		return graph;
	}

	Graph BuildNearestUpsampleGraph()
	{
		Graph graph;
		Subgraph sg;
		const auto input = sg.AddParam(DataType::Float32, { 1, 1, 2, 2 });
		const auto out = sg.AddNode(UpsampleNode{ .input = { input, 0 },
		                                          .mode = UpsampleMode::Nearest,
		                                          .outputSpatialShape = { 4, 4 },
		                                          .alignCorners = false },
		                            { OutputInfo{ DataType::Float32, { 1, 1, 4, 4 } } });
		sg.SetResults({ { out, 0 } });
		graph.AddSubgraph(std::move(sg));
		graph.SetForward(0);
		graph.SetInputNames({ "input" });
		graph.SetOutputNames({ "out" });
		return graph;
	}

	Graph BuildSliceGraph()
	{
		Graph graph;
		Subgraph sg;
		const auto input = sg.AddParam(DataType::Float32, { 2, 3 });
		const auto out = sg.AddNode(SliceNode{ { input, 0 }, 1, 1, 2 }, { OutputInfo{ DataType::Float32, { 2, 2 } } });
		sg.SetResults({ { out, 0 } });
		graph.AddSubgraph(std::move(sg));
		graph.SetForward(0);
		graph.SetInputNames({ "input" });
		graph.SetOutputNames({ "out" });
		return graph;
	}

	Graph BuildConcatGraph()
	{
		Graph graph;
		Subgraph sg;
		const auto lhs = sg.AddParam(DataType::Float32, { 2, 2 });
		const auto rhs = sg.AddParam(DataType::Float32, { 2, 1 });
		const auto out =
		    sg.AddNode(ConcatNode{ { { lhs, 0 }, { rhs, 0 } }, 1 }, { OutputInfo{ DataType::Float32, { 2, 3 } } });
		sg.SetResults({ { out, 0 } });
		graph.AddSubgraph(std::move(sg));
		graph.SetForward(0);
		graph.SetInputNames({ "lhs", "rhs" });
		graph.SetOutputNames({ "out" });
		return graph;
	}

	Graph BuildAffineGroupNormVariableGraph(std::vector<std::size_t> shape, std::size_t groupCount,
	                                        double epsilon = 1e-6)
	{
		Graph graph;
		auto groupedVolume = 1uz;
		for (auto dim = 0uz; dim < std::min<std::size_t>(shape.size(), 3); ++dim)
		{
			groupedVolume *= shape[dim];
		}
		std::vector<double> scale(groupedVolume);
		std::vector<double> bias(groupedVolume);
		for (std::size_t i = 0; i < groupedVolume; ++i)
		{
			scale[i] = 1.0 + 0.25 * static_cast<double>((i % 3) + 1);
			bias[i] = -0.25 + 0.125 * static_cast<double>(i % 5);
		}
		const auto scaleIndex =
		    graph.AddVariable(Variable::Create(Tensor<CPU>(std::move(scale), { groupedVolume }, DataType::Float32)));
		const auto biasIndex =
		    graph.AddVariable(Variable::Create(Tensor<CPU>(std::move(bias), { groupedVolume }, DataType::Float32)));
		graph.SetVariableName(scaleIndex, "group_norm_scale");
		graph.SetVariableName(biasIndex, "group_norm_bias");

		Subgraph sg;
		const auto input = sg.AddParam(DataType::Float32, shape);
		const auto scaleNode =
		    sg.AddNode(VariableRefNode{ scaleIndex }, { OutputInfo{ DataType::Float32, { groupedVolume } } });
		const auto biasNode =
		    sg.AddNode(VariableRefNode{ biasIndex }, { OutputInfo{ DataType::Float32, { groupedVolume } } });
		const auto out = sg.AddNode(NormalizationNode{ .input = { input, 0 },
		                                               .scale = NodeOutput{ scaleNode, 0 },
		                                               .bias = NodeOutput{ biasNode, 0 },
		                                               .mode = NormalizationMode::GroupNorm,
		                                               .axis = 0,
		                                               .groupCount = groupCount,
		                                               .epsilon = epsilon },
		                            { OutputInfo{ DataType::Float32, std::move(shape) } });
		sg.SetResults({ { out, 0 } });
		graph.AddSubgraph(std::move(sg));
		graph.SetForward(0);
		graph.SetInputNames({ "input" });
		graph.SetOutputNames({ "out" });
		return graph;
	}

	Graph BuildSimpleMatMulGraph()
	{
		Graph graph;
		Subgraph sg;
		const auto lhs = sg.AddParam(DataType::Float32, { 2, 3 });
		const auto rhs = sg.AddParam(DataType::Float32, { 3, 4 });
		const auto out = sg.AddNode(BinaryOpNode{ BinaryOp::MatMul, { lhs, 0 }, { rhs, 0 } },
		                            { OutputInfo{ DataType::Float32, { 2, 4 } } });
		sg.SetResults({ { out, 0 } });
		graph.AddSubgraph(std::move(sg));
		graph.SetForward(0);
		graph.SetInputNames({ "lhs", "rhs" });
		graph.SetOutputNames({ "out" });
		return graph;
	}

	Graph BuildSimpleMatMulBiasGraph(bool relu)
	{
		Graph graph;
		Subgraph sg;
		const auto lhs = sg.AddParam(DataType::Float32, { 2, 3 });
		const auto rhs = sg.AddParam(DataType::Float32, { 3, 4 });
		const auto bias = sg.AddParam(DataType::Float32, { 1, 4 });
		const auto matmul = sg.AddNode(BinaryOpNode{ BinaryOp::MatMul, { lhs, 0 }, { rhs, 0 } },
		                               { OutputInfo{ DataType::Float32, { 2, 4 } } });
		const auto shifted = sg.AddNode(BinaryOpNode{ BinaryOp::Add, { matmul, 0 }, { bias, 0 } },
		                                { OutputInfo{ DataType::Float32, { 2, 4 } } });
		NodeOutput result{ shifted, 0 };
		if (relu)
		{
			Tensor<CPU> zero({ 0.0f }, { 1, 1 }, DataType::Float32);
			const auto zeroNode = sg.AddNode(ConstantNode{ zero.CopyToDevice(PolymorphicDevice{ CPU{} }) },
			                                 { OutputInfo{ DataType::Float32, { 1, 1 } } });
			const auto reluOut = sg.AddNode(BinaryOpNode{ BinaryOp::Max, { shifted, 0 }, { zeroNode, 0 } },
			                                { OutputInfo{ DataType::Float32, { 2, 4 } } });
			result = { reluOut, 0 };
		}
		sg.SetResults({ result });
		graph.AddSubgraph(std::move(sg));
		graph.SetForward(0);
		graph.SetInputNames({ "lhs", "rhs", "bias" });
		graph.SetOutputNames({ relu ? "relu" : "out" });
		FusionPass{}.Run(graph);
		return graph;
	}

	Graph BuildSimpleMatMulBiasVariableGraph(bool relu)
	{
		Graph graph;
		const auto weightIndex = graph.AddVariable(Variable::Create(Tensor<CPU>(
		    { 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0, 12.0 }, { 3, 4 }, DataType::Float32)));
		const auto biasIndex =
		    graph.AddVariable(Variable::Create(Tensor<CPU>({ 1.0, -100.0, 3.0, -200.0 }, { 1, 4 }, DataType::Float32)));
		graph.SetVariableName(weightIndex, "weight");
		graph.SetVariableName(biasIndex, "bias");

		Subgraph sg;
		const auto lhs = sg.AddParam(DataType::Float32, { 2, 3 });
		const auto weight = sg.AddNode(VariableRefNode{ weightIndex }, { OutputInfo{ DataType::Float32, { 3, 4 } } });
		const auto matmul = sg.AddNode(BinaryOpNode{ BinaryOp::MatMul, { lhs, 0 }, { weight, 0 } },
		                               { OutputInfo{ DataType::Float32, { 2, 4 } } });
		const auto bias = sg.AddNode(VariableRefNode{ biasIndex }, { OutputInfo{ DataType::Float32, { 1, 4 } } });
		const auto shifted = sg.AddNode(BinaryOpNode{ BinaryOp::Add, { matmul, 0 }, { bias, 0 } },
		                                { OutputInfo{ DataType::Float32, { 2, 4 } } });
		NodeOutput result{ shifted, 0 };
		if (relu)
		{
			Tensor<CPU> zero({ 0.0f }, { 1, 1 }, DataType::Float32);
			const auto zeroNode = sg.AddNode(ConstantNode{ zero.CopyToDevice(PolymorphicDevice{ CPU{} }) },
			                                 { OutputInfo{ DataType::Float32, { 1, 1 } } });
			const auto reluOut = sg.AddNode(BinaryOpNode{ BinaryOp::Max, { shifted, 0 }, { zeroNode, 0 } },
			                                { OutputInfo{ DataType::Float32, { 2, 4 } } });
			result = { reluOut, 0 };
		}
		sg.SetResults({ result });
		graph.AddSubgraph(std::move(sg));
		graph.SetForward(0);
		graph.SetInputNames({ "lhs" });
		graph.SetOutputNames({ relu ? "relu" : "out" });
		FusionPass{}.Run(graph);
		return graph;
	}

	Graph BuildSimpleConv2DVariableGraph()
	{
		Graph graph;
		const auto weightIndex =
		    graph.AddVariable(Variable::Create(Tensor<CPU>({ 1.0, 0.0, 0.0, 1.0 }, { 1, 1, 2, 2 }, DataType::Float32)));
		const auto biasIndex = graph.AddVariable(Variable::Create(Tensor<CPU>({ 0.5 }, { 1 }, DataType::Float32)));
		graph.SetVariableName(weightIndex, "conv_weight");
		graph.SetVariableName(biasIndex, "conv_bias");

		Subgraph sg;
		const auto input = sg.AddParam(DataType::Float32, { 1, 1, 3, 3 });
		const auto weight =
		    sg.AddNode(VariableRefNode{ weightIndex }, { OutputInfo{ DataType::Float32, { 1, 1, 2, 2 } } });
		const auto bias = sg.AddNode(VariableRefNode{ biasIndex }, { OutputInfo{ DataType::Float32, { 1 } } });
		const auto out = sg.AddNode(Conv2DNode{ .input = { input, 0 },
		                                        .weight = { weight, 0 },
		                                        .bias = NodeOutput{ bias, 0 },
		                                        .strides = { 1, 1 },
		                                        .dilations = { 1, 1 },
		                                        .lowPads = { 0, 0 },
		                                        .highPads = { 0, 0 },
		                                        .groupCount = 1 },
		                            { OutputInfo{ DataType::Float32, { 1, 1, 2, 2 } } });
		sg.SetResults({ { out, 0 } });
		graph.AddSubgraph(std::move(sg));
		graph.SetForward(0);
		graph.SetInputNames({ "input" });
		graph.SetOutputNames({ "out" });
		return graph;
	}

	Graph BuildGroupedConv2DVariableGraph()
	{
		Graph graph;
		const auto weightIndex = graph.AddVariable(Variable::Create(
		    Tensor<CPU>({ 1.0, 0.0, 0.0, 1.0, 1.0, 1.0, 1.0, 1.0 }, { 2, 1, 2, 2 }, DataType::Float32)));
		const auto biasIndex = graph.AddVariable(Variable::Create(Tensor<CPU>({ 0.5, 1.0 }, { 2 }, DataType::Float32)));
		graph.SetVariableName(weightIndex, "grouped_conv_weight");
		graph.SetVariableName(biasIndex, "grouped_conv_bias");

		Subgraph sg;
		const auto input = sg.AddParam(DataType::Float32, { 1, 2, 3, 3 });
		const auto weight =
		    sg.AddNode(VariableRefNode{ weightIndex }, { OutputInfo{ DataType::Float32, { 2, 1, 2, 2 } } });
		const auto bias = sg.AddNode(VariableRefNode{ biasIndex }, { OutputInfo{ DataType::Float32, { 2 } } });
		const auto out = sg.AddNode(Conv2DNode{ .input = { input, 0 },
		                                        .weight = { weight, 0 },
		                                        .bias = NodeOutput{ bias, 0 },
		                                        .strides = { 1, 1 },
		                                        .dilations = { 1, 1 },
		                                        .lowPads = { 0, 0 },
		                                        .highPads = { 0, 0 },
		                                        .groupCount = 2 },
		                            { OutputInfo{ DataType::Float32, { 1, 2, 2, 2 } } });
		sg.SetResults({ { out, 0 } });
		graph.AddSubgraph(std::move(sg));
		graph.SetForward(0);
		graph.SetInputNames({ "input" });
		graph.SetOutputNames({ "out" });
		return graph;
	}

	Graph BuildSimpleConvTranspose2DVariableGraph()
	{
		Graph graph;
		const auto weightIndex =
		    graph.AddVariable(Variable::Create(Tensor<CPU>({ 1.0, 1.0, 1.0, 1.0 }, { 1, 1, 2, 2 }, DataType::Float32)));
		const auto biasIndex = graph.AddVariable(Variable::Create(Tensor<CPU>({ 0.5 }, { 1 }, DataType::Float32)));
		graph.SetVariableName(weightIndex, "conv_transpose_weight");
		graph.SetVariableName(biasIndex, "conv_transpose_bias");

		Subgraph sg;
		const auto input = sg.AddParam(DataType::Float32, { 1, 1, 2, 2 });
		const auto weight =
		    sg.AddNode(VariableRefNode{ weightIndex }, { OutputInfo{ DataType::Float32, { 1, 1, 2, 2 } } });
		const auto bias = sg.AddNode(VariableRefNode{ biasIndex }, { OutputInfo{ DataType::Float32, { 1 } } });
		const auto out = sg.AddNode(ConvTranspose2DNode{ .input = { input, 0 },
		                                                 .weight = { weight, 0 },
		                                                 .bias = NodeOutput{ bias, 0 },
		                                                 .strides = { 1, 1 },
		                                                 .dilations = { 1, 1 },
		                                                 .lowPads = { 0, 0 },
		                                                 .highPads = { 0, 0 },
		                                                 .outputPads = { 0, 0 },
		                                                 .groupCount = 1 },
		                            { OutputInfo{ DataType::Float32, { 1, 1, 3, 3 } } });
		sg.SetResults({ { out, 0 } });
		graph.AddSubgraph(std::move(sg));
		graph.SetForward(0);
		graph.SetInputNames({ "input" });
		graph.SetOutputNames({ "out" });
		return graph;
	}

	Graph BuildSimpleUnaryGraph(UnaryOp op, DataType dtype = DataType::Float32)
	{
		Graph graph;
		Subgraph sg;
		const auto input = sg.AddParam(dtype, { 4 });
		const auto out = sg.AddNode(UnaryOpNode{ op, { input, 0 } }, { OutputInfo{ dtype, { 4 } } });
		sg.SetResults({ { out, 0 } });
		graph.AddSubgraph(std::move(sg));
		graph.SetForward(0);
		graph.SetInputNames({ "input" });
		graph.SetOutputNames({ "output" });
		return graph;
	}

	Graph BuildSimpleCastGraph(DataType srcType, DataType dstType)
	{
		Graph graph;
		Subgraph sg;
		const auto input = sg.AddParam(srcType, { 4 });
		const auto out = sg.AddNode(CastNode{ { input, 0 }, dstType }, { OutputInfo{ dstType, { 4 } } });
		sg.SetResults({ { out, 0 } });
		graph.AddSubgraph(std::move(sg));
		graph.SetForward(0);
		graph.SetInputNames({ "input" });
		graph.SetOutputNames({ "output" });
		return graph;
	}

	std::array<float, 4> CopyToHost(const Tensor<Vulkan>& tensor)
	{
		Tensor<CPU> host(Uninitialized, tensor.Shape(), tensor.DType(), CPU{});
		auto device = tensor.CurDevice();
		DeviceTraits<Vulkan>::CopyToCPU(device, tensor.DType(), tensor.UnsafeRawData(), tensor.NumElements(),
		                                host.DType(), host.UnsafeRawData());
		const auto* values = static_cast<const float*>(host.UnsafeRawData());
		return { values[0], values[1], values[2], values[3] };
	}

	std::array<float, 4> CopyToHostAsFloat32(const Tensor<Vulkan>& tensor)
	{
		Tensor<CPU> host(Uninitialized, tensor.Shape(), DataType::Float32, CPU{});
		auto device = tensor.CurDevice();
		DeviceTraits<Vulkan>::CopyToCPU(device, tensor.DType(), tensor.UnsafeRawData(), tensor.NumElements(),
		                                host.DType(), host.UnsafeRawData());
		const auto* values = static_cast<const float*>(host.UnsafeRawData());
		return { values[0], values[1], values[2], values[3] };
	}

	std::vector<float> CopyToHostVector(const Tensor<Vulkan>& tensor)
	{
		Tensor<CPU> host(Uninitialized, tensor.Shape(), DataType::Float32, CPU{});
		auto device = tensor.CurDevice();
		DeviceTraits<Vulkan>::CopyToCPU(device, tensor.DType(), tensor.UnsafeRawData(), tensor.NumElements(),
		                                host.DType(), host.UnsafeRawData());
		const auto* values = static_cast<const float*>(host.UnsafeRawData());
		return std::vector<float>(values, values + host.NumElements());
	}

	CompiledModuleImage ImageWithInstructions(const CompiledModuleArtifact& artifact,
	                                          std::span<const std::byte> instructions)
	{
		return {
			.rodata = artifact.Rodata().data(),
			.rodataSize = artifact.Rodata().size(),
			.instructions = instructions.data(),
			.instructionSize = instructions.size(),
		};
	}

	void ExpectFloat16FeatureGateRejectsLoad(const CompiledModuleArtifact& artifact, Vulkan& device)
	{
		try
		{
			(void) artifact.Load(device);
			FAIL() << "Expected low-precision Vulkan artifact loading to require enabled device features";
		}
		catch (const std::runtime_error& ex)
		{
			const std::string message = ex.what();
			EXPECT_TRUE(message.find("shaderFloat16") != std::string::npos ||
			            message.find("storageBuffer16BitAccess") != std::string::npos)
			    << message;
			EXPECT_NE(message.find("enabled=false"), std::string::npos);
		}
	}

	struct BinaryCase
	{
		BinaryOp op;
		std::string_view mlirOp;
		std::array<float, 4> expected;
	};

	struct UnaryCase
	{
		UnaryOp op;
		std::string_view mlirOp;
		std::array<double, 4> input;
		float tolerance;
	};

	struct CastCase
	{
		DataType srcType;
		DataType dstType;
		std::string_view mlirOp;
		std::array<double, 4> input;
		std::array<float, 4> expected;
	};

	constexpr std::array kBinaryCases{
		BinaryCase{ BinaryOp::Add, "spirv.FAdd", { 11.0f, 22.0f, 33.0f, 44.0f } },
		BinaryCase{ BinaryOp::Subtract, "spirv.FSub", { -9.0f, -18.0f, -27.0f, -36.0f } },
		BinaryCase{ BinaryOp::Multiply, "spirv.FMul", { 10.0f, 40.0f, 90.0f, 160.0f } },
		BinaryCase{ BinaryOp::Divide, "spirv.FDiv", { 0.1f, 0.1f, 0.1f, 0.1f } },
		BinaryCase{ BinaryOp::Max, "spirv.GL.FMax", { 10.0f, 20.0f, 30.0f, 40.0f } },
		BinaryCase{ BinaryOp::Min, "spirv.GL.FMin", { 1.0f, 2.0f, 3.0f, 4.0f } },
	};

	constexpr std::array kUnaryCases{
		UnaryCase{ UnaryOp::Negate, "spirv.FNegate", { -4.0f, -1.0f, 0.0f, 9.0f }, 0.0f },
		UnaryCase{ UnaryOp::Abs, "spirv.GL.FAbs", { -4.0f, -1.0f, 0.0f, 9.0f }, 0.0f },
		UnaryCase{ UnaryOp::Sqrt, "spirv.GL.Sqrt", { 4.0f, 1.0f, 0.25f, 9.0f }, 1e-5f },
		UnaryCase{ UnaryOp::Exp, "spirv.GL.Exp", { -1.0f, 0.0f, 1.0f, 2.0f }, 1e-4f },
		UnaryCase{ UnaryOp::Log, "spirv.GL.Log", { 0.25f, 1.0f, 2.0f, 4.0f }, 1e-4f },
		UnaryCase{ UnaryOp::Sin, "spirv.GL.Sin", { -1.0f, 0.0f, 1.0f, 2.0f }, 1e-4f },
		UnaryCase{ UnaryOp::Cos, "spirv.GL.Cos", { -1.0f, 0.0f, 1.0f, 2.0f }, 1e-4f },
	};

	constexpr std::array kCastCases{
		CastCase{ DataType::Float32,
		          DataType::Int32,
		          "spirv.ConvertFToS",
		          { -3.5, -1.0, 0.75, 4.0 },
		          { -3.0f, -1.0f, 0.0f, 4.0f } },
		CastCase{ DataType::Int32,
		          DataType::Float32,
		          "spirv.ConvertSToF",
		          { -3.0, -1.0, 0.0, 4.0 },
		          { -3.0f, -1.0f, 0.0f, 4.0f } },
		CastCase{ DataType::Float32,
		          DataType::Float16,
		          "spirv.FConvert",
		          { -3.5, -1.0, 0.75, 4.0 },
		          { -3.5f, -1.0f, 0.75f, 4.0f } },
		CastCase{ DataType::Float16,
		          DataType::Float32,
		          "spirv.FConvert",
		          { -3.5, -1.0, 0.75, 4.0 },
		          { -3.5f, -1.0f, 0.75f, 4.0f } },
		CastCase{ DataType::Float32,
		          DataType::Int8,
		          "spirv.ConvertFToS",
		          { -3.5, -1.0, 0.75, 4.0 },
		          { -3.0f, -1.0f, 0.0f, 4.0f } },
		CastCase{ DataType::Int8,
		          DataType::Float32,
		          "spirv.ConvertSToF",
		          { -3.0, -1.0, 0.0, 4.0 },
		          { -3.0f, -1.0f, 0.0f, 4.0f } },
		CastCase{ DataType::Float32,
		          DataType::UInt8,
		          "spirv.ConvertFToU",
		          { 0.0, 1.0, 2.75, 4.0 },
		          { 0.0f, 1.0f, 2.0f, 4.0f } },
		CastCase{ DataType::UInt8,
		          DataType::Float32,
		          "spirv.ConvertUToF",
		          { 0.0, 1.0, 2.0, 4.0 },
		          { 0.0f, 1.0f, 2.0f, 4.0f } },
		CastCase{
		    DataType::Int32, DataType::Int8, "spirv.SConvert", { -3.0, -1.0, 0.0, 4.0 }, { -3.0f, -1.0f, 0.0f, 4.0f } },
		CastCase{
		    DataType::UInt8, DataType::Int32, "spirv.UConvert", { 0.0, 1.0, 2.0, 4.0 }, { 0.0f, 1.0f, 2.0f, 4.0f } },
	};

	constexpr std::array kRuntimeCastCases{
		kCastCases[0],
		kCastCases[1],
	};

	float ExpectedUnaryValue(UnaryOp op, double value)
	{
		switch (op)
		{
		case UnaryOp::Negate:
			return -value;
		case UnaryOp::Abs:
			return std::fabs(value);
		case UnaryOp::Sqrt:
			return std::sqrt(value);
		case UnaryOp::Exp:
			return std::exp(value);
		case UnaryOp::Log:
			return std::log(value);
		case UnaryOp::Sin:
			return std::sin(value);
		case UnaryOp::Cos:
			return std::cos(value);
		default:
			throw std::runtime_error("Unexpected unary test op");
		}
	}
} // namespace

TEST(CompiledModuleVulkanTest, GeneratesSimpleAddSPIRVFromMLIR)
{
	for (const auto& item : kBinaryCases)
	{
		const auto generated = VulkanNativeSameShapeBinaryF32SPIRV(item.op, kElementCount);
		EXPECT_FALSE(generated.words.empty());
		EXPECT_NE(generated.mlir.find("spirv.module"), std::string::npos);
		EXPECT_NE(generated.mlir.find(item.mlirOp), std::string::npos);
		EXPECT_NE(generated.mlir.find("spirv.ULessThan"), std::string::npos);
		EXPECT_NE(generated.mlir.find("spirv.mlir.selection"), std::string::npos);
		EXPECT_NE(generated.mlir.find("spirv.EntryPoint"), std::string::npos);
		EXPECT_NE(generated.mlir.find("LocalSize"), std::string::npos);
		EXPECT_NE(generated.mlir.find("64, 1, 1"), std::string::npos);
	}
}

TEST(CompiledModuleVulkanTest, GeneratesFloat16BinarySPIRVFromMLIR)
{
	const auto generated = VulkanNativeSameShapeBinarySPIRV(DataType::Float16, BinaryOp::Add, kElementCount);
	EXPECT_FALSE(generated.words.empty());
	EXPECT_NE(generated.mlir.find("spirv.module"), std::string::npos);
	EXPECT_NE(generated.mlir.find("spirv.FAdd"), std::string::npos);
	EXPECT_NE(generated.mlir.find("f16"), std::string::npos);
	EXPECT_NE(generated.mlir.find("StorageBuffer16BitAccess"), std::string::npos);
	EXPECT_NE(generated.mlir.find("SPV_KHR_16bit_storage"), std::string::npos);
}

TEST(CompiledModuleVulkanTest, GeneratesSimpleUnarySPIRVFromMLIR)
{
	for (const auto& item : kUnaryCases)
	{
		const auto generated = VulkanNativeSameShapeUnaryF32SPIRV(item.op, kElementCount);
		EXPECT_FALSE(generated.words.empty());
		EXPECT_NE(generated.mlir.find("spirv.module"), std::string::npos);
		EXPECT_NE(generated.mlir.find(item.mlirOp), std::string::npos);
		EXPECT_NE(generated.mlir.find("spirv.ULessThan"), std::string::npos);
		EXPECT_NE(generated.mlir.find("spirv.mlir.selection"), std::string::npos);
		EXPECT_NE(generated.mlir.find("spirv.EntryPoint"), std::string::npos);
		EXPECT_NE(generated.mlir.find("LocalSize"), std::string::npos);
		EXPECT_NE(generated.mlir.find("64, 1, 1"), std::string::npos);
	}
}

TEST(CompiledModuleVulkanTest, GeneratesFloat16UnarySPIRVFromMLIR)
{
	const auto generated = VulkanNativeSameShapeUnarySPIRV(DataType::Float16, UnaryOp::Abs, kElementCount);
	EXPECT_FALSE(generated.words.empty());
	EXPECT_NE(generated.mlir.find("spirv.module"), std::string::npos);
	EXPECT_NE(generated.mlir.find("spirv.GL.FAbs"), std::string::npos);
	EXPECT_NE(generated.mlir.find("f16"), std::string::npos);
	EXPECT_NE(generated.mlir.find("StorageBuffer16BitAccess"), std::string::npos);
	EXPECT_NE(generated.mlir.find("SPV_KHR_16bit_storage"), std::string::npos);
}

TEST(CompiledModuleVulkanTest, GeneratesSimpleCastSPIRVFromMLIR)
{
	for (const auto& item : kCastCases)
	{
		const auto generated = VulkanNativeSameShapeCastSPIRV(item.srcType, item.dstType, kElementCount);
		EXPECT_FALSE(generated.words.empty());
		EXPECT_NE(generated.mlir.find("spirv.module"), std::string::npos);
		EXPECT_NE(generated.mlir.find(item.mlirOp), std::string::npos);
		EXPECT_NE(generated.mlir.find("spirv.ULessThan"), std::string::npos);
		EXPECT_NE(generated.mlir.find("spirv.mlir.selection"), std::string::npos);
		EXPECT_NE(generated.mlir.find("spirv.EntryPoint"), std::string::npos);
		EXPECT_NE(generated.mlir.find("LocalSize"), std::string::npos);
		EXPECT_NE(generated.mlir.find("64, 1, 1"), std::string::npos);
	}
}

TEST(CompiledModuleVulkanTest, GeneratesSimpleMatMulSPIRVFromMLIR)
{
	const auto generated = VulkanNativeMatMulF32SPIRV(2, 3, 4);
	EXPECT_FALSE(generated.words.empty());
	EXPECT_NE(generated.mlir.find("spirv.module"), std::string::npos);
	EXPECT_NE(generated.mlir.find("spirv.FMul"), std::string::npos);
	EXPECT_NE(generated.mlir.find("spirv.FAdd"), std::string::npos);
	EXPECT_NE(generated.mlir.find("spirv.UDiv"), std::string::npos);
	EXPECT_NE(generated.mlir.find("spirv.UMod"), std::string::npos);
	EXPECT_NE(generated.mlir.find("spirv.EntryPoint"), std::string::npos);
	EXPECT_NE(generated.mlir.find("LocalSize"), std::string::npos);
	EXPECT_NE(generated.mlir.find("64, 1, 1"), std::string::npos);
}

TEST(CompiledModuleVulkanTest, GeneratesSimpleMatMulBiasReLUSPIRVFromMLIR)
{
	const auto generated = VulkanNativeMatMulBiasF32SPIRV(2, 3, 4, 1, true);
	EXPECT_FALSE(generated.words.empty());
	EXPECT_NE(generated.mlir.find("spirv.module"), std::string::npos);
	EXPECT_NE(generated.mlir.find("spirv.FMul"), std::string::npos);
	EXPECT_NE(generated.mlir.find("spirv.FAdd"), std::string::npos);
	EXPECT_NE(generated.mlir.find("spirv.GL.FMax"), std::string::npos);
	EXPECT_NE(generated.mlir.find("spirv.EntryPoint"), std::string::npos);
	EXPECT_NE(generated.mlir.find("LocalSize"), std::string::npos);
	EXPECT_NE(generated.mlir.find("64, 1, 1"), std::string::npos);
}

TEST(CompiledModuleVulkanTest, WritesVulkanNativePayloadForSimpleAdd)
{
	const auto graph = BuildSimpleBinaryGraph(BinaryOp::Add);
	const auto artifact = Compiler<Vulkan>::CompileArtifact(Detail::BuildExecutablePlanFromGraph(graph));
	EXPECT_EQ(artifact.Backend(), CompiledModuleBackend::VulkanNative);
	EXPECT_FALSE(artifact.Instructions().empty());

	const auto payload = DeserializeVulkanNativeInstructionPayload(artifact.Instructions());
	const auto generated = VulkanNativeSameShapeBinaryF32SPIRV(BinaryOp::Add, kElementCount);
	EXPECT_EQ(payload.spirv, generated.words);
	EXPECT_TRUE(payload.workspaceTensors.empty());
	ASSERT_EQ(payload.kernels.size(), 1u);
	EXPECT_EQ(payload.kernels[0].groups.x, 1u);
	EXPECT_EQ(payload.kernels[0].requirements.descriptorAbiVersion, 1u);
	EXPECT_EQ(payload.kernels[0].requirements.localSize.x, kVulkanNativeElementwiseWorkgroupSize);
	EXPECT_EQ(payload.kernels[0].requirements.localSize.y, 1u);
	EXPECT_EQ(payload.kernels[0].requirements.localSize.z, 1u);
	EXPECT_EQ(payload.kernels[0].requirements.deviceRequirements.flags, 0ull);
}

TEST(CompiledModuleVulkanTest, WritesVulkanNativePayloadForFloat16Add)
{
	const auto graph = BuildSimpleBinaryGraph(BinaryOp::Add, kElementCount, DataType::Float16);
	const auto artifact = Compiler<Vulkan>::CompileArtifact(Detail::BuildExecutablePlanFromGraph(graph));
	EXPECT_EQ(artifact.Backend(), CompiledModuleBackend::VulkanNative);
	EXPECT_FALSE(artifact.Instructions().empty());

	const auto payload = DeserializeVulkanNativeInstructionPayload(artifact.Instructions());
	const auto generated = VulkanNativeSameShapeBinarySPIRV(DataType::Float16, BinaryOp::Add, kElementCount);
	EXPECT_EQ(payload.spirv, generated.words);
	EXPECT_NE(payload.featureSet.flags &
	              (1ull << static_cast<std::uint32_t>(VulkanNativeFeature::SameShapeElementwiseBinaryLowPrecision)),
	          0ull);
	ASSERT_EQ(payload.kernels.size(), 1u);
	EXPECT_EQ(payload.kernels[0].arguments[0].byteSize, kElementCount * ElementByteSize(DataType::Float16));
	EXPECT_EQ(payload.kernels[0].arguments[1].byteSize, kElementCount * ElementByteSize(DataType::Float16));
	EXPECT_EQ(payload.kernels[0].arguments[2].byteSize, kElementCount * ElementByteSize(DataType::Float16));
	EXPECT_TRUE(payload.kernels[0].requirements.deviceRequirements.HasRequirement(
	    VulkanNativeDeviceRequirement::ShaderFloat16));
	EXPECT_TRUE(payload.kernels[0].requirements.deviceRequirements.HasRequirement(
	    VulkanNativeDeviceRequirement::StorageBuffer16BitAccess));
}

TEST(CompiledModuleVulkanTest, SerializesKernelRequirementMetadata)
{
	VulkanNativeInstructionPayload payload;
	payload.featureSet.AddFeature(VulkanNativeFeature::StaticShape);
	payload.featureSet.AddFeature(VulkanNativeFeature::SingleSubgraph);
	payload.spirv = { 0x07230203u };
	VulkanNativeKernelSpec kernel;
	kernel.entryPoint = "main";
	kernel.groups = { .x = 2, .y = 1, .z = 1 };
	kernel.requirements.descriptorAbiVersion = 1;
	kernel.requirements.localSize = { .x = kVulkanNativeElementwiseWorkgroupSize, .y = 1, .z = 1 };
	kernel.requirements.deviceRequirements.AddRequirement(VulkanNativeDeviceRequirement::ShaderInt8);
	kernel.requirements.deviceRequirements.AddRequirement(VulkanNativeDeviceRequirement::StorageBuffer8BitAccess);
	kernel.requirements.deviceRequirements.AddRequirement(VulkanNativeDeviceRequirement::SubgroupArithmetic);
	kernel.requirements.deviceRequirements.AddRequirement(VulkanNativeDeviceRequirement::RuntimeDescriptorArray);
	kernel.requirements.requiredSubgroupSize = 32;
	kernel.requirements.requiredStorageBufferOffsetAlignment = 16;
	kernel.specializationData = { std::byte{ 64 }, std::byte{ 0 }, std::byte{ 0 }, std::byte{ 0 } };
	kernel.specializationConstants.push_back({
	    .constantId = 7,
	    .byteOffset = 0,
	    .byteSize = 4,
	});
	kernel.arguments.push_back({
	    .kind = VulkanNativeArgumentKind::InputTensor,
	    .index = 0,
	    .binding = 0,
	    .byteOffset = 0,
	    .byteSize = 16,
	});
	payload.kernels.push_back(std::move(kernel));

	const auto decoded = DeserializeVulkanNativeInstructionPayload(SerializeVulkanNativeInstructionPayload(payload));
	ASSERT_EQ(decoded.kernels.size(), 1u);
	const auto& decodedRequirements = decoded.kernels[0].requirements;
	EXPECT_EQ(decodedRequirements.descriptorAbiVersion, 1u);
	EXPECT_EQ(decodedRequirements.localSize.x, kVulkanNativeElementwiseWorkgroupSize);
	EXPECT_TRUE(decodedRequirements.deviceRequirements.HasRequirement(VulkanNativeDeviceRequirement::ShaderInt8));
	EXPECT_TRUE(
	    decodedRequirements.deviceRequirements.HasRequirement(VulkanNativeDeviceRequirement::StorageBuffer8BitAccess));
	EXPECT_TRUE(
	    decodedRequirements.deviceRequirements.HasRequirement(VulkanNativeDeviceRequirement::SubgroupArithmetic));
	EXPECT_TRUE(
	    decodedRequirements.deviceRequirements.HasRequirement(VulkanNativeDeviceRequirement::RuntimeDescriptorArray));
	EXPECT_EQ(decodedRequirements.requiredSubgroupSize, 32u);
	EXPECT_EQ(decodedRequirements.requiredStorageBufferOffsetAlignment, 16u);
	ASSERT_EQ(decoded.kernels[0].specializationData.size(), 4u);
	ASSERT_EQ(decoded.kernels[0].specializationConstants.size(), 1u);
	EXPECT_EQ(decoded.kernels[0].specializationConstants[0].constantId, 7u);
	EXPECT_EQ(decoded.kernels[0].specializationConstants[0].byteOffset, 0u);
	EXPECT_EQ(decoded.kernels[0].specializationConstants[0].byteSize, 4u);

	payload.kernels[0].requirements.deviceRequirements.flags = 1ull << 63;
	EXPECT_THROW((void) SerializeVulkanNativeInstructionPayload(payload), std::runtime_error);
	payload.kernels[0].requirements.deviceRequirements.flags = 0;
	payload.kernels[0].specializationConstants[0].byteOffset = 2;
	EXPECT_THROW((void) SerializeVulkanNativeInstructionPayload(payload), std::runtime_error);
}

TEST(CompiledModuleVulkanTest, SerializesWorkspaceTensorMetadata)
{
	VulkanNativeInstructionPayload payload;
	payload.featureSet.AddFeature(VulkanNativeFeature::StaticShape);
	payload.featureSet.AddFeature(VulkanNativeFeature::SingleSubgraph);
	payload.spirv = { 0x07230203u };
	payload.workspaceTensors.push_back({
	    .byteSize = 256,
	    .alignment = 16,
	});
	VulkanNativeKernelSpec kernel;
	kernel.entryPoint = "main";
	kernel.groups = { .x = 1, .y = 1, .z = 1 };
	kernel.requirements.descriptorAbiVersion = 1;
	kernel.requirements.localSize = { .x = 1, .y = 1, .z = 1 };
	kernel.arguments.push_back({
	    .kind = VulkanNativeArgumentKind::WorkspaceTensor,
	    .index = 0,
	    .binding = 0,
	    .byteOffset = 0,
	    .byteSize = 256,
	});
	payload.kernels.push_back(std::move(kernel));

	const auto decoded = DeserializeVulkanNativeInstructionPayload(SerializeVulkanNativeInstructionPayload(payload));
	ASSERT_EQ(decoded.workspaceTensors.size(), 1u);
	EXPECT_EQ(decoded.workspaceTensors[0].byteSize, 256u);
	EXPECT_EQ(decoded.workspaceTensors[0].alignment, 16u);
	ASSERT_EQ(decoded.kernels.size(), 1u);
	ASSERT_EQ(decoded.kernels[0].arguments.size(), 1u);
	EXPECT_EQ(decoded.kernels[0].arguments[0].kind, VulkanNativeArgumentKind::WorkspaceTensor);
	EXPECT_EQ(decoded.kernels[0].arguments[0].index, 0u);

	payload.workspaceTensors[0].alignment = 24;
	EXPECT_THROW((void) SerializeVulkanNativeInstructionPayload(payload), std::runtime_error);
	payload.workspaceTensors[0].alignment = 16;
	payload.kernels[0].arguments[0].index = 1;
	EXPECT_THROW((void) SerializeVulkanNativeInstructionPayload(payload), std::runtime_error);
}

TEST(CompiledModuleVulkanTest, ReportsNativeSupportForSimpleAdd)
{
	const auto graph = BuildSimpleBinaryGraph(BinaryOp::Add);
	const auto report = Compiler<Vulkan>::QueryNativeSupport(Detail::BuildExecutablePlanFromGraph(graph));

	EXPECT_TRUE(report.supported);
	EXPECT_NE(report.capability.find("same-shape f32 binary"), std::string::npos);
	EXPECT_NE(report.capability.find("Add"), std::string::npos);
	EXPECT_TRUE(report.reason.empty());
}

TEST(CompiledModuleVulkanTest, ReportsNativeSupportForFloat16Add)
{
	const auto graph = BuildSimpleBinaryGraph(BinaryOp::Add, kElementCount, DataType::Float16);
	const auto report = Compiler<Vulkan>::QueryNativeSupport(Detail::BuildExecutablePlanFromGraph(graph));

	EXPECT_TRUE(report.supported);
	EXPECT_NE(report.capability.find("same-shape f16 binary"), std::string::npos);
	EXPECT_NE(report.capability.find("Add"), std::string::npos);
	EXPECT_TRUE(report.reason.empty());
}

TEST(CompiledModuleVulkanTest, ReportsNativeSupportForFloat16Unary)
{
	const auto graph = BuildSimpleUnaryGraph(UnaryOp::Abs, DataType::Float16);
	const auto report = Compiler<Vulkan>::QueryNativeSupport(Detail::BuildExecutablePlanFromGraph(graph));

	EXPECT_TRUE(report.supported);
	EXPECT_NE(report.capability.find("same-shape f16 unary"), std::string::npos);
	EXPECT_NE(report.capability.find("Abs"), std::string::npos);
	EXPECT_TRUE(report.reason.empty());
}

TEST(CompiledModuleVulkanTest, ReportsNativeSupportForSameOpBinaryChain)
{
	const auto graph = BuildBinaryChainGraph(BinaryOp::Add, BinaryOp::Add);
	const auto report = Compiler<Vulkan>::QueryNativeSupport(Detail::BuildExecutablePlanFromGraph(graph));

	EXPECT_TRUE(report.supported);
	EXPECT_NE(report.capability.find("binary chain"), std::string::npos);
	EXPECT_NE(report.capability.find("2 kernels"), std::string::npos);
	EXPECT_TRUE(report.reason.empty());
}

TEST(CompiledModuleVulkanTest, ReportsNativeSupportForMixedBinaryChain)
{
	const auto graph = BuildBinaryChainGraph(BinaryOp::Add, BinaryOp::Multiply);
	const auto report = Compiler<Vulkan>::QueryNativeSupport(Detail::BuildExecutablePlanFromGraph(graph));

	EXPECT_TRUE(report.supported);
	EXPECT_NE(report.capability.find("binary chain"), std::string::npos);
	EXPECT_NE(report.capability.find("2 kernels"), std::string::npos);
	EXPECT_TRUE(report.reason.empty());
}

TEST(CompiledModuleVulkanTest, ReportsNativeSupportForDiamondBinaryDAG)
{
	const auto graph = BuildBinaryDiamondGraph();
	const auto report = Compiler<Vulkan>::QueryNativeSupport(Detail::BuildExecutablePlanFromGraph(graph));

	EXPECT_TRUE(report.supported);
	EXPECT_NE(report.capability.find("binary DAG"), std::string::npos);
	EXPECT_NE(report.capability.find("3 kernels"), std::string::npos);
	EXPECT_TRUE(report.reason.empty());
}

TEST(CompiledModuleVulkanTest, ReportsNativeSupportForBranchedBinaryDAGWithTail)
{
	const auto graph = BuildBranchedBinaryDAGWithTailGraph();
	const auto report = Compiler<Vulkan>::QueryNativeSupport(Detail::BuildExecutablePlanFromGraph(graph));

	EXPECT_TRUE(report.supported);
	EXPECT_NE(report.capability.find("binary DAG"), std::string::npos);
	EXPECT_NE(report.capability.find("5 kernels"), std::string::npos);
	EXPECT_TRUE(report.reason.empty());
}

TEST(CompiledModuleVulkanTest, ReportsNativeSupportForMixedElementwiseDAG)
{
	const auto graph = BuildMixedElementwiseDAGGraph();
	const auto report = Compiler<Vulkan>::QueryNativeSupport(Detail::BuildExecutablePlanFromGraph(graph));

	EXPECT_TRUE(report.supported);
	EXPECT_NE(report.capability.find("elementwise DAG"), std::string::npos);
	EXPECT_NE(report.capability.find("3 kernels"), std::string::npos);
	EXPECT_TRUE(report.reason.empty());
}

TEST(CompiledModuleVulkanTest, ReportsNativeSupportForMatMul)
{
	const auto graph = BuildSimpleMatMulGraph();
	const auto report = Compiler<Vulkan>::QueryNativeSupport(Detail::BuildExecutablePlanFromGraph(graph));

	EXPECT_TRUE(report.supported);
	EXPECT_NE(report.capability.find("f32 matmul"), std::string::npos);
	EXPECT_TRUE(report.reason.empty());
}

TEST(CompiledModuleVulkanTest, ReportsNativeSupportForMatMulBiasReLU)
{
	const auto graph = BuildSimpleMatMulBiasGraph(true);
	const auto report = Compiler<Vulkan>::QueryNativeSupport(Detail::BuildExecutablePlanFromGraph(graph));

	EXPECT_TRUE(report.supported);
	EXPECT_NE(report.capability.find("f32 matmul bias relu"), std::string::npos);
	EXPECT_TRUE(report.reason.empty());
}

TEST(CompiledModuleVulkanTest, ReportsNativeSupportForMatMulBiasExternalWeights)
{
	const auto graph = BuildSimpleMatMulBiasVariableGraph(true);
	const auto report = Compiler<Vulkan>::QueryNativeSupport(Detail::BuildExecutablePlanFromGraph(graph));

	EXPECT_TRUE(report.supported);
	EXPECT_NE(report.capability.find("f32 matmul bias relu"), std::string::npos);
	EXPECT_TRUE(report.reason.empty());
}

TEST(CompiledModuleVulkanTest, ReportsNativeSupportForReduce)
{
	const auto graph = BuildReduceGraph(ReduceOp::Mean, 0, { 3 });
	const auto report = Compiler<Vulkan>::QueryNativeSupport(Detail::BuildExecutablePlanFromGraph(graph));

	EXPECT_TRUE(report.supported);
	EXPECT_NE(report.capability.find("f32 reduce Mean"), std::string::npos);
	EXPECT_NE(report.capability.find("axis=0"), std::string::npos);
	EXPECT_TRUE(report.reason.empty());
}

TEST(CompiledModuleVulkanTest, ReportsNativeSupportForSoftmax)
{
	const auto graph = BuildSoftmaxGraph(1);
	const auto report = Compiler<Vulkan>::QueryNativeSupport(Detail::BuildExecutablePlanFromGraph(graph));

	EXPECT_TRUE(report.supported);
	EXPECT_NE(report.capability.find("f32 softmax"), std::string::npos);
	EXPECT_NE(report.capability.find("axis=1"), std::string::npos);
	EXPECT_TRUE(report.reason.empty());
}

TEST(CompiledModuleVulkanTest, ReportsNativeSupportForNormalization)
{
	const auto graph = BuildNormalizationGraph(NormalizationMode::RMSNorm, 1);
	const auto report = Compiler<Vulkan>::QueryNativeSupport(Detail::BuildExecutablePlanFromGraph(graph));

	EXPECT_TRUE(report.supported);
	EXPECT_NE(report.capability.find("f32 normalization RMSNorm"), std::string::npos);
	EXPECT_NE(report.capability.find("axis=1"), std::string::npos);
	EXPECT_TRUE(report.reason.empty());
}

TEST(CompiledModuleVulkanTest, ReportsNativeSupportForGroupNorm)
{
	const auto graph = BuildGroupNormGraph({ 8 }, 4);
	const auto report = Compiler<Vulkan>::QueryNativeSupport(Detail::BuildExecutablePlanFromGraph(graph));

	EXPECT_TRUE(report.supported);
	EXPECT_NE(report.capability.find("f32 normalization GroupNorm"), std::string::npos);
	EXPECT_NE(report.capability.find("groupCount=4"), std::string::npos);
	EXPECT_TRUE(report.reason.empty());
}

TEST(CompiledModuleVulkanTest, ReportsNativeSupportForPool2D)
{
	const auto graph = BuildPool2DGraph(PoolMode::Max);
	const auto report = Compiler<Vulkan>::QueryNativeSupport(Detail::BuildExecutablePlanFromGraph(graph));

	EXPECT_TRUE(report.supported);
	EXPECT_NE(report.capability.find("f32 Pool2D Max"), std::string::npos);
	EXPECT_TRUE(report.reason.empty());
}

TEST(CompiledModuleVulkanTest, ReportsNativeSupportForNearestUpsample)
{
	const auto graph = BuildNearestUpsampleGraph();
	const auto report = Compiler<Vulkan>::QueryNativeSupport(Detail::BuildExecutablePlanFromGraph(graph));

	EXPECT_TRUE(report.supported);
	EXPECT_NE(report.capability.find("f32 nearest Upsample"), std::string::npos);
	EXPECT_TRUE(report.reason.empty());
}

TEST(CompiledModuleVulkanTest, ReportsNativeSupportForSlice)
{
	const auto graph = BuildSliceGraph();
	const auto report = Compiler<Vulkan>::QueryNativeSupport(Detail::BuildExecutablePlanFromGraph(graph));

	EXPECT_TRUE(report.supported);
	EXPECT_NE(report.capability.find("f32 Slice"), std::string::npos);
	EXPECT_TRUE(report.reason.empty());
}

TEST(CompiledModuleVulkanTest, ReportsNativeSupportForConcat)
{
	const auto graph = BuildConcatGraph();
	const auto report = Compiler<Vulkan>::QueryNativeSupport(Detail::BuildExecutablePlanFromGraph(graph));

	EXPECT_TRUE(report.supported);
	EXPECT_NE(report.capability.find("f32 Concat"), std::string::npos);
	EXPECT_TRUE(report.reason.empty());
}

TEST(CompiledModuleVulkanTest, ReportsNativeSupportForConv2D)
{
	const auto graph = BuildSimpleConv2DVariableGraph();
	const auto report = Compiler<Vulkan>::QueryNativeSupport(Detail::BuildExecutablePlanFromGraph(graph));

	EXPECT_TRUE(report.supported);
	EXPECT_NE(report.capability.find("f32 Conv2D"), std::string::npos);
	EXPECT_TRUE(report.reason.empty());
}

TEST(CompiledModuleVulkanTest, ReportsNativeSupportForConvTranspose2D)
{
	const auto graph = BuildSimpleConvTranspose2DVariableGraph();
	const auto report = Compiler<Vulkan>::QueryNativeSupport(Detail::BuildExecutablePlanFromGraph(graph));

	EXPECT_TRUE(report.supported);
	EXPECT_NE(report.capability.find("f32 ConvTranspose2D"), std::string::npos);
	EXPECT_TRUE(report.reason.empty());
}

TEST(CompiledModuleVulkanTest, UsesTunedWorkgroupDispatchForElementwisePayload)
{
	const auto graph = BuildSimpleBinaryGraph(BinaryOp::Add, kVulkanNativeElementwiseWorkgroupSize + 1);
	const auto artifact = Compiler<Vulkan>::CompileArtifact(Detail::BuildExecutablePlanFromGraph(graph));
	EXPECT_EQ(artifact.Backend(), CompiledModuleBackend::VulkanNative);

	const auto payload = DeserializeVulkanNativeInstructionPayload(artifact.Instructions());
	ASSERT_EQ(payload.kernels.size(), 1u);
	EXPECT_EQ(payload.kernels[0].groups.x, 2u);
	const auto generated =
	    VulkanNativeSameShapeBinaryF32SPIRV(BinaryOp::Add, kVulkanNativeElementwiseWorkgroupSize + 1);
	EXPECT_EQ(payload.spirv, generated.words);
}

TEST(CompiledModuleVulkanTest, WritesVulkanNativePayloadForBinaryChain)
{
	const auto graph = BuildBinaryChainGraph(BinaryOp::Add, BinaryOp::Multiply);
	const auto artifact = Compiler<Vulkan>::CompileArtifact(Detail::BuildExecutablePlanFromGraph(graph));
	EXPECT_EQ(artifact.Backend(), CompiledModuleBackend::VulkanNative);

	const auto payload = DeserializeVulkanNativeInstructionPayload(artifact.Instructions());
	const std::array ops{ BinaryOp::Add, BinaryOp::Multiply };
	const auto generated = VulkanNativeSameShapeBinaryF32ChainSPIRV(ops, kElementCount);
	EXPECT_EQ(payload.spirv, generated.words);
	ASSERT_EQ(payload.workspaceTensors.size(), 1u);
	EXPECT_EQ(payload.workspaceTensors[0].byteSize, kElementCount * sizeof(float));
	ASSERT_EQ(payload.kernels.size(), 2u);
	EXPECT_EQ(payload.kernels[0].entryPoint, VulkanNativeSameShapeBinaryF32KernelName(BinaryOp::Add));
	EXPECT_EQ(payload.kernels[1].entryPoint, VulkanNativeSameShapeBinaryF32KernelName(BinaryOp::Multiply));
	for (const auto& kernel : payload.kernels)
	{
		EXPECT_EQ(kernel.groups.x, 1u);
		ASSERT_EQ(kernel.arguments.size(), 3u);
		EXPECT_EQ(kernel.arguments[2].index, 0u);
	}
	EXPECT_EQ(payload.kernels[0].arguments[0].kind, VulkanNativeArgumentKind::InputTensor);
	EXPECT_EQ(payload.kernels[0].arguments[0].index, 0u);
	EXPECT_EQ(payload.kernels[0].arguments[1].kind, VulkanNativeArgumentKind::InputTensor);
	EXPECT_EQ(payload.kernels[0].arguments[1].index, 1u);
	EXPECT_EQ(payload.kernels[0].arguments[2].kind, VulkanNativeArgumentKind::WorkspaceTensor);
	EXPECT_EQ(payload.kernels[1].arguments[0].kind, VulkanNativeArgumentKind::WorkspaceTensor);
	EXPECT_EQ(payload.kernels[1].arguments[0].index, 0u);
	EXPECT_EQ(payload.kernels[1].arguments[1].kind, VulkanNativeArgumentKind::InputTensor);
	EXPECT_EQ(payload.kernels[1].arguments[1].index, 2u);
	EXPECT_EQ(payload.kernels[1].arguments[2].kind, VulkanNativeArgumentKind::OutputTensor);
	EXPECT_EQ(payload.kernels[1].arguments[2].index, 0u);
}

TEST(CompiledModuleVulkanTest, WritesVulkanNativePayloadForMixedElementwiseDAG)
{
	const auto graph = BuildMixedElementwiseDAGGraph();
	const auto artifact = Compiler<Vulkan>::CompileArtifact(Detail::BuildExecutablePlanFromGraph(graph));
	EXPECT_EQ(artifact.Backend(), CompiledModuleBackend::VulkanNative);

	const auto payload = DeserializeVulkanNativeInstructionPayload(artifact.Instructions());
	const std::array ops{
		VulkanNativeElementwiseF32KernelOp{ .kind = VulkanNativeElementwiseF32KernelKind::Binary,
		                                    .binaryOp = BinaryOp::Add },
		VulkanNativeElementwiseF32KernelOp{ .kind = VulkanNativeElementwiseF32KernelKind::Unary,
		                                    .unaryOp = UnaryOp::Abs },
		VulkanNativeElementwiseF32KernelOp{ .kind = VulkanNativeElementwiseF32KernelKind::Binary,
		                                    .binaryOp = BinaryOp::Multiply },
	};
	const auto generated = VulkanNativeSameShapeElementwiseF32DAGSPIRV(ops, kElementCount);
	EXPECT_EQ(payload.spirv, generated.words);
	ASSERT_EQ(payload.workspaceTensors.size(), 2u);
	ASSERT_EQ(payload.kernels.size(), 3u);
	EXPECT_EQ(payload.kernels[0].entryPoint, VulkanNativeSameShapeBinaryF32KernelName(BinaryOp::Add));
	EXPECT_EQ(payload.kernels[1].entryPoint, VulkanNativeSameShapeUnaryF32KernelName(UnaryOp::Abs));
	EXPECT_EQ(payload.kernels[2].entryPoint, VulkanNativeSameShapeBinaryF32KernelName(BinaryOp::Multiply));

	EXPECT_EQ(payload.kernels[0].arguments[0].kind, VulkanNativeArgumentKind::InputTensor);
	EXPECT_EQ(payload.kernels[0].arguments[0].index, 0u);
	EXPECT_EQ(payload.kernels[0].arguments[1].kind, VulkanNativeArgumentKind::InputTensor);
	EXPECT_EQ(payload.kernels[0].arguments[1].index, 1u);
	EXPECT_EQ(payload.kernels[0].arguments[2].kind, VulkanNativeArgumentKind::WorkspaceTensor);
	EXPECT_EQ(payload.kernels[0].arguments[2].index, 0u);

	ASSERT_EQ(payload.kernels[1].arguments.size(), 3u);
	EXPECT_EQ(payload.kernels[1].arguments[0].kind, VulkanNativeArgumentKind::WorkspaceTensor);
	EXPECT_EQ(payload.kernels[1].arguments[0].index, 0u);
	EXPECT_EQ(payload.kernels[1].arguments[1].kind, VulkanNativeArgumentKind::WorkspaceTensor);
	EXPECT_EQ(payload.kernels[1].arguments[1].index, 0u);
	EXPECT_EQ(payload.kernels[1].arguments[2].kind, VulkanNativeArgumentKind::WorkspaceTensor);
	EXPECT_EQ(payload.kernels[1].arguments[2].index, 1u);

	EXPECT_EQ(payload.kernels[2].arguments[0].kind, VulkanNativeArgumentKind::WorkspaceTensor);
	EXPECT_EQ(payload.kernels[2].arguments[0].index, 1u);
	EXPECT_EQ(payload.kernels[2].arguments[1].kind, VulkanNativeArgumentKind::InputTensor);
	EXPECT_EQ(payload.kernels[2].arguments[1].index, 2u);
	EXPECT_EQ(payload.kernels[2].arguments[2].kind, VulkanNativeArgumentKind::OutputTensor);
	EXPECT_EQ(payload.kernels[2].arguments[2].index, 0u);
}

TEST(CompiledModuleVulkanTest, ReusesOneWorkspaceForLongBinaryChain)
{
	const auto graph = BuildThreeStageBinaryChainGraph();
	const auto artifact = Compiler<Vulkan>::CompileArtifact(Detail::BuildExecutablePlanFromGraph(graph));
	EXPECT_EQ(artifact.Backend(), CompiledModuleBackend::VulkanNative);

	const auto payload = DeserializeVulkanNativeInstructionPayload(artifact.Instructions());
	ASSERT_EQ(payload.workspaceTensors.size(), 1u);
	EXPECT_EQ(payload.workspaceTensors[0].byteSize, kElementCount * sizeof(float));
	ASSERT_EQ(payload.kernels.size(), 3u);
	EXPECT_EQ(payload.kernels[0].arguments[2].kind, VulkanNativeArgumentKind::WorkspaceTensor);
	EXPECT_EQ(payload.kernels[1].arguments[0].kind, VulkanNativeArgumentKind::WorkspaceTensor);
	EXPECT_EQ(payload.kernels[1].arguments[2].kind, VulkanNativeArgumentKind::WorkspaceTensor);
	EXPECT_EQ(payload.kernels[2].arguments[0].kind, VulkanNativeArgumentKind::WorkspaceTensor);
	EXPECT_EQ(payload.kernels[2].arguments[2].kind, VulkanNativeArgumentKind::OutputTensor);
	for (const auto& kernel : payload.kernels)
	{
		ASSERT_EQ(kernel.arguments.size(), 3u);
		for (const auto& argument : kernel.arguments)
		{
			if (argument.kind == VulkanNativeArgumentKind::WorkspaceTensor)
			{
				EXPECT_EQ(argument.index, 0u);
			}
		}
	}
}

TEST(CompiledModuleVulkanTest, LowersFusedElementWiseChainToVulkanWorkspaceChain)
{
	auto graph = BuildThreeStageBinaryChainGraph();
	FusionPass{}.Run(graph);
	const auto& forward = graph.GetSubgraph(graph.Forward());
	ASSERT_EQ(forward.Results().size(), 1u);
	const auto& fusedEntry = forward.GetNodeEntry(forward.Results()[0].node);
	const auto* fused = std::get_if<FusedOpNode>(&fusedEntry.node);
	ASSERT_NE(fused, nullptr);
	EXPECT_EQ(fused->pattern, FusionPattern::ElementWiseChain);
	const auto report = Compiler<Vulkan>::QueryNativeSupport(Detail::BuildExecutablePlanFromGraph(graph));
	EXPECT_TRUE(report.supported) << report.reason;

	const auto artifact = Compiler<Vulkan>::CompileArtifact(Detail::BuildExecutablePlanFromGraph(graph));
	EXPECT_EQ(artifact.Backend(), CompiledModuleBackend::VulkanNative);

	const auto payload = DeserializeVulkanNativeInstructionPayload(artifact.Instructions());
	ASSERT_EQ(payload.workspaceTensors.size(), 1u);
	EXPECT_EQ(payload.workspaceTensors[0].byteSize, kElementCount * sizeof(float));
	ASSERT_EQ(payload.kernels.size(), 3u);
	EXPECT_EQ(payload.kernels[0].arguments[0].index, 0u);
	EXPECT_EQ(payload.kernels[0].arguments[1].index, 1u);
	EXPECT_EQ(payload.kernels[1].arguments[1].index, 2u);
	EXPECT_EQ(payload.kernels[2].arguments[1].index, 3u);
	EXPECT_EQ(payload.kernels[2].arguments[2].kind, VulkanNativeArgumentKind::OutputTensor);
}

TEST(CompiledModuleVulkanTest, LowersFusedMixedElementWiseChainToVulkanWorkspaceDAG)
{
	auto graph = BuildMixedElementwiseDAGGraph();
	FusionPass{}.Run(graph);
	const auto& forward = graph.GetSubgraph(graph.Forward());
	ASSERT_EQ(forward.Results().size(), 1u);
	const auto& fusedEntry = forward.GetNodeEntry(forward.Results()[0].node);
	const auto* fused = std::get_if<FusedOpNode>(&fusedEntry.node);
	ASSERT_NE(fused, nullptr);
	EXPECT_EQ(fused->pattern, FusionPattern::ElementWiseChain);
	const auto report = Compiler<Vulkan>::QueryNativeSupport(Detail::BuildExecutablePlanFromGraph(graph));
	EXPECT_TRUE(report.supported) << report.reason;
	EXPECT_NE(report.capability.find("elementwise DAG"), std::string::npos);

	const auto artifact = Compiler<Vulkan>::CompileArtifact(Detail::BuildExecutablePlanFromGraph(graph));
	EXPECT_EQ(artifact.Backend(), CompiledModuleBackend::VulkanNative);
	const auto payload = DeserializeVulkanNativeInstructionPayload(artifact.Instructions());
	ASSERT_EQ(payload.workspaceTensors.size(), 2u);
	ASSERT_EQ(payload.kernels.size(), 3u);
	EXPECT_EQ(payload.kernels[0].entryPoint, VulkanNativeSameShapeBinaryF32KernelName(BinaryOp::Add));
	EXPECT_EQ(payload.kernels[1].entryPoint, VulkanNativeSameShapeUnaryF32KernelName(UnaryOp::Abs));
	EXPECT_EQ(payload.kernels[2].entryPoint, VulkanNativeSameShapeBinaryF32KernelName(BinaryOp::Multiply));
}

TEST(CompiledModuleVulkanTest, PlansWorkspaceForDiamondBinaryDAG)
{
	const auto graph = BuildBinaryDiamondGraph();
	const auto artifact = Compiler<Vulkan>::CompileArtifact(Detail::BuildExecutablePlanFromGraph(graph));
	EXPECT_EQ(artifact.Backend(), CompiledModuleBackend::VulkanNative);

	const auto payload = DeserializeVulkanNativeInstructionPayload(artifact.Instructions());
	ASSERT_EQ(payload.kernels.size(), 3u);
	ASSERT_EQ(payload.workspaceTensors.size(), 2u);
	EXPECT_EQ(payload.workspaceTensors[0].byteSize, kElementCount * sizeof(float));
	EXPECT_EQ(payload.workspaceTensors[1].byteSize, kElementCount * sizeof(float));
	EXPECT_EQ(payload.kernels[0].arguments[2].kind, VulkanNativeArgumentKind::WorkspaceTensor);
	EXPECT_EQ(payload.kernels[1].arguments[2].kind, VulkanNativeArgumentKind::WorkspaceTensor);
	EXPECT_NE(payload.kernels[0].arguments[2].index, payload.kernels[1].arguments[2].index);
	EXPECT_EQ(payload.kernels[2].arguments[0].kind, VulkanNativeArgumentKind::WorkspaceTensor);
	EXPECT_EQ(payload.kernels[2].arguments[1].kind, VulkanNativeArgumentKind::WorkspaceTensor);
	EXPECT_EQ(payload.kernels[2].arguments[2].kind, VulkanNativeArgumentKind::OutputTensor);
}

TEST(CompiledModuleVulkanTest, PlansWorkspaceForFusedDiamondBinaryDAG)
{
	auto graph = BuildBinaryDiamondGraph();
	FusionPass{}.Run(graph);
	const auto report = Compiler<Vulkan>::QueryNativeSupport(Detail::BuildExecutablePlanFromGraph(graph));
	EXPECT_TRUE(report.supported) << report.reason;
	EXPECT_NE(report.capability.find("binary DAG"), std::string::npos);

	const auto artifact = Compiler<Vulkan>::CompileArtifact(Detail::BuildExecutablePlanFromGraph(graph));
	EXPECT_EQ(artifact.Backend(), CompiledModuleBackend::VulkanNative);

	const auto payload = DeserializeVulkanNativeInstructionPayload(artifact.Instructions());
	ASSERT_EQ(payload.kernels.size(), 3u);
	ASSERT_EQ(payload.workspaceTensors.size(), 2u);
	EXPECT_EQ(payload.kernels[2].arguments[0].kind, VulkanNativeArgumentKind::WorkspaceTensor);
	EXPECT_EQ(payload.kernels[2].arguments[1].kind, VulkanNativeArgumentKind::WorkspaceTensor);
	EXPECT_EQ(payload.kernels[2].arguments[2].kind, VulkanNativeArgumentKind::OutputTensor);
}

TEST(CompiledModuleVulkanTest, PlansWorkspaceForBranchedBinaryDAGWithTail)
{
	const auto graph = BuildBranchedBinaryDAGWithTailGraph();
	const auto artifact = Compiler<Vulkan>::CompileArtifact(Detail::BuildExecutablePlanFromGraph(graph));
	EXPECT_EQ(artifact.Backend(), CompiledModuleBackend::VulkanNative);

	const auto payload = DeserializeVulkanNativeInstructionPayload(artifact.Instructions());
	ASSERT_EQ(payload.kernels.size(), 5u);
	ASSERT_EQ(payload.workspaceTensors.size(), 3u);
	for (const auto& workspace : payload.workspaceTensors)
	{
		EXPECT_EQ(workspace.byteSize, kElementCount * sizeof(float));
	}
	EXPECT_EQ(payload.kernels[0].arguments[2].kind, VulkanNativeArgumentKind::WorkspaceTensor);
	EXPECT_EQ(payload.kernels[1].arguments[2].kind, VulkanNativeArgumentKind::WorkspaceTensor);
	EXPECT_EQ(payload.kernels[2].arguments[2].kind, VulkanNativeArgumentKind::WorkspaceTensor);
	EXPECT_EQ(payload.kernels[3].arguments[2].kind, VulkanNativeArgumentKind::WorkspaceTensor);
	EXPECT_EQ(payload.kernels[4].arguments[0].kind, VulkanNativeArgumentKind::WorkspaceTensor);
	EXPECT_EQ(payload.kernels[4].arguments[1].kind, VulkanNativeArgumentKind::WorkspaceTensor);
	EXPECT_EQ(payload.kernels[4].arguments[2].kind, VulkanNativeArgumentKind::OutputTensor);
	EXPECT_EQ(payload.kernels[3].arguments[0].index, payload.kernels[0].arguments[2].index);
	EXPECT_EQ(payload.kernels[3].arguments[2].index, payload.kernels[1].arguments[2].index);
}

TEST(CompiledModuleVulkanTest, WritesVulkanNativePayloadForSimpleUnary)
{
	const auto graph = BuildSimpleUnaryGraph(UnaryOp::Sqrt);
	const auto artifact = Compiler<Vulkan>::CompileArtifact(Detail::BuildExecutablePlanFromGraph(graph));
	EXPECT_EQ(artifact.Backend(), CompiledModuleBackend::VulkanNative);
	EXPECT_FALSE(artifact.Instructions().empty());

	const auto payload = DeserializeVulkanNativeInstructionPayload(artifact.Instructions());
	const auto generated = VulkanNativeSameShapeUnaryF32SPIRV(UnaryOp::Sqrt, kElementCount);
	EXPECT_EQ(payload.spirv, generated.words);
	ASSERT_EQ(payload.kernels.size(), 1u);
	EXPECT_EQ(payload.kernels[0].groups.x, 1u);
}

TEST(CompiledModuleVulkanTest, WritesVulkanNativePayloadForFloat16Unary)
{
	const auto graph = BuildSimpleUnaryGraph(UnaryOp::Abs, DataType::Float16);
	const auto artifact = Compiler<Vulkan>::CompileArtifact(Detail::BuildExecutablePlanFromGraph(graph));
	EXPECT_EQ(artifact.Backend(), CompiledModuleBackend::VulkanNative);
	EXPECT_FALSE(artifact.Instructions().empty());

	const auto payload = DeserializeVulkanNativeInstructionPayload(artifact.Instructions());
	const auto generated = VulkanNativeSameShapeUnarySPIRV(DataType::Float16, UnaryOp::Abs, kElementCount);
	EXPECT_EQ(payload.spirv, generated.words);
	EXPECT_NE(payload.featureSet.flags &
	              (1ull << static_cast<std::uint32_t>(VulkanNativeFeature::SameShapeElementwiseUnaryLowPrecision)),
	          0ull);
	ASSERT_EQ(payload.kernels.size(), 1u);
	EXPECT_EQ(payload.kernels[0].arguments[0].byteSize, kElementCount * ElementByteSize(DataType::Float16));
	EXPECT_EQ(payload.kernels[0].arguments[1].byteSize, kElementCount * ElementByteSize(DataType::Float16));
	EXPECT_TRUE(payload.kernels[0].requirements.deviceRequirements.HasRequirement(
	    VulkanNativeDeviceRequirement::ShaderFloat16));
	EXPECT_TRUE(payload.kernels[0].requirements.deviceRequirements.HasRequirement(
	    VulkanNativeDeviceRequirement::StorageBuffer16BitAccess));
}

TEST(CompiledModuleVulkanTest, WritesVulkanNativePayloadForSimpleCast)
{
	const auto graph = BuildSimpleCastGraph(DataType::Float32, DataType::Int32);
	const auto artifact = Compiler<Vulkan>::CompileArtifact(Detail::BuildExecutablePlanFromGraph(graph));
	EXPECT_EQ(artifact.Backend(), CompiledModuleBackend::VulkanNative);
	EXPECT_FALSE(artifact.Instructions().empty());

	const auto payload = DeserializeVulkanNativeInstructionPayload(artifact.Instructions());
	const auto generated = VulkanNativeSameShapeCastSPIRV(DataType::Float32, DataType::Int32, kElementCount);
	EXPECT_EQ(payload.spirv, generated.words);
	ASSERT_EQ(payload.kernels.size(), 1u);
	EXPECT_EQ(payload.kernels[0].groups.x, 1u);
}

TEST(CompiledModuleVulkanTest, WritesVulkanNativePayloadForSimpleMatMul)
{
	const auto graph = BuildSimpleMatMulGraph();
	const auto artifact = Compiler<Vulkan>::CompileArtifact(Detail::BuildExecutablePlanFromGraph(graph));
	EXPECT_EQ(artifact.Backend(), CompiledModuleBackend::VulkanNative);
	EXPECT_FALSE(artifact.Instructions().empty());

	const auto payload = DeserializeVulkanNativeInstructionPayload(artifact.Instructions());
	const auto generated = VulkanNativeMatMulF32SPIRV(2, 3, 4);
	EXPECT_EQ(payload.spirv, generated.words);
	EXPECT_NE(payload.featureSet.flags & (1ull << static_cast<std::uint32_t>(VulkanNativeFeature::MatMulF32)), 0ull);
	ASSERT_EQ(payload.kernels.size(), 1u);
	EXPECT_EQ(payload.kernels[0].groups.x, 1u);
	EXPECT_EQ(payload.kernels[0].requirements.localSize.x, kVulkanNativeMatMulWorkgroupSize);
	EXPECT_EQ(payload.kernels[0].requirements.deviceRequirements.flags, 0ull);
	ASSERT_EQ(payload.kernels[0].arguments.size(), 3u);
	EXPECT_EQ(payload.kernels[0].arguments[0].byteSize, 2u * 3u * sizeof(float));
	EXPECT_EQ(payload.kernels[0].arguments[1].byteSize, 3u * 4u * sizeof(float));
	EXPECT_EQ(payload.kernels[0].arguments[2].byteSize, 2u * 4u * sizeof(float));
}

TEST(CompiledModuleVulkanTest, WritesVulkanNativePayloadForSimpleMatMulBiasReLU)
{
	const auto graph = BuildSimpleMatMulBiasGraph(true);
	const auto artifact = Compiler<Vulkan>::CompileArtifact(Detail::BuildExecutablePlanFromGraph(graph));
	EXPECT_EQ(artifact.Backend(), CompiledModuleBackend::VulkanNative);
	EXPECT_FALSE(artifact.Instructions().empty());

	const auto payload = DeserializeVulkanNativeInstructionPayload(artifact.Instructions());
	const auto generated = VulkanNativeMatMulBiasF32SPIRV(2, 3, 4, 1, true);
	EXPECT_EQ(payload.spirv, generated.words);
	EXPECT_NE(payload.featureSet.flags &
	              (1ull << static_cast<std::uint32_t>(VulkanNativeFeature::MatMulBiasAddReLUF32)),
	          0ull);
	ASSERT_EQ(payload.kernels.size(), 1u);
	EXPECT_EQ(payload.kernels[0].groups.x, 1u);
	ASSERT_EQ(payload.kernels[0].arguments.size(), 4u);
	EXPECT_EQ(payload.kernels[0].arguments[0].byteSize, 2u * 3u * sizeof(float));
	EXPECT_EQ(payload.kernels[0].arguments[1].byteSize, 3u * 4u * sizeof(float));
	EXPECT_EQ(payload.kernels[0].arguments[2].byteSize, 1u * 4u * sizeof(float));
	EXPECT_EQ(payload.kernels[0].arguments[3].byteSize, 2u * 4u * sizeof(float));
}

TEST(CompiledModuleVulkanTest, WritesVulkanNativePayloadForMatMulBiasExternalWeights)
{
	const auto graph = BuildSimpleMatMulBiasVariableGraph(true);
	const auto artifact = Compiler<Vulkan>::CompileArtifact(Detail::BuildExecutablePlanFromGraph(graph));
	EXPECT_EQ(artifact.Backend(), CompiledModuleBackend::VulkanNative);
	EXPECT_EQ(artifact.InputSpecs().size(), 1u);
	EXPECT_EQ(artifact.ExternalTensorInfos().size(), 2u);
	EXPECT_EQ(artifact.Constants().size(), 0u);
	EXPECT_GT(artifact.Weights().size(), 0u);

	const auto payload = DeserializeVulkanNativeInstructionPayload(artifact.Instructions());
	ASSERT_EQ(payload.kernels.size(), 1u);
	ASSERT_EQ(payload.kernels[0].arguments.size(), 4u);
	EXPECT_EQ(payload.kernels[0].arguments[0].kind, VulkanNativeArgumentKind::InputTensor);
	EXPECT_EQ(payload.kernels[0].arguments[1].kind, VulkanNativeArgumentKind::ExternalTensor);
	EXPECT_EQ(payload.kernels[0].arguments[2].kind, VulkanNativeArgumentKind::ExternalTensor);
	EXPECT_EQ(payload.kernels[0].arguments[1].index, 0u);
	EXPECT_EQ(payload.kernels[0].arguments[2].index, 1u);
}

TEST(CompiledModuleVulkanTest, WritesVulkanNativePayloadForLowPrecisionCast)
{
	const auto graph = BuildSimpleCastGraph(DataType::Float32, DataType::Float16);
	const auto artifact = Compiler<Vulkan>::CompileArtifact(Detail::BuildExecutablePlanFromGraph(graph));
	EXPECT_EQ(artifact.Backend(), CompiledModuleBackend::VulkanNative);
	EXPECT_FALSE(artifact.Instructions().empty());

	const auto payload = DeserializeVulkanNativeInstructionPayload(artifact.Instructions());
	const auto generated = VulkanNativeSameShapeCastSPIRV(DataType::Float32, DataType::Float16, kElementCount);
	EXPECT_EQ(payload.spirv, generated.words);
	EXPECT_NE(payload.featureSet.flags &
	              (1ull << static_cast<std::uint32_t>(VulkanNativeFeature::SameShapeCastLowPrecision)),
	          0ull);
	ASSERT_EQ(payload.kernels.size(), 1u);
	EXPECT_EQ(payload.kernels[0].requirements.localSize.x, kVulkanNativeElementwiseWorkgroupSize);
	EXPECT_TRUE(payload.kernels[0].requirements.deviceRequirements.HasRequirement(
	    VulkanNativeDeviceRequirement::ShaderFloat16));
	EXPECT_TRUE(payload.kernels[0].requirements.deviceRequirements.HasRequirement(
	    VulkanNativeDeviceRequirement::StorageBuffer16BitAccess));
	EXPECT_FALSE(
	    payload.kernels[0].requirements.deviceRequirements.HasRequirement(VulkanNativeDeviceRequirement::ShaderInt8));
}

TEST(CompiledModuleVulkanTest, WritesVulkanNativePayloadForReduce)
{
	const auto graph = BuildReduceGraph(ReduceOp::Mean, 0, { 3 });
	const auto artifact = Compiler<Vulkan>::CompileArtifact(Detail::BuildExecutablePlanFromGraph(graph));
	EXPECT_EQ(artifact.Backend(), CompiledModuleBackend::VulkanNative);

	const auto payload = DeserializeVulkanNativeInstructionPayload(artifact.Instructions());
	const auto generated = VulkanNativeReduceF32SPIRV(ReduceOp::Mean, std::array<std::size_t, 2>{ 2, 3 }, 0);
	EXPECT_EQ(payload.spirv, generated.words);
	EXPECT_TRUE(payload.featureSet.CheckIsValid());
	EXPECT_NE(payload.featureSet.flags & (1ull << static_cast<std::uint32_t>(VulkanNativeFeature::ReduceF32)), 0ull);
	ASSERT_EQ(payload.kernels.size(), 1u);
	EXPECT_EQ(payload.kernels[0].entryPoint, VulkanNativeReduceF32KernelName(ReduceOp::Mean));
	EXPECT_EQ(payload.kernels[0].groups.x, 1u);
	EXPECT_EQ(payload.kernels[0].requirements.localSize.x, kVulkanNativeElementwiseWorkgroupSize);
	ASSERT_EQ(payload.kernels[0].arguments.size(), 2u);
	EXPECT_EQ(payload.kernels[0].arguments[0].kind, VulkanNativeArgumentKind::InputTensor);
	EXPECT_EQ(payload.kernels[0].arguments[0].index, 0u);
	EXPECT_EQ(payload.kernels[0].arguments[0].binding, 0u);
	EXPECT_EQ(payload.kernels[0].arguments[0].byteSize, 6u * sizeof(float));
	EXPECT_EQ(payload.kernels[0].arguments[1].kind, VulkanNativeArgumentKind::OutputTensor);
	EXPECT_EQ(payload.kernels[0].arguments[1].index, 0u);
	EXPECT_EQ(payload.kernels[0].arguments[1].binding, 1u);
	EXPECT_EQ(payload.kernels[0].arguments[1].byteSize, 3u * sizeof(float));
}

TEST(CompiledModuleVulkanTest, WritesVulkanNativePayloadForSoftmax)
{
	const auto graph = BuildSoftmaxGraph(1);
	const auto artifact = Compiler<Vulkan>::CompileArtifact(Detail::BuildExecutablePlanFromGraph(graph));
	EXPECT_EQ(artifact.Backend(), CompiledModuleBackend::VulkanNative);

	const auto payload = DeserializeVulkanNativeInstructionPayload(artifact.Instructions());
	const auto generated = VulkanNativeSoftmaxF32SPIRV(std::array<std::size_t, 2>{ 2, 3 }, 1);
	EXPECT_EQ(payload.spirv, generated.words);
	EXPECT_TRUE(payload.featureSet.CheckIsValid());
	EXPECT_NE(payload.featureSet.flags & (1ull << static_cast<std::uint32_t>(VulkanNativeFeature::SoftmaxF32)), 0ull);
	ASSERT_EQ(payload.workspaceTensors.size(), 2u);
	EXPECT_EQ(payload.workspaceTensors[0].byteSize, 2u * sizeof(float));
	EXPECT_EQ(payload.workspaceTensors[1].byteSize, 2u * sizeof(float));
	ASSERT_EQ(payload.kernels.size(), 3u);
	EXPECT_EQ(payload.kernels[0].entryPoint, VulkanNativeSoftmaxRowMaxF32KernelName());
	EXPECT_EQ(payload.kernels[1].entryPoint, VulkanNativeSoftmaxRowSumF32KernelName());
	EXPECT_EQ(payload.kernels[2].entryPoint, VulkanNativeSoftmaxWriteF32KernelName());
	EXPECT_EQ(payload.kernels[0].groups.x, 1u);
	EXPECT_EQ(payload.kernels[1].groups.x, 1u);
	EXPECT_EQ(payload.kernels[2].groups.x, 1u);
	EXPECT_EQ(payload.kernels[0].requirements.localSize.x, kVulkanNativeElementwiseWorkgroupSize);
	EXPECT_EQ(payload.kernels[1].requirements.localSize.x, kVulkanNativeElementwiseWorkgroupSize);
	EXPECT_EQ(payload.kernels[2].requirements.localSize.x, kVulkanNativeElementwiseWorkgroupSize);
	ASSERT_EQ(payload.kernels[0].arguments.size(), 2u);
	ASSERT_EQ(payload.kernels[1].arguments.size(), 3u);
	ASSERT_EQ(payload.kernels[2].arguments.size(), 4u);
	EXPECT_EQ(payload.kernels[0].arguments[0].kind, VulkanNativeArgumentKind::InputTensor);
	EXPECT_EQ(payload.kernels[0].arguments[0].binding, 0u);
	EXPECT_EQ(payload.kernels[0].arguments[0].byteSize, 6u * sizeof(float));
	EXPECT_EQ(payload.kernels[0].arguments[1].kind, VulkanNativeArgumentKind::WorkspaceTensor);
	EXPECT_EQ(payload.kernels[0].arguments[1].binding, 1u);
	EXPECT_EQ(payload.kernels[0].arguments[1].byteSize, 2u * sizeof(float));
	EXPECT_EQ(payload.kernels[1].arguments[1].kind, VulkanNativeArgumentKind::WorkspaceTensor);
	EXPECT_EQ(payload.kernels[1].arguments[1].binding, 1u);
	EXPECT_EQ(payload.kernels[1].arguments[2].kind, VulkanNativeArgumentKind::WorkspaceTensor);
	EXPECT_EQ(payload.kernels[1].arguments[2].binding, 2u);
	EXPECT_NE(payload.kernels[1].arguments[1].index, payload.kernels[1].arguments[2].index);
	EXPECT_EQ(payload.kernels[2].arguments[1].kind, VulkanNativeArgumentKind::WorkspaceTensor);
	EXPECT_EQ(payload.kernels[2].arguments[1].binding, 1u);
	EXPECT_EQ(payload.kernels[2].arguments[2].kind, VulkanNativeArgumentKind::WorkspaceTensor);
	EXPECT_EQ(payload.kernels[2].arguments[2].binding, 2u);
	EXPECT_EQ(payload.kernels[2].arguments[3].kind, VulkanNativeArgumentKind::OutputTensor);
	EXPECT_EQ(payload.kernels[2].arguments[3].binding, 3u);
	EXPECT_EQ(payload.kernels[2].arguments[3].byteSize, 6u * sizeof(float));
}

TEST(CompiledModuleVulkanTest, WritesVulkanNativePayloadForNormalization)
{
	const auto graph = BuildNormalizationGraph(NormalizationMode::LayerNorm, 1);
	const auto artifact = Compiler<Vulkan>::CompileArtifact(Detail::BuildExecutablePlanFromGraph(graph));
	EXPECT_EQ(artifact.Backend(), CompiledModuleBackend::VulkanNative);

	const auto payload = DeserializeVulkanNativeInstructionPayload(artifact.Instructions());
	const auto generated =
	    VulkanNativeAxisNormalizationF32SPIRV(NormalizationMode::LayerNorm, std::array<std::size_t, 2>{ 2, 3 }, 1,
	                                          1e-5);
	EXPECT_EQ(payload.spirv, generated.words);
	EXPECT_TRUE(payload.featureSet.CheckIsValid());
	EXPECT_NE(payload.featureSet.flags & (1ull << static_cast<std::uint32_t>(VulkanNativeFeature::NormalizationF32)),
	          0ull);
	ASSERT_EQ(payload.workspaceTensors.size(), 2u);
	EXPECT_EQ(payload.workspaceTensors[0].byteSize, 2u * sizeof(float));
	EXPECT_EQ(payload.workspaceTensors[1].byteSize, 2u * sizeof(float));
	ASSERT_EQ(payload.kernels.size(), 2u);
	EXPECT_EQ(payload.kernels[0].entryPoint,
	          VulkanNativeAxisNormalizationStatsF32KernelName(NormalizationMode::LayerNorm));
	EXPECT_EQ(payload.kernels[1].entryPoint,
	          VulkanNativeAxisNormalizationWriteF32KernelName(NormalizationMode::LayerNorm));
	EXPECT_EQ(payload.kernels[0].groups.x, 1u);
	EXPECT_EQ(payload.kernels[1].groups.x, 1u);
	EXPECT_EQ(payload.kernels[0].requirements.localSize.x, kVulkanNativeElementwiseWorkgroupSize);
	EXPECT_EQ(payload.kernels[1].requirements.localSize.x, kVulkanNativeElementwiseWorkgroupSize);
	ASSERT_EQ(payload.kernels[0].arguments.size(), 3u);
	ASSERT_EQ(payload.kernels[1].arguments.size(), 4u);
	EXPECT_EQ(payload.kernels[0].arguments[0].kind, VulkanNativeArgumentKind::InputTensor);
	EXPECT_EQ(payload.kernels[0].arguments[0].binding, 0u);
	EXPECT_EQ(payload.kernels[0].arguments[0].byteSize, 6u * sizeof(float));
	EXPECT_EQ(payload.kernels[0].arguments[1].kind, VulkanNativeArgumentKind::WorkspaceTensor);
	EXPECT_EQ(payload.kernels[0].arguments[1].binding, 1u);
	EXPECT_EQ(payload.kernels[0].arguments[1].byteSize, 2u * sizeof(float));
	EXPECT_EQ(payload.kernels[0].arguments[2].kind, VulkanNativeArgumentKind::WorkspaceTensor);
	EXPECT_EQ(payload.kernels[0].arguments[2].binding, 2u);
	EXPECT_EQ(payload.kernels[0].arguments[2].byteSize, 2u * sizeof(float));
	EXPECT_NE(payload.kernels[0].arguments[1].index, payload.kernels[0].arguments[2].index);
	EXPECT_EQ(payload.kernels[1].arguments[1].kind, VulkanNativeArgumentKind::WorkspaceTensor);
	EXPECT_EQ(payload.kernels[1].arguments[1].binding, 1u);
	EXPECT_EQ(payload.kernels[1].arguments[2].kind, VulkanNativeArgumentKind::WorkspaceTensor);
	EXPECT_EQ(payload.kernels[1].arguments[2].binding, 2u);
	EXPECT_EQ(payload.kernels[1].arguments[3].kind, VulkanNativeArgumentKind::OutputTensor);
	EXPECT_EQ(payload.kernels[1].arguments[3].binding, 3u);
	EXPECT_EQ(payload.kernels[1].arguments[3].byteSize, 6u * sizeof(float));
}

TEST(CompiledModuleVulkanTest, WritesVulkanNativePayloadForAffineNormalizationExternalWeights)
{
	const auto graph = BuildAffineNormalizationVariableGraph(NormalizationMode::LayerNorm, 1);
	const auto artifact = Compiler<Vulkan>::CompileArtifact(Detail::BuildExecutablePlanFromGraph(graph));
	EXPECT_EQ(artifact.Backend(), CompiledModuleBackend::VulkanNative);
	EXPECT_EQ(artifact.ExternalTensorInfos().size(), 2u);

	const auto payload = DeserializeVulkanNativeInstructionPayload(artifact.Instructions());
	const auto generated = VulkanNativeAxisNormalizationF32SPIRV(
	    NormalizationMode::LayerNorm, std::array<std::size_t, 2>{ 2, 3 }, 1, 1e-5, true, true);
	EXPECT_EQ(payload.spirv, generated.words);
	EXPECT_NE(payload.featureSet.flags & (1ull << static_cast<std::uint32_t>(VulkanNativeFeature::NormalizationF32)),
	          0ull);
	ASSERT_EQ(payload.workspaceTensors.size(), 2u);
	ASSERT_EQ(payload.kernels.size(), 2u);
	EXPECT_EQ(payload.kernels[0].entryPoint,
	          VulkanNativeAxisNormalizationStatsF32KernelName(NormalizationMode::LayerNorm));
	EXPECT_EQ(payload.kernels[1].entryPoint,
	          VulkanNativeAxisNormalizationWriteF32KernelName(NormalizationMode::LayerNorm));
	ASSERT_EQ(payload.kernels[0].arguments.size(), 3u);
	ASSERT_EQ(payload.kernels[1].arguments.size(), 6u);
	EXPECT_EQ(payload.kernels[0].arguments[0].kind, VulkanNativeArgumentKind::InputTensor);
	EXPECT_EQ(payload.kernels[0].arguments[0].binding, 0u);
	EXPECT_EQ(payload.kernels[0].arguments[1].kind, VulkanNativeArgumentKind::WorkspaceTensor);
	EXPECT_EQ(payload.kernels[0].arguments[1].binding, 1u);
	EXPECT_EQ(payload.kernels[0].arguments[2].kind, VulkanNativeArgumentKind::WorkspaceTensor);
	EXPECT_EQ(payload.kernels[0].arguments[2].binding, 2u);
	EXPECT_EQ(payload.kernels[1].arguments[1].kind, VulkanNativeArgumentKind::WorkspaceTensor);
	EXPECT_EQ(payload.kernels[1].arguments[1].binding, 1u);
	EXPECT_EQ(payload.kernels[1].arguments[2].kind, VulkanNativeArgumentKind::WorkspaceTensor);
	EXPECT_EQ(payload.kernels[1].arguments[2].binding, 2u);
	EXPECT_EQ(payload.kernels[1].arguments[3].kind, VulkanNativeArgumentKind::ExternalTensor);
	EXPECT_EQ(payload.kernels[1].arguments[3].binding, 3u);
	EXPECT_EQ(payload.kernels[1].arguments[3].byteSize, 3u * sizeof(float));
	EXPECT_EQ(payload.kernels[1].arguments[4].kind, VulkanNativeArgumentKind::ExternalTensor);
	EXPECT_EQ(payload.kernels[1].arguments[4].binding, 4u);
	EXPECT_EQ(payload.kernels[1].arguments[4].byteSize, 3u * sizeof(float));
	EXPECT_EQ(payload.kernels[1].arguments[5].kind, VulkanNativeArgumentKind::OutputTensor);
	EXPECT_EQ(payload.kernels[1].arguments[5].binding, 5u);
	EXPECT_EQ(payload.kernels[1].arguments[5].byteSize, 6u * sizeof(float));
}

TEST(CompiledModuleVulkanTest, WritesVulkanNativePayloadForGroupNorm)
{
	const auto graph = BuildGroupNormGraph({ 8 }, 4);
	const auto artifact = Compiler<Vulkan>::CompileArtifact(Detail::BuildExecutablePlanFromGraph(graph));
	EXPECT_EQ(artifact.Backend(), CompiledModuleBackend::VulkanNative);

	const auto payload = DeserializeVulkanNativeInstructionPayload(artifact.Instructions());
	const auto generated = VulkanNativeNormalizationF32SPIRV(NormalizationMode::GroupNorm,
	                                                         std::array<std::size_t, 1>{ 8 }, 0, 1e-6, false, false, 4);
	EXPECT_EQ(payload.spirv, generated.words);
	EXPECT_NE(payload.featureSet.flags & (1ull << static_cast<std::uint32_t>(VulkanNativeFeature::NormalizationF32)),
	          0ull);
	ASSERT_EQ(payload.kernels.size(), 1u);
	EXPECT_EQ(payload.kernels[0].entryPoint, "group_norm");
	EXPECT_EQ(payload.kernels[0].groups.x, 1u);
	ASSERT_EQ(payload.kernels[0].arguments.size(), 2u);
	EXPECT_EQ(payload.kernels[0].arguments[0].byteSize, 8u * sizeof(float));
	EXPECT_EQ(payload.kernels[0].arguments[1].byteSize, 8u * sizeof(float));
}

TEST(CompiledModuleVulkanTest, WritesVulkanNativePayloadForAffineGroupNormExternalWeights)
{
	const auto graph = BuildAffineGroupNormVariableGraph({ 8 }, 4);
	const auto artifact = Compiler<Vulkan>::CompileArtifact(Detail::BuildExecutablePlanFromGraph(graph));
	EXPECT_EQ(artifact.Backend(), CompiledModuleBackend::VulkanNative);
	EXPECT_EQ(artifact.ExternalTensorInfos().size(), 2u);

	const auto payload = DeserializeVulkanNativeInstructionPayload(artifact.Instructions());
	const auto generated = VulkanNativeNormalizationF32SPIRV(NormalizationMode::GroupNorm,
	                                                         std::array<std::size_t, 1>{ 8 }, 0, 1e-6, true, true, 4);
	EXPECT_EQ(payload.spirv, generated.words);
	ASSERT_EQ(payload.kernels.size(), 1u);
	EXPECT_EQ(payload.kernels[0].entryPoint, "group_norm");
	ASSERT_EQ(payload.kernels[0].arguments.size(), 4u);
	EXPECT_EQ(payload.kernels[0].arguments[0].kind, VulkanNativeArgumentKind::InputTensor);
	EXPECT_EQ(payload.kernels[0].arguments[0].binding, 0u);
	EXPECT_EQ(payload.kernels[0].arguments[1].kind, VulkanNativeArgumentKind::ExternalTensor);
	EXPECT_EQ(payload.kernels[0].arguments[1].binding, 1u);
	EXPECT_EQ(payload.kernels[0].arguments[1].byteSize, 8u * sizeof(float));
	EXPECT_EQ(payload.kernels[0].arguments[2].kind, VulkanNativeArgumentKind::ExternalTensor);
	EXPECT_EQ(payload.kernels[0].arguments[2].binding, 2u);
	EXPECT_EQ(payload.kernels[0].arguments[2].byteSize, 8u * sizeof(float));
	EXPECT_EQ(payload.kernels[0].arguments[3].kind, VulkanNativeArgumentKind::OutputTensor);
	EXPECT_EQ(payload.kernels[0].arguments[3].binding, 3u);
	EXPECT_EQ(payload.kernels[0].arguments[3].byteSize, 8u * sizeof(float));
}

TEST(CompiledModuleVulkanTest, WritesVulkanNativePayloadForPool2D)
{
	const auto graph = BuildPool2DGraph(PoolMode::Average);
	const auto artifact = Compiler<Vulkan>::CompileArtifact(Detail::BuildExecutablePlanFromGraph(graph));
	EXPECT_EQ(artifact.Backend(), CompiledModuleBackend::VulkanNative);

	const auto payload = DeserializeVulkanNativeInstructionPayload(artifact.Instructions());
	const auto generated = VulkanNativePool2DF32SPIRV(
	    PoolMode::Average, std::array<std::size_t, 4>{ 1, 1, 3, 3 }, std::array<std::size_t, 4>{ 1, 1, 2, 2 },
	    std::array<std::size_t, 2>{ 2, 2 }, std::array<std::size_t, 2>{ 1, 1 });
	EXPECT_EQ(payload.spirv, generated.words);
	EXPECT_NE(payload.featureSet.flags & (1ull << static_cast<std::uint32_t>(VulkanNativeFeature::Pool2DF32)), 0ull);
	ASSERT_EQ(payload.kernels.size(), 1u);
	EXPECT_EQ(payload.kernels[0].entryPoint, "pool2d_average");
	EXPECT_EQ(payload.kernels[0].groups.x, 1u);
	ASSERT_EQ(payload.kernels[0].arguments.size(), 2u);
	EXPECT_EQ(payload.kernels[0].arguments[0].byteSize, 9u * sizeof(float));
	EXPECT_EQ(payload.kernels[0].arguments[1].byteSize, 4u * sizeof(float));
}

TEST(CompiledModuleVulkanTest, WritesVulkanNativePayloadForPaddedPool2D)
{
	const auto graph = BuildPaddedPool2DGraph(PoolMode::Average, true);
	const auto artifact = Compiler<Vulkan>::CompileArtifact(Detail::BuildExecutablePlanFromGraph(graph));
	EXPECT_EQ(artifact.Backend(), CompiledModuleBackend::VulkanNative);

	const auto payload = DeserializeVulkanNativeInstructionPayload(artifact.Instructions());
	const auto generated = VulkanNativePool2DF32SPIRV(
	    PoolMode::Average, std::array<std::size_t, 4>{ 1, 1, 2, 2 }, std::array<std::size_t, 4>{ 1, 1, 3, 3 },
	    std::array<std::size_t, 2>{ 2, 2 }, std::array<std::size_t, 2>{ 1, 1 }, std::array<std::size_t, 2>{ 1, 1 },
	    std::array<std::size_t, 2>{ 1, 1 }, true);
	EXPECT_EQ(payload.spirv, generated.words);
	EXPECT_NE(payload.featureSet.flags & (1ull << static_cast<std::uint32_t>(VulkanNativeFeature::Pool2DF32)), 0ull);
	ASSERT_EQ(payload.kernels.size(), 1u);
	EXPECT_EQ(payload.kernels[0].entryPoint, "pool2d_average");
	EXPECT_EQ(payload.kernels[0].groups.x, 1u);
	ASSERT_EQ(payload.kernels[0].arguments.size(), 2u);
	EXPECT_EQ(payload.kernels[0].arguments[0].byteSize, 4u * sizeof(float));
	EXPECT_EQ(payload.kernels[0].arguments[1].byteSize, 9u * sizeof(float));
}

TEST(CompiledModuleVulkanTest, WritesVulkanNativePayloadForNearestUpsample)
{
	const auto graph = BuildNearestUpsampleGraph();
	const auto artifact = Compiler<Vulkan>::CompileArtifact(Detail::BuildExecutablePlanFromGraph(graph));
	EXPECT_EQ(artifact.Backend(), CompiledModuleBackend::VulkanNative);

	const auto payload = DeserializeVulkanNativeInstructionPayload(artifact.Instructions());
	const auto generated = VulkanNativeUpsampleNearestF32SPIRV(std::array<std::size_t, 4>{ 1, 1, 2, 2 },
	                                                           std::array<std::size_t, 4>{ 1, 1, 4, 4 }, false);
	EXPECT_EQ(payload.spirv, generated.words);
	EXPECT_NE(payload.featureSet.flags & (1ull << static_cast<std::uint32_t>(VulkanNativeFeature::UpsampleNearestF32)),
	          0ull);
	ASSERT_EQ(payload.kernels.size(), 1u);
	EXPECT_EQ(payload.kernels[0].entryPoint, "upsample_nearest");
	EXPECT_EQ(payload.kernels[0].groups.x, 1u);
	ASSERT_EQ(payload.kernels[0].arguments.size(), 2u);
	EXPECT_EQ(payload.kernels[0].arguments[0].kind, VulkanNativeArgumentKind::InputTensor);
	EXPECT_EQ(payload.kernels[0].arguments[0].binding, 0u);
	EXPECT_EQ(payload.kernels[0].arguments[0].byteSize, 4u * sizeof(float));
	EXPECT_EQ(payload.kernels[0].arguments[1].kind, VulkanNativeArgumentKind::OutputTensor);
	EXPECT_EQ(payload.kernels[0].arguments[1].binding, 1u);
	EXPECT_EQ(payload.kernels[0].arguments[1].byteSize, 16u * sizeof(float));
}

TEST(CompiledModuleVulkanTest, WritesVulkanNativePayloadForSlice)
{
	const auto graph = BuildSliceGraph();
	const auto artifact = Compiler<Vulkan>::CompileArtifact(Detail::BuildExecutablePlanFromGraph(graph));
	EXPECT_EQ(artifact.Backend(), CompiledModuleBackend::VulkanNative);

	const auto payload = DeserializeVulkanNativeInstructionPayload(artifact.Instructions());
	const auto generated =
	    VulkanNativeSliceF32SPIRV(std::array<std::size_t, 2>{ 2, 3 }, std::array<std::size_t, 2>{ 2, 2 }, 1, 1, 2);
	EXPECT_EQ(payload.spirv, generated.words);
	EXPECT_NE(payload.featureSet.flags & (1ull << static_cast<std::uint32_t>(VulkanNativeFeature::SliceF32)), 0ull);
	ASSERT_EQ(payload.kernels.size(), 1u);
	EXPECT_EQ(payload.kernels[0].entryPoint, "slice");
	EXPECT_EQ(payload.kernels[0].groups.x, 1u);
	ASSERT_EQ(payload.kernels[0].arguments.size(), 2u);
	EXPECT_EQ(payload.kernels[0].arguments[0].kind, VulkanNativeArgumentKind::InputTensor);
	EXPECT_EQ(payload.kernels[0].arguments[0].binding, 0u);
	EXPECT_EQ(payload.kernels[0].arguments[0].byteSize, 6u * sizeof(float));
	EXPECT_EQ(payload.kernels[0].arguments[1].kind, VulkanNativeArgumentKind::OutputTensor);
	EXPECT_EQ(payload.kernels[0].arguments[1].binding, 1u);
	EXPECT_EQ(payload.kernels[0].arguments[1].byteSize, 4u * sizeof(float));
}

TEST(CompiledModuleVulkanTest, WritesVulkanNativePayloadForConcat)
{
	const auto graph = BuildConcatGraph();
	const auto artifact = Compiler<Vulkan>::CompileArtifact(Detail::BuildExecutablePlanFromGraph(graph));
	EXPECT_EQ(artifact.Backend(), CompiledModuleBackend::VulkanNative);

	const auto payload = DeserializeVulkanNativeInstructionPayload(artifact.Instructions());
	const auto generated = VulkanNativeConcatF32SPIRV(
	    std::array<std::size_t, 2>{ 2, 2 }, std::array<std::size_t, 2>{ 2, 1 }, std::array<std::size_t, 2>{ 2, 3 }, 1);
	EXPECT_EQ(payload.spirv, generated.words);
	EXPECT_NE(payload.featureSet.flags & (1ull << static_cast<std::uint32_t>(VulkanNativeFeature::ConcatF32)), 0ull);
	ASSERT_EQ(payload.kernels.size(), 1u);
	EXPECT_EQ(payload.kernels[0].entryPoint, "concat");
	EXPECT_EQ(payload.kernels[0].groups.x, 1u);
	ASSERT_EQ(payload.kernels[0].arguments.size(), 3u);
	EXPECT_EQ(payload.kernels[0].arguments[0].kind, VulkanNativeArgumentKind::InputTensor);
	EXPECT_EQ(payload.kernels[0].arguments[0].binding, 0u);
	EXPECT_EQ(payload.kernels[0].arguments[0].byteSize, 4u * sizeof(float));
	EXPECT_EQ(payload.kernels[0].arguments[1].kind, VulkanNativeArgumentKind::InputTensor);
	EXPECT_EQ(payload.kernels[0].arguments[1].binding, 1u);
	EXPECT_EQ(payload.kernels[0].arguments[1].byteSize, 2u * sizeof(float));
	EXPECT_EQ(payload.kernels[0].arguments[2].kind, VulkanNativeArgumentKind::OutputTensor);
	EXPECT_EQ(payload.kernels[0].arguments[2].binding, 2u);
	EXPECT_EQ(payload.kernels[0].arguments[2].byteSize, 6u * sizeof(float));
}

TEST(CompiledModuleVulkanTest, WritesVulkanNativePayloadForConv2D)
{
	const auto graph = BuildSimpleConv2DVariableGraph();
	const auto artifact = Compiler<Vulkan>::CompileArtifact(Detail::BuildExecutablePlanFromGraph(graph));
	EXPECT_EQ(artifact.Backend(), CompiledModuleBackend::VulkanNative);

	const auto payload = DeserializeVulkanNativeInstructionPayload(artifact.Instructions());
	const auto generated =
	    VulkanNativeConv2DF32SPIRV(std::array<std::size_t, 4>{ 1, 1, 3, 3 }, std::array<std::size_t, 4>{ 1, 1, 2, 2 },
	                               std::array<std::size_t, 4>{ 1, 1, 2, 2 }, std::array<std::size_t, 2>{ 1, 1 },
	                               std::array<std::size_t, 2>{ 1, 1 }, std::array<std::size_t, 2>{ 0, 0 },
	                               std::array<std::size_t, 2>{ 0, 0 }, 1, true);
	EXPECT_EQ(payload.spirv, generated.words);
	EXPECT_NE(payload.featureSet.flags & (1ull << static_cast<std::uint32_t>(VulkanNativeFeature::Conv2DF32)), 0ull);
	ASSERT_EQ(payload.kernels.size(), 1u);
	EXPECT_EQ(payload.kernels[0].entryPoint, "conv2d");
	EXPECT_EQ(payload.kernels[0].groups.x, 1u);
	ASSERT_EQ(payload.kernels[0].arguments.size(), 4u);
	EXPECT_EQ(payload.kernels[0].arguments[0].kind, VulkanNativeArgumentKind::InputTensor);
	EXPECT_EQ(payload.kernels[0].arguments[0].binding, 0u);
	EXPECT_EQ(payload.kernels[0].arguments[0].byteSize, 9u * sizeof(float));
	EXPECT_EQ(payload.kernels[0].arguments[1].kind, VulkanNativeArgumentKind::ExternalTensor);
	EXPECT_EQ(payload.kernels[0].arguments[1].binding, 1u);
	EXPECT_EQ(payload.kernels[0].arguments[1].byteSize, 4u * sizeof(float));
	EXPECT_EQ(payload.kernels[0].arguments[2].kind, VulkanNativeArgumentKind::ExternalTensor);
	EXPECT_EQ(payload.kernels[0].arguments[2].binding, 2u);
	EXPECT_EQ(payload.kernels[0].arguments[2].byteSize, sizeof(float));
	EXPECT_EQ(payload.kernels[0].arguments[3].kind, VulkanNativeArgumentKind::OutputTensor);
	EXPECT_EQ(payload.kernels[0].arguments[3].binding, 3u);
	EXPECT_EQ(payload.kernels[0].arguments[3].byteSize, 4u * sizeof(float));
	ASSERT_EQ(artifact.ExternalTensorInfos().size(), 2u);
}

TEST(CompiledModuleVulkanTest, WritesVulkanNativePayloadForConvTranspose2D)
{
	const auto graph = BuildSimpleConvTranspose2DVariableGraph();
	const auto artifact = Compiler<Vulkan>::CompileArtifact(Detail::BuildExecutablePlanFromGraph(graph));
	EXPECT_EQ(artifact.Backend(), CompiledModuleBackend::VulkanNative);

	const auto payload = DeserializeVulkanNativeInstructionPayload(artifact.Instructions());
	const auto generated = VulkanNativeConvTranspose2DF32SPIRV(
	    std::array<std::size_t, 4>{ 1, 1, 2, 2 }, std::array<std::size_t, 4>{ 1, 1, 2, 2 },
	    std::array<std::size_t, 4>{ 1, 1, 3, 3 }, std::array<std::size_t, 2>{ 1, 1 },
	    std::array<std::size_t, 2>{ 1, 1 }, std::array<std::size_t, 2>{ 0, 0 }, std::array<std::size_t, 2>{ 0, 0 },
	    std::array<std::size_t, 2>{ 0, 0 }, 1, true);
	EXPECT_EQ(payload.spirv, generated.words);
	EXPECT_NE(payload.featureSet.flags & (1ull << static_cast<std::uint32_t>(VulkanNativeFeature::ConvTranspose2DF32)),
	          0ull);
	ASSERT_EQ(payload.kernels.size(), 1u);
	EXPECT_EQ(payload.kernels[0].entryPoint, "conv_transpose2d");
	ASSERT_EQ(payload.kernels[0].arguments.size(), 4u);
	EXPECT_EQ(payload.kernels[0].arguments[0].kind, VulkanNativeArgumentKind::InputTensor);
	EXPECT_EQ(payload.kernels[0].arguments[0].binding, 0u);
	EXPECT_EQ(payload.kernels[0].arguments[1].kind, VulkanNativeArgumentKind::ExternalTensor);
	EXPECT_EQ(payload.kernels[0].arguments[1].binding, 1u);
	EXPECT_EQ(payload.kernels[0].arguments[2].kind, VulkanNativeArgumentKind::ExternalTensor);
	EXPECT_EQ(payload.kernels[0].arguments[2].binding, 2u);
	EXPECT_EQ(payload.kernels[0].arguments[3].kind, VulkanNativeArgumentKind::OutputTensor);
	EXPECT_EQ(payload.kernels[0].arguments[3].binding, 3u);
	ASSERT_EQ(artifact.ExternalTensorInfos().size(), 2u);
}

TEST(CompiledModuleVulkanTest, RejectsLowPrecisionCastWhenDeviceFeaturesAreNotEnabled)
{
	if (!IsVulkanDeviceAvailable())
	{
		GTEST_SKIP() << "No Vulkan compute device is available";
	}

	Vulkan device;
	const auto capabilities = QueryVulkanDeviceCapabilities(device);
	if (capabilities.shaderFloat16Enabled && capabilities.storageBuffer16BitAccessEnabled)
	{
		GTEST_SKIP() << "Vulkan Float16 storage features are enabled by the runtime";
	}

	const auto graph = BuildSimpleCastGraph(DataType::Float32, DataType::Float16);
	const auto artifact = Compiler<Vulkan>::CompileArtifact(Detail::BuildExecutablePlanFromGraph(graph));
	ASSERT_EQ(artifact.Backend(), CompiledModuleBackend::VulkanNative);

	ExpectFloat16FeatureGateRejectsLoad(artifact, device);
}

TEST(CompiledModuleVulkanTest, RejectsFloat16BinaryWhenDeviceFeaturesAreNotEnabled)
{
	if (!IsVulkanDeviceAvailable())
	{
		GTEST_SKIP() << "No Vulkan compute device is available";
	}

	Vulkan device;
	const auto capabilities = QueryVulkanDeviceCapabilities(device);
	if (capabilities.shaderFloat16Enabled && capabilities.storageBuffer16BitAccessEnabled)
	{
		GTEST_SKIP() << "Vulkan Float16 storage features are enabled by the runtime";
	}

	const auto graph = BuildSimpleBinaryGraph(BinaryOp::Add, kElementCount, DataType::Float16);
	const auto artifact = Compiler<Vulkan>::CompileArtifact(Detail::BuildExecutablePlanFromGraph(graph));
	ASSERT_EQ(artifact.Backend(), CompiledModuleBackend::VulkanNative);

	ExpectFloat16FeatureGateRejectsLoad(artifact, device);
}

TEST(CompiledModuleVulkanTest, RejectsFloat16UnaryWhenDeviceFeaturesAreNotEnabled)
{
	if (!IsVulkanDeviceAvailable())
	{
		GTEST_SKIP() << "No Vulkan compute device is available";
	}

	Vulkan device;
	const auto capabilities = QueryVulkanDeviceCapabilities(device);
	if (capabilities.shaderFloat16Enabled && capabilities.storageBuffer16BitAccessEnabled)
	{
		GTEST_SKIP() << "Vulkan Float16 storage features are enabled by the runtime";
	}

	const auto graph = BuildSimpleUnaryGraph(UnaryOp::Abs, DataType::Float16);
	const auto artifact = Compiler<Vulkan>::CompileArtifact(Detail::BuildExecutablePlanFromGraph(graph));
	ASSERT_EQ(artifact.Backend(), CompiledModuleBackend::VulkanNative);

	ExpectFloat16FeatureGateRejectsLoad(artifact, device);
}

TEST(CompiledModuleVulkanTest, ReportsDescriptorAndDispatchLimits)
{
	if (!IsVulkanDeviceAvailable())
	{
		GTEST_SKIP() << "No Vulkan compute device is available";
	}

	const auto capabilities = QueryVulkanDeviceCapabilities(Vulkan{});
	EXPECT_GT(capabilities.maxComputeWorkGroupCount[0], 0u);
	EXPECT_GT(capabilities.maxComputeWorkGroupCount[1], 0u);
	EXPECT_GT(capabilities.maxComputeWorkGroupCount[2], 0u);
	EXPECT_GT(capabilities.maxPerStageDescriptorStorageBuffers, 0u);
	EXPECT_GT(capabilities.maxDescriptorSetStorageBuffers, 0u);
	EXPECT_GT(capabilities.maxBoundDescriptorSets, 0u);
	EXPECT_GT(capabilities.maxPerStageResources, 0u);
	EXPECT_GT(capabilities.maxComputeSharedMemorySize, 0u);
	EXPECT_GT(capabilities.maxPushConstantsSize, 0u);
	EXPECT_GE(capabilities.timestampPeriodNanoseconds, 0.0f);
	if (capabilities.computeQueueTimestampsAvailable)
	{
		EXPECT_GT(capabilities.computeQueueTimestampValidBits, 0u);
		EXPECT_GT(capabilities.timestampPeriodNanoseconds, 0.0f);
	}
	if (capabilities.subgroupComputeAvailable)
	{
		EXPECT_GT(capabilities.subgroupSize, 0u);
	}
	EXPECT_FALSE(capabilities.shaderStorageBufferArrayNonUniformIndexingEnabled &&
	             !capabilities.shaderStorageBufferArrayNonUniformIndexingAvailable);
	EXPECT_FALSE(capabilities.descriptorBindingStorageBufferUpdateAfterBindEnabled &&
	             !capabilities.descriptorBindingStorageBufferUpdateAfterBindAvailable);
	EXPECT_FALSE(capabilities.descriptorBindingPartiallyBoundEnabled &&
	             !capabilities.descriptorBindingPartiallyBoundAvailable);
	EXPECT_FALSE(capabilities.descriptorBindingVariableDescriptorCountEnabled &&
	             !capabilities.descriptorBindingVariableDescriptorCountAvailable);
	EXPECT_FALSE(capabilities.runtimeDescriptorArrayEnabled && !capabilities.runtimeDescriptorArrayAvailable);
}

TEST(CompiledModuleVulkanTest, RejectsPayloadWhenAdvancedDeviceRequirementIsNotEnabled)
{
	if (!IsVulkanDeviceAvailable())
	{
		GTEST_SKIP() << "No Vulkan compute device is available";
	}

	struct RequirementCase
	{
		VulkanNativeDeviceRequirement requirement;
		bool enabled;
		const char* name;
	};

	Vulkan device;
	const auto capabilities = QueryVulkanDeviceCapabilities(device);
	const std::array cases{
		RequirementCase{ VulkanNativeDeviceRequirement::SubgroupArithmetic,
		                 capabilities.subgroupComputeAvailable && capabilities.subgroupArithmeticAvailable,
		                 "subgroupArithmetic" },
		RequirementCase{ VulkanNativeDeviceRequirement::SubgroupBallot,
		                 capabilities.subgroupComputeAvailable && capabilities.subgroupBallotAvailable,
		                 "subgroupBallot" },
		RequirementCase{ VulkanNativeDeviceRequirement::SubgroupShuffle,
		                 capabilities.subgroupComputeAvailable && capabilities.subgroupShuffleAvailable,
		                 "subgroupShuffle" },
		RequirementCase{ VulkanNativeDeviceRequirement::ShaderStorageBufferArrayNonUniformIndexing,
		                 capabilities.shaderStorageBufferArrayNonUniformIndexingEnabled,
		                 "shaderStorageBufferArrayNonUniformIndexing" },
		RequirementCase{ VulkanNativeDeviceRequirement::DescriptorBindingStorageBufferUpdateAfterBind,
		                 capabilities.descriptorBindingStorageBufferUpdateAfterBindEnabled,
		                 "descriptorBindingStorageBufferUpdateAfterBind" },
		RequirementCase{ VulkanNativeDeviceRequirement::DescriptorBindingPartiallyBound,
		                 capabilities.descriptorBindingPartiallyBoundEnabled, "descriptorBindingPartiallyBound" },
		RequirementCase{ VulkanNativeDeviceRequirement::DescriptorBindingVariableDescriptorCount,
		                 capabilities.descriptorBindingVariableDescriptorCountEnabled,
		                 "descriptorBindingVariableDescriptorCount" },
		RequirementCase{ VulkanNativeDeviceRequirement::RuntimeDescriptorArray,
		                 capabilities.runtimeDescriptorArrayEnabled, "runtimeDescriptorArray" },
	};
	const RequirementCase* selected = nullptr;
	for (const auto& item : cases)
	{
		if (!item.enabled)
		{
			selected = &item;
			break;
		}
	}
	if (selected == nullptr)
	{
		GTEST_SKIP() << "All advanced Vulkan requirement bits are enabled by the selected logical device";
	}

	const auto graph = BuildSimpleBinaryGraph(BinaryOp::Add);
	const auto artifact = Compiler<Vulkan>::CompileArtifact(Detail::BuildExecutablePlanFromGraph(graph));
	auto payload = DeserializeVulkanNativeInstructionPayload(artifact.Instructions());
	ASSERT_FALSE(payload.kernels.empty());
	payload.kernels[0].requirements.deviceRequirements.AddRequirement(selected->requirement);
	const auto badInstructions = SerializeVulkanNativeInstructionPayload(payload);

	try
	{
		(void) CompiledModule<Vulkan>::Load(ImageWithInstructions(artifact, badInstructions), device);
		FAIL() << "Expected Vulkan payload loading to reject a disabled advanced device requirement";
	}
	catch (const std::runtime_error& ex)
	{
		const std::string message = ex.what();
		EXPECT_NE(message.find(selected->name), std::string::npos) << message;
		EXPECT_NE(message.find("enabled=false"), std::string::npos) << message;
	}
}

TEST(CompiledModuleVulkanTest, RejectsPayloadWhenDispatchGroupsExceedDeviceLimit)
{
	if (!IsVulkanDeviceAvailable())
	{
		GTEST_SKIP() << "No Vulkan compute device is available";
	}

	Vulkan device;
	const auto capabilities = QueryVulkanDeviceCapabilities(device);
	if (capabilities.maxComputeWorkGroupCount[0] == std::numeric_limits<std::uint32_t>::max())
	{
		GTEST_SKIP() << "Cannot construct a larger x dispatch group without overflowing uint32_t";
	}

	const auto graph = BuildSimpleBinaryGraph(BinaryOp::Add);
	const auto artifact = Compiler<Vulkan>::CompileArtifact(Detail::BuildExecutablePlanFromGraph(graph));
	auto payload = DeserializeVulkanNativeInstructionPayload(artifact.Instructions());
	ASSERT_FALSE(payload.kernels.empty());
	payload.kernels[0].groups.x = capabilities.maxComputeWorkGroupCount[0] + 1;
	const auto badInstructions = SerializeVulkanNativeInstructionPayload(payload);

	try
	{
		(void) CompiledModule<Vulkan>::Load(ImageWithInstructions(artifact, badInstructions), device);
		FAIL() << "Expected Vulkan payload loading to reject oversized dispatch groups";
	}
	catch (const std::runtime_error& ex)
	{
		const std::string message = ex.what();
		EXPECT_NE(message.find("maxComputeWorkGroupCount"), std::string::npos) << message;
	}
}

TEST(CompiledModuleVulkanTest, RejectsPayloadWhenDescriptorCountExceedsDeviceLimit)
{
	if (!IsVulkanDeviceAvailable())
	{
		GTEST_SKIP() << "No Vulkan compute device is available";
	}

	Vulkan device;
	const auto capabilities = QueryVulkanDeviceCapabilities(device);
	const auto descriptorLimit =
	    capabilities.maxPerStageDescriptorStorageBuffers < capabilities.maxDescriptorSetStorageBuffers
	        ? capabilities.maxPerStageDescriptorStorageBuffers
	        : capabilities.maxDescriptorSetStorageBuffers;
	if (descriptorLimit == 0 || descriptorLimit == std::numeric_limits<std::uint32_t>::max())
	{
		GTEST_SKIP() << "Cannot construct a larger descriptor binding for this device limit";
	}

	const auto graph = BuildSimpleBinaryGraph(BinaryOp::Add);
	const auto artifact = Compiler<Vulkan>::CompileArtifact(Detail::BuildExecutablePlanFromGraph(graph));
	auto payload = DeserializeVulkanNativeInstructionPayload(artifact.Instructions());
	ASSERT_FALSE(payload.kernels.empty());
	ASSERT_FALSE(payload.kernels[0].arguments.empty());
	payload.kernels[0].arguments[0].binding = descriptorLimit;
	const auto badInstructions = SerializeVulkanNativeInstructionPayload(payload);

	try
	{
		(void) CompiledModule<Vulkan>::Load(ImageWithInstructions(artifact, badInstructions), device);
		FAIL() << "Expected Vulkan payload loading to reject excessive descriptor bindings";
	}
	catch (const std::runtime_error& ex)
	{
		const std::string message = ex.what();
		EXPECT_NE(message.find("storage-buffer descriptor"), std::string::npos) << message;
	}
}

TEST(CompiledModuleVulkanTest, RunsLowPrecisionCastArithmeticWhenDeviceFeaturesAreEnabled)
{
	if (!IsVulkanDeviceAvailable())
	{
		GTEST_SKIP() << "No Vulkan compute device is available";
	}

	Vulkan device;
	const auto capabilities = QueryVulkanDeviceCapabilities(device);
	if (!capabilities.shaderFloat16Enabled || !capabilities.storageBuffer16BitAccessEnabled)
	{
		GTEST_SKIP() << "Vulkan Float16 storage features are not enabled by the runtime";
	}

	const auto graph = BuildSimpleCastGraph(DataType::Float32, DataType::Float16);
	auto module = Compiler<Vulkan>::Compile(Detail::BuildExecutablePlanFromGraph(graph), device);
	ASSERT_EQ(module.Backend(), CompiledModuleBackend::VulkanNative);

	std::array inputs{
		Tensor<Vulkan>({ -3.5, -1.0, 0.75, 4.0 }, { 4 }, DataType::Float32, device),
	};
	auto outputs = module.RunTensors(std::span<const Tensor<Vulkan>>(inputs));
	ASSERT_EQ(outputs.size(), 1);
	EXPECT_EQ(outputs[0].DType(), DataType::Float16);

	const auto actual = CopyToHostAsFloat32(outputs[0]);
	const std::array expected{ -3.5f, -1.0f, 0.75f, 4.0f };
	for (std::size_t i = 0; i < expected.size(); ++i)
	{
		EXPECT_FLOAT_EQ(actual[i], expected[i]);
	}
}

TEST(CompiledModuleVulkanTest, RunsFloat16BinaryArithmeticWhenDeviceFeaturesAreEnabled)
{
	if (!IsVulkanDeviceAvailable())
	{
		GTEST_SKIP() << "No Vulkan compute device is available";
	}

	Vulkan device;
	const auto capabilities = QueryVulkanDeviceCapabilities(device);
	if (!capabilities.shaderFloat16Enabled || !capabilities.storageBuffer16BitAccessEnabled)
	{
		GTEST_SKIP() << "Vulkan Float16 storage features are not enabled by the runtime";
	}

	const auto graph = BuildSimpleBinaryGraph(BinaryOp::Add, kElementCount, DataType::Float16);
	auto module = Compiler<Vulkan>::Compile(Detail::BuildExecutablePlanFromGraph(graph), device);
	ASSERT_EQ(module.Backend(), CompiledModuleBackend::VulkanNative);

	std::array inputs{
		Tensor<Vulkan>({ 1.0, 2.0, 3.0, 4.0 }, { 4 }, DataType::Float16, device),
		Tensor<Vulkan>({ 0.5, 1.0, 1.5, 2.0 }, { 4 }, DataType::Float16, device),
	};
	auto outputs = module.RunTensors(std::span<const Tensor<Vulkan>>(inputs));
	ASSERT_EQ(outputs.size(), 1);
	EXPECT_EQ(outputs[0].DType(), DataType::Float16);

	const auto actual = CopyToHostAsFloat32(outputs[0]);
	const std::array expected{ 1.5f, 3.0f, 4.5f, 6.0f };
	for (std::size_t i = 0; i < expected.size(); ++i)
	{
		EXPECT_NEAR(actual[i], expected[i], 1e-3f);
	}
}

TEST(CompiledModuleVulkanTest, RunsFloat16UnaryArithmeticWhenDeviceFeaturesAreEnabled)
{
	if (!IsVulkanDeviceAvailable())
	{
		GTEST_SKIP() << "No Vulkan compute device is available";
	}

	Vulkan device;
	const auto capabilities = QueryVulkanDeviceCapabilities(device);
	if (!capabilities.shaderFloat16Enabled || !capabilities.storageBuffer16BitAccessEnabled)
	{
		GTEST_SKIP() << "Vulkan Float16 storage features are not enabled by the runtime";
	}

	const auto graph = BuildSimpleUnaryGraph(UnaryOp::Abs, DataType::Float16);
	auto module = Compiler<Vulkan>::Compile(Detail::BuildExecutablePlanFromGraph(graph), device);
	ASSERT_EQ(module.Backend(), CompiledModuleBackend::VulkanNative);

	std::array inputs{
		Tensor<Vulkan>({ -1.5, -2.0, 3.0, -4.0 }, { 4 }, DataType::Float16, device),
	};
	auto outputs = module.RunTensors(std::span<const Tensor<Vulkan>>(inputs));
	ASSERT_EQ(outputs.size(), 1);
	EXPECT_EQ(outputs[0].DType(), DataType::Float16);

	const auto actual = CopyToHostAsFloat32(outputs[0]);
	const std::array expected{ 1.5f, 2.0f, 3.0f, 4.0f };
	for (std::size_t i = 0; i < expected.size(); ++i)
	{
		EXPECT_NEAR(actual[i], expected[i], 1e-3f);
	}
}

TEST(CompiledModuleVulkanTest, LoadsSeparatedArtifactForSimpleAdd)
{
	if (!IsVulkanDeviceAvailable())
	{
		GTEST_SKIP() << "No Vulkan compute device is available";
	}

	const auto graph = BuildSimpleBinaryGraph(BinaryOp::Add);
	const auto artifact = Compiler<Vulkan>::CompileArtifact(Detail::BuildExecutablePlanFromGraph(graph));
	auto separated = artifact.SeparateRodata();
	EXPECT_EQ(separated.Backend(), CompiledModuleBackend::VulkanNative);
	EXPECT_GT(separated.Metadata().size(), 0u);
	EXPECT_GT(separated.Instructions().size(), 0u);

	auto module = separated.LoadBorrowedExternalRegions(Vulkan{});
	ASSERT_EQ(module.Backend(), CompiledModuleBackend::VulkanNative);

	Vulkan device;
	std::array inputs{
		Tensor<Vulkan>({ 1.0, 2.0, 3.0, 4.0 }, { 4 }, DataType::Float32, device),
		Tensor<Vulkan>({ 10.0, 20.0, 30.0, 40.0 }, { 4 }, DataType::Float32, device),
	};
	auto outputs = module.RunTensors(std::span<const Tensor<Vulkan>>(inputs));
	ASSERT_EQ(outputs.size(), 1);

	const auto actual = CopyToHost(outputs[0]);
	const std::array expected{ 11.0f, 22.0f, 33.0f, 44.0f };
	for (std::size_t i = 0; i < expected.size(); ++i)
	{
		EXPECT_FLOAT_EQ(actual[i], expected[i]);
	}
}

TEST(CompiledModuleVulkanTest, RecordsVulkanNativeProfileEvents)
{
	if (!IsVulkanDeviceAvailable())
	{
		GTEST_SKIP() << "No Vulkan compute device is available";
	}

	const auto graph = BuildSimpleBinaryGraph(BinaryOp::Add);
	auto module = Compiler<Vulkan>::Compile(Detail::BuildExecutablePlanFromGraph(graph), Vulkan{});
	ASSERT_EQ(module.Backend(), CompiledModuleBackend::VulkanNative);

	Vulkan device;
	std::array inputs{
		Tensor<Vulkan>({ 1.0, 2.0, 3.0, 4.0 }, { 4 }, DataType::Float32, device),
		Tensor<Vulkan>({ 10.0, 20.0, 30.0, 40.0 }, { 4 }, DataType::Float32, device),
	};
	std::vector<CompiledModuleVulkanProfileEvent> events;
	auto outputs =
	    module.RunTensors(std::span<const Tensor<Vulkan>>(inputs), { .synchronize = true, .profileEvents = &events });
	ASSERT_EQ(outputs.size(), 1);
	ASSERT_EQ(events.size(), 1u);
	EXPECT_EQ(events[0].kernelIndex, 0u);
	EXPECT_EQ(events[0].entryPoint, "main");
	EXPECT_EQ(events[0].groups.x, 1u);
	EXPECT_EQ(events[0].localSize.x, kVulkanNativeElementwiseWorkgroupSize);
	EXPECT_EQ(events[0].descriptorCount, 3u);
	EXPECT_GE(events[0].moduleCreationWallMs, 0.0);
	EXPECT_GE(events[0].dispatchWallMs, 0.0);
	const auto capabilities = QueryVulkanDeviceCapabilities(device);
	EXPECT_EQ(events[0].gpuTimestampAvailable, capabilities.computeQueueTimestampsAvailable);
	if (events[0].gpuTimestampAvailable)
	{
		EXPECT_GE(events[0].gpuElapsedMs, 0.0);
	}
}

TEST(CompiledModuleVulkanTest, RunsSimpleCastArithmetic)
{
	if (!IsVulkanDeviceAvailable())
	{
		GTEST_SKIP() << "No Vulkan compute device is available";
	}

	for (const auto& item : kRuntimeCastCases)
	{
		const auto graph = BuildSimpleCastGraph(item.srcType, item.dstType);
		auto module = Compiler<Vulkan>::Compile(Detail::BuildExecutablePlanFromGraph(graph), Vulkan{});
		ASSERT_EQ(module.Backend(), CompiledModuleBackend::VulkanNative);

		Vulkan device;
		std::array inputs{
			Tensor<Vulkan>(item.input, { 4 }, item.srcType, device),
		};
		auto outputs = module.RunTensors(std::span<const Tensor<Vulkan>>(inputs));
		ASSERT_EQ(outputs.size(), 1);
		EXPECT_EQ(outputs[0].DType(), item.dstType);

		const auto actual = CopyToHostAsFloat32(outputs[0]);
		for (std::size_t i = 0; i < item.expected.size(); ++i)
		{
			EXPECT_FLOAT_EQ(actual[i], item.expected[i]);
		}
	}
}

TEST(CompiledModuleVulkanTest, RunsSimpleUnaryArithmetic)
{
	if (!IsVulkanDeviceAvailable())
	{
		GTEST_SKIP() << "No Vulkan compute device is available";
	}

	for (const auto& item : kUnaryCases)
	{
		const auto graph = BuildSimpleUnaryGraph(item.op);
		auto module = Compiler<Vulkan>::Compile(Detail::BuildExecutablePlanFromGraph(graph), Vulkan{});
		ASSERT_EQ(module.Backend(), CompiledModuleBackend::VulkanNative);

		Vulkan device;
		std::array inputs{
			Tensor<Vulkan>(item.input, { 4 }, DataType::Float32, device),
		};
		auto outputs = module.RunTensors(std::span<const Tensor<Vulkan>>(inputs));
		ASSERT_EQ(outputs.size(), 1);

		const auto actual = CopyToHost(outputs[0]);
		for (std::size_t i = 0; i < item.input.size(); ++i)
		{
			EXPECT_NEAR(actual[i], ExpectedUnaryValue(item.op, item.input[i]), item.tolerance);
		}
	}
}

TEST(CompiledModuleVulkanTest, RunsSimpleBinaryArithmetic)
{
	if (!IsVulkanDeviceAvailable())
	{
		GTEST_SKIP() << "No Vulkan compute device is available";
	}

	for (const auto& item : kBinaryCases)
	{
		const auto graph = BuildSimpleBinaryGraph(item.op);
		auto module = Compiler<Vulkan>::Compile(Detail::BuildExecutablePlanFromGraph(graph), Vulkan{});
		ASSERT_EQ(module.Backend(), CompiledModuleBackend::VulkanNative);

		Vulkan device;
		std::array inputs{
			Tensor<Vulkan>({ 1.0, 2.0, 3.0, 4.0 }, { 4 }, DataType::Float32, device),
			Tensor<Vulkan>({ 10.0, 20.0, 30.0, 40.0 }, { 4 }, DataType::Float32, device),
		};
		auto outputs = module.RunTensors(std::span<const Tensor<Vulkan>>(inputs));
		ASSERT_EQ(outputs.size(), 1);

		const auto actual = CopyToHost(outputs[0]);
		for (std::size_t i = 0; i < item.expected.size(); ++i)
		{
			EXPECT_FLOAT_EQ(actual[i], item.expected[i]);
		}
	}
}

TEST(CompiledModuleVulkanTest, RunsSimpleBinaryArithmeticWithDeviceLocalTensors)
{
	if (!IsVulkanDeviceAvailable())
	{
		GTEST_SKIP() << "No Vulkan compute device is available";
	}

	Vulkan device;
	device.bufferResidency = VulkanBufferResidency::DeviceLocal;

	const auto graph = BuildSimpleBinaryGraph(BinaryOp::Add);
	auto module = Compiler<Vulkan>::Compile(Detail::BuildExecutablePlanFromGraph(graph), device);
	ASSERT_EQ(module.Backend(), CompiledModuleBackend::VulkanNative);

	std::array inputs{
		Tensor<Vulkan>({ 1.0, 2.0, 3.0, 4.0 }, { 4 }, DataType::Float32, device),
		Tensor<Vulkan>({ 10.0, 20.0, 30.0, 40.0 }, { 4 }, DataType::Float32, device),
	};
	auto outputs = module.RunTensors(std::span<const Tensor<Vulkan>>(inputs));
	ASSERT_EQ(outputs.size(), 1);
	EXPECT_EQ(outputs[0].CurDevice().bufferResidency, VulkanBufferResidency::DeviceLocal);

	const auto actual = CopyToHost(outputs[0]);
	const std::array expected{ 11.0f, 22.0f, 33.0f, 44.0f };
	for (std::size_t i = 0; i < expected.size(); ++i)
	{
		EXPECT_FLOAT_EQ(actual[i], expected[i]);
	}
}

TEST(CompiledModuleVulkanTest, AllocatesDeviceLocalOutputsFromModulePolicy)
{
	if (!IsVulkanDeviceAvailable())
	{
		GTEST_SKIP() << "No Vulkan compute device is available";
	}

	Vulkan device;
	device.bufferResidency = VulkanBufferResidency::DeviceLocal;

	const auto graph = BuildSimpleBinaryGraph(BinaryOp::Add);
	auto module = Compiler<Vulkan>::Compile(Detail::BuildExecutablePlanFromGraph(graph), device);
	ASSERT_EQ(module.Backend(), CompiledModuleBackend::VulkanNative);

	std::array inputs{
		Tensor<Vulkan>({ 1.0, 2.0, 3.0, 4.0 }, { 4 }, DataType::Float32, device),
		Tensor<Vulkan>({ 10.0, 20.0, 30.0, 40.0 }, { 4 }, DataType::Float32, device),
	};
	auto outputs = module.AllocateOutputTensors();
	ASSERT_EQ(outputs.size(), 1);
	EXPECT_EQ(outputs[0].CurDevice().bufferResidency, VulkanBufferResidency::DeviceLocal);

	module.RunTensorsInto(std::span<const Tensor<Vulkan>>(inputs), std::span<Tensor<Vulkan>>(outputs));

	const auto actual = CopyToHost(outputs[0]);
	const std::array expected{ 11.0f, 22.0f, 33.0f, 44.0f };
	for (std::size_t i = 0; i < expected.size(); ++i)
	{
		EXPECT_FLOAT_EQ(actual[i], expected[i]);
	}
}

TEST(CompiledModuleVulkanTest, RunsPayloadWithUnusedWorkspaceTensorBinding)
{
	if (!IsVulkanDeviceAvailable())
	{
		GTEST_SKIP() << "No Vulkan compute device is available";
	}

	const auto graph = BuildSimpleBinaryGraph(BinaryOp::Add);
	const auto artifact = Compiler<Vulkan>::CompileArtifact(Detail::BuildExecutablePlanFromGraph(graph));
	auto payload = DeserializeVulkanNativeInstructionPayload(artifact.Instructions());
	ASSERT_EQ(payload.kernels.size(), 1u);
	payload.workspaceTensors.push_back({
	    .byteSize = 64,
	    .alignment = 16,
	});
	payload.kernels[0].arguments.push_back({
	    .kind = VulkanNativeArgumentKind::WorkspaceTensor,
	    .index = 0,
	    .binding = 3,
	    .byteOffset = 0,
	    .byteSize = 64,
	});
	auto instructions = SerializeVulkanNativeInstructionPayload(payload);
	auto patchedArtifact = CompiledModuleArtifact::CopyFromImage({
	    .rodata = artifact.Rodata().data(),
	    .rodataSize = artifact.Rodata().size(),
	    .instructions = instructions.data(),
	    .instructionSize = instructions.size(),
	});
	auto module = patchedArtifact.Load(Vulkan{});

	Vulkan device;
	std::array inputs{
		Tensor<Vulkan>({ 1.0, 2.0, 3.0, 4.0 }, { 4 }, DataType::Float32, device),
		Tensor<Vulkan>({ 10.0, 20.0, 30.0, 40.0 }, { 4 }, DataType::Float32, device),
	};
	auto outputs = module.RunTensors(std::span<const Tensor<Vulkan>>(inputs));
	ASSERT_EQ(outputs.size(), 1);

	const auto actual = CopyToHost(outputs[0]);
	const std::array expected{ 11.0f, 22.0f, 33.0f, 44.0f };
	for (std::size_t i = 0; i < expected.size(); ++i)
	{
		EXPECT_FLOAT_EQ(actual[i], expected[i]);
	}
}

TEST(CompiledModuleVulkanTest, ReusesRunWorkspaceOutputs)
{
	if (!IsVulkanDeviceAvailable())
	{
		GTEST_SKIP() << "No Vulkan compute device is available";
	}

	Vulkan device;
	device.bufferResidency = VulkanBufferResidency::DeviceLocal;

	const auto graph = BuildSimpleBinaryGraph(BinaryOp::Add);
	auto module = Compiler<Vulkan>::Compile(Detail::BuildExecutablePlanFromGraph(graph), device);
	ASSERT_EQ(module.Backend(), CompiledModuleBackend::VulkanNative);

	std::array inputs{
		Tensor<Vulkan>({ 1.0, 2.0, 3.0, 4.0 }, { 4 }, DataType::Float32, device),
		Tensor<Vulkan>({ 10.0, 20.0, 30.0, 40.0 }, { 4 }, DataType::Float32, device),
	};
	auto workspace = module.CreateRunWorkspace();
	ASSERT_EQ(workspace.Outputs().size(), 1);
	EXPECT_EQ(workspace.Outputs()[0].CurDevice().bufferResidency, VulkanBufferResidency::DeviceLocal);

	const auto* outputStorage = workspace.Outputs()[0].UnsafeRawData();
	auto outputs = module.RunTensors(std::span<const Tensor<Vulkan>>(inputs), workspace);
	ASSERT_EQ(outputs.size(), 1);
	EXPECT_EQ(outputs[0].UnsafeRawData(), outputStorage);

	outputs = module.RunTensors(std::span<const Tensor<Vulkan>>(inputs), workspace);
	ASSERT_EQ(outputs.size(), 1);
	EXPECT_EQ(outputs[0].UnsafeRawData(), outputStorage);

	const auto actual = CopyToHost(outputs[0]);
	const std::array expected{ 11.0f, 22.0f, 33.0f, 44.0f };
	for (std::size_t i = 0; i < expected.size(); ++i)
	{
		EXPECT_FLOAT_EQ(actual[i], expected[i]);
	}
}

TEST(CompiledModuleVulkanTest, ReusesRunWorkspaceForCPUBridge)
{
	if (!IsVulkanDeviceAvailable())
	{
		GTEST_SKIP() << "No Vulkan compute device is available";
	}

	Vulkan device;
	device.hostFallbackPolicy = VulkanHostFallbackPolicy::Allow;
	CompilerOptions options = CompilerOptions::Defaults();
	options.enableVulkanNativeAOT = false;

	const auto graph = BuildSimpleBinaryGraph(BinaryOp::Add);
	auto module = Compiler<Vulkan>::Compile(Detail::BuildExecutablePlanFromGraph(graph), device, options);
	ASSERT_EQ(module.Backend(), CompiledModuleBackend::CPUNative);

	std::array inputs{
		Tensor<Vulkan>({ 1.0, 2.0, 3.0, 4.0 }, { 4 }, DataType::Float32, device),
		Tensor<Vulkan>({ 10.0, 20.0, 30.0, 40.0 }, { 4 }, DataType::Float32, device),
	};
	auto workspace = module.CreateRunWorkspace();
	ASSERT_EQ(workspace.Outputs().size(), 1);
	const auto* outputStorage = workspace.Outputs()[0].UnsafeRawData();

	auto outputs = module.RunTensors(std::span<const Tensor<Vulkan>>(inputs), workspace);
	ASSERT_EQ(outputs.size(), 1);
	EXPECT_EQ(outputs[0].UnsafeRawData(), outputStorage);

	outputs = module.RunTensors(std::span<const Tensor<Vulkan>>(inputs), workspace);
	ASSERT_EQ(outputs.size(), 1);
	EXPECT_EQ(outputs[0].UnsafeRawData(), outputStorage);

	const auto actual = CopyToHost(outputs[0]);
	const std::array expected{ 11.0f, 22.0f, 33.0f, 44.0f };
	for (std::size_t i = 0; i < expected.size(); ++i)
	{
		EXPECT_FLOAT_EQ(actual[i], expected[i]);
	}
}

TEST(CompiledModuleVulkanTest, RunsBinaryChainArithmetic)
{
	if (!IsVulkanDeviceAvailable())
	{
		GTEST_SKIP() << "No Vulkan compute device is available";
	}

	const auto graph = BuildBinaryChainGraph(BinaryOp::Add, BinaryOp::Multiply);
	auto module = Compiler<Vulkan>::Compile(Detail::BuildExecutablePlanFromGraph(graph), Vulkan{});
	ASSERT_EQ(module.Backend(), CompiledModuleBackend::VulkanNative);

	Vulkan device;
	std::array inputs{
		Tensor<Vulkan>({ 1.0, 2.0, 3.0, 4.0 }, { 4 }, DataType::Float32, device),
		Tensor<Vulkan>({ 10.0, 20.0, 30.0, 40.0 }, { 4 }, DataType::Float32, device),
		Tensor<Vulkan>({ 100.0, 200.0, 300.0, 400.0 }, { 4 }, DataType::Float32, device),
	};
	std::vector<CompiledModuleVulkanProfileEvent> events;
	auto outputs =
	    module.RunTensors(std::span<const Tensor<Vulkan>>(inputs), { .synchronize = true, .profileEvents = &events });
	ASSERT_EQ(outputs.size(), 1);
	ASSERT_EQ(events.size(), 2u);

	const auto actual = CopyToHost(outputs[0]);
	EXPECT_FLOAT_EQ(actual[0], 1100.0f);
	EXPECT_FLOAT_EQ(actual[1], 4400.0f);
	EXPECT_FLOAT_EQ(actual[2], 9900.0f);
	EXPECT_FLOAT_EQ(actual[3], 17600.0f);
}

TEST(CompiledModuleVulkanTest, RunsLongBinaryChainWithWorkspaceReuse)
{
	if (!IsVulkanDeviceAvailable())
	{
		GTEST_SKIP() << "No Vulkan compute device is available";
	}

	const auto graph = BuildThreeStageBinaryChainGraph();
	auto module = Compiler<Vulkan>::Compile(Detail::BuildExecutablePlanFromGraph(graph), Vulkan{});
	ASSERT_EQ(module.Backend(), CompiledModuleBackend::VulkanNative);

	Vulkan device;
	std::array inputs{
		Tensor<Vulkan>({ 1.0, 2.0, 3.0, 4.0 }, { 4 }, DataType::Float32, device),
		Tensor<Vulkan>({ 10.0, 20.0, 30.0, 40.0 }, { 4 }, DataType::Float32, device),
		Tensor<Vulkan>({ 2.0, 3.0, 4.0, 5.0 }, { 4 }, DataType::Float32, device),
		Tensor<Vulkan>({ 1.0, 2.0, 3.0, 4.0 }, { 4 }, DataType::Float32, device),
	};
	auto outputs = module.RunTensors(std::span<const Tensor<Vulkan>>(inputs));
	ASSERT_EQ(outputs.size(), 1);

	const auto actual = CopyToHost(outputs[0]);
	const std::array expected{ 21.0f, 64.0f, 129.0f, 216.0f };
	for (std::size_t i = 0; i < expected.size(); ++i)
	{
		EXPECT_FLOAT_EQ(actual[i], expected[i]);
	}
}

TEST(CompiledModuleVulkanTest, RunsDiamondBinaryDAGWithWorkspace)
{
	if (!IsVulkanDeviceAvailable())
	{
		GTEST_SKIP() << "No Vulkan compute device is available";
	}

	const auto graph = BuildBinaryDiamondGraph();
	auto module = Compiler<Vulkan>::Compile(Detail::BuildExecutablePlanFromGraph(graph), Vulkan{});
	ASSERT_EQ(module.Backend(), CompiledModuleBackend::VulkanNative);

	Vulkan device;
	std::array inputs{
		Tensor<Vulkan>({ 1.0, 2.0, 3.0, 4.0 }, { 4 }, DataType::Float32, device),
		Tensor<Vulkan>({ 10.0, 20.0, 30.0, 40.0 }, { 4 }, DataType::Float32, device),
		Tensor<Vulkan>({ 100.0, 200.0, 300.0, 400.0 }, { 4 }, DataType::Float32, device),
	};
	auto outputs = module.RunTensors(std::span<const Tensor<Vulkan>>(inputs));
	ASSERT_EQ(outputs.size(), 1);

	const auto actual = CopyToHost(outputs[0]);
	const std::array expected{ 112.0f, 224.0f, 336.0f, 448.0f };
	for (std::size_t i = 0; i < expected.size(); ++i)
	{
		EXPECT_FLOAT_EQ(actual[i], expected[i]);
	}
}

TEST(CompiledModuleVulkanTest, RunsBranchedBinaryDAGWithTailWorkspace)
{
	if (!IsVulkanDeviceAvailable())
	{
		GTEST_SKIP() << "No Vulkan compute device is available";
	}

	const auto graph = BuildBranchedBinaryDAGWithTailGraph();
	auto module = Compiler<Vulkan>::Compile(Detail::BuildExecutablePlanFromGraph(graph), Vulkan{});
	ASSERT_EQ(module.Backend(), CompiledModuleBackend::VulkanNative);

	Vulkan device;
	std::array inputs{
		Tensor<Vulkan>({ 1.0, 2.0, 3.0, 4.0 }, { 4 }, DataType::Float32, device),
		Tensor<Vulkan>({ 10.0, 20.0, 30.0, 40.0 }, { 4 }, DataType::Float32, device),
		Tensor<Vulkan>({ 2.0, 3.0, 4.0, 5.0 }, { 4 }, DataType::Float32, device),
		Tensor<Vulkan>({ 1.0, 1.0, 1.0, 1.0 }, { 4 }, DataType::Float32, device),
		Tensor<Vulkan>({ 5.0, 6.0, 7.0, 8.0 }, { 4 }, DataType::Float32, device),
	};
	auto outputs = module.RunTensors(std::span<const Tensor<Vulkan>>(inputs));
	ASSERT_EQ(outputs.size(), 1);

	const auto actual = CopyToHost(outputs[0]);
	const std::array expected{ 39.0f, 104.0f, 191.0f, 300.0f };
	for (std::size_t i = 0; i < expected.size(); ++i)
	{
		EXPECT_FLOAT_EQ(actual[i], expected[i]);
	}
}

TEST(CompiledModuleVulkanTest, RunsMixedElementwiseDAGWithWorkspace)
{
	if (!IsVulkanDeviceAvailable())
	{
		GTEST_SKIP() << "No Vulkan compute device is available";
	}

	const auto graph = BuildMixedElementwiseDAGGraph();
	auto module = Compiler<Vulkan>::Compile(Detail::BuildExecutablePlanFromGraph(graph), Vulkan{});
	ASSERT_EQ(module.Backend(), CompiledModuleBackend::VulkanNative);

	Vulkan device;
	std::array inputs{
		Tensor<Vulkan>({ -2.0, -1.0, 3.0, -4.0 }, { 4 }, DataType::Float32, device),
		Tensor<Vulkan>({ 1.0, 3.0, -5.0, 2.0 }, { 4 }, DataType::Float32, device),
		Tensor<Vulkan>({ 2.0, 4.0, 3.0, 5.0 }, { 4 }, DataType::Float32, device),
	};
	std::vector<CompiledModuleVulkanProfileEvent> events;
	auto outputs =
	    module.RunTensors(std::span<const Tensor<Vulkan>>(inputs), { .synchronize = true, .profileEvents = &events });
	ASSERT_EQ(outputs.size(), 1);
	ASSERT_EQ(events.size(), 3u);
	EXPECT_EQ(events[0].entryPoint, VulkanNativeSameShapeBinaryF32KernelName(BinaryOp::Add));
	EXPECT_EQ(events[1].entryPoint, VulkanNativeSameShapeUnaryF32KernelName(UnaryOp::Abs));
	EXPECT_EQ(events[1].descriptorCount, 3u);
	EXPECT_EQ(events[2].entryPoint, VulkanNativeSameShapeBinaryF32KernelName(BinaryOp::Multiply));

	const auto actual = CopyToHost(outputs[0]);
	const std::array expected{ 2.0f, 8.0f, 6.0f, 10.0f };
	for (std::size_t i = 0; i < expected.size(); ++i)
	{
		EXPECT_FLOAT_EQ(actual[i], expected[i]);
	}
}

TEST(CompiledModuleVulkanTest, RunsSimpleReduceArithmetic)
{
	if (!IsVulkanDeviceAvailable())
	{
		GTEST_SKIP() << "No Vulkan compute device is available";
	}

	struct ReduceRunCase
	{
		Graph graph;
		std::vector<double> input;
		std::vector<float> expected;
	};

	std::vector<ReduceRunCase> cases;
	cases.push_back({
	    .graph = BuildReduceGraph(ReduceOp::Sum, 1, { 2 }),
	    .input = { 1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f },
	    .expected = { 6.0f, 15.0f },
	});
	cases.push_back({
	    .graph = BuildReduceGraph(ReduceOp::Mean, 0, { 3 }),
	    .input = { 1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f },
	    .expected = { 2.5f, 3.5f, 4.5f },
	});
	cases.push_back({
	    .graph = BuildReduceGraph(ReduceOp::Max, 1, { 2 }),
	    .input = { 1.0f, 7.0f, 3.0f, 4.0f, 5.0f, 6.0f },
	    .expected = { 7.0f, 6.0f },
	});
	cases.push_back({
	    .graph = BuildReduceGraph(ReduceOp::Min, 1, { 2 }),
	    .input = { 1.0f, 7.0f, 3.0f, 4.0f, 5.0f, 6.0f },
	    .expected = { 1.0f, 4.0f },
	});

	Vulkan device;
	for (const auto& testCase : cases)
	{
		auto module = Compiler<Vulkan>::Compile(Detail::BuildExecutablePlanFromGraph(testCase.graph), Vulkan{});
		ASSERT_EQ(module.Backend(), CompiledModuleBackend::VulkanNative);

		std::array inputs{
			Tensor<Vulkan>(testCase.input, { 2, 3 }, DataType::Float32, device),
		};
		auto outputs = module.RunTensors(std::span<const Tensor<Vulkan>>(inputs));
		ASSERT_EQ(outputs.size(), 1);

		const auto actual = CopyToHostVector(outputs[0]);
		ASSERT_EQ(actual.size(), testCase.expected.size());
		for (std::size_t i = 0; i < actual.size(); ++i)
		{
			EXPECT_FLOAT_EQ(actual[i], testCase.expected[i]);
		}
	}
}

TEST(CompiledModuleVulkanTest, RunsSimpleSoftmaxArithmetic)
{
	if (!IsVulkanDeviceAvailable())
	{
		GTEST_SKIP() << "No Vulkan compute device is available";
	}

	const auto graph = BuildSoftmaxGraph(1);
	auto module = Compiler<Vulkan>::Compile(Detail::BuildExecutablePlanFromGraph(graph), Vulkan{});
	ASSERT_EQ(module.Backend(), CompiledModuleBackend::VulkanNative);

	Vulkan device;
	std::array inputs{
		Tensor<Vulkan>({ 1000.0, 999.0, 998.0, -1.0, 0.0, 1.0 }, { 2, 3 }, DataType::Float32, device),
	};
	auto outputs = module.RunTensors(std::span<const Tensor<Vulkan>>(inputs));
	ASSERT_EQ(outputs.size(), 1);

	const auto actual = CopyToHostVector(outputs[0]);
	const std::array expected{
		0.66524096f, 0.24472847f, 0.09003057f, 0.09003057f, 0.24472847f, 0.66524096f,
	};
	ASSERT_EQ(actual.size(), expected.size());
	for (std::size_t i = 0; i < actual.size(); ++i)
	{
		EXPECT_NEAR(actual[i], expected[i], 1e-5f);
	}
}

TEST(CompiledModuleVulkanTest, RunsSimpleNormalizationArithmetic)
{
	if (!IsVulkanDeviceAvailable())
	{
		GTEST_SKIP() << "No Vulkan compute device is available";
	}

	const std::array input{ 1.0f, 2.0f, 3.0f, 3.0f, 4.0f, 0.0f };
	const auto expectedFor = [&](NormalizationMode mode) {
		std::vector<float> expected;
		expected.reserve(input.size());
		for (std::size_t row = 0; row < 2; ++row)
		{
			float mean = 0.0f;
			if (mode == NormalizationMode::LayerNorm)
			{
				for (std::size_t col = 0; col < 3; ++col)
				{
					mean += input[row * 3 + col];
				}
				mean /= 3.0f;
			}

			float variance = 0.0f;
			for (std::size_t col = 0; col < 3; ++col)
			{
				const auto centered =
				    mode == NormalizationMode::LayerNorm ? input[row * 3 + col] - mean : input[row * 3 + col];
				variance += centered * centered;
			}
			variance /= 3.0f;
			const auto denom = std::sqrt(variance + 1e-5f);
			for (std::size_t col = 0; col < 3; ++col)
			{
				const auto centered =
				    mode == NormalizationMode::LayerNorm ? input[row * 3 + col] - mean : input[row * 3 + col];
				expected.push_back(centered / denom);
			}
		}
		return expected;
	};

	Vulkan device;
	for (const auto mode : { NormalizationMode::LayerNorm, NormalizationMode::RMSNorm })
	{
		const auto graph = BuildNormalizationGraph(mode, 1);
		auto module = Compiler<Vulkan>::Compile(Detail::BuildExecutablePlanFromGraph(graph), Vulkan{});
		ASSERT_EQ(module.Backend(), CompiledModuleBackend::VulkanNative);

		std::array inputs{
			Tensor<Vulkan>(std::vector<double>(input.begin(), input.end()), { 2, 3 }, DataType::Float32, device),
		};
		auto outputs = module.RunTensors(std::span<const Tensor<Vulkan>>(inputs));
		ASSERT_EQ(outputs.size(), 1);

		const auto actual = CopyToHostVector(outputs[0]);
		const auto expected = expectedFor(mode);
		ASSERT_EQ(actual.size(), expected.size());
		for (std::size_t i = 0; i < actual.size(); ++i)
		{
			EXPECT_NEAR(actual[i], expected[i], 1e-5f);
		}
	}
}

TEST(CompiledModuleVulkanTest, RunsAffineNormalizationExternalWeightsArithmetic)
{
	if (!IsVulkanDeviceAvailable())
	{
		GTEST_SKIP() << "No Vulkan compute device is available";
	}

	const std::array input{ 1.0f, 2.0f, 3.0f, 3.0f, 4.0f, 0.0f };
	const std::array scale{ 2.0f, 3.0f, 4.0f };
	const std::array bias{ 0.5f, -0.5f, 1.0f };
	std::vector<float> expected;
	expected.reserve(input.size());
	for (std::size_t row = 0; row < 2; ++row)
	{
		float mean = 0.0f;
		for (std::size_t col = 0; col < 3; ++col)
		{
			mean += input[row * 3 + col];
		}
		mean /= 3.0f;
		float variance = 0.0f;
		for (std::size_t col = 0; col < 3; ++col)
		{
			const auto centered = input[row * 3 + col] - mean;
			variance += centered * centered;
		}
		variance /= 3.0f;
		const auto denom = std::sqrt(variance + 1e-5f);
		for (std::size_t col = 0; col < 3; ++col)
		{
			expected.push_back(((input[row * 3 + col] - mean) / denom) * scale[col] + bias[col]);
		}
	}

	const auto graph = BuildAffineNormalizationVariableGraph(NormalizationMode::LayerNorm, 1);
	auto module = Compiler<Vulkan>::Compile(Detail::BuildExecutablePlanFromGraph(graph), Vulkan{});
	ASSERT_EQ(module.Backend(), CompiledModuleBackend::VulkanNative);

	Vulkan device;
	std::array inputs{
		Tensor<Vulkan>(std::vector<double>(input.begin(), input.end()), { 2, 3 }, DataType::Float32, device),
	};
	auto outputs = module.RunTensors(std::span<const Tensor<Vulkan>>(inputs));
	ASSERT_EQ(outputs.size(), 1);

	const auto actual = CopyToHostVector(outputs[0]);
	ASSERT_EQ(actual.size(), expected.size());
	for (std::size_t i = 0; i < actual.size(); ++i)
	{
		EXPECT_NEAR(actual[i], expected[i], 1e-5f);
	}
}

TEST(CompiledModuleVulkanTest, RunsGroupNormArithmetic)
{
	if (!IsVulkanDeviceAvailable())
	{
		GTEST_SKIP() << "No Vulkan compute device is available";
	}

	struct GroupNormRunCase
	{
		Graph graph;
		std::vector<double> input;
		std::vector<float> expected;
		std::vector<std::size_t> shape;
	};

	std::vector<GroupNormRunCase> cases;
	cases.push_back({
	    .graph = BuildGroupNormGraph({ 8 }, 4),
	    .input = { 1.0f, 3.0f, 2.0f, 4.0f, 5.0f, 7.0f, 6.0f, 8.0f },
	    .expected = { -1.0f, 1.0f, -1.0f, 1.0f, -1.0f, 1.0f, -1.0f, 1.0f },
	    .shape = { 8 },
	});
	cases.push_back({
	    .graph = BuildGroupNormGraph({ 4, 1, 1, 2 }, 2),
	    .input = { 1.0f, 10.0f, 3.0f, 30.0f, 2.0f, 20.0f, 4.0f, 40.0f },
	    .expected = { -1.0f, -1.0f, 1.0f, 1.0f, -1.0f, -1.0f, 1.0f, 1.0f },
	    .shape = { 4, 1, 1, 2 },
	});

	Vulkan device;
	for (const auto& testCase : cases)
	{
		auto module = Compiler<Vulkan>::Compile(Detail::BuildExecutablePlanFromGraph(testCase.graph), Vulkan{});
		ASSERT_EQ(module.Backend(), CompiledModuleBackend::VulkanNative);

		std::array inputs{
			Tensor<Vulkan>(testCase.input, testCase.shape, DataType::Float32, device),
		};
		auto outputs = module.RunTensors(std::span<const Tensor<Vulkan>>(inputs));
		ASSERT_EQ(outputs.size(), 1);

		const auto actual = CopyToHostVector(outputs[0]);
		ASSERT_EQ(actual.size(), testCase.expected.size());
		for (std::size_t i = 0; i < actual.size(); ++i)
		{
			EXPECT_NEAR(actual[i], testCase.expected[i], 1e-3f);
		}
	}
}

TEST(CompiledModuleVulkanTest, RunsAffineGroupNormExternalWeightsArithmetic)
{
	if (!IsVulkanDeviceAvailable())
	{
		GTEST_SKIP() << "No Vulkan compute device is available";
	}

	const std::vector<double> input{ 1.0f, 3.0f, 2.0f, 4.0f, 5.0f, 7.0f, 6.0f, 8.0f };
	std::vector<float> expected;
	expected.reserve(input.size());
	for (std::size_t group = 0; group < 4; ++group)
	{
		const auto base = group * 2;
		const auto mean = static_cast<float>((input[base] + input[base + 1]) * 0.5);
		float variance = 0.0f;
		for (std::size_t member = 0; member < 2; ++member)
		{
			const auto centered = static_cast<float>(input[base + member]) - mean;
			variance += centered * centered;
		}
		variance *= 0.5f;
		const auto denom = std::sqrt(variance + 1e-6f);
		for (std::size_t member = 0; member < 2; ++member)
		{
			const auto index = base + member;
			const auto scale = static_cast<float>(1.0 + 0.25 * static_cast<double>((index % 3) + 1));
			const auto bias = static_cast<float>(-0.25 + 0.125 * static_cast<double>(index % 5));
			expected.push_back(((static_cast<float>(input[index]) - mean) / denom) * scale + bias);
		}
	}

	const auto graph = BuildAffineGroupNormVariableGraph({ 8 }, 4);
	auto module = Compiler<Vulkan>::Compile(Detail::BuildExecutablePlanFromGraph(graph), Vulkan{});
	ASSERT_EQ(module.Backend(), CompiledModuleBackend::VulkanNative);

	Vulkan device;
	std::array inputs{
		Tensor<Vulkan>(input, { 8 }, DataType::Float32, device),
	};
	auto outputs = module.RunTensors(std::span<const Tensor<Vulkan>>(inputs));
	ASSERT_EQ(outputs.size(), 1);

	const auto actual = CopyToHostVector(outputs[0]);
	ASSERT_EQ(actual.size(), expected.size());
	for (std::size_t i = 0; i < actual.size(); ++i)
	{
		EXPECT_NEAR(actual[i], expected[i], 1e-3f);
	}
}

TEST(CompiledModuleVulkanTest, RunsPool2DArithmetic)
{
	if (!IsVulkanDeviceAvailable())
	{
		GTEST_SKIP() << "No Vulkan compute device is available";
	}

	struct PoolRunCase
	{
		PoolMode mode{ PoolMode::Max };
		std::vector<float> expected;
	};

	const std::vector<double> input{ 1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f, 7.0f, 8.0f, 9.0f };
	const std::array cases{
		PoolRunCase{ .mode = PoolMode::Max, .expected = { 5.0f, 6.0f, 8.0f, 9.0f } },
		PoolRunCase{ .mode = PoolMode::Average, .expected = { 3.0f, 4.0f, 6.0f, 7.0f } },
	};

	Vulkan device;
	for (const auto& testCase : cases)
	{
		const auto graph = BuildPool2DGraph(testCase.mode);
		auto module = Compiler<Vulkan>::Compile(Detail::BuildExecutablePlanFromGraph(graph), Vulkan{});
		ASSERT_EQ(module.Backend(), CompiledModuleBackend::VulkanNative);

		std::array inputs{
			Tensor<Vulkan>(input, { 1, 1, 3, 3 }, DataType::Float32, device),
		};
		auto outputs = module.RunTensors(std::span<const Tensor<Vulkan>>(inputs));
		ASSERT_EQ(outputs.size(), 1);

		const auto actual = CopyToHostVector(outputs[0]);
		ASSERT_EQ(actual.size(), testCase.expected.size());
		for (std::size_t i = 0; i < actual.size(); ++i)
		{
			EXPECT_NEAR(actual[i], testCase.expected[i], 1e-5f);
		}
	}
}

TEST(CompiledModuleVulkanTest, RunsPaddedPool2DArithmetic)
{
	if (!IsVulkanDeviceAvailable())
	{
		GTEST_SKIP() << "No Vulkan compute device is available";
	}

	struct PoolRunCase
	{
		PoolMode mode{ PoolMode::Max };
		bool countIncludePad{};
		std::vector<float> expected;
	};

	const std::vector<double> input{ 1.0f, 2.0f, 3.0f, 4.0f };
	const std::array cases{
		PoolRunCase{ .mode = PoolMode::Max,
		             .countIncludePad = false,
		             .expected = { 1.0f, 2.0f, 2.0f, 3.0f, 4.0f, 4.0f, 3.0f, 4.0f, 4.0f } },
		PoolRunCase{ .mode = PoolMode::Average,
		             .countIncludePad = false,
		             .expected = { 1.0f, 1.5f, 2.0f, 2.0f, 2.5f, 3.0f, 3.0f, 3.5f, 4.0f } },
		PoolRunCase{ .mode = PoolMode::Average,
		             .countIncludePad = true,
		             .expected = { 0.25f, 0.75f, 0.5f, 1.0f, 2.5f, 1.5f, 0.75f, 1.75f, 1.0f } },
	};

	Vulkan device;
	for (const auto& testCase : cases)
	{
		const auto graph = BuildPaddedPool2DGraph(testCase.mode, testCase.countIncludePad);
		auto module = Compiler<Vulkan>::Compile(Detail::BuildExecutablePlanFromGraph(graph), Vulkan{});
		ASSERT_EQ(module.Backend(), CompiledModuleBackend::VulkanNative);

		std::array inputs{
			Tensor<Vulkan>(input, { 1, 1, 2, 2 }, DataType::Float32, device),
		};
		auto outputs = module.RunTensors(std::span<const Tensor<Vulkan>>(inputs));
		ASSERT_EQ(outputs.size(), 1);

		const auto actual = CopyToHostVector(outputs[0]);
		ASSERT_EQ(actual.size(), testCase.expected.size());
		for (std::size_t i = 0; i < actual.size(); ++i)
		{
			EXPECT_NEAR(actual[i], testCase.expected[i], 1e-5f);
		}
	}
}

TEST(CompiledModuleVulkanTest, RunsNearestUpsampleArithmetic)
{
	if (!IsVulkanDeviceAvailable())
	{
		GTEST_SKIP() << "No Vulkan compute device is available";
	}

	const auto graph = BuildNearestUpsampleGraph();
	auto module = Compiler<Vulkan>::Compile(Detail::BuildExecutablePlanFromGraph(graph), Vulkan{});
	ASSERT_EQ(module.Backend(), CompiledModuleBackend::VulkanNative);

	Vulkan device;
	std::array inputs{
		Tensor<Vulkan>({ 1.0, 2.0, 3.0, 4.0 }, { 1, 1, 2, 2 }, DataType::Float32, device),
	};
	auto outputs = module.RunTensors(std::span<const Tensor<Vulkan>>(inputs));
	ASSERT_EQ(outputs.size(), 1);

	const auto actual = CopyToHostVector(outputs[0]);
	const std::array expected{ 1.0f, 1.0f, 2.0f, 2.0f, 1.0f, 1.0f, 2.0f, 2.0f,
		                       3.0f, 3.0f, 4.0f, 4.0f, 3.0f, 3.0f, 4.0f, 4.0f };
	ASSERT_EQ(actual.size(), expected.size());
	for (std::size_t i = 0; i < actual.size(); ++i)
	{
		EXPECT_NEAR(actual[i], expected[i], 1e-5f);
	}
}

TEST(CompiledModuleVulkanTest, RunsSimpleConv2DArithmetic)
{
	if (!IsVulkanDeviceAvailable())
	{
		GTEST_SKIP() << "No Vulkan compute device is available";
	}

	const auto graph = BuildSimpleConv2DVariableGraph();
	auto module = Compiler<Vulkan>::Compile(Detail::BuildExecutablePlanFromGraph(graph), Vulkan{});
	ASSERT_EQ(module.Backend(), CompiledModuleBackend::VulkanNative);

	Vulkan device;
	std::array inputs{
		Tensor<Vulkan>({ 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0 }, { 1, 1, 3, 3 }, DataType::Float32, device),
	};
	auto outputs = module.RunTensors(std::span<const Tensor<Vulkan>>(inputs));
	ASSERT_EQ(outputs.size(), 1);

	const auto actual = CopyToHostVector(outputs[0]);
	const std::array expected{ 6.5f, 8.5f, 12.5f, 14.5f };
	ASSERT_EQ(actual.size(), expected.size());
	for (std::size_t i = 0; i < actual.size(); ++i)
	{
		EXPECT_NEAR(actual[i], expected[i], 1e-5f);
	}
}

TEST(CompiledModuleVulkanTest, RunsGroupedConv2DArithmetic)
{
	if (!IsVulkanDeviceAvailable())
	{
		GTEST_SKIP() << "No Vulkan compute device is available";
	}

	const auto graph = BuildGroupedConv2DVariableGraph();
	auto module = Compiler<Vulkan>::Compile(Detail::BuildExecutablePlanFromGraph(graph), Vulkan{});
	ASSERT_EQ(module.Backend(), CompiledModuleBackend::VulkanNative);

	Vulkan device;
	std::array inputs{
		Tensor<Vulkan>(
		    { 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0, 12.0, 13.0, 14.0, 15.0, 16.0, 17.0, 18.0 },
		    { 1, 2, 3, 3 }, DataType::Float32, device),
	};
	auto outputs = module.RunTensors(std::span<const Tensor<Vulkan>>(inputs));
	ASSERT_EQ(outputs.size(), 1);

	const auto actual = CopyToHostVector(outputs[0]);
	const std::array expected{ 6.5f, 8.5f, 12.5f, 14.5f, 49.0f, 53.0f, 61.0f, 65.0f };
	ASSERT_EQ(actual.size(), expected.size());
	for (std::size_t i = 0; i < actual.size(); ++i)
	{
		EXPECT_NEAR(actual[i], expected[i], 1e-5f);
	}
}

TEST(CompiledModuleVulkanTest, RunsSimpleConvTranspose2DArithmetic)
{
	if (!IsVulkanDeviceAvailable())
	{
		GTEST_SKIP() << "No Vulkan compute device is available";
	}

	const auto graph = BuildSimpleConvTranspose2DVariableGraph();
	auto module = Compiler<Vulkan>::Compile(Detail::BuildExecutablePlanFromGraph(graph), Vulkan{});
	ASSERT_EQ(module.Backend(), CompiledModuleBackend::VulkanNative);

	Vulkan device;
	std::array inputs{
		Tensor<Vulkan>({ 1.0, 2.0, 3.0, 4.0 }, { 1, 1, 2, 2 }, DataType::Float32, device),
	};
	auto outputs = module.RunTensors(std::span<const Tensor<Vulkan>>(inputs));
	ASSERT_EQ(outputs.size(), 1);

	const auto actual = CopyToHostVector(outputs[0]);
	const std::array expected{ 1.5f, 3.5f, 2.5f, 4.5f, 10.5f, 6.5f, 3.5f, 7.5f, 4.5f };
	ASSERT_EQ(actual.size(), expected.size());
	for (std::size_t i = 0; i < actual.size(); ++i)
	{
		EXPECT_NEAR(actual[i], expected[i], 1e-5f);
	}
}

TEST(CompiledModuleVulkanTest, RunsSliceArithmetic)
{
	if (!IsVulkanDeviceAvailable())
	{
		GTEST_SKIP() << "No Vulkan compute device is available";
	}

	const auto graph = BuildSliceGraph();
	auto module = Compiler<Vulkan>::Compile(Detail::BuildExecutablePlanFromGraph(graph), Vulkan{});
	ASSERT_EQ(module.Backend(), CompiledModuleBackend::VulkanNative);

	Vulkan device;
	std::array inputs{
		Tensor<Vulkan>({ 1.0, 2.0, 3.0, 4.0, 5.0, 6.0 }, { 2, 3 }, DataType::Float32, device),
	};
	auto outputs = module.RunTensors(std::span<const Tensor<Vulkan>>(inputs));
	ASSERT_EQ(outputs.size(), 1);

	const auto actual = CopyToHostVector(outputs[0]);
	const std::array expected{ 2.0f, 3.0f, 5.0f, 6.0f };
	ASSERT_EQ(actual.size(), expected.size());
	for (std::size_t i = 0; i < actual.size(); ++i)
	{
		EXPECT_NEAR(actual[i], expected[i], 1e-5f);
	}
}

TEST(CompiledModuleVulkanTest, RunsConcatArithmetic)
{
	if (!IsVulkanDeviceAvailable())
	{
		GTEST_SKIP() << "No Vulkan compute device is available";
	}

	const auto graph = BuildConcatGraph();
	auto module = Compiler<Vulkan>::Compile(Detail::BuildExecutablePlanFromGraph(graph), Vulkan{});
	ASSERT_EQ(module.Backend(), CompiledModuleBackend::VulkanNative);

	Vulkan device;
	std::array inputs{
		Tensor<Vulkan>({ 1.0, 2.0, 4.0, 5.0 }, { 2, 2 }, DataType::Float32, device),
		Tensor<Vulkan>({ 3.0, 6.0 }, { 2, 1 }, DataType::Float32, device),
	};
	auto outputs = module.RunTensors(std::span<const Tensor<Vulkan>>(inputs));
	ASSERT_EQ(outputs.size(), 1);

	const auto actual = CopyToHostVector(outputs[0]);
	const std::array expected{ 1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f };
	ASSERT_EQ(actual.size(), expected.size());
	for (std::size_t i = 0; i < actual.size(); ++i)
	{
		EXPECT_NEAR(actual[i], expected[i], 1e-5f);
	}
}

TEST(CompiledModuleVulkanTest, RunsSimpleMatMulArithmetic)
{
	if (!IsVulkanDeviceAvailable())
	{
		GTEST_SKIP() << "No Vulkan compute device is available";
	}

	const auto graph = BuildSimpleMatMulGraph();
	auto module = Compiler<Vulkan>::Compile(Detail::BuildExecutablePlanFromGraph(graph), Vulkan{});
	ASSERT_EQ(module.Backend(), CompiledModuleBackend::VulkanNative);

	Vulkan device;
	std::array inputs{
		Tensor<Vulkan>({ 1.0, 2.0, 3.0, 4.0, 5.0, 6.0 }, { 2, 3 }, DataType::Float32, device),
		Tensor<Vulkan>({ 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0, 12.0 }, { 3, 4 }, DataType::Float32,
		               device),
	};
	auto outputs = module.RunTensors(std::span<const Tensor<Vulkan>>(inputs));
	ASSERT_EQ(outputs.size(), 1);

	const auto actual = CopyToHostVector(outputs[0]);
	const std::array expected{ 38.0f, 44.0f, 50.0f, 56.0f, 83.0f, 98.0f, 113.0f, 128.0f };
	ASSERT_EQ(actual.size(), expected.size());
	for (std::size_t i = 0; i < expected.size(); ++i)
	{
		EXPECT_FLOAT_EQ(actual[i], expected[i]);
	}
}

TEST(CompiledModuleVulkanTest, RunsMatMulBiasExternalWeightsArithmetic)
{
	if (!IsVulkanDeviceAvailable())
	{
		GTEST_SKIP() << "No Vulkan compute device is available";
	}

	const auto graph = BuildSimpleMatMulBiasVariableGraph(true);
	const auto artifact = Compiler<Vulkan>::CompileArtifact(Detail::BuildExecutablePlanFromGraph(graph));
	ASSERT_EQ(artifact.Backend(), CompiledModuleBackend::VulkanNative);
	ASSERT_GT(artifact.Weights().size(), 0u);
	auto module = artifact.Load(Vulkan{});
	ASSERT_EQ(module.Backend(), CompiledModuleBackend::VulkanNative);

	Vulkan device;
	std::array inputs{
		Tensor<Vulkan>({ 1.0, 2.0, 3.0, 4.0, 5.0, 6.0 }, { 2, 3 }, DataType::Float32, device),
	};
	auto outputs = module.RunTensors(std::span<const Tensor<Vulkan>>(inputs));
	ASSERT_EQ(outputs.size(), 1);

	const auto actual = CopyToHostVector(outputs[0]);
	const std::array expected{ 39.0f, 0.0f, 53.0f, 0.0f, 84.0f, 0.0f, 116.0f, 0.0f };
	ASSERT_EQ(actual.size(), expected.size());
	for (std::size_t i = 0; i < expected.size(); ++i)
	{
		EXPECT_FLOAT_EQ(actual[i], expected[i]);
	}
}

TEST(CompiledModuleVulkanTest, RunsSeparatedExternalWeightsWithDeviceLocalTensors)
{
	if (!IsVulkanDeviceAvailable())
	{
		GTEST_SKIP() << "No Vulkan compute device is available";
	}

	const auto graph = BuildSimpleMatMulBiasVariableGraph(true);
	const auto artifact = Compiler<Vulkan>::CompileArtifact(Detail::BuildExecutablePlanFromGraph(graph));
	ASSERT_EQ(artifact.Backend(), CompiledModuleBackend::VulkanNative);
	ASSERT_GT(artifact.Weights().size(), 0u);

	Vulkan device;
	device.bufferResidency = VulkanBufferResidency::DeviceLocal;
	auto module = artifact.SeparateRodata().Load(device);
	ASSERT_EQ(module.Backend(), CompiledModuleBackend::VulkanNative);

	std::array inputs{
		Tensor<Vulkan>({ 1.0, 2.0, 3.0, 4.0, 5.0, 6.0 }, { 2, 3 }, DataType::Float32, device),
	};
	auto outputs = module.RunTensors(std::span<const Tensor<Vulkan>>(inputs));
	ASSERT_EQ(outputs.size(), 1);
	EXPECT_EQ(outputs[0].CurDevice().bufferResidency, VulkanBufferResidency::DeviceLocal);

	const auto actual = CopyToHostVector(outputs[0]);
	const std::array expected{ 39.0f, 0.0f, 53.0f, 0.0f, 84.0f, 0.0f, 116.0f, 0.0f };
	ASSERT_EQ(actual.size(), expected.size());
	for (std::size_t i = 0; i < expected.size(); ++i)
	{
		EXPECT_FLOAT_EQ(actual[i], expected[i]);
	}
}

TEST(CompiledModuleVulkanTest, RunsSimpleMatMulBiasReLUArithmetic)
{
	if (!IsVulkanDeviceAvailable())
	{
		GTEST_SKIP() << "No Vulkan compute device is available";
	}

	const auto graph = BuildSimpleMatMulBiasGraph(true);
	auto module = Compiler<Vulkan>::Compile(Detail::BuildExecutablePlanFromGraph(graph), Vulkan{});
	ASSERT_EQ(module.Backend(), CompiledModuleBackend::VulkanNative);

	Vulkan device;
	std::array inputs{
		Tensor<Vulkan>({ 1.0, 2.0, 3.0, 4.0, 5.0, 6.0 }, { 2, 3 }, DataType::Float32, device),
		Tensor<Vulkan>({ 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0, 12.0 }, { 3, 4 }, DataType::Float32,
		               device),
		Tensor<Vulkan>({ 1.0, -100.0, 3.0, -200.0 }, { 1, 4 }, DataType::Float32, device),
	};
	auto outputs = module.RunTensors(std::span<const Tensor<Vulkan>>(inputs));
	ASSERT_EQ(outputs.size(), 1);

	const auto actual = CopyToHostVector(outputs[0]);
	const std::array expected{ 39.0f, 0.0f, 53.0f, 0.0f, 84.0f, 0.0f, 116.0f, 0.0f };
	ASSERT_EQ(actual.size(), expected.size());
	for (std::size_t i = 0; i < expected.size(); ++i)
	{
		EXPECT_FLOAT_EQ(actual[i], expected[i]);
	}
}
