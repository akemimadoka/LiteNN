#include <gtest/gtest.h>

#include <LiteNN/Pass/EGraphPass.h>
#include <LiteNN/Runtime/Interpreter.h>

#include <array>
#include <random>
#include <span>
#include <string>
#include <vector>

using namespace LiteNN;

namespace
{
	float ReadFloat(const Tensor<CPU>& t, std::size_t i)
	{
		return static_cast<const float*>(t.RawData())[i];
	}

	void ExpectOutputsNear(const std::vector<Tensor<CPU>>& actual, const std::vector<Tensor<CPU>>& expected,
	                       float tolerance = 1e-5f)
	{
		ASSERT_EQ(actual.size(), expected.size());
		for (auto output = 0uz; output < actual.size(); ++output)
		{
			ASSERT_EQ(actual[output].DType(), expected[output].DType());
			ASSERT_EQ(actual[output].Shape(), expected[output].Shape());
			ASSERT_EQ(actual[output].NumElements(), expected[output].NumElements());
			for (auto i = 0uz; i < actual[output].NumElements(); ++i)
			{
				EXPECT_NEAR(ReadFloat(actual[output], i), ReadFloat(expected[output], i), tolerance);
			}
		}
	}

	NodeId AddConstant(Subgraph& sg, std::span<const double> values, std::span<const std::size_t> shape)
	{
		auto tensor = Tensor<CPU>(values, shape, DataType::Float32).CopyToDevice(PolymorphicDevice{ CPU{} });
		return sg.AddNode(ConstantNode{ std::move(tensor) },
		                  { OutputInfo{ DataType::Float32, std::vector(shape.begin(), shape.end()) } });
	}

	Graph BuildRedundantPureGraph()
	{
		Graph graph;
		Subgraph sg;
		const auto x = sg.AddParam(DataType::Float32, { 2, 2 });
		const std::array<double, 4> zeros = { 0.0, 0.0, 0.0, 0.0 };
		const std::array<double, 4> ones = { 1.0, 1.0, 1.0, 1.0 };
		const std::array<std::size_t, 2> matrixShape = { 2, 2 };
		const auto zero = AddConstant(sg, zeros, matrixShape);
		const auto one = AddConstant(sg, ones, matrixShape);
		const auto added = sg.AddNode(BinaryOpNode{ BinaryOp::Add, { x, 0 }, { zero, 0 } },
		                              { OutputInfo{ DataType::Float32, { 2, 2 } } });
		const auto neg1 = sg.AddNode(UnaryOpNode{ UnaryOp::Negate, { added, 0 } },
		                             { OutputInfo{ DataType::Float32, { 2, 2 } } });
		const auto neg2 = sg.AddNode(UnaryOpNode{ UnaryOp::Negate, { neg1, 0 } },
		                             { OutputInfo{ DataType::Float32, { 2, 2 } } });
		const auto flat = sg.AddNode(ReshapeNode{ { neg2, 0 }, { 4 } }, { OutputInfo{ DataType::Float32, { 4 } } });
		const auto matrix =
		    sg.AddNode(ReshapeNode{ { flat, 0 }, { 2, 2 } }, { OutputInfo{ DataType::Float32, { 2, 2 } } });
		const auto transposed =
		    sg.AddNode(PermuteNode{ { matrix, 0 }, { 1, 0 } }, { OutputInfo{ DataType::Float32, { 2, 2 } } });
		const auto restored =
		    sg.AddNode(PermuteNode{ { transposed, 0 }, { 1, 0 } }, { OutputInfo{ DataType::Float32, { 2, 2 } } });
		const auto broadcast =
		    sg.AddNode(BroadcastToNode{ { restored, 0 }, { 2, 2 } }, { OutputInfo{ DataType::Float32, { 2, 2 } } });
		const auto multiplied = sg.AddNode(BinaryOpNode{ BinaryOp::Multiply, { broadcast, 0 }, { one, 0 } },
		                                   { OutputInfo{ DataType::Float32, { 2, 2 } } });
		sg.SetResults({ { multiplied, 0 } });
		graph.SetForward(graph.AddSubgraph(std::move(sg)));
		return graph;
	}

	std::array<Tensor<CPU>, 1> MakeInput()
	{
		return { Tensor<CPU>({ 1.0, -2.0, 3.5, 0.25 }, { 2, 2 }, DataType::Float32) };
	}

	Graph BuildTwoInputIdentityGraph()
	{
		Graph graph;
		Subgraph sg;
		const auto x = sg.AddParam(DataType::Float32, { 2, 2 });
		const auto y = sg.AddParam(DataType::Float32, { 2, 2 });
		const std::array<double, 4> zeros = { 0.0, 0.0, 0.0, 0.0 };
		const std::array<double, 4> ones = { 1.0, 1.0, 1.0, 1.0 };
		const std::array<std::size_t, 2> shape = { 2, 2 };
		const auto zero = AddConstant(sg, zeros, shape);
		const auto one = AddConstant(sg, ones, shape);
		const auto sum = sg.AddNode(BinaryOpNode{ BinaryOp::Add, { x, 0 }, { y, 0 } },
		                            { OutputInfo{ DataType::Float32, { 2, 2 } } });
		const auto redundant = sg.AddNode(BinaryOpNode{ BinaryOp::Add, { sum, 0 }, { zero, 0 } },
		                                  { OutputInfo{ DataType::Float32, { 2, 2 } } });
		const auto out = sg.AddNode(BinaryOpNode{ BinaryOp::Multiply, { redundant, 0 }, { one, 0 } },
		                            { OutputInfo{ DataType::Float32, { 2, 2 } } });
		sg.SetResults({ { out, 0 } });
		graph.SetForward(graph.AddSubgraph(std::move(sg)));
		return graph;
	}
} // namespace

TEST(EGraphPass, EliminatesPureRedundancyAndExplainsRewrites)
{
	auto graph = BuildRedundantPureGraph();
	const auto input = MakeInput();
	Runtime::Interpreter<CPU> interpreter;
	const auto expected = interpreter.RunForward(BuildExecutablePlan(graph), std::span<const Tensor<CPU>>(input));

	EGraphPass pass;
	pass.Run(graph);
	const auto actual = interpreter.RunForward(BuildExecutablePlan(graph), std::span<const Tensor<CPU>>(input));

	ExpectOutputsNear(actual, expected);
	EXPECT_EQ(graph.GetSubgraph(graph.Forward()).NodeCount(), 1u);
	EXPECT_GE(pass.LastReport().rewrites, 6u);
	const auto dump = pass.DumpLastReport();
	EXPECT_NE(dump.find("double-negate"), std::string::npos);
	EXPECT_NE(dump.find("permute-compose-identity"), std::string::npos);
	EXPECT_NE(dump.find("broadcast-noop"), std::string::npos);
}

TEST(EGraphPass, PreservesNumericsOnRandomizedInputs)
{
	auto graph = BuildTwoInputIdentityGraph();
	auto optimized = BuildTwoInputIdentityGraph();
	EGraphPass{}.Run(optimized);

	std::mt19937 rng(1234);
	std::uniform_real_distribution<double> dist(-3.0, 3.0);
	for (auto trial = 0; trial < 16; ++trial)
	{
		std::array<double, 4> xValues{};
		std::array<double, 4> yValues{};
		for (auto i = 0uz; i < xValues.size(); ++i)
		{
			xValues[i] = dist(rng);
			yValues[i] = dist(rng);
		}
		std::array<Tensor<CPU>, 2> originalInputs = {
			Tensor<CPU>(xValues, { 2, 2 }, DataType::Float32),
			Tensor<CPU>(yValues, { 2, 2 }, DataType::Float32),
		};
		std::array<Tensor<CPU>, 2> optimizedInputs = {
			Tensor<CPU>(xValues, { 2, 2 }, DataType::Float32),
			Tensor<CPU>(yValues, { 2, 2 }, DataType::Float32),
		};
		Runtime::Interpreter<CPU> interpreter;
		const auto expected = interpreter.RunForward(BuildExecutablePlan(graph), std::span<const Tensor<CPU>>(originalInputs));
		const auto actual = interpreter.RunForward(BuildExecutablePlan(optimized), std::span<const Tensor<CPU>>(optimizedInputs));
		ExpectOutputsNear(actual, expected);
	}
}

TEST(EGraphPass, GuardrailsStopBeforeGraphBlowup)
{
	auto graph = BuildRedundantPureGraph();
	EGraphOptions options;
	options.maxTerms = 1;
	EGraphPass pass(options);
	pass.Run(graph);

	EXPECT_TRUE(pass.LastReport().hitLimit);
	EXPECT_EQ(pass.LastReport().rewrites, 0u);
	EXPECT_NE(pass.DumpLastReport().find("hitLimit: true"), std::string::npos);
}
