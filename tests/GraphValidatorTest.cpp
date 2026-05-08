#include <gtest/gtest.h>

#include <LiteNN.h>
#include <LiteNN/Runtime/Interpreter.h>
#include <LiteNN/Validation/GraphValidator.h>

#include <array>
#include <string>

using namespace LiteNN;

TEST(GraphValidator, AcceptsSimpleValidGraph)
{
	Graph graph;
	Subgraph subgraph;
	const auto x = subgraph.AddParam(DataType::Float32, { 2, 2 });
	const auto y = subgraph.AddNode(UnaryOpNode{ UnaryOp::Negate, { x, 0 } },
	                                { OutputInfo{ DataType::Float32, { 2, 2 } } });
	subgraph.SetResults({ { y, 0 } });
	graph.SetForward(graph.AddSubgraph(std::move(subgraph)));

	EXPECT_NO_THROW(Validation::ValidateGraph(graph));
}

TEST(GraphValidator, RejectsInvalidVariableRef)
{
	Graph graph;
	Subgraph subgraph;
	const auto var = subgraph.AddNode(VariableRefNode{ 3 }, { OutputInfo{ DataType::Float32, { 2, 2 } } });
	subgraph.SetResults({ { var, 0 } });
	graph.SetForward(graph.AddSubgraph(std::move(subgraph)));

	EXPECT_THROW(Validation::ValidateGraph(graph), Validation::GraphValidationError);
}

TEST(GraphValidator, RejectsIncorrectBinaryOutputMetadata)
{
	Graph graph;
	Subgraph subgraph;
	const auto a = subgraph.AddParam(DataType::Float32, { 2, 2 });
	const auto b = subgraph.AddParam(DataType::Float32, { 2, 2 });
	const auto y = subgraph.AddNode(BinaryOpNode{ BinaryOp::Add, { a, 0 }, { b, 0 } },
	                                { OutputInfo{ DataType::Float32, { 4 } } });
	subgraph.SetResults({ { y, 0 } });
	graph.SetForward(graph.AddSubgraph(std::move(subgraph)));

	try
	{
		Validation::ValidateGraph(graph);
		FAIL() << "expected graph validation to throw";
	}
	catch (const Validation::GraphValidationError& ex)
	{
		const std::string message = ex.what();
		EXPECT_NE(message.find("subgraph 0, node 2"), std::string::npos);
		EXPECT_NE(message.find("BinaryOp output expected Float32[2, 2], got Float32[4]"), std::string::npos);
		EXPECT_NE(message.find("nodeKind=BinaryOpNode"), std::string::npos);
	}
}

TEST(GraphValidator, RejectsCallSignatureMismatch)
{
	Graph graph;

	Subgraph callee;
	const auto calleeInput = callee.AddParam(DataType::Float32, { 3 });
	callee.SetResults({ { calleeInput, 0 } });
	const auto calleeId = graph.AddSubgraph(std::move(callee));

	Subgraph caller;
	const auto callerInput = caller.AddParam(DataType::Float32, { 2 });
	const auto call = caller.AddNode(CallNode{ calleeId, { { callerInput, 0 } } },
	                                 { OutputInfo{ DataType::Float32, { 3 } } });
	caller.SetResults({ { call, 0 } });
	graph.SetForward(graph.AddSubgraph(std::move(caller)));

	EXPECT_THROW(Validation::ValidateGraph(graph), Validation::GraphValidationError);
}

TEST(GraphValidator, RejectsRuntimeInputMismatch)
{
	Graph graph;
	Subgraph subgraph;
	const auto x = subgraph.AddParam(DataType::Float32, { 2 });
	subgraph.SetResults({ { x, 0 } });
	graph.SetForward(graph.AddSubgraph(std::move(subgraph)));

	Runtime::Interpreter<CPU> interpreter;
	Tensor<CPU> wrongInput({ 1, 2, 3 }, { 3 });
	std::array<Tensor<CPU>, 1> inputs = { std::move(wrongInput) };

	try
	{
		(void)interpreter.RunForward(graph, inputs);
		FAIL() << "expected runtime input validation to throw";
	}
	catch (const std::runtime_error& ex)
	{
		const std::string message = ex.what();
		EXPECT_NE(message.find("RunSubgraph input 0 mismatch for subgraph 0"), std::string::npos);
		EXPECT_NE(message.find("expected Float32[2]"), std::string::npos);
		EXPECT_NE(message.find("got Float32[3]"), std::string::npos);
	}
}
