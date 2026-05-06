#include <gtest/gtest.h>

#include <LiteNN.h>

#include "Dialect/LiteNNDialect.h"
#include "Dialect/LiteNNOps.h"
#include "Translation/GraphToMLIR.h"

#include "mlir/IR/MLIRContext.h"
#include "mlir/IR/Verifier.h"

using namespace LiteNN;

namespace
{

class CompilerTest : public ::testing::Test
{
protected:
	void SetUp() override
	{
		ctx_.disableMultithreading();

		ctx_.loadDialect<litenn::LiteNNDialect>();
	}

	mlir::MLIRContext ctx_;
};

TEST_F(CompilerTest, SimpleAdd)
{
	Graph graph;
	Subgraph sg;
	const auto a = sg.AddParam(DataType::Float32, { 2, 2 });
	const auto b = sg.AddParam(DataType::Float32, { 2, 2 });
	const auto y = sg.AddNode(BinaryOpNode{ BinaryOp::Add, { a, 0 }, { b, 0 } },
	                          { OutputInfo{ DataType::Float32, { 2, 2 } } });
	sg.SetResults({ { y, 0 } });
	const auto fwdId = graph.AddSubgraph(std::move(sg));
	graph.SetForward(fwdId);

	auto module = litenn::translateGraphToMLIR(graph, ctx_);
	ASSERT_TRUE(module);
	EXPECT_TRUE(mlir::succeeded(mlir::verify(*module)));

	// Dump for manual inspection
	module->dump();
}

TEST_F(CompilerTest, MatMulWithVariable)
{
	Graph graph;

	auto w = Variable::Create(Tensor<CPU>({ 1.0, 0.0, 0.0, 1.0 }, { 2, 2 }, DataType::Float32));
	const auto wIdx = graph.AddVariable(std::move(w));

	Subgraph sg;
	const auto x = sg.AddParam(DataType::Float32, { 1, 2 });
	const auto wRef = sg.AddNode(VariableRefNode{ wIdx }, { OutputInfo{ DataType::Float32, { 2, 2 } } });
	const auto y = sg.AddNode(BinaryOpNode{ BinaryOp::MatMul, { x, 0 }, { wRef, 0 } },
	                          { OutputInfo{ DataType::Float32, { 1, 2 } } });
	sg.SetResults({ { y, 0 } });
	graph.AddSubgraph(std::move(sg));
	graph.SetForward(0);

	auto module = litenn::translateGraphToMLIR(graph, ctx_);
	ASSERT_TRUE(module);
	EXPECT_TRUE(mlir::succeeded(mlir::verify(*module)));

	module->dump();
}

TEST_F(CompilerTest, CondNode)
{
	Graph graph;

	// Then branch: y = x * 2
	Subgraph thenSg;
	const auto thenX = thenSg.AddParam(DataType::Float32, { 2 });
	auto twoTensor = Tensor<CPU>({ 2.0, 2.0 }, { 2 }, DataType::Float32);
	const auto thenTwo = thenSg.AddNode(ConstantNode{ twoTensor.CopyToDevice(PolymorphicDevice{ CPU{} }) },
	                                    { OutputInfo{ DataType::Float32, { 2 } } });
	const auto thenMul = thenSg.AddNode(BinaryOpNode{ BinaryOp::Multiply, { thenX, 0 }, { thenTwo, 0 } },
	                                    { OutputInfo{ DataType::Float32, { 2 } } });
	thenSg.SetResults({ { thenMul, 0 } });
	const auto thenId = graph.AddSubgraph(std::move(thenSg));

	// Else branch: y = x + 1
	Subgraph elseSg;
	const auto elseX = elseSg.AddParam(DataType::Float32, { 2 });
	auto oneTensor = Tensor<CPU>({ 1.0, 1.0 }, { 2 }, DataType::Float32);
	const auto elseOne = elseSg.AddNode(ConstantNode{ oneTensor.CopyToDevice(PolymorphicDevice{ CPU{} }) },
	                                    { OutputInfo{ DataType::Float32, { 2 } } });
	const auto elseAdd = elseSg.AddNode(BinaryOpNode{ BinaryOp::Add, { elseX, 0 }, { elseOne, 0 } },
	                                    { OutputInfo{ DataType::Float32, { 2 } } });
	elseSg.SetResults({ { elseAdd, 0 } });
	const auto elseId = graph.AddSubgraph(std::move(elseSg));

	// Main subgraph: cond ? then(x) : else(x)
	Subgraph mainSg;
	const auto cond = mainSg.AddParam(DataType::Bool, { 1 });
	const auto x = mainSg.AddParam(DataType::Float32, { 2 });
	const auto condNode = mainSg.AddNode(CondNode{ { cond, 0 }, thenId, elseId, { { x, 0 } } },
	                                     { OutputInfo{ DataType::Float32, { 2 } } });
	mainSg.SetResults({ { condNode, 0 } });
	const auto mainId = graph.AddSubgraph(std::move(mainSg));
	graph.SetForward(mainId);

	auto module = litenn::translateGraphToMLIR(graph, ctx_);
	ASSERT_TRUE(module);
	EXPECT_TRUE(mlir::succeeded(mlir::verify(*module)));

	module->dump();
}

TEST_F(CompilerTest, WhileNode)
{
	Graph graph;

	// Cond: x < 100
	Subgraph condSg;
	const auto condX = condSg.AddParam(DataType::Float32, { 1 });
	auto hundredTensor = Tensor<CPU>({ 100.0 }, { 1 }, DataType::Float32);
	const auto hundred = condSg.AddNode(ConstantNode{ hundredTensor.CopyToDevice(PolymorphicDevice{ CPU{} }) },
	                                    { OutputInfo{ DataType::Float32, { 1 } } });
	const auto lt = condSg.AddNode(BinaryOpNode{ BinaryOp::Less, { condX, 0 }, { hundred, 0 } },
	                               { OutputInfo{ DataType::Bool, { 1 } } });
	condSg.SetResults({ { lt, 0 } });
	const auto condId = graph.AddSubgraph(std::move(condSg));

	// Body: x = x * 2
	Subgraph bodySg;
	const auto bodyX = bodySg.AddParam(DataType::Float32, { 1 });
	auto twoTensor = Tensor<CPU>({ 2.0 }, { 1 }, DataType::Float32);
	const auto two = bodySg.AddNode(ConstantNode{ twoTensor.CopyToDevice(PolymorphicDevice{ CPU{} }) },
	                                { OutputInfo{ DataType::Float32, { 1 } } });
	const auto mul = bodySg.AddNode(BinaryOpNode{ BinaryOp::Multiply, { bodyX, 0 }, { two, 0 } },
	                                { OutputInfo{ DataType::Float32, { 1 } } });
	bodySg.SetResults({ { mul, 0 } });
	const auto bodyId = graph.AddSubgraph(std::move(bodySg));

	// Main: while(x < 100) x = x * 2
	Subgraph mainSg;
	const auto initX = mainSg.AddParam(DataType::Float32, { 1 });
	const auto whileNode = mainSg.AddNode(WhileNode{ condId, bodyId, { { initX, 0 } } },
	                                      { OutputInfo{ DataType::Float32, { 1 } } });
	mainSg.SetResults({ { whileNode, 0 } });
	const auto mainId = graph.AddSubgraph(std::move(mainSg));
	graph.SetForward(mainId);

	auto module = litenn::translateGraphToMLIR(graph, ctx_);
	ASSERT_TRUE(module);
	EXPECT_TRUE(mlir::succeeded(mlir::verify(*module)));

	module->dump();
}

} // namespace
