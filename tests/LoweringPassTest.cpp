#include <gtest/gtest.h>

#include <LiteNN.h>

#include "Dialect/LiteNNDialect.h"
#include "Dialect/LiteNNOps.h"
#include "Translation/GraphToMLIR.h"
#include "Pass/LowerLiteNNPass.h"

#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Bufferization/IR/Bufferization.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/Dialect/Math/IR/Math.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/IR/Verifier.h"
#include "mlir/Pass/PassManager.h"

using namespace LiteNN;

namespace
{

class LoweringPassTest : public ::testing::Test
{
protected:
    void SetUp() override
    {
        ctx_.disableMultithreading();

        ctx_.loadDialect<
            litenn::LiteNNDialect,
            mlir::arith::ArithDialect,
            mlir::bufferization::BufferizationDialect,
            mlir::func::FuncDialect,
            mlir::linalg::LinalgDialect,
            mlir::math::MathDialect,
            mlir::memref::MemRefDialect,
            mlir::scf::SCFDialect,
            mlir::tensor::TensorDialect>();
    }

    // Translate graph, run the lowering pass, return the module.
    mlir::OwningOpRef<mlir::ModuleOp> translateAndLower(const Graph& graph)
    {
        auto module = litenn::translateGraphToMLIR(graph, ctx_);
        if (!module)
            return {};

        mlir::PassManager pm(&ctx_);
        pm.addPass(litenn::createLowerLiteNNPass());
        if (mlir::failed(pm.run(*module)))
            return {};

        return module;
    }

    // Assert no litenn.* ops remain in the module.
    static void expectNoLiteNNOps(mlir::ModuleOp mod)
    {
        bool found = false;
        mod->walk([&](mlir::Operation* op) {
            if (op->getName().getDialectNamespace() == "litenn")
                found = true;
        });
        EXPECT_FALSE(found) << "Module still contains litenn ops after lowering";
    }

    mlir::MLIRContext ctx_;
};

TEST_F(LoweringPassTest, SimpleAdd)
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

    auto module = translateAndLower(graph);
    ASSERT_TRUE(module);
    EXPECT_TRUE(mlir::succeeded(mlir::verify(*module)));
    expectNoLiteNNOps(*module);

    // Should contain linalg.generic for elementwise add
    bool hasLinalgGeneric = false;
    module->walk([&](mlir::linalg::GenericOp) { hasLinalgGeneric = true; });
    EXPECT_TRUE(hasLinalgGeneric) << "Expected linalg.generic from elementwise Add";
}

TEST_F(LoweringPassTest, MatMulWithVariable)
{
    Graph graph;

    auto w = Variable::Create(Tensor<CPU>({ 1.0f, 0.0f, 0.0f, 1.0f }, { 2, 2 }, DataType::Float32));
    const auto wIdx = graph.AddVariable(std::move(w));

    Subgraph sg;
    const auto x = sg.AddParam(DataType::Float32, { 1, 2 });
    const auto wRef = sg.AddNode(VariableRefNode{ wIdx }, { OutputInfo{ DataType::Float32, { 2, 2 } } });
    const auto y = sg.AddNode(BinaryOpNode{ BinaryOp::MatMul, { x, 0 }, { wRef, 0 } },
                              { OutputInfo{ DataType::Float32, { 1, 2 } } });
    sg.SetResults({ { y, 0 } });
    graph.AddSubgraph(std::move(sg));
    graph.SetForward(0);

    auto module = translateAndLower(graph);
    ASSERT_TRUE(module);
    EXPECT_TRUE(mlir::succeeded(mlir::verify(*module)));
    expectNoLiteNNOps(*module);

    // Should contain linalg.matmul and memref.global for the variable
    bool hasMatmul = false;
    module->walk([&](mlir::linalg::MatmulOp) { hasMatmul = true; });
    EXPECT_TRUE(hasMatmul) << "Expected linalg.matmul from MatMul";

    bool hasGlobal = false;
    module->walk([&](mlir::memref::GlobalOp) { hasGlobal = true; });
    EXPECT_TRUE(hasGlobal) << "Expected memref.global from Variable";
}

TEST_F(LoweringPassTest, CondNode)
{
    Graph graph;

    // Then branch: y = x * 2
    Subgraph thenSg;
    const auto thenX = thenSg.AddParam(DataType::Float32, { 2 });
    auto twoTensor = Tensor<CPU>({ 2.0f, 2.0f }, { 2 }, DataType::Float32);
    const auto thenTwo =
        thenSg.AddNode(ConstantNode{ twoTensor.CopyToDevice(PolymorphicDevice{ CPU{} }) },
                       { OutputInfo{ DataType::Float32, { 2 } } });
    const auto thenMul =
        thenSg.AddNode(BinaryOpNode{ BinaryOp::Multiply, { thenX, 0 }, { thenTwo, 0 } },
                       { OutputInfo{ DataType::Float32, { 2 } } });
    thenSg.SetResults({ { thenMul, 0 } });
    const auto thenId = graph.AddSubgraph(std::move(thenSg));

    // Else branch: y = x + 1
    Subgraph elseSg;
    const auto elseX = elseSg.AddParam(DataType::Float32, { 2 });
    auto oneTensor = Tensor<CPU>({ 1.0f, 1.0f }, { 2 }, DataType::Float32);
    const auto elseOne =
        elseSg.AddNode(ConstantNode{ oneTensor.CopyToDevice(PolymorphicDevice{ CPU{} }) },
                       { OutputInfo{ DataType::Float32, { 2 } } });
    const auto elseAdd =
        elseSg.AddNode(BinaryOpNode{ BinaryOp::Add, { elseX, 0 }, { elseOne, 0 } },
                       { OutputInfo{ DataType::Float32, { 2 } } });
    elseSg.SetResults({ { elseAdd, 0 } });
    const auto elseId = graph.AddSubgraph(std::move(elseSg));

    // Main: cond ? then(x) : else(x)
    Subgraph mainSg;
    const auto cond = mainSg.AddParam(DataType::Bool, { 1 });
    const auto x = mainSg.AddParam(DataType::Float32, { 2 });
    const auto condNode =
        mainSg.AddNode(CondNode{ { cond, 0 }, thenId, elseId, { { x, 0 } } },
                       { OutputInfo{ DataType::Float32, { 2 } } });
    mainSg.SetResults({ { condNode, 0 } });
    const auto mainId = graph.AddSubgraph(std::move(mainSg));
    graph.SetForward(mainId);

    auto module = translateAndLower(graph);
    ASSERT_TRUE(module);
    EXPECT_TRUE(mlir::succeeded(mlir::verify(*module)));
    expectNoLiteNNOps(*module);

    // Should contain scf.if
    bool hasScfIf = false;
    module->walk([&](mlir::scf::IfOp) { hasScfIf = true; });
    EXPECT_TRUE(hasScfIf) << "Expected scf.if from CondNode";
}

TEST_F(LoweringPassTest, WhileNode)
{
    Graph graph;

    // Cond: x < 100
    Subgraph condSg;
    const auto condX = condSg.AddParam(DataType::Float32, { 1 });
    auto hundredTensor = Tensor<CPU>({ 100.0f }, { 1 }, DataType::Float32);
    const auto hundred =
        condSg.AddNode(ConstantNode{ hundredTensor.CopyToDevice(PolymorphicDevice{ CPU{} }) },
                       { OutputInfo{ DataType::Float32, { 1 } } });
    const auto lt = condSg.AddNode(BinaryOpNode{ BinaryOp::Less, { condX, 0 }, { hundred, 0 } },
                                   { OutputInfo{ DataType::Bool, { 1 } } });
    condSg.SetResults({ { lt, 0 } });
    const auto condId = graph.AddSubgraph(std::move(condSg));

    // Body: x = x * 2
    Subgraph bodySg;
    const auto bodyX = bodySg.AddParam(DataType::Float32, { 1 });
    auto twoTensor = Tensor<CPU>({ 2.0f }, { 1 }, DataType::Float32);
    const auto two =
        bodySg.AddNode(ConstantNode{ twoTensor.CopyToDevice(PolymorphicDevice{ CPU{} }) },
                       { OutputInfo{ DataType::Float32, { 1 } } });
    const auto mul = bodySg.AddNode(BinaryOpNode{ BinaryOp::Multiply, { bodyX, 0 }, { two, 0 } },
                                    { OutputInfo{ DataType::Float32, { 1 } } });
    bodySg.SetResults({ { mul, 0 } });
    const auto bodyId = graph.AddSubgraph(std::move(bodySg));

    // Main: while(x < 100) x = x * 2
    Subgraph mainSg;
    const auto initX = mainSg.AddParam(DataType::Float32, { 1 });
    const auto whileNode =
        mainSg.AddNode(WhileNode{ condId, bodyId, { { initX, 0 } } },
                       { OutputInfo{ DataType::Float32, { 1 } } });
    mainSg.SetResults({ { whileNode, 0 } });
    const auto mainId = graph.AddSubgraph(std::move(mainSg));
    graph.SetForward(mainId);

    auto module = translateAndLower(graph);
    ASSERT_TRUE(module);
    EXPECT_TRUE(mlir::succeeded(mlir::verify(*module)));
    expectNoLiteNNOps(*module);

    // Should contain scf.while
    bool hasScfWhile = false;
    module->walk([&](mlir::scf::WhileOp) { hasScfWhile = true; });
    EXPECT_TRUE(hasScfWhile) << "Expected scf.while from WhileNode";
}

} // namespace
