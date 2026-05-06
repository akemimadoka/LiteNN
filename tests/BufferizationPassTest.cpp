#include <gtest/gtest.h>

#include <LiteNN.h>

#include "Dialect/LiteNNDialect.h"
#include "Dialect/LiteNNOps.h"
#include "Translation/GraphToMLIR.h"
#include "Pass/LowerLiteNNPass.h"
#include "Pass/BufferizationPipeline.h"

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

class BufferizationPassTest : public ::testing::Test
{
protected:
    void SetUp() override
    {
        ctx_.disableMultithreading();

        mlir::DialectRegistry registry;
        litenn::registerBufferizationModels(registry);
        ctx_.appendDialectRegistry(registry);
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

    mlir::OwningOpRef<mlir::ModuleOp> translateLowerAndBufferize(const Graph& graph)
    {
        auto module = litenn::translateGraphToMLIR(graph, ctx_);
        if (!module)
            return {};

        mlir::PassManager pm(&ctx_);
        pm.addPass(litenn::createLowerLiteNNPass());
        litenn::addBufferizationPipeline(pm);
        if (mlir::failed(pm.run(*module)))
            return {};

        return module;
    }

    static void expectNoTensorOrLinalgOps(mlir::ModuleOp mod)
    {
        // After bufferization, tensor-typed values should be gone (replaced by
        // memref). linalg.generic with memref operands is acceptable — full
        // linalg-to-loops lowering is Step 4.
        bool foundTensor = false;
        mod->walk([&](mlir::Operation* op) {
            for (auto t : op->getResultTypes())
                if (llvm::isa<mlir::RankedTensorType>(t))
                    foundTensor = true;
            for (auto t : op->getOperandTypes())
                if (llvm::isa<mlir::RankedTensorType>(t))
                    foundTensor = true;
        });
        EXPECT_FALSE(foundTensor) << "Module still contains tensor-typed values after bufferization";
    }

    static void expectHasMemRefAlloc(mlir::ModuleOp mod)
    {
        bool found = false;
        mod->walk([&](mlir::memref::AllocOp) { found = true; });
        mod->walk([&](mlir::memref::AllocaOp) { found = true; });
        EXPECT_TRUE(found) << "Expected memref.alloc/alloca from bufferized tensors";
    }

    mlir::MLIRContext ctx_;
};

TEST_F(BufferizationPassTest, SimpleAdd)
{
    Graph graph;
    Subgraph sg;
    const auto a = sg.AddParam(DataType::Float32, { 2, 2 });
    const auto b = sg.AddParam(DataType::Float32, { 2, 2 });
    const auto y = sg.AddNode(BinaryOpNode{ BinaryOp::Add, { a, 0 }, { b, 0 } },
                              { OutputInfo{ DataType::Float32, { 2, 2 } } });
    sg.SetResults({ { y, 0 } });
    graph.AddSubgraph(std::move(sg));
    graph.SetForward(0);

    auto module = translateLowerAndBufferize(graph);
    ASSERT_TRUE(module);
    EXPECT_TRUE(mlir::succeeded(mlir::verify(*module)));
    expectNoTensorOrLinalgOps(*module);
    expectHasMemRefAlloc(*module);
}

TEST_F(BufferizationPassTest, MatMulWithVariable)
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

    auto module = translateLowerAndBufferize(graph);
    ASSERT_TRUE(module);
    EXPECT_TRUE(mlir::succeeded(mlir::verify(*module)));
    expectNoTensorOrLinalgOps(*module);

    bool hasGlobal = false;
    module->walk([&](mlir::memref::GlobalOp) { hasGlobal = true; });
    EXPECT_TRUE(hasGlobal) << "Expected memref.global from Variable after bufferization";
}

TEST_F(BufferizationPassTest, WhileNode)
{
    Graph graph;

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

    Subgraph mainSg;
    const auto initX = mainSg.AddParam(DataType::Float32, { 1 });
    const auto whileNode =
        mainSg.AddNode(WhileNode{ condId, bodyId, { { initX, 0 } } },
                       { OutputInfo{ DataType::Float32, { 1 } } });
    mainSg.SetResults({ { whileNode, 0 } });
    graph.AddSubgraph(std::move(mainSg));
    graph.SetForward(2);

    auto module = translateLowerAndBufferize(graph);
    ASSERT_TRUE(module);
    EXPECT_TRUE(mlir::succeeded(mlir::verify(*module)));
    expectNoTensorOrLinalgOps(*module);

    bool hasScfWhile = false;
    module->walk([&](mlir::scf::WhileOp) { hasScfWhile = true; });
    EXPECT_TRUE(hasScfWhile) << "Expected scf.while to survive bufferization";
}

} // namespace
