#include <gtest/gtest.h>

#include <LiteNN.h>

#include "Dialect/LiteNNDialect.h"
#include "Dialect/LiteNNOps.h"
#include "Translation/GraphToMLIR.h"
#include "Pass/LowerLiteNNPass.h"
#include "Pass/BufferizationPipeline.h"
#include "Pass/LLVMCodegenPipeline.h"

#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Bufferization/IR/Bufferization.h"
#include "mlir/Dialect/ControlFlow/IR/ControlFlow.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/Dialect/Math/IR/Math.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/IR/Verifier.h"
#include "mlir/Pass/PassManager.h"

#include "llvm/IR/LLVMContext.h"
#include "llvm/IR/Module.h"

using namespace LiteNN;

namespace
{

class LLVMCodegenPassTest : public ::testing::Test
{
protected:
    void SetUp() override
    {
        ctx_.disableMultithreading();

        mlir::DialectRegistry registry;
        litenn::registerBufferizationModels(registry);
        litenn::registerLLVMTranslations(registry);
        ctx_.appendDialectRegistry(registry);
        ctx_.loadDialect<
            litenn::LiteNNDialect,
            mlir::arith::ArithDialect,
            mlir::bufferization::BufferizationDialect,
            mlir::cf::ControlFlowDialect,
            mlir::func::FuncDialect,
            mlir::linalg::LinalgDialect,
            mlir::LLVM::LLVMDialect,
            mlir::math::MathDialect,
            mlir::memref::MemRefDialect,
            mlir::scf::SCFDialect,
            mlir::tensor::TensorDialect>();
    }

    mlir::OwningOpRef<mlir::ModuleOp> fullPipeline(const Graph& graph)
    {
        auto module = litenn::translateGraphToMLIR(graph, ctx_);
        if (!module)
            return {};

        mlir::PassManager pm(&ctx_);
        pm.addPass(litenn::createLowerLiteNNPass());
        litenn::addBufferizationPipeline(pm);
        litenn::addLLVMCodegenPipeline(pm);
        if (mlir::failed(pm.run(*module)))
            return {};

        return module;
    }

    static void expectNoUnrealizedCasts(mlir::ModuleOp mod)
    {
        bool found = false;
        mod->walk([&](mlir::UnrealizedConversionCastOp) { found = true; });
        EXPECT_FALSE(found) << "Unrealized conversion casts remain after codegen";
    }

    mlir::MLIRContext ctx_;
};

TEST_F(LLVMCodegenPassTest, SimpleAdd)
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

    auto module = fullPipeline(graph);
    ASSERT_TRUE(module);
    EXPECT_TRUE(mlir::succeeded(mlir::verify(*module)));
    expectNoUnrealizedCasts(*module);

    llvm::LLVMContext llvmCtx;
    auto llvmModule = litenn::translateToLLVMIR(*module, llvmCtx);
    ASSERT_NE(llvmModule, nullptr);
}

TEST_F(LLVMCodegenPassTest, MatMulWithVariable)
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

    auto module = fullPipeline(graph);
    ASSERT_TRUE(module);
    EXPECT_TRUE(mlir::succeeded(mlir::verify(*module)));
    expectNoUnrealizedCasts(*module);

    llvm::LLVMContext llvmCtx;
    auto llvmModule = litenn::translateToLLVMIR(*module, llvmCtx);
    ASSERT_NE(llvmModule, nullptr);
}

TEST_F(LLVMCodegenPassTest, WhileNode)
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

    auto module = fullPipeline(graph);
    ASSERT_TRUE(module);
    EXPECT_TRUE(mlir::succeeded(mlir::verify(*module)));
    expectNoUnrealizedCasts(*module);

    llvm::LLVMContext llvmCtx;
    auto llvmModule = litenn::translateToLLVMIR(*module, llvmCtx);
    ASSERT_NE(llvmModule, nullptr);
}

} // namespace
