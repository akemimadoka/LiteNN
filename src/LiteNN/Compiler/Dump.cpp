#include <LiteNN/Compiler/Dump.h>

#include "Dialect/LiteNNDialect.h"
#include "Pass/BufferizationPipeline.h"
#include "Pass/LLVMCodegenPipeline.h"
#include "Pass/LowerLiteNNPass.h"
#include "Translation/GraphToMLIR.h"

#include <LiteNN/Validation/GraphValidator.h>

#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Bufferization/IR/Bufferization.h"
#include "mlir/Dialect/ControlFlow/IR/ControlFlowOps.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/Dialect/Math/IR/Math.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/Dialect/Vector/IR/VectorOps.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/IR/Verifier.h"
#include "mlir/Pass/PassManager.h"
#include "llvm/Support/raw_ostream.h"

#include <format>
#include <stdexcept>
#include <string_view>

namespace LiteNN::Debug
{
namespace
{
	template <typename Formatter>
	std::string JoinIndexed(std::size_t count, std::string_view separator, Formatter&& formatter)
	{
		std::string result;
		for (std::size_t index = 0; index < count; ++index)
		{
			if (index != 0)
			{
				result += separator;
			}
			result += formatter(index);
		}
		return result;
	}

	std::string FormatCompiledSpec(const CompiledTensorSpec& spec, std::string_view fallbackPrefix, std::size_t index)
	{
		const auto name = spec.name.empty() ? std::format("{}{}", fallbackPrefix, index) : spec.name;
		return std::format("{}: {}", name, Validation::FormatInfo(spec.dtype, spec.shape));
	}

	std::string_view FormatBackend(CompiledModuleBackend backend)
	{
		switch (backend)
		{
			case CompiledModuleBackend::CPUNative:
				return "cpu_native";
			case CompiledModuleBackend::CUDANative:
				return "cuda_native";
		}
		return "unknown";
	}

	std::string PrintModule(mlir::ModuleOp module)
	{
		std::string text;
		llvm::raw_string_ostream stream(text);
		module->print(stream);
		stream.flush();
		return text;
	}

	void SetupDumpMLIRContext(mlir::MLIRContext& ctx)
	{
		ctx.disableMultithreading();

		mlir::DialectRegistry registry;
		litenn::registerBufferizationModels(registry);
		litenn::registerLLVMTranslations(registry);

		ctx.appendDialectRegistry(registry);
		ctx.loadDialect<litenn::LiteNNDialect, mlir::arith::ArithDialect, mlir::bufferization::BufferizationDialect,
		                mlir::cf::ControlFlowDialect, mlir::func::FuncDialect, mlir::linalg::LinalgDialect,
		                mlir::LLVM::LLVMDialect, mlir::math::MathDialect, mlir::memref::MemRefDialect,
		                mlir::scf::SCFDialect, mlir::tensor::TensorDialect, mlir::vector::VectorDialect>();
	}

	mlir::OwningOpRef<mlir::ModuleOp> CreateTranslatedModule(const Graph& graph, mlir::MLIRContext& ctx)
	{
		auto module = litenn::translateGraphToMLIR(graph, ctx);
		if (!module)
		{
			throw std::runtime_error("Failed to translate LiteNN graph to MLIR");
		}
		if (mlir::failed(mlir::verify(*module)))
		{
			throw std::runtime_error("LiteNN input MLIR module verification failed");
		}
		return module;
	}

	std::string DumpCompiledModuleMetadataImpl(std::span<const std::byte> rodata,
	                                          std::span<const std::byte> instructions,
	                                          std::span<const CompiledTensorSpec> inputSpecs,
	                                          std::span<const CompiledTensorSpec> outputSpecs,
	                                          CompiledModuleBackend backend)
	{
		std::string out = "compiled_module {\n";
		out += std::format("  backend = {}\n", FormatBackend(backend));
		out += std::format("  rodata_size = {}\n", rodata.size());
		out += std::format("  instruction_size = {}\n", instructions.size());
		out += std::format("  inputs = [{}]\n", JoinIndexed(inputSpecs.size(), ", ", [&](std::size_t index) {
			return FormatCompiledSpec(inputSpecs[index], "input", index);
		}));
		out += std::format("  outputs = [{}]\n", JoinIndexed(outputSpecs.size(), ", ", [&](std::size_t index) {
			return FormatCompiledSpec(outputSpecs[index], "output", index);
		}));
		out += "}\n";
		return out;
	}
} // namespace

	std::string DumpMLIR(const Graph& graph, MLIRDumpStage stage)
	{
		mlir::MLIRContext ctx;
		SetupDumpMLIRContext(ctx);

		auto module = CreateTranslatedModule(graph, ctx);
		if (stage == MLIRDumpStage::InputDialect)
		{
			return PrintModule(*module);
		}

		mlir::PassManager pm(&ctx);
		pm.addPass(litenn::createLowerLiteNNPass());
		if (stage == MLIRDumpStage::AfterBufferization || stage == MLIRDumpStage::AfterLLVMCodegen)
		{
			litenn::addBufferizationPipeline(pm);
		}
		if (stage == MLIRDumpStage::AfterLLVMCodegen)
		{
			litenn::addLLVMCodegenPipeline(pm);
		}

		if (mlir::failed(pm.run(*module)))
		{
			throw std::runtime_error("LiteNN MLIR dump pipeline failed");
		}
		if (mlir::failed(mlir::verify(*module)))
		{
			throw std::runtime_error("LiteNN MLIR dump verification failed");
		}
		return PrintModule(*module);
	}

	std::string DumpCompiledModuleMetadata(const CompiledModuleArtifact& artifact)
	{
		return DumpCompiledModuleMetadataImpl(artifact.Rodata(), artifact.Instructions(), artifact.InputSpecs(),
		                                    artifact.OutputSpecs(), artifact.Backend());
	}

	std::string DumpCompiledModuleMetadata(const CompiledModule<CPU>& module)
	{
		return DumpCompiledModuleMetadataImpl(module.Rodata(), module.Instructions(), module.InputSpecs(),
		                                    module.OutputSpecs(), module.Backend());
	}
#ifdef LITENN_ENABLE_CUDA
	std::string DumpCompiledModuleMetadata(const CompiledModule<CUDA>& module)
	{
		return DumpCompiledModuleMetadataImpl(module.Rodata(), module.Instructions(), module.InputSpecs(),
		                                    module.OutputSpecs(), module.Backend());
	}
#endif
} // namespace LiteNN::Debug
