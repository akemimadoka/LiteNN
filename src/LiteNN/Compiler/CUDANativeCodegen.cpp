#include "CUDANativeCodegen.h"

#include "mlir/Conversion/Passes.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/GPU/IR/GPUDialect.h"
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/Dialect/LLVMIR/NVVMDialect.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/DialectRegistry.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/IR/Verifier.h"
#include "mlir/Pass/PassManager.h"
#include "mlir/Target/LLVMIR/Dialect/Builtin/BuiltinToLLVMIRTranslation.h"
#include "mlir/Target/LLVMIR/Dialect/LLVMIR/LLVMToLLVMIRTranslation.h"
#include "mlir/Target/LLVMIR/Dialect/NVVM/NVVMToLLVMIRTranslation.h"
#include "mlir/Target/LLVMIR/Export.h"
#include "llvm/ADT/APInt.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/IR/LLVMContext.h"
#include "llvm/IR/LegacyPassManager.h"
#include "llvm/IR/Module.h"
#include "llvm/IR/Verifier.h"
#include "llvm/MC/TargetRegistry.h"
#include "llvm/Support/CodeGen.h"
#include "llvm/Support/TargetSelect.h"
#include "llvm/Support/raw_ostream.h"
#include "llvm/Target/TargetMachine.h"
#include "llvm/TargetParser/Triple.h"

#include <cstdint>
#include <cstring>
#include <format>
#include <limits>
#include <mutex>
#include <optional>
#include <stdexcept>
#include <utility>

namespace LiteNN
{
namespace
{
	constexpr std::string_view kNVPTXTriple = "nvptx64-nvidia-cuda";
	constexpr std::string_view kNVPTXChip = "sm_30";
	constexpr std::string_view kNVPTXFeatures = "+ptx64";

	void InitializeNVPTXLLVM()
	{
		static std::once_flag once;
		std::call_once(once, [] {
			LLVMInitializeNVPTXTargetInfo();
			LLVMInitializeNVPTXTarget();
			LLVMInitializeNVPTXTargetMC();
			LLVMInitializeNVPTXAsmPrinter();
		});
	}

	std::unique_ptr<llvm::TargetMachine> CreateNVPTXTargetMachine()
	{
		InitializeNVPTXLLVM();

		std::string error;
		const auto* target = llvm::TargetRegistry::lookupTarget(llvm::Triple(std::string(kNVPTXTriple)), error);
		if (!target)
		{
			throw std::runtime_error("Failed to lookup NVPTX LLVM target: " + error);
		}

		llvm::TargetOptions options;
		auto targetMachine = std::unique_ptr<llvm::TargetMachine>(target->createTargetMachine(
		    llvm::Triple(std::string(kNVPTXTriple)), std::string(kNVPTXChip), std::string(kNVPTXFeatures), options,
		    std::nullopt, std::nullopt, llvm::CodeGenOptLevel::Aggressive));
		if (!targetMachine)
		{
			throw std::runtime_error("Failed to create NVPTX LLVM target machine");
		}
		return targetMachine;
	}

	std::string EmitNVPTXAssembly(llvm::Module& module)
	{
		auto targetMachine = CreateNVPTXTargetMachine();
		module.setTargetTriple(llvm::Triple(std::string(kNVPTXTriple)));
		module.setDataLayout(targetMachine->createDataLayout());

		std::string verifyMessage;
		llvm::raw_string_ostream verifyStream(verifyMessage);
		if (llvm::verifyModule(module, &verifyStream))
		{
			throw std::runtime_error("Generated NVPTX LLVM module verification failed: " + verifyStream.str());
		}

		llvm::SmallVector<char, 0> buffer;
		llvm::raw_svector_ostream stream(buffer);
		llvm::legacy::PassManager passManager;
		if (targetMachine->addPassesToEmitFile(passManager, stream, nullptr, llvm::CodeGenFileType::AssemblyFile))
		{
			throw std::runtime_error("NVPTX target cannot emit PTX assembly");
		}
		passManager.run(module);
		return std::string(buffer.begin(), buffer.end());
	}

	llvm::StringRef ToLLVMStringRef(std::string_view value)
	{
		return llvm::StringRef(value.data(), value.size());
	}

	std::optional<std::vector<std::uint32_t>> ContiguousStridesU32(std::span<const std::size_t> shape)
	{
		std::vector<std::uint32_t> strides(shape.size());
		std::uint64_t stride = 1;
		for (auto i = shape.size(); i > 0; --i)
		{
			if (stride > std::numeric_limits<std::uint32_t>::max())
			{
				return std::nullopt;
			}
			strides[i - 1] = static_cast<std::uint32_t>(stride);
			stride *= shape[i - 1];
		}
		return strides;
	}

	void ValidateShapeDimsU32(std::span<const std::size_t> shape)
	{
		for (const auto dim : shape)
		{
			if (dim > std::numeric_limits<std::uint32_t>::max())
			{
				throw std::runtime_error("CUDA native MLIR broadcast shape dimension is too large for u32 indexing");
			}
		}
	}

	struct CUDANativeBroadcastStrides
	{
		std::vector<std::uint32_t> output;
		std::vector<std::uint32_t> lhs;
		std::vector<std::uint32_t> rhs;
	};

	CUDANativeBroadcastStrides ComputeBroadcastStrides(
	    const CUDANativeBroadcastBinaryF32CodegenSpec& spec)
	{
		if (spec.outputShape.size() != spec.lhsShape.size() || spec.outputShape.size() != spec.rhsShape.size())
		{
			throw std::runtime_error("CUDA native MLIR broadcast codegen requires same-rank shapes");
		}
		ValidateShapeDimsU32(spec.outputShape);
		ValidateShapeDimsU32(spec.lhsShape);
		ValidateShapeDimsU32(spec.rhsShape);
		for (std::size_t i = 0; i < spec.outputShape.size(); ++i)
		{
			if ((spec.lhsShape[i] != 1 && spec.lhsShape[i] != spec.outputShape[i]) ||
			    (spec.rhsShape[i] != 1 && spec.rhsShape[i] != spec.outputShape[i]))
			{
				throw std::runtime_error("CUDA native MLIR broadcast codegen received incompatible shapes");
			}
		}

		auto outputStrides = ContiguousStridesU32(spec.outputShape);
		auto lhsStrides = ContiguousStridesU32(spec.lhsShape);
		auto rhsStrides = ContiguousStridesU32(spec.rhsShape);
		if (!outputStrides || !lhsStrides || !rhsStrides)
		{
			throw std::runtime_error("CUDA native MLIR broadcast shape is too large for u32 indexing");
		}
		return CUDANativeBroadcastStrides{ std::move(*outputStrides), std::move(*lhsStrides), std::move(*rhsStrides) };
	}

	struct CUDANativeMLIRKernelModule
	{
		mlir::OwningOpRef<mlir::ModuleOp> module;
		mlir::gpu::GPUModuleOp gpuModule;
	};

	struct CUDANativeLinearKernelBlocks
	{
		mlir::Block* entry = nullptr;
		mlir::Block* body = nullptr;
		mlir::Block* done = nullptr;
		mlir::Value index32;
	};

	struct CUDANativeBroadcastElementPointers
	{
		mlir::Value lhs;
		mlir::Value rhs;
	};

	class CUDANativeMLIRKernelBuilder
	{
	public:
		explicit CUDANativeMLIRKernelBuilder(mlir::MLIRContext& context)
		    : context_(context),
		      builder_(&context),
		      loc_(builder_.getUnknownLoc()),
		      indexType_(builder_.getIndexType()),
		      f32Type_(builder_.getF32Type()),
		      i32Type_(builder_.getI32Type()),
		      ptrType_(mlir::LLVM::LLVMPointerType::get(&context))
		{
		}

		mlir::OwningOpRef<mlir::ModuleOp> BuildUnaryF32(UnaryOp op)
		{
			auto kernelModule = CreateKernelModule();
			llvm::SmallVector<mlir::Type, 4> argTypes{ ptrType_, ptrType_, i32Type_ };
			auto func = CreateKernelFunc(kernelModule.gpuModule, CUDANativeUnaryF32KernelName(op), argTypes);
			auto blocks = EmitLinearIndexGuard(func, 2);

			builder_.setInsertionPointToStart(blocks.body);
			auto out = blocks.entry->getArgument(0);
			auto in = blocks.entry->getArgument(1);
			auto value = EmitLoadF32(EmitF32GEP(in, blocks.index32));
			auto result = EmitUnaryF32Result(op, value);
			EmitStoreF32(result, EmitF32GEP(out, blocks.index32));
			FinishLinearKernel(blocks);

			return FinalizeModule(std::move(kernelModule.module));
		}

		mlir::OwningOpRef<mlir::ModuleOp> BuildBinaryF32(BinaryOp op)
		{
			auto kernelModule = CreateKernelModule();
			llvm::SmallVector<mlir::Type, 4> argTypes{ ptrType_, ptrType_, ptrType_, i32Type_ };
			auto func = CreateKernelFunc(kernelModule.gpuModule, CUDANativeBinaryF32KernelName(op), argTypes);
			auto blocks = EmitLinearIndexGuard(func, 3);

			builder_.setInsertionPointToStart(blocks.body);
			auto out = blocks.entry->getArgument(0);
			auto lhs = blocks.entry->getArgument(1);
			auto rhs = blocks.entry->getArgument(2);
			auto lhsValue = EmitLoadF32(EmitF32GEP(lhs, blocks.index32));
			auto rhsValue = EmitLoadF32(EmitF32GEP(rhs, blocks.index32));
			auto result = EmitBinaryF32Result(op, lhsValue, rhsValue);
			EmitStoreF32(result, EmitF32GEP(out, blocks.index32));
			FinishLinearKernel(blocks);

			return FinalizeModule(std::move(kernelModule.module));
		}

		mlir::OwningOpRef<mlir::ModuleOp> BuildBinaryBroadcastF32(
		    const CUDANativeBroadcastBinaryF32CodegenSpec& spec)
		{
			const auto strides = ComputeBroadcastStrides(spec);

			auto kernelModule = CreateKernelModule();
			llvm::SmallVector<mlir::Type, 4> argTypes{ ptrType_, ptrType_, ptrType_, i32Type_ };
			auto func = CreateKernelFunc(kernelModule.gpuModule, CUDANativeBinaryF32KernelName(spec.op, true), argTypes);
			auto blocks = EmitLinearIndexGuard(func, 3);

			builder_.setInsertionPointToStart(blocks.body);
			auto out = blocks.entry->getArgument(0);
			auto lhs = blocks.entry->getArgument(1);
			auto rhs = blocks.entry->getArgument(2);
			auto operands = EmitBroadcastElementPointers(spec, strides, lhs, rhs, blocks.index32);
			auto lhsValue = EmitLoadF32(operands.lhs);
			auto rhsValue = EmitLoadF32(operands.rhs);
			auto result = EmitBinaryF32Result(spec.op, lhsValue, rhsValue);
			EmitStoreF32(result, EmitF32GEP(out, blocks.index32));
			FinishLinearKernel(blocks);

			return FinalizeModule(std::move(kernelModule.module));
		}

	private:
		CUDANativeMLIRKernelModule CreateKernelModule()
		{
			auto module = mlir::ModuleOp::create(loc_);
			builder_.setInsertionPointToStart(module.getBody());

			auto target = mlir::NVVM::NVVMTargetAttr::get(
			    &context_, 2, ToLLVMStringRef(kNVPTXTriple), ToLLVMStringRef(kNVPTXChip),
			    ToLLVMStringRef(kNVPTXFeatures), nullptr, nullptr, false);
			llvm::SmallVector<mlir::Attribute, 1> targets{ target };
			auto gpuModule = builder_.create<mlir::gpu::GPUModuleOp>(
			    loc_, "litenn_cuda_kernels", targets);
			return CUDANativeMLIRKernelModule{ mlir::OwningOpRef<mlir::ModuleOp>(module), gpuModule };
		}

		mlir::gpu::GPUFuncOp CreateKernelFunc(
		    mlir::gpu::GPUModuleOp gpuModule, std::string_view name, llvm::ArrayRef<mlir::Type> argTypes)
		{
			builder_.setInsertionPointToStart(gpuModule.getBody());
			auto funcType = builder_.getFunctionType(argTypes, mlir::TypeRange{});
			auto func = builder_.create<mlir::gpu::GPUFuncOp>(loc_, ToLLVMStringRef(name), funcType);
			func->setAttr(mlir::gpu::GPUDialect::getKernelFuncAttrName(), builder_.getUnitAttr());
			return func;
		}

		CUDANativeLinearKernelBlocks EmitLinearIndexGuard(mlir::gpu::GPUFuncOp func, unsigned countArgument)
		{
			auto* entry = &func.getBody().front();
			auto* body = builder_.createBlock(&func.getBody());
			auto* done = builder_.createBlock(&func.getBody());

			builder_.setInsertionPointToStart(entry);
			auto tid = builder_.create<mlir::gpu::ThreadIdOp>(loc_, indexType_, mlir::gpu::Dimension::x).getResult();
			auto blockId = builder_.create<mlir::gpu::BlockIdOp>(loc_, indexType_, mlir::gpu::Dimension::x).getResult();
			auto blockDim = builder_.create<mlir::gpu::BlockDimOp>(loc_, indexType_, mlir::gpu::Dimension::x).getResult();
			auto base = builder_.create<mlir::arith::MulIOp>(loc_, blockId, blockDim).getResult();
			auto index = builder_.create<mlir::arith::AddIOp>(loc_, base, tid).getResult();
			auto index32 = builder_.create<mlir::arith::IndexCastOp>(loc_, i32Type_, index).getResult();
			auto doneCond = builder_
			                    .create<mlir::LLVM::ICmpOp>(
			                        loc_, mlir::LLVM::ICmpPredicate::uge, index32, entry->getArgument(countArgument))
			                    .getResult();
			builder_.create<mlir::LLVM::CondBrOp>(loc_, doneCond, done, body);

			return CUDANativeLinearKernelBlocks{ entry, body, done, index32 };
		}

		void FinishLinearKernel(const CUDANativeLinearKernelBlocks& blocks)
		{
			builder_.create<mlir::LLVM::BrOp>(loc_, blocks.done);
			builder_.setInsertionPointToStart(blocks.done);
			builder_.create<mlir::gpu::ReturnOp>(loc_);
		}

		mlir::OwningOpRef<mlir::ModuleOp> FinalizeModule(mlir::OwningOpRef<mlir::ModuleOp> module)
		{
			if (mlir::failed(mlir::verify(module.get())))
			{
				throw std::runtime_error("Generated CUDA native MLIR GPU module verification failed");
			}
			return std::move(module);
		}

		mlir::Value EmitI32Constant(std::uint32_t value)
		{
			return builder_
			    .create<mlir::LLVM::ConstantOp>(
			        loc_, i32Type_, builder_.getIntegerAttr(i32Type_, llvm::APInt(32, value)))
			    .getResult();
		}

		mlir::Value EmitI32Add(mlir::Value lhs, mlir::Value rhs)
		{
			return builder_.create<mlir::LLVM::AddOp>(loc_, i32Type_, mlir::ValueRange{ lhs, rhs }).getResult();
		}

		mlir::Value EmitI32Mul(mlir::Value lhs, mlir::Value rhs)
		{
			return builder_.create<mlir::LLVM::MulOp>(loc_, i32Type_, mlir::ValueRange{ lhs, rhs }).getResult();
		}

		mlir::Value EmitI32UDiv(mlir::Value lhs, mlir::Value rhs)
		{
			return builder_.create<mlir::LLVM::UDivOp>(loc_, i32Type_, mlir::ValueRange{ lhs, rhs }).getResult();
		}

		mlir::Value EmitI32URem(mlir::Value lhs, mlir::Value rhs)
		{
			return builder_.create<mlir::LLVM::URemOp>(loc_, i32Type_, mlir::ValueRange{ lhs, rhs }).getResult();
		}

		mlir::Value EmitF32GEP(mlir::Value base, mlir::Value index)
		{
			return builder_
			    .create<mlir::LLVM::GEPOp>(loc_, ptrType_, f32Type_, base, mlir::ValueRange{ index })
			    .getResult();
		}

		mlir::Value EmitLoadF32(mlir::Value ptr)
		{
			return builder_.create<mlir::LLVM::LoadOp>(loc_, f32Type_, ptr).getResult();
		}

		void EmitStoreF32(mlir::Value value, mlir::Value ptr)
		{
			builder_.create<mlir::LLVM::StoreOp>(loc_, value, ptr);
		}

		mlir::Value EmitUnaryF32Result(UnaryOp op, mlir::Value value)
		{
			switch (op)
			{
			case UnaryOp::Negate:
				return builder_.create<mlir::LLVM::FNegOp>(loc_, f32Type_, mlir::ValueRange{ value }).getResult();
			case UnaryOp::Abs:
				return builder_
				    .create<mlir::LLVM::CallIntrinsicOp>(
				        loc_, f32Type_, builder_.getStringAttr("llvm.nvvm.fabs.ftz.f"), mlir::ValueRange{ value })
				    .getResult(0);
			case UnaryOp::Sqrt:
				return builder_
				    .create<mlir::LLVM::CallIntrinsicOp>(
				        loc_, f32Type_, builder_.getStringAttr("llvm.nvvm.sqrt.rn.ftz.f"), mlir::ValueRange{ value })
				    .getResult(0);
			default:
				throw std::runtime_error("Unsupported MLIR NVPTX CUDA native unary op");
			}
		}

		mlir::Value EmitBinaryF32Result(BinaryOp op, mlir::Value lhs, mlir::Value rhs)
		{
			switch (op)
			{
			case BinaryOp::Add:
				return builder_.create<mlir::LLVM::FAddOp>(loc_, f32Type_, mlir::ValueRange{ lhs, rhs }).getResult();
			case BinaryOp::Subtract:
				return builder_.create<mlir::LLVM::FSubOp>(loc_, f32Type_, mlir::ValueRange{ lhs, rhs }).getResult();
			case BinaryOp::Multiply:
				return builder_.create<mlir::LLVM::FMulOp>(loc_, f32Type_, mlir::ValueRange{ lhs, rhs }).getResult();
			case BinaryOp::Divide:
				return builder_.create<mlir::LLVM::FDivOp>(loc_, f32Type_, mlir::ValueRange{ lhs, rhs }).getResult();
			default:
				throw std::runtime_error("Unsupported MLIR NVPTX CUDA native binary op");
			}
		}

		CUDANativeBroadcastElementPointers EmitBroadcastElementPointers(
		    const CUDANativeBroadcastBinaryF32CodegenSpec& spec, const CUDANativeBroadcastStrides& strides,
		    mlir::Value lhs, mlir::Value rhs, mlir::Value outputIndex)
		{
			auto lhsOffset = EmitI32Constant(0);
			auto rhsOffset = EmitI32Constant(0);

			for (std::size_t i = 0; i < spec.outputShape.size(); ++i)
			{
				auto outStride = EmitI32Constant(strides.output[i]);
				auto outDim = EmitI32Constant(static_cast<std::uint32_t>(spec.outputShape[i]));
				auto coord = EmitI32URem(EmitI32UDiv(outputIndex, outStride), outDim);

				if (spec.lhsShape[i] != 1)
				{
					auto lhsTerm = EmitI32Mul(coord, EmitI32Constant(strides.lhs[i]));
					lhsOffset = EmitI32Add(lhsOffset, lhsTerm);
				}
				if (spec.rhsShape[i] != 1)
				{
					auto rhsTerm = EmitI32Mul(coord, EmitI32Constant(strides.rhs[i]));
					rhsOffset = EmitI32Add(rhsOffset, rhsTerm);
				}
			}

			return CUDANativeBroadcastElementPointers{ EmitF32GEP(lhs, lhsOffset), EmitF32GEP(rhs, rhsOffset) };
		}

		mlir::MLIRContext& context_;
		mlir::OpBuilder builder_;
		mlir::Location loc_;
		mlir::Type indexType_;
		mlir::Type f32Type_;
		mlir::Type i32Type_;
		mlir::Type ptrType_;
	};

	mlir::OwningOpRef<mlir::ModuleOp> ExtractLLVMKernelModule(mlir::ModuleOp loweredModule)
	{
		mlir::OpBuilder builder(loweredModule.getContext());
		auto llvmKernelModule = mlir::ModuleOp::create(loweredModule.getLoc());
		llvmKernelModule->setAttr("llvm.target_triple", builder.getStringAttr(kNVPTXTriple));
		builder.setInsertionPointToEnd(llvmKernelModule.getBody());

		std::size_t clonedKernelCount = 0;
		loweredModule.walk([&](mlir::LLVM::LLVMFuncOp func) {
			auto* cloned = builder.clone(*func.getOperation());
			cloned->removeAttr("gpu.kernel");
			++clonedKernelCount;
		});
		if (clonedKernelCount == 0)
		{
			throw std::runtime_error("MLIR GPU to NVVM lowering produced no LLVM kernel functions");
		}
		if (mlir::failed(mlir::verify(llvmKernelModule)))
		{
			throw std::runtime_error("Extracted MLIR NVVM kernel module verification failed");
		}
		return mlir::OwningOpRef<mlir::ModuleOp>(llvmKernelModule);
	}

	mlir::DialectRegistry CreateCUDANativeMLIRRegistry()
	{
		mlir::DialectRegistry registry;
		registry.insert<mlir::arith::ArithDialect, mlir::gpu::GPUDialect, mlir::LLVM::LLVMDialect,
		                mlir::NVVM::NVVMDialect>();
		mlir::registerBuiltinDialectTranslation(registry);
		mlir::registerLLVMDialectTranslation(registry);
		mlir::registerNVVMDialectTranslation(registry);
		return registry;
	}

	std::string EmitMLIRGPUToNVPTX(mlir::OwningOpRef<mlir::ModuleOp> module)
	{
		auto* context = module->getContext();
		mlir::PassManager passManager(context);
		passManager.addNestedPass<mlir::gpu::GPUModuleOp>(mlir::createConvertGpuOpsToNVVMOps());
		passManager.addPass(mlir::createArithToLLVMConversionPass());
		passManager.addPass(mlir::createReconcileUnrealizedCastsPass());
		if (mlir::failed(passManager.run(module.get())))
		{
			throw std::runtime_error("Failed to lower generated MLIR GPU module to NVVM dialect");
		}

		auto llvmKernelModule = ExtractLLVMKernelModule(module.get());
		llvm::LLVMContext llvmContext;
		auto llvmModule = mlir::translateModuleToLLVMIR(llvmKernelModule.get().getOperation(), llvmContext);
		if (!llvmModule)
		{
			throw std::runtime_error("Failed to translate lowered MLIR NVVM module to LLVM IR");
		}

		return EmitNVPTXAssembly(*llvmModule);
	}

	template <typename BuildFn>
	std::string BuildAndEmitMLIRGPUToNVPTX(BuildFn&& buildFn)
	{
		auto registry = CreateCUDANativeMLIRRegistry();
		mlir::MLIRContext context(registry);
		context.loadAllAvailableDialects();

		CUDANativeMLIRKernelBuilder builder(context);
		return EmitMLIRGPUToNVPTX(buildFn(builder));
	}

	std::string EmitUnaryF32PTXFromMLIRNVPTX(UnaryOp op)
	{
		return BuildAndEmitMLIRGPUToNVPTX([&](CUDANativeMLIRKernelBuilder& builder) {
			return builder.BuildUnaryF32(op);
		});
	}

	std::string EmitBinaryF32PTXFromMLIRNVPTX(BinaryOp op)
	{
		return BuildAndEmitMLIRGPUToNVPTX([&](CUDANativeMLIRKernelBuilder& builder) {
			return builder.BuildBinaryF32(op);
		});
	}

	std::string EmitBinaryBroadcastF32PTXFromMLIRNVPTX(const CUDANativeBroadcastBinaryF32CodegenSpec& spec)
	{
		return BuildAndEmitMLIRGPUToNVPTX([&](CUDANativeMLIRKernelBuilder& builder) {
			return builder.BuildBinaryBroadcastF32(spec);
		});
	}

	std::string_view CUDANativeBinaryF32Instruction(BinaryOp op)
	{
		switch (op)
		{
		case BinaryOp::Add:
			return "\tadd.rn.f32 %f3, %f1, %f2;\n";
		case BinaryOp::Subtract:
			return "\tsub.rn.f32 %f3, %f1, %f2;\n";
		case BinaryOp::Multiply:
			return "\tmul.rn.f32 %f3, %f1, %f2;\n";
		case BinaryOp::Divide:
			return "\tdiv.rn.f32 %f3, %f1, %f2;\n";
		default:
			throw std::runtime_error("Unsupported CUDA native binary op");
		}
	}

	std::string_view CUDANativeUnaryF32Instruction(UnaryOp op)
	{
		switch (op)
		{
		case UnaryOp::Negate:
			return "\tneg.f32 %f2, %f1;\n";
		case UnaryOp::Abs:
			return "\tabs.f32 %f2, %f1;\n";
		case UnaryOp::Sqrt:
			return "\tsqrt.rn.f32 %f2, %f1;\n";
		default:
			throw std::runtime_error("Unsupported CUDA native unary op");
		}
	}
} // namespace

std::string_view CUDANativeBinaryF32KernelName(BinaryOp op, bool broadcast)
{
	switch (op)
	{
	case BinaryOp::Add:
		return broadcast ? "litenn_add_broadcast_f32" : "litenn_add_f32";
	case BinaryOp::Subtract:
		return broadcast ? "litenn_subtract_broadcast_f32" : "litenn_subtract_f32";
	case BinaryOp::Multiply:
		return broadcast ? "litenn_multiply_broadcast_f32" : "litenn_multiply_f32";
	case BinaryOp::Divide:
		return broadcast ? "litenn_divide_broadcast_f32" : "litenn_divide_f32";
	default:
		throw std::runtime_error("Unsupported CUDA native binary op");
	}
}

std::string_view CUDANativeUnaryF32KernelName(UnaryOp op)
{
	switch (op)
	{
	case UnaryOp::Negate:
		return "litenn_negate_f32";
	case UnaryOp::Abs:
		return "litenn_abs_f32";
	case UnaryOp::Sqrt:
		return "litenn_sqrt_f32";
	default:
		throw std::runtime_error("Unsupported CUDA native unary op");
	}
}

std::string CUDANativeBinaryF32PTX(BinaryOp op)
{
	std::string ptx = R"ptx(.version 6.4
.target sm_30
.address_size 64

.visible .entry )ptx";
	ptx += CUDANativeBinaryF32KernelName(op);
	ptx += R"ptx((
	.param .u64 out_ptr,
	.param .u64 lhs_ptr,
	.param .u64 rhs_ptr,
	.param .u32 count
)
{
	.reg .pred %p<2>;
	.reg .b32 %r<6>;
	.reg .b64 %rd<10>;
	.reg .f32 %f<4>;

	ld.param.u64 %rd1, [out_ptr];
	ld.param.u64 %rd2, [lhs_ptr];
	ld.param.u64 %rd3, [rhs_ptr];
	ld.param.u32 %r1, [count];

	mov.u32 %r2, %tid.x;
	mov.u32 %r3, %ctaid.x;
	mov.u32 %r4, %ntid.x;
	mul.lo.u32 %r5, %r3, %r4;
	add.u32 %r5, %r5, %r2;
	setp.ge.u32 %p1, %r5, %r1;
	@%p1 bra DONE;

	mul.wide.u32 %rd4, %r5, 4;
	add.s64 %rd5, %rd2, %rd4;
	add.s64 %rd6, %rd3, %rd4;
	ld.global.f32 %f1, [%rd5];
	ld.global.f32 %f2, [%rd6];
)ptx";
	ptx += CUDANativeBinaryF32Instruction(op);
	ptx += R"ptx(	add.s64 %rd7, %rd1, %rd4;
	st.global.f32 [%rd7], %f3;

DONE:
	ret;
}
)ptx";
	return ptx;
}

std::string CUDANativeBinaryBroadcastF32PTX(const CUDANativeBroadcastBinaryF32CodegenSpec& spec)
{
	const auto outputStrides = ContiguousStridesU32(spec.outputShape);
	const auto lhsStrides = ContiguousStridesU32(spec.lhsShape);
	const auto rhsStrides = ContiguousStridesU32(spec.rhsShape);
	if (!outputStrides || !lhsStrides || !rhsStrides)
	{
		throw std::runtime_error("CUDA native broadcast shape is too large for u32 indexing");
	}
	if (spec.outputShape.size() != spec.lhsShape.size() || spec.outputShape.size() != spec.rhsShape.size())
	{
		throw std::runtime_error("CUDA native broadcast codegen requires same-rank shapes");
	}

	std::string ptx = R"ptx(.version 6.4
.target sm_30
.address_size 64

.visible .entry )ptx";
	ptx += CUDANativeBinaryF32KernelName(spec.op, true);
	ptx += R"ptx((
	.param .u64 out_ptr,
	.param .u64 lhs_ptr,
	.param .u64 rhs_ptr,
	.param .u32 count
)
{
	.reg .pred %p<2>;
	.reg .b32 %r<10>;
	.reg .b64 %rd<10>;
	.reg .f32 %f<4>;

	ld.param.u64 %rd1, [out_ptr];
	ld.param.u64 %rd2, [lhs_ptr];
	ld.param.u64 %rd3, [rhs_ptr];
	ld.param.u32 %r1, [count];

	mov.u32 %r2, %tid.x;
	mov.u32 %r3, %ctaid.x;
	mov.u32 %r4, %ntid.x;
	mul.lo.u32 %r5, %r3, %r4;
	add.u32 %r5, %r5, %r2;
	setp.ge.u32 %p1, %r5, %r1;
	@%p1 bra DONE;

	mov.u32 %r6, 0;
	mov.u32 %r7, 0;
)ptx";

	for (std::size_t i = 0; i < spec.outputShape.size(); ++i)
	{
		ptx += std::format("\tdiv.u32 %r8, %r5, {};\n", (*outputStrides)[i]);
		ptx += std::format("\trem.u32 %r9, %r8, {};\n", spec.outputShape[i]);
		if (spec.lhsShape[i] != 1)
		{
			ptx += std::format("\tmad.lo.u32 %r6, %r9, {}, %r6;\n", (*lhsStrides)[i]);
		}
		if (spec.rhsShape[i] != 1)
		{
			ptx += std::format("\tmad.lo.u32 %r7, %r9, {}, %r7;\n", (*rhsStrides)[i]);
		}
	}

	ptx += R"ptx(
	mul.wide.u32 %rd4, %r6, 4;
	mul.wide.u32 %rd5, %r7, 4;
	add.s64 %rd6, %rd2, %rd4;
	add.s64 %rd7, %rd3, %rd5;
	ld.global.f32 %f1, [%rd6];
	ld.global.f32 %f2, [%rd7];
)ptx";
	ptx += CUDANativeBinaryF32Instruction(spec.op);
	ptx += R"ptx(	mul.wide.u32 %rd8, %r5, 4;
	add.s64 %rd9, %rd1, %rd8;
	st.global.f32 [%rd9], %f3;

DONE:
	ret;
}
)ptx";
	return ptx;
}

std::string CUDANativeUnaryF32PTX(UnaryOp op)
{
	std::string ptx = R"ptx(.version 6.4
.target sm_30
.address_size 64

.visible .entry )ptx";
	ptx += CUDANativeUnaryF32KernelName(op);
	ptx += R"ptx((
	.param .u64 out_ptr,
	.param .u64 in_ptr,
	.param .u32 count
)
{
	.reg .pred %p<2>;
	.reg .b32 %r<6>;
	.reg .b64 %rd<8>;
	.reg .f32 %f<3>;

	ld.param.u64 %rd1, [out_ptr];
	ld.param.u64 %rd2, [in_ptr];
	ld.param.u32 %r1, [count];

	mov.u32 %r2, %tid.x;
	mov.u32 %r3, %ctaid.x;
	mov.u32 %r4, %ntid.x;
	mul.lo.u32 %r5, %r3, %r4;
	add.u32 %r5, %r5, %r2;
	setp.ge.u32 %p1, %r5, %r1;
	@%p1 bra DONE;

	mul.wide.u32 %rd3, %r5, 4;
	add.s64 %rd4, %rd2, %rd3;
	ld.global.f32 %f1, [%rd4];
)ptx";
	ptx += CUDANativeUnaryF32Instruction(op);
	ptx += R"ptx(	add.s64 %rd5, %rd1, %rd3;
	st.global.f32 [%rd5], %f2;

DONE:
	ret;
}
)ptx";
	return ptx;
}

std::string CUDANativeBinaryF32PTXFromMLIRNVPTX(BinaryOp op)
{
	return EmitBinaryF32PTXFromMLIRNVPTX(op);
}

std::optional<std::string> TryCUDANativeBinaryF32PTXFromMLIRNVPTX(BinaryOp op)
{
	try
	{
		return CUDANativeBinaryF32PTXFromMLIRNVPTX(op);
	}
	catch (const std::exception&)
	{
		return std::nullopt;
	}
}

std::string CUDANativeBinaryBroadcastF32PTXFromMLIRNVPTX(const CUDANativeBroadcastBinaryF32CodegenSpec& spec)
{
	return EmitBinaryBroadcastF32PTXFromMLIRNVPTX(spec);
}

std::optional<std::string> TryCUDANativeBinaryBroadcastF32PTXFromMLIRNVPTX(
    const CUDANativeBroadcastBinaryF32CodegenSpec& spec)
{
	try
	{
		return CUDANativeBinaryBroadcastF32PTXFromMLIRNVPTX(spec);
	}
	catch (const std::exception&)
	{
		return std::nullopt;
	}
}

std::string CUDANativeUnaryF32PTXFromMLIRNVPTX(UnaryOp op)
{
	return EmitUnaryF32PTXFromMLIRNVPTX(op);
}

std::optional<std::string> TryCUDANativeUnaryF32PTXFromMLIRNVPTX(UnaryOp op)
{
	try
	{
		return CUDANativeUnaryF32PTXFromMLIRNVPTX(op);
	}
	catch (const std::exception&)
	{
		return std::nullopt;
	}
}

std::vector<std::byte> CUDANativeTextBytes(std::string_view text)
{
	std::vector<std::byte> bytes(text.size() + 1);
	if (!text.empty())
	{
		std::memcpy(bytes.data(), text.data(), text.size());
	}
	bytes.back() = std::byte{ 0 };
	return bytes;
}
} // namespace LiteNN
