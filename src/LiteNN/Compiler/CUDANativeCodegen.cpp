#include "CUDANativeCodegen.h"

#include "mlir/Conversion/Passes.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/GPU/IR/GPUDialect.h"
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/Dialect/LLVMIR/NVVMDialect.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/DialectRegistry.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/IR/Verifier.h"
#include "mlir/Parser/Parser.h"
#include "mlir/Pass/PassManager.h"
#include "mlir/Target/LLVMIR/Dialect/Builtin/BuiltinToLLVMIRTranslation.h"
#include "mlir/Target/LLVMIR/Dialect/LLVMIR/LLVMToLLVMIRTranslation.h"
#include "mlir/Target/LLVMIR/Dialect/NVVM/NVVMToLLVMIRTranslation.h"
#include "mlir/Target/LLVMIR/Export.h"
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

	std::string_view CUDANativeUnaryF32MLIRResultOp(UnaryOp op)
	{
		switch (op)
		{
		case UnaryOp::Negate:
			return "    %result = llvm.fneg %value : f32\n";
		case UnaryOp::Abs:
			return "    %result = llvm.call_intrinsic \"llvm.nvvm.fabs.ftz.f\"(%value) : (f32) -> f32\n";
		case UnaryOp::Sqrt:
			return "    %result = llvm.call_intrinsic \"llvm.nvvm.sqrt.rn.ftz.f\"(%value) : (f32) -> f32\n";
		default:
			throw std::runtime_error("Unsupported MLIR NVPTX CUDA native unary op");
		}
	}

	std::string_view CUDANativeBinaryF32MLIRResultOp(BinaryOp op)
	{
		switch (op)
		{
		case BinaryOp::Add:
			return "    %result = llvm.fadd %lhsValue, %rhsValue : f32\n";
		case BinaryOp::Subtract:
			return "    %result = llvm.fsub %lhsValue, %rhsValue : f32\n";
		case BinaryOp::Multiply:
			return "    %result = llvm.fmul %lhsValue, %rhsValue : f32\n";
		case BinaryOp::Divide:
			return "    %result = llvm.fdiv %lhsValue, %rhsValue : f32\n";
		default:
			throw std::runtime_error("Unsupported MLIR NVPTX CUDA native binary op");
		}
	}

	std::string BuildUnaryF32MLIRGPUModule(UnaryOp op)
	{
		return std::format(R"mlir(module {{
  gpu.module @litenn_cuda_kernels [#nvvm.target<chip = "{}">] {{
    gpu.func @{}(%out: !llvm.ptr, %in: !llvm.ptr, %count: i32) kernel {{
    %tid = gpu.thread_id x
    %ctaid = gpu.block_id x
    %ntid = gpu.block_dim x
    %base = arith.muli %ctaid, %ntid : index
    %idx = arith.addi %base, %tid : index
    %idx32 = arith.index_cast %idx : index to i32
    %done = llvm.icmp "uge" %idx32, %count : i32
    llvm.cond_br %done, ^bb_done, ^bb_body
  ^bb_body:
    %elem = llvm.getelementptr %in[%idx32] : (!llvm.ptr, i32) -> !llvm.ptr, f32
    %value = llvm.load %elem : !llvm.ptr -> f32
{}
    %dst = llvm.getelementptr %out[%idx32] : (!llvm.ptr, i32) -> !llvm.ptr, f32
    llvm.store %result, %dst : f32, !llvm.ptr
    llvm.br ^bb_done
  ^bb_done:
    gpu.return
    }}
  }}
}}
)mlir",
		                   kNVPTXChip, CUDANativeUnaryF32KernelName(op), CUDANativeUnaryF32MLIRResultOp(op));
	}

	std::string BuildBinaryF32MLIRGPUModule(BinaryOp op)
	{
		return std::format(R"mlir(module {{
  gpu.module @litenn_cuda_kernels [#nvvm.target<chip = "{}">] {{
    gpu.func @{}(%out: !llvm.ptr, %lhs: !llvm.ptr, %rhs: !llvm.ptr, %count: i32) kernel {{
    %tid = gpu.thread_id x
    %ctaid = gpu.block_id x
    %ntid = gpu.block_dim x
    %base = arith.muli %ctaid, %ntid : index
    %idx = arith.addi %base, %tid : index
    %idx32 = arith.index_cast %idx : index to i32
    %done = llvm.icmp "uge" %idx32, %count : i32
    llvm.cond_br %done, ^bb_done, ^bb_body
  ^bb_body:
    %lhsElem = llvm.getelementptr %lhs[%idx32] : (!llvm.ptr, i32) -> !llvm.ptr, f32
    %rhsElem = llvm.getelementptr %rhs[%idx32] : (!llvm.ptr, i32) -> !llvm.ptr, f32
    %lhsValue = llvm.load %lhsElem : !llvm.ptr -> f32
    %rhsValue = llvm.load %rhsElem : !llvm.ptr -> f32
{}
    %dst = llvm.getelementptr %out[%idx32] : (!llvm.ptr, i32) -> !llvm.ptr, f32
    llvm.store %result, %dst : f32, !llvm.ptr
    llvm.br ^bb_done
  ^bb_done:
    gpu.return
    }}
  }}
}}
)mlir",
		                   kNVPTXChip, CUDANativeBinaryF32KernelName(op), CUDANativeBinaryF32MLIRResultOp(op));
	}

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

	std::string EmitMLIRGPUToNVPTX(std::string_view mlirText)
	{
		mlir::DialectRegistry registry;
		registry.insert<mlir::arith::ArithDialect, mlir::gpu::GPUDialect, mlir::LLVM::LLVMDialect,
		                mlir::NVVM::NVVMDialect>();
		mlir::registerBuiltinDialectTranslation(registry);
		mlir::registerLLVMDialectTranslation(registry);
		mlir::registerNVVMDialectTranslation(registry);

		mlir::MLIRContext context(registry);
		context.loadAllAvailableDialects();
		auto module = mlir::parseSourceString<mlir::ModuleOp>(mlirText, &context);
		if (!module)
		{
			throw std::runtime_error("Failed to parse generated MLIR GPU module");
		}

		mlir::PassManager passManager(&context);
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

	std::string EmitUnaryF32PTXFromMLIRNVPTX(UnaryOp op)
	{
		return EmitMLIRGPUToNVPTX(BuildUnaryF32MLIRGPUModule(op));
	}

	std::string EmitBinaryF32PTXFromMLIRNVPTX(BinaryOp op)
	{
		return EmitMLIRGPUToNVPTX(BuildBinaryF32MLIRGPUModule(op));
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

	std::string BuildBinaryBroadcastF32MLIRIndexCode(std::span<const std::size_t> outputShape,
	                                                 std::span<const std::size_t> lhsShape,
	                                                 std::span<const std::size_t> rhsShape,
	                                                 std::span<const std::uint32_t> outputStrides,
	                                                 std::span<const std::uint32_t> lhsStrides,
	                                                 std::span<const std::uint32_t> rhsStrides)
	{
		std::string code = "    %c0_i32 = llvm.mlir.constant(0 : i32) : i32\n";
		std::string lhsOffset = "%c0_i32";
		std::string rhsOffset = "%c0_i32";

		for (std::size_t i = 0; i < outputShape.size(); ++i)
		{
			code += std::format("    %outStride{} = llvm.mlir.constant({} : i32) : i32\n", i, outputStrides[i]);
			code += std::format("    %outDim{} = llvm.mlir.constant({} : i32) : i32\n", i, outputShape[i]);
			code += std::format("    %coordDiv{} = llvm.udiv %idx32, %outStride{} : i32\n", i, i);
			code += std::format("    %coord{} = llvm.urem %coordDiv{}, %outDim{} : i32\n", i, i, i);
			if (lhsShape[i] != 1)
			{
				code += std::format("    %lhsStride{} = llvm.mlir.constant({} : i32) : i32\n", i, lhsStrides[i]);
				code += std::format("    %lhsTerm{} = llvm.mul %coord{}, %lhsStride{} : i32\n", i, i, i);
				code += std::format("    %lhsOffset{} = llvm.add {}, %lhsTerm{} : i32\n", i, lhsOffset, i);
				lhsOffset = std::format("%lhsOffset{}", i);
			}
			if (rhsShape[i] != 1)
			{
				code += std::format("    %rhsStride{} = llvm.mlir.constant({} : i32) : i32\n", i, rhsStrides[i]);
				code += std::format("    %rhsTerm{} = llvm.mul %coord{}, %rhsStride{} : i32\n", i, i, i);
				code += std::format("    %rhsOffset{} = llvm.add {}, %rhsTerm{} : i32\n", i, rhsOffset, i);
				rhsOffset = std::format("%rhsOffset{}", i);
			}
		}

		code += std::format("    %lhsElem = llvm.getelementptr %lhs[{}] : (!llvm.ptr, i32) -> !llvm.ptr, f32\n",
		                    lhsOffset);
		code += std::format("    %rhsElem = llvm.getelementptr %rhs[{}] : (!llvm.ptr, i32) -> !llvm.ptr, f32\n",
		                    rhsOffset);
		return code;
	}

	std::string BuildBinaryBroadcastF32MLIRGPUModule(const CUDANativeBroadcastBinaryF32CodegenSpec& spec)
	{
		const auto outputStrides = ContiguousStridesU32(spec.outputShape);
		const auto lhsStrides = ContiguousStridesU32(spec.lhsShape);
		const auto rhsStrides = ContiguousStridesU32(spec.rhsShape);
		if (!outputStrides || !lhsStrides || !rhsStrides)
		{
			throw std::runtime_error("CUDA native MLIR broadcast shape is too large for u32 indexing");
		}
		if (spec.outputShape.size() != spec.lhsShape.size() || spec.outputShape.size() != spec.rhsShape.size())
		{
			throw std::runtime_error("CUDA native MLIR broadcast codegen requires same-rank shapes");
		}

		const auto indexCode = BuildBinaryBroadcastF32MLIRIndexCode(spec.outputShape, spec.lhsShape, spec.rhsShape,
		                                                            *outputStrides, *lhsStrides, *rhsStrides);
		return std::format(R"mlir(module {{
  gpu.module @litenn_cuda_kernels [#nvvm.target<chip = "{}">] {{
    gpu.func @{}(%out: !llvm.ptr, %lhs: !llvm.ptr, %rhs: !llvm.ptr, %count: i32) kernel {{
    %tid = gpu.thread_id x
    %ctaid = gpu.block_id x
    %ntid = gpu.block_dim x
    %base = arith.muli %ctaid, %ntid : index
    %idx = arith.addi %base, %tid : index
    %idx32 = arith.index_cast %idx : index to i32
    %done = llvm.icmp "uge" %idx32, %count : i32
    llvm.cond_br %done, ^bb_done, ^bb_body
  ^bb_body:
{}    %lhsValue = llvm.load %lhsElem : !llvm.ptr -> f32
    %rhsValue = llvm.load %rhsElem : !llvm.ptr -> f32
{}
    %dst = llvm.getelementptr %out[%idx32] : (!llvm.ptr, i32) -> !llvm.ptr, f32
    llvm.store %result, %dst : f32, !llvm.ptr
    llvm.br ^bb_done
  ^bb_done:
    gpu.return
    }}
  }}
}}
)mlir",
		                   kNVPTXChip, CUDANativeBinaryF32KernelName(spec.op, true), indexCode,
		                   CUDANativeBinaryF32MLIRResultOp(spec.op));
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
	return EmitMLIRGPUToNVPTX(BuildBinaryBroadcastF32MLIRGPUModule(spec));
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
