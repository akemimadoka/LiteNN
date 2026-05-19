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

#ifdef LITENN_ENABLE_CUDA
#include <cuda_runtime_api.h>
#endif

#include <cstdint>
#include <cstdlib>
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
	constexpr std::string_view kDefaultNVPTXChip = "sm_75";
	constexpr std::string_view kNVPTXFeatures = "+ptx64";

	bool IsValidNVPTXSMTarget(std::string_view value)
	{
		if (value.size() < 5 || value.substr(0, 3) != "sm_")
		{
			return false;
		}
		for (std::size_t i = 3; i < value.size(); ++i)
		{
			if (value[i] < '0' || value[i] > '9')
			{
				return false;
			}
		}
		return true;
	}

	std::string DetectNativeNVPTXTargetChip()
	{
#ifdef LITENN_ENABLE_CUDA
		int deviceIndex = 0;
		if (cudaGetDevice(&deviceIndex) != cudaSuccess)
		{
			(void)cudaGetLastError();
			throw std::runtime_error("LITENN_CUDA_AOT_TARGET=native requires an available CUDA runtime device");
		}
		cudaDeviceProp properties{};
		if (cudaGetDeviceProperties(&properties, deviceIndex) != cudaSuccess)
		{
			(void)cudaGetLastError();
			throw std::runtime_error("Failed to query CUDA device properties for LITENN_CUDA_AOT_TARGET=native");
		}
		if (properties.major <= 0 || properties.minor < 0)
		{
			throw std::runtime_error("CUDA device reported an invalid compute capability");
		}
		return std::format("sm_{}{}", properties.major, properties.minor);
#else
		throw std::runtime_error("LITENN_CUDA_AOT_TARGET=native requires LiteNN to be built with CUDA support");
#endif
	}

	std::string ResolveNVPTXTargetChip()
	{
		if (const char* env = std::getenv("LITENN_CUDA_AOT_TARGET"))
		{
			const std::string target = env;
			if (!target.empty())
			{
				if (target == "native")
				{
					return DetectNativeNVPTXTargetChip();
				}
				if (!IsValidNVPTXSMTarget(target))
				{
					throw std::runtime_error(
					    "LITENN_CUDA_AOT_TARGET must be native or an sm_<major><minor> target such as sm_75");
				}
				return target;
			}
		}
		return std::string(kDefaultNVPTXChip);
	}

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
		const auto chip = ResolveNVPTXTargetChip();
		auto targetMachine = std::unique_ptr<llvm::TargetMachine>(target->createTargetMachine(
		    llvm::Triple(std::string(kNVPTXTriple)), chip, std::string(kNVPTXFeatures), options,
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

	std::uint32_t NumElementsU32(std::span<const std::size_t> shape, std::string_view label)
	{
		std::uint64_t count = 1;
		for (const auto dim : shape)
		{
			if (dim == 0 || dim > std::numeric_limits<std::uint32_t>::max())
			{
				throw std::runtime_error(std::format("CUDA native MLIR {} shape dimension is invalid", label));
			}
			count *= dim;
			if (count > std::numeric_limits<std::uint32_t>::max())
			{
				throw std::runtime_error(std::format("CUDA native MLIR {} shape is too large for u32 indexing", label));
			}
		}
		return static_cast<std::uint32_t>(count);
	}

	std::vector<std::size_t> ReduceOutputShape(std::span<const std::size_t> inputShape, std::size_t axis)
	{
		if (axis >= inputShape.size())
		{
			throw std::runtime_error("CUDA native reduce axis is out of range");
		}

		std::vector<std::size_t> outputShape;
		outputShape.reserve(inputShape.size() - 1);
		for (std::size_t i = 0; i < inputShape.size(); ++i)
		{
			if (i != axis)
			{
				outputShape.push_back(inputShape[i]);
			}
		}
		if (outputShape.empty())
		{
			outputShape.push_back(1);
		}
		return outputShape;
	}

	std::uint32_t AxisInnerSizeU32(std::span<const std::size_t> shape, std::size_t axis)
	{
		std::uint64_t inner = 1;
		for (std::size_t i = axis + 1; i < shape.size(); ++i)
		{
			inner *= shape[i];
			if (inner > std::numeric_limits<std::uint32_t>::max())
			{
				throw std::runtime_error("CUDA native reduce inner size is too large for u32 indexing");
			}
		}
		return static_cast<std::uint32_t>(inner);
	}

	std::vector<std::uint32_t> ConcatAxisStartsU32(std::span<const std::vector<std::size_t>> inputShapes,
	                                             std::size_t axis)
	{
		std::vector<std::uint32_t> starts;
		starts.reserve(inputShapes.size());
		std::uint64_t offset = 0;
		for (const auto& shape : inputShapes)
		{
			if (axis >= shape.size())
			{
				throw std::runtime_error("CUDA native concat axis is out of range");
			}
			if (offset > std::numeric_limits<std::uint32_t>::max())
			{
				throw std::runtime_error("CUDA native concat axis offset is too large for u32 indexing");
			}
			starts.push_back(static_cast<std::uint32_t>(offset));
			offset += shape[axis];
		}
		return starts;
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

	bool IsSupportedCUDANativeCastScalarType(DataType type)
	{
		switch (type)
		{
		case DataType::Float64:
		case DataType::Float32:
		case DataType::Float16:
		case DataType::BFloat16:
		case DataType::Float8E4M3:
		case DataType::Float8E5M2:
		case DataType::Int64:
		case DataType::Int32:
		case DataType::Int8:
		case DataType::UInt8:
			return true;
		default:
			return false;
		}
	}

	bool IsCUDANativeCastFloatType(DataType type)
	{
		switch (type)
		{
		case DataType::Float64:
		case DataType::Float32:
		case DataType::Float16:
		case DataType::BFloat16:
		case DataType::Float8E4M3:
		case DataType::Float8E5M2:
			return true;
		default:
			return false;
		}
	}

	bool IsCUDANativeCastFloat8Type(DataType type)
	{
		switch (type)
		{
		case DataType::Float8E4M3:
		case DataType::Float8E5M2:
			return true;
		default:
			return false;
		}
	}

	bool IsSupportedCUDANativeMatMulBiasEpilogueType(DataType type)
	{
		switch (type)
		{
		case DataType::Float32:
		case DataType::Float16:
		case DataType::BFloat16:
		case DataType::Int8:
		case DataType::UInt8:
			return true;
		default:
			return false;
		}
	}

	DataType CUDANativeMatMulBiasEpilogueComputeType(DataType type)
	{
		switch (type)
		{
		case DataType::Float32:
		case DataType::Float16:
		case DataType::BFloat16:
			return DataType::Float32;
		case DataType::Int8:
		case DataType::UInt8:
			return DataType::Int32;
		default:
			throw std::runtime_error("Unsupported CUDA native MatMulBias epilogue dtype");
		}
	}

	bool IsCUDANativeCastUnsignedIntegerType(DataType type)
	{
		return type == DataType::UInt8;
	}

	std::string_view CUDANativeDataTypeShortName(DataType type)
	{
		switch (type)
		{
		case DataType::Float64:
			return "f64";
		case DataType::Float32:
			return "f32";
		case DataType::Float16:
			return "f16";
		case DataType::BFloat16:
			return "bf16";
		case DataType::Float8E4M3:
			return "f8e4m3";
		case DataType::Float8E5M2:
			return "f8e5m2";
		case DataType::Int64:
			return "i64";
		case DataType::Int32:
			return "i32";
		case DataType::Int8:
			return "i8";
		case DataType::UInt8:
			return "u8";
		default:
			throw std::runtime_error("Unsupported CUDA native cast dtype");
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

		mlir::OwningOpRef<mlir::ModuleOp> BuildReduceF32(const CUDANativeReduceF32CodegenSpec& spec)
		{
			ValidateShapeDimsU32(spec.inputShape);
			const auto outputShape = ReduceOutputShape(spec.inputShape, spec.axis);
			const auto outputCount = NumElementsU32(outputShape, "reduce output");
			const auto axisSize = static_cast<std::uint32_t>(spec.inputShape[spec.axis]);
			const auto innerSize = AxisInnerSizeU32(spec.inputShape, spec.axis);

			auto kernelModule = CreateKernelModule();
			llvm::SmallVector<mlir::Type, 3> argTypes{ ptrType_, ptrType_, i32Type_ };
			auto func = CreateKernelFunc(kernelModule.gpuModule, CUDANativeReduceF32KernelName(spec.op), argTypes);
			auto blocks = EmitLinearIndexGuard(func, 2);

			builder_.setInsertionPointToStart(blocks.body);
			auto out = blocks.entry->getArgument(0);
			auto in = blocks.entry->getArgument(1);
			auto outputIndex = blocks.index32;
			auto inner = EmitI32Constant(innerSize);
			auto axis = EmitI32Constant(axisSize);
			auto outerIndex = EmitI32UDiv(outputIndex, inner);
			auto innerIndex = EmitI32URem(outputIndex, inner);
			auto base = EmitI32Add(EmitI32Mul(EmitI32Mul(outerIndex, axis), inner), innerIndex);

			mlir::Value accumulator;
			for (std::uint32_t reduceIndex = 0; reduceIndex < axisSize; ++reduceIndex)
			{
				auto offset = EmitI32Add(base, EmitI32Mul(EmitI32Constant(reduceIndex), inner));
				auto value = EmitLoadF32(EmitF32GEP(in, offset));
				if (reduceIndex == 0)
				{
					accumulator = value;
					continue;
				}
				switch (spec.op)
				{
				case ReduceOp::Sum:
				case ReduceOp::Mean:
					accumulator = EmitF32Add(accumulator, value);
					break;
				case ReduceOp::Max:
					accumulator = EmitF32Intrinsic("llvm.nvvm.fmax.ftz.f", mlir::ValueRange{ accumulator, value });
					break;
				}
			}
			if (spec.op == ReduceOp::Mean)
			{
				accumulator = EmitF32Mul(accumulator, EmitF32Constant(1.0f / static_cast<float>(axisSize)));
			}
			EmitStoreF32(accumulator, EmitF32GEP(out, outputIndex));
			FinishLinearKernel(blocks);

			(void)outputCount;
			return FinalizeModule(std::move(kernelModule.module));
		}

		mlir::OwningOpRef<mlir::ModuleOp> BuildConcatF32(const CUDANativeConcatF32CodegenSpec& spec)
		{
			if (spec.inputShapes.empty())
			{
				throw std::runtime_error("CUDA native concat codegen requires at least one input");
			}
			ValidateShapeDimsU32(spec.outputShape);
			const auto outputStrides = ContiguousStridesU32(spec.outputShape);
			if (!outputStrides)
			{
				throw std::runtime_error("CUDA native concat output shape is too large for u32 indexing");
			}
			const auto axisStarts = ConcatAxisStartsU32(spec.inputShapes, spec.axis);

			auto kernelModule = CreateKernelModule();
			for (std::size_t inputIndex = 0; inputIndex < spec.inputShapes.size(); ++inputIndex)
			{
				const auto& inputShape = spec.inputShapes[inputIndex];
				ValidateShapeDimsU32(inputShape);
				const auto inputStrides = ContiguousStridesU32(inputShape);
				if (!inputStrides || inputShape.size() != spec.outputShape.size())
				{
					throw std::runtime_error("CUDA native concat input shape is invalid");
				}

				llvm::SmallVector<mlir::Type, 3> argTypes{ ptrType_, ptrType_, i32Type_ };
				auto func = CreateKernelFunc(kernelModule.gpuModule, CUDANativeConcatF32KernelName(inputIndex), argTypes);
				auto blocks = EmitLinearIndexGuard(func, 2);

				builder_.setInsertionPointToStart(blocks.body);
				auto out = blocks.entry->getArgument(0);
				auto in = blocks.entry->getArgument(1);
				auto outputOffset = EmitI32Constant(0);
				for (std::size_t dim = 0; dim < inputShape.size(); ++dim)
				{
					auto inStride = EmitI32Constant((*inputStrides)[dim]);
					auto inDim = EmitI32Constant(static_cast<std::uint32_t>(inputShape[dim]));
					auto coord = EmitI32URem(EmitI32UDiv(blocks.index32, inStride), inDim);
					if (dim == spec.axis)
					{
						coord = EmitI32Add(coord, EmitI32Constant(axisStarts[inputIndex]));
					}
					outputOffset = EmitI32Add(outputOffset, EmitI32Mul(coord, EmitI32Constant((*outputStrides)[dim])));
				}
				EmitStoreF32(EmitLoadF32(EmitF32GEP(in, blocks.index32)), EmitF32GEP(out, outputOffset));
				FinishLinearKernel(blocks);
			}

			return FinalizeModule(std::move(kernelModule.module));
		}

		mlir::OwningOpRef<mlir::ModuleOp> BuildSliceF32(const CUDANativeSliceF32CodegenSpec& spec)
		{
			ValidateShapeDimsU32(spec.inputShape);
			ValidateShapeDimsU32(spec.outputShape);
			if (spec.inputShape.size() != spec.outputShape.size() || spec.axis >= spec.inputShape.size())
			{
				throw std::runtime_error("CUDA native slice codegen received incompatible shapes");
			}
			const auto inputStrides = ContiguousStridesU32(spec.inputShape);
			const auto outputStrides = ContiguousStridesU32(spec.outputShape);
			if (!inputStrides || !outputStrides)
			{
				throw std::runtime_error("CUDA native slice shape is too large for u32 indexing");
			}

			auto kernelModule = CreateKernelModule();
			llvm::SmallVector<mlir::Type, 3> argTypes{ ptrType_, ptrType_, i32Type_ };
			auto func = CreateKernelFunc(kernelModule.gpuModule, CUDANativeSliceF32KernelName(), argTypes);
			auto blocks = EmitLinearIndexGuard(func, 2);

			builder_.setInsertionPointToStart(blocks.body);
			auto out = blocks.entry->getArgument(0);
			auto in = blocks.entry->getArgument(1);
			auto inputOffset = EmitI32Constant(0);
			for (std::size_t dim = 0; dim < spec.outputShape.size(); ++dim)
			{
				auto outStride = EmitI32Constant((*outputStrides)[dim]);
				auto outDim = EmitI32Constant(static_cast<std::uint32_t>(spec.outputShape[dim]));
				auto coord = EmitI32URem(EmitI32UDiv(blocks.index32, outStride), outDim);
				if (dim == spec.axis)
				{
					coord = EmitI32Add(coord, EmitI32Constant(static_cast<std::uint32_t>(spec.start)));
				}
				inputOffset = EmitI32Add(inputOffset, EmitI32Mul(coord, EmitI32Constant((*inputStrides)[dim])));
			}
			EmitStoreF32(EmitLoadF32(EmitF32GEP(in, inputOffset)), EmitF32GEP(out, blocks.index32));
			FinishLinearKernel(blocks);

			return FinalizeModule(std::move(kernelModule.module));
		}

		mlir::OwningOpRef<mlir::ModuleOp> BuildCast(const CUDANativeCastCodegenSpec& spec)
		{
			if (!CUDANativeSupportsCast(spec.srcType, spec.dstType))
			{
				throw std::runtime_error("CUDA native cast codegen received an unsupported dtype pair");
			}

			auto kernelModule = CreateKernelModule();
			llvm::SmallVector<mlir::Type, 4> argTypes{ ptrType_, ptrType_, i32Type_ };
			auto func = CreateKernelFunc(kernelModule.gpuModule,
			                            CUDANativeCastKernelName(spec.srcType, spec.dstType), argTypes);
			auto blocks = EmitLinearIndexGuard(func, 2);

			builder_.setInsertionPointToStart(blocks.body);
			auto out = blocks.entry->getArgument(0);
			auto in = blocks.entry->getArgument(1);
			auto srcScalarType = GetCastScalarType(spec.srcType);
			auto dstScalarType = GetCastScalarType(spec.dstType);
			auto srcStorageType = GetCastStorageType(spec.srcType);
			auto dstStorageType = GetCastStorageType(spec.dstType);
			auto rawValue = EmitLoad(EmitTypedGEP(in, srcStorageType, blocks.index32), srcStorageType);
			auto valueTypeId = spec.srcType;
			auto valueType = srcScalarType;
			auto value = rawValue;
			if (IsCUDANativeCastFloat8Type(spec.srcType))
			{
				value = DecodeFloat8StorageValue(spec.srcType, rawValue);
				valueTypeId = DataType::Float16;
				valueType = builder_.getF16Type();
			}
			else
			{
				value = DecodeCastStorageValue(spec.srcType, srcScalarType, rawValue);
			}

			mlir::Value stored;
			if (IsCUDANativeCastFloat8Type(spec.dstType))
			{
				auto f32Value = EmitCastValue(valueTypeId, DataType::Float32, valueType, f32Type_, value);
				stored = EncodeFloat8StorageValue(spec.dstType, f32Value);
			}
			else
			{
				auto result = EmitCastValue(valueTypeId, spec.dstType, valueType, dstScalarType, value);
				stored = EncodeCastStorageValue(spec.dstType, dstStorageType, result);
			}
			EmitStore(stored, EmitTypedGEP(out, dstStorageType, blocks.index32));
			FinishLinearKernel(blocks);

			return FinalizeModule(std::move(kernelModule.module));
		}

		mlir::OwningOpRef<mlir::ModuleOp> BuildMatMulBiasEpilogueF32(
		    const CUDANativeMatMulBiasEpilogueF32CodegenSpec& spec)
		{
			return BuildMatMulBiasEpilogue({
			    .kernelName = spec.kernelName,
			    .dtype = DataType::Float32,
			    .outputShape = spec.outputShape,
			    .biasShape = spec.biasShape,
			    .relu = spec.relu,
			});
		}

		mlir::OwningOpRef<mlir::ModuleOp> BuildMatMulBiasEpilogue(
		    const CUDANativeMatMulBiasEpilogueCodegenSpec& spec)
		{
			auto kernelModule = CreateKernelModule();
			EmitMatMulBiasEpilogue(kernelModule.gpuModule, spec);
			return FinalizeModule(std::move(kernelModule.module));
		}

		mlir::OwningOpRef<mlir::ModuleOp> BuildMatMulBiasEpiloguesF32(
		    std::span<const CUDANativeMatMulBiasEpilogueF32CodegenSpec> specs)
		{
			std::vector<CUDANativeMatMulBiasEpilogueCodegenSpec> genericSpecs;
			genericSpecs.reserve(specs.size());
			for (const auto& spec : specs)
			{
				genericSpecs.push_back({
				    .kernelName = spec.kernelName,
				    .dtype = DataType::Float32,
				    .outputShape = spec.outputShape,
				    .biasShape = spec.biasShape,
				    .relu = spec.relu,
				});
			}
			return BuildMatMulBiasEpilogues(genericSpecs);
		}

		mlir::OwningOpRef<mlir::ModuleOp> BuildMatMulBiasEpilogues(
		    std::span<const CUDANativeMatMulBiasEpilogueCodegenSpec> specs)
		{
			if (specs.empty())
			{
				throw std::runtime_error("CUDA native MatMulBias epilogue set must not be empty");
			}
			auto kernelModule = CreateKernelModule();
			for (const auto& spec : specs)
			{
				EmitMatMulBiasEpilogue(kernelModule.gpuModule, spec);
			}
			return FinalizeModule(std::move(kernelModule.module));
		}

	private:
		void EmitMatMulBiasEpilogue(mlir::gpu::GPUModuleOp gpuModule,
		                           const CUDANativeMatMulBiasEpilogueCodegenSpec& spec)
		{
			if (!IsSupportedCUDANativeMatMulBiasEpilogueType(spec.dtype))
			{
				throw std::runtime_error("CUDA native MatMulBias epilogue received an unsupported dtype");
			}
			const auto outputStrides = ContiguousStridesU32(spec.outputShape);
			const auto biasStrides = ContiguousStridesU32(spec.biasShape);
			if (!outputStrides || !biasStrides || spec.outputShape.size() != spec.biasShape.size())
			{
				throw std::runtime_error("CUDA native MatMulBias epilogue received invalid shapes");
			}

			llvm::SmallVector<mlir::Type, 3> argTypes{ ptrType_, ptrType_, i32Type_ };
			const auto name =
			    spec.kernelName.empty() ? CUDANativeMatMulBiasEpilogueKernelName(spec.dtype, spec.relu) : spec.kernelName;
			auto func = CreateKernelFunc(gpuModule, name, argTypes);
			auto blocks = EmitLinearIndexGuard(func, 2);

			builder_.setInsertionPointToStart(blocks.body);
			auto out = blocks.entry->getArgument(0);
			auto bias = blocks.entry->getArgument(1);
			auto elementType = GetCastScalarType(spec.dtype);
			const auto computeTypeId = CUDANativeMatMulBiasEpilogueComputeType(spec.dtype);
			auto computeType = GetCastScalarType(computeTypeId);
			auto biasOffset = EmitI32Constant(0);
			for (std::size_t dim = 0; dim < spec.outputShape.size(); ++dim)
			{
				auto outStride = EmitI32Constant((*outputStrides)[dim]);
				auto outDim = EmitI32Constant(static_cast<std::uint32_t>(spec.outputShape[dim]));
				auto coord = EmitI32URem(EmitI32UDiv(blocks.index32, outStride), outDim);
				if (spec.biasShape[dim] != 1)
				{
					biasOffset = EmitI32Add(biasOffset, EmitI32Mul(coord, EmitI32Constant((*biasStrides)[dim])));
				}
			}

			auto outputValue = EmitLoad(EmitTypedGEP(out, elementType, blocks.index32), elementType);
			auto biasValue = EmitLoad(EmitTypedGEP(bias, elementType, biasOffset), elementType);
			auto outputCompute = EmitCastValue(spec.dtype, computeTypeId, elementType, computeType, outputValue);
			auto biasCompute = EmitCastValue(spec.dtype, computeTypeId, elementType, computeType, biasValue);
			auto value = EmitScalarAdd(computeTypeId, computeType, outputCompute, biasCompute);
			if (spec.relu)
			{
				value = EmitReLU(computeTypeId, computeType, value);
			}
			auto result = EmitCastValue(computeTypeId, spec.dtype, computeType, elementType, value);
			EmitStore(result, EmitTypedGEP(out, elementType, blocks.index32));
			FinishLinearKernel(blocks);
		}

		CUDANativeMLIRKernelModule CreateKernelModule()
		{
			auto module = mlir::ModuleOp::create(loc_);
			builder_.setInsertionPointToStart(module.getBody());

			const auto chip = ResolveNVPTXTargetChip();
			auto target = mlir::NVVM::NVVMTargetAttr::get(
			    &context_, 2, ToLLVMStringRef(kNVPTXTriple), ToLLVMStringRef(chip), ToLLVMStringRef(kNVPTXFeatures),
			    nullptr, nullptr, false);
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

		mlir::Value EmitF32Constant(float value)
		{
			return builder_.create<mlir::LLVM::ConstantOp>(loc_, f32Type_, builder_.getF32FloatAttr(value)).getResult();
		}

		mlir::Value EmitF32Add(mlir::Value lhs, mlir::Value rhs)
		{
			return builder_.create<mlir::LLVM::FAddOp>(loc_, f32Type_, mlir::ValueRange{ lhs, rhs }).getResult();
		}

		mlir::Value EmitF32Mul(mlir::Value lhs, mlir::Value rhs)
		{
			return builder_.create<mlir::LLVM::FMulOp>(loc_, f32Type_, mlir::ValueRange{ lhs, rhs }).getResult();
		}

		mlir::Value EmitF32Intrinsic(std::string_view name, mlir::ValueRange args)
		{
			return builder_
			    .create<mlir::LLVM::CallIntrinsicOp>(loc_, f32Type_, builder_.getStringAttr(ToLLVMStringRef(name)), args)
			    .getResult(0);
		}

		mlir::Value EmitIntegerConstant(mlir::Type type, std::uint64_t value)
		{
			return builder_.create<mlir::arith::ConstantOp>(loc_, builder_.getIntegerAttr(type, value)).getResult();
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

		mlir::Type GetCastScalarType(DataType type)
		{
			switch (type)
			{
			case DataType::Float64:
				return builder_.getF64Type();
			case DataType::Float32:
				return f32Type_;
			case DataType::Float16:
				return builder_.getF16Type();
			case DataType::BFloat16:
				return builder_.getBF16Type();
			case DataType::Float8E4M3:
				return mlir::Float8E4M3FNType::get(&context_);
			case DataType::Float8E5M2:
				return mlir::Float8E5M2Type::get(&context_);
			case DataType::Int64:
				return builder_.getI64Type();
			case DataType::Int32:
				return i32Type_;
			case DataType::Int8:
			case DataType::UInt8:
				return builder_.getIntegerType(8);
			default:
				throw std::runtime_error("Unsupported CUDA native cast dtype");
			}
		}

		mlir::Type GetCastStorageType(DataType type)
		{
			switch (type)
			{
			case DataType::Float8E4M3:
			case DataType::Float8E5M2:
				return builder_.getIntegerType(8);
			default:
				return GetCastScalarType(type);
			}
		}

		mlir::Value EmitTypedGEP(mlir::Value base, mlir::Type elementType, mlir::Value index)
		{
			return builder_
			    .create<mlir::LLVM::GEPOp>(loc_, ptrType_, elementType, base, mlir::ValueRange{ index })
			    .getResult();
		}

		mlir::Value EmitLoad(mlir::Value ptr, mlir::Type type)
		{
			return builder_.create<mlir::LLVM::LoadOp>(loc_, type, ptr).getResult();
		}

		void EmitStore(mlir::Value value, mlir::Value ptr)
		{
			builder_.create<mlir::LLVM::StoreOp>(loc_, value, ptr);
		}

		mlir::Value DecodeCastStorageValue(DataType type, mlir::Type scalarType, mlir::Value value)
		{
			switch (type)
			{
			case DataType::Float8E4M3:
			case DataType::Float8E5M2:
				return builder_.create<mlir::arith::BitcastOp>(loc_, scalarType, value).getResult();
			default:
				return value;
			}
		}

		mlir::NVVM::ConvertFP8Type GetNVVMConvertFP8Type(DataType type)
		{
			switch (type)
			{
			case DataType::Float8E4M3:
				return mlir::NVVM::ConvertFP8Type::E4M3;
			case DataType::Float8E5M2:
				return mlir::NVVM::ConvertFP8Type::E5M2;
			default:
				throw std::runtime_error("Unsupported CUDA native float8 dtype");
			}
		}

		std::string_view Float8ToF16IntrinsicName(DataType type)
		{
			switch (type)
			{
			case DataType::Float8E4M3:
				return "llvm.nvvm.e4m3x2.to.f16x2.rn";
			case DataType::Float8E5M2:
				return "llvm.nvvm.e5m2x2.to.f16x2.rn";
			default:
				throw std::runtime_error("Unsupported CUDA native float8 dtype");
			}
		}

		mlir::Value DuplicateByteToPackedI16(mlir::Value value)
		{
			auto i16Type = builder_.getIntegerType(16);
			auto extended = builder_.create<mlir::arith::ExtUIOp>(loc_, i16Type, value).getResult();
			auto shifted =
			    builder_.create<mlir::arith::ShLIOp>(loc_, extended, EmitIntegerConstant(i16Type, 8)).getResult();
			return builder_.create<mlir::arith::OrIOp>(loc_, extended, shifted).getResult();
		}

		mlir::Value DecodeFloat8StorageValue(DataType type, mlir::Value value)
		{
			auto packedType = builder_.getIntegerType(16);
			auto packed = DuplicateByteToPackedI16(value);
			auto lanesType = mlir::LLVM::getVectorType(builder_.getF16Type(), 2);
			auto converted = builder_
			                     .create<mlir::LLVM::CallIntrinsicOp>(loc_, lanesType,
			                                                         builder_.getStringAttr(ToLLVMStringRef(
			                                                             Float8ToF16IntrinsicName(type))),
			                                                         mlir::ValueRange{ packed })
			                     .getResult(0);
			return builder_.create<mlir::LLVM::ExtractElementOp>(loc_, builder_.getF16Type(), converted,
			                                                     EmitI32Constant(0))
			    .getResult();
		}

		mlir::Value EncodeFloat8StorageValue(DataType type, mlir::Value value)
		{
			auto packed = builder_
			                  .create<mlir::NVVM::ConvertF32x2ToF8x2Op>(loc_, builder_.getIntegerType(16),
			                                                          GetNVVMConvertFP8Type(type), value, value,
			                                                          mlir::NVVM::FPRoundingMode::RN,
			                                                          mlir::NVVM::SaturationMode::SATFINITE)
			                  .getDst();
			return builder_.create<mlir::arith::TruncIOp>(loc_, builder_.getIntegerType(8), packed).getResult();
		}

		mlir::Value EncodeCastStorageValue(DataType type, mlir::Type storageType, mlir::Value value)
		{
			switch (type)
			{
			case DataType::Float8E4M3:
			case DataType::Float8E5M2:
				return builder_.create<mlir::arith::BitcastOp>(loc_, storageType, value).getResult();
			default:
				return value;
			}
		}

		mlir::Value EmitScalarZero(DataType type, mlir::Type valueType)
		{
			if (IsCUDANativeCastFloatType(type))
			{
				auto floatType = mlir::cast<mlir::FloatType>(valueType);
				return builder_.create<mlir::arith::ConstantFloatOp>(loc_, floatType, llvm::APFloat(floatType.getFloatSemantics(), 0)).getResult();
			}
			return builder_.create<mlir::arith::ConstantOp>(loc_, builder_.getIntegerAttr(valueType, 0)).getResult();
		}

		mlir::Value EmitScalarAdd(DataType type, mlir::Type valueType, mlir::Value lhs, mlir::Value rhs)
		{
			if (IsCUDANativeCastFloatType(type))
			{
				return builder_.create<mlir::arith::AddFOp>(loc_, lhs, rhs).getResult();
			}
			return builder_.create<mlir::arith::AddIOp>(loc_, lhs, rhs).getResult();
		}

		mlir::Value EmitReLU(DataType type, mlir::Type valueType, mlir::Value value)
		{
			auto zero = EmitScalarZero(type, valueType);
			if (IsCUDANativeCastFloatType(type))
			{
				return builder_.create<mlir::arith::MaximumFOp>(loc_, value, zero).getResult();
			}
			return builder_.create<mlir::arith::MaxSIOp>(loc_, value, zero).getResult();
		}

		mlir::Value EmitCastValue(DataType srcType, DataType dstType, mlir::Type srcElemType,
		                        mlir::Type dstElemType, mlir::Value value)
		{
			if (srcType == dstType)
			{
				return value;
			}

			const bool srcFloat = IsCUDANativeCastFloatType(srcType);
			const bool dstFloat = IsCUDANativeCastFloatType(dstType);
			const bool srcUnsigned = IsCUDANativeCastUnsignedIntegerType(srcType);
			const bool dstUnsigned = IsCUDANativeCastUnsignedIntegerType(dstType);

			if (srcFloat && dstFloat)
			{
				auto srcFloatTy = mlir::cast<mlir::FloatType>(srcElemType);
				auto dstFloatTy = mlir::cast<mlir::FloatType>(dstElemType);
				if (srcElemType == dstElemType)
				{
					return value;
				}
				if (srcFloatTy.getWidth() < dstFloatTy.getWidth())
				{
					return builder_.create<mlir::arith::ExtFOp>(loc_, dstElemType, value).getResult();
				}
				if (srcFloatTy.getWidth() > dstFloatTy.getWidth())
				{
					return builder_.create<mlir::arith::TruncFOp>(loc_, dstElemType, value).getResult();
				}
				auto widened = builder_.create<mlir::arith::ExtFOp>(loc_, f32Type_, value).getResult();
				return builder_.create<mlir::arith::TruncFOp>(loc_, dstElemType, widened).getResult();
			}

			if (srcFloat)
			{
				if (dstUnsigned)
				{
					return builder_.create<mlir::arith::FPToUIOp>(loc_, dstElemType, value).getResult();
				}
				return builder_.create<mlir::arith::FPToSIOp>(loc_, dstElemType, value).getResult();
			}

			if (dstFloat)
			{
				if (srcUnsigned)
				{
					return builder_.create<mlir::arith::UIToFPOp>(loc_, dstElemType, value).getResult();
				}
				return builder_.create<mlir::arith::SIToFPOp>(loc_, dstElemType, value).getResult();
			}

			auto srcIntTy = mlir::cast<mlir::IntegerType>(srcElemType);
			auto dstIntTy = mlir::cast<mlir::IntegerType>(dstElemType);
			if (srcIntTy.getWidth() < dstIntTy.getWidth())
			{
				if (srcUnsigned)
				{
					return builder_.create<mlir::arith::ExtUIOp>(loc_, dstElemType, value).getResult();
				}
				return builder_.create<mlir::arith::ExtSIOp>(loc_, dstElemType, value).getResult();
			}
			if (srcIntTy.getWidth() > dstIntTy.getWidth())
			{
				return builder_.create<mlir::arith::TruncIOp>(loc_, dstElemType, value).getResult();
			}
			return value;
		}

		mlir::Value EmitUnaryF32Result(UnaryOp op, mlir::Value value)
		{
			switch (op)
			{
			case UnaryOp::Negate:
				return builder_.create<mlir::LLVM::FNegOp>(loc_, f32Type_, mlir::ValueRange{ value }).getResult();
			case UnaryOp::Abs:
				return EmitF32Intrinsic("llvm.nvvm.fabs.ftz.f", mlir::ValueRange{ value });
			case UnaryOp::Sqrt:
				return EmitF32Intrinsic("llvm.nvvm.sqrt.rn.ftz.f", mlir::ValueRange{ value });
			case UnaryOp::Exp: {
				const auto log2e = EmitF32Constant(1.4426950408889634f);
				auto scaled = EmitF32Mul(value, log2e);
				return EmitF32Intrinsic("llvm.nvvm.ex2.approx.ftz.f", mlir::ValueRange{ scaled });
			}
			case UnaryOp::Log: {
				auto log2Value = EmitF32Intrinsic("llvm.nvvm.lg2.approx.ftz.f", mlir::ValueRange{ value });
				const auto ln2 = EmitF32Constant(0.6931471805599453f);
				return EmitF32Mul(log2Value, ln2);
			}
			case UnaryOp::Sin:
				return EmitF32Intrinsic("llvm.nvvm.sin.approx.ftz.f", mlir::ValueRange{ value });
			case UnaryOp::Cos:
				return EmitF32Intrinsic("llvm.nvvm.cos.approx.ftz.f", mlir::ValueRange{ value });
			default:
				throw std::runtime_error("Unsupported MLIR NVPTX CUDA native unary op");
			}
		}

		mlir::Value EmitBinaryF32Result(BinaryOp op, mlir::Value lhs, mlir::Value rhs)
		{
			switch (op)
			{
			case BinaryOp::Add:
				return EmitF32Add(lhs, rhs);
			case BinaryOp::Subtract:
				return builder_.create<mlir::LLVM::FSubOp>(loc_, f32Type_, mlir::ValueRange{ lhs, rhs }).getResult();
			case BinaryOp::Multiply:
				return EmitF32Mul(lhs, rhs);
			case BinaryOp::Divide:
				return builder_.create<mlir::LLVM::FDivOp>(loc_, f32Type_, mlir::ValueRange{ lhs, rhs }).getResult();
			case BinaryOp::Max:
				return EmitF32Intrinsic("llvm.nvvm.fmax.ftz.f", mlir::ValueRange{ lhs, rhs });
			case BinaryOp::Min:
				return EmitF32Intrinsic("llvm.nvvm.fmin.ftz.f", mlir::ValueRange{ lhs, rhs });
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
		context.disableMultithreading();
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

	std::string EmitReduceF32PTXFromMLIRNVPTX(const CUDANativeReduceF32CodegenSpec& spec)
	{
		return BuildAndEmitMLIRGPUToNVPTX([&](CUDANativeMLIRKernelBuilder& builder) {
			return builder.BuildReduceF32(spec);
		});
	}

	std::string EmitConcatF32PTXFromMLIRNVPTX(const CUDANativeConcatF32CodegenSpec& spec)
	{
		return BuildAndEmitMLIRGPUToNVPTX([&](CUDANativeMLIRKernelBuilder& builder) {
			return builder.BuildConcatF32(spec);
		});
	}

	std::string EmitSliceF32PTXFromMLIRNVPTX(const CUDANativeSliceF32CodegenSpec& spec)
	{
		return BuildAndEmitMLIRGPUToNVPTX([&](CUDANativeMLIRKernelBuilder& builder) {
			return builder.BuildSliceF32(spec);
		});
	}

	std::string EmitCastPTXFromMLIRNVPTX(const CUDANativeCastCodegenSpec& spec)
	{
		return BuildAndEmitMLIRGPUToNVPTX([&](CUDANativeMLIRKernelBuilder& builder) {
			return builder.BuildCast(spec);
		});
	}

	std::string EmitMatMulBiasEpilogueF32PTXFromMLIRNVPTX(
	    const CUDANativeMatMulBiasEpilogueF32CodegenSpec& spec)
	{
		return BuildAndEmitMLIRGPUToNVPTX([&](CUDANativeMLIRKernelBuilder& builder) {
			return builder.BuildMatMulBiasEpilogueF32(spec);
		});
	}

	std::string EmitMatMulBiasEpiloguePTXFromMLIRNVPTX(
	    const CUDANativeMatMulBiasEpilogueCodegenSpec& spec)
	{
		return BuildAndEmitMLIRGPUToNVPTX([&](CUDANativeMLIRKernelBuilder& builder) {
			return builder.BuildMatMulBiasEpilogue(spec);
		});
	}

	std::string EmitMatMulBiasEpiloguesF32PTXFromMLIRNVPTX(
	    std::span<const CUDANativeMatMulBiasEpilogueF32CodegenSpec> specs)
	{
		return BuildAndEmitMLIRGPUToNVPTX([&](CUDANativeMLIRKernelBuilder& builder) {
			return builder.BuildMatMulBiasEpiloguesF32(specs);
		});
	}

	std::string EmitMatMulBiasEpiloguesPTXFromMLIRNVPTX(
	    std::span<const CUDANativeMatMulBiasEpilogueCodegenSpec> specs)
	{
		return BuildAndEmitMLIRGPUToNVPTX([&](CUDANativeMLIRKernelBuilder& builder) {
			return builder.BuildMatMulBiasEpilogues(specs);
		});
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
	case BinaryOp::Max:
		return broadcast ? "litenn_max_broadcast_f32" : "litenn_max_f32";
	case BinaryOp::Min:
		return broadcast ? "litenn_min_broadcast_f32" : "litenn_min_f32";
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
	case UnaryOp::Exp:
		return "litenn_exp_f32";
	case UnaryOp::Log:
		return "litenn_log_f32";
	case UnaryOp::Sin:
		return "litenn_sin_f32";
	case UnaryOp::Cos:
		return "litenn_cos_f32";
	default:
		throw std::runtime_error("Unsupported CUDA native unary op");
	}
}

std::string_view CUDANativeReduceF32KernelName(ReduceOp op)
{
	switch (op)
	{
	case ReduceOp::Sum:
		return "litenn_reduce_sum_f32";
	case ReduceOp::Mean:
		return "litenn_reduce_mean_f32";
	case ReduceOp::Max:
		return "litenn_reduce_max_f32";
	}
	throw std::runtime_error("Unsupported CUDA native reduce op");
}

std::string CUDANativeConcatF32KernelName(std::size_t inputIndex)
{
	return std::format("litenn_concat_f32_input_{}", inputIndex);
}

std::string_view CUDANativeSliceF32KernelName()
{
	return "litenn_slice_f32";
}

std::string_view CUDANativeMatMulBiasEpilogueF32KernelName(bool relu)
{
	return relu ? "litenn_matmul_bias_relu_epilogue_f32" : "litenn_matmul_bias_add_epilogue_f32";
}

std::string CUDANativeMatMulBiasEpilogueKernelName(DataType dtype, bool relu)
{
	if (!IsSupportedCUDANativeMatMulBiasEpilogueType(dtype))
	{
		throw std::runtime_error("Unsupported CUDA native MatMulBias epilogue dtype");
	}
	if (dtype == DataType::Float32)
	{
		return std::string(CUDANativeMatMulBiasEpilogueF32KernelName(relu));
	}
	return std::format("litenn_matmul_bias_{}_epilogue_{}", relu ? "relu" : "add",
	                   CUDANativeDataTypeShortName(dtype));
}

bool CUDANativeSupportsCast(DataType srcType, DataType dstType)
{
	return srcType != dstType && IsSupportedCUDANativeCastScalarType(srcType) &&
	       IsSupportedCUDANativeCastScalarType(dstType);
}

std::string CUDANativeCastKernelName(DataType srcType, DataType dstType)
{
	if (!CUDANativeSupportsCast(srcType, dstType))
	{
		throw std::runtime_error("Unsupported CUDA native cast dtype pair");
	}
	return std::format("litenn_cast_{}_to_{}", CUDANativeDataTypeShortName(srcType),
	                   CUDANativeDataTypeShortName(dstType));
}

std::string CUDANativeNVPTXTargetChip()
{
	return ResolveNVPTXTargetChip();
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

std::string CUDANativeReduceF32PTXFromMLIRNVPTX(const CUDANativeReduceF32CodegenSpec& spec)
{
	return EmitReduceF32PTXFromMLIRNVPTX(spec);
}

std::optional<std::string> TryCUDANativeReduceF32PTXFromMLIRNVPTX(
    const CUDANativeReduceF32CodegenSpec& spec)
{
	try
	{
		return CUDANativeReduceF32PTXFromMLIRNVPTX(spec);
	}
	catch (const std::exception&)
	{
		return std::nullopt;
	}
}

std::string CUDANativeConcatF32PTXFromMLIRNVPTX(const CUDANativeConcatF32CodegenSpec& spec)
{
	return EmitConcatF32PTXFromMLIRNVPTX(spec);
}

std::optional<std::string> TryCUDANativeConcatF32PTXFromMLIRNVPTX(
    const CUDANativeConcatF32CodegenSpec& spec)
{
	try
	{
		return CUDANativeConcatF32PTXFromMLIRNVPTX(spec);
	}
	catch (const std::exception&)
	{
		return std::nullopt;
	}
}

std::string CUDANativeSliceF32PTXFromMLIRNVPTX(const CUDANativeSliceF32CodegenSpec& spec)
{
	return EmitSliceF32PTXFromMLIRNVPTX(spec);
}

std::optional<std::string> TryCUDANativeSliceF32PTXFromMLIRNVPTX(
    const CUDANativeSliceF32CodegenSpec& spec)
{
	try
	{
		return CUDANativeSliceF32PTXFromMLIRNVPTX(spec);
	}
	catch (const std::exception&)
	{
		return std::nullopt;
	}
}

std::string CUDANativeCastPTXFromMLIRNVPTX(const CUDANativeCastCodegenSpec& spec)
{
	return EmitCastPTXFromMLIRNVPTX(spec);
}

std::optional<std::string> TryCUDANativeCastPTXFromMLIRNVPTX(const CUDANativeCastCodegenSpec& spec)
{
	try
	{
		return CUDANativeCastPTXFromMLIRNVPTX(spec);
	}
	catch (const std::exception&)
	{
		return std::nullopt;
	}
}

std::string CUDANativeMatMulBiasEpilogueF32PTXFromMLIRNVPTX(
    const CUDANativeMatMulBiasEpilogueF32CodegenSpec& spec)
{
	return EmitMatMulBiasEpilogueF32PTXFromMLIRNVPTX(spec);
}

std::string CUDANativeMatMulBiasEpiloguePTXFromMLIRNVPTX(
	const CUDANativeMatMulBiasEpilogueCodegenSpec& spec)
{
	return EmitMatMulBiasEpiloguePTXFromMLIRNVPTX(spec);
}

std::optional<std::string> TryCUDANativeMatMulBiasEpilogueF32PTXFromMLIRNVPTX(
    const CUDANativeMatMulBiasEpilogueF32CodegenSpec& spec)
{
	try
	{
		return CUDANativeMatMulBiasEpilogueF32PTXFromMLIRNVPTX(spec);
	}
	catch (const std::exception&)
	{
		return std::nullopt;
	}
}

std::optional<std::string> TryCUDANativeMatMulBiasEpiloguePTXFromMLIRNVPTX(
    const CUDANativeMatMulBiasEpilogueCodegenSpec& spec)
{
	try
	{
		return CUDANativeMatMulBiasEpiloguePTXFromMLIRNVPTX(spec);
	}
	catch (const std::exception& ex)
	{
		if (std::getenv("LITENN_CUDA_NATIVE_CODEGEN_TRACE"))
		{
			llvm::errs() << "CUDA native MatMulBias epilogue codegen failed: " << ex.what() << '\n';
		}
		return std::nullopt;
	}
}

std::string CUDANativeMatMulBiasEpiloguesF32PTXFromMLIRNVPTX(
    std::span<const CUDANativeMatMulBiasEpilogueF32CodegenSpec> specs)
{
	return EmitMatMulBiasEpiloguesF32PTXFromMLIRNVPTX(specs);
}

std::string CUDANativeMatMulBiasEpiloguesPTXFromMLIRNVPTX(
    std::span<const CUDANativeMatMulBiasEpilogueCodegenSpec> specs)
{
	return EmitMatMulBiasEpiloguesPTXFromMLIRNVPTX(specs);
}

std::optional<std::string> TryCUDANativeMatMulBiasEpiloguesF32PTXFromMLIRNVPTX(
    std::span<const CUDANativeMatMulBiasEpilogueF32CodegenSpec> specs)
{
	try
	{
		return CUDANativeMatMulBiasEpiloguesF32PTXFromMLIRNVPTX(specs);
	}
	catch (const std::exception& ex)
	{
		if (std::getenv("LITENN_CUDA_NATIVE_CODEGEN_TRACE"))
		{
			llvm::errs() << "CUDA native MatMulBias epilogue set codegen failed: " << ex.what() << '\n';
		}
		return std::nullopt;
	}
}

std::optional<std::string> TryCUDANativeMatMulBiasEpiloguesPTXFromMLIRNVPTX(
    std::span<const CUDANativeMatMulBiasEpilogueCodegenSpec> specs)
{
	try
	{
		return CUDANativeMatMulBiasEpiloguesPTXFromMLIRNVPTX(specs);
	}
	catch (const std::exception& ex)
	{
		if (std::getenv("LITENN_CUDA_NATIVE_CODEGEN_TRACE"))
		{
			llvm::errs() << "CUDA native MatMulBias epilogue set codegen failed: " << ex.what() << '\n';
		}
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
