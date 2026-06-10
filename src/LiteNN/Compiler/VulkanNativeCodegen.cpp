#include "VulkanNativeCodegen.h"

#include "mlir/Dialect/SPIRV/IR/SPIRVAttributes.h"
#include "mlir/Dialect/SPIRV/IR/SPIRVDialect.h"
#include "mlir/Dialect/SPIRV/IR/SPIRVEnums.h"
#include "mlir/Dialect/SPIRV/IR/SPIRVOps.h"
#include "mlir/Dialect/SPIRV/IR/SPIRVTypes.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/Diagnostics.h"
#include "mlir/IR/Location.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/IR/Verifier.h"
#include "mlir/Support/LLVM.h"
#include "mlir/Target/SPIRV/Serialization.h"
#include "llvm/ADT/APInt.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/Support/raw_ostream.h"

#include <stdexcept>
#include <utility>

namespace LiteNN
{
	namespace
	{
		constexpr llvm::StringLiteral kEntryPointName = "main";

		mlir::Value EmitI32Constant(mlir::OpBuilder& builder, mlir::Location loc, std::uint32_t value)
		{
			auto type = builder.getI32Type();
			return builder
			    .create<mlir::spirv::ConstantOp>(loc, type,
			                                     builder.getIntegerAttr(type, llvm::APInt(32, value)))
			    .getResult();
		}

		mlir::spirv::StructType CreateF32StorageBufferStruct(mlir::OpBuilder& builder)
		{
			auto arrayType = mlir::spirv::RuntimeArrayType::get(builder.getF32Type(), sizeof(float));
			llvm::SmallVector<mlir::Type, 1> members{ arrayType };
			llvm::SmallVector<mlir::spirv::StructType::OffsetInfo, 1> offsets{ 0 };
			return mlir::spirv::StructType::get(members, offsets);
		}

		mlir::spirv::GlobalVariableOp CreateStorageBuffer(mlir::OpBuilder& builder,
		                                                  mlir::Location loc,
		                                                  mlir::Type structType,
		                                                  llvm::StringRef name,
		                                                  unsigned binding)
		{
			auto pointerType = mlir::spirv::PointerType::get(structType, mlir::spirv::StorageClass::StorageBuffer);
			return builder.create<mlir::spirv::GlobalVariableOp>(loc, pointerType, name, 0, binding);
		}

		mlir::Value EmitGlobalInvocationIndex(mlir::OpBuilder& builder,
		                                      mlir::Location loc,
		                                      mlir::spirv::GlobalVariableOp globalInvocationId)
		{
			auto zero = EmitI32Constant(builder, loc, 0);
			auto pointer = builder.create<mlir::spirv::AddressOfOp>(loc, globalInvocationId).getPointer();
			auto indexPointerType = mlir::spirv::PointerType::get(builder.getI32Type(), mlir::spirv::StorageClass::Input);
			auto indexPointer = builder
			                        .create<mlir::spirv::AccessChainOp>(loc, indexPointerType, pointer,
			                                                            mlir::ValueRange{ zero })
			                        .getComponentPtr();
			return builder.create<mlir::spirv::LoadOp>(loc, builder.getI32Type(), indexPointer, nullptr, nullptr)
			    .getValue();
		}

		mlir::Value EmitStorageBufferElementPointer(mlir::OpBuilder& builder,
		                                            mlir::Location loc,
		                                            mlir::spirv::GlobalVariableOp buffer,
		                                            mlir::Value index)
		{
			auto zero = EmitI32Constant(builder, loc, 0);
			auto pointer = builder.create<mlir::spirv::AddressOfOp>(loc, buffer).getPointer();
			auto elementPointerType =
			    mlir::spirv::PointerType::get(builder.getF32Type(), mlir::spirv::StorageClass::StorageBuffer);
			return builder
			    .create<mlir::spirv::AccessChainOp>(loc, elementPointerType, pointer,
			                                        mlir::ValueRange{ zero, index })
			    .getComponentPtr();
		}

		mlir::OwningOpRef<mlir::spirv::ModuleOp> BuildSameShapeBinaryF32SPIRVModule(BinaryOp op,
		                                                                            mlir::MLIRContext& context)
		{
			mlir::OpBuilder builder(&context);
			const auto loc = mlir::UnknownLoc::get(&context);

			mlir::OperationState state(loc, mlir::spirv::ModuleOp::getOperationName());
			state.addAttribute("addressing_model",
			                   builder.getAttr<mlir::spirv::AddressingModelAttr>(
			                       mlir::spirv::AddressingModel::Logical));
			state.addAttribute("memory_model",
			                   builder.getAttr<mlir::spirv::MemoryModelAttr>(mlir::spirv::MemoryModel::GLSL450));
			state.addAttribute("vce_triple",
			                   mlir::spirv::VerCapExtAttr::get(
			                       mlir::spirv::Version::V_1_0,
			                       llvm::ArrayRef<mlir::spirv::Capability>{ mlir::spirv::Capability::Shader },
			                       llvm::ArrayRef<mlir::spirv::Extension>{}, &context));
			mlir::spirv::ModuleOp::build(builder, state);
			auto module = mlir::cast<mlir::spirv::ModuleOp>(mlir::Operation::create(state));

			mlir::OpBuilder moduleBuilder(module.getRegion());
			auto bufferStruct = CreateF32StorageBufferStruct(moduleBuilder);
			auto lhs = CreateStorageBuffer(moduleBuilder, loc, bufferStruct, "lhs", 0);
			auto rhs = CreateStorageBuffer(moduleBuilder, loc, bufferStruct, "rhs", 1);
			auto out = CreateStorageBuffer(moduleBuilder, loc, bufferStruct, "out", 2);
			auto globalInvocationType =
			    mlir::spirv::PointerType::get(mlir::VectorType::get({ 3 }, moduleBuilder.getI32Type()),
			                                  mlir::spirv::StorageClass::Input);
			auto globalInvocationId = moduleBuilder.create<mlir::spirv::GlobalVariableOp>(
			    loc, globalInvocationType, "__builtin_var_GlobalInvocationId", mlir::spirv::BuiltIn::GlobalInvocationId);

			auto funcType = moduleBuilder.getFunctionType(mlir::TypeRange{}, mlir::TypeRange{});
			auto func = moduleBuilder.create<mlir::spirv::FuncOp>(loc, kEntryPointName, funcType);
			auto* entry = moduleBuilder.createBlock(&func.getBody());
			moduleBuilder.setInsertionPointToStart(entry);

			auto index = EmitGlobalInvocationIndex(moduleBuilder, loc, globalInvocationId);
			auto lhsValue = moduleBuilder
			                    .create<mlir::spirv::LoadOp>(
			                        loc, moduleBuilder.getF32Type(),
			                        EmitStorageBufferElementPointer(moduleBuilder, loc, lhs, index), nullptr, nullptr)
			                    .getValue();
			auto rhsValue = moduleBuilder
			                    .create<mlir::spirv::LoadOp>(
			                        loc, moduleBuilder.getF32Type(),
			                        EmitStorageBufferElementPointer(moduleBuilder, loc, rhs, index), nullptr, nullptr)
			                    .getValue();

			mlir::Value result;
			switch (op)
			{
			case BinaryOp::Add:
				result = moduleBuilder.create<mlir::spirv::FAddOp>(loc, lhsValue, rhsValue).getResult();
				break;
			case BinaryOp::Subtract:
				result = moduleBuilder.create<mlir::spirv::FSubOp>(loc, lhsValue, rhsValue).getResult();
				break;
			case BinaryOp::Multiply:
				result = moduleBuilder.create<mlir::spirv::FMulOp>(loc, lhsValue, rhsValue).getResult();
				break;
			case BinaryOp::Divide:
				result = moduleBuilder.create<mlir::spirv::FDivOp>(loc, lhsValue, rhsValue).getResult();
				break;
			default:
				throw std::runtime_error("Unsupported Vulkan native MLIR same-shape f32 binary op");
			}

			moduleBuilder.create<mlir::spirv::StoreOp>(
			    loc, EmitStorageBufferElementPointer(moduleBuilder, loc, out, index), result, nullptr, nullptr);
			moduleBuilder.create<mlir::spirv::ReturnOp>(loc);

			moduleBuilder.setInsertionPointAfter(func);
			moduleBuilder.create<mlir::spirv::EntryPointOp>(
			    loc, mlir::spirv::ExecutionModel::GLCompute, func,
			    llvm::ArrayRef<mlir::Attribute>{ mlir::FlatSymbolRefAttr::get(globalInvocationId) });
			moduleBuilder.create<mlir::spirv::ExecutionModeOp>(
			    loc, func, mlir::spirv::ExecutionMode::LocalSize, llvm::ArrayRef<int32_t>{ 1, 1, 1 });

			if (mlir::failed(mlir::verify(module)))
			{
				throw std::runtime_error("Generated Vulkan native MLIR SPIR-V module verification failed");
			}
			return mlir::OwningOpRef<mlir::spirv::ModuleOp>(module);
		}

		VulkanNativeGeneratedSPIRV SerializeSameShapeBinaryF32SPIRV(BinaryOp op)
		{
			mlir::MLIRContext context;
			context.getOrLoadDialect<mlir::spirv::SPIRVDialect>();

			auto module = BuildSameShapeBinaryF32SPIRVModule(op, context);

			std::string mlirText;
			llvm::raw_string_ostream mlirStream(mlirText);
			module.get().print(mlirStream);

			llvm::SmallVector<std::uint32_t, 0> binary;
			mlir::spirv::SerializationOptions options;
			options.emitSymbolName = false;
			options.emitDebugInfo = false;
			if (mlir::failed(mlir::spirv::serialize(module.get(), binary, options)))
			{
				throw std::runtime_error("Failed to serialize generated Vulkan native MLIR SPIR-V module");
			}

			return VulkanNativeGeneratedSPIRV{
				.words = std::vector<std::uint32_t>(binary.begin(), binary.end()),
				.mlir = mlirStream.str(),
			};
		}
	} // namespace

	bool VulkanNativeSupportsSameShapeBinaryF32(BinaryOp op)
	{
		switch (op)
		{
		case BinaryOp::Add:
		case BinaryOp::Subtract:
		case BinaryOp::Multiply:
		case BinaryOp::Divide:
			return true;
		default:
			return false;
		}
	}

	VulkanNativeGeneratedSPIRV VulkanNativeSameShapeBinaryF32SPIRV(BinaryOp op)
	{
		if (!VulkanNativeSupportsSameShapeBinaryF32(op))
		{
			throw std::runtime_error("Unsupported Vulkan native same-shape f32 binary op");
		}
		return SerializeSameShapeBinaryF32SPIRV(op);
	}
} // namespace LiteNN
