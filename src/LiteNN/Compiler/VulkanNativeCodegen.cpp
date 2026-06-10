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
#include "mlir/IR/SymbolTable.h"
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

		mlir::Type SPIRVScalarType(mlir::OpBuilder& builder, DataType dtype)
		{
			switch (dtype)
			{
			case DataType::Float32:
				return builder.getF32Type();
			case DataType::Int32:
				return builder.getI32Type();
			default:
				throw std::runtime_error("Unsupported Vulkan native SPIR-V scalar dtype");
			}
		}

		std::uint32_t SPIRVScalarByteSize(DataType dtype)
		{
			switch (dtype)
			{
			case DataType::Float32:
				return sizeof(float);
			case DataType::Int32:
				return sizeof(std::int32_t);
			default:
				throw std::runtime_error("Unsupported Vulkan native SPIR-V scalar dtype");
			}
		}

		mlir::spirv::StructType CreateStorageBufferStruct(mlir::OpBuilder& builder, mlir::Type elementType,
		                                                  std::uint32_t stride)
		{
			auto arrayType = mlir::spirv::RuntimeArrayType::get(elementType, stride);
			llvm::SmallVector<mlir::Type, 1> members{ arrayType };
			llvm::SmallVector<mlir::spirv::StructType::OffsetInfo, 1> offsets{ 0 };
			return mlir::spirv::StructType::get(members, offsets);
		}

		mlir::spirv::StructType CreateF32StorageBufferStruct(mlir::OpBuilder& builder)
		{
			return CreateStorageBufferStruct(builder, builder.getF32Type(), sizeof(float));
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
		                                            mlir::Type elementType,
		                                            mlir::spirv::GlobalVariableOp buffer,
		                                            mlir::Value index)
		{
			auto zero = EmitI32Constant(builder, loc, 0);
			auto pointer = builder.create<mlir::spirv::AddressOfOp>(loc, buffer).getPointer();
			auto elementPointerType =
			    mlir::spirv::PointerType::get(elementType, mlir::spirv::StorageClass::StorageBuffer);
			return builder
			    .create<mlir::spirv::AccessChainOp>(loc, elementPointerType, pointer,
			                                        mlir::ValueRange{ zero, index })
			    .getComponentPtr();
		}

		mlir::Value EmitF32StorageBufferElementPointer(mlir::OpBuilder& builder,
		                                               mlir::Location loc,
		                                               mlir::spirv::GlobalVariableOp buffer,
		                                               mlir::Value index)
		{
			return EmitStorageBufferElementPointer(builder, loc, builder.getF32Type(), buffer, index);
		}

		mlir::OwningOpRef<mlir::spirv::ModuleOp> BuildSameShapeUnaryF32SPIRVModule(UnaryOp op,
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
			auto input = CreateStorageBuffer(moduleBuilder, loc, bufferStruct, "input", 0);
			auto out = CreateStorageBuffer(moduleBuilder, loc, bufferStruct, "out", 1);
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
			auto inputValue = moduleBuilder
			                      .create<mlir::spirv::LoadOp>(
			                          loc, moduleBuilder.getF32Type(),
			                          EmitF32StorageBufferElementPointer(moduleBuilder, loc, input, index), nullptr,
			                          nullptr)
			                      .getValue();

			mlir::Value result;
			switch (op)
			{
			case UnaryOp::Negate:
				result = moduleBuilder.create<mlir::spirv::FNegateOp>(loc, inputValue).getResult();
				break;
			case UnaryOp::Abs:
				result = moduleBuilder.create<mlir::spirv::GLFAbsOp>(loc, inputValue).getResult();
				break;
			case UnaryOp::Sqrt:
				result = moduleBuilder.create<mlir::spirv::GLSqrtOp>(loc, inputValue).getResult();
				break;
			case UnaryOp::Exp:
				result = moduleBuilder.create<mlir::spirv::GLExpOp>(loc, inputValue).getResult();
				break;
			case UnaryOp::Log:
				result = moduleBuilder.create<mlir::spirv::GLLogOp>(loc, inputValue).getResult();
				break;
			case UnaryOp::Sin:
				result = moduleBuilder.create<mlir::spirv::GLSinOp>(loc, inputValue).getResult();
				break;
			case UnaryOp::Cos:
				result = moduleBuilder.create<mlir::spirv::GLCosOp>(loc, inputValue).getResult();
				break;
			default:
				throw std::runtime_error("Unsupported Vulkan native MLIR same-shape f32 unary op");
			}

			moduleBuilder.create<mlir::spirv::StoreOp>(
			    loc, EmitF32StorageBufferElementPointer(moduleBuilder, loc, out, index), result, nullptr, nullptr);
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
			                        EmitF32StorageBufferElementPointer(moduleBuilder, loc, lhs, index), nullptr, nullptr)
			                    .getValue();
			auto rhsValue = moduleBuilder
			                    .create<mlir::spirv::LoadOp>(
			                        loc, moduleBuilder.getF32Type(),
			                        EmitF32StorageBufferElementPointer(moduleBuilder, loc, rhs, index), nullptr, nullptr)
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
			case BinaryOp::Max:
				result = moduleBuilder.create<mlir::spirv::GLFMaxOp>(loc, lhsValue, rhsValue).getResult();
				break;
			case BinaryOp::Min:
				result = moduleBuilder.create<mlir::spirv::GLFMinOp>(loc, lhsValue, rhsValue).getResult();
				break;
			default:
				throw std::runtime_error("Unsupported Vulkan native MLIR same-shape f32 binary op");
			}

			moduleBuilder.create<mlir::spirv::StoreOp>(
			    loc, EmitF32StorageBufferElementPointer(moduleBuilder, loc, out, index), result, nullptr, nullptr);
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

		mlir::OwningOpRef<mlir::spirv::ModuleOp> BuildSameShapeCastSPIRVModule(DataType srcType,
		                                                                       DataType dstType,
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
			auto srcElementType = SPIRVScalarType(moduleBuilder, srcType);
			auto dstElementType = SPIRVScalarType(moduleBuilder, dstType);
			auto inputStruct = CreateStorageBufferStruct(moduleBuilder, srcElementType, SPIRVScalarByteSize(srcType));
			auto outStruct = CreateStorageBufferStruct(moduleBuilder, dstElementType, SPIRVScalarByteSize(dstType));
			auto input = CreateStorageBuffer(moduleBuilder, loc, inputStruct, "input", 0);
			auto out = CreateStorageBuffer(moduleBuilder, loc, outStruct, "out", 1);
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
			auto inputValue = moduleBuilder
			                      .create<mlir::spirv::LoadOp>(
			                          loc, srcElementType,
			                          EmitStorageBufferElementPointer(moduleBuilder, loc, srcElementType, input, index),
			                          nullptr, nullptr)
			                      .getValue();

			mlir::Value result;
			if (srcType == DataType::Float32 && dstType == DataType::Int32)
			{
				result = moduleBuilder.create<mlir::spirv::ConvertFToSOp>(loc, dstElementType, inputValue).getResult();
			}
			else if (srcType == DataType::Int32 && dstType == DataType::Float32)
			{
				result = moduleBuilder.create<mlir::spirv::ConvertSToFOp>(loc, dstElementType, inputValue).getResult();
			}
			else
			{
				throw std::runtime_error("Unsupported Vulkan native MLIR same-shape cast");
			}

			moduleBuilder.create<mlir::spirv::StoreOp>(
			    loc, EmitStorageBufferElementPointer(moduleBuilder, loc, dstElementType, out, index), result, nullptr,
			    nullptr);
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

		std::int32_t IntegerAttrToI32(mlir::Attribute attr)
		{
			auto integer = mlir::dyn_cast<mlir::IntegerAttr>(attr);
			if (!integer)
			{
				throw std::runtime_error("Vulkan SPIR-V execution mode parameter must be an integer");
			}
			return static_cast<std::int32_t>(integer.getInt());
		}

		void ValidateVulkanStorageBufferGlobal(mlir::spirv::GlobalVariableOp global)
		{
			if (!global.getDescriptorSet() || !global.getBinding())
			{
				throw std::runtime_error("Vulkan SPIR-V storage buffer global requires descriptor set and binding");
			}
			if (*global.getDescriptorSet() != 0)
			{
				throw std::runtime_error("Vulkan SPIR-V storage buffer globals must use descriptor set 0");
			}
			auto pointerType = mlir::dyn_cast<mlir::spirv::PointerType>(global.getType());
			if (!pointerType || !mlir::isa<mlir::spirv::StructType>(pointerType.getPointeeType()))
			{
				throw std::runtime_error("Vulkan SPIR-V storage buffer global must point at a block struct");
			}
		}

		void ValidateVulkanInputGlobal(mlir::spirv::GlobalVariableOp global)
		{
			auto builtin = global->getAttrOfType<mlir::StringAttr>(
			    mlir::spirv::SPIRVDialect::getAttributeName(mlir::spirv::Decoration::BuiltIn));
			if (!builtin || builtin.getValue() != "GlobalInvocationId")
			{
				const auto actual = builtin ? builtin.getValue().str() : std::string("<none>");
				throw std::runtime_error("Vulkan SPIR-V input globals must be known builtins, got " + actual);
			}
		}

		void ValidateVulkanGlobal(mlir::spirv::GlobalVariableOp global)
		{
			switch (global.storageClass())
			{
			case mlir::spirv::StorageClass::StorageBuffer:
				ValidateVulkanStorageBufferGlobal(global);
				break;
			case mlir::spirv::StorageClass::Input:
				ValidateVulkanInputGlobal(global);
				break;
			default:
				throw std::runtime_error("Vulkan SPIR-V module contains an unsupported global storage class");
			}
		}

		void ValidateVulkanEntryPoint(mlir::spirv::ModuleOp module, mlir::spirv::EntryPointOp entryPoint)
		{
			if (entryPoint.getExecutionModel() != mlir::spirv::ExecutionModel::GLCompute)
			{
				throw std::runtime_error("Vulkan SPIR-V module must use a GLCompute entry point");
			}
			if (!module.lookupSymbol<mlir::spirv::FuncOp>(entryPoint.getFnAttr()))
			{
				throw std::runtime_error("Vulkan SPIR-V entry point references a missing function");
			}
			for (auto attr : entryPoint.getInterface())
			{
				auto symbol = mlir::dyn_cast<mlir::FlatSymbolRefAttr>(attr);
				if (!symbol)
				{
					throw std::runtime_error("Vulkan SPIR-V entry point interface must use symbol refs");
				}
				auto global = module.lookupSymbol<mlir::spirv::GlobalVariableOp>(symbol);
				if (!global)
				{
					throw std::runtime_error("Vulkan SPIR-V entry point references a missing interface global");
				}
				ValidateVulkanGlobal(global);
			}
		}

		void ValidateVulkanExecutionMode(mlir::spirv::ExecutionModeOp mode)
		{
			if (mode.getExecutionMode() != mlir::spirv::ExecutionMode::LocalSize)
			{
				throw std::runtime_error("Vulkan SPIR-V module contains an unsupported execution mode");
			}
			auto values = mode.getValues();
			if (values.size() != 3)
			{
				throw std::runtime_error("Vulkan SPIR-V LocalSize execution mode requires three dimensions");
			}
			for (auto value : values)
			{
				if (IntegerAttrToI32(value) <= 0)
				{
					throw std::runtime_error("Vulkan SPIR-V LocalSize dimensions must be positive");
				}
			}
		}

		void ValidateVulkanShaderModule(mlir::spirv::ModuleOp module)
		{
			if (module.getAddressingModel() != mlir::spirv::AddressingModel::Logical ||
			    module.getMemoryModel() != mlir::spirv::MemoryModel::GLSL450)
			{
				throw std::runtime_error("Vulkan SPIR-V module must use Logical addressing and GLSL450 memory model");
			}
			auto vce = module.getVceTriple();
			if (!vce)
			{
				throw std::runtime_error("Vulkan SPIR-V module must declare a version/capability/extension triple");
			}
			bool hasShaderCapability = false;
			for (auto capability : vce->getCapabilities())
			{
				hasShaderCapability |= capability == mlir::spirv::Capability::Shader;
			}
			if (!hasShaderCapability)
			{
				throw std::runtime_error("Vulkan SPIR-V module must declare the Shader capability");
			}

			std::size_t entryPointCount = 0;
			std::size_t executionModeCount = 0;
			module.walk([&](mlir::spirv::GlobalVariableOp global) { ValidateVulkanGlobal(global); });
			module.walk([&](mlir::spirv::EntryPointOp entryPoint) {
				++entryPointCount;
				ValidateVulkanEntryPoint(module, entryPoint);
			});
			module.walk([&](mlir::spirv::ExecutionModeOp mode) {
				++executionModeCount;
				ValidateVulkanExecutionMode(mode);
			});
			if (entryPointCount != 1)
			{
				throw std::runtime_error("Vulkan SPIR-V module must contain exactly one entry point");
			}
			if (executionModeCount != 1)
			{
				throw std::runtime_error("Vulkan SPIR-V module must contain exactly one execution mode");
			}
		}

		VulkanNativeGeneratedSPIRV SerializeSameShapeUnaryF32SPIRV(UnaryOp op)
		{
			mlir::MLIRContext context;
			context.getOrLoadDialect<mlir::spirv::SPIRVDialect>();

			auto module = BuildSameShapeUnaryF32SPIRVModule(op, context);
			ValidateVulkanShaderModule(module.get());

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

		VulkanNativeGeneratedSPIRV SerializeSameShapeBinaryF32SPIRV(BinaryOp op)
		{
			mlir::MLIRContext context;
			context.getOrLoadDialect<mlir::spirv::SPIRVDialect>();

			auto module = BuildSameShapeBinaryF32SPIRVModule(op, context);
			ValidateVulkanShaderModule(module.get());

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

		VulkanNativeGeneratedSPIRV SerializeSameShapeCastSPIRV(DataType srcType, DataType dstType)
		{
			mlir::MLIRContext context;
			context.getOrLoadDialect<mlir::spirv::SPIRVDialect>();

			auto module = BuildSameShapeCastSPIRVModule(srcType, dstType, context);
			ValidateVulkanShaderModule(module.get());

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

	bool VulkanNativeSupportsSameShapeUnaryF32(UnaryOp op)
	{
		switch (op)
		{
		case UnaryOp::Negate:
		case UnaryOp::Abs:
		case UnaryOp::Sqrt:
		case UnaryOp::Exp:
		case UnaryOp::Log:
		case UnaryOp::Sin:
		case UnaryOp::Cos:
			return true;
		default:
			return false;
		}
	}

	VulkanNativeGeneratedSPIRV VulkanNativeSameShapeUnaryF32SPIRV(UnaryOp op)
	{
		if (!VulkanNativeSupportsSameShapeUnaryF32(op))
		{
			throw std::runtime_error("Unsupported Vulkan native same-shape f32 unary op");
		}
		return SerializeSameShapeUnaryF32SPIRV(op);
	}

	bool VulkanNativeSupportsSameShapeBinaryF32(BinaryOp op)
	{
		switch (op)
		{
		case BinaryOp::Add:
		case BinaryOp::Subtract:
		case BinaryOp::Multiply:
		case BinaryOp::Divide:
		case BinaryOp::Max:
		case BinaryOp::Min:
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

	bool VulkanNativeSupportsSameShapeCast(DataType srcType, DataType dstType)
	{
		return (srcType == DataType::Float32 && dstType == DataType::Int32) ||
		       (srcType == DataType::Int32 && dstType == DataType::Float32);
	}

	VulkanNativeGeneratedSPIRV VulkanNativeSameShapeCastSPIRV(DataType srcType, DataType dstType)
	{
		if (!VulkanNativeSupportsSameShapeCast(srcType, dstType))
		{
			throw std::runtime_error("Unsupported Vulkan native same-shape cast");
		}
		return SerializeSameShapeCastSPIRV(srcType, dstType);
	}
} // namespace LiteNN
