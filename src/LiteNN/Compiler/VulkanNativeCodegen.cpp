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
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/Support/raw_ostream.h"

#include <array>
#include <limits>
#include <span>
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
			    .create<mlir::spirv::ConstantOp>(loc, type, builder.getIntegerAttr(type, llvm::APInt(32, value)))
			    .getResult();
		}

		mlir::Value EmitF32Constant(mlir::OpBuilder& builder, mlir::Location loc, float value)
		{
			auto type = builder.getF32Type();
			return builder.create<mlir::spirv::ConstantOp>(loc, type, builder.getFloatAttr(type, value)).getResult();
		}

		mlir::Type SPIRVScalarType(mlir::OpBuilder& builder, DataType dtype)
		{
			switch (dtype)
			{
			case DataType::Float32:
				return builder.getF32Type();
			case DataType::Float16:
				return builder.getF16Type();
			case DataType::Int32:
				return builder.getI32Type();
			case DataType::Int8:
			case DataType::UInt8:
				return builder.getIntegerType(8);
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
			case DataType::Float16:
				return sizeof(std::uint16_t);
			case DataType::Int32:
				return sizeof(std::int32_t);
			case DataType::Int8:
				return sizeof(std::int8_t);
			case DataType::UInt8:
				return sizeof(std::uint8_t);
			default:
				throw std::runtime_error("Unsupported Vulkan native SPIR-V scalar dtype");
			}
		}

		bool VulkanSPIRVScalarUses8BitStorage(DataType dtype)
		{
			return dtype == DataType::Int8 || dtype == DataType::UInt8;
		}

		bool VulkanSPIRVScalarUses16BitStorage(DataType dtype)
		{
			return dtype == DataType::Float16;
		}

		bool VulkanSPIRVScalarIsFloat(DataType dtype)
		{
			return dtype == DataType::Float32 || dtype == DataType::Float16;
		}

		bool VulkanSPIRVScalarIsSignedInteger(DataType dtype)
		{
			return dtype == DataType::Int32 || dtype == DataType::Int8;
		}

		bool VulkanSPIRVScalarIsUnsignedInteger(DataType dtype)
		{
			return dtype == DataType::UInt8;
		}

		bool VulkanSPIRVScalarIsInteger(DataType dtype)
		{
			return VulkanSPIRVScalarIsSignedInteger(dtype) || VulkanSPIRVScalarIsUnsignedInteger(dtype);
		}

		mlir::spirv::VerCapExtAttr MakeVulkanShaderVCE(mlir::MLIRContext& context,
		                                               std::span<const DataType> storageTypes)
		{
			llvm::SmallVector<mlir::spirv::Capability, 4> capabilities{ mlir::spirv::Capability::Shader };
			llvm::SmallVector<mlir::spirv::Extension, 2> extensions;
			const auto addCapability = [&](mlir::spirv::Capability capability) {
				if (!llvm::is_contained(capabilities, capability))
				{
					capabilities.push_back(capability);
				}
			};
			const auto addExtension = [&](mlir::spirv::Extension extension) {
				if (!llvm::is_contained(extensions, extension))
				{
					extensions.push_back(extension);
				}
			};

			for (const auto dtype : storageTypes)
			{
				if (VulkanSPIRVScalarUses8BitStorage(dtype))
				{
					addCapability(mlir::spirv::Capability::Int8);
					addCapability(mlir::spirv::Capability::StorageBuffer8BitAccess);
					addExtension(mlir::spirv::Extension::SPV_KHR_8bit_storage);
				}
				if (VulkanSPIRVScalarUses16BitStorage(dtype))
				{
					addCapability(mlir::spirv::Capability::Float16);
					addCapability(mlir::spirv::Capability::StorageBuffer16BitAccess);
					addExtension(mlir::spirv::Extension::SPV_KHR_16bit_storage);
				}
			}

			return mlir::spirv::VerCapExtAttr::get(mlir::spirv::Version::V_1_0, capabilities, extensions, &context);
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

		mlir::spirv::GlobalVariableOp CreateStorageBuffer(mlir::OpBuilder& builder, mlir::Location loc,
		                                                  mlir::Type structType, llvm::StringRef name, unsigned binding)
		{
			auto pointerType = mlir::spirv::PointerType::get(structType, mlir::spirv::StorageClass::StorageBuffer);
			return builder.create<mlir::spirv::GlobalVariableOp>(loc, pointerType, name, 0, binding);
		}

		mlir::Value EmitGlobalInvocationIndex(mlir::OpBuilder& builder, mlir::Location loc,
		                                      mlir::spirv::GlobalVariableOp globalInvocationId)
		{
			auto zero = EmitI32Constant(builder, loc, 0);
			auto pointer = builder.create<mlir::spirv::AddressOfOp>(loc, globalInvocationId).getPointer();
			auto indexPointerType =
			    mlir::spirv::PointerType::get(builder.getI32Type(), mlir::spirv::StorageClass::Input);
			auto indexPointer =
			    builder.create<mlir::spirv::AccessChainOp>(loc, indexPointerType, pointer, mlir::ValueRange{ zero })
			        .getComponentPtr();
			return builder.create<mlir::spirv::LoadOp>(loc, builder.getI32Type(), indexPointer, nullptr, nullptr)
			    .getValue();
		}

		mlir::Value EmitStorageBufferElementPointer(mlir::OpBuilder& builder, mlir::Location loc,
		                                            mlir::Type elementType, mlir::spirv::GlobalVariableOp buffer,
		                                            mlir::Value index)
		{
			auto zero = EmitI32Constant(builder, loc, 0);
			auto pointer = builder.create<mlir::spirv::AddressOfOp>(loc, buffer).getPointer();
			auto elementPointerType =
			    mlir::spirv::PointerType::get(elementType, mlir::spirv::StorageClass::StorageBuffer);
			return builder
			    .create<mlir::spirv::AccessChainOp>(loc, elementPointerType, pointer, mlir::ValueRange{ zero, index })
			    .getComponentPtr();
		}

		mlir::Value EmitF32StorageBufferElementPointer(mlir::OpBuilder& builder, mlir::Location loc,
		                                               mlir::spirv::GlobalVariableOp buffer, mlir::Value index)
		{
			return EmitStorageBufferElementPointer(builder, loc, builder.getF32Type(), buffer, index);
		}

		mlir::Value EmitElementwiseInBounds(mlir::OpBuilder& builder, mlir::Location loc, mlir::Value index,
		                                    std::uint32_t elementCount)
		{
			auto bound = EmitI32Constant(builder, loc, elementCount);
			return builder.create<mlir::spirv::ULessThanOp>(loc, index, bound).getResult();
		}

		mlir::OwningOpRef<mlir::spirv::ModuleOp>
		BuildSameShapeUnaryF32SPIRVModule(UnaryOp op, std::uint32_t elementCount, mlir::MLIRContext& context)
		{
			mlir::OpBuilder builder(&context);
			const auto loc = mlir::UnknownLoc::get(&context);

			mlir::OperationState state(loc, mlir::spirv::ModuleOp::getOperationName());
			state.addAttribute("addressing_model", builder.getAttr<mlir::spirv::AddressingModelAttr>(
			                                           mlir::spirv::AddressingModel::Logical));
			state.addAttribute("memory_model",
			                   builder.getAttr<mlir::spirv::MemoryModelAttr>(mlir::spirv::MemoryModel::GLSL450));
			state.addAttribute("vce_triple", MakeVulkanShaderVCE(context, std::span<const DataType>{}));
			mlir::spirv::ModuleOp::build(builder, state);
			auto module = mlir::cast<mlir::spirv::ModuleOp>(mlir::Operation::create(state));

			mlir::OpBuilder moduleBuilder(module.getRegion());
			auto bufferStruct = CreateF32StorageBufferStruct(moduleBuilder);
			auto input = CreateStorageBuffer(moduleBuilder, loc, bufferStruct, "input", 0);
			auto out = CreateStorageBuffer(moduleBuilder, loc, bufferStruct, "out", 1);
			auto globalInvocationType = mlir::spirv::PointerType::get(
			    mlir::VectorType::get({ 3 }, moduleBuilder.getI32Type()), mlir::spirv::StorageClass::Input);
			auto globalInvocationId = moduleBuilder.create<mlir::spirv::GlobalVariableOp>(
			    loc, globalInvocationType, "__builtin_var_GlobalInvocationId",
			    mlir::spirv::BuiltIn::GlobalInvocationId);

			auto funcType = moduleBuilder.getFunctionType(mlir::TypeRange{}, mlir::TypeRange{});
			auto func = moduleBuilder.create<mlir::spirv::FuncOp>(loc, kEntryPointName, funcType);
			auto* entry = moduleBuilder.createBlock(&func.getBody());
			moduleBuilder.setInsertionPointToStart(entry);

			auto index = EmitGlobalInvocationIndex(moduleBuilder, loc, globalInvocationId);
			auto inBounds = EmitElementwiseInBounds(moduleBuilder, loc, index, elementCount);
			mlir::spirv::SelectionOp::createIfThen(
			    loc, inBounds,
			    [&](mlir::OpBuilder& bodyBuilder) {
				    auto inputValue =
				        bodyBuilder
				            .create<mlir::spirv::LoadOp>(
				                loc, bodyBuilder.getF32Type(),
				                EmitF32StorageBufferElementPointer(bodyBuilder, loc, input, index), nullptr, nullptr)
				            .getValue();

				    mlir::Value result;
				    switch (op)
				    {
				    case UnaryOp::Negate:
					    result = bodyBuilder.create<mlir::spirv::FNegateOp>(loc, inputValue).getResult();
					    break;
				    case UnaryOp::Abs:
					    result = bodyBuilder.create<mlir::spirv::GLFAbsOp>(loc, inputValue).getResult();
					    break;
				    case UnaryOp::Sqrt:
					    result = bodyBuilder.create<mlir::spirv::GLSqrtOp>(loc, inputValue).getResult();
					    break;
				    case UnaryOp::Exp:
					    result = bodyBuilder.create<mlir::spirv::GLExpOp>(loc, inputValue).getResult();
					    break;
				    case UnaryOp::Log:
					    result = bodyBuilder.create<mlir::spirv::GLLogOp>(loc, inputValue).getResult();
					    break;
				    case UnaryOp::Sin:
					    result = bodyBuilder.create<mlir::spirv::GLSinOp>(loc, inputValue).getResult();
					    break;
				    case UnaryOp::Cos:
					    result = bodyBuilder.create<mlir::spirv::GLCosOp>(loc, inputValue).getResult();
					    break;
				    default:
					    throw std::runtime_error("Unsupported Vulkan native MLIR same-shape f32 unary op");
				    }

				    bodyBuilder.create<mlir::spirv::StoreOp>(
				        loc, EmitF32StorageBufferElementPointer(bodyBuilder, loc, out, index), result, nullptr,
				        nullptr);
			    },
			    moduleBuilder);
			moduleBuilder.create<mlir::spirv::ReturnOp>(loc);

			moduleBuilder.setInsertionPointAfter(func);
			moduleBuilder.create<mlir::spirv::EntryPointOp>(
			    loc, mlir::spirv::ExecutionModel::GLCompute, func,
			    llvm::ArrayRef<mlir::Attribute>{ mlir::FlatSymbolRefAttr::get(globalInvocationId) });
			moduleBuilder.create<mlir::spirv::ExecutionModeOp>(
			    loc, func, mlir::spirv::ExecutionMode::LocalSize,
			    llvm::ArrayRef<int32_t>{ static_cast<int32_t>(kVulkanNativeElementwiseWorkgroupSize), 1, 1 });

			if (mlir::failed(mlir::verify(module)))
			{
				throw std::runtime_error("Generated Vulkan native MLIR SPIR-V module verification failed");
			}
			return mlir::OwningOpRef<mlir::spirv::ModuleOp>(module);
		}

		mlir::OwningOpRef<mlir::spirv::ModuleOp>
		BuildSameShapeBinaryF32SPIRVModule(BinaryOp op, std::uint32_t elementCount, mlir::MLIRContext& context)
		{
			mlir::OpBuilder builder(&context);
			const auto loc = mlir::UnknownLoc::get(&context);

			mlir::OperationState state(loc, mlir::spirv::ModuleOp::getOperationName());
			state.addAttribute("addressing_model", builder.getAttr<mlir::spirv::AddressingModelAttr>(
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
			auto globalInvocationType = mlir::spirv::PointerType::get(
			    mlir::VectorType::get({ 3 }, moduleBuilder.getI32Type()), mlir::spirv::StorageClass::Input);
			auto globalInvocationId = moduleBuilder.create<mlir::spirv::GlobalVariableOp>(
			    loc, globalInvocationType, "__builtin_var_GlobalInvocationId",
			    mlir::spirv::BuiltIn::GlobalInvocationId);

			auto funcType = moduleBuilder.getFunctionType(mlir::TypeRange{}, mlir::TypeRange{});
			auto func = moduleBuilder.create<mlir::spirv::FuncOp>(loc, kEntryPointName, funcType);
			auto* entry = moduleBuilder.createBlock(&func.getBody());
			moduleBuilder.setInsertionPointToStart(entry);

			auto index = EmitGlobalInvocationIndex(moduleBuilder, loc, globalInvocationId);
			auto inBounds = EmitElementwiseInBounds(moduleBuilder, loc, index, elementCount);
			mlir::spirv::SelectionOp::createIfThen(
			    loc, inBounds,
			    [&](mlir::OpBuilder& bodyBuilder) {
				    auto lhsValue =
				        bodyBuilder
				            .create<mlir::spirv::LoadOp>(
				                loc, bodyBuilder.getF32Type(),
				                EmitF32StorageBufferElementPointer(bodyBuilder, loc, lhs, index), nullptr, nullptr)
				            .getValue();
				    auto rhsValue =
				        bodyBuilder
				            .create<mlir::spirv::LoadOp>(
				                loc, bodyBuilder.getF32Type(),
				                EmitF32StorageBufferElementPointer(bodyBuilder, loc, rhs, index), nullptr, nullptr)
				            .getValue();

				    mlir::Value result;
				    switch (op)
				    {
				    case BinaryOp::Add:
					    result = bodyBuilder.create<mlir::spirv::FAddOp>(loc, lhsValue, rhsValue).getResult();
					    break;
				    case BinaryOp::Subtract:
					    result = bodyBuilder.create<mlir::spirv::FSubOp>(loc, lhsValue, rhsValue).getResult();
					    break;
				    case BinaryOp::Multiply:
					    result = bodyBuilder.create<mlir::spirv::FMulOp>(loc, lhsValue, rhsValue).getResult();
					    break;
				    case BinaryOp::Divide:
					    result = bodyBuilder.create<mlir::spirv::FDivOp>(loc, lhsValue, rhsValue).getResult();
					    break;
				    case BinaryOp::Max:
					    result = bodyBuilder.create<mlir::spirv::GLFMaxOp>(loc, lhsValue, rhsValue).getResult();
					    break;
				    case BinaryOp::Min:
					    result = bodyBuilder.create<mlir::spirv::GLFMinOp>(loc, lhsValue, rhsValue).getResult();
					    break;
				    default:
					    throw std::runtime_error("Unsupported Vulkan native MLIR same-shape f32 binary op");
				    }

				    bodyBuilder.create<mlir::spirv::StoreOp>(
				        loc, EmitF32StorageBufferElementPointer(bodyBuilder, loc, out, index), result, nullptr,
				        nullptr);
			    },
			    moduleBuilder);
			moduleBuilder.create<mlir::spirv::ReturnOp>(loc);

			moduleBuilder.setInsertionPointAfter(func);
			moduleBuilder.create<mlir::spirv::EntryPointOp>(
			    loc, mlir::spirv::ExecutionModel::GLCompute, func,
			    llvm::ArrayRef<mlir::Attribute>{ mlir::FlatSymbolRefAttr::get(globalInvocationId) });
			moduleBuilder.create<mlir::spirv::ExecutionModeOp>(
			    loc, func, mlir::spirv::ExecutionMode::LocalSize,
			    llvm::ArrayRef<int32_t>{ static_cast<int32_t>(kVulkanNativeElementwiseWorkgroupSize), 1, 1 });

			if (mlir::failed(mlir::verify(module)))
			{
				throw std::runtime_error("Generated Vulkan native MLIR SPIR-V module verification failed");
			}
			return mlir::OwningOpRef<mlir::spirv::ModuleOp>(module);
		}

		mlir::OwningOpRef<mlir::spirv::ModuleOp> BuildMatMulF32SPIRVModule(std::uint32_t m, std::uint32_t k,
		                                                                   std::uint32_t n, mlir::MLIRContext& context)
		{
			mlir::OpBuilder builder(&context);
			const auto loc = mlir::UnknownLoc::get(&context);
			const auto outputElementCount = m * n;

			mlir::OperationState state(loc, mlir::spirv::ModuleOp::getOperationName());
			state.addAttribute("addressing_model", builder.getAttr<mlir::spirv::AddressingModelAttr>(
			                                           mlir::spirv::AddressingModel::Logical));
			state.addAttribute("memory_model",
			                   builder.getAttr<mlir::spirv::MemoryModelAttr>(mlir::spirv::MemoryModel::GLSL450));
			state.addAttribute("vce_triple", MakeVulkanShaderVCE(context, std::span<const DataType>{}));
			mlir::spirv::ModuleOp::build(builder, state);
			auto module = mlir::cast<mlir::spirv::ModuleOp>(mlir::Operation::create(state));

			mlir::OpBuilder moduleBuilder(module.getRegion());
			auto bufferStruct = CreateF32StorageBufferStruct(moduleBuilder);
			auto lhs = CreateStorageBuffer(moduleBuilder, loc, bufferStruct, "lhs", 0);
			auto rhs = CreateStorageBuffer(moduleBuilder, loc, bufferStruct, "rhs", 1);
			auto out = CreateStorageBuffer(moduleBuilder, loc, bufferStruct, "out", 2);
			auto globalInvocationType = mlir::spirv::PointerType::get(
			    mlir::VectorType::get({ 3 }, moduleBuilder.getI32Type()), mlir::spirv::StorageClass::Input);
			auto globalInvocationId = moduleBuilder.create<mlir::spirv::GlobalVariableOp>(
			    loc, globalInvocationType, "__builtin_var_GlobalInvocationId",
			    mlir::spirv::BuiltIn::GlobalInvocationId);

			auto funcType = moduleBuilder.getFunctionType(mlir::TypeRange{}, mlir::TypeRange{});
			auto func = moduleBuilder.create<mlir::spirv::FuncOp>(loc, kEntryPointName, funcType);
			auto* entry = moduleBuilder.createBlock(&func.getBody());
			moduleBuilder.setInsertionPointToStart(entry);

			auto index = EmitGlobalInvocationIndex(moduleBuilder, loc, globalInvocationId);
			auto inBounds = EmitElementwiseInBounds(moduleBuilder, loc, index, outputElementCount);
			mlir::spirv::SelectionOp::createIfThen(
			    loc, inBounds,
			    [&](mlir::OpBuilder& bodyBuilder) {
				    auto nValue = EmitI32Constant(bodyBuilder, loc, n);
				    auto row = bodyBuilder.create<mlir::spirv::UDivOp>(loc, index, nValue).getResult();
				    auto col = bodyBuilder.create<mlir::spirv::UModOp>(loc, index, nValue).getResult();
				    auto rowBase =
				        bodyBuilder.create<mlir::spirv::IMulOp>(loc, row, EmitI32Constant(bodyBuilder, loc, k))
				            .getResult();
				    auto sum = EmitF32Constant(bodyBuilder, loc, 0.0f);

				    for (std::uint32_t kk = 0; kk < k; ++kk)
				    {
					    auto lhsIndex =
					        bodyBuilder.create<mlir::spirv::IAddOp>(loc, rowBase, EmitI32Constant(bodyBuilder, loc, kk))
					            .getResult();
					    auto rhsIndex =
					        bodyBuilder.create<mlir::spirv::IAddOp>(loc, EmitI32Constant(bodyBuilder, loc, kk * n), col)
					            .getResult();
					    auto lhsValue = bodyBuilder
					                        .create<mlir::spirv::LoadOp>(
					                            loc, bodyBuilder.getF32Type(),
					                            EmitF32StorageBufferElementPointer(bodyBuilder, loc, lhs, lhsIndex),
					                            nullptr, nullptr)
					                        .getValue();
					    auto rhsValue = bodyBuilder
					                        .create<mlir::spirv::LoadOp>(
					                            loc, bodyBuilder.getF32Type(),
					                            EmitF32StorageBufferElementPointer(bodyBuilder, loc, rhs, rhsIndex),
					                            nullptr, nullptr)
					                        .getValue();
					    auto product = bodyBuilder.create<mlir::spirv::FMulOp>(loc, lhsValue, rhsValue).getResult();
					    sum = bodyBuilder.create<mlir::spirv::FAddOp>(loc, sum, product).getResult();
				    }

				    bodyBuilder.create<mlir::spirv::StoreOp>(
				        loc, EmitF32StorageBufferElementPointer(bodyBuilder, loc, out, index), sum, nullptr, nullptr);
			    },
			    moduleBuilder);
			moduleBuilder.create<mlir::spirv::ReturnOp>(loc);

			moduleBuilder.setInsertionPointAfter(func);
			moduleBuilder.create<mlir::spirv::EntryPointOp>(
			    loc, mlir::spirv::ExecutionModel::GLCompute, func,
			    llvm::ArrayRef<mlir::Attribute>{ mlir::FlatSymbolRefAttr::get(globalInvocationId) });
			moduleBuilder.create<mlir::spirv::ExecutionModeOp>(
			    loc, func, mlir::spirv::ExecutionMode::LocalSize,
			    llvm::ArrayRef<int32_t>{ static_cast<int32_t>(kVulkanNativeMatMulWorkgroupSize), 1, 1 });

			if (mlir::failed(mlir::verify(module)))
			{
				throw std::runtime_error("Generated Vulkan native MLIR SPIR-V MatMul module verification failed");
			}
			return mlir::OwningOpRef<mlir::spirv::ModuleOp>(module);
		}

		mlir::OwningOpRef<mlir::spirv::ModuleOp> BuildMatMulBiasF32SPIRVModule(std::uint32_t m, std::uint32_t k,
		                                                                       std::uint32_t n, std::uint32_t biasRows,
		                                                                       bool relu, mlir::MLIRContext& context)
		{
			mlir::OpBuilder builder(&context);
			const auto loc = mlir::UnknownLoc::get(&context);
			const auto outputElementCount = m * n;

			mlir::OperationState state(loc, mlir::spirv::ModuleOp::getOperationName());
			state.addAttribute("addressing_model", builder.getAttr<mlir::spirv::AddressingModelAttr>(
			                                           mlir::spirv::AddressingModel::Logical));
			state.addAttribute("memory_model",
			                   builder.getAttr<mlir::spirv::MemoryModelAttr>(mlir::spirv::MemoryModel::GLSL450));
			state.addAttribute("vce_triple", MakeVulkanShaderVCE(context, std::span<const DataType>{}));
			mlir::spirv::ModuleOp::build(builder, state);
			auto module = mlir::cast<mlir::spirv::ModuleOp>(mlir::Operation::create(state));

			mlir::OpBuilder moduleBuilder(module.getRegion());
			auto bufferStruct = CreateF32StorageBufferStruct(moduleBuilder);
			auto lhs = CreateStorageBuffer(moduleBuilder, loc, bufferStruct, "lhs", 0);
			auto rhs = CreateStorageBuffer(moduleBuilder, loc, bufferStruct, "rhs", 1);
			auto bias = CreateStorageBuffer(moduleBuilder, loc, bufferStruct, "bias", 2);
			auto out = CreateStorageBuffer(moduleBuilder, loc, bufferStruct, "out", 3);
			auto globalInvocationType = mlir::spirv::PointerType::get(
			    mlir::VectorType::get({ 3 }, moduleBuilder.getI32Type()), mlir::spirv::StorageClass::Input);
			auto globalInvocationId = moduleBuilder.create<mlir::spirv::GlobalVariableOp>(
			    loc, globalInvocationType, "__builtin_var_GlobalInvocationId",
			    mlir::spirv::BuiltIn::GlobalInvocationId);

			auto funcType = moduleBuilder.getFunctionType(mlir::TypeRange{}, mlir::TypeRange{});
			auto func = moduleBuilder.create<mlir::spirv::FuncOp>(loc, kEntryPointName, funcType);
			auto* entry = moduleBuilder.createBlock(&func.getBody());
			moduleBuilder.setInsertionPointToStart(entry);

			auto index = EmitGlobalInvocationIndex(moduleBuilder, loc, globalInvocationId);
			auto inBounds = EmitElementwiseInBounds(moduleBuilder, loc, index, outputElementCount);
			mlir::spirv::SelectionOp::createIfThen(
			    loc, inBounds,
			    [&](mlir::OpBuilder& bodyBuilder) {
				    auto nValue = EmitI32Constant(bodyBuilder, loc, n);
				    auto row = bodyBuilder.create<mlir::spirv::UDivOp>(loc, index, nValue).getResult();
				    auto col = bodyBuilder.create<mlir::spirv::UModOp>(loc, index, nValue).getResult();
				    auto rowBase =
				        bodyBuilder.create<mlir::spirv::IMulOp>(loc, row, EmitI32Constant(bodyBuilder, loc, k))
				            .getResult();
				    mlir::Value biasIndex = col;
				    if (biasRows != 1)
				    {
					    auto biasRowBase = bodyBuilder.create<mlir::spirv::IMulOp>(loc, row, nValue).getResult();
					    biasIndex = bodyBuilder.create<mlir::spirv::IAddOp>(loc, biasRowBase, col).getResult();
				    }
				    auto sum =
				        bodyBuilder
				            .create<mlir::spirv::LoadOp>(
				                loc, bodyBuilder.getF32Type(),
				                EmitF32StorageBufferElementPointer(bodyBuilder, loc, bias, biasIndex), nullptr, nullptr)
				            .getValue();

				    for (std::uint32_t kk = 0; kk < k; ++kk)
				    {
					    auto lhsIndex =
					        bodyBuilder.create<mlir::spirv::IAddOp>(loc, rowBase, EmitI32Constant(bodyBuilder, loc, kk))
					            .getResult();
					    auto rhsIndex =
					        bodyBuilder.create<mlir::spirv::IAddOp>(loc, EmitI32Constant(bodyBuilder, loc, kk * n), col)
					            .getResult();
					    auto lhsValue = bodyBuilder
					                        .create<mlir::spirv::LoadOp>(
					                            loc, bodyBuilder.getF32Type(),
					                            EmitF32StorageBufferElementPointer(bodyBuilder, loc, lhs, lhsIndex),
					                            nullptr, nullptr)
					                        .getValue();
					    auto rhsValue = bodyBuilder
					                        .create<mlir::spirv::LoadOp>(
					                            loc, bodyBuilder.getF32Type(),
					                            EmitF32StorageBufferElementPointer(bodyBuilder, loc, rhs, rhsIndex),
					                            nullptr, nullptr)
					                        .getValue();
					    auto product = bodyBuilder.create<mlir::spirv::FMulOp>(loc, lhsValue, rhsValue).getResult();
					    sum = bodyBuilder.create<mlir::spirv::FAddOp>(loc, sum, product).getResult();
				    }
				    if (relu)
				    {
					    sum =
					        bodyBuilder.create<mlir::spirv::GLFMaxOp>(loc, sum, EmitF32Constant(bodyBuilder, loc, 0.0f))
					            .getResult();
				    }

				    bodyBuilder.create<mlir::spirv::StoreOp>(
				        loc, EmitF32StorageBufferElementPointer(bodyBuilder, loc, out, index), sum, nullptr, nullptr);
			    },
			    moduleBuilder);
			moduleBuilder.create<mlir::spirv::ReturnOp>(loc);

			moduleBuilder.setInsertionPointAfter(func);
			moduleBuilder.create<mlir::spirv::EntryPointOp>(
			    loc, mlir::spirv::ExecutionModel::GLCompute, func,
			    llvm::ArrayRef<mlir::Attribute>{ mlir::FlatSymbolRefAttr::get(globalInvocationId) });
			moduleBuilder.create<mlir::spirv::ExecutionModeOp>(
			    loc, func, mlir::spirv::ExecutionMode::LocalSize,
			    llvm::ArrayRef<int32_t>{ static_cast<int32_t>(kVulkanNativeMatMulWorkgroupSize), 1, 1 });

			if (mlir::failed(mlir::verify(module)))
			{
				throw std::runtime_error("Generated Vulkan native MLIR SPIR-V MatMulBias module verification failed");
			}
			return mlir::OwningOpRef<mlir::spirv::ModuleOp>(module);
		}

		mlir::OwningOpRef<mlir::spirv::ModuleOp> BuildSameShapeCastSPIRVModule(DataType srcType, DataType dstType,
		                                                                       std::uint32_t elementCount,
		                                                                       mlir::MLIRContext& context)
		{
			mlir::OpBuilder builder(&context);
			const auto loc = mlir::UnknownLoc::get(&context);

			mlir::OperationState state(loc, mlir::spirv::ModuleOp::getOperationName());
			state.addAttribute("addressing_model", builder.getAttr<mlir::spirv::AddressingModelAttr>(
			                                           mlir::spirv::AddressingModel::Logical));
			state.addAttribute("memory_model",
			                   builder.getAttr<mlir::spirv::MemoryModelAttr>(mlir::spirv::MemoryModel::GLSL450));
			const std::array storageTypes{ srcType, dstType };
			state.addAttribute("vce_triple", MakeVulkanShaderVCE(context, storageTypes));
			mlir::spirv::ModuleOp::build(builder, state);
			auto module = mlir::cast<mlir::spirv::ModuleOp>(mlir::Operation::create(state));

			mlir::OpBuilder moduleBuilder(module.getRegion());
			auto srcElementType = SPIRVScalarType(moduleBuilder, srcType);
			auto dstElementType = SPIRVScalarType(moduleBuilder, dstType);
			auto inputStruct = CreateStorageBufferStruct(moduleBuilder, srcElementType, SPIRVScalarByteSize(srcType));
			auto outStruct = CreateStorageBufferStruct(moduleBuilder, dstElementType, SPIRVScalarByteSize(dstType));
			auto input = CreateStorageBuffer(moduleBuilder, loc, inputStruct, "input", 0);
			auto out = CreateStorageBuffer(moduleBuilder, loc, outStruct, "out", 1);
			auto globalInvocationType = mlir::spirv::PointerType::get(
			    mlir::VectorType::get({ 3 }, moduleBuilder.getI32Type()), mlir::spirv::StorageClass::Input);
			auto globalInvocationId = moduleBuilder.create<mlir::spirv::GlobalVariableOp>(
			    loc, globalInvocationType, "__builtin_var_GlobalInvocationId",
			    mlir::spirv::BuiltIn::GlobalInvocationId);

			auto funcType = moduleBuilder.getFunctionType(mlir::TypeRange{}, mlir::TypeRange{});
			auto func = moduleBuilder.create<mlir::spirv::FuncOp>(loc, kEntryPointName, funcType);
			auto* entry = moduleBuilder.createBlock(&func.getBody());
			moduleBuilder.setInsertionPointToStart(entry);

			auto index = EmitGlobalInvocationIndex(moduleBuilder, loc, globalInvocationId);
			auto inBounds = EmitElementwiseInBounds(moduleBuilder, loc, index, elementCount);
			mlir::spirv::SelectionOp::createIfThen(
			    loc, inBounds,
			    [&](mlir::OpBuilder& bodyBuilder) {
				    auto inputValue =
				        bodyBuilder
				            .create<mlir::spirv::LoadOp>(
				                loc, srcElementType,
				                EmitStorageBufferElementPointer(bodyBuilder, loc, srcElementType, input, index),
				                nullptr, nullptr)
				            .getValue();

				    mlir::Value result;
				    if (VulkanSPIRVScalarIsFloat(srcType) && VulkanSPIRVScalarIsFloat(dstType))
				    {
					    result = srcElementType == dstElementType
					                 ? inputValue
					                 : bodyBuilder.create<mlir::spirv::FConvertOp>(loc, dstElementType, inputValue)
					                       .getResult();
				    }
				    else if (VulkanSPIRVScalarIsFloat(srcType) && VulkanSPIRVScalarIsSignedInteger(dstType))
				    {
					    result =
					        bodyBuilder.create<mlir::spirv::ConvertFToSOp>(loc, dstElementType, inputValue).getResult();
				    }
				    else if (VulkanSPIRVScalarIsFloat(srcType) && VulkanSPIRVScalarIsUnsignedInteger(dstType))
				    {
					    result =
					        bodyBuilder.create<mlir::spirv::ConvertFToUOp>(loc, dstElementType, inputValue).getResult();
				    }
				    else if (VulkanSPIRVScalarIsSignedInteger(srcType) && VulkanSPIRVScalarIsFloat(dstType))
				    {
					    result =
					        bodyBuilder.create<mlir::spirv::ConvertSToFOp>(loc, dstElementType, inputValue).getResult();
				    }
				    else if (VulkanSPIRVScalarIsUnsignedInteger(srcType) && VulkanSPIRVScalarIsFloat(dstType))
				    {
					    result =
					        bodyBuilder.create<mlir::spirv::ConvertUToFOp>(loc, dstElementType, inputValue).getResult();
				    }
				    else if (VulkanSPIRVScalarIsInteger(srcType) && VulkanSPIRVScalarIsInteger(dstType))
				    {
					    if (srcElementType == dstElementType)
					    {
						    result = inputValue;
					    }
					    else if (VulkanSPIRVScalarIsUnsignedInteger(srcType) ||
					             VulkanSPIRVScalarIsUnsignedInteger(dstType))
					    {
						    result = bodyBuilder.create<mlir::spirv::UConvertOp>(loc, dstElementType, inputValue)
						                 .getResult();
					    }
					    else
					    {
						    result = bodyBuilder.create<mlir::spirv::SConvertOp>(loc, dstElementType, inputValue)
						                 .getResult();
					    }
				    }
				    else
				    {
					    throw std::runtime_error("Unsupported Vulkan native MLIR same-shape cast");
				    }

				    bodyBuilder.create<mlir::spirv::StoreOp>(
				        loc, EmitStorageBufferElementPointer(bodyBuilder, loc, dstElementType, out, index), result,
				        nullptr, nullptr);
			    },
			    moduleBuilder);
			moduleBuilder.create<mlir::spirv::ReturnOp>(loc);

			moduleBuilder.setInsertionPointAfter(func);
			moduleBuilder.create<mlir::spirv::EntryPointOp>(
			    loc, mlir::spirv::ExecutionModel::GLCompute, func,
			    llvm::ArrayRef<mlir::Attribute>{ mlir::FlatSymbolRefAttr::get(globalInvocationId) });
			moduleBuilder.create<mlir::spirv::ExecutionModeOp>(
			    loc, func, mlir::spirv::ExecutionMode::LocalSize,
			    llvm::ArrayRef<int32_t>{ static_cast<int32_t>(kVulkanNativeElementwiseWorkgroupSize), 1, 1 });

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

		VulkanNativeGeneratedSPIRV SerializeSameShapeUnaryF32SPIRV(UnaryOp op, std::uint32_t elementCount)
		{
			mlir::MLIRContext context;
			context.getOrLoadDialect<mlir::spirv::SPIRVDialect>();

			auto module = BuildSameShapeUnaryF32SPIRVModule(op, elementCount, context);
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

		VulkanNativeGeneratedSPIRV SerializeSameShapeBinaryF32SPIRV(BinaryOp op, std::uint32_t elementCount)
		{
			mlir::MLIRContext context;
			context.getOrLoadDialect<mlir::spirv::SPIRVDialect>();

			auto module = BuildSameShapeBinaryF32SPIRVModule(op, elementCount, context);
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

		VulkanNativeGeneratedSPIRV SerializeMatMulF32SPIRV(std::uint32_t m, std::uint32_t k, std::uint32_t n)
		{
			mlir::MLIRContext context;
			context.getOrLoadDialect<mlir::spirv::SPIRVDialect>();

			auto module = BuildMatMulF32SPIRVModule(m, k, n, context);
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
				throw std::runtime_error("Failed to serialize generated Vulkan native MLIR SPIR-V MatMul module");
			}

			return VulkanNativeGeneratedSPIRV{
				.words = std::vector<std::uint32_t>(binary.begin(), binary.end()),
				.mlir = mlirStream.str(),
			};
		}

		VulkanNativeGeneratedSPIRV SerializeMatMulBiasF32SPIRV(std::uint32_t m, std::uint32_t k, std::uint32_t n,
		                                                       std::uint32_t biasRows, bool relu)
		{
			mlir::MLIRContext context;
			context.getOrLoadDialect<mlir::spirv::SPIRVDialect>();

			auto module = BuildMatMulBiasF32SPIRVModule(m, k, n, biasRows, relu, context);
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
				throw std::runtime_error("Failed to serialize generated Vulkan native MLIR SPIR-V MatMulBias module");
			}

			return VulkanNativeGeneratedSPIRV{
				.words = std::vector<std::uint32_t>(binary.begin(), binary.end()),
				.mlir = mlirStream.str(),
			};
		}

		VulkanNativeGeneratedSPIRV SerializeSameShapeCastSPIRV(DataType srcType, DataType dstType,
		                                                       std::uint32_t elementCount)
		{
			mlir::MLIRContext context;
			context.getOrLoadDialect<mlir::spirv::SPIRVDialect>();

			auto module = BuildSameShapeCastSPIRVModule(srcType, dstType, elementCount, context);
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

	VulkanNativeGeneratedSPIRV VulkanNativeSameShapeUnaryF32SPIRV(UnaryOp op, std::uint32_t elementCount)
	{
		if (!VulkanNativeSupportsSameShapeUnaryF32(op))
		{
			throw std::runtime_error("Unsupported Vulkan native same-shape f32 unary op");
		}
		if (elementCount == 0)
		{
			throw std::runtime_error("Vulkan native same-shape unary requires a non-empty static shape");
		}
		return SerializeSameShapeUnaryF32SPIRV(op, elementCount);
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

	VulkanNativeGeneratedSPIRV VulkanNativeSameShapeBinaryF32SPIRV(BinaryOp op, std::uint32_t elementCount)
	{
		if (!VulkanNativeSupportsSameShapeBinaryF32(op))
		{
			throw std::runtime_error("Unsupported Vulkan native same-shape f32 binary op");
		}
		if (elementCount == 0)
		{
			throw std::runtime_error("Vulkan native same-shape binary requires a non-empty static shape");
		}
		return SerializeSameShapeBinaryF32SPIRV(op, elementCount);
	}

	bool VulkanNativeSupportsMatMulF32(std::uint32_t m, std::uint32_t k, std::uint32_t n)
	{
		if (m == 0 || k == 0 || n == 0)
		{
			return false;
		}
		const auto max = static_cast<std::uint64_t>(std::numeric_limits<std::uint32_t>::max());
		return static_cast<std::uint64_t>(m) * k <= max && static_cast<std::uint64_t>(k) * n <= max &&
		       static_cast<std::uint64_t>(m) * n <= max;
	}

	VulkanNativeGeneratedSPIRV VulkanNativeMatMulF32SPIRV(std::uint32_t m, std::uint32_t k, std::uint32_t n)
	{
		if (!VulkanNativeSupportsMatMulF32(m, k, n))
		{
			throw std::runtime_error("Vulkan native MatMul requires non-empty dimensions with uint32_t-sized buffers");
		}
		return SerializeMatMulF32SPIRV(m, k, n);
	}

	bool VulkanNativeSupportsMatMulBiasF32(std::uint32_t m, std::uint32_t k, std::uint32_t n, std::uint32_t biasRows)
	{
		if (biasRows != 1 && biasRows != m)
		{
			return false;
		}
		if (!VulkanNativeSupportsMatMulF32(m, k, n))
		{
			return false;
		}
		const auto max = static_cast<std::uint64_t>(std::numeric_limits<std::uint32_t>::max());
		return static_cast<std::uint64_t>(biasRows) * n <= max;
	}

	VulkanNativeGeneratedSPIRV VulkanNativeMatMulBiasF32SPIRV(std::uint32_t m, std::uint32_t k, std::uint32_t n,
	                                                          std::uint32_t biasRows, bool relu)
	{
		if (!VulkanNativeSupportsMatMulBiasF32(m, k, n, biasRows))
		{
			throw std::runtime_error(
			    "Vulkan native MatMulBias requires non-empty dimensions and bias rows equal to 1 or M");
		}
		return SerializeMatMulBiasF32SPIRV(m, k, n, biasRows, relu);
	}

	bool VulkanNativeSupportsSameShapeCast(DataType srcType, DataType dstType)
	{
		if (srcType == dstType)
		{
			return false;
		}
		return (VulkanSPIRVScalarIsFloat(srcType) || VulkanSPIRVScalarIsInteger(srcType)) &&
		       (VulkanSPIRVScalarIsFloat(dstType) || VulkanSPIRVScalarIsInteger(dstType));
	}

	VulkanNativeGeneratedSPIRV VulkanNativeSameShapeCastSPIRV(DataType srcType, DataType dstType,
	                                                          std::uint32_t elementCount)
	{
		if (!VulkanNativeSupportsSameShapeCast(srcType, dstType))
		{
			throw std::runtime_error("Unsupported Vulkan native same-shape cast");
		}
		if (elementCount == 0)
		{
			throw std::runtime_error("Vulkan native same-shape cast requires a non-empty static shape");
		}
		return SerializeSameShapeCastSPIRV(srcType, dstType, elementCount);
	}
} // namespace LiteNN
