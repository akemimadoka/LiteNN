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
#include <optional>
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

		std::optional<std::uint32_t> NumElementsU32(std::span<const std::size_t> shape)
		{
			std::uint64_t count = 1;
			for (const auto dim : shape)
			{
				if (dim == 0)
				{
					return std::nullopt;
				}
				count *= static_cast<std::uint64_t>(dim);
				if (count > std::numeric_limits<std::uint32_t>::max())
				{
					return std::nullopt;
				}
			}
			return static_cast<std::uint32_t>(count);
		}

		std::vector<std::size_t> ReduceOutputShape(std::span<const std::size_t> inputShape, std::size_t axis)
		{
			if (axis >= inputShape.size())
			{
				throw std::runtime_error("Vulkan native reduce axis is out of range");
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
			return outputShape;
		}

		std::optional<std::uint32_t> AxisInnerSizeU32(std::span<const std::size_t> shape, std::size_t axis)
		{
			if (axis >= shape.size())
			{
				return std::nullopt;
			}
			std::uint64_t count = 1;
			for (std::size_t i = axis + 1; i < shape.size(); ++i)
			{
				if (shape[i] == 0)
				{
					return std::nullopt;
				}
				count *= static_cast<std::uint64_t>(shape[i]);
				if (count > std::numeric_limits<std::uint32_t>::max())
				{
					return std::nullopt;
				}
			}
			return static_cast<std::uint32_t>(count);
		}

		std::string_view ReduceF32KernelName(ReduceOp op)
		{
			switch (op)
			{
			case ReduceOp::Sum:
				return "reduce_sum";
			case ReduceOp::Mean:
				return "reduce_mean";
			case ReduceOp::Max:
				return "reduce_max";
			default:
				throw std::runtime_error("Unsupported Vulkan native f32 reduce op");
			}
		}

		std::string_view NormalizationF32KernelName(NormalizationMode mode)
		{
			switch (mode)
			{
			case NormalizationMode::LayerNorm:
				return "layer_norm";
			case NormalizationMode::RMSNorm:
				return "rms_norm";
			case NormalizationMode::GroupNorm:
				return "group_norm";
			default:
				throw std::runtime_error("Unsupported Vulkan native f32 normalization mode");
			}
		}

		std::string_view Pool2DF32KernelName(PoolMode mode)
		{
			switch (mode)
			{
			case PoolMode::Max:
				return "pool2d_max";
			case PoolMode::Average:
				return "pool2d_average";
			default:
				throw std::runtime_error("Unsupported Vulkan native f32 Pool2D mode");
			}
		}

		std::string SameShapeBinaryF32KernelName(BinaryOp op)
		{
			switch (op)
			{
			case BinaryOp::Add:
				return "binary_add";
			case BinaryOp::Subtract:
				return "binary_subtract";
			case BinaryOp::Multiply:
				return "binary_multiply";
			case BinaryOp::Divide:
				return "binary_divide";
			case BinaryOp::Max:
				return "binary_max";
			case BinaryOp::Min:
				return "binary_min";
			default:
				throw std::runtime_error("Unsupported Vulkan native same-shape f32 binary op");
			}
		}

		mlir::spirv::FuncOp EmitSameShapeBinaryF32Function(
		    mlir::OpBuilder& moduleBuilder, mlir::Location loc, BinaryOp op, llvm::StringRef entryPoint,
		    std::uint32_t elementCount, mlir::spirv::GlobalVariableOp lhs, mlir::spirv::GlobalVariableOp rhs,
		    mlir::spirv::GlobalVariableOp out, mlir::spirv::GlobalVariableOp globalInvocationId)
		{
			auto funcType = moduleBuilder.getFunctionType(mlir::TypeRange{}, mlir::TypeRange{});
			auto func = moduleBuilder.create<mlir::spirv::FuncOp>(loc, entryPoint, funcType);
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
			return func;
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

			auto func = EmitSameShapeBinaryF32Function(moduleBuilder, loc, op, kEntryPointName, elementCount, lhs, rhs,
			                                           out, globalInvocationId);
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
		BuildSameShapeBinaryF32ChainSPIRVModule(std::span<const BinaryOp> ops, std::uint32_t elementCount,
		                                        mlir::MLIRContext& context)
		{
			if (ops.empty())
			{
				throw std::runtime_error("Vulkan native binary chain requires at least one op");
			}

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
			auto lhs = CreateStorageBuffer(moduleBuilder, loc, bufferStruct, "lhs", 0);
			auto rhs = CreateStorageBuffer(moduleBuilder, loc, bufferStruct, "rhs", 1);
			auto out = CreateStorageBuffer(moduleBuilder, loc, bufferStruct, "out", 2);
			auto globalInvocationType = mlir::spirv::PointerType::get(
			    mlir::VectorType::get({ 3 }, moduleBuilder.getI32Type()), mlir::spirv::StorageClass::Input);
			auto globalInvocationId = moduleBuilder.create<mlir::spirv::GlobalVariableOp>(
			    loc, globalInvocationType, "__builtin_var_GlobalInvocationId",
			    mlir::spirv::BuiltIn::GlobalInvocationId);

			std::vector<BinaryOp> emittedOps;
			for (const auto op : ops)
			{
				if (llvm::is_contained(emittedOps, op))
				{
					continue;
				}
				auto entryName = SameShapeBinaryF32KernelName(op);
				auto func = EmitSameShapeBinaryF32Function(moduleBuilder, loc, op, entryName, elementCount, lhs, rhs,
				                                           out, globalInvocationId);
				moduleBuilder.setInsertionPointAfter(func);
				moduleBuilder.create<mlir::spirv::EntryPointOp>(
				    loc, mlir::spirv::ExecutionModel::GLCompute, func,
				    llvm::ArrayRef<mlir::Attribute>{ mlir::FlatSymbolRefAttr::get(globalInvocationId) });
				moduleBuilder.create<mlir::spirv::ExecutionModeOp>(
				    loc, func, mlir::spirv::ExecutionMode::LocalSize,
				    llvm::ArrayRef<int32_t>{ static_cast<int32_t>(kVulkanNativeElementwiseWorkgroupSize), 1, 1 });
				emittedOps.push_back(op);
			}

			if (mlir::failed(mlir::verify(module)))
			{
				throw std::runtime_error("Generated Vulkan native MLIR SPIR-V binary chain module verification failed");
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

		mlir::OwningOpRef<mlir::spirv::ModuleOp> BuildReduceF32SPIRVModule(ReduceOp op,
		                                                                   std::span<const std::size_t> inputShape,
		                                                                   std::size_t axis,
		                                                                   mlir::MLIRContext& context)
		{
			const auto outputShape = ReduceOutputShape(inputShape, axis);
			const auto inputElementCount = NumElementsU32(inputShape);
			const auto outputElementCount = NumElementsU32(outputShape);
			const auto innerSize = AxisInnerSizeU32(inputShape, axis);
			if (!inputElementCount || !outputElementCount || !innerSize ||
			    inputShape[axis] > std::numeric_limits<std::uint32_t>::max())
			{
				throw std::runtime_error("Vulkan native reduce shape is too large or empty");
			}
			const auto axisSize = static_cast<std::uint32_t>(inputShape[axis]);
			if (axisSize == 0)
			{
				throw std::runtime_error("Vulkan native reduce axis size must not be zero");
			}

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
			auto func = moduleBuilder.create<mlir::spirv::FuncOp>(loc, ReduceF32KernelName(op), funcType);
			auto* entry = moduleBuilder.createBlock(&func.getBody());
			moduleBuilder.setInsertionPointToStart(entry);

			auto outputIndex = EmitGlobalInvocationIndex(moduleBuilder, loc, globalInvocationId);
			auto inBounds = EmitElementwiseInBounds(moduleBuilder, loc, outputIndex, *outputElementCount);
			mlir::spirv::SelectionOp::createIfThen(
			    loc, inBounds,
			    [&](mlir::OpBuilder& bodyBuilder) {
				    auto inner = EmitI32Constant(bodyBuilder, loc, *innerSize);
				    auto axisValue = EmitI32Constant(bodyBuilder, loc, axisSize);
				    auto outerIndex = bodyBuilder.create<mlir::spirv::UDivOp>(loc, outputIndex, inner).getResult();
				    auto innerIndex = bodyBuilder.create<mlir::spirv::UModOp>(loc, outputIndex, inner).getResult();
				    auto outerAxis =
				        bodyBuilder.create<mlir::spirv::IMulOp>(loc, outerIndex, axisValue).getResult();
				    auto base = bodyBuilder
				                    .create<mlir::spirv::IAddOp>(
				                        loc, bodyBuilder.create<mlir::spirv::IMulOp>(loc, outerAxis, inner).getResult(),
				                        innerIndex)
				                    .getResult();

				    mlir::Value accumulator;
				    for (std::uint32_t reduceIndex = 0; reduceIndex < axisSize; ++reduceIndex)
				    {
					    auto offset = bodyBuilder
					                      .create<mlir::spirv::IAddOp>(
					                          loc, base,
					                          bodyBuilder
					                              .create<mlir::spirv::IMulOp>(
					                                  loc, EmitI32Constant(bodyBuilder, loc, reduceIndex), inner)
					                              .getResult())
					                      .getResult();
					    auto value =
					        bodyBuilder
					            .create<mlir::spirv::LoadOp>(
					                loc, bodyBuilder.getF32Type(),
					                EmitF32StorageBufferElementPointer(bodyBuilder, loc, input, offset), nullptr,
					                nullptr)
					            .getValue();
					    if (reduceIndex == 0)
					    {
						    accumulator = value;
						    continue;
					    }
					    switch (op)
					    {
					    case ReduceOp::Sum:
					    case ReduceOp::Mean:
						    accumulator = bodyBuilder.create<mlir::spirv::FAddOp>(loc, accumulator, value).getResult();
						    break;
					    case ReduceOp::Max:
						    accumulator =
						        bodyBuilder.create<mlir::spirv::GLFMaxOp>(loc, accumulator, value).getResult();
						    break;
					    default:
						    throw std::runtime_error("Unsupported Vulkan native MLIR f32 reduce op");
					    }
				    }
				    if (op == ReduceOp::Mean)
				    {
					    accumulator =
					        bodyBuilder
					            .create<mlir::spirv::FMulOp>(
					                loc, accumulator,
					                EmitF32Constant(bodyBuilder, loc, 1.0f / static_cast<float>(axisSize)))
					            .getResult();
				    }
				    bodyBuilder.create<mlir::spirv::StoreOp>(
				        loc, EmitF32StorageBufferElementPointer(bodyBuilder, loc, out, outputIndex), accumulator,
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
				throw std::runtime_error("Generated Vulkan native MLIR SPIR-V Reduce module verification failed");
			}
			(void)inputElementCount;
			return mlir::OwningOpRef<mlir::spirv::ModuleOp>(module);
		}

		mlir::OwningOpRef<mlir::spirv::ModuleOp> BuildPool2DF32SPIRVModule(
		    PoolMode mode, std::span<const std::size_t> inputShape, std::span<const std::size_t> outputShape,
		    std::span<const std::size_t> kernelShape, std::span<const std::size_t> strides,
		    std::span<const std::size_t> lowPads, bool countIncludePad, mlir::MLIRContext& context)
		{
			const auto inputElementCount = NumElementsU32(inputShape);
			const auto outputElementCount = NumElementsU32(outputShape);
			if (!inputElementCount || !outputElementCount || inputShape.size() != 4 || outputShape.size() != 4 ||
			    kernelShape.size() != 2 || strides.size() != 2 || lowPads.size() != 2)
			{
				throw std::runtime_error("Vulkan native Pool2D requires static rank-4 input/output and rank-2 params");
			}
			for (const auto value : { inputShape[0], inputShape[1], inputShape[2], inputShape[3], outputShape[2],
				                      outputShape[3], kernelShape[0], kernelShape[1], strides[0], strides[1],
				                      lowPads[0], lowPads[1] })
			{
				if (value > std::numeric_limits<std::uint32_t>::max())
				{
					throw std::runtime_error("Vulkan native Pool2D shape or parameter is too large or empty");
				}
			}
			for (const auto value : { inputShape[0], inputShape[1], inputShape[2], inputShape[3], outputShape[2],
				                      outputShape[3], kernelShape[0], kernelShape[1], strides[0], strides[1] })
			{
				if (value == 0)
				{
					throw std::runtime_error("Vulkan native Pool2D shape or parameter is too large or empty");
				}
			}
			const auto channels = static_cast<std::uint32_t>(inputShape[1]);
			const auto inHeight = static_cast<std::uint32_t>(inputShape[2]);
			const auto inWidth = static_cast<std::uint32_t>(inputShape[3]);
			const auto outHeight = static_cast<std::uint32_t>(outputShape[2]);
			const auto outWidth = static_cast<std::uint32_t>(outputShape[3]);
			const auto kernelH = static_cast<std::uint32_t>(kernelShape[0]);
			const auto kernelW = static_cast<std::uint32_t>(kernelShape[1]);
			const auto strideH = static_cast<std::uint32_t>(strides[0]);
			const auto strideW = static_cast<std::uint32_t>(strides[1]);
			const auto lowPadH = static_cast<std::uint32_t>(lowPads[0]);
			const auto lowPadW = static_cast<std::uint32_t>(lowPads[1]);

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
			auto func = moduleBuilder.create<mlir::spirv::FuncOp>(loc, Pool2DF32KernelName(mode), funcType);
			auto* entry = moduleBuilder.createBlock(&func.getBody());
			moduleBuilder.setInsertionPointToStart(entry);

			auto outputIndex = EmitGlobalInvocationIndex(moduleBuilder, loc, globalInvocationId);
			auto inBounds = EmitElementwiseInBounds(moduleBuilder, loc, outputIndex, *outputElementCount);
			mlir::spirv::SelectionOp::createIfThen(
			    loc, inBounds,
			    [&](mlir::OpBuilder& bodyBuilder) {
				    auto outW = EmitI32Constant(bodyBuilder, loc, outWidth);
				    auto outH = EmitI32Constant(bodyBuilder, loc, outHeight);
				    auto channelCount = EmitI32Constant(bodyBuilder, loc, channels);
				    auto inH = EmitI32Constant(bodyBuilder, loc, inHeight);
				    auto inW = EmitI32Constant(bodyBuilder, loc, inWidth);
				    auto zero = EmitI32Constant(bodyBuilder, loc, 0);

				    auto ow = bodyBuilder.create<mlir::spirv::UModOp>(loc, outputIndex, outW).getResult();
				    auto tmp0 = bodyBuilder.create<mlir::spirv::UDivOp>(loc, outputIndex, outW).getResult();
				    auto oh = bodyBuilder.create<mlir::spirv::UModOp>(loc, tmp0, outH).getResult();
				    auto tmp1 = bodyBuilder.create<mlir::spirv::UDivOp>(loc, tmp0, outH).getResult();
				    auto channel = bodyBuilder.create<mlir::spirv::UModOp>(loc, tmp1, channelCount).getResult();
				    auto batch = bodyBuilder.create<mlir::spirv::UDivOp>(loc, tmp1, channelCount).getResult();

				    auto inPlane = bodyBuilder.create<mlir::spirv::IMulOp>(loc, inH, inW).getResult();
				    auto baseNC = bodyBuilder
				                      .create<mlir::spirv::IMulOp>(
				                          loc,
				                          bodyBuilder
				                              .create<mlir::spirv::IAddOp>(
				                                  loc,
				                                  bodyBuilder.create<mlir::spirv::IMulOp>(loc, batch, channelCount)
				                                      .getResult(),
				                                  channel)
				                              .getResult(),
				                          inPlane)
				                      .getResult();
				    auto startH = bodyBuilder
				                      .create<mlir::spirv::IMulOp>(loc, oh, EmitI32Constant(bodyBuilder, loc, strideH))
				                      .getResult();
				    auto startW = bodyBuilder
				                      .create<mlir::spirv::IMulOp>(loc, ow, EmitI32Constant(bodyBuilder, loc, strideW))
				                      .getResult();

				    auto accumulator = mode == PoolMode::Max ? EmitF32Constant(bodyBuilder, loc, -3.402823466e38f)
				                                             : EmitF32Constant(bodyBuilder, loc, 0.0f);
				    auto validCount = EmitI32Constant(bodyBuilder, loc, 0);
				    auto one = EmitI32Constant(bodyBuilder, loc, 1);
				    auto invalidMax = EmitF32Constant(bodyBuilder, loc, -3.402823466e38f);
				    auto zeroF32 = EmitF32Constant(bodyBuilder, loc, 0.0f);
				    auto lowPadHValue = EmitI32Constant(bodyBuilder, loc, lowPadH);
				    auto lowPadWValue = EmitI32Constant(bodyBuilder, loc, lowPadW);
				    auto paddedLimitH =
				        EmitI32Constant(bodyBuilder, loc, static_cast<std::uint32_t>(lowPadH + inHeight));
				    auto paddedLimitW =
				        EmitI32Constant(bodyBuilder, loc, static_cast<std::uint32_t>(lowPadW + inWidth));
				    for (std::uint32_t kh = 0; kh < kernelH; ++kh)
				    {
					    auto paddedH = bodyBuilder
					                       .create<mlir::spirv::IAddOp>(loc, startH, EmitI32Constant(bodyBuilder, loc, kh))
					                       .getResult();
					    auto validH = bodyBuilder
					                      .create<mlir::spirv::LogicalAndOp>(
					                          loc,
					                          bodyBuilder
					                              .create<mlir::spirv::UGreaterThanEqualOp>(loc, paddedH, lowPadHValue)
					                              .getResult(),
					                          bodyBuilder
					                              .create<mlir::spirv::ULessThanOp>(loc, paddedH, paddedLimitH)
					                              .getResult())
					                      .getResult();
					    auto rawIh = bodyBuilder.create<mlir::spirv::ISubOp>(loc, paddedH, lowPadHValue).getResult();
					    auto safeIh =
					        bodyBuilder.create<mlir::spirv::SelectOp>(loc, validH, rawIh, zero).getResult();
					    for (std::uint32_t kw = 0; kw < kernelW; ++kw)
					    {
						    auto paddedW = bodyBuilder
						                       .create<mlir::spirv::IAddOp>(loc, startW, EmitI32Constant(bodyBuilder, loc, kw))
						                       .getResult();
						    auto validW = bodyBuilder
						                      .create<mlir::spirv::LogicalAndOp>(
						                          loc,
						                          bodyBuilder
						                              .create<mlir::spirv::UGreaterThanEqualOp>(loc, paddedW, lowPadWValue)
						                              .getResult(),
						                          bodyBuilder
						                              .create<mlir::spirv::ULessThanOp>(loc, paddedW, paddedLimitW)
						                              .getResult())
						                      .getResult();
						    auto valid = bodyBuilder.create<mlir::spirv::LogicalAndOp>(loc, validH, validW).getResult();
						    auto rawIw = bodyBuilder.create<mlir::spirv::ISubOp>(loc, paddedW, lowPadWValue).getResult();
						    auto safeIw =
						        bodyBuilder.create<mlir::spirv::SelectOp>(loc, validW, rawIw, zero).getResult();
						    auto inputOffset = bodyBuilder
						                           .create<mlir::spirv::IAddOp>(
						                               loc, baseNC,
						                               bodyBuilder
						                                   .create<mlir::spirv::IAddOp>(
						                                       loc,
						                                       bodyBuilder.create<mlir::spirv::IMulOp>(loc, safeIh, inW)
						                                           .getResult(),
						                                       safeIw)
						                                   .getResult())
						                           .getResult();
						    auto value =
						        bodyBuilder
						            .create<mlir::spirv::LoadOp>(
						                loc, bodyBuilder.getF32Type(),
						                EmitF32StorageBufferElementPointer(bodyBuilder, loc, input, inputOffset),
						                nullptr, nullptr)
						            .getValue();
						    if (mode == PoolMode::Max)
						    {
							    auto candidate =
							        bodyBuilder.create<mlir::spirv::SelectOp>(loc, valid, value, invalidMax).getResult();
							    accumulator =
							        bodyBuilder.create<mlir::spirv::GLFMaxOp>(loc, accumulator, candidate).getResult();
							    auto increment = bodyBuilder.create<mlir::spirv::SelectOp>(loc, valid, one, zero).getResult();
							    validCount =
							        bodyBuilder.create<mlir::spirv::IAddOp>(loc, validCount, increment).getResult();
						    }
						    else
						    {
							    auto contribution =
							        bodyBuilder.create<mlir::spirv::SelectOp>(loc, valid, value, zeroF32).getResult();
							    accumulator =
							        bodyBuilder.create<mlir::spirv::FAddOp>(loc, accumulator, contribution).getResult();
							    if (countIncludePad)
							    {
								    validCount = bodyBuilder.create<mlir::spirv::IAddOp>(loc, validCount, one).getResult();
							    }
							    else
							    {
								    auto increment =
								        bodyBuilder.create<mlir::spirv::SelectOp>(loc, valid, one, zero).getResult();
								    validCount =
								        bodyBuilder.create<mlir::spirv::IAddOp>(loc, validCount, increment).getResult();
							    }
						    }
					    }
				    }
				    auto countIsZero = bodyBuilder.create<mlir::spirv::IEqualOp>(loc, validCount, zero).getResult();
				    if (mode == PoolMode::Average)
				    {
					    auto divisor =
					        bodyBuilder.create<mlir::spirv::ConvertSToFOp>(loc, bodyBuilder.getF32Type(), validCount)
					            .getResult();
					    auto safeDivisor =
					        bodyBuilder.create<mlir::spirv::SelectOp>(loc, countIsZero, EmitF32Constant(bodyBuilder, loc, 1.0f),
					                                                   divisor)
					            .getResult();
					    auto average = bodyBuilder.create<mlir::spirv::FDivOp>(loc, accumulator, safeDivisor).getResult();
					    accumulator = bodyBuilder.create<mlir::spirv::SelectOp>(loc, countIsZero, zeroF32, average).getResult();
				    }
				    else
				    {
					    accumulator =
					        bodyBuilder.create<mlir::spirv::SelectOp>(loc, countIsZero, zeroF32, accumulator).getResult();
				    }
				    bodyBuilder.create<mlir::spirv::StoreOp>(
				        loc, EmitF32StorageBufferElementPointer(bodyBuilder, loc, out, outputIndex), accumulator,
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
				throw std::runtime_error("Generated Vulkan native MLIR SPIR-V Pool2D module verification failed");
			}
			return mlir::OwningOpRef<mlir::spirv::ModuleOp>(module);
		}

		mlir::OwningOpRef<mlir::spirv::ModuleOp> BuildSoftmaxF32SPIRVModule(std::span<const std::size_t> inputShape,
		                                                                    std::size_t axis,
		                                                                    mlir::MLIRContext& context)
		{
			const auto elementCount = NumElementsU32(inputShape);
			const auto innerSize = AxisInnerSizeU32(inputShape, axis);
			if (!elementCount || !innerSize || axis >= inputShape.size() ||
			    inputShape[axis] == 0 || inputShape[axis] > std::numeric_limits<std::uint32_t>::max())
			{
				throw std::runtime_error("Vulkan native softmax shape is too large or empty");
			}
			const auto axisSize = static_cast<std::uint32_t>(inputShape[axis]);

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
			auto func = moduleBuilder.create<mlir::spirv::FuncOp>(loc, "softmax", funcType);
			auto* entry = moduleBuilder.createBlock(&func.getBody());
			moduleBuilder.setInsertionPointToStart(entry);

			auto outputIndex = EmitGlobalInvocationIndex(moduleBuilder, loc, globalInvocationId);
			auto inBounds = EmitElementwiseInBounds(moduleBuilder, loc, outputIndex, *elementCount);
			mlir::spirv::SelectionOp::createIfThen(
			    loc, inBounds,
			    [&](mlir::OpBuilder& bodyBuilder) {
				    auto inner = EmitI32Constant(bodyBuilder, loc, *innerSize);
				    auto axisValue = EmitI32Constant(bodyBuilder, loc, axisSize);
				    auto axisIndex =
				        bodyBuilder
				            .create<mlir::spirv::UModOp>(
				                loc, bodyBuilder.create<mlir::spirv::UDivOp>(loc, outputIndex, inner).getResult(),
				                axisValue)
				            .getResult();
				    auto base = bodyBuilder
				                    .create<mlir::spirv::ISubOp>(
				                        loc, outputIndex,
				                        bodyBuilder.create<mlir::spirv::IMulOp>(loc, axisIndex, inner).getResult())
				                    .getResult();

				    mlir::Value maxValue;
				    for (std::uint32_t reduceIndex = 0; reduceIndex < axisSize; ++reduceIndex)
				    {
					    auto offset = bodyBuilder
					                      .create<mlir::spirv::IAddOp>(
					                          loc, base,
					                          bodyBuilder
					                              .create<mlir::spirv::IMulOp>(
					                                  loc, EmitI32Constant(bodyBuilder, loc, reduceIndex), inner)
					                              .getResult())
					                      .getResult();
					    auto value =
					        bodyBuilder
					            .create<mlir::spirv::LoadOp>(
					                loc, bodyBuilder.getF32Type(),
					                EmitF32StorageBufferElementPointer(bodyBuilder, loc, input, offset), nullptr,
					                nullptr)
					            .getValue();
					    if (reduceIndex == 0)
					    {
						    maxValue = value;
					    }
					    else
					    {
						    maxValue = bodyBuilder.create<mlir::spirv::GLFMaxOp>(loc, maxValue, value).getResult();
					    }
				    }

				    auto sum = EmitF32Constant(bodyBuilder, loc, 0.0f);
				    for (std::uint32_t reduceIndex = 0; reduceIndex < axisSize; ++reduceIndex)
				    {
					    auto offset = bodyBuilder
					                      .create<mlir::spirv::IAddOp>(
					                          loc, base,
					                          bodyBuilder
					                              .create<mlir::spirv::IMulOp>(
					                                  loc, EmitI32Constant(bodyBuilder, loc, reduceIndex), inner)
					                              .getResult())
					                      .getResult();
					    auto value =
					        bodyBuilder
					            .create<mlir::spirv::LoadOp>(
					                loc, bodyBuilder.getF32Type(),
					                EmitF32StorageBufferElementPointer(bodyBuilder, loc, input, offset), nullptr,
					                nullptr)
					            .getValue();
					    auto shifted = bodyBuilder.create<mlir::spirv::FSubOp>(loc, value, maxValue).getResult();
					    auto expValue = bodyBuilder.create<mlir::spirv::GLExpOp>(loc, shifted).getResult();
					    sum = bodyBuilder.create<mlir::spirv::FAddOp>(loc, sum, expValue).getResult();
				    }

				    auto current =
				        bodyBuilder
				            .create<mlir::spirv::LoadOp>(
				                loc, bodyBuilder.getF32Type(),
				                EmitF32StorageBufferElementPointer(bodyBuilder, loc, input, outputIndex), nullptr,
				                nullptr)
				            .getValue();
				    auto shifted = bodyBuilder.create<mlir::spirv::FSubOp>(loc, current, maxValue).getResult();
				    auto numerator = bodyBuilder.create<mlir::spirv::GLExpOp>(loc, shifted).getResult();
				    auto probability = bodyBuilder.create<mlir::spirv::FDivOp>(loc, numerator, sum).getResult();
				    bodyBuilder.create<mlir::spirv::StoreOp>(
				        loc, EmitF32StorageBufferElementPointer(bodyBuilder, loc, out, outputIndex), probability,
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
				throw std::runtime_error("Generated Vulkan native MLIR SPIR-V Softmax module verification failed");
			}
			return mlir::OwningOpRef<mlir::spirv::ModuleOp>(module);
		}

		mlir::OwningOpRef<mlir::spirv::ModuleOp> BuildNormalizationF32SPIRVModule(
		    NormalizationMode mode, std::span<const std::size_t> inputShape, std::size_t axis, double epsilon,
		    bool hasScale, bool hasBias, std::size_t groupCount, mlir::MLIRContext& context)
		{
			const auto elementCount = NumElementsU32(inputShape);
			const auto isGroupNorm = mode == NormalizationMode::GroupNorm;
			std::uint32_t reductionSize = 0;
			std::uint32_t innerSizeValue = 1;
			std::uint32_t batchSize = 1;
			if (isGroupNorm)
			{
				if (!elementCount || inputShape.empty() || inputShape.size() > 4 || groupCount == 0 ||
				    groupCount > std::numeric_limits<std::uint32_t>::max())
				{
					throw std::runtime_error("Vulkan native GroupNorm shape or group count is invalid");
				}
				if (inputShape.size() == 4)
				{
					if (inputShape[3] == 0 || inputShape[3] > std::numeric_limits<std::uint32_t>::max())
					{
						throw std::runtime_error("Vulkan native GroupNorm batch dimension is invalid");
					}
					batchSize = static_cast<std::uint32_t>(inputShape[3]);
				}
				std::uint64_t groupedVolume = 1;
				for (std::size_t dim = 0; dim < std::min<std::size_t>(inputShape.size(), 3); ++dim)
				{
					if (inputShape[dim] == 0)
					{
						throw std::runtime_error("Vulkan native GroupNorm shape is empty");
					}
					groupedVolume *= static_cast<std::uint64_t>(inputShape[dim]);
				}
				if (groupedVolume % groupCount != 0 ||
				    groupedVolume / groupCount > std::numeric_limits<std::uint32_t>::max())
				{
					throw std::runtime_error("Vulkan native GroupNorm grouped volume is invalid");
				}
				reductionSize = static_cast<std::uint32_t>(groupedVolume / groupCount);
			}
			else
			{
				const auto innerSize = AxisInnerSizeU32(inputShape, axis);
				if (!elementCount || !innerSize || axis >= inputShape.size() || inputShape[axis] == 0 ||
				    inputShape[axis] > std::numeric_limits<std::uint32_t>::max())
				{
					throw std::runtime_error("Vulkan native normalization shape is too large or empty");
				}
				reductionSize = static_cast<std::uint32_t>(inputShape[axis]);
				innerSizeValue = *innerSize;
			}
			if (reductionSize == 0)
			{
				throw std::runtime_error("Vulkan native normalization shape is too large or empty");
			}

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
			std::uint32_t nextBinding = 1;
			std::optional<mlir::spirv::GlobalVariableOp> scale;
			std::optional<mlir::spirv::GlobalVariableOp> bias;
			if (hasScale)
			{
				scale = CreateStorageBuffer(moduleBuilder, loc, bufferStruct, "scale", nextBinding++);
			}
			if (hasBias)
			{
				bias = CreateStorageBuffer(moduleBuilder, loc, bufferStruct, "bias", nextBinding++);
			}
			auto out = CreateStorageBuffer(moduleBuilder, loc, bufferStruct, "out", nextBinding);
			auto globalInvocationType = mlir::spirv::PointerType::get(
			    mlir::VectorType::get({ 3 }, moduleBuilder.getI32Type()), mlir::spirv::StorageClass::Input);
			auto globalInvocationId = moduleBuilder.create<mlir::spirv::GlobalVariableOp>(
			    loc, globalInvocationType, "__builtin_var_GlobalInvocationId",
			    mlir::spirv::BuiltIn::GlobalInvocationId);

			auto funcType = moduleBuilder.getFunctionType(mlir::TypeRange{}, mlir::TypeRange{});
			auto func = moduleBuilder.create<mlir::spirv::FuncOp>(
			    loc, llvm::StringRef(NormalizationF32KernelName(mode).data(), NormalizationF32KernelName(mode).size()),
			    funcType);
			auto* entry = moduleBuilder.createBlock(&func.getBody());
			moduleBuilder.setInsertionPointToStart(entry);

			auto outputIndex = EmitGlobalInvocationIndex(moduleBuilder, loc, globalInvocationId);
			auto inBounds = EmitElementwiseInBounds(moduleBuilder, loc, outputIndex, *elementCount);
			mlir::spirv::SelectionOp::createIfThen(
			    loc, inBounds,
			    [&](mlir::OpBuilder& bodyBuilder) {
				    const auto loadAtOffset = [&](mlir::OpBuilder& b, mlir::Value offset) {
					    return b.create<mlir::spirv::LoadOp>(
					                loc, b.getF32Type(), EmitF32StorageBufferElementPointer(b, loc, input, offset),
					                nullptr, nullptr)
					        .getValue();
				    };
				    auto inner = EmitI32Constant(bodyBuilder, loc, innerSizeValue);
				    auto reductionValue = EmitI32Constant(bodyBuilder, loc, reductionSize);
				    auto batchValue = EmitI32Constant(bodyBuilder, loc, batchSize);
				    mlir::Value affineIndex;
				    mlir::Value base;
				    if (isGroupNorm)
				    {
					    auto batchIndex = bodyBuilder.create<mlir::spirv::UModOp>(loc, outputIndex, batchValue).getResult();
					    auto groupedIndex =
					        bodyBuilder.create<mlir::spirv::UDivOp>(loc, outputIndex, batchValue).getResult();
					    auto groupIndex =
					        bodyBuilder.create<mlir::spirv::UDivOp>(loc, groupedIndex, reductionValue).getResult();
					    auto groupBase = bodyBuilder.create<mlir::spirv::IMulOp>(loc, groupIndex, reductionValue).getResult();
					    affineIndex = groupedIndex;
					    base = bodyBuilder
					               .create<mlir::spirv::IAddOp>(
					                   loc,
					                   bodyBuilder.create<mlir::spirv::IMulOp>(loc, groupBase, batchValue).getResult(),
					                   batchIndex)
					               .getResult();
				    }
				    else
				    {
					    auto axisValue = reductionValue;
					    auto axisIndex =
					        bodyBuilder
					            .create<mlir::spirv::UModOp>(
					                loc, bodyBuilder.create<mlir::spirv::UDivOp>(loc, outputIndex, inner).getResult(),
					                axisValue)
					            .getResult();
					    affineIndex = axisIndex;
					    base = bodyBuilder
					               .create<mlir::spirv::ISubOp>(
					                   loc, outputIndex,
					                   bodyBuilder.create<mlir::spirv::IMulOp>(loc, axisIndex, inner).getResult())
					               .getResult();
				    }
				    const auto memberOffset = [&](mlir::OpBuilder& b, std::uint32_t reduceIndex) {
					    const auto step = isGroupNorm ? batchValue : inner;
					    return b.create<mlir::spirv::IAddOp>(
					                loc, base,
					                b.create<mlir::spirv::IMulOp>(loc, EmitI32Constant(b, loc, reduceIndex), step)
					                    .getResult())
					        .getResult();
				    };

				    auto mean = EmitF32Constant(bodyBuilder, loc, 0.0f);
				    if (mode == NormalizationMode::LayerNorm || mode == NormalizationMode::GroupNorm)
				    {
					    for (std::uint32_t reduceIndex = 0; reduceIndex < reductionSize; ++reduceIndex)
					    {
						    auto value = loadAtOffset(bodyBuilder, memberOffset(bodyBuilder, reduceIndex));
						    mean = bodyBuilder.create<mlir::spirv::FAddOp>(loc, mean, value).getResult();
					    }
					    mean = bodyBuilder
					               .create<mlir::spirv::FMulOp>(
					                   loc, mean,
					                   EmitF32Constant(bodyBuilder, loc, 1.0f / static_cast<float>(reductionSize)))
					               .getResult();
				    }

				    auto variance = EmitF32Constant(bodyBuilder, loc, 0.0f);
				    for (std::uint32_t reduceIndex = 0; reduceIndex < reductionSize; ++reduceIndex)
				    {
					    auto value = loadAtOffset(bodyBuilder, memberOffset(bodyBuilder, reduceIndex));
					    auto centered = (mode == NormalizationMode::LayerNorm || mode == NormalizationMode::GroupNorm)
					                        ? bodyBuilder.create<mlir::spirv::FSubOp>(loc, value, mean).getResult()
					                        : value;
					    auto squared = bodyBuilder.create<mlir::spirv::FMulOp>(loc, centered, centered).getResult();
					    variance = bodyBuilder.create<mlir::spirv::FAddOp>(loc, variance, squared).getResult();
				    }
				    variance = bodyBuilder
				                   .create<mlir::spirv::FMulOp>(
				                       loc, variance,
				                       EmitF32Constant(bodyBuilder, loc, 1.0f / static_cast<float>(reductionSize)))
				                   .getResult();
				    auto denom = bodyBuilder
				                     .create<mlir::spirv::GLSqrtOp>(
				                         loc, bodyBuilder
				                                  .create<mlir::spirv::FAddOp>(
				                                      loc, variance,
				                                      EmitF32Constant(bodyBuilder, loc, static_cast<float>(epsilon)))
				                                  .getResult())
				                     .getResult();
				    auto current =
				        bodyBuilder
				            .create<mlir::spirv::LoadOp>(
				                loc, bodyBuilder.getF32Type(),
				                EmitF32StorageBufferElementPointer(bodyBuilder, loc, input, outputIndex), nullptr,
				                nullptr)
				            .getValue();
				    auto centered = (mode == NormalizationMode::LayerNorm || mode == NormalizationMode::GroupNorm)
				                        ? bodyBuilder.create<mlir::spirv::FSubOp>(loc, current, mean).getResult()
				                        : current;
				    auto normalized = bodyBuilder.create<mlir::spirv::FDivOp>(loc, centered, denom).getResult();
				    if (scale)
				    {
					    auto scaleValue =
					        bodyBuilder
					            .create<mlir::spirv::LoadOp>(
					                loc, bodyBuilder.getF32Type(),
					                EmitF32StorageBufferElementPointer(bodyBuilder, loc, *scale, affineIndex), nullptr,
					                nullptr)
					            .getValue();
					    normalized = bodyBuilder.create<mlir::spirv::FMulOp>(loc, normalized, scaleValue).getResult();
				    }
				    if (bias)
				    {
					    auto biasValue =
					        bodyBuilder
					            .create<mlir::spirv::LoadOp>(
					                loc, bodyBuilder.getF32Type(),
					                EmitF32StorageBufferElementPointer(bodyBuilder, loc, *bias, affineIndex), nullptr,
					                nullptr)
					            .getValue();
					    normalized = bodyBuilder.create<mlir::spirv::FAddOp>(loc, normalized, biasValue).getResult();
				    }
				    bodyBuilder.create<mlir::spirv::StoreOp>(
				        loc, EmitF32StorageBufferElementPointer(bodyBuilder, loc, out, outputIndex), normalized,
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
				throw std::runtime_error("Generated Vulkan native MLIR SPIR-V Normalization module verification failed");
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

		void ValidateVulkanShaderModule(mlir::spirv::ModuleOp module, std::size_t expectedEntryPointCount = 1)
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
			if (entryPointCount != expectedEntryPointCount)
			{
				throw std::runtime_error("Vulkan SPIR-V module contains an unexpected entry point count");
			}
			if (executionModeCount != expectedEntryPointCount)
			{
				throw std::runtime_error("Vulkan SPIR-V module contains an unexpected execution mode count");
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

		VulkanNativeGeneratedSPIRV SerializeSameShapeBinaryF32ChainSPIRV(std::span<const BinaryOp> ops,
		                                                                 std::uint32_t elementCount)
		{
			mlir::MLIRContext context;
			context.getOrLoadDialect<mlir::spirv::SPIRVDialect>();

			std::vector<BinaryOp> uniqueOps;
			for (const auto op : ops)
			{
				if (!llvm::is_contained(uniqueOps, op))
				{
					uniqueOps.push_back(op);
				}
			}
			auto module = BuildSameShapeBinaryF32ChainSPIRVModule(ops, elementCount, context);
			ValidateVulkanShaderModule(module.get(), uniqueOps.size());

			std::string mlirText;
			llvm::raw_string_ostream mlirStream(mlirText);
			module.get().print(mlirStream);

			llvm::SmallVector<std::uint32_t, 0> binary;
			mlir::spirv::SerializationOptions options;
			options.emitSymbolName = false;
			options.emitDebugInfo = false;
			if (mlir::failed(mlir::spirv::serialize(module.get(), binary, options)))
			{
				throw std::runtime_error("Failed to serialize generated Vulkan native MLIR SPIR-V binary chain module");
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

		VulkanNativeGeneratedSPIRV SerializeReduceF32SPIRV(ReduceOp op, std::span<const std::size_t> inputShape,
		                                                   std::size_t axis)
		{
			mlir::MLIRContext context;
			context.getOrLoadDialect<mlir::spirv::SPIRVDialect>();

			auto module = BuildReduceF32SPIRVModule(op, inputShape, axis, context);
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
				throw std::runtime_error("Failed to serialize generated Vulkan native MLIR SPIR-V Reduce module");
			}

			return VulkanNativeGeneratedSPIRV{
				.words = std::vector<std::uint32_t>(binary.begin(), binary.end()),
				.mlir = mlirStream.str(),
			};
		}

		VulkanNativeGeneratedSPIRV SerializeSoftmaxF32SPIRV(std::span<const std::size_t> inputShape, std::size_t axis)
		{
			mlir::MLIRContext context;
			context.getOrLoadDialect<mlir::spirv::SPIRVDialect>();

			auto module = BuildSoftmaxF32SPIRVModule(inputShape, axis, context);
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
				throw std::runtime_error("Failed to serialize generated Vulkan native MLIR SPIR-V Softmax module");
			}

			return VulkanNativeGeneratedSPIRV{
				.words = std::vector<std::uint32_t>(binary.begin(), binary.end()),
				.mlir = mlirStream.str(),
			};
		}

		VulkanNativeGeneratedSPIRV SerializePool2DF32SPIRV(PoolMode mode, std::span<const std::size_t> inputShape,
		                                                   std::span<const std::size_t> outputShape,
		                                                   std::span<const std::size_t> kernelShape,
		                                                   std::span<const std::size_t> strides,
		                                                   std::span<const std::size_t> lowPads,
		                                                   bool countIncludePad)
		{
			mlir::MLIRContext context;
			context.getOrLoadDialect<mlir::spirv::SPIRVDialect>();

			auto module =
			    BuildPool2DF32SPIRVModule(mode, inputShape, outputShape, kernelShape, strides, lowPads,
			                              countIncludePad, context);
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
				throw std::runtime_error("Failed to serialize generated Vulkan native MLIR SPIR-V Pool2D module");
			}

			return VulkanNativeGeneratedSPIRV{
				.words = std::vector<std::uint32_t>(binary.begin(), binary.end()),
				.mlir = mlirStream.str(),
			};
		}

		VulkanNativeGeneratedSPIRV SerializeNormalizationF32SPIRV(NormalizationMode mode,
		                                                          std::span<const std::size_t> inputShape,
		                                                          std::size_t axis, double epsilon, bool hasScale,
		                                                          bool hasBias, std::size_t groupCount)
		{
			mlir::MLIRContext context;
			context.getOrLoadDialect<mlir::spirv::SPIRVDialect>();

			auto module =
			    BuildNormalizationF32SPIRVModule(mode, inputShape, axis, epsilon, hasScale, hasBias, groupCount, context);
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
				throw std::runtime_error("Failed to serialize generated Vulkan native MLIR SPIR-V Normalization module");
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

	std::string VulkanNativeSameShapeBinaryF32KernelName(BinaryOp op)
	{
		if (!VulkanNativeSupportsSameShapeBinaryF32(op))
		{
			throw std::runtime_error("Unsupported Vulkan native same-shape f32 binary op");
		}
		return SameShapeBinaryF32KernelName(op);
	}

	VulkanNativeGeneratedSPIRV VulkanNativeSameShapeBinaryF32ChainSPIRV(std::span<const BinaryOp> ops,
	                                                                    std::uint32_t elementCount)
	{
		if (ops.empty())
		{
			throw std::runtime_error("Vulkan native same-shape binary chain requires at least one op");
		}
		if (elementCount == 0)
		{
			throw std::runtime_error("Vulkan native same-shape binary chain requires a non-empty static shape");
		}
		for (const auto op : ops)
		{
			if (!VulkanNativeSupportsSameShapeBinaryF32(op))
			{
				throw std::runtime_error("Unsupported Vulkan native same-shape f32 binary chain op");
			}
		}
		return SerializeSameShapeBinaryF32ChainSPIRV(ops, elementCount);
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

	std::string_view VulkanNativeReduceF32KernelName(ReduceOp op)
	{
		return ReduceF32KernelName(op);
	}

	bool VulkanNativeSupportsReduceF32(ReduceOp op, std::span<const std::size_t> inputShape, std::size_t axis)
	{
		switch (op)
		{
		case ReduceOp::Sum:
		case ReduceOp::Mean:
		case ReduceOp::Max:
			break;
		default:
			return false;
		}
		if (axis >= inputShape.size())
		{
			return false;
		}
		if (inputShape[axis] == 0 || inputShape[axis] > std::numeric_limits<std::uint32_t>::max())
		{
			return false;
		}
		const auto outputShape = ReduceOutputShape(inputShape, axis);
		return NumElementsU32(inputShape).has_value() && NumElementsU32(outputShape).has_value() &&
		       AxisInnerSizeU32(inputShape, axis).has_value();
	}

	VulkanNativeGeneratedSPIRV VulkanNativeReduceF32SPIRV(ReduceOp op, std::span<const std::size_t> inputShape,
	                                                      std::size_t axis)
	{
		if (!VulkanNativeSupportsReduceF32(op, inputShape, axis))
		{
			throw std::runtime_error("Vulkan native f32 reduce requires a supported op, static non-empty shape, and "
			                         "an in-range axis");
		}
		return SerializeReduceF32SPIRV(op, inputShape, axis);
	}

	bool VulkanNativeSupportsSoftmaxF32(std::span<const std::size_t> inputShape, std::size_t axis)
	{
		if (axis >= inputShape.size())
		{
			return false;
		}
		if (inputShape[axis] == 0 || inputShape[axis] > std::numeric_limits<std::uint32_t>::max())
		{
			return false;
		}
		return NumElementsU32(inputShape).has_value() && AxisInnerSizeU32(inputShape, axis).has_value();
	}

	VulkanNativeGeneratedSPIRV VulkanNativeSoftmaxF32SPIRV(std::span<const std::size_t> inputShape,
	                                                       std::size_t axis)
	{
		if (!VulkanNativeSupportsSoftmaxF32(inputShape, axis))
		{
			throw std::runtime_error("Vulkan native f32 softmax requires a static non-empty shape and an in-range axis");
		}
		return SerializeSoftmaxF32SPIRV(inputShape, axis);
	}

	std::string_view VulkanNativePool2DF32KernelName(PoolMode mode)
	{
		return Pool2DF32KernelName(mode);
	}

	bool VulkanNativeSupportsPool2DF32(PoolMode mode, std::span<const std::size_t> inputShape,
	                                   std::span<const std::size_t> outputShape,
	                                   std::span<const std::size_t> kernelShape,
	                                   std::span<const std::size_t> strides,
	                                   std::span<const std::size_t> lowPads,
	                                   std::span<const std::size_t> highPads,
	                                   bool countIncludePad)
	{
		if (mode != PoolMode::Max && mode != PoolMode::Average)
		{
			return false;
		}
		if (inputShape.size() != 4 || outputShape.size() != 4 || kernelShape.size() != 2 || strides.size() != 2 ||
		    lowPads.size() != 2 || highPads.size() != 2)
		{
			return false;
		}
		if (!NumElementsU32(inputShape).has_value() || !NumElementsU32(outputShape).has_value())
		{
			return false;
		}
		for (const auto value : { inputShape[0], inputShape[1], inputShape[2], inputShape[3], outputShape[0],
			                      outputShape[1], outputShape[2], outputShape[3], kernelShape[0], kernelShape[1],
			                      strides[0], strides[1] })
		{
			if (value == 0 || value > std::numeric_limits<std::uint32_t>::max())
			{
				return false;
			}
		}
		for (const auto value : { lowPads[0], lowPads[1], highPads[0], highPads[1] })
		{
			if (value > std::numeric_limits<std::uint32_t>::max())
			{
				return false;
			}
		}
		const auto paddedH = static_cast<std::uint64_t>(lowPads[0]) + inputShape[2] + highPads[0];
		const auto paddedW = static_cast<std::uint64_t>(lowPads[1]) + inputShape[3] + highPads[1];
		if (paddedH > std::numeric_limits<std::uint32_t>::max() ||
		    paddedW > std::numeric_limits<std::uint32_t>::max() ||
		    kernelShape[0] > paddedH || kernelShape[1] > paddedW)
		{
			return false;
		}
		const auto expectedH = (paddedH - kernelShape[0]) / strides[0] + 1;
		const auto expectedW = (paddedW - kernelShape[1]) / strides[1] + 1;
		return inputShape[0] == outputShape[0] && inputShape[1] == outputShape[1] &&
		       outputShape[2] == expectedH && outputShape[3] == expectedW;
	}

	VulkanNativeGeneratedSPIRV VulkanNativePool2DF32SPIRV(PoolMode mode,
	                                                      std::span<const std::size_t> inputShape,
	                                                      std::span<const std::size_t> outputShape,
	                                                      std::span<const std::size_t> kernelShape,
	                                                      std::span<const std::size_t> strides,
	                                                      std::span<const std::size_t> lowPads,
	                                                      std::span<const std::size_t> highPads,
	                                                      bool countIncludePad)
	{
		constexpr std::array<std::size_t, 2> zeroPads{ 0, 0 };
		const auto effectiveLowPads = lowPads.empty() ? std::span<const std::size_t>(zeroPads) : lowPads;
		const auto effectiveHighPads = highPads.empty() ? std::span<const std::size_t>(zeroPads) : highPads;
		if (!VulkanNativeSupportsPool2DF32(mode, inputShape, outputShape, kernelShape, strides, effectiveLowPads,
		                                   effectiveHighPads, countIncludePad))
		{
			throw std::runtime_error("Vulkan native f32 Pool2D requires static rank-4 shape and rank-2 params");
		}
		return SerializePool2DF32SPIRV(mode, inputShape, outputShape, kernelShape, strides, effectiveLowPads,
		                               countIncludePad);
	}

	std::string_view VulkanNativeNormalizationF32KernelName(NormalizationMode mode)
	{
		return NormalizationF32KernelName(mode);
	}

	bool VulkanNativeSupportsNormalizationF32(NormalizationMode mode, std::span<const std::size_t> inputShape,
	                                          std::size_t axis, std::size_t groupCount)
	{
		if (mode != NormalizationMode::LayerNorm && mode != NormalizationMode::RMSNorm &&
		    mode != NormalizationMode::GroupNorm)
		{
			return false;
		}
		if (mode == NormalizationMode::GroupNorm)
		{
			if (inputShape.empty() || inputShape.size() > 4 || groupCount == 0 ||
			    groupCount > std::numeric_limits<std::uint32_t>::max())
			{
				return false;
			}
			const auto elementCount = NumElementsU32(inputShape);
			if (!elementCount)
			{
				return false;
			}
			if (inputShape.size() == 4 && inputShape[3] > std::numeric_limits<std::uint32_t>::max())
			{
				return false;
			}
			std::uint64_t groupedVolume = 1;
			for (std::size_t dim = 0; dim < std::min<std::size_t>(inputShape.size(), 3); ++dim)
			{
				if (inputShape[dim] == 0)
				{
					return false;
				}
				groupedVolume *= static_cast<std::uint64_t>(inputShape[dim]);
			}
			return groupedVolume % groupCount == 0 &&
			       groupedVolume / groupCount <= std::numeric_limits<std::uint32_t>::max();
		}
		if (axis >= inputShape.size())
		{
			return false;
		}
		if (inputShape[axis] == 0 || inputShape[axis] > std::numeric_limits<std::uint32_t>::max())
		{
			return false;
		}
		return NumElementsU32(inputShape).has_value() && AxisInnerSizeU32(inputShape, axis).has_value();
	}

	VulkanNativeGeneratedSPIRV VulkanNativeNormalizationF32SPIRV(NormalizationMode mode,
	                                                             std::span<const std::size_t> inputShape,
	                                                             std::size_t axis, double epsilon,
	                                                             bool hasScale, bool hasBias,
	                                                             std::size_t groupCount)
	{
		if (!VulkanNativeSupportsNormalizationF32(mode, inputShape, axis, groupCount))
		{
			throw std::runtime_error(
			    "Vulkan native f32 normalization requires LayerNorm/RMSNorm, static non-empty shape, and an in-range axis");
		}
		return SerializeNormalizationF32SPIRV(mode, inputShape, axis, epsilon, hasScale, hasBias, groupCount);
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
