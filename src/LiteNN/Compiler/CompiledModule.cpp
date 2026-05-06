#include "CompiledModule.h"

#include "Dialect/LiteNNDialect.h"
#include "Dialect/LiteNNOps.h"
#include "Pass/BufferizationPipeline.h"
#include "Pass/LLVMCodegenPipeline.h"
#include "Pass/LowerLiteNNPass.h"
#include "Translation/GraphToMLIR.h"

#include "llvm/ADT/SmallVector.h"
#include "llvm/ExecutionEngine/ExecutionEngine.h"
#include "llvm/ExecutionEngine/MCJIT.h"
#include "llvm/IR/Constants.h"
#include "llvm/IR/IRBuilder.h"
#include "llvm/IR/LegacyPassManager.h"
#include "llvm/IR/LLVMContext.h"
#include "llvm/IR/Module.h"
#include "llvm/IR/Verifier.h"
#include "llvm/MC/TargetRegistry.h"
#include "llvm/Object/ObjectFile.h"
#include "llvm/Passes/PassBuilder.h"
#include "llvm/Support/CodeGen.h"
#include "llvm/Support/Error.h"
#include "llvm/Support/MemoryBuffer.h"
#include "llvm/Support/TargetSelect.h"
#include "llvm/Support/raw_ostream.h"
#include "llvm/Target/TargetMachine.h"
#include "llvm/TargetParser/Host.h"
#include "llvm/TargetParser/Triple.h"
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

#include <algorithm>
#include <array>
#include <cstring>
#include <cstdint>
#include <fstream>
#include <limits>
#include <optional>
#include <stdexcept>
#include <utility>

using namespace LiteNN;

namespace
{
	constexpr std::string_view kEntrySymbol = "litenn_forward";
	constexpr std::array<std::byte, 8> kRodataMagic = {
	    std::byte{ 'L' }, std::byte{ 'T' }, std::byte{ 'N' }, std::byte{ 'N' },
	    std::byte{ 'C' }, std::byte{ 'M' }, std::byte{ '0' }, std::byte{ 0 },
	};
	constexpr std::uint32_t kRodataVersion = 1;

	using EntryFn = void (*)(void**, void**);

	struct NativeTargetConfig
	{
		std::string triple;
		std::unique_ptr<llvm::TargetMachine> targetMachine;
	};

	std::string ToString(llvm::Error error)
	{
		std::string message;
		llvm::raw_string_ostream os(message);
		llvm::logAllUnhandledErrors(std::move(error), os);
		return os.str();
	}

	template <typename T>
	T TakeExpected(llvm::Expected<T> expected, std::string_view what)
	{
		if (!expected)
		{
			throw std::runtime_error(std::string(what) + ": " + ToString(expected.takeError()));
		}
		return std::move(*expected);
	}

	void CheckError(llvm::Error error, std::string_view what)
	{
		if (error)
		{
			throw std::runtime_error(std::string(what) + ": " + ToString(std::move(error)));
		}
	}

	void InitializeNativeLLVM()
	{
		static const bool initialized = [] {
			llvm::InitializeNativeTarget();
			llvm::InitializeNativeTargetAsmPrinter();
			llvm::InitializeNativeTargetAsmParser();
			return true;
		}();
		(void)initialized;
	}

	NativeTargetConfig CreateNativeTargetMachine()
	{
		InitializeNativeLLVM();

		auto triple = llvm::sys::getDefaultTargetTriple();
		std::string error;
		const auto* target = llvm::TargetRegistry::lookupTarget(triple, error);
		if (!target)
		{
			throw std::runtime_error("Failed to lookup native LLVM target: " + error);
		}

		llvm::TargetOptions options;
		auto relocModel = std::optional<llvm::Reloc::Model>(llvm::Reloc::PIC_);
		auto targetMachine = std::unique_ptr<llvm::TargetMachine>(
		    target->createTargetMachine(triple, "generic", "", options, relocModel));
		if (!targetMachine)
		{
			throw std::runtime_error("Failed to create native LLVM target machine");
		}

		return { std::move(triple), std::move(targetMachine) };
	}

	void ConfigureForNativeObject(llvm::Module& module, const NativeTargetConfig& config)
	{
		module.setTargetTriple(llvm::Triple(config.triple));
		module.setDataLayout(config.targetMachine->createDataLayout());
	}

	std::vector<std::byte> EmitObjectFile(llvm::Module& module)
	{
		auto config = CreateNativeTargetMachine();
		ConfigureForNativeObject(module, config);

		if (llvm::verifyModule(module, &llvm::errs()))
		{
			throw std::runtime_error("LLVM module verification failed before object emission");
		}

		llvm::SmallVector<char, 0> buffer;
		llvm::raw_svector_ostream stream(buffer);
		llvm::legacy::PassManager passManager;
		if (config.targetMachine->addPassesToEmitFile(passManager, stream, nullptr,
		                                              llvm::CodeGenFileType::ObjectFile))
		{
			throw std::runtime_error("Native target cannot emit object files");
		}
		passManager.run(module);

		std::vector<std::byte> bytes(buffer.size());
		std::memcpy(bytes.data(), buffer.data(), buffer.size());
		return bytes;
	}

	void AppendU32(std::vector<std::byte>& out, std::uint32_t value)
	{
		for (int i = 0; i < 4; ++i)
		{
			out.push_back(static_cast<std::byte>((value >> (i * 8)) & 0xffu));
		}
	}

	void AppendU64(std::vector<std::byte>& out, std::uint64_t value)
	{
		for (int i = 0; i < 8; ++i)
		{
			out.push_back(static_cast<std::byte>((value >> (i * 8)) & 0xffu));
		}
	}

	std::uint32_t ReadU32(std::span<const std::byte> bytes, std::size_t& offset)
	{
		if (offset + 4 > bytes.size())
		{
			throw std::runtime_error("Compiled module rodata is truncated");
		}
		std::uint32_t value = 0;
		for (int i = 0; i < 4; ++i)
		{
			value |= std::to_integer<std::uint32_t>(bytes[offset + i]) << (i * 8);
		}
		offset += 4;
		return value;
	}

	std::uint64_t ReadU64(std::span<const std::byte> bytes, std::size_t& offset)
	{
		if (offset + 8 > bytes.size())
		{
			throw std::runtime_error("Compiled module rodata is truncated");
		}
		std::uint64_t value = 0;
		for (int i = 0; i < 8; ++i)
		{
			value |= std::to_integer<std::uint64_t>(bytes[offset + i]) << (i * 8);
		}
		offset += 8;
		return value;
	}

	std::vector<std::byte> SerializeRodata(std::span<const CompiledTensorSpec> inputs,
	                                       std::span<const CompiledTensorSpec> outputs)
	{
		std::vector<std::byte> rodata;
		rodata.insert(rodata.end(), kRodataMagic.begin(), kRodataMagic.end());
		AppendU32(rodata, kRodataVersion);
		AppendU32(rodata, static_cast<std::uint32_t>(inputs.size()));
		AppendU32(rodata, static_cast<std::uint32_t>(outputs.size()));

		const auto appendSpec = [&](const CompiledTensorSpec& spec) {
			AppendU32(rodata, static_cast<std::uint32_t>(spec.dtype));
			AppendU32(rodata, static_cast<std::uint32_t>(spec.shape.size()));
			for (auto dim : spec.shape)
			{
				AppendU64(rodata, static_cast<std::uint64_t>(dim));
			}
		};

		for (const auto& spec : inputs)
		{
			appendSpec(spec);
		}
		for (const auto& spec : outputs)
		{
			appendSpec(spec);
		}
		return rodata;
	}

	std::pair<std::vector<CompiledTensorSpec>, std::vector<CompiledTensorSpec>>
	DeserializeRodata(std::span<const std::byte> rodata)
	{
		if (rodata.size() < kRodataMagic.size() ||
		    !std::equal(kRodataMagic.begin(), kRodataMagic.end(), rodata.begin()))
		{
			throw std::runtime_error("Compiled module rodata has an invalid magic header");
		}

		std::size_t offset = kRodataMagic.size();
		const auto version = ReadU32(rodata, offset);
		if (version != kRodataVersion)
		{
			throw std::runtime_error("Unsupported compiled module rodata version");
		}

		const auto inputCount = ReadU32(rodata, offset);
		const auto outputCount = ReadU32(rodata, offset);

		const auto readSpec = [&]() {
			CompiledTensorSpec spec;
			const auto dtypeValue = ReadU32(rodata, offset);
			if (dtypeValue > static_cast<std::uint32_t>(DataType::Bool))
			{
				throw std::runtime_error("Compiled module rodata contains an invalid data type");
			}
			spec.dtype = static_cast<DataType>(dtypeValue);

			const auto rank = ReadU32(rodata, offset);
			spec.shape.reserve(rank);
			for (std::uint32_t i = 0; i < rank; ++i)
			{
				const auto dim = ReadU64(rodata, offset);
				if (dim > std::numeric_limits<std::size_t>::max())
				{
					throw std::runtime_error("Compiled module rodata shape dimension is too large");
				}
				spec.shape.push_back(static_cast<std::size_t>(dim));
			}
			return spec;
		};

		std::vector<CompiledTensorSpec> inputs;
		inputs.reserve(inputCount);
		for (std::uint32_t i = 0; i < inputCount; ++i)
		{
			inputs.push_back(readSpec());
		}

		std::vector<CompiledTensorSpec> outputs;
		outputs.reserve(outputCount);
		for (std::uint32_t i = 0; i < outputCount; ++i)
		{
			outputs.push_back(readSpec());
		}

		if (offset != rodata.size())
		{
			throw std::runtime_error("Compiled module rodata contains trailing bytes");
		}

		return { std::move(inputs), std::move(outputs) };
	}

	std::vector<CompiledTensorSpec> BuildInputSpecs(const Subgraph& subgraph)
	{
		std::vector<CompiledTensorSpec> specs;
		specs.reserve(subgraph.Params().size());
		for (const auto& param : subgraph.Params())
		{
			specs.push_back({ param.dtype, param.shape });
		}
		return specs;
	}

	std::vector<CompiledTensorSpec> BuildOutputSpecs(const Subgraph& subgraph)
	{
		std::vector<CompiledTensorSpec> specs;
		specs.reserve(subgraph.Results().size());
		for (const auto& result : subgraph.Results())
		{
			const auto& info = subgraph.GetOutputInfo(result);
			specs.push_back({ info.dtype, info.shape });
		}
		return specs;
	}

	std::size_t ElementByteSize(DataType dtype)
	{
		switch (dtype)
		{
		case DataType::Float32:
			return sizeof(float);
		case DataType::Float64:
			return sizeof(double);
		case DataType::Int32:
			return sizeof(std::int32_t);
		case DataType::Int64:
			return sizeof(std::int64_t);
		case DataType::Bool:
			return sizeof(bool);
		}
		throw std::runtime_error("Invalid data type");
	}

	std::uint64_t NumElements(const CompiledTensorSpec& spec)
	{
		std::uint64_t n = 1;
		for (const auto dim : spec.shape)
		{
			n *= static_cast<std::uint64_t>(dim);
		}
		return n;
	}

	std::vector<std::uint64_t> ContiguousStrides(const CompiledTensorSpec& spec)
	{
		std::vector<std::uint64_t> strides(spec.shape.size());
		if (!strides.empty())
		{
			strides.back() = 1;
			for (std::size_t i = strides.size() - 1; i > 0; --i)
			{
				strides[i - 1] = strides[i] * static_cast<std::uint64_t>(spec.shape[i]);
			}
		}
		return strides;
	}

	llvm::Type* GetElementType(llvm::LLVMContext& ctx, DataType dtype)
	{
		switch (dtype)
		{
		case DataType::Float32:
			return llvm::Type::getFloatTy(ctx);
		case DataType::Float64:
			return llvm::Type::getDoubleTy(ctx);
		case DataType::Int32:
			return llvm::Type::getInt32Ty(ctx);
		case DataType::Int64:
			return llvm::Type::getInt64Ty(ctx);
		case DataType::Bool:
			return llvm::Type::getInt1Ty(ctx);
		}
		throw std::runtime_error("Invalid data type");
	}

	std::string LLVMTypeToString(llvm::Type* type)
	{
		std::string text;
		llvm::raw_string_ostream os(text);
		type->print(os);
		return os.str();
	}

	llvm::StructType* GetMemRefDescriptorType(llvm::LLVMContext& ctx, std::size_t rank)
	{
		auto* ptrTy = llvm::PointerType::get(ctx, 0);
		auto* i64Ty = llvm::Type::getInt64Ty(ctx);
		auto* shapeArrayTy = llvm::ArrayType::get(i64Ty, rank);
		return llvm::StructType::get(ctx, { ptrTy, ptrTy, i64Ty, shapeArrayTy, shapeArrayTy });
	}

	bool IsMemRefDescriptorType(llvm::LLVMContext& ctx, llvm::Type* type, std::size_t rank)
	{
		auto* descTy = llvm::dyn_cast<llvm::StructType>(type);
		if (!descTy)
		{
			return false;
		}

		auto* ptrTy = llvm::PointerType::get(ctx, 0);
		auto* i64Ty = llvm::Type::getInt64Ty(ctx);
		auto* shapeArrayTy = llvm::ArrayType::get(i64Ty, rank);
		if (descTy->getNumElements() == 5)
		{
			return descTy->getElementType(0) == ptrTy && descTy->getElementType(1) == ptrTy &&
			       descTy->getElementType(2) == i64Ty && descTy->getElementType(3) == shapeArrayTy &&
			       descTy->getElementType(4) == shapeArrayTy;
		}
		if (descTy->getNumElements() == 4)
		{
			return descTy->getElementType(0) == ptrTy && descTy->getElementType(1) == i64Ty &&
			       descTy->getElementType(2) == shapeArrayTy && descTy->getElementType(3) == shapeArrayTy;
		}
		return false;
	}

	llvm::Value* BuildI64Array(llvm::IRBuilder<>& builder, std::span<const std::uint64_t> values)
	{
		auto& ctx = builder.getContext();
		auto* i64Ty = llvm::Type::getInt64Ty(ctx);
		auto* arrayTy = llvm::ArrayType::get(i64Ty, values.size());
		llvm::Value* array = llvm::PoisonValue::get(arrayTy);
		for (std::size_t i = 0; i < values.size(); ++i)
		{
			array = builder.CreateInsertValue(array, builder.getInt64(values[i]), { static_cast<unsigned>(i) });
		}
		return array;
	}

	llvm::Value* BuildMemRefDescriptor(llvm::IRBuilder<>& builder, llvm::Value* data,
	                                   const CompiledTensorSpec& spec)
	{
		auto& ctx = builder.getContext();
		auto* descTy = GetMemRefDescriptorType(ctx, spec.shape.size());
		llvm::Value* desc = llvm::PoisonValue::get(descTy);
		desc = builder.CreateInsertValue(desc, data, { 0 });
		desc = builder.CreateInsertValue(desc, data, { 1 });
		desc = builder.CreateInsertValue(desc, builder.getInt64(0), { 2 });

		std::vector<std::uint64_t> sizes(spec.shape.begin(), spec.shape.end());
		desc = builder.CreateInsertValue(desc, BuildI64Array(builder, sizes), { 3 });

		const auto strides = ContiguousStrides(spec);
		desc = builder.CreateInsertValue(desc, BuildI64Array(builder, strides), { 4 });
		return desc;
	}

	void AppendDescriptorCallArgument(llvm::IRBuilder<>& builder, llvm::FunctionType* calleeType,
	                                  std::size_t& paramIndex, llvm::Value* descriptor,
	                                  std::vector<llvm::Value*>& args)
	{
		if (paramIndex >= calleeType->getNumParams())
		{
			throw std::runtime_error("Compiled subgraph function has fewer arguments than expected");
		}

		auto* expectedTy = calleeType->getParamType(paramIndex);
		if (expectedTy == descriptor->getType())
		{
			args.push_back(descriptor);
			++paramIndex;
			return;
		}

		auto* descTy = llvm::cast<llvm::StructType>(descriptor->getType());
		const auto appendValue = [&](llvm::Value* value) -> bool {
			if (paramIndex >= calleeType->getNumParams() || calleeType->getParamType(paramIndex) != value->getType())
			{
				return false;
			}
			args.push_back(value);
			++paramIndex;
			return true;
		};

		const auto tryAppendPattern = [&](const auto& appendPattern) {
			const auto savedParamIndex = paramIndex;
			const auto savedArgCount = args.size();
			if (appendPattern())
			{
				return true;
			}
			paramIndex = savedParamIndex;
			args.resize(savedArgCount);
			return false;
		};

		const auto appendWholeField = [&](unsigned index) -> bool {
			return appendValue(builder.CreateExtractValue(descriptor, { index }));
		};

		const auto* sizesTy = llvm::cast<llvm::ArrayType>(descTy->getElementType(3));
		const auto* stridesTy = llvm::cast<llvm::ArrayType>(descTy->getElementType(4));
		const auto appendArrayScalars = [&](unsigned index, const llvm::ArrayType* arrayTy) -> bool {
			for (unsigned i = 0; i < arrayTy->getNumElements(); ++i)
			{
				if (!appendValue(builder.CreateExtractValue(descriptor, { index, i })))
				{
					return false;
				}
			}
			return true;
		};

		if (tryAppendPattern([&] {
			    return appendWholeField(0) && appendWholeField(1) && appendWholeField(2) &&
			           appendWholeField(3) && appendWholeField(4);
		    }) ||
		    tryAppendPattern([&] {
			    return appendWholeField(1) && appendWholeField(2) && appendWholeField(3) &&
			           appendWholeField(4);
		    }) ||
		    tryAppendPattern([&] {
			    return appendWholeField(1) && appendWholeField(2) && appendArrayScalars(3, sizesTy) &&
			           appendArrayScalars(4, stridesTy);
		    }) ||
		    tryAppendPattern([&] {
			    return appendWholeField(0) && appendWholeField(1) && appendWholeField(2) &&
			           appendArrayScalars(3, sizesTy) && appendArrayScalars(4, stridesTy);
		    }))
		{
			return;
		}

		if (expectedTy->isPointerTy())
		{
			auto* alloca = builder.CreateAlloca(descriptor->getType());
			builder.CreateStore(descriptor, alloca);
			args.push_back(alloca);
			++paramIndex;
			return;
		}

		std::string message = "Compiled subgraph function has an unsupported memref ABI at parameter " +
		                      std::to_string(paramIndex);
		if (paramIndex < calleeType->getNumParams())
		{
			message += ": expected " + LLVMTypeToString(calleeType->getParamType(paramIndex));
		}
		message += ", got " + LLVMTypeToString(descriptor->getType());
		throw std::runtime_error(message);
	}

	void CopyDescriptorToOutput(llvm::IRBuilder<>& builder, llvm::Value* descriptor, llvm::Value* outputArray,
	                            std::size_t outputIndex, const CompiledTensorSpec& spec)
	{
		auto& ctx = builder.getContext();
		auto* ptrTy = llvm::PointerType::get(ctx, 0);
		auto* outputSlot = builder.CreateGEP(ptrTy, outputArray, builder.getInt64(outputIndex));
		auto* outputData = builder.CreateLoad(ptrTy, outputSlot);
		auto* descTy = llvm::cast<llvm::StructType>(descriptor->getType());
		const unsigned dataField = descTy->getNumElements() == 5 ? 1 : 0;
		auto* sourceData = builder.CreateExtractValue(descriptor, { dataField });
		const auto byteCount = NumElements(spec) * ElementByteSize(spec.dtype);
		builder.CreateMemCpy(outputData, llvm::Align(1), sourceData, llvm::Align(1), builder.getInt64(byteCount));
	}

	std::optional<std::size_t> FindStructRetParamIndex(const llvm::Function& function)
	{
		std::size_t index = 0;
		for (const auto& arg : function.args())
		{
			if (arg.hasStructRetAttr())
			{
				return index;
			}
			++index;
		}
		return std::nullopt;
	}

	void AddUniformEntryWrapper(llvm::Module& module, std::string_view calleeName,
	                            std::span<const CompiledTensorSpec> inputs,
	                            std::span<const CompiledTensorSpec> outputs)
	{
		auto* callee = module.getFunction(calleeName);
		if (!callee)
		{
			throw std::runtime_error("Compiled subgraph function was not found in LLVM module");
		}

		auto& ctx = module.getContext();
		auto* voidTy = llvm::Type::getVoidTy(ctx);
		auto* ptrTy = llvm::PointerType::get(ctx, 0);
		auto* entryType = llvm::FunctionType::get(voidTy, { ptrTy, ptrTy }, false);
		auto* entry = llvm::Function::Create(entryType, llvm::GlobalValue::ExternalLinkage,
		                                     std::string(kEntrySymbol), module);

		auto* block = llvm::BasicBlock::Create(ctx, "entry", entry);
		llvm::IRBuilder<> builder(block);
		auto argIt = entry->arg_begin();
		llvm::Value* inputArray = &*argIt++;
		llvm::Value* outputArray = &*argIt;

		std::vector<llvm::Value*> descriptors;
		descriptors.reserve(inputs.size());
		for (std::size_t i = 0; i < inputs.size(); ++i)
		{
			auto* inputSlot = builder.CreateGEP(ptrTy, inputArray, builder.getInt64(i));
			auto* inputData = builder.CreateLoad(ptrTy, inputSlot);
			descriptors.push_back(BuildMemRefDescriptor(builder, inputData, inputs[i]));
		}

		auto* calleeType = callee->getFunctionType();
		std::vector<llvm::Value*> callArgs;
		std::size_t paramIndex = 0;
		const auto sretParamIndex = FindStructRetParamIndex(*callee);
		llvm::AllocaInst* sretStorage = nullptr;
		llvm::Type* sretType = nullptr;

		if (sretParamIndex)
		{
			sretType = callee->getParamStructRetType(static_cast<unsigned>(*sretParamIndex));
			if (!sretType)
			{
				throw std::runtime_error("Compiled subgraph function has an invalid sret ABI");
			}
			sretStorage = builder.CreateAlloca(sretType);
		}

		const auto appendStructRetIfNeeded = [&]() {
			if (sretStorage && paramIndex == *sretParamIndex)
			{
				callArgs.push_back(sretStorage);
				++paramIndex;
				return true;
			}
			return false;
		};

		appendStructRetIfNeeded();
		for (auto* descriptor : descriptors)
		{
			appendStructRetIfNeeded();
			AppendDescriptorCallArgument(builder, calleeType, paramIndex, descriptor, callArgs);
		}

		std::vector<llvm::Value*> outputDescriptors;
		outputDescriptors.reserve(outputs.size());
		for (std::size_t i = 0; i < outputs.size(); ++i)
		{
			auto* outputSlot = builder.CreateGEP(ptrTy, outputArray, builder.getInt64(i));
			auto* outputData = builder.CreateLoad(ptrTy, outputSlot);
			outputDescriptors.push_back(BuildMemRefDescriptor(builder, outputData, outputs[i]));
		}
		for (auto* descriptor : outputDescriptors)
		{
			appendStructRetIfNeeded();
			if (paramIndex >= calleeType->getNumParams())
			{
				break;
			}
			AppendDescriptorCallArgument(builder, calleeType, paramIndex, descriptor, callArgs);
		}
		appendStructRetIfNeeded();

		if (paramIndex != calleeType->getNumParams())
		{
			throw std::runtime_error("Compiled subgraph function has more arguments than expected");
		}

		auto* call = builder.CreateCall(callee, callArgs);
		call->setCallingConv(callee->getCallingConv());
		call->setAttributes(callee->getAttributes());
		auto* retTy = calleeType->getReturnType();

		if (sretStorage)
		{
			if (outputs.size() == 1 && IsMemRefDescriptorType(ctx, sretType, outputs[0].shape.size()))
			{
				auto* descriptor = builder.CreateLoad(sretType, sretStorage);
				CopyDescriptorToOutput(builder, descriptor, outputArray, 0, outputs[0]);
				builder.CreateRetVoid();
				return;
			}

			auto* resultTupleTy = llvm::dyn_cast<llvm::StructType>(sretType);
			if (!resultTupleTy || resultTupleTy->getNumElements() != outputs.size())
			{
				throw std::runtime_error("Compiled subgraph function has an unsupported sret ABI");
			}

			auto* result = builder.CreateLoad(sretType, sretStorage);
			for (std::size_t i = 0; i < outputs.size(); ++i)
			{
				auto* descriptor = builder.CreateExtractValue(result, { static_cast<unsigned>(i) });
				CopyDescriptorToOutput(builder, descriptor, outputArray, i, outputs[i]);
			}
			builder.CreateRetVoid();
			return;
		}

		if (retTy->isVoidTy())
		{
			builder.CreateRetVoid();
			return;
		}

		if (outputs.size() == 1)
		{
			auto* expectedDescTy = GetMemRefDescriptorType(ctx, outputs[0].shape.size());
			if (retTy == expectedDescTy)
			{
				CopyDescriptorToOutput(builder, call, outputArray, 0, outputs[0]);
				builder.CreateRetVoid();
				return;
			}
		}

		auto* resultTupleTy = llvm::dyn_cast<llvm::StructType>(retTy);
		if (!resultTupleTy || resultTupleTy->getNumElements() != outputs.size())
		{
			throw std::runtime_error("Compiled subgraph function has an unsupported result ABI");
		}

		for (std::size_t i = 0; i < outputs.size(); ++i)
		{
			auto* descriptor = builder.CreateExtractValue(call, { static_cast<unsigned>(i) });
			CopyDescriptorToOutput(builder, descriptor, outputArray, i, outputs[i]);
		}
		builder.CreateRetVoid();
	}

	std::vector<std::byte> ToByteVector(const void* data, std::size_t size)
	{
		if (size != 0 && data == nullptr)
		{
			throw std::runtime_error("Compiled module image has a null data pointer");
		}
		std::vector<std::byte> bytes(size);
		if (size != 0)
		{
			std::memcpy(bytes.data(), data, size);
		}
		return bytes;
	}

	struct LoadedJIT
	{
		std::unique_ptr<llvm::LLVMContext> context;
		std::unique_ptr<llvm::ExecutionEngine> engine;
		EntryFn entry{};
	};

	LoadedJIT LoadJIT(std::span<const std::byte> instructions)
	{
		InitializeNativeLLVM();

		LoadedJIT loaded;
		loaded.context = std::make_unique<llvm::LLVMContext>();
		auto module = std::make_unique<llvm::Module>("litenn_jit_loader", *loaded.context);

		std::string error;
		llvm::EngineBuilder builder(std::move(module));
		builder.setErrorStr(&error);
		builder.setEngineKind(llvm::EngineKind::JIT);
		loaded.engine.reset(builder.create());
		if (!loaded.engine)
		{
			throw std::runtime_error("Failed to create LiteNN JIT: " + error);
		}

		auto buffer = llvm::MemoryBuffer::getMemBufferCopy(
		    llvm::StringRef(reinterpret_cast<const char*>(instructions.data()), instructions.size()),
		    "litenn-compiled-module.o");
		auto object = TakeExpected(llvm::object::ObjectFile::createObjectFile(buffer->getMemBufferRef()),
		                           "Failed to parse LiteNN object image");
		loaded.engine->addObjectFile(
		    llvm::object::OwningBinary<llvm::object::ObjectFile>(std::move(object), std::move(buffer)));
		loaded.engine->finalizeObject();

		const auto address = loaded.engine->getFunctionAddress(std::string(kEntrySymbol));
		if (address == 0)
		{
			throw std::runtime_error("Failed to lookup LiteNN entry symbol");
		}
		loaded.entry = reinterpret_cast<EntryFn>(address);
		return loaded;
	}

	llvm::Constant* ByteArrayConstant(llvm::LLVMContext& ctx, std::span<const std::byte> bytes)
	{
		return llvm::ConstantDataArray::getString(
		    ctx, llvm::StringRef(reinterpret_cast<const char*>(bytes.data()), bytes.size()),
		    /*AddNull=*/false);
	}

	void AddByteArraySymbol(llvm::Module& module, std::string_view name, std::span<const std::byte> bytes)
	{
		auto& ctx = module.getContext();
		auto* init = ByteArrayConstant(ctx, bytes);
		auto* global = new llvm::GlobalVariable(module, init->getType(), true,
		                                        llvm::GlobalValue::ExternalLinkage, init, std::string(name));
		global->setAlignment(llvm::Align(1));
	}

	void AddSizeSymbol(llvm::Module& module, std::string_view name, std::size_t size)
	{
		auto& ctx = module.getContext();
		auto* i64Ty = llvm::Type::getInt64Ty(ctx);
		auto* init = llvm::ConstantInt::get(i64Ty, static_cast<std::uint64_t>(size));
		new llvm::GlobalVariable(module, i64Ty, true, llvm::GlobalValue::ExternalLinkage, init, std::string(name));
	}

	std::vector<std::byte> EmitCarrierObject(std::span<const std::byte> rodata,
	                                         std::span<const std::byte> instructions,
	                                         std::string_view symbolPrefix)
	{
		llvm::LLVMContext ctx;
		llvm::Module module("litenn_compiled_module_carrier", ctx);

		const auto prefix = std::string(symbolPrefix);
		AddByteArraySymbol(module, prefix + "_rodata", rodata);
		AddSizeSymbol(module, prefix + "_rodata_size", rodata.size());
		AddByteArraySymbol(module, prefix + "_instructions", instructions);
		AddSizeSymbol(module, prefix + "_instructions_size", instructions.size());

		return EmitObjectFile(module);
	}

	void WriteAllBytes(const std::filesystem::path& path, std::span<const std::byte> bytes)
	{
		std::ofstream out(path, std::ios::binary);
		if (!out)
		{
			throw std::runtime_error("Failed to open output object file");
		}
		out.write(reinterpret_cast<const char*>(bytes.data()), static_cast<std::streamsize>(bytes.size()));
		if (!out)
		{
			throw std::runtime_error("Failed to write output object file");
		}
	}

	void ValidateTensorAgainstSpec(const Tensor<CPU>& tensor, const CompiledTensorSpec& spec)
	{
		if (tensor.DType() != spec.dtype)
		{
			throw std::runtime_error("CompiledModule input dtype mismatch");
		}
		if (!std::ranges::equal(tensor.Shape().Dims, spec.shape))
		{
			throw std::runtime_error("CompiledModule input shape mismatch");
		}
	}

	mlir::OwningOpRef<mlir::ModuleOp> BuildLoweredMLIRModule(const Graph& graph, mlir::MLIRContext& ctx)
	{
		auto module = litenn::translateGraphToMLIR(graph, ctx);
		if (!module)
		{
			throw std::runtime_error("Failed to translate LiteNN graph to MLIR");
		}

		mlir::PassManager pm(&ctx);
		pm.addPass(litenn::createLowerLiteNNPass());
		litenn::addBufferizationPipeline(pm);
		litenn::addLLVMCodegenPipeline(pm);
		if (mlir::failed(pm.run(*module)))
		{
			throw std::runtime_error("LiteNN MLIR lowering/codegen pipeline failed");
		}
		if (mlir::failed(mlir::verify(*module)))
		{
			throw std::runtime_error("LiteNN lowered MLIR module verification failed");
		}
		return module;
	}

	void SetupCompilerMLIRContext(mlir::MLIRContext& ctx)
	{
		ctx.disableMultithreading();

		mlir::DialectRegistry registry;
		litenn::registerBufferizationModels(registry);
		litenn::registerLLVMTranslations(registry);

		ctx.appendDialectRegistry(registry);
		ctx.loadDialect<litenn::LiteNNDialect, mlir::arith::ArithDialect, mlir::bufferization::BufferizationDialect,
		                mlir::cf::ControlFlowDialect, mlir::func::FuncDialect, mlir::linalg::LinalgDialect,
		                mlir::LLVM::LLVMDialect, mlir::math::MathDialect, mlir::memref::MemRefDialect,
		                mlir::scf::SCFDialect, mlir::tensor::TensorDialect>();
	}
} // namespace

struct CompiledModule<CPU>::Impl
{
	std::vector<std::byte> rodata;
	std::vector<std::byte> instructions;
	std::vector<CompiledTensorSpec> inputSpecs;
	std::vector<CompiledTensorSpec> outputSpecs;
	std::unique_ptr<llvm::LLVMContext> jitContext;
	std::unique_ptr<llvm::ExecutionEngine> jit;
	EntryFn entry{};
};

CompiledModule<CPU>::CompiledModule() = default;
CompiledModule<CPU>::CompiledModule(const CompiledModule&) = default;
CompiledModule<CPU>::CompiledModule(CompiledModule&&) noexcept = default;
CompiledModule<CPU>& CompiledModule<CPU>::operator=(const CompiledModule&) = default;
CompiledModule<CPU>& CompiledModule<CPU>::operator=(CompiledModule&&) noexcept = default;
CompiledModule<CPU>::~CompiledModule() = default;

CompiledModule<CPU>::CompiledModule(std::shared_ptr<Impl> impl) : impl_(std::move(impl)) {}

CompiledModule<CPU> CompiledModule<CPU>::Load(CompiledModuleImage image)
{
	auto impl = std::make_shared<Impl>();
	impl->rodata = ToByteVector(image.rodata, image.rodataSize);
	impl->instructions = ToByteVector(image.instructions, image.instructionSize);

	auto [inputSpecs, outputSpecs] = DeserializeRodata(impl->rodata);
	impl->inputSpecs = std::move(inputSpecs);
	impl->outputSpecs = std::move(outputSpecs);
	auto loadedJit = LoadJIT(impl->instructions);
	impl->jitContext = std::move(loadedJit.context);
	impl->jit = std::move(loadedJit.engine);
	impl->entry = loadedJit.entry;
	return CompiledModule(std::move(impl));
}

std::vector<Tensor<CPU>> CompiledModule<CPU>::Run(std::span<const Tensor<CPU>> inputs) const
{
	if (!impl_ || !impl_->entry)
	{
		throw std::runtime_error("CompiledModule is empty");
	}
	if (inputs.size() != impl_->inputSpecs.size())
	{
		throw std::runtime_error("CompiledModule input count mismatch");
	}

	std::vector<void*> inputPtrs;
	inputPtrs.reserve(inputs.size());
	for (std::size_t i = 0; i < inputs.size(); ++i)
	{
		ValidateTensorAgainstSpec(inputs[i], impl_->inputSpecs[i]);
		inputPtrs.push_back(const_cast<void*>(inputs[i].RawData()));
	}

	std::vector<Tensor<CPU>> outputs;
	outputs.reserve(impl_->outputSpecs.size());
	std::vector<void*> outputPtrs;
	outputPtrs.reserve(impl_->outputSpecs.size());
	for (const auto& spec : impl_->outputSpecs)
	{
		outputs.emplace_back(Uninitialized, ShapeView{ spec.shape }, spec.dtype, CPU{});
		outputPtrs.push_back(outputs.back().RawData());
	}

	impl_->entry(inputPtrs.data(), outputPtrs.data());
	return outputs;
}

CompiledModuleImage CompiledModule<CPU>::Image() const
{
	if (!impl_)
	{
		return {};
	}
	return {
		.rodata = impl_->rodata.data(),
		.rodataSize = impl_->rodata.size(),
		.instructions = impl_->instructions.data(),
		.instructionSize = impl_->instructions.size(),
	};
}

std::span<const std::byte> CompiledModule<CPU>::Rodata() const
{
	if (!impl_)
	{
		return {};
	}
	return impl_->rodata;
}

std::span<const std::byte> CompiledModule<CPU>::Instructions() const
{
	if (!impl_)
	{
		return {};
	}
	return impl_->instructions;
}

std::span<const CompiledTensorSpec> CompiledModule<CPU>::InputSpecs() const
{
	if (!impl_)
	{
		return {};
	}
	return impl_->inputSpecs;
}

std::span<const CompiledTensorSpec> CompiledModule<CPU>::OutputSpecs() const
{
	if (!impl_)
	{
		return {};
	}
	return impl_->outputSpecs;
}

void CompiledModule<CPU>::WriteObjectFile(const std::filesystem::path& path, std::string_view symbolPrefix) const
{
	if (!impl_)
	{
		throw std::runtime_error("CompiledModule is empty");
	}
	const auto objectBytes = EmitCarrierObject(impl_->rodata, impl_->instructions, symbolPrefix);
	WriteAllBytes(path, objectBytes);
}

CompiledModule<CPU> Compiler<CPU>::Compile(const Graph& graph)
{
	mlir::MLIRContext ctx;
	SetupCompilerMLIRContext(ctx);
	auto mlirModule = BuildLoweredMLIRModule(graph, ctx);

	llvm::LLVMContext llvmCtx;
	auto llvmModule = litenn::translateToLLVMIR(*mlirModule, llvmCtx);
	if (!llvmModule)
	{
		throw std::runtime_error("Failed to translate lowered MLIR module to LLVM IR");
	}

	auto config = CreateNativeTargetMachine();
	ConfigureForNativeObject(*llvmModule, config);

	const auto& forward = graph.GetSubgraph(graph.Forward());
	const auto inputSpecs = BuildInputSpecs(forward);
	const auto outputSpecs = BuildOutputSpecs(forward);
	AddUniformEntryWrapper(*llvmModule, "subgraph_" + std::to_string(graph.Forward()), inputSpecs, outputSpecs);

	const auto rodata = SerializeRodata(inputSpecs, outputSpecs);
	auto instructions = EmitObjectFile(*llvmModule);
	return CompiledModule<CPU>::Load({
	    .rodata = rodata.data(),
	    .rodataSize = rodata.size(),
	    .instructions = instructions.data(),
	    .instructionSize = instructions.size(),
	});
}
