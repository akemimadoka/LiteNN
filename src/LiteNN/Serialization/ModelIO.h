#include <LiteNN/Graph.h>
#include <LiteNN/Validation/GraphValidator.h>

#include <array>
#include <concepts>
#include <cstddef>
#include <cstdint>
#include <filesystem>
#include <fstream>
#include <optional>
#include <span>
#include <stdexcept>
#include <string>
#include <string_view>
#include <type_traits>
#include <vector>

#ifndef LITENN_SERIALIZATION_MODELIO_H
#define LITENN_SERIALIZATION_MODELIO_H

namespace LiteNN::Serialization
{
	namespace Detail
	{
		constexpr std::array<char, 8> kModelMagic = { 'L', 'T', 'N', 'N', 'M', 'D', 'L', '\0' };
		constexpr std::uint32_t kModelVersion = 9;

		enum class MetadataValueKind : std::uint32_t
		{
			Int64 = 0,
			UInt64,
			Float64,
			Bool,
			String,
			Int64List,
			UInt64List,
			Float64List,
			BoolList,
			StringList,
		};

		enum class NodeKind : std::uint32_t
		{
			ParamRef = 0,
			Constant,
			VariableRef,
			UnaryOp,
			BinaryOp,
			Call,
			Cast,
			Cond,
			While,
			SaveActivation,
			LoadActivation,
			TapeSaveActivation,
			TapeLoadActivation,
			ReduceOp,
			Reshape,
			Concat,
			Slice,
			FusedOp,
			QuantizedConstant,
			Quantize,
			Dequantize,
			GetRows,
			Argsort,
			MulMatId,
		};

		inline void EnsureWrite(const std::ostream& out)
		{
			if (!out)
			{
				throw std::runtime_error("Failed to write LiteNN model");
			}
		}

		inline void EnsureRead(const std::istream& in)
		{
			if (!in)
			{
				throw std::runtime_error("LiteNN model is truncated or unreadable");
			}
		}

		template <typename T>
		void WriteScalar(std::ostream& out, T value)
		{
			out.write(reinterpret_cast<const char*>(&value), sizeof(T));
			EnsureWrite(out);
		}

		template <typename T>
		T ReadScalar(std::istream& in)
		{
			T value{};
			in.read(reinterpret_cast<char*>(&value), sizeof(T));
			EnsureRead(in);
			return value;
		}

		inline void WriteSize(std::ostream& out, std::size_t value)
		{
			WriteScalar(out, static_cast<std::uint64_t>(value));
		}

		inline std::size_t ReadSize(std::istream& in)
		{
			return static_cast<std::size_t>(ReadScalar<std::uint64_t>(in));
		}

		inline void WriteDataType(std::ostream& out, DataType dtype)
		{
			WriteScalar(out, static_cast<std::uint32_t>(dtype));
		}

		inline DataType ReadDataType(std::istream& in)
		{
			return static_cast<DataType>(ReadScalar<std::uint32_t>(in));
		}

		inline void WriteFloatList(std::ostream& out, std::span<const float> values)
		{
			WriteSize(out, values.size());
			for (const auto value : values)
			{
				WriteScalar(out, value);
			}
		}

		inline std::vector<float> ReadFloatList(std::istream& in)
		{
			std::vector<float> values(ReadSize(in));
			for (auto& value : values)
			{
				value = ReadScalar<float>(in);
			}
			return values;
		}

		inline void WriteI32List(std::ostream& out, std::span<const std::int32_t> values)
		{
			WriteSize(out, values.size());
			for (const auto value : values)
			{
				WriteScalar(out, value);
			}
		}

		inline std::vector<std::int32_t> ReadI32List(std::istream& in)
		{
			std::vector<std::int32_t> values(ReadSize(in));
			for (auto& value : values)
			{
				value = ReadScalar<std::int32_t>(in);
			}
			return values;
		}

		inline void WriteI64List(std::ostream& out, std::span<const std::int64_t> values)
		{
			WriteSize(out, values.size());
			for (const auto value : values)
			{
				WriteScalar(out, value);
			}
		}

		inline std::vector<std::int64_t> ReadI64List(std::istream& in)
		{
			std::vector<std::int64_t> values(ReadSize(in));
			for (auto& value : values)
			{
				value = ReadScalar<std::int64_t>(in);
			}
			return values;
		}

		inline void WriteU64List(std::ostream& out, std::span<const std::uint64_t> values)
		{
			WriteSize(out, values.size());
			for (const auto value : values)
			{
				WriteScalar(out, value);
			}
		}

		inline std::vector<std::uint64_t> ReadU64List(std::istream& in)
		{
			std::vector<std::uint64_t> values(ReadSize(in));
			for (auto& value : values)
			{
				value = ReadScalar<std::uint64_t>(in);
			}
			return values;
		}

		inline void WriteF64List(std::ostream& out, std::span<const double> values)
		{
			WriteSize(out, values.size());
			for (const auto value : values)
			{
				WriteScalar(out, value);
			}
		}

		inline std::vector<double> ReadF64List(std::istream& in)
		{
			std::vector<double> values(ReadSize(in));
			for (auto& value : values)
			{
				value = ReadScalar<double>(in);
			}
			return values;
		}

		inline void WriteBoolList(std::ostream& out, const std::vector<bool>& values)
		{
			WriteSize(out, values.size());
			for (const auto value : values)
			{
				WriteScalar(out, static_cast<std::uint8_t>(value ? 1 : 0));
			}
		}

		inline std::vector<bool> ReadBoolList(std::istream& in)
		{
			std::vector<bool> values(ReadSize(in));
			for (std::size_t i = 0; i < values.size(); ++i)
			{
				const auto value = ReadScalar<std::uint8_t>(in);
				if (value > 1)
				{
					throw std::runtime_error("Invalid boolean list value in LiteNN model metadata");
				}
				values[i] = value != 0;
			}
			return values;
		}

		inline void WriteSizeList(std::ostream& out, std::span<const std::size_t> values)
		{
			WriteSize(out, values.size());
			for (const auto value : values)
			{
				WriteSize(out, value);
			}
		}

		inline std::vector<std::size_t> ReadSizeList(std::istream& in)
		{
			std::vector<std::size_t> values(ReadSize(in));
			for (auto& value : values)
			{
				value = ReadSize(in);
			}
			return values;
		}

		inline void WriteQuantizationParams(std::ostream& out, const QuantizationParams& params)
		{
			WriteScalar(out, static_cast<std::uint32_t>(params.scheme));
			WriteScalar(out, static_cast<std::uint32_t>(params.granularity));
			WriteScalar(out, static_cast<std::uint32_t>(params.blockFormat));
			WriteDataType(out, params.storageType);
			WriteDataType(out, params.expressedType);
			WriteScalar(out, params.axis);
			WriteSize(out, params.groupSize);
			WriteFloatList(out, params.scales);
			WriteI32List(out, params.zeroPoints);
			WriteSizeList(out, params.expressedShape);
		}

		inline QuantizationParams ReadQuantizationParams(std::istream& in, std::uint32_t version)
		{
			QuantizationParams params;
			params.scheme = static_cast<QuantizationScheme>(ReadScalar<std::uint32_t>(in));
			params.granularity = static_cast<QuantizationGranularity>(ReadScalar<std::uint32_t>(in));
			params.blockFormat = static_cast<QuantizedBlockFormat>(ReadScalar<std::uint32_t>(in));
			params.storageType = ReadDataType(in);
			params.expressedType = ReadDataType(in);
			params.axis = ReadScalar<std::int64_t>(in);
			params.groupSize = ReadSize(in);
			params.scales = ReadFloatList(in);
			params.zeroPoints = ReadI32List(in);
			if (version >= 5)
			{
				params.expressedShape = ReadSizeList(in);
			}
			return params;
		}

		inline void WriteOptionalQuantizationParams(std::ostream& out,
		                                            const std::optional<QuantizationParams>& params)
		{
			WriteScalar(out, static_cast<std::uint8_t>(params.has_value() ? 1 : 0));
			if (!params)
			{
				return;
			}
			WriteQuantizationParams(out, *params);
		}

		inline std::optional<QuantizationParams> ReadOptionalQuantizationParams(std::istream& in,
		                                                                        std::uint32_t version)
		{
			const auto hasValue = ReadScalar<std::uint8_t>(in);
			if (hasValue == 0)
			{
				return std::nullopt;
			}
			if (hasValue != 1)
			{
				throw std::runtime_error("Invalid quantization metadata presence flag");
			}
			return ReadQuantizationParams(in, version);
		}

		inline void WriteShape(std::ostream& out, std::span<const std::size_t> shape)
		{
			WriteSize(out, shape.size());
			for (const auto dim : shape)
			{
				WriteSize(out, dim);
			}
		}

		inline std::vector<std::size_t> ReadShape(std::istream& in)
		{
			std::vector<std::size_t> shape(ReadSize(in));
			for (auto& dim : shape)
			{
				dim = ReadSize(in);
			}
			return shape;
		}

		inline void WriteString(std::ostream& out, std::string_view value)
		{
			WriteSize(out, value.size());
			out.write(value.data(), static_cast<std::streamsize>(value.size()));
			EnsureWrite(out);
		}

		inline std::string ReadString(std::istream& in)
		{
			std::string value(ReadSize(in), '\0');
			in.read(value.data(), static_cast<std::streamsize>(value.size()));
			EnsureRead(in);
			return value;
		}

		inline void WriteStringList(std::ostream& out, std::span<const std::string> values)
		{
			WriteSize(out, values.size());
			for (const auto& value : values)
			{
				WriteString(out, value);
			}
		}

		inline std::vector<std::string> ReadStringList(std::istream& in)
		{
			std::vector<std::string> values(ReadSize(in));
			for (auto& value : values)
			{
				value = ReadString(in);
			}
			return values;
		}

		inline void WriteMetadataValue(std::ostream& out, const ModelMetadataValue& value)
		{
			std::visit(
			    [&](const auto& current) {
				    using T = std::decay_t<decltype(current)>;
				    if constexpr (std::same_as<T, std::int64_t>)
				    {
					    WriteScalar(out, static_cast<std::uint32_t>(MetadataValueKind::Int64));
					    WriteScalar(out, current);
				    }
				    else if constexpr (std::same_as<T, std::uint64_t>)
				    {
					    WriteScalar(out, static_cast<std::uint32_t>(MetadataValueKind::UInt64));
					    WriteScalar(out, current);
				    }
				    else if constexpr (std::same_as<T, double>)
				    {
					    WriteScalar(out, static_cast<std::uint32_t>(MetadataValueKind::Float64));
					    WriteScalar(out, current);
				    }
				    else if constexpr (std::same_as<T, bool>)
				    {
					    WriteScalar(out, static_cast<std::uint32_t>(MetadataValueKind::Bool));
					    WriteScalar(out, static_cast<std::uint8_t>(current ? 1 : 0));
				    }
				    else if constexpr (std::same_as<T, std::string>)
				    {
					    WriteScalar(out, static_cast<std::uint32_t>(MetadataValueKind::String));
					    WriteString(out, current);
				    }
				    else if constexpr (std::same_as<T, std::vector<std::int64_t>>)
				    {
					    WriteScalar(out, static_cast<std::uint32_t>(MetadataValueKind::Int64List));
					    WriteI64List(out, current);
				    }
				    else if constexpr (std::same_as<T, std::vector<std::uint64_t>>)
				    {
					    WriteScalar(out, static_cast<std::uint32_t>(MetadataValueKind::UInt64List));
					    WriteU64List(out, current);
				    }
				    else if constexpr (std::same_as<T, std::vector<double>>)
				    {
					    WriteScalar(out, static_cast<std::uint32_t>(MetadataValueKind::Float64List));
					    WriteF64List(out, current);
				    }
				    else if constexpr (std::same_as<T, std::vector<bool>>)
				    {
					    WriteScalar(out, static_cast<std::uint32_t>(MetadataValueKind::BoolList));
					    WriteBoolList(out, current);
				    }
				    else if constexpr (std::same_as<T, std::vector<std::string>>)
				    {
					    WriteScalar(out, static_cast<std::uint32_t>(MetadataValueKind::StringList));
					    WriteStringList(out, current);
				    }
			    },
			    value);
		}

		inline ModelMetadataValue ReadMetadataValue(std::istream& in)
		{
			const auto kind = static_cast<MetadataValueKind>(ReadScalar<std::uint32_t>(in));
			switch (kind)
			{
			case MetadataValueKind::Int64:
				return ReadScalar<std::int64_t>(in);
			case MetadataValueKind::UInt64:
				return ReadScalar<std::uint64_t>(in);
			case MetadataValueKind::Float64:
				return ReadScalar<double>(in);
			case MetadataValueKind::Bool: {
				const auto value = ReadScalar<std::uint8_t>(in);
				if (value > 1)
				{
					throw std::runtime_error("Invalid boolean value in LiteNN model metadata");
				}
				return value != 0;
			}
			case MetadataValueKind::String:
				return ReadString(in);
			case MetadataValueKind::Int64List:
				return ReadI64List(in);
			case MetadataValueKind::UInt64List:
				return ReadU64List(in);
			case MetadataValueKind::Float64List:
				return ReadF64List(in);
			case MetadataValueKind::BoolList:
				return ReadBoolList(in);
			case MetadataValueKind::StringList:
				return ReadStringList(in);
			}
			throw std::runtime_error("LiteNN model contains an unknown metadata value kind");
		}

		inline void WriteMetadataEntries(std::ostream& out, std::span<const ModelMetadataEntry> entries)
		{
			WriteSize(out, entries.size());
			for (const auto& entry : entries)
			{
				WriteString(out, entry.key);
				WriteMetadataValue(out, entry.value);
			}
		}

		inline std::vector<ModelMetadataEntry> ReadMetadataEntries(std::istream& in)
		{
			std::vector<ModelMetadataEntry> entries(ReadSize(in));
			for (auto& entry : entries)
			{
				entry.key = ReadString(in);
				entry.value = ReadMetadataValue(in);
			}
			return entries;
		}

		inline void WriteNodeOutput(std::ostream& out, NodeOutput output)
		{
			WriteSize(out, output.node);
			WriteSize(out, output.port);
		}

		inline NodeOutput ReadNodeOutput(std::istream& in)
		{
			return { ReadSize(in), ReadSize(in) };
		}

		inline void WriteNodeOutputList(std::ostream& out, std::span<const NodeOutput> outputs)
		{
			WriteSize(out, outputs.size());
			for (const auto output : outputs)
			{
				WriteNodeOutput(out, output);
			}
		}

		inline std::vector<NodeOutput> ReadNodeOutputList(std::istream& in)
		{
			std::vector<NodeOutput> outputs(ReadSize(in));
			for (auto& output : outputs)
			{
				output = ReadNodeOutput(in);
			}
			return outputs;
		}

		inline void WriteOutputInfo(std::ostream& out, const OutputInfo& info)
		{
			WriteDataType(out, info.dtype);
			WriteShape(out, info.shape);
		}

		inline OutputInfo ReadOutputInfo(std::istream& in)
		{
			return { ReadDataType(in), ReadShape(in) };
		}

		inline void WriteOutputInfoList(std::ostream& out, std::span<const OutputInfo> infos)
		{
			WriteSize(out, infos.size());
			for (const auto& info : infos)
			{
				WriteOutputInfo(out, info);
			}
		}

		inline std::vector<OutputInfo> ReadOutputInfoList(std::istream& in)
		{
			std::vector<OutputInfo> infos(ReadSize(in));
			for (auto& info : infos)
			{
				info = ReadOutputInfo(in);
			}
			return infos;
		}

		template <Device D>
		void WriteTensor(std::ostream& out, const Tensor<D>& tensor)
		{
			auto cpuTensor = tensor.CopyToDevice(CPU{});
			WriteDataType(out, cpuTensor.DType());
			WriteShape(out, cpuTensor.Shape().Dims);
			const auto byteCount = cpuTensor.NumElements() * LiteNN::ElementByteSize(cpuTensor.DType());
			out.write(static_cast<const char*>(cpuTensor.RawData()), static_cast<std::streamsize>(byteCount));
			EnsureWrite(out);
		}

		inline Tensor<CPU> ReadTensor(std::istream& in)
		{
			const auto dtype = ReadDataType(in);
			const auto shape = ReadShape(in);
			Tensor<CPU> tensor(Uninitialized, ShapeView{ shape }, dtype, CPU{});
			const auto byteCount = tensor.NumElements() * LiteNN::ElementByteSize(dtype);
			in.read(static_cast<char*>(tensor.RawData()), static_cast<std::streamsize>(byteCount));
			EnsureRead(in);
			return tensor;
		}

		inline void WriteNode(std::ostream& out, const NodeEntry& entry)
		{
			WriteOutputInfoList(out, entry.outputInfos);
			std::visit(
			    [&](const auto& node) {
				    using T = std::decay_t<decltype(node)>;
				    if constexpr (std::same_as<T, ParamRefNode>)
				    {
					    WriteScalar(out, static_cast<std::uint32_t>(NodeKind::ParamRef));
					    WriteSize(out, node.paramIndex);
				    }
				    else if constexpr (std::same_as<T, ConstantNode>)
				    {
					    WriteScalar(out, static_cast<std::uint32_t>(NodeKind::Constant));
					    WriteTensor(out, node.value);
				    }
				    else if constexpr (std::same_as<T, QuantizedConstantNode>)
				    {
					    WriteScalar(out, static_cast<std::uint32_t>(NodeKind::QuantizedConstant));
					    WriteTensor(out, node.storage);
					    WriteQuantizationParams(out, node.params);
				    }
				    else if constexpr (std::same_as<T, VariableRefNode>)
				    {
					    WriteScalar(out, static_cast<std::uint32_t>(NodeKind::VariableRef));
					    WriteSize(out, node.variableIndex);
				    }
				    else if constexpr (std::same_as<T, UnaryOpNode>)
				    {
					    WriteScalar(out, static_cast<std::uint32_t>(NodeKind::UnaryOp));
					    WriteScalar(out, static_cast<std::uint32_t>(node.op));
					    WriteNodeOutput(out, node.input);
				    }
				    else if constexpr (std::same_as<T, BinaryOpNode>)
				    {
					    WriteScalar(out, static_cast<std::uint32_t>(NodeKind::BinaryOp));
					    WriteScalar(out, static_cast<std::uint32_t>(node.op));
					    WriteNodeOutput(out, node.lhs);
					    WriteNodeOutput(out, node.rhs);
				    }
				    else if constexpr (std::same_as<T, CallNode>)
				    {
					    WriteScalar(out, static_cast<std::uint32_t>(NodeKind::Call));
					    WriteSize(out, node.callee);
					    WriteNodeOutputList(out, node.args);
				    }
				    else if constexpr (std::same_as<T, CastNode>)
				    {
					    WriteScalar(out, static_cast<std::uint32_t>(NodeKind::Cast));
					    WriteNodeOutput(out, node.input);
					    WriteDataType(out, node.targetType);
				    }
				    else if constexpr (std::same_as<T, QuantizeNode>)
				    {
					    WriteScalar(out, static_cast<std::uint32_t>(NodeKind::Quantize));
					    WriteNodeOutput(out, node.input);
					    WriteQuantizationParams(out, node.params);
				    }
				    else if constexpr (std::same_as<T, DequantizeNode>)
				    {
					    WriteScalar(out, static_cast<std::uint32_t>(NodeKind::Dequantize));
					    WriteNodeOutput(out, node.input);
					    WriteQuantizationParams(out, node.params);
					    WriteDataType(out, node.targetType);
				    }
				    else if constexpr (std::same_as<T, CondNode>)
				    {
					    WriteScalar(out, static_cast<std::uint32_t>(NodeKind::Cond));
					    WriteNodeOutput(out, node.condition);
					    WriteSize(out, node.thenBranch);
					    WriteSize(out, node.elseBranch);
					    WriteNodeOutputList(out, node.args);
				    }
				    else if constexpr (std::same_as<T, WhileNode>)
				    {
					    WriteScalar(out, static_cast<std::uint32_t>(NodeKind::While));
					    WriteSize(out, node.condBranch);
					    WriteSize(out, node.bodyBranch);
					    WriteNodeOutputList(out, node.initArgs);
				    }
				    else if constexpr (std::same_as<T, SaveActivationNode>)
				    {
					    WriteScalar(out, static_cast<std::uint32_t>(NodeKind::SaveActivation));
					    WriteNodeOutput(out, node.input);
					    WriteSize(out, node.slotId);
				    }
				    else if constexpr (std::same_as<T, LoadActivationNode>)
				    {
					    WriteScalar(out, static_cast<std::uint32_t>(NodeKind::LoadActivation));
					    WriteSize(out, node.slotId);
				    }
				    else if constexpr (std::same_as<T, TapeSaveActivationNode>)
				    {
					    WriteScalar(out, static_cast<std::uint32_t>(NodeKind::TapeSaveActivation));
					    WriteNodeOutput(out, node.input);
					    WriteSize(out, node.tapeSlotId);
				    }
				    else if constexpr (std::same_as<T, TapeLoadActivationNode>)
				    {
					    WriteScalar(out, static_cast<std::uint32_t>(NodeKind::TapeLoadActivation));
					    WriteSize(out, node.tapeSlotId);
				    }
				    else if constexpr (std::same_as<T, ReduceOpNode>)
				    {
					    WriteScalar(out, static_cast<std::uint32_t>(NodeKind::ReduceOp));
					    WriteScalar(out, static_cast<std::uint32_t>(node.op));
					    WriteNodeOutput(out, node.input);
					    WriteSize(out, node.axis);
				    }
				    else if constexpr (std::same_as<T, ReshapeNode>)
				    {
					    WriteScalar(out, static_cast<std::uint32_t>(NodeKind::Reshape));
					    WriteNodeOutput(out, node.input);
					    WriteShape(out, node.targetShape);
				    }
				    else if constexpr (std::same_as<T, ConcatNode>)
				    {
					    WriteScalar(out, static_cast<std::uint32_t>(NodeKind::Concat));
					    WriteNodeOutputList(out, node.inputs);
					    WriteSize(out, node.axis);
				    }
				    else if constexpr (std::same_as<T, SliceNode>)
				    {
					    WriteScalar(out, static_cast<std::uint32_t>(NodeKind::Slice));
					    WriteNodeOutput(out, node.input);
					    WriteSize(out, node.axis);
					    WriteSize(out, node.start);
					    WriteSize(out, node.length);
				    }
				    else if constexpr (std::same_as<T, GetRowsNode>)
				    {
					    WriteScalar(out, static_cast<std::uint32_t>(NodeKind::GetRows));
					    WriteNodeOutput(out, node.data);
					    WriteNodeOutput(out, node.indices);
				    }
				    else if constexpr (std::same_as<T, ArgsortNode>)
				    {
					    WriteScalar(out, static_cast<std::uint32_t>(NodeKind::Argsort));
					    WriteNodeOutput(out, node.input);
					    WriteScalar(out, static_cast<std::uint32_t>(node.order));
				    }
				    else if constexpr (std::same_as<T, MulMatIdNode>)
				    {
					    WriteScalar(out, static_cast<std::uint32_t>(NodeKind::MulMatId));
					    WriteNodeOutput(out, node.as);
					    WriteNodeOutput(out, node.b);
					    WriteNodeOutput(out, node.ids);
				    }
				    else if constexpr (std::same_as<T, FusedOpNode>)
				    {
					    WriteScalar(out, static_cast<std::uint32_t>(NodeKind::FusedOp));
					    WriteScalar(out, static_cast<std::uint32_t>(node.pattern));
					    WriteSize(out, node.body);
					    WriteNodeOutputList(out, node.args);
				    }
			    },
			    entry.node);
		}

		inline NodeVariant ReadNodePayload(std::istream& in, std::uint32_t version)
		{
			const auto kind = static_cast<NodeKind>(ReadScalar<std::uint32_t>(in));
			switch (kind)
			{
			case NodeKind::ParamRef:
				return ParamRefNode{ ReadSize(in) };
			case NodeKind::Constant:
				return ConstantNode{ ReadTensor(in).CopyToDevice(PolymorphicDevice{ CPU{} }) };
			case NodeKind::QuantizedConstant: {
				auto storage = ReadTensor(in).CopyToDevice(PolymorphicDevice{ CPU{} });
				auto params = ReadQuantizationParams(in, version);
				return QuantizedConstantNode{ std::move(storage), std::move(params) };
			}
			case NodeKind::VariableRef:
				return VariableRefNode{ ReadSize(in) };
			case NodeKind::UnaryOp: {
				const auto op = static_cast<UnaryOp>(ReadScalar<std::uint32_t>(in));
				return UnaryOpNode{ op, ReadNodeOutput(in) };
			}
			case NodeKind::BinaryOp: {
				const auto op = static_cast<BinaryOp>(ReadScalar<std::uint32_t>(in));
				const auto lhs = ReadNodeOutput(in);
				const auto rhs = ReadNodeOutput(in);
				return BinaryOpNode{ op, lhs, rhs };
			}
			case NodeKind::Call: {
				const auto callee = ReadSize(in);
				return CallNode{ callee, ReadNodeOutputList(in) };
			}
			case NodeKind::Cast: {
				const auto input = ReadNodeOutput(in);
				return CastNode{ input, ReadDataType(in) };
			}
			case NodeKind::Quantize: {
				const auto input = ReadNodeOutput(in);
				auto params = ReadQuantizationParams(in, version);
				return QuantizeNode{ input, std::move(params) };
			}
			case NodeKind::Dequantize: {
				const auto input = ReadNodeOutput(in);
				auto params = ReadQuantizationParams(in, version);
				return DequantizeNode{ input, std::move(params), ReadDataType(in) };
			}
			case NodeKind::Cond: {
				const auto condition = ReadNodeOutput(in);
				const auto thenBranch = ReadSize(in);
				const auto elseBranch = ReadSize(in);
				return CondNode{ condition, thenBranch, elseBranch, ReadNodeOutputList(in) };
			}
			case NodeKind::While: {
				const auto condBranch = ReadSize(in);
				const auto bodyBranch = ReadSize(in);
				return WhileNode{ condBranch, bodyBranch, ReadNodeOutputList(in) };
			}
			case NodeKind::SaveActivation: {
				const auto input = ReadNodeOutput(in);
				return SaveActivationNode{ input, ReadSize(in) };
			}
			case NodeKind::LoadActivation:
				return LoadActivationNode{ ReadSize(in) };
			case NodeKind::TapeSaveActivation: {
				const auto input = ReadNodeOutput(in);
				return TapeSaveActivationNode{ input, ReadSize(in) };
			}
			case NodeKind::TapeLoadActivation:
				return TapeLoadActivationNode{ ReadSize(in) };
			case NodeKind::ReduceOp: {
				const auto op = static_cast<ReduceOp>(ReadScalar<std::uint32_t>(in));
				const auto input = ReadNodeOutput(in);
				return ReduceOpNode{ op, input, ReadSize(in) };
			}
			case NodeKind::Reshape: {
				const auto input = ReadNodeOutput(in);
				return ReshapeNode{ input, ReadShape(in) };
			}
			case NodeKind::Concat: {
				auto inputs = ReadNodeOutputList(in);
				return ConcatNode{ std::move(inputs), ReadSize(in) };
			}
			case NodeKind::Slice: {
				const auto input = ReadNodeOutput(in);
				const auto axis = ReadSize(in);
				const auto start = ReadSize(in);
				return SliceNode{ input, axis, start, ReadSize(in) };
			}
			case NodeKind::GetRows: {
				const auto data = ReadNodeOutput(in);
				const auto indices = ReadNodeOutput(in);
				return GetRowsNode{ data, indices };
			}
			case NodeKind::Argsort: {
				const auto input = ReadNodeOutput(in);
				const auto order = static_cast<SortOrder>(ReadScalar<std::uint32_t>(in));
				return ArgsortNode{ input, order };
			}
			case NodeKind::MulMatId: {
				const auto as = ReadNodeOutput(in);
				const auto b = ReadNodeOutput(in);
				const auto ids = ReadNodeOutput(in);
				return MulMatIdNode{ as, b, ids };
			}
			case NodeKind::FusedOp: {
				const auto pattern = static_cast<FusionPattern>(ReadScalar<std::uint32_t>(in));
				const auto body = ReadSize(in);
				return FusedOpNode{ pattern, body, ReadNodeOutputList(in) };
			}
			}
			throw std::runtime_error("LiteNN model contains an unknown node kind");
		}

		inline void WriteSubgraph(std::ostream& out, const Subgraph& subgraph)
		{
			WriteSize(out, subgraph.Params().size());
			for (const auto& param : subgraph.Params())
			{
				WriteDataType(out, param.dtype);
				WriteShape(out, param.shape);
			}

			WriteSize(out, subgraph.NodeCount());
			for (const auto& node : subgraph.Nodes())
			{
				WriteNode(out, node);
			}
			WriteNodeOutputList(out, subgraph.Results());
		}

		inline Subgraph ReadSubgraph(std::istream& in, std::uint32_t version)
		{
			Subgraph subgraph;
			const auto paramCount = ReadSize(in);
			for (std::size_t i = 0; i < paramCount; ++i)
			{
				const auto dtype = ReadDataType(in);
				auto shape = ReadShape(in);
				(void)subgraph.AddParam(dtype, std::move(shape));
			}

			const auto nodeCount = ReadSize(in);
			if (nodeCount < paramCount)
			{
				throw std::runtime_error("LiteNN model subgraph node count is smaller than parameter count");
			}
			for (std::size_t nodeId = 0; nodeId < nodeCount; ++nodeId)
			{
				auto outputInfos = ReadOutputInfoList(in);
				auto node = ReadNodePayload(in, version);
				if (nodeId < paramCount)
				{
					const auto* param = std::get_if<ParamRefNode>(&node);
					if (!param || param->paramIndex != nodeId)
					{
						throw std::runtime_error("LiteNN model parameter nodes must be serialized first");
					}
					continue;
				}
				(void)subgraph.AddNode(std::move(node), std::move(outputInfos));
			}
			subgraph.SetResults(ReadNodeOutputList(in));
			return subgraph;
		}
	} // namespace Detail

	inline void SaveModel(const Graph& graph, const std::filesystem::path& path)
	{
		Validation::ValidateGraph(graph);
		std::ofstream out(path, std::ios::binary);
		if (!out)
		{
			throw std::runtime_error("Failed to open LiteNN model file for writing");
		}

		out.write(Detail::kModelMagic.data(), static_cast<std::streamsize>(Detail::kModelMagic.size()));
		Detail::EnsureWrite(out);
		Detail::WriteScalar(out, Detail::kModelVersion);
		Detail::WriteSize(out, graph.Forward());
		Detail::WriteScalar(out, static_cast<std::uint8_t>(graph.Backward().has_value() ? 1 : 0));
		if (graph.Backward())
		{
			Detail::WriteSize(out, *graph.Backward());
		}
		Detail::WriteStringList(out, graph.InputNames());
		Detail::WriteStringList(out, graph.OutputNames());

		Detail::WriteSize(out, graph.VariableCount());
		Detail::WriteStringList(out, graph.VariableNames());
		Detail::WriteMetadataEntries(out, graph.Metadata());
		for (const auto& variable : graph.Variables())
		{
			Detail::WriteTensor(out, variable->Data());
			Detail::WriteOptionalQuantizationParams(out, variable->Quantization());
		}

		Detail::WriteSize(out, graph.ActivationSlotCount());
		for (std::size_t i = 0; i < graph.ActivationSlotCount(); ++i)
		{
			const auto& slot = graph.GetActivationSlot(i);
			Detail::WriteDataType(out, slot.dtype);
			Detail::WriteShape(out, slot.shape);
		}

		Detail::WriteSize(out, graph.TapeSlotCount());
		for (std::size_t i = 0; i < graph.TapeSlotCount(); ++i)
		{
			const auto& slot = graph.GetTapeSlot(i);
			Detail::WriteDataType(out, slot.dtype);
			Detail::WriteShape(out, slot.shape);
		}

		Detail::WriteSize(out, graph.SubgraphCount());
		for (SubgraphId id = 0; id < graph.SubgraphCount(); ++id)
		{
			Detail::WriteSubgraph(out, graph.GetSubgraph(id));
		}
	}

	inline Graph LoadModel(const std::filesystem::path& path)
	{
		std::ifstream in(path, std::ios::binary);
		if (!in)
		{
			throw std::runtime_error("Failed to open LiteNN model file for reading");
		}

		std::array<char, Detail::kModelMagic.size()> magic{};
		in.read(magic.data(), static_cast<std::streamsize>(magic.size()));
		Detail::EnsureRead(in);
		if (magic != Detail::kModelMagic)
		{
			throw std::runtime_error("Invalid LiteNN model magic header");
		}

		const auto version = Detail::ReadScalar<std::uint32_t>(in);
		if (version == 0 || version > Detail::kModelVersion)
		{
			throw std::runtime_error("Unsupported LiteNN model version");
		}

		const auto forward = Detail::ReadSize(in);
		const auto hasBackward = Detail::ReadScalar<std::uint8_t>(in) != 0;
		std::optional<SubgraphId> backward;
		if (hasBackward)
		{
			backward = Detail::ReadSize(in);
		}
		std::vector<std::string> inputNames;
		std::vector<std::string> outputNames;
		if (version >= 2)
		{
			inputNames = Detail::ReadStringList(in);
			outputNames = Detail::ReadStringList(in);
		}

		Graph graph;
		const auto variableCount = Detail::ReadSize(in);
		std::vector<std::string> variableNames;
		std::vector<ModelMetadataEntry> metadata;
		if (version >= 6)
		{
			variableNames = Detail::ReadStringList(in);
			metadata = Detail::ReadMetadataEntries(in);
		}
		for (std::size_t i = 0; i < variableCount; ++i)
		{
			auto tensor = Detail::ReadTensor(in);
			std::optional<QuantizationParams> quantization;
			if (version >= 4)
			{
				quantization = Detail::ReadOptionalQuantizationParams(in, version);
			}
			auto variable = Variable::Create(std::move(tensor));
			variable->SetQuantization(std::move(quantization));
			graph.AddVariable(std::move(variable));
		}

		const auto activationSlotCount = Detail::ReadSize(in);
		for (std::size_t i = 0; i < activationSlotCount; ++i)
		{
			graph.AddActivationSlot({ Detail::ReadDataType(in), Detail::ReadShape(in) });
		}

		const auto tapeSlotCount = Detail::ReadSize(in);
		for (std::size_t i = 0; i < tapeSlotCount; ++i)
		{
			graph.AddTapeSlot({ Detail::ReadDataType(in), Detail::ReadShape(in) });
		}

		const auto subgraphCount = Detail::ReadSize(in);
		for (std::size_t i = 0; i < subgraphCount; ++i)
		{
			graph.AddSubgraph(Detail::ReadSubgraph(in, version));
		}
		graph.SetForward(forward);
		if (backward)
		{
			graph.SetBackward(*backward);
		}
		graph.SetInputNames(std::move(inputNames));
		graph.SetVariableNames(std::move(variableNames));
		graph.SetOutputNames(std::move(outputNames));
		graph.SetMetadata(std::move(metadata));

		if (in.peek() != std::char_traits<char>::eof())
		{
			throw std::runtime_error("LiteNN model contains trailing bytes");
		}

		Validation::ValidateGraph(graph);
		return graph;
	}
} // namespace LiteNN::Serialization

#endif
