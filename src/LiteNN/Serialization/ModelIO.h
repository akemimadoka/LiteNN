#include <LiteNN/ExecutablePlan.h>
#include <LiteNN/Graph.h>
#include <LiteNN/Validation/GraphValidator.h>

#include <algorithm>
#include <array>
#include <concepts>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <memory>
#include <optional>
#include <span>
#include <stdexcept>
#include <string>
#include <string_view>
#include <type_traits>
#include <utility>
#include <vector>

#ifndef LITENN_SERIALIZATION_MODELIO_H
#define LITENN_SERIALIZATION_MODELIO_H

namespace LiteNN::Serialization
{
	/// Controls when variable tensor payloads are written to a sibling external weight file.
	struct ExternalWeightSaveOptions
	{
		std::uint64_t minVariableBytes{ 0 };
		std::uint64_t alignment{ 64 };
	};

	namespace Detail
	{
		constexpr std::array<char, 8> kGraphArchiveMagic = { 'L', 'T', 'N', 'N', 'M', 'D', 'L', '\0' };
		constexpr std::uint32_t kGraphArchiveVersion = 21;

		enum class VariablePayloadKind : std::uint8_t
		{
			Inline = 0,
			External = 1,
		};

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

		enum class GraphArchiveNodeKind : std::uint32_t
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
			Permute,
			BroadcastTo,
			Pad,
			Gather,
			Scatter,
			Scan,
			SSMScan,
			RWKVWKV,
			Softmax,
			Normalization,
			BatchMatMul,
			Im2Col,
			Conv2D,
			Pool2D,
			ConvTranspose2D,
			Upsample,
			OutProd,
			TimestepEmbedding,
			SolveTri,
			SGDStep,
			AdamWStep,
			CrossEntropyLoss,
			CrossEntropyLossBackward,
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

		inline QuantizationParams ReadQuantizationParams(std::istream& in)
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
			params.expressedShape = ReadSizeList(in);
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

		inline std::optional<QuantizationParams> ReadOptionalQuantizationParams(std::istream& in)
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
			return ReadQuantizationParams(in);
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

		inline std::uint64_t AlignUp(std::uint64_t value, std::uint64_t alignment)
		{
			if (alignment <= 1)
			{
				return value;
			}
			const auto remainder = value % alignment;
			return remainder == 0 ? value : value + (alignment - remainder);
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

		inline std::filesystem::path ResolveExternalPath(const std::filesystem::path& modelPath,
		                                                 const std::string& externalPath)
		{
			std::filesystem::path path(externalPath);
			if (path.is_relative() && modelPath.has_parent_path())
			{
				path = modelPath.parent_path() / path;
			}
			return path.lexically_normal();
		}

		inline std::string ExternalPathText(const std::filesystem::path& modelPath,
		                                    const std::filesystem::path& externalPath)
		{
			const auto modelDirectory =
			    modelPath.has_parent_path() ? std::filesystem::absolute(modelPath.parent_path()).lexically_normal()
			                               : std::filesystem::current_path().lexically_normal();
			const auto externalAbsolute = std::filesystem::absolute(externalPath).lexically_normal();
			const auto relative = externalAbsolute.lexically_relative(modelDirectory);
			if (!relative.empty())
			{
				return relative.string();
			}
			return externalPath.string();
		}

		inline std::vector<std::byte> ReadExternalWeightBytes(const std::filesystem::path& path)
		{
			std::ifstream in(path, std::ios::binary);
			if (!in)
			{
				throw std::runtime_error("Failed to open external LiteNN weight file for reading: " + path.string());
			}
			in.seekg(0, std::ios::end);
			const auto end = in.tellg();
			if (end == std::streampos(-1))
			{
				throw std::runtime_error("Failed to determine external LiteNN weight file size: " + path.string());
			}
			std::vector<std::byte> bytes(static_cast<std::size_t>(end));
			in.seekg(0, std::ios::beg);
			std::size_t offset = 0;
			constexpr std::size_t kChunkSize = 64uz * 1024uz * 1024uz;
			while (offset < bytes.size())
			{
				const auto chunk = std::min(kChunkSize, bytes.size() - offset);
				in.read(reinterpret_cast<char*>(bytes.data() + offset), static_cast<std::streamsize>(chunk));
				EnsureRead(in);
				offset += chunk;
			}
			return bytes;
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

		inline void WriteOptionalNodeOutput(std::ostream& out, const std::optional<NodeOutput>& output)
		{
			WriteScalar(out, static_cast<std::uint8_t>(output.has_value() ? 1 : 0));
			if (output)
			{
				WriteNodeOutput(out, *output);
			}
		}

		inline std::optional<NodeOutput> ReadOptionalNodeOutput(std::istream& in)
		{
			const auto hasValue = ReadScalar<std::uint8_t>(in);
			if (hasValue == 0)
			{
				return std::nullopt;
			}
			if (hasValue != 1)
			{
				throw std::runtime_error("Invalid optional NodeOutput flag in LiteNN model");
			}
			return ReadNodeOutput(in);
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

		template <Device D>
		std::uint64_t TensorByteSize(const Tensor<D>& tensor)
		{
			return static_cast<std::uint64_t>(tensor.NumElements()) * LiteNN::ElementByteSize(tensor.DType());
		}

		inline std::uint64_t TensorSpecByteSize(const TensorSpec& spec)
		{
			std::uint64_t elementCount = 1;
			for (const auto dim : spec.shape)
			{
				elementCount *= static_cast<std::uint64_t>(dim);
			}
			return elementCount * LiteNN::ElementByteSize(spec.dtype);
		}

		template <Device D>
		void WriteTensorMetadata(std::ostream& out, const Tensor<D>& tensor)
		{
			WriteDataType(out, tensor.DType());
			WriteShape(out, tensor.Shape().Dims);
		}

		inline void WriteZeroBytes(std::ostream& out, std::uint64_t count)
		{
			std::array<char, 256> zeros{};
			while (count != 0)
			{
				const auto chunk = std::min<std::uint64_t>(count, zeros.size());
				out.write(zeros.data(), static_cast<std::streamsize>(chunk));
				EnsureWrite(out);
				count -= chunk;
			}
		}

		template <Device D>
		void WriteTensorPayload(std::ostream& out, const Tensor<D>& tensor)
		{
			if constexpr (std::same_as<D, CPU>)
			{
				out.write(static_cast<const char*>(tensor.RawData()), static_cast<std::streamsize>(TensorByteSize(tensor)));
				EnsureWrite(out);
			}
			else
			{
				auto cpuTensor = tensor.CopyToDevice(CPU{});
				out.write(static_cast<const char*>(cpuTensor.RawData()),
				          static_cast<std::streamsize>(TensorByteSize(cpuTensor)));
				EnsureWrite(out);
			}
		}

		inline TensorSpec ReadTensorMetadata(std::istream& in)
		{
			return { ReadDataType(in), ReadShape(in) };
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
					    WriteScalar(out, static_cast<std::uint32_t>(GraphArchiveNodeKind::ParamRef));
					    WriteSize(out, node.paramIndex);
				    }
				    else if constexpr (std::same_as<T, ConstantNode>)
				    {
					    WriteScalar(out, static_cast<std::uint32_t>(GraphArchiveNodeKind::Constant));
					    WriteTensor(out, node.value);
				    }
				    else if constexpr (std::same_as<T, QuantizedConstantNode>)
				    {
					    WriteScalar(out, static_cast<std::uint32_t>(GraphArchiveNodeKind::QuantizedConstant));
					    WriteTensor(out, node.storage);
					    WriteQuantizationParams(out, node.params);
				    }
				    else if constexpr (std::same_as<T, VariableRefNode>)
				    {
					    WriteScalar(out, static_cast<std::uint32_t>(GraphArchiveNodeKind::VariableRef));
					    WriteSize(out, node.variableIndex);
				    }
				    else if constexpr (std::same_as<T, UnaryOpNode>)
				    {
					    WriteScalar(out, static_cast<std::uint32_t>(GraphArchiveNodeKind::UnaryOp));
					    WriteScalar(out, static_cast<std::uint32_t>(node.op));
					    WriteNodeOutput(out, node.input);
				    }
				    else if constexpr (std::same_as<T, BinaryOpNode>)
				    {
					    WriteScalar(out, static_cast<std::uint32_t>(GraphArchiveNodeKind::BinaryOp));
					    WriteScalar(out, static_cast<std::uint32_t>(node.op));
					    WriteNodeOutput(out, node.lhs);
					    WriteNodeOutput(out, node.rhs);
				    }
				    else if constexpr (std::same_as<T, CallNode>)
				    {
					    WriteScalar(out, static_cast<std::uint32_t>(GraphArchiveNodeKind::Call));
					    WriteSize(out, node.callee);
					    WriteNodeOutputList(out, node.args);
				    }
				    else if constexpr (std::same_as<T, CastNode>)
				    {
					    WriteScalar(out, static_cast<std::uint32_t>(GraphArchiveNodeKind::Cast));
					    WriteNodeOutput(out, node.input);
					    WriteDataType(out, node.targetType);
				    }
				    else if constexpr (std::same_as<T, QuantizeNode>)
				    {
					    WriteScalar(out, static_cast<std::uint32_t>(GraphArchiveNodeKind::Quantize));
					    WriteNodeOutput(out, node.input);
					    WriteQuantizationParams(out, node.params);
				    }
				    else if constexpr (std::same_as<T, DequantizeNode>)
				    {
					    WriteScalar(out, static_cast<std::uint32_t>(GraphArchiveNodeKind::Dequantize));
					    WriteNodeOutput(out, node.input);
					    WriteQuantizationParams(out, node.params);
					    WriteDataType(out, node.targetType);
				    }
				    else if constexpr (std::same_as<T, CondNode>)
				    {
					    WriteScalar(out, static_cast<std::uint32_t>(GraphArchiveNodeKind::Cond));
					    WriteNodeOutput(out, node.condition);
					    WriteSize(out, node.thenBranch);
					    WriteSize(out, node.elseBranch);
					    WriteNodeOutputList(out, node.args);
				    }
				    else if constexpr (std::same_as<T, WhileNode>)
				    {
					    WriteScalar(out, static_cast<std::uint32_t>(GraphArchiveNodeKind::While));
					    WriteSize(out, node.condBranch);
					    WriteSize(out, node.bodyBranch);
					    WriteNodeOutputList(out, node.initArgs);
				    }
				    else if constexpr (std::same_as<T, SaveActivationNode>)
				    {
					    WriteScalar(out, static_cast<std::uint32_t>(GraphArchiveNodeKind::SaveActivation));
					    WriteNodeOutput(out, node.input);
					    WriteSize(out, node.slotId);
				    }
				    else if constexpr (std::same_as<T, LoadActivationNode>)
				    {
					    WriteScalar(out, static_cast<std::uint32_t>(GraphArchiveNodeKind::LoadActivation));
					    WriteSize(out, node.slotId);
				    }
				    else if constexpr (std::same_as<T, TapeSaveActivationNode>)
				    {
					    WriteScalar(out, static_cast<std::uint32_t>(GraphArchiveNodeKind::TapeSaveActivation));
					    WriteNodeOutput(out, node.input);
					    WriteSize(out, node.tapeSlotId);
				    }
				    else if constexpr (std::same_as<T, TapeLoadActivationNode>)
				    {
					    WriteScalar(out, static_cast<std::uint32_t>(GraphArchiveNodeKind::TapeLoadActivation));
					    WriteSize(out, node.tapeSlotId);
				    }
				    else if constexpr (std::same_as<T, ReduceOpNode>)
				    {
					    WriteScalar(out, static_cast<std::uint32_t>(GraphArchiveNodeKind::ReduceOp));
					    WriteScalar(out, static_cast<std::uint32_t>(node.op));
					    WriteNodeOutput(out, node.input);
					    WriteSize(out, node.axis);
				    }
				    else if constexpr (std::same_as<T, ReshapeNode>)
				    {
					    WriteScalar(out, static_cast<std::uint32_t>(GraphArchiveNodeKind::Reshape));
					    WriteNodeOutput(out, node.input);
					    WriteShape(out, node.targetShape);
				    }
				    else if constexpr (std::same_as<T, PermuteNode>)
				    {
					    WriteScalar(out, static_cast<std::uint32_t>(GraphArchiveNodeKind::Permute));
					    WriteNodeOutput(out, node.input);
					    WriteShape(out, node.permutation);
				    }
				    else if constexpr (std::same_as<T, BroadcastToNode>)
				    {
					    WriteScalar(out, static_cast<std::uint32_t>(GraphArchiveNodeKind::BroadcastTo));
					    WriteNodeOutput(out, node.input);
					    WriteShape(out, node.targetShape);
				    }
				    else if constexpr (std::same_as<T, PadNode>)
				    {
					    WriteScalar(out, static_cast<std::uint32_t>(GraphArchiveNodeKind::Pad));
					    WriteNodeOutput(out, node.input);
					    WriteShape(out, node.lowPads);
					    WriteShape(out, node.highPads);
					    WriteScalar(out, static_cast<std::uint32_t>(node.mode));
					    WriteScalar(out, node.constantValue);
				    }
				    else if constexpr (std::same_as<T, GatherNode>)
				    {
					    WriteScalar(out, static_cast<std::uint32_t>(GraphArchiveNodeKind::Gather));
					    WriteNodeOutput(out, node.data);
					    WriteNodeOutput(out, node.indices);
					    WriteSize(out, node.axis);
				    }
				    else if constexpr (std::same_as<T, ScatterNode>)
				    {
					    WriteScalar(out, static_cast<std::uint32_t>(GraphArchiveNodeKind::Scatter));
					    WriteNodeOutput(out, node.data);
					    WriteNodeOutput(out, node.indices);
					    WriteNodeOutput(out, node.updates);
					    WriteSize(out, node.axis);
					    WriteScalar(out, static_cast<std::uint32_t>(node.mode));
				    }
				    else if constexpr (std::same_as<T, ScanNode>)
				    {
					    WriteScalar(out, static_cast<std::uint32_t>(GraphArchiveNodeKind::Scan));
					    WriteNodeOutput(out, node.input);
					    WriteSize(out, node.axis);
					    WriteScalar(out, static_cast<std::uint32_t>(node.op));
				    }
				    else if constexpr (std::same_as<T, SSMScanNode>)
				    {
					    WriteScalar(out, static_cast<std::uint32_t>(GraphArchiveNodeKind::SSMScan));
					    WriteNodeOutput(out, node.state);
					    WriteNodeOutput(out, node.dt);
					    WriteNodeOutput(out, node.a);
					    WriteNodeOutput(out, node.b);
					    WriteNodeOutput(out, node.c);
					    WriteOptionalNodeOutput(out, node.d);
				    }
				    else if constexpr (std::same_as<T, RWKVWKVNode>)
				    {
					    WriteScalar(out, static_cast<std::uint32_t>(GraphArchiveNodeKind::RWKVWKV));
					    WriteNodeOutput(out, node.key);
					    WriteNodeOutput(out, node.value);
					    WriteNodeOutput(out, node.receptance);
					    WriteNodeOutput(out, node.timeDecay);
					    WriteNodeOutput(out, node.timeFirst);
				    }
				    else if constexpr (std::same_as<T, SoftmaxNode>)
				    {
					    WriteScalar(out, static_cast<std::uint32_t>(GraphArchiveNodeKind::Softmax));
					    WriteNodeOutput(out, node.input);
					    WriteSize(out, node.axis);
				    }
				    else if constexpr (std::same_as<T, CrossEntropyLossNode>)
				    {
					    WriteScalar(out, static_cast<std::uint32_t>(GraphArchiveNodeKind::CrossEntropyLoss));
					    WriteNodeOutput(out, node.logits);
					    WriteNodeOutput(out, node.labels);
				    }
				    else if constexpr (std::same_as<T, CrossEntropyLossBackwardNode>)
				    {
					    WriteScalar(out, static_cast<std::uint32_t>(GraphArchiveNodeKind::CrossEntropyLossBackward));
					    WriteNodeOutput(out, node.grad);
					    WriteNodeOutput(out, node.logits);
					    WriteNodeOutput(out, node.labels);
				    }
				    else if constexpr (std::same_as<T, NormalizationNode>)
				    {
					    WriteScalar(out, static_cast<std::uint32_t>(GraphArchiveNodeKind::Normalization));
					    WriteNodeOutput(out, node.input);
					    WriteOptionalNodeOutput(out, node.scale);
					    WriteOptionalNodeOutput(out, node.bias);
					    WriteScalar(out, static_cast<std::uint32_t>(node.mode));
					    WriteSize(out, node.axis);
					    WriteSize(out, node.groupCount);
					    WriteScalar(out, node.epsilon);
				    }
				    else if constexpr (std::same_as<T, BatchMatMulNode>)
				    {
					    WriteScalar(out, static_cast<std::uint32_t>(GraphArchiveNodeKind::BatchMatMul));
					    WriteNodeOutput(out, node.lhs);
					    WriteNodeOutput(out, node.rhs);
				    }
				    else if constexpr (std::same_as<T, OutProdNode>)
				    {
					    WriteScalar(out, static_cast<std::uint32_t>(GraphArchiveNodeKind::OutProd));
					    WriteNodeOutput(out, node.lhs);
					    WriteNodeOutput(out, node.rhs);
				    }
				    else if constexpr (std::same_as<T, TimestepEmbeddingNode>)
				    {
					    WriteScalar(out, static_cast<std::uint32_t>(GraphArchiveNodeKind::TimestepEmbedding));
					    WriteNodeOutput(out, node.timesteps);
					    WriteSize(out, node.dim);
					    WriteSize(out, node.maxPeriod);
				    }
				    else if constexpr (std::same_as<T, SolveTriNode>)
				    {
					    WriteScalar(out, static_cast<std::uint32_t>(GraphArchiveNodeKind::SolveTri));
					    WriteNodeOutput(out, node.a);
					    WriteNodeOutput(out, node.b);
					    WriteScalar(out, node.lower);
					    WriteScalar(out, node.unitDiagonal);
				    }
				    else if constexpr (std::same_as<T, SGDStepNode>)
				    {
					    WriteScalar(out, static_cast<std::uint32_t>(GraphArchiveNodeKind::SGDStep));
					    WriteNodeOutput(out, node.parameter);
					    WriteNodeOutput(out, node.gradient);
					    WriteOptionalNodeOutput(out, node.velocity);
					    WriteScalar(out, node.learningRate);
					    WriteScalar(out, node.momentum);
					    WriteScalar(out, node.weightDecay);
					    WriteScalar(out, node.nesterov);
				    }
				    else if constexpr (std::same_as<T, AdamWStepNode>)
				    {
					    WriteScalar(out, static_cast<std::uint32_t>(GraphArchiveNodeKind::AdamWStep));
					    WriteNodeOutput(out, node.parameter);
					    WriteNodeOutput(out, node.gradient);
					    WriteNodeOutput(out, node.firstMoment);
					    WriteNodeOutput(out, node.secondMoment);
					    WriteScalar(out, node.learningRate);
					    WriteScalar(out, node.beta1);
					    WriteScalar(out, node.beta2);
					    WriteScalar(out, node.epsilon);
					    WriteScalar(out, node.weightDecay);
					    WriteSize(out, node.step);
				    }
				    else if constexpr (std::same_as<T, Im2ColNode>)
				    {
					    WriteScalar(out, static_cast<std::uint32_t>(GraphArchiveNodeKind::Im2Col));
					    WriteNodeOutput(out, node.input);
					    WriteShape(out, node.kernelShape);
					    WriteShape(out, node.strides);
					    WriteShape(out, node.dilations);
					    WriteShape(out, node.lowPads);
					    WriteShape(out, node.highPads);
				    }
				    else if constexpr (std::same_as<T, Conv2DNode>)
				    {
					    WriteScalar(out, static_cast<std::uint32_t>(GraphArchiveNodeKind::Conv2D));
					    WriteNodeOutput(out, node.input);
					    WriteNodeOutput(out, node.weight);
					    WriteOptionalNodeOutput(out, node.bias);
					    WriteShape(out, node.strides);
					    WriteShape(out, node.dilations);
					    WriteShape(out, node.lowPads);
					    WriteShape(out, node.highPads);
					    WriteSize(out, node.groupCount);
				    }
				    else if constexpr (std::same_as<T, ConvTranspose2DNode>)
				    {
					    WriteScalar(out, static_cast<std::uint32_t>(GraphArchiveNodeKind::ConvTranspose2D));
					    WriteNodeOutput(out, node.input);
					    WriteNodeOutput(out, node.weight);
					    WriteOptionalNodeOutput(out, node.bias);
					    WriteShape(out, node.strides);
					    WriteShape(out, node.dilations);
					    WriteShape(out, node.lowPads);
					    WriteShape(out, node.highPads);
					    WriteShape(out, node.outputPads);
					    WriteSize(out, node.groupCount);
				    }
				    else if constexpr (std::same_as<T, Pool2DNode>)
				    {
					    WriteScalar(out, static_cast<std::uint32_t>(GraphArchiveNodeKind::Pool2D));
					    WriteNodeOutput(out, node.input);
					    WriteScalar(out, static_cast<std::uint32_t>(node.mode));
					    WriteShape(out, node.kernelShape);
					    WriteShape(out, node.strides);
					    WriteShape(out, node.lowPads);
					    WriteShape(out, node.highPads);
					    WriteScalar(out, node.countIncludePad);
				    }
				    else if constexpr (std::same_as<T, UpsampleNode>)
				    {
					    WriteScalar(out, static_cast<std::uint32_t>(GraphArchiveNodeKind::Upsample));
					    WriteNodeOutput(out, node.input);
					    WriteScalar(out, static_cast<std::uint32_t>(node.mode));
					    WriteShape(out, node.outputSpatialShape);
					    WriteScalar(out, node.alignCorners);
				    }
				    else if constexpr (std::same_as<T, ConcatNode>)
				    {
					    WriteScalar(out, static_cast<std::uint32_t>(GraphArchiveNodeKind::Concat));
					    WriteNodeOutputList(out, node.inputs);
					    WriteSize(out, node.axis);
				    }
				    else if constexpr (std::same_as<T, SliceNode>)
				    {
					    WriteScalar(out, static_cast<std::uint32_t>(GraphArchiveNodeKind::Slice));
					    WriteNodeOutput(out, node.input);
					    WriteSize(out, node.axis);
					    WriteSize(out, node.start);
					    WriteSize(out, node.length);
				    }
				    else if constexpr (std::same_as<T, GetRowsNode>)
				    {
					    WriteScalar(out, static_cast<std::uint32_t>(GraphArchiveNodeKind::GetRows));
					    WriteNodeOutput(out, node.data);
					    WriteNodeOutput(out, node.indices);
				    }
				    else if constexpr (std::same_as<T, ArgsortNode>)
				    {
					    WriteScalar(out, static_cast<std::uint32_t>(GraphArchiveNodeKind::Argsort));
					    WriteNodeOutput(out, node.input);
					    WriteSize(out, node.axis);
					    WriteScalar(out, static_cast<std::uint32_t>(node.order));
				    }
				    else if constexpr (std::same_as<T, MulMatIdNode>)
				    {
					    WriteScalar(out, static_cast<std::uint32_t>(GraphArchiveNodeKind::MulMatId));
					    WriteNodeOutput(out, node.as);
					    WriteNodeOutput(out, node.b);
					    WriteNodeOutput(out, node.ids);
				    }
				    else if constexpr (std::same_as<T, FusedOpNode>)
				    {
					    WriteScalar(out, static_cast<std::uint32_t>(GraphArchiveNodeKind::FusedOp));
					    WriteScalar(out, static_cast<std::uint32_t>(node.pattern));
					    WriteSize(out, node.body);
					    WriteNodeOutputList(out, node.args);
				    }
			    },
			    entry.node);
		}

		inline NodeVariant ReadNodePayload(std::istream& in)
		{
			const auto kind = static_cast<GraphArchiveNodeKind>(ReadScalar<std::uint32_t>(in));
			switch (kind)
			{
			case GraphArchiveNodeKind::ParamRef:
				return ParamRefNode{ ReadSize(in) };
			case GraphArchiveNodeKind::Constant:
				return ConstantNode{ ReadTensor(in).CopyToDevice(PolymorphicDevice{ CPU{} }) };
			case GraphArchiveNodeKind::QuantizedConstant: {
				auto storage = ReadTensor(in).CopyToDevice(PolymorphicDevice{ CPU{} });
				auto params = ReadQuantizationParams(in);
				return QuantizedConstantNode{ std::move(storage), std::move(params) };
			}
			case GraphArchiveNodeKind::VariableRef:
				return VariableRefNode{ ReadSize(in) };
			case GraphArchiveNodeKind::UnaryOp: {
				const auto op = static_cast<UnaryOp>(ReadScalar<std::uint32_t>(in));
				return UnaryOpNode{ op, ReadNodeOutput(in) };
			}
			case GraphArchiveNodeKind::BinaryOp: {
				const auto op = static_cast<BinaryOp>(ReadScalar<std::uint32_t>(in));
				const auto lhs = ReadNodeOutput(in);
				const auto rhs = ReadNodeOutput(in);
				return BinaryOpNode{ op, lhs, rhs };
			}
			case GraphArchiveNodeKind::Call: {
				const auto callee = ReadSize(in);
				return CallNode{ callee, ReadNodeOutputList(in) };
			}
			case GraphArchiveNodeKind::Cast: {
				const auto input = ReadNodeOutput(in);
				return CastNode{ input, ReadDataType(in) };
			}
			case GraphArchiveNodeKind::Quantize: {
				const auto input = ReadNodeOutput(in);
				auto params = ReadQuantizationParams(in);
				return QuantizeNode{ input, std::move(params) };
			}
			case GraphArchiveNodeKind::Dequantize: {
				const auto input = ReadNodeOutput(in);
				auto params = ReadQuantizationParams(in);
				return DequantizeNode{ input, std::move(params), ReadDataType(in) };
			}
			case GraphArchiveNodeKind::Cond: {
				const auto condition = ReadNodeOutput(in);
				const auto thenBranch = ReadSize(in);
				const auto elseBranch = ReadSize(in);
				return CondNode{ condition, thenBranch, elseBranch, ReadNodeOutputList(in) };
			}
			case GraphArchiveNodeKind::While: {
				const auto condBranch = ReadSize(in);
				const auto bodyBranch = ReadSize(in);
				return WhileNode{ condBranch, bodyBranch, ReadNodeOutputList(in) };
			}
			case GraphArchiveNodeKind::SaveActivation: {
				const auto input = ReadNodeOutput(in);
				return SaveActivationNode{ input, ReadSize(in) };
			}
			case GraphArchiveNodeKind::LoadActivation:
				return LoadActivationNode{ ReadSize(in) };
			case GraphArchiveNodeKind::TapeSaveActivation: {
				const auto input = ReadNodeOutput(in);
				return TapeSaveActivationNode{ input, ReadSize(in) };
			}
			case GraphArchiveNodeKind::TapeLoadActivation:
				return TapeLoadActivationNode{ ReadSize(in) };
			case GraphArchiveNodeKind::ReduceOp: {
				const auto op = static_cast<ReduceOp>(ReadScalar<std::uint32_t>(in));
				const auto input = ReadNodeOutput(in);
				return ReduceOpNode{ op, input, ReadSize(in) };
			}
			case GraphArchiveNodeKind::Reshape: {
				const auto input = ReadNodeOutput(in);
				return ReshapeNode{ input, ReadShape(in) };
			}
			case GraphArchiveNodeKind::Permute: {
				const auto input = ReadNodeOutput(in);
				return PermuteNode{ input, ReadShape(in) };
			}
			case GraphArchiveNodeKind::BroadcastTo: {
				const auto input = ReadNodeOutput(in);
				return BroadcastToNode{ input, ReadShape(in) };
			}
			case GraphArchiveNodeKind::Pad: {
				const auto input = ReadNodeOutput(in);
				auto lowPads = ReadShape(in);
				auto highPads = ReadShape(in);
				const auto mode = static_cast<PadMode>(ReadScalar<std::uint32_t>(in));
				const auto constantValue = ReadScalar<double>(in);
				return PadNode{ input, std::move(lowPads), std::move(highPads), mode, constantValue };
			}
			case GraphArchiveNodeKind::Gather: {
				const auto data = ReadNodeOutput(in);
				const auto indices = ReadNodeOutput(in);
				return GatherNode{ data, indices, ReadSize(in) };
			}
			case GraphArchiveNodeKind::Scatter: {
				const auto data = ReadNodeOutput(in);
				const auto indices = ReadNodeOutput(in);
				const auto updates = ReadNodeOutput(in);
				const auto axis = ReadSize(in);
				const auto mode = static_cast<ScatterMode>(ReadScalar<std::uint32_t>(in));
				return ScatterNode{ data, indices, updates, axis, mode };
			}
			case GraphArchiveNodeKind::Scan: {
				const auto input = ReadNodeOutput(in);
				const auto axis = ReadSize(in);
				const auto op = static_cast<ScanOp>(ReadScalar<std::uint32_t>(in));
				return ScanNode{ input, axis, op };
			}
			case GraphArchiveNodeKind::SSMScan: {
				const auto state = ReadNodeOutput(in);
				const auto dt = ReadNodeOutput(in);
				const auto a = ReadNodeOutput(in);
				const auto b = ReadNodeOutput(in);
				const auto c = ReadNodeOutput(in);
				return SSMScanNode{ state, dt, a, b, c, ReadOptionalNodeOutput(in) };
			}
			case GraphArchiveNodeKind::RWKVWKV: {
				const auto key = ReadNodeOutput(in);
				const auto value = ReadNodeOutput(in);
				const auto receptance = ReadNodeOutput(in);
				const auto timeDecay = ReadNodeOutput(in);
				const auto timeFirst = ReadNodeOutput(in);
				return RWKVWKVNode{ key, value, receptance, timeDecay, timeFirst };
			}
			case GraphArchiveNodeKind::Softmax: {
				const auto input = ReadNodeOutput(in);
				return SoftmaxNode{ input, ReadSize(in) };
			}
			case GraphArchiveNodeKind::CrossEntropyLoss: {
				const auto logits = ReadNodeOutput(in);
				const auto labels = ReadNodeOutput(in);
				return CrossEntropyLossNode{ logits, labels };
			}
			case GraphArchiveNodeKind::CrossEntropyLossBackward: {
				const auto grad = ReadNodeOutput(in);
				const auto logits = ReadNodeOutput(in);
				const auto labels = ReadNodeOutput(in);
				return CrossEntropyLossBackwardNode{ grad, logits, labels };
			}
			case GraphArchiveNodeKind::Normalization: {
				const auto input = ReadNodeOutput(in);
				auto scale = ReadOptionalNodeOutput(in);
				auto bias = ReadOptionalNodeOutput(in);
				const auto mode = static_cast<NormalizationMode>(ReadScalar<std::uint32_t>(in));
				const auto axis = ReadSize(in);
				const auto groupCount = ReadSize(in);
				const auto epsilon = ReadScalar<double>(in);
				return NormalizationNode{ input, std::move(scale), std::move(bias), mode, axis, groupCount, epsilon };
			}
			case GraphArchiveNodeKind::BatchMatMul: {
				const auto lhs = ReadNodeOutput(in);
				const auto rhs = ReadNodeOutput(in);
				return BatchMatMulNode{ lhs, rhs };
			}
			case GraphArchiveNodeKind::OutProd: {
				const auto lhs = ReadNodeOutput(in);
				const auto rhs = ReadNodeOutput(in);
				return OutProdNode{ lhs, rhs };
			}
			case GraphArchiveNodeKind::TimestepEmbedding: {
				const auto timesteps = ReadNodeOutput(in);
				const auto dim = ReadSize(in);
				const auto maxPeriod = ReadSize(in);
				return TimestepEmbeddingNode{ timesteps, dim, maxPeriod };
			}
			case GraphArchiveNodeKind::SolveTri: {
				const auto a = ReadNodeOutput(in);
				const auto b = ReadNodeOutput(in);
				const auto lower = ReadScalar<bool>(in);
				const auto unitDiagonal = ReadScalar<bool>(in);
				return SolveTriNode{ a, b, lower, unitDiagonal };
			}
			case GraphArchiveNodeKind::SGDStep: {
				const auto parameter = ReadNodeOutput(in);
				const auto gradient = ReadNodeOutput(in);
				auto velocity = ReadOptionalNodeOutput(in);
				const auto learningRate = ReadScalar<double>(in);
				const auto momentum = ReadScalar<double>(in);
				const auto weightDecay = ReadScalar<double>(in);
				const auto nesterov = ReadScalar<bool>(in);
				return SGDStepNode{ parameter, gradient, std::move(velocity), learningRate,
				                    momentum, weightDecay, nesterov };
			}
			case GraphArchiveNodeKind::AdamWStep: {
				const auto parameter = ReadNodeOutput(in);
				const auto gradient = ReadNodeOutput(in);
				const auto firstMoment = ReadNodeOutput(in);
				const auto secondMoment = ReadNodeOutput(in);
				const auto learningRate = ReadScalar<double>(in);
				const auto beta1 = ReadScalar<double>(in);
				const auto beta2 = ReadScalar<double>(in);
				const auto epsilon = ReadScalar<double>(in);
				const auto weightDecay = ReadScalar<double>(in);
				const auto step = ReadSize(in);
				return AdamWStepNode{ parameter, gradient, firstMoment, secondMoment, learningRate,
				                      beta1, beta2, epsilon, weightDecay, step };
			}
			case GraphArchiveNodeKind::Im2Col: {
				const auto input = ReadNodeOutput(in);
				auto kernelShape = ReadShape(in);
				auto strides = ReadShape(in);
				auto dilations = ReadShape(in);
				auto lowPads = ReadShape(in);
				auto highPads = ReadShape(in);
				return Im2ColNode{ input, std::move(kernelShape), std::move(strides), std::move(dilations),
				                   std::move(lowPads), std::move(highPads) };
			}
			case GraphArchiveNodeKind::Conv2D: {
				const auto input = ReadNodeOutput(in);
				const auto weight = ReadNodeOutput(in);
				auto bias = ReadOptionalNodeOutput(in);
				auto strides = ReadShape(in);
				auto dilations = ReadShape(in);
				auto lowPads = ReadShape(in);
				auto highPads = ReadShape(in);
				const auto groupCount = ReadSize(in);
				return Conv2DNode{ input, weight, std::move(bias), std::move(strides), std::move(dilations),
				                   std::move(lowPads), std::move(highPads), groupCount };
			}
			case GraphArchiveNodeKind::ConvTranspose2D: {
				const auto input = ReadNodeOutput(in);
				const auto weight = ReadNodeOutput(in);
				auto bias = ReadOptionalNodeOutput(in);
				auto strides = ReadShape(in);
				auto dilations = ReadShape(in);
				auto lowPads = ReadShape(in);
				auto highPads = ReadShape(in);
				auto outputPads = ReadShape(in);
				const auto groupCount = ReadSize(in);
				return ConvTranspose2DNode{ input, weight, std::move(bias), std::move(strides),
				                            std::move(dilations), std::move(lowPads), std::move(highPads),
				                            std::move(outputPads), groupCount };
			}
			case GraphArchiveNodeKind::Pool2D: {
				const auto input = ReadNodeOutput(in);
				const auto mode = static_cast<PoolMode>(ReadScalar<std::uint32_t>(in));
				auto kernelShape = ReadShape(in);
				auto strides = ReadShape(in);
				auto lowPads = ReadShape(in);
				auto highPads = ReadShape(in);
				const auto countIncludePad = ReadScalar<bool>(in);
				return Pool2DNode{ input, mode, std::move(kernelShape), std::move(strides), std::move(lowPads),
				                   std::move(highPads), countIncludePad };
			}
			case GraphArchiveNodeKind::Upsample: {
				const auto input = ReadNodeOutput(in);
				const auto mode = static_cast<UpsampleMode>(ReadScalar<std::uint32_t>(in));
				auto outputSpatialShape = ReadShape(in);
				const auto alignCorners = ReadScalar<bool>(in);
				return UpsampleNode{ input, mode, std::move(outputSpatialShape), alignCorners };
			}
			case GraphArchiveNodeKind::Concat: {
				auto inputs = ReadNodeOutputList(in);
				return ConcatNode{ std::move(inputs), ReadSize(in) };
			}
			case GraphArchiveNodeKind::Slice: {
				const auto input = ReadNodeOutput(in);
				const auto axis = ReadSize(in);
				const auto start = ReadSize(in);
				return SliceNode{ input, axis, start, ReadSize(in) };
			}
			case GraphArchiveNodeKind::GetRows: {
				const auto data = ReadNodeOutput(in);
				const auto indices = ReadNodeOutput(in);
				return GetRowsNode{ data, indices };
			}
			case GraphArchiveNodeKind::Argsort: {
				const auto input = ReadNodeOutput(in);
				const auto axis = ReadSize(in);
				const auto order = static_cast<SortOrder>(ReadScalar<std::uint32_t>(in));
				return ArgsortNode{ input, axis, order };
			}
			case GraphArchiveNodeKind::MulMatId: {
				const auto as = ReadNodeOutput(in);
				const auto b = ReadNodeOutput(in);
				const auto ids = ReadNodeOutput(in);
				return MulMatIdNode{ as, b, ids };
			}
			case GraphArchiveNodeKind::FusedOp: {
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

		inline Subgraph ReadSubgraph(std::istream& in)
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
				auto node = ReadNodePayload(in);
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

	inline void SaveGraphArchiveImpl(const Graph& graph, const std::filesystem::path& path,
	                          const std::optional<std::filesystem::path>& externalWeightsPath,
	                          ExternalWeightSaveOptions externalOptions)
	{
		Validation::ValidateGraph(graph);
		ValidateExecutablePlan(BuildExecutablePlan(graph));
		std::optional<std::ofstream> externalOut;
		std::string externalPathText;
		if (externalWeightsPath)
		{
			if (std::filesystem::absolute(*externalWeightsPath).lexically_normal() ==
			    std::filesystem::absolute(path).lexically_normal())
			{
				throw std::runtime_error("LiteNN external weight file must be different from the model file");
			}
			externalOut.emplace(*externalWeightsPath, std::ios::binary);
			if (!*externalOut)
			{
				throw std::runtime_error("Failed to open LiteNN external weight file for writing");
			}
			externalPathText = Detail::ExternalPathText(path, *externalWeightsPath);
		}

		std::ofstream out(path, std::ios::binary);
		if (!out)
		{
			throw std::runtime_error("Failed to open LiteNN graph archive file for writing");
		}

		out.write(Detail::kGraphArchiveMagic.data(), static_cast<std::streamsize>(Detail::kGraphArchiveMagic.size()));
		Detail::EnsureWrite(out);
		Detail::WriteScalar(out, Detail::kGraphArchiveVersion);
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
			const auto byteCount = Detail::TensorByteSize(variable->Data());
			if (externalOut && byteCount >= externalOptions.minVariableBytes)
			{
				const auto rawPosition = externalOut->tellp();
				if (rawPosition == std::streampos(-1))
				{
					throw std::runtime_error("Failed to determine LiteNN external weight output offset");
				}
				const auto rawOffset = static_cast<std::uint64_t>(rawPosition);
				const auto alignedOffset = Detail::AlignUp(rawOffset, externalOptions.alignment);
				Detail::WriteZeroBytes(*externalOut, alignedOffset - rawOffset);
				Detail::WriteTensorPayload(*externalOut, variable->Data());

				Detail::WriteScalar(out, static_cast<std::uint8_t>(Detail::VariablePayloadKind::External));
				Detail::WriteTensorMetadata(out, variable->Data());
				Detail::WriteString(out, externalPathText);
				Detail::WriteScalar(out, alignedOffset);
				Detail::WriteScalar(out, byteCount);
			}
			else
			{
				Detail::WriteScalar(out, static_cast<std::uint8_t>(Detail::VariablePayloadKind::Inline));
				Detail::WriteTensor(out, variable->Data());
			}
			Detail::WriteScalar(out, static_cast<std::uint8_t>(variable->HasGradStorage() ? 1 : 0));
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

	namespace Migration
	{
	inline void SaveGraphArchive(const Graph& graph, const std::filesystem::path& path)
	{
		SaveGraphArchiveImpl(graph, path, std::nullopt, {});
	}

	/// Save graph structure and variable metadata to `path`, with selected variable payloads in `externalWeightsPath`.
	inline void SaveGraphArchiveExternalWeights(const Graph& graph, const std::filesystem::path& path,
	                                            const std::filesystem::path& externalWeightsPath,
	                                            const ExternalWeightSaveOptions& externalOptions = {})
	{
		SaveGraphArchiveImpl(graph, path, externalWeightsPath, externalOptions);
	}

	inline Graph LoadGraphArchive(const std::filesystem::path& path)
	{
		std::ifstream in(path, std::ios::binary);
		if (!in)
		{
			throw std::runtime_error("Failed to open LiteNN graph archive file for reading");
		}

		std::array<char, Detail::kGraphArchiveMagic.size()> magic{};
		in.read(magic.data(), static_cast<std::streamsize>(magic.size()));
		Detail::EnsureRead(in);
		if (magic != Detail::kGraphArchiveMagic)
		{
			throw std::runtime_error("Invalid LiteNN graph archive magic header");
		}

		const auto version = Detail::ReadScalar<std::uint32_t>(in);
		if (version != Detail::kGraphArchiveVersion)
		{
			throw std::runtime_error(std::format(
			    "Unsupported LiteNN graph archive version {}; this vNext branch only loads version {}",
			    version, Detail::kGraphArchiveVersion));
		}

		const auto forward = Detail::ReadSize(in);
		const auto hasBackward = Detail::ReadScalar<std::uint8_t>(in) != 0;
		std::optional<SubgraphId> backward;
		if (hasBackward)
		{
			backward = Detail::ReadSize(in);
		}
		auto inputNames = Detail::ReadStringList(in);
		auto outputNames = Detail::ReadStringList(in);

		Graph graph;
		const auto variableCount = Detail::ReadSize(in);
		auto variableNames = Detail::ReadStringList(in);
		auto metadata = Detail::ReadMetadataEntries(in);
		std::vector<std::pair<std::filesystem::path, std::shared_ptr<std::vector<std::byte>>>> externalWeightCache;
		const auto loadExternalWeightFile = [&](const std::string& externalPathText) {
			auto resolvedPath = Detail::ResolveExternalPath(path, externalPathText);
			for (const auto& [cachedPath, storage] : externalWeightCache)
			{
				if (cachedPath == resolvedPath)
				{
					return storage;
				}
			}
			auto storage = std::make_shared<std::vector<std::byte>>(Detail::ReadExternalWeightBytes(resolvedPath));
			externalWeightCache.emplace_back(std::move(resolvedPath), storage);
			graph.AddExternalStorage(storage);
			return storage;
		};
		for (std::size_t i = 0; i < variableCount; ++i)
		{
			std::optional<Tensor<PolymorphicDevice>> tensor;
			const auto payloadKind =
			    static_cast<Detail::VariablePayloadKind>(Detail::ReadScalar<std::uint8_t>(in));
			if (payloadKind == Detail::VariablePayloadKind::Inline)
			{
				tensor.emplace(Detail::ReadTensor(in).CopyToDevice(PolymorphicDevice{ CPU{} }));
			}
			else if (payloadKind == Detail::VariablePayloadKind::External)
			{
				auto spec = Detail::ReadTensorMetadata(in);
				const auto externalPathText = Detail::ReadString(in);
				const auto offset = Detail::ReadScalar<std::uint64_t>(in);
				const auto byteCount = Detail::ReadScalar<std::uint64_t>(in);
				const auto expectedByteCount = Detail::TensorSpecByteSize(spec);
				if (byteCount != expectedByteCount)
				{
					throw std::runtime_error("LiteNN external variable byte size does not match tensor metadata");
				}
				auto storage = loadExternalWeightFile(externalPathText);
				if (offset > storage->size() || byteCount > storage->size() - offset)
				{
					throw std::runtime_error("LiteNN external variable payload is outside the weight file");
				}
				auto* data = static_cast<void*>(storage->data() + static_cast<std::size_t>(offset));
				tensor.emplace(data, ShapeView{ spec.shape }, spec.dtype, PolymorphicDevice{ CPU{} });
			}
			else
			{
				throw std::runtime_error("LiteNN model contains an unknown variable payload kind");
			}
			const auto hasGradStorage = Detail::ReadScalar<std::uint8_t>(in) != 0;
			auto quantization = Detail::ReadOptionalQuantizationParams(in);
			auto variable = Variable::Create(std::move(*tensor), hasGradStorage ? VariableGradStorage::Allocate
			                                                                    : VariableGradStorage::None);
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
			graph.AddSubgraph(Detail::ReadSubgraph(in));
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
	} // namespace Migration
} // namespace LiteNN::Serialization

#endif
