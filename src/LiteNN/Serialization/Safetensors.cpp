#include <LiteNN/Serialization/Safetensors.h>

#include <LiteNN/Validation/GraphValidator.h>

#include <simdjson.h>

#include <algorithm>
#include <array>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <fstream>
#include <limits>
#include <map>
#include <optional>
#include <set>
#include <stdexcept>
#include <string>
#include <utility>

namespace LiteNN::Serialization
{
	std::size_t SafetensorsTensorInfo::ByteSize() const
	{
		return dataEnd - dataBegin;
	}

	std::optional<DataType> TryMapSafetensorsDataType(std::string_view dtype)
	{
		if (dtype == "F64")
		{
			return DataType::Float64;
		}
		if (dtype == "F32")
		{
			return DataType::Float32;
		}
		if (dtype == "F16")
		{
			return DataType::Float16;
		}
		if (dtype == "BF16")
		{
			return DataType::BFloat16;
		}
		if (dtype == "F8_E4M3")
		{
			return DataType::Float8E4M3;
		}
		if (dtype == "F8_E5M2")
		{
			return DataType::Float8E5M2;
		}
		if (dtype == "I64")
		{
			return DataType::Int64;
		}
		if (dtype == "I32")
		{
			return DataType::Int32;
		}
		if (dtype == "I8")
		{
			return DataType::Int8;
		}
		if (dtype == "U8")
		{
			return DataType::UInt8;
		}
		if (dtype == "BOOL")
		{
			return DataType::Bool;
		}
		return std::nullopt;
	}

	DataType MapSafetensorsDataType(std::string_view dtype)
	{
		if (auto mapped = TryMapSafetensorsDataType(dtype))
		{
			return *mapped;
		}
		throw std::runtime_error(std::string("Unsupported safetensors dtype: ") + std::string(dtype));
	}

	namespace Detail
	{
		constexpr std::size_t kMaxSafetensorsHeaderBytes = 128uz * 1024uz * 1024uz;

		std::runtime_error JsonError(std::string_view label, simdjson::error_code error)
		{
			return std::runtime_error(std::string("Safetensors header JSON ") + std::string(label) + ": " +
			                          simdjson::error_message(error));
		}

		std::uint64_t ReadU64LE(std::span<const std::byte> data, std::size_t offset)
		{
			if (offset + sizeof(std::uint64_t) > data.size())
			{
				throw std::runtime_error("Safetensors file is truncated while reading u64");
			}
			std::uint64_t value = 0;
			for (std::size_t i = 0; i < sizeof(std::uint64_t); ++i)
			{
				value |= static_cast<std::uint64_t>(std::to_integer<unsigned char>(data[offset + i])) << (8 * i);
			}
			return value;
		}

		std::size_t CheckedToSize(std::uint64_t value, std::string_view label)
		{
			if (value > static_cast<std::uint64_t>(std::numeric_limits<std::size_t>::max()))
			{
				throw std::runtime_error(std::string("Safetensors ") + std::string(label) + " is too large");
			}
			return static_cast<std::size_t>(value);
		}

		std::size_t CheckedMul(std::size_t lhs, std::size_t rhs, std::string_view label)
		{
			if (lhs != 0 && rhs > std::numeric_limits<std::size_t>::max() / lhs)
			{
				throw std::runtime_error(std::string("Safetensors ") + std::string(label) + " overflows size_t");
			}
			return lhs * rhs;
		}

		std::size_t TensorByteSize(ShapeView shape, DataType dtype)
		{
			std::size_t elements = 1;
			for (const auto dim : shape.Dims)
			{
				if (dim == 0)
				{
					throw std::runtime_error(
					    "Safetensors tensor shape contains a zero dimension unsupported by LiteNN");
				}
				elements = CheckedMul(elements, dim, "tensor element count");
			}
			return CheckedMul(elements, ElementByteSize(dtype), "tensor byte size");
		}

		simdjson::dom::object RequireObject(simdjson::dom::element value, std::string_view label)
		{
			simdjson::dom::object object;
			if (const auto error = value.get_object().get(object))
			{
				throw JsonError(std::string(label) + " must be an object", error);
			}
			return object;
		}

		simdjson::dom::array RequireArray(simdjson::dom::element value, std::string_view label)
		{
			simdjson::dom::array array;
			if (const auto error = value.get_array().get(array))
			{
				throw JsonError(std::string(label) + " must be an array", error);
			}
			return array;
		}

		std::string_view RequireString(simdjson::dom::element value, std::string_view label)
		{
			std::string_view string;
			if (const auto error = value.get_string().get(string))
			{
				throw JsonError(std::string(label) + " must be a string", error);
			}
			return string;
		}

		std::uint64_t RequireUInt(simdjson::dom::element value, std::string_view label)
		{
			std::uint64_t integer{};
			if (const auto error = value.get_uint64().get(integer))
			{
				throw JsonError(std::string(label) + " must be an unsigned integer", error);
			}
			return integer;
		}

		std::optional<simdjson::dom::element> FindMember(simdjson::dom::object object, std::string_view key)
		{
			for (auto field : object)
			{
				if (field.key == key)
				{
					return field.value;
				}
			}
			return std::nullopt;
		}

		simdjson::dom::element RequireMember(simdjson::dom::object object, std::string_view key, std::string_view label)
		{
			if (auto member = FindMember(object, key))
			{
				return *member;
			}
			throw std::runtime_error(std::string("Safetensors tensor ") + std::string(label) +
			                         " is missing required field '" + std::string(key) + "'");
		}

		std::vector<std::size_t> ParseShape(simdjson::dom::element value, std::string_view tensorName)
		{
			const auto array = RequireArray(value, std::string("shape for ") + std::string(tensorName));
			std::vector<std::size_t> shape;
			for (auto dimValue : array)
			{
				const auto dim = CheckedToSize(RequireUInt(dimValue, "shape dimension"), "shape dimension");
				if (dim == 0)
				{
					throw std::runtime_error("Safetensors tensor shape contains zero dimension for " +
					                         std::string(tensorName));
				}
				shape.push_back(dim);
			}
			return shape;
		}

		std::pair<std::size_t, std::size_t> ParseOffsets(simdjson::dom::element value, std::string_view tensorName)
		{
			const auto array = RequireArray(value, std::string("data_offsets for ") + std::string(tensorName));
			std::vector<std::size_t> offsets;
			for (auto offsetValue : array)
			{
				offsets.push_back(CheckedToSize(RequireUInt(offsetValue, "data_offsets value"), "data_offsets value"));
			}
			if (offsets.size() != 2)
			{
				throw std::runtime_error("Safetensors data_offsets must contain exactly two integers for " +
				                         std::string(tensorName));
			}
			const auto begin = offsets[0];
			const auto end = offsets[1];
			if (begin > end)
			{
				throw std::runtime_error("Safetensors data_offsets begin is greater than end for " +
				                         std::string(tensorName));
			}
			return { begin, end };
		}

		std::vector<std::byte> ReadAllBytes(const std::filesystem::path& path)
		{
			std::ifstream in(path, std::ios::binary | std::ios::ate);
			if (!in)
			{
				throw std::runtime_error("Failed to open safetensors file for reading");
			}
			const auto size = in.tellg();
			if (size < 0)
			{
				throw std::runtime_error("Failed to determine safetensors file size");
			}
			std::vector<std::byte> bytes(static_cast<std::size_t>(size));
			in.seekg(0, std::ios::beg);
			if (!bytes.empty())
			{
				in.read(reinterpret_cast<char*>(bytes.data()), static_cast<std::streamsize>(bytes.size()));
				if (!in)
				{
					throw std::runtime_error("Failed to read safetensors file");
				}
			}
			return bytes;
		}

		std::streamoff FileSize(std::ifstream& in)
		{
			const auto current = in.tellg();
			in.seekg(0, std::ios::end);
			const auto size = in.tellg();
			in.seekg(current, std::ios::beg);
			if (size < 0)
			{
				throw std::runtime_error("Failed to determine safetensors file size");
			}
			return size;
		}

		std::vector<std::byte> ReadFileRange(const std::filesystem::path& path, std::uint64_t offset,
		                                     std::size_t byteCount)
		{
			std::ifstream in(path, std::ios::binary);
			if (!in)
			{
				throw std::runtime_error("Failed to open safetensors file for reading");
			}
			in.seekg(static_cast<std::streamoff>(offset), std::ios::beg);
			if (!in)
			{
				throw std::runtime_error("Failed to seek safetensors file");
			}
			std::vector<std::byte> bytes(byteCount);
			if (!bytes.empty())
			{
				in.read(reinterpret_cast<char*>(bytes.data()), static_cast<std::streamsize>(bytes.size()));
				if (!in)
				{
					throw std::runtime_error("Failed to read safetensors tensor payload");
				}
			}
			return bytes;
		}

		struct SafetensorsArchiveBuilder
		{
			static SafetensorsArchive Build(std::span<const std::byte> bytes)
			{
				if (bytes.size() < sizeof(std::uint64_t))
				{
					throw std::runtime_error("Safetensors file is too small to contain a header length");
				}

				const auto headerSizeU64 = ReadU64LE(bytes, 0);
				if (headerSizeU64 > kMaxSafetensorsHeaderBytes)
				{
					throw std::runtime_error("Safetensors header is too large");
				}
				const auto headerSize = CheckedToSize(headerSizeU64, "header size");
				const auto payloadOffset = sizeof(std::uint64_t) + headerSize;
				if (payloadOffset > bytes.size())
				{
					throw std::runtime_error("Safetensors file is truncated before tensor payload data");
				}

				const auto* headerBegin = reinterpret_cast<const char*>(bytes.data() + sizeof(std::uint64_t));
				simdjson::padded_string header(headerBegin, headerSize);
				simdjson::dom::parser parser;
				simdjson::dom::element root;
				if (const auto error = parser.parse(header).get(root))
				{
					throw JsonError("parse failed", error);
				}
				const auto rootObject = RequireObject(root, "header");

				SafetensorsArchive archive;
				archive.storage_.assign(bytes.begin(), bytes.end());
				archive.payloadOffset_ = payloadOffset;
				ParseHeader(archive, rootObject, bytes.size() - payloadOffset);
				return archive;
			}

			static SafetensorsArchive BuildFromFile(const std::filesystem::path& path)
			{
				std::ifstream in(path, std::ios::binary);
				if (!in)
				{
					throw std::runtime_error("Failed to open safetensors file for reading");
				}
				const auto fileSize = FileSize(in);
				if (fileSize < static_cast<std::streamoff>(sizeof(std::uint64_t)))
				{
					throw std::runtime_error("Safetensors file is too small to contain a header length");
				}

				std::array<std::byte, sizeof(std::uint64_t)> headerLengthBytes{};
				in.read(reinterpret_cast<char*>(headerLengthBytes.data()),
				        static_cast<std::streamsize>(headerLengthBytes.size()));
				if (!in)
				{
					throw std::runtime_error("Failed to read safetensors header length");
				}
				const auto headerSizeU64 = ReadU64LE(headerLengthBytes, 0);
				if (headerSizeU64 > kMaxSafetensorsHeaderBytes)
				{
					throw std::runtime_error("Safetensors header is too large");
				}
				const auto headerSize = CheckedToSize(headerSizeU64, "header size");
				const auto payloadOffset = sizeof(std::uint64_t) + headerSize;
				if (payloadOffset > static_cast<std::size_t>(fileSize))
				{
					throw std::runtime_error("Safetensors file is truncated before tensor payload data");
				}

				std::vector<char> header(headerSize);
				if (!header.empty())
				{
					in.read(header.data(), static_cast<std::streamsize>(header.size()));
					if (!in)
					{
						throw std::runtime_error("Failed to read safetensors header");
					}
				}

				simdjson::padded_string paddedHeader(header.data(), header.size());
				simdjson::dom::parser parser;
				simdjson::dom::element root;
				if (const auto error = parser.parse(paddedHeader).get(root))
				{
					throw JsonError("parse failed", error);
				}

				SafetensorsArchive archive;
				archive.backingPath_ = path;
				archive.payloadOffset_ = payloadOffset;
				ParseHeader(archive, RequireObject(root, "header"), static_cast<std::size_t>(fileSize) - payloadOffset);
				return archive;
			}

			static void ParseHeader(SafetensorsArchive& archive, simdjson::dom::object rootObject,
			                        std::size_t payloadSize)
			{
				std::vector<std::pair<std::size_t, std::size_t>> intervals;
				std::set<std::string> seenFields;
				for (auto field : rootObject)
				{
					const auto name = std::string(field.key);
					if (!seenFields.insert(name).second)
					{
						throw std::runtime_error("Safetensors header JSON contains duplicate top-level key: " + name);
					}
					if (name == "__metadata__")
					{
						ParseMetadata(archive, field.value);
						continue;
					}
					ParseTensor(archive, name, field.value, payloadSize, intervals);
				}

				std::sort(intervals.begin(), intervals.end());
				for (std::size_t i = 1; i < intervals.size(); ++i)
				{
					if (intervals[i - 1].second > intervals[i].first)
					{
						throw std::runtime_error("Safetensors tensor payload ranges overlap");
					}
				}
			}

			static void ParseMetadata(SafetensorsArchive& archive, simdjson::dom::element value)
			{
				const auto object = RequireObject(value, "__metadata__");
				std::set<std::string> seenMetadataKeys;
				for (auto field : object)
				{
					const auto key = std::string(field.key);
					if (!seenMetadataKeys.insert(key).second)
					{
						throw std::runtime_error("Safetensors metadata contains duplicate key: " + key);
					}
					archive.metadata_.push_back({ key, std::string(RequireString(field.value, "__metadata__ value")) });
				}
			}

			static void ParseTensor(SafetensorsArchive& archive, std::string_view name, simdjson::dom::element value,
			                        std::size_t payloadSize,
			                        std::vector<std::pair<std::size_t, std::size_t>>& intervals)
			{
				const auto object = RequireObject(value, std::string("metadata for tensor ") + std::string(name));
				const auto storageDType = std::string(RequireString(RequireMember(object, "dtype", name), "dtype"));
				const auto dtype = MapSafetensorsDataType(storageDType);
				auto shape = ParseShape(RequireMember(object, "shape", name), name);
				const auto [begin, end] = ParseOffsets(RequireMember(object, "data_offsets", name), name);
				if (end > payloadSize)
				{
					throw std::runtime_error("Safetensors tensor payload exceeds file size for " + std::string(name));
				}
				const auto expectedBytes = TensorByteSize(ShapeView{ shape }, dtype);
				if (end - begin != expectedBytes)
				{
					throw std::runtime_error("Safetensors tensor byte size does not match dtype and shape for " +
					                         std::string(name));
				}
				if (dtype == DataType::Bool && !archive.storage_.empty())
				{
					const auto absoluteBegin = archive.payloadOffset_ + begin;
					for (std::size_t i = absoluteBegin; i < absoluteBegin + expectedBytes; ++i)
					{
						const auto valueByte = std::to_integer<unsigned char>(archive.storage_[i]);
						if (valueByte > 1)
						{
							throw std::runtime_error("Safetensors BOOL tensor contains a non-boolean byte for " +
							                         std::string(name));
						}
					}
				}

				intervals.emplace_back(begin, end);
				archive.tensors_.push_back({
				    .name = std::string(name),
				    .storageDType = storageDType,
				    .type = TensorType::Dense(dtype, ShapeView{ shape }),
				    .dataBegin = begin,
				    .dataEnd = end,
				});
			}
		};
	} // namespace Detail

	SafetensorsArchive SafetensorsArchive::Load(std::span<const std::byte> bytes)
	{
		return Detail::SafetensorsArchiveBuilder::Build(bytes);
	}

	SafetensorsArchive SafetensorsArchive::LoadFile(const std::filesystem::path& path)
	{
		return Detail::SafetensorsArchiveBuilder::BuildFromFile(path);
	}

	std::span<const SafetensorsTensorInfo> SafetensorsArchive::Tensors() const
	{
		return tensors_;
	}

	std::span<const ModelMetadataEntry> SafetensorsArchive::Metadata() const
	{
		return metadata_;
	}

	const SafetensorsTensorInfo* SafetensorsArchive::FindTensor(std::string_view name) const
	{
		for (const auto& tensor : tensors_)
		{
			if (tensor.name == name)
			{
				return &tensor;
			}
		}
		return nullptr;
	}

	std::span<const std::byte> SafetensorsArchive::TensorData(const SafetensorsTensorInfo& tensor) const
	{
		if (!storage_.empty())
		{
			const auto begin = payloadOffset_ + tensor.dataBegin;
			return std::span<const std::byte>(storage_).subspan(begin, tensor.ByteSize());
		}
		if (backingPath_.empty())
		{
			throw std::runtime_error("Safetensors archive has no in-memory payload or backing file");
		}
		tensorReadBuffer_ = Detail::ReadFileRange(backingPath_, payloadOffset_ + tensor.dataBegin, tensor.ByteSize());
		return tensorReadBuffer_;
	}

	Tensor<CPU> SafetensorsArchive::TensorAsCPU(const SafetensorsTensorInfo& tensor, bool transpose2D) const
	{
		const auto shape = tensor.type.StaticShape();
		Tensor<CPU> result(Uninitialized, ShapeView{ shape }, tensor.type.dtype);
		const auto bytes = TensorData(tensor);
		std::memcpy(result.UnsafeRawData(), bytes.data(), bytes.size());
		if (transpose2D)
		{
			if (shape.size() != 2)
			{
				throw std::runtime_error("Safetensors transpose hook requires a rank-2 tensor: " + tensor.name);
			}
			return result.Transpose();
		}
		return result;
	}

	Graph ImportSafetensorsVariables(const SafetensorsArchive& archive, const SafetensorsImportOptions& options)
	{
		Graph graph;
		graph.SetForward(graph.AddSubgraph(Subgraph{}));
		std::set<std::string> seenNames;
		for (const auto& metadata : archive.Metadata())
		{
			graph.SetMetadataEntry("safetensors.metadata." + metadata.key, metadata.value);
		}

		for (const auto& tensor : archive.Tensors())
		{
			auto targetName = options.renameTensor ? options.renameTensor(tensor.name) : tensor.name;
			if (targetName.empty())
			{
				throw std::runtime_error("Safetensors import produced an empty variable name for " + tensor.name);
			}
			if (options.failOnDuplicateVariableNames && !seenNames.insert(targetName).second)
			{
				throw std::runtime_error("Safetensors import produced duplicate variable name: " + targetName);
			}

			const auto shouldTranspose = options.transpose2D && options.transpose2D(tensor.name);
			auto importedTensor = archive.TensorAsCPU(tensor, shouldTranspose);
			const auto index = graph.AddVariable(Variable::Create(std::move(importedTensor)));
			graph.SetVariableName(index, std::move(targetName));
		}

		Validation::ValidateGraph(graph);
		return graph;
	}

	Graph LoadSafetensorsVariables(const std::filesystem::path& path, const SafetensorsImportOptions& options)
	{
		return ImportSafetensorsVariables(SafetensorsArchive::LoadFile(path), options);
	}

	ImporterOwnedManifest ImportSafetensorsVariablesManifest(const SafetensorsArchive& archive,
	                                                         const SafetensorsImportOptions& options)
	{
		auto graph = ImportSafetensorsVariables(archive, options);
		auto manifest = BuildImporterOwnedManifest("safetensors", std::move(graph));
		manifest.configMetadata.assign(archive.Metadata().begin(), archive.Metadata().end());
		manifest.diagnostics.push_back(MakeImportDiagnostic(
		    ImportDiagnosticKind::MissingMetadata, "safetensors",
		    "safetensors is tensor storage only; production graph construction requires an explicit manifest/config"));

		manifest.weights.reserve(archive.Tensors().size());
		for (const auto& tensor : archive.Tensors())
		{
			const auto targetName = options.renameTensor ? options.renameTensor(tensor.name) : tensor.name;
			const auto variableIndex = manifest.model.UnsafeGraphView().FindVariable(targetName);
			if (!variableIndex)
			{
				throw std::runtime_error("Safetensors manifest could not find imported variable: " + targetName);
			}
			const auto& imported = manifest.model.UnsafeGraphView().GetVariable(*variableIndex)->Data();
			const auto transposed = options.transpose2D && options.transpose2D(tensor.name);
			manifest.weights.push_back({
			    .sourceName = tensor.name,
			    .graphName = targetName,
			    .sourceType = tensor.type,
			    .graphType = TensorType::Dense(imported.DType(), imported.Shape()),
			    .layoutConversion = transposed ? "transpose2d" : "identity",
			    .quantizationMapping = "none",
			    .loraBinding = "none",
			});
			manifest.diagnostics.push_back(MakeImportDiagnostic(
			    ImportDiagnosticKind::MissingMetadata, tensor.name,
			    "tensor imported without architecture role; bind it through a model manifest before production use"));
		}

		ValidateImporterOwnedManifest(manifest);
		return manifest;
	}

	SafetensorsLoRAImportResult ImportLinearLoRAAdapters(Graph& graph, const SafetensorsArchive& archive,
	                                                     const SafetensorsLoRAImportOptions& options)
	{
		struct PendingAdapter
		{
			std::string targetName;
			std::string adapterName;
			const SafetensorsTensorInfo* a = nullptr;
			const SafetensorsTensorInfo* b = nullptr;
		};

		std::map<std::string, PendingAdapter> pending;
		SafetensorsLoRAImportResult result;
		for (const auto& tensor : archive.Tensors())
		{
			const auto parsed = Layer::ParsePEFTLoRATensorName(tensor.name);
			if (!parsed)
			{
				continue;
			}

			auto targetName = options.renameTarget ? options.renameTarget(parsed->targetName) : parsed->targetName;
			if (targetName.empty())
			{
				throw std::runtime_error("Safetensors LoRA import produced an empty target name for " + tensor.name);
			}
			const auto key = targetName + "\n" + parsed->adapterName;
			auto& entry = pending[key];
			entry.targetName = std::move(targetName);
			entry.adapterName = parsed->adapterName;
			auto*& slot = parsed->role == Layer::LoRATensorRole::A ? entry.a : entry.b;
			if (slot != nullptr)
			{
				throw std::runtime_error("Safetensors LoRA import found duplicate adapter tensor for " + tensor.name);
			}
			slot = &tensor;
		}

		for (const auto& [_, entry] : pending)
		{
			if (entry.a == nullptr || entry.b == nullptr)
			{
				result.diagnostics.push_back("LoRA adapter '" + entry.targetName + "'/'" + entry.adapterName +
				                             "' is missing " + (entry.a == nullptr ? "A" : "B") + " tensor");
				continue;
			}

			auto a = archive.TensorAsCPU(*entry.a, options.transposePEFTWeights);
			auto b = archive.TensorAsCPU(*entry.b, options.transposePEFTWeights);
			if (a.Shape().NumDim() != 2 || b.Shape().NumDim() != 2)
			{
				throw std::runtime_error("Safetensors LoRA adapter tensors must be rank-2: " + entry.targetName);
			}
			if (a.DType() != b.DType())
			{
				throw std::runtime_error("Safetensors LoRA adapter A/B tensors must have matching dtype: " +
				                         entry.targetName);
			}
			const auto rank = a.Shape()[1];
			const auto alpha = options.defaultAlpha == 0.0f ? static_cast<float>(rank) : options.defaultAlpha;
			result.adapters.push_back(
			    Layer::CreateLinearLoRA(graph,
			                            Layer::LoRAAdapterMetadata{ .targetName = entry.targetName,
			                                                        .adapterName = entry.adapterName,
			                                                        .rank = rank,
			                                                        .alpha = alpha,
			                                                        .dtype = a.DType(),
			                                                        .mergeMode = Layer::LoRAMergeMode::Unmerged },
			                            std::move(a), std::move(b)));
		}
		return result;
	}
} // namespace LiteNN::Serialization
