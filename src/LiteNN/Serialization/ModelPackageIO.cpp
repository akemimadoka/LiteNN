#include <LiteNN/Serialization/ModelPackageIO.h>
#include <LiteNN/Validation/GraphValidator.h>

#include <algorithm>
#include <cstring>
#include <fstream>
#include <map>
#include <optional>
#include <sstream>
#include <stdexcept>
#include <string>
#include <string_view>
#include <system_error>
#include <type_traits>

#include <simdjson.h>

namespace LiteNN::Serialization
{
	namespace
	{
		constexpr std::string_view kFormat = "litenn.model.vnext";

		template <typename Enum>
		constexpr auto EnumValue(Enum value)
		{
			return static_cast<std::underlying_type_t<Enum>>(value);
		}

		std::uint64_t ChecksumBytes(std::span<const std::byte> bytes)
		{
			std::uint64_t hash = 1469598103934665603ull;
			for (const auto byte : bytes)
			{
				hash ^= std::to_integer<std::uint8_t>(byte);
				hash *= 1099511628211ull;
			}
			return hash;
		}

		std::vector<std::byte> ReadAllBytes(const std::filesystem::path& path)
		{
			std::ifstream in(path, std::ios::binary);
			if (!in)
			{
				throw std::runtime_error("Failed to open LiteNN vNext artifact region: " + path.string());
			}
			in.seekg(0, std::ios::end);
			const auto size = in.tellg();
			if (size < 0)
			{
				throw std::runtime_error("Failed to determine LiteNN vNext artifact region size: " + path.string());
			}
			in.seekg(0, std::ios::beg);
			std::vector<std::byte> bytes(static_cast<std::size_t>(size));
			if (!bytes.empty())
			{
				in.read(reinterpret_cast<char*>(bytes.data()), static_cast<std::streamsize>(bytes.size()));
			}
			if (!in && !in.eof())
			{
				throw std::runtime_error("Failed to read LiteNN vNext artifact region: " + path.string());
			}
			return bytes;
		}

		std::filesystem::path ResolvePackageRelativePath(const std::filesystem::path& baseDirectory,
		                                                 const std::string& relativePath)
		{
			const std::filesystem::path path(relativePath);
			if (path.is_absolute())
			{
				return path;
			}
			return baseDirectory / path;
		}

		std::runtime_error JsonError(std::string_view label, simdjson::error_code error)
		{
			return std::runtime_error(std::string(label) + ": " + simdjson::error_message(error));
		}

		simdjson::dom::object AsObject(simdjson::dom::element value, std::string_view label)
		{
			simdjson::dom::object object;
			if (const auto error = value.get_object().get(object))
			{
				throw JsonError(label, error);
			}
			return object;
		}

		simdjson::dom::array AsArray(simdjson::dom::element value, std::string_view label)
		{
			simdjson::dom::array array;
			if (const auto error = value.get_array().get(array))
			{
				throw JsonError(label, error);
			}
			return array;
		}

		simdjson::dom::element Member(simdjson::dom::object object, std::string_view key, std::string_view label)
		{
			simdjson::dom::element value;
			if (const auto error = object.at_key(key).get(value))
			{
				throw JsonError(label, error);
			}
			return value;
		}

		std::optional<simdjson::dom::element> FindMember(simdjson::dom::object object, std::string_view key)
		{
			simdjson::dom::element value;
			if (const auto error = object.at_key(key).get(value))
			{
				if (error == simdjson::NO_SUCH_FIELD)
				{
					return std::nullopt;
				}
				throw JsonError(key, error);
			}
			return value;
		}

		std::string AsString(simdjson::dom::element value, std::string_view label)
		{
			std::string_view text;
			if (const auto error = value.get_string().get(text))
			{
				throw JsonError(label, error);
			}
			return std::string(text);
		}

		std::uint64_t AsUInt(simdjson::dom::element value, std::string_view label)
		{
			std::uint64_t number{};
			if (const auto error = value.get_uint64().get(number))
			{
				throw JsonError(label, error);
			}
			return number;
		}

		std::int64_t AsInt(simdjson::dom::element value, std::string_view label)
		{
			std::int64_t number{};
			if (const auto error = value.get_int64().get(number))
			{
				throw JsonError(label, error);
			}
			return number;
		}

		double AsDouble(simdjson::dom::element value, std::string_view label)
		{
			double number{};
			if (const auto error = value.get_double().get(number))
			{
				throw JsonError(label, error);
			}
			return number;
		}

		bool AsBool(simdjson::dom::element value, std::string_view label)
		{
			bool flag{};
			if (const auto error = value.get_bool().get(flag))
			{
				throw JsonError(label, error);
			}
			return flag;
		}

		void JsonString(std::ostream& out, std::string_view value)
		{
			out << '"';
			for (const char ch : value)
			{
				switch (ch)
				{
				case '\\':
					out << "\\\\";
					break;
				case '"':
					out << "\\\"";
					break;
				case '\n':
					out << "\\n";
					break;
				case '\r':
					out << "\\r";
					break;
				case '\t':
					out << "\\t";
					break;
				default:
					out << ch;
					break;
				}
			}
			out << '"';
		}

		template <typename T>
		void NumberList(std::ostream& out, const std::vector<T>& values)
		{
			out << '[';
			for (std::size_t i = 0; i < values.size(); ++i)
			{
				if (i != 0)
				{
					out << ',';
				}
				out << values[i];
			}
			out << ']';
		}

		void ShapeJson(std::ostream& out, const TensorShape& shape)
		{
			out << '[';
			for (std::size_t i = 0; i < shape.dims.size(); ++i)
			{
				const auto& dim = shape.dims[i];
				if (i != 0)
				{
					out << ',';
				}
				out << "{\"kind\":" << EnumValue(dim.kind) << ",\"extent\":" << dim.extent << ",\"symbol\":";
				JsonString(out, dim.symbol);
				out << '}';
			}
			out << ']';
		}

		void TensorTypeJson(std::ostream& out, const TensorType& type)
		{
			out << "{\"dtype\":" << EnumValue(type.dtype) << ",\"shape\":";
			ShapeJson(out, type.shape);
			out << ",\"layout\":{\"kind\":" << EnumValue(type.layout.kind) << ",\"strides\":";
			NumberList(out, type.layout.strides);
			out << ",\"tag\":";
			JsonString(out, type.layout.tag);
			out << "},\"memorySpace\":" << EnumValue(type.memorySpace) << '}';
		}

		void NodeOutputJson(std::ostream& out, NodeOutput output)
		{
			out << "{\"node\":" << output.node << ",\"port\":" << output.port << '}';
		}

		void NodeOutputListJson(std::ostream& out, const std::vector<NodeOutput>& outputs)
		{
			out << '[';
			for (std::size_t i = 0; i < outputs.size(); ++i)
			{
				if (i != 0)
				{
					out << ',';
				}
				NodeOutputJson(out, outputs[i]);
			}
			out << ']';
		}

		void StringListJson(std::ostream& out, const std::vector<std::string>& values)
		{
			out << '[';
			for (std::size_t i = 0; i < values.size(); ++i)
			{
				if (i != 0)
				{
					out << ',';
				}
				JsonString(out, values[i]);
			}
			out << ']';
		}

		void PlanOpJson(std::ostream& out, const ExecutablePlanOp& op)
		{
			out << "{\"kind\":";
			JsonString(out, op.kind);
			out << ",\"schemaId\":" << op.schemaId << ",\"category\":" << EnumValue(op.category)
			    << ",\"effect\":" << EnumValue(op.effect) << ",\"attributes\":[";
			for (std::size_t i = 0; i < op.attributes.size(); ++i)
			{
				const auto& attr = op.attributes[i];
				if (i != 0)
				{
					out << ',';
				}
				out << "{\"name\":";
				JsonString(out, attr.name);
				out << ",\"value\":";
				JsonString(out, attr.value);
				out << '}';
			}
			out << "]}";
		}

		void TensorRefJson(std::ostream& out, const VNextExternalTensorRef& tensor)
		{
			out << "{\"name\":";
			JsonString(out, tensor.name);
			out << ",\"type\":";
			TensorTypeJson(out, tensor.type);
			out << ",\"kind\":" << EnumValue(tensor.kind) << ",\"relativePath\":";
			JsonString(out, tensor.relativePath);
			out << ",\"byteOffset\":" << tensor.byteOffset << ",\"byteSize\":" << tensor.byteSize
			    << ",\"alignment\":" << tensor.alignment << ",\"checksum\":" << tensor.checksum
			    << ",\"mutability\":" << EnumValue(tensor.mutability)
			    << ",\"rebindPolicy\":" << EnumValue(tensor.rebindPolicy);
			if (tensor.quantization)
			{
				const auto& q = *tensor.quantization;
				out << ",\"quantization\":{\"scheme\":" << EnumValue(q.scheme)
				    << ",\"granularity\":" << EnumValue(q.granularity)
				    << ",\"blockFormat\":" << EnumValue(q.blockFormat)
				    << ",\"packedFormat\":" << EnumValue(q.packedFormat)
				    << ",\"packedOrder\":" << EnumValue(q.packedOrder)
				    << ",\"blockScaleLayout\":" << EnumValue(q.blockScaleLayout)
				    << ",\"storageLayout\":" << EnumValue(q.storageLayout)
				    << ",\"storageType\":" << EnumValue(q.storageType)
				    << ",\"expressedType\":" << EnumValue(q.expressedType) << ",\"axis\":" << q.axis
				    << ",\"groupSize\":" << q.groupSize << ",\"scales\":";
				NumberList(out, q.scales);
				out << ",\"zeroPoints\":";
				NumberList(out, q.zeroPoints);
				out << ",\"expressedShape\":";
				NumberList(out, q.expressedShape);
				out << '}';
			}
			out << '}';
		}

		void TensorStorageRefJson(std::ostream& out, const TensorStorageRef& storage)
		{
			const auto name = storage.region.name.empty() ? std::string("variable") : storage.region.name;
			TensorRefJson(out, ToVNextExternalTensorRef(name, storage));
		}

		void MemoryJson(std::ostream& out, const MemoryPlan& memory)
		{
			out << "{\"workspaceBytes\":" << memory.workspaceBytes << ",\"persistentBytes\":" << memory.persistentBytes
			    << ",\"externalBytes\":" << memory.externalBytes << ",\"constantBytes\":" << memory.constantBytes
			    << ",\"buffers\":[";
			for (std::size_t i = 0; i < memory.buffers.size(); ++i)
			{
				const auto& b = memory.buffers[i];
				if (i != 0)
				{
					out << ',';
				}
				out << "{\"id\":" << b.id << ",\"kind\":" << EnumValue(b.kind)
				    << ",\"memorySpace\":" << EnumValue(b.memorySpace) << ",\"byteSize\":" << b.byteSize
				    << ",\"alignment\":" << b.alignment << ",\"aliasSet\":" << b.aliasSet << '}';
			}
			out << "]}";
		}

		void RuntimeBufferBindingJson(std::ostream& out, const RuntimeBufferBinding& binding)
		{
			out << "{\"name\":";
			JsonString(out, binding.name);
			out << ",\"type\":";
			TensorTypeJson(out, binding.type);
			out << ",\"ownership\":" << EnumValue(binding.ownership)
			    << ",\"externalKind\":" << EnumValue(binding.externalKind)
			    << ",\"memorySpace\":" << EnumValue(binding.memorySpace) << ",\"memoryBuffer\":" << binding.memoryBuffer
			    << ",\"byteOffset\":" << binding.byteOffset << ",\"byteSize\":" << binding.byteSize
			    << ",\"alignment\":" << binding.alignment << ",\"checksum\":" << binding.checksum
			    << ",\"mutability\":" << EnumValue(binding.mutability)
			    << ",\"rebindPolicy\":" << EnumValue(binding.rebindPolicy) << ",\"strides\":";
			NumberList(out, binding.strides);
			out << ",\"layoutTag\":";
			JsonString(out, binding.layoutTag);
			out << ",\"aliasSet\":" << binding.aliasSet << '}';
		}

		void RuntimeStateBindingJson(std::ostream& out, const Runtime::RuntimeStateBinding& binding)
		{
			out << "{\"name\":";
			JsonString(out, binding.name);
			out << ",\"kind\":" << EnumValue(binding.kind) << ",\"role\":";
			JsonString(out, binding.role);
			out << ",\"type\":";
			TensorTypeJson(out, binding.type);
			out << ",\"mutability\":" << EnumValue(binding.mutability) << ",\"effects\":";
			StringListJson(out, binding.effects);
			out << ",\"memoryBuffer\":";
			if (binding.memoryBuffer)
			{
				out << *binding.memoryBuffer;
			}
			else
			{
				out << "null";
			}
			if (binding.layout)
			{
				const auto& layout = *binding.layout;
				out << ",\"layout\":{\"kind\":" << EnumValue(layout.kind)
				    << ",\"pageSizeTokens\":" << layout.pageSizeTokens
				    << ",\"maxLogicalTokens\":" << layout.maxLogicalTokens
				    << ",\"residentPageCount\":" << layout.residentPageCount
				    << ",\"keyValuePlaneCount\":" << layout.keyValuePlaneCount
				    << ",\"keyPlaneOffsetBytes\":" << layout.keyPlaneOffsetBytes
				    << ",\"valuePlaneOffsetBytes\":" << layout.valuePlaneOffsetBytes
				    << ",\"tokenByteStride\":" << layout.tokenByteStride
				    << ",\"pageByteStride\":" << layout.pageByteStride << ",\"pageTableState\":";
				JsonString(out, layout.pageTableState);
				out << ",\"pageDescriptorState\":";
				JsonString(out, layout.pageDescriptorState);
				out << ",\"activeLengthState\":";
				JsonString(out, layout.activeLengthState);
				out << '}';
			}
			out << '}';
		}

		void RuntimeStateValueBindingJson(std::ostream& out, const Runtime::RuntimeStateValueBinding& binding)
		{
			out << "{\"stateName\":";
			JsonString(out, binding.stateName);
			out << ",\"function\":" << binding.function << ",\"kind\":" << EnumValue(binding.kind)
			    << ",\"valueIndex\":" << binding.valueIndex << ",\"stateByteOffset\":" << binding.stateByteOffset
			    << '}';
		}

		void RuntimeExecutionSegmentJson(std::ostream& out, const Runtime::RuntimeExecutionSegment& segment)
		{
			out << "{\"id\":" << segment.id << ",\"subgraph\":" << segment.subgraph << ",\"backend\":";
			JsonString(out, segment.backend);
			out << ",\"nodes\":";
			NumberList(out, segment.nodes);
			out << ",\"inputBuffers\":";
			NumberList(out, segment.inputBuffers);
			out << ",\"outputBuffers\":";
			NumberList(out, segment.outputBuffers);
			out << '}';
		}

		void AdapterRefJson(std::ostream& out, const VNextAdapterRef& adapter)
		{
			out << "{\"targetName\":";
			JsonString(out, adapter.targetName);
			out << ",\"adapterName\":";
			JsonString(out, adapter.adapterName);
			out << ",\"kind\":";
			JsonString(out, adapter.kind);
			out << ",\"aTensor\":" << adapter.aTensor << ",\"bTensor\":" << adapter.bTensor
			    << ",\"rank\":" << adapter.rank << ",\"alpha\":" << adapter.alpha << ",\"dropout\":" << adapter.dropout
			    << ",\"dtype\":" << EnumValue(adapter.dtype) << ",\"mergeMode\":";
			JsonString(out, adapter.mergeMode);
			out << '}';
		}

		void BackendRequirementJson(std::ostream& out, const VNextBackendRequirementRef& requirement)
		{
			out << "{\"segment\":";
			if (requirement.segment)
			{
				out << *requirement.segment;
			}
			else
			{
				out << "null";
			}
			out << ",\"backend\":";
			JsonString(out, requirement.backend);
			out << ",\"requiredCapabilities\":";
			StringListJson(out, requirement.requiredCapabilities);
			out << ",\"transferABI\":";
			JsonString(out, requirement.transferABI);
			out << ",\"allowsFallback\":" << (requirement.allowsFallback ? "true" : "false") << '}';
		}

		void PlanJson(std::ostream& out, const ExecutablePlan& plan)
		{
			out << "{\"forward\":" << plan.forward << ",\"backward\":";
			if (plan.backward)
			{
				out << *plan.backward;
			}
			else
			{
				out << "null";
			}
			out << ",\"variables\":[";
			for (std::size_t i = 0; i < plan.variables.size(); ++i)
			{
				if (i != 0)
				{
					out << ',';
				}
				TensorStorageRefJson(out, plan.variables[i]);
			}
			out << "],\"variableNames\":";
			StringListJson(out, plan.variableNames);
			out << ",\"activationSlots\":[";
			for (std::size_t i = 0; i < plan.activationSlots.size(); ++i)
			{
				if (i != 0)
				{
					out << ',';
				}
				TensorTypeJson(out, plan.activationSlots[i]);
			}
			out << "],\"tapeSlots\":[";
			for (std::size_t i = 0; i < plan.tapeSlots.size(); ++i)
			{
				if (i != 0)
				{
					out << ',';
				}
				TensorTypeJson(out, plan.tapeSlots[i]);
			}
			const auto writeValues = [&](const std::vector<ExecutablePlanValue>& values) {
				out << '[';
				for (std::size_t i = 0; i < values.size(); ++i)
				{
					if (i != 0)
					{
						out << ',';
					}
					out << "{\"source\":";
					NodeOutputJson(out, values[i].source);
					out << ",\"type\":";
					TensorTypeJson(out, values[i].type);
					out << ",\"name\":";
					JsonString(out, values[i].name);
					out << '}';
				}
				out << ']';
			};
			out << "],\"inputs\":";
			writeValues(plan.inputs);
			out << ",\"outputs\":";
			writeValues(plan.outputs);
			out << ",\"subgraphs\":[";
			for (std::size_t s = 0; s < plan.subgraphs.size(); ++s)
			{
				const auto& subgraph = plan.subgraphs[s];
				if (s != 0)
				{
					out << ',';
				}
				out << "{\"sourceSubgraph\":" << subgraph.sourceSubgraph << ",\"params\":[";
				for (std::size_t i = 0; i < subgraph.params.size(); ++i)
				{
					if (i != 0)
					{
						out << ',';
					}
					TensorTypeJson(out, subgraph.params[i]);
				}
				out << "],\"nodes\":[";
				for (std::size_t n = 0; n < subgraph.nodes.size(); ++n)
				{
					const auto& node = subgraph.nodes[n];
					if (n != 0)
					{
						out << ',';
					}
					out << "{\"sourceNode\":" << node.sourceNode << ",\"op\":";
					PlanOpJson(out, node.op);
					out << ",\"inputs\":";
					NodeOutputListJson(out, node.inputs);
					out << ",\"outputs\":[";
					for (std::size_t o = 0; o < node.outputs.size(); ++o)
					{
						if (o != 0)
						{
							out << ',';
						}
						TensorTypeJson(out, node.outputs[o]);
					}
					out << "]}";
				}
				out << "],\"results\":";
				NodeOutputListJson(out, subgraph.results);
				out << '}';
			}
			out << "]}";
		}

		void ManifestJson(std::ostream& out, const VNextPackageManifest& manifest)
		{
			out << "{\"versions\":{\"manifest\":" << manifest.versions.manifest
			    << ",\"opSet\":" << manifest.versions.opSet << ",\"dtypeSet\":" << manifest.versions.dtypeSet
			    << ",\"layoutSet\":" << manifest.versions.layoutSet
			    << ",\"quantizationSet\":" << manifest.versions.quantizationSet
			    << ",\"artifactABI\":" << manifest.versions.artifactABI << "},\"layout\":{\"mode\":";
			JsonString(out, manifest.layout.mode);
			out << ",\"manifestPath\":";
			JsonString(out, manifest.layout.manifestPath);
			out << ",\"tensorDirectory\":";
			JsonString(out, manifest.layout.tensorDirectory);
			out << ",\"artifactDirectory\":";
			JsonString(out, manifest.layout.artifactDirectory);
			out << "},\"functions\":[";
			for (std::size_t i = 0; i < manifest.functions.size(); ++i)
			{
				const auto& f = manifest.functions[i];
				if (i != 0)
				{
					out << ',';
				}
				out << "{\"id\":" << f.id << ",\"name\":";
				JsonString(out, f.name);
				out << ",\"body\":" << f.body << ",\"inputs\":[";
				for (std::size_t j = 0; j < f.inputs.size(); ++j)
				{
					if (j != 0)
					{
						out << ',';
					}
					TensorTypeJson(out, f.inputs[j]);
				}
				out << "],\"outputs\":[";
				for (std::size_t j = 0; j < f.outputs.size(); ++j)
				{
					if (j != 0)
					{
						out << ',';
					}
					TensorTypeJson(out, f.outputs[j]);
				}
				out << "]}";
			}
			out << "],\"regions\":[";
			for (std::size_t i = 0; i < manifest.regions.size(); ++i)
			{
				const auto& r = manifest.regions[i];
				if (i != 0)
				{
					out << ',';
				}
				out << "{\"id\":" << r.id << ",\"name\":";
				JsonString(out, r.name);
				out << ",\"function\":" << r.function << ",\"subgraph\":" << r.subgraph << ",\"nodes\":";
				NumberList(out, r.nodes);
				out << '}';
			}
			out << "],\"partitions\":[";
			for (std::size_t i = 0; i < manifest.partitions.size(); ++i)
			{
				const auto& p = manifest.partitions[i];
				if (i != 0)
				{
					out << ',';
				}
				out << "{\"id\":" << p.id << ",\"backend\":";
				JsonString(out, p.backend);
				out << ",\"regions\":";
				NumberList(out, p.regions);
				std::vector<std::uint32_t> spaces;
				for (const auto space : p.memorySpaces)
				{
					spaces.push_back(EnumValue(space));
				}
				out << ",\"memorySpaces\":";
				NumberList(out, spaces);
				out << '}';
			}
			out << "],\"memory\":";
			MemoryJson(out, manifest.memory);
			out << ",\"runtimeSegments\":[";
			for (std::size_t i = 0; i < manifest.runtimeSegments.size(); ++i)
			{
				if (i != 0)
				{
					out << ',';
				}
				RuntimeExecutionSegmentJson(out, manifest.runtimeSegments[i]);
			}
			out << "],\"runtimeSteps\":[";
			for (std::size_t i = 0; i < manifest.runtimeSteps.size(); ++i)
			{
				const auto& step = manifest.runtimeSteps[i];
				if (i != 0)
				{
					out << ',';
				}
				out << "{\"id\":" << step.id << ",\"kind\":" << EnumValue(step.kind)
				    << ",\"function\":" << step.function << ",\"region\":" << step.region << ",\"backend\":";
				JsonString(out, step.backend);
				out << ",\"segment\":";
				if (step.segment)
				{
					out << *step.segment;
				}
				else
				{
					out << "null";
				}
				out << ",\"fallbackBackend\":";
				JsonString(out, step.fallbackBackend);
				out << ",\"streamOwner\":";
				JsonString(out, step.streamOwner);
				out << ",\"eventOwner\":";
				JsonString(out, step.eventOwner);
				out << ",\"syncScope\":";
				JsonString(out, step.syncScope);
				out << ",\"inputBuffers\":";
				NumberList(out, step.inputBuffers);
				out << ",\"outputBuffers\":";
				NumberList(out, step.outputBuffers);
				out << '}';
			}
			out << "],\"runtimeStates\":[";
			for (std::size_t i = 0; i < manifest.runtimeStates.size(); ++i)
			{
				if (i != 0)
				{
					out << ',';
				}
				RuntimeStateBindingJson(out, manifest.runtimeStates[i]);
			}
			out << "],\"stateValueBindings\":[";
			for (std::size_t i = 0; i < manifest.stateValueBindings.size(); ++i)
			{
				if (i != 0)
				{
					out << ',';
				}
				RuntimeStateValueBindingJson(out, manifest.stateValueBindings[i]);
			}
			out << "],\"bufferBindings\":[";
			for (std::size_t i = 0; i < manifest.bufferBindings.size(); ++i)
			{
				if (i != 0)
				{
					out << ',';
				}
				RuntimeBufferBindingJson(out, manifest.bufferBindings[i]);
			}
			out << "],\"tensors\":[";
			for (std::size_t i = 0; i < manifest.tensors.size(); ++i)
			{
				if (i != 0)
				{
					out << ',';
				}
				TensorRefJson(out, manifest.tensors[i]);
			}
			out << "],\"adapters\":[";
			for (std::size_t i = 0; i < manifest.adapters.size(); ++i)
			{
				if (i != 0)
				{
					out << ',';
				}
				AdapterRefJson(out, manifest.adapters[i]);
			}
			out << "],\"artifacts\":[";
			for (std::size_t i = 0; i < manifest.artifacts.size(); ++i)
			{
				const auto& a = manifest.artifacts[i];
				if (i != 0)
				{
					out << ',';
				}
				out << "{\"name\":";
				JsonString(out, a.name);
				out << ",\"backend\":";
				JsonString(out, a.backend);
				out << ",\"entries\":[";
				for (std::size_t j = 0; j < a.entries.size(); ++j)
				{
					const auto& entry = a.entries[j];
					if (j != 0)
					{
						out << ',';
					}
					out << "{\"name\":";
					JsonString(out, entry.name);
					out << ",\"kind\":" << EnumValue(entry.kind) << ",\"function\":";
					if (entry.function)
					{
						out << *entry.function;
					}
					else
					{
						out << "null";
					}
					out << ",\"sourceSubgraph\":";
					if (entry.sourceSubgraph)
					{
						out << *entry.sourceSubgraph;
					}
					else
					{
						out << "null";
					}
					out << ",\"requiredStateBindings\":";
					StringListJson(out, entry.requiredStateBindings);
					out << ",\"requiredBufferBindings\":";
					StringListJson(out, entry.requiredBufferBindings);
					out << '}';
				}
				out << "],\"regions\":[";
				for (std::size_t j = 0; j < a.regions.size(); ++j)
				{
					const auto& r = a.regions[j];
					if (j != 0)
					{
						out << ',';
					}
					out << "{\"name\":";
					JsonString(out, r.name);
					out << ",\"kind\":" << EnumValue(r.kind) << ",\"relativePath\":";
					JsonString(out, r.relativePath);
					out << ",\"byteOffset\":" << r.byteOffset << ",\"byteSize\":" << r.byteSize
					    << ",\"checksum\":" << r.checksum << '}';
				}
				out << "],\"externalTensors\":[";
				for (std::size_t j = 0; j < a.externalTensors.size(); ++j)
				{
					if (j != 0)
					{
						out << ',';
					}
					TensorRefJson(out, a.externalTensors[j]);
				}
				out << "],\"backendRequirements\":[";
				for (std::size_t j = 0; j < a.backendRequirements.size(); ++j)
				{
					if (j != 0)
					{
						out << ',';
					}
					BackendRequirementJson(out, a.backendRequirements[j]);
				}
				out << "]}";
			}
			out << "],\"opCoverageCount\":" << manifest.opCoverage.size() << '}';
		}

		std::vector<std::size_t> SizeList(simdjson::dom::element value, std::string_view label)
		{
			std::vector<std::size_t> result;
			for (const auto item : AsArray(value, label))
			{
				result.push_back(static_cast<std::size_t>(AsUInt(item, label)));
			}
			return result;
		}

		TensorType ParseTensorType(simdjson::dom::element value, std::string_view label)
		{
			const auto object = AsObject(value, label);
			TensorShape shape;
			for (const auto dimElement : AsArray(Member(object, "shape", label), label))
			{
				const auto dim = AsObject(dimElement, label);
				shape.dims.push_back({ static_cast<TensorDimKind>(AsUInt(Member(dim, "kind", label), label)),
				                       static_cast<std::size_t>(AsUInt(Member(dim, "extent", label), label)),
				                       AsString(Member(dim, "symbol", label), label) });
			}
			const auto layoutObject = AsObject(Member(object, "layout", label), label);
			TensorLayout layout;
			layout.kind = static_cast<TensorLayoutKind>(AsUInt(Member(layoutObject, "kind", label), label));
			layout.strides = SizeList(Member(layoutObject, "strides", label), label);
			layout.tag = AsString(Member(layoutObject, "tag", label), label);
			return { static_cast<DataType>(AsUInt(Member(object, "dtype", label), label)), std::move(shape),
				     std::move(layout),
				     static_cast<TensorMemorySpace>(AsUInt(Member(object, "memorySpace", label), label)) };
		}

		std::vector<TensorType> TensorTypeList(simdjson::dom::element value, std::string_view label)
		{
			std::vector<TensorType> result;
			for (const auto item : AsArray(value, label))
			{
				result.push_back(ParseTensorType(item, label));
			}
			return result;
		}

		NodeOutput ParseNodeOutput(simdjson::dom::element value, std::string_view label)
		{
			const auto object = AsObject(value, label);
			return { static_cast<NodeId>(AsUInt(Member(object, "node", label), label)),
				     static_cast<std::size_t>(AsUInt(Member(object, "port", label), label)) };
		}

		std::vector<NodeOutput> NodeOutputList(simdjson::dom::element value, std::string_view label)
		{
			std::vector<NodeOutput> result;
			for (const auto item : AsArray(value, label))
			{
				result.push_back(ParseNodeOutput(item, label));
			}
			return result;
		}

		std::vector<std::string> StringList(simdjson::dom::element value, std::string_view label)
		{
			std::vector<std::string> result;
			for (const auto item : AsArray(value, label))
			{
				result.push_back(AsString(item, label));
			}
			return result;
		}

		ExecutablePlanOp ParsePlanOp(simdjson::dom::element value, std::string_view label)
		{
			const auto object = AsObject(value, label);
			ExecutablePlanOp op;
			op.kind = AsString(Member(object, "kind", label), label);
			op.schemaId = static_cast<std::uint32_t>(AsUInt(Member(object, "schemaId", label), label));
			op.category = static_cast<OpCategory>(AsUInt(Member(object, "category", label), label));
			op.effect = static_cast<OpEffect>(AsUInt(Member(object, "effect", label), label));
			for (const auto item : AsArray(Member(object, "attributes", label), label))
			{
				const auto attr = AsObject(item, label);
				op.attributes.push_back({ .name = AsString(Member(attr, "name", label), label),
				                          .value = AsString(Member(attr, "value", label), label) });
			}
			return op;
		}

		VNextExternalTensorRef ParseTensorRef(simdjson::dom::element value, std::string_view label)
		{
			const auto object = AsObject(value, label);
			VNextExternalTensorRef tensor;
			tensor.name = AsString(Member(object, "name", label), label);
			tensor.type = ParseTensorType(Member(object, "type", label), label);
			tensor.kind = static_cast<ExternalBufferKind>(AsUInt(Member(object, "kind", label), label));
			tensor.relativePath = AsString(Member(object, "relativePath", label), label);
			tensor.byteOffset = static_cast<std::size_t>(AsUInt(Member(object, "byteOffset", label), label));
			tensor.byteSize = static_cast<std::size_t>(AsUInt(Member(object, "byteSize", label), label));
			tensor.alignment = static_cast<std::size_t>(AsUInt(Member(object, "alignment", label), label));
			tensor.checksum = AsUInt(Member(object, "checksum", label), label);
			tensor.mutability = static_cast<BufferMutability>(AsUInt(Member(object, "mutability", label), label));
			tensor.rebindPolicy = static_cast<BufferRebindPolicy>(AsUInt(Member(object, "rebindPolicy", label), label));
			if (const auto quantElement = FindMember(object, "quantization"))
			{
				const auto quantObject = AsObject(*quantElement, label);
				QuantizationParams q;
				q.scheme = static_cast<QuantizationScheme>(AsUInt(Member(quantObject, "scheme", label), label));
				q.granularity =
				    static_cast<QuantizationGranularity>(AsUInt(Member(quantObject, "granularity", label), label));
				q.blockFormat =
				    static_cast<QuantizedBlockFormat>(AsUInt(Member(quantObject, "blockFormat", label), label));
				q.packedFormat =
				    static_cast<PackedNibbleFormat>(AsUInt(Member(quantObject, "packedFormat", label), label));
				q.packedOrder =
				    static_cast<PackedNibbleOrder>(AsUInt(Member(quantObject, "packedOrder", label), label));
				q.blockScaleLayout =
				    static_cast<BlockScaleLayout>(AsUInt(Member(quantObject, "blockScaleLayout", label), label));
				q.storageLayout =
				    static_cast<QuantizedStorageLayout>(AsUInt(Member(quantObject, "storageLayout", label), label));
				q.storageType = static_cast<DataType>(AsUInt(Member(quantObject, "storageType", label), label));
				q.expressedType = static_cast<DataType>(AsUInt(Member(quantObject, "expressedType", label), label));
				q.axis = AsInt(Member(quantObject, "axis", label), label);
				q.groupSize = static_cast<std::size_t>(AsUInt(Member(quantObject, "groupSize", label), label));
				for (const auto item : AsArray(Member(quantObject, "scales", label), label))
				{
					q.scales.push_back(static_cast<float>(AsDouble(item, label)));
				}
				for (const auto item : AsArray(Member(quantObject, "zeroPoints", label), label))
				{
					q.zeroPoints.push_back(static_cast<std::int32_t>(AsInt(item, label)));
				}
				q.expressedShape = SizeList(Member(quantObject, "expressedShape", label), label);
				tensor.quantization = std::move(q);
			}
			return tensor;
		}

		TensorStorageRef ParseTensorStorageRef(simdjson::dom::element value, std::string_view label)
		{
			const auto tensor = ParseTensorRef(value, label);
			return { .type = tensor.type,
				     .quantization = tensor.quantization,
				     .region = { .ownership = tensor.kind == ExternalBufferKind::None ? BufferOwnership::Owned
				                                                                      : BufferOwnership::External,
				                 .externalKind = tensor.kind,
				                 .memorySpace = tensor.type.memorySpace,
				                 .name = tensor.relativePath.empty() ? tensor.name : tensor.relativePath,
				                 .data = nullptr,
				                 .byteOffset = tensor.byteOffset,
				                 .byteSize = tensor.byteSize,
				                 .alignment = tensor.alignment,
				                 .checksum = tensor.checksum,
				                 .mutability = tensor.mutability,
				                 .rebindPolicy = tensor.rebindPolicy },
				     .viewMutability = tensor.mutability };
		}

		std::shared_ptr<const std::vector<std::byte>> ReadExternalTensorFile(const std::filesystem::path& path)
		{
			std::ifstream in(path, std::ios::binary | std::ios::ate);
			if (!in)
			{
				throw std::runtime_error("Failed to open vNext external tensor file: " + path.string());
			}
			const auto size = in.tellg();
			if (size < 0)
			{
				throw std::runtime_error("Failed to determine vNext external tensor file size: " + path.string());
			}
			auto storage = std::make_shared<std::vector<std::byte>>(static_cast<std::size_t>(size));
			in.seekg(0, std::ios::beg);
			if (!storage->empty())
			{
				in.read(reinterpret_cast<char*>(storage->data()), static_cast<std::streamsize>(storage->size()));
				if (!in)
				{
					throw std::runtime_error("Failed to read vNext external tensor file: " + path.string());
				}
			}
			return storage;
		}

		void BindExternalTensorFiles(ExecutablePlan& plan, const std::filesystem::path& packagePath)
		{
			std::map<std::filesystem::path, std::shared_ptr<const std::vector<std::byte>>> cache;
			const auto base = packagePath.parent_path();
			for (auto& variable : plan.variables)
			{
				if (!variable.IsExternal())
				{
					continue;
				}
				if (variable.region.name.empty())
				{
					throw std::runtime_error("vNext external tensor variable has an empty relative path");
				}
				const auto relative = std::filesystem::path(variable.region.name);
				const auto resolved = relative.is_absolute() ? relative : base / relative;
				auto [it, inserted] = cache.try_emplace(resolved);
				if (inserted)
				{
					it->second = ReadExternalTensorFile(resolved);
				}
				const auto& storage = it->second;
				if (variable.region.byteOffset > storage->size() ||
				    variable.region.byteSize > storage->size() - variable.region.byteOffset)
				{
					throw std::runtime_error("vNext external tensor view exceeds file size: " + resolved.string());
				}
				variable.region.data = storage->data();
				variable.region.owner = storage;
			}
		}

		const std::string& RequirePlanAttribute(const ExecutablePlanOp& op, std::string_view name)
		{
			const auto it = std::ranges::find_if(
			    op.attributes, [&](const ExecutablePlanAttribute& attr) { return attr.name == name; });
			if (it == op.attributes.end())
			{
				throw std::runtime_error("vNext node descriptor for '" + op.kind +
				                         "' is missing attribute: " + std::string(name));
			}
			return it->value;
		}

		std::size_t PlanAttributeSize(const ExecutablePlanOp& op, std::string_view name)
		{
			return static_cast<std::size_t>(std::stoull(RequirePlanAttribute(op, name)));
		}

		std::int64_t PlanAttributeInt(const ExecutablePlanOp& op, std::string_view name)
		{
			return std::stoll(RequirePlanAttribute(op, name));
		}

		double PlanAttributeDouble(const ExecutablePlanOp& op, std::string_view name)
		{
			return std::stod(RequirePlanAttribute(op, name));
		}

		bool PlanAttributeBool(const ExecutablePlanOp& op, std::string_view name)
		{
			const auto& value = RequirePlanAttribute(op, name);
			if (value == "true")
			{
				return true;
			}
			if (value == "false")
			{
				return false;
			}
			throw std::runtime_error("vNext node descriptor for '" + op.kind +
			                         "' has an invalid boolean attribute: " + std::string(name));
		}

		template <typename Enum>
		    requires std::is_enum_v<Enum>
		Enum PlanAttributeEnum(const ExecutablePlanOp& op, std::string_view name)
		{
			return static_cast<Enum>(PlanAttributeSize(op, name));
		}

		std::vector<std::size_t> PlanAttributeSizeList(const ExecutablePlanOp& op, std::string_view name)
		{
			const auto& text = RequirePlanAttribute(op, name);
			std::vector<std::size_t> values;
			if (text.empty())
			{
				return values;
			}
			std::stringstream stream(text);
			std::string item;
			while (std::getline(stream, item, ','))
			{
				if (item.empty())
				{
					throw std::runtime_error("vNext node descriptor for '" + op.kind +
					                         "' has an empty size-list item: " + std::string(name));
				}
				values.push_back(static_cast<std::size_t>(std::stoull(item)));
			}
			return values;
		}

		std::vector<float> PlanAttributeFloatList(const ExecutablePlanOp& op, std::string_view name)
		{
			const auto& text = RequirePlanAttribute(op, name);
			std::vector<float> values;
			if (text.empty())
			{
				return values;
			}
			std::stringstream stream(text);
			std::string item;
			while (std::getline(stream, item, ','))
			{
				if (item.empty())
				{
					throw std::runtime_error("vNext node descriptor has an empty float-list item");
				}
				values.push_back(std::stof(item));
			}
			return values;
		}

		std::vector<std::int32_t> PlanAttributeIntList(const ExecutablePlanOp& op, std::string_view name)
		{
			const auto& text = RequirePlanAttribute(op, name);
			std::vector<std::int32_t> values;
			if (text.empty())
			{
				return values;
			}
			std::stringstream stream(text);
			std::string item;
			while (std::getline(stream, item, ','))
			{
				if (item.empty())
				{
					throw std::runtime_error("vNext node descriptor has an empty integer-list item");
				}
				values.push_back(static_cast<std::int32_t>(std::stoll(item)));
			}
			return values;
		}

		std::uint8_t HexNibble(char c)
		{
			if (c >= '0' && c <= '9')
			{
				return static_cast<std::uint8_t>(c - '0');
			}
			if (c >= 'a' && c <= 'f')
			{
				return static_cast<std::uint8_t>(10 + c - 'a');
			}
			if (c >= 'A' && c <= 'F')
			{
				return static_cast<std::uint8_t>(10 + c - 'A');
			}
			throw std::runtime_error("vNext tensor payload contains a non-hex character");
		}

		std::vector<std::byte> PlanAttributeHexBytes(const ExecutablePlanOp& op, std::string_view name)
		{
			const auto& text = RequirePlanAttribute(op, name);
			if ((text.size() % 2) != 0)
			{
				throw std::runtime_error("vNext tensor payload hex string must have an even length");
			}
			std::vector<std::byte> bytes(text.size() / 2);
			for (std::size_t i = 0; i < bytes.size(); ++i)
			{
				const auto high = HexNibble(text[2 * i]);
				const auto low = HexNibble(text[2 * i + 1]);
				bytes[i] = static_cast<std::byte>((high << 4) | low);
			}
			return bytes;
		}

		Tensor<PolymorphicDevice> PlanConstantTensor(const ExecutablePlanOp& op)
		{
			const auto dtype = PlanAttributeEnum<DataType>(op, "dtype");
			const auto shape = PlanAttributeSizeList(op, "shape");
			const auto data = PlanAttributeHexBytes(op, "dataHex");
			const auto expectedBytes = Detail::Product(shape) * ElementByteSize(dtype);
			if (data.size() != expectedBytes)
			{
				throw std::runtime_error(std::format("vNext ConstantNode payload size mismatch: expected {}, got {}",
				                                     expectedBytes, data.size()));
			}
			CPU cpu;
			Tensor<CPU> tensor(Uninitialized, shape, dtype, cpu);
			std::memcpy(tensor.UnsafeRawData(), data.data(), data.size());
			return tensor.CopyToDevice(PolymorphicDevice{ CPU{} });
		}

		QuantizationParams PlanQuantizationParams(const ExecutablePlanOp& op)
		{
			return {
				.scheme = PlanAttributeEnum<QuantizationScheme>(op, "scheme"),
				.granularity = PlanAttributeEnum<QuantizationGranularity>(op, "granularity"),
				.blockFormat = PlanAttributeEnum<QuantizedBlockFormat>(op, "blockFormat"),
				.packedFormat = PlanAttributeEnum<PackedNibbleFormat>(op, "packedFormat"),
				.packedOrder = PlanAttributeEnum<PackedNibbleOrder>(op, "packedOrder"),
				.blockScaleLayout = PlanAttributeEnum<BlockScaleLayout>(op, "blockScaleLayout"),
				.storageType = PlanAttributeEnum<DataType>(op, "storageType"),
				.expressedType = PlanAttributeEnum<DataType>(op, "expressedType"),
				.axis = PlanAttributeInt(op, "axis"),
				.groupSize = PlanAttributeSize(op, "groupSize"),
				.scales = PlanAttributeFloatList(op, "scales"),
				.zeroPoints = PlanAttributeIntList(op, "zeroPoints"),
				.expressedShape = PlanAttributeSizeList(op, "expressedShape"),
				.storageLayout = PlanAttributeEnum<QuantizedStorageLayout>(op, "storageLayout"),
			};
		}

		NodeOutput RequireNodeInput(std::span<const NodeOutput> inputs, std::size_t index, std::string_view opKind)
		{
			if (index >= inputs.size())
			{
				throw std::runtime_error("vNext node descriptor for '" + std::string(opKind) + "' has too few inputs");
			}
			return inputs[index];
		}

		NodeVariant HydrateExecutablePlanNodePayload(const ExecutablePlanOp& op, std::span<const NodeOutput> inputs)
		{
			if (op.kind == "ParamRefNode")
			{
				return ParamRefNode{ PlanAttributeSize(op, "paramIndex") };
			}
			if (op.kind == "ConstantNode")
			{
				return ConstantNode{ PlanConstantTensor(op) };
			}
			if (op.kind == "QuantizedConstantNode")
			{
				return QuantizedConstantNode{ PlanConstantTensor(op), PlanQuantizationParams(op) };
			}
			if (op.kind == "VariableRefNode")
			{
				return VariableRefNode{ PlanAttributeSize(op, "variableIndex") };
			}
			if (op.kind == "UnaryOpNode")
			{
				return UnaryOpNode{ PlanAttributeEnum<UnaryOp>(op, "op"), RequireNodeInput(inputs, 0, op.kind) };
			}
			if (op.kind == "BinaryOpNode")
			{
				return BinaryOpNode{ PlanAttributeEnum<BinaryOp>(op, "op"), RequireNodeInput(inputs, 0, op.kind),
					                 RequireNodeInput(inputs, 1, op.kind) };
			}
			if (op.kind == "CallNode")
			{
				return CallNode{ PlanAttributeSize(op, "callee"),
					             std::vector<NodeOutput>(inputs.begin(), inputs.end()) };
			}
			if (op.kind == "CastNode")
			{
				return CastNode{ RequireNodeInput(inputs, 0, op.kind), PlanAttributeEnum<DataType>(op, "targetType") };
			}
			if (op.kind == "QuantizeNode")
			{
				return QuantizeNode{ RequireNodeInput(inputs, 0, op.kind), PlanQuantizationParams(op) };
			}
			if (op.kind == "DequantizeNode")
			{
				return DequantizeNode{ RequireNodeInput(inputs, 0, op.kind), PlanQuantizationParams(op),
					                   PlanAttributeEnum<DataType>(op, "targetType") };
			}
			if (op.kind == "QuantizedMatMulNode")
			{
				return QuantizedMatMulNode{ RequireNodeInput(inputs, 0, op.kind), RequireNodeInput(inputs, 1, op.kind),
					                        PlanQuantizationParams(op), PlanAttributeBool(op, "transposeRhs") };
			}
			if (op.kind == "GroupedQuantizedMatMulNode")
			{
				if (inputs.size() < 3 || inputs.size() > 4)
				{
					throw std::runtime_error(
					    "vNext GroupedQuantizedMatMulNode descriptor requires three or four inputs");
				}
				return GroupedQuantizedMatMulNode{
					.lhs = RequireNodeInput(inputs, 0, op.kind),
					.rhsStorages = std::vector<NodeOutput>(inputs.begin() + 1, inputs.end()),
					.params = PlanQuantizationParams(op),
					.outputWidths = PlanAttributeSizeList(op, "outputWidths"),
					.transposeRhs = PlanAttributeBool(op, "transposeRhs"),
				};
			}
			if (op.kind == "QuantizedGetRowsNode")
			{
				return QuantizedGetRowsNode{ RequireNodeInput(inputs, 0, op.kind), RequireNodeInput(inputs, 1, op.kind),
					                         PlanQuantizationParams(op) };
			}
			if (op.kind == "ReduceOpNode")
			{
				return ReduceOpNode{ PlanAttributeEnum<ReduceOp>(op, "op"), RequireNodeInput(inputs, 0, op.kind),
					                 PlanAttributeSize(op, "axis") };
			}
			if (op.kind == "ReshapeNode")
			{
				return ReshapeNode{ RequireNodeInput(inputs, 0, op.kind), PlanAttributeSizeList(op, "targetShape") };
			}
			if (op.kind == "PermuteNode")
			{
				return PermuteNode{ RequireNodeInput(inputs, 0, op.kind), PlanAttributeSizeList(op, "permutation") };
			}
			if (op.kind == "BroadcastToNode")
			{
				return BroadcastToNode{ RequireNodeInput(inputs, 0, op.kind),
					                    PlanAttributeSizeList(op, "targetShape") };
			}
			if (op.kind == "SoftmaxNode")
			{
				return SoftmaxNode{ RequireNodeInput(inputs, 0, op.kind), PlanAttributeSize(op, "axis") };
			}
			if (op.kind == "ActivePrefixAttentionNode")
			{
				if (inputs.size() != 4)
				{
					throw std::runtime_error("vNext ActivePrefixAttentionNode descriptor requires four inputs");
				}
				return ActivePrefixAttentionNode{ .query = RequireNodeInput(inputs, 0, op.kind),
					                              .keys = RequireNodeInput(inputs, 1, op.kind),
					                              .values = RequireNodeInput(inputs, 2, op.kind),
					                              .currentPosition = RequireNodeInput(inputs, 3, op.kind),
					                              .scale = PlanAttributeDouble(op, "scale"),
					                              .kvHeadIndex = PlanAttributeSize(op, "kvHeadIndex") };
			}
			if (op.kind == "GroupedActivePrefixAttentionNode")
			{
				if (inputs.size() != 4)
				{
					throw std::runtime_error("vNext GroupedActivePrefixAttentionNode descriptor requires four inputs");
				}
				return GroupedActivePrefixAttentionNode{ .queries = RequireNodeInput(inputs, 0, op.kind),
					                                     .keys = RequireNodeInput(inputs, 1, op.kind),
					                                     .values = RequireNodeInput(inputs, 2, op.kind),
					                                     .currentPosition = RequireNodeInput(inputs, 3, op.kind),
					                                     .scale = PlanAttributeDouble(op, "scale"),
					                                     .queryGroupsPerKVHead =
					                                         PlanAttributeSize(op, "queryGroupsPerKVHead") };
			}
			if (op.kind == "GroupedPagedAttentionNode")
			{
				if (inputs.size() != 5)
				{
					throw std::runtime_error("vNext GroupedPagedAttentionNode descriptor requires five inputs");
				}
				return GroupedPagedAttentionNode{ .queries = RequireNodeInput(inputs, 0, op.kind),
					                              .kvState = RequireNodeInput(inputs, 1, op.kind),
					                              .pageTable = RequireNodeInput(inputs, 2, op.kind),
					                              .pageDescriptors = RequireNodeInput(inputs, 3, op.kind),
					                              .activeLength = RequireNodeInput(inputs, 4, op.kind),
					                              .scale = PlanAttributeDouble(op, "scale"),
					                              .queryGroupsPerKVHead =
					                                  PlanAttributeSize(op, "queryGroupsPerKVHead") };
			}
			if (op.kind == "PagedKVAppendNode")
			{
				if (inputs.size() != 7)
				{
					throw std::runtime_error("vNext PagedKVAppendNode descriptor requires seven inputs");
				}
				return PagedKVAppendNode{ .kvState = RequireNodeInput(inputs, 0, op.kind),
					                      .pageTable = RequireNodeInput(inputs, 1, op.kind),
					                      .pageDescriptors = RequireNodeInput(inputs, 2, op.kind),
					                      .activeLength = RequireNodeInput(inputs, 3, op.kind),
					                      .keys = RequireNodeInput(inputs, 4, op.kind),
					                      .values = RequireNodeInput(inputs, 5, op.kind),
					                      .position = RequireNodeInput(inputs, 6, op.kind) };
			}
			if (op.kind == "RoPENode")
			{
				const auto hasPositions = PlanAttributeBool(op, "hasPositions");
				if (inputs.size() != (hasPositions ? 2u : 1u))
				{
					throw std::runtime_error("vNext RoPENode descriptor has unexpected inputs");
				}
				return RoPENode{ .input = RequireNodeInput(inputs, 0, op.kind),
					             .positions = hasPositions
					                              ? std::optional<NodeOutput>{ RequireNodeInput(inputs, 1, op.kind) }
					                              : std::nullopt,
					             .base = PlanAttributeDouble(op, "base"),
					             .frequencyScale = PlanAttributeDouble(op, "frequencyScale"),
					             .positionOffset = PlanAttributeSize(op, "positionOffset") };
			}
			if (op.kind == "GetRowsNode")
			{
				return GetRowsNode{ RequireNodeInput(inputs, 0, op.kind), RequireNodeInput(inputs, 1, op.kind) };
			}
			if (op.kind == "ScatterNode")
			{
				return ScatterNode{ RequireNodeInput(inputs, 0, op.kind), RequireNodeInput(inputs, 1, op.kind),
					                RequireNodeInput(inputs, 2, op.kind), PlanAttributeSize(op, "axis"),
					                PlanAttributeEnum<ScatterMode>(op, "mode") };
			}
			if (op.kind == "NormalizationNode")
			{
				const auto hasScale = PlanAttributeBool(op, "hasScale");
				const auto hasBias = PlanAttributeBool(op, "hasBias");
				std::size_t inputIndex = 1;
				const auto scale = hasScale
				                       ? std::optional<NodeOutput>{ RequireNodeInput(inputs, inputIndex++, op.kind) }
				                       : std::nullopt;
				const auto bias = hasBias ? std::optional<NodeOutput>{ RequireNodeInput(inputs, inputIndex++, op.kind) }
				                          : std::nullopt;
				if (inputIndex != inputs.size())
				{
					throw std::runtime_error("vNext NormalizationNode descriptor has unexpected inputs");
				}
				return NormalizationNode{ .input = RequireNodeInput(inputs, 0, op.kind),
					                      .scale = scale,
					                      .bias = bias,
					                      .mode = PlanAttributeEnum<NormalizationMode>(op, "mode"),
					                      .axis = PlanAttributeSize(op, "axis"),
					                      .groupCount = PlanAttributeSize(op, "groupCount"),
					                      .epsilon = PlanAttributeDouble(op, "epsilon") };
			}
			if (op.kind == "ConcatNode")
			{
				return ConcatNode{ std::vector<NodeOutput>(inputs.begin(), inputs.end()),
					               PlanAttributeSize(op, "axis") };
			}
			if (op.kind == "SliceNode")
			{
				return SliceNode{ RequireNodeInput(inputs, 0, op.kind), PlanAttributeSize(op, "axis"),
					              PlanAttributeSize(op, "start"), PlanAttributeSize(op, "length") };
			}
			if (op.kind == "SGDStepNode")
			{
				if (inputs.size() != 2 && inputs.size() != 3)
				{
					throw std::runtime_error("vNext SGDStepNode descriptor requires two or three inputs");
				}
				return SGDStepNode{ RequireNodeInput(inputs, 0, op.kind),
					                RequireNodeInput(inputs, 1, op.kind),
					                inputs.size() == 3 ? std::optional<NodeOutput>{ inputs[2] } : std::nullopt,
					                PlanAttributeDouble(op, "learningRate"),
					                PlanAttributeDouble(op, "momentum"),
					                PlanAttributeDouble(op, "weightDecay"),
					                PlanAttributeBool(op, "nesterov") };
			}
			if (op.kind == "AdamWStepNode")
			{
				if (inputs.size() != 4)
				{
					throw std::runtime_error("vNext AdamWStepNode descriptor requires four inputs");
				}
				return AdamWStepNode{ RequireNodeInput(inputs, 0, op.kind),    RequireNodeInput(inputs, 1, op.kind),
					                  RequireNodeInput(inputs, 2, op.kind),    RequireNodeInput(inputs, 3, op.kind),
					                  PlanAttributeDouble(op, "learningRate"), PlanAttributeDouble(op, "beta1"),
					                  PlanAttributeDouble(op, "beta2"),        PlanAttributeDouble(op, "epsilon"),
					                  PlanAttributeDouble(op, "weightDecay"),  PlanAttributeSize(op, "step") };
			}
			throw std::runtime_error("vNext node descriptor cannot hydrate executable payload for op: " + op.kind);
		}

		MemoryPlan ParseMemory(simdjson::dom::element value, std::string_view label)
		{
			const auto object = AsObject(value, label);
			MemoryPlan memory;
			memory.workspaceBytes = static_cast<std::size_t>(AsUInt(Member(object, "workspaceBytes", label), label));
			memory.persistentBytes = static_cast<std::size_t>(AsUInt(Member(object, "persistentBytes", label), label));
			memory.externalBytes = static_cast<std::size_t>(AsUInt(Member(object, "externalBytes", label), label));
			memory.constantBytes = static_cast<std::size_t>(AsUInt(Member(object, "constantBytes", label), label));
			for (const auto item : AsArray(Member(object, "buffers", label), label))
			{
				const auto buffer = AsObject(item, label);
				memory.buffers.push_back({
				    .id = static_cast<std::size_t>(AsUInt(Member(buffer, "id", label), label)),
				    .kind = static_cast<MemoryBufferKind>(AsUInt(Member(buffer, "kind", label), label)),
				    .memorySpace = static_cast<TensorMemorySpace>(AsUInt(Member(buffer, "memorySpace", label), label)),
				    .byteSize = static_cast<std::size_t>(AsUInt(Member(buffer, "byteSize", label), label)),
				    .alignment = static_cast<std::size_t>(AsUInt(Member(buffer, "alignment", label), label)),
				    .aliasSet = static_cast<std::size_t>(AsUInt(Member(buffer, "aliasSet", label), label)),
				});
			}
			return memory;
		}

		RuntimeBufferBinding ParseRuntimeBufferBinding(simdjson::dom::element value, std::string_view label)
		{
			const auto object = AsObject(value, label);
			RuntimeBufferBinding binding;
			binding.name = AsString(Member(object, "name", label), label);
			binding.type = ParseTensorType(Member(object, "type", label), label);
			binding.ownership = static_cast<BufferOwnership>(AsUInt(Member(object, "ownership", label), label));
			binding.externalKind =
			    static_cast<ExternalBufferKind>(AsUInt(Member(object, "externalKind", label), label));
			binding.memorySpace = static_cast<TensorMemorySpace>(AsUInt(Member(object, "memorySpace", label), label));
			binding.memoryBuffer = static_cast<std::size_t>(AsUInt(Member(object, "memoryBuffer", label), label));
			binding.byteOffset = static_cast<std::size_t>(AsUInt(Member(object, "byteOffset", label), label));
			binding.byteSize = static_cast<std::size_t>(AsUInt(Member(object, "byteSize", label), label));
			binding.alignment = static_cast<std::size_t>(AsUInt(Member(object, "alignment", label), label));
			binding.checksum = AsUInt(Member(object, "checksum", label), label);
			binding.mutability = static_cast<BufferMutability>(AsUInt(Member(object, "mutability", label), label));
			binding.rebindPolicy =
			    static_cast<BufferRebindPolicy>(AsUInt(Member(object, "rebindPolicy", label), label));
			binding.strides = SizeList(Member(object, "strides", label), label);
			binding.layoutTag = AsString(Member(object, "layoutTag", label), label);
			binding.aliasSet = static_cast<std::size_t>(AsUInt(Member(object, "aliasSet", label), label));
			ValidateRuntimeBufferBinding(binding);
			return binding;
		}

		Runtime::RuntimeStateValueBinding ParseRuntimeStateValueBinding(simdjson::dom::element value,
		                                                                std::string_view label)
		{
			const auto object = AsObject(value, label);
			return {
				.stateName = AsString(Member(object, "stateName", label), label),
				.function = static_cast<FunctionId>(AsUInt(Member(object, "function", label), label)),
				.kind = static_cast<Runtime::RuntimeStateValueKind>(AsUInt(Member(object, "kind", label), label)),
				.valueIndex = static_cast<std::size_t>(AsUInt(Member(object, "valueIndex", label), label)),
				.stateByteOffset = static_cast<std::size_t>(AsUInt(Member(object, "stateByteOffset", label), label)),
			};
		}

		Runtime::RuntimeExecutionSegment ParseRuntimeExecutionSegment(simdjson::dom::element value,
		                                                              std::string_view label)
		{
			const auto object = AsObject(value, label);
			return {
				.id = static_cast<std::size_t>(AsUInt(Member(object, "id", label), label)),
				.subgraph = static_cast<SubgraphId>(AsUInt(Member(object, "subgraph", label), label)),
				.backend = AsString(Member(object, "backend", label), label),
				.nodes = SizeList(Member(object, "nodes", label), label),
				.inputBuffers = SizeList(Member(object, "inputBuffers", label), label),
				.outputBuffers = SizeList(Member(object, "outputBuffers", label), label),
			};
		}

		Runtime::RuntimeStateLayout ParseRuntimeStateLayout(simdjson::dom::element value, std::string_view label)
		{
			const auto object = AsObject(value, label);
			return {
				.kind = static_cast<Runtime::RuntimeStateLayoutKind>(AsUInt(Member(object, "kind", label), label)),
				.pageSizeTokens = static_cast<std::size_t>(AsUInt(Member(object, "pageSizeTokens", label), label)),
				.maxLogicalTokens = static_cast<std::size_t>(AsUInt(Member(object, "maxLogicalTokens", label), label)),
				.residentPageCount =
				    static_cast<std::size_t>(AsUInt(Member(object, "residentPageCount", label), label)),
				.keyValuePlaneCount =
				    static_cast<std::size_t>(AsUInt(Member(object, "keyValuePlaneCount", label), label)),
				.keyPlaneOffsetBytes =
				    static_cast<std::size_t>(AsUInt(Member(object, "keyPlaneOffsetBytes", label), label)),
				.valuePlaneOffsetBytes =
				    static_cast<std::size_t>(AsUInt(Member(object, "valuePlaneOffsetBytes", label), label)),
				.tokenByteStride = static_cast<std::size_t>(AsUInt(Member(object, "tokenByteStride", label), label)),
				.pageByteStride = static_cast<std::size_t>(AsUInt(Member(object, "pageByteStride", label), label)),
				.pageTableState = AsString(Member(object, "pageTableState", label), label),
				.pageDescriptorState = AsString(Member(object, "pageDescriptorState", label), label),
				.activeLengthState = AsString(Member(object, "activeLengthState", label), label),
			};
		}

		Runtime::RuntimeStateBinding ParseRuntimeStateBinding(simdjson::dom::element value, std::string_view label)
		{
			const auto object = AsObject(value, label);
			Runtime::RuntimeStateBinding binding;
			binding.name = AsString(Member(object, "name", label), label);
			binding.kind = static_cast<Runtime::RuntimeStateKind>(AsUInt(Member(object, "kind", label), label));
			binding.role = AsString(Member(object, "role", label), label);
			binding.type = ParseTensorType(Member(object, "type", label), label);
			binding.mutability = static_cast<BufferMutability>(AsUInt(Member(object, "mutability", label), label));
			binding.effects = StringList(Member(object, "effects", label), label);
			const auto memoryBuffer = Member(object, "memoryBuffer", label);
			if (!memoryBuffer.is_null())
			{
				binding.memoryBuffer = static_cast<std::size_t>(AsUInt(memoryBuffer, label));
			}
			if (const auto layout = FindMember(object, "layout"))
			{
				binding.layout = ParseRuntimeStateLayout(*layout, label);
			}
			Runtime::ValidateRuntimeStateBinding(binding);
			return binding;
		}

		VNextAdapterRef ParseAdapterRef(simdjson::dom::element value, std::string_view label)
		{
			const auto object = AsObject(value, label);
			VNextAdapterRef adapter;
			adapter.targetName = AsString(Member(object, "targetName", label), label);
			adapter.adapterName = AsString(Member(object, "adapterName", label), label);
			adapter.kind = AsString(Member(object, "kind", label), label);
			adapter.aTensor = static_cast<std::size_t>(AsUInt(Member(object, "aTensor", label), label));
			adapter.bTensor = static_cast<std::size_t>(AsUInt(Member(object, "bTensor", label), label));
			adapter.rank = static_cast<std::size_t>(AsUInt(Member(object, "rank", label), label));
			adapter.alpha = static_cast<float>(AsDouble(Member(object, "alpha", label), label));
			adapter.dropout = static_cast<float>(AsDouble(Member(object, "dropout", label), label));
			adapter.dtype = static_cast<DataType>(AsUInt(Member(object, "dtype", label), label));
			adapter.mergeMode = AsString(Member(object, "mergeMode", label), label);
			return adapter;
		}

		VNextBackendRequirementRef ParseBackendRequirementRef(simdjson::dom::element value, std::string_view label)
		{
			const auto object = AsObject(value, label);
			VNextBackendRequirementRef requirement;
			const auto segment = Member(object, "segment", label);
			if (!segment.is_null())
			{
				requirement.segment = static_cast<std::size_t>(AsUInt(segment, label));
			}
			requirement.backend = AsString(Member(object, "backend", label), label);
			requirement.requiredCapabilities = StringList(Member(object, "requiredCapabilities", label), label);
			requirement.transferABI = AsString(Member(object, "transferABI", label), label);
			requirement.allowsFallback = AsBool(Member(object, "allowsFallback", label), label);
			return requirement;
		}

		ExecutablePlan ParsePlan(simdjson::dom::element value)
		{
			const auto object = AsObject(value, "plan");
			ExecutablePlan plan;
			plan.forward = static_cast<SubgraphId>(AsUInt(Member(object, "forward", "plan.forward"), "plan.forward"));
			const auto backward = Member(object, "backward", "plan.backward");
			if (!backward.is_null())
			{
				plan.backward = static_cast<SubgraphId>(AsUInt(backward, "plan.backward"));
			}
			for (const auto item : AsArray(Member(object, "variables", "plan.variables"), "plan.variables"))
			{
				plan.variables.push_back(ParseTensorStorageRef(item, "plan.variables"));
			}
			plan.variableNames =
			    StringList(Member(object, "variableNames", "plan.variableNames"), "plan.variableNames");
			plan.activationSlots =
			    TensorTypeList(Member(object, "activationSlots", "plan.activationSlots"), "plan.activationSlots");
			plan.tapeSlots = TensorTypeList(Member(object, "tapeSlots", "plan.tapeSlots"), "plan.tapeSlots");
			const auto parseValues = [](simdjson::dom::element list, std::string_view label) {
				std::vector<ExecutablePlanValue> result;
				for (const auto item : AsArray(list, label))
				{
					const auto value = AsObject(item, label);
					result.push_back({ .source = ParseNodeOutput(Member(value, "source", label), label),
					                   .type = ParseTensorType(Member(value, "type", label), label),
					                   .name = AsString(Member(value, "name", label), label) });
				}
				return result;
			};
			plan.inputs = parseValues(Member(object, "inputs", "plan.inputs"), "plan.inputs");
			plan.outputs = parseValues(Member(object, "outputs", "plan.outputs"), "plan.outputs");
			for (const auto item : AsArray(Member(object, "subgraphs", "plan.subgraphs"), "plan.subgraphs"))
			{
				const auto subgraphObject = AsObject(item, "plan.subgraphs");
				ExecutablePlanSubgraph subgraph;
				subgraph.sourceSubgraph = static_cast<SubgraphId>(
				    AsUInt(Member(subgraphObject, "sourceSubgraph", "plan.subgraph.sourceSubgraph"),
				           "plan.subgraph.sourceSubgraph"));
				subgraph.params =
				    TensorTypeList(Member(subgraphObject, "params", "plan.subgraph.params"), "plan.subgraph.params");
				for (const auto nodeItem :
				     AsArray(Member(subgraphObject, "nodes", "plan.subgraph.nodes"), "plan.subgraph.nodes"))
				{
					const auto nodeObject = AsObject(nodeItem, "plan.subgraph.nodes");
					ExecutablePlanNode node;
					node.sourceNode = static_cast<NodeId>(
					    AsUInt(Member(nodeObject, "sourceNode", "plan.node.sourceNode"), "plan.node.sourceNode"));
					node.op = ParsePlanOp(Member(nodeObject, "op", "plan.node.op"), "plan.node.op");
					node.opKind = node.op.kind;
					node.category = node.op.category;
					node.effect = node.op.effect;
					node.inputs = NodeOutputList(Member(nodeObject, "inputs", "plan.node.inputs"), "plan.node.inputs");
					node.outputs =
					    TensorTypeList(Member(nodeObject, "outputs", "plan.node.outputs"), "plan.node.outputs");
					node.node = HydrateExecutablePlanNodePayload(node.op, node.inputs);
					subgraph.nodes.push_back(std::move(node));
				}
				subgraph.results =
				    NodeOutputList(Member(subgraphObject, "results", "plan.subgraph.results"), "plan.subgraph.results");
				plan.subgraphs.push_back(std::move(subgraph));
			}
			return plan;
		}

		VNextPackageManifest ParseManifest(simdjson::dom::element value)
		{
			const auto object = AsObject(value, "manifest");
			VNextPackageManifest manifest;
			const auto versions = AsObject(Member(object, "versions", "manifest.versions"), "manifest.versions");
			manifest.versions.manifest = static_cast<std::uint32_t>(
			    AsUInt(Member(versions, "manifest", "manifest.versions.manifest"), "manifest.versions.manifest"));
			manifest.versions.opSet = static_cast<std::uint32_t>(
			    AsUInt(Member(versions, "opSet", "manifest.versions.opSet"), "manifest.versions.opSet"));
			manifest.versions.dtypeSet = static_cast<std::uint32_t>(
			    AsUInt(Member(versions, "dtypeSet", "manifest.versions.dtypeSet"), "manifest.versions.dtypeSet"));
			manifest.versions.layoutSet = static_cast<std::uint32_t>(
			    AsUInt(Member(versions, "layoutSet", "manifest.versions.layoutSet"), "manifest.versions.layoutSet"));
			manifest.versions.quantizationSet = static_cast<std::uint32_t>(
			    AsUInt(Member(versions, "quantizationSet", "manifest.versions.quantizationSet"),
			           "manifest.versions.quantizationSet"));
			manifest.versions.artifactABI = static_cast<std::uint32_t>(AsUInt(
			    Member(versions, "artifactABI", "manifest.versions.artifactABI"), "manifest.versions.artifactABI"));
			const auto layout = AsObject(Member(object, "layout", "manifest.layout"), "manifest.layout");
			manifest.layout.mode = AsString(Member(layout, "mode", "manifest.layout.mode"), "manifest.layout.mode");
			manifest.layout.manifestPath = AsString(Member(layout, "manifestPath", "manifest.layout.manifestPath"),
			                                        "manifest.layout.manifestPath");
			manifest.layout.tensorDirectory =
			    AsString(Member(layout, "tensorDirectory", "manifest.layout.tensorDirectory"),
			             "manifest.layout.tensorDirectory");
			manifest.layout.artifactDirectory =
			    AsString(Member(layout, "artifactDirectory", "manifest.layout.artifactDirectory"),
			             "manifest.layout.artifactDirectory");
			for (const auto item : AsArray(Member(object, "functions", "manifest.functions"), "manifest.functions"))
			{
				const auto f = AsObject(item, "manifest.functions");
				manifest.functions.push_back({
				    .id = static_cast<FunctionId>(AsUInt(Member(f, "id", "function.id"), "function.id")),
				    .name = AsString(Member(f, "name", "function.name"), "function.name"),
				    .body = static_cast<SubgraphId>(AsUInt(Member(f, "body", "function.body"), "function.body")),
				    .inputs = TensorTypeList(Member(f, "inputs", "function.inputs"), "function.inputs"),
				    .outputs = TensorTypeList(Member(f, "outputs", "function.outputs"), "function.outputs"),
				});
			}
			for (const auto item : AsArray(Member(object, "regions", "manifest.regions"), "manifest.regions"))
			{
				const auto r = AsObject(item, "manifest.regions");
				manifest.regions.push_back({
				    .id = static_cast<RegionId>(AsUInt(Member(r, "id", "region.id"), "region.id")),
				    .name = AsString(Member(r, "name", "region.name"), "region.name"),
				    .function =
				        static_cast<FunctionId>(AsUInt(Member(r, "function", "region.function"), "region.function")),
				    .subgraph =
				        static_cast<SubgraphId>(AsUInt(Member(r, "subgraph", "region.subgraph"), "region.subgraph")),
				    .nodes = SizeList(Member(r, "nodes", "region.nodes"), "region.nodes"),
				});
			}
			for (const auto item : AsArray(Member(object, "partitions", "manifest.partitions"), "manifest.partitions"))
			{
				const auto p = AsObject(item, "manifest.partitions");
				ExecutablePartition partition;
				partition.id = static_cast<PartitionId>(AsUInt(Member(p, "id", "partition.id"), "partition.id"));
				partition.backend = AsString(Member(p, "backend", "partition.backend"), "partition.backend");
				partition.regions = SizeList(Member(p, "regions", "partition.regions"), "partition.regions");
				partition.memorySpaces.clear();
				for (const auto space :
				     AsArray(Member(p, "memorySpaces", "partition.memorySpaces"), "partition.memorySpaces"))
				{
					partition.memorySpaces.push_back(
					    static_cast<TensorMemorySpace>(AsUInt(space, "partition.memorySpace")));
				}
				manifest.partitions.push_back(std::move(partition));
			}
			manifest.memory = ParseMemory(Member(object, "memory", "manifest.memory"), "manifest.memory");
			for (const auto item :
			     AsArray(Member(object, "runtimeStates", "manifest.runtimeStates"), "manifest.runtimeStates"))
			{
				manifest.runtimeStates.push_back(ParseRuntimeStateBinding(item, "manifest.runtimeStates"));
			}
			for (const auto item : AsArray(Member(object, "stateValueBindings", "manifest.stateValueBindings"),
			                               "manifest.stateValueBindings"))
			{
				manifest.stateValueBindings.push_back(
				    ParseRuntimeStateValueBinding(item, "manifest.stateValueBindings"));
			}
			for (const auto item :
			     AsArray(Member(object, "bufferBindings", "manifest.bufferBindings"), "manifest.bufferBindings"))
			{
				manifest.bufferBindings.push_back(ParseRuntimeBufferBinding(item, "manifest.bufferBindings"));
			}
			for (const auto item :
			     AsArray(Member(object, "runtimeSegments", "manifest.runtimeSegments"), "manifest.runtimeSegments"))
			{
				manifest.runtimeSegments.push_back(ParseRuntimeExecutionSegment(item, "manifest.runtimeSegments"));
			}
			for (const auto item :
			     AsArray(Member(object, "runtimeSteps", "manifest.runtimeSteps"), "manifest.runtimeSteps"))
			{
				const auto step = AsObject(item, "manifest.runtimeSteps");
				Runtime::RuntimeScheduleStep runtimeStep{
					.id = static_cast<std::size_t>(AsUInt(Member(step, "id", "runtimeStep.id"), "runtimeStep.id")),
					.kind = static_cast<Runtime::RuntimeScheduleStepKind>(
					    AsUInt(Member(step, "kind", "runtimeStep.kind"), "runtimeStep.kind")),
					.function = static_cast<FunctionId>(
					    AsUInt(Member(step, "function", "runtimeStep.function"), "runtimeStep.function")),
					.region = static_cast<RegionId>(
					    AsUInt(Member(step, "region", "runtimeStep.region"), "runtimeStep.region")),
					.backend = AsString(Member(step, "backend", "runtimeStep.backend"), "runtimeStep.backend"),
					.fallbackBackend = AsString(Member(step, "fallbackBackend", "runtimeStep.fallbackBackend"),
					                            "runtimeStep.fallbackBackend"),
					.streamOwner =
					    AsString(Member(step, "streamOwner", "runtimeStep.streamOwner"), "runtimeStep.streamOwner"),
					.eventOwner =
					    AsString(Member(step, "eventOwner", "runtimeStep.eventOwner"), "runtimeStep.eventOwner"),
					.syncScope = AsString(Member(step, "syncScope", "runtimeStep.syncScope"), "runtimeStep.syncScope"),
					.inputBuffers =
					    SizeList(Member(step, "inputBuffers", "runtimeStep.inputBuffers"), "runtimeStep.inputBuffers"),
					.outputBuffers = SizeList(Member(step, "outputBuffers", "runtimeStep.outputBuffers"),
					                          "runtimeStep.outputBuffers"),
				};
				const auto segment = Member(step, "segment", "runtimeStep.segment");
				if (!segment.is_null())
				{
					runtimeStep.segment = static_cast<std::size_t>(AsUInt(segment, "runtimeStep.segment"));
				}
				manifest.runtimeSteps.push_back(std::move(runtimeStep));
			}
			for (const auto item : AsArray(Member(object, "tensors", "manifest.tensors"), "manifest.tensors"))
			{
				manifest.tensors.push_back(ParseTensorRef(item, "manifest.tensors"));
			}
			for (const auto item : AsArray(Member(object, "adapters", "manifest.adapters"), "manifest.adapters"))
			{
				manifest.adapters.push_back(ParseAdapterRef(item, "manifest.adapters"));
			}
			for (const auto item : AsArray(Member(object, "artifacts", "manifest.artifacts"), "manifest.artifacts"))
			{
				const auto a = AsObject(item, "manifest.artifacts");
				VNextArtifactRef artifact;
				artifact.name = AsString(Member(a, "name", "artifact.name"), "artifact.name");
				artifact.backend = AsString(Member(a, "backend", "artifact.backend"), "artifact.backend");
				for (const auto entryItem : AsArray(Member(a, "entries", "artifact.entries"), "artifact.entries"))
				{
					const auto entryObject = AsObject(entryItem, "artifact.entries");
					VNextArtifactEntryRef entry;
					entry.name = AsString(Member(entryObject, "name", "artifact.entry.name"), "artifact.entry.name");
					entry.kind = static_cast<VNextArtifactEntryKind>(
					    AsUInt(Member(entryObject, "kind", "artifact.entry.kind"), "artifact.entry.kind"));
					const auto function = Member(entryObject, "function", "artifact.entry.function");
					if (!function.is_null())
					{
						entry.function = static_cast<FunctionId>(AsUInt(function, "artifact.entry.function"));
					}
					if (const auto sourceSubgraph = FindMember(entryObject, "sourceSubgraph");
					    sourceSubgraph && !sourceSubgraph->is_null())
					{
						entry.sourceSubgraph =
						    static_cast<SubgraphId>(AsUInt(*sourceSubgraph, "artifact.entry.sourceSubgraph"));
					}
					entry.requiredStateBindings =
					    StringList(Member(entryObject, "requiredStateBindings", "artifact.entry.requiredStateBindings"),
					               "artifact.entry.requiredStateBindings");
					entry.requiredBufferBindings = StringList(
					    Member(entryObject, "requiredBufferBindings", "artifact.entry.requiredBufferBindings"),
					    "artifact.entry.requiredBufferBindings");
					artifact.entries.push_back(std::move(entry));
				}
				for (const auto regionItem : AsArray(Member(a, "regions", "artifact.regions"), "artifact.regions"))
				{
					const auto r = AsObject(regionItem, "artifact.regions");
					artifact.regions.push_back({
					    .name = AsString(Member(r, "name", "artifact.region.name"), "artifact.region.name"),
					    .kind = static_cast<ExternalBufferKind>(
					        AsUInt(Member(r, "kind", "artifact.region.kind"), "artifact.region.kind")),
					    .relativePath = AsString(Member(r, "relativePath", "artifact.region.relativePath"),
					                             "artifact.region.relativePath"),
					    .byteOffset = static_cast<std::size_t>(AsUInt(
					        Member(r, "byteOffset", "artifact.region.byteOffset"), "artifact.region.byteOffset")),
					    .byteSize = static_cast<std::size_t>(
					        AsUInt(Member(r, "byteSize", "artifact.region.byteSize"), "artifact.region.byteSize")),
					    .checksum =
					        AsUInt(Member(r, "checksum", "artifact.region.checksum"), "artifact.region.checksum"),
					});
				}
				for (const auto tensorItem :
				     AsArray(Member(a, "externalTensors", "artifact.externalTensors"), "artifact.externalTensors"))
				{
					artifact.externalTensors.push_back(ParseTensorRef(tensorItem, "artifact.externalTensors"));
				}
				for (const auto requirementItem :
				     AsArray(Member(a, "backendRequirements", "artifact.backendRequirements"),
				             "artifact.backendRequirements"))
				{
					artifact.backendRequirements.push_back(
					    ParseBackendRequirementRef(requirementItem, "artifact.backendRequirements"));
				}
				manifest.artifacts.push_back(std::move(artifact));
			}
			return manifest;
		}

		void WriteVNextModelPackageFile(const std::filesystem::path& path, const VNextPackageManifest& manifest,
		                                const ExecutablePlan& plan)
		{
			std::ofstream out(path, std::ios::binary);
			if (!out)
			{
				throw std::runtime_error("Failed to open LiteNN vNext model package for writing: " + path.string());
			}
			out << "{\"format\":";
			JsonString(out, kFormat);
			out << ",\"manifest\":";
			ManifestJson(out, manifest);
			out << ",\"plan\":";
			PlanJson(out, plan);
			out << "}\n";
			if (!out)
			{
				throw std::runtime_error("Failed to write LiteNN vNext model package: " + path.string());
			}
		}
	} // namespace

	void SaveVNextModelPackage(const ExecutableModule& module, const std::filesystem::path& path,
	                           std::vector<VNextArtifactRef> artifacts, VNextPackageLayout layout,
	                           std::vector<VNextAdapterRef> adapters,
	                           std::vector<Runtime::RuntimeStateBinding> runtimeStates)
	{
		ValidateExecutablePlan(module.plan);
		auto manifest = BuildVNextPackageManifest(module, std::move(artifacts), std::move(layout), std::move(adapters),
		                                          std::move(runtimeStates));
		ValidateVNextPackageManifest(manifest);

		WriteVNextModelPackageFile(path, manifest, module.plan);
	}

	void SaveVNextModelPackage(const Runtime::RuntimeSchedule& schedule, const std::filesystem::path& path,
	                           std::vector<VNextArtifactRef> artifacts, VNextPackageLayout layout,
	                           std::vector<VNextAdapterRef> adapters)
	{
		Runtime::ValidateRuntimeSchedule(schedule);
		auto manifest =
		    BuildVNextPackageManifest(schedule, std::move(artifacts), std::move(layout), std::move(adapters));
		ValidateVNextPackageManifest(manifest);
		WriteVNextModelPackageFile(path, manifest, schedule.module.plan);
	}

	void SaveVNextModelPackageExternalWeights(const Runtime::RuntimeSchedule& sourceSchedule,
	                                          const std::filesystem::path& path,
	                                          const std::filesystem::path& externalWeightsPath,
	                                          const ExternalWeightSaveOptions& externalOptions)
	{
		if (std::filesystem::absolute(externalWeightsPath).lexically_normal() ==
		    std::filesystem::absolute(path).lexically_normal())
		{
			throw std::runtime_error("LiteNN vNext external weight file must be different from the package file");
		}

		auto schedule = sourceSchedule;
		std::ofstream weights(externalWeightsPath, std::ios::binary);
		if (!weights)
		{
			throw std::runtime_error("Failed to open LiteNN vNext external weight file for writing: " +
			                         externalWeightsPath.string());
		}
		std::error_code ec;
		auto relativePath = std::filesystem::relative(externalWeightsPath, path.parent_path(), ec);
		const auto externalPathText = ec ? externalWeightsPath.string() : relativePath.string();
		const auto alignment = std::max<std::uint64_t>(externalOptions.alignment, 1);

		for (std::size_t i = 0; i < schedule.module.plan.variables.size(); ++i)
		{
			const auto bindingName = schedule.bufferBindings.at(i).name;
			auto& storage = schedule.module.plan.variables[i];
			const auto byteCount = storage.region.byteSize;
			if (storage.region.data == nullptr && byteCount != 0)
			{
				throw std::runtime_error("Runtime schedule variable has no payload data: " + storage.region.name);
			}
			const auto rawPosition = weights.tellp();
			if (rawPosition == std::streampos(-1))
			{
				throw std::runtime_error("Failed to determine LiteNN vNext external weight output offset");
			}
			const auto rawOffset = static_cast<std::uint64_t>(rawPosition);
			const auto alignedOffset = ((rawOffset + alignment - 1) / alignment) * alignment;
			std::vector<char> padding(static_cast<std::size_t>(alignedOffset - rawOffset), '\0');
			if (!padding.empty())
			{
				weights.write(padding.data(), static_cast<std::streamsize>(padding.size()));
			}
			const auto* payload = static_cast<const char*>(storage.region.data) + storage.storageOffsetBytes;
			weights.write(payload, static_cast<std::streamsize>(byteCount));
			if (!weights)
			{
				throw std::runtime_error("Failed to write LiteNN vNext external weight payload");
			}

			storage.region.ownership = BufferOwnership::External;
			storage.region.externalKind = ExternalBufferKind::User;
			storage.region.name = externalPathText;
			storage.region.data = nullptr;
			storage.region.byteOffset = static_cast<std::size_t>(alignedOffset);
			storage.region.alignment = static_cast<std::size_t>(alignment);
			storage.region.mutability = BufferMutability::Immutable;
			storage.region.rebindPolicy = BufferRebindPolicy::ExactMetadataAndChecksum;
			storage.storageOffsetBytes = 0;

			auto& memoryBuffer = schedule.memory.buffers.at(i);
			memoryBuffer.kind = MemoryBufferKind::External;
			memoryBuffer.alignment = static_cast<std::size_t>(alignment);
			schedule.bufferBindings.at(i) =
			    ToRuntimeBufferBinding(bindingName.empty() ? std::format("variable.{}", i) : bindingName,
			                           schedule.module.plan.variables[i], i);
		}

		schedule.memory.workspaceBytes = 0;
		schedule.memory.persistentBytes = 0;
		schedule.memory.externalBytes = 0;
		schedule.memory.constantBytes = 0;
		for (const auto& buffer : schedule.memory.buffers)
		{
			switch (buffer.kind)
			{
			case MemoryBufferKind::Workspace:
				schedule.memory.workspaceBytes += buffer.byteSize;
				break;
			case MemoryBufferKind::Persistent:
				schedule.memory.persistentBytes += buffer.byteSize;
				break;
			case MemoryBufferKind::External:
				schedule.memory.externalBytes += buffer.byteSize;
				break;
			case MemoryBufferKind::Constant:
				schedule.memory.constantBytes += buffer.byteSize;
				break;
			}
		}
		SaveVNextModelPackage(schedule, path);
	}

	void SaveVNextModelPackageExternalWeights(const Graph& graph, const std::filesystem::path& path,
	                                          const std::filesystem::path& externalWeightsPath,
	                                          const ExternalWeightSaveOptions& externalOptions)
	{
		if (std::filesystem::absolute(externalWeightsPath).lexically_normal() ==
		    std::filesystem::absolute(path).lexically_normal())
		{
			throw std::runtime_error("LiteNN vNext external weight file must be different from the package file");
		}

		Validation::ValidateGraph(graph);
		auto module = ::LiteNN::Detail::BuildExecutableModuleFromGraph(graph);
		std::ofstream weights(externalWeightsPath, std::ios::binary);
		if (!weights)
		{
			throw std::runtime_error("Failed to open LiteNN vNext external weight file for writing: " +
			                         externalWeightsPath.string());
		}

		std::error_code ec;
		auto relativePath = std::filesystem::relative(externalWeightsPath, path.parent_path(), ec);
		const auto externalPathText = ec ? externalWeightsPath.string() : relativePath.string();

		for (std::size_t i = 0; i < graph.VariableCount(); ++i)
		{
			const auto& tensor = graph.GetVariable(i)->Data();
			const auto byteCount = static_cast<std::size_t>(tensor.NumElements() * ElementByteSize(tensor.DType()));
			const auto rawPosition = weights.tellp();
			if (rawPosition == std::streampos(-1))
			{
				throw std::runtime_error("Failed to determine LiteNN vNext external weight output offset");
			}
			const auto rawOffset = static_cast<std::uint64_t>(rawPosition);
			const auto alignment = std::max<std::uint64_t>(externalOptions.alignment, 1);
			const auto alignedOffset = ((rawOffset + alignment - 1) / alignment) * alignment;
			std::vector<char> padding(static_cast<std::size_t>(alignedOffset - rawOffset), '\0');
			if (!padding.empty())
			{
				weights.write(padding.data(), static_cast<std::streamsize>(padding.size()));
			}
			weights.write(static_cast<const char*>(tensor.UnsafeRawData()), static_cast<std::streamsize>(byteCount));
			if (!weights)
			{
				throw std::runtime_error("Failed to write LiteNN vNext external weight payload");
			}

			auto& storage = module.plan.variables.at(i);
			storage.region.ownership = BufferOwnership::External;
			storage.region.externalKind = ExternalBufferKind::User;
			storage.region.memorySpace = TensorMemorySpaceFor(tensor.CurDevice());
			storage.region.name = externalPathText;
			storage.region.data = nullptr;
			storage.region.byteOffset = static_cast<std::size_t>(alignedOffset);
			storage.region.byteSize = byteCount;
			storage.region.alignment = static_cast<std::size_t>(alignment);
			storage.region.mutability = BufferMutability::Immutable;
			storage.region.rebindPolicy = BufferRebindPolicy::ExactMetadataAndChecksum;
			storage.storageOffsetBytes = 0;
		}

		SaveVNextModelPackage(module, path);
	}

	VNextModelPackage LoadVNextModelPackage(const std::filesystem::path& path)
	{
		std::ifstream in(path, std::ios::binary);
		if (!in)
		{
			throw std::runtime_error("Failed to open LiteNN vNext model package: " + path.string());
		}
		std::stringstream buffer;
		buffer << in.rdbuf();
		const auto json = buffer.str();

		simdjson::padded_string padded(json.data(), json.size());
		simdjson::dom::parser parser;
		simdjson::dom::element root;
		if (const auto error = parser.parse(padded).get(root))
		{
			throw JsonError("LiteNN vNext model package root", error);
		}
		const auto rootObject = AsObject(root, "LiteNN vNext model package root");
		const auto format = AsString(Member(rootObject, "format", "package.format"), "package.format");
		if (format != kFormat)
		{
			throw std::runtime_error("Unsupported LiteNN model package format: " + format);
		}

		VNextModelPackage package;
		package.manifest = ParseManifest(Member(rootObject, "manifest", "package.manifest"));
		package.plan = ParsePlan(Member(rootObject, "plan", "package.plan"));
		package.sourcePath = path;
		BindExternalTensorFiles(package.plan, path);
		ValidateVNextPackageManifest(package.manifest);
		ValidateExecutablePlan(package.plan);
		return package;
	}

	const VNextLoadedArtifactRegion* VNextLoadedArtifactRegions::FindRegion(std::string_view name) const
	{
		const auto it = std::ranges::find_if(
		    regions, [&](const VNextLoadedArtifactRegion& region) { return region.ref.name == name; });
		return it == regions.end() ? nullptr : &*it;
	}

	VNextLoadedArtifactRegions LoadVNextArtifactRegions(const VNextModelPackage& package, std::string_view artifactName)
	{
		if (package.sourcePath.empty())
		{
			throw std::runtime_error(
			    "LiteNN vNext package source path is unknown; pass an explicit artifact base directory");
		}
		return LoadVNextArtifactRegions(package, package.sourcePath.parent_path(), artifactName);
	}

	VNextLoadedArtifactRegions LoadVNextArtifactRegions(const VNextModelPackage& package,
	                                                    const std::filesystem::path& baseDirectory,
	                                                    std::string_view artifactName)
	{
		const auto artifactIt = std::ranges::find_if(package.manifest.artifacts, [&](const VNextArtifactRef& artifact) {
			return artifact.name == artifactName;
		});
		if (artifactIt == package.manifest.artifacts.end())
		{
			throw std::runtime_error("LiteNN vNext package has no artifact named: " + std::string(artifactName));
		}

		VNextLoadedArtifactRegions loaded;
		loaded.artifact = *artifactIt;
		loaded.regions.reserve(artifactIt->regions.size());
		for (const auto& region : artifactIt->regions)
		{
			auto bytes = ReadAllBytes(ResolvePackageRelativePath(baseDirectory, region.relativePath));
			if (region.byteSize != bytes.size())
			{
				throw std::runtime_error("LiteNN vNext artifact region '" + region.name + "' size mismatch");
			}
			if (region.checksum != 0 && ChecksumBytes(bytes) != region.checksum)
			{
				throw std::runtime_error("LiteNN vNext artifact region '" + region.name + "' checksum mismatch");
			}
			loaded.regions.push_back({ .ref = region, .bytes = std::move(bytes) });
		}
		if (loaded.FindRegion("rodata") == nullptr && loaded.FindRegion("metadata") == nullptr)
		{
			throw std::runtime_error("LiteNN vNext artifact '" + std::string(artifactName) +
			                         "' has no rodata or metadata region");
		}
		if (loaded.FindRegion("instructions") == nullptr)
		{
			throw std::runtime_error("LiteNN vNext artifact '" + std::string(artifactName) +
			                         "' has no instructions region");
		}
		return loaded;
	}
} // namespace LiteNN::Serialization
