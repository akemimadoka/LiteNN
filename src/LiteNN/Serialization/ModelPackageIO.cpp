#include <LiteNN/Serialization/ModelPackageIO.h>

#include <fstream>
#include <optional>
#include <sstream>
#include <stdexcept>
#include <string>
#include <string_view>

#include <simdjson.h>

namespace LiteNN::Serialization
{
	namespace
	{
		constexpr std::string_view kFormat = "litenn.model.vnext";

		template <typename Enum>
		std::uint32_t EnumValue(Enum value)
		{
			return static_cast<std::uint32_t>(value);
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
			    << ",\"mutability\":" << EnumValue(tensor.mutability) << ",\"rebindPolicy\":"
			    << EnumValue(tensor.rebindPolicy);
			if (tensor.quantization)
			{
				const auto& q = *tensor.quantization;
				out << ",\"quantization\":{\"scheme\":" << EnumValue(q.scheme) << ",\"granularity\":"
				    << EnumValue(q.granularity) << ",\"blockFormat\":" << EnumValue(q.blockFormat)
				    << ",\"storageType\":" << EnumValue(q.storageType) << ",\"expressedType\":"
				    << EnumValue(q.expressedType) << ",\"axis\":" << q.axis << ",\"groupSize\":"
				    << q.groupSize << ",\"scales\":";
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
				out << "{\"id\":" << b.id << ",\"kind\":" << EnumValue(b.kind) << ",\"memorySpace\":"
				    << EnumValue(b.memorySpace) << ",\"byteSize\":" << b.byteSize << ",\"alignment\":"
				    << b.alignment << ",\"aliasSet\":" << b.aliasSet << '}';
			}
			out << "]}";
		}

		void RuntimeBufferBindingJson(std::ostream& out, const RuntimeBufferBinding& binding)
		{
			out << "{\"name\":";
			JsonString(out, binding.name);
			out << ",\"type\":";
			TensorTypeJson(out, binding.type);
			out << ",\"ownership\":" << EnumValue(binding.ownership) << ",\"externalKind\":"
			    << EnumValue(binding.externalKind) << ",\"memorySpace\":" << EnumValue(binding.memorySpace)
			    << ",\"memoryBuffer\":" << binding.memoryBuffer << ",\"byteOffset\":" << binding.byteOffset
			    << ",\"byteSize\":" << binding.byteSize << ",\"alignment\":" << binding.alignment
			    << ",\"checksum\":" << binding.checksum << ",\"mutability\":" << EnumValue(binding.mutability)
			    << ",\"rebindPolicy\":" << EnumValue(binding.rebindPolicy) << ",\"strides\":";
			NumberList(out, binding.strides);
			out << ",\"layoutTag\":";
			JsonString(out, binding.layoutTag);
			out << ",\"aliasSet\":" << binding.aliasSet << '}';
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
			out << "],\"activationSlots\":[";
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
			out << "{\"versions\":{\"manifest\":" << manifest.versions.manifest << ",\"opSet\":"
			    << manifest.versions.opSet << ",\"dtypeSet\":" << manifest.versions.dtypeSet << ",\"layoutSet\":"
			    << manifest.versions.layoutSet << ",\"quantizationSet\":" << manifest.versions.quantizationSet
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
			out << ",\"runtimeSteps\":[";
			for (std::size_t i = 0; i < manifest.runtimeSteps.size(); ++i)
			{
				const auto& step = manifest.runtimeSteps[i];
				if (i != 0)
				{
					out << ',';
				}
				out << "{\"id\":" << step.id << ",\"kind\":" << EnumValue(step.kind) << ",\"function\":"
				    << step.function << ",\"region\":" << step.region << ",\"backend\":";
				JsonString(out, step.backend);
				out << ",\"inputBuffers\":";
				NumberList(out, step.inputBuffers);
				out << ",\"outputBuffers\":";
				NumberList(out, step.outputBuffers);
				out << '}';
			}
			out << "],\"runtimeStates\":[],\"bufferBindings\":[";
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
				out << ",\"entryFunction\":" << a.entryFunction << ",\"regions\":[";
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
			tensor.rebindPolicy =
			    static_cast<BufferRebindPolicy>(AsUInt(Member(object, "rebindPolicy", label), label));
			if (const auto quantElement = FindMember(object, "quantization"))
			{
				const auto quantObject = AsObject(*quantElement, label);
				QuantizationParams q;
				q.scheme = static_cast<QuantizationScheme>(AsUInt(Member(quantObject, "scheme", label), label));
				q.granularity =
				    static_cast<QuantizationGranularity>(AsUInt(Member(quantObject, "granularity", label), label));
				q.blockFormat =
				    static_cast<QuantizedBlockFormat>(AsUInt(Member(quantObject, "blockFormat", label), label));
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
			binding.memorySpace =
			    static_cast<TensorMemorySpace>(AsUInt(Member(object, "memorySpace", label), label));
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
			plan.activationSlots = TensorTypeList(Member(object, "activationSlots", "plan.activationSlots"),
			                                      "plan.activationSlots");
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
				subgraph.params = TensorTypeList(Member(subgraphObject, "params", "plan.subgraph.params"),
				                                 "plan.subgraph.params");
				for (const auto nodeItem :
				     AsArray(Member(subgraphObject, "nodes", "plan.subgraph.nodes"), "plan.subgraph.nodes"))
				{
					const auto nodeObject = AsObject(nodeItem, "plan.subgraph.nodes");
					ExecutablePlanNode node;
					node.sourceNode =
					    static_cast<NodeId>(AsUInt(Member(nodeObject, "sourceNode", "plan.node.sourceNode"),
					                              "plan.node.sourceNode"));
					node.op = ParsePlanOp(Member(nodeObject, "op", "plan.node.op"), "plan.node.op");
					node.opKind = node.op.kind;
					node.category = node.op.category;
					node.effect = node.op.effect;
					node.inputs = NodeOutputList(Member(nodeObject, "inputs", "plan.node.inputs"), "plan.node.inputs");
					node.outputs =
					    TensorTypeList(Member(nodeObject, "outputs", "plan.node.outputs"), "plan.node.outputs");
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
			manifest.versions.manifest =
			    static_cast<std::uint32_t>(AsUInt(Member(versions, "manifest", "manifest.versions.manifest"),
			                                     "manifest.versions.manifest"));
			manifest.versions.opSet =
			    static_cast<std::uint32_t>(AsUInt(Member(versions, "opSet", "manifest.versions.opSet"),
			                                     "manifest.versions.opSet"));
			manifest.versions.dtypeSet =
			    static_cast<std::uint32_t>(AsUInt(Member(versions, "dtypeSet", "manifest.versions.dtypeSet"),
			                                     "manifest.versions.dtypeSet"));
			manifest.versions.layoutSet =
			    static_cast<std::uint32_t>(AsUInt(Member(versions, "layoutSet", "manifest.versions.layoutSet"),
			                                     "manifest.versions.layoutSet"));
			manifest.versions.quantizationSet = static_cast<std::uint32_t>(
			    AsUInt(Member(versions, "quantizationSet", "manifest.versions.quantizationSet"),
			           "manifest.versions.quantizationSet"));
			manifest.versions.artifactABI =
			    static_cast<std::uint32_t>(AsUInt(Member(versions, "artifactABI", "manifest.versions.artifactABI"),
			                                     "manifest.versions.artifactABI"));
			const auto layout = AsObject(Member(object, "layout", "manifest.layout"), "manifest.layout");
			manifest.layout.mode = AsString(Member(layout, "mode", "manifest.layout.mode"), "manifest.layout.mode");
			manifest.layout.manifestPath =
			    AsString(Member(layout, "manifestPath", "manifest.layout.manifestPath"), "manifest.layout.manifestPath");
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
				    .function = static_cast<FunctionId>(AsUInt(Member(r, "function", "region.function"),
				                                               "region.function")),
				    .subgraph = static_cast<SubgraphId>(AsUInt(Member(r, "subgraph", "region.subgraph"),
				                                               "region.subgraph")),
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
				for (const auto space : AsArray(Member(p, "memorySpaces", "partition.memorySpaces"),
				                                "partition.memorySpaces"))
				{
					partition.memorySpaces.push_back(static_cast<TensorMemorySpace>(AsUInt(space, "partition.memorySpace")));
				}
				manifest.partitions.push_back(std::move(partition));
			}
			manifest.memory = ParseMemory(Member(object, "memory", "manifest.memory"), "manifest.memory");
			for (const auto item : AsArray(Member(object, "bufferBindings", "manifest.bufferBindings"),
			                              "manifest.bufferBindings"))
			{
				manifest.bufferBindings.push_back(ParseRuntimeBufferBinding(item, "manifest.bufferBindings"));
			}
			for (const auto item : AsArray(Member(object, "runtimeSteps", "manifest.runtimeSteps"),
			                              "manifest.runtimeSteps"))
			{
				const auto step = AsObject(item, "manifest.runtimeSteps");
				manifest.runtimeSteps.push_back({
				    .id = static_cast<std::size_t>(AsUInt(Member(step, "id", "runtimeStep.id"), "runtimeStep.id")),
				    .kind = static_cast<Runtime::RuntimeScheduleStepKind>(
				        AsUInt(Member(step, "kind", "runtimeStep.kind"), "runtimeStep.kind")),
				    .function = static_cast<FunctionId>(AsUInt(Member(step, "function", "runtimeStep.function"),
				                                               "runtimeStep.function")),
				    .region = static_cast<RegionId>(AsUInt(Member(step, "region", "runtimeStep.region"),
				                                           "runtimeStep.region")),
				    .backend = AsString(Member(step, "backend", "runtimeStep.backend"), "runtimeStep.backend"),
				    .inputBuffers = SizeList(Member(step, "inputBuffers", "runtimeStep.inputBuffers"),
				                             "runtimeStep.inputBuffers"),
				    .outputBuffers = SizeList(Member(step, "outputBuffers", "runtimeStep.outputBuffers"),
				                              "runtimeStep.outputBuffers"),
				});
			}
			for (const auto item : AsArray(Member(object, "tensors", "manifest.tensors"), "manifest.tensors"))
			{
				manifest.tensors.push_back(ParseTensorRef(item, "manifest.tensors"));
			}
			for (const auto item : AsArray(Member(object, "artifacts", "manifest.artifacts"), "manifest.artifacts"))
			{
				const auto a = AsObject(item, "manifest.artifacts");
				VNextArtifactRef artifact;
				artifact.name = AsString(Member(a, "name", "artifact.name"), "artifact.name");
				artifact.backend = AsString(Member(a, "backend", "artifact.backend"), "artifact.backend");
				artifact.entryFunction =
				    static_cast<FunctionId>(AsUInt(Member(a, "entryFunction", "artifact.entryFunction"),
				                                   "artifact.entryFunction"));
				for (const auto regionItem : AsArray(Member(a, "regions", "artifact.regions"), "artifact.regions"))
				{
					const auto r = AsObject(regionItem, "artifact.regions");
					artifact.regions.push_back({
					    .name = AsString(Member(r, "name", "artifact.region.name"), "artifact.region.name"),
					    .kind = static_cast<ExternalBufferKind>(AsUInt(Member(r, "kind", "artifact.region.kind"),
					                                                   "artifact.region.kind")),
					    .relativePath = AsString(Member(r, "relativePath", "artifact.region.relativePath"),
					                             "artifact.region.relativePath"),
					    .byteOffset = static_cast<std::size_t>(
					        AsUInt(Member(r, "byteOffset", "artifact.region.byteOffset"), "artifact.region.byteOffset")),
					    .byteSize = static_cast<std::size_t>(
					        AsUInt(Member(r, "byteSize", "artifact.region.byteSize"), "artifact.region.byteSize")),
					    .checksum = AsUInt(Member(r, "checksum", "artifact.region.checksum"),
					                       "artifact.region.checksum"),
					});
				}
				for (const auto tensorItem :
				     AsArray(Member(a, "externalTensors", "artifact.externalTensors"), "artifact.externalTensors"))
				{
					artifact.externalTensors.push_back(ParseTensorRef(tensorItem, "artifact.externalTensors"));
				}
				manifest.artifacts.push_back(std::move(artifact));
			}
			return manifest;
		}
	} // namespace

	void SaveVNextModelPackage(const ExecutableModule& module, const std::filesystem::path& path,
	                           std::vector<VNextArtifactRef> artifacts,
	                           VNextPackageLayout layout)
	{
		ValidateExecutablePlan(module.plan);
		auto manifest = BuildVNextPackageManifest(module, std::move(artifacts), std::move(layout));
		ValidateVNextPackageManifest(manifest);

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
		PlanJson(out, module.plan);
		out << "}\n";
		if (!out)
		{
			throw std::runtime_error("Failed to write LiteNN vNext model package: " + path.string());
		}
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
		ValidateVNextPackageManifest(package.manifest);
		ValidateExecutablePlan(package.plan);
		return package;
	}
} // namespace LiteNN::Serialization
