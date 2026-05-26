#include <LiteNN/Serialization/TorchManifest.h>

#include <LiteNN/Layer/Activation.h>
#include <LiteNN/Layer/BatchMatMul.h>
#include <LiteNN/Layer/Conv2D.h>
#include <LiteNN/Layer/ConvTranspose2D.h>
#include <LiteNN/Layer/Gather.h>
#include <LiteNN/Layer/GroupNorm.h>
#include <LiteNN/Layer/LayerUtils.h>
#include <LiteNN/Layer/Normalization.h>
#include <LiteNN/Layer/Pad.h>
#include <LiteNN/Layer/Permute.h>
#include <LiteNN/Layer/Reshape.h>
#include <LiteNN/Layer/Softmax.h>
#include <LiteNN/Layer/TimestepEmbedding.h>
#include <LiteNN/Layer/Upsample.h>
#include <LiteNN/Validation/GraphValidator.h>

#include <simdjson.h>

#include <algorithm>
#include <array>
#include <cctype>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <filesystem>
#include <format>
#include <fstream>
#include <iterator>
#include <limits>
#include <map>
#include <optional>
#include <set>
#include <stdexcept>
#include <string>
#include <string_view>
#include <utility>
#include <vector>

namespace LiteNN::Serialization
{
	namespace
	{
		constexpr std::string_view kManifestFormat = "litenn.torch_manifest.v1";

		struct VariableBinding
		{
			std::size_t index{};
			DataType dtype{};
			std::vector<std::size_t> shape;
			std::string source;
		};

		struct ImportContext
		{
			Graph graph;
			Subgraph subgraph;
			TorchManifestReport report;
			std::map<std::string, NodeOutput, std::less<>> values;
			std::map<std::string, VariableBinding, std::less<>> variables;
			std::set<std::string, std::less<>> consumedSources;
			std::vector<std::string> inputNames;
		};

		std::runtime_error JsonError(std::string_view label, simdjson::error_code error)
		{
			return std::runtime_error(std::string("Torch manifest JSON ") + std::string(label) + ": " +
			                          simdjson::error_message(error));
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

		bool RequireBool(simdjson::dom::element value, std::string_view label)
		{
			bool result{};
			if (const auto error = value.get_bool().get(result))
			{
				throw JsonError(std::string(label) + " must be a bool", error);
			}
			return result;
		}

		std::uint64_t RequireUInt(simdjson::dom::element value, std::string_view label)
		{
			std::uint64_t result{};
			if (const auto error = value.get_uint64().get(result))
			{
				throw JsonError(std::string(label) + " must be an unsigned integer", error);
			}
			return result;
		}

		double RequireDouble(simdjson::dom::element value, std::string_view label)
		{
			double result{};
			if (const auto error = value.get_double().get(result))
			{
				throw JsonError(std::string(label) + " must be a number", error);
			}
			return result;
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

		simdjson::dom::element RequireMember(simdjson::dom::object object, std::string_view key,
		                                     std::string_view label)
		{
			if (auto member = FindMember(object, key))
			{
				return *member;
			}
			throw std::runtime_error(std::format("Torch manifest {} is missing required field '{}'", label, key));
		}

		std::optional<std::string> FindString(simdjson::dom::object object, std::string_view key,
		                                      std::string_view label)
		{
			if (auto member = FindMember(object, key))
			{
				return std::string(RequireString(*member, label));
			}
			return std::nullopt;
		}

		bool FindBool(simdjson::dom::object object, std::string_view key, bool fallback, std::string_view label)
		{
			if (auto member = FindMember(object, key))
			{
				return RequireBool(*member, label);
			}
			return fallback;
		}

		double FindDouble(simdjson::dom::object object, std::string_view key, double fallback,
		                  std::string_view label)
		{
			if (auto member = FindMember(object, key))
			{
				return RequireDouble(*member, label);
			}
			return fallback;
		}

		std::size_t CheckedToSize(std::uint64_t value, std::string_view label)
		{
			if (value > static_cast<std::uint64_t>(std::numeric_limits<std::size_t>::max()))
			{
				throw std::runtime_error(std::string("Torch manifest ") + std::string(label) + " is too large");
			}
			return static_cast<std::size_t>(value);
		}

		std::vector<std::size_t> ParseShape(simdjson::dom::element value, std::string_view label)
		{
			std::vector<std::size_t> shape;
			for (auto dimValue : RequireArray(value, label))
			{
				const auto dim = CheckedToSize(RequireUInt(dimValue, label), label);
				if (dim == 0)
				{
					throw std::runtime_error(std::string("Torch manifest ") + std::string(label) +
					                         " contains a zero dimension unsupported by LiteNN");
				}
				shape.push_back(dim);
			}
			return shape;
		}

		std::vector<std::size_t> ParseSizeList(simdjson::dom::element value, std::string_view label,
		                                       bool allowZero)
		{
			std::vector<std::size_t> values;
			for (auto dimValue : RequireArray(value, label))
			{
				const auto dim = CheckedToSize(RequireUInt(dimValue, label), label);
				if (!allowZero && dim == 0)
				{
					throw std::runtime_error(std::string("Torch manifest ") + std::string(label) +
					                         " contains a zero value");
				}
				values.push_back(dim);
			}
			return values;
		}

		std::optional<std::vector<std::size_t>> FindSizeList(simdjson::dom::object object, std::string_view key,
		                                                     std::string_view label, bool allowZero = false)
		{
			if (auto member = FindMember(object, key))
			{
				return ParseSizeList(*member, label, allowZero);
			}
			return std::nullopt;
		}

		std::optional<simdjson::dom::object> FindObject(simdjson::dom::object object, std::string_view key,
		                                                std::string_view label)
		{
			if (auto member = FindMember(object, key))
			{
				return RequireObject(*member, label);
			}
			return std::nullopt;
		}

		simdjson::dom::object RequireObjectMember(simdjson::dom::object object, std::string_view key,
		                                          std::string_view label)
		{
			return RequireObject(RequireMember(object, key, label), label);
		}

		std::vector<std::size_t> FindSizeListOr(simdjson::dom::object object, std::string_view key,
		                                        std::vector<std::size_t> fallback, std::string_view label,
		                                        bool allowZero = false)
		{
			if (auto value = FindSizeList(object, key, label, allowZero))
			{
				return *value;
			}
			return fallback;
		}

		std::size_t FindSizeOr(simdjson::dom::object object, std::string_view key, std::size_t fallback,
		                       std::string_view label)
		{
			if (auto member = FindMember(object, key))
			{
				return CheckedToSize(RequireUInt(*member, label), label);
			}
			return fallback;
		}

		std::optional<std::vector<std::size_t>> FindShape(simdjson::dom::object object, std::string_view key,
		                                                  std::string_view label)
		{
			if (auto member = FindMember(object, key))
			{
				return ParseShape(*member, label);
			}
			return std::nullopt;
		}

		std::string ShapeToString(ShapeView shape)
		{
			std::string result = "[";
			for (std::size_t i = 0; i < shape.NumDim(); ++i)
			{
				if (i != 0)
				{
					result += ", ";
				}
				result += std::to_string(shape[i]);
			}
			result += "]";
			return result;
		}

		std::string NormalizeToken(std::string_view value)
		{
			std::string result;
			result.reserve(value.size());
			for (const char c : value)
			{
				if (c == '-' || c == '_' || c == '.' || c == ' ')
				{
					continue;
				}
				result.push_back(static_cast<char>(std::tolower(static_cast<unsigned char>(c))));
			}
			return result;
		}

		std::vector<std::byte> ReadAllBytes(const std::filesystem::path& path)
		{
			std::ifstream in(path, std::ios::binary);
			if (!in)
			{
				throw std::runtime_error("Failed to open Torch manifest file for reading");
			}
			const std::vector<char> chars{ std::istreambuf_iterator<char>(in), std::istreambuf_iterator<char>() };
			std::vector<std::byte> bytes(chars.size());
			std::memcpy(bytes.data(), chars.data(), chars.size());
			return bytes;
		}

		template <BinaryOp Op>
		OutputInfo InferBinaryInfo(const OutputInfo& lhs, const OutputInfo& rhs, std::string_view opName)
		{
			auto dtype = BinaryOpTraits<Op>::ResultType(lhs.dtype, rhs.dtype);
			if (!dtype)
			{
				throw std::runtime_error(std::format("Torch manifest {} dtype inference failed: {}", opName,
				                                     dtype.error()));
			}
			auto shape = BinaryOpTraits<Op>::ResultShape(lhs.shape, rhs.shape);
			if (!shape)
			{
				throw std::runtime_error(std::format("Torch manifest {} shape inference failed: {}", opName,
				                                     shape.error()));
			}
			return { *dtype, *shape };
		}

		OutputInfo InferBinaryInfo(BinaryOp op, const OutputInfo& lhs, const OutputInfo& rhs,
		                           std::string_view opName)
		{
			switch (op)
			{
			case BinaryOp::Add:
				return InferBinaryInfo<BinaryOp::Add>(lhs, rhs, opName);
			case BinaryOp::Subtract:
				return InferBinaryInfo<BinaryOp::Subtract>(lhs, rhs, opName);
			case BinaryOp::Multiply:
				return InferBinaryInfo<BinaryOp::Multiply>(lhs, rhs, opName);
			case BinaryOp::Divide:
				return InferBinaryInfo<BinaryOp::Divide>(lhs, rhs, opName);
			case BinaryOp::MatMul:
				return InferBinaryInfo<BinaryOp::MatMul>(lhs, rhs, opName);
			default:
				throw std::runtime_error("Torch manifest internal error: unsupported binary inference op");
			}
		}

		NodeOutput RequireValue(const ImportContext& context, std::string_view name, std::string_view nodeName)
		{
			if (const auto it = context.values.find(name); it != context.values.end())
			{
				return it->second;
			}
			throw std::runtime_error(std::format("Torch manifest node '{}' references unknown value '{}'", nodeName,
			                                     name));
		}

		const VariableBinding& RequireVariable(const ImportContext& context, std::string_view name,
		                                       std::string_view nodeName)
		{
			if (const auto it = context.variables.find(name); it != context.variables.end())
			{
				return it->second;
			}
			throw std::runtime_error(std::format("Torch manifest node '{}' references unknown tensor '{}'",
			                                     nodeName, name));
		}

		NodeOutput AddVariableRef(ImportContext& context, std::string_view variableName,
		                          std::string_view nodeName)
		{
			const auto& variable = RequireVariable(context, variableName, nodeName);
			const auto id = context.subgraph.AddNode(VariableRefNode{ variable.index },
			                                         { OutputInfo{ variable.dtype, variable.shape } });
			return { id, 0 };
		}

		void BindValue(ImportContext& context, std::string name, NodeOutput output, std::string_view nodeName)
		{
			if (!context.values.emplace(name, output).second)
			{
				throw std::runtime_error(std::format("Torch manifest node '{}' produces duplicate value '{}'",
				                                     nodeName, name));
			}
		}

		bool SameShape(ShapeView lhs, ShapeView rhs)
		{
			return lhs == rhs;
		}

		Tensor<CPU> ApplyTensorLayout(Tensor<CPU> tensor, std::string_view layout,
		                              std::string_view manifestName, TorchManifestReport& report)
		{
			const auto normalized = NormalizeToken(layout);
			if (normalized.empty() || normalized == "none" || normalized == "identity" ||
			    normalized == "torchembeddingweight" || normalized == "torchconv2dweight" ||
			    normalized == "torchconvtranspose2dweight" || normalized == "torchdeconv2dweight" ||
			    normalized == "torchconvweight")
			{
				return tensor;
			}

			if (normalized == "transpose2d" || normalized == "torchlinearweight" ||
			    normalized == "torchattentionprojectionweight" || normalized == "peftloraaweight" ||
			    normalized == "peftlorabweight")
			{
				if (tensor.Shape().NumDim() != 2)
				{
					throw std::runtime_error(std::format(
					    "Torch manifest tensor '{}' layout '{}' expects rank-2 source tensor, got {}",
					    manifestName, layout, ShapeToString(tensor.Shape())));
				}
				report.foldedConstants.push_back(
				    std::format("tensor {}: materialized {} via 2D transpose", manifestName, layout));
				return tensor.Transpose();
			}

			if (normalized == "torchbias1d" || normalized == "torchnormweight" ||
			    normalized == "torchnormbias" || normalized == "torchlayernormweight" ||
			    normalized == "torchlayernormbias" || normalized == "torchrmsnormweight")
			{
				if (tensor.Shape().NumDim() != 1)
				{
					throw std::runtime_error(std::format(
					    "Torch manifest tensor '{}' layout '{}' expects rank-1 source tensor, got {}",
					    manifestName, layout, ShapeToString(tensor.Shape())));
				}
				tensor.Reshape({ 1uz, tensor.Shape()[0] });
				report.foldedConstants.push_back(
				    std::format("tensor {}: materialized {} via [1, features] reshape", manifestName, layout));
				return tensor;
			}

			if (normalized == "torchchannelweight" || normalized == "torchchannelbias" ||
			    normalized == "torchgroupnormweight" || normalized == "torchgroupnormbias" ||
			    normalized == "torchconvbias1d")
			{
				if (tensor.Shape().NumDim() != 1)
				{
					throw std::runtime_error(std::format(
					    "Torch manifest tensor '{}' layout '{}' expects rank-1 source tensor, got {}",
					    manifestName, layout, ShapeToString(tensor.Shape())));
				}
				const auto channels = tensor.Shape()[0];
				tensor.Reshape({ 1uz, channels, 1uz, 1uz });
				report.foldedConstants.push_back(
				    std::format("tensor {}: materialized {} via [1, channels, 1, 1] reshape",
				                manifestName, layout));
				return tensor;
			}

			throw std::runtime_error(std::format("Torch manifest tensor '{}' uses unsupported layout preset '{}'",
			                                     manifestName, layout));
		}

		Tensor<CPU> CastManifestTensor(Tensor<CPU> tensor, DataType targetType,
		                               std::string_view manifestName, TorchManifestReport& report)
		{
			if (tensor.DType() == targetType)
			{
				return tensor;
			}

			CPU device;
			Tensor<CPU> converted(Uninitialized, tensor.Shape(), targetType, device);
			DeviceTraits<CPU>::ConvertTo(device, tensor.DType(), tensor.RawData(), tensor.NumElements(), targetType,
			                             converted.RawData());
			report.foldedConstants.push_back(std::format("tensor {}: converted {} -> {}", manifestName,
			                                             DataTypeName(tensor.DType()), DataTypeName(targetType)));
			return converted;
		}

		void ImportManifestTensors(ImportContext& context, simdjson::dom::object rootObject,
		                           const SafetensorsArchive& archive, const TorchManifestImportOptions& options)
		{
			const auto tensorsMember = FindMember(rootObject, "tensors");
			if (!tensorsMember)
			{
				return;
			}

			for (auto tensorValue : RequireArray(*tensorsMember, "tensors"))
			{
				const auto object = RequireObject(tensorValue, "tensor entry");
				const auto name = std::string(RequireString(RequireMember(object, "name", "tensor"), "tensor name"));
				const auto source = FindString(object, "source", "tensor source").value_or(name);
				auto layout = FindString(object, "layout", "tensor layout");
				if (!layout)
				{
					layout = FindBool(object, "transpose", false, "tensor transpose") ? std::string("transpose2d")
					                                                                  : std::string("identity");
				}

				if (context.variables.contains(name))
				{
					throw std::runtime_error("Torch manifest defines duplicate tensor name: " + name);
				}

				const auto* tensorInfo = archive.FindTensor(source);
				if (tensorInfo == nullptr)
				{
					throw std::runtime_error(std::format(
					    "Torch manifest tensor '{}' source '{}' was not found in safetensors archive", name, source));
				}
				context.consumedSources.insert(source);

				if (auto dtypeText = FindString(object, "dtype", "tensor dtype"))
				{
					const auto expected = MapTorchManifestDataType(*dtypeText);
					if (tensorInfo->dtype != expected)
					{
						throw std::runtime_error(std::format(
						    "Torch manifest tensor '{}' dtype mismatch: expected {}, got {}", name,
						    DataTypeName(expected), DataTypeName(tensorInfo->dtype)));
					}
				}
				if (auto sourceShape = FindShape(object, "source_shape", "tensor source_shape"))
				{
					if (!SameShape(tensorInfo->shape, *sourceShape))
					{
						throw std::runtime_error(std::format(
						    "Torch manifest tensor '{}' source shape mismatch: expected {}, got {}", name,
						    ShapeToString(*sourceShape), ShapeToString(tensorInfo->shape)));
					}
				}

				auto tensor = ApplyTensorLayout(archive.TensorAsCPU(*tensorInfo), *layout, name, context.report);
				if (auto shape = FindShape(object, "shape", "tensor shape"))
				{
					if (!SameShape(tensor.Shape(), *shape))
					{
						throw std::runtime_error(std::format(
						    "Torch manifest tensor '{}' shape mismatch after layout '{}': expected {}, got {}",
						    name, *layout, ShapeToString(*shape), ShapeToString(tensor.Shape())));
					}
				}
				std::optional<Tensor<CPU>> convertedTensor;
				const Tensor<CPU>* finalTensor = &tensor;
				if (auto targetDTypeText = FindString(object, "target_dtype", "tensor target_dtype"))
				{
					convertedTensor.emplace(CastManifestTensor(std::move(tensor), MapTorchManifestDataType(*targetDTypeText),
					                                           name, context.report));
					finalTensor = &*convertedTensor;
				}

				const auto dtype = finalTensor->DType();
				auto shape = finalTensor->Shape().ToOwned();
				auto variableTensor = convertedTensor ? std::move(*convertedTensor) : std::move(tensor);
				auto variable = options.trainableVariables ? Variable::Create(std::move(variableTensor))
				                                           : Variable::CreateFrozen(std::move(variableTensor));
				const auto index = context.graph.AddVariable(std::move(variable));
				context.graph.SetVariableName(index, name);
				context.variables.emplace(name, VariableBinding{ index, dtype, std::move(shape), source });
				context.report.importedTensors.push_back(
				    std::format("{} <- {} ({})", name, source, *layout));
			}

			if (options.failOnUnusedWeights)
			{
				for (const auto& tensor : archive.Tensors())
				{
					if (!context.consumedSources.contains(tensor.name))
					{
						throw std::runtime_error("Safetensors archive contains extra tensor not consumed by manifest: " +
						                         tensor.name);
					}
				}
			}
		}

		void ImportManifestInputs(ImportContext& context, simdjson::dom::object rootObject)
		{
			for (auto inputValue : RequireArray(RequireMember(rootObject, "inputs", "root"), "inputs"))
			{
				const auto object = RequireObject(inputValue, "input entry");
				const auto name = std::string(RequireString(RequireMember(object, "name", "input"), "input name"));
				if (context.values.contains(name))
				{
					throw std::runtime_error("Torch manifest defines duplicate input name: " + name);
				}
				const auto dtype =
				    MapTorchManifestDataType(RequireString(RequireMember(object, "dtype", "input"), "input dtype"));
				auto shape = ParseShape(RequireMember(object, "shape", "input"), "input shape");
				const auto id = context.subgraph.AddParam(dtype, shape);
				context.values.emplace(name, NodeOutput{ id, 0 });
				context.inputNames.push_back(name);
			}
		}

		NodeOutput AddBinary(ImportContext& context, BinaryOp op, NodeOutput lhs, NodeOutput rhs,
		                     std::string_view nodeName, std::string_view opName)
		{
			const auto lhsInfo = context.subgraph.GetOutputInfo(lhs);
			const auto rhsInfo = context.subgraph.GetOutputInfo(rhs);
			const auto outputInfo = InferBinaryInfo(op, lhsInfo, rhsInfo, opName);
			const auto id = context.subgraph.AddNode(BinaryOpNode{ op, lhs, rhs }, { outputInfo });
			return { id, 0 };
		}

		NodeOutput AddScalarBinary(ImportContext& context, BinaryOp op, NodeOutput input, double value,
		                           std::string_view nodeName, std::string_view opName)
		{
			const auto info = context.subgraph.GetOutputInfo(input);
			const auto scalar = Layer::Detail::AddConstant(context.subgraph,
			                                               Layer::Detail::MakeScalarTensor(info.dtype, value));
			return AddBinary(context, op, input, { scalar, 0 }, nodeName, opName);
		}

		NodeOutput AddReshapeChecked(ImportContext& context, NodeOutput input, std::vector<std::size_t> targetShape,
		                             std::string_view nodeName)
		{
			const auto info = context.subgraph.GetOutputInfo(input);
			const auto elementCount = [](std::span<const std::size_t> shape) {
				std::size_t count = 1;
				for (const auto dim : shape)
				{
					count *= dim;
				}
				return count;
			};
			if (elementCount(info.shape) != elementCount(targetShape))
			{
				throw std::runtime_error(std::format("Torch manifest node '{}' reshape element count mismatch: {} -> {}",
				                                     nodeName, ShapeToString(info.shape),
				                                     ShapeToString(targetShape)));
			}
			auto outputShape = targetShape;
			const auto id = context.subgraph.AddNode(ReshapeNode{ input, std::move(targetShape) },
			                                         { OutputInfo{ info.dtype, std::move(outputShape) } });
			return { id, 0 };
		}

		NodeOutput AddActivationByName(ImportContext& context, NodeOutput input, std::string_view activation,
		                               std::string_view nodeName)
		{
			const auto normalized = NormalizeToken(activation);
			if (normalized.empty() || normalized == "identity" || normalized == "none")
			{
				return input;
			}
			if (normalized == "relu")
			{
				return Layer::AddReLU(context.subgraph, input);
			}
			if (normalized == "gelu")
			{
				return Layer::AddGELU(context.subgraph, input);
			}
			if (normalized == "geluerf")
			{
				return Layer::AddGELUErf(context.subgraph, input);
			}
			if (normalized == "silu" || normalized == "swish")
			{
				return Layer::AddSiLU(context.subgraph, input);
			}
			if (normalized == "sigmoid")
			{
				return Layer::AddSigmoid(context.subgraph, input);
			}
			if (normalized == "tanh")
			{
				return Layer::AddTanh(context.subgraph, input);
			}
			throw std::runtime_error(std::format("Torch manifest node '{}' uses unsupported activation '{}'",
			                                     nodeName, activation));
		}

		NodeOutput AddLinearSpec(ImportContext& context, NodeOutput input, simdjson::dom::object spec,
		                         std::string_view nodeName, std::string_view label)
		{
			const auto weightName = RequireString(RequireMember(spec, "weight", label), label);
			const auto weight = AddVariableRef(context, weightName, nodeName);
			auto output = AddBinary(context, BinaryOp::MatMul, input, weight, nodeName, "linear matmul");
			if (auto biasName = FindString(spec, "bias", label))
			{
				const auto bias = AddVariableRef(context, *biasName, nodeName);
				output = AddBinary(context, BinaryOp::Add, output, bias, nodeName, "linear bias add");
			}
			return output;
		}

		NodeOutput ImportLinear(ImportContext& context, simdjson::dom::object object, std::string_view nodeName)
		{
			const auto inputName = RequireString(RequireMember(object, "input", nodeName), "linear input");
			const auto input = RequireValue(context, inputName, nodeName);
			auto output = AddLinearSpec(context, input, object, nodeName, "linear");
			context.report.loweredOps.push_back(std::format("{}: linear -> MatMul(+Add)", nodeName));
			return output;
		}

		NodeOutput ImportEmbedding(ImportContext& context, simdjson::dom::object object, std::string_view nodeName)
		{
			auto inputName = FindString(object, "indices", "embedding indices");
			if (!inputName)
			{
				inputName = std::string(RequireString(RequireMember(object, "input", nodeName), "embedding input"));
			}
			const auto weightName = RequireString(RequireMember(object, "weight", nodeName), "embedding weight");
			const auto indices = RequireValue(context, *inputName, nodeName);
			const auto weight = AddVariableRef(context, weightName, nodeName);
			auto output = Layer::AddGather(context.subgraph, weight, indices, 0);
			context.report.loweredOps.push_back(std::format("{}: embedding -> Gather(axis=0)", nodeName));
			return output;
		}

		std::vector<std::size_t> FindSpatialList(simdjson::dom::object object, std::initializer_list<std::string_view> keys,
		                                         std::vector<std::size_t> fallback, std::string_view label,
		                                         bool allowZero)
		{
			for (const auto key : keys)
			{
				if (auto value = FindSizeList(object, key, label, allowZero))
				{
					return *value;
				}
			}
			return fallback;
		}

		std::pair<std::vector<std::size_t>, std::vector<std::size_t>>
		FindLowHighPads(simdjson::dom::object object, std::string_view opName)
		{
			auto lowPads = FindSizeList(object, "low_pads", "low_pads", true);
			auto highPads = FindSizeList(object, "high_pads", "high_pads", true);
			if (lowPads || highPads)
			{
				if (!lowPads || !highPads)
				{
					throw std::runtime_error(std::format("Torch manifest {} requires both low_pads and high_pads",
					                                     opName));
				}
				return { *lowPads, *highPads };
			}
			auto symmetric = FindSpatialList(object, { "padding", "pads" }, { 0uz, 0uz }, "padding", true);
			return { symmetric, std::move(symmetric) };
		}

		PadMode ParsePadMode(std::string_view text)
		{
			const auto normalized = NormalizeToken(text);
			if (normalized.empty() || normalized == "constant")
			{
				return PadMode::Constant;
			}
			if (normalized == "reflect")
			{
				return PadMode::Reflect;
			}
			if (normalized == "replicate" || normalized == "edge")
			{
				return PadMode::Replicate;
			}
			throw std::runtime_error("Torch manifest unsupported pad mode: " + std::string(text));
		}

		UpsampleMode ParseUpsampleMode(std::string_view text)
		{
			const auto normalized = NormalizeToken(text);
			if (normalized.empty() || normalized == "nearest")
			{
				return UpsampleMode::Nearest;
			}
			if (normalized == "bilinear" || normalized == "linear")
			{
				return UpsampleMode::Bilinear;
			}
			if (normalized == "bicubic" || normalized == "cubic")
			{
				return UpsampleMode::Bicubic;
			}
			throw std::runtime_error("Torch manifest unsupported upsample mode: " + std::string(text));
		}

		std::optional<NodeOutput> FindVariableRef(ImportContext& context, simdjson::dom::object object,
		                                           std::initializer_list<std::string_view> keys,
		                                           std::string_view nodeName, std::string_view label)
		{
			for (const auto key : keys)
			{
				if (auto name = FindString(object, key, label))
				{
					return AddVariableRef(context, *name, nodeName);
				}
			}
			return std::nullopt;
		}

		NodeOutput AddConv2DSpec(ImportContext& context, NodeOutput input, simdjson::dom::object spec,
		                         std::string_view nodeName, std::string_view label)
		{
			const auto weightName = RequireString(RequireMember(spec, "weight", label), label);
			const auto weight = AddVariableRef(context, weightName, nodeName);
			auto bias = FindVariableRef(context, spec, { "bias" }, nodeName, label);
			auto [lowPads, highPads] = FindLowHighPads(spec, label);
			const auto groups = FindSizeOr(spec, "groups", FindSizeOr(spec, "group", 1, label), label);
			return Layer::AddConv2D(
			    context.subgraph, input, weight, bias,
			    FindSpatialList(spec, { "strides", "stride" }, { 1uz, 1uz }, label, false),
			    FindSpatialList(spec, { "dilations", "dilation" }, { 1uz, 1uz }, label, false),
			    std::move(lowPads), std::move(highPads), groups);
		}

		NodeOutput AddConvTranspose2DSpec(ImportContext& context, NodeOutput input, simdjson::dom::object spec,
		                                  std::string_view nodeName, std::string_view label)
		{
			const auto weightName = RequireString(RequireMember(spec, "weight", label), label);
			const auto weight = AddVariableRef(context, weightName, nodeName);
			auto bias = FindVariableRef(context, spec, { "bias" }, nodeName, label);
			auto [lowPads, highPads] = FindLowHighPads(spec, label);
			const auto groups = FindSizeOr(spec, "groups", FindSizeOr(spec, "group", 1, label), label);
			auto outputPads = FindSpatialList(spec, { "output_padding", "output_pads", "outputPads" },
			                                  { 0uz, 0uz }, label, true);
			return Layer::AddConvTranspose2D(
			    context.subgraph, input, weight, bias,
			    FindSpatialList(spec, { "strides", "stride" }, { 1uz, 1uz }, label, false),
			    FindSpatialList(spec, { "dilations", "dilation" }, { 1uz, 1uz }, label, false),
			    std::move(lowPads), std::move(highPads), std::move(outputPads), groups);
		}

		NodeOutput ImportConv2D(ImportContext& context, simdjson::dom::object object, std::string_view nodeName)
		{
			const auto inputName = RequireString(RequireMember(object, "input", nodeName), "conv2d input");
			const auto input = RequireValue(context, inputName, nodeName);
			const auto output = AddConv2DSpec(context, input, object, nodeName, "conv2d");
			context.report.loweredOps.push_back(std::format("{}: conv2d -> Conv2DNode", nodeName));
			return output;
		}

		NodeOutput ImportConvTranspose2D(ImportContext& context, simdjson::dom::object object,
		                                 std::string_view nodeName)
		{
			const auto inputName =
			    RequireString(RequireMember(object, "input", nodeName), "conv_transpose2d input");
			const auto input = RequireValue(context, inputName, nodeName);
			const auto output = AddConvTranspose2DSpec(context, input, object, nodeName, "conv_transpose2d");
			context.report.loweredOps.push_back(std::format("{}: conv_transpose2d -> ConvTranspose2DNode", nodeName));
			return output;
		}

		NodeOutput ImportNormalization(ImportContext& context, simdjson::dom::object object,
		                               std::string_view nodeName, NormalizationMode mode)
		{
			const auto inputName = RequireString(RequireMember(object, "input", nodeName), "normalization input");
			const auto input = RequireValue(context, inputName, nodeName);
			const auto inputInfo = context.subgraph.GetOutputInfo(input);
			if (inputInfo.shape.empty())
			{
				throw std::runtime_error("Torch manifest normalization input must not be scalar");
			}
			const auto axisMember = FindMember(object, "axis");
			const auto axis = CheckedToSize(axisMember ? RequireUInt(*axisMember, "normalization axis")
			                                           : static_cast<std::uint64_t>(inputInfo.shape.size() - 1),
			                                "normalization axis");
			const auto eps = FindDouble(object, "eps", FindDouble(object, "epsilon", 1e-5, "normalization epsilon"),
			                            "normalization eps");

			std::optional<NodeOutput> scale;
			if (auto weight = FindString(object, "weight", "normalization weight"))
			{
				scale = AddVariableRef(context, *weight, nodeName);
			}
			else if (auto gamma = FindString(object, "gamma", "normalization gamma"))
			{
				scale = AddVariableRef(context, *gamma, nodeName);
			}

			std::optional<NodeOutput> bias;
			if (auto beta = FindString(object, "bias", "normalization bias"))
			{
				bias = AddVariableRef(context, *beta, nodeName);
			}
			else if (auto beta = FindString(object, "beta", "normalization beta"))
			{
				bias = AddVariableRef(context, *beta, nodeName);
			}

			const auto output = Layer::AddNormalization(context.subgraph, input, mode, axis, eps, scale, bias);
			context.report.loweredOps.push_back(std::format(
			    "{}: {} -> NormalizationNode", nodeName,
			    mode == NormalizationMode::LayerNorm ? "layer_norm" : "rms_norm"));
			return output;
		}

		NodeOutput AddPyTorchGroupNorm(ImportContext& context, NodeOutput input, std::size_t groups, double eps,
		                               std::optional<NodeOutput> scale, std::optional<NodeOutput> bias,
		                               std::string_view nodeName)
		{
			const auto inputInfo = context.subgraph.GetOutputInfo(input);
			if (inputInfo.shape.size() < 2 || inputInfo.shape.size() > 4)
			{
				throw std::runtime_error(std::format(
				    "Torch manifest node '{}' PyTorch group_norm expects rank [N, C, ...] with rank 2-4",
				    nodeName));
			}
			const auto batch = inputInfo.shape[0];
			const auto channels = inputInfo.shape[1];
			if (groups == 0 || channels % groups != 0)
			{
				throw std::runtime_error(std::format(
				    "Torch manifest node '{}' group_norm requires channels divisible by num_groups", nodeName));
			}

			std::size_t groupSize = channels / groups;
			for (auto dim = 2uz; dim < inputInfo.shape.size(); ++dim)
			{
				groupSize *= inputInfo.shape[dim];
			}

			auto normalized = AddReshapeChecked(context, input, { batch, groups, groupSize }, nodeName);
			normalized = Layer::AddNormalization(context.subgraph, normalized, NormalizationMode::LayerNorm, 2,
			                                     eps, std::nullopt, std::nullopt);
			normalized = AddReshapeChecked(context, normalized, inputInfo.shape, nodeName);
			if (scale)
			{
				normalized = AddBinary(context, BinaryOp::Multiply, normalized, *scale, nodeName,
				                       "group_norm affine scale");
			}
			if (bias)
			{
				normalized = AddBinary(context, BinaryOp::Add, normalized, *bias, nodeName,
				                       "group_norm affine bias");
			}
			return normalized;
		}

		NodeOutput AddGroupNormSpec(ImportContext& context, NodeOutput input, simdjson::dom::object spec,
		                            std::string_view nodeName, std::string_view label)
		{
			const auto groups = FindSizeOr(spec, "num_groups", FindSizeOr(spec, "groups", 1, label), label);
			const auto eps = FindDouble(spec, "eps", FindDouble(spec, "epsilon", 1e-5, label), label);
			auto scale = FindVariableRef(context, spec, { "weight", "gamma", "scale" }, nodeName, label);
			auto bias = FindVariableRef(context, spec, { "bias", "beta" }, nodeName, label);
			const auto layout = NormalizeToken(FindString(spec, "layout", label).value_or("nchw"));
			if (layout == "litenn" || layout == "ggml" || layout == "native")
			{
				return Layer::AddNormalization(context.subgraph, input, NormalizationMode::GroupNorm, 0,
				                               eps, scale, bias, groups);
			}
			if (layout == "nchw" || layout == "torch" || layout == "pytorch" || layout == "channelfirst")
			{
				return AddPyTorchGroupNorm(context, input, groups, eps, scale, bias, nodeName);
			}
			throw std::runtime_error(std::format("Torch manifest node '{}' unsupported group_norm layout '{}'",
			                                     nodeName, layout));
		}

		NodeOutput ImportGroupNorm(ImportContext& context, simdjson::dom::object object,
		                           std::string_view nodeName)
		{
			const auto inputName = RequireString(RequireMember(object, "input", nodeName), "group_norm input");
			const auto input = RequireValue(context, inputName, nodeName);
			const auto output = AddGroupNormSpec(context, input, object, nodeName, "group_norm");
			context.report.loweredOps.push_back(
			    std::format("{}: group_norm -> PyTorch-compatible reshape+LayerNorm affine", nodeName));
			return output;
		}

		NodeOutput ImportTimestepEmbedding(ImportContext& context, simdjson::dom::object object,
		                                   std::string_view nodeName)
		{
			auto inputName = FindString(object, "timesteps", "timestep_embedding timesteps");
			if (!inputName)
			{
				inputName = std::string(RequireString(RequireMember(object, "input", nodeName),
				                                      "timestep_embedding input"));
			}
			const auto input = RequireValue(context, *inputName, nodeName);
			const auto dim = CheckedToSize(RequireUInt(RequireMember(object, "dim", nodeName),
			                                           "timestep_embedding dim"),
			                               "timestep_embedding dim");
			const auto maxPeriod = FindSizeOr(object, "max_period",
			                                  FindSizeOr(object, "maxPeriod", 10000, "timestep_embedding maxPeriod"),
			                                  "timestep_embedding max_period");
			const auto output = Layer::AddTimestepEmbedding(context.subgraph, input, dim, maxPeriod);
			context.report.loweredOps.push_back(std::format("{}: timestep_embedding -> TimestepEmbeddingNode", nodeName));
			return output;
		}

		NodeOutput ImportUpsample(ImportContext& context, simdjson::dom::object object,
		                          std::string_view nodeName)
		{
			const auto inputName = RequireString(RequireMember(object, "input", nodeName), "upsample input");
			const auto input = RequireValue(context, inputName, nodeName);
			const auto mode = ParseUpsampleMode(FindString(object, "mode", "upsample mode").value_or("nearest"));
			auto outputSpatial = FindSpatialList(object,
			                                     { "output_spatial_shape", "output_size", "size", "spatial_shape" },
			                                     {}, "upsample output spatial shape", false);
			if (outputSpatial.empty())
			{
				throw std::runtime_error(std::format("Torch manifest node '{}' requires output_spatial_shape/size",
				                                     nodeName));
			}
			const auto alignCorners = FindBool(object, "align_corners", false, "upsample align_corners");
			const auto output = Layer::AddUpsample(context.subgraph, input, mode, std::move(outputSpatial),
			                                       alignCorners);
			context.report.loweredOps.push_back(std::format("{}: upsample -> UpsampleNode", nodeName));
			return output;
		}

		NodeOutput ImportPad(ImportContext& context, simdjson::dom::object object, std::string_view nodeName)
		{
			const auto inputName = RequireString(RequireMember(object, "input", nodeName), "pad input");
			const auto input = RequireValue(context, inputName, nodeName);
			const auto inputInfo = context.subgraph.GetOutputInfo(input);
			auto lowPads = FindSizeListOr(object, "low_pads", std::vector<std::size_t>(inputInfo.shape.size(), 0uz),
			                              "pad low_pads", true);
			auto highPads = FindSizeListOr(object, "high_pads", std::vector<std::size_t>(inputInfo.shape.size(), 0uz),
			                               "pad high_pads", true);
			const auto mode = ParsePadMode(FindString(object, "mode", "pad mode").value_or("constant"));
			const auto constantValue = FindDouble(object, "constant_value",
			                                      FindDouble(object, "value", 0.0, "pad value"),
			                                      "pad constant_value");
			const auto output = Layer::AddPad(context.subgraph, input, lowPads, highPads, mode, constantValue);
			context.report.loweredOps.push_back(std::format("{}: pad -> PadNode", nodeName));
			return output;
		}

		NodeOutput ImportClamp(ImportContext& context, simdjson::dom::object object, std::string_view nodeName)
		{
			const auto inputName = RequireString(RequireMember(object, "input", nodeName), "clamp input");
			const auto input = RequireValue(context, inputName, nodeName);
			const auto minValue = FindDouble(object, "min", -std::numeric_limits<double>::infinity(), "clamp min");
			const auto maxValue = FindDouble(object, "max", std::numeric_limits<double>::infinity(), "clamp max");
			if (!std::isfinite(minValue) || !std::isfinite(maxValue))
			{
				throw std::runtime_error("Torch manifest clamp requires finite min and max fields");
			}
			const auto output = Layer::AddClamp(context.subgraph, input, minValue, maxValue);
			context.report.loweredOps.push_back(std::format("{}: clamp -> Min(Max(x, min), max)", nodeName));
			return output;
		}

		NodeOutput ImportResidualBlock(ImportContext& context, simdjson::dom::object object,
		                               std::string_view nodeName)
		{
			const auto inputName = RequireString(RequireMember(object, "input", nodeName), "residual_block input");
			const auto input = RequireValue(context, inputName, nodeName);
			const auto activation = FindString(object, "activation", "residual_block activation").value_or("silu");

			auto current = AddGroupNormSpec(context, input, RequireObjectMember(object, "norm1", "residual_block norm1"),
			                                nodeName, "residual_block norm1");
			current = AddActivationByName(context, current, activation, nodeName);
			current = AddConv2DSpec(context, current, RequireObjectMember(object, "conv1", "residual_block conv1"),
			                        nodeName, "residual_block conv1");

			if (auto tembName = FindString(object, "temb", "residual_block temb"))
			{
				const auto temb = RequireValue(context, *tembName, nodeName);
				auto projected = AddLinearSpec(context, temb,
				                               RequireObjectMember(object, "temb_projection",
				                                                   "residual_block temb_projection"),
				                               nodeName, "residual_block temb_projection");
				const auto currentInfo = context.subgraph.GetOutputInfo(current);
				const auto projectedInfo = context.subgraph.GetOutputInfo(projected);
				if (currentInfo.shape.size() != 4 || projectedInfo.shape.size() != 2 ||
				    projectedInfo.shape[0] != currentInfo.shape[0] || projectedInfo.shape[1] != currentInfo.shape[1])
				{
					throw std::runtime_error(std::format(
					    "Torch manifest node '{}' residual_block temb projection must be [N, C] for NCHW feature",
					    nodeName));
				}
				projected = AddReshapeChecked(context, projected,
				                              { currentInfo.shape[0], currentInfo.shape[1], 1uz, 1uz }, nodeName);
				current = AddBinary(context, BinaryOp::Add, current, projected, nodeName,
				                    "residual_block temb add");
			}

			current = AddGroupNormSpec(context, current, RequireObjectMember(object, "norm2", "residual_block norm2"),
			                            nodeName, "residual_block norm2");
			current = AddActivationByName(context, current, activation, nodeName);
			current = AddConv2DSpec(context, current, RequireObjectMember(object, "conv2", "residual_block conv2"),
			                        nodeName, "residual_block conv2");

			auto residual = input;
			if (auto skip = FindObject(object, "skip", "residual_block skip"))
			{
				residual = AddConv2DSpec(context, input, *skip, nodeName, "residual_block skip");
			}
			const auto output = AddBinary(context, BinaryOp::Add, current, residual, nodeName,
			                              "residual_block residual add");
			context.report.loweredOps.push_back(
			    std::format("{}: residual_block -> GroupNorm/SiLU/Conv2D(+temb)+residual", nodeName));
			return output;
		}

		NodeOutput AddGEGLUFeedForwardSpec(ImportContext& context, NodeOutput input, simdjson::dom::object spec,
		                                   std::string_view nodeName, std::string_view label);

		NodeOutput AddAttentionBlockSpec(ImportContext& context, NodeOutput input, NodeOutput keyValueInput,
		                                 simdjson::dom::object spec, std::string_view nodeName,
		                                 std::string_view label);

		NodeOutput AddLayerNormSpec(ImportContext& context, NodeOutput input, simdjson::dom::object spec,
		                             std::string_view nodeName, std::string_view label)
		{
			const auto inputInfo = context.subgraph.GetOutputInfo(input);
			if (inputInfo.shape.empty())
			{
				throw std::runtime_error(std::format("Torch manifest node '{}' {} input must not be scalar",
				                                     nodeName, label));
			}
			const auto axis = FindSizeOr(spec, "axis", inputInfo.shape.size() - 1, label);
			if (axis >= inputInfo.shape.size())
			{
				throw std::runtime_error(std::format("Torch manifest node '{}' {} axis {} out of range for rank {}",
				                                     nodeName, label, axis, inputInfo.shape.size()));
			}
			const auto eps = FindDouble(spec, "eps", FindDouble(spec, "epsilon", 1e-5, label), label);
			auto scale = FindVariableRef(context, spec, { "weight", "gamma", "scale" }, nodeName, label);
			auto bias = FindVariableRef(context, spec, { "bias", "beta" }, nodeName, label);
			return Layer::AddNormalization(context.subgraph, input, NormalizationMode::LayerNorm, axis, eps,
			                               scale, bias);
		}

		NodeOutput ImportFeedForward(ImportContext& context, simdjson::dom::object object,
		                             std::string_view nodeName)
		{
			const auto inputName = RequireString(RequireMember(object, "input", nodeName), "feed_forward input");
			const auto input = RequireValue(context, inputName, nodeName);
			const auto activation = FindString(object, "activation", "feed_forward activation").value_or("gelu");

			auto hidden = AddLinearSpec(context, input, RequireObjectMember(object, "up", "feed_forward up"),
			                            nodeName, "feed_forward up");
			if (auto gateSpec = FindObject(object, "gate", "feed_forward gate"))
			{
				auto gate = AddLinearSpec(context, input, *gateSpec, nodeName, "feed_forward gate");
				gate = AddActivationByName(context, gate, activation, nodeName);
				hidden = AddBinary(context, BinaryOp::Multiply, hidden, gate, nodeName, "feed_forward gate multiply");
			}
			else
			{
				hidden = AddActivationByName(context, hidden, activation, nodeName);
			}

			auto output = AddLinearSpec(context, hidden, RequireObjectMember(object, "down", "feed_forward down"),
			                            nodeName, "feed_forward down");
			if (FindBool(object, "residual", true, "feed_forward residual"))
			{
				output = AddBinary(context, BinaryOp::Add, output, input, nodeName, "feed_forward residual add");
			}
			context.report.loweredOps.push_back(
			    std::format("{}: feed_forward -> Linear/activation/Linear(+residual)", nodeName));
			return output;
		}

		NodeOutput ImportGEGLUFeedForward(ImportContext& context, simdjson::dom::object object,
		                                  std::string_view nodeName)
		{
			const auto inputName = RequireString(RequireMember(object, "input", nodeName), "geglu_feed_forward input");
			const auto input = RequireValue(context, inputName, nodeName);
			return AddGEGLUFeedForwardSpec(context, input, object, nodeName, "geglu_feed_forward");
		}

		NodeOutput AddGEGLUFeedForwardSpec(ImportContext& context, NodeOutput input, simdjson::dom::object spec,
		                                   std::string_view nodeName, std::string_view label)
		{
			auto projected = AddLinearSpec(context, input, RequireObjectMember(spec, "proj", label), nodeName, label);
			const auto projectedInfo = context.subgraph.GetOutputInfo(projected);
			if (projectedInfo.shape.empty())
			{
				throw std::runtime_error("Torch manifest geglu_feed_forward projection must not be scalar");
			}
			const auto axis = FindSizeOr(spec, "axis", projectedInfo.shape.size() - 1, "geglu_feed_forward axis");
			if (axis >= projectedInfo.shape.size())
			{
				throw std::runtime_error(std::format(
				    "Torch manifest node '{}' geglu_feed_forward axis {} out of range for rank {}",
				    nodeName, axis, projectedInfo.shape.size()));
			}
			const auto width = projectedInfo.shape[axis];
			if (width % 2 != 0)
			{
				throw std::runtime_error(std::format(
				    "Torch manifest node '{}' geglu_feed_forward projection axis width {} must be even",
				    nodeName, width));
			}
			const auto inner = width / 2;
			auto partShape = projectedInfo.shape;
			partShape[axis] = inner;
			const auto valueId = context.subgraph.AddNode(SliceNode{ projected, axis, 0, inner },
			                                               { OutputInfo{ projectedInfo.dtype, partShape } });
			const auto gateId = context.subgraph.AddNode(SliceNode{ projected, axis, inner, inner },
			                                              { OutputInfo{ projectedInfo.dtype, partShape } });
			auto gate = AddActivationByName(context, { gateId, 0 }, "gelu", nodeName);
			auto hidden = AddBinary(context, BinaryOp::Multiply, { valueId, 0 }, gate, nodeName,
			                        "geglu_feed_forward gate multiply");
			auto output = AddLinearSpec(context, hidden, RequireObjectMember(spec, "down", label), nodeName, label);
			if (FindBool(spec, "residual", true, "geglu_feed_forward residual"))
			{
				output = AddBinary(context, BinaryOp::Add, output, input, nodeName,
				                   "geglu_feed_forward residual add");
			}
			context.report.loweredOps.push_back(
			    std::format("{}: geglu_feed_forward -> Linear/Slice/GELU/Gate/Linear(+residual)", nodeName));
			return output;
		}

		NodeOutput ImportAttentionBlock(ImportContext& context, simdjson::dom::object object,
		                                std::string_view nodeName)
		{
			const auto inputName = RequireString(RequireMember(object, "input", nodeName), "attention_block input");
			const auto input = RequireValue(context, inputName, nodeName);
			const auto contextName = FindString(object, "context", "attention_block context").value_or(std::string(inputName));
			const auto keyValueInput = RequireValue(context, contextName, nodeName);
			const auto output = AddAttentionBlockSpec(context, input, keyValueInput, object, nodeName,
			                                          "attention_block");
			context.report.loweredOps.push_back(
			    std::format("{}: attention_block -> QKV/head reshape/SDPA/output projection(+residual)", nodeName));
			return output;
		}

		NodeOutput AddAttentionBlockSpec(ImportContext& context, NodeOutput input, NodeOutput keyValueInput,
		                                 simdjson::dom::object spec, std::string_view nodeName,
		                                 std::string_view label)
		{
			const auto heads = FindSizeOr(spec, "heads", FindSizeOr(spec, "num_heads", 1, label), label);
			if (heads == 0)
			{
				throw std::runtime_error("Torch manifest attention_block requires heads > 0");
			}

			auto q = AddLinearSpec(context, input, RequireObjectMember(spec, "q", label), nodeName, label);
			auto k = AddLinearSpec(context, keyValueInput, RequireObjectMember(spec, "k", label), nodeName, label);
			auto v = AddLinearSpec(context, keyValueInput, RequireObjectMember(spec, "v", label), nodeName, label);
			const auto qInfo = context.subgraph.GetOutputInfo(q);
			const auto kInfo = context.subgraph.GetOutputInfo(k);
			const auto vInfo = context.subgraph.GetOutputInfo(v);
			if (qInfo.shape.size() != 2 || kInfo.shape.size() != 2 || vInfo.shape.size() != 2)
			{
				throw std::runtime_error("Torch manifest attention_block currently expects 2D [tokens, channels] tensors");
			}
			if (qInfo.shape[1] % heads != 0 || kInfo.shape[1] % heads != 0 || vInfo.shape[1] % heads != 0)
			{
				throw std::runtime_error("Torch manifest attention_block projection widths must be divisible by heads");
			}
			const auto headDim = qInfo.shape[1] / heads;
			const auto keyHeadDim = kInfo.shape[1] / heads;
			const auto valueHeadDim = vInfo.shape[1] / heads;
			if (headDim != keyHeadDim)
			{
				throw std::runtime_error("Torch manifest attention_block q/k head dimensions must match");
			}

			q = AddReshapeChecked(context, q, { qInfo.shape[0], heads, headDim }, nodeName);
			q = Layer::AddPermute(context.subgraph, q, { 1uz, 0uz, 2uz });
			k = AddReshapeChecked(context, k, { kInfo.shape[0], heads, headDim }, nodeName);
			k = Layer::AddPermute(context.subgraph, k, { 1uz, 2uz, 0uz });
			v = AddReshapeChecked(context, v, { vInfo.shape[0], heads, valueHeadDim }, nodeName);
			v = Layer::AddPermute(context.subgraph, v, { 1uz, 0uz, 2uz });

			auto scores = Layer::AddBatchMatMul(context.subgraph, q, k);
			const auto scale = FindDouble(spec, "scale", 1.0 / std::sqrt(static_cast<double>(headDim)), label);
			scores = AddScalarBinary(context, BinaryOp::Multiply, scores, scale, nodeName, "attention scale");
			if (auto maskName = FindString(spec, "mask", label))
			{
				const auto mask = RequireValue(context, *maskName, nodeName);
				scores = AddBinary(context, BinaryOp::Add, scores, mask, nodeName, "attention mask add");
			}
			const auto scoreInfo = context.subgraph.GetOutputInfo(scores);
			auto probs = Layer::AddSoftmax(context.subgraph, scores, scoreInfo.shape.size() - 1);
			auto attended = Layer::AddBatchMatMul(context.subgraph, probs, v);
			attended = Layer::AddPermute(context.subgraph, attended, { 1uz, 0uz, 2uz });
			attended = AddReshapeChecked(context, attended, { qInfo.shape[0], heads * valueHeadDim }, nodeName);
			auto output = AddLinearSpec(context, attended, RequireObjectMember(spec, "out", label), nodeName, label);
			if (FindBool(spec, "residual", true, label))
			{
				output = AddBinary(context, BinaryOp::Add, output, input, nodeName, "attention residual add");
			}
			return output;
		}

		NodeOutput ImportSpatialTransformer2D(ImportContext& context, simdjson::dom::object object,
		                                      std::string_view nodeName)
		{
			const auto inputName = RequireString(RequireMember(object, "input", nodeName),
			                                     "spatial_transformer_2d input");
			const auto input = RequireValue(context, inputName, nodeName);
			const auto inputInfo = context.subgraph.GetOutputInfo(input);
			if (inputInfo.shape.size() != 4)
			{
				throw std::runtime_error(std::format(
				    "Torch manifest node '{}' spatial_transformer_2d expects NCHW rank-4 input", nodeName));
			}
			if (inputInfo.shape[0] != 1)
			{
				throw std::runtime_error(std::format(
				    "Torch manifest node '{}' spatial_transformer_2d currently supports batch=1 to avoid cross-batch attention",
				    nodeName));
			}

			const auto batch = inputInfo.shape[0];
			const auto inputChannels = inputInfo.shape[1];
			const auto height = inputInfo.shape[2];
			const auto width = inputInfo.shape[3];
			const auto spatialTokens = height * width;
			const auto useLinear = FindBool(object, "use_linear", true, "spatial_transformer_2d use_linear");
			const auto contextName = FindString(object, "context", "spatial_transformer_2d context");
			const std::optional<NodeOutput> contextValue =
			    contextName ? std::optional<NodeOutput>{ RequireValue(context, *contextName, nodeName) } : std::nullopt;

			auto current = AddGroupNormSpec(context, input, RequireObjectMember(object, "norm",
			                                                                    "spatial_transformer_2d norm"),
			                                nodeName, "spatial_transformer_2d norm");
			std::size_t tokenWidth = inputChannels;
			if (useLinear)
			{
				current = AddReshapeChecked(context, current, { inputChannels, spatialTokens }, nodeName);
				current = Layer::AddPermute(context.subgraph, current, { 1uz, 0uz });
				current = AddLinearSpec(context, current,
				                        RequireObjectMember(object, "proj_in", "spatial_transformer_2d proj_in"),
				                        nodeName, "spatial_transformer_2d proj_in");
				tokenWidth = context.subgraph.GetOutputInfo(current).shape[1];
			}
			else
			{
				current = AddConv2DSpec(context, current,
				                        RequireObjectMember(object, "proj_in", "spatial_transformer_2d proj_in"),
				                        nodeName, "spatial_transformer_2d proj_in");
				const auto projectedInfo = context.subgraph.GetOutputInfo(current);
				tokenWidth = projectedInfo.shape[1];
				current = AddReshapeChecked(context, current, { tokenWidth, spatialTokens }, nodeName);
				current = Layer::AddPermute(context.subgraph, current, { 1uz, 0uz });
			}

			std::size_t blockIndex = 0;
			for (auto blockValue : RequireArray(RequireMember(object, "blocks", nodeName),
			                                    "spatial_transformer_2d blocks"))
			{
				const auto block = RequireObject(blockValue, "spatial_transformer_2d block");
				const auto blockLabel = std::format("spatial_transformer_2d block {}", blockIndex);

				auto norm1 = AddLayerNormSpec(context, current, RequireObjectMember(block, "norm1", blockLabel),
				                              nodeName, blockLabel + " norm1");
				auto attn1 = AddAttentionBlockSpec(context, norm1, norm1,
				                                   RequireObjectMember(block, "attn1", blockLabel),
				                                   nodeName, blockLabel + " attn1");
				current = AddBinary(context, BinaryOp::Add, current, attn1, nodeName,
				                    blockLabel + " self-attention residual");

				auto norm2 = AddLayerNormSpec(context, current, RequireObjectMember(block, "norm2", blockLabel),
				                              nodeName, blockLabel + " norm2");
				auto keyValue = contextValue.value_or(norm2);
				auto attn2 = AddAttentionBlockSpec(context, norm2, keyValue,
				                                   RequireObjectMember(block, "attn2", blockLabel),
				                                   nodeName, blockLabel + " attn2");
				current = AddBinary(context, BinaryOp::Add, current, attn2, nodeName,
				                    blockLabel + " cross-attention residual");

				auto norm3 = AddLayerNormSpec(context, current, RequireObjectMember(block, "norm3", blockLabel),
				                              nodeName, blockLabel + " norm3");
				auto ff = AddGEGLUFeedForwardSpec(context, norm3, RequireObjectMember(block, "ff", blockLabel),
				                                  nodeName, blockLabel + " ff");
				current = AddBinary(context, BinaryOp::Add, current, ff, nodeName, blockLabel + " ff residual");
				++blockIndex;
			}
			if (blockIndex == 0)
			{
				throw std::runtime_error("Torch manifest spatial_transformer_2d requires at least one block");
			}

			if (useLinear)
			{
				current = AddLinearSpec(context, current,
				                        RequireObjectMember(object, "proj_out", "spatial_transformer_2d proj_out"),
				                        nodeName, "spatial_transformer_2d proj_out");
				const auto projectedInfo = context.subgraph.GetOutputInfo(current);
				if (projectedInfo.shape.size() != 2 || projectedInfo.shape[0] != spatialTokens ||
				    projectedInfo.shape[1] != inputChannels)
				{
					throw std::runtime_error(std::format(
					    "Torch manifest node '{}' spatial_transformer_2d proj_out must return [H*W, input_channels]",
					    nodeName));
				}
				current = Layer::AddPermute(context.subgraph, current, { 1uz, 0uz });
				current = AddReshapeChecked(context, current, { batch, inputChannels, height, width }, nodeName);
			}
			else
			{
				current = Layer::AddPermute(context.subgraph, current, { 1uz, 0uz });
				current = AddReshapeChecked(context, current, { batch, tokenWidth, height, width }, nodeName);
				current = AddConv2DSpec(context, current,
				                        RequireObjectMember(object, "proj_out", "spatial_transformer_2d proj_out"),
				                        nodeName, "spatial_transformer_2d proj_out");
			}

			const auto output = AddBinary(context, BinaryOp::Add, current, input, nodeName,
			                              "spatial_transformer_2d residual");
			context.report.loweredOps.push_back(std::format(
			    "{}: spatial_transformer_2d -> GroupNorm/proj_in/flatten/TransformerBlock*/proj_out/residual",
			    nodeName));
			return output;
		}

		NodeOutput ApplyVAEDecodeStep(ImportContext& context, NodeOutput input, simdjson::dom::object step,
		                              std::string_view nodeName)
		{
			const auto opText = std::string(RequireString(RequireMember(step, "op", nodeName), "vae_decode step op"));
			const auto op = NormalizeToken(opText);
			if (op == "conv2d")
			{
				return AddConv2DSpec(context, input, step, nodeName, "vae_decode conv2d");
			}
			if (op == "convtranspose2d" || op == "deconv2d")
			{
				return AddConvTranspose2DSpec(context, input, step, nodeName, "vae_decode conv_transpose2d");
			}
			if (op == "groupnorm" || op == "groupnormalization")
			{
				return AddGroupNormSpec(context, input, step, nodeName, "vae_decode group_norm");
			}
			if (op == "silu" || op == "swish" || op == "relu" || op == "gelu" || op == "tanh" ||
			    op == "sigmoid")
			{
				return AddActivationByName(context, input, opText, nodeName);
			}
			if (op == "upsample" || op == "interpolate" || op == "resize")
			{
				const auto mode = ParseUpsampleMode(FindString(step, "mode", "vae_decode upsample mode").value_or("nearest"));
				auto outputSpatial = FindSpatialList(step,
				                                     { "output_spatial_shape", "output_size", "size", "spatial_shape" },
				                                     {}, "vae_decode upsample output spatial shape", false);
				if (outputSpatial.empty())
				{
					throw std::runtime_error("Torch manifest vae_decode upsample step requires output_spatial_shape");
				}
				const auto alignCorners = FindBool(step, "align_corners", false, "vae_decode upsample align_corners");
				return Layer::AddUpsample(context.subgraph, input, mode, std::move(outputSpatial), alignCorners);
			}
			if (op == "pad")
			{
				const auto inputInfo = context.subgraph.GetOutputInfo(input);
				auto lowPads = FindSizeListOr(step, "low_pads", std::vector<std::size_t>(inputInfo.shape.size(), 0uz),
				                              "vae_decode pad low_pads", true);
				auto highPads = FindSizeListOr(step, "high_pads", std::vector<std::size_t>(inputInfo.shape.size(), 0uz),
				                               "vae_decode pad high_pads", true);
				const auto mode = ParsePadMode(FindString(step, "mode", "vae_decode pad mode").value_or("constant"));
				const auto constantValue = FindDouble(step, "constant_value",
				                                      FindDouble(step, "value", 0.0, "vae_decode pad value"),
				                                      "vae_decode pad constant_value");
				return Layer::AddPad(context.subgraph, input, lowPads, highPads, mode, constantValue);
			}
			throw std::runtime_error("Torch manifest vae_decode unsupported step op: " + opText);
		}

		NodeOutput ImportVAEDecode(ImportContext& context, simdjson::dom::object object,
		                           std::string_view nodeName)
		{
			const auto inputName = RequireString(RequireMember(object, "input", nodeName), "vae_decode input");
			auto current = RequireValue(context, inputName, nodeName);
			if (auto scale = FindMember(object, "latent_scale"))
			{
				current = AddScalarBinary(context, BinaryOp::Multiply, current,
				                          RequireDouble(*scale, "vae_decode latent_scale"), nodeName,
				                          "vae_decode latent scale");
			}
			for (auto stepValue : RequireArray(RequireMember(object, "steps", nodeName), "vae_decode steps"))
			{
				current = ApplyVAEDecodeStep(context, current, RequireObject(stepValue, "vae_decode step"), nodeName);
			}
			if (auto scale = FindMember(object, "output_scale"))
			{
				current = AddScalarBinary(context, BinaryOp::Multiply, current,
				                          RequireDouble(*scale, "vae_decode output_scale"), nodeName,
				                          "vae_decode output scale");
			}
			if (auto bias = FindMember(object, "output_bias"))
			{
				current = AddScalarBinary(context, BinaryOp::Add, current,
				                          RequireDouble(*bias, "vae_decode output_bias"), nodeName,
				                          "vae_decode output bias");
			}
			if (auto clamp = FindObject(object, "clamp", "vae_decode clamp"))
			{
				const auto minValue = FindDouble(*clamp, "min", 0.0, "vae_decode clamp min");
				const auto maxValue = FindDouble(*clamp, "max", 1.0, "vae_decode clamp max");
				current = Layer::AddClamp(context.subgraph, current, minValue, maxValue);
			}
			context.report.loweredOps.push_back(
			    std::format("{}: vae_decode -> fixed step Conv/Norm/Upsample/ConvTranspose/scale/clamp", nodeName));
			return current;
		}

		NodeOutput ImportConcat(ImportContext& context, simdjson::dom::object object,
		                        std::string_view nodeName)
		{
			std::vector<NodeOutput> inputs;
			for (auto inputValue : RequireArray(RequireMember(object, "inputs", nodeName), "concat inputs"))
			{
				inputs.push_back(RequireValue(context, RequireString(inputValue, "concat input"), nodeName));
			}
			if (inputs.empty())
			{
				throw std::runtime_error("Torch manifest concat requires at least one input");
			}

			const auto axis = FindSizeOr(object, "axis", 0, "concat axis");
			const auto firstInfo = context.subgraph.GetOutputInfo(inputs.front());
			if (axis >= firstInfo.shape.size())
			{
				throw std::runtime_error(std::format("Torch manifest node '{}' concat axis {} out of range for rank {}",
				                                     nodeName, axis, firstInfo.shape.size()));
			}

			auto outputShape = firstInfo.shape;
			outputShape[axis] = 0;
			for (std::size_t i = 0; i < inputs.size(); ++i)
			{
				const auto info = context.subgraph.GetOutputInfo(inputs[i]);
				if (info.dtype != firstInfo.dtype)
				{
					throw std::runtime_error(std::format(
					    "Torch manifest node '{}' concat input {} dtype mismatch: expected {}, got {}",
					    nodeName, i, DataTypeName(firstInfo.dtype), DataTypeName(info.dtype)));
				}
				if (info.shape.size() != firstInfo.shape.size())
				{
					throw std::runtime_error(std::format(
					    "Torch manifest node '{}' concat input {} rank mismatch: expected {}, got {}",
					    nodeName, i, firstInfo.shape.size(), info.shape.size()));
				}
				for (std::size_t dim = 0; dim < info.shape.size(); ++dim)
				{
					if (dim == axis)
					{
						outputShape[dim] += info.shape[dim];
						continue;
					}
					if (info.shape[dim] != firstInfo.shape[dim])
					{
						throw std::runtime_error(std::format(
						    "Torch manifest node '{}' concat input {} dim {} mismatch: expected {}, got {}",
						    nodeName, i, dim, firstInfo.shape[dim], info.shape[dim]));
					}
				}
			}

			const auto id = context.subgraph.AddNode(ConcatNode{ std::move(inputs), axis },
			                                         { OutputInfo{ firstInfo.dtype, std::move(outputShape) } });
			context.report.loweredOps.push_back(std::format("{}: concat -> ConcatNode", nodeName));
			return { id, 0 };
		}

		NodeOutput ImportNode(ImportContext& context, simdjson::dom::object object)
		{
			const auto nodeName =
			    FindString(object, "name", "node name").value_or(std::format("node{}", context.report.loweredOps.size()));
			const auto opText = std::string(RequireString(RequireMember(object, "op", nodeName), "node op"));
			const auto op = NormalizeToken(opText);

			if (op == "linear" || op == "torchnnlinear" || op == "attentionprojection")
			{
				return ImportLinear(context, object, nodeName);
			}
			if (op == "embedding" || op == "torchnnembedding")
			{
				return ImportEmbedding(context, object, nodeName);
			}
			if (op == "conv2d" || op == "torchnnconv2d")
			{
				return ImportConv2D(context, object, nodeName);
			}
			if (op == "convtranspose2d" || op == "convtransposed2d" || op == "torchnnconvtranspose2d" ||
			    op == "deconv2d")
			{
				return ImportConvTranspose2D(context, object, nodeName);
			}
			if (op == "layernorm" || op == "torchnnlayernorm")
			{
				return ImportNormalization(context, object, nodeName, NormalizationMode::LayerNorm);
			}
			if (op == "rmsnorm")
			{
				return ImportNormalization(context, object, nodeName, NormalizationMode::RMSNorm);
			}
			if (op == "groupnorm" || op == "groupnormalization" || op == "torchnngroupnorm")
			{
				return ImportGroupNorm(context, object, nodeName);
			}
			if (op == "timestepembedding" || op == "timeembedding" || op == "sinusoidaltimestepembedding")
			{
				return ImportTimestepEmbedding(context, object, nodeName);
			}
			if (op == "residualblock" || op == "resnetblock2d" || op == "unetresidualblock")
			{
				return ImportResidualBlock(context, object, nodeName);
			}
			if (op == "feedforward" || op == "ffn" || op == "swiglu")
			{
				return ImportFeedForward(context, object, nodeName);
			}
			if (op == "geglu" || op == "geglufeedforward" || op == "gegluffn" || op == "torchgeglu")
			{
				return ImportGEGLUFeedForward(context, object, nodeName);
			}
			if (op == "spatialtransformer2d" || op == "spatialtransformer" || op == "spatialtransformerblock")
			{
				return ImportSpatialTransformer2D(context, object, nodeName);
			}
			if (op == "attentionblock" || op == "crossattention" || op == "selfattention")
			{
				return ImportAttentionBlock(context, object, nodeName);
			}
			if (op == "vaedecode" || op == "vaedecoder" || op == "autoencoderkldecode")
			{
				return ImportVAEDecode(context, object, nodeName);
			}
			if (op == "concat" || op == "cat" || op == "torchcat")
			{
				return ImportConcat(context, object, nodeName);
			}

			const auto requireInputName = [&] {
				return std::string(RequireString(RequireMember(object, "input", nodeName), "node input"));
			};
			if (op == "relu")
			{
				const auto output = Layer::AddReLU(context.subgraph, RequireValue(context, requireInputName(), nodeName));
				context.report.loweredOps.push_back(std::format("{}: relu -> Max(x, 0)", nodeName));
				return output;
			}
			if (op == "gelu")
			{
				const auto output = Layer::AddGELU(context.subgraph, RequireValue(context, requireInputName(), nodeName));
				context.report.loweredOps.push_back(std::format("{}: gelu -> GELU(tanh approximation)", nodeName));
				return output;
			}
			if (op == "geluerf")
			{
				const auto output =
				    Layer::AddGELUErf(context.subgraph, RequireValue(context, requireInputName(), nodeName));
				context.report.loweredOps.push_back(std::format("{}: gelu_erf -> GELU(erf)", nodeName));
				return output;
			}
			if (op == "silu" || op == "swish")
			{
				const auto output = Layer::AddSiLU(context.subgraph, RequireValue(context, requireInputName(), nodeName));
				context.report.loweredOps.push_back(std::format("{}: silu -> x * sigmoid(x)", nodeName));
				return output;
			}
			if (op == "sigmoid")
			{
				const auto output =
				    Layer::AddSigmoid(context.subgraph, RequireValue(context, requireInputName(), nodeName));
				context.report.loweredOps.push_back(std::format("{}: sigmoid -> primitive expression", nodeName));
				return output;
			}
			if (op == "tanh")
			{
				const auto output = Layer::AddTanh(context.subgraph, RequireValue(context, requireInputName(), nodeName));
				context.report.loweredOps.push_back(std::format("{}: tanh -> primitive expression", nodeName));
				return output;
			}
			if (op == "softmax")
			{
				const auto input = RequireValue(context, requireInputName(), nodeName);
				const auto inputInfo = context.subgraph.GetOutputInfo(input);
				if (inputInfo.shape.empty())
				{
					throw std::runtime_error("Torch manifest softmax input must not be scalar");
				}
				const auto axisMember = FindMember(object, "axis");
				const auto axis = CheckedToSize(axisMember ? RequireUInt(*axisMember, "softmax axis")
				                                           : static_cast<std::uint64_t>(inputInfo.shape.size() - 1),
				                                "softmax axis");
				const auto output = Layer::AddSoftmax(context.subgraph, input, axis);
				context.report.loweredOps.push_back(std::format("{}: softmax -> SoftmaxNode", nodeName));
				return output;
			}
			if (op == "upsample" || op == "interpolate" || op == "resize")
			{
				return ImportUpsample(context, object, nodeName);
			}
			if (op == "pad")
			{
				return ImportPad(context, object, nodeName);
			}
			if (op == "clamp" || op == "clip")
			{
				return ImportClamp(context, object, nodeName);
			}
			if (op == "cast" || op == "to")
			{
				const auto input = RequireValue(context, requireInputName(), nodeName);
				const auto inputInfo = context.subgraph.GetOutputInfo(input);
				const auto targetType =
				    MapTorchManifestDataType(RequireString(RequireMember(object, "dtype", nodeName), "cast dtype"));
				const auto id = context.subgraph.AddNode(CastNode{ input, targetType },
				                                         { OutputInfo{ targetType, inputInfo.shape } });
				context.report.loweredOps.push_back(std::format("{}: cast -> CastNode", nodeName));
				return { id, 0 };
			}
			if (op == "scale")
			{
				const auto input = RequireValue(context, requireInputName(), nodeName);
				auto output = input;
				if (auto factor = FindMember(object, "factor"))
				{
					output = AddScalarBinary(context, BinaryOp::Multiply, output,
					                         RequireDouble(*factor, "scale factor"), nodeName, "scale factor");
				}
				if (auto bias = FindMember(object, "bias"))
				{
					output = AddScalarBinary(context, BinaryOp::Add, output,
					                         RequireDouble(*bias, "scale bias"), nodeName, "scale bias");
				}
				context.report.loweredOps.push_back(std::format("{}: scale -> scalar Multiply/Add", nodeName));
				return output;
			}
			if (op == "reshape")
			{
				const auto output = Layer::AddReshape(context.subgraph, RequireValue(context, requireInputName(), nodeName),
				                                      ParseShape(RequireMember(object, "shape", nodeName),
				                                                 "reshape shape"));
				context.report.loweredOps.push_back(std::format("{}: reshape -> ReshapeNode", nodeName));
				return output;
			}
			if (op == "slice" || op == "narrow")
			{
				const auto input = RequireValue(context, requireInputName(), nodeName);
				const auto inputInfo = context.subgraph.GetOutputInfo(input);
				const auto axis = FindSizeOr(object, "axis", inputInfo.shape.empty() ? 0 : inputInfo.shape.size() - 1,
				                             "slice axis");
				if (axis >= inputInfo.shape.size())
				{
					throw std::runtime_error(std::format("Torch manifest node '{}' slice axis {} out of range for rank {}",
					                                     nodeName, axis, inputInfo.shape.size()));
				}
				const auto start = FindSizeOr(object, "start", 0, "slice start");
				const auto length = CheckedToSize(RequireUInt(RequireMember(object, "length", nodeName),
				                                              "slice length"),
				                                  "slice length");
				if (length == 0 || start > inputInfo.shape[axis] || length > inputInfo.shape[axis] - start)
				{
					throw std::runtime_error(std::format(
					    "Torch manifest node '{}' slice range [{}, {}) is out of bounds for axis dim {}",
					    nodeName, start, start + length, inputInfo.shape[axis]));
				}
				auto outputShape = inputInfo.shape;
				outputShape[axis] = length;
				const auto id = context.subgraph.AddNode(SliceNode{ input, axis, start, length },
				                                         { OutputInfo{ inputInfo.dtype, std::move(outputShape) } });
				context.report.loweredOps.push_back(std::format("{}: slice -> SliceNode", nodeName));
				return { id, 0 };
			}
			if (op == "permute")
			{
				const auto output = Layer::AddPermute(
				    context.subgraph, RequireValue(context, requireInputName(), nodeName),
				    ParseSizeList(RequireMember(object, "permutation", nodeName), "permute permutation", true));
				context.report.loweredOps.push_back(std::format("{}: permute -> PermuteNode", nodeName));
				return output;
			}
			if (op == "transpose")
			{
				const auto input = RequireValue(context, requireInputName(), nodeName);
				const auto inputInfo = context.subgraph.GetOutputInfo(input);
				auto resultShape = UnaryOpTraits<UnaryOp::Transpose>::ResultShape(inputInfo.shape);
				if (!resultShape)
				{
					throw std::runtime_error(std::format("Torch manifest transpose shape inference failed: {}",
					                                     resultShape.error()));
				}
				const auto id = context.subgraph.AddNode(UnaryOpNode{ UnaryOp::Transpose, input },
				                                         { OutputInfo{ inputInfo.dtype, *resultShape } });
				context.report.loweredOps.push_back(std::format("{}: transpose -> UnaryOp::Transpose", nodeName));
				return { id, 0 };
			}
			if (op == "add" || op == "sub" || op == "subtract" || op == "mul" || op == "multiply" ||
			    op == "div" || op == "divide" || op == "matmul")
			{
				const auto lhs = RequireValue(context, RequireString(RequireMember(object, "lhs", nodeName), "lhs"),
				                              nodeName);
				const auto rhs = RequireValue(context, RequireString(RequireMember(object, "rhs", nodeName), "rhs"),
				                              nodeName);
				BinaryOp binaryOp = BinaryOp::Add;
				if (op == "sub" || op == "subtract")
				{
					binaryOp = BinaryOp::Subtract;
				}
				else if (op == "mul" || op == "multiply")
				{
					binaryOp = BinaryOp::Multiply;
				}
				else if (op == "div" || op == "divide")
				{
					binaryOp = BinaryOp::Divide;
				}
				else if (op == "matmul")
				{
					binaryOp = BinaryOp::MatMul;
				}
				const auto output = AddBinary(context, binaryOp, lhs, rhs, nodeName, opText);
				context.report.loweredOps.push_back(std::format("{}: {} -> BinaryOp", nodeName, opText));
				return output;
			}

			context.report.unsupportedOps.push_back(std::format("{}: {}", nodeName, opText));
			throw std::runtime_error("Torch manifest unsupported op: " + opText);
		}

		void ImportManifestNodes(ImportContext& context, simdjson::dom::object rootObject)
		{
			for (auto nodeValue : RequireArray(RequireMember(rootObject, "nodes", "root"), "nodes"))
			{
				const auto object = RequireObject(nodeValue, "node entry");
				const auto outputName =
				    std::string(RequireString(RequireMember(object, "output", "node"), "node output"));
				const auto output = ImportNode(context, object);
				BindValue(context, outputName, output, outputName);
			}
		}

		std::vector<NodeOutput> ParseOutputs(ImportContext& context, simdjson::dom::object rootObject,
		                                     std::vector<std::string>& outputNames)
		{
			std::vector<NodeOutput> outputs;
			for (auto outputValue : RequireArray(RequireMember(rootObject, "outputs", "root"), "outputs"))
			{
				std::string name;
				std::string source;
				std::string_view directOutput;
				if (!outputValue.get_string().get(directOutput))
				{
					name = std::string(directOutput);
					source = name;
				}
				else
				{
					const auto object = RequireObject(outputValue, "output entry");
					name = std::string(RequireString(RequireMember(object, "name", "output"), "output name"));
					source = FindString(object, "source", "output source").value_or(name);
				}
				outputs.push_back(RequireValue(context, source, "outputs"));
				outputNames.push_back(std::move(name));
			}
			return outputs;
		}
	} // namespace

	std::span<const TorchManifestOpMapping> SupportedTorchManifestOpMappings()
	{
		static constexpr std::array<TorchManifestOpMapping, 35> mappings{ {
		    { "linear", "VariableRef -> MatMul -> optional Add", "expects torch_linear_weight layout for PyTorch weights" },
		    { "attention_projection", "VariableRef -> MatMul -> optional Add", "same layout contract as linear" },
		    { "embedding", "VariableRef -> Gather(axis=0)", "indices input may be named input or indices" },
		    { "conv2d", "Conv2DNode", "channel-first [N, C, H, W], PyTorch weight layout is already compatible" },
		    { "conv_transpose2d", "ConvTranspose2DNode", "PyTorch ConvTranspose2d weight layout is compatible" },
		    { "layer_norm", "NormalizationNode(LayerNorm)", "weight/bias usually use torch_norm_* layout" },
		    { "rms_norm", "NormalizationNode(RMSNorm)", "weight usually uses torch_norm_weight layout" },
		    { "group_norm", "reshape + NormalizationNode(LayerNorm) + affine", "PyTorch NCHW semantics by default; native LiteNN layout is opt-in" },
		    { "timestep_embedding", "TimestepEmbeddingNode", "sinusoidal diffusion timestep embedding" },
		    { "residual_block", "GroupNorm/activation/Conv2D(+temb)+residual", "fixed-shape SDXL UNet ResNet block template" },
		    { "feed_forward", "Linear/activation-or-gate/Linear(+residual)", "fixed-shape transformer MLP template" },
		    { "geglu_feed_forward", "Linear -> Slice(value/gate) -> GELU(gate) -> Multiply -> Linear(+residual)", "SDXL GEGLU combined projection template" },
		    { "attention_block", "QKV projection + head reshape/permute + SDPA + output projection", "fixed-shape self/cross attention over [tokens, channels]" },
		    { "spatial_transformer_2d", "GroupNorm + proj_in + NCHW/token reshape + transformer blocks + proj_out", "batch=1 SDXL use_linear_in_transformer path" },
		    { "vae_decode", "fixed step Conv/Norm/Upsample/ConvTranspose/scale/clamp", "VAE decoder assembly template" },
		    { "concat", "ConcatNode", "used for UNet skip-connection channel joins" },
		    { "matmul", "BinaryOp(MatMul)", "2D matmul" },
		    { "add", "BinaryOp(Add)", "LiteNN broadcast rules" },
		    { "subtract", "BinaryOp(Subtract)", "LiteNN broadcast rules" },
		    { "multiply", "BinaryOp(Multiply)", "LiteNN broadcast rules" },
		    { "divide", "BinaryOp(Divide)", "LiteNN broadcast rules" },
		    { "scale", "scalar Multiply/Add", "convenience op for latent/output scaling" },
		    { "relu", "Max(x, 0)", "lowered through Layer::AddReLU" },
		    { "gelu", "primitive GELU tanh approximation", "PyTorch approximate='tanh' style" },
		    { "gelu_erf", "primitive GELU erf formula", "PyTorch default exact-style path" },
		    { "silu", "x * sigmoid(x)", "also accepts swish" },
		    { "softmax", "SoftmaxNode", "axis defaults to last dimension" },
		    { "pad", "PadNode", "explicit low_pads/high_pads, constant/reflect/replicate" },
		    { "upsample", "UpsampleNode", "nearest/bilinear/bicubic channel-first 2D" },
		    { "clamp", "Min(Max(x, min), max)", "final image clamp/clip policy helper" },
		    { "cast", "CastNode", "dtype conversion helper for mixed precision manifests" },
		    { "reshape", "ReshapeNode", "element count must match" },
		    { "slice", "SliceNode", "axis/start/length narrow helper" },
		    { "permute", "PermuteNode", "explicit multi-axis permutation" },
		    { "transpose", "UnaryOp(Transpose)", "2D only" },
		} };
		return mappings;
	}

	std::optional<DataType> TryMapTorchManifestDataType(std::string_view dtype)
	{
		if (auto mapped = TryMapSafetensorsDataType(dtype))
		{
			return mapped;
		}
		const auto normalized = NormalizeToken(dtype);
		if (normalized == "torchfloat64" || normalized == "float64" || normalized == "double" ||
		    normalized == "f64")
		{
			return DataType::Float64;
		}
		if (normalized == "torchfloat32" || normalized == "float32" || normalized == "float" ||
		    normalized == "f32")
		{
			return DataType::Float32;
		}
		if (normalized == "torchfloat16" || normalized == "float16" || normalized == "half" ||
		    normalized == "f16")
		{
			return DataType::Float16;
		}
		if (normalized == "torchbfloat16" || normalized == "bfloat16" || normalized == "bf16")
		{
			return DataType::BFloat16;
		}
		if (normalized == "torchfloat8e4m3fn" || normalized == "float8e4m3" || normalized == "f8e4m3")
		{
			return DataType::Float8E4M3;
		}
		if (normalized == "torchfloat8e5m2" || normalized == "float8e5m2" || normalized == "f8e5m2")
		{
			return DataType::Float8E5M2;
		}
		if (normalized == "torchint64" || normalized == "torchlong" || normalized == "int64" || normalized == "long" ||
		    normalized == "i64")
		{
			return DataType::Int64;
		}
		if (normalized == "torchint32" || normalized == "int32" || normalized == "int" ||
		    normalized == "i32")
		{
			return DataType::Int32;
		}
		if (normalized == "torchint8" || normalized == "int8" || normalized == "i8")
		{
			return DataType::Int8;
		}
		if (normalized == "torchuint8" || normalized == "uint8" || normalized == "u8")
		{
			return DataType::UInt8;
		}
		if (normalized == "torchbool" || normalized == "bool")
		{
			return DataType::Bool;
		}
		return std::nullopt;
	}

	DataType MapTorchManifestDataType(std::string_view dtype)
	{
		if (auto mapped = TryMapTorchManifestDataType(dtype))
		{
			return *mapped;
		}
		throw std::runtime_error("Unsupported Torch manifest dtype: " + std::string(dtype));
	}

	TorchManifestImportResult ImportTorchManifest(std::string_view manifestJson,
	                                              const SafetensorsArchive& archive,
	                                              const TorchManifestImportOptions& options)
	{
		simdjson::padded_string padded(manifestJson.data(), manifestJson.size());
		simdjson::dom::parser parser;
		simdjson::dom::element root;
		if (const auto error = parser.parse(padded).get(root))
		{
			throw JsonError("parse failed", error);
		}
		const auto rootObject = RequireObject(root, "root");

		if (auto format = FindString(rootObject, "format", "format"); format && *format != kManifestFormat)
		{
			throw std::runtime_error(std::format("Unsupported Torch manifest format '{}', expected '{}'",
			                                     *format, kManifestFormat));
		}

		ImportContext context;
		context.graph.SetMetadataEntry("torch_manifest.format", std::string(kManifestFormat));
		ImportManifestTensors(context, rootObject, archive, options);
		ImportManifestInputs(context, rootObject);
		ImportManifestNodes(context, rootObject);

		std::vector<std::string> outputNames;
		context.subgraph.SetResults(ParseOutputs(context, rootObject, outputNames));
		const auto forward = context.graph.AddSubgraph(std::move(context.subgraph));
		context.graph.SetForward(forward);
		context.graph.SetInputNames(std::move(context.inputNames));
		context.graph.SetOutputNames(std::move(outputNames));
		Validation::ValidateGraph(context.graph);

		return { std::move(context.graph), std::move(context.report) };
	}

	TorchManifestImportResult LoadTorchManifest(const std::filesystem::path& manifestPath,
	                                            const std::filesystem::path& safetensorsPath,
	                                            const TorchManifestImportOptions& options)
	{
		const auto manifestBytes = ReadAllBytes(manifestPath);
		const auto manifest = std::string_view(reinterpret_cast<const char*>(manifestBytes.data()),
		                                      manifestBytes.size());
		const auto archive = SafetensorsArchive::LoadFile(safetensorsPath);
		return ImportTorchManifest(manifest, archive, options);
	}
} // namespace LiteNN::Serialization
