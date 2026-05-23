#include <LiteNN/Serialization/TorchManifest.h>

#include <LiteNN/Layer/Activation.h>
#include <LiteNN/Layer/Gather.h>
#include <LiteNN/Layer/Normalization.h>
#include <LiteNN/Layer/Reshape.h>
#include <LiteNN/Layer/Softmax.h>
#include <LiteNN/Validation/GraphValidator.h>

#include <simdjson.h>

#include <algorithm>
#include <array>
#include <cctype>
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
			    normalized == "torchembeddingweight")
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

			throw std::runtime_error(std::format("Torch manifest tensor '{}' uses unsupported layout preset '{}'",
			                                     manifestName, layout));
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

				const auto dtype = tensor.DType();
				auto shape = tensor.Shape().ToOwned();
				const auto index = context.graph.AddVariable(Variable::Create(std::move(tensor)));
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

		NodeOutput ImportLinear(ImportContext& context, simdjson::dom::object object, std::string_view nodeName)
		{
			const auto inputName = RequireString(RequireMember(object, "input", nodeName), "linear input");
			const auto weightName = RequireString(RequireMember(object, "weight", nodeName), "linear weight");
			const auto input = RequireValue(context, inputName, nodeName);
			const auto weight = AddVariableRef(context, weightName, nodeName);
			auto output = AddBinary(context, BinaryOp::MatMul, input, weight, nodeName, "linear matmul");
			if (auto biasName = FindString(object, "bias", "linear bias"))
			{
				const auto bias = AddVariableRef(context, *biasName, nodeName);
				output = AddBinary(context, BinaryOp::Add, output, bias, nodeName, "linear bias add");
			}
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
			if (op == "layernorm" || op == "torchnnlayernorm")
			{
				return ImportNormalization(context, object, nodeName, NormalizationMode::LayerNorm);
			}
			if (op == "rmsnorm")
			{
				return ImportNormalization(context, object, nodeName, NormalizationMode::RMSNorm);
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
			if (op == "reshape")
			{
				const auto output = Layer::AddReshape(context.subgraph, RequireValue(context, requireInputName(), nodeName),
				                                      ParseShape(RequireMember(object, "shape", nodeName),
				                                                 "reshape shape"));
				context.report.loweredOps.push_back(std::format("{}: reshape -> ReshapeNode", nodeName));
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
		static constexpr std::array<TorchManifestOpMapping, 17> mappings{ {
		    { "linear", "VariableRef -> MatMul -> optional Add", "expects torch_linear_weight layout for PyTorch weights" },
		    { "attention_projection", "VariableRef -> MatMul -> optional Add", "same layout contract as linear" },
		    { "embedding", "VariableRef -> Gather(axis=0)", "indices input may be named input or indices" },
		    { "layer_norm", "NormalizationNode(LayerNorm)", "weight/bias usually use torch_norm_* layout" },
		    { "rms_norm", "NormalizationNode(RMSNorm)", "weight usually uses torch_norm_weight layout" },
		    { "matmul", "BinaryOp(MatMul)", "2D matmul" },
		    { "add", "BinaryOp(Add)", "LiteNN broadcast rules" },
		    { "subtract", "BinaryOp(Subtract)", "LiteNN broadcast rules" },
		    { "multiply", "BinaryOp(Multiply)", "LiteNN broadcast rules" },
		    { "divide", "BinaryOp(Divide)", "LiteNN broadcast rules" },
		    { "relu", "Max(x, 0)", "lowered through Layer::AddReLU" },
		    { "gelu", "primitive GELU tanh approximation", "PyTorch approximate='tanh' style" },
		    { "gelu_erf", "primitive GELU erf formula", "PyTorch default exact-style path" },
		    { "silu", "x * sigmoid(x)", "also accepts swish" },
		    { "softmax", "SoftmaxNode", "axis defaults to last dimension" },
		    { "reshape", "ReshapeNode", "element count must match" },
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
