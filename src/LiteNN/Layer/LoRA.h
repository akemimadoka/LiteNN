#include <LiteNN/Layer/LayerUtils.h>
#include <LiteNN/Layer/Linear.h>
#include <LiteNN/ModelBuilder.h>

#include <format>
#include <optional>
#include <stdexcept>
#include <string>
#include <string_view>
#include <utility>
#include <vector>

#ifndef LITENN_LAYER_LORA_H
#define LITENN_LAYER_LORA_H

namespace LiteNN::Layer
{
	enum class LoRAMergeMode
	{
		Unmerged,
		Merged,
	};

	struct LoRAAdapterMetadata
	{
		std::string targetName;
		std::string adapterName{ "default" };
		std::size_t rank{};
		float alpha = 1.0f;
		float dropout = 0.0f;
		DataType dtype{ DataType::Float32 };
		LoRAMergeMode mergeMode{ LoRAMergeMode::Unmerged };
	};

	struct LinearLoRAAdapter
	{
		LoRAAdapterMetadata metadata;
		std::size_t aVariable{};
		std::size_t bVariable{};
		std::size_t inFeatures{};
		std::size_t outFeatures{};
	};

	enum class LoRATensorRole
	{
		A,
		B,
	};

	struct ParsedLoRATensorName
	{
		std::string targetName;
		std::string adapterName;
		LoRATensorRole role{};
	};

	inline void ValidateLoRAMetadata(const LoRAAdapterMetadata& metadata)
	{
		if (metadata.rank == 0)
		{
			throw std::runtime_error("LoRA rank must be greater than zero");
		}
		if (metadata.alpha == 0.0f)
		{
			throw std::runtime_error("LoRA alpha must be non-zero");
		}
		if (metadata.dropout < 0.0f || metadata.dropout >= 1.0f)
		{
			throw std::runtime_error("LoRA dropout must be in [0, 1)");
		}
		if (metadata.mergeMode != LoRAMergeMode::Unmerged)
		{
			throw std::runtime_error("Only unmerged LoRA adapters are supported by the current layer helper");
		}
	}

	inline float LoRAScale(const LoRAAdapterMetadata& metadata)
	{
		ValidateLoRAMetadata(metadata);
		return metadata.alpha / static_cast<float>(metadata.rank);
	}

	inline bool IsLoRAFloatingAdapterDType(DataType dtype)
	{
		return dtype == DataType::Float32 || dtype == DataType::Float16 || dtype == DataType::BFloat16;
	}

	inline std::optional<ParsedLoRATensorName> ParsePEFTLoRATensorName(std::string_view name)
	{
		const auto parse = [&](std::string_view marker, LoRATensorRole role) -> std::optional<ParsedLoRATensorName> {
			const auto markerPos = name.find(marker);
			if (markerPos == std::string_view::npos || markerPos == 0)
			{
				return std::nullopt;
			}
			const auto suffix = name.substr(markerPos + marker.size());
			if (!suffix.ends_with(".weight"))
			{
				return std::nullopt;
			}
			auto adapter = suffix.substr(0, suffix.size() - std::string_view(".weight").size());
			if (adapter.starts_with("."))
			{
				adapter.remove_prefix(1);
			}
			if (adapter.empty())
			{
				adapter = "default";
			}
			return ParsedLoRATensorName{ .targetName = std::string(name.substr(0, markerPos)),
			                             .adapterName = std::string(adapter),
			                             .role = role };
		};
		if (auto parsed = parse(".lora_A", LoRATensorRole::A))
		{
			return parsed;
		}
		if (auto parsed = parse(".lora_B", LoRATensorRole::B))
		{
			return parsed;
		}
		return std::nullopt;
	}

	inline void ValidateLinearLoRACompatibility(const LinearLayer& linear, const LinearLoRAAdapter& adapter)
	{
		if (linear.inFeatures != adapter.inFeatures || linear.outFeatures != adapter.outFeatures)
		{
			throw std::runtime_error("LoRA adapter feature dimensions do not match the target Linear layer");
		}
		if (linear.dtype != adapter.metadata.dtype)
		{
			throw std::runtime_error("LoRA adapter dtype must match the target Linear layer dtype");
		}
		if (!IsLoRAFloatingAdapterDType(adapter.metadata.dtype))
		{
			throw std::runtime_error("LoRA adapters currently require Float32, Float16, or BFloat16 tensors");
		}
	}

	namespace Detail
	{
		inline LinearLoRAAdapter CreateLinearLoRAImpl(Graph& graph, LoRAAdapterMetadata metadata, Tensor<CPU> a,
		                                             Tensor<CPU> b)
		{
			ValidateLoRAMetadata(metadata);
			if (a.DType() != metadata.dtype || b.DType() != metadata.dtype)
			{
				throw std::runtime_error("LoRA adapter tensors must match metadata dtype");
			}
			if (a.Shape().NumDim() != 2 || b.Shape().NumDim() != 2)
			{
				throw std::runtime_error("LoRA adapter tensors must be rank-2 matrices");
			}
			if (a.Shape()[1] != metadata.rank || b.Shape()[0] != metadata.rank)
			{
				throw std::runtime_error("LoRA adapter tensor shapes must be [inFeatures, rank] and [rank, outFeatures]");
			}

			LinearLoRAAdapter adapter;
			adapter.metadata = std::move(metadata);
			adapter.inFeatures = a.Shape()[0];
			adapter.outFeatures = b.Shape()[1];
			adapter.aVariable = graph.AddVariable(Variable::Create(std::move(a)));
			adapter.bVariable = graph.AddVariable(Variable::Create(std::move(b)));
			return adapter;
		}
	} // namespace Detail

	inline LinearLoRAAdapter CreateLinearLoRA(ModelBuilder& builder, LoRAAdapterMetadata metadata, Tensor<CPU> a,
	                                         Tensor<CPU> b)
	{
		return Detail::CreateLinearLoRAImpl(builder.UnsafeMutableGraph(), std::move(metadata), std::move(a), std::move(b));
	}

	inline LinearLoRAAdapter CreateLinearLoRA(Graph& graph, LoRAAdapterMetadata metadata, Tensor<CPU> a, Tensor<CPU> b)
	{
		return Detail::CreateLinearLoRAImpl(graph, std::move(metadata), std::move(a), std::move(b));
	}

	inline NodeOutput AddLinearWithLoRA(Subgraph& subgraph, const LinearLayer& linear,
	                                    const LinearLoRAAdapter& adapter, NodeOutput input)
	{
		ValidateLinearLoRACompatibility(linear, adapter);

		const auto inputInfo = subgraph.GetOutputInfo(input);
		if (inputInfo.dtype != adapter.metadata.dtype || inputInfo.shape.size() != 2 ||
		    inputInfo.shape[1] != adapter.inFeatures)
		{
			throw std::runtime_error(std::format("LoRA input must have shape [batch, {}] and matching dtype",
			                                    adapter.inFeatures));
		}

		const auto base = AddLinear(subgraph, linear, input);
		const std::vector<std::size_t> aShape{ adapter.inFeatures, adapter.metadata.rank };
		const std::vector<std::size_t> hiddenShape{ inputInfo.shape[0], adapter.metadata.rank };
		const std::vector<std::size_t> bShape{ adapter.metadata.rank, adapter.outFeatures };
		const std::vector<std::size_t> outputShape{ inputInfo.shape[0], adapter.outFeatures };

		const auto a = subgraph.AddNode(VariableRefNode{ adapter.aVariable }, { OutputInfo{ adapter.metadata.dtype, aShape } });
		const auto hidden = subgraph.AddNode(BinaryOpNode{ BinaryOp::MatMul, input, { a, 0 } },
		                                     { OutputInfo{ adapter.metadata.dtype, hiddenShape } });
		const auto b = subgraph.AddNode(VariableRefNode{ adapter.bVariable }, { OutputInfo{ adapter.metadata.dtype, bShape } });
		const auto delta = subgraph.AddNode(BinaryOpNode{ BinaryOp::MatMul, { hidden, 0 }, { b, 0 } },
		                                    { OutputInfo{ adapter.metadata.dtype, outputShape } });
		const auto scaleTensor = Detail::MakeFilledTensor(outputShape, adapter.metadata.dtype, LoRAScale(adapter.metadata));
		const auto scale = Detail::AddConstant(subgraph, scaleTensor);
		const auto scaledDelta = subgraph.AddNode(BinaryOpNode{ BinaryOp::Multiply, { delta, 0 }, { scale, 0 } },
		                                          { OutputInfo{ adapter.metadata.dtype, outputShape } });
		const auto result = subgraph.AddNode(BinaryOpNode{ BinaryOp::Add, base, { scaledDelta, 0 } },
		                                     { OutputInfo{ adapter.metadata.dtype, outputShape } });
		return { result, 0 };
	}

	inline LinearLayer MergeLinearLoRA(Graph& graph, const LinearLayer& linear, const LinearLoRAAdapter& adapter)
	{
		ValidateLinearLoRACompatibility(linear, adapter);
		if (linear.dtype != DataType::Float32)
		{
			throw std::runtime_error("Merged LoRA export currently supports Float32 Linear layers only");
		}
		if (graph.GetVariable(linear.weightVariable)->IsQuantized() || graph.GetVariable(adapter.aVariable)->IsQuantized() ||
		    graph.GetVariable(adapter.bVariable)->IsQuantized())
		{
			throw std::runtime_error("Merged LoRA export currently requires dequantized Float32 variables");
		}

		const auto weight = graph.GetVariable(linear.weightVariable)->Data().CopyToDevice(CPU{});
		const auto a = graph.GetVariable(adapter.aVariable)->Data().CopyToDevice(CPU{});
		const auto b = graph.GetVariable(adapter.bVariable)->Data().CopyToDevice(CPU{});
		if (weight.DType() != DataType::Float32 || a.DType() != DataType::Float32 || b.DType() != DataType::Float32)
		{
			throw std::runtime_error("Merged LoRA export variable dtype metadata must be Float32");
		}

		Tensor<CPU> merged(Uninitialized, { linear.inFeatures, linear.outFeatures }, DataType::Float32);
		const auto weightData = weight.Data<float>();
		const auto aData = a.Data<float>();
		const auto bData = b.Data<float>();
		auto mergedData = merged.MutableData<float>();
		const auto scale = LoRAScale(adapter.metadata);
		for (auto in = 0uz; in < linear.inFeatures; ++in)
		{
			for (auto out = 0uz; out < linear.outFeatures; ++out)
			{
				auto delta = 0.0f;
				for (auto r = 0uz; r < adapter.metadata.rank; ++r)
				{
					delta += aData[in * adapter.metadata.rank + r] * bData[r * linear.outFeatures + out];
				}
				mergedData[in * linear.outFeatures + out] = weightData[in * linear.outFeatures + out] + scale * delta;
			}
		}

		if (!linear.biasVariable)
		{
			return Detail::MakeLinearLayerImpl(graph, std::move(merged));
		}

		auto bias = graph.GetVariable(*linear.biasVariable)->Data().CopyToDevice(CPU{});
		if (bias.DType() != DataType::Float32)
		{
			throw std::runtime_error("Merged LoRA export bias dtype metadata must be Float32");
		}
		return Detail::MakeLinearLayerImpl(graph, std::move(merged), std::move(bias));
	}
} // namespace LiteNN::Layer

#endif
