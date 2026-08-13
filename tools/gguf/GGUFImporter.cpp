#include "GGUFImporter.h"

#include <LiteNN/ExecutablePlan.h>
#include <LiteNN/Serialization/ModelPackageIO.h>

#include <algorithm>
#include <cctype>
#include <cstring>
#include <fstream>
#include <limits>
#include <map>
#include <memory>
#include <optional>
#include <stdexcept>
#include <string>
#include <string_view>
#include <vector>

#include <ggml.h>
#include <gguf.h>

namespace LiteNN::GGUF
{
	std::size_t LLaMAHyperparameters::HeadDimension() const
	{
		if (attentionHeadCount == 0)
		{
			throw std::runtime_error("LLaMA attention.head_count must be greater than zero");
		}
		if ((embeddingLength % attentionHeadCount) != 0)
		{
			throw std::runtime_error(
			    std::format("LLaMA embedding_length {} must be divisible by attention.head_count {}", embeddingLength,
			                attentionHeadCount));
		}
		return embeddingLength / attentionHeadCount;
	}

	std::size_t LLaMAHyperparameters::QueryGroupsPerKVHead() const
	{
		if (attentionHeadCountKV == 0)
		{
			throw std::runtime_error("LLaMA attention.head_count_kv must be greater than zero");
		}
		if ((attentionHeadCount % attentionHeadCountKV) != 0)
		{
			throw std::runtime_error(
			    std::format("LLaMA attention.head_count {} must be divisible by attention.head_count_kv {}",
			                attentionHeadCount, attentionHeadCountKV));
		}
		return attentionHeadCount / attentionHeadCountKV;
	}

	namespace
	{
		const ModelMetadataEntry& RequireMetadata(const Graph& graph, std::string_view key)
		{
			const auto* entry = graph.FindMetadata(key);
			if (!entry)
			{
				throw std::runtime_error(std::format("Missing GGUF metadata key '{}'", key));
			}
			return *entry;
		}

		std::optional<const ModelMetadataEntry*> FindMetadata(const Graph& graph, std::string_view key)
		{
			if (const auto* entry = graph.FindMetadata(key))
			{
				return entry;
			}
			return std::nullopt;
		}

		std::size_t NarrowToSize(std::uint64_t value, std::string_view key)
		{
			if (value > std::numeric_limits<std::size_t>::max())
			{
				throw std::runtime_error(std::format("GGUF metadata key '{}' exceeds size_t range", key));
			}
			return static_cast<std::size_t>(value);
		}

		std::size_t ReadSizeValue(const ModelMetadataEntry& entry)
		{
			return std::visit(
			    [&entry](const auto& value) -> std::size_t {
				    using T = std::decay_t<decltype(value)>;
				    if constexpr (std::same_as<T, std::uint64_t>)
				    {
					    return NarrowToSize(value, entry.key);
				    }
				    else if constexpr (std::same_as<T, std::int64_t>)
				    {
					    if (value < 0)
					    {
						    throw std::runtime_error(
						        std::format("GGUF metadata key '{}' must be non-negative", entry.key));
					    }
					    return NarrowToSize(static_cast<std::uint64_t>(value), entry.key);
				    }
				    else
				    {
					    throw std::runtime_error(std::format("GGUF metadata key '{}' must be an integer", entry.key));
				    }
			    },
			    entry.value);
		}

		double ReadDoubleValue(const ModelMetadataEntry& entry)
		{
			return std::visit(
			    [&entry](const auto& value) -> double {
				    using T = std::decay_t<decltype(value)>;
				    if constexpr (std::same_as<T, double>)
				    {
					    return value;
				    }
				    else if constexpr (std::same_as<T, std::uint64_t>)
				    {
					    return static_cast<double>(value);
				    }
				    else if constexpr (std::same_as<T, std::int64_t>)
				    {
					    return static_cast<double>(value);
				    }
				    else
				    {
					    throw std::runtime_error(std::format("GGUF metadata key '{}' must be numeric", entry.key));
				    }
			    },
			    entry.value);
		}

		std::string ReadStringValue(const ModelMetadataEntry& entry)
		{
			if (const auto* value = std::get_if<std::string>(&entry.value))
			{
				return *value;
			}
			throw std::runtime_error(std::format("GGUF metadata key '{}' must be a string", entry.key));
		}

		bool ReadBoolValue(const ModelMetadataEntry& entry)
		{
			if (const auto* value = std::get_if<bool>(&entry.value))
			{
				return *value;
			}
			throw std::runtime_error(std::format("GGUF metadata key '{}' must be a bool", entry.key));
		}

		std::string NormalizeRopeScalingType(std::string value)
		{
			for (auto& ch : value)
			{
				ch = static_cast<char>(std::tolower(static_cast<unsigned char>(ch)));
			}
			return value;
		}

		RoPELayout RoPELayoutForArchitecture(std::string_view architecture)
		{
			if (architecture == "llama")
			{
				return RoPELayout::Normal;
			}
			if (architecture == "qwen2")
			{
				return RoPELayout::NeoX;
			}
			throw std::runtime_error(
			    std::format("Unsupported GGUF architecture '{}' for explicit RoPE layout", architecture));
		}

		struct GGUFContextDeleter
		{
			void operator()(gguf_context* ctx) const
			{
				if (ctx)
				{
					gguf_free(ctx);
				}
			}
		};

		struct GGMLContextDeleter
		{
			void operator()(ggml_context* ctx) const
			{
				if (ctx)
				{
					ggml_free(ctx);
				}
			}
		};

		using GGUFContextPtr = std::unique_ptr<gguf_context, GGUFContextDeleter>;
		using GGMLContextPtr = std::unique_ptr<ggml_context, GGMLContextDeleter>;

		struct LoadedGGUF
		{
			GGUFContextPtr gguf;
			GGMLContextPtr ggml;
		};

		LoadedGGUF LoadGGUFContext(const std::filesystem::path& inputPath)
		{
			ggml_context* ggmlContext = nullptr;
			gguf_init_params params{};
			params.no_alloc = true;
			params.ctx = &ggmlContext;

			const auto narrowPath = inputPath.string();
			GGUFContextPtr gguf{ gguf_init_from_file(narrowPath.c_str(), params) };
			if (!gguf)
			{
				throw std::runtime_error(std::format("Failed to open GGUF file '{}'", inputPath.string()));
			}

			return { std::move(gguf), GGMLContextPtr{ ggmlContext } };
		}

		std::vector<std::size_t> TensorShape(const ggml_tensor& tensor)
		{
			const auto rank = ggml_n_dims(&tensor);
			std::vector<std::size_t> shape;
			shape.reserve(rank);
			for (int i = rank - 1; i >= 0; --i)
			{
				if (tensor.ne[i] <= 0)
				{
					throw std::runtime_error(
					    std::format("GGUF tensor '{}' has non-positive dimension {}", tensor.name, tensor.ne[i]));
				}
				shape.push_back(static_cast<std::size_t>(tensor.ne[i]));
			}
			return shape;
		}

		std::size_t TensorByteSize(const ggml_tensor& tensor)
		{
			const auto bytes = ggml_nbytes(&tensor);
			if (bytes == 0)
			{
				throw std::runtime_error(
				    std::format("GGUF tensor '{}' has an invalid serialized byte size", tensor.name));
			}
			return bytes;
		}

		std::optional<DataType> TryMapPlainDataType(ggml_type type)
		{
			switch (type)
			{
			case GGML_TYPE_F32:
				return DataType::Float32;
			case GGML_TYPE_F64:
				return DataType::Float64;
			case GGML_TYPE_F16:
				return DataType::Float16;
			case GGML_TYPE_BF16:
				return DataType::BFloat16;
			case GGML_TYPE_I8:
				return DataType::Int8;
			case GGML_TYPE_I32:
				return DataType::Int32;
			case GGML_TYPE_I64:
				return DataType::Int64;
			default:
				return std::nullopt;
			}
		}

		std::optional<QuantizedBlockFormat> TryMapQuantizedBlockFormat(ggml_type type)
		{
			switch (type)
			{
			case GGML_TYPE_Q4_0:
				return QuantizedBlockFormat::GGML_Q4_0;
			case GGML_TYPE_Q4_1:
				return QuantizedBlockFormat::GGML_Q4_1;
			case GGML_TYPE_Q5_0:
				return QuantizedBlockFormat::GGML_Q5_0;
			case GGML_TYPE_Q5_1:
				return QuantizedBlockFormat::GGML_Q5_1;
			case GGML_TYPE_Q8_0:
				return QuantizedBlockFormat::GGML_Q8_0;
			case GGML_TYPE_Q8_1:
				return QuantizedBlockFormat::GGML_Q8_1;
			case GGML_TYPE_Q2_K:
				return QuantizedBlockFormat::GGML_Q2_K;
			case GGML_TYPE_Q3_K:
				return QuantizedBlockFormat::GGML_Q3_K;
			case GGML_TYPE_Q4_K:
				return QuantizedBlockFormat::GGML_Q4_K;
			case GGML_TYPE_Q5_K:
				return QuantizedBlockFormat::GGML_Q5_K;
			case GGML_TYPE_Q6_K:
				return QuantizedBlockFormat::GGML_Q6_K;
			case GGML_TYPE_Q8_K:
				return QuantizedBlockFormat::GGML_Q8_K;
			case GGML_TYPE_IQ2_XXS:
				return QuantizedBlockFormat::GGML_IQ2_XXS;
			case GGML_TYPE_IQ2_XS:
				return QuantizedBlockFormat::GGML_IQ2_XS;
			case GGML_TYPE_IQ3_XXS:
				return QuantizedBlockFormat::GGML_IQ3_XXS;
			case GGML_TYPE_IQ1_S:
				return QuantizedBlockFormat::GGML_IQ1_S;
			case GGML_TYPE_IQ4_NL:
				return QuantizedBlockFormat::GGML_IQ4_NL;
			case GGML_TYPE_IQ3_S:
				return QuantizedBlockFormat::GGML_IQ3_S;
			case GGML_TYPE_IQ2_S:
				return QuantizedBlockFormat::GGML_IQ2_S;
			case GGML_TYPE_IQ4_XS:
				return QuantizedBlockFormat::GGML_IQ4_XS;
			default:
				return std::nullopt;
			}
		}

		void SeekTo(std::ifstream& input, std::size_t offset)
		{
			input.clear();
			input.seekg(static_cast<std::streamoff>(offset), std::ios::beg);
			if (!input)
			{
				throw std::runtime_error(std::format("Failed to seek to GGUF data offset {}", offset));
			}
		}

		void ReadExact(std::ifstream& input, void* destination, std::size_t bytes, std::string_view label)
		{
			input.read(static_cast<char*>(destination), static_cast<std::streamsize>(bytes));
			if (static_cast<std::size_t>(input.gcount()) != bytes)
			{
				throw std::runtime_error(std::format("Failed to read {} bytes for GGUF tensor '{}'", bytes, label));
			}
		}

		template <typename Src, typename Dst>
		std::vector<Dst> WidenArray(const void* data, std::size_t count)
		{
			const auto* typed = static_cast<const Src*>(data);
			std::vector<Dst> result;
			result.reserve(count);
			for (std::size_t i = 0; i < count; ++i)
			{
				result.push_back(static_cast<Dst>(typed[i]));
			}
			return result;
		}

		ModelMetadataValue ReadArrayMetadata(const gguf_context* ctx, std::int64_t keyId)
		{
			const auto arrayType = gguf_get_arr_type(ctx, keyId);
			const auto count = gguf_get_arr_n(ctx, keyId);
			if (arrayType == GGUF_TYPE_STRING)
			{
				std::vector<std::string> values;
				values.reserve(count);
				for (std::size_t i = 0; i < count; ++i)
				{
					values.emplace_back(gguf_get_arr_str(ctx, keyId, i));
				}
				return values;
			}

			const auto* data = gguf_get_arr_data(ctx, keyId);
			switch (arrayType)
			{
			case GGUF_TYPE_UINT8:
				return WidenArray<std::uint8_t, std::uint64_t>(data, count);
			case GGUF_TYPE_INT8:
				return WidenArray<std::int8_t, std::int64_t>(data, count);
			case GGUF_TYPE_UINT16:
				return WidenArray<std::uint16_t, std::uint64_t>(data, count);
			case GGUF_TYPE_INT16:
				return WidenArray<std::int16_t, std::int64_t>(data, count);
			case GGUF_TYPE_UINT32:
				return WidenArray<std::uint32_t, std::uint64_t>(data, count);
			case GGUF_TYPE_INT32:
				return WidenArray<std::int32_t, std::int64_t>(data, count);
			case GGUF_TYPE_FLOAT32:
				return WidenArray<float, double>(data, count);
			case GGUF_TYPE_UINT64:
				return WidenArray<std::uint64_t, std::uint64_t>(data, count);
			case GGUF_TYPE_INT64:
				return WidenArray<std::int64_t, std::int64_t>(data, count);
			case GGUF_TYPE_FLOAT64:
				return WidenArray<double, double>(data, count);
			case GGUF_TYPE_BOOL: {
				const auto* typed = static_cast<const std::int8_t*>(data);
				std::vector<bool> values(count);
				for (std::size_t i = 0; i < count; ++i)
				{
					values[i] = typed[i] != 0;
				}
				return values;
			}
			default:
				throw std::runtime_error(
				    std::format("Unsupported GGUF metadata array type '{}'", gguf_type_name(arrayType)));
			}
		}

		ModelMetadataValue ReadMetadataValue(const gguf_context* ctx, std::int64_t keyId)
		{
			switch (gguf_get_kv_type(ctx, keyId))
			{
			case GGUF_TYPE_UINT8:
				return static_cast<std::uint64_t>(gguf_get_val_u8(ctx, keyId));
			case GGUF_TYPE_INT8:
				return static_cast<std::int64_t>(gguf_get_val_i8(ctx, keyId));
			case GGUF_TYPE_UINT16:
				return static_cast<std::uint64_t>(gguf_get_val_u16(ctx, keyId));
			case GGUF_TYPE_INT16:
				return static_cast<std::int64_t>(gguf_get_val_i16(ctx, keyId));
			case GGUF_TYPE_UINT32:
				return static_cast<std::uint64_t>(gguf_get_val_u32(ctx, keyId));
			case GGUF_TYPE_INT32:
				return static_cast<std::int64_t>(gguf_get_val_i32(ctx, keyId));
			case GGUF_TYPE_FLOAT32:
				return static_cast<double>(gguf_get_val_f32(ctx, keyId));
			case GGUF_TYPE_BOOL:
				return gguf_get_val_bool(ctx, keyId);
			case GGUF_TYPE_STRING:
				return std::string(gguf_get_val_str(ctx, keyId));
			case GGUF_TYPE_UINT64:
				return gguf_get_val_u64(ctx, keyId);
			case GGUF_TYPE_INT64:
				return gguf_get_val_i64(ctx, keyId);
			case GGUF_TYPE_FLOAT64:
				return gguf_get_val_f64(ctx, keyId);
			case GGUF_TYPE_ARRAY:
				return ReadArrayMetadata(ctx, keyId);
			default:
				throw std::runtime_error(
				    std::format("Unsupported GGUF metadata type '{}'", gguf_type_name(gguf_get_kv_type(ctx, keyId))));
			}
		}

		void ImportMetadata(Graph& graph, const gguf_context* ctx)
		{
			std::vector<ModelMetadataEntry> metadata;
			metadata.reserve(static_cast<std::size_t>(gguf_get_n_kv(ctx)));
			for (std::int64_t keyId = 0; keyId < gguf_get_n_kv(ctx); ++keyId)
			{
				metadata.push_back({ gguf_get_key(ctx, keyId), ReadMetadataValue(ctx, keyId) });
			}
			graph.SetMetadata(std::move(metadata));
		}

		Tensor<CPU> ReadPlainTensor(std::ifstream& input, std::size_t offset, const std::vector<std::size_t>& shape,
		                            DataType dtype, std::string_view tensorName)
		{
			Tensor<CPU> tensor(Uninitialized, shape, dtype);
			SeekTo(input, offset);
			ReadExact(input, tensor.UnsafeRawData(), tensor.NumElements() * ElementByteSize(dtype), tensorName);
			return tensor;
		}

		Tensor<CPU> ReadPayloadTensor(std::ifstream& input, std::size_t offset, std::size_t bytes,
		                              std::string_view tensorName)
		{
			Tensor<CPU> tensor(Uninitialized, { bytes }, DataType::UInt8);
			SeekTo(input, offset);
			ReadExact(input, tensor.UnsafeRawData(), bytes, tensorName);
			return tensor;
		}

		void ImportTensor(Graph& graph, std::ifstream& input, const gguf_context* gguf, ggml_context* ggml,
		                  std::int64_t tensorId, std::string_view tensorName)
		{
			auto* tensor = ggml_get_tensor(ggml, std::string(tensorName).c_str());
			if (!tensor)
			{
				throw std::runtime_error(
				    std::format("GGUF tensor '{}' is missing from the ggml metadata context", tensorName));
			}

			const auto shape = TensorShape(*tensor);
			const auto expectedBytes = TensorByteSize(*tensor);
			const auto storedBytes = gguf_get_tensor_size(gguf, tensorId);
			if (expectedBytes != storedBytes)
			{
				throw std::runtime_error(std::format(
				    "GGUF tensor '{}' has inconsistent byte size: header reports {}, ggml metadata reports {}",
				    tensorName, storedBytes, expectedBytes));
			}

			const auto dataOffset = gguf_get_data_offset(gguf) + gguf_get_tensor_offset(gguf, tensorId);
			if (ggml_is_quantized(tensor->type))
			{
				const auto format = TryMapQuantizedBlockFormat(tensor->type);
				if (!format)
				{
					throw std::runtime_error(std::format("Unsupported ggml tensor type '{}' for tensor '{}'",
					                                     ggml_type_name(tensor->type), tensorName));
				}

				auto storage = ReadPayloadTensor(input, dataOffset, storedBytes, tensorName);
				auto quantization = BlockQuantization(*format, shape, DataType::Float32);
				graph.AddVariable(Variable::CreateFrozenQuantized(std::move(storage), std::move(quantization)));
				return;
			}

			const auto dtype = TryMapPlainDataType(tensor->type);
			if (!dtype)
			{
				throw std::runtime_error(std::format("Unsupported ggml tensor type '{}' for tensor '{}'",
				                                     ggml_type_name(tensor->type), tensorName));
			}

			auto plainTensor = ReadPlainTensor(input, dataOffset, shape, *dtype, tensorName);
			graph.AddVariable(Variable::CreateFrozen(std::move(plainTensor)));
		}
	} // namespace

	std::string_view LLaMACompatibilityProfileName(LLaMACompatibilityProfileKind kind)
	{
		switch (kind)
		{
		case LLaMACompatibilityProfileKind::TinyFixture:
			return "tiny-fixture";
		case LLaMACompatibilityProfileKind::LLaMA2LikeCausalLM:
			return "llama2-like-causal-lm";
		case LLaMACompatibilityProfileKind::LLaMA3LikeCausalLM:
			return "llama3-like-causal-lm";
		case LLaMACompatibilityProfileKind::Qwen2LikeCausalLM:
			return "qwen2-like-causal-lm";
		}
		return "unknown";
	}

	LLaMACompatibilityProfileDescriptor QueryLLaMACompatibilityProfile(LLaMACompatibilityProfileKind kind)
	{
		switch (kind)
		{
		case LLaMACompatibilityProfileKind::TinyFixture:
			return { kind,
				     LLaMACompatibilityProfileName(kind),
				     "llama",
				     false,
				     true,
				     true,
				     true,
				     false,
				     true,
				     false,
				     "Self-contained tiny LLaMA-family fixtures used for metadata, lowering, AOT, and deterministic "
				     "logit regression tests.",
				     "Unsupported metadata, tensor shapes, and quantization formats fail with importer/lowering "
				     "diagnostics.",
				     "Internal fixture parity is sufficient only for regression coverage, not production model "
				     "acceptance." };
		case LLaMACompatibilityProfileKind::LLaMA2LikeCausalLM:
			return { kind,
				     LLaMACompatibilityProfileName(kind),
				     "llama",
				     true,
				     true,
				     true,
				     true,
				     false,
				     true,
				     true,
				     "LLaMA-family causal LM archives with static-shape prefill/decode, token_embd/output weights, "
				     "RMSNorm, SwiGLU MLP, GQA/MQA head layout, none/linear RoPE, and ggml block weights "
				     "dequantized during import.",
				     "Unsupported llama.cpp ops, tensor layouts, RoPE variants, state ABI, or quantization formats are "
				     "rejected instead of guessed.",
				     "A new production profile requires external llama.cpp golden logits for prefill and decode plus "
				     "dtype/quantization-specific tolerance." };
		case LLaMACompatibilityProfileKind::LLaMA3LikeCausalLM:
			return { kind,
				     LLaMACompatibilityProfileName(kind),
				     "llama",
				     true,
				     true,
				     true,
				     true,
				     false,
				     true,
				     true,
				     "LLaMA-3-like causal LM archives that stay inside the same static causal-LM contract as the "
				     "LLaMA2-like profile: GQA, RMSNorm, SwiGLU, tokenizer metadata preservation, and none/linear "
				     "RoPE execution.",
				     "YaRN/LongRoPE, sliding-window attention, mixture-of-experts, custom tokenizer runtime behavior, "
				     "and unsupported block layouts remain blocking diagnostics.",
				     "Acceptance requires external llama.cpp golden logits for representative prompt, prefill, and "
				     "decode cases before claiming production support." };
		case LLaMACompatibilityProfileKind::Qwen2LikeCausalLM:
			return { kind,
				     LLaMACompatibilityProfileName(kind),
				     "qwen2",
				     false,
				     true,
				     true,
				     true,
				     true,
				     true,
				     true,
				     "Qwen2/Qwen2.5-style causal LM GGUF archives that use the LLaMA-like decoder skeleton but require "
				     "Qwen-specific tokenizer/config validation, RoPE scaling semantics, native quantized projection "
				     "execution, and external logits before production use.",
				     "Current LiteNN support is an analysis/lowering target only: tokenizer runtime, chat template, "
				     "long-context RoPE variants, full decode-loop ABI, and external golden-logit evidence must remain "
				     "explicit diagnostics. Q4_K_M projection formats are now reported as native CUDA quantized paths.",
				     "Acceptance requires a real Qwen2.5 GGUF fixture, llama.cpp golden prefill/decode logits, "
				     "tokenizer/chat-template parity, and CUDA/native or explicitly configured fallback evidence." };
		}
		return { kind,
			     "unknown",
			     "unknown",
			     false,
			     false,
			     false,
			     false,
			     false,
			     false,
			     true,
			     "Unknown profile.",
			     "Unknown profile.",
			     "Unknown profile." };
	}

	std::vector<LLaMACompatibilityProfileDescriptor> QueryLLaMACompatibilityProfiles()
	{
		return {
			QueryLLaMACompatibilityProfile(LLaMACompatibilityProfileKind::TinyFixture),
			QueryLLaMACompatibilityProfile(LLaMACompatibilityProfileKind::LLaMA2LikeCausalLM),
			QueryLLaMACompatibilityProfile(LLaMACompatibilityProfileKind::LLaMA3LikeCausalLM),
			QueryLLaMACompatibilityProfile(LLaMACompatibilityProfileKind::Qwen2LikeCausalLM),
		};
	}

	std::optional<LLaMACompatibilityProfileKind> TryInferLLaMACompatibilityProfile(std::string_view architecture)
	{
		if (architecture == "llama")
		{
			return LLaMACompatibilityProfileKind::LLaMA2LikeCausalLM;
		}
		if (architecture == "qwen2")
		{
			return LLaMACompatibilityProfileKind::Qwen2LikeCausalLM;
		}
		if (architecture == "qwen2moe" || architecture == "mistral" || architecture == "gemma")
		{
			return std::nullopt;
		}
		return std::nullopt;
	}

	LLMTokenizerMetadataSummary SummarizeLLMTokenizerMetadata(const Graph& graph)
	{
		const auto readIntMetadata = [&graph](std::string_view key) -> std::optional<std::int64_t> {
			if (const auto entry = FindMetadata(graph, key))
			{
				if (const auto* value = std::get_if<std::int64_t>(&(*entry)->value))
				{
					return *value;
				}
				if (const auto* value = std::get_if<std::uint64_t>(&(*entry)->value))
				{
					if (*value <= static_cast<std::uint64_t>(std::numeric_limits<std::int64_t>::max()))
					{
						return static_cast<std::int64_t>(*value);
					}
				}
			}
			return std::nullopt;
		};

		LLMTokenizerMetadataSummary summary;
		if (const auto model = FindMetadata(graph, "tokenizer.ggml.model"))
		{
			if (const auto* value = std::get_if<std::string>(&(*model)->value))
			{
				summary.model = *value;
			}
		}
		if (const auto tokens = FindMetadata(graph, "tokenizer.ggml.tokens"))
		{
			if (const auto* value = std::get_if<std::vector<std::string>>(&(*tokens)->value))
			{
				summary.tokenCount = value->size();
			}
		}
		if (const auto tokenTypes = FindMetadata(graph, "tokenizer.ggml.token_type"))
		{
			if (const auto* value = std::get_if<std::vector<std::int64_t>>(&(*tokenTypes)->value))
			{
				summary.tokenTypeCount = value->size();
			}
			else if (const auto* unsignedValue = std::get_if<std::vector<std::uint64_t>>(&(*tokenTypes)->value))
			{
				summary.tokenTypeCount = unsignedValue->size();
			}
		}
		for (const auto key : { "tokenizer.chat_template", "tokenizer.ggml.chat_template" })
		{
			if (const auto chatTemplate = FindMetadata(graph, key))
			{
				if (const auto* value = std::get_if<std::string>(&(*chatTemplate)->value))
				{
					summary.hasChatTemplate = true;
					summary.chatTemplateBytes = value->size();
					break;
				}
			}
		}
		summary.bosTokenId = readIntMetadata("tokenizer.ggml.bos_token_id");
		summary.eosTokenId = readIntMetadata("tokenizer.ggml.eos_token_id");
		summary.unknownTokenId = readIntMetadata("tokenizer.ggml.unknown_token_id");
		summary.hasBosTokenId = summary.bosTokenId.has_value();
		summary.hasEosTokenId = summary.eosTokenId.has_value();
		summary.hasUnknownTokenId = summary.unknownTokenId.has_value();
		return summary;
	}

	std::string_view LLaMAQuantizedExecutionPolicyName(LLaMAQuantizedExecutionPolicy policy)
	{
		switch (policy)
		{
		case LLaMAQuantizedExecutionPolicy::None:
			return "none";
		case LLaMAQuantizedExecutionPolicy::Reject:
			return "reject";
		case LLaMAQuantizedExecutionPolicy::CPUNativeQuantized:
			return "cpu-native-quantized";
		case LLaMAQuantizedExecutionPolicy::CPUReferenceDequantize:
			return "cpu-reference-dequantize";
		case LLaMAQuantizedExecutionPolicy::CUDADequantizeThenGEMM:
			return "cuda-dequantize-then-gemm";
		case LLaMAQuantizedExecutionPolicy::CUDANativeQuantized:
			return "cuda-native-quantized";
		}
		return "unknown";
	}

	LLaMAQuantizedExecutionPlan PlanLLaMAQuantizedWeightExecution(const Graph& archive,
	                                                              std::size_t dequantizedMemoryBudgetBytes)
	{
		struct Stats
		{
			std::size_t tensorCount{};
			std::size_t storedBytes{};
			std::size_t dequantizedBytes{};
		};

		const auto addChecked = [](std::size_t lhs, std::size_t rhs, std::string_view label) {
			if (rhs > std::numeric_limits<std::size_t>::max() - lhs)
			{
				throw std::runtime_error(std::format("LLaMA quantized execution {} byte count overflow", label));
			}
			return lhs + rhs;
		};
		const auto multiplyChecked = [](std::span<const std::size_t> dims, std::size_t elementBytes) {
			std::size_t elements = 1;
			for (const auto dim : dims)
			{
				if (dim != 0 && elements > std::numeric_limits<std::size_t>::max() / dim)
				{
					throw std::runtime_error("LLaMA quantized execution element count overflow");
				}
				elements *= dim;
			}
			if (elementBytes != 0 && elements > std::numeric_limits<std::size_t>::max() / elementBytes)
			{
				throw std::runtime_error("LLaMA quantized execution byte count overflow");
			}
			return elements * elementBytes;
		};

		std::map<QuantizedBlockFormat, Stats> statsByFormat;
		for (std::size_t variableIndex = 0; variableIndex < archive.VariableCount(); ++variableIndex)
		{
			const auto& variable = *archive.GetVariable(variableIndex);
			if (!variable.IsQuantized())
			{
				continue;
			}
			const auto& params = *variable.Quantization();
			auto& stats = statsByFormat[params.blockFormat];
			++stats.tensorCount;
			stats.storedBytes = addChecked(
			    stats.storedBytes, variable.Data().NumElements() * ElementByteSize(variable.Data().DType()), "stored");
			stats.dequantizedBytes = addChecked(
			    stats.dequantizedBytes, multiplyChecked(params.expressedShape, ElementByteSize(params.expressedType)),
			    "dequantized");
		}

		LLaMAQuantizedExecutionPlan plan{
			.dequantizedMemoryBudgetBytes = dequantizedMemoryBudgetBytes,
		};
		for (const auto& [format, stats] : statsByFormat)
		{
			plan.tensorCount = addChecked(plan.tensorCount, stats.tensorCount, "tensor");
			plan.storedBytes = addChecked(plan.storedBytes, stats.storedBytes, "stored");
			plan.dequantizedBytes = addChecked(plan.dequantizedBytes, stats.dequantizedBytes, "dequantized");
			const auto nativeKQuant =
			    format == QuantizedBlockFormat::GGML_Q4_K || format == QuantizedBlockFormat::GGML_Q5_K ||
			    format == QuantizedBlockFormat::GGML_Q6_K || format == QuantizedBlockFormat::GGML_Q8_0;
			const auto overBudget = !nativeKQuant && dequantizedMemoryBudgetBytes != 0 &&
			                        stats.dequantizedBytes > dequantizedMemoryBudgetBytes;
			const auto policy = overBudget     ? LLaMAQuantizedExecutionPolicy::Reject
			                    : nativeKQuant ? LLaMAQuantizedExecutionPolicy::CUDANativeQuantized
			                                   : LLaMAQuantizedExecutionPolicy::CPUReferenceDequantize;
			if (overBudget)
			{
				plan.lowerable = false;
			}
			plan.decisions.push_back({
			    .format = format,
			    .tensorCount = stats.tensorCount,
			    .storedBytes = stats.storedBytes,
			    .dequantizedBytes = stats.dequantizedBytes,
			    .selectedPolicy = policy,
			    .blocking = overBudget,
			    .reason = nativeKQuant
			                  ? "CPU Interpreter/AOT and CUDA native projection kernels consume the original GGML "
			                    "block payload directly"
			              : overBudget
			                  ? std::format("dequantized bytes {} exceed budget {}", stats.dequantizedBytes,
			                                dequantizedMemoryBudgetBytes)
			                  : "CPU reference dequantization is required for this block format; CUDA native quantized "
			                    "kernels are not selected yet",
			});
		}
		return plan;
	}

	LLaMACompatibilityReport AnalyzeLLaMACompatibility(const Graph& archive, LLaMACompatibilityProfileKind kind,
	                                                   std::size_t dequantizedMemoryBudgetBytes)
	{
		LLaMACompatibilityReport report{
			.profile = QueryLLaMACompatibilityProfile(kind),
			.lowerable = true,
			.externalGoldenRequired = QueryLLaMACompatibilityProfile(kind).requiresExternalLLaMACppGolden,
		};

		const auto addDiagnostic = [&report](std::string subject, std::string message, bool blocking) {
			if (blocking)
			{
				report.lowerable = false;
			}
			report.diagnostics.push_back({
			    .subject = std::move(subject),
			    .message = std::move(message),
			    .blocking = blocking,
			});
		};

		std::optional<LLaMAHyperparameters> hyperparameters;
		try
		{
			hyperparameters = ParseLLaMAHyperparameters(archive);
		}
		catch (const std::exception& ex)
		{
			addDiagnostic("metadata", ex.what(), true);
			return report;
		}

		if (hyperparameters->architecture != report.profile.architecture)
		{
			addDiagnostic("general.architecture",
			              std::format("Profile '{}' expects architecture '{}', got '{}'", report.profile.name,
			                          report.profile.architecture, hyperparameters->architecture),
			              true);
		}
		if (kind == LLaMACompatibilityProfileKind::Qwen2LikeCausalLM)
		{
			const auto tokenizer = SummarizeLLMTokenizerMetadata(archive);
			addDiagnostic(
			    "qwen2.tokenizer",
			    std::format("Qwen2 tokenizer metadata: model={}, tokens={}, token_types={}, "
			                "chat_template={} ({} bytes), bos={}, eos={}, unk={}. Token-id parity against "
			                "llama.cpp is still required; current LiteNN lowering accepts token ids directly.",
			                tokenizer.model.value_or("<missing>"), tokenizer.tokenCount, tokenizer.tokenTypeCount,
			                tokenizer.hasChatTemplate ? "yes" : "no", tokenizer.chatTemplateBytes,
			                tokenizer.hasBosTokenId ? "yes" : "no", tokenizer.hasEosTokenId ? "yes" : "no",
			                tokenizer.hasUnknownTokenId ? "yes" : "no"),
			    false);
			addDiagnostic("qwen2.quantized-cuda",
			              "Qwen2 Q4_K_M projection formats (Q4_K/Q6_K, plus Q5_K/Q8_0 coverage) are reported as "
			              "native CUDA quantized paths for projection kernels. Full-model acceptance still requires "
			              "decode-loop coverage, tokenizer parity, and external golden-logit evidence.",
			              false);
			addDiagnostic("qwen2.decode-loop",
			              "LiteNN provides tokenizer-adapter integration, sampler control, mutable KV-cache rebinding, "
			              "and EOS-aware decode-loop tooling; external llama.cpp golden logits remain required for "
			              "production acceptance.",
			              false);
		}
		if (hyperparameters->ropeScalingType != "none" && hyperparameters->ropeScalingType != "linear")
		{
			addDiagnostic(std::format("{}.rope.scaling.type", hyperparameters->architecture),
			              std::format("Current LiteNN LLaMA lowering preserves '{}' metadata but only executes "
			                          "none/linear RoPE scaling; use an external golden-gated profile before enabling "
			                          "this variant.",
			                          hyperparameters->ropeScalingType),
			              true);
		}
		if (hyperparameters->ropeDimensionCount == 0 ||
		    hyperparameters->ropeDimensionCount > hyperparameters->HeadDimension() ||
		    (hyperparameters->ropeDimensionCount % 2) != 0)
		{
			addDiagnostic(std::format("{}.rope.dimension_count", hyperparameters->architecture),
			              "RoPE dimension count must be an even value in [2, headDim] for current lowering.", true);
		}

		const auto quantizedExecution = PlanLLaMAQuantizedWeightExecution(archive, dequantizedMemoryBudgetBytes);
		if (!quantizedExecution.decisions.empty())
		{
			std::string summary;
			for (const auto& decision : quantizedExecution.decisions)
			{
				if (!summary.empty())
				{
					summary += ", ";
				}
				summary +=
				    std::format("{}:{} tensors/stored={} bytes/dequantized={} bytes/policy={}",
				                QuantizedBlockFormatName(decision.format), decision.tensorCount, decision.storedBytes,
				                decision.dequantizedBytes, LLaMAQuantizedExecutionPolicyName(decision.selectedPolicy));
			}
			addDiagnostic(
			    "quantization.mix",
			    std::format("GGUF archive contains block-quantized weights: {}. LiteNN preserves block payloads "
			                "for reported native policies; unsupported formats still require explicit fallback.",
			                summary),
			    !quantizedExecution.lowerable);
		}
		if (std::ranges::any_of(quantizedExecution.decisions, [](const auto& decision) {
			    return decision.format == QuantizedBlockFormat::GGML_Q4_K;
		    }))
		{
			addDiagnostic("quantization.q4_k_m",
			              "Detected GGML_Q4_K tensors, which are part of common Q4_K_M model mixes. CPU native block "
			              "execution and CUDA native K-quant projection kernels are available for Q4_K/Q6_K payloads; "
			              "full-model production still needs golden-logit and decode-loop gates.",
			              false);
		}

		const auto requireTensor = [&](std::string name) {
			if (!archive.FindVariable(name))
			{
				addDiagnostic(std::move(name), "Required LLaMA tensor is missing from the GGUF archive.", true);
			}
		};
		requireTensor("token_embd.weight");
		requireTensor("output_norm.weight");
		for (std::size_t blockIndex = 0; blockIndex < hyperparameters->blockCount; ++blockIndex)
		{
			const auto prefix = std::format("blk.{}.", blockIndex);
			requireTensor(prefix + "attn_norm.weight");
			requireTensor(prefix + "attn_q.weight");
			requireTensor(prefix + "attn_k.weight");
			requireTensor(prefix + "attn_v.weight");
			requireTensor(prefix + "attn_output.weight");
			requireTensor(prefix + "ffn_norm.weight");
			requireTensor(prefix + "ffn_gate.weight");
			requireTensor(prefix + "ffn_up.weight");
			requireTensor(prefix + "ffn_down.weight");
		}

		if (report.externalGoldenRequired)
		{
			addDiagnostic("external-golden",
			              "Production acceptance requires external llama.cpp golden logits for matching prompt, "
			              "prefill/decode shape, dtype, and quantization profile.",
			              false);
		}
		return report;
	}

	LLaMAHyperparameters ParseLLaMAHyperparameters(const Graph& graph)
	{
		const auto architecture = ReadStringValue(RequireMetadata(graph, "general.architecture"));
		if (architecture.empty())
		{
			throw std::runtime_error("GGUF metadata key 'general.architecture' must not be empty");
		}

		const auto key = [&architecture](std::string_view suffix) {
			return std::format("{}.{}", architecture, suffix);
		};

		LLaMAHyperparameters hyperparameters{
			.architecture = architecture,
			.ropeLayout = RoPELayoutForArchitecture(architecture),
			.contextLength = ReadSizeValue(RequireMetadata(graph, key("context_length"))),
			.embeddingLength = ReadSizeValue(RequireMetadata(graph, key("embedding_length"))),
			.blockCount = ReadSizeValue(RequireMetadata(graph, key("block_count"))),
			.feedForwardLength = ReadSizeValue(RequireMetadata(graph, key("feed_forward_length"))),
			.attentionHeadCount = ReadSizeValue(RequireMetadata(graph, key("attention.head_count"))),
			.attentionHeadCountKV = 0,
			.rmsNormEpsilon = ReadDoubleValue(RequireMetadata(graph, key("attention.layer_norm_rms_epsilon"))),
			.ropeFrequencyBase = 10000.0,
			.ropeFrequencyScale = 1.0,
			.ropeDimensionCount = 0,
			.ropeScalingType = "none",
		};

		if (const auto headCountKV = FindMetadata(graph, key("attention.head_count_kv")))
		{
			hyperparameters.attentionHeadCountKV = ReadSizeValue(**headCountKV);
		}
		else
		{
			hyperparameters.attentionHeadCountKV = hyperparameters.attentionHeadCount;
		}

		if (const auto ropeFrequencyBase = FindMetadata(graph, key("rope.freq_base")))
		{
			hyperparameters.ropeFrequencyBase = ReadDoubleValue(**ropeFrequencyBase);
		}
		if (const auto ropeFrequencyScale = FindMetadata(graph, key("rope.freq_scale")))
		{
			hyperparameters.ropeFrequencyScale = ReadDoubleValue(**ropeFrequencyScale);
		}
		if (const auto ropeDimensionCount = FindMetadata(graph, key("rope.dimension_count")))
		{
			hyperparameters.ropeDimensionCount = ReadSizeValue(**ropeDimensionCount);
		}
		if (const auto ropeScalingType = FindMetadata(graph, key("rope.scaling.type")))
		{
			hyperparameters.ropeScalingType = NormalizeRopeScalingType(ReadStringValue(**ropeScalingType));
		}
		if (const auto ropeScalingFactor = FindMetadata(graph, key("rope.scaling.factor")))
		{
			hyperparameters.ropeScalingFactor = ReadDoubleValue(**ropeScalingFactor);
			if (*hyperparameters.ropeScalingFactor < 0.0)
			{
				throw std::runtime_error("LLaMA rope.scaling.factor must be non-negative");
			}
			hyperparameters.ropeFrequencyScale =
			    *hyperparameters.ropeScalingFactor == 0.0 ? 1.0 : 1.0 / *hyperparameters.ropeScalingFactor;
		}
		if (const auto ropeScalingAlpha = FindMetadata(graph, key("rope.scaling.alpha")))
		{
			hyperparameters.ropeScalingAlpha = ReadDoubleValue(**ropeScalingAlpha);
		}
		if (const auto ropeScalingAttentionFactor = FindMetadata(graph, key("rope.scaling.attn_factor")))
		{
			hyperparameters.ropeScalingAttentionFactor = ReadDoubleValue(**ropeScalingAttentionFactor);
		}
		if (const auto ropeScalingOriginalContextLength =
		        FindMetadata(graph, key("rope.scaling.original_context_length")))
		{
			hyperparameters.ropeScalingOriginalContextLength = ReadSizeValue(**ropeScalingOriginalContextLength);
		}
		if (const auto ropeScalingFinetuned = FindMetadata(graph, key("rope.scaling.finetuned")))
		{
			hyperparameters.ropeScalingFinetuned = ReadBoolValue(**ropeScalingFinetuned);
		}
		if (const auto ropeScalingYarnLogMultiplier = FindMetadata(graph, key("rope.scaling.yarn_log_multiplier")))
		{
			hyperparameters.ropeScalingYarnLogMultiplier = ReadDoubleValue(**ropeScalingYarnLogMultiplier);
		}
		if (const auto ropeScalingYarnExtFactor = FindMetadata(graph, key("rope.scaling.yarn_ext_factor")))
		{
			hyperparameters.ropeScalingYarnExtFactor = ReadDoubleValue(**ropeScalingYarnExtFactor);
		}
		if (const auto ropeScalingYarnAttentionFactor = FindMetadata(graph, key("rope.scaling.yarn_attn_factor")))
		{
			hyperparameters.ropeScalingYarnAttentionFactor = ReadDoubleValue(**ropeScalingYarnAttentionFactor);
		}
		if (const auto ropeScalingYarnBetaFast = FindMetadata(graph, key("rope.scaling.yarn_beta_fast")))
		{
			hyperparameters.ropeScalingYarnBetaFast = ReadDoubleValue(**ropeScalingYarnBetaFast);
		}
		if (const auto ropeScalingYarnBetaSlow = FindMetadata(graph, key("rope.scaling.yarn_beta_slow")))
		{
			hyperparameters.ropeScalingYarnBetaSlow = ReadDoubleValue(**ropeScalingYarnBetaSlow);
		}

		if (hyperparameters.contextLength == 0 || hyperparameters.embeddingLength == 0 ||
		    hyperparameters.blockCount == 0 || hyperparameters.feedForwardLength == 0)
		{
			throw std::runtime_error("LLaMA hyperparameters must be greater than zero");
		}
		if (!(hyperparameters.rmsNormEpsilon > 0.0))
		{
			throw std::runtime_error("LLaMA attention.layer_norm_rms_epsilon must be greater than zero");
		}
		if (!(hyperparameters.ropeFrequencyBase > 0.0))
		{
			throw std::runtime_error("LLaMA rope.freq_base must be greater than zero");
		}
		if (!(hyperparameters.ropeFrequencyScale > 0.0))
		{
			throw std::runtime_error("LLaMA rope.freq_scale must be greater than zero");
		}
		if (hyperparameters.ropeScalingType != "none" && hyperparameters.ropeScalingType != "linear" &&
		    hyperparameters.ropeScalingType != "yarn" && hyperparameters.ropeScalingType != "longrope")
		{
			throw std::runtime_error(
			    std::format("Unsupported LLaMA rope.scaling.type '{}'", hyperparameters.ropeScalingType));
		}

		const auto headDimension = hyperparameters.HeadDimension();
		if (hyperparameters.ropeDimensionCount == 0)
		{
			hyperparameters.ropeDimensionCount = headDimension;
		}
		if (hyperparameters.ropeDimensionCount > headDimension || (hyperparameters.ropeDimensionCount % 2) != 0)
		{
			throw std::runtime_error("LLaMA rope.dimension_count must be an even value in [2, headDim]");
		}
		static_cast<void>(hyperparameters.QueryGroupsPerKVHead());
		return hyperparameters;
	}

	LLaMAContextValidationReport ValidateLLaMAContextRequest(const LLaMAHyperparameters& hyperparameters,
	                                                         std::size_t requestedTokenCount,
	                                                         std::size_t maxCacheLength)
	{
		LLaMAContextValidationReport report{
			.requestedTokenCount = requestedTokenCount,
			.maxCacheLength = maxCacheLength,
			.modelContextLength = hyperparameters.contextLength,
			.trainedContextLength =
			    hyperparameters.ropeScalingOriginalContextLength.value_or(hyperparameters.contextLength),
			.ropeScalingType = hyperparameters.ropeScalingType,
		};
		const auto addDiagnostic = [&report](std::string subject, std::string message, bool blocking) {
			if (blocking)
			{
				report.accepted = false;
			}
			report.diagnostics.push_back({
			    .subject = std::move(subject),
			    .message = std::move(message),
			    .blocking = blocking,
			});
		};

		if (requestedTokenCount == 0 || maxCacheLength == 0)
		{
			addDiagnostic("decode.context", "Requested token count and max cache length must be positive.", true);
			return report;
		}
		if (maxCacheLength < requestedTokenCount)
		{
			addDiagnostic("decode.max_cache_length",
			              std::format("max-cache-length {} is smaller than requested token count {}", maxCacheLength,
			                          requestedTokenCount),
			              true);
		}
		if (hyperparameters.contextLength > 0 && requestedTokenCount > hyperparameters.contextLength)
		{
			addDiagnostic("decode.requested_token_count",
			              std::format("requested {} total tokens but model context length is {}", requestedTokenCount,
			                          hyperparameters.contextLength),
			              true);
		}
		if (hyperparameters.contextLength > 0 && maxCacheLength > hyperparameters.contextLength)
		{
			addDiagnostic("decode.max_cache_length",
			              std::format("max-cache-length {} exceeds model context length {}", maxCacheLength,
			                          hyperparameters.contextLength),
			              true);
		}

		const auto validationLength = std::max(requestedTokenCount, maxCacheLength);
		report.usesContextExtension = report.trainedContextLength > 0 && validationLength > report.trainedContextLength;
		if (!report.usesContextExtension)
		{
			return report;
		}

		const auto scalingSubject = std::format("{}.rope.scaling.type", hyperparameters.architecture);
		if (hyperparameters.ropeScalingType == "none")
		{
			addDiagnostic(scalingSubject,
			              std::format("requested context {} exceeds trained context {} but RoPE scaling is absent",
			                          validationLength, report.trainedContextLength),
			              true);
			return report;
		}
		if (hyperparameters.ropeScalingType != "linear")
		{
			addDiagnostic(scalingSubject,
			              std::format("requested context {} requires '{}' RoPE scaling, but current LiteNN decode "
			                          "execution only validates none/linear scaling against long-context gates",
			                          validationLength, hyperparameters.ropeScalingType),
			              true);
			return report;
		}
		if (!hyperparameters.ropeScalingFactor)
		{
			addDiagnostic(std::format("{}.rope.scaling.factor", hyperparameters.architecture),
			              "linear RoPE context extension requires rope.scaling.factor metadata.", true);
			return report;
		}
		if (!(*hyperparameters.ropeScalingFactor > 1.0))
		{
			addDiagnostic(std::format("{}.rope.scaling.factor", hyperparameters.architecture),
			              std::format("linear RoPE scaling factor {} does not extend the trained context.",
			                          *hyperparameters.ropeScalingFactor),
			              true);
		}
		const auto scaledLimit = static_cast<double>(report.trainedContextLength) * *hyperparameters.ropeScalingFactor;
		if (static_cast<double>(validationLength) > scaledLimit + 0.5)
		{
			addDiagnostic(std::format("{}.rope.scaling.factor", hyperparameters.architecture),
			              std::format("requested context {} exceeds trained_context * scaling_factor ({:.0f})",
			                          validationLength, scaledLimit),
			              true);
		}
		if (!hyperparameters.ropeScalingFinetuned.value_or(false))
		{
			addDiagnostic(std::format("{}.rope.scaling.finetuned", hyperparameters.architecture),
			              "linear RoPE context extension is not marked finetuned; require external long-context golden "
			              "evidence before production acceptance.",
			              false);
		}
		return report;
	}

	ImportResult ImportGGUFArchive(const std::filesystem::path& inputPath)
	{
		auto loaded = LoadGGUFContext(inputPath);

		std::ifstream input(inputPath, std::ios::binary);
		if (!input)
		{
			throw std::runtime_error(std::format("Failed to open GGUF payload stream '{}'", inputPath.string()));
		}

		Graph graph;
		ImportMetadata(graph, loaded.gguf.get());

		const auto tensorCount = gguf_get_n_tensors(loaded.gguf.get());
		std::vector<std::string> variableNames;
		variableNames.reserve(static_cast<std::size_t>(tensorCount));
		for (std::int64_t tensorId = 0; tensorId < tensorCount; ++tensorId)
		{
			std::string tensorName = gguf_get_tensor_name(loaded.gguf.get(), tensorId);
			ImportTensor(graph, input, loaded.gguf.get(), loaded.ggml.get(), tensorId, tensorName);
			variableNames.push_back(std::move(tensorName));
		}
		graph.SetVariableNames(std::move(variableNames));

		Subgraph archive;
		graph.SetForward(graph.AddSubgraph(std::move(archive)));

		return {
			.model = ModelGraph(std::move(graph)),
			.summary = {
				.tensorCount = static_cast<std::size_t>(tensorCount),
				.metadataCount = static_cast<std::size_t>(gguf_get_n_kv(loaded.gguf.get())),
			},
		};
	}

	ImportResult ImportGGUFMetadata(const std::filesystem::path& inputPath)
	{
		auto loaded = LoadGGUFContext(inputPath);

		Graph graph;
		ImportMetadata(graph, loaded.gguf.get());
		graph.SetForward(graph.AddSubgraph(Subgraph{}));

		return {
			.model = ModelGraph(std::move(graph)),
			.summary = {
				.tensorCount = static_cast<std::size_t>(gguf_get_n_tensors(loaded.gguf.get())),
				.metadataCount = static_cast<std::size_t>(gguf_get_n_kv(loaded.gguf.get())),
			},
		};
	}

	ImportSummary ConvertGGUFArchive(const std::filesystem::path& inputPath, const std::filesystem::path& outputPath)
	{
		auto result = ImportGGUFArchive(inputPath);
		Serialization::SaveVNextModelPackage(Detail::BuildExecutableModuleFromGraph(result.model.UnsafeGraphView()),
		                                     outputPath);
		return result.summary;
	}

	ImportSummary ConvertGGUFArchiveExternalWeights(const std::filesystem::path& inputPath,
	                                                const std::filesystem::path& outputPath,
	                                                const std::filesystem::path& weightsPath,
	                                                const Serialization::ExternalWeightSaveOptions& options)
	{
		auto result = ImportGGUFArchive(inputPath);
		Serialization::SaveVNextModelPackageExternalWeights(result.model.UnsafeGraphView(), outputPath, weightsPath,
		                                                    options);
		return result.summary;
	}
} // namespace LiteNN::GGUF
