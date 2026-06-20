#include <gtest/gtest.h>

#include <GGUFImporter.h>
#include <LLMGeneration.h>
#include <LLaMABuilder.h>

#ifdef LITENN_ENABLE_MLIR
#include <LiteNN/Compiler/CompiledModule.h>
#endif
#include <LiteNN/Runtime/Interpreter.h>
#include <LiteNN/Serialization/ModelPackageIO.h>

#include <algorithm>
#include <array>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <filesystem>
#include <format>
#include <memory>
#include <span>
#include <stdexcept>
#include <string>
#include <string_view>
#include <vector>

#include <ggml.h>
#include <gguf.h>

using namespace LiteNN;

namespace
{
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

	float ReadFloat(const Tensor<CPU>& tensor, std::size_t index)
	{
		const auto cpuTensor = tensor.CopyToDevice(CPU{});
		return static_cast<const float*>(cpuTensor.UnsafeRawData())[index];
	}

	float ReadFloat(const Tensor<PolymorphicDevice>& tensor, std::size_t index)
	{
		return ReadFloat(tensor.CopyToDevice(CPU{}), index);
	}

	template <Device D>
	void ExpectTensorNear(const Tensor<D>& tensor, std::span<const float> expected,
	                      GGUF::LLaMAParityTolerance tolerance)
	{
		ASSERT_EQ(tensor.NumElements(), expected.size());
		for (std::size_t i = 0; i < expected.size(); ++i)
		{
			const auto actual = ReadFloat(tensor, i);
			const auto allowed = tolerance.absolute + tolerance.relative * std::fabs(expected[i]);
			EXPECT_LE(std::fabs(actual - expected[i]), allowed) << "at element " << i;
		}
	}

	template <Device ActualDevice, Device ExpectedDevice>
	void ExpectTensorNear(const Tensor<ActualDevice>& actual, const Tensor<ExpectedDevice>& expected,
	                      GGUF::LLaMAParityTolerance tolerance)
	{
		const auto actualCpu = actual.CopyToDevice(CPU{});
		const auto expectedCpu = expected.CopyToDevice(CPU{});
		ASSERT_EQ(actualCpu.Shape().ToOwned(), expectedCpu.Shape().ToOwned());
		ASSERT_EQ(actualCpu.NumElements(), expectedCpu.NumElements());
		for (std::size_t i = 0; i < actualCpu.NumElements(); ++i)
		{
			const auto actualValue = ReadFloat(actualCpu, i);
			const auto expectedValue = ReadFloat(expectedCpu, i);
			const auto allowed = tolerance.absolute + tolerance.relative * std::fabs(expectedValue);
			EXPECT_LE(std::fabs(actualValue - expectedValue), allowed) << "at element " << i;
		}
	}

	Tensor<CPU> MakeInt32Tensor(std::initializer_list<std::int32_t> values, std::initializer_list<std::size_t> shape)
	{
		CPU device;
		Tensor<CPU> tensor(Uninitialized, shape, DataType::Int32, device);
		DeviceTraits<CPU>::CopyFromCPU(device, DataType::Int32, tensor.UnsafeRawData(), DataType::Int32, values.begin(),
		                               values.size());
		return tensor;
	}

	Tensor<CPU> MakeFloatTensor(std::span<const float> values, std::initializer_list<std::size_t> shape)
	{
		CPU device;
		Tensor<CPU> tensor(Uninitialized, shape, DataType::Float32, device);
		DeviceTraits<CPU>::CopyFromCPU(device, DataType::Float32, tensor.UnsafeRawData(), DataType::Float32,
		                               values.data(), values.size());
		return tensor;
	}

	GGMLContextPtr CreateTensorContext()
	{
		ggml_init_params params{};
		params.mem_size = ggml_tensor_overhead() * 8;
		params.no_alloc = true;
		return GGMLContextPtr{ ggml_init(params) };
	}

	std::filesystem::path MakeTempFixturePath(std::string_view stem, std::string_view extension)
	{
		const auto now = std::chrono::steady_clock::now().time_since_epoch().count();
		const auto salt = reinterpret_cast<std::uintptr_t>(&stem);
		return std::filesystem::temp_directory_path() / std::format("{}_{}_{}{}", stem, now, salt, extension);
	}

	void AddTensor(gguf_context* gguf, ggml_context* ggml, ggml_type type, std::string_view name,
	               std::span<const std::int64_t> dims, const void* data)
	{
		auto* tensor = ggml_new_tensor(ggml, type, static_cast<int>(dims.size()), dims.data());
		if (!tensor)
		{
			throw std::runtime_error(std::format("Failed to allocate GGML tensor '{}'", name));
		}
		ggml_set_name(tensor, std::string(name).c_str());
		gguf_add_tensor(gguf, tensor);
		gguf_set_tensor_data(gguf, tensor->name, data);
	}

	std::filesystem::path WriteSupportedFixture()
	{
		const auto path = MakeTempFixturePath("litenn_gguf_importer_fixture", ".gguf");
		std::filesystem::remove(path);

		GGUFContextPtr gguf{ gguf_init_empty() };
		auto ggml = CreateTensorContext();
		if (!gguf || !ggml)
		{
			throw std::runtime_error("Failed to initialize GGUF fixture contexts");
		}

		gguf_set_val_str(gguf.get(), "general.architecture", "llama");
		gguf_set_val_u32(gguf.get(), "llama.context_length", 4096);
		gguf_set_val_f32(gguf.get(), "llama.rope.freq_base", 500000.0F);

		const char* tokens[] = { "<s>", "hello", "world" };
		gguf_set_arr_str(gguf.get(), "tokenizer.ggml.tokens", tokens, 3);

		const std::array<std::int32_t, 3> tokenTypes = { 1, 3, 3 };
		gguf_set_arr_data(gguf.get(), "tokenizer.ggml.token_type", GGUF_TYPE_INT32, tokenTypes.data(),
		                  tokenTypes.size());

		const std::array<float, 6> embedding = { 1.0F, 2.0F, 3.0F, 4.0F, 5.0F, 6.0F };
		const std::array<std::int64_t, 2> embeddingShape = { 2, 3 };
		AddTensor(gguf.get(), ggml.get(), GGML_TYPE_F32, "token_embd.weight", embeddingShape, embedding.data());

		const std::array<std::uint8_t, 18> q4Payload = {
			0x10, 0x00, 0x22, 0x44, 0x66, 0x88, 0xaa, 0xcc, 0xee, 0x11, 0x33, 0x55, 0x77, 0x99, 0xbb, 0xdd, 0xff, 0x7f,
		};
		const std::array<std::int64_t, 1> q4Shape = { 32 };
		AddTensor(gguf.get(), ggml.get(), GGML_TYPE_Q4_0, "blk.0.attn_q.weight", q4Shape, q4Payload.data());

		if (!gguf_write_to_file(gguf.get(), path.string().c_str(), false))
		{
			throw std::runtime_error("Failed to write GGUF importer fixture");
		}

		return path;
	}

	std::filesystem::path WriteUnsupportedFixture()
	{
		const auto path = MakeTempFixturePath("litenn_gguf_importer_unsupported_fixture", ".gguf");
		std::filesystem::remove(path);

		GGUFContextPtr gguf{ gguf_init_empty() };
		auto ggml = CreateTensorContext();
		if (!gguf || !ggml)
		{
			throw std::runtime_error("Failed to initialize unsupported GGUF fixture contexts");
		}

		gguf_set_val_str(gguf.get(), "general.architecture", "llama");
		const std::array<std::int16_t, 4> payload = { 1, 2, 3, 4 };
		const std::array<std::int64_t, 1> shape = { 4 };
		AddTensor(gguf.get(), ggml.get(), GGML_TYPE_I16, "bad.weight", shape, payload.data());

		if (!gguf_write_to_file(gguf.get(), path.string().c_str(), false))
		{
			throw std::runtime_error("Failed to write unsupported GGUF importer fixture");
		}

		return path;
	}

	std::size_t AddNamedVariable(Graph& graph, std::string_view name, Tensor<CPU> tensor)
	{
		const auto index = graph.AddVariable(Variable::Create(std::move(tensor)));
		graph.SetVariableName(index, std::string(name));
		return index;
	}

	Graph BuildTinyLLaMAArchive()
	{
		Graph graph;
		graph.SetMetadata({
		    { "general.architecture", std::string("llama") },
		    { "llama.context_length", std::uint64_t{ 8 } },
		    { "llama.embedding_length", std::uint64_t{ 4 } },
		    { "llama.block_count", std::uint64_t{ 1 } },
		    { "llama.feed_forward_length", std::uint64_t{ 8 } },
		    { "llama.attention.head_count", std::uint64_t{ 2 } },
		    { "llama.attention.head_count_kv", std::uint64_t{ 1 } },
		    { "llama.attention.layer_norm_rms_epsilon", 1.0e-6 },
		    { "llama.rope.freq_base", 10000.0 },
		});

		AddNamedVariable(
		    graph, "token_embd.weight",
		    Tensor<CPU>({ 1.0f, 0.0f, 0.0f, 0.0f, 1.0f, 0.0f, 0.0f, 0.0f, 1.0f, 0.0f, 0.0f, 0.0f }, { 4, 3 }));
		AddNamedVariable(graph, "output_norm.weight", Tensor<CPU>({ 1.0f, 1.0f, 1.0f, 1.0f }, { 1, 4 }));
		AddNamedVariable(
		    graph, "output.weight",
		    Tensor<CPU>({ 1.0f, 0.0f, 0.0f, 0.0f, 1.0f, 0.0f, 0.0f, 0.0f, 1.0f, 0.0f, 0.0f, 0.0f }, { 4, 3 }));

		AddNamedVariable(graph, "blk.0.attn_norm.weight", Tensor<CPU>({ 1.0f, 1.0f, 1.0f, 1.0f }, { 1, 4 }));
		AddNamedVariable(graph, "blk.0.ffn_norm.weight", Tensor<CPU>({ 1.0f, 1.0f, 1.0f, 1.0f }, { 1, 4 }));
		AddNamedVariable(graph, "blk.0.attn_q.weight",
		                 Tensor<CPU>({ 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f,
		                               0.0f, 0.0f, 0.0f },
		                             { 4, 4 }));
		AddNamedVariable(graph, "blk.0.attn_k.weight",
		                 Tensor<CPU>({ 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f }, { 4, 2 }));
		AddNamedVariable(graph, "blk.0.attn_v.weight",
		                 Tensor<CPU>({ 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f }, { 4, 2 }));
		AddNamedVariable(graph, "blk.0.attn_output.weight",
		                 Tensor<CPU>({ 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f,
		                               0.0f, 0.0f, 0.0f },
		                             { 4, 4 }));
		AddNamedVariable(graph, "blk.0.ffn_gate.weight",
		                 Tensor<CPU>({ 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f,
		                               0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f,
		                               0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f },
		                             { 4, 8 }));
		AddNamedVariable(graph, "blk.0.ffn_up.weight",
		                 Tensor<CPU>({ 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f,
		                               0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f,
		                               0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f },
		                             { 4, 8 }));
		AddNamedVariable(graph, "blk.0.ffn_down.weight",
		                 Tensor<CPU>({ 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f,
		                               0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f,
		                               0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f },
		                             { 8, 4 }));
		return graph;
	}

	Graph BuildTinyQwen2Archive()
	{
		auto graph = BuildTinyLLaMAArchive();
		graph.SetMetadata({
		    { "general.architecture", std::string("qwen2") },
		    { "qwen2.context_length", std::uint64_t{ 8 } },
		    { "qwen2.embedding_length", std::uint64_t{ 4 } },
		    { "qwen2.block_count", std::uint64_t{ 1 } },
		    { "qwen2.feed_forward_length", std::uint64_t{ 8 } },
		    { "qwen2.attention.head_count", std::uint64_t{ 2 } },
		    { "qwen2.attention.head_count_kv", std::uint64_t{ 1 } },
		    { "qwen2.attention.layer_norm_rms_epsilon", 1.0e-6 },
		    { "qwen2.rope.freq_base", 10000.0 },
		    { "tokenizer.ggml.model", std::string("gpt2") },
		    { "tokenizer.ggml.tokens", std::vector<std::string>{ "<|im_start|>", "hello", "<|im_end|>" } },
		    { "tokenizer.ggml.token_type", std::vector<std::int64_t>{ 3, 1, 3 } },
		    { "tokenizer.ggml.bos_token_id", std::int64_t{ 0 } },
		    { "tokenizer.ggml.eos_token_id", std::int64_t{ 2 } },
		    { "tokenizer.ggml.unknown_token_id", std::int64_t{ 1 } },
		    { "tokenizer.chat_template",
		      std::string("{% for message in messages %}{{ message.content }}{% endfor %}") },
		});
		return graph;
	}

	Graph BuildTinyQwen2ArchiveWithQ4KPayload()
	{
		auto graph = BuildTinyQwen2Archive();
		Tensor<CPU> storage(Uninitialized, { 144 }, DataType::UInt8);
		const auto variable = graph.AddVariable(Variable::CreateQuantized(
		    std::move(storage), BlockQuantization(QuantizedBlockFormat::GGML_Q4_K, { 256 }, DataType::Float32)));
		graph.SetVariableName(variable, "diagnostic.q4_k.weight");
		return graph;
	}

	Graph BuildQuantizedFriendlyLLaMAArchive()
	{
		constexpr std::size_t kEmbeddingLength = 32;
		constexpr std::size_t kFeedForwardLength = 64;

		Graph graph;
		graph.SetMetadata({
		    { "general.architecture", std::string("llama") },
		    { "llama.context_length", std::uint64_t{ 16 } },
		    { "llama.embedding_length", std::uint64_t{ kEmbeddingLength } },
		    { "llama.block_count", std::uint64_t{ 1 } },
		    { "llama.feed_forward_length", std::uint64_t{ kFeedForwardLength } },
		    { "llama.attention.head_count", std::uint64_t{ 4 } },
		    { "llama.attention.head_count_kv", std::uint64_t{ 4 } },
		    { "llama.attention.layer_norm_rms_epsilon", 1.0e-6 },
		    { "llama.rope.freq_base", 10000.0 },
		});

		std::vector<float> identity(kEmbeddingLength * kEmbeddingLength, 0.0f);
		for (std::size_t i = 0; i < kEmbeddingLength; ++i)
		{
			identity[i * kEmbeddingLength + i] = 1.0f;
		}
		const std::vector<float> ones(kEmbeddingLength, 1.0f);
		const std::vector<float> zeros32x32(kEmbeddingLength * kEmbeddingLength, 0.0f);
		const std::vector<float> zeros32x64(kEmbeddingLength * kFeedForwardLength, 0.0f);
		const std::vector<float> zeros64x32(kFeedForwardLength * kEmbeddingLength, 0.0f);

		AddNamedVariable(graph, "token_embd.weight", MakeFloatTensor(identity, { kEmbeddingLength, kEmbeddingLength }));
		AddNamedVariable(graph, "output_norm.weight", MakeFloatTensor(ones, { 1, kEmbeddingLength }));
		AddNamedVariable(graph, "output.weight", MakeFloatTensor(identity, { kEmbeddingLength, kEmbeddingLength }));

		AddNamedVariable(graph, "blk.0.attn_norm.weight", MakeFloatTensor(ones, { 1, kEmbeddingLength }));
		AddNamedVariable(graph, "blk.0.ffn_norm.weight", MakeFloatTensor(ones, { 1, kEmbeddingLength }));
		AddNamedVariable(graph, "blk.0.attn_q.weight",
		                 MakeFloatTensor(zeros32x32, { kEmbeddingLength, kEmbeddingLength }));
		AddNamedVariable(graph, "blk.0.attn_k.weight",
		                 MakeFloatTensor(zeros32x32, { kEmbeddingLength, kEmbeddingLength }));
		AddNamedVariable(graph, "blk.0.attn_v.weight",
		                 MakeFloatTensor(zeros32x32, { kEmbeddingLength, kEmbeddingLength }));
		AddNamedVariable(graph, "blk.0.attn_output.weight",
		                 MakeFloatTensor(zeros32x32, { kEmbeddingLength, kEmbeddingLength }));
		AddNamedVariable(graph, "blk.0.ffn_gate.weight",
		                 MakeFloatTensor(zeros32x64, { kEmbeddingLength, kFeedForwardLength }));
		AddNamedVariable(graph, "blk.0.ffn_up.weight",
		                 MakeFloatTensor(zeros32x64, { kEmbeddingLength, kFeedForwardLength }));
		AddNamedVariable(graph, "blk.0.ffn_down.weight",
		                 MakeFloatTensor(zeros64x32, { kFeedForwardLength, kEmbeddingLength }));
		return graph;
	}

	std::shared_ptr<Variable> QuantizeQ80Variable(const Variable& source)
	{
		const auto data = source.Data().CopyToDevice(CPU{});
		if (data.DType() != DataType::Float32)
		{
			throw std::runtime_error("QuantizeQ80Variable expects Float32 source tensors");
		}
		if (data.Shape().NumDim() == 0)
		{
			throw std::runtime_error("QuantizeQ80Variable requires at least 1D tensors");
		}

		const auto* traits = ggml_get_type_traits(GGML_TYPE_Q8_0);
		if (!traits || !traits->from_float_ref)
		{
			throw std::runtime_error("GGML_TYPE_Q8_0 reference quantizer is unavailable");
		}

		const auto rowSize = data.Shape()[0];
		if ((rowSize % static_cast<std::size_t>(traits->blck_size)) != 0)
		{
			throw std::runtime_error("QuantizeQ80Variable requires the leading dimension to be a multiple of 32");
		}

		const auto rowCount = data.NumElements() / rowSize;
		const auto rowBytes = (rowSize / static_cast<std::size_t>(traits->blck_size)) * traits->type_size;
		Tensor<CPU> storage(Uninitialized, { rowCount * rowBytes }, DataType::UInt8);
		const auto* src = static_cast<const float*>(data.UnsafeRawData());
		auto* dst = static_cast<std::uint8_t*>(storage.UnsafeRawData());
		for (std::size_t row = 0; row < rowCount; ++row)
		{
			traits->from_float_ref(src + row * rowSize, dst + row * rowBytes, static_cast<int64_t>(rowSize));
		}

		return Variable::CreateQuantized(
		    std::move(storage),
		    BlockQuantization(QuantizedBlockFormat::GGML_Q8_0, data.Shape().ToOwned(), DataType::Float32));
	}

	bool IsQ80QuantizationTarget(std::string_view name)
	{
		return name == "blk.0.attn_q.weight" || name == "blk.0.attn_k.weight" || name == "blk.0.attn_v.weight" ||
		       name == "blk.0.attn_output.weight" || name == "blk.0.ffn_gate.weight" || name == "blk.0.ffn_up.weight" ||
		       name == "blk.0.ffn_down.weight";
	}

	bool ShouldQuantizeQ80Weight(std::string_view name, const Variable& variable)
	{
		if (variable.IsQuantized() || !IsFloatingDataType(variable.Data().DType()) ||
		    variable.Data().Shape().NumDim() < 1)
		{
			return false;
		}

		return IsQ80QuantizationTarget(name);
	}

	Graph QuantizeQ80Weights(const Graph& archive)
	{
		Graph copy;
		copy.SetMetadata(std::vector<ModelMetadataEntry>(archive.Metadata().begin(), archive.Metadata().end()));
		for (std::size_t i = 0; i < archive.VariableCount(); ++i)
		{
			const auto& variable = *archive.GetVariable(i);
			const auto name = archive.VariableName(i);
			const auto shouldQuantize = ShouldQuantizeQ80Weight(name, variable);
			const auto index =
			    copy.AddVariable(shouldQuantize ? QuantizeQ80Variable(variable) : archive.GetVariable(i));
			copy.SetVariableName(index, name);
		}
		return copy;
	}

	Graph CopyArchiveExcludingVariables(const Graph& archive, std::initializer_list<std::string_view> excludedNames)
	{
		Graph copy;
		copy.SetMetadata(std::vector<ModelMetadataEntry>(archive.Metadata().begin(), archive.Metadata().end()));
		for (std::size_t i = 0; i < archive.VariableCount(); ++i)
		{
			const auto name = archive.VariableName(i);
			bool excluded = false;
			for (const auto excludedName : excludedNames)
			{
				if (name == excludedName)
				{
					excluded = true;
					break;
				}
			}
			if (excluded)
			{
				continue;
			}
			const auto index = copy.AddVariable(archive.GetVariable(i));
			copy.SetVariableName(index, name);
		}
		return copy;
	}

	Graph CopyArchiveWithMetadataOverride(const Graph& archive, std::string key, ModelMetadataValue value)
	{
		Graph copy;
		std::vector<ModelMetadataEntry> metadata(archive.Metadata().begin(), archive.Metadata().end());
		bool replaced = false;
		for (auto& entry : metadata)
		{
			if (entry.key == key)
			{
				entry.value = std::move(value);
				replaced = true;
				break;
			}
		}
		if (!replaced)
		{
			metadata.push_back({ std::move(key), std::move(value) });
		}
		copy.SetMetadata(std::move(metadata));
		for (std::size_t i = 0; i < archive.VariableCount(); ++i)
		{
			const auto index = copy.AddVariable(archive.GetVariable(i));
			copy.SetVariableName(index, archive.VariableName(i));
		}
		return copy;
	}
} // namespace

TEST(GGUFImporter, ImportsMetadataTensorNamesAndQuantizedPayloads)
{
	const auto path = WriteSupportedFixture();
	auto imported = GGUF::ImportGGUFArchive(path);
	std::filesystem::remove(path);

	EXPECT_EQ(imported.summary.tensorCount, 2u);
	EXPECT_EQ(imported.summary.metadataCount, 5u);

	ASSERT_EQ(imported.model.UnsafeGraphView().VariableCount(), 2);
	ASSERT_EQ(imported.model.UnsafeGraphView().VariableNames().size(), 2);
	EXPECT_EQ(imported.model.UnsafeGraphView().VariableName(0), "token_embd.weight");
	EXPECT_EQ(imported.model.UnsafeGraphView().VariableName(1), "blk.0.attn_q.weight");
	EXPECT_TRUE(imported.model.UnsafeGraphView().InputSignature().empty());
	EXPECT_TRUE(imported.model.UnsafeGraphView().OutputSignature().empty());

	const auto* architecture = imported.model.UnsafeGraphView().FindMetadata("general.architecture");
	ASSERT_NE(architecture, nullptr);
	EXPECT_EQ(std::get<std::string>(architecture->value), "llama");

	const auto* contextLength = imported.model.UnsafeGraphView().FindMetadata("llama.context_length");
	ASSERT_NE(contextLength, nullptr);
	EXPECT_EQ(std::get<std::uint64_t>(contextLength->value), 4096u);

	const auto* ropeBase = imported.model.UnsafeGraphView().FindMetadata("llama.rope.freq_base");
	ASSERT_NE(ropeBase, nullptr);
	EXPECT_DOUBLE_EQ(std::get<double>(ropeBase->value), 500000.0);

	const auto* tokens = imported.model.UnsafeGraphView().FindMetadata("tokenizer.ggml.tokens");
	ASSERT_NE(tokens, nullptr);
	const auto& tokenList = std::get<std::vector<std::string>>(tokens->value);
	ASSERT_EQ(tokenList.size(), 3);
	EXPECT_EQ(tokenList[0], "<s>");
	EXPECT_EQ(tokenList[1], "hello");
	EXPECT_EQ(tokenList[2], "world");

	const auto* tokenTypes = imported.model.UnsafeGraphView().FindMetadata("tokenizer.ggml.token_type");
	ASSERT_NE(tokenTypes, nullptr);
	const auto& tokenTypeList = std::get<std::vector<std::int64_t>>(tokenTypes->value);
	ASSERT_EQ(tokenTypeList.size(), 3);
	EXPECT_EQ(tokenTypeList[0], 1);
	EXPECT_EQ(tokenTypeList[1], 3);
	EXPECT_EQ(tokenTypeList[2], 3);

	ASSERT_EQ(imported.model.UnsafeGraphView().GetVariable(0)->Data().Shape().ToOwned(),
	          std::vector<std::size_t>({ 3, 2 }));
	EXPECT_FLOAT_EQ(ReadFloat(imported.model.UnsafeGraphView().GetVariable(0)->Data(), 0), 1.0F);
	EXPECT_FLOAT_EQ(ReadFloat(imported.model.UnsafeGraphView().GetVariable(0)->Data(), 1), 2.0F);
	EXPECT_FLOAT_EQ(ReadFloat(imported.model.UnsafeGraphView().GetVariable(0)->Data(), 2), 3.0F);
	EXPECT_FLOAT_EQ(ReadFloat(imported.model.UnsafeGraphView().GetVariable(0)->Data(), 3), 4.0F);
	EXPECT_FLOAT_EQ(ReadFloat(imported.model.UnsafeGraphView().GetVariable(0)->Data(), 4), 5.0F);
	EXPECT_FLOAT_EQ(ReadFloat(imported.model.UnsafeGraphView().GetVariable(0)->Data(), 5), 6.0F);

	const auto& quantized = *imported.model.UnsafeGraphView().GetVariable(1);
	ASSERT_TRUE(quantized.IsQuantized());
	EXPECT_EQ(quantized.Data().DType(), DataType::UInt8);
	EXPECT_EQ(quantized.Data().NumElements(), 18u);
	const auto& params = *quantized.Quantization();
	EXPECT_EQ(params.scheme, QuantizationScheme::Block);
	EXPECT_EQ(params.blockFormat, QuantizedBlockFormat::GGML_Q4_0);
	EXPECT_EQ(params.expressedType, DataType::Float32);
	EXPECT_EQ(params.expressedShape, std::vector<std::size_t>({ 32 }));

	const auto quantizedBytes = quantized.Data().CopyToDevice(CPU{});
	const auto* rawBytes = static_cast<const std::uint8_t*>(quantizedBytes.UnsafeRawData());
	EXPECT_EQ(rawBytes[0], 0x10u);
	EXPECT_EQ(rawBytes[17], 0x7fu);
}

TEST(GGUFImporter, RejectsUnsupportedTensorTypes)
{
	const auto path = WriteUnsupportedFixture();
	try
	{
		static_cast<void>(GGUF::ImportGGUFArchive(path));
		std::filesystem::remove(path);
		FAIL() << "Expected importer to reject unsupported GGML tensor types";
	}
	catch (const std::runtime_error& ex)
	{
		std::filesystem::remove(path);
		const std::string_view message = ex.what();
		EXPECT_NE(message.find("Unsupported ggml tensor type 'i16'"), std::string_view::npos);
		EXPECT_NE(message.find("bad.weight"), std::string_view::npos);
	}
}

TEST(GGUFImporter, ConvertGGUFArchiveWritesLoadableLiteNNModel)
{
	const auto inputPath = WriteSupportedFixture();
	const auto outputPath = MakeTempFixturePath("litenn_gguf_imported_archive", ".ltnn");
	std::filesystem::remove(outputPath);

	const auto summary = GGUF::ConvertGGUFArchive(inputPath, outputPath);
	auto loaded = Serialization::LoadVNextModelPackage(outputPath);

	std::filesystem::remove(inputPath);
	std::filesystem::remove(outputPath);

	EXPECT_EQ(summary.tensorCount, 2u);
	EXPECT_EQ(summary.metadataCount, 5u);
	ASSERT_EQ(loaded.plan.variables.size(), 2);
	ASSERT_EQ(loaded.manifest.tensors.size(), 2);
	EXPECT_EQ(loaded.manifest.tensors[0].name, "token_embd.weight");
	EXPECT_EQ(loaded.manifest.tensors[1].name, "blk.0.attn_q.weight");
	EXPECT_EQ(loaded.plan.variables[0].type.StaticShape(), std::vector<std::size_t>({ 3, 2 }));
	EXPECT_EQ(loaded.plan.variables[0].type.dtype, DataType::Float32);
	ASSERT_TRUE(loaded.plan.variables[1].quantization.has_value());
	EXPECT_EQ(loaded.plan.variables[1].quantization->blockFormat, QuantizedBlockFormat::GGML_Q4_0);
}

TEST(GGUFLLaMAHyperparameters, ParsesRequiredKeysAndDefaultsOptionalOnes)
{
	Graph graph;
	graph.SetMetadata({
	    { "general.architecture", std::string("llama") },
	    { "llama.context_length", std::uint64_t{ 4096 } },
	    { "llama.embedding_length", std::uint64_t{ 128 } },
	    { "llama.block_count", std::uint64_t{ 2 } },
	    { "llama.feed_forward_length", std::uint64_t{ 512 } },
	    { "llama.attention.head_count", std::uint64_t{ 8 } },
	    { "llama.attention.layer_norm_rms_epsilon", 1.0e-5 },
	});

	const auto hyperparameters = GGUF::ParseLLaMAHyperparameters(graph);
	EXPECT_EQ(hyperparameters.architecture, "llama");
	EXPECT_EQ(hyperparameters.contextLength, 4096u);
	EXPECT_EQ(hyperparameters.embeddingLength, 128u);
	EXPECT_EQ(hyperparameters.blockCount, 2u);
	EXPECT_EQ(hyperparameters.feedForwardLength, 512u);
	EXPECT_EQ(hyperparameters.attentionHeadCount, 8u);
	EXPECT_EQ(hyperparameters.attentionHeadCountKV, 8u);
	EXPECT_DOUBLE_EQ(hyperparameters.rmsNormEpsilon, 1.0e-5);
	EXPECT_DOUBLE_EQ(hyperparameters.ropeFrequencyBase, 10000.0);
	EXPECT_DOUBLE_EQ(hyperparameters.ropeFrequencyScale, 1.0);
	EXPECT_EQ(hyperparameters.HeadDimension(), 16u);
	EXPECT_EQ(hyperparameters.ropeDimensionCount, 16u);
	EXPECT_EQ(hyperparameters.QueryGroupsPerKVHead(), 1u);
}

TEST(GGUFLLaMACompatibility, ReportsNamedProductionProfiles)
{
	const auto profiles = GGUF::QueryLLaMACompatibilityProfiles();
	ASSERT_GE(profiles.size(), 4u);

	const auto tiny = GGUF::QueryLLaMACompatibilityProfile(GGUF::LLaMACompatibilityProfileKind::TinyFixture);
	EXPECT_EQ(tiny.name, "tiny-fixture");
	EXPECT_FALSE(tiny.selectedProductionProfile);
	EXPECT_TRUE(tiny.supportsPrefill);
	EXPECT_TRUE(tiny.supportsDecode);
	EXPECT_FALSE(tiny.requiresExternalLLaMACppGolden);

	const auto llama2 = GGUF::QueryLLaMACompatibilityProfile(GGUF::LLaMACompatibilityProfileKind::LLaMA2LikeCausalLM);
	EXPECT_EQ(llama2.name, "llama2-like-causal-lm");
	EXPECT_TRUE(llama2.selectedProductionProfile);
	EXPECT_TRUE(llama2.supportsLinearRoPE);
	EXPECT_FALSE(llama2.supportsYaRNOrLongRoPE);
	EXPECT_TRUE(llama2.importsQuantizedWeightsByDequantizing);
	EXPECT_TRUE(llama2.requiresExternalLLaMACppGolden);
	EXPECT_NE(llama2.unsupportedPolicy.find("rejected"), std::string_view::npos);
	EXPECT_NE(llama2.acceptancePolicy.find("llama.cpp golden"), std::string_view::npos);

	const auto qwen2 = GGUF::QueryLLaMACompatibilityProfile(GGUF::LLaMACompatibilityProfileKind::Qwen2LikeCausalLM);
	EXPECT_EQ(qwen2.name, "qwen2-like-causal-lm");
	EXPECT_EQ(qwen2.architecture, "qwen2");
	EXPECT_FALSE(qwen2.selectedProductionProfile);
	EXPECT_TRUE(qwen2.supportsPrefill);
	EXPECT_TRUE(qwen2.supportsDecode);
	EXPECT_TRUE(qwen2.supportsYaRNOrLongRoPE);
	EXPECT_TRUE(qwen2.requiresExternalLLaMACppGolden);
	EXPECT_NE(qwen2.unsupportedPolicy.find("Q4_K_M"), std::string_view::npos);
	EXPECT_NE(qwen2.acceptancePolicy.find("Qwen2.5"), std::string_view::npos);
}

TEST(GGUFLLaMACompatibility, InfersProfileFromArchitectureAliases)
{
	EXPECT_EQ(GGUF::TryInferLLaMACompatibilityProfile("llama"),
	          GGUF::LLaMACompatibilityProfileKind::LLaMA2LikeCausalLM);
	EXPECT_EQ(GGUF::TryInferLLaMACompatibilityProfile("qwen2"), GGUF::LLaMACompatibilityProfileKind::Qwen2LikeCausalLM);
	EXPECT_FALSE(GGUF::TryInferLLaMACompatibilityProfile("qwen2moe").has_value());
	EXPECT_FALSE(GGUF::TryInferLLaMACompatibilityProfile("mistral").has_value());
	EXPECT_FALSE(GGUF::TryInferLLaMACompatibilityProfile("gemma").has_value());
	EXPECT_FALSE(GGUF::TryInferLLaMACompatibilityProfile("unknown").has_value());
}

TEST(GGUFLLaMACompatibility, AnalyzesTinyArchiveAgainstProductionProfile)
{
	const auto archive = BuildTinyLLaMAArchive();
	const auto report =
	    GGUF::AnalyzeLLaMACompatibility(archive, GGUF::LLaMACompatibilityProfileKind::LLaMA2LikeCausalLM);

	EXPECT_TRUE(report.lowerable);
	EXPECT_TRUE(report.externalGoldenRequired);
	ASSERT_EQ(report.diagnostics.size(), 1u);
	EXPECT_EQ(report.diagnostics[0].subject, "external-golden");
	EXPECT_FALSE(report.diagnostics[0].blocking);
	EXPECT_NE(report.diagnostics[0].message.find("llama.cpp golden logits"), std::string::npos);
}

TEST(GGUFLLaMACompatibility, ReportsUnsupportedRopeVariantAsBlockingDiagnostic)
{
	const auto archive =
	    CopyArchiveWithMetadataOverride(BuildTinyLLaMAArchive(), "llama.rope.scaling.type", std::string("yarn"));
	const auto report =
	    GGUF::AnalyzeLLaMACompatibility(archive, GGUF::LLaMACompatibilityProfileKind::LLaMA3LikeCausalLM);

	EXPECT_FALSE(report.lowerable);
	EXPECT_TRUE(std::ranges::any_of(report.diagnostics, [](const GGUF::LLaMACompatibilityDiagnostic& diagnostic) {
		return diagnostic.blocking && diagnostic.subject == "llama.rope.scaling.type" &&
		       diagnostic.message.find("only executes none/linear") != std::string::npos;
	}));
}

TEST(GGUFLLaMACompatibility, AnalyzesQwen2ArchiveWithActionableProductionDiagnostics)
{
	const auto report = GGUF::AnalyzeLLaMACompatibility(BuildTinyQwen2Archive(),
	                                                    GGUF::LLaMACompatibilityProfileKind::Qwen2LikeCausalLM);

	EXPECT_TRUE(report.lowerable);
	EXPECT_TRUE(report.externalGoldenRequired);
	EXPECT_TRUE(std::ranges::any_of(report.diagnostics, [](const GGUF::LLaMACompatibilityDiagnostic& diagnostic) {
		return !diagnostic.blocking && diagnostic.subject == "qwen2.tokenizer" &&
		       diagnostic.message.find("tokens=3") != std::string::npos &&
		       diagnostic.message.find("chat_template=yes") != std::string::npos &&
		       diagnostic.message.find("Token-id parity") != std::string::npos;
	}));
	EXPECT_TRUE(std::ranges::any_of(report.diagnostics, [](const GGUF::LLaMACompatibilityDiagnostic& diagnostic) {
		return !diagnostic.blocking && diagnostic.subject == "qwen2.quantized-cuda" &&
		       diagnostic.message.find("Q4_K_M CUDA") != std::string::npos;
	}));
	EXPECT_TRUE(std::ranges::any_of(report.diagnostics, [](const GGUF::LLaMACompatibilityDiagnostic& diagnostic) {
		return !diagnostic.blocking && diagnostic.subject == "qwen2.decode-loop" &&
		       diagnostic.message.find("runtime decode loop") != std::string::npos;
	}));
}

TEST(GGUFLLaMACompatibility, SummarizesTokenizerMetadata)
{
	const auto summary = GGUF::SummarizeLLMTokenizerMetadata(BuildTinyQwen2Archive());

	ASSERT_TRUE(summary.model.has_value());
	EXPECT_EQ(*summary.model, "gpt2");
	EXPECT_EQ(summary.tokenCount, 3u);
	EXPECT_EQ(summary.tokenTypeCount, 3u);
	EXPECT_TRUE(summary.hasChatTemplate);
	EXPECT_GT(summary.chatTemplateBytes, 0u);
	EXPECT_TRUE(summary.hasBosTokenId);
	EXPECT_TRUE(summary.hasEosTokenId);
	EXPECT_TRUE(summary.hasUnknownTokenId);
	EXPECT_EQ(summary.bosTokenId, 0);
	EXPECT_EQ(summary.eosTokenId, 2);
	EXPECT_EQ(summary.unknownTokenId, 1);
}

TEST(GGUFLLMGeneration, AcceptsCallerProvidedTokenIdsAndTracksEOS)
{
	const auto tokenizer = GGUF::SummarizeLLMTokenizerMetadata(BuildTinyQwen2Archive());
	const std::array<std::int32_t, 2> tokenIds{ 0, 1 };
	auto prompt = GGUF::MakeCallerProvidedPromptTokens(tokenIds, tokenizer);

	EXPECT_TRUE(prompt.callerProvided);
	EXPECT_EQ(prompt.tokenIds, std::vector<std::int32_t>({ 0, 1 }));

	auto generation = GGUF::BeginGeneration(std::move(prompt), static_cast<std::int32_t>(*tokenizer.eosTokenId));
	GGUF::LLMSamplerState sampler{ .config = { .mode = GGUF::LLMSamplingMode::Greedy } };
	const std::array<float, 3> logits{ -1.0f, 0.0f, 4.0f };

	EXPECT_EQ(GGUF::StepGeneration(generation, logits, sampler), 2);
	EXPECT_TRUE(generation.finished);
	EXPECT_EQ(generation.generatedTokenCount, 1u);
	EXPECT_EQ(generation.tokens, std::vector<std::int32_t>({ 0, 1, 2 }));
}

TEST(GGUFLLMGeneration, RejectsOutOfVocabularyCallerProvidedTokenIds)
{
	const auto tokenizer = GGUF::SummarizeLLMTokenizerMetadata(BuildTinyQwen2Archive());
	const std::array<std::int32_t, 1> tokenIds{ 3 };

	EXPECT_THROW(static_cast<void>(GGUF::MakeCallerProvidedPromptTokens(tokenIds, tokenizer)), std::runtime_error);
}

TEST(GGUFLLMGeneration, AppliesRepeatPenaltyBeforeGreedySampling)
{
	GGUF::LLMSamplerState sampler{ .config = { .mode = GGUF::LLMSamplingMode::Greedy, .repeatPenalty = 2.0f } };
	const std::array<float, 3> logits{ 1.0f, 3.0f, 2.0f };
	const std::array<std::int32_t, 1> history{ 1 };

	EXPECT_EQ(GGUF::SelectNextToken(logits, sampler, history), 2);
}

TEST(GGUFLLMGeneration, SelectsFromLastRowOfLiteNNTensorLogits)
{
	auto generation = GGUF::BeginGeneration({ .tokenIds = { 0 }, .callerProvided = true });
	GGUF::LLMSamplerState sampler{ .config = { .mode = GGUF::LLMSamplingMode::Greedy } };
	const Tensor<CPU> logits({ 9.0f, 0.0f, 1.0f, -1.0f, 4.0f, 2.0f }, { 2, 3 });

	EXPECT_EQ(GGUF::ExtractLastTokenLogits(logits), std::vector<float>({ -1.0f, 4.0f, 2.0f }));
	EXPECT_EQ(GGUF::StepGeneration(generation, logits, sampler), 1);
	EXPECT_EQ(generation.tokens, std::vector<std::int32_t>({ 0, 1 }));
}

TEST(GGUFLLMGeneration, SamplesDeterministicallyWithTopKAndTopP)
{
	GGUF::LLMSamplerState first{
		.config = { .mode = GGUF::LLMSamplingMode::Random, .temperature = 0.7f, .topK = 3, .topP = 0.9f, .seed = 42 }
	};
	GGUF::LLMSamplerState second = first;
	const std::array<float, 5> logits{ -4.0f, 2.0f, 1.0f, 0.5f, -2.0f };

	EXPECT_EQ(GGUF::SelectNextToken(logits, first), GGUF::SelectNextToken(logits, second));
	EXPECT_EQ(first.drawCount, 1u);
}

TEST(GGUFLLaMAArtifacts, PlansPrefillAndDecodeStepEntries)
{
	const auto plan = GGUF::PlanLLaMAArtifacts(BuildTinyQwen2Archive(), 4, 3);

	EXPECT_EQ(plan.hyperparameters.architecture, "qwen2");
	EXPECT_EQ(plan.dtype, DataType::Float32);
	EXPECT_EQ(plan.vocabSize, 3u);
	EXPECT_EQ(plan.prefill.kind, GGUF::LLaMAArtifactKind::Prefill);
	EXPECT_EQ(plan.prefill.name, "prefill");
	EXPECT_EQ(plan.prefill.sequenceLength, 4u);
	EXPECT_EQ(plan.prefill.inputNames, std::vector<std::string>({ "token_ids" }));
	EXPECT_EQ(plan.prefill.outputNames, std::vector<std::string>({ "logits" }));
	EXPECT_TRUE(plan.prefill.kvCaches.empty());

	EXPECT_EQ(plan.decodeStep.kind, GGUF::LLaMAArtifactKind::DecodeStep);
	EXPECT_EQ(plan.decodeStep.name, "decode_step");
	EXPECT_EQ(plan.decodeStep.sequenceLength, 1u);
	EXPECT_EQ(plan.decodeStep.pastLength, 3u);
	EXPECT_EQ(plan.decodeStep.positionOffset, 3u);
	EXPECT_EQ(plan.decodeStep.inputNames, std::vector<std::string>({ "token_ids", "past_key_0", "past_value_0" }));
	EXPECT_EQ(plan.decodeStep.outputNames, std::vector<std::string>({ "logits", "updated_key_0", "updated_value_0" }));
	ASSERT_EQ(plan.decodeStep.kvCaches.size(), 1u);
	EXPECT_EQ(plan.decodeStep.kvCaches[0].blockIndex, 0u);
	EXPECT_EQ(plan.decodeStep.kvCaches[0].pastKeyInput, "past_key_0");
	EXPECT_EQ(plan.decodeStep.kvCaches[0].updatedValueOutput, "updated_value_0");
	EXPECT_EQ(plan.decodeStep.kvCaches[0].cacheType.dtype, DataType::Float32);
	EXPECT_EQ(plan.decodeStep.kvCaches[0].cacheType.StaticShape(), std::vector<std::size_t>({ 3, 1, 2 }));
	EXPECT_EQ(plan.decodeStep.kvCaches[0].stateType.StaticShape(), std::vector<std::size_t>({ 2, 3, 1, 2 }));
	EXPECT_EQ(plan.decodeStep.kvCaches[0].stateBinding.name, "kv.layer0");
	EXPECT_EQ(plan.decodeStep.kvCaches[0].stateBinding.kind, Runtime::RuntimeStateKind::KVCache);
	EXPECT_EQ(plan.decodeStep.kvCaches[0].keyByteOffset, 0u);
	EXPECT_EQ(plan.decodeStep.kvCaches[0].valueByteOffset, 24u);
	EXPECT_EQ(plan.decodeStep.kvCaches[0].layerByteStride, 48u);
	EXPECT_EQ(plan.decodeStep.kvCaches[0].tokenByteStride, 8u);
	ASSERT_EQ(plan.decodeStateABI.kvCaches.size(), 1u);
	EXPECT_EQ(plan.decodeStateABI.kvCaches[0].name, "kv.layer0");
	EXPECT_TRUE(std::ranges::contains(plan.decodeStateABI.kvCaches[0].effects, std::string("write")));
	ASSERT_TRUE(plan.decodeStateABI.currentPosition.has_value());
	EXPECT_EQ(plan.decodeStateABI.currentPosition->name, "decode.position");
	EXPECT_TRUE(std::ranges::contains(plan.decodeStateABI.currentPosition->effects, std::string("increment")));
}

TEST(GGUFLLaMAArtifacts, ReportsInspectableTensorLayouts)
{
	const auto plan = GGUF::PlanLLaMAArtifacts(BuildTinyQwen2Archive(), 4, 3);

	const auto findLayout = [&plan](std::string_view name) -> const GGUF::LLaMATensorLayoutRecord* {
		const auto found =
		    std::ranges::find_if(plan.tensorLayouts, [name](const auto& record) { return record.name == name; });
		return found == plan.tensorLayouts.end() ? nullptr : std::to_address(found);
	};

	const auto* hidden = findLayout("litenn.hidden_state");
	ASSERT_NE(hidden, nullptr);
	EXPECT_EQ(hidden->domain, "litenn-semantic");
	EXPECT_EQ(hidden->axes, std::vector<std::string>({ "sequence", "embedding" }));

	const auto* mutableCache = findLayout("runtime.mutable_kv_state");
	ASSERT_NE(mutableCache, nullptr);
	EXPECT_EQ(mutableCache->axes, std::vector<std::string>({ "key_value", "capacity", "kv_head", "head_dim" }));
	EXPECT_NE(mutableCache->note.find("value plane"), std::string::npos);

	EXPECT_NE(findLayout("gguf.imported_weight"), nullptr);
	EXPECT_NE(findLayout("runtime.functional_kv_cache"), nullptr);
}

TEST(GGUFLLaMAArtifacts, SeparatesDecodePositionFromCacheCapacity)
{
	const auto plan = GGUF::PlanLLaMAArtifacts(BuildTinyQwen2Archive(), GGUF::LLaMAArtifactPlanningOptions{
	                                                                        .prefillSequenceLength = 4,
	                                                                        .decodePastLength = 3,
	                                                                        .maxCacheLength = 8,
	                                                                    });

	EXPECT_EQ(plan.prefill.maxCacheLength, 8u);
	EXPECT_EQ(plan.decodeStep.pastLength, 3u);
	EXPECT_EQ(plan.decodeStep.maxCacheLength, 8u);
	ASSERT_EQ(plan.decodeStep.kvCaches.size(), 1u);
	EXPECT_EQ(plan.decodeStep.kvCaches[0].cacheType.StaticShape(), std::vector<std::size_t>({ 3, 1, 2 }));
	EXPECT_EQ(plan.decodeStep.kvCaches[0].stateType.StaticShape(), std::vector<std::size_t>({ 2, 8, 1, 2 }));
	EXPECT_EQ(plan.decodeStep.kvCaches[0].valueByteOffset, 64u);
	EXPECT_EQ(plan.decodeStep.kvCaches[0].layerByteStride, 128u);
}

TEST(GGUFLLaMAArtifacts, RejectsCacheCapacitySmallerThanDecodePosition)
{
	EXPECT_THROW(static_cast<void>(GGUF::PlanLLaMAArtifacts(BuildTinyQwen2Archive(),
	                                                        GGUF::LLaMAArtifactPlanningOptions{
	                                                            .prefillSequenceLength = 4,
	                                                            .decodePastLength = 3,
	                                                            .maxCacheLength = 2,
	                                                        })),
	             std::runtime_error);
}

TEST(GGUFLLaMAQuantizedExecution, PlansReferenceDequantizedFallbackAndBudgetRejection)
{
	const auto archive = BuildTinyQwen2ArchiveWithQ4KPayload();
	const auto plan = GGUF::PlanLLaMAQuantizedWeightExecution(archive);

	EXPECT_TRUE(plan.lowerable);
	EXPECT_EQ(plan.tensorCount, 1u);
	EXPECT_EQ(plan.storedBytes, 144u);
	EXPECT_EQ(plan.dequantizedBytes, 1024u);
	ASSERT_EQ(plan.decisions.size(), 1u);
	EXPECT_EQ(plan.decisions[0].format, QuantizedBlockFormat::GGML_Q4_K);
	EXPECT_EQ(plan.decisions[0].selectedPolicy, GGUF::LLaMAQuantizedExecutionPolicy::CPUReferenceDequantize);
	EXPECT_EQ(GGUF::LLaMAQuantizedExecutionPolicyName(plan.decisions[0].selectedPolicy), "cpu-reference-dequantize");
	EXPECT_FALSE(plan.decisions[0].blocking);

	const auto rejected = GGUF::PlanLLaMAQuantizedWeightExecution(archive, 512);
	EXPECT_FALSE(rejected.lowerable);
	ASSERT_EQ(rejected.decisions.size(), 1u);
	EXPECT_EQ(rejected.decisions[0].selectedPolicy, GGUF::LLaMAQuantizedExecutionPolicy::Reject);
	EXPECT_TRUE(rejected.decisions[0].blocking);
}

TEST(GGUFLLaMACompatibility, ReportsQuantizationMixAndQ4KDiagnostic)
{
	const auto report = GGUF::AnalyzeLLaMACompatibility(BuildTinyQwen2ArchiveWithQ4KPayload(),
	                                                    GGUF::LLaMACompatibilityProfileKind::Qwen2LikeCausalLM);

	EXPECT_TRUE(report.lowerable);
	EXPECT_TRUE(std::ranges::any_of(report.diagnostics, [](const GGUF::LLaMACompatibilityDiagnostic& diagnostic) {
		return !diagnostic.blocking && diagnostic.subject == "quantization.mix" &&
		       diagnostic.message.find("GGML_Q4_K") != std::string::npos &&
		       diagnostic.message.find("policy=cpu-reference-dequantize") != std::string::npos;
	}));
	EXPECT_TRUE(std::ranges::any_of(report.diagnostics, [](const GGUF::LLaMACompatibilityDiagnostic& diagnostic) {
		return !diagnostic.blocking && diagnostic.subject == "quantization.q4_k_m" &&
		       diagnostic.message.find("native K-quant projection kernels") != std::string::npos;
	}));
}

TEST(GGUFLLaMACompatibility, AppliesQuantizedDequantizationBudgetAsBlockingDiagnostic)
{
	const auto report = GGUF::AnalyzeLLaMACompatibility(BuildTinyQwen2ArchiveWithQ4KPayload(),
	                                                    GGUF::LLaMACompatibilityProfileKind::Qwen2LikeCausalLM, 512);

	EXPECT_FALSE(report.lowerable);
	EXPECT_TRUE(std::ranges::any_of(report.diagnostics, [](const GGUF::LLaMACompatibilityDiagnostic& diagnostic) {
		return diagnostic.blocking && diagnostic.subject == "quantization.mix" &&
		       diagnostic.message.find("policy=reject") != std::string::npos;
	}));
}

TEST(GGUFLLaMAHyperparameters, UsesExplicitKVHeadCountAndRopeBase)
{
	Graph graph;
	graph.SetMetadata({
	    { "general.architecture", std::string("llama") },
	    { "llama.context_length", std::uint64_t{ 8192 } },
	    { "llama.embedding_length", std::uint64_t{ 256 } },
	    { "llama.block_count", std::uint64_t{ 4 } },
	    { "llama.feed_forward_length", std::uint64_t{ 768 } },
	    { "llama.attention.head_count", std::uint64_t{ 8 } },
	    { "llama.attention.head_count_kv", std::uint64_t{ 2 } },
	    { "llama.attention.layer_norm_rms_epsilon", 1.0e-6 },
	    { "llama.rope.freq_base", 500000.0 },
	    { "llama.rope.freq_scale", 1.0 },
	    { "llama.rope.dimension_count", std::uint64_t{ 32 } },
	});

	const auto hyperparameters = GGUF::ParseLLaMAHyperparameters(graph);
	EXPECT_EQ(hyperparameters.attentionHeadCountKV, 2u);
	EXPECT_DOUBLE_EQ(hyperparameters.ropeFrequencyBase, 500000.0);
	EXPECT_EQ(hyperparameters.ropeDimensionCount, 32u);
	EXPECT_EQ(hyperparameters.HeadDimension(), 32u);
	EXPECT_EQ(hyperparameters.QueryGroupsPerKVHead(), 4u);
}

TEST(GGUFLLaMAHyperparameters, ParsesRopeScalingMetadata)
{
	Graph graph;
	graph.SetMetadata({
	    { "general.architecture", std::string("llama") },
	    { "llama.context_length", std::uint64_t{ 8192 } },
	    { "llama.embedding_length", std::uint64_t{ 256 } },
	    { "llama.block_count", std::uint64_t{ 4 } },
	    { "llama.feed_forward_length", std::uint64_t{ 768 } },
	    { "llama.attention.head_count", std::uint64_t{ 8 } },
	    { "llama.attention.layer_norm_rms_epsilon", 1.0e-6 },
	    { "llama.rope.scaling.type", std::string("linear") },
	    { "llama.rope.scaling.factor", 4.0 },
	    { "llama.rope.scaling.original_context_length", std::uint64_t{ 2048 } },
	    { "llama.rope.scaling.finetuned", true },
	});

	const auto hyperparameters = GGUF::ParseLLaMAHyperparameters(graph);
	EXPECT_EQ(hyperparameters.ropeScalingType, "linear");
	ASSERT_TRUE(hyperparameters.ropeScalingFactor);
	EXPECT_DOUBLE_EQ(*hyperparameters.ropeScalingFactor, 4.0);
	EXPECT_DOUBLE_EQ(hyperparameters.ropeFrequencyScale, 0.25);
	ASSERT_TRUE(hyperparameters.ropeScalingOriginalContextLength);
	EXPECT_EQ(*hyperparameters.ropeScalingOriginalContextLength, 2048u);
	ASSERT_TRUE(hyperparameters.ropeScalingFinetuned);
	EXPECT_TRUE(*hyperparameters.ropeScalingFinetuned);
}

TEST(GGUFLLaMAHyperparameters, RejectsMissingRequiredMetadata)
{
	Graph graph;
	graph.SetMetadata({
	    { "general.architecture", std::string("llama") },
	    { "llama.context_length", std::uint64_t{ 4096 } },
	    { "llama.block_count", std::uint64_t{ 2 } },
	    { "llama.feed_forward_length", std::uint64_t{ 512 } },
	    { "llama.attention.head_count", std::uint64_t{ 8 } },
	    { "llama.attention.layer_norm_rms_epsilon", 1.0e-5 },
	});

	EXPECT_THROW(static_cast<void>(GGUF::ParseLLaMAHyperparameters(graph)), std::runtime_error);
}

TEST(GGUFLLaMAHyperparameters, RejectsIncompatibleHeadLayout)
{
	Graph graph;
	graph.SetMetadata({
	    { "general.architecture", std::string("llama") },
	    { "llama.context_length", std::uint64_t{ 4096 } },
	    { "llama.embedding_length", std::uint64_t{ 130 } },
	    { "llama.block_count", std::uint64_t{ 2 } },
	    { "llama.feed_forward_length", std::uint64_t{ 512 } },
	    { "llama.attention.head_count", std::uint64_t{ 8 } },
	    { "llama.attention.head_count_kv", std::uint64_t{ 3 } },
	    { "llama.attention.layer_norm_rms_epsilon", 1.0e-5 },
	});

	EXPECT_THROW(static_cast<void>(GGUF::ParseLLaMAHyperparameters(graph)), std::runtime_error);
}

TEST(GGUFLLaMADecoderBlock, LowersNamedArchiveBlockAndActsAsIdentityWithZeroProjections)
{
	const auto archive = BuildTinyLLaMAArchive();
	const auto hyperparameters = GGUF::ParseLLaMAHyperparameters(archive);

	Graph graph;
	const auto block = GGUF::CreateLLaMADecoderBlock(graph, archive, hyperparameters, 0);
	Subgraph sg;
	const auto hiddenState = sg.AddParam(DataType::Float32, { 2, 4 });
	const auto result = GGUF::AddLLaMADecoderBlock(sg, block, hyperparameters, { hiddenState, 0 });
	sg.SetResults({ result });
	graph.SetForward(graph.AddSubgraph(std::move(sg)));

	Runtime::Interpreter<CPU> interpreter;
	std::array<Tensor<CPU>, 1> inputs = { Tensor<CPU>({ 1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f, 7.0f, 8.0f }, { 2, 4 }) };
	const auto outputs = interpreter.RunForward(Detail::BuildExecutablePlanFromGraph(graph), inputs);
	ASSERT_EQ(outputs.size(), 1u);
	for (std::size_t i = 0; i < 8; ++i)
	{
		EXPECT_NEAR(ReadFloat(outputs[0], i), ReadFloat(inputs[0], i), 1e-5f);
	}
}

TEST(GGUFLLaMADecoderBlock, RejectsMissingNamedWeights)
{
	auto archive = BuildTinyLLaMAArchive();
	const auto incompleteArchive = CopyArchiveExcludingVariables(archive, { "blk.0.ffn_down.weight" });

	Graph graph;
	const auto hyperparameters = GGUF::ParseLLaMAHyperparameters(incompleteArchive);
	EXPECT_THROW(static_cast<void>(GGUF::CreateLLaMADecoderBlock(graph, incompleteArchive, hyperparameters, 0)),
	             std::runtime_error);
}

TEST(GGUFLLaMACausalLM, LowersFullGraphAndRunsCPUForwardOnTokenIds)
{
	const auto archive = BuildTinyLLaMAArchive();
	const auto lowered = GGUF::LowerLLaMACausalLM(archive, 2);

	EXPECT_EQ(lowered.InputName(0), "token_ids");
	EXPECT_EQ(lowered.OutputName(0), "logits");

	Runtime::Interpreter<CPU> interpreter;
	std::array<Tensor<CPU>, 1> inputs = { MakeInt32Tensor({ 0, 1 }, { 2 }) };
	const auto outputs = interpreter.RunForward(Detail::BuildExecutablePlanFromGraph(lowered), inputs);
	ASSERT_EQ(outputs.size(), 1u);
	ASSERT_EQ(outputs[0].Shape().NumElements(), 6u);

	const float expected = 1.0f / std::sqrt(0.25f + 1.0e-6f);
	EXPECT_NEAR(ReadFloat(outputs[0], 0), expected, 1e-4f);
	EXPECT_NEAR(ReadFloat(outputs[0], 1), 0.0f, 1e-5f);
	EXPECT_NEAR(ReadFloat(outputs[0], 2), 0.0f, 1e-5f);
	EXPECT_NEAR(ReadFloat(outputs[0], 3), 0.0f, 1e-5f);
	EXPECT_NEAR(ReadFloat(outputs[0], 4), expected, 1e-4f);
	EXPECT_NEAR(ReadFloat(outputs[0], 5), 0.0f, 1e-5f);
}

TEST(GGUFLLaMACausalLM, MatchesDeterministicGoldenPrefillLogits)
{
	const auto archive = BuildTinyLLaMAArchive();
	const auto lowered = GGUF::LowerLLaMACausalLM(archive, 2);

	Runtime::Interpreter<CPU> interpreter;
	std::array<Tensor<CPU>, 1> inputs = { MakeInt32Tensor({ 0, 1 }, { 2 }) };
	const auto outputs = interpreter.RunForward(Detail::BuildExecutablePlanFromGraph(lowered), inputs);
	ASSERT_EQ(outputs.size(), 1u);

	const float expectedScalar = 1.0f / std::sqrt(0.25f + 1.0e-6f);
	const std::array<float, 6> goldenLogits = {
		expectedScalar, 0.0f, 0.0f, 0.0f, expectedScalar, 0.0f,
	};
	ExpectTensorNear(outputs[0], goldenLogits, GGUF::GetLLaMAParityTolerance(DataType::Float32));
}

TEST(GGUFLLaMACausalLM, FallsBackToTokenEmbeddingWhenOutputWeightIsMissing)
{
	const auto archive = BuildTinyLLaMAArchive();
	const auto tiedArchive = CopyArchiveExcludingVariables(archive, { "output.weight" });
	const auto lowered = GGUF::LowerLLaMACausalLM(tiedArchive, 1);

	Runtime::Interpreter<CPU> interpreter;
	std::array<Tensor<CPU>, 1> inputs = { MakeInt32Tensor({ 2 }, { 1 }) };
	const auto outputs = interpreter.RunForward(Detail::BuildExecutablePlanFromGraph(lowered), inputs);
	ASSERT_EQ(outputs.size(), 1u);
	ASSERT_EQ(outputs[0].Shape().NumElements(), 3u);

	const float expected = 1.0f / std::sqrt(0.25f + 1.0e-6f);
	EXPECT_NEAR(ReadFloat(outputs[0], 0), 0.0f, 1e-5f);
	EXPECT_NEAR(ReadFloat(outputs[0], 1), 0.0f, 1e-5f);
	EXPECT_NEAR(ReadFloat(outputs[0], 2), expected, 1e-4f);
}

TEST(GGUFLLaMACausalLM, TransposesImportedNonSquareLinearWeightsIntoLiteNNLayout)
{
	auto archive = CopyArchiveExcludingVariables(BuildTinyLLaMAArchive(), { "output.weight" });
	AddNamedVariable(archive, "output.weight",
	                 Tensor<CPU>(
	                     {
	                         1.0f,
	                         2.0f,
	                         3.0f,
	                         4.0f,
	                         5.0f,
	                         6.0f,
	                         7.0f,
	                         8.0f,
	                         9.0f,
	                         10.0f,
	                         11.0f,
	                         12.0f,
	                     },
	                     { 3, 4 }));

	const auto lowered = GGUF::LowerLLaMACausalLM(archive, 2);
	const auto outputWeightIndex = lowered.FindVariable("output.weight");
	ASSERT_TRUE(outputWeightIndex.has_value());

	const auto loweredWeight = lowered.GetVariable(*outputWeightIndex)->Data().CopyToDevice(CPU{});
	ASSERT_EQ(loweredWeight.Shape().NumDim(), 2u);
	EXPECT_EQ(loweredWeight.Shape()[0], 4u);
	EXPECT_EQ(loweredWeight.Shape()[1], 3u);

	const std::array<float, 12> expected = {
		1.0f, 5.0f, 9.0f, 2.0f, 6.0f, 10.0f, 3.0f, 7.0f, 11.0f, 4.0f, 8.0f, 12.0f,
	};
	for (std::size_t i = 0; i < expected.size(); ++i)
	{
		EXPECT_NEAR(ReadFloat(loweredWeight, i), expected[i], 1e-5f);
	}
}

TEST(GGUFLLaMACausalLM, LowersDecodeGraphWithExplicitKVCacheInputsAndOutputs)
{
	const auto archive = BuildTinyLLaMAArchive();
	const auto lowered = GGUF::LowerLLaMACausalLMDecode(archive, 1, 1, 1);

	ASSERT_EQ(lowered.InputSignature().size(), 3u);
	ASSERT_EQ(lowered.OutputSignature().size(), 3u);
	EXPECT_EQ(lowered.InputName(0), "token_ids");
	EXPECT_EQ(lowered.InputName(1), "past_key_0");
	EXPECT_EQ(lowered.InputName(2), "past_value_0");
	EXPECT_EQ(lowered.OutputName(0), "logits");
	EXPECT_EQ(lowered.OutputName(1), "updated_key_0");
	EXPECT_EQ(lowered.OutputName(2), "updated_value_0");

	Runtime::Interpreter<CPU> interpreter;
	const std::vector<float> zeroCache{ 0.0f, 0.0f };
	std::array<Tensor<CPU>, 3> inputs = {
		MakeInt32Tensor({ 1 }, { 1 }),
		MakeFloatTensor(zeroCache, { 1, 1, 2 }),
		MakeFloatTensor(zeroCache, { 1, 1, 2 }),
	};
	const auto outputs = interpreter.RunForward(Detail::BuildExecutablePlanFromGraph(lowered), inputs);
	ASSERT_EQ(outputs.size(), 3u);
	EXPECT_EQ(outputs[0].Shape().ToOwned(), std::vector<std::size_t>({ 1, 3 }));
	EXPECT_EQ(outputs[1].Shape().ToOwned(), std::vector<std::size_t>({ 2, 1, 2 }));
	EXPECT_EQ(outputs[2].Shape().ToOwned(), std::vector<std::size_t>({ 2, 1, 2 }));
}

TEST(GGUFLLaMACausalLM, MatchesDeterministicGoldenDecodeLogitsAndCacheUpdate)
{
	const auto archive = BuildTinyLLaMAArchive();
	const auto lowered = GGUF::LowerLLaMACausalLMDecode(archive, 1, 1, 1);

	Runtime::Interpreter<CPU> interpreter;
	const std::vector<float> zeroCache{ 0.0f, 0.0f };
	std::array<Tensor<CPU>, 3> inputs = {
		MakeInt32Tensor({ 2 }, { 1 }),
		MakeFloatTensor(zeroCache, { 1, 1, 2 }),
		MakeFloatTensor(zeroCache, { 1, 1, 2 }),
	};
	const auto outputs = interpreter.RunForward(Detail::BuildExecutablePlanFromGraph(lowered), inputs);
	ASSERT_EQ(outputs.size(), 3u);

	const float expectedScalar = 1.0f / std::sqrt(0.25f + 1.0e-6f);
	const std::array<float, 3> goldenLogits = { 0.0f, 0.0f, expectedScalar };
	ExpectTensorNear(outputs[0], goldenLogits, GGUF::GetLLaMAParityTolerance(DataType::Float32));
	EXPECT_EQ(outputs[1].Shape().ToOwned(), std::vector<std::size_t>({ 2, 1, 2 }));
	EXPECT_EQ(outputs[2].Shape().ToOwned(), std::vector<std::size_t>({ 2, 1, 2 }));
}

TEST(GGUFLLaMACausalLM, PrefillThenDecodeMatchesFullPrefillLogit)
{
	const auto archive = BuildTinyLLaMAArchive();
	const auto fullPrefill = GGUF::LowerLLaMACausalLM(archive, 2);
	const auto secondStep = GGUF::LowerLLaMACausalLMDecode(archive, 1, 1, 1);

	Runtime::Interpreter<CPU> interpreter;
	std::array<Tensor<CPU>, 1> fullInputs = { MakeInt32Tensor({ 0, 1 }, { 2 }) };
	const auto fullOutputs = interpreter.RunForward(Detail::BuildExecutablePlanFromGraph(fullPrefill), fullInputs);
	ASSERT_EQ(fullOutputs.size(), 1u);

	const std::vector<float> oneTokenPastCache{ 0.0f, 0.0f };
	std::array<Tensor<CPU>, 3> secondInputs = {
		MakeInt32Tensor({ 1 }, { 1 }),
		MakeFloatTensor(oneTokenPastCache, { 1, 1, 2 }),
		MakeFloatTensor(oneTokenPastCache, { 1, 1, 2 }),
	};
	const auto secondOutputs = interpreter.RunForward(Detail::BuildExecutablePlanFromGraph(secondStep), secondInputs);
	ASSERT_EQ(secondOutputs.size(), 3u);
	EXPECT_EQ(secondOutputs[1].Shape().ToOwned(), std::vector<std::size_t>({ 2, 1, 2 }));
	EXPECT_EQ(secondOutputs[2].Shape().ToOwned(), std::vector<std::size_t>({ 2, 1, 2 }));

	const std::array<float, 3> fullSecondLogit = {
		ReadFloat(fullOutputs[0], 3),
		ReadFloat(fullOutputs[0], 4),
		ReadFloat(fullOutputs[0], 5),
	};
	ExpectTensorNear(secondOutputs[0], fullSecondLogit, GGUF::GetLLaMAParityTolerance(DataType::Float32));
}

#ifdef LITENN_ENABLE_MLIR
TEST(GGUFLLaMACausalLM, CompilesTwoTokenFullGraphToCPUArtifactAndLoads)
{
	const auto archive = BuildTinyLLaMAArchive();
	const auto lowered = GGUF::LowerLLaMACausalLM(archive, 2);

	auto artifact = Compiler<CPU>::CompileArtifact(Detail::BuildExecutablePlanFromGraph(lowered));
	EXPECT_EQ(artifact.InputSpecs().size(), 1u);
	EXPECT_EQ(artifact.OutputSpecs().size(), 1u);
	auto compiled = artifact.Load();
	EXPECT_EQ(compiled.InputSpecs().size(), 1u);
	EXPECT_EQ(compiled.OutputSpecs().size(), 1u);
	EXPECT_EQ(compiled.FindInput("token_ids"), 0u);
	EXPECT_EQ(compiled.FindOutput("logits"), 0u);
}

TEST(GGUFLLaMACausalLM, CompilesSingleTokenFullGraphToCPUArtifactAndMatchesInterpreter)
{
	const auto archive = BuildTinyLLaMAArchive();
	const auto lowered = GGUF::LowerLLaMACausalLM(archive, 1);
	const auto tolerance = GGUF::GetLLaMAParityTolerance(DataType::Float32);

	Runtime::Interpreter<CPU> interpreter;
	std::array<Tensor<CPU>, 1> inputs = { MakeInt32Tensor({ 2 }, { 1 }) };
	const auto expected = interpreter.RunForward(Detail::BuildExecutablePlanFromGraph(lowered), inputs);

	auto artifact = Compiler<CPU>::CompileArtifact(Detail::BuildExecutablePlanFromGraph(lowered));
	auto compiled = artifact.Load();
	const auto outputs = compiled.RunTensors(inputs);

	ASSERT_EQ(expected.size(), 1u);
	ASSERT_EQ(outputs.size(), 1u);
	ExpectTensorNear(outputs[0], expected[0], tolerance);
}

TEST(GGUFLLaMACausalLM, CompilesDecodeGraphToCPUArtifactAndMatchesInterpreter)
{
	const auto archive = BuildTinyLLaMAArchive();
	const auto lowered = GGUF::LowerLLaMACausalLMDecode(archive, 1, 1, 1);
	const auto tolerance = GGUF::GetLLaMAParityTolerance(DataType::Float32);

	Runtime::Interpreter<CPU> interpreter;
	const std::vector<float> zeroCache{ 0.0f, 0.0f };
	std::array<Tensor<CPU>, 3> inputs = {
		MakeInt32Tensor({ 1 }, { 1 }),
		MakeFloatTensor(zeroCache, { 1, 1, 2 }),
		MakeFloatTensor(zeroCache, { 1, 1, 2 }),
	};
	const auto expected = interpreter.RunForward(Detail::BuildExecutablePlanFromGraph(lowered), inputs);

	auto artifact = Compiler<CPU>::CompileArtifact(Detail::BuildExecutablePlanFromGraph(lowered));
	EXPECT_EQ(artifact.InputSpecs().size(), 3u);
	EXPECT_EQ(artifact.OutputSpecs().size(), 3u);
	auto compiled = artifact.Load();
	const auto outputs = compiled.RunTensors(inputs);

	ASSERT_EQ(expected.size(), 3u);
	ASSERT_EQ(outputs.size(), 3u);
	for (std::size_t i = 0; i < outputs.size(); ++i)
	{
		ExpectTensorNear(outputs[i], expected[i], tolerance);
	}
}
#endif

TEST(GGUFLLaMACausalLM, LowersLinearRopeScaling)
{
	auto archive =
	    CopyArchiveWithMetadataOverride(BuildTinyLLaMAArchive(), "llama.rope.scaling.type", std::string("linear"));
	archive = CopyArchiveWithMetadataOverride(archive, "llama.rope.scaling.factor", 2.0);
	const auto lowered = GGUF::LowerLLaMACausalLM(archive, 2, 1);

	Runtime::Interpreter<CPU> interpreter;
	std::array<Tensor<CPU>, 1> inputs = { MakeInt32Tensor({ 0, 1 }, { 2 }) };
	const auto outputs = interpreter.RunForward(Detail::BuildExecutablePlanFromGraph(lowered), inputs);
	ASSERT_EQ(outputs.size(), 1u);
	EXPECT_EQ(outputs[0].Shape().ToOwned(), std::vector<std::size_t>({ 2, 3 }));
}

TEST(GGUFLLaMACausalLM, RejectsUnsupportedRopeScalingTypeWithActionableDiagnostic)
{
	const auto archive =
	    CopyArchiveWithMetadataOverride(BuildTinyLLaMAArchive(), "llama.rope.scaling.type", std::string("yarn"));
	try
	{
		static_cast<void>(GGUF::LowerLLaMACausalLM(archive, 1));
		FAIL() << "Expected unsupported RoPE scaling type to fail";
	}
	catch (const std::runtime_error& ex)
	{
		const std::string message = ex.what();
		EXPECT_NE(message.find("only executes none/linear scaling"), std::string::npos);
	}
}

TEST(GGUFLLaMACausalLM, RejectsDecodePositionOffsetMismatchWithActionableDiagnostic)
{
	const auto archive = BuildTinyLLaMAArchive();

	try
	{
		static_cast<void>(GGUF::LowerLLaMACausalLMDecode(archive, 1, 2, 1));
		FAIL() << "Expected unsupported decode cache position mismatch to fail";
	}
	catch (const std::runtime_error& ex)
	{
		const std::string message = ex.what();
		EXPECT_NE(message.find("positionOffset == pastLength"), std::string::npos);
	}
}

TEST(GGUFLLaMACausalLM, DefinesParityToleranceByDTypeAndQuantizationFormat)
{
	const auto f32 = GGUF::GetLLaMAParityTolerance(DataType::Float32);
	EXPECT_LE(f32.absolute, 1.0e-5);
	EXPECT_LE(f32.relative, 1.0e-5);

	const auto f16 = GGUF::GetLLaMAParityTolerance(DataType::Float16);
	EXPECT_GT(f16.absolute, f32.absolute);
	EXPECT_GT(f16.relative, f32.relative);

	const auto q8 = GGUF::GetLLaMAParityTolerance(DataType::Float32, QuantizedBlockFormat::GGML_Q8_0);
	const auto q4 = GGUF::GetLLaMAParityTolerance(DataType::Float32, QuantizedBlockFormat::GGML_Q4_0);
	EXPECT_GT(q8.absolute, f32.absolute);
	EXPECT_GT(q4.absolute, q8.absolute);
}

TEST(GGUFLLaMACausalLM, LowersQuantizedWeightsByDequantizingDuringImport)
{
	const auto archive = BuildQuantizedFriendlyLLaMAArchive();
	const auto quantizedArchive = QuantizeQ80Weights(archive);
	const auto plainLowered = GGUF::LowerLLaMACausalLM(archive, 2);
	const auto quantizedLowered = GGUF::LowerLLaMACausalLM(quantizedArchive, 2);

	for (std::size_t i = 0; i < quantizedArchive.VariableCount(); ++i)
	{
		if (IsQ80QuantizationTarget(quantizedArchive.VariableName(i)))
		{
			ASSERT_TRUE(quantizedArchive.GetVariable(i)->IsQuantized());
		}
	}
	for (std::size_t i = 0; i < quantizedLowered.VariableCount(); ++i)
	{
		EXPECT_FALSE(quantizedLowered.GetVariable(i)->IsQuantized());
	}

	Runtime::Interpreter<CPU> interpreter;
	std::array<Tensor<CPU>, 1> inputs = { MakeInt32Tensor({ 0, 1 }, { 2 }) };
	const auto plainOutputs = interpreter.RunForward(Detail::BuildExecutablePlanFromGraph(plainLowered), inputs);
	const auto quantizedOutputs =
	    interpreter.RunForward(Detail::BuildExecutablePlanFromGraph(quantizedLowered), inputs);
	ASSERT_EQ(plainOutputs.size(), 1u);
	ASSERT_EQ(quantizedOutputs.size(), 1u);
	ASSERT_EQ(plainOutputs[0].NumElements(), quantizedOutputs[0].NumElements());
	for (std::size_t i = 0; i < plainOutputs[0].NumElements(); ++i)
	{
		EXPECT_NEAR(ReadFloat(quantizedOutputs[0], i), ReadFloat(plainOutputs[0], i), 1e-6f);
	}
}
