#include <gtest/gtest.h>

#include <GGMLQuantizedKernels.h>
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

#ifdef LITENN_ENABLE_MLIR
extern "C" void
litenn_cpu_ggml_block_matmul_f32(const float*, const float* lhsAligned, std::int64_t lhsOffset, std::int64_t lhsRows,
                                 std::int64_t lhsColumns, std::int64_t lhsRowStride, std::int64_t lhsColumnStride,
                                 const std::uint8_t*, const std::uint8_t* rhsAligned, std::int64_t rhsOffset,
                                 std::int64_t rhsBytes, std::int64_t rhsStride, float*, float* outAligned,
                                 std::int64_t outOffset, std::int64_t outRows, std::int64_t outColumns,
                                 std::int64_t outRowStride, std::int64_t outColumnStride, std::uint64_t formatValue,
                                 std::uint64_t requestedThreadCount, std::uint64_t affinityPolicyValue);

extern "C" void litenn_cpu_ggml_block_matmul_q8k_staged_f32(
    const float*, const float* lhsAligned, std::int64_t lhsOffset, std::int64_t lhsRows, std::int64_t lhsColumns,
    std::int64_t lhsRowStride, std::int64_t lhsColumnStride, const std::uint8_t*, const std::uint8_t* rhsAligned,
    std::int64_t rhsOffset, std::int64_t rhsBytes, std::int64_t rhsStride, float*, float* outAligned,
    std::int64_t outOffset, std::int64_t outRows, std::int64_t outColumns, std::int64_t outRowStride,
    std::int64_t outColumnStride, std::uint64_t formatValue, std::uint64_t requestedThreadCount,
    std::uint64_t affinityPolicyValue);

extern "C" void litenn_cpu_scatter_update_axis0_f32_rank3(
    const float*, const float* dataAligned, std::int64_t dataOffset, std::int64_t dataDim0, std::int64_t dataDim1,
    std::int64_t dataDim2, std::int64_t dataStride0, std::int64_t dataStride1, std::int64_t dataStride2,
    const std::int64_t*, const std::int64_t* indicesAligned, std::int64_t indicesOffset, std::int64_t indicesSize,
    std::int64_t indicesStride, const float*, const float* updatesAligned, std::int64_t updatesOffset,
    std::int64_t updatesDim0, std::int64_t updatesDim1, std::int64_t updatesDim2, std::int64_t updatesStride0,
    std::int64_t updatesStride1, std::int64_t updatesStride2, float*, float* outAligned, std::int64_t outOffset,
    std::int64_t outDim0, std::int64_t outDim1, std::int64_t outDim2, std::int64_t outStride0, std::int64_t outStride1,
    std::int64_t outStride2);
#endif

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

	bool ByteSpanContains(std::span<const std::byte> bytes, std::string_view text)
	{
		const auto* begin = reinterpret_cast<const char*>(bytes.data());
		return std::string_view(begin, bytes.size()).find(text) != std::string_view::npos;
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

	void ExpectValuesNear(std::span<const float> actual, std::span<const float> expected,
	                      GGUF::LLaMAParityTolerance tolerance)
	{
		ASSERT_EQ(actual.size(), expected.size());
		for (std::size_t i = 0; i < expected.size(); ++i)
		{
			const auto allowed = tolerance.absolute + tolerance.relative * std::fabs(expected[i]);
			EXPECT_LE(std::fabs(actual[i] - expected[i]), allowed) << "at element " << i;
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

	Tensor<CPU> MakeInt64Tensor(std::initializer_list<std::int64_t> values, std::initializer_list<std::size_t> shape)
	{
		CPU device;
		Tensor<CPU> tensor(Uninitialized, shape, DataType::Int64, device);
		DeviceTraits<CPU>::CopyFromCPU(device, DataType::Int64, tensor.UnsafeRawData(), DataType::Int64, values.begin(),
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

	Graph BuildTinyQwen2ArchiveWithQ4_0Payload()
	{
		auto graph = BuildTinyQwen2Archive();
		Tensor<CPU> storage(Uninitialized, { 18 }, DataType::UInt8);
		const auto variable = graph.AddVariable(Variable::CreateQuantized(
		    std::move(storage), BlockQuantization(QuantizedBlockFormat::GGML_Q4_0, { 32 }, DataType::Float32)));
		graph.SetVariableName(variable, "diagnostic.q4_0.weight");
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

	std::shared_ptr<Variable> QuantizeGGMLVariable(const Variable& source, ggml_type type,
	                                               QuantizedBlockFormat blockFormat)
	{
		const auto data = source.Data().CopyToDevice(CPU{});
		if (data.DType() != DataType::Float32)
		{
			throw std::runtime_error("QuantizeGGMLVariable expects Float32 source tensors");
		}
		if (data.Shape().NumDim() == 0)
		{
			throw std::runtime_error("QuantizeGGMLVariable requires at least 1D tensors");
		}

		const auto* traits = ggml_get_type_traits(type);
		if (!traits || !traits->from_float_ref)
		{
			throw std::runtime_error("GGML reference quantizer is unavailable");
		}

		const auto rowSize = data.Shape()[data.Shape().NumDim() - 1];
		if ((rowSize % static_cast<std::size_t>(traits->blck_size)) != 0)
		{
			throw std::runtime_error("QuantizeGGMLVariable row width is incompatible with the block size");
		}

		const auto rowCount = data.NumElements() / rowSize;
		const auto rowBytes = ggml_row_size(type, static_cast<std::int64_t>(rowSize));
		Tensor<CPU> storage(Uninitialized, { rowCount * rowBytes }, DataType::UInt8);
		const auto* src = static_cast<const float*>(data.UnsafeRawData());
		auto* dst = static_cast<std::uint8_t*>(storage.UnsafeRawData());
		for (std::size_t row = 0; row < rowCount; ++row)
		{
			traits->from_float_ref(src + row * rowSize, dst + row * rowBytes, static_cast<int64_t>(rowSize));
		}

		return Variable::CreateQuantized(std::move(storage),
		                                 BlockQuantization(blockFormat, data.Shape().ToOwned(), DataType::Float32));
	}

	std::shared_ptr<Variable> QuantizeQ80Variable(const Variable& source)
	{
		return QuantizeGGMLVariable(source, GGML_TYPE_Q8_0, QuantizedBlockFormat::GGML_Q8_0);
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

TEST(GGUFImporter, ConvertGGUFArchivePreservesQuantizedExternalWeights)
{
	const auto inputPath = WriteSupportedFixture();
	const auto outputPath = MakeTempFixturePath("litenn_gguf_external_archive", ".ltnn");
	const auto weightsPath = MakeTempFixturePath("litenn_gguf_external_archive", ".weights.bin");
	std::filesystem::remove(outputPath);
	std::filesystem::remove(weightsPath);

	const auto summary = GGUF::ConvertGGUFArchiveExternalWeights(inputPath, outputPath, weightsPath);
	const auto loaded = Serialization::LoadVNextModelPackage(outputPath);

	EXPECT_EQ(summary.tensorCount, 2u);
	ASSERT_EQ(loaded.manifest.tensors.size(), 2u);
	EXPECT_EQ(loaded.manifest.tensors[0].kind, ExternalBufferKind::User);
	EXPECT_EQ(loaded.manifest.tensors[1].kind, ExternalBufferKind::User);
	EXPECT_EQ(loaded.manifest.tensors[0].relativePath, weightsPath.filename().string());
	EXPECT_EQ(loaded.manifest.tensors[1].relativePath, weightsPath.filename().string());
	ASSERT_TRUE(loaded.manifest.tensors[1].quantization.has_value());
	EXPECT_EQ(loaded.manifest.tensors[1].quantization->blockFormat, QuantizedBlockFormat::GGML_Q4_0);
	EXPECT_EQ(loaded.manifest.tensors[1].quantization->expressedShape, std::vector<std::size_t>({ 32 }));
	ASSERT_EQ(loaded.plan.variables.size(), 2u);
	ASSERT_TRUE(loaded.plan.variables[1].quantization.has_value());
	EXPECT_EQ(loaded.plan.variables[1].type.dtype, DataType::UInt8);
	EXPECT_EQ(loaded.plan.variables[1].region.byteOffset, 64u);
	EXPECT_EQ(loaded.plan.variables[1].region.byteSize, 18u);
	EXPECT_EQ(std::filesystem::file_size(weightsPath), 82u);

	std::filesystem::remove(inputPath);
	std::filesystem::remove(outputPath);
	std::filesystem::remove(weightsPath);
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
	EXPECT_NE(qwen2.unsupportedPolicy.find("native CUDA quantized"), std::string_view::npos);
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
		       diagnostic.message.find("native CUDA quantized paths") != std::string::npos;
	}));
	EXPECT_TRUE(std::ranges::any_of(report.diagnostics, [](const GGUF::LLaMACompatibilityDiagnostic& diagnostic) {
		return !diagnostic.blocking && diagnostic.subject == "qwen2.decode-loop" &&
		       diagnostic.message.find("decode-loop tooling") != std::string::npos;
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

TEST(GGUFLLMGeneration, TokenizesExactVocabularyPromptWithOptionalBos)
{
	const auto archive = BuildTinyQwen2Archive();

	auto withBos = GGUF::MakeExactVocabularyPromptTokens("hello", archive);
	EXPECT_FALSE(withBos.callerProvided);
	EXPECT_EQ(withBos.tokenIds, std::vector<std::int32_t>({ 0, 1 }));

	auto withoutBos = GGUF::MakeExactVocabularyPromptTokens("hellohello", archive, false);
	EXPECT_EQ(withoutBos.tokenIds, std::vector<std::int32_t>({ 1, 1 }));
}

TEST(GGUFLLMGeneration, RejectsPromptTextOutsideExactVocabulary)
{
	const auto archive = BuildTinyQwen2Archive();

	EXPECT_THROW(static_cast<void>(GGUF::MakeExactVocabularyPromptTokens("unknown", archive)), std::runtime_error);
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
	EXPECT_EQ(plan.decodeStep.kvCaches[0].stateType.StaticShape(), std::vector<std::size_t>({ 2, 4, 1, 2 }));
	EXPECT_EQ(plan.decodeStep.kvCaches[0].stateBinding.name, "kv.layer0");
	EXPECT_EQ(plan.decodeStep.kvCaches[0].stateBinding.kind, Runtime::RuntimeStateKind::KVCache);
	EXPECT_EQ(plan.decodeStep.kvCaches[0].keyByteOffset, 0u);
	EXPECT_EQ(plan.decodeStep.kvCaches[0].valueByteOffset, 32u);
	EXPECT_EQ(plan.decodeStep.kvCaches[0].layerByteStride, 64u);
	EXPECT_EQ(plan.decodeStep.kvCaches[0].tokenByteStride, 8u);
	ASSERT_EQ(plan.decodeStep.stateValueBindings.size(), 4u);
	EXPECT_EQ(plan.decodeStep.stateValueBindings[0].kind, Runtime::RuntimeStateValueKind::FunctionInput);
	EXPECT_EQ(plan.decodeStep.stateValueBindings[0].valueIndex, 1u);
	EXPECT_EQ(plan.decodeStep.stateValueBindings[1].stateByteOffset, 32u);
	EXPECT_EQ(plan.decodeStep.stateValueBindings[2].kind, Runtime::RuntimeStateValueKind::FunctionOutput);
	EXPECT_EQ(plan.decodeStep.stateValueBindings[3].stateByteOffset, 32u);
	ASSERT_EQ(plan.decodeStateABI.kvCaches.size(), 1u);
	EXPECT_EQ(plan.decodeStateABI.kvCaches[0].name, "kv.layer0");
	EXPECT_TRUE(std::ranges::contains(plan.decodeStateABI.kvCaches[0].effects, std::string("write")));
	ASSERT_TRUE(plan.decodeStateABI.currentPosition.has_value());
	EXPECT_EQ(plan.decodeStateABI.currentPosition->name, "decode.position");
	EXPECT_TRUE(std::ranges::contains(plan.decodeStateABI.currentPosition->effects, std::string("increment")));
}

TEST(GGUFLLaMAArtifacts, BuildsDecodeRuntimeScheduleWithPersistentCacheAliases)
{
	const auto archive = BuildTinyQwen2Archive();
	const auto schedule = GGUF::BuildLLaMADecodeRuntimeSchedule(
	    archive, { .prefillSequenceLength = 4, .decodePastLength = 3, .maxCacheLength = 8 });

	ASSERT_EQ(schedule.states.size(), 2u);
	EXPECT_EQ(schedule.states[0].name, "kv.layer0");
	EXPECT_EQ(schedule.states[1].name, "decode.position");
	ASSERT_EQ(schedule.stateValueBindings.size(), 4u);
	const auto cacheBuffer = *schedule.states[0].memoryBuffer;
	const auto& subgraph = schedule.module.plan.subgraphs[schedule.module.plan.forward];
	const auto findParam = [&](std::size_t paramIndex) {
		const auto node = std::ranges::find_if(subgraph.nodes, [&](const auto& entry) {
			const auto* param = std::get_if<ParamRefNode>(&entry.node);
			return param != nullptr && param->paramIndex == paramIndex;
		});
		return NodeOutput{ node->sourceNode, 0 };
	};
	const auto* keyInput = FindMemoryAssignment(schedule.memory, subgraph.sourceSubgraph, findParam(1));
	const auto* valueInput = FindMemoryAssignment(schedule.memory, subgraph.sourceSubgraph, findParam(2));
	const auto* keyOutput = FindMemoryAssignment(schedule.memory, subgraph.sourceSubgraph, subgraph.results[1]);
	const auto* valueOutput = FindMemoryAssignment(schedule.memory, subgraph.sourceSubgraph, subgraph.results[2]);
	ASSERT_NE(keyInput, nullptr);
	ASSERT_NE(valueInput, nullptr);
	ASSERT_NE(keyOutput, nullptr);
	ASSERT_NE(valueOutput, nullptr);
	EXPECT_EQ(keyInput->buffer, cacheBuffer);
	EXPECT_EQ(keyInput->offset, 0u);
	EXPECT_EQ(valueInput->buffer, cacheBuffer);
	EXPECT_EQ(valueInput->offset, 64u);
	EXPECT_EQ(keyOutput->buffer, cacheBuffer);
	EXPECT_EQ(keyOutput->offset, 0u);
	EXPECT_EQ(valueOutput->buffer, cacheBuffer);
	EXPECT_EQ(valueOutput->offset, 64u);
	EXPECT_NO_THROW(Runtime::ValidateRuntimeSchedule(schedule));

	const auto path = MakeTempFixturePath("litenn_llama_decode_schedule", ".ltnn");
	const auto weightsPath = MakeTempFixturePath("litenn_llama_decode_schedule", ".weights.bin");
	Serialization::SaveVNextModelPackageExternalWeights(schedule, path, weightsPath);
	const auto loaded = Serialization::LoadVNextModelPackage(path);
	std::filesystem::remove(path);
	std::filesystem::remove(weightsPath);
	ASSERT_EQ(loaded.manifest.runtimeStates.size(), 2u);
	ASSERT_EQ(loaded.manifest.stateValueBindings.size(), 4u);
	EXPECT_EQ(loaded.manifest.stateValueBindings[1].stateName, "kv.layer0");
	EXPECT_EQ(loaded.manifest.stateValueBindings[1].stateByteOffset, 64u);
	ASSERT_FALSE(loaded.plan.variables.empty());
	EXPECT_NE(loaded.plan.variables[0].region.data, nullptr);
	EXPECT_EQ(loaded.manifest.tensors[0].name, "token_embd.weight");
	EXPECT_EQ(loaded.manifest.tensors[0].kind, ExternalBufferKind::User);

#ifdef LITENN_ENABLE_MLIR
	const std::vector<float> zeroCache(6, 0.0f);
	std::array<Tensor<CPU>, 3> inputs = {
		MakeInt32Tensor({ 1 }, { 1 }),
		MakeFloatTensor(zeroCache, { 3, 1, 2 }),
		MakeFloatTensor(zeroCache, { 3, 1, 2 }),
	};
	Runtime::Interpreter<CPU> interpreter;
	const auto expected = interpreter.RunForward(loaded.plan, inputs);
	auto compiled = Compiler<CPU>::CompileArtifact(loaded.plan).Load();
	const auto actual = compiled.RunTensors(inputs);
	ASSERT_EQ(actual.size(), expected.size());
	for (std::size_t i = 0; i < actual.size(); ++i)
	{
		ExpectTensorNear(actual[i], expected[i], GGUF::GetLLaMAParityTolerance(DataType::Float32));
	}
#endif
}

TEST(GGUFLLaMAArtifacts, PlansCapacityDecodeWithDynamicPositionState)
{
	const auto plan = GGUF::PlanLLaMAArtifacts(
	    BuildTinyQwen2Archive(),
	    { .prefillSequenceLength = 4, .decodePastLength = 0, .maxCacheLength = 8, .dynamicDecodePosition = true });

	EXPECT_TRUE(plan.decodeStep.dynamicPosition);
	EXPECT_EQ(plan.decodeStep.inputNames,
	          std::vector<std::string>({ "token_ids", "current_position", "past_key_0", "past_value_0" }));
	EXPECT_EQ(plan.decodeStep.outputNames,
	          std::vector<std::string>({ "logits", "next_position", "updated_key_0", "updated_value_0" }));
	ASSERT_EQ(plan.decodeStep.kvCaches.size(), 1u);
	EXPECT_EQ(plan.decodeStep.kvCaches[0].cacheType.StaticShape(), std::vector<std::size_t>({ 8, 1, 2 }));
	ASSERT_EQ(plan.decodeStep.stateValueBindings.size(), 6u);
	EXPECT_EQ(plan.decodeStep.stateValueBindings[0].valueIndex, 2u);
	EXPECT_EQ(plan.decodeStep.stateValueBindings[2].valueIndex, 2u);
	EXPECT_EQ(plan.decodeStep.stateValueBindings[4].stateName, "decode.position");
	EXPECT_EQ(plan.decodeStep.stateValueBindings[4].kind, Runtime::RuntimeStateValueKind::FunctionInput);
	EXPECT_EQ(plan.decodeStep.stateValueBindings[4].valueIndex, 1u);
	EXPECT_EQ(plan.decodeStep.stateValueBindings[5].kind, Runtime::RuntimeStateValueKind::FunctionOutput);
	EXPECT_EQ(plan.decodeStep.stateValueBindings[5].valueIndex, 1u);
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

TEST(GGUFLLaMAQuantizedExecution, PlansNativeAndReferenceQuantizedPolicies)
{
	const auto archive = BuildTinyQwen2ArchiveWithQ4KPayload();
	const auto plan = GGUF::PlanLLaMAQuantizedWeightExecution(archive);

	EXPECT_TRUE(plan.lowerable);
	EXPECT_EQ(plan.tensorCount, 1u);
	EXPECT_EQ(plan.storedBytes, 144u);
	EXPECT_EQ(plan.dequantizedBytes, 1024u);
	ASSERT_EQ(plan.decisions.size(), 1u);
	EXPECT_EQ(plan.decisions[0].format, QuantizedBlockFormat::GGML_Q4_K);
	EXPECT_EQ(plan.decisions[0].selectedPolicy, GGUF::LLaMAQuantizedExecutionPolicy::CUDANativeQuantized);
	EXPECT_EQ(GGUF::LLaMAQuantizedExecutionPolicyName(plan.decisions[0].selectedPolicy), "cuda-native-quantized");
	EXPECT_FALSE(plan.decisions[0].blocking);

	const auto nativeUnderBudget = GGUF::PlanLLaMAQuantizedWeightExecution(archive, 512);
	EXPECT_TRUE(nativeUnderBudget.lowerable);
	EXPECT_EQ(nativeUnderBudget.decisions[0].selectedPolicy, GGUF::LLaMAQuantizedExecutionPolicy::CUDANativeQuantized);

	const auto rejected = GGUF::PlanLLaMAQuantizedWeightExecution(BuildTinyQwen2ArchiveWithQ4_0Payload(), 64);
	EXPECT_FALSE(rejected.lowerable);
	ASSERT_EQ(rejected.decisions.size(), 1u);
	EXPECT_EQ(rejected.decisions[0].selectedPolicy, GGUF::LLaMAQuantizedExecutionPolicy::Reject);
	EXPECT_TRUE(rejected.decisions[0].blocking);
}

TEST(GGUFLLaMAQuantizedExecution, RunsOutputMajorQ4KMatMulWithoutMaterializingWeight)
{
	constexpr std::size_t inFeatures = 256;
	constexpr std::size_t outFeatures = 2;
	std::vector<float> weightValues(outFeatures * inFeatures);
	for (std::size_t i = 0; i < weightValues.size(); ++i)
	{
		weightValues[i] = static_cast<float>(static_cast<int>(i % 17) - 8) * 0.125F;
	}
	const auto plainWeight = Variable::Create(MakeFloatTensor(weightValues, { outFeatures, inFeatures }));
	const auto quantizedWeight = QuantizeGGMLVariable(*plainWeight, GGML_TYPE_Q4_K, QuantizedBlockFormat::GGML_Q4_K);

	std::vector<float> inputValues(2 * inFeatures, 1.0F);
	std::fill(inputValues.begin() + inFeatures, inputValues.end(), -0.5F);
	const auto input = MakeFloatTensor(inputValues, { 2, inFeatures });
	const auto actual = GGUF::EvalGGMLQuantizedMatMul(input, *quantizedWeight, true);
	const auto dequantized = GGUF::DequantizeGGMLBlockVariable(*quantizedWeight, "q4_k.weight");

	ASSERT_EQ(actual.Shape(), (ShapeView{ 2, outFeatures }));
	const auto* inputData = static_cast<const float*>(input.UnsafeRawData());
	const auto* weightData = static_cast<const float*>(dequantized.UnsafeRawData());
	for (std::size_t row = 0; row < 2; ++row)
	{
		for (std::size_t column = 0; column < outFeatures; ++column)
		{
			float expected = 0.0F;
			for (std::size_t k = 0; k < inFeatures; ++k)
			{
				expected += inputData[row * inFeatures + k] * weightData[column * inFeatures + k];
			}
			EXPECT_NEAR(ReadFloat(actual, row * outFeatures + column), expected, 1.0e-3F);
		}
	}
	EXPECT_THROW((void) GGUF::EvalGGMLQuantizedMatMul(input, *quantizedWeight, false), std::runtime_error);

	Graph graph;
	const auto weightVariable = graph.AddVariable(quantizedWeight);
	const Layer::LinearLayer layer{
		.weightVariable = weightVariable,
		.inFeatures = inFeatures,
		.outFeatures = outFeatures,
		.dtype = DataType::Float32,
		.weightQuantization = *quantizedWeight->Quantization(),
		.weightStorageShape = quantizedWeight->Data().Shape().ToOwned(),
		.transposeWeight = true,
	};
	Subgraph forward;
	const auto inputNode = forward.AddParam(DataType::Float32, { 2, inFeatures });
	const auto output = Layer::AddLinear(forward, layer, { inputNode, 0 });
	forward.SetResults({ output });
	graph.SetForward(graph.AddSubgraph(std::move(forward)));
	graph.SetInputNames({ "input" });
	graph.SetOutputNames({ "output" });

	const auto packagePath = MakeTempFixturePath("litenn_ggml_q4k_matmul", ".ltnn");
	const auto weightsPath = MakeTempFixturePath("litenn_ggml_q4k_matmul", ".weights.bin");
	Serialization::ExternalWeightSaveOptions saveOptions;
	saveOptions.minVariableBytes = 0;
	Serialization::SaveVNextModelPackageExternalWeights(graph, packagePath, weightsPath, saveOptions);
	const auto loaded = Serialization::LoadVNextModelPackage(packagePath);
	std::filesystem::remove(packagePath);
	std::filesystem::remove(weightsPath);
	Runtime::Interpreter<CPU> interpreter(GGUF::TryEvalGGMLQuantizedMatMul);
	std::array<Tensor<CPU>, 1> inputs{ input };
	const auto graphOutputs = interpreter.RunForward(loaded.plan, inputs);
	ASSERT_EQ(graphOutputs.size(), 1u);
	for (std::size_t i = 0; i < actual.NumElements(); ++i)
	{
		EXPECT_FLOAT_EQ(ReadFloat(graphOutputs[0], i), ReadFloat(actual, i));
	}

#ifdef LITENN_ENABLE_MLIR
	CompilerOptions compileOptions;
	compileOptions.cpuAOTThreadCount = 2;
	compileOptions.cpuAOTAffinityPolicy = CPUAOTAffinityPolicy::Compact;
	const auto artifact = Compiler<CPU>::CompileArtifact(loaded.plan, compileOptions);
	EXPECT_TRUE(ByteSpanContains(artifact.Instructions(), "litenn_cpu_ggml_block_matmul_f32"));
	auto compiled = artifact.Load();
	const auto compiledOutputs = compiled.RunTensors(inputs);
	ASSERT_EQ(compiledOutputs.size(), 1u);
	ExpectTensorNear(compiledOutputs[0], actual, { .absolute = 1.0e-3, .relative = 1.0e-5 });
#endif
}

#ifdef LITENN_ENABLE_MLIR
TEST(GGUFLLaMAQuantizedExecution, CompilesQ4KTokenEmbeddingGatherWithoutInterpreter)
{
	constexpr std::size_t rowCount = 3;
	constexpr std::size_t rowWidth = 256;
	std::vector<float> tableValues(rowCount * rowWidth);
	for (std::size_t i = 0; i < tableValues.size(); ++i)
	{
		tableValues[i] = static_cast<float>(static_cast<int>(i % 29) - 14) * 0.0625F;
	}
	const auto plainTable = Variable::Create(MakeFloatTensor(tableValues, { rowCount, rowWidth }));
	const auto quantizedTable = QuantizeGGMLVariable(*plainTable, GGML_TYPE_Q4_K, QuantizedBlockFormat::GGML_Q4_K);
	const auto dequantized = GGUF::DequantizeGGMLBlockVariable(*quantizedTable, "q4_k.embedding");

	Graph graph;
	const auto tableVariable = graph.AddVariable(quantizedTable);
	Subgraph forward;
	const auto indices = forward.AddParam(DataType::Int32, { 2 });
	const auto storage =
	    forward.AddNode(VariableRefNode{ tableVariable },
	                    std::vector<OutputInfo>{
	                        OutputInfo{ quantizedTable->Data().DType(), quantizedTable->Data().Shape().ToOwned() } });
	const auto rows =
	    forward.AddNode(QuantizedGetRowsNode{ { storage, 0 }, { indices, 0 }, *quantizedTable->Quantization() },
	                    std::vector<OutputInfo>{ OutputInfo{ DataType::Float32, { 2, rowWidth } } });
	forward.SetResults(std::vector<NodeOutput>{ { rows, 0 } });
	graph.SetForward(graph.AddSubgraph(std::move(forward)));
	graph.SetInputNames({ "token_ids" });
	graph.SetOutputNames({ "embeddings" });

	Tensor<CPU> tokenIds(Uninitialized, { 2 }, DataType::Int32);
	const std::array<std::int32_t, 2> tokenIdValues{ 2, 0 };
	CPU cpu;
	DeviceTraits<CPU>::CopyFromCPU(cpu, DataType::Int32, tokenIds.UnsafeRawData(), DataType::Int32,
	                               tokenIdValues.data(), tokenIdValues.size());
	const std::array<Tensor<CPU>, 1> inputs{ tokenIds };
	const auto artifact = Compiler<CPU>::CompileArtifact(Detail::BuildExecutablePlanFromGraph(graph));
	EXPECT_TRUE(ByteSpanContains(artifact.Instructions(), "litenn_cpu_ggml_block_get_rows_i32_f32"));
	auto compiled = artifact.Load();
	const auto outputs = compiled.RunTensors(inputs);

	ASSERT_EQ(outputs.size(), 1u);
	ASSERT_EQ(outputs[0].Shape(), (ShapeView{ 2, rowWidth }));
	const auto* expected = static_cast<const float*>(dequantized.UnsafeRawData());
	for (std::size_t column = 0; column < rowWidth; ++column)
	{
		EXPECT_NEAR(ReadFloat(outputs[0], column), expected[2 * rowWidth + column], 1.0e-6F);
		EXPECT_NEAR(ReadFloat(outputs[0], rowWidth + column), expected[column], 1.0e-6F);
	}
}
#endif

#ifdef LITENN_ENABLE_MLIR
TEST(GGUFLLaMAQuantizedExecution, CompilesOutputMajorQ5KQ6KAndQ8_0MatMulWithoutMaterializingWeight)
{
	constexpr std::size_t inFeatures = 256;
	constexpr std::size_t outFeatures = 3;
	const std::array cases = {
		std::tuple{ GGML_TYPE_Q5_K, QuantizedBlockFormat::GGML_Q5_K, "q5_k.weight" },
		std::tuple{ GGML_TYPE_Q6_K, QuantizedBlockFormat::GGML_Q6_K, "q6_k.weight" },
		std::tuple{ GGML_TYPE_Q8_0, QuantizedBlockFormat::GGML_Q8_0, "q8_0.weight" },
	};
	for (const auto& [ggmlType, blockFormat, name] : cases)
	{
		SCOPED_TRACE(name);
		std::vector<float> weightValues(outFeatures * inFeatures);
		for (std::size_t i = 0; i < weightValues.size(); ++i)
		{
			weightValues[i] = static_cast<float>(static_cast<int>(i % 13) - 6) * 0.25F;
		}
		const auto plainWeight = Variable::Create(MakeFloatTensor(weightValues, { outFeatures, inFeatures }));
		const auto quantizedWeight = QuantizeGGMLVariable(*plainWeight, ggmlType, blockFormat);

		Graph graph;
		const auto weightVariable = graph.AddVariable(quantizedWeight);
		const Layer::LinearLayer layer{
			.weightVariable = weightVariable,
			.inFeatures = inFeatures,
			.outFeatures = outFeatures,
			.dtype = DataType::Float32,
			.weightQuantization = *quantizedWeight->Quantization(),
			.weightStorageShape = quantizedWeight->Data().Shape().ToOwned(),
			.transposeWeight = true,
		};
		Subgraph forward;
		const auto inputNode = forward.AddParam(DataType::Float32, { 2, inFeatures });
		const auto output = Layer::AddLinear(forward, layer, { inputNode, 0 });
		forward.SetResults({ output });
		graph.SetForward(graph.AddSubgraph(std::move(forward)));

		std::vector<float> inputValues(2 * inFeatures);
		for (std::size_t i = 0; i < inputValues.size(); ++i)
		{
			inputValues[i] = static_cast<float>(static_cast<int>(i % 9) - 4) * 0.5F;
		}
		std::array<Tensor<CPU>, 1> inputs = { MakeFloatTensor(inputValues, { 2, inFeatures }) };
		const auto plan = Detail::BuildExecutablePlanFromGraph(graph);
		const auto dequantized = GGUF::DequantizeGGMLBlockVariable(*quantizedWeight, name);
		Tensor<CPU> expected(Uninitialized, { 2, outFeatures }, DataType::Float32);
		const auto* weightData = static_cast<const float*>(dequantized.UnsafeRawData());
		auto* expectedData = static_cast<float*>(expected.UnsafeRawData());
		for (std::size_t row = 0; row < 2; ++row)
		{
			for (std::size_t column = 0; column < outFeatures; ++column)
			{
				float sum = 0.0F;
				for (std::size_t reduction = 0; reduction < inFeatures; ++reduction)
				{
					sum += inputValues[row * inFeatures + reduction] * weightData[column * inFeatures + reduction];
				}
				expectedData[row * outFeatures + column] = sum;
			}
		}
		const auto artifact = Compiler<CPU>::CompileArtifact(plan);
		EXPECT_TRUE(ByteSpanContains(artifact.Instructions(), "litenn_cpu_ggml_block_matmul_f32"));
		auto compiled = artifact.Load();
		const auto actual = compiled.RunTensors(inputs);
		ASSERT_EQ(actual.size(), 1u);
		ExpectTensorNear(actual[0], expected, { .absolute = 2.0e-3, .relative = 1.0e-5 });
	}
}

TEST(GGUFLLaMAQuantizedExecution, Q8KStagedHelperMatchesDirectHelperForExactActivationRows)
{
	constexpr std::size_t inFeatures = 256;
	constexpr std::size_t outFeatures = 5;
	constexpr std::size_t rows = 2;
	const std::array cases = {
		std::tuple{ GGML_TYPE_Q4_K, QuantizedBlockFormat::GGML_Q4_K, "q4_k.weight" },
		std::tuple{ GGML_TYPE_Q5_K, QuantizedBlockFormat::GGML_Q5_K, "q5_k.weight" },
		std::tuple{ GGML_TYPE_Q6_K, QuantizedBlockFormat::GGML_Q6_K, "q6_k.weight" },
	};

	std::vector<float> inputValues(rows * inFeatures, 1.0F);
	std::fill(inputValues.begin() + inFeatures, inputValues.end(), -0.5F);

	for (const auto& [ggmlType, blockFormat, name] : cases)
	{
		SCOPED_TRACE(name);
		std::vector<float> weightValues(outFeatures * inFeatures);
		for (std::size_t i = 0; i < weightValues.size(); ++i)
		{
			weightValues[i] = static_cast<float>(static_cast<int>(i % 23) - 11) * 0.125F;
		}
		const auto plainWeight = Variable::Create(MakeFloatTensor(weightValues, { outFeatures, inFeatures }));
		const auto quantizedWeight = QuantizeGGMLVariable(*plainWeight, ggmlType, blockFormat);
		const auto& storage = quantizedWeight->Data();
		const auto* storageBytes = static_cast<const std::uint8_t*>(storage.UnsafeRawData());
		std::vector<float> direct(rows * outFeatures);
		std::vector<float> staged(rows * outFeatures);

		litenn_cpu_ggml_block_matmul_f32(nullptr, inputValues.data(), 0, static_cast<std::int64_t>(rows),
		                                 static_cast<std::int64_t>(inFeatures), static_cast<std::int64_t>(inFeatures),
		                                 1, nullptr, storageBytes, 0, static_cast<std::int64_t>(storage.NumElements()),
		                                 1, nullptr, direct.data(), 0, static_cast<std::int64_t>(rows),
		                                 static_cast<std::int64_t>(outFeatures), static_cast<std::int64_t>(outFeatures),
		                                 1, static_cast<std::uint64_t>(blockFormat), 2,
		                                 static_cast<std::uint64_t>(CPUAOTAffinityPolicy::None));
		litenn_cpu_ggml_block_matmul_q8k_staged_f32(
		    nullptr, inputValues.data(), 0, static_cast<std::int64_t>(rows), static_cast<std::int64_t>(inFeatures),
		    static_cast<std::int64_t>(inFeatures), 1, nullptr, storageBytes, 0,
		    static_cast<std::int64_t>(storage.NumElements()), 1, nullptr, staged.data(), 0,
		    static_cast<std::int64_t>(rows), static_cast<std::int64_t>(outFeatures),
		    static_cast<std::int64_t>(outFeatures), 1, static_cast<std::uint64_t>(blockFormat), 2,
		    static_cast<std::uint64_t>(CPUAOTAffinityPolicy::None));

		for (std::size_t i = 0; i < direct.size(); ++i)
		{
			EXPECT_NEAR(staged[i], direct[i], 1.0e-4F);
		}
	}
}
#endif

TEST(GGUFLLaMACompatibility, ReportsQuantizationMixAndQ4KDiagnostic)
{
	const auto report = GGUF::AnalyzeLLaMACompatibility(BuildTinyQwen2ArchiveWithQ4KPayload(),
	                                                    GGUF::LLaMACompatibilityProfileKind::Qwen2LikeCausalLM);

	EXPECT_TRUE(report.lowerable);
	EXPECT_TRUE(std::ranges::any_of(report.diagnostics, [](const GGUF::LLaMACompatibilityDiagnostic& diagnostic) {
		return !diagnostic.blocking && diagnostic.subject == "quantization.mix" &&
		       diagnostic.message.find("GGML_Q4_K") != std::string::npos &&
		       diagnostic.message.find("policy=cuda-native-quantized") != std::string::npos;
	}));
	EXPECT_TRUE(std::ranges::any_of(report.diagnostics, [](const GGUF::LLaMACompatibilityDiagnostic& diagnostic) {
		return !diagnostic.blocking && diagnostic.subject == "quantization.q4_k_m" &&
		       diagnostic.message.find("CUDA native K-quant projection kernels") != std::string::npos;
	}));
}

TEST(GGUFLLaMACompatibility, AppliesQuantizedDequantizationBudgetAsBlockingDiagnostic)
{
	const auto report = GGUF::AnalyzeLLaMACompatibility(BuildTinyQwen2ArchiveWithQ4_0Payload(),
	                                                    GGUF::LLaMACompatibilityProfileKind::Qwen2LikeCausalLM, 64);

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

TEST(GGUFLLaMACausalLM, ReusesCapacityDecodeGraphAcrossRuntimePositions)
{
	const auto archive = BuildTinyLLaMAArchive();
	const auto capacityDecode = GGUF::LowerLLaMACausalLMDecodeCapacity(archive, 4);
	const auto hyperparameters = GGUF::ParseLLaMAHyperparameters(archive);
	EXPECT_EQ(capacityDecode.SubgraphCount(), hyperparameters.blockCount + 1);
	const auto& capacityForward = capacityDecode.GetSubgraph(capacityDecode.Forward());
	EXPECT_EQ(
	    std::ranges::count_if(capacityForward.Nodes(),
	                          [](const NodeEntry& entry) { return std::holds_alternative<CallNode>(entry.node); }),
	    hyperparameters.blockCount);
	const auto fullPrefill = GGUF::LowerLLaMACausalLM(archive, 2);
	const auto capacityPlan = Detail::BuildExecutablePlanFromGraph(capacityDecode);
	Runtime::Interpreter<CPU> interpreter;
	const std::vector<float> zeroCache(8, 0.0f);
	std::array<Tensor<CPU>, 4> firstInputs = {
		MakeInt32Tensor({ 0 }, { 1 }),
		MakeInt64Tensor({ 0 }, { 1 }),
		MakeFloatTensor(zeroCache, { 4, 1, 2 }),
		MakeFloatTensor(zeroCache, { 4, 1, 2 }),
	};
	const auto firstOutputs = interpreter.RunForward(capacityPlan, firstInputs);
	ASSERT_EQ(firstOutputs.size(), 4u);
	EXPECT_EQ(firstOutputs[1].DType(), DataType::Int64);
	EXPECT_EQ(static_cast<const std::int64_t*>(firstOutputs[1].UnsafeRawData())[0], 1);

	std::array<Tensor<CPU>, 4> secondInputs = {
		MakeInt32Tensor({ 1 }, { 1 }),
		firstOutputs[1],
		firstOutputs[2],
		firstOutputs[3],
	};
	const auto secondOutputs = interpreter.RunForward(capacityPlan, secondInputs);
	ASSERT_EQ(secondOutputs.size(), 4u);
	EXPECT_EQ(static_cast<const std::int64_t*>(secondOutputs[1].UnsafeRawData())[0], 2);

	std::array<Tensor<CPU>, 1> fullInputs = { MakeInt32Tensor({ 0, 1 }, { 2 }) };
	const auto fullOutputs = interpreter.RunForward(Detail::BuildExecutablePlanFromGraph(fullPrefill), fullInputs);
	const std::array<float, 3> fullSecondLogit = {
		ReadFloat(fullOutputs[0], 3),
		ReadFloat(fullOutputs[0], 4),
		ReadFloat(fullOutputs[0], 5),
	};
	ExpectTensorNear(secondOutputs[0], fullSecondLogit, GGUF::GetLLaMAParityTolerance(DataType::Float32));
}

TEST(GGUFLLaMACausalLM, CapacityDecodeUsesGroupedActivePrefixAttention)
{
	const auto archive = BuildTinyLLaMAArchive();
	const auto capacityDecode = GGUF::LowerLLaMACausalLMDecodeCapacity(archive, 4);
	const auto hyperparameters = GGUF::ParseLLaMAHyperparameters(archive);
	std::size_t groupedAttentionNodeCount = 0;
	std::size_t singleHeadAttentionNodeCount = 0;
	for (SubgraphId subgraphId = 0; subgraphId < capacityDecode.SubgraphCount(); ++subgraphId)
	{
		const auto& subgraph = capacityDecode.GetSubgraph(subgraphId);
		for (NodeId nodeId = 0; nodeId < subgraph.NodeCount(); ++nodeId)
		{
			const auto& node = subgraph.GetNodeEntry(nodeId).node;
			if (std::holds_alternative<GroupedActivePrefixAttentionNode>(node))
			{
				++groupedAttentionNodeCount;
			}
			if (std::holds_alternative<ActivePrefixAttentionNode>(node))
			{
				++singleHeadAttentionNodeCount;
			}
		}
	}
	EXPECT_EQ(groupedAttentionNodeCount, hyperparameters.blockCount);
	EXPECT_EQ(singleHeadAttentionNodeCount, 0u);
}

TEST(GGUFLLaMACausalLM, ReusesCapacityPrefillGraphAcrossPromptLengths)
{
	const auto archive = BuildTinyLLaMAArchive();
	const auto capacityPrefill = GGUF::LowerLLaMACausalLMPrefillCapacity(archive, 4);
	const auto fullOne = GGUF::LowerLLaMACausalLM(archive, 1);
	const auto fullTwo = GGUF::LowerLLaMACausalLM(archive, 2);
	const auto capacityPlan = Detail::BuildExecutablePlanFromGraph(capacityPrefill);
	Runtime::Interpreter<CPU> interpreter;

	std::array<Tensor<CPU>, 1> oneInputs = {
		MakeInt32Tensor({ 2, 0, 0, 0 }, { 4 }),
	};
	const auto oneOutputs = interpreter.RunForward(capacityPlan, oneInputs);
	ASSERT_EQ(oneOutputs.size(), 1u);
	const auto fullOneOutputs = interpreter.RunForward(Detail::BuildExecutablePlanFromGraph(fullOne),
	                                                   std::array{ MakeInt32Tensor({ 2 }, { 1 }) });
	const std::array<float, 3> capacityFirstLogit = {
		ReadFloat(oneOutputs[0], 0),
		ReadFloat(oneOutputs[0], 1),
		ReadFloat(oneOutputs[0], 2),
	};
	ExpectTensorNear(fullOneOutputs[0], capacityFirstLogit, GGUF::GetLLaMAParityTolerance(DataType::Float32));

	std::array<Tensor<CPU>, 1> twoInputs = {
		MakeInt32Tensor({ 0, 1, 0, 0 }, { 4 }),
	};
	const auto twoOutputs = interpreter.RunForward(capacityPlan, twoInputs);
	ASSERT_EQ(twoOutputs.size(), 1u);
	const auto fullTwoOutputs = interpreter.RunForward(Detail::BuildExecutablePlanFromGraph(fullTwo),
	                                                   std::array{ MakeInt32Tensor({ 0, 1 }, { 2 }) });
	const std::array<float, 3> fullSecondLogit = {
		ReadFloat(fullTwoOutputs[0], 3),
		ReadFloat(fullTwoOutputs[0], 4),
		ReadFloat(fullTwoOutputs[0], 5),
	};
	const std::array<float, 3> capacitySecondLogit = {
		ReadFloat(twoOutputs[0], 3),
		ReadFloat(twoOutputs[0], 4),
		ReadFloat(twoOutputs[0], 5),
	};
	ExpectValuesNear(capacitySecondLogit, fullSecondLogit, GGUF::GetLLaMAParityTolerance(DataType::Float32));
}

TEST(GGUFLLaMAArtifacts, CapacityDecodeScheduleRoundTripsPositionAndFullCacheBindings)
{
	const auto schedule = GGUF::BuildLLaMADecodeRuntimeSchedule(
	    BuildTinyQwen2Archive(),
	    { .prefillSequenceLength = 1, .decodePastLength = 0, .maxCacheLength = 4, .dynamicDecodePosition = true });
	ASSERT_EQ(schedule.states.size(), 2u);
	ASSERT_EQ(schedule.stateValueBindings.size(), 6u);
	const auto forward = schedule.module.plan.forward;
	EXPECT_EQ(schedule.module.functions[forward].inputs[1].dtype, DataType::Int64);
	EXPECT_EQ(schedule.module.functions[forward].inputs[2].StaticShape(), std::vector<std::size_t>({ 4, 1, 2 }));
	EXPECT_EQ(Runtime::RuntimeScheduleStateOutputIndices(schedule, forward), std::vector<std::size_t>({ 1, 2, 3 }));
	EXPECT_EQ(Runtime::RuntimeSchedulePublicOutputIndices(schedule, forward), std::vector<std::size_t>({ 0 }));
	const auto publicTypes = Runtime::RuntimeSchedulePublicOutputTypes(schedule, forward);
	ASSERT_EQ(publicTypes.size(), 1u);
	EXPECT_EQ(publicTypes[0].dtype, DataType::Float32);
	EXPECT_EQ(publicTypes[0].StaticShape(), std::vector<std::size_t>({ 1, 3 }));
	const auto stateAliases = Runtime::RuntimeScheduleStateOutputAliases(schedule, forward);
	ASSERT_EQ(stateAliases.size(), 3u);
	EXPECT_EQ(stateAliases[0].outputIndex, 1u);
	EXPECT_EQ(stateAliases[0].inputIndex, 1u);
	EXPECT_EQ(stateAliases[1].outputIndex, 2u);
	EXPECT_EQ(stateAliases[1].inputIndex, 2u);
	EXPECT_EQ(stateAliases[2].outputIndex, 3u);
	EXPECT_EQ(stateAliases[2].inputIndex, 3u);
	const auto projection = Runtime::RuntimeScheduleOutputProjectionForFunction(schedule, forward);
	EXPECT_EQ(projection.functionalOutputCount, 4u);
	EXPECT_EQ(projection.publicOutputIndices, std::vector<std::size_t>({ 0 }));
	ASSERT_EQ(projection.publicOutputTypes.size(), 1u);
	EXPECT_EQ(projection.publicOutputTypes[0].StaticShape(), std::vector<std::size_t>({ 1, 3 }));
	EXPECT_EQ(projection.stateAliases.size(), 3u);
	EXPECT_NO_THROW(Runtime::ValidateRuntimeSchedule(schedule));

	const auto path = MakeTempFixturePath("litenn_llama_capacity_decode", ".ltnn");
	Serialization::SaveVNextModelPackage(schedule, path);
	const auto loaded = Serialization::LoadVNextModelPackage(path);
	std::filesystem::remove(path);
	ASSERT_EQ(loaded.manifest.runtimeStates.size(), 2u);
	ASSERT_EQ(loaded.manifest.stateValueBindings.size(), 6u);
	EXPECT_EQ(loaded.manifest.stateValueBindings[4].stateName, "decode.position");
	EXPECT_EQ(loaded.manifest.stateValueBindings[5].kind, Runtime::RuntimeStateValueKind::FunctionOutput);
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
TEST(GGUFLLaMACausalLM, CompilesCapacityDecodeOnceAndMatchesInterpreterAtRuntimePosition)
{
	const auto lowered = GGUF::LowerLLaMACausalLMDecodeCapacity(BuildTinyLLaMAArchive(), 4);
	const auto plan = Detail::BuildExecutablePlanFromGraph(lowered);
	const auto tolerance = GGUF::GetLLaMAParityTolerance(DataType::Float32);
	const std::vector<float> zeroCache(8, 0.0f);
	std::array<Tensor<CPU>, 4> inputs = {
		MakeInt32Tensor({ 1 }, { 1 }),
		MakeInt64Tensor({ 0 }, { 1 }),
		MakeFloatTensor(zeroCache, { 4, 1, 2 }),
		MakeFloatTensor(zeroCache, { 4, 1, 2 }),
	};
	Runtime::Interpreter<CPU> interpreter;
	const auto expected = interpreter.RunForward(plan, inputs);
	auto compiled = Compiler<CPU>::CompileArtifact(plan).Load();
	const auto actual = compiled.RunTensors(inputs);
	ASSERT_EQ(actual.size(), expected.size());
	ExpectTensorNear(actual[0], expected[0], tolerance);
	EXPECT_EQ(static_cast<const std::int64_t*>(actual[1].UnsafeRawData())[0], 1);
	ExpectTensorNear(actual[2], expected[2], tolerance);
	ExpectTensorNear(actual[3], expected[3], tolerance);
}

TEST(GGUFLLaMACausalLM, CompilesStatefulDecodeScheduleWithPublicLogitsOnly)
{
	const auto lowered = GGUF::LowerLLaMACausalLMDecodeCapacity(BuildTinyLLaMAArchive(), 4);
	auto module = Detail::BuildExecutableModuleFromGraph(lowered);
	std::vector<Runtime::RuntimeStateBinding> states{
		{ .name = "past_key_0",
		  .kind = Runtime::RuntimeStateKind::KVCache,
		  .role = "key",
		  .type = TensorType::Dense(DataType::Float32, { 4, 1, 2 }) },
		{ .name = "past_value_0",
		  .kind = Runtime::RuntimeStateKind::KVCache,
		  .role = "value",
		  .type = TensorType::Dense(DataType::Float32, { 4, 1, 2 }) },
		{ .name = "decode.position",
		  .kind = Runtime::RuntimeStateKind::Generic,
		  .role = "position",
		  .type = TensorType::Dense(DataType::Int64, { 1 }) },
	};
	std::vector<Runtime::RuntimeStateValueBinding> bindings{
		{ "decode.position", module.plan.forward, Runtime::RuntimeStateValueKind::FunctionInput, 1, 0 },
		{ "decode.position", module.plan.forward, Runtime::RuntimeStateValueKind::FunctionOutput, 1, 0 },
		{ "past_key_0", module.plan.forward, Runtime::RuntimeStateValueKind::FunctionInput, 2, 0 },
		{ "past_key_0", module.plan.forward, Runtime::RuntimeStateValueKind::FunctionOutput, 2, 0 },
		{ "past_value_0", module.plan.forward, Runtime::RuntimeStateValueKind::FunctionInput, 3, 0 },
		{ "past_value_0", module.plan.forward, Runtime::RuntimeStateValueKind::FunctionOutput, 3, 0 },
	};
	auto schedule = Runtime::BuildRuntimeSchedule(std::move(module), std::move(states), std::move(bindings));
	const auto plan = schedule.module.plan;
	const auto tolerance = GGUF::GetLLaMAParityTolerance(DataType::Float32);
	const std::vector<float> zeroCache(8, 0.0f);
	std::array<Tensor<CPU>, 4> interpreterInputs = {
		MakeInt32Tensor({ 1 }, { 1 }),
		MakeInt64Tensor({ 0 }, { 1 }),
		MakeFloatTensor(zeroCache, { 4, 1, 2 }),
		MakeFloatTensor(zeroCache, { 4, 1, 2 }),
	};
	Runtime::Interpreter<CPU> interpreter;
	const auto expected = interpreter.RunForward(plan, interpreterInputs);
	ASSERT_EQ(expected.size(), 4u);

	auto artifact = Compiler<CPU>::CompileArtifact(schedule);
	ASSERT_EQ(artifact.InputSpecs().size(), 4u);
	ASSERT_EQ(artifact.OutputSpecs().size(), 1u);
	EXPECT_EQ(artifact.OutputSpecs()[0].name, "logits");
	auto compiled = artifact.Load();
	ASSERT_EQ(compiled.OutputSpecs().size(), 1u);

	std::array<Tensor<CPU>, 4> inputs = {
		MakeInt32Tensor({ 1 }, { 1 }),
		MakeInt64Tensor({ 0 }, { 1 }),
		MakeFloatTensor(zeroCache, { 4, 1, 2 }),
		MakeFloatTensor(zeroCache, { 4, 1, 2 }),
	};
	const auto actual = compiled.RunTensors(inputs);
	ASSERT_EQ(actual.size(), 1u);
	ExpectTensorNear(actual[0], expected[0], tolerance);
	EXPECT_EQ(static_cast<const std::int64_t*>(inputs[1].UnsafeRawData())[0], 1);
	ExpectTensorNear(inputs[2], expected[2], tolerance);
	ExpectTensorNear(inputs[3], expected[3], tolerance);
}

TEST(GGUFLLaMACausalLM, CompilesBuilderStatefulDecodeScheduleWithPublicLogitsOnly)
{
	auto schedule = GGUF::BuildLLaMADecodeRuntimeSchedule(
	    BuildTinyLLaMAArchive(),
	    { .prefillSequenceLength = 1, .decodePastLength = 0, .maxCacheLength = 4, .dynamicDecodePosition = true });
	ASSERT_FALSE(schedule.module.plan.variables.empty());
	for (const auto& variable : schedule.module.plan.variables)
	{
		EXPECT_EQ(variable.region.ownership, BufferOwnership::Borrowed);
		EXPECT_NE(variable.region.owner, nullptr);
		EXPECT_NE(variable.region.data, nullptr);
	}
	const auto plan = schedule.module.plan;
	const auto tolerance = GGUF::GetLLaMAParityTolerance(DataType::Float32);
	const std::vector<float> zeroCache(8, 0.0f);
	std::array<Tensor<CPU>, 4> interpreterInputs = {
		MakeInt32Tensor({ 1 }, { 1 }),
		MakeInt64Tensor({ 0 }, { 1 }),
		MakeFloatTensor(zeroCache, { 4, 1, 2 }),
		MakeFloatTensor(zeroCache, { 4, 1, 2 }),
	};
	Runtime::Interpreter<CPU> interpreter;
	const auto expected = interpreter.RunForward(plan, interpreterInputs);
	ASSERT_EQ(expected.size(), 4u);

	auto artifact = Compiler<CPU>::CompileArtifact(schedule);
	ASSERT_EQ(artifact.InputSpecs().size(), 4u);
	ASSERT_EQ(artifact.OutputSpecs().size(), 1u);
	EXPECT_EQ(artifact.OutputSpecs()[0].name, "logits");
	auto compiled = artifact.Load();
	ASSERT_EQ(compiled.OutputSpecs().size(), 1u);

	std::array<Tensor<CPU>, 4> inputs = {
		MakeInt32Tensor({ 1 }, { 1 }),
		MakeInt64Tensor({ 0 }, { 1 }),
		MakeFloatTensor(zeroCache, { 4, 1, 2 }),
		MakeFloatTensor(zeroCache, { 4, 1, 2 }),
	};
	const auto actual = compiled.RunTensors(inputs);
	ASSERT_EQ(actual.size(), 1u);
	ExpectTensorNear(actual[0], expected[0], tolerance);
	EXPECT_EQ(static_cast<const std::int64_t*>(inputs[1].UnsafeRawData())[0], 1);
	ExpectTensorNear(inputs[2], expected[2], tolerance);
	ExpectTensorNear(inputs[3], expected[3], tolerance);
}

TEST(GGUFLLaMACausalLM, KVScatterUpdateHelperSupportsInPlaceAppend)
{
	constexpr std::int64_t capacity = 4;
	constexpr std::int64_t kvHeads = 2;
	constexpr std::int64_t headDim = 3;
	constexpr std::int64_t rowStride = kvHeads * headDim;
	std::vector<float> initial(static_cast<std::size_t>(capacity * rowStride));
	for (std::size_t i = 0; i < initial.size(); ++i)
	{
		initial[i] = static_cast<float>(i + 1);
	}
	const std::array<std::int64_t, 1> indices{ 2 };
	const std::array<float, 6> updates{ 100.0F, 101.0F, 102.0F, 103.0F, 104.0F, 105.0F };

	auto expected = initial;
	std::copy(updates.begin(), updates.end(), expected.begin() + static_cast<std::ptrdiff_t>(indices[0] * rowStride));

	auto inPlace = initial;
	litenn_cpu_scatter_update_axis0_f32_rank3(
	    nullptr, inPlace.data(), 0, capacity, kvHeads, headDim, rowStride, headDim, 1, nullptr, indices.data(), 0,
	    static_cast<std::int64_t>(indices.size()), 1, nullptr, updates.data(), 0, 1, kvHeads, headDim, rowStride,
	    headDim, 1, nullptr, inPlace.data(), 0, capacity, kvHeads, headDim, rowStride, headDim, 1);
	EXPECT_EQ(inPlace, expected);

	std::vector<float> copied(initial.size(), -1.0F);
	litenn_cpu_scatter_update_axis0_f32_rank3(
	    nullptr, initial.data(), 0, capacity, kvHeads, headDim, rowStride, headDim, 1, nullptr, indices.data(), 0,
	    static_cast<std::int64_t>(indices.size()), 1, nullptr, updates.data(), 0, 1, kvHeads, headDim, rowStride,
	    headDim, 1, nullptr, copied.data(), 0, capacity, kvHeads, headDim, rowStride, headDim, 1);
	EXPECT_EQ(copied, expected);
}

TEST(GGUFLLaMACausalLM, CompilesCapacityPrefillOnceAndExposesMaxCapacityLogits)
{
	const auto lowered = GGUF::LowerLLaMACausalLMPrefillCapacity(BuildTinyLLaMAArchive(), 4);
	const auto plan = Detail::BuildExecutablePlanFromGraph(lowered);
	auto artifact = Compiler<CPU>::CompileArtifact(plan);
	ASSERT_EQ(artifact.InputSpecs().size(), 1u);
	ASSERT_EQ(artifact.OutputSpecs().size(), 1u);
	EXPECT_EQ(artifact.InputSpecs()[0].name, "token_ids");
	EXPECT_EQ(artifact.OutputSpecs()[0].name, "logits");
	EXPECT_EQ(artifact.OutputSpecs()[0].type.StaticShape(), std::vector<std::size_t>({ 4, 3 }));
	auto compiled = artifact.Load();
	ASSERT_EQ(compiled.InputSpecs().size(), 1u);
	ASSERT_EQ(compiled.OutputSpecs().size(), 1u);

	Runtime::Interpreter<CPU> interpreter;
	std::array<Tensor<CPU>, 1> inputs = { MakeInt32Tensor({ 0, 1, 0, 0 }, { 4 }) };
	const auto expected = interpreter.RunForward(plan, inputs);
	const auto actual = compiled.RunTensors(inputs);
	ASSERT_EQ(actual.size(), expected.size());
	ExpectTensorNear(actual[0], expected[0], GGUF::GetLLaMAParityTolerance(DataType::Float32));
}

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

TEST(GGUFLLaMACausalLM, PreservesQuantizedProjectionStorageWithQuantizedMatMulNodes)
{
	const auto quantizedArchive = QuantizeQ80Weights(BuildQuantizedFriendlyLLaMAArchive());
	const auto lowered = GGUF::LowerLLaMACausalLM(quantizedArchive, 2, 0, { .preserveQuantizedWeights = true });

	std::size_t quantizedVariableCount = 0;
	for (std::size_t i = 0; i < lowered.VariableCount(); ++i)
	{
		if (lowered.GetVariable(i)->IsQuantized())
		{
			++quantizedVariableCount;
			EXPECT_EQ(lowered.GetVariable(i)->Quantization()->blockFormat, QuantizedBlockFormat::GGML_Q8_0);
		}
	}
	EXPECT_EQ(quantizedVariableCount, 7u);

	const auto& forward = lowered.GetSubgraph(lowered.Forward());
	std::size_t quantizedMatMulNodeCount = 0;
	for (NodeId nodeId = 0; nodeId < forward.NodeCount(); ++nodeId)
	{
		const auto* quantizedMatMul = std::get_if<QuantizedMatMulNode>(&forward.GetNodeEntry(nodeId).node);
		if (!quantizedMatMul)
		{
			continue;
		}
		++quantizedMatMulNodeCount;
		EXPECT_EQ(quantizedMatMul->params.blockFormat, QuantizedBlockFormat::GGML_Q8_0);
	}
	EXPECT_EQ(quantizedMatMulNodeCount, 7u);

	const auto plan = Detail::BuildExecutablePlanFromGraph(lowered);
	EXPECT_NO_THROW(ValidateExecutablePlan(plan));
}

TEST(GGUFLLaMACausalLM, ImportsQwenProjectionBiasesWithVectorShapes)
{
	auto archive = BuildTinyQwen2Archive();
	AddNamedVariable(archive, "blk.0.attn_q.bias", Tensor<CPU>({ 0.1F, 0.2F, 0.3F, 0.4F }, { 4 }));
	AddNamedVariable(archive, "blk.0.attn_k.bias", Tensor<CPU>({ 0.5F, 0.6F }, { 2 }));
	AddNamedVariable(archive, "blk.0.attn_v.bias", Tensor<CPU>({ 0.7F, 0.8F }, { 2 }));

	Graph lowered;
	const auto hyperparameters = GGUF::ParseLLaMAHyperparameters(archive);
	const auto model = GGUF::CreateLLaMACausalLM(lowered, archive, hyperparameters);
	ASSERT_EQ(model.blocks.size(), 1u);
	ASSERT_TRUE(model.blocks[0].queryProjection.biasVariable.has_value());
	ASSERT_TRUE(model.blocks[0].keyProjection.biasVariable.has_value());
	ASSERT_TRUE(model.blocks[0].valueProjection.biasVariable.has_value());
	EXPECT_EQ(model.blocks[0].queryProjection.biasShape, (std::vector<std::size_t>{ 1, 4 }));
	EXPECT_EQ(model.blocks[0].keyProjection.biasShape, (std::vector<std::size_t>{ 1, 2 }));
	EXPECT_EQ(model.blocks[0].valueProjection.biasShape, (std::vector<std::size_t>{ 1, 2 }));

	const auto forward = GGUF::BuildLLaMACausalLM(lowered, model, hyperparameters, 1);
	lowered.SetForward(forward);
	EXPECT_NO_THROW(ValidateExecutablePlan(Detail::BuildExecutablePlanFromGraph(lowered)));
}
