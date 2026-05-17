#include <GGUFImporter.h>
#include <LLaMABuilder.h>

#include <LiteNN/Runtime/Interpreter.h>
#include <LiteNN/Serialization/ModelIO.h>

#include <array>
#include <filesystem>
#include <format>
#include <iostream>
#include <memory>
#include <span>
#include <stdexcept>
#include <string_view>
#include <vector>

#include <ggml.h>
#include <gguf.h>

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

	GGMLContextPtr CreateTensorContext()
	{
		ggml_init_params params{};
		params.mem_size = ggml_tensor_overhead() * 32;
		params.no_alloc = true;
		return GGMLContextPtr{ ggml_init(params) };
	}

	void AddTensor(gguf_context* gguf, ggml_context* ggml, ggml_type type, std::string_view name,
	              std::span<const std::int64_t> dims, const void* data)
	{
		auto* tensor = ggml_new_tensor(ggml, type, static_cast<int>(dims.size()), dims.data());
		if (!tensor)
		{
			throw std::runtime_error(std::format("failed to allocate GGML tensor '{}'", name));
		}
		ggml_set_name(tensor, std::string(name).c_str());
		gguf_add_tensor(gguf, tensor);
		gguf_set_tensor_data(gguf, tensor->name, data);
	}

	std::vector<float> Zeros(std::size_t count)
	{
		return std::vector<float>(count, 0.0f);
	}

	std::filesystem::path WriteTinyLLaMAGGUF(const std::filesystem::path& path)
	{
		std::filesystem::remove(path);

		GGUFContextPtr gguf{ gguf_init_empty() };
		auto ggml = CreateTensorContext();
		if (!gguf || !ggml)
		{
			throw std::runtime_error("failed to initialize GGUF fixture contexts");
		}

		constexpr std::int64_t kEmbedding = 4;
		constexpr std::int64_t kVocab = 3;
		constexpr std::int64_t kHeadCount = 2;
		constexpr std::int64_t kKVHeadCount = 1;
		constexpr std::int64_t kHeadDim = kEmbedding / kHeadCount;
		constexpr std::int64_t kKVWidth = kKVHeadCount * kHeadDim;
		constexpr std::int64_t kFeedForward = 8;

		gguf_set_val_str(gguf.get(), "general.architecture", "llama");
		gguf_set_val_u32(gguf.get(), "llama.context_length", 8);
		gguf_set_val_u32(gguf.get(), "llama.embedding_length", kEmbedding);
		gguf_set_val_u32(gguf.get(), "llama.block_count", 1);
		gguf_set_val_u32(gguf.get(), "llama.feed_forward_length", kFeedForward);
		gguf_set_val_u32(gguf.get(), "llama.attention.head_count", kHeadCount);
		gguf_set_val_u32(gguf.get(), "llama.attention.head_count_kv", kKVHeadCount);
		gguf_set_val_f32(gguf.get(), "llama.attention.layer_norm_rms_epsilon", 1.0e-6F);
		gguf_set_val_f32(gguf.get(), "llama.rope.freq_base", 10000.0F);

		const char* tokens[] = { "<s>", "hello", "world" };
		gguf_set_arr_str(gguf.get(), "tokenizer.ggml.tokens", tokens, 3);

		const std::array<float, kEmbedding * kVocab> embedding = {
			1.0F, 0.0F, 0.0F, 0.0F,
			0.0F, 1.0F, 0.0F, 0.0F,
			0.0F, 0.0F, 1.0F, 0.0F,
		};
		AddTensor(gguf.get(), ggml.get(), GGML_TYPE_F32, "token_embd.weight",
		          std::array<std::int64_t, 2>{ kEmbedding, kVocab }, embedding.data());
		AddTensor(gguf.get(), ggml.get(), GGML_TYPE_F32, "output.weight",
		          std::array<std::int64_t, 2>{ kEmbedding, kVocab }, embedding.data());

		const std::array<float, kEmbedding> ones = { 1.0F, 1.0F, 1.0F, 1.0F };
		AddTensor(gguf.get(), ggml.get(), GGML_TYPE_F32, "output_norm.weight",
		          std::array<std::int64_t, 1>{ kEmbedding }, ones.data());
		AddTensor(gguf.get(), ggml.get(), GGML_TYPE_F32, "blk.0.attn_norm.weight",
		          std::array<std::int64_t, 1>{ kEmbedding }, ones.data());
		AddTensor(gguf.get(), ggml.get(), GGML_TYPE_F32, "blk.0.ffn_norm.weight",
		          std::array<std::int64_t, 1>{ kEmbedding }, ones.data());

		const auto q = Zeros(kEmbedding * kEmbedding);
		const auto kv = Zeros(kEmbedding * kKVWidth);
		const auto ffnUp = Zeros(kEmbedding * kFeedForward);
		const auto ffnDown = Zeros(kFeedForward * kEmbedding);
		AddTensor(gguf.get(), ggml.get(), GGML_TYPE_F32, "blk.0.attn_q.weight",
		          std::array<std::int64_t, 2>{ kEmbedding, kEmbedding }, q.data());
		AddTensor(gguf.get(), ggml.get(), GGML_TYPE_F32, "blk.0.attn_k.weight",
		          std::array<std::int64_t, 2>{ kEmbedding, kKVWidth }, kv.data());
		AddTensor(gguf.get(), ggml.get(), GGML_TYPE_F32, "blk.0.attn_v.weight",
		          std::array<std::int64_t, 2>{ kEmbedding, kKVWidth }, kv.data());
		AddTensor(gguf.get(), ggml.get(), GGML_TYPE_F32, "blk.0.attn_output.weight",
		          std::array<std::int64_t, 2>{ kEmbedding, kEmbedding }, q.data());
		AddTensor(gguf.get(), ggml.get(), GGML_TYPE_F32, "blk.0.ffn_gate.weight",
		          std::array<std::int64_t, 2>{ kEmbedding, kFeedForward }, ffnUp.data());
		AddTensor(gguf.get(), ggml.get(), GGML_TYPE_F32, "blk.0.ffn_up.weight",
		          std::array<std::int64_t, 2>{ kEmbedding, kFeedForward }, ffnUp.data());
		AddTensor(gguf.get(), ggml.get(), GGML_TYPE_F32, "blk.0.ffn_down.weight",
		          std::array<std::int64_t, 2>{ kFeedForward, kEmbedding }, ffnDown.data());

		if (!gguf_write_to_file(gguf.get(), path.string().c_str(), false))
		{
			throw std::runtime_error("failed to write tiny GGUF fixture");
		}
		return path;
	}
} // namespace

int main(int argc, char** argv)
{
	try
	{
		const auto outputDir = argc >= 2 ? std::filesystem::path(argv[1]) : std::filesystem::current_path();
		std::filesystem::create_directories(outputDir);

		const auto ggufPath = WriteTinyLLaMAGGUF(outputDir / "tiny_llama.gguf");
		const auto archivePath = outputDir / "tiny_llama.archive.ltnn";
		const auto loweredPath = outputDir / "tiny_llama.prefill.ltnn";
		const auto decodePath = outputDir / "tiny_llama.decode.ltnn";

		const auto imported = LiteNN::GGUF::ImportGGUFArchive(ggufPath);
		LiteNN::Serialization::SaveModel(imported.graph, archivePath);

		auto lowered = LiteNN::GGUF::LowerLLaMACausalLM(imported.graph, 2);
		LiteNN::Serialization::SaveModel(lowered, loweredPath);
		auto decode = LiteNN::GGUF::LowerLLaMACausalLMDecode(imported.graph, 1, 1, 1);
		LiteNN::Serialization::SaveModel(decode, decodePath);

		LiteNN::Runtime::Interpreter<LiteNN::CPU> interpreter;
		LiteNN::CPU cpu;
		LiteNN::Tensor<LiteNN::CPU> tokenIds(LiteNN::Uninitialized, { 2 }, LiteNN::DataType::Int32, cpu);
		const std::array<std::int32_t, 2> tokenIdValues = { 0, 1 };
		LiteNN::DeviceTraits<LiteNN::CPU>::CopyFromCPU(cpu, LiteNN::DataType::Int32, tokenIds.RawData(),
		                                               LiteNN::DataType::Int32, tokenIdValues.data(),
		                                               tokenIdValues.size());
		std::array<LiteNN::Tensor<LiteNN::CPU>, 1> inputs = { std::move(tokenIds) };
		const auto outputs = interpreter.RunForward(lowered, inputs);

		LiteNN::Tensor<LiteNN::CPU> decodeTokenIds(LiteNN::Uninitialized, { 1 }, LiteNN::DataType::Int32, cpu);
		const std::array<std::int32_t, 1> decodeTokenIdValues = { 2 };
		LiteNN::DeviceTraits<LiteNN::CPU>::CopyFromCPU(cpu, LiteNN::DataType::Int32, decodeTokenIds.RawData(),
		                                               LiteNN::DataType::Int32, decodeTokenIdValues.data(),
		                                               decodeTokenIdValues.size());
		LiteNN::Tensor<LiteNN::CPU> pastKeys({ 0.0F, 0.0F }, { 1, 1, 2 });
		LiteNN::Tensor<LiteNN::CPU> pastValues({ 0.0F, 0.0F }, { 1, 1, 2 });
		std::array<LiteNN::Tensor<LiteNN::CPU>, 3> decodeInputs = {
			std::move(decodeTokenIds),
			std::move(pastKeys),
			std::move(pastValues),
		};
		const auto decodeOutputs = interpreter.RunForward(decode, decodeInputs);

		std::cout << "Imported " << imported.summary.tensorCount << " tensors and "
		          << imported.summary.metadataCount << " metadata entries\n";
		std::cout << "Archive: " << archivePath.string() << '\n';
		std::cout << "Lowered prefill graph: " << loweredPath.string() << '\n';
		std::cout << "Lowered decode graph: " << decodePath.string() << '\n';
		std::cout << "Logits shape: [" << outputs[0].Shape()[0] << ", " << outputs[0].Shape()[1] << "]\n";
		std::cout << "Decode updated key shape: [" << decodeOutputs[1].Shape()[0] << ", "
		          << decodeOutputs[1].Shape()[1] << ", " << decodeOutputs[1].Shape()[2] << "]\n";
		return 0;
	}
	catch (const std::exception& ex)
	{
		std::cerr << "litenn_gguf_conversion_example: " << ex.what() << '\n';
		return 1;
	}
}
