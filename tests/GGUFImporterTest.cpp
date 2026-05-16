#include <gtest/gtest.h>

#include <GGUFImporter.h>
#include <LLaMABuilder.h>

#include <LiteNN/Serialization/ModelIO.h>
#include <LiteNN/Runtime/Interpreter.h>

#include <array>
#include <filesystem>
#include <memory>
#include <span>
#include <stdexcept>
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
		return static_cast<const float*>(cpuTensor.RawData())[index];
	}

	float ReadFloat(const Tensor<PolymorphicDevice>& tensor, std::size_t index)
	{
		return ReadFloat(tensor.CopyToDevice(CPU{}), index);
	}

	GGMLContextPtr CreateTensorContext()
	{
		ggml_init_params params{};
		params.mem_size = ggml_tensor_overhead() * 8;
		params.no_alloc = true;
		return GGMLContextPtr{ ggml_init(params) };
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
		const auto path = std::filesystem::path("litenn_gguf_importer_fixture.gguf");
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

		const char* tokens[] = { "<s>", "hello" };
		gguf_set_arr_str(gguf.get(), "tokenizer.ggml.tokens", tokens, 2);

		const std::array<std::int32_t, 2> tokenTypes = { 1, 3 };
		gguf_set_arr_data(gguf.get(), "tokenizer.ggml.token_type", GGUF_TYPE_INT32, tokenTypes.data(),
		                  tokenTypes.size());

		const std::array<float, 4> embedding = { 1.0F, 2.0F, 3.0F, 4.0F };
		const std::array<std::int64_t, 2> embeddingShape = { 2, 2 };
		AddTensor(gguf.get(), ggml.get(), GGML_TYPE_F32, "token_embd.weight", embeddingShape, embedding.data());

		const std::array<std::uint8_t, 18> q4Payload = {
			0x10, 0x00, 0x22, 0x44, 0x66, 0x88, 0xaa, 0xcc, 0xee,
			0x11, 0x33, 0x55, 0x77, 0x99, 0xbb, 0xdd, 0xff, 0x7f,
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
		const auto path = std::filesystem::path("litenn_gguf_importer_unsupported_fixture.gguf");
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

		AddNamedVariable(graph, "token_embd.weight", Tensor<CPU>({ 1.0f, 0.0f, 0.0f,
		                                                          0.0f, 1.0f, 0.0f,
		                                                          0.0f, 0.0f, 1.0f,
		                                                          0.0f, 0.0f, 0.0f },
		                                                         { 4, 3 }));
		AddNamedVariable(graph, "output_norm.weight", Tensor<CPU>({ 1.0f, 1.0f, 1.0f, 1.0f }, { 1, 4 }));
		AddNamedVariable(graph, "output.weight", Tensor<CPU>({ 1.0f, 0.0f, 0.0f,
		                                                      0.0f, 1.0f, 0.0f,
		                                                      0.0f, 0.0f, 1.0f,
		                                                      0.0f, 0.0f, 0.0f },
		                                                     { 4, 3 }));

		AddNamedVariable(graph, "blk.0.attn_norm.weight", Tensor<CPU>({ 1.0f, 1.0f, 1.0f, 1.0f }, { 1, 4 }));
		AddNamedVariable(graph, "blk.0.ffn_norm.weight", Tensor<CPU>({ 1.0f, 1.0f, 1.0f, 1.0f }, { 1, 4 }));
		AddNamedVariable(graph, "blk.0.attn_q.weight", Tensor<CPU>(
		                                                { 0.0f, 0.0f, 0.0f, 0.0f,
		                                                  0.0f, 0.0f, 0.0f, 0.0f,
		                                                  0.0f, 0.0f, 0.0f, 0.0f,
		                                                  0.0f, 0.0f, 0.0f, 0.0f },
		                                                { 4, 4 }));
		AddNamedVariable(graph, "blk.0.attn_k.weight", Tensor<CPU>(
		                                                { 0.0f, 0.0f,
		                                                  0.0f, 0.0f,
		                                                  0.0f, 0.0f,
		                                                  0.0f, 0.0f },
		                                                { 4, 2 }));
		AddNamedVariable(graph, "blk.0.attn_v.weight", Tensor<CPU>(
		                                                { 0.0f, 0.0f,
		                                                  0.0f, 0.0f,
		                                                  0.0f, 0.0f,
		                                                  0.0f, 0.0f },
		                                                { 4, 2 }));
		AddNamedVariable(graph, "blk.0.attn_output.weight", Tensor<CPU>(
		                                                     { 0.0f, 0.0f, 0.0f, 0.0f,
		                                                       0.0f, 0.0f, 0.0f, 0.0f,
		                                                       0.0f, 0.0f, 0.0f, 0.0f,
		                                                       0.0f, 0.0f, 0.0f, 0.0f },
		                                                     { 4, 4 }));
		AddNamedVariable(graph, "blk.0.ffn_gate.weight", Tensor<CPU>(
		                                                  { 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f,
		                                                    0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f,
		                                                    0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f,
		                                                    0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f },
		                                                  { 4, 8 }));
		AddNamedVariable(graph, "blk.0.ffn_up.weight", Tensor<CPU>(
		                                                { 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f,
		                                                  0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f,
		                                                  0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f,
		                                                  0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f },
		                                                { 4, 8 }));
		AddNamedVariable(graph, "blk.0.ffn_down.weight", Tensor<CPU>(
		                                                  { 0.0f, 0.0f, 0.0f, 0.0f,
		                                                    0.0f, 0.0f, 0.0f, 0.0f,
		                                                    0.0f, 0.0f, 0.0f, 0.0f,
		                                                    0.0f, 0.0f, 0.0f, 0.0f,
		                                                    0.0f, 0.0f, 0.0f, 0.0f,
		                                                    0.0f, 0.0f, 0.0f, 0.0f,
		                                                    0.0f, 0.0f, 0.0f, 0.0f,
		                                                    0.0f, 0.0f, 0.0f, 0.0f },
		                                                  { 8, 4 }));
		return graph;
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
} // namespace

TEST(GGUFImporter, ImportsMetadataTensorNamesAndQuantizedPayloads)
{
	const auto path = WriteSupportedFixture();
	auto imported = GGUF::ImportGGUFArchive(path);
	std::filesystem::remove(path);

	EXPECT_EQ(imported.summary.tensorCount, 2u);
	EXPECT_EQ(imported.summary.metadataCount, 5u);

	ASSERT_EQ(imported.graph.VariableCount(), 2);
	ASSERT_EQ(imported.graph.VariableNames().size(), 2);
	EXPECT_EQ(imported.graph.VariableName(0), "token_embd.weight");
	EXPECT_EQ(imported.graph.VariableName(1), "blk.0.attn_q.weight");
	EXPECT_TRUE(imported.graph.InputSignature().empty());
	EXPECT_TRUE(imported.graph.OutputSignature().empty());

	const auto* architecture = imported.graph.FindMetadata("general.architecture");
	ASSERT_NE(architecture, nullptr);
	EXPECT_EQ(std::get<std::string>(architecture->value), "llama");

	const auto* contextLength = imported.graph.FindMetadata("llama.context_length");
	ASSERT_NE(contextLength, nullptr);
	EXPECT_EQ(std::get<std::uint64_t>(contextLength->value), 4096u);

	const auto* ropeBase = imported.graph.FindMetadata("llama.rope.freq_base");
	ASSERT_NE(ropeBase, nullptr);
	EXPECT_DOUBLE_EQ(std::get<double>(ropeBase->value), 500000.0);

	const auto* tokens = imported.graph.FindMetadata("tokenizer.ggml.tokens");
	ASSERT_NE(tokens, nullptr);
	const auto& tokenList = std::get<std::vector<std::string>>(tokens->value);
	ASSERT_EQ(tokenList.size(), 2);
	EXPECT_EQ(tokenList[0], "<s>");
	EXPECT_EQ(tokenList[1], "hello");

	const auto* tokenTypes = imported.graph.FindMetadata("tokenizer.ggml.token_type");
	ASSERT_NE(tokenTypes, nullptr);
	const auto& tokenTypeList = std::get<std::vector<std::int64_t>>(tokenTypes->value);
	ASSERT_EQ(tokenTypeList.size(), 2);
	EXPECT_EQ(tokenTypeList[0], 1);
	EXPECT_EQ(tokenTypeList[1], 3);

	EXPECT_FLOAT_EQ(ReadFloat(imported.graph.GetVariable(0)->Data(), 0), 1.0F);
	EXPECT_FLOAT_EQ(ReadFloat(imported.graph.GetVariable(0)->Data(), 1), 2.0F);
	EXPECT_FLOAT_EQ(ReadFloat(imported.graph.GetVariable(0)->Data(), 2), 3.0F);
	EXPECT_FLOAT_EQ(ReadFloat(imported.graph.GetVariable(0)->Data(), 3), 4.0F);

	const auto& quantized = *imported.graph.GetVariable(1);
	ASSERT_TRUE(quantized.IsQuantized());
	EXPECT_EQ(quantized.Data().DType(), DataType::UInt8);
	EXPECT_EQ(quantized.Data().NumElements(), 18u);
	const auto& params = *quantized.Quantization();
	EXPECT_EQ(params.scheme, QuantizationScheme::Block);
	EXPECT_EQ(params.blockFormat, QuantizedBlockFormat::GGML_Q4_0);
	EXPECT_EQ(params.expressedType, DataType::Float32);
	EXPECT_EQ(params.expressedShape, std::vector<std::size_t>({ 32 }));

	const auto quantizedBytes = quantized.Data().CopyToDevice(CPU{});
	const auto* rawBytes = static_cast<const std::uint8_t*>(quantizedBytes.RawData());
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
	const auto outputPath = std::filesystem::path("litenn_gguf_imported_archive.ltnn");
	std::filesystem::remove(outputPath);

	const auto summary = GGUF::ConvertGGUFArchive(inputPath, outputPath);
	auto loaded = Serialization::LoadModel(outputPath);

	std::filesystem::remove(inputPath);
	std::filesystem::remove(outputPath);

	EXPECT_EQ(summary.tensorCount, 2u);
	EXPECT_EQ(summary.metadataCount, 5u);
	ASSERT_EQ(loaded.VariableCount(), 2);
	EXPECT_EQ(loaded.VariableName(0), "token_embd.weight");
	EXPECT_EQ(loaded.VariableName(1), "blk.0.attn_q.weight");

	const auto* architecture = loaded.FindMetadata("general.architecture");
	ASSERT_NE(architecture, nullptr);
	EXPECT_EQ(std::get<std::string>(architecture->value), "llama");

	const auto* tokens = loaded.FindMetadata("tokenizer.ggml.tokens");
	ASSERT_NE(tokens, nullptr);
	const auto& tokenList = std::get<std::vector<std::string>>(tokens->value);
	ASSERT_EQ(tokenList.size(), 2);
	EXPECT_EQ(tokenList[0], "<s>");
	EXPECT_EQ(tokenList[1], "hello");

	EXPECT_FLOAT_EQ(ReadFloat(loaded.GetVariable(0)->Data(), 0), 1.0F);
	EXPECT_TRUE(loaded.GetVariable(1)->IsQuantized());
	EXPECT_EQ(loaded.GetVariable(1)->Quantization()->blockFormat, QuantizedBlockFormat::GGML_Q4_0);
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
	EXPECT_EQ(hyperparameters.HeadDimension(), 16u);
	EXPECT_EQ(hyperparameters.QueryGroupsPerKVHead(), 1u);
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
	});

	const auto hyperparameters = GGUF::ParseLLaMAHyperparameters(graph);
	EXPECT_EQ(hyperparameters.attentionHeadCountKV, 2u);
	EXPECT_DOUBLE_EQ(hyperparameters.ropeFrequencyBase, 500000.0);
	EXPECT_EQ(hyperparameters.HeadDimension(), 32u);
	EXPECT_EQ(hyperparameters.QueryGroupsPerKVHead(), 4u);
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
	std::array<Tensor<CPU>, 1> inputs = { Tensor<CPU>({ 1.0f, 2.0f, 3.0f, 4.0f,
	                                                   5.0f, 6.0f, 7.0f, 8.0f },
	                                                  { 2, 4 }) };
	const auto outputs = interpreter.RunForward(graph, inputs);
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

TEST(GGUFLLaMACausalLM, LowersFullGraphAndRunsCPUForwardOnTokenPlanes)
{
	const auto archive = BuildTinyLLaMAArchive();
	const auto lowered = GGUF::LowerLLaMACausalLM(archive, 2);

	EXPECT_EQ(lowered.InputName(0), "token_plane");
	EXPECT_EQ(lowered.OutputName(0), "logits");

	Runtime::Interpreter<CPU> interpreter;
	std::array<Tensor<CPU>, 1> inputs = { Tensor<CPU>({ 1.0f, 0.0f, 0.0f,
	                                                   0.0f, 1.0f, 0.0f },
	                                                  { 2, 3 }) };
	const auto outputs = interpreter.RunForward(lowered, inputs);
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

TEST(GGUFLLaMACausalLM, FallsBackToTokenEmbeddingWhenOutputWeightIsMissing)
{
	const auto archive = BuildTinyLLaMAArchive();
	const auto tiedArchive = CopyArchiveExcludingVariables(archive, { "output.weight" });
	const auto lowered = GGUF::LowerLLaMACausalLM(tiedArchive, 1);

	Runtime::Interpreter<CPU> interpreter;
	std::array<Tensor<CPU>, 1> inputs = { Tensor<CPU>({ 0.0f, 0.0f, 1.0f }, { 1, 3 }) };
	const auto outputs = interpreter.RunForward(lowered, inputs);
	ASSERT_EQ(outputs.size(), 1u);
	ASSERT_EQ(outputs[0].Shape().NumElements(), 3u);

	const float expected = 1.0f / std::sqrt(0.25f + 1.0e-6f);
	EXPECT_NEAR(ReadFloat(outputs[0], 0), 0.0f, 1e-5f);
	EXPECT_NEAR(ReadFloat(outputs[0], 1), 0.0f, 1e-5f);
	EXPECT_NEAR(ReadFloat(outputs[0], 2), expected, 1e-4f);
}