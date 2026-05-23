#include <gtest/gtest.h>

#include <LiteNN.h>
#include <LiteNN/Runtime/Interpreter.h>
#include <LiteNN/Serialization/TorchManifest.h>

#ifdef LITENN_ENABLE_MLIR
#include <LiteNN/Compiler/CompiledModule.h>
#endif

#include <algorithm>
#include <array>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <span>
#include <stdexcept>
#include <string>
#include <string_view>
#include <utility>
#include <vector>

using namespace LiteNN;

namespace
{
	void AppendU64LE(std::vector<std::byte>& out, std::uint64_t value)
	{
		for (int i = 0; i < 8; ++i)
		{
			out.push_back(static_cast<std::byte>((value >> (8 * i)) & 0xffU));
		}
	}

	template <typename T>
	void AppendValue(std::vector<std::byte>& out, const T& value)
	{
		const auto oldSize = out.size();
		out.resize(oldSize + sizeof(T));
		std::memcpy(out.data() + oldSize, &value, sizeof(T));
	}

	std::vector<std::byte> BuildSafetensors(std::string header, std::vector<std::byte> payload)
	{
		std::vector<std::byte> bytes;
		AppendU64LE(bytes, header.size());
		for (const auto c : header)
		{
			bytes.push_back(static_cast<std::byte>(static_cast<unsigned char>(c)));
		}
		bytes.insert(bytes.end(), payload.begin(), payload.end());
		return bytes;
	}

	struct FloatTensorSpec
	{
		std::string name;
		std::vector<std::size_t> shape;
		std::vector<float> values;
	};

	std::string ShapeJson(std::span<const std::size_t> shape)
	{
		std::string result = "[";
		for (std::size_t i = 0; i < shape.size(); ++i)
		{
			if (i != 0)
			{
				result += ",";
			}
			result += std::to_string(shape[i]);
		}
		result += "]";
		return result;
	}

	Serialization::SafetensorsArchive BuildFloatArchive(std::span<const FloatTensorSpec> specs)
	{
		std::vector<std::byte> payload;
		std::string header = "{";
		for (std::size_t i = 0; i < specs.size(); ++i)
		{
			const auto& spec = specs[i];
			const auto begin = payload.size();
			for (const auto value : spec.values)
			{
				AppendValue(payload, value);
			}
			const auto end = payload.size();
			if (i != 0)
			{
				header += ",";
			}
			header += "\"" + spec.name + "\":{\"dtype\":\"F32\",\"shape\":" + ShapeJson(spec.shape) +
			          ",\"data_offsets\":[" + std::to_string(begin) + "," + std::to_string(end) + "]}";
		}
		header += "}";
		return Serialization::SafetensorsArchive::Load(BuildSafetensors(std::move(header), std::move(payload)));
	}

	Serialization::SafetensorsArchive BuildLinearArchive(bool includeExtra = false)
	{
		std::vector<std::byte> payload;
		for (const auto value : std::array<float, 6>{ 0.25F, 2.0F, -0.75F, -1.0F, 0.5F, 1.5F })
		{
			AppendValue(payload, value);
		}
		for (const auto value : std::array<float, 2>{ 0.1F, -0.2F })
		{
			AppendValue(payload, value);
		}
		if (includeExtra)
		{
			AppendValue(payload, 42.0F);
		}

		const auto header = includeExtra
		                        ? R"({"linear.weight":{"dtype":"F32","shape":[2,3],"data_offsets":[0,24]},"linear.bias":{"dtype":"F32","shape":[2],"data_offsets":[24,32]},"unused.weight":{"dtype":"F32","shape":[1],"data_offsets":[32,36]}})"
		                        : R"({"linear.weight":{"dtype":"F32","shape":[2,3],"data_offsets":[0,24]},"linear.bias":{"dtype":"F32","shape":[2],"data_offsets":[24,32]}})";
		return Serialization::SafetensorsArchive::Load(BuildSafetensors(header, std::move(payload)));
	}

	Serialization::SafetensorsArchive BuildHalfLinearArchive()
	{
		std::vector<std::byte> payload;
		for (const auto value : std::array<Float16, 2>{ Float16{ 0.5F }, Float16{ 1.0F } })
		{
			AppendValue(payload, value);
		}
		AppendValue(payload, Float16{ 0.25F });

		return Serialization::SafetensorsArchive::Load(BuildSafetensors(
		    R"({"linear.weight":{"dtype":"F16","shape":[1,2],"data_offsets":[0,4]},"linear.bias":{"dtype":"F16","shape":[1],"data_offsets":[4,6]}})",
		    std::move(payload)));
	}

	Serialization::SafetensorsArchive BuildDiffusionBlockArchive()
	{
		std::vector<std::byte> payload;
		for (const auto value : std::array<float, 4>{ 2.0F, 1.0F, 1.0F, 0.0F })
		{
			AppendValue(payload, value);
		}

		return Serialization::SafetensorsArchive::Load(BuildSafetensors(
		    R"({"conv.weight":{"dtype":"F32","shape":[1,1,1,1],"data_offsets":[0,4]},"conv.bias":{"dtype":"F32","shape":[1],"data_offsets":[4,8]},"norm.weight":{"dtype":"F32","shape":[1],"data_offsets":[8,12]},"norm.bias":{"dtype":"F32","shape":[1],"data_offsets":[12,16]}})",
		    std::move(payload)));
	}

	Serialization::SafetensorsArchive BuildCompositeDiffusionArchive()
	{
		const std::array<FloatTensorSpec, 26> specs{ {
		    { "res.norm1.weight", { 1 }, { 1.0F } },
		    { "res.norm1.bias", { 1 }, { 0.0F } },
		    { "res.norm2.weight", { 1 }, { 1.0F } },
		    { "res.norm2.bias", { 1 }, { 0.0F } },
		    { "res.conv1.weight", { 1, 1, 1, 1 }, { 0.0F } },
		    { "res.conv1.bias", { 1 }, { 0.0F } },
		    { "res.conv2.weight", { 1, 1, 1, 1 }, { 0.0F } },
		    { "res.conv2.bias", { 1 }, { 0.0F } },
		    { "ff.up.weight", { 2, 2 }, { 1.0F, 0.0F, 0.0F, 1.0F } },
		    { "ff.up.bias", { 2 }, { 0.0F, 0.0F } },
		    { "ff.down.weight", { 2, 2 }, { 1.0F, 0.0F, 0.0F, 1.0F } },
		    { "ff.down.bias", { 2 }, { 0.0F, 0.0F } },
		    { "attn.q.weight", { 2, 2 }, { 1.0F, 0.0F, 0.0F, 1.0F } },
		    { "attn.q.bias", { 2 }, { 0.0F, 0.0F } },
		    { "attn.k.weight", { 2, 2 }, { 1.0F, 0.0F, 0.0F, 1.0F } },
		    { "attn.k.bias", { 2 }, { 0.0F, 0.0F } },
		    { "attn.v.weight", { 2, 2 }, { 1.0F, 0.0F, 0.0F, 1.0F } },
		    { "attn.v.bias", { 2 }, { 0.0F, 0.0F } },
		    { "attn.out.weight", { 2, 2 }, { 1.0F, 0.0F, 0.0F, 1.0F } },
		    { "attn.out.bias", { 2 }, { 0.0F, 0.0F } },
		    { "vae.conv.weight", { 1, 1, 1, 1 }, { 0.0F } },
		    { "vae.conv.bias", { 1 }, { 1.0F } },
		    { "vae.norm.weight", { 1 }, { 1.0F } },
		    { "vae.norm.bias", { 1 }, { 0.0F } },
		    { "vae.deconv.weight", { 1, 1, 1, 1 }, { 0.0F } },
		    { "vae.deconv.bias", { 1 }, { 1.0F } },
		} };
		return BuildFloatArchive(specs);
	}

	std::string BuildLinearManifest(std::string_view weightDType = "F32",
	                                std::string_view weightShape = "[3,2]",
	                                std::string_view weightSource = "linear.weight",
	                                std::string_view weightLayout = "torch_linear_weight",
	                                std::string_view weightSourceShape = "[2,3]")
	{
		return std::string(R"({
  "format":"litenn.torch_manifest.v1",
  "inputs":[{"name":"x","dtype":"torch.float32","shape":[2,3]}],
  "tensors":[
    {"name":"fc.weight","source":")") + std::string(weightSource) + R"(","dtype":")" +
		       std::string(weightDType) + R"(","source_shape":)" + std::string(weightSourceShape) + R"(,"layout":")" +
		       std::string(weightLayout) + R"(","shape":)" + std::string(weightShape) + R"(},
    {"name":"fc.bias","source":"linear.bias","dtype":"F32","source_shape":[2],"layout":"torch_bias_1d","shape":[1,2]}
  ],
  "nodes":[
    {"name":"fc","op":"linear","input":"x","weight":"fc.weight","bias":"fc.bias","output":"fc_out"},
    {"name":"relu","op":"relu","input":"fc_out","output":"relu_linear"}
  ],
  "outputs":["relu_linear"]
})";
	}

	std::string BuildDiffusionBlockManifest()
	{
		return R"({
  "format":"litenn.torch_manifest.v1",
  "inputs":[
    {"name":"x","dtype":"torch.float32","shape":[1,1,2,2]},
    {"name":"t","dtype":"torch.float32","shape":[1]}
  ],
  "tensors":[
    {"name":"conv.weight","source":"conv.weight","dtype":"F32","shape":[1,1,1,1],"layout":"torch_conv2d_weight"},
    {"name":"conv.bias","source":"conv.bias","dtype":"F32","shape":[1],"layout":"identity"},
    {"name":"norm.weight","source":"norm.weight","dtype":"F32","shape":[1,1,1,1],"layout":"torch_groupnorm_weight"},
    {"name":"norm.bias","source":"norm.bias","dtype":"F32","shape":[1,1,1,1],"layout":"torch_groupnorm_bias"}
  ],
  "nodes":[
    {"name":"pad","op":"pad","input":"x","low_pads":[0,0,0,0],"high_pads":[0,0,1,1],"output":"padded"},
    {"name":"conv","op":"conv2d","input":"padded","weight":"conv.weight","bias":"conv.bias","strides":[1,1],"dilations":[1,1],"padding":[0,0],"groups":1,"output":"conv_out"},
    {"name":"norm","op":"group_norm","input":"conv_out","weight":"norm.weight","bias":"norm.bias","num_groups":1,"eps":0.00001,"output":"norm_out"},
    {"name":"act","op":"silu","input":"norm_out","output":"act_out"},
    {"name":"up","op":"upsample","input":"act_out","mode":"nearest","output_spatial_shape":[6,6],"output":"features"},
    {"name":"temb","op":"timestep_embedding","timesteps":"t","dim":4,"max_period":10000,"output":"temb"}
  ],
  "outputs":[
    {"name":"features","source":"features"},
    {"name":"temb","source":"temb"}
  ]
})";
	}

	std::string BuildCompositeDiffusionManifest()
	{
		return R"({
  "format":"litenn.torch_manifest.v1",
  "inputs":[
    {"name":"res_x","dtype":"torch.float32","shape":[1,1,2,2]},
    {"name":"ff_x","dtype":"torch.float32","shape":[1,2]},
    {"name":"attn_x","dtype":"torch.float32","shape":[2,2]},
    {"name":"vae_z","dtype":"torch.float32","shape":[1,1,1,2]}
  ],
  "tensors":[
    {"name":"res.norm1.weight","source":"res.norm1.weight","dtype":"F32","shape":[1,1,1,1],"layout":"torch_groupnorm_weight"},
    {"name":"res.norm1.bias","source":"res.norm1.bias","dtype":"F32","shape":[1,1,1,1],"layout":"torch_groupnorm_bias"},
    {"name":"res.norm2.weight","source":"res.norm2.weight","dtype":"F32","shape":[1,1,1,1],"layout":"torch_groupnorm_weight"},
    {"name":"res.norm2.bias","source":"res.norm2.bias","dtype":"F32","shape":[1,1,1,1],"layout":"torch_groupnorm_bias"},
    {"name":"res.conv1.weight","source":"res.conv1.weight","dtype":"F32","shape":[1,1,1,1],"layout":"torch_conv2d_weight"},
    {"name":"res.conv1.bias","source":"res.conv1.bias","dtype":"F32","shape":[1],"layout":"identity"},
    {"name":"res.conv2.weight","source":"res.conv2.weight","dtype":"F32","shape":[1,1,1,1],"layout":"torch_conv2d_weight"},
    {"name":"res.conv2.bias","source":"res.conv2.bias","dtype":"F32","shape":[1],"layout":"identity"},
    {"name":"ff.up.weight","source":"ff.up.weight","dtype":"F32","shape":[2,2],"layout":"torch_linear_weight"},
    {"name":"ff.up.bias","source":"ff.up.bias","dtype":"F32","shape":[1,2],"layout":"torch_bias_1d"},
    {"name":"ff.down.weight","source":"ff.down.weight","dtype":"F32","shape":[2,2],"layout":"torch_linear_weight"},
    {"name":"ff.down.bias","source":"ff.down.bias","dtype":"F32","shape":[1,2],"layout":"torch_bias_1d"},
    {"name":"attn.q.weight","source":"attn.q.weight","dtype":"F32","shape":[2,2],"layout":"torch_linear_weight"},
    {"name":"attn.q.bias","source":"attn.q.bias","dtype":"F32","shape":[1,2],"layout":"torch_bias_1d"},
    {"name":"attn.k.weight","source":"attn.k.weight","dtype":"F32","shape":[2,2],"layout":"torch_linear_weight"},
    {"name":"attn.k.bias","source":"attn.k.bias","dtype":"F32","shape":[1,2],"layout":"torch_bias_1d"},
    {"name":"attn.v.weight","source":"attn.v.weight","dtype":"F32","shape":[2,2],"layout":"torch_linear_weight"},
    {"name":"attn.v.bias","source":"attn.v.bias","dtype":"F32","shape":[1,2],"layout":"torch_bias_1d"},
    {"name":"attn.out.weight","source":"attn.out.weight","dtype":"F32","shape":[2,2],"layout":"torch_linear_weight"},
    {"name":"attn.out.bias","source":"attn.out.bias","dtype":"F32","shape":[1,2],"layout":"torch_bias_1d"},
    {"name":"vae.conv.weight","source":"vae.conv.weight","dtype":"F32","shape":[1,1,1,1],"layout":"torch_conv2d_weight"},
    {"name":"vae.conv.bias","source":"vae.conv.bias","dtype":"F32","shape":[1],"layout":"identity"},
    {"name":"vae.norm.weight","source":"vae.norm.weight","dtype":"F32","shape":[1,1,1,1],"layout":"torch_groupnorm_weight"},
    {"name":"vae.norm.bias","source":"vae.norm.bias","dtype":"F32","shape":[1,1,1,1],"layout":"torch_groupnorm_bias"},
    {"name":"vae.deconv.weight","source":"vae.deconv.weight","dtype":"F32","shape":[1,1,1,1],"layout":"torch_conv_transpose2d_weight"},
    {"name":"vae.deconv.bias","source":"vae.deconv.bias","dtype":"F32","shape":[1],"layout":"identity"}
  ],
  "nodes":[
    {
      "name":"tiny_residual",
      "op":"residual_block",
      "input":"res_x",
      "activation":"silu",
      "norm1":{"num_groups":1,"weight":"res.norm1.weight","bias":"res.norm1.bias"},
      "conv1":{"weight":"res.conv1.weight","bias":"res.conv1.bias","padding":[0,0]},
      "norm2":{"num_groups":1,"weight":"res.norm2.weight","bias":"res.norm2.bias"},
      "conv2":{"weight":"res.conv2.weight","bias":"res.conv2.bias","padding":[0,0]},
      "output":"res_out"
    },
    {
      "name":"tiny_ff",
      "op":"feed_forward",
      "input":"ff_x",
      "activation":"identity",
      "up":{"weight":"ff.up.weight","bias":"ff.up.bias"},
      "down":{"weight":"ff.down.weight","bias":"ff.down.bias"},
      "residual":true,
      "output":"ff_out"
    },
    {
      "name":"tiny_attn",
      "op":"attention_block",
      "input":"attn_x",
      "heads":1,
      "q":{"weight":"attn.q.weight","bias":"attn.q.bias"},
      "k":{"weight":"attn.k.weight","bias":"attn.k.bias"},
      "v":{"weight":"attn.v.weight","bias":"attn.v.bias"},
      "out":{"weight":"attn.out.weight","bias":"attn.out.bias"},
      "residual":false,
      "output":"attn_out"
    },
    {
      "name":"tiny_vae_decode",
      "op":"vae_decode",
      "input":"vae_z",
      "steps":[
        {"op":"conv2d","weight":"vae.conv.weight","bias":"vae.conv.bias","padding":[0,0]},
        {"op":"group_norm","num_groups":1,"weight":"vae.norm.weight","bias":"vae.norm.bias"},
        {"op":"silu"},
        {"op":"upsample","mode":"nearest","output_spatial_shape":[2,4]},
        {"op":"conv_transpose2d","weight":"vae.deconv.weight","bias":"vae.deconv.bias"}
      ],
      "output_scale":0.5,
      "output_bias":0.5,
      "clamp":{"min":0.0,"max":1.0},
      "output":"image"
    }
  ],
  "outputs":[
    {"name":"res_out","source":"res_out"},
    {"name":"ff_out","source":"ff_out"},
    {"name":"attn_out","source":"attn_out"},
    {"name":"image","source":"image"}
  ]
})";
	}

	float ReadFloat(const Tensor<CPU>& tensor, std::size_t index)
	{
		return static_cast<const float*>(tensor.RawData())[index];
	}

	std::array<Tensor<CPU>, 1> MakeInputs()
	{
		return {
			Tensor<CPU>({
			                1.0, -2.0, 0.5,
			                0.0, 3.0, -1.0,
			            },
			            { 2, 3 })
		};
	}

	void ExpectPyTorchLinearReluGolden(const Tensor<CPU>& output)
	{
		// Generated by PyTorch: torch.relu(torch.nn.functional.linear(x, weight, bias)).
		const std::array<float, 4> expected{ 0.0F, 0.0F, 6.85F, 0.0F };
		ASSERT_EQ(output.Shape().ToOwned(), (std::vector<std::size_t>{ 2, 2 }));
		for (std::size_t i = 0; i < expected.size(); ++i)
		{
			EXPECT_NEAR(ReadFloat(output, i), expected[i], 1e-5F) << i;
		}
	}

	void ExpectManifestError(std::string manifest, const Serialization::SafetensorsArchive& archive,
	                         std::string_view expected)
	{
		try
		{
			(void)Serialization::ImportTorchManifest(manifest, archive);
			FAIL() << "expected manifest import to throw";
		}
		catch (const std::runtime_error& ex)
		{
			const std::string message = ex.what();
			EXPECT_NE(message.find(expected), std::string::npos) << message;
		}
	}
} // namespace

TEST(TorchManifest, ReportsSupportedMappingsAndDTypeAliases)
{
	const auto mappings = Serialization::SupportedTorchManifestOpMappings();
	const auto hasLinear = std::ranges::any_of(mappings, [](const auto& mapping) {
		return mapping.torchOp == "linear";
	});
	const auto hasLayerNorm = std::ranges::any_of(mappings, [](const auto& mapping) {
		return mapping.torchOp == "layer_norm";
	});
	const auto hasConv2D = std::ranges::any_of(mappings, [](const auto& mapping) {
		return mapping.torchOp == "conv2d";
	});
	const auto hasGroupNorm = std::ranges::any_of(mappings, [](const auto& mapping) {
		return mapping.torchOp == "group_norm";
	});
	EXPECT_TRUE(hasLinear);
	EXPECT_TRUE(hasLayerNorm);
	EXPECT_TRUE(hasConv2D);
	EXPECT_TRUE(hasGroupNorm);
	EXPECT_EQ(Serialization::MapTorchManifestDataType("torch.float32"), DataType::Float32);
	EXPECT_EQ(Serialization::MapTorchManifestDataType("torch.long"), DataType::Int64);
}

TEST(TorchManifest, ImportsTorchLinearReluManifestAndRunsGolden)
{
	const auto archive = BuildLinearArchive();
	auto result = Serialization::ImportTorchManifest(BuildLinearManifest(), archive);

	ASSERT_EQ(result.graph.VariableCount(), 2u);
	ASSERT_TRUE(result.graph.FindVariable("fc.weight").has_value());
	ASSERT_TRUE(result.graph.FindVariable("fc.bias").has_value());
	EXPECT_EQ(result.graph.GetVariable(*result.graph.FindVariable("fc.weight"))->Data().Shape().ToOwned(),
	          (std::vector<std::size_t>{ 3, 2 }));
	EXPECT_EQ(result.graph.GetVariable(*result.graph.FindVariable("fc.bias"))->Data().Shape().ToOwned(),
	          (std::vector<std::size_t>{ 1, 2 }));
	EXPECT_EQ(result.graph.InputSignature()[0].name, "x");
	EXPECT_EQ(result.graph.OutputSignature()[0].name, "relu_linear");
	EXPECT_EQ(result.report.importedTensors.size(), 2u);
	EXPECT_EQ(result.report.loweredOps.size(), 2u);
	EXPECT_FALSE(result.report.foldedConstants.empty());

	auto inputs = MakeInputs();
	Runtime::Interpreter<CPU> interpreter;
	const auto outputs = interpreter.RunForward(result.graph, std::span<const Tensor<CPU>>(inputs));
	ASSERT_EQ(outputs.size(), 1u);
	ExpectPyTorchLinearReluGolden(outputs[0]);

#ifdef LITENN_ENABLE_MLIR
	auto module = Compiler<CPU>::Compile(result.graph);
	const auto compiledOutputs = module.Run(std::span<const Tensor<CPU>>(inputs));
	ASSERT_EQ(compiledOutputs.size(), 1u);
	ExpectPyTorchLinearReluGolden(compiledOutputs[0]);
#endif
}

TEST(TorchManifest, ReportsManifestTensorDiagnostics)
{
	const auto archive = BuildLinearArchive();
	ExpectManifestError(BuildLinearManifest("BF16"), archive, "dtype mismatch");
	ExpectManifestError(BuildLinearManifest("F32", "[4,2]"), archive, "shape mismatch");
	ExpectManifestError(BuildLinearManifest("F32", "[3,2]", "missing.weight"), archive, "was not found");
	ExpectManifestError(BuildLinearManifest("F32", "[2]", "linear.bias", "torch_linear_weight", "[2]"), archive,
	                    "expects rank-2");

	const auto extraArchive = BuildLinearArchive(true);
	ExpectManifestError(BuildLinearManifest(), extraArchive, "extra tensor");
}

TEST(TorchManifest, ConvertsManifestTensorTargetDType)
{
	const auto archive = BuildHalfLinearArchive();
	const auto manifest = R"({
  "format":"litenn.torch_manifest.v1",
  "inputs":[{"name":"x","dtype":"F32","shape":[1,2]}],
  "tensors":[
    {"name":"fc.weight","source":"linear.weight","dtype":"F16","target_dtype":"F32","layout":"torch_linear_weight","shape":[2,1]},
    {"name":"fc.bias","source":"linear.bias","dtype":"F16","target_dtype":"F32","layout":"torch_bias_1d","shape":[1,1]}
  ],
  "nodes":[
    {"name":"fc","op":"linear","input":"x","weight":"fc.weight","bias":"fc.bias","output":"y"}
  ],
  "outputs":[{"name":"y","source":"y"}]
})";

	auto result = Serialization::ImportTorchManifest(manifest, archive);
	ASSERT_TRUE(result.graph.FindVariable("fc.weight").has_value());
	ASSERT_TRUE(result.graph.FindVariable("fc.bias").has_value());
	EXPECT_EQ(result.graph.GetVariable(*result.graph.FindVariable("fc.weight"))->Data().DType(), DataType::Float32);
	EXPECT_EQ(result.graph.GetVariable(*result.graph.FindVariable("fc.bias"))->Data().DType(), DataType::Float32);
	EXPECT_GE(result.report.foldedConstants.size(), 2u);

	std::array<Tensor<CPU>, 1> inputs{ Tensor<CPU>({ 2.0, 3.0 }, { 1, 2 }, DataType::Float32) };
	Runtime::Interpreter<CPU> interpreter;
	const auto outputs = interpreter.RunForward(result.graph, std::span<const Tensor<CPU>>(inputs));
	ASSERT_EQ(outputs.size(), 1u);
	ASSERT_EQ(outputs[0].Shape().ToOwned(), (std::vector<std::size_t>{ 1, 1 }));
	EXPECT_NEAR(ReadFloat(outputs[0], 0), 4.25F, 1e-5F);
}

TEST(TorchManifest, ImportsDiffusionFoundationOps)
{
	const auto archive = BuildDiffusionBlockArchive();
	auto result = Serialization::ImportTorchManifest(BuildDiffusionBlockManifest(), archive);

	ASSERT_EQ(result.graph.VariableCount(), 4u);
	EXPECT_EQ(result.graph.OutputSignature().size(), 2u);
	EXPECT_EQ(result.graph.OutputSignature()[0].shape, (std::vector<std::size_t>{ 1, 1, 6, 6 }));
	EXPECT_EQ(result.graph.OutputSignature()[1].shape, (std::vector<std::size_t>{ 1, 4 }));
	EXPECT_EQ(result.report.loweredOps.size(), 6u);
	EXPECT_GE(result.report.foldedConstants.size(), 2u);

	std::array<Tensor<CPU>, 2> inputs = {
		Tensor<CPU>({ 1.0, 2.0, 3.0, 4.0 }, { 1, 1, 2, 2 }),
		Tensor<CPU>({ 10.0 }, { 1 }),
	};
	Runtime::Interpreter<CPU> interpreter;
	const auto outputs = interpreter.RunForward(result.graph, std::span<const Tensor<CPU>>(inputs));
	ASSERT_EQ(outputs.size(), 2u);
	EXPECT_EQ(outputs[0].Shape().ToOwned(), (std::vector<std::size_t>{ 1, 1, 6, 6 }));
	EXPECT_EQ(outputs[1].Shape().ToOwned(), (std::vector<std::size_t>{ 1, 4 }));
	for (const auto& output : outputs)
	{
		for (std::size_t i = 0; i < output.NumElements(); ++i)
		{
			EXPECT_TRUE(std::isfinite(ReadFloat(output, i))) << i;
		}
	}
}

TEST(TorchManifest, ImportsSDXLCompositePatternsWithTinyParityFixture)
{
	const auto archive = BuildCompositeDiffusionArchive();
	auto result = Serialization::ImportTorchManifest(BuildCompositeDiffusionManifest(), archive);

	ASSERT_EQ(result.graph.OutputSignature().size(), 4u);
	EXPECT_EQ(result.graph.OutputSignature()[0].shape, (std::vector<std::size_t>{ 1, 1, 2, 2 }));
	EXPECT_EQ(result.graph.OutputSignature()[1].shape, (std::vector<std::size_t>{ 1, 2 }));
	EXPECT_EQ(result.graph.OutputSignature()[2].shape, (std::vector<std::size_t>{ 2, 2 }));
	EXPECT_EQ(result.graph.OutputSignature()[3].shape, (std::vector<std::size_t>{ 1, 1, 2, 4 }));
	EXPECT_GE(result.report.loweredOps.size(), 4u);

	std::array<Tensor<CPU>, 4> inputs = {
		Tensor<CPU>({ 1.0, -2.0, 3.0, -4.0 }, { 1, 1, 2, 2 }),
		Tensor<CPU>({ 1.0, -2.0 }, { 1, 2 }),
		Tensor<CPU>({ 1.0, 0.0, 0.0, 1.0 }, { 2, 2 }),
		Tensor<CPU>({ 2.0, 3.0 }, { 1, 1, 1, 2 }),
	};
	Runtime::Interpreter<CPU> interpreter;
	const auto outputs = interpreter.RunForward(result.graph, std::span<const Tensor<CPU>>(inputs));
	ASSERT_EQ(outputs.size(), 4u);

	const std::array<float, 4> expectedResidual{ 1.0F, -2.0F, 3.0F, -4.0F };
	for (std::size_t i = 0; i < expectedResidual.size(); ++i)
	{
		EXPECT_NEAR(ReadFloat(outputs[0], i), expectedResidual[i], 1e-5F) << i;
	}
	const std::array<float, 2> expectedFeedForward{ 2.0F, -4.0F };
	for (std::size_t i = 0; i < expectedFeedForward.size(); ++i)
	{
		EXPECT_NEAR(ReadFloat(outputs[1], i), expectedFeedForward[i], 1e-5F) << i;
	}
	const std::array<float, 4> expectedAttention{ 0.66976154F, 0.33023846F, 0.33023846F, 0.66976154F };
	for (std::size_t i = 0; i < expectedAttention.size(); ++i)
	{
		EXPECT_NEAR(ReadFloat(outputs[2], i), expectedAttention[i], 1e-5F) << i;
	}
	for (std::size_t i = 0; i < outputs[3].NumElements(); ++i)
	{
		EXPECT_NEAR(ReadFloat(outputs[3], i), 1.0F, 1e-5F) << i;
	}
}
