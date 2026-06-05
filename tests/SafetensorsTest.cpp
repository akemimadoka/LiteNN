#include <gtest/gtest.h>

#include <LiteNN/Serialization/Safetensors.h>

#include <array>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <span>
#include <string>
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

	std::vector<std::byte> BuildFixture()
	{
		std::vector<std::byte> payload;
		for (const auto value : std::array<float, 6>{ 1.0F, 2.0F, 3.0F, 4.0F, 5.0F, 6.0F })
		{
			AppendValue(payload, value);
		}
		for (const auto value : std::array<std::int64_t, 2>{ 7, 8 })
		{
			AppendValue(payload, value);
		}

		return BuildSafetensors(
		    R"({"__metadata__":{"format":"pt"},"linear.weight":{"dtype":"F32","shape":[2,3],"data_offsets":[0,24]},"linear.bias":{"dtype":"I64","shape":[2],"data_offsets":[24,40]}})",
		    std::move(payload));
	}

	float ReadFloat(const Tensor<CPU>& tensor, std::size_t index)
	{
		return static_cast<const float*>(tensor.RawData())[index];
	}

	std::int64_t ReadI64(const Tensor<CPU>& tensor, std::size_t index)
	{
		return static_cast<const std::int64_t*>(tensor.RawData())[index];
	}
} // namespace

TEST(Safetensors, ReadsMetadataDTypesAndPayloads)
{
	const auto bytes = BuildFixture();
	const auto archive = Serialization::SafetensorsArchive::Load(std::span<const std::byte>(bytes));

	ASSERT_EQ(archive.Metadata().size(), 1u);
	EXPECT_EQ(archive.Metadata()[0].key, "format");
	ASSERT_TRUE(std::holds_alternative<std::string>(archive.Metadata()[0].value));
	EXPECT_EQ(std::get<std::string>(archive.Metadata()[0].value), "pt");

	ASSERT_EQ(archive.Tensors().size(), 2u);
	const auto* weightInfo = archive.FindTensor("linear.weight");
	ASSERT_NE(weightInfo, nullptr);
	EXPECT_EQ(weightInfo->type.dtype, DataType::Float32);
	EXPECT_EQ(weightInfo->storageDType, "F32");
	EXPECT_EQ(weightInfo->type.StaticShape(), (std::vector<std::size_t>{ 2, 3 }));
	EXPECT_EQ(weightInfo->ByteSize(), 24u);

	const auto weight = archive.TensorAsCPU(*weightInfo);
	EXPECT_EQ(weight.Shape().ToOwned(), (std::vector<std::size_t>{ 2, 3 }));
	EXPECT_FLOAT_EQ(ReadFloat(weight, 0), 1.0F);
	EXPECT_FLOAT_EQ(ReadFloat(weight, 5), 6.0F);

	const auto* biasInfo = archive.FindTensor("linear.bias");
	ASSERT_NE(biasInfo, nullptr);
	EXPECT_EQ(biasInfo->type.dtype, DataType::Int64);
	const auto bias = archive.TensorAsCPU(*biasInfo);
	EXPECT_EQ(ReadI64(bias, 0), 7);
	EXPECT_EQ(ReadI64(bias, 1), 8);
}

TEST(Safetensors, LoadFileReadsHeaderAndPayloadOnDemand)
{
	const auto bytes = BuildFixture();
	const auto path = std::filesystem::temp_directory_path() / "litenn_safetensors_loadfile_fixture.safetensors";
	{
		std::ofstream out(path, std::ios::binary);
		ASSERT_TRUE(out);
		out.write(reinterpret_cast<const char*>(bytes.data()), static_cast<std::streamsize>(bytes.size()));
		ASSERT_TRUE(out);
	}

	const auto archive = Serialization::SafetensorsArchive::LoadFile(path);
	ASSERT_EQ(archive.Tensors().size(), 2u);
	const auto* weightInfo = archive.FindTensor("linear.weight");
	ASSERT_NE(weightInfo, nullptr);
	const auto weight = archive.TensorAsCPU(*weightInfo);
	EXPECT_FLOAT_EQ(ReadFloat(weight, 0), 1.0F);
	EXPECT_FLOAT_EQ(ReadFloat(weight, 5), 6.0F);

	std::filesystem::remove(path);
}

TEST(Safetensors, ImportsVariablesWithRenameAndTransposeHooks)
{
	const auto bytes = BuildFixture();
	const auto archive = Serialization::SafetensorsArchive::Load(std::span<const std::byte>(bytes));

	Serialization::SafetensorsImportOptions options;
	options.renameTensor = [](std::string_view name) {
		if (name == "linear.weight")
		{
			return std::string("fc.weight");
		}
		return std::string(name);
	};
	options.transpose2D = [](std::string_view name) { return name == "linear.weight"; };

	const auto graph = Serialization::ImportSafetensorsVariables(archive, options);
	ASSERT_EQ(graph.VariableCount(), 2u);
	const auto weightIndex = graph.FindVariable("fc.weight");
	ASSERT_TRUE(weightIndex.has_value());
	const auto weight = graph.GetVariable(*weightIndex)->Data().CopyToDevice(CPU{});
	EXPECT_EQ(weight.Shape().ToOwned(), (std::vector<std::size_t>{ 3, 2 }));
	EXPECT_FLOAT_EQ(ReadFloat(weight, 0), 1.0F);
	EXPECT_FLOAT_EQ(ReadFloat(weight, 1), 4.0F);
	EXPECT_FLOAT_EQ(ReadFloat(weight, 2), 2.0F);
	EXPECT_FLOAT_EQ(ReadFloat(weight, 3), 5.0F);
	EXPECT_FLOAT_EQ(ReadFloat(weight, 4), 3.0F);
	EXPECT_FLOAT_EQ(ReadFloat(weight, 5), 6.0F);

	const auto biasIndex = graph.FindVariable("linear.bias");
	ASSERT_TRUE(biasIndex.has_value());
	const auto bias = graph.GetVariable(*biasIndex)->Data().CopyToDevice(CPU{});
	EXPECT_EQ(ReadI64(bias, 0), 7);
	EXPECT_EQ(ReadI64(bias, 1), 8);
	ASSERT_NE(graph.FindMetadata("safetensors.metadata.format"), nullptr);
}

TEST(Safetensors, RejectsCorruptHeadersAndPayloads)
{
	EXPECT_THROW(Serialization::SafetensorsArchive::Load(std::span<const std::byte>()), std::runtime_error);

	std::vector<std::byte> payload(4);
	EXPECT_THROW(Serialization::SafetensorsArchive::Load(BuildSafetensors(
	                 R"({"x":{"dtype":"U64","shape":[1],"data_offsets":[0,8]}})", payload)),
	             std::runtime_error);

	EXPECT_THROW(Serialization::SafetensorsArchive::Load(BuildSafetensors(
	                 R"({"x":{"dtype":"F32","shape":[2],"data_offsets":[0,4]}})", payload)),
	             std::runtime_error);

	EXPECT_THROW(Serialization::SafetensorsArchive::Load(BuildSafetensors(
	                 R"({"x":{"dtype":"F32","shape":[1],"data_offsets":[0,8]}})", payload)),
	             std::runtime_error);

	EXPECT_THROW(Serialization::SafetensorsArchive::Load(BuildSafetensors(
	                 R"({"x":{"dtype":"BOOL","shape":[1],"data_offsets":[0,1]}})",
	                 std::vector<std::byte>{ std::byte{ 2 } })),
	             std::runtime_error);
}
