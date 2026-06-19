#include <LiteNN/DType.h>
#include <LiteNN/Graph.h>
#include <LiteNN/Layer/LoRA.h>
#include <LiteNN/Serialization/ImportManifest.h>
#include <LiteNN/Tensor.h>
#include <LiteNN/TensorType.h>

#include <cstddef>
#include <filesystem>
#include <functional>
#include <optional>
#include <span>
#include <string>
#include <string_view>
#include <vector>

#ifndef LITENN_SERIALIZATION_SAFETENSORS_H
#define LITENN_SERIALIZATION_SAFETENSORS_H

namespace LiteNN::Serialization
{
	struct SafetensorsTensorInfo
	{
		std::string name;
		std::string storageDType;
		TensorType type;
		std::size_t dataBegin{};
		std::size_t dataEnd{};

		std::size_t ByteSize() const;
	};

	struct SafetensorsImportOptions
	{
		std::function<std::string(std::string_view)> renameTensor;
		std::function<bool(std::string_view)> transpose2D;
		bool failOnDuplicateVariableNames{ true };
	};

	struct SafetensorsLoRAImportOptions
	{
		std::function<std::string(std::string_view)> renameTarget;
		float defaultAlpha = 0.0f;
		bool transposePEFTWeights = true;
	};

	struct SafetensorsLoRAImportResult
	{
		std::vector<Layer::LinearLoRAAdapter> adapters;
		std::vector<std::string> diagnostics;
	};

	std::optional<DataType> TryMapSafetensorsDataType(std::string_view dtype);
	DataType MapSafetensorsDataType(std::string_view dtype);

	namespace Detail
	{
		struct SafetensorsArchiveBuilder;
	}

	class SafetensorsArchive
	{
	public:
		static SafetensorsArchive Load(std::span<const std::byte> bytes);
		static SafetensorsArchive LoadFile(const std::filesystem::path& path);

		std::span<const SafetensorsTensorInfo> Tensors() const;
		std::span<const ModelMetadataEntry> Metadata() const;
		const SafetensorsTensorInfo* FindTensor(std::string_view name) const;
		std::span<const std::byte> TensorData(const SafetensorsTensorInfo& tensor) const;
		Tensor<CPU> TensorAsCPU(const SafetensorsTensorInfo& tensor, bool transpose2D = false) const;

	private:
		friend struct Detail::SafetensorsArchiveBuilder;

		std::vector<std::byte> storage_;
		std::filesystem::path backingPath_;
		mutable std::vector<std::byte> tensorReadBuffer_;
		std::size_t payloadOffset_{};
		std::vector<SafetensorsTensorInfo> tensors_;
		std::vector<ModelMetadataEntry> metadata_;
	};

	Graph ImportSafetensorsVariables(const SafetensorsArchive& archive, const SafetensorsImportOptions& options = {});
	Graph LoadSafetensorsVariables(const std::filesystem::path& path, const SafetensorsImportOptions& options = {});
	ImporterOwnedManifest ImportSafetensorsVariablesManifest(const SafetensorsArchive& archive,
	                                                         const SafetensorsImportOptions& options = {});
	SafetensorsLoRAImportResult ImportLinearLoRAAdapters(Graph& graph, const SafetensorsArchive& archive,
	                                                     const SafetensorsLoRAImportOptions& options = {});
} // namespace LiteNN::Serialization

#endif
