#include "LlamaCppTokenizerAdapter.h"

#include <ggml-backend.h>
#include <ggml.h>
#include <llama.h>

#include <algorithm>
#include <array>
#include <charconv>
#include <chrono>
#include <cmath>
#include <cstddef>
#include <cstring>
#include <filesystem>
#include <format>
#include <fstream>
#include <iomanip>
#include <limits>
#include <map>
#include <memory>
#include <numeric>
#include <optional>
#include <set>
#include <stdexcept>
#include <string>
#include <utility>

namespace LiteNN::LlamaCppAdapter
{
	namespace
	{
		void EnsureBackendInitialized()
		{
			static const bool initialized = [] {
				llama_backend_init();
				return true;
			}();
			(void) initialized;
		}

		std::vector<llama_token> ToLlamaTokens(std::span<const std::int32_t> tokenIds)
		{
			std::vector<llama_token> tokens;
			tokens.reserve(tokenIds.size());
			for (const auto token : tokenIds)
			{
				if (token < 0)
				{
					throw std::runtime_error("token ids must be non-negative");
				}
				tokens.push_back(static_cast<llama_token>(token));
			}
			return tokens;
		}

		std::vector<std::int32_t> FromLlamaTokens(std::span<const llama_token> tokens)
		{
			std::vector<std::int32_t> result;
			result.reserve(tokens.size());
			for (const auto token : tokens)
			{
				if (token < 0 || token > std::numeric_limits<std::int32_t>::max())
				{
					throw std::runtime_error("llama.cpp returned an out-of-range token id");
				}
				result.push_back(static_cast<std::int32_t>(token));
			}
			return result;
		}

		void WriteLogits(const float* logits, std::int32_t vocabularySize, const std::filesystem::path& path)
		{
			if (logits == nullptr)
			{
				throw std::runtime_error("llama.cpp returned null logits");
			}
			std::ofstream output(path);
			if (!output)
			{
				throw std::runtime_error("failed to open logits output: " + path.string());
			}
			output << std::setprecision(9);
			for (std::int32_t i = 0; i < vocabularySize; ++i)
			{
				output << i << ": " << logits[i] << '\n';
			}
		}

		std::optional<std::size_t> ParseLayerIndex(const ggml_tensor* tensor)
		{
			constexpr std::string_view prefix = "l_out-";
			const std::string_view name = ggml_get_name(tensor);
			if (!name.starts_with(prefix))
			{
				return std::nullopt;
			}
			const auto suffix = name.substr(prefix.size());
			std::size_t layer{};
			const auto parsed = std::from_chars(suffix.data(), suffix.data() + suffix.size(), layer);
			if (suffix.empty() || parsed.ec != std::errc{} || parsed.ptr != suffix.data() + suffix.size())
			{
				return std::nullopt;
			}
			return layer;
		}

		std::vector<float> ReadLastTensorRow(const ggml_tensor* tensor, std::size_t expectedWidth)
		{
			if (tensor->ne[0] != static_cast<std::int64_t>(expectedWidth) || tensor->ne[1] <= 0 || tensor->ne[2] != 1 ||
			    tensor->ne[3] != 1)
			{
				throw std::runtime_error("llama.cpp l_out tensor has an unexpected shape");
			}
			std::size_t elementSize{};
			switch (tensor->type)
			{
			case GGML_TYPE_F32:
				elementSize = sizeof(float);
				break;
			case GGML_TYPE_F16:
				elementSize = sizeof(ggml_fp16_t);
				break;
			case GGML_TYPE_BF16:
				elementSize = sizeof(ggml_bf16_t);
				break;
			default:
				throw std::runtime_error("llama.cpp l_out tensor has unsupported dtype " +
				                         std::string(ggml_type_name(tensor->type)));
			}
			if (tensor->nb[0] != elementSize || tensor->nb[1] < expectedWidth * elementSize)
			{
				throw std::runtime_error("llama.cpp l_out tensor has unsupported row strides");
			}

			const auto byteCount = expectedWidth * elementSize;
			const auto byteOffset = static_cast<std::size_t>(tensor->ne[1] - 1) * tensor->nb[1];
			std::vector<std::byte> bytes(byteCount);
			ggml_backend_tensor_get(tensor, bytes.data(), byteOffset, byteCount);
			std::vector<float> values(expectedWidth);
			if (tensor->type == GGML_TYPE_F32)
			{
				std::memcpy(values.data(), bytes.data(), byteCount);
			}
			else if (tensor->type == GGML_TYPE_F16)
			{
				ggml_fp16_to_fp32_row(reinterpret_cast<const ggml_fp16_t*>(bytes.data()), values.data(),
				                      static_cast<std::int64_t>(expectedWidth));
			}
			else
			{
				ggml_bf16_to_fp32_row(reinterpret_cast<const ggml_bf16_t*>(bytes.data()), values.data(),
				                      static_cast<std::int64_t>(expectedWidth));
			}
			return values;
		}

		std::uint64_t FNV1a(std::span<const float> values)
		{
			const auto* bytes = reinterpret_cast<const unsigned char*>(values.data());
			const auto byteCount = values.size_bytes();
			std::uint64_t hash = 14695981039346656037ull;
			for (std::size_t i = 0; i < byteCount; ++i)
			{
				hash ^= bytes[i];
				hash *= 1099511628211ull;
			}
			return hash;
		}

		struct LayerSummary
		{
			double minimum{ std::numeric_limits<double>::infinity() };
			double maximum{ -std::numeric_limits<double>::infinity() };
			double mean{};
			double rms{};
			std::size_t nonFinite{};
		};

		LayerSummary Summarize(std::span<const float> values)
		{
			LayerSummary result;
			double sum = 0.0;
			double squareSum = 0.0;
			for (const auto value : values)
			{
				if (!std::isfinite(value))
				{
					++result.nonFinite;
					continue;
				}
				result.minimum = std::min(result.minimum, static_cast<double>(value));
				result.maximum = std::max(result.maximum, static_cast<double>(value));
				sum += value;
				squareSum += static_cast<double>(value) * value;
			}
			const auto finiteCount = values.size() - result.nonFinite;
			if (finiteCount == 0)
			{
				result.minimum = result.maximum = result.mean = result.rms = std::numeric_limits<double>::quiet_NaN();
			}
			else
			{
				result.mean = sum / static_cast<double>(finiteCount);
				result.rms = std::sqrt(squareSum / static_cast<double>(finiteCount));
			}
			return result;
		}

		class LayerCheckpointCapture
		{
		public:
			LayerCheckpointCapture(std::size_t layerCount, std::size_t hiddenWidth,
			                       const std::filesystem::path& outputDirectory)
			    : layerCount_(layerCount), hiddenWidth_(hiddenWidth), outputDirectory_(outputDirectory)
			{
				std::filesystem::create_directories(outputDirectory_);
				manifest_.open(outputDirectory_ / "manifest.tsv", std::ios::binary | std::ios::trunc);
				if (!manifest_)
				{
					throw std::runtime_error("failed to open llama.cpp layer checkpoint manifest");
				}
				manifest_
				    << "# litenn-layer-checkpoints-v1\n"
				    << "generated_index\tabsolute_step\tposition\tinput_token_id\tfile\tlayer\tname\tdtype\tshape\t"
				       "byte_offset\tbyte_size\tminimum\tmaximum\tmean\trms\tnon_finite\tchecksum_fnv1a64\n";
			}

			void Begin(std::size_t generatedIndex)
			{
				activeGeneratedIndex_ = generatedIndex;
				layers_.clear();
				error_.clear();
			}

			void End(std::size_t absoluteStep, std::size_t position, std::int32_t inputTokenId)
			{
				if (!error_.empty())
				{
					throw std::runtime_error(error_);
				}
				if (!activeGeneratedIndex_ || layers_.size() != layerCount_)
				{
					throw std::runtime_error("llama.cpp did not capture every decoder layer");
				}
				for (std::size_t layer = 0; layer < layerCount_; ++layer)
				{
					if (!layers_.contains(layer))
					{
						throw std::runtime_error("llama.cpp layer checkpoints are not contiguous");
					}
				}

				const auto fileName = "generated-" + SixDigit(*activeGeneratedIndex_) + ".bin";
				std::ofstream payload(outputDirectory_ / fileName, std::ios::binary | std::ios::trunc);
				if (!payload)
				{
					throw std::runtime_error("failed to open llama.cpp layer checkpoint payload");
				}
				std::size_t byteOffset = 0;
				for (std::size_t layer = 0; layer < layerCount_; ++layer)
				{
					const auto& values = layers_.at(layer);
					const auto byteSize = values.size() * sizeof(float);
					payload.write(reinterpret_cast<const char*>(values.data()), static_cast<std::streamsize>(byteSize));
					const auto summary = Summarize(values);
					manifest_ << std::setprecision(17) << *activeGeneratedIndex_ << '\t' << absoluteStep << '\t'
					          << position << '\t' << inputTokenId << '\t' << fileName << '\t' << layer << '\t'
					          << "layer_hidden_" << layer << "\tFloat32\t1x" << hiddenWidth_ << '\t' << byteOffset
					          << '\t' << byteSize << '\t' << summary.minimum << '\t' << summary.maximum << '\t'
					          << summary.mean << '\t' << summary.rms << '\t' << summary.nonFinite << '\t' << std::hex
					          << std::setw(16) << std::setfill('0') << FNV1a(values) << std::dec << std::setfill(' ')
					          << '\n';
					byteOffset += byteSize;
				}
				if (!payload || !manifest_)
				{
					throw std::runtime_error("failed to write llama.cpp layer checkpoints");
				}
				manifest_.flush();
				activeGeneratedIndex_.reset();
				layers_.clear();
			}

			static bool Callback(ggml_tensor* tensor, bool ask, void* userData) noexcept
			{
				auto& self = *static_cast<LayerCheckpointCapture*>(userData);
				if (!self.activeGeneratedIndex_ || !ParseLayerIndex(tensor))
				{
					return false;
				}
				if (ask)
				{
					return true;
				}
				try
				{
					const auto layer = *ParseLayerIndex(tensor);
					if (layer >= self.layerCount_ || self.layers_.contains(layer))
					{
						throw std::runtime_error("llama.cpp emitted an invalid or duplicate l_out tensor");
					}
					self.layers_.emplace(layer, ReadLastTensorRow(tensor, self.hiddenWidth_));
					return true;
				}
				catch (const std::exception& error)
				{
					self.error_ = error.what();
					return false;
				}
			}

		private:
			static std::string SixDigit(std::size_t value)
			{
				std::string result = std::to_string(value);
				if (result.size() < 6)
				{
					result.insert(result.begin(), 6 - result.size(), '0');
				}
				return result;
			}

			std::size_t layerCount_{};
			std::size_t hiddenWidth_{};
			std::filesystem::path outputDirectory_;
			std::ofstream manifest_;
			std::optional<std::size_t> activeGeneratedIndex_;
			std::map<std::size_t, std::vector<float>> layers_;
			std::string error_;
		};

		constexpr std::array<std::string_view, 15> SubLayerBoundaries{
			"attention_norm",    "query_pre_rope",   "key_pre_rope",       "query_rotated", "key_rotated", "value",
			"attention_context", "attention_output", "attention_residual", "ffn_norm",      "ffn_gate",    "ffn_up",
			"ffn_swiglu",        "ffn_down",         "post_ffn",
		};

		struct NamedLayerTensor
		{
			std::string_view base;
			std::optional<std::size_t> layer;
		};

		NamedLayerTensor ParseNamedLayerTensor(const ggml_tensor* tensor)
		{
			const std::string_view name = ggml_get_name(tensor);
			const auto separator = name.rfind('-');
			if (separator == std::string_view::npos)
			{
				return { .base = name };
			}
			const auto suffix = name.substr(separator + 1);
			std::size_t layer{};
			const auto parsed = std::from_chars(suffix.data(), suffix.data() + suffix.size(), layer);
			if (suffix.empty() || parsed.ec != std::errc{} || parsed.ptr != suffix.data() + suffix.size())
			{
				return { .base = name };
			}
			return { .base = name.substr(0, separator), .layer = layer };
		}

		std::vector<float> ReadContiguousTensor(const ggml_tensor* tensor)
		{
			if (!ggml_is_contiguous(tensor))
			{
				throw std::runtime_error("llama.cpp sub-layer checkpoint tensor is not contiguous: " +
				                         std::string(ggml_get_name(tensor)));
			}
			const auto count = static_cast<std::size_t>(ggml_nelements(tensor));
			std::size_t elementSize{};
			switch (tensor->type)
			{
			case GGML_TYPE_F32:
				elementSize = sizeof(float);
				break;
			case GGML_TYPE_F16:
				elementSize = sizeof(ggml_fp16_t);
				break;
			case GGML_TYPE_BF16:
				elementSize = sizeof(ggml_bf16_t);
				break;
			default:
				throw std::runtime_error("llama.cpp sub-layer checkpoint tensor has unsupported dtype " +
				                         std::string(ggml_type_name(tensor->type)));
			}
			std::vector<std::byte> bytes(count * elementSize);
			ggml_backend_tensor_get(tensor, bytes.data(), 0, bytes.size());
			std::vector<float> values(count);
			if (tensor->type == GGML_TYPE_F32)
			{
				std::memcpy(values.data(), bytes.data(), bytes.size());
			}
			else if (tensor->type == GGML_TYPE_F16)
			{
				ggml_fp16_to_fp32_row(reinterpret_cast<const ggml_fp16_t*>(bytes.data()), values.data(),
				                      static_cast<std::int64_t>(count));
			}
			else
			{
				ggml_bf16_to_fp32_row(reinterpret_cast<const ggml_bf16_t*>(bytes.data()), values.data(),
				                      static_cast<std::int64_t>(count));
			}
			return values;
		}

		class SubLayerCheckpointCapture
		{
		public:
			SubLayerCheckpointCapture(std::size_t layerCount, std::size_t hiddenWidth, std::size_t attentionHeads,
			                          std::size_t kvHeads, std::span<const std::size_t> selectedBlocks,
			                          const std::filesystem::path& outputDirectory)
			    : layerCount_(layerCount), hiddenWidth_(hiddenWidth), attentionHeads_(attentionHeads),
			      kvHeads_(kvHeads), headWidth_(hiddenWidth / attentionHeads),
			      selectedBlocks_(selectedBlocks.begin(), selectedBlocks.end())
			{
				if (attentionHeads == 0 || kvHeads == 0 || hiddenWidth % attentionHeads != 0 || selectedBlocks_.empty())
				{
					throw std::runtime_error("invalid llama.cpp sub-layer checkpoint dimensions or block selection");
				}
				if (selectedBlocks_.size() != selectedBlocks.size() || *selectedBlocks_.rbegin() >= layerCount_)
				{
					throw std::runtime_error("sub-layer checkpoint block indices must be unique and within the model");
				}
				for (const auto boundary : SubLayerBoundaries)
				{
					auto [it, inserted] = groups_.try_emplace(std::string(boundary));
					(void) inserted;
					it->second.directory = outputDirectory / boundary;
					std::filesystem::create_directories(it->second.directory);
					it->second.manifest.open(it->second.directory / "manifest.tsv", std::ios::binary | std::ios::trunc);
					if (!it->second.manifest)
					{
						throw std::runtime_error("failed to open llama.cpp sub-layer checkpoint manifest");
					}
					it->second.manifest
					    << "# litenn-layer-checkpoints-v1\n"
					    << "generated_index\tabsolute_step\tposition\tinput_token_id\tfile\tlayer\tname\tdtype\tshape\t"
					       "byte_offset\tbyte_size\tminimum\tmaximum\tmean\trms\tnon_finite\tchecksum_fnv1a64\n";
				}
			}

			void Begin(std::size_t generatedIndex)
			{
				activeGeneratedIndex_ = generatedIndex;
				values_.clear();
				layerOutputs_.clear();
				inputEmbedding_.clear();
				error_.clear();
			}

			void End(std::size_t absoluteStep, std::size_t position, std::int32_t inputTokenId)
			{
				if (!error_.empty())
				{
					throw std::runtime_error(error_);
				}
				if (!activeGeneratedIndex_)
				{
					throw std::runtime_error("llama.cpp sub-layer checkpoint capture was not active");
				}
				for (const auto block : selectedBlocks_)
				{
					const auto& residual = Require("attention_residual", block);
					const auto& blockInput = block == 0 ? inputEmbedding_ : RequireLayerOutput(block - 1);
					if (residual.size() != blockInput.size())
					{
						throw std::runtime_error(
						    std::format("llama.cpp attention residual and block input shapes differ: {} vs {}",
						                residual.size(), blockInput.size()));
					}
					auto attentionOutput = residual;
					for (std::size_t i = 0; i < attentionOutput.size(); ++i)
					{
						attentionOutput[i] -= blockInput[i];
					}
					values_["attention_output"][block] = std::move(attentionOutput);
				}

				for (const auto boundary : SubLayerBoundaries)
				{
					auto& group = groups_.at(std::string(boundary));
					const auto fileName = "generated-" + SixDigit(*activeGeneratedIndex_) + ".bin";
					std::ofstream payload(group.directory / fileName, std::ios::binary | std::ios::trunc);
					if (!payload)
					{
						throw std::runtime_error("failed to open llama.cpp sub-layer checkpoint payload");
					}
					std::size_t byteOffset = 0;
					for (const auto block : selectedBlocks_)
					{
						const auto& values = Require(boundary, block);
						const auto byteSize = values.size() * sizeof(float);
						payload.write(reinterpret_cast<const char*>(values.data()),
						              static_cast<std::streamsize>(byteSize));
						const auto summary = Summarize(values);
						group.manifest << std::setprecision(17) << *activeGeneratedIndex_ << '\t' << absoluteStep
						               << '\t' << position << '\t' << inputTokenId << '\t' << fileName << '\t' << block
						               << '\t' << "layer_checkpoint_" << boundary << '_' << block << "\tFloat32\t"
						               << ShapeFor(boundary, values.size()) << '\t' << byteOffset << '\t' << byteSize
						               << '\t' << summary.minimum << '\t' << summary.maximum << '\t' << summary.mean
						               << '\t' << summary.rms << '\t' << summary.nonFinite << '\t' << std::hex
						               << std::setw(16) << std::setfill('0') << FNV1a(values) << std::dec
						               << std::setfill(' ') << '\n';
						byteOffset += byteSize;
					}
					group.manifest.flush();
					if (!payload || !group.manifest)
					{
						throw std::runtime_error("failed to write llama.cpp sub-layer checkpoints");
					}
				}
				activeGeneratedIndex_.reset();
				values_.clear();
				layerOutputs_.clear();
				inputEmbedding_.clear();
			}

			static bool Callback(ggml_tensor* tensor, bool ask, void* userData) noexcept
			{
				auto& self = *static_cast<SubLayerCheckpointCapture*>(userData);
				if (!self.activeGeneratedIndex_ || !self.Wants(tensor))
				{
					return false;
				}
				if (ask)
				{
					return true;
				}
				try
				{
					self.Store(tensor);
					return true;
				}
				catch (const std::exception& error)
				{
					self.error_ = error.what();
					return false;
				}
			}

		private:
			struct Group
			{
				std::filesystem::path directory;
				std::ofstream manifest;
			};

			static std::optional<std::string_view> BoundaryFor(const ggml_tensor* tensor)
			{
				const auto base = ParseNamedLayerTensor(tensor).base;
				if (base == "attn_norm")
				{
					return "attention_norm";
				}
				if (base == "Qcur")
				{
					return tensor->op == GGML_OP_ROPE ? "query_rotated" : "query_pre_rope";
				}
				if (base == "Kcur")
				{
					return tensor->op == GGML_OP_ROPE ? "key_rotated" : "key_pre_rope";
				}
				if (base == "Vcur")
				{
					return "value";
				}
				if (base == "kqv_out")
				{
					return "attention_context";
				}
				if (base == "ffn_inp")
				{
					return "attention_residual";
				}
				if (base == "ffn_norm")
				{
					return "ffn_norm";
				}
				if (base == "ffn_gate")
				{
					return "ffn_gate";
				}
				if (base == "ffn_up")
				{
					return "ffn_up";
				}
				if (base == "ffn_swiglu")
				{
					return "ffn_swiglu";
				}
				if (base == "ffn_out")
				{
					return "ffn_down";
				}
				if (base == "l_out")
				{
					return "post_ffn";
				}
				return std::nullopt;
			}

			bool Wants(const ggml_tensor* tensor) const
			{
				const auto named = ParseNamedLayerTensor(tensor);
				if (named.base == "embd")
				{
					return selectedBlocks_.contains(0);
				}
				if (!named.layer)
				{
					return false;
				}
				if (named.base == "l_out")
				{
					return selectedBlocks_.contains(*named.layer) || selectedBlocks_.contains(*named.layer + 1);
				}
				return selectedBlocks_.contains(*named.layer) && BoundaryFor(tensor).has_value();
			}

			void Store(const ggml_tensor* tensor)
			{
				const auto named = ParseNamedLayerTensor(tensor);
				auto values = ReadContiguousTensor(tensor);
				if (named.base == "embd")
				{
					inputEmbedding_ = std::move(values);
					return;
				}
				if (!named.layer)
				{
					throw std::runtime_error("llama.cpp sub-layer checkpoint tensor is missing a layer suffix");
				}
				if (named.base == "l_out")
				{
					layerOutputs_[*named.layer] = values;
				}
				if (const auto boundary = BoundaryFor(tensor); boundary && selectedBlocks_.contains(*named.layer))
				{
					values_[std::string(*boundary)][*named.layer] = std::move(values);
				}
			}

			const std::vector<float>& Require(std::string_view boundary, std::size_t block) const
			{
				const auto boundaryIt = values_.find(std::string(boundary));
				if (boundaryIt == values_.end() || !boundaryIt->second.contains(block))
				{
					throw std::runtime_error(std::format("llama.cpp did not capture {} for block {}", boundary, block));
				}
				return boundaryIt->second.at(block);
			}

			const std::vector<float>& RequireLayerOutput(std::size_t block) const
			{
				if (!layerOutputs_.contains(block))
				{
					throw std::runtime_error("llama.cpp did not capture the previous block output");
				}
				return layerOutputs_.at(block);
			}

			std::string ShapeFor(std::string_view boundary, std::size_t count) const
			{
				if (boundary == "query_pre_rope" || boundary == "query_rotated" || boundary == "attention_context")
				{
					if (count != attentionHeads_ * headWidth_)
					{
						throw std::runtime_error("invalid attention tensor width");
					}
					return std::format("{}x{}", attentionHeads_, headWidth_);
				}
				if (boundary == "key_pre_rope" || boundary == "key_rotated" || boundary == "value")
				{
					if (count != kvHeads_ * headWidth_)
					{
						throw std::runtime_error("invalid KV tensor width");
					}
					return std::format("{}x{}", kvHeads_, headWidth_);
				}
				if (boundary == "attention_norm" || boundary == "attention_output" ||
				    boundary == "attention_residual" || boundary == "ffn_norm" || boundary == "ffn_down" ||
				    boundary == "post_ffn")
				{
					if (count != hiddenWidth_)
					{
						throw std::runtime_error("invalid hidden-state tensor width");
					}
				}
				return std::format("1x{}", count);
			}

			static std::string SixDigit(std::size_t value)
			{
				auto result = std::to_string(value);
				if (result.size() < 6)
				{
					result.insert(result.begin(), 6 - result.size(), '0');
				}
				return result;
			}

			std::size_t layerCount_{};
			std::size_t hiddenWidth_{};
			std::size_t attentionHeads_{};
			std::size_t kvHeads_{};
			std::size_t headWidth_{};
			std::set<std::size_t> selectedBlocks_;
			std::map<std::string, Group> groups_;
			std::optional<std::size_t> activeGeneratedIndex_;
			std::map<std::string, std::map<std::size_t, std::vector<float>>> values_;
			std::map<std::size_t, std::vector<float>> layerOutputs_;
			std::vector<float> inputEmbedding_;
			std::string error_;
		};
	} // namespace

	struct Model::Impl
	{
		explicit Impl(const std::filesystem::path& path)
		{
			EnsureBackendInitialized();
			auto params = llama_model_default_params();
			params.n_gpu_layers = 0;
			model = llama_model_load_from_file(path.string().c_str(), params);
			if (model == nullptr)
			{
				throw std::runtime_error("failed to load llama.cpp model");
			}
		}

		~Impl()
		{
			if (model != nullptr)
			{
				llama_model_free(model);
			}
		}

		llama_model* model{};
	};

	Model::Model(const std::filesystem::path& path) : impl_(std::make_unique<Impl>(path))
	{
	}

	Model::~Model() = default;
	Model::Model(Model&&) noexcept = default;
	Model& Model::operator=(Model&&) noexcept = default;

	TokenizationResult Model::Tokenize(std::string_view text) const
	{
		if (text.size() > static_cast<std::size_t>(std::numeric_limits<std::int32_t>::max()))
		{
			throw std::runtime_error("text is too large for llama.cpp tokenization");
		}
		const auto* vocabulary = llama_model_get_vocab(impl_->model);
		const auto textLength = static_cast<std::int32_t>(text.size());
		const bool addSpecial = llama_vocab_get_add_bos(vocabulary);
		const auto required = llama_tokenize(vocabulary, text.data(), textLength, nullptr, 0, addSpecial, true);
		if (required == std::numeric_limits<std::int32_t>::min())
		{
			throw std::runtime_error("llama.cpp tokenization size overflow");
		}
		std::vector<llama_token> tokens(static_cast<std::size_t>(required < 0 ? -required : required));
		const auto count = llama_tokenize(vocabulary, text.data(), textLength, tokens.data(),
		                                  static_cast<std::int32_t>(tokens.size()), addSpecial, true);
		if (count < 0)
		{
			throw std::runtime_error("llama.cpp tokenization failed");
		}
		tokens.resize(static_cast<std::size_t>(count));
		return { .tokenIds = FromLlamaTokens(tokens), .addBos = addSpecial, .parseSpecial = true };
	}

	std::string Model::Detokenize(std::span<const std::int32_t> tokenIds) const
	{
		const auto tokens = ToLlamaTokens(tokenIds);
		const auto tokenCount = static_cast<std::int32_t>(tokens.size());
		const auto* vocabulary = llama_model_get_vocab(impl_->model);
		const auto required = llama_detokenize(vocabulary, tokens.data(), tokenCount, nullptr, 0, false, true);
		if (required == std::numeric_limits<std::int32_t>::min())
		{
			throw std::runtime_error("llama.cpp detokenization size overflow");
		}
		std::string text(static_cast<std::size_t>(required < 0 ? -required : required), '\0');
		const auto count = llama_detokenize(vocabulary, tokens.data(), tokenCount, text.data(),
		                                    static_cast<std::int32_t>(text.size()), false, true);
		if (count < 0)
		{
			throw std::runtime_error("llama.cpp detokenization failed");
		}
		text.resize(static_cast<std::size_t>(count));
		return text;
	}

	std::string Model::ApplyChatTemplate(std::string_view userText) const
	{
		const auto* chatTemplate = llama_model_chat_template(impl_->model, nullptr);
		if (chatTemplate == nullptr)
		{
			throw std::runtime_error("GGUF model does not define a supported chat template");
		}
		const std::string content(userText);
		const llama_chat_message message{ "user", content.c_str() };
		const auto required = llama_chat_apply_template(chatTemplate, &message, 1, true, nullptr, 0);
		if (required < 0)
		{
			throw std::runtime_error("llama.cpp failed to size the chat template output");
		}
		std::string prompt(static_cast<std::size_t>(required), '\0');
		const auto count = llama_chat_apply_template(chatTemplate, &message, 1, true, prompt.data(), required);
		if (count < 0 || count > required)
		{
			throw std::runtime_error("llama.cpp failed to apply the chat template");
		}
		prompt.resize(static_cast<std::size_t>(count));
		return prompt;
	}

	NaturalGenerationResult Model::CaptureGreedyGeneration(std::span<const std::int32_t> promptTokenIds,
	                                                       std::size_t maximumGeneratedTokens,
	                                                       const std::filesystem::path& logitsOutputDirectory) const
	{
		if (promptTokenIds.empty() || maximumGeneratedTokens == 0)
		{
			throw std::runtime_error("natural generation requires a non-empty prompt and positive token count");
		}
		if (promptTokenIds.size() + maximumGeneratedTokens > std::numeric_limits<std::uint32_t>::max())
		{
			throw std::runtime_error("natural generation context exceeds llama.cpp's uint32 capacity");
		}
		std::filesystem::create_directories(logitsOutputDirectory);
		for (const auto& entry : std::filesystem::directory_iterator(logitsOutputDirectory))
		{
			if (entry.is_regular_file() && entry.path().filename().string().starts_with("decision-step-") &&
			    entry.path().extension() == ".txt")
			{
				std::filesystem::remove(entry.path());
			}
		}

		auto contextParams = llama_context_default_params();
		contextParams.n_ctx = static_cast<std::uint32_t>(promptTokenIds.size() + maximumGeneratedTokens);
		contextParams.n_batch = static_cast<std::uint32_t>(promptTokenIds.size());
		contextParams.no_perf = true;
		auto* rawContext = llama_init_from_model(impl_->model, contextParams);
		if (rawContext == nullptr)
		{
			throw std::runtime_error("failed to create llama.cpp natural-generation context");
		}
		const std::unique_ptr<llama_context, decltype(&llama_free)> context(rawContext, llama_free);
		auto prompt = ToLlamaTokens(promptTokenIds);
		if (llama_decode(context.get(), llama_batch_get_one(prompt.data(), static_cast<std::int32_t>(prompt.size()))) !=
		    0)
		{
			throw std::runtime_error("llama.cpp natural-generation prompt decode failed");
		}

		const auto* vocabulary = llama_model_get_vocab(impl_->model);
		const auto vocabularySize = llama_vocab_n_tokens(vocabulary);
		NaturalGenerationResult result{ .requestedTokenCount = maximumGeneratedTokens };
		result.generatedTokenIds.reserve(maximumGeneratedTokens);
		for (std::size_t decisionStep = 0; decisionStep < maximumGeneratedTokens; ++decisionStep)
		{
			const auto* logits = llama_get_logits_ith(context.get(), -1);
			WriteLogits(logits, vocabularySize,
			            logitsOutputDirectory / std::format("decision-step-{:06}.txt", decisionStep));
			std::int32_t selectedToken = 0;
			float selectedLogit = -std::numeric_limits<float>::infinity();
			for (std::int32_t token = 0; token < vocabularySize; ++token)
			{
				if (!std::isfinite(logits[token]))
				{
					throw std::runtime_error("llama.cpp natural generation produced a non-finite logit");
				}
				if (logits[token] > selectedLogit)
				{
					selectedToken = token;
					selectedLogit = logits[token];
				}
			}
			result.generatedTokenIds.push_back(selectedToken);
			if (llama_vocab_is_eog(vocabulary, selectedToken))
			{
				result.stoppedOnEos = true;
				break;
			}
			if (decisionStep + 1 < maximumGeneratedTokens)
			{
				auto token = static_cast<llama_token>(selectedToken);
				if (llama_decode(context.get(), llama_batch_get_one(&token, 1)) != 0)
				{
					throw std::runtime_error("llama.cpp natural generation failed at decision step " +
					                         std::to_string(decisionStep + 1));
				}
			}
		}
		return result;
	}

	void Model::CaptureTeacherForcedLogits(std::span<const std::int32_t> promptTokenIds,
	                                       std::span<const std::int32_t> targetTokenIds,
	                                       const std::filesystem::path& logitsOutputDirectory) const
	{
		if (promptTokenIds.empty() || targetTokenIds.empty())
		{
			throw std::runtime_error("teacher-forced-logits requires non-empty prompt and target token ids");
		}
		if (promptTokenIds.size() + targetTokenIds.size() > std::numeric_limits<std::uint32_t>::max())
		{
			throw std::runtime_error("teacher-forced-logits context exceeds llama.cpp's uint32 capacity");
		}
		std::filesystem::create_directories(logitsOutputDirectory);
		for (const auto& entry : std::filesystem::directory_iterator(logitsOutputDirectory))
		{
			if (entry.is_regular_file() && entry.path().filename().string().starts_with("decision-step-") &&
			    entry.path().extension() == ".txt")
			{
				std::filesystem::remove(entry.path());
			}
		}

		auto contextParams = llama_context_default_params();
		contextParams.n_ctx = static_cast<std::uint32_t>(promptTokenIds.size() + targetTokenIds.size());
		contextParams.n_batch = static_cast<std::uint32_t>(promptTokenIds.size());
		contextParams.no_perf = true;
		auto* rawContext = llama_init_from_model(impl_->model, contextParams);
		if (rawContext == nullptr)
		{
			throw std::runtime_error("failed to create llama.cpp teacher-forced context");
		}
		const std::unique_ptr<llama_context, decltype(&llama_free)> context(rawContext, llama_free);
		auto prompt = ToLlamaTokens(promptTokenIds);
		if (llama_decode(context.get(), llama_batch_get_one(prompt.data(), static_cast<std::int32_t>(prompt.size()))) !=
		    0)
		{
			throw std::runtime_error("llama.cpp teacher-forced prompt decode failed");
		}

		const auto vocabularySize = llama_vocab_n_tokens(llama_model_get_vocab(impl_->model));
		const auto targets = ToLlamaTokens(targetTokenIds);
		for (std::size_t decisionStep = 0; decisionStep < targets.size(); ++decisionStep)
		{
			WriteLogits(llama_get_logits_ith(context.get(), -1), vocabularySize,
			            logitsOutputDirectory / std::format("decision-step-{:06}.txt", decisionStep));
			if (decisionStep + 1 < targets.size())
			{
				auto token = targets[decisionStep];
				if (llama_decode(context.get(), llama_batch_get_one(&token, 1)) != 0)
				{
					throw std::runtime_error("llama.cpp teacher-forced token decode failed at decision step " +
					                         std::to_string(decisionStep));
				}
			}
		}
	}

	void Model::BenchmarkFixedDecode(std::span<const std::int32_t> promptTokenIds,
	                                 std::span<const std::int32_t> generatedTokenIds, std::size_t warmupWindowCount,
	                                 std::size_t measuredWindowCount, std::size_t contextLength,
	                                 std::size_t threadCount, const std::filesystem::path& reportPath) const
	{
		if (promptTokenIds.empty() || generatedTokenIds.size() < 2 || measuredWindowCount == 0)
		{
			throw std::runtime_error("benchmark-fixed-decode requires a non-empty prompt, at least two generated "
			                         "tokens, and measured windows");
		}
		if (contextLength < promptTokenIds.size() + generatedTokenIds.size())
		{
			throw std::runtime_error("benchmark-fixed-decode context is shorter than prompt plus generated tokens");
		}
		if (contextLength > std::numeric_limits<std::uint32_t>::max() ||
		    threadCount > static_cast<std::size_t>(std::numeric_limits<std::int32_t>::max()))
		{
			throw std::runtime_error("benchmark-fixed-decode context or thread count exceeds llama.cpp limits");
		}

		auto contextParams = llama_context_default_params();
		contextParams.n_ctx = static_cast<std::uint32_t>(contextLength);
		contextParams.n_batch = 2048;
		contextParams.n_ubatch = 512;
		contextParams.n_threads = static_cast<std::int32_t>(threadCount);
		contextParams.n_threads_batch = static_cast<std::int32_t>(threadCount);
		contextParams.flash_attn_type = LLAMA_FLASH_ATTN_TYPE_DISABLED;
		contextParams.no_perf = true;
		auto* rawContext = llama_init_from_model(impl_->model, contextParams);
		if (rawContext == nullptr)
		{
			throw std::runtime_error("failed to create llama.cpp fixed-decode benchmark context");
		}
		const std::unique_ptr<llama_context, decltype(&llama_free)> context(rawContext, llama_free);
		auto prompt = ToLlamaTokens(promptTokenIds);
		const auto generated = ToLlamaTokens(generatedTokenIds);

		struct Window
		{
			bool warmup{};
			std::size_t index{};
			double stateResetMs{};
			double prefillMs{};
			double decodeWallMs{};
			std::size_t decodeTokens{};
		};
		std::vector<Window> windows;
		windows.reserve(warmupWindowCount + measuredWindowCount);
		for (std::size_t windowIndex = 0; windowIndex < warmupWindowCount + measuredWindowCount; ++windowIndex)
		{
			const auto resetStart = std::chrono::steady_clock::now();
			llama_memory_clear(llama_get_memory(context.get()), false);
			const auto resetEnd = std::chrono::steady_clock::now();

			const auto prefillStart = std::chrono::steady_clock::now();
			if (llama_decode(context.get(),
			                 llama_batch_get_one(prompt.data(), static_cast<std::int32_t>(prompt.size()))) != 0)
			{
				throw std::runtime_error("llama.cpp fixed-decode benchmark prompt decode failed");
			}
			const auto prefillEnd = std::chrono::steady_clock::now();

			const auto decodeStart = std::chrono::steady_clock::now();
			for (std::size_t tokenIndex = 0; tokenIndex + 1 < generated.size(); ++tokenIndex)
			{
				auto token = generated[tokenIndex];
				if (llama_decode(context.get(), llama_batch_get_one(&token, 1)) != 0)
				{
					throw std::runtime_error("llama.cpp fixed-decode benchmark failed at token " +
					                         std::to_string(tokenIndex));
				}
			}
			const auto decodeEnd = std::chrono::steady_clock::now();
			windows.push_back(
			    { .warmup = windowIndex < warmupWindowCount,
			      .index = windowIndex < warmupWindowCount ? windowIndex : windowIndex - warmupWindowCount,
			      .stateResetMs = std::chrono::duration<double, std::milli>(resetEnd - resetStart).count(),
			      .prefillMs = std::chrono::duration<double, std::milli>(prefillEnd - prefillStart).count(),
			      .decodeWallMs = std::chrono::duration<double, std::milli>(decodeEnd - decodeStart).count(),
			      .decodeTokens = generated.size() - 1 });
		}

		std::vector<double> measuredThroughputs;
		std::vector<double> measuredLatencies;
		for (const auto& window : windows)
		{
			if (!window.warmup)
			{
				measuredLatencies.push_back(window.decodeWallMs / static_cast<double>(window.decodeTokens));
				measuredThroughputs.push_back(static_cast<double>(window.decodeTokens) * 1000.0 / window.decodeWallMs);
			}
		}
		const auto median = [](std::vector<double> values) {
			std::ranges::sort(values);
			const auto middle = values.size() / 2;
			return values.size() % 2 == 0 ? (values[middle - 1] + values[middle]) / 2.0 : values[middle];
		};
		const auto meanThroughput = std::reduce(measuredThroughputs.begin(), measuredThroughputs.end()) /
		                            static_cast<double>(measuredThroughputs.size());
		double throughputSquaredDeviation = 0.0;
		for (const auto throughput : measuredThroughputs)
		{
			throughputSquaredDeviation += (throughput - meanThroughput) * (throughput - meanThroughput);
		}
		const auto throughputStandardDeviation =
		    measuredThroughputs.size() < 2
		        ? 0.0
		        : std::sqrt(throughputSquaredDeviation / static_cast<double>(measuredThroughputs.size() - 1));

		if (!reportPath.parent_path().empty())
		{
			std::filesystem::create_directories(reportPath.parent_path());
		}
		std::ofstream output(reportPath, std::ios::binary | std::ios::trunc);
		if (!output)
		{
			throw std::runtime_error("failed to open llama.cpp fixed-decode benchmark report: " + reportPath.string());
		}
		output << std::setprecision(17) << "{\n  \"schema\": \"litenn.in_process_decode_windows.v1\",\n"
		       << "  \"producer\": \"llama.cpp\",\n  \"runtime\": \"cpu\",\n"
		       << "  \"stateReset\": \"llama_memory_clear_metadata\",\n"
		       << "  \"modelMappedOnce\": true,\n  \"contextCreatedOnce\": true,\n"
		       << "  \"warmupWindows\": " << warmupWindowCount << ",\n  \"measuredWindows\": " << measuredWindowCount
		       << ",\n  \"promptTokens\": " << promptTokenIds.size()
		       << ",\n  \"decodeTokensPerWindow\": " << generatedTokenIds.size() - 1
		       << ",\n  \"threadCount\": " << threadCount << ",\n  \"requestedContextLength\": " << contextLength
		       << ",\n  \"contextLength\": " << llama_n_ctx(context.get()) << ",\n  \"windows\": [\n";
		for (std::size_t index = 0; index < windows.size(); ++index)
		{
			const auto& window = windows[index];
			const auto millisecondsPerToken = window.decodeWallMs / static_cast<double>(window.decodeTokens);
			const auto tokensPerSecond = static_cast<double>(window.decodeTokens) * 1000.0 / window.decodeWallMs;
			output << "    {\"phase\": \"" << (window.warmup ? "warmup" : "measured")
			       << "\", \"index\": " << window.index << ", \"stateResetMs\": " << window.stateResetMs
			       << ", \"prefillMs\": " << window.prefillMs << ", \"decodeWallMs\": " << window.decodeWallMs
			       << ", \"moduleRunMs\": " << window.decodeWallMs << ", \"decodeTokens\": " << window.decodeTokens
			       << ", \"msPerToken\": " << millisecondsPerToken << ", \"tokensPerSecond\": " << tokensPerSecond
			       << '}';
			if (index + 1 != windows.size())
			{
				output << ',';
			}
			output << '\n';
		}
		output << "  ],\n  \"summary\": {\"tokensPerSecondMean\": " << meanThroughput
		       << ", \"tokensPerSecondMedian\": " << median(measuredThroughputs)
		       << ", \"tokensPerSecondStandardDeviation\": " << throughputStandardDeviation
		       << ", \"tokensPerSecondCVPercent\": "
		       << (meanThroughput == 0.0 ? 0.0 : throughputStandardDeviation * 100.0 / meanThroughput)
		       << ", \"msPerTokenMedian\": " << median(measuredLatencies) << "}\n}\n";
		if (!output)
		{
			throw std::runtime_error("failed to write llama.cpp fixed-decode benchmark report: " + reportPath.string());
		}
	}

	void Model::CaptureDecodeLogits(std::span<const std::int32_t> promptTokenIds,
	                                std::span<const std::int32_t> generatedTokenIds,
	                                const std::filesystem::path& outputDirectory) const
	{
		if (promptTokenIds.empty() || generatedTokenIds.empty())
		{
			throw std::runtime_error("decode-logits requires non-empty prompt and generated token ids");
		}
		std::filesystem::create_directories(outputDirectory);
		auto contextParams = llama_context_default_params();
		contextParams.n_ctx = static_cast<std::uint32_t>(promptTokenIds.size() + generatedTokenIds.size());
		contextParams.n_batch = static_cast<std::uint32_t>(promptTokenIds.size());
		contextParams.no_perf = true;
		auto* rawContext = llama_init_from_model(impl_->model, contextParams);
		if (rawContext == nullptr)
		{
			throw std::runtime_error("failed to create llama.cpp context");
		}
		const std::unique_ptr<llama_context, decltype(&llama_free)> context(rawContext, llama_free);
		auto prompt = ToLlamaTokens(promptTokenIds);
		if (llama_decode(context.get(), llama_batch_get_one(prompt.data(), static_cast<std::int32_t>(prompt.size()))) !=
		    0)
		{
			throw std::runtime_error("llama.cpp prompt decode failed");
		}
		const auto vocabularySize = llama_vocab_n_tokens(llama_model_get_vocab(impl_->model));
		const auto generated = ToLlamaTokens(generatedTokenIds);
		for (std::size_t step = 0; step < generated.size(); ++step)
		{
			auto token = generated[step];
			if (llama_decode(context.get(), llama_batch_get_one(&token, 1)) != 0)
			{
				throw std::runtime_error("llama.cpp token decode failed at step " + std::to_string(step + 1));
			}
			WriteLogits(llama_get_logits_ith(context.get(), -1), vocabularySize,
			            outputDirectory / ("decode-step-" + std::to_string(step + 1) + ".txt"));
		}
	}

	void Model::CaptureDecodeLayerCheckpoints(std::span<const std::int32_t> promptTokenIds,
	                                          std::span<const std::int32_t> generatedTokenIds,
	                                          std::span<const std::size_t> generatedIndices,
	                                          const std::filesystem::path& outputDirectory) const
	{
		if (promptTokenIds.empty() || generatedIndices.empty())
		{
			throw std::runtime_error("decode-layer-checkpoints requires a non-empty prompt and generated indices");
		}
		std::set<std::size_t> selected(generatedIndices.begin(), generatedIndices.end());
		if (selected.size() != generatedIndices.size())
		{
			throw std::runtime_error("decode-layer-checkpoints generated indices must be unique");
		}
		const auto maximumIndex = *selected.rbegin();
		if (maximumIndex > generatedTokenIds.size())
		{
			throw std::runtime_error(
			    "decode-layer-checkpoints needs generated token ids through selected index minus one");
		}

		LayerCheckpointCapture capture(static_cast<std::size_t>(llama_model_n_layer(impl_->model)),
		                               static_cast<std::size_t>(llama_model_n_embd(impl_->model)), outputDirectory);
		auto contextParams = llama_context_default_params();
		contextParams.n_ctx = static_cast<std::uint32_t>(promptTokenIds.size() + maximumIndex);
		contextParams.n_batch = 1;
		contextParams.flash_attn_type = LLAMA_FLASH_ATTN_TYPE_DISABLED;
		contextParams.no_perf = true;
		contextParams.cb_eval = &LayerCheckpointCapture::Callback;
		contextParams.cb_eval_user_data = &capture;
		auto* rawContext = llama_init_from_model(impl_->model, contextParams);
		if (rawContext == nullptr)
		{
			throw std::runtime_error("failed to create llama.cpp checkpoint context");
		}
		const std::unique_ptr<llama_context, decltype(&llama_free)> context(rawContext, llama_free);
		auto prompt = ToLlamaTokens(promptTokenIds);
		for (std::size_t promptIndex = 0; promptIndex < prompt.size(); ++promptIndex)
		{
			const auto finalPromptToken = promptIndex + 1 == prompt.size();
			if (finalPromptToken && selected.contains(0))
			{
				capture.Begin(0);
			}
			auto token = prompt[promptIndex];
			if (llama_decode(context.get(), llama_batch_get_one(&token, 1)) != 0)
			{
				throw std::runtime_error("llama.cpp prompt checkpoint decode failed at prompt index " +
				                         std::to_string(promptIndex));
			}
			if (finalPromptToken && selected.contains(0))
			{
				capture.End(promptTokenIds.size(), promptTokenIds.size() - 1, promptTokenIds.back());
			}
		}

		const auto generated = ToLlamaTokens(generatedTokenIds);
		for (std::size_t generatedIndex = 1; generatedIndex <= maximumIndex; ++generatedIndex)
		{
			if (selected.contains(generatedIndex))
			{
				capture.Begin(generatedIndex);
			}
			auto token = generated[generatedIndex - 1];
			if (llama_decode(context.get(), llama_batch_get_one(&token, 1)) != 0)
			{
				throw std::runtime_error("llama.cpp layer checkpoint decode failed at generated index " +
				                         std::to_string(generatedIndex));
			}
			if (selected.contains(generatedIndex))
			{
				capture.End(promptTokenIds.size() + generatedIndex, promptTokenIds.size() + generatedIndex - 1,
				            generatedTokenIds[generatedIndex - 1]);
			}
		}
	}

	void Model::CaptureDecodeSubLayerCheckpoints(std::span<const std::int32_t> promptTokenIds,
	                                             std::span<const std::int32_t> generatedTokenIds,
	                                             std::span<const std::size_t> generatedIndices,
	                                             std::span<const std::size_t> blockIndices,
	                                             const std::filesystem::path& outputDirectory,
	                                             const std::filesystem::path& logitsOutputDirectory) const
	{
		if (promptTokenIds.empty() || generatedIndices.empty() || blockIndices.empty())
		{
			throw std::runtime_error(
			    "decode-sub-layer-checkpoints requires a non-empty prompt, generated indices, and block indices");
		}
		const std::set<std::size_t> selectedGenerated(generatedIndices.begin(), generatedIndices.end());
		if (selectedGenerated.size() != generatedIndices.size())
		{
			throw std::runtime_error("decode-sub-layer-checkpoints generated indices must be unique");
		}
		const auto maximumIndex = *selectedGenerated.rbegin();
		if (maximumIndex > generatedTokenIds.size())
		{
			throw std::runtime_error(
			    "decode-sub-layer-checkpoints needs generated token ids through selected index minus one");
		}

		SubLayerCheckpointCapture capture(static_cast<std::size_t>(llama_model_n_layer(impl_->model)),
		                                  static_cast<std::size_t>(llama_model_n_embd(impl_->model)),
		                                  static_cast<std::size_t>(llama_model_n_head(impl_->model)),
		                                  static_cast<std::size_t>(llama_model_n_head_kv(impl_->model)), blockIndices,
		                                  outputDirectory);
		if (!logitsOutputDirectory.empty())
		{
			std::filesystem::create_directories(logitsOutputDirectory);
		}
		const auto vocabularySize = llama_vocab_n_tokens(llama_model_get_vocab(impl_->model));
		auto contextParams = llama_context_default_params();
		contextParams.n_ctx = static_cast<std::uint32_t>(promptTokenIds.size() + maximumIndex);
		contextParams.n_batch = 1;
		contextParams.flash_attn_type = LLAMA_FLASH_ATTN_TYPE_DISABLED;
		contextParams.no_perf = true;
		contextParams.cb_eval = &SubLayerCheckpointCapture::Callback;
		contextParams.cb_eval_user_data = &capture;
		auto* rawContext = llama_init_from_model(impl_->model, contextParams);
		if (rawContext == nullptr)
		{
			throw std::runtime_error("failed to create llama.cpp sub-layer checkpoint context");
		}
		const std::unique_ptr<llama_context, decltype(&llama_free)> context(rawContext, llama_free);
		auto prompt = ToLlamaTokens(promptTokenIds);
		for (std::size_t promptIndex = 0; promptIndex < prompt.size(); ++promptIndex)
		{
			const auto finalPromptToken = promptIndex + 1 == prompt.size();
			if (finalPromptToken && selectedGenerated.contains(0))
			{
				capture.Begin(0);
			}
			auto token = prompt[promptIndex];
			if (llama_decode(context.get(), llama_batch_get_one(&token, 1)) != 0)
			{
				throw std::runtime_error("llama.cpp prompt sub-layer checkpoint decode failed at prompt index " +
				                         std::to_string(promptIndex));
			}
			if (finalPromptToken && selectedGenerated.contains(0))
			{
				capture.End(promptTokenIds.size(), promptTokenIds.size() - 1, promptTokenIds.back());
				if (!logitsOutputDirectory.empty())
				{
					WriteLogits(llama_get_logits_ith(context.get(), -1), vocabularySize,
					            logitsOutputDirectory / "generated-000000.txt");
				}
			}
		}

		const auto generated = ToLlamaTokens(generatedTokenIds);
		for (std::size_t generatedIndex = 1; generatedIndex <= maximumIndex; ++generatedIndex)
		{
			if (selectedGenerated.contains(generatedIndex))
			{
				capture.Begin(generatedIndex);
			}
			auto token = generated[generatedIndex - 1];
			if (llama_decode(context.get(), llama_batch_get_one(&token, 1)) != 0)
			{
				throw std::runtime_error("llama.cpp sub-layer checkpoint decode failed at generated index " +
				                         std::to_string(generatedIndex));
			}
			if (selectedGenerated.contains(generatedIndex))
			{
				capture.End(promptTokenIds.size() + generatedIndex, promptTokenIds.size() + generatedIndex - 1,
				            generatedTokenIds[generatedIndex - 1]);
				if (!logitsOutputDirectory.empty())
				{
					WriteLogits(llama_get_logits_ith(context.get(), -1), vocabularySize,
					            logitsOutputDirectory / std::format("generated-{:06}.txt", generatedIndex));
				}
			}
		}
	}

	void WriteTokensJson(const TokenizationResult& result, const std::filesystem::path& path)
	{
		std::ofstream output(path);
		if (!output)
		{
			throw std::runtime_error("failed to open token output: " + path.string());
		}
		output << "{\n  \"schema\": \"litenn.llamacpp_tokens.v1\",\n"
		       << "  \"addBos\": " << (result.addBos ? "true" : "false") << ",\n"
		       << "  \"parseSpecial\": " << (result.parseSpecial ? "true" : "false") << ",\n  \"tokenIds\": [";
		for (std::size_t i = 0; i < result.tokenIds.size(); ++i)
		{
			if (i != 0)
			{
				output << ", ";
			}
			output << result.tokenIds[i];
		}
		output << "]\n}\n";
	}

	void WriteNaturalGenerationManifest(std::span<const std::int32_t> promptTokenIds,
	                                    const NaturalGenerationResult& result,
	                                    const std::filesystem::path& outputDirectory)
	{
		std::filesystem::create_directories(outputDirectory);
		std::ofstream output(outputDirectory / "manifest.json");
		if (!output)
		{
			throw std::runtime_error("failed to open natural-generation manifest");
		}
		const auto writeTokens = [&output](std::span<const std::int32_t> tokens) {
			output << '[';
			for (std::size_t index = 0; index < tokens.size(); ++index)
			{
				if (index != 0)
				{
					output << ", ";
				}
				output << tokens[index];
			}
			output << ']';
		};
		output << "{\n  \"schema\": \"litenn.natural_generation.v1\",\n"
		       << "  \"producer\": \"llama.cpp\",\n  \"sampling\": \"greedy\",\n  \"promptTokenIds\": ";
		writeTokens(promptTokenIds);
		output << ",\n  \"generatedTokenIds\": ";
		writeTokens(result.generatedTokenIds);
		output << ",\n  \"requestedTokenCount\": " << result.requestedTokenCount
		       << ",\n  \"stoppedOnEos\": " << (result.stoppedOnEos ? "true" : "false")
		       << ",\n  \"fallbackUsed\": false,\n  \"logitsArtifacts\": [\n";
		for (std::size_t step = 0; step < result.generatedTokenIds.size(); ++step)
		{
			output << "    {\"decisionStep\": " << step << ", \"position\": " << promptTokenIds.size() + step
			       << ", \"path\": \"logits/decision-step-" << std::setw(6) << std::setfill('0') << step << ".txt\"}";
			if (step + 1 != result.generatedTokenIds.size())
			{
				output << ',';
			}
			output << '\n';
		}
		output << "  ]\n}\n";
	}

	void WriteTeacherForcedManifest(std::span<const std::int32_t> promptTokenIds,
	                                std::span<const std::int32_t> targetTokenIds,
	                                const std::filesystem::path& outputDirectory)
	{
		std::filesystem::create_directories(outputDirectory);
		std::ofstream output(outputDirectory / "manifest.json");
		if (!output)
		{
			throw std::runtime_error("failed to open teacher-forced manifest");
		}
		const auto writeTokens = [&output](std::span<const std::int32_t> tokens) {
			output << '[';
			for (std::size_t index = 0; index < tokens.size(); ++index)
			{
				if (index != 0)
				{
					output << ", ";
				}
				output << tokens[index];
			}
			output << ']';
		};
		output << "{\n  \"schema\": \"litenn.teacher_forced_logits.v1\",\n"
		       << "  \"producer\": \"llama.cpp\",\n  \"captureBoundary\": \"pre-target\",\n"
		       << "  \"promptTokenIds\": ";
		writeTokens(promptTokenIds);
		output << ",\n  \"targetTokenIds\": ";
		writeTokens(targetTokenIds);
		output << ",\n  \"fallbackUsed\": false,\n  \"logitsArtifacts\": [\n";
		for (std::size_t step = 0; step < targetTokenIds.size(); ++step)
		{
			output << "    {\"decisionStep\": " << step << ", \"position\": " << promptTokenIds.size() + step
			       << ", \"targetTokenId\": " << targetTokenIds[step] << ", \"path\": \"logits/decision-step-"
			       << std::setw(6) << std::setfill('0') << step << ".txt\"}";
			if (step + 1 != targetTokenIds.size())
			{
				output << ',';
			}
			output << '\n';
		}
		output << "  ]\n}\n";
	}

	std::vector<std::int32_t> ParseCommaTokenIds(std::string_view text, std::string_view label)
	{
		std::vector<std::int32_t> result;
		while (!text.empty())
		{
			const auto separator = text.find(',');
			const auto part = text.substr(0, separator);
			std::int32_t value{};
			const auto parsed = std::from_chars(part.data(), part.data() + part.size(), value);
			if (part.empty() || parsed.ec != std::errc{} || parsed.ptr != part.data() + part.size() || value < 0)
			{
				throw std::runtime_error(std::string(label) + " must contain comma-separated non-negative integers");
			}
			result.push_back(value);
			if (separator == std::string_view::npos)
			{
				break;
			}
			text.remove_prefix(separator + 1);
		}
		if (result.empty())
		{
			throw std::runtime_error(std::string(label) + " must not be empty");
		}
		return result;
	}
} // namespace LiteNN::LlamaCppAdapter
