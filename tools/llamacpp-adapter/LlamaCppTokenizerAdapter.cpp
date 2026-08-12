#include "LlamaCppTokenizerAdapter.h"

#include <ggml-backend.h>
#include <ggml.h>
#include <llama.h>

#include <algorithm>
#include <charconv>
#include <cmath>
#include <cstddef>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <limits>
#include <map>
#include <memory>
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
