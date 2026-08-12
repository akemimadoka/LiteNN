#include "DownProjectionVerifier.h"

#include "GGMLQuantizedKernels.h"
#include "GGUFImporter.h"

#include <LiteNN/Compiler/CompiledModule.h>

#include <algorithm>
#include <array>
#include <bit>
#include <charconv>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <format>
#include <fstream>
#include <iomanip>
#include <limits>
#include <map>
#include <numeric>
#include <optional>
#include <span>
#include <sstream>
#include <stdexcept>
#include <string_view>
#include <utility>

#ifdef LITENN_GGUF_CONVERT_ENABLE_AOT
extern "C" void litenn_cpu_ggml_block_matmul_f32(const float*, const float*, std::int64_t, std::int64_t, std::int64_t,
                                                 std::int64_t, std::int64_t, const std::uint8_t*, const std::uint8_t*,
                                                 std::int64_t, std::int64_t, std::int64_t, float*, float*, std::int64_t,
                                                 std::int64_t, std::int64_t, std::int64_t, std::int64_t, std::uint64_t,
                                                 std::uint64_t, std::uint64_t);
extern "C" std::uint64_t litenn_cpu_ggml_field_interleaved_v4_bytes(std::uint64_t, std::int64_t, std::int64_t);
extern "C" void litenn_cpu_ggml_prepack_field_interleaved_v4(const std::uint8_t*, const std::uint8_t*, std::int64_t,
                                                             std::int64_t, std::int64_t, std::int64_t, std::int64_t,
                                                             std::uint64_t, std::uint8_t*, std::uint8_t*, std::int64_t,
                                                             std::int64_t, std::int64_t);
extern "C" void litenn_cpu_ggml_block_matmul_field_interleaved_v4_q8k_f32(
    const float*, const float*, std::int64_t, std::int64_t, std::int64_t, std::int64_t, std::int64_t,
    const std::uint8_t*, const std::uint8_t*, std::int64_t, std::int64_t, std::int64_t, float*, float*, std::int64_t,
    std::int64_t, std::int64_t, std::int64_t, std::int64_t, std::uint64_t, std::uint64_t, std::uint64_t);
extern "C" void litenn_cpu_ggml_block_grouped_matmul2_field_interleaved_v4_q8k_f32(
    const float*, const float*, std::int64_t, std::int64_t, std::int64_t, std::int64_t, std::int64_t,
    const std::uint8_t*, const std::uint8_t*, std::int64_t, std::int64_t, std::int64_t, const std::uint8_t*,
    const std::uint8_t*, std::int64_t, std::int64_t, std::int64_t, float*, float*, std::int64_t, std::int64_t,
    std::int64_t, std::int64_t, std::int64_t, std::uint64_t, std::uint64_t, std::uint64_t, std::uint64_t,
    std::uint64_t);
extern "C" void litenn_cpu_swiglu_f32(const float*, const float*, std::int64_t, std::int64_t, std::int64_t,
                                      std::int64_t, std::int64_t, const float*, const float*, std::int64_t,
                                      std::int64_t, std::int64_t, std::int64_t, std::int64_t, float*, float*,
                                      std::int64_t, std::int64_t, std::int64_t, std::int64_t, std::int64_t);
#endif

namespace LiteNN::GGUF::Tooling
{
	namespace
	{
		struct ErrorMetrics
		{
			double maximumAbsolute{};
			double meanAbsolute{};
			double rmsError{};
			double referenceRms{};
			double normalizedRmsError{};
			double cosineSimilarity{};
		};

		struct CandidateMetrics
		{
			std::string name;
			ErrorMetrics versusExact;
			ErrorMetrics versusCaptured;
		};

		struct BlockResult
		{
			std::size_t blockIndex{};
			std::string weightName;
			std::size_t inputFeatures{};
			std::size_t outputFeatures{};
			std::size_t storedBytes{};
			std::size_t packedBytes{};
			std::string closestToExact;
			std::vector<CandidateMetrics> candidates;
		};

		struct FFNCandidateMetrics
		{
			std::string name;
			ErrorMetrics gateVersusExact;
			ErrorMetrics gateVersusCaptured;
			ErrorMetrics upVersusExact;
			ErrorMetrics upVersusCaptured;
			ErrorMetrics swigluVersusExact;
			ErrorMetrics swigluVersusCaptured;
		};

		struct FFNBlockResult
		{
			std::size_t blockIndex{};
			std::size_t inputFeatures{};
			std::size_t hiddenFeatures{};
			std::vector<FFNCandidateMetrics> candidates;
		};

		std::vector<std::string_view> Split(std::string_view text, char delimiter)
		{
			std::vector<std::string_view> fields;
			while (true)
			{
				const auto position = text.find(delimiter);
				fields.push_back(text.substr(0, position));
				if (position == std::string_view::npos)
				{
					break;
				}
				text.remove_prefix(position + 1);
			}
			return fields;
		}

		template <typename T>
		T ParseInteger(std::string_view text, std::string_view field)
		{
			T value{};
			const auto result = std::from_chars(text.data(), text.data() + text.size(), value);
			if (result.ec != std::errc{} || result.ptr != text.data() + text.size())
			{
				throw std::runtime_error(std::format("invalid {} in checkpoint manifest", field));
			}
			return value;
		}

		std::uint64_t FNV1a(std::span<const std::byte> bytes)
		{
			std::uint64_t hash = 14695981039346656037ULL;
			for (const auto value : bytes)
			{
				hash ^= std::to_integer<std::uint8_t>(value);
				hash *= 1099511628211ULL;
			}
			return hash;
		}

		Tensor<CPU> LoadCheckpointTensor(const std::filesystem::path& root, std::string_view boundary,
		                                 std::size_t generatedIndex, std::size_t blockIndex,
		                                 std::size_t expectedElements)
		{
			const auto directory = root / boundary;
			std::ifstream manifest(directory / "manifest.tsv", std::ios::binary);
			if (!manifest)
			{
				throw std::runtime_error(std::format("failed to open {} checkpoint manifest", boundary));
			}
			std::string line;
			if (!std::getline(manifest, line) || line != "# litenn-layer-checkpoints-v1")
			{
				throw std::runtime_error(std::format("{} checkpoint manifest has an unsupported schema", boundary));
			}
			if (!std::getline(manifest, line))
			{
				throw std::runtime_error(std::format("{} checkpoint manifest has no header", boundary));
			}

			std::optional<Tensor<CPU>> result;
			while (std::getline(manifest, line))
			{
				if (!line.empty() && line.back() == '\r')
				{
					line.pop_back();
				}
				const auto fields = Split(line, '\t');
				if (fields.size() != 17)
				{
					throw std::runtime_error(
					    std::format("{} checkpoint manifest row must contain 17 fields", boundary));
				}
				if (ParseInteger<std::size_t>(fields[0], "generated_index") != generatedIndex ||
				    ParseInteger<std::size_t>(fields[5], "layer") != blockIndex)
				{
					continue;
				}
				if (result)
				{
					throw std::runtime_error(std::format("duplicate {} checkpoint for generated index {} block {}",
					                                     boundary, generatedIndex, blockIndex));
				}
				if (fields[7] != "Float32" || fields[8] != std::format("1x{}", expectedElements) ||
				    ParseInteger<std::size_t>(fields[10], "byte_size") != expectedElements * sizeof(float) ||
				    ParseInteger<std::size_t>(fields[15], "non_finite") != 0)
				{
					throw std::runtime_error(
					    std::format("{} checkpoint dtype, shape, size, or finiteness is invalid", boundary));
				}

				const std::filesystem::path fileName{ fields[4] };
				if (fileName.is_absolute() || fileName.has_parent_path())
				{
					throw std::runtime_error("checkpoint payload name must be a relative file name");
				}
				const auto byteOffset = ParseInteger<std::uint64_t>(fields[9], "byte_offset");
				const auto byteSize = ParseInteger<std::uint64_t>(fields[10], "byte_size");
				const auto payloadPath = directory / fileName;
				if (byteOffset > std::filesystem::file_size(payloadPath) ||
				    byteSize > std::filesystem::file_size(payloadPath) - byteOffset)
				{
					throw std::runtime_error("checkpoint payload range exceeds its file");
				}

				Tensor<CPU> tensor(Uninitialized, { 1, expectedElements }, DataType::Float32);
				std::ifstream payload(payloadPath, std::ios::binary);
				payload.seekg(static_cast<std::streamoff>(byteOffset));
				payload.read(static_cast<char*>(tensor.UnsafeRawData()), static_cast<std::streamsize>(byteSize));
				if (!payload)
				{
					throw std::runtime_error("failed to read checkpoint payload range");
				}
				const auto bytes = std::span{ static_cast<const std::byte*>(tensor.UnsafeRawData()),
					                          static_cast<std::size_t>(byteSize) };
				if (std::format("{:016x}", FNV1a(bytes)) != fields[16])
				{
					throw std::runtime_error("checkpoint payload checksum does not match manifest");
				}
				result.emplace(std::move(tensor));
			}
			if (!result)
			{
				throw std::runtime_error(std::format("missing {} checkpoint for generated index {} block {}", boundary,
				                                     generatedIndex, blockIndex));
			}
			return std::move(*result);
		}

		ErrorMetrics Compare(std::span<const float> candidate, std::span<const float> reference)
		{
			if (candidate.size() != reference.size() || candidate.empty())
			{
				throw std::runtime_error("projection comparison tensors must have the same nonzero size");
			}
			double absoluteSum = 0.0;
			double squaredError = 0.0;
			double referenceSquared = 0.0;
			double candidateSquared = 0.0;
			double dot = 0.0;
			double maximumAbsolute = 0.0;
			for (std::size_t i = 0; i < candidate.size(); ++i)
			{
				if (!std::isfinite(candidate[i]) || !std::isfinite(reference[i]))
				{
					throw std::runtime_error("projection comparison contains a non-finite value");
				}
				const auto difference = static_cast<double>(candidate[i]) - static_cast<double>(reference[i]);
				const auto absolute = std::abs(difference);
				maximumAbsolute = std::max(maximumAbsolute, absolute);
				absoluteSum += absolute;
				squaredError += difference * difference;
				referenceSquared += static_cast<double>(reference[i]) * reference[i];
				candidateSquared += static_cast<double>(candidate[i]) * candidate[i];
				dot += static_cast<double>(candidate[i]) * reference[i];
			}
			const auto count = static_cast<double>(candidate.size());
			const auto referenceRms = std::sqrt(referenceSquared / count);
			const auto rmsError = std::sqrt(squaredError / count);
			const auto magnitude = std::sqrt(candidateSquared * referenceSquared);
			return {
				.maximumAbsolute = maximumAbsolute,
				.meanAbsolute = absoluteSum / count,
				.rmsError = rmsError,
				.referenceRms = referenceRms,
				.normalizedRmsError =
				    referenceRms == 0.0 ? std::numeric_limits<double>::infinity() : rmsError / referenceRms,
				.cosineSimilarity = magnitude == 0.0 ? 1.0 : dot / magnitude,
			};
		}

		std::span<const float> Values(const Tensor<CPU>& tensor)
		{
			return { static_cast<const float*>(tensor.UnsafeRawData()), tensor.NumElements() };
		}

		Tensor<CPU> RunSourceHelper(const Tensor<CPU>& input, const Variable& weight, std::size_t threads)
		{
#ifdef LITENN_GGUF_CONVERT_ENABLE_AOT
			const auto& params = *weight.Quantization();
			const auto storage = weight.Data().CopyToDevice(CPU{});
			const auto rows = input.Shape()[0];
			const auto inputFeatures = input.Shape()[1];
			const auto outputFeatures = params.expressedShape[0];
			Tensor<CPU> result(Uninitialized, { rows, outputFeatures }, DataType::Float32);
			std::fill_n(static_cast<float*>(result.UnsafeRawData()), result.NumElements(),
			            std::numeric_limits<float>::quiet_NaN());
			litenn_cpu_ggml_block_matmul_f32(
			    nullptr, static_cast<const float*>(input.UnsafeRawData()), 0, static_cast<std::int64_t>(rows),
			    static_cast<std::int64_t>(inputFeatures), static_cast<std::int64_t>(inputFeatures), 1, nullptr,
			    static_cast<const std::uint8_t*>(storage.UnsafeRawData()), 0,
			    static_cast<std::int64_t>(storage.NumElements()), 1, nullptr,
			    static_cast<float*>(result.UnsafeRawData()), 0, static_cast<std::int64_t>(rows),
			    static_cast<std::int64_t>(outputFeatures), static_cast<std::int64_t>(outputFeatures), 1,
			    static_cast<std::uint64_t>(params.blockFormat), threads,
			    static_cast<std::uint64_t>(CPUAOTAffinityPolicy::None));
			return result;
#else
			(void) input;
			(void) weight;
			(void) threads;
			throw std::runtime_error("Down projection verification requires an AOT-enabled build");
#endif
		}

		std::pair<Tensor<CPU>, std::size_t> RunProductionHelper(const Tensor<CPU>& input, const Variable& weight,
		                                                        std::size_t threads)
		{
#ifdef LITENN_GGUF_CONVERT_ENABLE_AOT
			const auto& params = *weight.Quantization();
			const auto storage = weight.Data().CopyToDevice(CPU{});
			const auto rows = input.Shape()[0];
			const auto inputFeatures = input.Shape()[1];
			const auto outputFeatures = params.expressedShape[0];
			const auto format = static_cast<std::uint64_t>(params.blockFormat);
			const auto packedBytes = litenn_cpu_ggml_field_interleaved_v4_bytes(
			    format, static_cast<std::int64_t>(outputFeatures), static_cast<std::int64_t>(inputFeatures));
			if (packedBytes == 0 || packedBytes > static_cast<std::uint64_t>(std::numeric_limits<std::size_t>::max()))
			{
				throw std::runtime_error("field-interleaved-v4 prepack size is invalid");
			}
			std::vector<std::uint8_t> packed(static_cast<std::size_t>(packedBytes));
			litenn_cpu_ggml_prepack_field_interleaved_v4(
			    nullptr, static_cast<const std::uint8_t*>(storage.UnsafeRawData()), 0,
			    static_cast<std::int64_t>(storage.NumElements()), 1, static_cast<std::int64_t>(outputFeatures),
			    static_cast<std::int64_t>(inputFeatures), format, nullptr, packed.data(), 0,
			    static_cast<std::int64_t>(packed.size()), 1);

			Tensor<CPU> result(Uninitialized, { rows, outputFeatures }, DataType::Float32);
			std::fill_n(static_cast<float*>(result.UnsafeRawData()), result.NumElements(),
			            std::numeric_limits<float>::quiet_NaN());
			litenn_cpu_ggml_block_matmul_field_interleaved_v4_q8k_f32(
			    nullptr, static_cast<const float*>(input.UnsafeRawData()), 0, static_cast<std::int64_t>(rows),
			    static_cast<std::int64_t>(inputFeatures), static_cast<std::int64_t>(inputFeatures), 1, nullptr,
			    packed.data(), 0, static_cast<std::int64_t>(packed.size()), 1, nullptr,
			    static_cast<float*>(result.UnsafeRawData()), 0, static_cast<std::int64_t>(rows),
			    static_cast<std::int64_t>(outputFeatures), static_cast<std::int64_t>(outputFeatures), 1, format,
			    threads, static_cast<std::uint64_t>(CPUAOTAffinityPolicy::None));
			return { std::move(result), packed.size() };
#else
			(void) input;
			(void) weight;
			(void) threads;
			throw std::runtime_error("Down projection verification requires an AOT-enabled build");
#endif
		}

		std::pair<Tensor<CPU>, Tensor<CPU>> RunGroupedFFNProductionHelper(const Tensor<CPU>& input,
		                                                                  const Variable& gateWeight,
		                                                                  const Variable& upWeight, std::size_t threads)
		{
#ifdef LITENN_GGUF_CONVERT_ENABLE_AOT
			const auto& gateParams = *gateWeight.Quantization();
			const auto& upParams = *upWeight.Quantization();
			if (gateParams.blockFormat != QuantizedBlockFormat::GGML_Q4_K ||
			    upParams.blockFormat != QuantizedBlockFormat::GGML_Q4_K ||
			    gateParams.expressedShape != upParams.expressedShape || gateParams.expressedShape.size() != 2)
			{
				throw std::runtime_error("grouped FFN verification requires shape-matched Q4_K Gate/Up weights");
			}
			const auto rows = input.Shape()[0];
			const auto inputFeatures = input.Shape()[1];
			const auto hiddenFeatures = gateParams.expressedShape[0];
			const auto format = static_cast<std::uint64_t>(QuantizedBlockFormat::GGML_Q4_K);
			const auto pack = [&](const Variable& weight) {
				const auto storage = weight.Data().CopyToDevice(CPU{});
				const auto packedBytes = litenn_cpu_ggml_field_interleaved_v4_bytes(
				    format, static_cast<std::int64_t>(hiddenFeatures), static_cast<std::int64_t>(inputFeatures));
				if (packedBytes == 0 ||
				    packedBytes > static_cast<std::uint64_t>(std::numeric_limits<std::size_t>::max()))
				{
					throw std::runtime_error("grouped FFN field-v4 prepack size is invalid");
				}
				std::vector<std::uint8_t> packed(static_cast<std::size_t>(packedBytes));
				litenn_cpu_ggml_prepack_field_interleaved_v4(
				    nullptr, static_cast<const std::uint8_t*>(storage.UnsafeRawData()), 0,
				    static_cast<std::int64_t>(storage.NumElements()), 1, static_cast<std::int64_t>(hiddenFeatures),
				    static_cast<std::int64_t>(inputFeatures), format, nullptr, packed.data(), 0,
				    static_cast<std::int64_t>(packed.size()), 1);
				return packed;
			};
			auto packedGate = pack(gateWeight);
			auto packedUp = pack(upWeight);
			Tensor<CPU> grouped(Uninitialized, { rows, hiddenFeatures * 2 }, DataType::Float32);
			std::fill_n(static_cast<float*>(grouped.UnsafeRawData()), grouped.NumElements(),
			            std::numeric_limits<float>::quiet_NaN());
			litenn_cpu_ggml_block_grouped_matmul2_field_interleaved_v4_q8k_f32(
			    nullptr, static_cast<const float*>(input.UnsafeRawData()), 0, static_cast<std::int64_t>(rows),
			    static_cast<std::int64_t>(inputFeatures), static_cast<std::int64_t>(inputFeatures), 1, nullptr,
			    packedGate.data(), 0, static_cast<std::int64_t>(packedGate.size()), 1, nullptr, packedUp.data(), 0,
			    static_cast<std::int64_t>(packedUp.size()), 1, nullptr, static_cast<float*>(grouped.UnsafeRawData()), 0,
			    static_cast<std::int64_t>(rows), static_cast<std::int64_t>(hiddenFeatures * 2),
			    static_cast<std::int64_t>(hiddenFeatures * 2), 1, format, hiddenFeatures, hiddenFeatures, threads,
			    static_cast<std::uint64_t>(CPUAOTAffinityPolicy::None));

			Tensor<CPU> gate(Uninitialized, { rows, hiddenFeatures }, DataType::Float32);
			Tensor<CPU> up(Uninitialized, { rows, hiddenFeatures }, DataType::Float32);
			const auto* groupedValues = static_cast<const float*>(grouped.UnsafeRawData());
			auto* gateValues = static_cast<float*>(gate.UnsafeRawData());
			auto* upValues = static_cast<float*>(up.UnsafeRawData());
			for (std::size_t row = 0; row < rows; ++row)
			{
				std::copy_n(groupedValues + row * hiddenFeatures * 2, hiddenFeatures,
				            gateValues + row * hiddenFeatures);
				std::copy_n(groupedValues + row * hiddenFeatures * 2 + hiddenFeatures, hiddenFeatures,
				            upValues + row * hiddenFeatures);
			}
			return { std::move(gate), std::move(up) };
#else
			(void) input;
			(void) gateWeight;
			(void) upWeight;
			(void) threads;
			throw std::runtime_error("FFN activation verification requires an AOT-enabled build");
#endif
		}

		Tensor<CPU> RunStrictSwiGLU(const Tensor<CPU>& gate, const Tensor<CPU>& up)
		{
#ifdef LITENN_GGUF_CONVERT_ENABLE_AOT
			if (gate.DType() != DataType::Float32 || up.DType() != DataType::Float32 || gate.Shape() != up.Shape() ||
			    gate.Shape().NumDim() != 2)
			{
				throw std::runtime_error("strict SwiGLU verification requires shape-matched 2D Float32 tensors");
			}
			const auto rows = gate.Shape()[0];
			const auto columns = gate.Shape()[1];
			Tensor<CPU> result(Uninitialized, { rows, columns }, DataType::Float32);
			litenn_cpu_swiglu_f32(nullptr, static_cast<const float*>(gate.UnsafeRawData()), 0,
			                      static_cast<std::int64_t>(rows), static_cast<std::int64_t>(columns),
			                      static_cast<std::int64_t>(columns), 1, nullptr,
			                      static_cast<const float*>(up.UnsafeRawData()), 0, static_cast<std::int64_t>(rows),
			                      static_cast<std::int64_t>(columns), static_cast<std::int64_t>(columns), 1, nullptr,
			                      static_cast<float*>(result.UnsafeRawData()), 0, static_cast<std::int64_t>(rows),
			                      static_cast<std::int64_t>(columns), static_cast<std::int64_t>(columns), 1);
			return result;
#else
			(void) gate;
			(void) up;
			throw std::runtime_error("FFN activation verification requires an AOT-enabled build");
#endif
		}

		void WriteMetrics(std::ostream& output, const ErrorMetrics& metrics, std::string_view indent)
		{
			output << indent << "{\n"
			       << indent << "  \"max_abs\": " << metrics.maximumAbsolute << ",\n"
			       << indent << "  \"mean_abs\": " << metrics.meanAbsolute << ",\n"
			       << indent << "  \"rms_error\": " << metrics.rmsError << ",\n"
			       << indent << "  \"reference_rms\": " << metrics.referenceRms << ",\n"
			       << indent << "  \"nrmse\": " << metrics.normalizedRmsError << ",\n"
			       << indent << "  \"cosine_similarity\": " << metrics.cosineSimilarity << '\n'
			       << indent << '}';
		}

		void WriteReport(const DownProjectionVerificationOptions& options, std::span<const BlockResult> blocks,
		                 double maximumProductionVersusCapturedNRMSE,
		                 double maximumProductionVersusCapturedAbsoluteError)
		{
			if (!options.outputPath.parent_path().empty())
			{
				std::filesystem::create_directories(options.outputPath.parent_path());
			}
			std::ofstream output(options.outputPath, std::ios::binary | std::ios::trunc);
			if (!output)
			{
				throw std::runtime_error("failed to open Down projection verification report");
			}
			output << std::setprecision(17) << "{\n"
			       << "  \"schema\": \"litenn.qwen_down_projection_verification.v1\",\n"
			       << "  \"generated_index\": " << options.generatedIndex << ",\n"
			       << "  \"thread_count\": " << options.threadCount << ",\n"
			       << "  \"activation_source\": \"captured_ffn_swiglu\",\n"
			       << "  \"exact_accumulation\": \"float64\",\n"
			       << "  \"summary\": {\n"
			       << "    \"maximum_field_v4_vs_captured_nrmse\": " << maximumProductionVersusCapturedNRMSE << ",\n"
			       << "    \"maximum_field_v4_vs_captured_max_abs\": " << maximumProductionVersusCapturedAbsoluteError
			       << "\n"
			       << "  },\n"
			       << "  \"blocks\": [\n";
			for (std::size_t blockIndex = 0; blockIndex < blocks.size(); ++blockIndex)
			{
				const auto& block = blocks[blockIndex];
				output << "    {\n"
				       << "      \"block\": " << block.blockIndex << ",\n"
				       << "      \"weight\": \"" << block.weightName << "\",\n"
				       << "      \"format\": \"GGML_Q6_K\",\n"
				       << "      \"input_features\": " << block.inputFeatures << ",\n"
				       << "      \"output_features\": " << block.outputFeatures << ",\n"
				       << "      \"stored_bytes\": " << block.storedBytes << ",\n"
				       << "      \"field_v4_bytes\": " << block.packedBytes << ",\n"
				       << "      \"closest_to_exact\": \"" << block.closestToExact << "\",\n"
				       << "      \"candidates\": {\n";
				for (std::size_t candidateIndex = 0; candidateIndex < block.candidates.size(); ++candidateIndex)
				{
					const auto& candidate = block.candidates[candidateIndex];
					output << "        \"" << candidate.name << "\": {\n"
					       << "          \"versus_exact\": ";
					WriteMetrics(output, candidate.versusExact, "          ");
					output << ",\n          \"versus_captured\": ";
					WriteMetrics(output, candidate.versusCaptured, "          ");
					output << "\n        }" << (candidateIndex + 1 == block.candidates.size() ? "\n" : ",\n");
				}
				output << "      }\n    }" << (blockIndex + 1 == blocks.size() ? "\n" : ",\n");
			}
			output << "  ]\n}\n";
			if (!output)
			{
				throw std::runtime_error("failed to write Down projection verification report");
			}
		}

		void WriteFFNReport(const DownProjectionVerificationOptions& options, std::span<const FFNBlockResult> blocks,
		                    const FFNActivationVerificationSummary& summary)
		{
			if (!options.outputPath.parent_path().empty())
			{
				std::filesystem::create_directories(options.outputPath.parent_path());
			}
			std::ofstream output(options.outputPath, std::ios::binary | std::ios::trunc);
			if (!output)
			{
				throw std::runtime_error("failed to open FFN activation verification report");
			}
			output << std::setprecision(17) << "{\n"
			       << "  \"schema\": \"litenn.qwen_ffn_activation_verification.v1\",\n"
			       << "  \"generated_index\": " << options.generatedIndex << ",\n"
			       << "  \"thread_count\": " << options.threadCount << ",\n"
			       << "  \"activation_source\": \"captured_ffn_norm\",\n"
			       << "  \"summary\": {\n"
			       << "    \"maximum_field_v4_gate_vs_captured_nrmse\": "
			       << summary.maximumProductionGateVersusCapturedNRMSE << ",\n"
			       << "    \"maximum_field_v4_up_vs_captured_nrmse\": "
			       << summary.maximumProductionUpVersusCapturedNRMSE << ",\n"
			       << "    \"maximum_field_v4_swiglu_vs_captured_nrmse\": "
			       << summary.maximumProductionSwiGLUVersusCapturedNRMSE << ",\n"
			       << "    \"maximum_captured_input_swiglu_vs_captured_nrmse\": "
			       << summary.maximumCapturedInputSwiGLUVersusCapturedNRMSE << "\n"
			       << "  },\n"
			       << "  \"blocks\": [\n";
			for (std::size_t blockIndex = 0; blockIndex < blocks.size(); ++blockIndex)
			{
				const auto& block = blocks[blockIndex];
				output << "    {\n"
				       << "      \"block\": " << block.blockIndex << ",\n"
				       << "      \"format\": \"GGML_Q4_K\",\n"
				       << "      \"input_features\": " << block.inputFeatures << ",\n"
				       << "      \"hidden_features\": " << block.hiddenFeatures << ",\n"
				       << "      \"candidates\": {\n";
				for (std::size_t candidateIndex = 0; candidateIndex < block.candidates.size(); ++candidateIndex)
				{
					const auto& candidate = block.candidates[candidateIndex];
					output << "        \"" << candidate.name << "\": {\n";
					const auto writeStage = [&](std::string_view name, const ErrorMetrics& versusExact,
					                            const ErrorMetrics& versusCaptured, bool last) {
						output << "          \"" << name << "\": {\n"
						       << "            \"versus_exact\": ";
						WriteMetrics(output, versusExact, "            ");
						output << ",\n            \"versus_captured\": ";
						WriteMetrics(output, versusCaptured, "            ");
						output << "\n          }" << (last ? "\n" : ",\n");
					};
					writeStage("gate", candidate.gateVersusExact, candidate.gateVersusCaptured, false);
					writeStage("up", candidate.upVersusExact, candidate.upVersusCaptured, false);
					writeStage("swiglu", candidate.swigluVersusExact, candidate.swigluVersusCaptured, true);
					output << "        }" << (candidateIndex + 1 == block.candidates.size() ? "\n" : ",\n");
				}
				output << "      }\n    }" << (blockIndex + 1 == blocks.size() ? "\n" : ",\n");
			}
			output << "  ]\n}\n";
			if (!output)
			{
				throw std::runtime_error("failed to write FFN activation verification report");
			}
		}
	} // namespace

	DownProjectionVerificationSummary
	VerifyLLaMADownProjectionCheckpoints(const DownProjectionVerificationOptions& options)
	{
		if (options.blockIndices.empty() || options.threadCount == 0)
		{
			throw std::runtime_error("Down projection verification requires blocks and a positive thread count");
		}
		if (!std::ranges::is_sorted(options.blockIndices) ||
		    std::ranges::adjacent_find(options.blockIndices) != options.blockIndices.end())
		{
			throw std::runtime_error("Down projection verification block indices must be sorted and unique");
		}

		const auto imported = ImportGGUFArchive(options.modelPath);
		const auto& graph = imported.model.UnsafeGraphView();
		const auto hyperparameters = ParseLLaMAHyperparameters(graph);
		std::vector<BlockResult> results;
		results.reserve(options.blockIndices.size());
		std::map<std::string, std::size_t, std::less<>> closestCounts;
		double maximumProductionVersusCapturedNRMSE = 0.0;
		double maximumProductionVersusCapturedAbsoluteError = 0.0;
		for (const auto blockIndex : options.blockIndices)
		{
			if (blockIndex >= hyperparameters.blockCount)
			{
				throw std::runtime_error("Down projection verification block index exceeds model block count");
			}
			const auto weightName = std::format("blk.{}.ffn_down.weight", blockIndex);
			const auto variableIndex = graph.FindVariable(weightName);
			if (!variableIndex)
			{
				throw std::runtime_error(std::format("model does not contain {}", weightName));
			}
			const auto& weight = *graph.GetVariable(*variableIndex);
			if (!weight.IsQuantized() || weight.Quantization()->blockFormat != QuantizedBlockFormat::GGML_Q6_K ||
			    weight.Quantization()->expressedType != DataType::Float32 ||
			    weight.Quantization()->expressedShape.size() != 2)
			{
				throw std::runtime_error(std::format("{} must be a 2D Float32-expressed GGML_Q6_K tensor", weightName));
			}
			const auto outputFeatures = weight.Quantization()->expressedShape[0];
			const auto inputFeatures = weight.Quantization()->expressedShape[1];
			auto activation = LoadCheckpointTensor(options.checkpointDirectory, "ffn_swiglu", options.generatedIndex,
			                                       blockIndex, inputFeatures);
			auto captured = LoadCheckpointTensor(options.checkpointDirectory, "ffn_down", options.generatedIndex,
			                                     blockIndex, outputFeatures);
			auto exact = EvalGGMLExactDequantizedMatMul(activation, weight, true);
			auto ggml = EvalGGMLQuantizedMatMul(activation, weight, true);
			auto source = RunSourceHelper(activation, weight, options.threadCount);
			auto [production, packedBytes] = RunProductionHelper(activation, weight, options.threadCount);

			const auto exactValues = Values(exact);
			const auto capturedValues = Values(captured);
			std::vector<std::pair<std::string, const Tensor<CPU>*>> candidates{
				{ "llama_cpp_captured", &captured },
				{ "ggml_vec_dot", &ggml },
				{ "litenn_source_f32", &source },
				{ "litenn_field_v4_q8k", &production },
			};
			BlockResult result{
				.blockIndex = blockIndex,
				.weightName = weightName,
				.inputFeatures = inputFeatures,
				.outputFeatures = outputFeatures,
				.storedBytes = weight.Data().NumElements(),
				.packedBytes = packedBytes,
			};
			double closestError = std::numeric_limits<double>::infinity();
			for (const auto& [name, tensor] : candidates)
			{
				const auto versusExact = Compare(Values(*tensor), exactValues);
				const auto versusCaptured = Compare(Values(*tensor), capturedValues);
				result.candidates.push_back({
				    .name = name,
				    .versusExact = versusExact,
				    .versusCaptured = versusCaptured,
				});
				if (name == "litenn_field_v4_q8k")
				{
					maximumProductionVersusCapturedNRMSE =
					    std::max(maximumProductionVersusCapturedNRMSE, versusCaptured.normalizedRmsError);
					maximumProductionVersusCapturedAbsoluteError =
					    std::max(maximumProductionVersusCapturedAbsoluteError, versusCaptured.maximumAbsolute);
				}
				if (versusExact.rmsError < closestError)
				{
					closestError = versusExact.rmsError;
					result.closestToExact = name;
				}
			}
			++closestCounts[result.closestToExact];
			results.push_back(std::move(result));
		}

		WriteReport(options, results, maximumProductionVersusCapturedNRMSE,
		            maximumProductionVersusCapturedAbsoluteError);
		std::ostringstream counts;
		bool first = true;
		for (const auto& [name, count] : closestCounts)
		{
			counts << (std::exchange(first, false) ? "" : ",") << name << ':' << count;
		}
		return {
			.blockCount = results.size(),
			.closestCandidateCounts = std::move(counts).str(),
			.maximumProductionVersusCapturedNRMSE = maximumProductionVersusCapturedNRMSE,
			.maximumProductionVersusCapturedAbsoluteError = maximumProductionVersusCapturedAbsoluteError,
		};
	}

	FFNActivationVerificationSummary
	VerifyLLaMAFFNActivationCheckpoints(const DownProjectionVerificationOptions& options)
	{
		if (options.blockIndices.empty() || options.threadCount == 0)
		{
			throw std::runtime_error("FFN activation verification requires blocks and a positive thread count");
		}
		if (!std::ranges::is_sorted(options.blockIndices) ||
		    std::ranges::adjacent_find(options.blockIndices) != options.blockIndices.end())
		{
			throw std::runtime_error("FFN activation verification block indices must be sorted and unique");
		}

		const auto imported = ImportGGUFArchive(options.modelPath);
		const auto& graph = imported.model.UnsafeGraphView();
		const auto hyperparameters = ParseLLaMAHyperparameters(graph);
		std::vector<FFNBlockResult> results;
		results.reserve(options.blockIndices.size());
		FFNActivationVerificationSummary summary{ .blockCount = options.blockIndices.size() };
		for (const auto blockIndex : options.blockIndices)
		{
			if (blockIndex >= hyperparameters.blockCount)
			{
				throw std::runtime_error("FFN activation verification block index exceeds model block count");
			}
			const auto gateName = std::format("blk.{}.ffn_gate.weight", blockIndex);
			const auto upName = std::format("blk.{}.ffn_up.weight", blockIndex);
			const auto gateIndex = graph.FindVariable(gateName);
			const auto upIndex = graph.FindVariable(upName);
			if (!gateIndex || !upIndex)
			{
				throw std::runtime_error(std::format("model does not contain {} and {}", gateName, upName));
			}
			const auto& gateWeight = *graph.GetVariable(*gateIndex);
			const auto& upWeight = *graph.GetVariable(*upIndex);
			const auto validateWeight = [](const Variable& weight, std::string_view name) {
				if (!weight.IsQuantized() || weight.Quantization()->blockFormat != QuantizedBlockFormat::GGML_Q4_K ||
				    weight.Quantization()->expressedType != DataType::Float32 ||
				    weight.Quantization()->expressedShape.size() != 2)
				{
					throw std::runtime_error(std::format("{} must be a 2D Float32-expressed GGML_Q4_K tensor", name));
				}
			};
			validateWeight(gateWeight, gateName);
			validateWeight(upWeight, upName);
			if (gateWeight.Quantization()->expressedShape != upWeight.Quantization()->expressedShape)
			{
				throw std::runtime_error("FFN Gate/Up expressed shapes differ");
			}
			const auto hiddenFeatures = gateWeight.Quantization()->expressedShape[0];
			const auto inputFeatures = gateWeight.Quantization()->expressedShape[1];
			auto activation = LoadCheckpointTensor(options.checkpointDirectory, "ffn_norm", options.generatedIndex,
			                                       blockIndex, inputFeatures);
			auto capturedGate = LoadCheckpointTensor(options.checkpointDirectory, "ffn_gate", options.generatedIndex,
			                                         blockIndex, hiddenFeatures);
			auto capturedUp = LoadCheckpointTensor(options.checkpointDirectory, "ffn_up", options.generatedIndex,
			                                       blockIndex, hiddenFeatures);
			auto capturedSwiGLU = LoadCheckpointTensor(options.checkpointDirectory, "ffn_swiglu",
			                                           options.generatedIndex, blockIndex, hiddenFeatures);

			auto exactGate = EvalGGMLExactDequantizedMatMul(activation, gateWeight, true);
			auto exactUp = EvalGGMLExactDequantizedMatMul(activation, upWeight, true);
			auto exactSwiGLU = RunStrictSwiGLU(exactGate, exactUp);
			auto ggmlGate = EvalGGMLQuantizedMatMul(activation, gateWeight, true);
			auto ggmlUp = EvalGGMLQuantizedMatMul(activation, upWeight, true);
			auto ggmlSwiGLU = RunStrictSwiGLU(ggmlGate, ggmlUp);
			auto sourceGate = RunSourceHelper(activation, gateWeight, options.threadCount);
			auto sourceUp = RunSourceHelper(activation, upWeight, options.threadCount);
			auto sourceSwiGLU = RunStrictSwiGLU(sourceGate, sourceUp);
			auto [productionGate, productionUp] =
			    RunGroupedFFNProductionHelper(activation, gateWeight, upWeight, options.threadCount);
			auto productionSwiGLU = RunStrictSwiGLU(productionGate, productionUp);
			auto capturedInputSwiGLU = RunStrictSwiGLU(capturedGate, capturedUp);

			struct Candidate
			{
				std::string name;
				const Tensor<CPU>* gate;
				const Tensor<CPU>* up;
				const Tensor<CPU>* swiglu;
			};
			const std::array candidates{
				Candidate{ "llama_cpp_captured", &capturedGate, &capturedUp, &capturedSwiGLU },
				Candidate{ "ggml_vec_dot", &ggmlGate, &ggmlUp, &ggmlSwiGLU },
				Candidate{ "litenn_source_f32", &sourceGate, &sourceUp, &sourceSwiGLU },
				Candidate{ "litenn_grouped_field_v4_q8k", &productionGate, &productionUp, &productionSwiGLU },
				Candidate{ "captured_gate_up_litenn_strict_swiglu", &capturedGate, &capturedUp, &capturedInputSwiGLU },
			};
			FFNBlockResult result{
				.blockIndex = blockIndex,
				.inputFeatures = inputFeatures,
				.hiddenFeatures = hiddenFeatures,
			};
			for (const auto& candidate : candidates)
			{
				FFNCandidateMetrics metrics{
					.name = candidate.name,
					.gateVersusExact = Compare(Values(*candidate.gate), Values(exactGate)),
					.gateVersusCaptured = Compare(Values(*candidate.gate), Values(capturedGate)),
					.upVersusExact = Compare(Values(*candidate.up), Values(exactUp)),
					.upVersusCaptured = Compare(Values(*candidate.up), Values(capturedUp)),
					.swigluVersusExact = Compare(Values(*candidate.swiglu), Values(exactSwiGLU)),
					.swigluVersusCaptured = Compare(Values(*candidate.swiglu), Values(capturedSwiGLU)),
				};
				if (candidate.name == "litenn_grouped_field_v4_q8k")
				{
					summary.maximumProductionGateVersusCapturedNRMSE =
					    std::max(summary.maximumProductionGateVersusCapturedNRMSE,
					             metrics.gateVersusCaptured.normalizedRmsError);
					summary.maximumProductionUpVersusCapturedNRMSE = std::max(
					    summary.maximumProductionUpVersusCapturedNRMSE, metrics.upVersusCaptured.normalizedRmsError);
					summary.maximumProductionSwiGLUVersusCapturedNRMSE =
					    std::max(summary.maximumProductionSwiGLUVersusCapturedNRMSE,
					             metrics.swigluVersusCaptured.normalizedRmsError);
				}
				if (candidate.name == "captured_gate_up_litenn_strict_swiglu")
				{
					summary.maximumCapturedInputSwiGLUVersusCapturedNRMSE =
					    std::max(summary.maximumCapturedInputSwiGLUVersusCapturedNRMSE,
					             metrics.swigluVersusCaptured.normalizedRmsError);
				}
				result.candidates.push_back(std::move(metrics));
			}
			results.push_back(std::move(result));
		}

		WriteFFNReport(options, results, summary);
		return summary;
	}
} // namespace LiteNN::GGUF::Tooling
