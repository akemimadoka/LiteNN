#include "ggml-cpu.h"
#include "llama.h"

#include <algorithm>
#include <chrono>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <map>
#include <numeric>
#include <string>
#include <vector>

namespace
{

	using Clock = std::chrono::steady_clock;

	void SilentLog(ggml_log_level, const char*, void*)
	{
	}

	struct StageStats
	{
		double totalMilliseconds = 0.0;
		std::uint64_t calls = 0;
	};

	enum class ProfileMode
	{
		Baseline,
#ifdef LITENN_LLAMA_CPP_HAS_STAGE_COUNTERS
		Aggregate,
#endif
		Coarse,
		FFN,
		Layer,
		Cut
	};

	struct CallbackState
	{
		bool record = false;
		ProfileMode mode = ProfileMode::Baseline;
		int targetLayer = -1;
		std::string cutBoundary;
		bool segmentOpen = false;
		Clock::time_point segmentStart;
		std::string pendingStage;
		std::map<std::string, StageStats> totals;
	};

	struct RunResult
	{
		double totalMilliseconds = 0.0;
		std::map<std::string, StageStats> totals;
	};

	std::vector<llama_token> ParseTokenIds(const char* text, const char* option)
	{
		std::vector<llama_token> tokens;
		const char* cursor = text;
		while (*cursor != '\0')
		{
			char* end = nullptr;
			const long parsed = std::strtol(cursor, &end, 10);
			if (end == cursor || parsed < 0 || parsed > INT32_MAX)
			{
				std::fprintf(stderr, "invalid token id in %s: %s\n", option, text);
				std::exit(2);
			}
			tokens.push_back(static_cast<llama_token>(parsed));
			if (*end == '\0')
			{
				break;
			}
			if (*end != ',')
			{
				std::fprintf(stderr, "invalid token list in %s: %s\n", option, text);
				std::exit(2);
			}
			cursor = end + 1;
			if (*cursor == '\0')
			{
				std::fprintf(stderr, "trailing comma in %s: %s\n", option, text);
				std::exit(2);
			}
		}
		if (tokens.empty())
		{
			std::fprintf(stderr, "%s must not be empty\n", option);
			std::exit(2);
		}
		return tokens;
	}

	bool Contains(const char* text, const char* needle)
	{
		return text != nullptr && std::strstr(text, needle) != nullptr;
	}

	std::string MatMulStage(const ggml_tensor* tensor)
	{
		const char* weight = tensor->src[0] != nullptr ? tensor->src[0]->name : "";
		if (Contains(weight, "ffn_gate"))
		{
			return "projection.ffn_gate";
		}
		if (Contains(weight, "ffn_up"))
		{
			return "projection.ffn_up";
		}
		if (Contains(weight, "ffn_down"))
		{
			return "projection.ffn_down";
		}
		if (Contains(weight, "attn_q"))
		{
			return "projection.attn_q";
		}
		if (Contains(weight, "attn_k"))
		{
			return "projection.attn_k";
		}
		if (Contains(weight, "attn_v"))
		{
			return "projection.attn_v";
		}
		if (Contains(weight, "attn_output"))
		{
			return "projection.attn_output";
		}
		if (Contains(weight, "output.weight"))
		{
			return "projection.logits";
		}
		return "attention.matmul";
	}

	bool IsLayerWeight(const char* weight, int layer)
	{
		const std::string prefix = "blk." + std::to_string(layer) + ".";
		return weight != nullptr && std::strncmp(weight, prefix.c_str(), prefix.size()) == 0;
	}

	std::string BoundaryStage(const ggml_tensor* tensor, const CallbackState& state)
	{
		const ProfileMode mode = state.mode;
		if (mode == ProfileMode::Coarse)
		{
			if (tensor->op == GGML_OP_MUL_MAT)
			{
				const std::string stage = MatMulStage(tensor);
				if (stage == "projection.attn_output")
				{
					return "attention_block";
				}
				if (stage == "projection.ffn_down")
				{
					return "ffn_block";
				}
				if (stage == "projection.logits")
				{
					return "final_logits";
				}
			}
		}
		else if (mode == ProfileMode::FFN)
		{
			if (tensor->op == GGML_OP_MUL_MAT)
			{
				const std::string stage = MatMulStage(tensor);
				if (stage == "projection.attn_output")
				{
					return "attention_block";
				}
				const char* type = tensor->src[0] != nullptr ? ggml_type_name(tensor->src[0]->type) : "unknown";
				if (stage == "projection.ffn_gate")
				{
					return std::string("ffn.prefix_and_gate.") + type;
				}
				if (stage == "projection.ffn_up")
				{
					return std::string("ffn.up.") + type;
				}
				if (stage == "projection.ffn_down")
				{
					return std::string("ffn.activation_and_down.") + type;
				}
				if (stage == "projection.logits")
				{
					return "final_logits";
				}
			}
		}
		else if (mode == ProfileMode::Layer && tensor->op == GGML_OP_MUL_MAT)
		{
			const char* weight = tensor->src[0] != nullptr ? tensor->src[0]->name : "";
			if (state.targetLayer > 0 && IsLayerWeight(weight, state.targetLayer - 1) && Contains(weight, "ffn_down"))
			{
				return "setup.before_layer";
			}
			if (IsLayerWeight(weight, state.targetLayer))
			{
				const char* type = tensor->src[0] != nullptr ? ggml_type_name(tensor->src[0]->type) : "unknown";
				if (Contains(weight, "attn_output"))
				{
					return "layer.attention";
				}
				if (Contains(weight, "ffn_gate"))
				{
					return std::string("layer.ffn_prefix_and_gate.") + type;
				}
				if (Contains(weight, "ffn_up"))
				{
					return std::string("layer.ffn_up.") + type;
				}
				if (Contains(weight, "ffn_down"))
				{
					return std::string("layer.ffn_activation_and_down.") + type;
				}
			}
		}
		else if (mode == ProfileMode::Cut && tensor->op == GGML_OP_MUL_MAT)
		{
			const char* weight = tensor->src[0] != nullptr ? tensor->src[0]->name : "";
			const bool selected =
			    (state.cutBoundary == "prev" && state.targetLayer > 0 && IsLayerWeight(weight, state.targetLayer - 1) &&
			     Contains(weight, "ffn_down")) ||
			    (state.cutBoundary == "attn" && IsLayerWeight(weight, state.targetLayer) &&
			     Contains(weight, "attn_output")) ||
			    (state.cutBoundary == "gate" && IsLayerWeight(weight, state.targetLayer) &&
			     Contains(weight, "ffn_gate")) ||
			    (state.cutBoundary == "up" && IsLayerWeight(weight, state.targetLayer) && Contains(weight, "ffn_up")) ||
			    (state.cutBoundary == "down" && IsLayerWeight(weight, state.targetLayer) &&
			     Contains(weight, "ffn_down"));
			if (selected)
			{
				return "cut.prefix";
			}
		}
		if (mode == ProfileMode::Cut)
		{
			return {};
		}
		if ((tensor->flags & GGML_TENSOR_FLAG_OUTPUT) != 0)
		{
			return "graph.trailing";
		}
		return {};
	}

	bool ProfileCallback(ggml_tensor* tensor, bool ask, void* userData)
	{
		auto& state = *static_cast<CallbackState*>(userData);
		if (ask)
		{
			if (!state.segmentOpen)
			{
				state.segmentStart = Clock::now();
				state.segmentOpen = true;
			}
			state.pendingStage = BoundaryStage(tensor, state);
			return !state.pendingStage.empty();
		}

		const auto end = Clock::now();
		if (state.record && state.segmentOpen)
		{
			auto& stats = state.totals[state.pendingStage];
			stats.totalMilliseconds += std::chrono::duration<double, std::milli>(end - state.segmentStart).count();
			++stats.calls;
		}
		state.segmentOpen = false;
		state.pendingStage.clear();
		return true;
	}

	ProfileMode ParseMode(const char* value, int& targetLayer)
	{
		if (std::strcmp(value, "baseline") == 0)
		{
			return ProfileMode::Baseline;
		}
#ifdef LITENN_LLAMA_CPP_HAS_STAGE_COUNTERS
		if (std::strcmp(value, "aggregate") == 0)
		{
			return ProfileMode::Aggregate;
		}
#endif
		if (std::strcmp(value, "coarse") == 0)
		{
			return ProfileMode::Coarse;
		}
		if (std::strcmp(value, "ffn") == 0)
		{
			return ProfileMode::FFN;
		}
		constexpr const char* layerPrefix = "layer-";
		if (std::strncmp(value, layerPrefix, std::strlen(layerPrefix)) == 0)
		{
			const char* layerText = value + std::strlen(layerPrefix);
			char* end = nullptr;
			const long parsedLayer = std::strtol(layerText, &end, 10);
			if (end != layerText && *end == '\0' && parsedLayer > 0 && parsedLayer <= 100000)
			{
				targetLayer = static_cast<int>(parsedLayer);
				return ProfileMode::Layer;
			}
		}
		std::fprintf(stderr,
#ifdef LITENN_LLAMA_CPP_HAS_STAGE_COUNTERS
		             "invalid mode %s; expected baseline, aggregate, coarse, ffn, or layer-N\n",
#else
		             "invalid mode %s; expected baseline, coarse, ffn, or layer-N\n",
#endif
		             value);
		std::exit(2);
	}

	int ParseScanLayer(const char* value)
	{
		constexpr const char* scanPrefix = "scan-layer-";
		if (std::strncmp(value, scanPrefix, std::strlen(scanPrefix)) != 0)
		{
			return -1;
		}
		const char* layerText = value + std::strlen(scanPrefix);
		char* end = nullptr;
		const long parsedLayer = std::strtol(layerText, &end, 10);
		if (end == layerText || *end != '\0' || parsedLayer <= 0 || parsedLayer > 100000)
		{
			std::fprintf(stderr, "invalid scan layer in mode %s\n", value);
			std::exit(2);
		}
		return static_cast<int>(parsedLayer);
	}

	RunResult RunDecode(llama_model* model, ProfileMode mode, int targetLayer, const std::string& cutBoundary,
	                    int threads, int warmup, int steps)
	{
		CallbackState callbackState;
		callbackState.mode = mode;
		callbackState.targetLayer = targetLayer;
		callbackState.cutBoundary = cutBoundary;
		llama_context_params contextParams = llama_context_default_params();
		contextParams.n_ctx = std::max(64, warmup + steps + 1);
		contextParams.n_batch = 1;
		contextParams.n_ubatch = 1;
		contextParams.n_threads = threads;
		contextParams.n_threads_batch = threads;
		contextParams.flash_attn_type = LLAMA_FLASH_ATTN_TYPE_DISABLED;
		contextParams.no_perf = false;
		if (mode != ProfileMode::Baseline)
		{
			contextParams.cb_eval = ProfileCallback;
			contextParams.cb_eval_user_data = &callbackState;
		}

		llama_context* context = llama_init_from_model(model, contextParams);
		if (context == nullptr)
		{
			std::fprintf(stderr, "failed to create context\n");
			std::exit(1);
		}
		llama_token token = 151644;
		std::vector<double> decodeMilliseconds;
		decodeMilliseconds.reserve(static_cast<std::size_t>(steps));
		for (int index = 0; index < warmup + steps; ++index)
		{
			callbackState.record = index >= warmup;
			callbackState.segmentOpen = false;
			const auto start = Clock::now();
			const int status = llama_decode(context, llama_batch_get_one(&token, 1));
			const auto end = Clock::now();
			if (status != 0)
			{
				std::fprintf(stderr, "decode failed at step %d: %d\n", index, status);
				llama_free(context);
				std::exit(1);
			}
			if (index >= warmup)
			{
				decodeMilliseconds.push_back(std::chrono::duration<double, std::milli>(end - start).count());
			}
		}
		RunResult result;
		result.totalMilliseconds = std::accumulate(decodeMilliseconds.begin(), decodeMilliseconds.end(), 0.0);
		result.totals = std::move(callbackState.totals);
		llama_free(context);
		return result;
	}

} // namespace

int main(int argc, char** argv)
{
	if (argc < 3)
	{
		std::fprintf(stderr,
#ifdef LITENN_LLAMA_CPP_HAS_STAGE_COUNTERS
		             "usage: %s MODEL baseline|aggregate|coarse|ffn|layer-N|scan-layer-N [threads] [warmup] [steps] "
		             "[--prefill-token-ids CSV --decode-token-ids CSV]\n",
#else
		             "usage: %s MODEL baseline|coarse|ffn|layer-N|scan-layer-N [threads] [warmup] [steps] "
		             "[--prefill-token-ids CSV --decode-token-ids CSV]\n",
#endif
		             argv[0]);
		return 2;
	}

	const int scanLayer = ParseScanLayer(argv[2]);
	int targetLayer = -1;
	const ProfileMode mode = scanLayer >= 0 ? ProfileMode::Baseline : ParseMode(argv[2], targetLayer);
	const int threads = argc > 3 ? std::atoi(argv[3]) : 2;
	const int warmup = argc > 4 ? std::atoi(argv[4]) : 9;
	const int steps = argc > 5 ? std::atoi(argv[5]) : 15;
	std::vector<llama_token> prefillTokens;
	std::vector<llama_token> decodeTokens;
	for (int index = 6; index < argc; index += 2)
	{
		if (index + 1 >= argc)
		{
			std::fprintf(stderr, "missing value for %s\n", argv[index]);
			return 2;
		}
		if (std::strcmp(argv[index], "--prefill-token-ids") == 0)
		{
			prefillTokens = ParseTokenIds(argv[index + 1], argv[index]);
		}
		else if (std::strcmp(argv[index], "--decode-token-ids") == 0)
		{
			decodeTokens = ParseTokenIds(argv[index + 1], argv[index]);
		}
		else
		{
			std::fprintf(stderr, "unknown option %s\n", argv[index]);
			return 2;
		}
	}
	if (threads <= 0 || warmup < 0 || steps <= 0)
	{
		std::fprintf(stderr, "threads and steps must be positive and warmup must be non-negative\n");
		return 2;
	}
	if (prefillTokens.empty() != decodeTokens.empty())
	{
		std::fprintf(stderr, "--prefill-token-ids and --decode-token-ids must be supplied together\n");
		return 2;
	}
	if (!decodeTokens.empty() && (warmup != 0 || decodeTokens.size() != static_cast<std::size_t>(steps)))
	{
		std::fprintf(stderr, "exact token replay requires warmup=0 and one decode token per measured step\n");
		return 2;
	}
	if (scanLayer >= 0 && !decodeTokens.empty())
	{
		std::fprintf(stderr, "exact token replay is not supported by scan-layer modes\n");
		return 2;
	}

	llama_log_set(SilentLog, nullptr);
	ggml_backend_load_all();

	llama_model_params modelParams = llama_model_default_params();
	modelParams.n_gpu_layers = 0;
	modelParams.use_mmap = true;
	llama_model* model = llama_model_load_from_file(argv[1], modelParams);
	if (model == nullptr)
	{
		std::fprintf(stderr, "failed to load model\n");
		return 1;
	}
	if (scanLayer >= 0)
	{
		const RunResult baselineBefore = RunDecode(model, ProfileMode::Baseline, scanLayer, {}, threads, warmup, steps);
		const std::vector<std::string> boundaries = { "prev", "attn", "gate", "up", "down" };
		std::map<std::string, double> prefixMilliseconds;
		for (const std::string& boundary : boundaries)
		{
			const RunResult result = RunDecode(model, ProfileMode::Cut, scanLayer, boundary, threads, warmup, steps);
			const auto prefix = result.totals.find("cut.prefix");
			if (prefix == result.totals.end() || prefix->second.calls != static_cast<std::uint64_t>(steps))
			{
				std::fprintf(stderr, "cut %s produced %llu calls; expected %d\n", boundary.c_str(),
				             prefix == result.totals.end() ? 0ULL
				                                           : static_cast<unsigned long long>(prefix->second.calls),
				             steps);
				llama_model_free(model);
				llama_backend_free();
				return 1;
			}
			prefixMilliseconds.emplace(boundary, prefix->second.totalMilliseconds / steps);
		}
		const RunResult baselineAfter = RunDecode(model, ProfileMode::Baseline, scanLayer, {}, threads, warmup, steps);
		const double baselineMilliseconds =
		    (baselineBefore.totalMilliseconds + baselineAfter.totalMilliseconds) / (2.0 * steps);
		const std::vector<std::pair<std::string, double>> stages = {
			{ "layer.attention", prefixMilliseconds.at("attn") - prefixMilliseconds.at("prev") },
			{ "layer.ffn_prefix_and_gate", prefixMilliseconds.at("gate") - prefixMilliseconds.at("attn") },
			{ "layer.ffn_up", prefixMilliseconds.at("up") - prefixMilliseconds.at("gate") },
			{ "layer.ffn_activation_and_down", prefixMilliseconds.at("down") - prefixMilliseconds.at("up") },
		};
		if (std::any_of(stages.begin(), stages.end(), [](const auto& stage) { return stage.second <= 0.0; }))
		{
			std::fprintf(stderr, "non-positive stage delta in scan-layer-%d\n", scanLayer);
			for (const auto& [boundary, milliseconds] : prefixMilliseconds)
			{
				std::fprintf(stderr, "cut=%s prefix_ms_per_token=%.6f\n", boundary.c_str(), milliseconds);
			}
			llama_model_free(model);
			llama_backend_free();
			return 1;
		}
		std::printf("mode=%s threads=%d warmup=%d steps=%d mean_decode_ms=%.6f tokens_per_second=%.6f\n", argv[2],
		            threads, warmup, steps, baselineMilliseconds, 1000.0 / baselineMilliseconds);
		for (const auto& [stage, milliseconds] : stages)
		{
			std::printf("stage=%s ms_per_token=%.6f calls_per_token=1.000 percent_of_decode=%.3f\n", stage.c_str(),
			            milliseconds, 100.0 * milliseconds / baselineMilliseconds);
		}
		for (const auto& [boundary, milliseconds] : prefixMilliseconds)
		{
			std::printf("cut=%s prefix_ms_per_token=%.6f\n", boundary.c_str(), milliseconds);
		}
		llama_model_free(model);
		llama_backend_free();
		return 0;
	}

	CallbackState callbackState;
	callbackState.mode = mode;
	callbackState.targetLayer = targetLayer;
	llama_context_params contextParams = llama_context_default_params();
	contextParams.n_ctx = std::max(64, static_cast<int>(prefillTokens.size()) + warmup + steps + 1);
	contextParams.n_batch = std::max(1, static_cast<int>(prefillTokens.size()));
	contextParams.n_ubatch = contextParams.n_batch;
	contextParams.n_threads = threads;
	contextParams.n_threads_batch = threads;
	contextParams.flash_attn_type = LLAMA_FLASH_ATTN_TYPE_DISABLED;
	contextParams.no_perf = false;
	if (mode != ProfileMode::Baseline
#ifdef LITENN_LLAMA_CPP_HAS_STAGE_COUNTERS
	    && mode != ProfileMode::Aggregate
#endif
	)
	{
		contextParams.cb_eval = ProfileCallback;
		contextParams.cb_eval_user_data = &callbackState;
	}

	llama_context* context = llama_init_from_model(model, contextParams);
	if (context == nullptr)
	{
		std::fprintf(stderr, "failed to create context\n");
		llama_model_free(model);
		return 1;
	}
	if (!prefillTokens.empty())
	{
		const int status = llama_decode(
		    context, llama_batch_get_one(prefillTokens.data(), static_cast<int32_t>(prefillTokens.size())));
		if (status != 0)
		{
			std::fprintf(stderr, "prefill failed: %d\n", status);
			llama_free(context);
			llama_model_free(model);
			return 1;
		}
	}

	llama_token token = 151644;
	std::vector<double> decodeMilliseconds;
	decodeMilliseconds.reserve(static_cast<std::size_t>(steps));
#ifdef LITENN_LLAMA_CPP_HAS_STAGE_COUNTERS
	ggml_cpu_stage_profile_snapshot aggregateSnapshot{};
#endif
	for (int index = 0; index < warmup + steps; ++index)
	{
#ifdef LITENN_LLAMA_CPP_HAS_STAGE_COUNTERS
		if (mode == ProfileMode::Aggregate && index == warmup)
		{
			ggml_cpu_stage_profile_set_enabled(false);
			ggml_cpu_stage_profile_reset();
			ggml_cpu_stage_profile_set_enabled(true);
		}
#endif
		callbackState.record = index >= warmup;
		callbackState.segmentOpen = false;
		if (!decodeTokens.empty())
		{
			token = decodeTokens[static_cast<std::size_t>(index)];
		}
		const auto start = Clock::now();
		const int status = llama_decode(context, llama_batch_get_one(&token, 1));
		const auto end = Clock::now();
		if (status != 0)
		{
#ifdef LITENN_LLAMA_CPP_HAS_STAGE_COUNTERS
			ggml_cpu_stage_profile_set_enabled(false);
#endif
			std::fprintf(stderr, "decode failed at step %d: %d\n", index, status);
			llama_free(context);
			llama_model_free(model);
			return 1;
		}
		if (index >= warmup)
		{
			decodeMilliseconds.push_back(std::chrono::duration<double, std::milli>(end - start).count());
		}
	}
#ifdef LITENN_LLAMA_CPP_HAS_STAGE_COUNTERS
	if (mode == ProfileMode::Aggregate)
	{
		ggml_cpu_stage_profile_set_enabled(false);
		ggml_cpu_stage_profile_get_snapshot(&aggregateSnapshot);
	}
#endif

	const double totalMilliseconds = std::accumulate(decodeMilliseconds.begin(), decodeMilliseconds.end(), 0.0);
	std::printf("mode=%s threads=%d warmup=%d steps=%d mean_decode_ms=%.6f tokens_per_second=%.6f\n", argv[2], threads,
	            warmup, steps, totalMilliseconds / steps, 1000.0 * steps / totalMilliseconds);
	for (const auto& [stage, stats] : callbackState.totals)
	{
		std::printf("stage=%s ms_per_token=%.6f calls_per_token=%.3f percent_of_decode=%.3f\n", stage.c_str(),
		            stats.totalMilliseconds / steps, static_cast<double>(stats.calls) / steps,
		            100.0 * stats.totalMilliseconds / totalMilliseconds);
	}
#ifdef LITENN_LLAMA_CPP_HAS_STAGE_COUNTERS
	if (mode == ProfileMode::Aggregate)
	{
		const char* stageNames[GGML_CPU_STAGE_PROFILE_COUNT] = {
			"attention",
			"ffn.gate_up",
			"ffn.down",
			"logits",
		};
		for (int stage = 0; stage < GGML_CPU_STAGE_PROFILE_COUNT; ++stage)
		{
			const double milliseconds = static_cast<double>(aggregateSnapshot.microseconds[stage]) / 1000.0;
			std::printf("stage=%s ms_per_token=%.6f calls_per_token=%.3f percent_of_decode=%.3f\n", stageNames[stage],
			            milliseconds / steps, static_cast<double>(aggregateSnapshot.segments[stage]) / steps,
			            100.0 * milliseconds / totalMilliseconds);
		}
	}
#endif

	llama_perf_context_print(context);
	llama_free(context);
	llama_model_free(model);
	llama_backend_free();
	return 0;
}
