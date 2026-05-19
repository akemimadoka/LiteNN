#include <benchmark/benchmark.h>

#include <LiteNN.h>
#include <LiteNN/Initializer/Initializer.h>
#include <LiteNN/Layer/Layer.h>
#include <LiteNN/Optimizer/Loss.h>
#include <LiteNN/Optimizer/OptimizerUtils.h>
#include <LiteNN/Optimizer/SGD.h>
#include <LiteNN/Pass/AutogradPass.h>
#include <LiteNN/Pass/ConstFoldPass.h>
#include <LiteNN/Pass/FusionPass.h>
#include <LiteNN/Pass/InlinePass.h>
#include <LiteNN/Runtime/Interpreter.h>
#include <LiteNN/Training/Trainer.h>

#ifdef LITENN_TRAIN_BENCH_HAS_AOT
#include <LiteNN/Compiler/CompiledModule.h>
#endif

#include <array>
#include <cstddef>
#include <cstdlib>
#include <format>
#include <optional>
#include <random>
#include <span>
#include <string>
#include <string_view>
#include <utility>
#include <vector>

using namespace LiteNN;

namespace {

enum class TrainModelKind : std::size_t
{
	MLP128,
	MLP512,
};

struct TrainModelSpec
{
	std::string_view name;
	std::span<const std::size_t> hiddenSizes;
};

constexpr std::array<std::size_t, 1> kMLP128Hidden = { 128 };
constexpr std::array<std::size_t, 2> kMLP512Hidden = { 512, 256 };
constexpr std::array<TrainModelKind, 2> kTrainModelKinds = {
	TrainModelKind::MLP128,
	TrainModelKind::MLP512,
};
constexpr std::array<std::size_t, 3> kTrainBatchSizes = { 32, 128, 512 };
constexpr int kWarmupIterations = 5;

const TrainModelSpec& GetTrainModelSpec(TrainModelKind kind)
{
	static constexpr std::array<TrainModelSpec, 2> specs = {
		TrainModelSpec{ "MNIST-MLP128", kMLP128Hidden },
		TrainModelSpec{ "MNIST-MLP512", kMLP512Hidden },
	};
	return specs[static_cast<std::size_t>(kind)];
}

Layer::LinearLayer CreateLinear(Graph& graph, std::size_t inputSize, std::size_t outputSize, std::mt19937& rng)
{
	return Layer::CreateLinear(graph,
	    Initializer::XavierUniform({ inputSize, outputSize }, rng),
	    Initializer::Zeros({ 1, outputSize }));
}

Graph BuildMNISTMLPGraph(TrainModelKind kind, std::size_t batch, std::mt19937& rng)
{
	const auto& spec = GetTrainModelSpec(kind);
	Graph graph;
	std::vector<Layer::LinearLayer> layers;
	std::size_t inputSize = 784;
	for (const auto hiddenSize : spec.hiddenSizes)
	{
		layers.push_back(CreateLinear(graph, inputSize, hiddenSize, rng));
		inputSize = hiddenSize;
	}
	layers.push_back(CreateLinear(graph, inputSize, 10, rng));

	Subgraph forward;
	NodeOutput value{ forward.AddParam(DataType::Float32, { batch, 784 }), 0 };
	for (std::size_t i = 0; i + 1 < layers.size(); ++i)
	{
		value = Layer::AddReLU(forward, Layer::AddLinear(forward, layers[i], value));
	}
	forward.SetResults({ Layer::AddLinear(forward, layers.back(), value) });
	graph.SetForward(graph.AddSubgraph(std::move(forward)));
	graph.SetInputNames({ "image" });
	graph.SetOutputNames({ "logits" });
	return graph;
}

void OptimizeInferenceGraph(Graph& graph)
{
	InlinePass{}.Run(graph);
	ConstFoldPass{}.Run(graph);
	FusionPass{}.Run(graph);
}

Graph BuildTrainingGraph(TrainModelKind kind, std::size_t batch)
{
	std::mt19937 rng(42);
	auto graph = BuildMNISTMLPGraph(kind, batch, rng);
	AutogradPass{}.Run(graph);
	return graph;
}

Graph BuildInferenceGraph(TrainModelKind kind, std::size_t batch)
{
	std::mt19937 rng(42);
	auto graph = BuildMNISTMLPGraph(kind, batch, rng);
	OptimizeInferenceGraph(graph);
	return graph;
}

std::vector<float> MakeInputData(std::size_t batch)
{
	std::mt19937 rng(7);
	std::uniform_real_distribution<float> dist(-1.0f, 1.0f);
	std::vector<float> data(batch * 784);
	for (auto& value : data)
	{
		value = dist(rng);
	}
	return data;
}

std::vector<std::size_t> MakeTargets(std::size_t batch)
{
	std::vector<std::size_t> targets(batch);
	for (std::size_t i = 0; i < batch; ++i)
	{
		targets[i] = i % 10;
	}
	return targets;
}

std::vector<Tensor<CPU>> MakeCPUInputs(const std::vector<float>& inputData, std::size_t batch)
{
	std::vector<Tensor<CPU>> inputs;
	inputs.push_back(Optimizer::MakeFloatTensor(std::span<const float>(inputData), { batch, 784 }));
	return inputs;
}

#ifdef LITENN_ENABLE_CUDA
std::vector<Tensor<CUDA>> MakeCUDAInputs(const std::vector<float>& inputData, std::size_t batch)
{
	std::vector<Tensor<CUDA>> inputs;
	auto cpuInput = Optimizer::MakeFloatTensor(std::span<const float>(inputData), { batch, 784 });
	inputs.push_back(cpuInput.CopyToDevice(CUDA{}));
	return inputs;
}
#endif

void SetThroughputCounters(benchmark::State& state, std::size_t batch)
{
	state.SetItemsProcessed(static_cast<int64_t>(state.iterations()) * static_cast<int64_t>(batch));
	state.counters["samples_per_second"] = benchmark::Counter(
	    static_cast<double>(batch), benchmark::Counter::kIsIterationInvariantRate);
}

class ScopedEnvVar
{
public:
	ScopedEnvVar(const char* name, const char* value) : name_(name)
	{
		if (const char* oldValue = std::getenv(name))
		{
			oldValue_ = oldValue;
		}
		Set(value);
	}

	~ScopedEnvVar()
	{
		if (oldValue_.empty())
		{
			Unset();
		}
		else
		{
			Set(oldValue_.c_str());
		}
	}

	ScopedEnvVar(const ScopedEnvVar&) = delete;
	ScopedEnvVar& operator=(const ScopedEnvVar&) = delete;

private:
	void Set(const char* value) const
	{
#ifdef _WIN32
		_putenv_s(name_, value);
#else
		setenv(name_, value, 1);
#endif
	}

	void Unset() const
	{
#ifdef _WIN32
		_putenv_s(name_, "");
#else
		unsetenv(name_);
#endif
	}

	const char* name_{};
	std::string oldValue_;
};

std::vector<Tensor<CPU>> MakeBackwardInputs(std::span<const Tensor<CPU>> forwardInputs,
                                            std::span<const Tensor<CPU>> outputGradients)
{
	std::vector<Tensor<CPU>> backwardInputs;
	backwardInputs.reserve(forwardInputs.size() + outputGradients.size());
	for (const auto& input : forwardInputs)
	{
		backwardInputs.push_back(input);
	}
	for (const auto& gradient : outputGradients)
	{
		backwardInputs.push_back(gradient);
	}
	return backwardInputs;
}

void BMTrainCPUForward(benchmark::State& state, TrainModelKind kind, std::size_t batch)
{
	auto graph = BuildTrainingGraph(kind, batch);
	const auto inputData = MakeInputData(batch);
	auto inputs = MakeCPUInputs(inputData, batch);
	Runtime::Interpreter<CPU> interpreter;

	for (int i = 0; i < kWarmupIterations; ++i)
	{
		auto outputs = interpreter.RunForward(graph, inputs);
		benchmark::DoNotOptimize(outputs);
	}

	for (auto _ : state)
	{
		auto outputs = interpreter.RunForward(graph, inputs);
		benchmark::DoNotOptimize(outputs);
		benchmark::ClobberMemory();
	}
	SetThroughputCounters(state, batch);
}

void BMTrainCPUBackward(benchmark::State& state, TrainModelKind kind, std::size_t batch)
{
	auto graph = BuildTrainingGraph(kind, batch);
	const auto inputData = MakeInputData(batch);
	const auto targets = MakeTargets(batch);
	auto inputs = MakeCPUInputs(inputData, batch);
	Runtime::Interpreter<CPU> interpreter;
	auto outputs = interpreter.RunForward(graph, inputs);
	auto lossGradient = Optimizer::SoftmaxCrossEntropyWithLogitsBatch(outputs[0], targets);
	std::vector<Tensor<CPU>> outputGradients;
	outputGradients.push_back(std::move(lossGradient.gradient));
	auto backwardInputs = MakeBackwardInputs(inputs, outputGradients);

	for (int i = 0; i < kWarmupIterations; ++i)
	{
		auto backwardResults = interpreter.RunBackward(graph, backwardInputs);
		benchmark::DoNotOptimize(backwardResults);
	}

	for (auto _ : state)
	{
		auto backwardResults = interpreter.RunBackward(graph, backwardInputs);
		benchmark::DoNotOptimize(backwardResults);
		benchmark::ClobberMemory();
	}
	SetThroughputCounters(state, batch);
}

void BMTrainCPUOptimizerStep(benchmark::State& state, TrainModelKind kind, std::size_t batch)
{
	auto graph = BuildTrainingGraph(kind, batch);
	const auto inputData = MakeInputData(batch);
	const auto targets = MakeTargets(batch);
	auto inputs = MakeCPUInputs(inputData, batch);
	Runtime::Interpreter<CPU> interpreter;
	auto outputs = interpreter.RunForward(graph, inputs);
	auto lossGradient = Optimizer::SoftmaxCrossEntropyWithLogitsBatch(outputs[0], targets);
	std::vector<Tensor<CPU>> outputGradients;
	outputGradients.push_back(std::move(lossGradient.gradient));
	auto backwardInputs = MakeBackwardInputs(inputs, outputGradients);
	auto backwardResults = interpreter.RunBackward(graph, backwardInputs);
	const auto inputGradientCount = Optimizer::InferInputGradientCount(graph);
	Optimizer::SGD optimizer(Optimizer::SGDOptions{ .learningRate = 1.0e-3f });

	for (int i = 0; i < kWarmupIterations; ++i)
	{
		optimizer.Step(graph, backwardResults, inputGradientCount);
	}

	for (auto _ : state)
	{
		optimizer.Step(graph, backwardResults, inputGradientCount);
		benchmark::ClobberMemory();
	}
	SetThroughputCounters(state, batch);
}

void BMTrainCPUFullStep(benchmark::State& state, TrainModelKind kind, std::size_t batch)
{
	auto graph = BuildTrainingGraph(kind, batch);
	const auto inputData = MakeInputData(batch);
	const auto targets = MakeTargets(batch);
	auto inputs = MakeCPUInputs(inputData, batch);
	Training::CPUTrainer<Optimizer::SGD> trainer(
	    graph, Optimizer::SGD(Optimizer::SGDOptions{ .learningRate = 1.0e-3f }),
	    Training::TrainerOptions{ .buildBackwardIfMissing = false });

	for (int i = 0; i < kWarmupIterations; ++i)
	{
		auto result = trainer.StepSoftmaxCrossEntropyBatch(inputs, targets);
		benchmark::DoNotOptimize(result.loss);
	}

	for (auto _ : state)
	{
		auto result = trainer.StepSoftmaxCrossEntropyBatch(inputs, targets);
		benchmark::DoNotOptimize(result.loss);
		benchmark::DoNotOptimize(result.outputs);
		benchmark::DoNotOptimize(result.backwardResults);
		benchmark::ClobberMemory();
	}
	SetThroughputCounters(state, batch);
}

#ifdef LITENN_TRAIN_BENCH_HAS_AOT
std::vector<Tensor<CPU>> AllocateCPUOutputs(const CompiledModule<CPU>& module)
{
	std::vector<Tensor<CPU>> outputs;
	for (const auto& spec : module.OutputSpecs())
	{
		outputs.emplace_back(Uninitialized, ShapeView{ spec.shape }, spec.dtype, CPU{});
	}
	return outputs;
}

void BMTrainCPUAOTForwardConfigured(benchmark::State& state, TrainModelKind kind, std::size_t batch,
                                    const char* threadCount)
{
	std::optional<ScopedEnvVar> threadCountEnv;
	if (threadCount != nullptr)
	{
		threadCountEnv.emplace("LITENN_CPU_AOT_THREADS", threadCount);
	}

	auto graph = BuildInferenceGraph(kind, batch);
	auto module = Compiler<CPU>::Compile(graph);
	const auto inputData = MakeInputData(batch);
	auto inputs = MakeCPUInputs(inputData, batch);
	auto outputs = AllocateCPUOutputs(module);

	for (int i = 0; i < kWarmupIterations; ++i)
	{
		module.RunInto(inputs, outputs);
	}

	for (auto _ : state)
	{
		module.RunInto(inputs, outputs);
		benchmark::DoNotOptimize(outputs.data());
		benchmark::ClobberMemory();
	}
	SetThroughputCounters(state, batch);
}

void BMTrainCPUAOTForward(benchmark::State& state, TrainModelKind kind, std::size_t batch)
{
	BMTrainCPUAOTForwardConfigured(state, kind, batch, nullptr);
}

void BMTrainCPUAOTForwardT1(benchmark::State& state, TrainModelKind kind, std::size_t batch)
{
	BMTrainCPUAOTForwardConfigured(state, kind, batch, "1");
}

void BMTrainCPUAOTForwardT16(benchmark::State& state, TrainModelKind kind, std::size_t batch)
{
	BMTrainCPUAOTForwardConfigured(state, kind, batch, "16");
}

#ifdef LITENN_ENABLE_CUDA
std::vector<Tensor<CUDA>> AllocateCUDAOutputs(const CompiledModule<CUDA>& module)
{
	std::vector<Tensor<CUDA>> outputs;
	for (const auto& spec : module.OutputSpecs())
	{
		outputs.emplace_back(Uninitialized, ShapeView{ spec.shape }, spec.dtype, CUDA{});
	}
	return outputs;
}

void BMTrainCUDACPUFallbackForward(benchmark::State& state, TrainModelKind kind, std::size_t batch)
{
	if (!IsCUDADeviceAvailable())
	{
		state.SkipWithError("CUDA device is not available");
		return;
	}

	ScopedEnvVar disableNative("LITENN_CUDA_DISABLE_NATIVE_AOT", "1");
	auto graph = BuildInferenceGraph(kind, batch);
	auto module = Compiler<CUDA>::Compile(graph, CUDA{});
	if (module.Backend() != CompiledModuleBackend::CPUNative)
	{
		state.SkipWithError("expected CUDA CPU fallback backend");
		return;
	}

	const auto inputData = MakeInputData(batch);
	auto inputs = MakeCUDAInputs(inputData, batch);
	auto outputs = AllocateCUDAOutputs(module);

	for (int i = 0; i < kWarmupIterations; ++i)
	{
		module.RunInto(inputs, outputs);
	}

	for (auto _ : state)
	{
		module.RunInto(inputs, outputs);
		benchmark::DoNotOptimize(outputs.data());
		benchmark::ClobberMemory();
	}
	SetThroughputCounters(state, batch);
}

void BMTrainCUDANativeForward(benchmark::State& state, TrainModelKind kind, std::size_t batch)
{
	if (!IsCUDADeviceAvailable())
	{
		state.SkipWithError("CUDA device is not available");
		return;
	}

	auto graph = BuildInferenceGraph(kind, batch);
	auto module = Compiler<CUDA>::Compile(graph, CUDA{});
	if (module.Backend() != CompiledModuleBackend::CUDANative)
	{
		state.SkipWithError("expected CUDA native backend");
		return;
	}

	const auto inputData = MakeInputData(batch);
	auto inputs = MakeCUDAInputs(inputData, batch);
	auto outputs = AllocateCUDAOutputs(module);

	for (int i = 0; i < kWarmupIterations; ++i)
	{
		module.RunInto(inputs, outputs);
	}

	for (auto _ : state)
	{
		module.RunInto(inputs, outputs);
		benchmark::DoNotOptimize(outputs.data());
		benchmark::ClobberMemory();
	}
	SetThroughputCounters(state, batch);
}
#else
void BMTrainCUDACPUFallbackForward(benchmark::State& state, TrainModelKind, std::size_t)
{
	state.SkipWithError("LiteNN benchmark build has no CUDA support");
}

void BMTrainCUDANativeForward(benchmark::State& state, TrainModelKind, std::size_t)
{
	state.SkipWithError("LiteNN benchmark build has no CUDA support");
}
#endif
#endif

template <typename Fn>
void RegisterTrainBenchmark(std::string_view backend, std::string_view phase,
                            TrainModelKind kind, std::size_t batch, Fn&& fn)
{
	const auto& spec = GetTrainModelSpec(kind);
	auto* benchmarkCase = benchmark::RegisterBenchmark(
	    std::format("{}/{}/{}/batch:{}", backend, phase, spec.name, batch),
	    std::forward<Fn>(fn));
	benchmarkCase->UseRealTime()->Unit(benchmark::kMillisecond);
}

void RegisterTrainBenchmarks()
{
	for (const auto kind : kTrainModelKinds)
	{
		for (const auto batch : kTrainBatchSizes)
		{
			RegisterTrainBenchmark("TrainCPUInterpreter", "Forward", kind, batch,
			    [=](benchmark::State& state) { BMTrainCPUForward(state, kind, batch); });
			RegisterTrainBenchmark("TrainCPUInterpreter", "Backward", kind, batch,
			    [=](benchmark::State& state) { BMTrainCPUBackward(state, kind, batch); });
			RegisterTrainBenchmark("TrainCPUInterpreter", "OptimizerStep", kind, batch,
			    [=](benchmark::State& state) { BMTrainCPUOptimizerStep(state, kind, batch); });
			RegisterTrainBenchmark("TrainCPUInterpreter", "FullStep", kind, batch,
			    [=](benchmark::State& state) { BMTrainCPUFullStep(state, kind, batch); });

#ifdef LITENN_TRAIN_BENCH_HAS_AOT
			RegisterTrainBenchmark("TrainCPUAOT", "Forward", kind, batch,
			    [=](benchmark::State& state) { BMTrainCPUAOTForward(state, kind, batch); });
			RegisterTrainBenchmark("TrainCPUAOTT1", "Forward", kind, batch,
			    [=](benchmark::State& state) { BMTrainCPUAOTForwardT1(state, kind, batch); });
			RegisterTrainBenchmark("TrainCPUAOTT16", "Forward", kind, batch,
			    [=](benchmark::State& state) { BMTrainCPUAOTForwardT16(state, kind, batch); });
			RegisterTrainBenchmark("TrainCUDACPUFallback", "Forward", kind, batch,
			    [=](benchmark::State& state) { BMTrainCUDACPUFallbackForward(state, kind, batch); });
			RegisterTrainBenchmark("TrainCUDANative", "Forward", kind, batch,
			    [=](benchmark::State& state) { BMTrainCUDANativeForward(state, kind, batch); });
#endif
		}
	}
}

const bool kRegisteredTrainBenchmarks = [] {
	RegisterTrainBenchmarks();
	return true;
}();

} // namespace

BENCHMARK_MAIN();
