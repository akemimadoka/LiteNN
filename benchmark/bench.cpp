#include <benchmark/benchmark.h>

#include <LiteNN.h>
#include <LiteNN/Initializer/Initializer.h>
#include <LiteNN/Layer/Layer.h>
#include <LiteNN/Optimizer/Loss.h>
#include <LiteNN/Pass/ConstFoldPass.h>
#include <LiteNN/Pass/FusionPass.h>
#include <LiteNN/Pass/InlinePass.h>
#include <LiteNN/Runtime/Interpreter.h>

#ifdef LITENN_BENCH_HAS_AOT
#include <LiteNN/Compiler/CompiledModule.h>
#endif

#include <array>
#include <cstddef>
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

enum class ModelKind : std::size_t
{
	Linear,
	MLP128,
	MLP512,
};

struct ModelSpec
{
	std::string_view name;
	Graph (*build)(std::size_t, std::mt19937&);
};

constexpr std::array<ModelKind, 3> kModelKinds = {
	ModelKind::Linear,
	ModelKind::MLP128,
	ModelKind::MLP512,
};

constexpr std::array<std::size_t, 4> kBatchSizes = { 1, 32, 128, 512 };
constexpr int kWarmupIterations = 5;

Graph BuildLinear(std::size_t batch, std::mt19937& rng)
{
	Graph graph;
	const auto fc = Layer::CreateLinear(graph,
	    Initializer::XavierUniform({ 784, 10 }, rng),
	    Initializer::Zeros({ 1, 10 }));
	Subgraph fwd;
	const auto in = fwd.AddParam(DataType::Float32, { batch, 784 });
	fwd.SetResults({ Layer::AddLinear(fwd, fc, { in, 0 }) });
	graph.SetForward(graph.AddSubgraph(std::move(fwd)));
	return graph;
}

Graph BuildMLP128(std::size_t batch, std::mt19937& rng)
{
	Graph graph;
	const auto h1 = Layer::CreateLinear(graph,
	    Initializer::XavierUniform({ 784, 128 }, rng),
	    Initializer::Zeros({ 1, 128 }));
	const auto h2 = Layer::CreateLinear(graph,
	    Initializer::XavierUniform({ 128, 10 }, rng),
	    Initializer::Zeros({ 1, 10 }));
	Subgraph fwd;
	const auto in = fwd.AddParam(DataType::Float32, { batch, 784 });
	const auto a1 = Layer::AddReLU(fwd, Layer::AddLinear(fwd, h1, { in, 0 }));
	fwd.SetResults({ Layer::AddLinear(fwd, h2, a1) });
	graph.SetForward(graph.AddSubgraph(std::move(fwd)));
	return graph;
}

Graph BuildMLP512(std::size_t batch, std::mt19937& rng)
{
	Graph graph;
	const auto h1 = Layer::CreateLinear(graph,
	    Initializer::XavierUniform({ 784, 512 }, rng),
	    Initializer::Zeros({ 1, 512 }));
	const auto h2 = Layer::CreateLinear(graph,
	    Initializer::XavierUniform({ 512, 256 }, rng),
	    Initializer::Zeros({ 1, 256 }));
	const auto h3 = Layer::CreateLinear(graph,
	    Initializer::XavierUniform({ 256, 10 }, rng),
	    Initializer::Zeros({ 1, 10 }));
	Subgraph fwd;
	const auto in = fwd.AddParam(DataType::Float32, { batch, 784 });
	const auto a1 = Layer::AddReLU(fwd, Layer::AddLinear(fwd, h1, { in, 0 }));
	const auto a2 = Layer::AddReLU(fwd, Layer::AddLinear(fwd, h2, a1));
	fwd.SetResults({ Layer::AddLinear(fwd, h3, a2) });
	graph.SetForward(graph.AddSubgraph(std::move(fwd)));
	return graph;
}

void Optimize(Graph& graph)
{
	InlinePass{}.Run(graph);
	ConstFoldPass{}.Run(graph);
	FusionPass{}.Run(graph);
}

const ModelSpec& GetModelSpec(ModelKind kind)
{
	static const std::array<ModelSpec, 3> specs = {
		ModelSpec{ "Linear(784->10)", &BuildLinear },
		ModelSpec{ "MLP(784->128->10)", &BuildMLP128 },
		ModelSpec{ "MLP(784->512->256->10)", &BuildMLP512 },
	};
	return specs[static_cast<std::size_t>(kind)];
}

std::vector<float> MakeInputData(std::size_t batch)
{
	std::mt19937 rng(0);
	std::uniform_real_distribution<float> dist(-1.0f, 1.0f);
	std::vector<float> data(batch * 784);
	for (float& value : data)
		value = dist(rng);
	return data;
}

std::vector<Tensor<CPU>> MakeInputs(const std::vector<float>& data, std::size_t batch)
{
	std::vector<Tensor<CPU>> inputs;
	inputs.emplace_back(Optimizer::MakeFloatTensor(std::span<const float>(data), { batch, 784 }));
	return inputs;
}

#ifdef LITENN_ENABLE_CUDA
struct TensorInputSpec
{
	std::vector<float> values;
	std::vector<std::size_t> shape;
};

std::vector<Tensor<CUDA>> MakeCUDAInputs(const std::vector<float>& data, std::size_t batch)
{
	std::vector<Tensor<CUDA>> inputs;
	auto cpuInput = Optimizer::MakeFloatTensor(std::span<const float>(data), { batch, 784 });
	inputs.push_back(cpuInput.CopyToDevice(CUDA{}));
	return inputs;
}

Graph BuildNativeMatMul(std::size_t batch, std::size_t width)
{
	Graph graph;
	Subgraph fwd;
	const auto lhs = fwd.AddParam(DataType::Float32, { batch, width });
	const auto rhs = fwd.AddParam(DataType::Float32, { width, width });
	const auto out = fwd.AddNode(BinaryOpNode{ BinaryOp::MatMul, { lhs, 0 }, { rhs, 0 } },
	                            { OutputInfo{ DataType::Float32, { batch, width } } });
	fwd.SetResults({ { out, 0 } });
	graph.SetForward(graph.AddSubgraph(std::move(fwd)));
	return graph;
}

std::vector<TensorInputSpec> MakeNativeMatMulInputs(std::size_t batch, std::size_t width)
{
	std::mt19937 rng(7);
	std::uniform_real_distribution<float> dist(-1.0f, 1.0f);
	std::vector<float> lhs(batch * width);
	std::vector<float> rhs(width * width);
	for (auto& value : lhs)
		value = dist(rng);
	for (auto& value : rhs)
		value = dist(rng);

	std::vector<TensorInputSpec> specs;
	specs.push_back(TensorInputSpec{ .values = std::move(lhs), .shape = { batch, width } });
	specs.push_back(TensorInputSpec{ .values = std::move(rhs), .shape = { width, width } });
	return specs;
}

std::vector<Tensor<CUDA>> MakeCUDAInputs(std::span<const TensorInputSpec> specs)
{
	std::vector<Tensor<CUDA>> inputs;
	inputs.reserve(specs.size());
	for (const auto& spec : specs)
	{
		auto cpuInput = Optimizer::MakeFloatTensor(std::span<const float>(spec.values), ShapeView{ spec.shape });
		inputs.push_back(cpuInput.CopyToDevice(CUDA{}));
	}
	return inputs;
}
#endif

void SetThroughputCounters(benchmark::State& state, std::size_t batch)
{
	state.SetItemsProcessed(static_cast<int64_t>(state.iterations()) * static_cast<int64_t>(batch));
	state.counters["samples_per_second"] = benchmark::Counter(
	    static_cast<double>(batch), benchmark::Counter::kIsIterationInvariantRate);
}

#ifdef LITENN_ENABLE_CUDA
std::vector<double> MakeCUDADeviceMatMulData(std::size_t count, DataType dtype)
{
	std::vector<double> values(count);
	if (dtype == DataType::Int8 || dtype == DataType::UInt8)
	{
		for (auto i = 0uz; i < values.size(); ++i)
			values[i] = static_cast<double>(i % 3);
		return values;
	}

	std::mt19937 rng(11);
	std::uniform_real_distribution<double> dist(-1.0, 1.0);
	for (auto& value : values)
		value = dist(rng);
	return values;
}

void BMCUDADeviceMatMul(benchmark::State& state, std::size_t batch, std::size_t width, DataType dtype)
{
	if (!IsCUDADeviceAvailable())
	{
		state.SkipWithError("CUDA device is not available");
		return;
	}
	if (dtype != DataType::Float32 && dtype != DataType::Float64 && !CUDASupportsLowPrecisionStorage(dtype))
	{
		state.SkipWithError("CUDA device does not support requested dtype storage");
		return;
	}

	const auto lhsData = MakeCUDADeviceMatMulData(batch * width, dtype);
	const auto rhsData = MakeCUDADeviceMatMulData(width * width, dtype);
	Tensor<CPU> lhsCpu(std::span<const double>(lhsData), { batch, width }, dtype);
	Tensor<CPU> rhsCpu(std::span<const double>(rhsData), { width, width }, dtype);
	auto lhs = lhsCpu.CopyToDevice(CUDA{});
	auto rhs = rhsCpu.CopyToDevice(CUDA{});

	for (int i = 0; i < kWarmupIterations; ++i)
	{
		auto output = lhs.MatMul(rhs);
		benchmark::DoNotOptimize(output.RawData());
	}

	for (auto _ : state)
	{
		auto output = lhs.MatMul(rhs);
		benchmark::DoNotOptimize(output.RawData());
		benchmark::ClobberMemory();
	}

	SetThroughputCounters(state, batch);
}
#endif

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

template <typename Fn>
void RegisterBenchmarkCase(std::string_view backend, ModelKind kind, std::size_t batch, Fn&& fn)
{
	auto* benchmarkCase = benchmark::RegisterBenchmark(
	    std::format("{}/{}/batch:{}", backend, GetModelSpec(kind).name, batch),
	    std::forward<Fn>(fn));
	benchmarkCase->UseRealTime()->Unit(benchmark::kMillisecond);
}

void BMInterpreter(benchmark::State& state, ModelKind kind, std::size_t batch)
{
	std::mt19937 rng(42);
	auto graph = GetModelSpec(kind).build(batch, rng);
	Optimize(graph);

	const auto inputData = MakeInputData(batch);
	auto inputs = MakeInputs(inputData, batch);
	Runtime::Interpreter<CPU> interp;

	for (int i = 0; i < kWarmupIterations; ++i)
	{
		auto outputs = interp.RunForward(graph, std::span<const Tensor<CPU>>(inputs));
		benchmark::DoNotOptimize(outputs);
	}

	for (auto _ : state)
	{
		auto outputs = interp.RunForward(graph, std::span<const Tensor<CPU>>(inputs));
		benchmark::DoNotOptimize(outputs);
		benchmark::ClobberMemory();
	}

	SetThroughputCounters(state, batch);
}

#ifdef LITENN_BENCH_HAS_AOT
std::vector<Tensor<CPU>> AllocateOutputs(const CompiledModule<CPU>& module)
{
	std::vector<Tensor<CPU>> outputs;
	outputs.reserve(module.OutputSpecs().size());
	for (const auto& spec : module.OutputSpecs())
		outputs.emplace_back(Uninitialized, ShapeView{ spec.shape }, spec.dtype, CPU{});
	return outputs;
}

#ifdef LITENN_ENABLE_CUDA
std::vector<Tensor<CUDA>> AllocateCUDAOutputs(const CompiledModule<CUDA>& module)
{
	std::vector<Tensor<CUDA>> outputs;
	outputs.reserve(module.OutputSpecs().size());
	for (const auto& spec : module.OutputSpecs())
		outputs.emplace_back(Uninitialized, ShapeView{ spec.shape }, spec.dtype, CUDA{});
	return outputs;
}

void BMCUDACPUFallbackRunInto(benchmark::State& state, ModelKind kind, std::size_t batch)
{
	if (!IsCUDADeviceAvailable())
	{
		state.SkipWithError("CUDA device is not available");
		return;
	}

	ScopedEnvVar disableNative("LITENN_CUDA_DISABLE_NATIVE_AOT", "1");
	std::mt19937 rng(42);
	auto graph = GetModelSpec(kind).build(batch, rng);
	Optimize(graph);
	auto module = Compiler<CUDA>::Compile(graph, CUDA{});
	if (module.Backend() != CompiledModuleBackend::CPUNative)
	{
		state.SkipWithError("expected CUDA CPU-bridge backend for model benchmark");
		return;
	}

	const auto inputData = MakeInputData(batch);
	auto inputs = MakeCUDAInputs(inputData, batch);
	auto outputs = AllocateCUDAOutputs(module);

	for (int i = 0; i < kWarmupIterations; ++i)
	{
		module.RunInto(std::span<const Tensor<CUDA>>(inputs), std::span<Tensor<CUDA>>(outputs));
	}

	for (auto _ : state)
	{
		module.RunInto(std::span<const Tensor<CUDA>>(inputs), std::span<Tensor<CUDA>>(outputs));
		benchmark::DoNotOptimize(outputs.data());
		benchmark::ClobberMemory();
	}

	SetThroughputCounters(state, batch);
}

void BMCUDANativeModelRunInto(benchmark::State& state, ModelKind kind, std::size_t batch)
{
	if (!IsCUDADeviceAvailable())
	{
		state.SkipWithError("CUDA device is not available");
		return;
	}

	std::mt19937 rng(42);
	auto graph = GetModelSpec(kind).build(batch, rng);
	Optimize(graph);
	auto module = Compiler<CUDA>::Compile(graph, CUDA{});
	if (module.Backend() != CompiledModuleBackend::CUDANative)
	{
		state.SkipWithError("expected CUDA native backend for model benchmark");
		return;
	}

	const auto inputData = MakeInputData(batch);
	auto inputs = MakeCUDAInputs(inputData, batch);
	auto outputs = AllocateCUDAOutputs(module);

	for (int i = 0; i < kWarmupIterations; ++i)
	{
		module.RunInto(std::span<const Tensor<CUDA>>(inputs), std::span<Tensor<CUDA>>(outputs));
	}

	for (auto _ : state)
	{
		module.RunInto(std::span<const Tensor<CUDA>>(inputs), std::span<Tensor<CUDA>>(outputs));
		benchmark::DoNotOptimize(outputs.data());
		benchmark::ClobberMemory();
	}

	SetThroughputCounters(state, batch);
}

void BMCUDANativeGraphModelRunInto(benchmark::State& state, ModelKind kind, std::size_t batch)
{
	ScopedEnvVar enableGraphReplay("LITENN_CUDA_ENABLE_GRAPH_REPLAY", "1");
	BMCUDANativeModelRunInto(state, kind, batch);
}

void BMCUDANativeMatMulRunInto(benchmark::State& state, std::size_t batch, std::size_t width)
{
	if (!IsCUDADeviceAvailable())
	{
		state.SkipWithError("CUDA device is not available");
		return;
	}

	auto graph = BuildNativeMatMul(batch, width);
	auto module = Compiler<CUDA>::Compile(graph, CUDA{});
	if (module.Backend() != CompiledModuleBackend::CUDANative)
	{
		state.SkipWithError("expected CUDA native backend for MatMul benchmark");
		return;
	}

	auto specs = MakeNativeMatMulInputs(batch, width);
	auto inputs = MakeCUDAInputs(specs);
	auto outputs = AllocateCUDAOutputs(module);

	for (int i = 0; i < kWarmupIterations; ++i)
	{
		module.RunInto(std::span<const Tensor<CUDA>>(inputs), std::span<Tensor<CUDA>>(outputs));
	}

	for (auto _ : state)
	{
		module.RunInto(std::span<const Tensor<CUDA>>(inputs), std::span<Tensor<CUDA>>(outputs));
		benchmark::DoNotOptimize(outputs.data());
		benchmark::ClobberMemory();
	}

	SetThroughputCounters(state, batch);
}
#else
void BMCUDACPUFallbackRunInto(benchmark::State& state, ModelKind, std::size_t)
{
	state.SkipWithError("LiteNN benchmark build has no CUDA support");
}

void BMCUDANativeMatMulRunInto(benchmark::State& state, std::size_t, std::size_t)
{
	state.SkipWithError("LiteNN benchmark build has no CUDA support");
}

void BMCUDANativeModelRunInto(benchmark::State& state, ModelKind, std::size_t)
{
	state.SkipWithError("LiteNN benchmark build has no CUDA support");
}

void BMCUDANativeGraphModelRunInto(benchmark::State& state, ModelKind, std::size_t)
{
	state.SkipWithError("LiteNN benchmark build has no CUDA support");
}
#endif

void BMAOTRun(benchmark::State& state, ModelKind kind, std::size_t batch)
{
	ScopedEnvVar disableFastPath("LITENN_CPU_AOT_LINEAR_CHAIN_FASTPATH", "0");
	std::mt19937 rng(42);
	auto graph = GetModelSpec(kind).build(batch, rng);
	Optimize(graph);

	auto compiled = Compiler<CPU>::Compile(graph);
	auto module = CompiledModule<CPU>::Load(compiled.Image());
	const auto inputData = MakeInputData(batch);
	auto inputs = MakeInputs(inputData, batch);

	for (int i = 0; i < kWarmupIterations; ++i)
	{
		auto outputs = module.Run(std::span<const Tensor<CPU>>(inputs));
		benchmark::DoNotOptimize(outputs);
	}

	for (auto _ : state)
	{
		auto outputs = module.Run(std::span<const Tensor<CPU>>(inputs));
		benchmark::DoNotOptimize(outputs);
		benchmark::ClobberMemory();
	}

	SetThroughputCounters(state, batch);
}

void BMAOTRunIntoConfigured(benchmark::State& state, ModelKind kind, std::size_t batch,
                            bool enableFastPath, const char* threadCount)
{
	ScopedEnvVar fastPathEnv("LITENN_CPU_AOT_LINEAR_CHAIN_FASTPATH", enableFastPath ? "1" : "0");
	std::optional<ScopedEnvVar> threadCountEnv;
	if (threadCount != nullptr)
	{
		threadCountEnv.emplace("LITENN_CPU_AOT_THREADS", threadCount);
	}

	std::mt19937 rng(42);
	auto graph = GetModelSpec(kind).build(batch, rng);
	Optimize(graph);

	auto compiled = Compiler<CPU>::Compile(graph);
	auto module = CompiledModule<CPU>::Load(compiled.Image());
	const auto inputData = MakeInputData(batch);
	auto inputs = MakeInputs(inputData, batch);
	auto outputs = AllocateOutputs(module);

	for (int i = 0; i < kWarmupIterations; ++i)
	{
		module.RunInto(std::span<const Tensor<CPU>>(inputs), std::span<Tensor<CPU>>(outputs));
	}

	for (auto _ : state)
	{
		module.RunInto(std::span<const Tensor<CPU>>(inputs), std::span<Tensor<CPU>>(outputs));
		benchmark::DoNotOptimize(outputs.data());
		benchmark::ClobberMemory();
	}

	SetThroughputCounters(state, batch);
}

void BMAOTRunInto(benchmark::State& state, ModelKind kind, std::size_t batch)
{
	BMAOTRunIntoConfigured(state, kind, batch, false, nullptr);
}

void BMAOTFastPathRunIntoT1(benchmark::State& state, ModelKind kind, std::size_t batch)
{
	BMAOTRunIntoConfigured(state, kind, batch, true, "1");
}

void BMAOTFastPathRunIntoT16(benchmark::State& state, ModelKind kind, std::size_t batch)
{
	BMAOTRunIntoConfigured(state, kind, batch, true, "16");
}
#endif

void RegisterBenchmarks()
{
	for (const auto kind : kModelKinds)
	{
		for (const auto batch : kBatchSizes)
		{
			RegisterBenchmarkCase("Interpreter", kind, batch,
			    [=](benchmark::State& state) { BMInterpreter(state, kind, batch); });
#ifdef LITENN_BENCH_HAS_AOT
			RegisterBenchmarkCase("AOTRun", kind, batch,
			    [=](benchmark::State& state) { BMAOTRun(state, kind, batch); });
			RegisterBenchmarkCase("AOTRunInto", kind, batch,
			    [=](benchmark::State& state) { BMAOTRunInto(state, kind, batch); });
			RegisterBenchmarkCase("AOTFastPathRunIntoT1", kind, batch,
			    [=](benchmark::State& state) { BMAOTFastPathRunIntoT1(state, kind, batch); });
			RegisterBenchmarkCase("AOTFastPathRunIntoT16", kind, batch,
			    [=](benchmark::State& state) { BMAOTFastPathRunIntoT16(state, kind, batch); });
			RegisterBenchmarkCase("CUDACPUFallbackRunInto", kind, batch,
			    [=](benchmark::State& state) { BMCUDACPUFallbackRunInto(state, kind, batch); });
			RegisterBenchmarkCase("CUDANativeRunInto", kind, batch,
			    [=](benchmark::State& state) { BMCUDANativeModelRunInto(state, kind, batch); });
			RegisterBenchmarkCase("CUDANativeGraphRunInto", kind, batch,
			    [=](benchmark::State& state) { BMCUDANativeGraphModelRunInto(state, kind, batch); });
#endif
		}
	}

#ifdef LITENN_BENCH_HAS_AOT
	constexpr std::size_t nativeMatMulWidth = 128;
	for (const auto batch : kBatchSizes)
	{
		auto* benchmarkCase = benchmark::RegisterBenchmark(
		    std::format("CUDANativeMatMul/batch:{}/width:{}", batch, nativeMatMulWidth),
		    [=](benchmark::State& state) { BMCUDANativeMatMulRunInto(state, batch, nativeMatMulWidth); });
		benchmarkCase->UseRealTime()->Unit(benchmark::kMillisecond);
	}
#endif

#ifdef LITENN_ENABLE_CUDA
	constexpr std::size_t cudaDeviceMatMulWidth = 128;
	constexpr std::array cudaDeviceMatMulDTypes{
		DataType::Float32,
		DataType::Float16,
		DataType::BFloat16,
		DataType::Float8E4M3,
		DataType::Float8E5M2,
		DataType::Int8,
		DataType::UInt8,
	};
	for (const auto batch : kBatchSizes)
	{
		for (const auto dtype : cudaDeviceMatMulDTypes)
		{
			auto* benchmarkCase = benchmark::RegisterBenchmark(
			    std::format("CUDADeviceMatMul/{}/batch:{}/width:{}", DataTypeName(dtype), batch,
			                cudaDeviceMatMulWidth),
			    [=](benchmark::State& state) { BMCUDADeviceMatMul(state, batch, cudaDeviceMatMulWidth, dtype); });
			benchmarkCase->UseRealTime()->Unit(benchmark::kMillisecond);
		}
	}
#endif
}

const bool kRegisteredBenchmarks = [] {
	RegisterBenchmarks();
	return true;
}();

} // namespace

BENCHMARK_MAIN();
