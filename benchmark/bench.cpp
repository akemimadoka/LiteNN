#include <benchmark/benchmark.h>

#include <LiteNN.h>
#include <LiteNN/Initializer/Initializer.h>
#include <LiteNN/Layer/Layer.h>
#include <LiteNN/Optimizer/Loss.h>
#include <LiteNN/Pass/ConstFoldPass.h>
#include <LiteNN/Pass/FusionPass.h>
#include <LiteNN/Pass/InlinePass.h>
#include <LiteNN/Runtime/Interpreter.h>

#include <ggml-cpp.h>
#include <ggml-cpu.h>

#ifdef LITENN_BENCH_HAS_AOT
#include <LiteNN/Compiler/CompiledModule.h>
#endif

#include <array>
#include <cmath>
#include <cstddef>
#include <format>
#include <optional>
#include <random>
#include <span>
#include <stdexcept>
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
constexpr std::array<int, 2> kGGMLThreadCounts = { 1, 16 };
constexpr int kWarmupIterations = 5;

struct GGMLLayerSpec
{
	std::size_t inputWidth;
	std::size_t outputWidth;
	bool relu;
};

constexpr std::array<GGMLLayerSpec, 1> kGGMLLinearLayers = {
	GGMLLayerSpec{ 784, 10, false },
};

constexpr std::array<GGMLLayerSpec, 2> kGGMLMLP128Layers = {
	GGMLLayerSpec{ 784, 128, true },
	GGMLLayerSpec{ 128, 10, false },
};

constexpr std::array<GGMLLayerSpec, 3> kGGMLMLP512Layers = {
	GGMLLayerSpec{ 784, 512, true },
	GGMLLayerSpec{ 512, 256, true },
	GGMLLayerSpec{ 256, 10, false },
};

void SetThroughputCounters(benchmark::State& state, std::size_t batch);

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

std::span<const GGMLLayerSpec> GetGGMLLayerSpecs(ModelKind kind)
{
	switch (kind)
	{
	case ModelKind::Linear:
		return kGGMLLinearLayers;
	case ModelKind::MLP128:
		return kGGMLMLP128Layers;
	case ModelKind::MLP512:
		return kGGMLMLP512Layers;
	}
	throw std::invalid_argument("unsupported GGML benchmark model kind");
}

std::vector<float> MakeXavierUniform(std::size_t inputWidth, std::size_t outputWidth, std::mt19937& rng)
{
	const auto limit = std::sqrt(6.0f / static_cast<float>(inputWidth + outputWidth));
	std::uniform_real_distribution<float> dist(-limit, limit);
	std::vector<float> values(inputWidth * outputWidth);
	for (auto& value : values)
		value = dist(rng);
	return values;
}

void UploadGGMLTensor(struct ggml_tensor* tensor, std::span<const float> values)
{
	ggml_backend_tensor_set(tensor, values.data(), 0, values.size() * sizeof(float));
}

class GGMLModelRunner
{
public:
	GGMLModelRunner(ModelKind kind, std::size_t batch, int threadCount)
	    : graphBuffer_(ggml_tensor_overhead() * GGML_DEFAULT_GRAPH_SIZE + ggml_graph_overhead())
	{
		if (threadCount <= 0)
			throw std::invalid_argument("GGML thread count must be positive");

		backend_.reset(ggml_backend_cpu_init());
		if (!backend_)
			throw std::runtime_error("ggml_backend_cpu_init failed");
		ggml_backend_cpu_set_n_threads(backend_.get(), threadCount);

		const auto layerSpecs = GetGGMLLayerSpecs(kind);
		const auto tensorCount = 1 + static_cast<int>(layerSpecs.size() * 2);
		tensorContext_.reset(ggml_init({
		    .mem_size = ggml_tensor_overhead() * tensorCount,
		    .mem_buffer = nullptr,
		    .no_alloc = true,
		}));
		if (!tensorContext_)
			throw std::runtime_error("ggml_init failed for tensor context");

		input_ = ggml_new_tensor_2d(tensorContext_.get(), GGML_TYPE_F32, 784, batch);
		weights_.reserve(layerSpecs.size());
		biases_.reserve(layerSpecs.size());
		for (const auto& layer : layerSpecs)
		{
			weights_.push_back(
			    ggml_new_tensor_2d(tensorContext_.get(), GGML_TYPE_F32, layer.inputWidth, layer.outputWidth));
			biases_.push_back(ggml_new_tensor_2d(tensorContext_.get(), GGML_TYPE_F32, layer.outputWidth, 1));
		}

		tensorBuffer_.reset(ggml_backend_alloc_ctx_tensors(tensorContext_.get(), backend_.get()));
		if (!tensorBuffer_)
			throw std::runtime_error("ggml_backend_alloc_ctx_tensors failed");

		UploadGGMLTensor(input_, MakeInputData(batch));
		std::mt19937 rng(42);
		for (auto i = 0uz; i < layerSpecs.size(); ++i)
		{
			const auto& layer = layerSpecs[i];
			UploadGGMLTensor(weights_[i], MakeXavierUniform(layer.inputWidth, layer.outputWidth, rng));
			UploadGGMLTensor(biases_[i], std::vector<float>(layer.outputWidth, 0.0f));
		}

		graphContext_.reset(ggml_init({
		    .mem_size = graphBuffer_.size(),
		    .mem_buffer = graphBuffer_.data(),
		    .no_alloc = true,
		}));
		if (!graphContext_)
			throw std::runtime_error("ggml_init failed for graph context");

		graph_ = ggml_new_graph(graphContext_.get());
		if (!graph_)
			throw std::runtime_error("ggml_new_graph failed");

		auto* current = input_;
		for (auto i = 0uz; i < layerSpecs.size(); ++i)
			current = AddLinearLayer(graphContext_.get(), current, weights_[i], biases_[i], layerSpecs[i].relu);

		ggml_build_forward_expand(graph_, current);
		result_ = ggml_graph_node(graph_, -1);
		allocator_.reset(ggml_gallocr_new(ggml_backend_get_default_buffer_type(backend_.get())));
		if (!allocator_)
			throw std::runtime_error("ggml_gallocr_new failed");
		if (!ggml_gallocr_alloc_graph(allocator_.get(), graph_))
			throw std::runtime_error("ggml_gallocr_alloc_graph failed");
	}

	bool Run() const
	{
		return ggml_backend_graph_compute(backend_.get(), graph_) == GGML_STATUS_SUCCESS;
	}

	const void* ResultData() const
	{
		return ggml_get_data(result_);
	}

private:
	static struct ggml_tensor* AddLinearLayer(struct ggml_context* ctx, struct ggml_tensor* input,
	                                          struct ggml_tensor* weight, struct ggml_tensor* bias, bool relu)
	{
		auto* linear = ggml_mul_mat(ctx, weight, input);
		auto* shifted = ggml_add(ctx, linear, ggml_repeat(ctx, bias, linear));
		return relu ? ggml_relu(ctx, shifted) : shifted;
	}

	ggml_backend_ptr backend_;
	ggml_gallocr_ptr allocator_;
	ggml_context_ptr tensorContext_;
	ggml_backend_buffer_ptr tensorBuffer_;
	std::vector<std::byte> graphBuffer_;
	ggml_context_ptr graphContext_;
	struct ggml_cgraph* graph_{};
	struct ggml_tensor* input_{};
	std::vector<struct ggml_tensor*> weights_;
	std::vector<struct ggml_tensor*> biases_;
	struct ggml_tensor* result_{};
};

void BMLlamaCppGGML(benchmark::State& state, ModelKind kind, std::size_t batch, int threadCount)
{
	try
	{
		GGMLModelRunner runner(kind, batch, threadCount);

		for (int i = 0; i < kWarmupIterations; ++i)
		{
			if (!runner.Run())
			{
				state.SkipWithError("llama.cpp ggml graph compute failed during warmup");
				return;
			}
			benchmark::DoNotOptimize(runner.ResultData());
		}

		for (auto _ : state)
		{
			if (!runner.Run())
			{
				state.SkipWithError("llama.cpp ggml graph compute failed");
				return;
			}
			benchmark::DoNotOptimize(runner.ResultData());
			benchmark::ClobberMemory();
		}

		SetThroughputCounters(state, batch);
	}
	catch (const std::exception& ex)
	{
		state.SkipWithError(ex.what());
	}
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
			for (const auto threadCount : kGGMLThreadCounts)
			{
				RegisterBenchmarkCase(std::format("LlamaCppGGMLT{}", threadCount), kind, batch,
				    [=](benchmark::State& state) { BMLlamaCppGGML(state, kind, batch, threadCount); });
			}
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
