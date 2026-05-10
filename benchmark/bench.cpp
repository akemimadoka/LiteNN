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

void SetThroughputCounters(benchmark::State& state, std::size_t batch)
{
	state.SetItemsProcessed(static_cast<int64_t>(state.iterations()) * static_cast<int64_t>(batch));
	state.counters["samples_per_second"] = benchmark::Counter(
	    static_cast<double>(batch), benchmark::Counter::kIsIterationInvariantRate);
}

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

void BMAOTRun(benchmark::State& state, ModelKind kind, std::size_t batch)
{
	std::mt19937 rng(42);
	auto graph = GetModelSpec(kind).build(batch, rng);
	Optimize(graph);

	auto compiled = Compiler<CPU>::Compile(graph);
	auto module = CompiledModule<CPU>::Load(compiled.Image());
	const auto inputData = MakeInputData(batch);
	auto inputs = MakeInputs(inputData, batch);

	for (auto _ : state)
	{
		auto outputs = module.Run(std::span<const Tensor<CPU>>(inputs));
		benchmark::DoNotOptimize(outputs);
		benchmark::ClobberMemory();
	}

	SetThroughputCounters(state, batch);
}

void BMAOTRunInto(benchmark::State& state, ModelKind kind, std::size_t batch)
{
	std::mt19937 rng(42);
	auto graph = GetModelSpec(kind).build(batch, rng);
	Optimize(graph);

	auto compiled = Compiler<CPU>::Compile(graph);
	auto module = CompiledModule<CPU>::Load(compiled.Image());
	const auto inputData = MakeInputData(batch);
	auto inputs = MakeInputs(inputData, batch);
	auto outputs = AllocateOutputs(module);

	for (auto _ : state)
	{
		module.RunInto(std::span<const Tensor<CPU>>(inputs), std::span<Tensor<CPU>>(outputs));
		benchmark::DoNotOptimize(outputs.data());
		benchmark::ClobberMemory();
	}

	SetThroughputCounters(state, batch);
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
#endif
		}
	}
}

const bool kRegisteredBenchmarks = [] {
	RegisterBenchmarks();
	return true;
}();

} // namespace

BENCHMARK_MAIN();
