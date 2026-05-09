// LiteNN CPU Inference Benchmark
//
// 测试三种模型在不同批次大小下的推理吞吐量，与 bench.py（PyTorch）结果对比。
//
// 模型：
//   Linear      784 → 10
//   MLP-128     784 → 128 → ReLU → 10
//   MLP-512     784 → 512 → ReLU → 256 → ReLU → 10
//
// 批次大小：1 / 32 / 128 / 512；自适应迭代次数（目标计时约 2 秒）。

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

#include <algorithm>
#include <chrono>
#include <format>
#include <iostream>
#include <random>
#include <span>
#include <string>
#include <string_view>
#include <vector>

using namespace LiteNN;

// ---------------------------------------------------------------------------
// Model builders — each returns an inference-ready Graph for the given batch
// ---------------------------------------------------------------------------

static Graph BuildLinear(std::size_t batch, std::mt19937& rng)
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

static Graph BuildMLP128(std::size_t batch, std::mt19937& rng)
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

static Graph BuildMLP512(std::size_t batch, std::mt19937& rng)
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

// ---------------------------------------------------------------------------
// Optimization: InlinePass → ConstFoldPass → FusionPass
// ---------------------------------------------------------------------------

static void Optimize(Graph& graph)
{
	InlinePass{}.Run(graph);
	ConstFoldPass{}.Run(graph);
	FusionPass{}.Run(graph);
}

// ---------------------------------------------------------------------------
// Single benchmark run
// ---------------------------------------------------------------------------

struct BenchResult
{
	double meanMs;    // milliseconds per forward call
	double throughput; // samples per second
};

struct BenchOptions
{
	bool useRunInto = false;
};

static BenchResult RunBench(Graph& graph, std::size_t batch)
{
	// Build a fixed random input (reused every call — measures kernel time, not alloc time)
	std::mt19937 rng(0);
	std::uniform_real_distribution<float> dist(-1.0f, 1.0f);
	std::vector<float> data(batch * 784);
	for (float& v : data)
		v = dist(rng);

	auto inputTensor = Optimizer::MakeFloatTensor(std::span<const float>(data), { batch, 784 });
	const std::vector<Tensor<CPU>> inputs = { std::move(inputTensor) };

	Runtime::Interpreter<CPU> interp;

	// Warmup (fixed 5 runs)
	for (std::size_t i = 0; i < 5; ++i)
		(void)interp.RunForward(graph, std::span<const Tensor<CPU>>(inputs));

	// Probe: 1 iteration to estimate per-call cost
	const auto tp0 = std::chrono::steady_clock::now();
	(void)interp.RunForward(graph, std::span<const Tensor<CPU>>(inputs));
	const auto tp1 = std::chrono::steady_clock::now();
	const double probeMs = std::chrono::duration<double, std::milli>(tp1 - tp0).count();

	// Target ~2 seconds total, at least 10 iters, at most 1000
	const auto iters = static_cast<std::size_t>(
	    std::clamp(2000.0 / std::max(probeMs, 0.001), 10.0, 1000.0));

	// Timed loop
	const auto t0 = std::chrono::steady_clock::now();
	for (std::size_t i = 0; i < iters; ++i)
		(void)interp.RunForward(graph, std::span<const Tensor<CPU>>(inputs));
	const auto t1 = std::chrono::steady_clock::now();

	const double totalMs = std::chrono::duration<double, std::milli>(t1 - t0).count();
	return {
		totalMs / static_cast<double>(iters),
		static_cast<double>(batch * iters) / (totalMs * 1e-3)
	};
}

// ---------------------------------------------------------------------------
// AOT benchmark — compile once, then measure module.Run()
// ---------------------------------------------------------------------------

#ifdef LITENN_BENCH_HAS_AOT
static std::vector<Tensor<CPU>> AllocateOutputs(const CompiledModule<CPU>& module)
{
	std::vector<Tensor<CPU>> outputs;
	outputs.reserve(module.OutputSpecs().size());
	for (const auto& spec : module.OutputSpecs())
		outputs.emplace_back(Uninitialized, ShapeView{ spec.shape }, spec.dtype, CPU{});
	return outputs;
}

static BenchResult RunBenchCompiled(const CompiledModule<CPU>& module, std::size_t batch,
                                    bool useRunInto)
{
	std::mt19937 rng(0);
	std::uniform_real_distribution<float> dist(-1.0f, 1.0f);
	std::vector<float> data(batch * 784);
	for (float& v : data)
		v = dist(rng);

	auto inputTensor = Optimizer::MakeFloatTensor(std::span<const float>(data), { batch, 784 });
	const std::vector<Tensor<CPU>> inputs = { std::move(inputTensor) };
	auto outputs = useRunInto ? AllocateOutputs(module) : std::vector<Tensor<CPU>>{};

	const auto runOnce = [&] {
		if (useRunInto)
			module.RunInto(std::span<const Tensor<CPU>>(inputs), std::span<Tensor<CPU>>(outputs));
		else
			(void)module.Run(std::span<const Tensor<CPU>>(inputs));
	};

	for (std::size_t i = 0; i < 5; ++i)
		runOnce();

	const auto tp0 = std::chrono::steady_clock::now();
	runOnce();
	const auto tp1 = std::chrono::steady_clock::now();
	const double probeMs = std::chrono::duration<double, std::milli>(tp1 - tp0).count();

	const auto iters = static_cast<std::size_t>(
	    std::clamp(2000.0 / std::max(probeMs, 0.001), 10.0, 1000.0));

	const auto t0 = std::chrono::steady_clock::now();
	for (std::size_t i = 0; i < iters; ++i)
		runOnce();
	const auto t1 = std::chrono::steady_clock::now();

	const double totalMs = std::chrono::duration<double, std::milli>(t1 - t0).count();
	return {
		totalMs / static_cast<double>(iters),
		static_cast<double>(batch * iters) / (totalMs * 1e-3)
	};
}
#endif // LITENN_BENCH_HAS_AOT

// ---------------------------------------------------------------------------
// main
// ---------------------------------------------------------------------------

int main(int argc, char** argv)
{
	BenchOptions options;
	for (int i = 1; i < argc; ++i)
	{
		const std::string_view arg(argv[i]);
		if (arg == "--use-runinto")
		{
			options.useRunInto = true;
		}
		else if (arg == "--help" || arg == "-h")
		{
			std::cout << "Usage: litenn_bench [--use-runinto]\n";
			return 0;
		}
		else
		{
			std::cerr << "Unknown argument: " << arg << "\n";
			std::cerr << "Usage: litenn_bench [--use-runinto]\n";
			return 2;
		}
	}

	const std::size_t batchSizes[] = { 1, 32, 128, 512 };

	std::cout << "LiteNN CPU Inference Benchmark\n";
#ifdef LITENN_BENCH_HAS_AOT
	if (options.useRunInto)
		std::cout << "AOT measurement mode: RunInto\n";
#endif
	std::cout << std::string(72, '=') << "\n";
	std::cout << std::format("{:<24} {:>6} {:>10} {:>14} {:>14}\n",
	    "Model", "Batch", "Backend", "ms/batch", "samples/sec");
	std::cout << std::string(72, '-') << "\n";

	for (const std::size_t batch : batchSizes)
	{
		std::mt19937 rng(42);

		auto run = [&](std::string_view name, Graph graph) {
			Optimize(graph);

			const auto ri = RunBench(graph, batch);
			std::cout << std::format("{:<24} {:>6} {:>10} {:>12.3f}ms {:>12.0f}/s\n",
			    name, batch, "Interp", ri.meanMs, ri.throughput);

#ifdef LITENN_BENCH_HAS_AOT
			auto compiled = Compiler<CPU>::Compile(graph);
			const auto module = CompiledModule<CPU>::Load(compiled.Image());
			const auto ra = RunBenchCompiled(module, batch, options.useRunInto);
			const auto aotBackend = options.useRunInto ? "AOTInto" : "AOT";
			std::cout << std::format("{:<24} {:>6} {:>10} {:>12.3f}ms {:>12.0f}/s\n",
			    name, batch, aotBackend, ra.meanMs, ra.throughput);
#endif
		};

		run("Linear(784->10)",        BuildLinear(batch, rng));
		run("MLP(784->128->10)",      BuildMLP128(batch, rng));
		run("MLP(784->512->256->10)", BuildMLP512(batch, rng));

		if (batch != 512)
			std::cout << "\n";
	}

	std::cout << std::string(72, '=') << "\n";
	return 0;
}
