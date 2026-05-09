// LiteNN AOT 性能瓶颈分析工具
//
// 1) 测量 Run() 的分配开销 vs 纯 entry 调用开销
// 2) 将编译产物 (.o) 写到磁盘，便于用 objdump -d 反汇编查看是否向量化
// 3) 测量 Compile() 自身的耗时（一次性成本）

#include <LiteNN.h>
#include <LiteNN/Compiler/CompiledModule.h>
#include <LiteNN/Initializer/Initializer.h>
#include <LiteNN/Layer/Layer.h>
#include <LiteNN/Optimizer/Loss.h>
#include <LiteNN/Pass/ConstFoldPass.h>
#include <LiteNN/Pass/FusionPass.h>
#include <LiteNN/Pass/InlinePass.h>

#include <algorithm>
#include <chrono>
#include <filesystem>
#include <format>
#include <fstream>
#include <iostream>
#include <random>
#include <span>
#include <string>
#include <vector>

using namespace LiteNN;

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

static void Optimize(Graph& graph)
{
	InlinePass{}.Run(graph);
	ConstFoldPass{}.Run(graph);
	FusionPass{}.Run(graph);
}

namespace clk = std::chrono;
using Clock = clk::steady_clock;

struct Timing
{
	double meanMs;
	double throughput;
};

static Timing TimedRun(const CompiledModule<CPU>& module,
                       std::span<const Tensor<CPU>> inputs,
                       std::size_t batch)
{
	for (int i = 0; i < 5; ++i) (void)module.Run(inputs);
	auto t0 = Clock::now();
	(void)module.Run(inputs);
	auto t1 = Clock::now();
	const double probeMs = clk::duration<double, std::milli>(t1 - t0).count();
	const auto iters = static_cast<std::size_t>(
	    std::clamp(2000.0 / std::max(probeMs, 0.001), 10.0, 2000.0));
	auto a = Clock::now();
	for (std::size_t i = 0; i < iters; ++i) (void)module.Run(inputs);
	auto b = Clock::now();
	const double total = clk::duration<double, std::milli>(b - a).count();
	return { total / iters, batch * iters / (total * 1e-3) };
}

static Timing TimedRunInto(const CompiledModule<CPU>& module,
                           std::span<const Tensor<CPU>> inputs,
                           std::span<Tensor<CPU>> outputs,
                           std::size_t batch)
{
	for (int i = 0; i < 5; ++i) module.RunInto(inputs, outputs);
	auto t0 = Clock::now();
	module.RunInto(inputs, outputs);
	auto t1 = Clock::now();
	const double probeMs = clk::duration<double, std::milli>(t1 - t0).count();
	const auto iters = static_cast<std::size_t>(
	    std::clamp(2000.0 / std::max(probeMs, 0.001), 10.0, 2000.0));
	auto a = Clock::now();
	for (std::size_t i = 0; i < iters; ++i) module.RunInto(inputs, outputs);
	auto b = Clock::now();
	const double total = clk::duration<double, std::milli>(b - a).count();
	return { total / iters, batch * iters / (total * 1e-3) };
}

struct Case
{
	std::string name;
	Graph (*build)(std::size_t, std::mt19937&);
	std::vector<std::size_t> outShape;  // single-output models
};

int main(int argc, char** argv)
{
	const std::filesystem::path outDir = (argc >= 2)
	    ? std::filesystem::path(argv[1])
	    : std::filesystem::current_path() / "profile_out";
	std::filesystem::create_directories(outDir);

	std::cout << "LiteNN AOT Profile Report\n";
	std::cout << "Object files written to: " << outDir.string() << "\n";
	std::cout << std::string(86, '=') << "\n";

	std::vector<Case> cases = {
	    { "linear_b1",   BuildLinear,  { 1,   10  } },
	    { "linear_b32",  BuildLinear,  { 32,  10  } },
	    { "linear_b128", BuildLinear,  { 128, 10  } },
	    { "linear_b512", BuildLinear,  { 512, 10  } },
	    { "mlp128_b1",   BuildMLP128,  { 1,   10  } },
	    { "mlp128_b32",  BuildMLP128,  { 32,  10  } },
	    { "mlp128_b128", BuildMLP128,  { 128, 10  } },
	    { "mlp128_b512", BuildMLP128,  { 512, 10  } },
	    { "mlp512_b1",   BuildMLP512,  { 1,   10  } },
	    { "mlp512_b32",  BuildMLP512,  { 32,  10  } },
	    { "mlp512_b128", BuildMLP512,  { 128, 10  } },
	    { "mlp512_b512", BuildMLP512,  { 512, 10  } },
	};

	std::cout << std::format("{:<14} {:>8} {:>10} {:>12} {:>12} {:>10} {:>12}\n",
	    "Case", "Batch", "Compile/ms", "Run/ms", "RunInto/ms",
	    "Alloc/us", "Speedup");
	std::cout << std::string(86, '-') << "\n";

	for (const auto& c : cases)
	{
		const std::size_t batch = c.outShape[0];
		std::mt19937 rng(0);
		Graph g = c.build(batch, rng);
		Optimize(g);

		// Time compile
		auto cs = Clock::now();
		auto compiled = Compiler<CPU>::Compile(g);
		auto ce = Clock::now();
		const double compileMs = clk::duration<double, std::milli>(ce - cs).count();

		// Write the *raw* compiled object (the JIT-loaded code) for disassembly.
		// Note: WriteObjectFile() emits a "carrier" wrapper, not the executable code.
		try {
			const auto bytes = compiled.Instructions();
			std::ofstream f(outDir / (c.name + ".o"), std::ios::binary);
			f.write(reinterpret_cast<const char*>(bytes.data()),
			        static_cast<std::streamsize>(bytes.size()));
		} catch (...) {}

		// Build inputs once
		std::mt19937 rng2(0);
		std::uniform_real_distribution<float> dist(-1.0f, 1.0f);
		std::vector<float> data(batch * 784);
		for (auto& v : data) v = dist(rng2);
		auto in = Optimizer::MakeFloatTensor(std::span<const float>(data), { batch, 784 });
		std::vector<Tensor<CPU>> inputs; inputs.emplace_back(std::move(in));
		std::vector<Tensor<CPU>> outputs;
		outputs.emplace_back(Uninitialized,
		    ShapeView{ std::vector<std::size_t>{ batch, c.outShape[1] } },
		    DataType::Float32, CPU{});

		const auto tRun     = TimedRun(compiled, inputs, batch);
		const auto tRunInto = TimedRunInto(compiled, inputs, outputs, batch);

		const double allocUs = (tRun.meanMs - tRunInto.meanMs) * 1000.0;
		const double speedup = tRun.meanMs / std::max(tRunInto.meanMs, 1e-6);

		std::cout << std::format("{:<14} {:>8} {:>9.2f}ms {:>10.4f}ms {:>10.4f}ms {:>8.2f}us {:>10.2f}x\n",
		    c.name, batch, compileMs, tRun.meanMs, tRunInto.meanMs, allocUs, speedup);
	}

	std::cout << std::string(86, '=') << "\n";
	std::cout << "\nNext step: disassemble hot kernels:\n";
	std::cout << "  objdump -d -M intel " << (outDir / "linear_b512.o").string()
	          << " > linear_b512.s\n";
	std::cout << "  Then look for SIMD ops (vmovups/vmulps/vfmadd...).\n";
	return 0;
}
