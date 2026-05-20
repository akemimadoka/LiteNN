// LiteNN AOT 性能瓶颈分析工具
//
// 1) 测量 Run() 的分配开销 vs 纯 entry 调用开销
// 2) 将编译产物 (.o) 写到磁盘，并用 objdump 生成 first-class 指令统计
// 3) 测量 Compile() 自身的耗时（一次性成本）

#include <LiteNN.h>
#include <LiteNN/Compiler/CompiledModule.h>
#ifdef LITENN_ENABLE_CUDA
#include <LiteNN/Compiler/CUDANativePayload.h>
#endif
#include <LiteNN/Initializer/Initializer.h>
#include <LiteNN/Layer/Layer.h>
#include <LiteNN/Optimizer/Loss.h>
#include <LiteNN/Pass/ConstFoldPass.h>
#include <LiteNN/Pass/FusionPass.h>
#include <LiteNN/Pass/InlinePass.h>

#include <algorithm>
#include <chrono>
#include <cctype>
#include <cstdlib>
#include <filesystem>
#include <format>
#include <fstream>
#include <iostream>
#include <optional>
#include <random>
#include <span>
#include <stdexcept>
#include <string>
#include <string_view>
#include <utility>
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

struct InstructionStats
{
	bool available{};
	std::string message;
	std::string function;
	std::size_t lines{};
	std::size_t packedFMA{};
	std::size_t scalarFMA{};
	std::size_t zmmPackedFMA{};
	std::size_t ymmPackedFMA{};
	std::size_t xmmPackedFMA{};
	std::size_t gather{};
	std::size_t scatter{};
	std::size_t stackVectorOp{};
	std::size_t vectorLoad{};
	std::size_t scalarMove{};
	std::size_t broadcast{};
	std::size_t prefetch{};
};

struct CaseInstructionStats
{
	std::string name;
	std::filesystem::path asmPath;
	InstructionStats stats;
};

class ScopedEnvVar
{
public:
	ScopedEnvVar(const char* name, const char* value) : name_(name)
	{
		if (const char* current = std::getenv(name))
		{
			oldValue_ = current;
		}
		Set(name_, value);
	}

	~ScopedEnvVar()
	{
		if (oldValue_)
		{
			Set(name_, oldValue_->c_str());
		}
		else
		{
			Unset(name_);
		}
	}

	ScopedEnvVar(const ScopedEnvVar&) = delete;
	ScopedEnvVar& operator=(const ScopedEnvVar&) = delete;

private:
	static void Set(const std::string& name, const char* value)
	{
#ifdef _WIN32
		_putenv_s(name.c_str(), value);
#else
		setenv(name.c_str(), value, 1);
#endif
	}

	static void Unset(const std::string& name)
	{
#ifdef _WIN32
		_putenv_s(name.c_str(), "");
#else
		unsetenv(name.c_str());
#endif
	}

	std::string name_;
	std::optional<std::string> oldValue_;
};

static bool EnvFlagEnabled(const char* name)
{
	if (const char* value = std::getenv(name))
	{
		const std::string_view text = value;
		return text == "1" || text == "true" || text == "TRUE" || text == "on" || text == "ON";
	}
	return false;
}

static std::string QuoteForShell(std::string_view text)
{
	std::string quoted;
	quoted.reserve(text.size() + 2);
	quoted.push_back('"');
	for (const char ch : text)
	{
		if (ch == '"')
		{
			quoted += "\\\"";
		}
		else
		{
			quoted.push_back(ch);
		}
	}
	quoted.push_back('"');
	return quoted;
}

static std::string QuoteProgramForShell(std::string_view text)
{
#ifdef _WIN32
	if (text.find_first_of(" \t\"") == std::string_view::npos)
	{
		return std::string(text);
	}
	return std::format("call {}", QuoteForShell(text));
#else
	return QuoteForShell(text);
#endif
}

static std::string ToLowerASCII(std::string_view text)
{
	std::string lowered;
	lowered.reserve(text.size());
	for (const unsigned char ch : text)
	{
		lowered.push_back(static_cast<char>(std::tolower(ch)));
	}
	return lowered;
}

static bool IsObjdumpFunctionHeader(std::string_view line)
{
	std::size_t pos = 0;
	while (pos < line.size() && std::isspace(static_cast<unsigned char>(line[pos])))
	{
		++pos;
	}
	const auto begin = pos;
	while (pos < line.size() && std::isxdigit(static_cast<unsigned char>(line[pos])))
	{
		++pos;
	}
	return pos > begin && line.find('<', pos) != std::string_view::npos && line.find(">:", pos) != std::string_view::npos;
}

static std::string ObjdumpFunctionName(std::string_view line)
{
	const auto begin = line.find('<');
	const auto end = line.find(">:", begin == std::string_view::npos ? 0 : begin);
	if (begin == std::string_view::npos || end == std::string_view::npos || end <= begin + 1)
	{
		return {};
	}
	return std::string(line.substr(begin + 1, end - begin - 1));
}

static std::optional<std::size_t> FindFunctionHeader(std::span<const std::string> lines, std::string_view function)
{
	const auto needle = std::format("<{}>:", function);
	for (std::size_t i = 0; i < lines.size(); ++i)
	{
		if (IsObjdumpFunctionHeader(lines[i]) && lines[i].find(needle) != std::string::npos)
		{
			return i;
		}
	}
	return std::nullopt;
}

static std::optional<std::size_t> FindFirstFunctionHeader(std::span<const std::string> lines)
{
	for (std::size_t i = 0; i < lines.size(); ++i)
	{
		if (IsObjdumpFunctionHeader(lines[i]))
		{
			return i;
		}
	}
	return std::nullopt;
}

static bool ContainsAny(std::string_view line, std::span<const std::string_view> needles)
{
	for (const auto needle : needles)
	{
		if (line.find(needle) != std::string_view::npos)
		{
			return true;
		}
	}
	return false;
}

static void AccumulateInstructionLine(InstructionStats& stats, std::string_view line)
{
	const auto lower = ToLowerASCII(line);
	const bool packedFma = lower.find("vfmadd") != std::string::npos && lower.find("ps") != std::string::npos;
	if (packedFma)
	{
		++stats.packedFMA;
		if (lower.find("zmm") != std::string::npos) ++stats.zmmPackedFMA;
		if (lower.find("ymm") != std::string::npos) ++stats.ymmPackedFMA;
		if (lower.find("xmm") != std::string::npos) ++stats.xmmPackedFMA;
	}
	if (lower.find("vfmadd") != std::string::npos && lower.find("ss") != std::string::npos)
	{
		++stats.scalarFMA;
	}
	if (lower.find("gather") != std::string::npos) ++stats.gather;
	if (lower.find("scatter") != std::string::npos) ++stats.scatter;
	if (lower.find("vmovups") != std::string::npos || lower.find("vmovaps") != std::string::npos) ++stats.vectorLoad;
	if (lower.find("vmovss") != std::string::npos) ++stats.scalarMove;
	if (lower.find("vbroadcast") != std::string::npos) ++stats.broadcast;
	if (lower.find("prefetch") != std::string::npos) ++stats.prefetch;

	constexpr std::string_view kVectorOps[] = { "vmov", "vadd", "vmul", "vfmadd" };
	const bool stackReference = lower.find("[rsp") != std::string::npos || lower.find("[rbp") != std::string::npos;
	if (stackReference && ContainsAny(lower, kVectorOps))
	{
		++stats.stackVectorOp;
	}
}

static InstructionStats AnalyzeObjectInstructions(const std::filesystem::path& objectPath,
                                                  const std::filesystem::path& asmPath,
                                                  std::string_view function = "subgraph_0")
{
	if (EnvFlagEnabled("LITENN_PROFILE_SKIP_OBJDUMP"))
	{
		return { .message = "skipped by LITENN_PROFILE_SKIP_OBJDUMP" };
	}

	const char* objdumpEnv = std::getenv("LITENN_OBJDUMP");
	const std::string objdump = objdumpEnv ? objdumpEnv : "objdump";
	const auto errPath = asmPath.string() + ".err";
	const auto command = std::format("{} -d -M intel {} > {} 2> {}",
	    QuoteProgramForShell(objdump), QuoteForShell(objectPath.string()), QuoteForShell(asmPath.string()),
	    QuoteForShell(errPath));
	if (std::system(command.c_str()) != 0)
	{
		return { .message = std::format("objdump failed; set LITENN_OBJDUMP or inspect {}", errPath) };
	}

	std::ifstream file(asmPath);
	if (!file)
	{
		return { .message = std::format("could not read {}", asmPath.string()) };
	}

	std::vector<std::string> lines;
	std::string line;
	while (std::getline(file, line))
	{
		lines.push_back(std::move(line));
	}

	auto start = FindFunctionHeader(lines, function);
	if (!start)
	{
		start = FindFirstFunctionHeader(lines);
		if (!start)
		{
			return { .message = "no function headers found in disassembly" };
		}
	}
	std::size_t end = lines.size();
	for (std::size_t i = *start + 1; i < lines.size(); ++i)
	{
		if (IsObjdumpFunctionHeader(lines[i]))
		{
			end = i;
			break;
		}
	}

	InstructionStats stats{ .available = true, .message = "ok",
	                        .function = ObjdumpFunctionName(lines[*start]), .lines = end - *start };
	for (std::size_t i = *start; i < end; ++i)
	{
		AccumulateInstructionLine(stats, lines[i]);
	}
	return stats;
}

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

template <typename Fn>
static double TimedOnceMs(Fn&& fn)
{
	auto begin = Clock::now();
	std::forward<Fn>(fn)();
	auto end = Clock::now();
	return clk::duration<double, std::milli>(end - begin).count();
}

template <typename Fn>
static Timing TimedRepeated(Fn&& fn, std::size_t batch, double targetMs = 500.0)
{
	for (int i = 0; i < 5; ++i) std::forward<Fn>(fn)();
	const double probeMs = std::max(TimedOnceMs(fn), 0.001);
	const auto iters = static_cast<std::size_t>(std::clamp(targetMs / probeMs, 10.0, 2000.0));
	auto begin = Clock::now();
	for (std::size_t i = 0; i < iters; ++i)
	{
		std::forward<Fn>(fn)();
	}
	auto end = Clock::now();
	const double totalMs = clk::duration<double, std::milli>(end - begin).count();
	return { totalMs / iters, batch * iters / (totalMs * 1e-3) };
}

struct Case
{
	std::string name;
	Graph (*build)(std::size_t, std::mt19937&);
	std::vector<std::size_t> outShape;  // single-output models
};

#ifdef LITENN_ENABLE_CUDA
struct CUDALaunchBreakdown
{
	std::string name;
	std::size_t batch{};
	std::string backend;
	std::string binaryKind;
	std::uint64_t featureFlags{};
	std::size_t kernelCount{};
	std::size_t libraryKernelCount{};
	std::size_t ptxKernelCount{};
	std::size_t workspaceBytes{};
	double compileMs{};
	double loadMs{};
	double nativeFirstMs{};
	double nativeMeanMs{};
	double graphFirstMs{};
	double graphMeanMs{};
	std::string message;
};

static std::string CUDABinaryKindName(CUDANativeBinaryKind kind)
{
	switch (kind)
	{
	case CUDANativeBinaryKind::PTX:
		return "ptx";
	case CUDANativeBinaryKind::Cubin:
		return "cubin";
	case CUDANativeBinaryKind::Fatbin:
		return "fatbin";
	case CUDANativeBinaryKind::LibraryCall:
		return "library";
	}
	return "unknown";
}

static std::vector<Tensor<CUDA>> MakeCUDAProfileInputs(std::size_t batch)
{
	std::mt19937 rng(0);
	std::uniform_real_distribution<float> dist(-1.0f, 1.0f);
	std::vector<float> data(batch * 784);
	for (auto& value : data)
	{
		value = dist(rng);
	}
	auto cpuInput = Optimizer::MakeFloatTensor(std::span<const float>(data), { batch, 784 });
	std::vector<Tensor<CUDA>> inputs;
	inputs.emplace_back(cpuInput.CopyToDevice(CUDA{}));
	return inputs;
}

static std::vector<Tensor<CUDA>> AllocateCUDAProfileOutputs(const CompiledModule<CUDA>& module)
{
	std::vector<Tensor<CUDA>> outputs;
	outputs.reserve(module.OutputSpecs().size());
	for (const auto& spec : module.OutputSpecs())
	{
		outputs.emplace_back(Uninitialized, ShapeView{ spec.shape }, spec.dtype, CUDA{});
	}
	return outputs;
}

static CUDALaunchBreakdown ProfileCUDALaunches(const Case& profileCase)
{
	CUDALaunchBreakdown result{ .name = profileCase.name, .batch = profileCase.outShape[0] };
	if (!IsCUDADeviceAvailable())
	{
		result.message = "CUDA device is not available";
		return result;
	}

	try
	{
		std::mt19937 rng(0);
		Graph graph = profileCase.build(result.batch, rng);
		Optimize(graph);

		CompiledModuleArtifact artifact;
		{
			ScopedEnvVar disableGraph("LITENN_CUDA_ENABLE_GRAPH_REPLAY", "0");
			auto begin = Clock::now();
			artifact = Compiler<CUDA>::CompileArtifact(graph);
			auto end = Clock::now();
			result.compileMs = clk::duration<double, std::milli>(end - begin).count();
		}
		result.backend = artifact.Backend() == CompiledModuleBackend::CUDANative ? "cuda_native" : "cpu_bridge";
		if (artifact.Backend() != CompiledModuleBackend::CUDANative)
		{
			result.message = "compiled artifact did not use CUDA native backend";
			return result;
		}

		const auto payload = DeserializeCUDANativeInstructionPayload(artifact.Instructions());
		result.binaryKind = CUDABinaryKindName(payload.binaryKind);
		result.featureFlags = payload.featureFlags;
		result.kernelCount = payload.kernels.size();
		result.workspaceBytes = static_cast<std::size_t>(payload.workspaceBytes);
		if (payload.binaryKind == CUDANativeBinaryKind::LibraryCall)
		{
			result.libraryKernelCount = payload.kernels.size();
		}
		else
		{
			result.ptxKernelCount = payload.kernels.size();
		}
		for (const auto& kernel : payload.kernels)
		{
			result.workspaceBytes = std::max(result.workspaceBytes, static_cast<std::size_t>(kernel.workspaceBytes));
		}

		auto loadBegin = Clock::now();
		auto module = artifact.Load(CUDA{});
		auto loadEnd = Clock::now();
		result.loadMs = clk::duration<double, std::milli>(loadEnd - loadBegin).count();

		auto inputs = MakeCUDAProfileInputs(result.batch);
		auto outputs = AllocateCUDAProfileOutputs(module);
		const auto runInto = [&] {
			module.RunInto(std::span<const Tensor<CUDA>>(inputs), std::span<Tensor<CUDA>>(outputs));
		};

		{
			ScopedEnvVar disableGraph("LITENN_CUDA_ENABLE_GRAPH_REPLAY", "0");
			result.nativeFirstMs = TimedOnceMs(runInto);
			const auto timing = TimedRepeated(runInto, result.batch, 300.0);
			result.nativeMeanMs = timing.meanMs;
		}
		{
			ScopedEnvVar enableGraph("LITENN_CUDA_ENABLE_GRAPH_REPLAY", "1");
			result.graphFirstMs = TimedOnceMs(runInto);
			const auto timing = TimedRepeated(runInto, result.batch, 300.0);
			result.graphMeanMs = timing.meanMs;
		}
		result.message = "ok";
	}
	catch (const std::exception& ex)
	{
		result.message = ex.what();
	}
	return result;
}
#endif

int main(int argc, char** argv)
{
	const std::filesystem::path outDir = (argc >= 2)
	    ? std::filesystem::path(argv[1])
	    : std::filesystem::current_path() / "profile_out";
	std::filesystem::create_directories(outDir);

	std::cout << "LiteNN AOT Profile Report\n";
	std::cout << "Object files written to: " << outDir.string() << "\n";
	std::cout << "Instruction stats use objdump; set LITENN_OBJDUMP to override or LITENN_PROFILE_SKIP_OBJDUMP=1 to skip.\n";
	std::cout << std::string(116, '=') << "\n";

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

	std::vector<CaseInstructionStats> instructionStats;
	instructionStats.reserve(cases.size());

	std::cout << std::format("{:<14} {:>8} {:>10} {:>12} {:>12} {:>10} {:>12} {:>7} {:>7} {:>8}\n",
	    "Case", "Batch", "Compile/ms", "Run/ms", "RunInto/ms",
	    "Alloc/us", "Speedup", "FMAps", "VecLd", "StackVec");
	std::cout << std::string(116, '-') << "\n";

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
		const auto statsBeforeWrite = instructionStats.size();
		try {
			const auto bytes = compiled.Instructions();
			const auto objectPath = outDir / (c.name + ".o");
			std::ofstream f(objectPath, std::ios::binary);
			f.write(reinterpret_cast<const char*>(bytes.data()),
			        static_cast<std::streamsize>(bytes.size()));
			f.close();
			if (!f)
			{
				throw std::runtime_error(std::format("failed to write {}", objectPath.string()));
			}
			instructionStats.push_back({ c.name, outDir / (c.name + ".s"),
			                             AnalyzeObjectInstructions(objectPath, outDir / (c.name + ".s")) });
		} catch (...) {}
		if (instructionStats.size() == statsBeforeWrite)
		{
			instructionStats.push_back({ c.name, outDir / (c.name + ".s"), { .message = "object write failed" } });
		}

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

		const auto& stats = instructionStats.back().stats;
		const auto packedFMA = stats.available ? std::format("{}", stats.packedFMA) : "n/a";
		const auto vectorLoad = stats.available ? std::format("{}", stats.vectorLoad) : "n/a";
		const auto stackVectorOp = stats.available ? std::format("{}", stats.stackVectorOp) : "n/a";
		std::cout << std::format("{:<14} {:>8} {:>9.2f}ms {:>10.4f}ms {:>10.4f}ms {:>8.2f}us {:>10.2f}x {:>7} {:>7} {:>8}\n",
		    c.name, batch, compileMs, tRun.meanMs, tRunInto.meanMs, allocUs, speedup,
		    packedFMA, vectorLoad, stackVectorOp);
	}

	std::cout << std::string(116, '=') << "\n";
	std::cout << "\nInstruction stats for subgraph_0, falling back to the first function when needed\n";
	std::cout << std::format("{:<14} {:<16} {:>7} {:>7} {:>7} {:>5} {:>5} {:>5} {:>7} {:>7} {:>7} {:>8} {:>7} {:>7}\n",
	    "Case", "Function", "Lines", "FMAps", "FMAss", "zmm", "ymm", "xmm", "Gather", "Scatter", "VecLd",
	    "ScalarMv", "Bcast", "StackV");
	std::cout << std::string(132, '-') << "\n";
	for (const auto& row : instructionStats)
	{
		const auto& s = row.stats;
		if (!s.available)
		{
			std::cout << std::format("{:<14} {:<16} {}\n", row.name, "-", s.message);
			continue;
		}
		std::cout << std::format("{:<14} {:<16} {:>7} {:>7} {:>7} {:>5} {:>5} {:>5} {:>7} {:>7} {:>7} {:>8} {:>7} {:>7}\n",
		    row.name, s.function, s.lines, s.packedFMA, s.scalarFMA, s.zmmPackedFMA, s.ymmPackedFMA,
		    s.xmmPackedFMA, s.gather, s.scatter, s.vectorLoad, s.scalarMove,
		    s.broadcast, s.stackVectorOp);
	}
	std::cout << "\nAssembly files are written beside the object files when objdump succeeds.\n";

	std::cout << "\nCUDA launch breakdowns\n";
#ifdef LITENN_ENABLE_CUDA
	if (EnvFlagEnabled("LITENN_PROFILE_SKIP_CUDA"))
	{
		std::cout << "Skipped by LITENN_PROFILE_SKIP_CUDA.\n";
	}
	else if (!IsCUDADeviceAvailable())
	{
		std::cout << "CUDA device is not available.\n";
	}
	else
	{
		std::cout << std::format(
		    "{:<14} {:>8} {:<11} {:<8} {:>7} {:>7} {:>7} {:>10} {:>10} {:>10} {:>12} {:>11} {:>10} {:>10} {}\n",
		    "Case", "Batch", "Backend", "Binary", "Kernels", "Lib", "PTX", "Workspace", "Compile",
		    "Load", "Native1", "NativeAvg", "Graph1", "GraphAvg", "Status");
		std::cout << std::string(170, '-') << "\n";
		for (const auto& c : cases)
		{
			const auto row = ProfileCUDALaunches(c);
			std::cout << std::format(
			    "{:<14} {:>8} {:<11} {:<8} {:>7} {:>7} {:>7} {:>10} {:>8.2f}ms {:>8.2f}ms {:>10.4f}ms {:>9.4f}ms {:>8.4f}ms {:>8.4f}ms {}\n",
			    row.name, row.batch, row.backend.empty() ? "-" : row.backend, row.binaryKind.empty() ? "-" : row.binaryKind,
			    row.kernelCount, row.libraryKernelCount, row.ptxKernelCount, row.workspaceBytes,
			    row.compileMs, row.loadMs, row.nativeFirstMs, row.nativeMeanMs, row.graphFirstMs, row.graphMeanMs, row.message);
		}
		std::cout << "Native1 is the first synchronized native RunInto. NativeAvg is steady synchronized native RunInto.\n";
		std::cout << "Graph1 is first graph capture+run. GraphAvg is steady synchronized RunInto with LITENN_CUDA_ENABLE_GRAPH_REPLAY=1.\n";
	}
#else
	std::cout << "Unavailable: LiteNN was built without LITENN_ENABLE_CUDA.\n";
#endif
	return 0;
}
