// LiteNN AOT 性能瓶颈分析工具
//
// 1) 测量 Run() 的分配开销 vs 纯 entry 调用开销
// 2) 将编译产物 (.o) 写到磁盘，并用 objdump 生成 first-class 指令统计
// 3) 测量 Compile() 自身的耗时（一次性成本）

#include "CompilerOptionsEnv.h"

#include <LiteNN.h>
#include <LiteNN/Compiler/CompiledModule.h>
#ifdef LITENN_ENABLE_CUDA
#include <LiteNN/Compiler/CUDANativePayload.h>
#endif
#ifdef LITENN_ENABLE_VULKAN
#include <LiteNN/Compiler/VulkanNativePayload.h>
#endif
#include <LiteNN/Initializer/Initializer.h>
#include <LiteNN/Layer/Layer.h>
#include <LiteNN/Optimizer/Loss.h>
#include <LiteNN/Pass/ConstFoldPass.h>
#include <LiteNN/Pass/FusionPass.h>
#include <LiteNN/Pass/InlinePass.h>

#include <algorithm>
#include <cctype>
#include <chrono>
#include <cstdlib>
#include <filesystem>
#include <format>
#include <fstream>
#include <iostream>
#include <limits>
#include <optional>
#include <random>
#include <span>
#include <stdexcept>
#include <string>
#include <string_view>
#include <thread>
#include <utility>
#include <vector>

using namespace LiteNN;

static Graph BuildLinear(std::size_t batch, std::mt19937& rng)
{
	ModelBuilder builder;
	Graph& graph = builder.UnsafeMutableGraph();
	const auto fc =
	    Layer::CreateLinear(builder, Initializer::XavierUniform({ 784, 10 }, rng), Initializer::Zeros({ 1, 10 }));
	Subgraph fwd;
	const auto in = fwd.AddParam(DataType::Float32, { batch, 784 });
	fwd.SetResults({ Layer::AddLinear(fwd, fc, { in, 0 }) });
	graph.SetForward(graph.AddSubgraph(std::move(fwd)));
	return builder.UnsafeTakeGraph();
}

static Graph BuildMLP128(std::size_t batch, std::mt19937& rng)
{
	ModelBuilder builder;
	Graph& graph = builder.UnsafeMutableGraph();
	const auto h1 =
	    Layer::CreateLinear(builder, Initializer::XavierUniform({ 784, 128 }, rng), Initializer::Zeros({ 1, 128 }));
	const auto h2 =
	    Layer::CreateLinear(builder, Initializer::XavierUniform({ 128, 10 }, rng), Initializer::Zeros({ 1, 10 }));
	Subgraph fwd;
	const auto in = fwd.AddParam(DataType::Float32, { batch, 784 });
	const auto a1 = Layer::AddReLU(fwd, Layer::AddLinear(fwd, h1, { in, 0 }));
	fwd.SetResults({ Layer::AddLinear(fwd, h2, a1) });
	graph.SetForward(graph.AddSubgraph(std::move(fwd)));
	return builder.UnsafeTakeGraph();
}

static Graph BuildMLP512(std::size_t batch, std::mt19937& rng)
{
	ModelBuilder builder;
	Graph& graph = builder.UnsafeMutableGraph();
	const auto h1 =
	    Layer::CreateLinear(builder, Initializer::XavierUniform({ 784, 512 }, rng), Initializer::Zeros({ 1, 512 }));
	const auto h2 =
	    Layer::CreateLinear(builder, Initializer::XavierUniform({ 512, 256 }, rng), Initializer::Zeros({ 1, 256 }));
	const auto h3 =
	    Layer::CreateLinear(builder, Initializer::XavierUniform({ 256, 10 }, rng), Initializer::Zeros({ 1, 10 }));
	Subgraph fwd;
	const auto in = fwd.AddParam(DataType::Float32, { batch, 784 });
	const auto a1 = Layer::AddReLU(fwd, Layer::AddLinear(fwd, h1, { in, 0 }));
	const auto a2 = Layer::AddReLU(fwd, Layer::AddLinear(fwd, h2, a1));
	fwd.SetResults({ Layer::AddLinear(fwd, h3, a2) });
	graph.SetForward(graph.AddSubgraph(std::move(fwd)));
	return builder.UnsafeTakeGraph();
}

static Graph BuildBinaryChainProfileGraph(std::size_t batch, std::mt19937&)
{
	Graph graph;
	Subgraph sg;
	const std::vector<std::size_t> shape{ batch, 784 };
	const auto a = sg.AddParam(DataType::Float32, shape);
	const auto b = sg.AddParam(DataType::Float32, shape);
	const auto c = sg.AddParam(DataType::Float32, shape);
	const auto d = sg.AddParam(DataType::Float32, shape);
	const auto first =
	    sg.AddNode(BinaryOpNode{ BinaryOp::Add, { a, 0 }, { b, 0 } }, { OutputInfo{ DataType::Float32, shape } });
	const auto second = sg.AddNode(BinaryOpNode{ BinaryOp::Multiply, { first, 0 }, { c, 0 } },
	                               { OutputInfo{ DataType::Float32, shape } });
	const auto third = sg.AddNode(BinaryOpNode{ BinaryOp::Subtract, { second, 0 }, { d, 0 } },
	                              { OutputInfo{ DataType::Float32, shape } });
	sg.SetResults({ { third, 0 } });
	graph.AddSubgraph(std::move(sg));
	graph.SetForward(0);
	graph.SetInputNames({ "a", "b", "c", "d" });
	graph.SetOutputNames({ "out" });
	return graph;
}

static Graph BuildBinaryDAGProfileGraph(std::size_t batch, std::mt19937&)
{
	Graph graph;
	Subgraph sg;
	const std::vector<std::size_t> shape{ batch, 784 };
	const auto lhs = sg.AddParam(DataType::Float32, shape);
	const auto rhs = sg.AddParam(DataType::Float32, shape);
	const auto tail = sg.AddParam(DataType::Float32, shape);
	const auto first =
	    sg.AddNode(BinaryOpNode{ BinaryOp::Add, { lhs, 0 }, { rhs, 0 } }, { OutputInfo{ DataType::Float32, shape } });
	const auto second =
	    sg.AddNode(BinaryOpNode{ BinaryOp::Add, { lhs, 0 }, { tail, 0 } }, { OutputInfo{ DataType::Float32, shape } });
	const auto out = sg.AddNode(BinaryOpNode{ BinaryOp::Add, { first, 0 }, { second, 0 } },
	                            { OutputInfo{ DataType::Float32, shape } });
	sg.SetResults({ { out, 0 } });
	graph.AddSubgraph(std::move(sg));
	graph.SetForward(0);
	graph.SetInputNames({ "lhs", "rhs", "tail" });
	graph.SetOutputNames({ "out" });
	return graph;
}

static Graph BuildMixedElementwiseDAGProfileGraph(std::size_t batch, std::mt19937&)
{
	Graph graph;
	Subgraph sg;
	const std::vector<std::size_t> shape{ batch, 784 };
	const auto lhs = sg.AddParam(DataType::Float32, shape);
	const auto rhs = sg.AddParam(DataType::Float32, shape);
	const auto tail = sg.AddParam(DataType::Float32, shape);
	const auto added =
	    sg.AddNode(BinaryOpNode{ BinaryOp::Add, { lhs, 0 }, { rhs, 0 } }, { OutputInfo{ DataType::Float32, shape } });
	const auto abs = sg.AddNode(UnaryOpNode{ UnaryOp::Abs, { added, 0 } }, { OutputInfo{ DataType::Float32, shape } });
	const auto out = sg.AddNode(BinaryOpNode{ BinaryOp::Multiply, { abs, 0 }, { tail, 0 } },
	                            { OutputInfo{ DataType::Float32, shape } });
	sg.SetResults({ { out, 0 } });
	graph.AddSubgraph(std::move(sg));
	graph.SetForward(0);
	graph.SetInputNames({ "lhs", "rhs", "tail" });
	graph.SetOutputNames({ "out" });
	return graph;
}

static Graph BuildBranchedBinaryDAGProfileGraph(std::size_t batch, std::mt19937&)
{
	Graph graph;
	Subgraph sg;
	const std::vector<std::size_t> shape{ batch, 784 };
	const auto a = sg.AddParam(DataType::Float32, shape);
	const auto b = sg.AddParam(DataType::Float32, shape);
	const auto c = sg.AddParam(DataType::Float32, shape);
	const auto d = sg.AddParam(DataType::Float32, shape);
	const auto e = sg.AddParam(DataType::Float32, shape);
	const auto first =
	    sg.AddNode(BinaryOpNode{ BinaryOp::Add, { a, 0 }, { b, 0 } }, { OutputInfo{ DataType::Float32, shape } });
	const auto second =
	    sg.AddNode(BinaryOpNode{ BinaryOp::Add, { c, 0 }, { d, 0 } }, { OutputInfo{ DataType::Float32, shape } });
	const auto merged = sg.AddNode(BinaryOpNode{ BinaryOp::Multiply, { first, 0 }, { second, 0 } },
	                               { OutputInfo{ DataType::Float32, shape } });
	const auto tail = sg.AddNode(BinaryOpNode{ BinaryOp::Subtract, { first, 0 }, { e, 0 } },
	                             { OutputInfo{ DataType::Float32, shape } });
	const auto out = sg.AddNode(BinaryOpNode{ BinaryOp::Add, { merged, 0 }, { tail, 0 } },
	                            { OutputInfo{ DataType::Float32, shape } });
	sg.SetResults({ { out, 0 } });
	graph.AddSubgraph(std::move(sg));
	graph.SetForward(0);
	graph.SetInputNames({ "a", "b", "c", "d", "e" });
	graph.SetOutputNames({ "out" });
	return graph;
}

static Graph BuildReduceProfileGraph(ReduceOp op, std::size_t batch)
{
	Graph graph;
	Subgraph sg;
	const std::vector<std::size_t> inputShape{ batch, 784 };
	const auto input = sg.AddParam(DataType::Float32, inputShape);
	const auto out = sg.AddNode(ReduceOpNode{ op, { input, 0 }, 1 }, { OutputInfo{ DataType::Float32, { batch } } });
	sg.SetResults({ { out, 0 } });
	graph.AddSubgraph(std::move(sg));
	graph.SetForward(0);
	graph.SetInputNames({ "input" });
	graph.SetOutputNames({ "out" });
	return graph;
}

static Graph BuildReduceSumProfileGraph(std::size_t batch, std::mt19937&)
{
	return BuildReduceProfileGraph(ReduceOp::Sum, batch);
}

static Graph BuildReduceMeanProfileGraph(std::size_t batch, std::mt19937&)
{
	return BuildReduceProfileGraph(ReduceOp::Mean, batch);
}

static Graph BuildReduceMaxProfileGraph(std::size_t batch, std::mt19937&)
{
	return BuildReduceProfileGraph(ReduceOp::Max, batch);
}

static Graph BuildReduceMinProfileGraph(std::size_t batch, std::mt19937&)
{
	return BuildReduceProfileGraph(ReduceOp::Min, batch);
}

static Graph BuildSoftmaxProfileGraph(std::size_t batch, std::mt19937&)
{
	Graph graph;
	Subgraph sg;
	const std::vector<std::size_t> shape{ batch, 784 };
	const auto input = sg.AddParam(DataType::Float32, shape);
	const auto out = sg.AddNode(SoftmaxNode{ { input, 0 }, 1 }, { OutputInfo{ DataType::Float32, shape } });
	sg.SetResults({ { out, 0 } });
	graph.AddSubgraph(std::move(sg));
	graph.SetForward(0);
	graph.SetInputNames({ "input" });
	graph.SetOutputNames({ "out" });
	return graph;
}

static Graph BuildNormalizationProfileGraph(NormalizationMode mode, std::size_t batch)
{
	Graph graph;
	Subgraph sg;
	const std::vector<std::size_t> shape{ batch, 784 };
	const auto input = sg.AddParam(DataType::Float32, shape);
	const auto out = sg.AddNode(NormalizationNode{ .input = { input, 0 },
	                                               .scale = std::nullopt,
	                                               .bias = std::nullopt,
	                                               .mode = mode,
	                                               .axis = 1,
	                                               .groupCount = 1,
	                                               .epsilon = 1e-5 },
	                            { OutputInfo{ DataType::Float32, shape } });
	sg.SetResults({ { out, 0 } });
	graph.AddSubgraph(std::move(sg));
	graph.SetForward(0);
	graph.SetInputNames({ "input" });
	graph.SetOutputNames({ "out" });
	return graph;
}

static Graph BuildAffineLayerNormProfileGraph(std::size_t batch, std::mt19937&)
{
	Graph graph;
	std::vector<double> scale(784);
	std::vector<double> bias(784);
	for (std::size_t i = 0; i < 784; ++i)
	{
		scale[i] = 0.75 + 0.01 * static_cast<double>(i % 17);
		bias[i] = -0.05 + 0.001 * static_cast<double>(i % 23);
	}
	const auto scaleIndex =
	    graph.AddVariable(Variable::Create(Tensor<CPU>(std::move(scale), { 784 }, DataType::Float32)));
	const auto biasIndex =
	    graph.AddVariable(Variable::Create(Tensor<CPU>(std::move(bias), { 784 }, DataType::Float32)));
	graph.SetVariableName(scaleIndex, "layernorm_scale");
	graph.SetVariableName(biasIndex, "layernorm_bias");

	Subgraph sg;
	const std::vector<std::size_t> shape{ batch, 784 };
	const auto input = sg.AddParam(DataType::Float32, shape);
	const auto scaleRef = sg.AddNode(VariableRefNode{ scaleIndex }, { OutputInfo{ DataType::Float32, { 784 } } });
	const auto biasRef = sg.AddNode(VariableRefNode{ biasIndex }, { OutputInfo{ DataType::Float32, { 784 } } });
	const auto out = sg.AddNode(NormalizationNode{ .input = { input, 0 },
	                                               .scale = NodeOutput{ scaleRef, 0 },
	                                               .bias = NodeOutput{ biasRef, 0 },
	                                               .mode = NormalizationMode::LayerNorm,
	                                               .axis = 1,
	                                               .groupCount = 1,
	                                               .epsilon = 1e-5 },
	                            { OutputInfo{ DataType::Float32, shape } });
	sg.SetResults({ { out, 0 } });
	graph.AddSubgraph(std::move(sg));
	graph.SetForward(0);
	graph.SetInputNames({ "input" });
	graph.SetOutputNames({ "out" });
	return graph;
}

static Graph BuildLayerNormProfileGraph(std::size_t batch, std::mt19937&)
{
	return BuildNormalizationProfileGraph(NormalizationMode::LayerNorm, batch);
}

static Graph BuildRMSNormProfileGraph(std::size_t batch, std::mt19937&)
{
	return BuildNormalizationProfileGraph(NormalizationMode::RMSNorm, batch);
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

struct CPUAOTLayerSelection
{
	std::size_t m{};
	std::size_t k{};
	std::size_t n{};
	std::uint64_t flops{};
	std::size_t selectedThreads{ 1 };
	std::string reason;
};

struct CPUAOTParallelSelection
{
	std::string name;
	std::size_t batch{};
	std::size_t configuredThreads{};
	std::size_t fusedLayerCount{};
	std::size_t parallelLayerCount{};
	std::uint64_t totalFlops{};
	bool predictedSidecar{};
	bool objectUsesSidecar{};
	std::string gate;
	std::vector<CPUAOTLayerSelection> layers;
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
	return pos > begin && line.find('<', pos) != std::string_view::npos &&
	       line.find(">:", pos) != std::string_view::npos;
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

static std::uint64_t SaturatingMul(std::uint64_t lhs, std::uint64_t rhs)
{
	if (lhs == 0 || rhs == 0)
	{
		return 0;
	}
	if (lhs > std::numeric_limits<std::uint64_t>::max() / rhs)
	{
		return std::numeric_limits<std::uint64_t>::max();
	}
	return lhs * rhs;
}

static std::uint64_t SaturatingAdd(std::uint64_t lhs, std::uint64_t rhs)
{
	if (lhs > std::numeric_limits<std::uint64_t>::max() - rhs)
	{
		return std::numeric_limits<std::uint64_t>::max();
	}
	return lhs + rhs;
}

static std::size_t ResolveProfileCPUAOTThreadCount(const CompilerOptions& options)
{
	if (options.cpuAOTThreadCount != 0)
	{
		return options.cpuAOTThreadCount;
	}
	const auto hardware = std::thread::hardware_concurrency();
	return hardware == 0 ? 1 : hardware;
}

static bool ProfileSidecarLayerGate(std::uint64_t m, std::uint64_t k, std::uint64_t n, std::uint64_t flops)
{
	constexpr std::uint64_t kMinLayerFlops = 1ull << 26;
	constexpr std::uint64_t kMaxRowsBeforePackedMLIR = 256;
	constexpr std::uint64_t kMinOutputColumns = 64;
	return flops >= kMinLayerFlops && m <= kMaxRowsBeforePackedMLIR && k >= 64 && n >= kMinOutputColumns;
}

static bool ObjectBytesContain(std::span<const std::byte> bytes, std::string_view needle)
{
	if (needle.empty() || bytes.size() < needle.size())
	{
		return false;
	}
	const auto* raw = reinterpret_cast<const char*>(bytes.data());
	for (std::size_t i = 0; i + needle.size() <= bytes.size(); ++i)
	{
		if (std::string_view(raw + i, needle.size()) == needle)
		{
			return true;
		}
	}
	return false;
}

static CPUAOTParallelSelection AnalyzeCPUAOTParallelSelection(const std::string& name, std::size_t batch,
                                                              const Graph& graph, const CompilerOptions& options)
{
	CPUAOTParallelSelection result{
		.name = name,
		.batch = batch,
		.configuredThreads = ResolveProfileCPUAOTThreadCount(options),
		.gate = "not a supported fused linear chain",
	};
	if (result.configuredThreads <= 1)
	{
		result.gate = "thread_count<=1";
		return result;
	}
	if (graph.Backward().has_value() || graph.ActivationSlotCount() != 0 || graph.TapeSlotCount() != 0 ||
	    graph.SubgraphCount() == 0)
	{
		result.gate = "graph has backward/tape/activation state or no forward subgraph";
		return result;
	}
	const auto& subgraph = graph.GetSubgraph(graph.Forward());
	if (subgraph.Results().size() != 1)
	{
		result.gate = "requires exactly one forward result";
		return result;
	}
	const bool forceSidecarShapeGate = options.cpuAOTParallelMinFlops <= 1;
	for (NodeId nodeId = 0; nodeId < subgraph.NodeCount(); ++nodeId)
	{
		const auto& entry = subgraph.GetNodeEntry(nodeId);
		const auto* fused = std::get_if<FusedOpNode>(&entry.node);
		if (fused == nullptr ||
		    (fused->pattern != FusionPattern::MatMulBiasAdd && fused->pattern != FusionPattern::MatMulBiasAddReLU) ||
		    fused->args.size() < 3 || entry.outputInfos.empty())
		{
			continue;
		}
		const auto lhsOutput = fused->args[0];
		const auto rhsOutput = fused->args[1];
		if (lhsOutput.node >= subgraph.NodeCount() || rhsOutput.node >= subgraph.NodeCount())
		{
			continue;
		}
		const auto& lhsEntry = subgraph.GetNodeEntry(lhsOutput.node);
		const auto& rhsEntry = subgraph.GetNodeEntry(rhsOutput.node);
		if (lhsOutput.port >= lhsEntry.outputInfos.size() || rhsOutput.port >= rhsEntry.outputInfos.size())
		{
			continue;
		}
		const auto& lhsInfo = lhsEntry.outputInfos[lhsOutput.port];
		const auto& rhsInfo = rhsEntry.outputInfos[rhsOutput.port];
		const auto& outInfo = entry.outputInfos[0];
		if (lhsInfo.dtype != DataType::Float32 || rhsInfo.dtype != DataType::Float32 ||
		    outInfo.dtype != DataType::Float32 || lhsInfo.shape.size() != 2 || rhsInfo.shape.size() != 2 ||
		    outInfo.shape.size() != 2)
		{
			continue;
		}
		const auto m = static_cast<std::uint64_t>(outInfo.shape[0]);
		const auto k = static_cast<std::uint64_t>(lhsInfo.shape[1]);
		const auto n = static_cast<std::uint64_t>(outInfo.shape[1]);
		const auto flops = SaturatingMul(SaturatingMul(SaturatingMul(m, k), n), 2);
		CPUAOTLayerSelection layer{
			.m = outInfo.shape[0],
			.k = lhsInfo.shape[1],
			.n = outInfo.shape[1],
			.flops = flops,
			.selectedThreads = 1,
			.reason = "single-thread: layer gate rejected",
		};
		if (!forceSidecarShapeGate && m > 256)
		{
			layer.reason = "chain rejected: m>256 keeps packed MLIR fallback";
			result.layers.push_back(std::move(layer));
			result.totalFlops = SaturatingAdd(result.totalFlops, flops);
			++result.fusedLayerCount;
			result.gate = "m>256";
			return result;
		}
		if (forceSidecarShapeGate || ProfileSidecarLayerGate(m, k, n, flops))
		{
			layer.selectedThreads = result.configuredThreads;
			layer.reason = forceSidecarShapeGate ? "parallel: forced by cpuAOTParallelMinFlops<=1"
			                                     : "parallel: layer gate accepted";
			++result.parallelLayerCount;
		}
		result.layers.push_back(std::move(layer));
		result.totalFlops = SaturatingAdd(result.totalFlops, flops);
		++result.fusedLayerCount;
	}
	if (result.fusedLayerCount == 0)
	{
		result.gate = "no fused MatMulBiasAdd/ReLU layers";
		return result;
	}
	if (result.parallelLayerCount == 0)
	{
		result.gate = "no layer selected more than one helper thread";
		return result;
	}
	if (result.totalFlops < options.cpuAOTParallelMinFlops)
	{
		result.gate = "total_flops below cpuAOTParallelMinFlops";
		return result;
	}
	result.predictedSidecar = true;
	result.gate = "sidecar predicted";
	return result;
}

static void AccumulateInstructionLine(InstructionStats& stats, std::string_view line)
{
	const auto lower = ToLowerASCII(line);
	const bool packedFma = lower.find("vfmadd") != std::string::npos && lower.find("ps") != std::string::npos;
	if (packedFma)
	{
		++stats.packedFMA;
		if (lower.find("zmm") != std::string::npos)
		{
			++stats.zmmPackedFMA;
		}
		if (lower.find("ymm") != std::string::npos)
		{
			++stats.ymmPackedFMA;
		}
		if (lower.find("xmm") != std::string::npos)
		{
			++stats.xmmPackedFMA;
		}
	}
	if (lower.find("vfmadd") != std::string::npos && lower.find("ss") != std::string::npos)
	{
		++stats.scalarFMA;
	}
	if (lower.find("gather") != std::string::npos)
	{
		++stats.gather;
	}
	if (lower.find("scatter") != std::string::npos)
	{
		++stats.scatter;
	}
	if (lower.find("vmovups") != std::string::npos || lower.find("vmovaps") != std::string::npos)
	{
		++stats.vectorLoad;
	}
	if (lower.find("vmovss") != std::string::npos)
	{
		++stats.scalarMove;
	}
	if (lower.find("vbroadcast") != std::string::npos)
	{
		++stats.broadcast;
	}
	if (lower.find("prefetch") != std::string::npos)
	{
		++stats.prefetch;
	}

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
	const auto command =
	    std::format("{} -d -M intel {} > {} 2> {}", QuoteProgramForShell(objdump), QuoteForShell(objectPath.string()),
	                QuoteForShell(asmPath.string()), QuoteForShell(errPath));
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

	InstructionStats stats{
		.available = true, .message = "ok", .function = ObjdumpFunctionName(lines[*start]), .lines = end - *start
	};
	for (std::size_t i = *start; i < end; ++i)
	{
		AccumulateInstructionLine(stats, lines[i]);
	}
	return stats;
}

static Timing TimedRun(const CompiledModule<CPU>& module, std::span<const Tensor<CPU>> inputs, std::size_t batch)
{
	for (int i = 0; i < 5; ++i)
	{
		(void) module.RunTensors(inputs);
	}
	auto t0 = Clock::now();
	(void) module.RunTensors(inputs);
	auto t1 = Clock::now();
	const double probeMs = clk::duration<double, std::milli>(t1 - t0).count();
	const auto iters = static_cast<std::size_t>(std::clamp(2000.0 / std::max(probeMs, 0.001), 10.0, 2000.0));
	auto a = Clock::now();
	for (std::size_t i = 0; i < iters; ++i)
	{
		(void) module.RunTensors(inputs);
	}
	auto b = Clock::now();
	const double total = clk::duration<double, std::milli>(b - a).count();
	return { total / iters, batch * iters / (total * 1e-3) };
}

static Timing TimedRunTensorsInto(const CompiledModule<CPU>& module, std::span<const Tensor<CPU>> inputs,
                                  std::span<Tensor<CPU>> outputs, std::size_t batch)
{
	for (int i = 0; i < 5; ++i)
	{
		module.RunTensorsInto(inputs, outputs);
	}
	auto t0 = Clock::now();
	module.RunTensorsInto(inputs, outputs);
	auto t1 = Clock::now();
	const double probeMs = clk::duration<double, std::milli>(t1 - t0).count();
	const auto iters = static_cast<std::size_t>(std::clamp(2000.0 / std::max(probeMs, 0.001), 10.0, 2000.0));
	auto a = Clock::now();
	for (std::size_t i = 0; i < iters; ++i)
	{
		module.RunTensorsInto(inputs, outputs);
	}
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
	for (int i = 0; i < 5; ++i)
	{
		std::forward<Fn>(fn)();
	}
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
	std::vector<std::size_t> outShape; // single-output models
};

static std::string CsvEscape(std::string_view value)
{
	if (value.find_first_of(",\"\n\r") == std::string_view::npos)
	{
		return std::string(value);
	}
	std::string escaped;
	escaped.reserve(value.size() + 2);
	escaped.push_back('"');
	for (const char ch : value)
	{
		if (ch == '"')
		{
			escaped.push_back('"');
		}
		escaped.push_back(ch);
	}
	escaped.push_back('"');
	return escaped;
}

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
	std::size_t inputBytes{};
	std::size_t outputBytes{};
	std::size_t constantBytes{};
	std::size_t estimatedBytesPerRun{};
	double compileMs{};
	double loadMs{};
	double nativeFirstMs{};
	double nativeMeanMs{};
	double graphFirstMs{};
	double graphMeanMs{};
	double nativeEstimatedGBps{};
	double graphEstimatedGBps{};
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

static std::size_t SumCompiledTensorBytes(std::span<const CompiledTensorSpec> specs)
{
	std::size_t bytes = 0;
	for (const auto& spec : specs)
	{
		bytes += spec.type.ByteSize().value_or(0);
	}
	return bytes;
}

static double EstimatedGBps(std::size_t bytes, double milliseconds)
{
	if (bytes == 0 || milliseconds <= 0.0)
	{
		return 0.0;
	}
	return static_cast<double>(bytes) / (milliseconds * 1.0e6);
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
		outputs.emplace_back(Uninitialized, ShapeView{ spec.type.StaticShape() }, spec.type.dtype, CUDA{});
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
			auto begin = Clock::now();
			artifact = Compiler<CUDA>::CompileArtifact(Detail::BuildExecutablePlanFromGraph(graph),
			                                           LiteNNBenchCompilerOptionsFromEnvironment());
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
		result.featureFlags = payload.featureSet.flags;
		result.kernelCount = payload.kernels.size();
		result.workspaceBytes = static_cast<std::size_t>(payload.workspaceBytes);
		result.inputBytes = SumCompiledTensorBytes(artifact.InputSpecs());
		result.outputBytes = SumCompiledTensorBytes(artifact.OutputSpecs());
		result.constantBytes = payload.constantData.size();
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
		result.estimatedBytesPerRun =
		    result.inputBytes + result.outputBytes + result.constantBytes + result.workspaceBytes;

		auto loadBegin = Clock::now();
		auto module = artifact.Load(CUDA{});
		auto loadEnd = Clock::now();
		result.loadMs = clk::duration<double, std::milli>(loadEnd - loadBegin).count();

		auto inputs = MakeCUDAProfileInputs(result.batch);
		auto outputs = AllocateCUDAProfileOutputs(module);
		const auto runInto = [&](bool enableGraphReplay) {
			module.RunTensorsInto(std::span<const Tensor<CUDA>>(inputs), std::span<Tensor<CUDA>>(outputs),
			                      CompiledModuleCUDARunOptions{ .graphReplay = enableGraphReplay
			                                                                       ? CUDAGraphReplayMode::Enabled
			                                                                       : CUDAGraphReplayMode::Disabled });
		};

		{
			const auto nativeRun = [&] { runInto(false); };
			result.nativeFirstMs = TimedOnceMs(nativeRun);
			const auto timing = TimedRepeated(nativeRun, result.batch, 300.0);
			result.nativeMeanMs = timing.meanMs;
			result.nativeEstimatedGBps = EstimatedGBps(result.estimatedBytesPerRun, result.nativeMeanMs);
		}
		{
			const auto graphRun = [&] { runInto(true); };
			result.graphFirstMs = TimedOnceMs(graphRun);
			const auto timing = TimedRepeated(graphRun, result.batch, 300.0);
			result.graphMeanMs = timing.meanMs;
			result.graphEstimatedGBps = EstimatedGBps(result.estimatedBytesPerRun, result.graphMeanMs);
		}
		result.message = "ok";
	}
	catch (const std::exception& ex)
	{
		result.message = ex.what();
	}
	return result;
}

static void WriteCUDAProfileCsv(const std::filesystem::path& path, std::span<const CUDALaunchBreakdown> rows)
{
	std::ofstream out(path);
	if (!out)
	{
		throw std::runtime_error(std::format("Failed to open CUDA profile CSV '{}'", path.string()));
	}
	out << "case,batch,backend,binary,kernels,library_kernels,ptx_kernels,workspace_bytes,input_bytes,output_bytes,"
	       "constant_bytes,estimated_bytes_per_run,compile_ms,load_ms,native_first_ms,native_mean_ms,"
	       "native_estimated_gbps,graph_first_ms,graph_mean_ms,graph_estimated_gbps,status\n";
	for (const auto& row : rows)
	{
		out << CsvEscape(row.name) << ',' << row.batch << ',' << CsvEscape(row.backend) << ','
		    << CsvEscape(row.binaryKind) << ',' << row.kernelCount << ',' << row.libraryKernelCount << ','
		    << row.ptxKernelCount << ',' << row.workspaceBytes << ',' << row.inputBytes << ',' << row.outputBytes << ','
		    << row.constantBytes << ',' << row.estimatedBytesPerRun << ',' << row.compileMs << ',' << row.loadMs << ','
		    << row.nativeFirstMs << ',' << row.nativeMeanMs << ',' << row.nativeEstimatedGBps << ',' << row.graphFirstMs
		    << ',' << row.graphMeanMs << ',' << row.graphEstimatedGBps << ',' << CsvEscape(row.message) << '\n';
	}
}
#endif

#ifdef LITENN_ENABLE_VULKAN
struct VulkanLaunchBreakdown
{
	std::string name;
	std::size_t batch{};
	std::string backend;
	std::string target;
	std::uint64_t featureFlags{};
	std::size_t kernelCount{};
	std::size_t externalTensorCount{};
	std::size_t workspaceTensorCount{};
	std::size_t workspaceBytes{};
	double compileMs{};
	double loadMs{};
	double inputUploadMs{};
	double firstMs{};
	double meanMs{};
	double lastDispatchMs{};
	double lastGpuMs{};
	bool gpuTimestampAvailable{};
	double outputDownloadMs{};
	double moduleCreationMs{};
	std::string message;
};

struct VulkanProfileInputs
{
	std::vector<Tensor<Vulkan>> tensors;
	double uploadMs{};
};

struct VulkanProfileCase
{
	std::string name;
	Graph (*build)(std::size_t, std::mt19937&);
	std::size_t batch{};
	VulkanProfileInputs (*makeInputs)(std::size_t);
};

static VulkanProfileInputs MakeVulkanProfileInputs(std::size_t batch)
{
	std::mt19937 rng(0);
	std::uniform_real_distribution<float> dist(-1.0f, 1.0f);
	std::vector<float> data(batch * 784);
	for (auto& value : data)
	{
		value = dist(rng);
	}
	auto cpuInput = Optimizer::MakeFloatTensor(std::span<const float>(data), { batch, 784 });
	std::vector<Tensor<Vulkan>> inputs;
	const auto uploadMs = TimedOnceMs([&] { inputs.emplace_back(cpuInput.CopyToDevice(Vulkan{})); });
	return { .tensors = std::move(inputs), .uploadMs = uploadMs };
}

static VulkanProfileInputs MakeVulkanSameShapeProfileInputs(std::size_t batch, std::size_t inputCount);

static VulkanProfileInputs MakeVulkanBinaryChainProfileInputs(std::size_t batch)
{
	return MakeVulkanSameShapeProfileInputs(batch, 4);
}

static VulkanProfileInputs MakeVulkanBinaryDAGProfileInputs(std::size_t batch)
{
	return MakeVulkanSameShapeProfileInputs(batch, 3);
}

static VulkanProfileInputs MakeVulkanMixedElementwiseDAGProfileInputs(std::size_t batch)
{
	return MakeVulkanSameShapeProfileInputs(batch, 3);
}

static VulkanProfileInputs MakeVulkanBranchedBinaryDAGProfileInputs(std::size_t batch)
{
	return MakeVulkanSameShapeProfileInputs(batch, 5);
}

static VulkanProfileInputs MakeVulkanSameShapeProfileInputs(std::size_t batch, std::size_t inputCount)
{
	std::mt19937 rng(0);
	std::uniform_real_distribution<float> dist(-1.0f, 1.0f);
	std::vector<Tensor<CPU>> cpuInputs;
	cpuInputs.reserve(inputCount);
	for (std::size_t inputIndex = 0; inputIndex < inputCount; ++inputIndex)
	{
		std::vector<float> data(batch * 784);
		for (auto& value : data)
		{
			value = dist(rng);
		}
		cpuInputs.emplace_back(Optimizer::MakeFloatTensor(std::span<const float>(data), { batch, 784 }));
	}

	std::vector<Tensor<Vulkan>> inputs;
	inputs.reserve(cpuInputs.size());
	const auto uploadMs = TimedOnceMs([&] {
		for (const auto& input : cpuInputs)
		{
			inputs.emplace_back(input.CopyToDevice(Vulkan{}));
		}
	});
	return { .tensors = std::move(inputs), .uploadMs = uploadMs };
}

static std::vector<Tensor<Vulkan>> AllocateVulkanProfileOutputs(const CompiledModule<Vulkan>& module)
{
	return module.AllocateOutputTensors();
}

static double MeasureVulkanOutputDownloadMs(std::span<const Tensor<Vulkan>> outputs)
{
	return TimedOnceMs([&] {
		std::vector<Tensor<CPU>> hostOutputs;
		hostOutputs.reserve(outputs.size());
		for (const auto& output : outputs)
		{
			hostOutputs.push_back(output.CopyToDevice(CPU{}));
		}
	});
}

static double SumVulkanDispatchWallMs(std::span<const CompiledModuleVulkanProfileEvent> events)
{
	double total = 0.0;
	for (const auto& event : events)
	{
		total += event.dispatchWallMs;
	}
	return total;
}

static double SumVulkanModuleCreationMs(std::span<const CompiledModuleVulkanProfileEvent> events)
{
	double total = 0.0;
	for (const auto& event : events)
	{
		total += event.moduleCreationWallMs;
	}
	return total;
}

static double SumVulkanGpuMs(std::span<const CompiledModuleVulkanProfileEvent> events)
{
	double total = 0.0;
	for (const auto& event : events)
	{
		if (!event.gpuTimestampAvailable)
		{
			return 0.0;
		}
		total += event.gpuElapsedMs;
	}
	return total;
}

static bool AllVulkanGpuTimestampsAvailable(std::span<const CompiledModuleVulkanProfileEvent> events)
{
	return !events.empty() && std::ranges::all_of(events, [](const CompiledModuleVulkanProfileEvent& event) {
		return event.gpuTimestampAvailable;
	});
}

static std::size_t SumVulkanWorkspaceBytes(const VulkanNativeInstructionPayload& payload)
{
	std::size_t total = 0;
	for (const auto& workspace : payload.workspaceTensors)
	{
		if (workspace.byteSize > std::numeric_limits<std::size_t>::max() - total)
		{
			throw std::runtime_error("Vulkan native workspace byte total overflows size_t");
		}
		total += static_cast<std::size_t>(workspace.byteSize);
	}
	return total;
}

static VulkanLaunchBreakdown ProfileVulkanLaunches(const VulkanProfileCase& profileCase)
{
	VulkanLaunchBreakdown result{ .name = profileCase.name, .batch = profileCase.batch };
	if (!IsVulkanDeviceAvailable())
	{
		result.message = "Vulkan compute device is not available";
		return result;
	}

	try
	{
		std::mt19937 rng(0);
		Graph graph = profileCase.build(result.batch, rng);
		Optimize(graph);

		CompiledModuleArtifact artifact;
		{
			auto begin = Clock::now();
			artifact = Compiler<Vulkan>::CompileArtifact(Detail::BuildExecutablePlanFromGraph(graph),
			                                             LiteNNBenchCompilerOptionsFromEnvironment());
			auto end = Clock::now();
			result.compileMs = clk::duration<double, std::milli>(end - begin).count();
		}
		result.backend = artifact.Backend() == CompiledModuleBackend::VulkanNative ? "vulkan_native" : "cpu_bridge";
		if (artifact.Backend() != CompiledModuleBackend::VulkanNative)
		{
			result.message = "compiled artifact did not use Vulkan native backend";
			return result;
		}

		const auto payload = DeserializeVulkanNativeInstructionPayload(artifact.Instructions());
		result.target = payload.target;
		result.featureFlags = payload.featureSet.flags;
		result.kernelCount = payload.kernels.size();
		result.externalTensorCount = artifact.ExternalTensorInfos().size();
		result.workspaceTensorCount = payload.workspaceTensors.size();
		result.workspaceBytes = SumVulkanWorkspaceBytes(payload);

		auto loadBegin = Clock::now();
		auto module = artifact.Load(Vulkan{});
		auto loadEnd = Clock::now();
		result.loadMs = clk::duration<double, std::milli>(loadEnd - loadBegin).count();

		auto profileInputs = profileCase.makeInputs(result.batch);
		result.inputUploadMs = profileInputs.uploadMs;
		auto& inputs = profileInputs.tensors;
		auto outputs = AllocateVulkanProfileOutputs(module);
		std::vector<CompiledModuleVulkanProfileEvent> events;
		const auto runInto = [&] {
			events.clear();
			module.RunTensorsInto(std::span<const Tensor<Vulkan>>(inputs), std::span<Tensor<Vulkan>>(outputs),
			                      CompiledModuleVulkanRunOptions{ .synchronize = true, .profileEvents = &events });
		};

		result.firstMs = TimedOnceMs(runInto);
		result.lastDispatchMs = SumVulkanDispatchWallMs(events);
		result.gpuTimestampAvailable = AllVulkanGpuTimestampsAvailable(events);
		result.lastGpuMs = result.gpuTimestampAvailable ? SumVulkanGpuMs(events) : 0.0;
		result.moduleCreationMs = SumVulkanModuleCreationMs(events);
		const auto timing = TimedRepeated(runInto, result.batch, 300.0);
		result.meanMs = timing.meanMs;
		result.lastDispatchMs = SumVulkanDispatchWallMs(events);
		result.gpuTimestampAvailable = AllVulkanGpuTimestampsAvailable(events);
		result.lastGpuMs = result.gpuTimestampAvailable ? SumVulkanGpuMs(events) : 0.0;
		result.moduleCreationMs = SumVulkanModuleCreationMs(events);
		result.outputDownloadMs = MeasureVulkanOutputDownloadMs(outputs);
		result.message = "ok";
	}
	catch (const std::exception& ex)
	{
		result.message = ex.what();
	}
	return result;
}

static VulkanLaunchBreakdown ProfileVulkanLaunches(const Case& profileCase)
{
	return ProfileVulkanLaunches(VulkanProfileCase{ .name = profileCase.name,
	                                                .build = profileCase.build,
	                                                .batch = profileCase.outShape[0],
	                                                .makeInputs = MakeVulkanProfileInputs });
}

static void WriteVulkanProfileCsv(const std::filesystem::path& path, std::span<const VulkanLaunchBreakdown> rows)
{
	std::ofstream out(path);
	if (!out)
	{
		throw std::runtime_error(std::format("Failed to open Vulkan profile CSV '{}'", path.string()));
	}
	out << "case,batch,backend,target,kernels,external_tensors,workspace_tensors,workspace_bytes,compile_ms,load_ms,"
	       "upload_ms,first_run_ms,"
	       "mean_run_ms,last_dispatch_wall_ms,gpu_timestamp_available,gpu_time_ms,download_ms,status\n";
	for (const auto& row : rows)
	{
		out << CsvEscape(row.name) << ',' << row.batch << ',' << CsvEscape(row.backend) << ',' << CsvEscape(row.target)
		    << ',' << row.kernelCount << ',' << row.externalTensorCount << ',' << row.workspaceTensorCount << ','
		    << row.workspaceBytes << ',' << row.compileMs << ',' << row.loadMs << ',' << row.inputUploadMs << ','
		    << row.firstMs << ',' << row.meanMs << ',' << row.lastDispatchMs << ','
		    << (row.gpuTimestampAvailable ? "true" : "false") << ',' << row.lastGpuMs << ',' << row.outputDownloadMs
		    << ',' << CsvEscape(row.message) << '\n';
	}
}
#endif

namespace
{

	struct ProfileCLIOptions
	{
		std::filesystem::path outDir = std::filesystem::current_path() / "profile_out";
		bool showHelp = false;
	};

	static bool HasPrefix(std::string_view value, std::string_view prefix)
	{
		return value.size() >= prefix.size() && value.substr(0, prefix.size()) == prefix;
	}

	static void PrintProfileUsage(std::ostream& os)
	{
		os << "Usage: litenn_profile [--out-dir <dir>] [out_dir]\n"
		   << "\n"
		   << "Options:\n"
		   << "  --out-dir <dir>   Directory for raw object files, assembly, and CSV profile output.\n"
		   << "  --out-dir=<dir>   Same as --out-dir <dir>.\n"
		   << "  -h, --help        Show this help text.\n"
		   << "\n"
		   << "The positional out_dir form is retained for existing scripts.\n";
	}

	static ProfileCLIOptions ParseProfileCLIOptions(int argc, char** argv)
	{
		ProfileCLIOptions options;
		bool outDirSet = false;

		for (int i = 1; i < argc; ++i)
		{
			const std::string_view arg(argv[i] ? argv[i] : "");
			if (arg == "-h" || arg == "--help")
			{
				options.showHelp = true;
				continue;
			}

			if (arg == "--out-dir")
			{
				if (outDirSet)
				{
					throw std::runtime_error("--out-dir was specified more than once");
				}
				if (i + 1 >= argc || std::string_view(argv[i + 1] ? argv[i + 1] : "").empty())
				{
					throw std::runtime_error("--out-dir requires a non-empty path");
				}
				options.outDir = std::filesystem::path(argv[++i]);
				outDirSet = true;
				continue;
			}

			constexpr std::string_view kOutDirPrefix = "--out-dir=";
			if (HasPrefix(arg, kOutDirPrefix))
			{
				if (outDirSet)
				{
					throw std::runtime_error("--out-dir was specified more than once");
				}
				const auto value = arg.substr(kOutDirPrefix.size());
				if (value.empty())
				{
					throw std::runtime_error("--out-dir requires a non-empty path");
				}
				options.outDir = std::filesystem::path(std::string(value));
				outDirSet = true;
				continue;
			}

			if (!arg.empty() && arg.front() == '-')
			{
				throw std::runtime_error(std::format("Unknown argument '{}'", arg));
			}

			if (outDirSet)
			{
				throw std::runtime_error(std::format("Unexpected positional argument '{}'", arg));
			}
			options.outDir = std::filesystem::path(argv[i]);
			outDirSet = true;
		}

		return options;
	}

} // namespace

int main(int argc, char** argv)
{
	ProfileCLIOptions cliOptions;
	try
	{
		cliOptions = ParseProfileCLIOptions(argc, argv);
	}
	catch (const std::exception& e)
	{
		std::cerr << "litenn_profile: " << e.what() << "\n\n";
		PrintProfileUsage(std::cerr);
		return 2;
	}

	if (cliOptions.showHelp)
	{
		PrintProfileUsage(std::cout);
		return 0;
	}

	const std::filesystem::path outDir = cliOptions.outDir;
	std::filesystem::create_directories(outDir);

	std::cout << "LiteNN AOT Profile Report\n";
	std::cout << "Object files written to: " << outDir.string() << "\n";
	std::cout
	    << "Instruction stats use objdump; set LITENN_OBJDUMP to override or LITENN_PROFILE_SKIP_OBJDUMP=1 to skip.\n";
	std::cout << std::string(116, '=') << "\n";

	std::vector<Case> cases = {
		{ "linear_b1", BuildLinear, { 1, 10 } },     { "linear_b32", BuildLinear, { 32, 10 } },
		{ "linear_b128", BuildLinear, { 128, 10 } }, { "linear_b512", BuildLinear, { 512, 10 } },
		{ "mlp128_b1", BuildMLP128, { 1, 10 } },     { "mlp128_b32", BuildMLP128, { 32, 10 } },
		{ "mlp128_b128", BuildMLP128, { 128, 10 } }, { "mlp128_b512", BuildMLP128, { 512, 10 } },
		{ "mlp512_b1", BuildMLP512, { 1, 10 } },     { "mlp512_b32", BuildMLP512, { 32, 10 } },
		{ "mlp512_b128", BuildMLP512, { 128, 10 } }, { "mlp512_b512", BuildMLP512, { 512, 10 } },
	};

	std::vector<CaseInstructionStats> instructionStats;
	instructionStats.reserve(cases.size());
	std::vector<CPUAOTParallelSelection> cpuAOTSelections;
	cpuAOTSelections.reserve(cases.size());

	std::cout << std::format("{:<14} {:>8} {:>10} {:>12} {:>12} {:>10} {:>12} {:>7} {:>7} {:>8}\n", "Case", "Batch",
	                         "Compile/ms", "Run/ms", "RunInto/ms", "Alloc/us", "Speedup", "FMAps", "VecLd", "StackVec");
	std::cout << std::string(116, '-') << "\n";

	for (const auto& c : cases)
	{
		const std::size_t batch = c.outShape[0];
		std::mt19937 rng(0);
		Graph g = c.build(batch, rng);
		Optimize(g);
		auto compilerOptions = LiteNNBenchCompilerOptionsFromEnvironment();
		auto cpuAOTSelection = AnalyzeCPUAOTParallelSelection(c.name, batch, g, compilerOptions);

		// Time compile
		auto cs = Clock::now();
		auto compiled = Compiler<CPU>::Compile(Detail::BuildExecutablePlanFromGraph(g), compilerOptions);
		auto ce = Clock::now();
		const double compileMs = clk::duration<double, std::milli>(ce - cs).count();
		cpuAOTSelection.objectUsesSidecar =
		    ObjectBytesContain(compiled.Instructions(), "litenn_cpu_matmul_bias_relu_parallel_f32");
		cpuAOTSelections.push_back(std::move(cpuAOTSelection));

		// Write the *raw* compiled object (the JIT-loaded code) for disassembly.
		// Note: WriteObjectFile() emits a "carrier" wrapper, not the executable code.
		const auto statsBeforeWrite = instructionStats.size();
		try
		{
			const auto bytes = compiled.Instructions();
			const auto objectPath = outDir / (c.name + ".o");
			std::ofstream f(objectPath, std::ios::binary);
			f.write(reinterpret_cast<const char*>(bytes.data()), static_cast<std::streamsize>(bytes.size()));
			f.close();
			if (!f)
			{
				throw std::runtime_error(std::format("failed to write {}", objectPath.string()));
			}
			instructionStats.push_back(
			    { c.name, outDir / (c.name + ".s"), AnalyzeObjectInstructions(objectPath, outDir / (c.name + ".s")) });
		}
		catch (...)
		{
		}
		if (instructionStats.size() == statsBeforeWrite)
		{
			instructionStats.push_back({ c.name, outDir / (c.name + ".s"), { .message = "object write failed" } });
		}

		// Build inputs once
		std::mt19937 rng2(0);
		std::uniform_real_distribution<float> dist(-1.0f, 1.0f);
		std::vector<float> data(batch * 784);
		for (auto& v : data)
		{
			v = dist(rng2);
		}
		auto in = Optimizer::MakeFloatTensor(std::span<const float>(data), { batch, 784 });
		std::vector<Tensor<CPU>> inputs;
		inputs.emplace_back(std::move(in));
		std::vector<Tensor<CPU>> outputs;
		outputs.emplace_back(Uninitialized, ShapeView{ std::vector<std::size_t>{ batch, c.outShape[1] } },
		                     DataType::Float32, CPU{});

		const auto tRun = TimedRun(compiled, inputs, batch);
		const auto tRunInto = TimedRunTensorsInto(compiled, inputs, outputs, batch);

		const double allocUs = (tRun.meanMs - tRunInto.meanMs) * 1000.0;
		const double speedup = tRun.meanMs / std::max(tRunInto.meanMs, 1e-6);

		const auto& stats = instructionStats.back().stats;
		const auto packedFMA = stats.available ? std::format("{}", stats.packedFMA) : "n/a";
		const auto vectorLoad = stats.available ? std::format("{}", stats.vectorLoad) : "n/a";
		const auto stackVectorOp = stats.available ? std::format("{}", stats.stackVectorOp) : "n/a";
		std::cout << std::format(
		    "{:<14} {:>8} {:>9.2f}ms {:>10.4f}ms {:>10.4f}ms {:>8.2f}us {:>10.2f}x {:>7} {:>7} {:>8}\n", c.name, batch,
		    compileMs, tRun.meanMs, tRunInto.meanMs, allocUs, speedup, packedFMA, vectorLoad, stackVectorOp);
	}

	std::cout << std::string(116, '=') << "\n";
	std::cout << "\nInstruction stats for subgraph_0, falling back to the first function when needed\n";
	std::cout << std::format("{:<14} {:<16} {:>7} {:>7} {:>7} {:>5} {:>5} {:>5} {:>7} {:>7} {:>7} {:>8} {:>7} {:>7}\n",
	                         "Case", "Function", "Lines", "FMAps", "FMAss", "zmm", "ymm", "xmm", "Gather", "Scatter",
	                         "VecLd", "ScalarMv", "Bcast", "StackV");
	std::cout << std::string(132, '-') << "\n";
	for (const auto& row : instructionStats)
	{
		const auto& s = row.stats;
		if (!s.available)
		{
			std::cout << std::format("{:<14} {:<16} {}\n", row.name, "-", s.message);
			continue;
		}
		std::cout << std::format(
		    "{:<14} {:<16} {:>7} {:>7} {:>7} {:>5} {:>5} {:>5} {:>7} {:>7} {:>7} {:>8} {:>7} {:>7}\n", row.name,
		    s.function, s.lines, s.packedFMA, s.scalarFMA, s.zmmPackedFMA, s.ymmPackedFMA, s.xmmPackedFMA, s.gather,
		    s.scatter, s.vectorLoad, s.scalarMove, s.broadcast, s.stackVectorOp);
	}
	std::cout << "\nAssembly files are written beside the object files when objdump succeeds.\n";

	std::cout << "\nCPU AOT parallel selection\n";
	std::cout << std::format("{:<14} {:>8} {:>7} {:>7} {:>9} {:>10} {:>9} {:>8} {}\n", "Case", "Batch", "Threads",
	                         "Layers", "Parallel", "TotalFLOPs", "Predicted", "Object", "Gate");
	std::cout << std::string(112, '-') << "\n";
	for (const auto& row : cpuAOTSelections)
	{
		std::cout << std::format("{:<14} {:>8} {:>7} {:>7} {:>9} {:>10} {:>9} {:>8} {}\n", row.name, row.batch,
		                         row.configuredThreads, row.fusedLayerCount, row.parallelLayerCount, row.totalFlops,
		                         row.predictedSidecar ? "sidecar" : "mlir", row.objectUsesSidecar ? "sidecar" : "mlir",
		                         row.gate);
		for (std::size_t i = 0; i < row.layers.size(); ++i)
		{
			const auto& layer = row.layers[i];
			std::cout << std::format("  layer {:<2} shape=({}x{}x{}) flops={} threads={} {}\n", i, layer.m, layer.k,
			                         layer.n, layer.flops, layer.selectedThreads, layer.reason);
		}
	}
	std::cout << "Predicted mirrors the public shape/thread gate; Object is detected from the emitted object symbol.\n";

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
		std::vector<CUDALaunchBreakdown> cudaRows;
		cudaRows.reserve(cases.size());
		std::cout << std::format("{:<14} {:>8} {:<11} {:<8} {:>7} {:>7} {:>7} {:>10} {:>10} {:>10} {:>10} {:>10} "
		                         "{:>11} {:>10} {:>10} {:>10} {:>10} {}\n",
		                         "Case", "Batch", "Backend", "Binary", "Kernels", "Lib", "PTX", "Workspace",
		                         "Bytes/Run", "Compile", "Load", "Native1", "NativeAvg", "NativeGB/s", "GraphAvg",
		                         "GraphGB/s", "Graph1", "Status");
		std::cout << std::string(202, '-') << "\n";
		for (const auto& c : cases)
		{
			const auto row = ProfileCUDALaunches(c);
			cudaRows.push_back(row);
			std::cout << std::format("{:<14} {:>8} {:<11} {:<8} {:>7} {:>7} {:>7} {:>10} {:>10} {:>8.2f}ms "
			                         "{:>8.2f}ms {:>8.4f}ms {:>9.4f}ms {:>8.2f} {:>8.4f}ms {:>8.2f} "
			                         "{:>8.4f}ms {}\n",
			                         row.name, row.batch, row.backend.empty() ? "-" : row.backend,
			                         row.binaryKind.empty() ? "-" : row.binaryKind, row.kernelCount,
			                         row.libraryKernelCount, row.ptxKernelCount, row.workspaceBytes,
			                         row.estimatedBytesPerRun, row.compileMs, row.loadMs, row.nativeFirstMs,
			                         row.nativeMeanMs, row.nativeEstimatedGBps, row.graphMeanMs, row.graphEstimatedGBps,
			                         row.graphFirstMs, row.message);
		}
		std::cout
		    << "Native1 is the first synchronized native RunInto. NativeAvg is steady synchronized native RunInto.\n";
		std::cout
		    << "Graph1 is first graph capture+run. GraphAvg is steady synchronized RunInto with graphReplay=Enabled.\n";
		std::cout << "Bytes/Run is a payload-visible estimate: inputs + outputs + constants + max workspace.\n";
		const auto csvPath = outDir / "cuda_profile.csv";
		WriteCUDAProfileCsv(csvPath, cudaRows);
		std::cout << "CUDA profile CSV written to: " << csvPath.string() << "\n";
	}
#else
	std::cout << "Unavailable: LiteNN was built without LITENN_ENABLE_CUDA.\n";
#endif

	std::cout << "\nVulkan native breakdowns\n";
#ifdef LITENN_ENABLE_VULKAN
	if (EnvFlagEnabled("LITENN_PROFILE_SKIP_VULKAN"))
	{
		std::cout << "Skipped by LITENN_PROFILE_SKIP_VULKAN.\n";
	}
	else if (!IsVulkanDeviceAvailable())
	{
		std::cout << "Vulkan compute device is not available.\n";
	}
	else
	{
		std::vector<VulkanProfileCase> vulkanOnlyCases = {
			{ .name = "binary_chain_b1",
			  .build = BuildBinaryChainProfileGraph,
			  .batch = 1,
			  .makeInputs = MakeVulkanBinaryChainProfileInputs },
			{ .name = "binary_chain_b32",
			  .build = BuildBinaryChainProfileGraph,
			  .batch = 32,
			  .makeInputs = MakeVulkanBinaryChainProfileInputs },
			{ .name = "binary_chain_b128",
			  .build = BuildBinaryChainProfileGraph,
			  .batch = 128,
			  .makeInputs = MakeVulkanBinaryChainProfileInputs },
			{ .name = "binary_chain_b512",
			  .build = BuildBinaryChainProfileGraph,
			  .batch = 512,
			  .makeInputs = MakeVulkanBinaryChainProfileInputs },
			{ .name = "binary_dag_b1",
			  .build = BuildBinaryDAGProfileGraph,
			  .batch = 1,
			  .makeInputs = MakeVulkanBinaryDAGProfileInputs },
			{ .name = "binary_dag_b32",
			  .build = BuildBinaryDAGProfileGraph,
			  .batch = 32,
			  .makeInputs = MakeVulkanBinaryDAGProfileInputs },
			{ .name = "binary_dag_b128",
			  .build = BuildBinaryDAGProfileGraph,
			  .batch = 128,
			  .makeInputs = MakeVulkanBinaryDAGProfileInputs },
			{ .name = "binary_dag_b512",
			  .build = BuildBinaryDAGProfileGraph,
			  .batch = 512,
			  .makeInputs = MakeVulkanBinaryDAGProfileInputs },
			{ .name = "mixed_elementwise_dag_b1",
			  .build = BuildMixedElementwiseDAGProfileGraph,
			  .batch = 1,
			  .makeInputs = MakeVulkanMixedElementwiseDAGProfileInputs },
			{ .name = "mixed_elementwise_dag_b32",
			  .build = BuildMixedElementwiseDAGProfileGraph,
			  .batch = 32,
			  .makeInputs = MakeVulkanMixedElementwiseDAGProfileInputs },
			{ .name = "mixed_elementwise_dag_b128",
			  .build = BuildMixedElementwiseDAGProfileGraph,
			  .batch = 128,
			  .makeInputs = MakeVulkanMixedElementwiseDAGProfileInputs },
			{ .name = "mixed_elementwise_dag_b512",
			  .build = BuildMixedElementwiseDAGProfileGraph,
			  .batch = 512,
			  .makeInputs = MakeVulkanMixedElementwiseDAGProfileInputs },
			{ .name = "branch_dag_b1",
			  .build = BuildBranchedBinaryDAGProfileGraph,
			  .batch = 1,
			  .makeInputs = MakeVulkanBranchedBinaryDAGProfileInputs },
			{ .name = "branch_dag_b32",
			  .build = BuildBranchedBinaryDAGProfileGraph,
			  .batch = 32,
			  .makeInputs = MakeVulkanBranchedBinaryDAGProfileInputs },
			{ .name = "branch_dag_b128",
			  .build = BuildBranchedBinaryDAGProfileGraph,
			  .batch = 128,
			  .makeInputs = MakeVulkanBranchedBinaryDAGProfileInputs },
			{ .name = "branch_dag_b512",
			  .build = BuildBranchedBinaryDAGProfileGraph,
			  .batch = 512,
			  .makeInputs = MakeVulkanBranchedBinaryDAGProfileInputs },
			{ .name = "reduce_sum_b1",
			  .build = BuildReduceSumProfileGraph,
			  .batch = 1,
			  .makeInputs = MakeVulkanProfileInputs },
			{ .name = "reduce_sum_b512",
			  .build = BuildReduceSumProfileGraph,
			  .batch = 512,
			  .makeInputs = MakeVulkanProfileInputs },
			{ .name = "reduce_mean_b512",
			  .build = BuildReduceMeanProfileGraph,
			  .batch = 512,
			  .makeInputs = MakeVulkanProfileInputs },
			{ .name = "reduce_max_b512",
			  .build = BuildReduceMaxProfileGraph,
			  .batch = 512,
			  .makeInputs = MakeVulkanProfileInputs },
			{ .name = "reduce_min_b512",
			  .build = BuildReduceMinProfileGraph,
			  .batch = 512,
			  .makeInputs = MakeVulkanProfileInputs },
			{ .name = "softmax_b1",
			  .build = BuildSoftmaxProfileGraph,
			  .batch = 1,
			  .makeInputs = MakeVulkanProfileInputs },
			{ .name = "softmax_b512",
			  .build = BuildSoftmaxProfileGraph,
			  .batch = 512,
			  .makeInputs = MakeVulkanProfileInputs },
			{ .name = "layernorm_b1",
			  .build = BuildLayerNormProfileGraph,
			  .batch = 1,
			  .makeInputs = MakeVulkanProfileInputs },
			{ .name = "layernorm_b512",
			  .build = BuildLayerNormProfileGraph,
			  .batch = 512,
			  .makeInputs = MakeVulkanProfileInputs },
			{ .name = "affine_layernorm_b512",
			  .build = BuildAffineLayerNormProfileGraph,
			  .batch = 512,
			  .makeInputs = MakeVulkanProfileInputs },
			{ .name = "rmsnorm_b512",
			  .build = BuildRMSNormProfileGraph,
			  .batch = 512,
			  .makeInputs = MakeVulkanProfileInputs },
		};
		std::cout << std::format("{:<14} {:>8} {:<13} {:<10} {:>7} {:>7} {:>7} {:>10} {:>10} {:>10} {:>10} {:>12} "
		                         "{:>11} {:>12} {:>10} {:>10} {}\n",
		                         "Case", "Batch", "Backend", "Target", "Kernels", "Ext", "WS", "WSBytes", "Compile",
		                         "Load", "Upload", "FirstRun", "MeanRun", "LastDispatch", "GPUTime", "Download",
		                         "Status");
		std::cout << std::string(198, '-') << "\n";
		std::vector<VulkanLaunchBreakdown> vulkanRows;
		vulkanRows.reserve(cases.size() + vulkanOnlyCases.size());
		const auto printVulkanRow = [](const VulkanLaunchBreakdown& row) {
			const std::string gpuTime =
			    row.gpuTimestampAvailable ? std::format("{:.4f}ms", row.lastGpuMs) : std::string("n/a");
			std::cout << std::format("{:<14} {:>8} {:<13} {:<10} {:>7} {:>7} {:>7} {:>10} {:>8.2f}ms {:>8.2f}ms "
			                         "{:>8.4f}ms {:>10.4f}ms {:>9.4f}ms {:>10.4f}ms {:>10} {:>8.4f}ms {}\n",
			                         row.name, row.batch, row.backend.empty() ? "-" : row.backend,
			                         row.target.empty() ? "-" : row.target, row.kernelCount, row.externalTensorCount,
			                         row.workspaceTensorCount, row.workspaceBytes, row.compileMs, row.loadMs,
			                         row.inputUploadMs, row.firstMs, row.meanMs, row.lastDispatchMs, gpuTime,
			                         row.outputDownloadMs, row.message);
		};
		for (const auto& c : cases)
		{
			const auto row = ProfileVulkanLaunches(c);
			vulkanRows.push_back(row);
			printVulkanRow(row);
		}
		for (const auto& c : vulkanOnlyCases)
		{
			const auto row = ProfileVulkanLaunches(c);
			vulkanRows.push_back(row);
			printVulkanRow(row);
		}
		std::cout << "FirstRun and MeanRun are synchronized RunInto wall times. LastDispatch is the sum of CPU-side\n";
		std::cout << "Vulkan dispatch wall times captured from the last profiled RunInto. GPUTime is the sum of\n";
		std::cout << "Vulkan timestamp-query elapsed time for devices whose compute queue supports timestamps.\n";
		std::cout << "Upload and Download are one-shot host/device tensor copy measurements outside RunInto.\n";
		const auto csvPath = outDir / "vulkan_profile.csv";
		WriteVulkanProfileCsv(csvPath, vulkanRows);
		std::cout << "Vulkan profile CSV written to: " << csvPath.string() << "\n";
	}
#else
	std::cout << "Unavailable: LiteNN was built without LITENN_ENABLE_VULKAN.\n";
#endif
	return 0;
}
