#include <benchmark/benchmark.h>

#include "CompilerOptionsEnv.h"

#include <LiteNN.h>
#include <LiteNN/ComputePrimitives.h>
#include <LiteNN/Initializer/Initializer.h>
#include <LiteNN/Layer/Layer.h>
#include <LiteNN/Layer/LoRA.h>
#include <LiteNN/Optimizer/Loss.h>
#include <LiteNN/Pass/ConstFoldPass.h>
#include <LiteNN/Pass/EGraphPass.h>
#include <LiteNN/Pass/FusionPass.h>
#include <LiteNN/Pass/InlinePass.h>
#include <LiteNN/Runtime/Interpreter.h>

#include <ggml-cpp.h>
#include <ggml-cpu.h>

#ifdef LITENN_BENCH_HAS_AOT
#include <LiteNN/Compiler/CompiledModule.h>
#endif

#include <algorithm>
#include <array>
#include <cmath>
#include <cstddef>
#include <cstdint>
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

namespace
{

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
	constexpr std::size_t kInputWidth = 784;
	constexpr std::size_t kLoRAOutputWidth = 512;
	constexpr std::size_t kLoRARank = 8;
	constexpr float kQuantizedBenchmarkTolerance = 1.0e-5F;

	struct GGMLLayerSpec
	{
		std::size_t inputWidth;
		std::size_t outputWidth;
		bool relu;
	};

	struct QuantizedLayerSpec
	{
		std::size_t inputWidth;
		std::size_t outputWidth;
		bool relu;
	};

	enum class QuantizedWeightKind
	{
		AffineInt8,
		PackedInt4,
		PackedFP4E2M1,
	};

	struct QuantizedLayer
	{
		Tensor<CPU> storage;
		QuantizationParams params;
		Tensor<CPU> dequantized;
		Tensor<CPU> bias;
		bool relu{};
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

	constexpr std::array<QuantizedLayerSpec, 1> kQuantizedLinearLayers = {
		QuantizedLayerSpec{ 784, 10, false },
	};

	constexpr std::array<QuantizedLayerSpec, 2> kQuantizedMLP128Layers = {
		QuantizedLayerSpec{ 784, 128, true },
		QuantizedLayerSpec{ 128, 10, false },
	};

	constexpr std::array<QuantizedLayerSpec, 3> kQuantizedMLP512Layers = {
		QuantizedLayerSpec{ 784, 512, true },
		QuantizedLayerSpec{ 512, 256, true },
		QuantizedLayerSpec{ 256, 10, false },
	};

	void SetThroughputCounters(benchmark::State& state, std::size_t batch);

	Graph BuildLinear(std::size_t batch, std::mt19937& rng)
	{
		ModelBuilder builder;
		Graph& graph = builder.UnsafeMutableGraph();
		const auto fc =
		    Layer::CreateLinear(builder, Initializer::XavierUniform({ 784, 10 }, rng), Initializer::Zeros({ 1, 10 }));
		Subgraph fwd;
		const auto in = fwd.AddParam(DataType::Float32, { batch, kInputWidth });
		fwd.SetResults({ Layer::AddLinear(fwd, fc, { in, 0 }) });
		graph.SetForward(graph.AddSubgraph(std::move(fwd)));
		return builder.UnsafeTakeGraph();
	}

	Graph BuildMLP128(std::size_t batch, std::mt19937& rng)
	{
		ModelBuilder builder;
		Graph& graph = builder.UnsafeMutableGraph();
		const auto h1 =
		    Layer::CreateLinear(builder, Initializer::XavierUniform({ 784, 128 }, rng), Initializer::Zeros({ 1, 128 }));
		const auto h2 =
		    Layer::CreateLinear(builder, Initializer::XavierUniform({ 128, 10 }, rng), Initializer::Zeros({ 1, 10 }));
		Subgraph fwd;
		const auto in = fwd.AddParam(DataType::Float32, { batch, kInputWidth });
		const auto a1 = Layer::AddReLU(fwd, Layer::AddLinear(fwd, h1, { in, 0 }));
		fwd.SetResults({ Layer::AddLinear(fwd, h2, a1) });
		graph.SetForward(graph.AddSubgraph(std::move(fwd)));
		return builder.UnsafeTakeGraph();
	}

	Graph BuildMLP512(std::size_t batch, std::mt19937& rng)
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
		const auto in = fwd.AddParam(DataType::Float32, { batch, kInputWidth });
		const auto a1 = Layer::AddReLU(fwd, Layer::AddLinear(fwd, h1, { in, 0 }));
		const auto a2 = Layer::AddReLU(fwd, Layer::AddLinear(fwd, h2, a1));
		fwd.SetResults({ Layer::AddLinear(fwd, h3, a2) });
		graph.SetForward(graph.AddSubgraph(std::move(fwd)));
		return builder.UnsafeTakeGraph();
	}

	Graph BuildLoRALinear(std::size_t batch, std::mt19937& rng, bool merged)
	{
		ModelBuilder builder;
		auto& graph = builder.UnsafeMutableGraph();
		const auto base =
		    Layer::CreateLinear(builder, Initializer::XavierUniform({ kInputWidth, kLoRAOutputWidth }, rng),
		                        Initializer::Zeros({ 1, kLoRAOutputWidth }));
		const auto adapter = Layer::CreateLinearLoRA(builder,
		                                             Layer::LoRAAdapterMetadata{ .targetName = "linear",
		                                                                         .rank = kLoRARank,
		                                                                         .alpha = static_cast<float>(kLoRARank),
		                                                                         .dtype = DataType::Float32 },
		                                             Initializer::XavierUniform({ kInputWidth, kLoRARank }, rng),
		                                             Initializer::XavierUniform({ kLoRARank, kLoRAOutputWidth }, rng));
		const auto active = merged ? Layer::MergeLinearLoRA(graph, base, adapter) : base;

		Subgraph fwd;
		const auto in = fwd.AddParam(DataType::Float32, { batch, kInputWidth });
		const auto out = merged ? Layer::AddLinear(fwd, active, { in, 0 })
		                        : Layer::AddLinearWithLoRA(fwd, active, adapter, { in, 0 });
		fwd.SetResults(std::vector<NodeOutput>{ out });
		graph.SetForward(graph.AddSubgraph(std::move(fwd)));
		return builder.UnsafeTakeGraph();
	}

	void Optimize(Graph& graph)
	{
		InlinePass{}.Run(graph);
		ConstFoldPass{}.Run(graph);
		FusionPass{}.Run(graph);
	}

	void OptimizeWithEGraph(Graph& graph)
	{
		InlinePass{}.Run(graph);
		ConstFoldPass{}.Run(graph);
		EGraphPass{}.Run(graph);
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
		std::vector<float> data(batch * kInputWidth);
		for (float& value : data)
		{
			value = dist(rng);
		}
		return data;
	}

	std::vector<Tensor<CPU>> MakeInputs(const std::vector<float>& data, std::size_t batch)
	{
		std::vector<Tensor<CPU>> inputs;
		inputs.emplace_back(Optimizer::MakeFloatTensor(std::span<const float>(data), { batch, kInputWidth }));
		return inputs;
	}

	float ReadF32(const Tensor<CPU>& tensor, std::size_t index)
	{
		return static_cast<const float*>(tensor.UnsafeRawData())[index];
	}

	std::vector<float> MakeRandomVector(std::size_t count, std::mt19937& rng, float minValue = -1.0F,
	                                    float maxValue = 1.0F)
	{
		std::uniform_real_distribution<float> dist(minValue, maxValue);
		std::vector<float> values(count);
		for (auto& value : values)
		{
			value = dist(rng);
		}
		return values;
	}

	Tensor<CPU> MakeRandomTensor(std::span<const std::size_t> shape, std::mt19937& rng, float minValue = -1.0F,
	                             float maxValue = 1.0F)
	{
		const auto values = MakeRandomVector(ShapeView{ shape }.NumElements(), rng, minValue, maxValue);
		return Optimizer::MakeFloatTensor(std::span<const float>(values), ShapeView{ shape });
	}

	const char* QuantizedWeightKindName(QuantizedWeightKind kind)
	{
		switch (kind)
		{
		case QuantizedWeightKind::AffineInt8:
			return "AffineInt8";
		case QuantizedWeightKind::PackedInt4:
			return "PackedInt4";
		case QuantizedWeightKind::PackedFP4E2M1:
			return "PackedFP4E2M1";
		}
		return "Unknown";
	}

	std::span<const QuantizedLayerSpec> GetQuantizedLayerSpecs(ModelKind kind)
	{
		switch (kind)
		{
		case ModelKind::Linear:
			return kQuantizedLinearLayers;
		case ModelKind::MLP128:
			return kQuantizedMLP128Layers;
		case ModelKind::MLP512:
			return kQuantizedMLP512Layers;
		}
		throw std::invalid_argument("unsupported quantized benchmark model kind");
	}

	QuantizedLayer MakeQuantizedLayer(const QuantizedLayerSpec& spec, QuantizedWeightKind kind, std::mt19937& rng)
	{
		const std::array shape{ spec.inputWidth, spec.outputWidth };
		const auto weight = MakeRandomTensor(shape, rng, -1.0F, 1.0F);
		const auto bias = MakeRandomTensor(std::array{ std::size_t{ 1 }, spec.outputWidth }, rng, -0.25F, 0.25F);
		switch (kind)
		{
		case QuantizedWeightKind::AffineInt8: {
			const auto quantized = QuantizeAffine(weight, PerTensorAffineQuantization(DataType::Int8, 1.0F / 64.0F));
			return { quantized.Storage(), quantized.Params(), DequantizeAffine(quantized), bias, spec.relu };
		}
		case QuantizedWeightKind::PackedInt4: {
			const auto quantized = QuantizeAffine(weight, PerTensorAffineQuantization(DataType::Int8, 1.0F / 4.0F));
			auto params =
			    PackedNibbleQuantization(PackedNibbleFormat::Int4, { spec.inputWidth, spec.outputWidth }, 1.0F / 4.0F);
			const auto packed = PackInteger4(quantized.Storage(), params);
			return { packed, params, DequantizePackedNibble(packed, params), bias, spec.relu };
		}
		case QuantizedWeightKind::PackedFP4E2M1: {
			auto params = PackedNibbleQuantization(PackedNibbleFormat::FP4E2M1, { spec.inputWidth, spec.outputWidth });
			const auto packed = PackFloat4(weight, params);
			return { packed, params, DequantizePackedNibble(packed, params), bias, spec.relu };
		}
		}
		throw std::invalid_argument("unsupported quantized weight kind");
	}

	std::vector<QuantizedLayer> MakeQuantizedLayers(ModelKind modelKind, QuantizedWeightKind weightKind)
	{
		std::mt19937 rng(42);
		std::vector<QuantizedLayer> layers;
		for (const auto& spec : GetQuantizedLayerSpecs(modelKind))
		{
			layers.push_back(MakeQuantizedLayer(spec, weightKind, rng));
		}
		return layers;
	}

	void AddBiasAndOptionalReLU(Tensor<CPU>& value, const Tensor<CPU>& bias, bool relu)
	{
		auto* data = static_cast<float*>(value.UnsafeRawData());
		const auto rows = value.Shape()[0];
		const auto cols = value.Shape()[1];
		for (std::size_t row = 0; row < rows; ++row)
		{
			for (std::size_t col = 0; col < cols; ++col)
			{
				auto& item = data[row * cols + col];
				item += ReadF32(bias, col);
				if (relu && item < 0.0F)
				{
					item = 0.0F;
				}
			}
		}
	}

	void ApplyReLU(Tensor<CPU>& value)
	{
		auto* data = static_cast<float*>(value.UnsafeRawData());
		for (std::size_t i = 0; i < value.NumElements(); ++i)
		{
			if (data[i] < 0.0F)
			{
				data[i] = 0.0F;
			}
		}
	}

	Tensor<CPU> RunNativeQuantizedModel(const Tensor<CPU>& input, const std::vector<QuantizedLayer>& layers)
	{
		Tensor<CPU> value = input;
		for (const auto& layer : layers)
		{
			value = EvalQuantizedLinear(value, layer.storage, layer.params, &layer.bias);
			if (layer.relu)
			{
				ApplyReLU(value);
			}
		}
		return value;
	}

	Tensor<CPU> RunDequantizedReferenceModel(const Tensor<CPU>& input, const std::vector<QuantizedLayer>& layers)
	{
		Tensor<CPU> value = input;
		for (const auto& layer : layers)
		{
			value = Detail::EvalBatchMatMul(value, layer.dequantized);
			AddBiasAndOptionalReLU(value, layer.bias, layer.relu);
		}
		return value;
	}

	float MaxAbsError(const Tensor<CPU>& lhs, const Tensor<CPU>& rhs)
	{
		if (lhs.Shape() != rhs.Shape() || lhs.DType() != DataType::Float32 || rhs.DType() != DataType::Float32)
		{
			throw std::runtime_error("MaxAbsError requires matching Float32 tensors");
		}
		float maxError = 0.0F;
		for (std::size_t i = 0; i < lhs.NumElements(); ++i)
		{
			maxError = std::max(maxError, std::fabs(ReadF32(lhs, i) - ReadF32(rhs, i)));
		}
		return maxError;
	}

	NodeId AddFloatConstant(Subgraph& sg, std::vector<float> values, std::vector<std::size_t> shape)
	{
		auto tensor = Optimizer::MakeFloatTensor(std::span<const float>(values), ShapeView{ shape })
		                  .CopyToDevice(PolymorphicDevice{ CPU{} });
		return sg.AddNode(ConstantNode{ std::move(tensor) }, { OutputInfo{ DataType::Float32, std::move(shape) } });
	}

	Graph BuildRedundantAOTGraph(std::size_t batch)
	{
		Graph graph;
		Subgraph fwd;
		const std::vector<std::size_t> matrixShape{ batch, kInputWidth };
		const auto elementCount = batch * kInputWidth;
		const auto in = fwd.AddParam(DataType::Float32, matrixShape);
		const auto zero = AddFloatConstant(fwd, std::vector<float>(elementCount, 0.0f), matrixShape);
		const auto one = AddFloatConstant(fwd, std::vector<float>(elementCount, 1.0f), matrixShape);
		const auto added = fwd.AddNode(BinaryOpNode{ BinaryOp::Add, { in, 0 }, { zero, 0 } },
		                               { OutputInfo{ DataType::Float32, matrixShape } });
		const auto neg1 =
		    fwd.AddNode(UnaryOpNode{ UnaryOp::Negate, { added, 0 } }, { OutputInfo{ DataType::Float32, matrixShape } });
		const auto neg2 =
		    fwd.AddNode(UnaryOpNode{ UnaryOp::Negate, { neg1, 0 } }, { OutputInfo{ DataType::Float32, matrixShape } });
		const auto flat = fwd.AddNode(ReshapeNode{ { neg2, 0 }, { elementCount } },
		                              { OutputInfo{ DataType::Float32, { elementCount } } });
		const auto restoredShape =
		    fwd.AddNode(ReshapeNode{ { flat, 0 }, matrixShape }, { OutputInfo{ DataType::Float32, matrixShape } });
		const auto transposed = fwd.AddNode(PermuteNode{ { restoredShape, 0 }, { 1, 0 } },
		                                    { OutputInfo{ DataType::Float32, { kInputWidth, batch } } });
		const auto restoredOrder =
		    fwd.AddNode(PermuteNode{ { transposed, 0 }, { 1, 0 } }, { OutputInfo{ DataType::Float32, matrixShape } });
		const auto broadcast = fwd.AddNode(BroadcastToNode{ { restoredOrder, 0 }, matrixShape },
		                                   { OutputInfo{ DataType::Float32, matrixShape } });
		const auto multiplied = fwd.AddNode(BinaryOpNode{ BinaryOp::Multiply, { broadcast, 0 }, { one, 0 } },
		                                    { OutputInfo{ DataType::Float32, matrixShape } });
		fwd.SetResults({ { multiplied, 0 } });
		graph.SetForward(graph.AddSubgraph(std::move(fwd)));
		return graph;
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
		{
			value = dist(rng);
		}
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
			{
				throw std::invalid_argument("GGML thread count must be positive");
			}

			backend_.reset(ggml_backend_cpu_init());
			if (!backend_)
			{
				throw std::runtime_error("ggml_backend_cpu_init failed");
			}
			ggml_backend_cpu_set_n_threads(backend_.get(), threadCount);

			const auto layerSpecs = GetGGMLLayerSpecs(kind);
			const auto tensorCount = 1 + static_cast<int>(layerSpecs.size() * 2);
			tensorContext_.reset(ggml_init({
			    .mem_size = ggml_tensor_overhead() * tensorCount,
			    .mem_buffer = nullptr,
			    .no_alloc = true,
			}));
			if (!tensorContext_)
			{
				throw std::runtime_error("ggml_init failed for tensor context");
			}

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
			{
				throw std::runtime_error("ggml_backend_alloc_ctx_tensors failed");
			}

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
			{
				throw std::runtime_error("ggml_init failed for graph context");
			}

			graph_ = ggml_new_graph(graphContext_.get());
			if (!graph_)
			{
				throw std::runtime_error("ggml_new_graph failed");
			}

			auto* current = input_;
			for (auto i = 0uz; i < layerSpecs.size(); ++i)
			{
				current = AddLinearLayer(graphContext_.get(), current, weights_[i], biases_[i], layerSpecs[i].relu);
			}

			ggml_build_forward_expand(graph_, current);
			result_ = ggml_graph_node(graph_, -1);
			allocator_.reset(ggml_gallocr_new(ggml_backend_get_default_buffer_type(backend_.get())));
			if (!allocator_)
			{
				throw std::runtime_error("ggml_gallocr_new failed");
			}
			if (!ggml_gallocr_alloc_graph(allocator_.get(), graph_))
			{
				throw std::runtime_error("ggml_gallocr_alloc_graph failed");
			}
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
		std::vector<double> values;
		std::vector<std::size_t> shape;
		DataType dtype = DataType::Float32;
	};

	std::vector<double> MakeCUDADeviceMatMulData(std::size_t count, DataType dtype);

	std::vector<Tensor<CUDA>> MakeCUDAInputs(const std::vector<float>& data, std::size_t batch)
	{
		std::vector<Tensor<CUDA>> inputs;
		auto cpuInput = Optimizer::MakeFloatTensor(std::span<const float>(data), { batch, 784 });
		inputs.push_back(cpuInput.CopyToDevice(CUDA{}));
		return inputs;
	}

	Graph BuildNativeMatMul(std::size_t batch, std::size_t width, DataType dtype)
	{
		Graph graph;
		Subgraph fwd;
		const auto lhs = fwd.AddParam(dtype, { batch, width });
		const auto rhs = fwd.AddParam(dtype, { width, width });
		const auto out = fwd.AddNode(BinaryOpNode{ BinaryOp::MatMul, { lhs, 0 }, { rhs, 0 } },
		                             { OutputInfo{ dtype, { batch, width } } });
		fwd.SetResults({ { out, 0 } });
		graph.SetForward(graph.AddSubgraph(std::move(fwd)));
		return graph;
	}

	std::vector<TensorInputSpec> MakeNativeMatMulInputs(std::size_t batch, std::size_t width, DataType dtype)
	{
		auto lhs = MakeCUDADeviceMatMulData(batch * width, dtype);
		auto rhs = MakeCUDADeviceMatMulData(width * width, dtype);

		std::vector<TensorInputSpec> specs;
		specs.push_back(TensorInputSpec{ .values = std::move(lhs), .shape = { batch, width }, .dtype = dtype });
		specs.push_back(TensorInputSpec{ .values = std::move(rhs), .shape = { width, width }, .dtype = dtype });
		return specs;
	}

	std::vector<Tensor<CUDA>> MakeCUDAInputs(std::span<const TensorInputSpec> specs)
	{
		std::vector<Tensor<CUDA>> inputs;
		inputs.reserve(specs.size());
		for (const auto& spec : specs)
		{
			Tensor<CPU> cpuInput(std::span<const double>(spec.values), ShapeView{ spec.shape }, spec.dtype);
			inputs.push_back(cpuInput.CopyToDevice(CUDA{}));
		}
		return inputs;
	}
#endif

	void SetThroughputCounters(benchmark::State& state, std::size_t batch)
	{
		state.SetItemsProcessed(static_cast<int64_t>(state.iterations()) * static_cast<int64_t>(batch));
		state.counters["samples_per_second"] =
		    benchmark::Counter(static_cast<double>(batch), benchmark::Counter::kIsIterationInvariantRate);
	}

#ifdef LITENN_ENABLE_CUDA
	std::vector<double> MakeCUDADeviceMatMulData(std::size_t count, DataType dtype)
	{
		std::vector<double> values(count);
		if (dtype == DataType::Int8 || dtype == DataType::UInt8)
		{
			for (auto i = 0uz; i < values.size(); ++i)
			{
				values[i] = static_cast<double>(i % 3);
			}
			return values;
		}

		std::mt19937 rng(11);
		std::uniform_real_distribution<double> dist(-1.0, 1.0);
		for (auto& value : values)
		{
			value = dist(rng);
		}
		return values;
	}

	bool SupportsCUDANativeMatMulBenchmarkDType(DataType dtype)
	{
		return dtype == DataType::Float32 || CUDASupportsNativeMatMul(dtype);
	}

	bool SupportsCUDADeviceMatMulBenchmarkDType(DataType dtype)
	{
		if (dtype != DataType::Float32 && dtype != DataType::Float64 && !CUDASupportsLowPrecisionStorage(dtype))
		{
			return false;
		}
		return dtype == DataType::Float32 || dtype == DataType::Float64 || CUDASupportsNativeMatMul(dtype);
	}

	void BMCUDADeviceMatMul(benchmark::State& state, std::size_t batch, std::size_t width, DataType dtype)
	{
		if (!IsCUDADeviceAvailable())
		{
			state.SkipWithError("CUDA device is not available");
			return;
		}
		if (!SupportsCUDADeviceMatMulBenchmarkDType(dtype))
		{
			state.SkipWithError("CUDA device does not support requested MatMul dtype");
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
			benchmark::DoNotOptimize(output.UnsafeRawData());
		}

		for (auto _ : state)
		{
			auto output = lhs.MatMul(rhs);
			benchmark::DoNotOptimize(output.UnsafeRawData());
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
		    std::format("{}/{}/batch:{}", backend, GetModelSpec(kind).name, batch), std::forward<Fn>(fn));
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
			auto outputs =
			    interp.RunForward(Detail::BuildExecutablePlanFromGraph(graph), std::span<const Tensor<CPU>>(inputs));
			benchmark::DoNotOptimize(outputs);
		}

		for (auto _ : state)
		{
			auto outputs =
			    interp.RunForward(Detail::BuildExecutablePlanFromGraph(graph), std::span<const Tensor<CPU>>(inputs));
			benchmark::DoNotOptimize(outputs);
			benchmark::ClobberMemory();
		}

		SetThroughputCounters(state, batch);
	}

	void BMInterpreterLoRA(benchmark::State& state, std::size_t batch, bool merged)
	{
		std::mt19937 rng(123);
		auto graph = BuildLoRALinear(batch, rng, merged);
		Optimize(graph);
		auto plan = Detail::BuildExecutablePlanFromGraph(graph);
		const auto inputData = MakeInputData(batch);
		const auto inputs = MakeInputs(inputData, batch);
		Runtime::Interpreter<CPU> interp;
		for (int i = 0; i < kWarmupIterations; ++i)
		{
			auto outputs = interp.RunForward(plan, std::span<const Tensor<CPU>>(inputs));
			benchmark::DoNotOptimize(outputs);
		}
		for (auto _ : state)
		{
			auto outputs = interp.RunForward(plan, std::span<const Tensor<CPU>>(inputs));
			benchmark::DoNotOptimize(outputs);
		}
		SetThroughputCounters(state, batch);
	}

	void BMNativeQuantizedLinearRun(benchmark::State& state, ModelKind kind, std::size_t batch,
	                                QuantizedWeightKind weightKind)
	{
		const auto layers = MakeQuantizedLayers(kind, weightKind);
		const auto inputData = MakeInputData(batch);
		const auto input = Optimizer::MakeFloatTensor(std::span<const float>(inputData), { batch, kInputWidth });
		const auto expected = RunDequantizedReferenceModel(input, layers);
		const auto actual = RunNativeQuantizedModel(input, layers);
		const auto maxError = MaxAbsError(actual, expected);
		if (maxError > kQuantizedBenchmarkTolerance)
		{
			state.SkipWithError(std::format("quantized native parity failed: max_abs_error={}", maxError).c_str());
			return;
		}

		for (int i = 0; i < kWarmupIterations; ++i)
		{
			auto output = RunNativeQuantizedModel(input, layers);
			benchmark::DoNotOptimize(output);
		}

		for (auto _ : state)
		{
			auto output = RunNativeQuantizedModel(input, layers);
			benchmark::DoNotOptimize(output);
			benchmark::ClobberMemory();
		}

		SetThroughputCounters(state, batch);
		state.counters["max_abs_error"] = benchmark::Counter(maxError, benchmark::Counter::kAvgIterations);
	}

	void BMDequantizedQuantizedLinearReferenceRun(benchmark::State& state, ModelKind kind, std::size_t batch,
	                                              QuantizedWeightKind weightKind)
	{
		const auto layers = MakeQuantizedLayers(kind, weightKind);
		const auto inputData = MakeInputData(batch);
		const auto input = Optimizer::MakeFloatTensor(std::span<const float>(inputData), { batch, kInputWidth });
		const auto expected = RunNativeQuantizedModel(input, layers);
		const auto actual = RunDequantizedReferenceModel(input, layers);
		const auto maxError = MaxAbsError(actual, expected);
		if (maxError > kQuantizedBenchmarkTolerance)
		{
			state.SkipWithError(std::format("quantized reference parity failed: max_abs_error={}", maxError).c_str());
			return;
		}

		for (int i = 0; i < kWarmupIterations; ++i)
		{
			auto output = RunDequantizedReferenceModel(input, layers);
			benchmark::DoNotOptimize(output);
		}

		for (auto _ : state)
		{
			auto output = RunDequantizedReferenceModel(input, layers);
			benchmark::DoNotOptimize(output);
			benchmark::ClobberMemory();
		}

		SetThroughputCounters(state, batch);
		state.counters["max_abs_error"] = benchmark::Counter(maxError, benchmark::Counter::kAvgIterations);
	}

#ifdef LITENN_BENCH_HAS_AOT
	std::vector<Tensor<CPU>> AllocateOutputs(const CompiledModule<CPU>& module)
	{
		std::vector<Tensor<CPU>> outputs;
		outputs.reserve(module.OutputSpecs().size());
		for (const auto& spec : module.OutputSpecs())
		{
			outputs.emplace_back(Uninitialized, ShapeView{ spec.type.StaticShape() }, spec.type.dtype, CPU{});
		}
		return outputs;
	}

#ifdef LITENN_ENABLE_CUDA
	std::vector<Tensor<CUDA>> AllocateCUDAOutputs(const CompiledModule<CUDA>& module)
	{
		std::vector<Tensor<CUDA>> outputs;
		outputs.reserve(module.OutputSpecs().size());
		for (const auto& spec : module.OutputSpecs())
		{
			outputs.emplace_back(Uninitialized, ShapeView{ spec.type.StaticShape() }, spec.type.dtype, CUDA{});
		}
		return outputs;
	}

	void BMCUDACPUFallbackRunTensorsInto(benchmark::State& state, ModelKind kind, std::size_t batch)
	{
		if (!IsCUDADeviceAvailable())
		{
			state.SkipWithError("CUDA device is not available");
			return;
		}

		std::mt19937 rng(42);
		auto graph = GetModelSpec(kind).build(batch, rng);
		Optimize(graph);
		auto options = LiteNNBenchCompilerOptionsFromEnvironment();
		options.enableCUDANativeAOT = false;
		auto module = Compiler<CUDA>::Compile(
		    Detail::BuildExecutablePlanFromGraph(graph),
		    CUDA{ .deviceIndex = 0, .hostFallbackPolicy = CUDAHostFallbackPolicy::Allow }, options);
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
			module.RunTensorsInto(std::span<const Tensor<CUDA>>(inputs), std::span<Tensor<CUDA>>(outputs));
		}

		for (auto _ : state)
		{
			module.RunTensorsInto(std::span<const Tensor<CUDA>>(inputs), std::span<Tensor<CUDA>>(outputs));
			benchmark::DoNotOptimize(outputs.data());
			benchmark::ClobberMemory();
		}

		SetThroughputCounters(state, batch);
	}

	void BMCUDANativeModelRunTensorsInto(benchmark::State& state, ModelKind kind, std::size_t batch,
	                                     bool enableGraphReplay = false)
	{
		if (!IsCUDADeviceAvailable())
		{
			state.SkipWithError("CUDA device is not available");
			return;
		}

		std::mt19937 rng(42);
		auto graph = GetModelSpec(kind).build(batch, rng);
		Optimize(graph);
		auto module = Compiler<CUDA>::Compile(Detail::BuildExecutablePlanFromGraph(graph), CUDA{},
		                                      LiteNNBenchCompilerOptionsFromEnvironment());
		if (module.Backend() != CompiledModuleBackend::CUDANative)
		{
			state.SkipWithError("expected CUDA native backend for model benchmark");
			return;
		}

		const auto inputData = MakeInputData(batch);
		auto inputs = MakeCUDAInputs(inputData, batch);
		auto outputs = AllocateCUDAOutputs(module);
		const CompiledModuleCUDARunOptions runOptions{ .graphReplay = enableGraphReplay
			                                                              ? CUDAGraphReplayMode::Enabled
			                                                              : CUDAGraphReplayMode::Disabled };

		for (int i = 0; i < kWarmupIterations; ++i)
		{
			module.RunTensorsInto(std::span<const Tensor<CUDA>>(inputs), std::span<Tensor<CUDA>>(outputs), runOptions);
		}

		for (auto _ : state)
		{
			module.RunTensorsInto(std::span<const Tensor<CUDA>>(inputs), std::span<Tensor<CUDA>>(outputs), runOptions);
			benchmark::DoNotOptimize(outputs.data());
			benchmark::ClobberMemory();
		}

		SetThroughputCounters(state, batch);
	}

	void BMCUDANativeGraphModelRunTensorsInto(benchmark::State& state, ModelKind kind, std::size_t batch)
	{
		BMCUDANativeModelRunTensorsInto(state, kind, batch, true);
	}

	void BMCUDANativeMatMulRunTensorsInto(benchmark::State& state, std::size_t batch, std::size_t width, DataType dtype)
	{
		if (!IsCUDADeviceAvailable())
		{
			state.SkipWithError("CUDA device is not available");
			return;
		}
		if (!SupportsCUDANativeMatMulBenchmarkDType(dtype))
		{
			state.SkipWithError("CUDA device does not support requested native MatMul dtype");
			return;
		}

		auto graph = BuildNativeMatMul(batch, width, dtype);
		auto module = Compiler<CUDA>::Compile(Detail::BuildExecutablePlanFromGraph(graph), CUDA{},
		                                      LiteNNBenchCompilerOptionsFromEnvironment());
		if (module.Backend() != CompiledModuleBackend::CUDANative)
		{
			state.SkipWithError("expected CUDA native backend for MatMul benchmark");
			return;
		}

		auto specs = MakeNativeMatMulInputs(batch, width, dtype);
		auto inputs = MakeCUDAInputs(specs);
		auto outputs = AllocateCUDAOutputs(module);

		for (int i = 0; i < kWarmupIterations; ++i)
		{
			module.RunTensorsInto(std::span<const Tensor<CUDA>>(inputs), std::span<Tensor<CUDA>>(outputs));
		}

		for (auto _ : state)
		{
			module.RunTensorsInto(std::span<const Tensor<CUDA>>(inputs), std::span<Tensor<CUDA>>(outputs));
			benchmark::DoNotOptimize(outputs.data());
			benchmark::ClobberMemory();
		}

		SetThroughputCounters(state, batch);
	}
#else
	void BMCUDACPUFallbackRunTensorsInto(benchmark::State& state, ModelKind, std::size_t)
	{
		state.SkipWithError("LiteNN benchmark build has no CUDA support");
	}

	void BMCUDANativeMatMulRunTensorsInto(benchmark::State& state, std::size_t, std::size_t, DataType)
	{
		state.SkipWithError("LiteNN benchmark build has no CUDA support");
	}

	void BMCUDANativeModelRunTensorsInto(benchmark::State& state, ModelKind, std::size_t)
	{
		state.SkipWithError("LiteNN benchmark build has no CUDA support");
	}

	void BMCUDANativeGraphModelRunTensorsInto(benchmark::State& state, ModelKind, std::size_t)
	{
		state.SkipWithError("LiteNN benchmark build has no CUDA support");
	}
#endif

#ifdef LITENN_ENABLE_VULKAN
	Graph BuildVulkanElementwiseAddGraph(std::size_t elementCount, DataType dtype = DataType::Float32)
	{
		Graph graph;
		Subgraph sg;
		const auto lhs = sg.AddParam(dtype, { elementCount });
		const auto rhs = sg.AddParam(dtype, { elementCount });
		const auto out = sg.AddNode(BinaryOpNode{ BinaryOp::Add, { lhs, 0 }, { rhs, 0 } },
		                            { OutputInfo{ dtype, { elementCount } } });
		sg.SetResults({ { out, 0 } });
		graph.SetForward(graph.AddSubgraph(std::move(sg)));
		graph.SetInputNames({ "lhs", "rhs" });
		graph.SetOutputNames({ "sum" });
		return graph;
	}

	Graph BuildVulkanUnaryAbsGraph(std::size_t elementCount, DataType dtype = DataType::Float32)
	{
		Graph graph;
		Subgraph sg;
		const auto input = sg.AddParam(dtype, { elementCount });
		const auto out =
		    sg.AddNode(UnaryOpNode{ UnaryOp::Abs, { input, 0 } }, { OutputInfo{ dtype, { elementCount } } });
		sg.SetResults({ { out, 0 } });
		graph.SetForward(graph.AddSubgraph(std::move(sg)));
		graph.SetInputNames({ "input" });
		graph.SetOutputNames({ "abs" });
		return graph;
	}

	Graph BuildVulkanBinaryChainGraph(std::size_t elementCount)
	{
		Graph graph;
		Subgraph sg;
		const auto a = sg.AddParam(DataType::Float32, { elementCount });
		const auto b = sg.AddParam(DataType::Float32, { elementCount });
		const auto c = sg.AddParam(DataType::Float32, { elementCount });
		const auto d = sg.AddParam(DataType::Float32, { elementCount });
		const auto first = sg.AddNode(BinaryOpNode{ BinaryOp::Add, { a, 0 }, { b, 0 } },
		                              { OutputInfo{ DataType::Float32, { elementCount } } });
		const auto second = sg.AddNode(BinaryOpNode{ BinaryOp::Multiply, { first, 0 }, { c, 0 } },
		                               { OutputInfo{ DataType::Float32, { elementCount } } });
		const auto out = sg.AddNode(BinaryOpNode{ BinaryOp::Subtract, { second, 0 }, { d, 0 } },
		                            { OutputInfo{ DataType::Float32, { elementCount } } });
		sg.SetResults({ { out, 0 } });
		graph.SetForward(graph.AddSubgraph(std::move(sg)));
		graph.SetInputNames({ "a", "b", "c", "d" });
		graph.SetOutputNames({ "out" });
		FusionPass{}.Run(graph);
		return graph;
	}

	Graph BuildVulkanBinaryDAGGraph(std::size_t elementCount)
	{
		Graph graph;
		Subgraph sg;
		const auto lhs = sg.AddParam(DataType::Float32, { elementCount });
		const auto rhs = sg.AddParam(DataType::Float32, { elementCount });
		const auto tail = sg.AddParam(DataType::Float32, { elementCount });
		const auto first = sg.AddNode(BinaryOpNode{ BinaryOp::Add, { lhs, 0 }, { rhs, 0 } },
		                              { OutputInfo{ DataType::Float32, { elementCount } } });
		const auto second = sg.AddNode(BinaryOpNode{ BinaryOp::Add, { lhs, 0 }, { tail, 0 } },
		                               { OutputInfo{ DataType::Float32, { elementCount } } });
		const auto out = sg.AddNode(BinaryOpNode{ BinaryOp::Add, { first, 0 }, { second, 0 } },
		                            { OutputInfo{ DataType::Float32, { elementCount } } });
		sg.SetResults({ { out, 0 } });
		graph.SetForward(graph.AddSubgraph(std::move(sg)));
		graph.SetInputNames({ "lhs", "rhs", "tail" });
		graph.SetOutputNames({ "out" });
		FusionPass{}.Run(graph);
		return graph;
	}

	Graph BuildVulkanMixedElementwiseDAGGraph(std::size_t elementCount)
	{
		Graph graph;
		Subgraph sg;
		const auto lhs = sg.AddParam(DataType::Float32, { elementCount });
		const auto rhs = sg.AddParam(DataType::Float32, { elementCount });
		const auto tail = sg.AddParam(DataType::Float32, { elementCount });
		const auto added = sg.AddNode(BinaryOpNode{ BinaryOp::Add, { lhs, 0 }, { rhs, 0 } },
		                              { OutputInfo{ DataType::Float32, { elementCount } } });
		const auto abs = sg.AddNode(UnaryOpNode{ UnaryOp::Abs, { added, 0 } },
		                            { OutputInfo{ DataType::Float32, { elementCount } } });
		const auto out = sg.AddNode(BinaryOpNode{ BinaryOp::Multiply, { abs, 0 }, { tail, 0 } },
		                            { OutputInfo{ DataType::Float32, { elementCount } } });
		sg.SetResults({ { out, 0 } });
		graph.SetForward(graph.AddSubgraph(std::move(sg)));
		graph.SetInputNames({ "lhs", "rhs", "tail" });
		graph.SetOutputNames({ "out" });
		return graph;
	}

	Graph BuildVulkanBranchedBinaryDAGGraph(std::size_t elementCount)
	{
		Graph graph;
		Subgraph sg;
		const auto a = sg.AddParam(DataType::Float32, { elementCount });
		const auto b = sg.AddParam(DataType::Float32, { elementCount });
		const auto c = sg.AddParam(DataType::Float32, { elementCount });
		const auto d = sg.AddParam(DataType::Float32, { elementCount });
		const auto e = sg.AddParam(DataType::Float32, { elementCount });
		const auto first = sg.AddNode(BinaryOpNode{ BinaryOp::Add, { a, 0 }, { b, 0 } },
		                              { OutputInfo{ DataType::Float32, { elementCount } } });
		const auto second = sg.AddNode(BinaryOpNode{ BinaryOp::Add, { c, 0 }, { d, 0 } },
		                               { OutputInfo{ DataType::Float32, { elementCount } } });
		const auto merged = sg.AddNode(BinaryOpNode{ BinaryOp::Multiply, { first, 0 }, { second, 0 } },
		                               { OutputInfo{ DataType::Float32, { elementCount } } });
		const auto tail = sg.AddNode(BinaryOpNode{ BinaryOp::Subtract, { first, 0 }, { e, 0 } },
		                             { OutputInfo{ DataType::Float32, { elementCount } } });
		const auto out = sg.AddNode(BinaryOpNode{ BinaryOp::Add, { merged, 0 }, { tail, 0 } },
		                            { OutputInfo{ DataType::Float32, { elementCount } } });
		sg.SetResults({ { out, 0 } });
		graph.SetForward(graph.AddSubgraph(std::move(sg)));
		graph.SetInputNames({ "a", "b", "c", "d", "e" });
		graph.SetOutputNames({ "out" });
		FusionPass{}.Run(graph);
		return graph;
	}

	Graph BuildVulkanMatMulGraph(std::size_t batch, std::size_t width)
	{
		Graph graph;
		Subgraph sg;
		const auto lhs = sg.AddParam(DataType::Float32, { batch, width });
		const auto rhs = sg.AddParam(DataType::Float32, { width, width });
		const auto out = sg.AddNode(BinaryOpNode{ BinaryOp::MatMul, { lhs, 0 }, { rhs, 0 } },
		                            { OutputInfo{ DataType::Float32, { batch, width } } });
		sg.SetResults({ { out, 0 } });
		graph.SetForward(graph.AddSubgraph(std::move(sg)));
		graph.SetInputNames({ "lhs", "rhs" });
		graph.SetOutputNames({ "out" });
		return graph;
	}

	Graph BuildVulkanMatMulBiasGraph(std::size_t batch, std::size_t width)
	{
		Graph graph;
		Subgraph sg;
		const auto lhs = sg.AddParam(DataType::Float32, { batch, width });
		const auto rhs = sg.AddParam(DataType::Float32, { width, width });
		const auto bias = sg.AddParam(DataType::Float32, { 1, width });
		const auto matmul = sg.AddNode(BinaryOpNode{ BinaryOp::MatMul, { lhs, 0 }, { rhs, 0 } },
		                               { OutputInfo{ DataType::Float32, { batch, width } } });
		const auto out = sg.AddNode(BinaryOpNode{ BinaryOp::Add, { matmul, 0 }, { bias, 0 } },
		                            { OutputInfo{ DataType::Float32, { batch, width } } });
		sg.SetResults({ { out, 0 } });
		graph.SetForward(graph.AddSubgraph(std::move(sg)));
		graph.SetInputNames({ "lhs", "rhs", "bias" });
		graph.SetOutputNames({ "out" });
		FusionPass{}.Run(graph);
		return graph;
	}

	Graph BuildVulkanLinearVariableGraph(std::size_t batch, std::size_t inputWidth, std::size_t outputWidth, bool relu,
	                                     std::uint32_t seed)
	{
		Graph graph;
		std::vector<double> weightData(inputWidth * outputWidth);
		std::vector<double> biasData(outputWidth);
		for (std::size_t i = 0; i < weightData.size(); ++i)
		{
			weightData[i] = static_cast<double>(static_cast<int>((i + seed) % 17) - 8) * 0.01;
		}
		for (std::size_t i = 0; i < biasData.size(); ++i)
		{
			biasData[i] = static_cast<double>(static_cast<int>((i + seed) % 7) - 3) * 0.001;
		}
		const auto weightIndex = graph.AddVariable(
		    Variable::Create(Tensor<CPU>(std::move(weightData), { inputWidth, outputWidth }, DataType::Float32)));
		const auto biasIndex = graph.AddVariable(
		    Variable::Create(Tensor<CPU>(std::move(biasData), { 1, outputWidth }, DataType::Float32)));
		graph.SetVariableName(weightIndex, std::format("linear{}_weight", seed));
		graph.SetVariableName(biasIndex, std::format("linear{}_bias", seed));

		Subgraph sg;
		const auto input = sg.AddParam(DataType::Float32, { batch, inputWidth });
		const auto weight = sg.AddNode(VariableRefNode{ weightIndex },
		                               { OutputInfo{ DataType::Float32, { inputWidth, outputWidth } } });
		const auto bias =
		    sg.AddNode(VariableRefNode{ biasIndex }, { OutputInfo{ DataType::Float32, { 1, outputWidth } } });
		const auto matmul = sg.AddNode(BinaryOpNode{ BinaryOp::MatMul, { input, 0 }, { weight, 0 } },
		                               { OutputInfo{ DataType::Float32, { batch, outputWidth } } });
		const auto shifted = sg.AddNode(BinaryOpNode{ BinaryOp::Add, { matmul, 0 }, { bias, 0 } },
		                                { OutputInfo{ DataType::Float32, { batch, outputWidth } } });
		NodeOutput result{ shifted, 0 };
		if (relu)
		{
			Tensor<CPU> zero({ 0.0f }, { 1, 1 }, DataType::Float32);
			const auto zeroNode = sg.AddNode(ConstantNode{ zero.CopyToDevice(PolymorphicDevice{ CPU{} }) },
			                                 { OutputInfo{ DataType::Float32, { 1, 1 } } });
			const auto reluOut = sg.AddNode(BinaryOpNode{ BinaryOp::Max, { shifted, 0 }, { zeroNode, 0 } },
			                                { OutputInfo{ DataType::Float32, { batch, outputWidth } } });
			result = { reluOut, 0 };
		}
		sg.SetResults({ result });
		graph.SetForward(graph.AddSubgraph(std::move(sg)));
		graph.SetInputNames({ "input" });
		graph.SetOutputNames({ relu ? "relu" : "out" });
		FusionPass{}.Run(graph);
		return graph;
	}

	Graph BuildVulkanHomogeneousLinearChainVariableGraph(std::size_t batch, std::size_t width)
	{
		Graph graph;
		const auto makeVariable = [&](std::string name, std::vector<double> data, std::vector<std::size_t> shape) {
			const auto index =
			    graph.AddVariable(Variable::Create(Tensor<CPU>(std::move(data), std::move(shape), DataType::Float32)));
			graph.SetVariableName(index, std::move(name));
			return index;
		};
		std::vector<double> weight0Data(width * width);
		std::vector<double> weight1Data(width * width);
		std::vector<double> bias0Data(width);
		std::vector<double> bias1Data(width);
		for (std::size_t i = 0; i < weight0Data.size(); ++i)
		{
			weight0Data[i] = i % (width + 1) == 0 ? 1.0 : 0.0;
			weight1Data[i] = i % (width + 1) == 0 ? 0.5 : 0.0;
		}
		for (std::size_t i = 0; i < width; ++i)
		{
			bias0Data[i] = 0.001;
			bias1Data[i] = -0.001;
		}
		const auto weight0Index = makeVariable("linear_chain_weight0", std::move(weight0Data), { width, width });
		const auto bias0Index = makeVariable("linear_chain_bias0", std::move(bias0Data), { 1, width });
		const auto weight1Index = makeVariable("linear_chain_weight1", std::move(weight1Data), { width, width });
		const auto bias1Index = makeVariable("linear_chain_bias1", std::move(bias1Data), { 1, width });

		Subgraph sg;
		const auto input = sg.AddParam(DataType::Float32, { batch, width });
		const auto weight0 =
		    sg.AddNode(VariableRefNode{ weight0Index }, { OutputInfo{ DataType::Float32, { width, width } } });
		const auto bias0 = sg.AddNode(VariableRefNode{ bias0Index }, { OutputInfo{ DataType::Float32, { 1, width } } });
		const auto matmul0 = sg.AddNode(BinaryOpNode{ BinaryOp::MatMul, { input, 0 }, { weight0, 0 } },
		                                { OutputInfo{ DataType::Float32, { batch, width } } });
		const auto hidden = sg.AddNode(BinaryOpNode{ BinaryOp::Add, { matmul0, 0 }, { bias0, 0 } },
		                               { OutputInfo{ DataType::Float32, { batch, width } } });
		const auto weight1 =
		    sg.AddNode(VariableRefNode{ weight1Index }, { OutputInfo{ DataType::Float32, { width, width } } });
		const auto bias1 = sg.AddNode(VariableRefNode{ bias1Index }, { OutputInfo{ DataType::Float32, { 1, width } } });
		const auto matmul1 = sg.AddNode(BinaryOpNode{ BinaryOp::MatMul, { hidden, 0 }, { weight1, 0 } },
		                                { OutputInfo{ DataType::Float32, { batch, width } } });
		const auto output = sg.AddNode(BinaryOpNode{ BinaryOp::Add, { matmul1, 0 }, { bias1, 0 } },
		                               { OutputInfo{ DataType::Float32, { batch, width } } });
		sg.SetResults({ { output, 0 } });
		graph.SetForward(graph.AddSubgraph(std::move(sg)));
		graph.SetInputNames({ "input" });
		graph.SetOutputNames({ "out" });
		FusionPass{}.Run(graph);
		return graph;
	}

	double EstimateLayerFlops(std::span<const GGMLLayerSpec> layers, std::size_t batch)
	{
		std::uint64_t flops = 0;
		for (const auto& layer : layers)
		{
			flops += 2ull * static_cast<std::uint64_t>(batch) * static_cast<std::uint64_t>(layer.inputWidth) *
			         static_cast<std::uint64_t>(layer.outputWidth);
		}
		return static_cast<double>(flops);
	}

	Graph BuildVulkanMLPVariableGraph(ModelKind kind, std::size_t batch)
	{
		const auto layers = GetGGMLLayerSpecs(kind);
		if (layers.size() < 2)
		{
			throw std::runtime_error("Vulkan graph MLP benchmark requires at least two layers");
		}

		Graph graph;
		const auto makeVariable = [&](std::string name, std::vector<double> data, std::vector<std::size_t> shape) {
			const auto index =
			    graph.AddVariable(Variable::Create(Tensor<CPU>(std::move(data), std::move(shape), DataType::Float32)));
			graph.SetVariableName(index, std::move(name));
			return index;
		};
		const auto makeWeightData = [](std::size_t inputWidth, std::size_t outputWidth, std::uint32_t seed) {
			std::vector<double> data(inputWidth * outputWidth);
			for (std::size_t i = 0; i < data.size(); ++i)
			{
				data[i] = static_cast<double>(static_cast<int>((i + seed) % 17) - 8) * 0.01;
			}
			return data;
		};
		const auto makeBiasData = [](std::size_t outputWidth, std::uint32_t seed) {
			std::vector<double> data(outputWidth);
			for (std::size_t i = 0; i < data.size(); ++i)
			{
				data[i] = static_cast<double>(static_cast<int>((i + seed) % 7) - 3) * 0.001;
			}
			return data;
		};

		Subgraph sg;
		const auto input = sg.AddParam(DataType::Float32, { batch, kInputWidth });
		NodeOutput activation{ input, 0 };
		std::optional<NodeId> zeroNode;
		for (std::size_t layerIndex = 0; layerIndex < layers.size(); ++layerIndex)
		{
			const auto& layer = layers[layerIndex];
			const auto seed = static_cast<std::uint32_t>(31u + layerIndex * 16u);
			const auto weightIndex = makeVariable(std::format("vulkan_mlp_weight{}", layerIndex),
			                                      makeWeightData(layer.inputWidth, layer.outputWidth, seed),
			                                      { layer.inputWidth, layer.outputWidth });
			const auto biasIndex = makeVariable(std::format("vulkan_mlp_bias{}", layerIndex),
			                                    makeBiasData(layer.outputWidth, seed), { 1, layer.outputWidth });
			const auto weight =
			    sg.AddNode(VariableRefNode{ weightIndex },
			               { OutputInfo{ DataType::Float32, { layer.inputWidth, layer.outputWidth } } });
			const auto bias =
			    sg.AddNode(VariableRefNode{ biasIndex }, { OutputInfo{ DataType::Float32, { 1, layer.outputWidth } } });
			const auto matmul = sg.AddNode(BinaryOpNode{ BinaryOp::MatMul, activation, { weight, 0 } },
			                               { OutputInfo{ DataType::Float32, { batch, layer.outputWidth } } });
			const auto shifted = sg.AddNode(BinaryOpNode{ BinaryOp::Add, { matmul, 0 }, { bias, 0 } },
			                                { OutputInfo{ DataType::Float32, { batch, layer.outputWidth } } });
			activation = { shifted, 0 };
			if (layer.relu)
			{
				if (!zeroNode)
				{
					Tensor<CPU> zero({ 0.0f }, { 1, 1 }, DataType::Float32);
					zeroNode = sg.AddNode(ConstantNode{ zero.CopyToDevice(PolymorphicDevice{ CPU{} }) },
					                      { OutputInfo{ DataType::Float32, { 1, 1 } } });
				}
				const auto relu = sg.AddNode(BinaryOpNode{ BinaryOp::Max, activation, { *zeroNode, 0 } },
				                             { OutputInfo{ DataType::Float32, { batch, layer.outputWidth } } });
				activation = { relu, 0 };
			}
		}
		sg.SetResults({ activation });
		graph.SetForward(graph.AddSubgraph(std::move(sg)));
		graph.SetInputNames({ "input" });
		graph.SetOutputNames({ "out" });
		FusionPass{}.Run(graph);
		return graph;
	}

	Graph BuildVulkanCastGraph(DataType srcType, DataType dstType, std::size_t elementCount)
	{
		Graph graph;
		Subgraph sg;
		const auto input = sg.AddParam(srcType, { elementCount });
		const auto out = sg.AddNode(CastNode{ { input, 0 }, dstType }, { OutputInfo{ dstType, { elementCount } } });
		sg.SetResults({ { out, 0 } });
		graph.SetForward(graph.AddSubgraph(std::move(sg)));
		graph.SetInputNames({ "input" });
		graph.SetOutputNames({ "out" });
		return graph;
	}

	Graph BuildVulkanReduceGraph(ReduceOp op, std::size_t batch, std::size_t width)
	{
		Graph graph;
		Subgraph sg;
		const auto input = sg.AddParam(DataType::Float32, { batch, width });
		const auto out =
		    sg.AddNode(ReduceOpNode{ op, { input, 0 }, 1 }, { OutputInfo{ DataType::Float32, { batch } } });
		sg.SetResults({ { out, 0 } });
		graph.SetForward(graph.AddSubgraph(std::move(sg)));
		graph.SetInputNames({ "input" });
		graph.SetOutputNames({ "out" });
		return graph;
	}

	Graph BuildVulkanSoftmaxGraph(std::size_t batch, std::size_t width)
	{
		Graph graph;
		Subgraph sg;
		const auto input = sg.AddParam(DataType::Float32, { batch, width });
		const auto out =
		    sg.AddNode(SoftmaxNode{ { input, 0 }, 1 }, { OutputInfo{ DataType::Float32, { batch, width } } });
		sg.SetResults({ { out, 0 } });
		graph.SetForward(graph.AddSubgraph(std::move(sg)));
		graph.SetInputNames({ "input" });
		graph.SetOutputNames({ "out" });
		return graph;
	}

	Graph BuildVulkanNormalizationGraph(NormalizationMode mode, std::size_t batch, std::size_t width)
	{
		Graph graph;
		Subgraph sg;
		const auto input = sg.AddParam(DataType::Float32, { batch, width });
		const auto out = sg.AddNode(NormalizationNode{ .input = { input, 0 },
		                                               .scale = std::nullopt,
		                                               .bias = std::nullopt,
		                                               .mode = mode,
		                                               .axis = 1,
		                                               .groupCount = 1,
		                                               .epsilon = 1e-5 },
		                            { OutputInfo{ DataType::Float32, { batch, width } } });
		sg.SetResults({ { out, 0 } });
		graph.SetForward(graph.AddSubgraph(std::move(sg)));
		graph.SetInputNames({ "input" });
		graph.SetOutputNames({ "out" });
		return graph;
	}

	Graph BuildVulkanAffineNormalizationGraph(NormalizationMode mode, std::size_t batch, std::size_t width)
	{
		Graph graph;
		std::vector<double> scale(width);
		std::vector<double> bias(width);
		for (std::size_t i = 0; i < width; ++i)
		{
			scale[i] = 0.75 + 0.01 * static_cast<double>(i % 17);
			bias[i] = -0.05 + 0.001 * static_cast<double>(i % 23);
		}
		const auto scaleIndex =
		    graph.AddVariable(Variable::Create(Tensor<CPU>(std::move(scale), { width }, DataType::Float32)));
		const auto biasIndex =
		    graph.AddVariable(Variable::Create(Tensor<CPU>(std::move(bias), { width }, DataType::Float32)));
		graph.SetVariableName(scaleIndex, "norm_scale");
		graph.SetVariableName(biasIndex, "norm_bias");

		Subgraph sg;
		const auto input = sg.AddParam(DataType::Float32, { batch, width });
		const auto scaleNode =
		    sg.AddNode(VariableRefNode{ scaleIndex }, { OutputInfo{ DataType::Float32, { width } } });
		const auto biasNode = sg.AddNode(VariableRefNode{ biasIndex }, { OutputInfo{ DataType::Float32, { width } } });
		const auto out = sg.AddNode(NormalizationNode{ .input = { input, 0 },
		                                               .scale = NodeOutput{ scaleNode, 0 },
		                                               .bias = NodeOutput{ biasNode, 0 },
		                                               .mode = mode,
		                                               .axis = 1,
		                                               .groupCount = 1,
		                                               .epsilon = 1e-5 },
		                            { OutputInfo{ DataType::Float32, { batch, width } } });
		sg.SetResults({ { out, 0 } });
		graph.SetForward(graph.AddSubgraph(std::move(sg)));
		graph.SetInputNames({ "input" });
		graph.SetOutputNames({ "out" });
		return graph;
	}

	Graph BuildVulkanGroupNormGraph(std::size_t elementCount, std::size_t groupCount)
	{
		Graph graph;
		Subgraph sg;
		const auto input = sg.AddParam(DataType::Float32, { elementCount });
		const auto out = sg.AddNode(NormalizationNode{ .input = { input, 0 },
		                                               .scale = std::nullopt,
		                                               .bias = std::nullopt,
		                                               .mode = NormalizationMode::GroupNorm,
		                                               .axis = 0,
		                                               .groupCount = groupCount,
		                                               .epsilon = 1e-5 },
		                            { OutputInfo{ DataType::Float32, { elementCount } } });
		sg.SetResults({ { out, 0 } });
		graph.SetForward(graph.AddSubgraph(std::move(sg)));
		graph.SetInputNames({ "input" });
		graph.SetOutputNames({ "out" });
		return graph;
	}

	Graph BuildVulkanPool2DGraph(PoolMode mode, std::size_t batch, std::size_t channels, std::size_t height,
	                             std::size_t width, std::array<std::size_t, 2> lowPads = { 0, 0 },
	                             std::array<std::size_t, 2> highPads = { 0, 0 }, bool countIncludePad = false)
	{
		Graph graph;
		Subgraph sg;
		const auto input = sg.AddParam(DataType::Float32, { batch, channels, height, width });
		const auto outHeight = lowPads[0] + height + highPads[0] - 1;
		const auto outWidth = lowPads[1] + width + highPads[1] - 1;
		const auto out = sg.AddNode(Pool2DNode{ .input = { input, 0 },
		                                        .mode = mode,
		                                        .kernelShape = { 2, 2 },
		                                        .strides = { 1, 1 },
		                                        .lowPads = { lowPads[0], lowPads[1] },
		                                        .highPads = { highPads[0], highPads[1] },
		                                        .countIncludePad = countIncludePad },
		                            { OutputInfo{ DataType::Float32, { batch, channels, outHeight, outWidth } } });
		sg.SetResults({ { out, 0 } });
		graph.SetForward(graph.AddSubgraph(std::move(sg)));
		graph.SetInputNames({ "input" });
		graph.SetOutputNames({ "out" });
		return graph;
	}

	Graph BuildVulkanConv2DGraph(std::size_t batch, std::size_t channels, std::size_t outChannels, std::size_t height,
	                             std::size_t width)
	{
		Graph graph;
		std::vector<double> weights(outChannels * channels * 3 * 3);
		std::vector<double> bias(outChannels);
		for (std::size_t i = 0; i < weights.size(); ++i)
		{
			weights[i] = static_cast<double>(static_cast<int>(i % 11) - 5) * 0.01;
		}
		for (std::size_t i = 0; i < bias.size(); ++i)
		{
			bias[i] = static_cast<double>(static_cast<int>(i % 7) - 3) * 0.001;
		}
		const auto weightIndex = graph.AddVariable(
		    Variable::Create(Tensor<CPU>(std::move(weights), { outChannels, channels, 3, 3 }, DataType::Float32)));
		const auto biasIndex =
		    graph.AddVariable(Variable::Create(Tensor<CPU>(std::move(bias), { outChannels }, DataType::Float32)));
		graph.SetVariableName(weightIndex, "conv_weight");
		graph.SetVariableName(biasIndex, "conv_bias");

		Subgraph sg;
		const auto input = sg.AddParam(DataType::Float32, { batch, channels, height, width });
		const auto weight = sg.AddNode(VariableRefNode{ weightIndex },
		                               { OutputInfo{ DataType::Float32, { outChannels, channels, 3, 3 } } });
		const auto biasNode =
		    sg.AddNode(VariableRefNode{ biasIndex }, { OutputInfo{ DataType::Float32, { outChannels } } });
		const auto out = sg.AddNode(Conv2DNode{ .input = { input, 0 },
		                                        .weight = { weight, 0 },
		                                        .bias = NodeOutput{ biasNode, 0 },
		                                        .strides = { 1, 1 },
		                                        .dilations = { 1, 1 },
		                                        .lowPads = { 1, 1 },
		                                        .highPads = { 1, 1 },
		                                        .groupCount = 1 },
		                            { OutputInfo{ DataType::Float32, { batch, outChannels, height, width } } });
		sg.SetResults({ { out, 0 } });
		graph.SetForward(graph.AddSubgraph(std::move(sg)));
		graph.SetInputNames({ "input" });
		graph.SetOutputNames({ "out" });
		return graph;
	}

	Graph BuildVulkanNearestUpsampleGraph(std::size_t batch, std::size_t channels, std::size_t height,
	                                      std::size_t width, std::size_t scale)
	{
		Graph graph;
		Subgraph sg;
		const auto input = sg.AddParam(DataType::Float32, { batch, channels, height, width });
		const auto outHeight = height * scale;
		const auto outWidth = width * scale;
		const auto out = sg.AddNode(UpsampleNode{ .input = { input, 0 },
		                                          .mode = UpsampleMode::Nearest,
		                                          .outputSpatialShape = { outHeight, outWidth },
		                                          .alignCorners = false },
		                            { OutputInfo{ DataType::Float32, { batch, channels, outHeight, outWidth } } });
		sg.SetResults({ { out, 0 } });
		graph.SetForward(graph.AddSubgraph(std::move(sg)));
		graph.SetInputNames({ "input" });
		graph.SetOutputNames({ "out" });
		return graph;
	}

	Graph BuildVulkanSliceGraph(std::size_t batch, std::size_t channels, std::size_t height, std::size_t width)
	{
		Graph graph;
		Subgraph sg;
		const auto sliceChannels = channels / 2;
		const auto input = sg.AddParam(DataType::Float32, { batch, channels, height, width });
		const auto out = sg.AddNode(SliceNode{ { input, 0 }, 1, sliceChannels / 2, sliceChannels },
		                            { OutputInfo{ DataType::Float32, { batch, sliceChannels, height, width } } });
		sg.SetResults({ { out, 0 } });
		graph.SetForward(graph.AddSubgraph(std::move(sg)));
		graph.SetInputNames({ "input" });
		graph.SetOutputNames({ "out" });
		return graph;
	}

	Graph BuildVulkanConcatGraph(std::size_t batch, std::size_t lhsChannels, std::size_t rhsChannels,
	                             std::size_t height, std::size_t width)
	{
		Graph graph;
		Subgraph sg;
		const auto lhs = sg.AddParam(DataType::Float32, { batch, lhsChannels, height, width });
		const auto rhs = sg.AddParam(DataType::Float32, { batch, rhsChannels, height, width });
		const auto out =
		    sg.AddNode(ConcatNode{ { { lhs, 0 }, { rhs, 0 } }, 1 },
		               { OutputInfo{ DataType::Float32, { batch, lhsChannels + rhsChannels, height, width } } });
		sg.SetResults({ { out, 0 } });
		graph.SetForward(graph.AddSubgraph(std::move(sg)));
		graph.SetInputNames({ "lhs", "rhs" });
		graph.SetOutputNames({ "out" });
		return graph;
	}

	Graph BuildVulkanConvTranspose2DGraph(std::size_t batch, std::size_t channels, std::size_t outChannels,
	                                      std::size_t height, std::size_t width)
	{
		Graph graph;
		std::vector<double> weights(channels * outChannels * 3 * 3);
		std::vector<double> bias(outChannels);
		for (std::size_t i = 0; i < weights.size(); ++i)
		{
			weights[i] = static_cast<double>(static_cast<int>(i % 13) - 6) * 0.01;
		}
		for (std::size_t i = 0; i < bias.size(); ++i)
		{
			bias[i] = static_cast<double>(static_cast<int>(i % 5) - 2) * 0.001;
		}
		const auto weightIndex = graph.AddVariable(
		    Variable::Create(Tensor<CPU>(std::move(weights), { channels, outChannels, 3, 3 }, DataType::Float32)));
		const auto biasIndex =
		    graph.AddVariable(Variable::Create(Tensor<CPU>(std::move(bias), { outChannels }, DataType::Float32)));
		graph.SetVariableName(weightIndex, "conv_transpose_weight");
		graph.SetVariableName(biasIndex, "conv_transpose_bias");

		Subgraph sg;
		const auto input = sg.AddParam(DataType::Float32, { batch, channels, height, width });
		const auto weight = sg.AddNode(VariableRefNode{ weightIndex },
		                               { OutputInfo{ DataType::Float32, { channels, outChannels, 3, 3 } } });
		const auto biasNode =
		    sg.AddNode(VariableRefNode{ biasIndex }, { OutputInfo{ DataType::Float32, { outChannels } } });
		const auto outHeight = (height - 1) * 2 + 3 - 2;
		const auto outWidth = (width - 1) * 2 + 3 - 2;
		const auto out = sg.AddNode(ConvTranspose2DNode{ .input = { input, 0 },
		                                                 .weight = { weight, 0 },
		                                                 .bias = NodeOutput{ biasNode, 0 },
		                                                 .strides = { 2, 2 },
		                                                 .dilations = { 1, 1 },
		                                                 .lowPads = { 1, 1 },
		                                                 .highPads = { 1, 1 },
		                                                 .outputPads = { 0, 0 },
		                                                 .groupCount = 1 },
		                            { OutputInfo{ DataType::Float32, { batch, outChannels, outHeight, outWidth } } });
		sg.SetResults({ { out, 0 } });
		graph.SetForward(graph.AddSubgraph(std::move(sg)));
		graph.SetInputNames({ "input" });
		graph.SetOutputNames({ "out" });
		return graph;
	}

	Graph BuildVulkanAffineGroupNormGraph(std::size_t elementCount, std::size_t groupCount)
	{
		Graph graph;
		std::vector<double> scale(elementCount);
		std::vector<double> bias(elementCount);
		for (std::size_t i = 0; i < elementCount; ++i)
		{
			scale[i] = 0.75 + 0.01 * static_cast<double>(i % 13);
			bias[i] = -0.05 + 0.001 * static_cast<double>(i % 19);
		}
		const auto scaleIndex =
		    graph.AddVariable(Variable::Create(Tensor<CPU>(std::move(scale), { elementCount }, DataType::Float32)));
		const auto biasIndex =
		    graph.AddVariable(Variable::Create(Tensor<CPU>(std::move(bias), { elementCount }, DataType::Float32)));
		graph.SetVariableName(scaleIndex, "group_norm_scale");
		graph.SetVariableName(biasIndex, "group_norm_bias");

		Subgraph sg;
		const auto input = sg.AddParam(DataType::Float32, { elementCount });
		const auto scaleNode =
		    sg.AddNode(VariableRefNode{ scaleIndex }, { OutputInfo{ DataType::Float32, { elementCount } } });
		const auto biasNode =
		    sg.AddNode(VariableRefNode{ biasIndex }, { OutputInfo{ DataType::Float32, { elementCount } } });
		const auto out = sg.AddNode(NormalizationNode{ .input = { input, 0 },
		                                               .scale = NodeOutput{ scaleNode, 0 },
		                                               .bias = NodeOutput{ biasNode, 0 },
		                                               .mode = NormalizationMode::GroupNorm,
		                                               .axis = 0,
		                                               .groupCount = groupCount,
		                                               .epsilon = 1e-5 },
		                            { OutputInfo{ DataType::Float32, { elementCount } } });
		sg.SetResults({ { out, 0 } });
		graph.SetForward(graph.AddSubgraph(std::move(sg)));
		graph.SetInputNames({ "input" });
		graph.SetOutputNames({ "out" });
		return graph;
	}

	std::string_view ReduceOpBenchmarkName(ReduceOp op)
	{
		switch (op)
		{
		case ReduceOp::Sum:
			return "SumAxis1";
		case ReduceOp::Mean:
			return "MeanAxis1";
		case ReduceOp::Max:
			return "MaxAxis1";
		case ReduceOp::Min:
			return "MinAxis1";
		default:
			return "Unknown";
		}
	}

	std::string_view NormalizationModeBenchmarkName(NormalizationMode mode)
	{
		switch (mode)
		{
		case NormalizationMode::LayerNorm:
			return "LayerNormAxis1";
		case NormalizationMode::RMSNorm:
			return "RMSNormAxis1";
		default:
			return "Unknown";
		}
	}

	std::string_view PoolModeBenchmarkName(PoolMode mode)
	{
		switch (mode)
		{
		case PoolMode::Max:
			return "Max";
		case PoolMode::Average:
			return "Average";
		default:
			return "Unknown";
		}
	}

	bool SupportsVulkanNativeCastBenchmarkDType(DataType dstType)
	{
		if (!IsVulkanDeviceAvailable())
		{
			return false;
		}
		const auto capabilities = QueryVulkanDeviceCapabilities(Vulkan{});
		switch (dstType)
		{
		case DataType::Float16:
			return capabilities.shaderFloat16Enabled && capabilities.storageBuffer16BitAccessEnabled;
		case DataType::Int8:
		case DataType::UInt8:
			return capabilities.shaderInt8Enabled && capabilities.storageBuffer8BitAccessEnabled;
		default:
			return dstType == DataType::Float32 || dstType == DataType::Int32;
		}
	}

	bool SupportsVulkanNativeElementwiseBenchmarkDType(DataType dtype)
	{
		if (!IsVulkanDeviceAvailable())
		{
			return false;
		}
		const auto capabilities = QueryVulkanDeviceCapabilities(Vulkan{});
		switch (dtype)
		{
		case DataType::Float32:
			return true;
		case DataType::Float16:
			return capabilities.shaderFloat16Enabled && capabilities.storageBuffer16BitAccessEnabled;
		case DataType::Int8:
		case DataType::UInt8:
			return capabilities.shaderInt8Enabled && capabilities.storageBuffer8BitAccessEnabled;
		default:
			return false;
		}
	}

	std::vector<float> MakeElementwiseInputData(std::size_t elementCount, unsigned int seed)
	{
		std::mt19937 rng(seed);
		std::uniform_real_distribution<float> dist(-1.0f, 1.0f);
		std::vector<float> data(elementCount);
		for (float& value : data)
		{
			value = dist(rng);
		}
		return data;
	}

	std::vector<Tensor<Vulkan>> MakeVulkanElementwiseInputs(const std::vector<float>& lhsData,
	                                                        const std::vector<float>& rhsData,
	                                                        DataType dtype = DataType::Float32)
	{
		if (lhsData.size() != rhsData.size())
		{
			throw std::invalid_argument("Vulkan elementwise benchmark inputs must have identical element counts");
		}

		std::vector<Tensor<Vulkan>> inputs;
		const std::vector<double> lhsValues(lhsData.begin(), lhsData.end());
		const std::vector<double> rhsValues(rhsData.begin(), rhsData.end());
		const Tensor<CPU> lhsCpu(std::span<const double>(lhsValues), { lhsData.size() }, dtype);
		const Tensor<CPU> rhsCpu(std::span<const double>(rhsValues), { rhsData.size() }, dtype);
		inputs.push_back(lhsCpu.CopyToDevice(Vulkan{}));
		inputs.push_back(rhsCpu.CopyToDevice(Vulkan{}));
		return inputs;
	}

	std::vector<Tensor<Vulkan>> MakeVulkanUnaryInputs(const std::vector<float>& data,
	                                                  DataType dtype = DataType::Float32)
	{
		std::vector<Tensor<Vulkan>> inputs;
		const std::vector<double> values(data.begin(), data.end());
		const Tensor<CPU> cpu(std::span<const double>(values), { data.size() }, dtype);
		inputs.push_back(cpu.CopyToDevice(Vulkan{}));
		return inputs;
	}

	std::vector<Tensor<Vulkan>> MakeVulkanSameShapeInputs(std::span<const std::vector<float>> inputData)
	{
		if (inputData.empty())
		{
			throw std::invalid_argument("Vulkan same-shape benchmark requires at least one input");
		}
		const auto elementCount = inputData.front().size();
		std::vector<Tensor<Vulkan>> inputs;
		inputs.reserve(inputData.size());
		for (const auto& data : inputData)
		{
			if (data.size() != elementCount)
			{
				throw std::invalid_argument("Vulkan same-shape benchmark inputs must have identical element counts");
			}
			const auto cpu = Optimizer::MakeFloatTensor(std::span<const float>(data), { elementCount });
			inputs.push_back(cpu.CopyToDevice(Vulkan{}));
		}
		return inputs;
	}

	std::vector<Tensor<Vulkan>> MakeVulkanMatMulInputs(const std::vector<float>& lhsData,
	                                                   const std::vector<float>& rhsData, std::size_t batch,
	                                                   std::size_t width)
	{
		if (lhsData.size() != batch * width || rhsData.size() != width * width)
		{
			throw std::invalid_argument("Vulkan MatMul benchmark inputs do not match the requested shape");
		}

		std::vector<Tensor<Vulkan>> inputs;
		const auto lhsCpu = Optimizer::MakeFloatTensor(std::span<const float>(lhsData), { batch, width });
		const auto rhsCpu = Optimizer::MakeFloatTensor(std::span<const float>(rhsData), { width, width });
		inputs.push_back(lhsCpu.CopyToDevice(Vulkan{}));
		inputs.push_back(rhsCpu.CopyToDevice(Vulkan{}));
		return inputs;
	}

	std::vector<Tensor<Vulkan>> MakeVulkanMatMulBiasInputs(const std::vector<float>& lhsData,
	                                                       const std::vector<float>& rhsData,
	                                                       const std::vector<float>& biasData, std::size_t batch,
	                                                       std::size_t width)
	{
		if (lhsData.size() != batch * width || rhsData.size() != width * width || biasData.size() != width)
		{
			throw std::invalid_argument("Vulkan MatMulBias benchmark inputs do not match the requested shape");
		}

		std::vector<Tensor<Vulkan>> inputs;
		const auto lhsCpu = Optimizer::MakeFloatTensor(std::span<const float>(lhsData), { batch, width });
		const auto rhsCpu = Optimizer::MakeFloatTensor(std::span<const float>(rhsData), { width, width });
		const auto biasCpu = Optimizer::MakeFloatTensor(std::span<const float>(biasData), { 1, width });
		inputs.push_back(lhsCpu.CopyToDevice(Vulkan{}));
		inputs.push_back(rhsCpu.CopyToDevice(Vulkan{}));
		inputs.push_back(biasCpu.CopyToDevice(Vulkan{}));
		return inputs;
	}

	std::vector<Tensor<Vulkan>> MakeVulkanModelInputs(const std::vector<float>& data, std::size_t batch,
	                                                  const Vulkan& device = Vulkan{})
	{
		if (data.size() != batch * kInputWidth)
		{
			throw std::invalid_argument("Vulkan model benchmark inputs do not match the requested shape");
		}

		std::vector<Tensor<Vulkan>> inputs;
		const auto inputCpu = Optimizer::MakeFloatTensor(std::span<const float>(data), { batch, kInputWidth });
		inputs.push_back(inputCpu.CopyToDevice(device));
		return inputs;
	}

	std::vector<Tensor<Vulkan>> MakeVulkanCastInputs(const std::vector<float>& data, std::size_t elementCount)
	{
		if (data.size() != elementCount)
		{
			throw std::invalid_argument("Vulkan cast benchmark input does not match the requested shape");
		}

		std::vector<Tensor<Vulkan>> inputs;
		const auto inputCpu = Optimizer::MakeFloatTensor(std::span<const float>(data), { elementCount });
		inputs.push_back(inputCpu.CopyToDevice(Vulkan{}));
		return inputs;
	}

	std::vector<Tensor<Vulkan>> MakeVulkanReduceInputs(const std::vector<float>& data, std::size_t batch,
	                                                   std::size_t width)
	{
		if (data.size() != batch * width)
		{
			throw std::invalid_argument("Vulkan reduce benchmark input does not match the requested shape");
		}

		std::vector<Tensor<Vulkan>> inputs;
		const auto inputCpu = Optimizer::MakeFloatTensor(std::span<const float>(data), { batch, width });
		inputs.push_back(inputCpu.CopyToDevice(Vulkan{}));
		return inputs;
	}

	std::vector<Tensor<Vulkan>> MakeVulkanPool2DInputs(const std::vector<float>& data, std::size_t batch,
	                                                   std::size_t channels, std::size_t height, std::size_t width)
	{
		if (data.size() != batch * channels * height * width)
		{
			throw std::invalid_argument("Vulkan Pool2D benchmark input does not match the requested shape");
		}

		std::vector<Tensor<Vulkan>> inputs;
		const auto inputCpu =
		    Optimizer::MakeFloatTensor(std::span<const float>(data), { batch, channels, height, width });
		inputs.push_back(inputCpu.CopyToDevice(Vulkan{}));
		return inputs;
	}

	std::vector<Tensor<Vulkan>> AllocateVulkanOutputs(const CompiledModule<Vulkan>& module)
	{
		return module.AllocateOutputTensors();
	}

	void BMVulkanNativeElementwiseAddRunTensorsInto(benchmark::State& state, std::size_t elementCount,
	                                                DataType dtype = DataType::Float32)
	{
		auto graph = BuildVulkanElementwiseAddGraph(elementCount, dtype);
		auto module = Compiler<Vulkan>::Compile(Detail::BuildExecutablePlanFromGraph(graph), Vulkan{},
		                                        LiteNNBenchCompilerOptionsFromEnvironment());
		if (module.Backend() != CompiledModuleBackend::VulkanNative)
		{
			state.SkipWithError("expected Vulkan native backend for elementwise Add benchmark");
			return;
		}

		const auto lhsData = MakeElementwiseInputData(elementCount, 0);
		const auto rhsData = MakeElementwiseInputData(elementCount, 1);
		auto inputs = MakeVulkanElementwiseInputs(lhsData, rhsData, dtype);
		auto outputs = AllocateVulkanOutputs(module);

		for (int i = 0; i < kWarmupIterations; ++i)
		{
			module.RunTensorsInto(std::span<const Tensor<Vulkan>>(inputs), std::span<Tensor<Vulkan>>(outputs));
		}

		for (auto _ : state)
		{
			module.RunTensorsInto(std::span<const Tensor<Vulkan>>(inputs), std::span<Tensor<Vulkan>>(outputs));
			benchmark::DoNotOptimize(outputs.data());
			benchmark::ClobberMemory();
		}

		state.SetItemsProcessed(static_cast<int64_t>(state.iterations()) * static_cast<int64_t>(elementCount));
		state.counters["elements_per_second"] =
		    benchmark::Counter(static_cast<double>(elementCount), benchmark::Counter::kIsIterationInvariantRate);
	}

	void BMVulkanNativeUnaryAbsRunTensorsInto(benchmark::State& state, std::size_t elementCount,
	                                          DataType dtype = DataType::Float32)
	{
		auto graph = BuildVulkanUnaryAbsGraph(elementCount, dtype);
		auto module = Compiler<Vulkan>::Compile(Detail::BuildExecutablePlanFromGraph(graph), Vulkan{},
		                                        LiteNNBenchCompilerOptionsFromEnvironment());
		if (module.Backend() != CompiledModuleBackend::VulkanNative)
		{
			state.SkipWithError("expected Vulkan native backend for unary Abs benchmark");
			return;
		}

		const auto inputData = MakeElementwiseInputData(elementCount, 17);
		auto inputs = MakeVulkanUnaryInputs(inputData, dtype);
		auto outputs = AllocateVulkanOutputs(module);

		for (int i = 0; i < kWarmupIterations; ++i)
		{
			module.RunTensorsInto(std::span<const Tensor<Vulkan>>(inputs), std::span<Tensor<Vulkan>>(outputs));
		}

		for (auto _ : state)
		{
			module.RunTensorsInto(std::span<const Tensor<Vulkan>>(inputs), std::span<Tensor<Vulkan>>(outputs));
			benchmark::DoNotOptimize(outputs.data());
			benchmark::ClobberMemory();
		}

		state.SetItemsProcessed(static_cast<int64_t>(state.iterations()) * static_cast<int64_t>(elementCount));
		state.counters["elements_per_second"] =
		    benchmark::Counter(static_cast<double>(elementCount), benchmark::Counter::kIsIterationInvariantRate);
	}

	void BMVulkanNativeBinaryChainRunTensorsInto(benchmark::State& state, std::size_t elementCount)
	{
		auto graph = BuildVulkanBinaryChainGraph(elementCount);
		auto module = Compiler<Vulkan>::Compile(Detail::BuildExecutablePlanFromGraph(graph), Vulkan{},
		                                        LiteNNBenchCompilerOptionsFromEnvironment());
		if (module.Backend() != CompiledModuleBackend::VulkanNative)
		{
			state.SkipWithError("expected Vulkan native backend for binary chain benchmark");
			return;
		}

		std::array inputData{
			MakeElementwiseInputData(elementCount, 25u),
			MakeElementwiseInputData(elementCount, 26u),
			MakeElementwiseInputData(elementCount, 27u),
			MakeElementwiseInputData(elementCount, 28u),
		};
		auto inputs = MakeVulkanSameShapeInputs(std::span<const std::vector<float>>(inputData));
		auto outputs = AllocateVulkanOutputs(module);

		for (int i = 0; i < kWarmupIterations; ++i)
		{
			module.RunTensorsInto(std::span<const Tensor<Vulkan>>(inputs), std::span<Tensor<Vulkan>>(outputs));
		}

		for (auto _ : state)
		{
			module.RunTensorsInto(std::span<const Tensor<Vulkan>>(inputs), std::span<Tensor<Vulkan>>(outputs));
			benchmark::DoNotOptimize(outputs.data());
			benchmark::ClobberMemory();
		}

		state.SetItemsProcessed(static_cast<int64_t>(state.iterations()) * static_cast<int64_t>(elementCount));
		state.counters["elements_per_second"] =
		    benchmark::Counter(static_cast<double>(elementCount), benchmark::Counter::kIsIterationInvariantRate);
	}

	void BMVulkanNativeBinaryDAGRunTensorsInto(benchmark::State& state, std::size_t elementCount)
	{
		auto graph = BuildVulkanBinaryDAGGraph(elementCount);
		auto module = Compiler<Vulkan>::Compile(Detail::BuildExecutablePlanFromGraph(graph), Vulkan{},
		                                        LiteNNBenchCompilerOptionsFromEnvironment());
		if (module.Backend() != CompiledModuleBackend::VulkanNative)
		{
			state.SkipWithError("expected Vulkan native backend for binary DAG benchmark");
			return;
		}

		std::array inputData{
			MakeElementwiseInputData(elementCount, 29u),
			MakeElementwiseInputData(elementCount, 30u),
			MakeElementwiseInputData(elementCount, 31u),
		};
		auto inputs = MakeVulkanSameShapeInputs(std::span<const std::vector<float>>(inputData));
		auto outputs = AllocateVulkanOutputs(module);

		for (int i = 0; i < kWarmupIterations; ++i)
		{
			module.RunTensorsInto(std::span<const Tensor<Vulkan>>(inputs), std::span<Tensor<Vulkan>>(outputs));
		}

		for (auto _ : state)
		{
			module.RunTensorsInto(std::span<const Tensor<Vulkan>>(inputs), std::span<Tensor<Vulkan>>(outputs));
			benchmark::DoNotOptimize(outputs.data());
			benchmark::ClobberMemory();
		}

		state.SetItemsProcessed(static_cast<int64_t>(state.iterations()) * static_cast<int64_t>(elementCount));
		state.counters["elements_per_second"] =
		    benchmark::Counter(static_cast<double>(elementCount), benchmark::Counter::kIsIterationInvariantRate);
	}

	void BMVulkanNativeMixedElementwiseDAGRunTensorsInto(benchmark::State& state, std::size_t elementCount)
	{
		auto graph = BuildVulkanMixedElementwiseDAGGraph(elementCount);
		auto module = Compiler<Vulkan>::Compile(Detail::BuildExecutablePlanFromGraph(graph), Vulkan{},
		                                        LiteNNBenchCompilerOptionsFromEnvironment());
		if (module.Backend() != CompiledModuleBackend::VulkanNative)
		{
			state.SkipWithError("expected Vulkan native backend for mixed elementwise DAG benchmark");
			return;
		}

		std::array inputData{
			MakeElementwiseInputData(elementCount, 35u),
			MakeElementwiseInputData(elementCount, 36u),
			MakeElementwiseInputData(elementCount, 37u),
		};
		auto inputs = MakeVulkanSameShapeInputs(std::span<const std::vector<float>>(inputData));
		auto outputs = AllocateVulkanOutputs(module);

		for (int i = 0; i < kWarmupIterations; ++i)
		{
			module.RunTensorsInto(std::span<const Tensor<Vulkan>>(inputs), std::span<Tensor<Vulkan>>(outputs));
		}

		for (auto _ : state)
		{
			module.RunTensorsInto(std::span<const Tensor<Vulkan>>(inputs), std::span<Tensor<Vulkan>>(outputs));
			benchmark::DoNotOptimize(outputs.data());
			benchmark::ClobberMemory();
		}

		state.SetItemsProcessed(static_cast<int64_t>(state.iterations()) * static_cast<int64_t>(elementCount));
		state.counters["elements_per_second"] =
		    benchmark::Counter(static_cast<double>(elementCount), benchmark::Counter::kIsIterationInvariantRate);
	}

	void BMVulkanNativeBranchedBinaryDAGRunTensorsInto(benchmark::State& state, std::size_t elementCount)
	{
		auto graph = BuildVulkanBranchedBinaryDAGGraph(elementCount);
		auto module = Compiler<Vulkan>::Compile(Detail::BuildExecutablePlanFromGraph(graph), Vulkan{},
		                                        LiteNNBenchCompilerOptionsFromEnvironment());
		if (module.Backend() != CompiledModuleBackend::VulkanNative)
		{
			state.SkipWithError("expected Vulkan native backend for branched binary DAG benchmark");
			return;
		}

		std::array inputData{
			MakeElementwiseInputData(elementCount, 32u), MakeElementwiseInputData(elementCount, 33u),
			MakeElementwiseInputData(elementCount, 34u), MakeElementwiseInputData(elementCount, 35u),
			MakeElementwiseInputData(elementCount, 36u),
		};
		auto inputs = MakeVulkanSameShapeInputs(std::span<const std::vector<float>>(inputData));
		auto outputs = AllocateVulkanOutputs(module);

		for (int i = 0; i < kWarmupIterations; ++i)
		{
			module.RunTensorsInto(std::span<const Tensor<Vulkan>>(inputs), std::span<Tensor<Vulkan>>(outputs));
		}

		for (auto _ : state)
		{
			module.RunTensorsInto(std::span<const Tensor<Vulkan>>(inputs), std::span<Tensor<Vulkan>>(outputs));
			benchmark::DoNotOptimize(outputs.data());
			benchmark::ClobberMemory();
		}

		state.SetItemsProcessed(static_cast<int64_t>(state.iterations()) * static_cast<int64_t>(elementCount));
		state.counters["elements_per_second"] =
		    benchmark::Counter(static_cast<double>(elementCount), benchmark::Counter::kIsIterationInvariantRate);
	}

	void BMVulkanNativeMatMulRunTensorsInto(benchmark::State& state, std::size_t batch, std::size_t width)
	{
		auto graph = BuildVulkanMatMulGraph(batch, width);
		auto module = Compiler<Vulkan>::Compile(Detail::BuildExecutablePlanFromGraph(graph), Vulkan{},
		                                        LiteNNBenchCompilerOptionsFromEnvironment());
		if (module.Backend() != CompiledModuleBackend::VulkanNative)
		{
			state.SkipWithError("expected Vulkan native backend for MatMul benchmark");
			return;
		}

		const auto lhsData = MakeElementwiseInputData(batch * width, 2);
		const auto rhsData = MakeElementwiseInputData(width * width, 3);
		auto inputs = MakeVulkanMatMulInputs(lhsData, rhsData, batch, width);
		auto outputs = AllocateVulkanOutputs(module);

		for (int i = 0; i < kWarmupIterations; ++i)
		{
			module.RunTensorsInto(std::span<const Tensor<Vulkan>>(inputs), std::span<Tensor<Vulkan>>(outputs));
		}

		for (auto _ : state)
		{
			module.RunTensorsInto(std::span<const Tensor<Vulkan>>(inputs), std::span<Tensor<Vulkan>>(outputs));
			benchmark::DoNotOptimize(outputs.data());
			benchmark::ClobberMemory();
		}

		SetThroughputCounters(state, batch);
		state.counters["flops"] = benchmark::Counter(static_cast<double>(2 * batch * width * width),
		                                             benchmark::Counter::kIsIterationInvariantRate);
	}

	void BMVulkanNativeMatMulBiasAddRunTensorsInto(benchmark::State& state, std::size_t batch, std::size_t width)
	{
		auto graph = BuildVulkanMatMulBiasGraph(batch, width);
		auto module = Compiler<Vulkan>::Compile(Detail::BuildExecutablePlanFromGraph(graph), Vulkan{},
		                                        LiteNNBenchCompilerOptionsFromEnvironment());
		if (module.Backend() != CompiledModuleBackend::VulkanNative)
		{
			state.SkipWithError("expected Vulkan native backend for MatMulBias benchmark");
			return;
		}

		const auto lhsData = MakeElementwiseInputData(batch * width, 4);
		const auto rhsData = MakeElementwiseInputData(width * width, 5);
		const auto biasData = MakeElementwiseInputData(width, 6);
		auto inputs = MakeVulkanMatMulBiasInputs(lhsData, rhsData, biasData, batch, width);
		auto outputs = AllocateVulkanOutputs(module);

		for (int i = 0; i < kWarmupIterations; ++i)
		{
			module.RunTensorsInto(std::span<const Tensor<Vulkan>>(inputs), std::span<Tensor<Vulkan>>(outputs));
		}

		for (auto _ : state)
		{
			module.RunTensorsInto(std::span<const Tensor<Vulkan>>(inputs), std::span<Tensor<Vulkan>>(outputs));
			benchmark::DoNotOptimize(outputs.data());
			benchmark::ClobberMemory();
		}

		SetThroughputCounters(state, batch);
		state.counters["flops"] = benchmark::Counter(static_cast<double>(2 * batch * width * width),
		                                             benchmark::Counter::kIsIterationInvariantRate);
	}

	void BMVulkanNativeManualMLP128RunTensorsInto(benchmark::State& state, std::size_t batch)
	{
		auto firstGraph = BuildVulkanLinearVariableGraph(batch, kInputWidth, 128, true, 31u);
		auto secondGraph = BuildVulkanLinearVariableGraph(batch, 128, 10, false, 47u);
		auto firstModule = Compiler<Vulkan>::Compile(Detail::BuildExecutablePlanFromGraph(firstGraph), Vulkan{},
		                                             LiteNNBenchCompilerOptionsFromEnvironment());
		auto secondModule = Compiler<Vulkan>::Compile(Detail::BuildExecutablePlanFromGraph(secondGraph), Vulkan{},
		                                              LiteNNBenchCompilerOptionsFromEnvironment());
		if (firstModule.Backend() != CompiledModuleBackend::VulkanNative ||
		    secondModule.Backend() != CompiledModuleBackend::VulkanNative)
		{
			state.SkipWithError("expected Vulkan native backend for manual MLP128 benchmark");
			return;
		}

		const auto inputData = MakeInputData(batch);
		std::vector<Tensor<Vulkan>> firstInputs;
		firstInputs.reserve(1);
		const auto inputCpu = Optimizer::MakeFloatTensor(std::span<const float>(inputData), { batch, kInputWidth });
		firstInputs.push_back(inputCpu.CopyToDevice(Vulkan{}));
		auto hidden = AllocateVulkanOutputs(firstModule);
		auto outputs = AllocateVulkanOutputs(secondModule);

		for (int i = 0; i < kWarmupIterations; ++i)
		{
			firstModule.RunTensorsInto(std::span<const Tensor<Vulkan>>(firstInputs), std::span<Tensor<Vulkan>>(hidden));
			secondModule.RunTensorsInto(std::span<const Tensor<Vulkan>>(hidden), std::span<Tensor<Vulkan>>(outputs));
		}

		for (auto _ : state)
		{
			firstModule.RunTensorsInto(std::span<const Tensor<Vulkan>>(firstInputs), std::span<Tensor<Vulkan>>(hidden));
			secondModule.RunTensorsInto(std::span<const Tensor<Vulkan>>(hidden), std::span<Tensor<Vulkan>>(outputs));
			benchmark::DoNotOptimize(outputs.data());
			benchmark::ClobberMemory();
		}

		SetThroughputCounters(state, batch);
		const auto flops = 2 * batch * ((kInputWidth * 128) + (128 * 10));
		state.counters["flops"] =
		    benchmark::Counter(static_cast<double>(flops), benchmark::Counter::kIsIterationInvariantRate);
	}

	void
	BMVulkanNativeGraphMLPRunTensorsInto(benchmark::State& state, ModelKind kind, std::size_t batch,
	                                     VulkanBufferResidency residency = VulkanBufferResidency::HostVisibleCoherent)
	{
		auto graph = BuildVulkanMLPVariableGraph(kind, batch);
		Vulkan device;
		device.bufferResidency = residency;
		auto module = Compiler<Vulkan>::Compile(Detail::BuildExecutablePlanFromGraph(graph), device,
		                                        LiteNNBenchCompilerOptionsFromEnvironment());
		if (module.Backend() != CompiledModuleBackend::VulkanNative)
		{
			state.SkipWithError("expected Vulkan native backend for whole-graph MLP benchmark");
			return;
		}

		const auto inputData = MakeInputData(batch);
		std::vector<Tensor<Vulkan>> inputs;
		inputs.reserve(1);
		const auto inputCpu = Optimizer::MakeFloatTensor(std::span<const float>(inputData), { batch, kInputWidth });
		inputs.push_back(inputCpu.CopyToDevice(device));
		auto outputs = AllocateVulkanOutputs(module);

		for (int i = 0; i < kWarmupIterations; ++i)
		{
			module.RunTensorsInto(std::span<const Tensor<Vulkan>>(inputs), std::span<Tensor<Vulkan>>(outputs));
		}

		for (auto _ : state)
		{
			module.RunTensorsInto(std::span<const Tensor<Vulkan>>(inputs), std::span<Tensor<Vulkan>>(outputs));
			benchmark::DoNotOptimize(outputs.data());
			benchmark::ClobberMemory();
		}

		SetThroughputCounters(state, batch);
		state.counters["flops"] = benchmark::Counter(EstimateLayerFlops(GetGGMLLayerSpecs(kind), batch),
		                                             benchmark::Counter::kIsIterationInvariantRate);
	}

	void BMVulkanNativeHomogeneousLinearChainRunTensorsInto(benchmark::State& state, std::size_t batch,
	                                                        std::size_t width)
	{
		auto graph = BuildVulkanHomogeneousLinearChainVariableGraph(batch, width);
		auto module = Compiler<Vulkan>::Compile(Detail::BuildExecutablePlanFromGraph(graph), Vulkan{},
		                                        LiteNNBenchCompilerOptionsFromEnvironment());
		if (module.Backend() != CompiledModuleBackend::VulkanNative)
		{
			state.SkipWithError("expected Vulkan native backend for homogeneous linear-chain benchmark");
			return;
		}

		std::vector<float> inputData(batch * width);
		for (std::size_t i = 0; i < inputData.size(); ++i)
		{
			inputData[i] = static_cast<float>(static_cast<int>(i % 29) - 14) * 0.01f;
		}
		std::vector<Tensor<Vulkan>> inputs;
		inputs.reserve(1);
		const auto inputCpu = Optimizer::MakeFloatTensor(std::span<const float>(inputData), { batch, width });
		inputs.push_back(inputCpu.CopyToDevice(Vulkan{}));
		auto outputs = AllocateVulkanOutputs(module);

		for (int i = 0; i < kWarmupIterations; ++i)
		{
			module.RunTensorsInto(std::span<const Tensor<Vulkan>>(inputs), std::span<Tensor<Vulkan>>(outputs));
		}

		for (auto _ : state)
		{
			module.RunTensorsInto(std::span<const Tensor<Vulkan>>(inputs), std::span<Tensor<Vulkan>>(outputs));
			benchmark::DoNotOptimize(outputs.data());
			benchmark::ClobberMemory();
		}

		SetThroughputCounters(state, batch);
		const auto flops = 2 * batch * width * width * 2;
		state.counters["flops"] =
		    benchmark::Counter(static_cast<double>(flops), benchmark::Counter::kIsIterationInvariantRate);
	}

	void BMVulkanNativeCastRunTensorsInto(benchmark::State& state, DataType dstType, std::size_t elementCount)
	{
		if (!SupportsVulkanNativeCastBenchmarkDType(dstType))
		{
			state.SkipWithError("Vulkan native cast benchmark dtype is not enabled on this device");
			return;
		}
		auto graph = BuildVulkanCastGraph(DataType::Float32, dstType, elementCount);
		auto module = Compiler<Vulkan>::Compile(Detail::BuildExecutablePlanFromGraph(graph), Vulkan{},
		                                        LiteNNBenchCompilerOptionsFromEnvironment());
		if (module.Backend() != CompiledModuleBackend::VulkanNative)
		{
			state.SkipWithError("expected Vulkan native backend for cast benchmark");
			return;
		}

		auto inputData = MakeElementwiseInputData(elementCount, dstType == DataType::UInt8 ? 8u : 7u);
		if (dstType == DataType::UInt8)
		{
			for (float& value : inputData)
			{
				value = std::abs(value) * 4.0f;
			}
		}
		auto inputs = MakeVulkanCastInputs(inputData, elementCount);
		auto outputs = AllocateVulkanOutputs(module);

		for (int i = 0; i < kWarmupIterations; ++i)
		{
			module.RunTensorsInto(std::span<const Tensor<Vulkan>>(inputs), std::span<Tensor<Vulkan>>(outputs));
		}

		for (auto _ : state)
		{
			module.RunTensorsInto(std::span<const Tensor<Vulkan>>(inputs), std::span<Tensor<Vulkan>>(outputs));
			benchmark::DoNotOptimize(outputs.data());
			benchmark::ClobberMemory();
		}

		state.SetItemsProcessed(static_cast<int64_t>(state.iterations()) * static_cast<int64_t>(elementCount));
		state.counters["elements_per_second"] =
		    benchmark::Counter(static_cast<double>(elementCount), benchmark::Counter::kIsIterationInvariantRate);
	}

	void BMVulkanNativeReduceRunTensorsInto(benchmark::State& state, ReduceOp op, std::size_t batch, std::size_t width)
	{
		auto graph = BuildVulkanReduceGraph(op, batch, width);
		auto module = Compiler<Vulkan>::Compile(Detail::BuildExecutablePlanFromGraph(graph), Vulkan{},
		                                        LiteNNBenchCompilerOptionsFromEnvironment());
		if (module.Backend() != CompiledModuleBackend::VulkanNative)
		{
			state.SkipWithError("expected Vulkan native backend for Reduce benchmark");
			return;
		}

		auto inputData = MakeElementwiseInputData(batch * width, 9);
		auto inputs = MakeVulkanReduceInputs(inputData, batch, width);
		auto outputs = AllocateVulkanOutputs(module);

		for (int i = 0; i < kWarmupIterations; ++i)
		{
			module.RunTensorsInto(std::span<const Tensor<Vulkan>>(inputs), std::span<Tensor<Vulkan>>(outputs));
		}

		for (auto _ : state)
		{
			module.RunTensorsInto(std::span<const Tensor<Vulkan>>(inputs), std::span<Tensor<Vulkan>>(outputs));
			benchmark::DoNotOptimize(outputs.data());
			benchmark::ClobberMemory();
		}

		SetThroughputCounters(state, batch);
		state.counters["reduced_elements"] =
		    benchmark::Counter(static_cast<double>(batch * width), benchmark::Counter::kIsIterationInvariantRate);
	}

	void BMVulkanNativeSoftmaxRunTensorsInto(benchmark::State& state, std::size_t batch, std::size_t width)
	{
		auto graph = BuildVulkanSoftmaxGraph(batch, width);
		auto module = Compiler<Vulkan>::Compile(Detail::BuildExecutablePlanFromGraph(graph), Vulkan{},
		                                        LiteNNBenchCompilerOptionsFromEnvironment());
		if (module.Backend() != CompiledModuleBackend::VulkanNative)
		{
			state.SkipWithError("expected Vulkan native backend for Softmax benchmark");
			return;
		}

		auto inputData = MakeElementwiseInputData(batch * width, 10);
		auto inputs = MakeVulkanReduceInputs(inputData, batch, width);
		auto outputs = AllocateVulkanOutputs(module);

		for (int i = 0; i < kWarmupIterations; ++i)
		{
			module.RunTensorsInto(std::span<const Tensor<Vulkan>>(inputs), std::span<Tensor<Vulkan>>(outputs));
		}

		for (auto _ : state)
		{
			module.RunTensorsInto(std::span<const Tensor<Vulkan>>(inputs), std::span<Tensor<Vulkan>>(outputs));
			benchmark::DoNotOptimize(outputs.data());
			benchmark::ClobberMemory();
		}

		SetThroughputCounters(state, batch);
		state.counters["softmax_elements"] =
		    benchmark::Counter(static_cast<double>(batch * width), benchmark::Counter::kIsIterationInvariantRate);
	}

	void BMVulkanNativeNormalizationRunTensorsInto(benchmark::State& state, NormalizationMode mode, std::size_t batch,
	                                               std::size_t width)
	{
		auto graph = BuildVulkanNormalizationGraph(mode, batch, width);
		auto module = Compiler<Vulkan>::Compile(Detail::BuildExecutablePlanFromGraph(graph), Vulkan{},
		                                        LiteNNBenchCompilerOptionsFromEnvironment());
		if (module.Backend() != CompiledModuleBackend::VulkanNative)
		{
			state.SkipWithError("expected Vulkan native backend for Normalization benchmark");
			return;
		}

		auto inputData = MakeElementwiseInputData(batch * width, mode == NormalizationMode::LayerNorm ? 11u : 12u);
		auto inputs = MakeVulkanReduceInputs(inputData, batch, width);
		auto outputs = AllocateVulkanOutputs(module);

		for (int i = 0; i < kWarmupIterations; ++i)
		{
			module.RunTensorsInto(std::span<const Tensor<Vulkan>>(inputs), std::span<Tensor<Vulkan>>(outputs));
		}

		for (auto _ : state)
		{
			module.RunTensorsInto(std::span<const Tensor<Vulkan>>(inputs), std::span<Tensor<Vulkan>>(outputs));
			benchmark::DoNotOptimize(outputs.data());
			benchmark::ClobberMemory();
		}

		SetThroughputCounters(state, batch);
		state.counters["normalized_elements"] =
		    benchmark::Counter(static_cast<double>(batch * width), benchmark::Counter::kIsIterationInvariantRate);
	}

	void BMVulkanNativeAffineNormalizationRunTensorsInto(benchmark::State& state, NormalizationMode mode,
	                                                     std::size_t batch, std::size_t width)
	{
		auto graph = BuildVulkanAffineNormalizationGraph(mode, batch, width);
		auto module = Compiler<Vulkan>::Compile(Detail::BuildExecutablePlanFromGraph(graph), Vulkan{},
		                                        LiteNNBenchCompilerOptionsFromEnvironment());
		if (module.Backend() != CompiledModuleBackend::VulkanNative)
		{
			state.SkipWithError("expected Vulkan native backend for affine Normalization benchmark");
			return;
		}

		auto inputData = MakeElementwiseInputData(batch * width, mode == NormalizationMode::LayerNorm ? 13u : 14u);
		auto inputs = MakeVulkanReduceInputs(inputData, batch, width);
		auto outputs = AllocateVulkanOutputs(module);

		for (int i = 0; i < kWarmupIterations; ++i)
		{
			module.RunTensorsInto(std::span<const Tensor<Vulkan>>(inputs), std::span<Tensor<Vulkan>>(outputs));
		}

		for (auto _ : state)
		{
			module.RunTensorsInto(std::span<const Tensor<Vulkan>>(inputs), std::span<Tensor<Vulkan>>(outputs));
			benchmark::DoNotOptimize(outputs.data());
			benchmark::ClobberMemory();
		}

		SetThroughputCounters(state, batch);
		state.counters["normalized_elements"] =
		    benchmark::Counter(static_cast<double>(batch * width), benchmark::Counter::kIsIterationInvariantRate);
	}

	void BMVulkanNativeGroupNormRunTensorsInto(benchmark::State& state, std::size_t elementCount,
	                                           std::size_t groupCount)
	{
		auto graph = BuildVulkanGroupNormGraph(elementCount, groupCount);
		auto module = Compiler<Vulkan>::Compile(Detail::BuildExecutablePlanFromGraph(graph), Vulkan{},
		                                        LiteNNBenchCompilerOptionsFromEnvironment());
		if (module.Backend() != CompiledModuleBackend::VulkanNative)
		{
			state.SkipWithError("expected Vulkan native backend for GroupNorm benchmark");
			return;
		}

		auto inputData = MakeElementwiseInputData(elementCount, 15u);
		auto inputs = MakeVulkanCastInputs(inputData, elementCount);
		auto outputs = AllocateVulkanOutputs(module);

		for (int i = 0; i < kWarmupIterations; ++i)
		{
			module.RunTensorsInto(std::span<const Tensor<Vulkan>>(inputs), std::span<Tensor<Vulkan>>(outputs));
		}

		for (auto _ : state)
		{
			module.RunTensorsInto(std::span<const Tensor<Vulkan>>(inputs), std::span<Tensor<Vulkan>>(outputs));
			benchmark::DoNotOptimize(outputs.data());
			benchmark::ClobberMemory();
		}

		state.SetItemsProcessed(static_cast<int64_t>(state.iterations()) * static_cast<int64_t>(elementCount));
		state.counters["normalized_elements"] =
		    benchmark::Counter(static_cast<double>(elementCount), benchmark::Counter::kIsIterationInvariantRate);
	}

	void BMVulkanNativeAffineGroupNormRunTensorsInto(benchmark::State& state, std::size_t elementCount,
	                                                 std::size_t groupCount)
	{
		auto graph = BuildVulkanAffineGroupNormGraph(elementCount, groupCount);
		auto module = Compiler<Vulkan>::Compile(Detail::BuildExecutablePlanFromGraph(graph), Vulkan{},
		                                        LiteNNBenchCompilerOptionsFromEnvironment());
		if (module.Backend() != CompiledModuleBackend::VulkanNative)
		{
			state.SkipWithError("expected Vulkan native backend for affine GroupNorm benchmark");
			return;
		}

		auto inputData = MakeElementwiseInputData(elementCount, 16u);
		auto inputs = MakeVulkanCastInputs(inputData, elementCount);
		auto outputs = AllocateVulkanOutputs(module);

		for (int i = 0; i < kWarmupIterations; ++i)
		{
			module.RunTensorsInto(std::span<const Tensor<Vulkan>>(inputs), std::span<Tensor<Vulkan>>(outputs));
		}

		for (auto _ : state)
		{
			module.RunTensorsInto(std::span<const Tensor<Vulkan>>(inputs), std::span<Tensor<Vulkan>>(outputs));
			benchmark::DoNotOptimize(outputs.data());
			benchmark::ClobberMemory();
		}

		state.SetItemsProcessed(static_cast<int64_t>(state.iterations()) * static_cast<int64_t>(elementCount));
		state.counters["normalized_elements"] =
		    benchmark::Counter(static_cast<double>(elementCount), benchmark::Counter::kIsIterationInvariantRate);
	}

	void BMVulkanNativePool2DRunTensorsInto(benchmark::State& state, PoolMode mode, std::size_t batch,
	                                        std::size_t channels, std::size_t height, std::size_t width,
	                                        std::array<std::size_t, 2> lowPads = { 0, 0 },
	                                        std::array<std::size_t, 2> highPads = { 0, 0 },
	                                        bool countIncludePad = false)
	{
		auto graph = BuildVulkanPool2DGraph(mode, batch, channels, height, width, lowPads, highPads, countIncludePad);
		auto module = Compiler<Vulkan>::Compile(Detail::BuildExecutablePlanFromGraph(graph), Vulkan{},
		                                        LiteNNBenchCompilerOptionsFromEnvironment());
		if (module.Backend() != CompiledModuleBackend::VulkanNative)
		{
			state.SkipWithError("expected Vulkan native backend for Pool2D benchmark");
			return;
		}

		const auto inputElementCount = batch * channels * height * width;
		auto inputData = MakeElementwiseInputData(inputElementCount, mode == PoolMode::Max ? 17u : 18u);
		auto inputs = MakeVulkanPool2DInputs(inputData, batch, channels, height, width);
		auto outputs = AllocateVulkanOutputs(module);

		for (int i = 0; i < kWarmupIterations; ++i)
		{
			module.RunTensorsInto(std::span<const Tensor<Vulkan>>(inputs), std::span<Tensor<Vulkan>>(outputs));
		}

		for (auto _ : state)
		{
			module.RunTensorsInto(std::span<const Tensor<Vulkan>>(inputs), std::span<Tensor<Vulkan>>(outputs));
			benchmark::DoNotOptimize(outputs.data());
			benchmark::ClobberMemory();
		}

		const auto outputElementCount =
		    batch * channels * (lowPads[0] + height + highPads[0] - 1) * (lowPads[1] + width + highPads[1] - 1);
		state.SetItemsProcessed(static_cast<int64_t>(state.iterations()) * static_cast<int64_t>(outputElementCount));
		state.counters["output_elements"] =
		    benchmark::Counter(static_cast<double>(outputElementCount), benchmark::Counter::kIsIterationInvariantRate);
	}

	void BMVulkanNativeConv2DRunTensorsInto(benchmark::State& state, std::size_t batch, std::size_t channels,
	                                        std::size_t outChannels, std::size_t height, std::size_t width)
	{
		auto graph = BuildVulkanConv2DGraph(batch, channels, outChannels, height, width);
		auto module = Compiler<Vulkan>::Compile(Detail::BuildExecutablePlanFromGraph(graph), Vulkan{},
		                                        LiteNNBenchCompilerOptionsFromEnvironment());
		if (module.Backend() != CompiledModuleBackend::VulkanNative)
		{
			state.SkipWithError("expected Vulkan native backend for Conv2D benchmark");
			return;
		}

		const auto inputElementCount = batch * channels * height * width;
		auto inputData = MakeElementwiseInputData(inputElementCount, 19u);
		auto inputs = MakeVulkanPool2DInputs(inputData, batch, channels, height, width);
		auto outputs = AllocateVulkanOutputs(module);

		for (int i = 0; i < kWarmupIterations; ++i)
		{
			module.RunTensorsInto(std::span<const Tensor<Vulkan>>(inputs), std::span<Tensor<Vulkan>>(outputs));
		}

		for (auto _ : state)
		{
			module.RunTensorsInto(std::span<const Tensor<Vulkan>>(inputs), std::span<Tensor<Vulkan>>(outputs));
			benchmark::DoNotOptimize(outputs.data());
			benchmark::ClobberMemory();
		}

		const auto outputElementCount = batch * outChannels * height * width;
		state.SetItemsProcessed(static_cast<int64_t>(state.iterations()) * static_cast<int64_t>(outputElementCount));
		state.counters["output_elements"] =
		    benchmark::Counter(static_cast<double>(outputElementCount), benchmark::Counter::kIsIterationInvariantRate);
	}

	void BMVulkanNativeNearestUpsampleRunTensorsInto(benchmark::State& state, std::size_t batch, std::size_t channels,
	                                                 std::size_t height, std::size_t width, std::size_t scale)
	{
		auto graph = BuildVulkanNearestUpsampleGraph(batch, channels, height, width, scale);
		auto module = Compiler<Vulkan>::Compile(Detail::BuildExecutablePlanFromGraph(graph), Vulkan{},
		                                        LiteNNBenchCompilerOptionsFromEnvironment());
		if (module.Backend() != CompiledModuleBackend::VulkanNative)
		{
			state.SkipWithError("expected Vulkan native backend for nearest Upsample benchmark");
			return;
		}

		const auto inputElementCount = batch * channels * height * width;
		auto inputData = MakeElementwiseInputData(inputElementCount, 20u);
		auto inputs = MakeVulkanPool2DInputs(inputData, batch, channels, height, width);
		auto outputs = AllocateVulkanOutputs(module);

		for (int i = 0; i < kWarmupIterations; ++i)
		{
			module.RunTensorsInto(std::span<const Tensor<Vulkan>>(inputs), std::span<Tensor<Vulkan>>(outputs));
		}

		for (auto _ : state)
		{
			module.RunTensorsInto(std::span<const Tensor<Vulkan>>(inputs), std::span<Tensor<Vulkan>>(outputs));
			benchmark::DoNotOptimize(outputs.data());
			benchmark::ClobberMemory();
		}

		const auto outputElementCount = batch * channels * height * scale * width * scale;
		state.SetItemsProcessed(static_cast<int64_t>(state.iterations()) * static_cast<int64_t>(outputElementCount));
		state.counters["output_elements"] =
		    benchmark::Counter(static_cast<double>(outputElementCount), benchmark::Counter::kIsIterationInvariantRate);
	}

	void BMVulkanNativeSliceRunTensorsInto(benchmark::State& state, std::size_t batch, std::size_t channels,
	                                       std::size_t height, std::size_t width)
	{
		auto graph = BuildVulkanSliceGraph(batch, channels, height, width);
		auto module = Compiler<Vulkan>::Compile(Detail::BuildExecutablePlanFromGraph(graph), Vulkan{},
		                                        LiteNNBenchCompilerOptionsFromEnvironment());
		if (module.Backend() != CompiledModuleBackend::VulkanNative)
		{
			state.SkipWithError("expected Vulkan native backend for Slice benchmark");
			return;
		}

		const auto inputElementCount = batch * channels * height * width;
		auto inputData = MakeElementwiseInputData(inputElementCount, 22u);
		auto inputs = MakeVulkanPool2DInputs(inputData, batch, channels, height, width);
		auto outputs = AllocateVulkanOutputs(module);

		for (int i = 0; i < kWarmupIterations; ++i)
		{
			module.RunTensorsInto(std::span<const Tensor<Vulkan>>(inputs), std::span<Tensor<Vulkan>>(outputs));
		}

		for (auto _ : state)
		{
			module.RunTensorsInto(std::span<const Tensor<Vulkan>>(inputs), std::span<Tensor<Vulkan>>(outputs));
			benchmark::DoNotOptimize(outputs.data());
			benchmark::ClobberMemory();
		}

		const auto outputElementCount = batch * (channels / 2) * height * width;
		state.SetItemsProcessed(static_cast<int64_t>(state.iterations()) * static_cast<int64_t>(outputElementCount));
		state.counters["output_elements"] =
		    benchmark::Counter(static_cast<double>(outputElementCount), benchmark::Counter::kIsIterationInvariantRate);
	}

	void BMVulkanNativeConcatRunTensorsInto(benchmark::State& state, std::size_t batch, std::size_t lhsChannels,
	                                        std::size_t rhsChannels, std::size_t height, std::size_t width)
	{
		auto graph = BuildVulkanConcatGraph(batch, lhsChannels, rhsChannels, height, width);
		auto module = Compiler<Vulkan>::Compile(Detail::BuildExecutablePlanFromGraph(graph), Vulkan{},
		                                        LiteNNBenchCompilerOptionsFromEnvironment());
		if (module.Backend() != CompiledModuleBackend::VulkanNative)
		{
			state.SkipWithError("expected Vulkan native backend for Concat benchmark");
			return;
		}

		const auto lhsElementCount = batch * lhsChannels * height * width;
		const auto rhsElementCount = batch * rhsChannels * height * width;
		auto lhsData = MakeElementwiseInputData(lhsElementCount, 23u);
		auto rhsData = MakeElementwiseInputData(rhsElementCount, 24u);
		std::vector<Tensor<Vulkan>> inputs;
		inputs.reserve(2);
		const auto lhsCpu =
		    Optimizer::MakeFloatTensor(std::span<const float>(lhsData), { batch, lhsChannels, height, width });
		const auto rhsCpu =
		    Optimizer::MakeFloatTensor(std::span<const float>(rhsData), { batch, rhsChannels, height, width });
		inputs.push_back(lhsCpu.CopyToDevice(Vulkan{}));
		inputs.push_back(rhsCpu.CopyToDevice(Vulkan{}));
		auto outputs = AllocateVulkanOutputs(module);

		for (int i = 0; i < kWarmupIterations; ++i)
		{
			module.RunTensorsInto(std::span<const Tensor<Vulkan>>(inputs), std::span<Tensor<Vulkan>>(outputs));
		}

		for (auto _ : state)
		{
			module.RunTensorsInto(std::span<const Tensor<Vulkan>>(inputs), std::span<Tensor<Vulkan>>(outputs));
			benchmark::DoNotOptimize(outputs.data());
			benchmark::ClobberMemory();
		}

		const auto outputElementCount = batch * (lhsChannels + rhsChannels) * height * width;
		state.SetItemsProcessed(static_cast<int64_t>(state.iterations()) * static_cast<int64_t>(outputElementCount));
		state.counters["output_elements"] =
		    benchmark::Counter(static_cast<double>(outputElementCount), benchmark::Counter::kIsIterationInvariantRate);
	}

	void BMVulkanNativeConvTranspose2DRunTensorsInto(benchmark::State& state, std::size_t batch, std::size_t channels,
	                                                 std::size_t outChannels, std::size_t height, std::size_t width)
	{
		auto graph = BuildVulkanConvTranspose2DGraph(batch, channels, outChannels, height, width);
		auto module = Compiler<Vulkan>::Compile(Detail::BuildExecutablePlanFromGraph(graph), Vulkan{},
		                                        LiteNNBenchCompilerOptionsFromEnvironment());
		if (module.Backend() != CompiledModuleBackend::VulkanNative)
		{
			state.SkipWithError("expected Vulkan native backend for ConvTranspose2D benchmark");
			return;
		}

		const auto inputElementCount = batch * channels * height * width;
		auto inputData = MakeElementwiseInputData(inputElementCount, 21u);
		auto inputs = MakeVulkanPool2DInputs(inputData, batch, channels, height, width);
		auto outputs = AllocateVulkanOutputs(module);

		for (int i = 0; i < kWarmupIterations; ++i)
		{
			module.RunTensorsInto(std::span<const Tensor<Vulkan>>(inputs), std::span<Tensor<Vulkan>>(outputs));
		}

		for (auto _ : state)
		{
			module.RunTensorsInto(std::span<const Tensor<Vulkan>>(inputs), std::span<Tensor<Vulkan>>(outputs));
			benchmark::DoNotOptimize(outputs.data());
			benchmark::ClobberMemory();
		}

		const auto outputElementCount = batch * outChannels * ((height - 1) * 2 + 1) * ((width - 1) * 2 + 1);
		state.SetItemsProcessed(static_cast<int64_t>(state.iterations()) * static_cast<int64_t>(outputElementCount));
		state.counters["output_elements"] =
		    benchmark::Counter(static_cast<double>(outputElementCount), benchmark::Counter::kIsIterationInvariantRate);
	}

	void BMVulkanNativeModelRunTensorsInto(benchmark::State& state, ModelKind kind, std::size_t batch)
	{
		std::mt19937 rng(42);
		auto graph = GetModelSpec(kind).build(batch, rng);
		Optimize(graph);
		auto module = Compiler<Vulkan>::Compile(Detail::BuildExecutablePlanFromGraph(graph), Vulkan{},
		                                        LiteNNBenchCompilerOptionsFromEnvironment());
		if (module.Backend() != CompiledModuleBackend::VulkanNative)
		{
			state.SkipWithError("expected Vulkan native backend for model benchmark");
			return;
		}

		const auto inputData = MakeInputData(batch);
		auto inputs = MakeVulkanModelInputs(inputData, batch);
		auto outputs = AllocateVulkanOutputs(module);

		for (int i = 0; i < kWarmupIterations; ++i)
		{
			module.RunTensorsInto(std::span<const Tensor<Vulkan>>(inputs), std::span<Tensor<Vulkan>>(outputs));
		}

		for (auto _ : state)
		{
			module.RunTensorsInto(std::span<const Tensor<Vulkan>>(inputs), std::span<Tensor<Vulkan>>(outputs));
			benchmark::DoNotOptimize(outputs.data());
			benchmark::ClobberMemory();
		}

		SetThroughputCounters(state, batch);
	}
#endif

	void BMAOTRun(benchmark::State& state, ModelKind kind, std::size_t batch)
	{
		std::mt19937 rng(42);
		auto graph = GetModelSpec(kind).build(batch, rng);
		Optimize(graph);

		auto options = LiteNNBenchCompilerOptionsFromEnvironment();
		auto compiled = Compiler<CPU>::Compile(Detail::BuildExecutablePlanFromGraph(graph), options);
		auto module = CompiledModule<CPU>::Load(compiled.Image());
		const auto inputData = MakeInputData(batch);
		auto inputs = MakeInputs(inputData, batch);

		for (int i = 0; i < kWarmupIterations; ++i)
		{
			auto outputs = module.RunTensors(std::span<const Tensor<CPU>>(inputs));
			benchmark::DoNotOptimize(outputs);
		}

		for (auto _ : state)
		{
			auto outputs = module.RunTensors(std::span<const Tensor<CPU>>(inputs));
			benchmark::DoNotOptimize(outputs);
			benchmark::ClobberMemory();
		}

		SetThroughputCounters(state, batch);
	}

	void BMAOTRunIntoConfigured(benchmark::State& state, ModelKind kind, std::size_t batch, const char* threadCount,
	                            void (*optimizer)(Graph&) = Optimize)
	{
		auto options = LiteNNBenchCompilerOptionsFromEnvironment();
		if (threadCount != nullptr)
		{
			options.cpuAOTThreadCount = static_cast<std::size_t>(std::stoull(threadCount));
		}

		std::mt19937 rng(42);
		auto graph = GetModelSpec(kind).build(batch, rng);
		optimizer(graph);

		auto compiled = Compiler<CPU>::Compile(Detail::BuildExecutablePlanFromGraph(graph), options);
		auto module = CompiledModule<CPU>::Load(compiled.Image());
		const auto inputData = MakeInputData(batch);
		auto inputs = MakeInputs(inputData, batch);
		auto outputs = AllocateOutputs(module);

		for (int i = 0; i < kWarmupIterations; ++i)
		{
			module.RunTensorsInto(std::span<const Tensor<CPU>>(inputs), std::span<Tensor<CPU>>(outputs));
		}

		for (auto _ : state)
		{
			module.RunTensorsInto(std::span<const Tensor<CPU>>(inputs), std::span<Tensor<CPU>>(outputs));
			benchmark::DoNotOptimize(outputs.data());
			benchmark::ClobberMemory();
		}

		SetThroughputCounters(state, batch);
	}

	void BMAOTRunTensorsInto(benchmark::State& state, ModelKind kind, std::size_t batch)
	{
		BMAOTRunIntoConfigured(state, kind, batch, nullptr);
	}

	void BMAOTLoRARunTensorsInto(benchmark::State& state, std::size_t batch, bool merged)
	{
		std::mt19937 rng(123);
		auto graph = BuildLoRALinear(batch, rng, merged);
		Optimize(graph);
		auto options = LiteNNBenchCompilerOptionsFromEnvironment();
		auto compiled = Compiler<CPU>::Compile(Detail::BuildExecutablePlanFromGraph(graph), options);
		auto module = CompiledModule<CPU>::Load(compiled.Image());
		const auto inputData = MakeInputData(batch);
		const auto inputs = MakeInputs(inputData, batch);
		auto outputs = AllocateOutputs(module);
		for (int i = 0; i < kWarmupIterations; ++i)
		{
			module.RunTensorsInto(std::span<const Tensor<CPU>>(inputs), std::span<Tensor<CPU>>(outputs));
			benchmark::DoNotOptimize(outputs);
		}
		for (auto _ : state)
		{
			module.RunTensorsInto(std::span<const Tensor<CPU>>(inputs), std::span<Tensor<CPU>>(outputs));
			benchmark::DoNotOptimize(outputs);
		}
		SetThroughputCounters(state, batch);
	}

	void BMAOTRunIntoT1(benchmark::State& state, ModelKind kind, std::size_t batch)
	{
		BMAOTRunIntoConfigured(state, kind, batch, "1");
	}

	void BMAOTRunIntoT16(benchmark::State& state, ModelKind kind, std::size_t batch)
	{
		BMAOTRunIntoConfigured(state, kind, batch, "16");
	}

	void BMEGraphAOTRunTensorsInto(benchmark::State& state, ModelKind kind, std::size_t batch)
	{
		BMAOTRunIntoConfigured(state, kind, batch, nullptr, OptimizeWithEGraph);
	}

	void BMAOTRedundantRunIntoConfigured(benchmark::State& state, std::size_t batch, bool enableEGraph)
	{
		auto options = LiteNNBenchCompilerOptionsFromEnvironment();
		auto graph = BuildRedundantAOTGraph(batch);
		if (enableEGraph)
		{
			EGraphPass{}.Run(graph);
		}

		auto compiled = Compiler<CPU>::Compile(Detail::BuildExecutablePlanFromGraph(graph), options);
		auto module = CompiledModule<CPU>::Load(compiled.Image());
		const auto inputData = MakeInputData(batch);
		auto inputs = MakeInputs(inputData, batch);
		auto outputs = AllocateOutputs(module);

		for (int i = 0; i < kWarmupIterations; ++i)
		{
			module.RunTensorsInto(std::span<const Tensor<CPU>>(inputs), std::span<Tensor<CPU>>(outputs));
		}

		for (auto _ : state)
		{
			module.RunTensorsInto(std::span<const Tensor<CPU>>(inputs), std::span<Tensor<CPU>>(outputs));
			benchmark::DoNotOptimize(outputs.data());
			benchmark::ClobberMemory();
		}

		SetThroughputCounters(state, batch);
	}

	void BMAOTRedundantRawRunTensorsInto(benchmark::State& state, std::size_t batch)
	{
		BMAOTRedundantRunIntoConfigured(state, batch, false);
	}

	void BMEGraphAOTRedundantRunTensorsInto(benchmark::State& state, std::size_t batch)
	{
		BMAOTRedundantRunIntoConfigured(state, batch, true);
	}
#endif

	void RegisterBenchmarks()
	{
#ifdef LITENN_ENABLE_CUDA
		const bool cudaDeviceAvailable = IsCUDADeviceAvailable();
#endif
#ifdef LITENN_ENABLE_VULKAN
		const bool vulkanDeviceAvailable = IsVulkanDeviceAvailable();
#endif
		constexpr std::array quantizedWeightKinds{
			QuantizedWeightKind::AffineInt8,
			QuantizedWeightKind::PackedInt4,
			QuantizedWeightKind::PackedFP4E2M1,
		};
		for (const auto kind : kModelKinds)
		{
			for (const auto batch : kBatchSizes)
			{
				RegisterBenchmarkCase("Interpreter", kind, batch,
				                      [=](benchmark::State& state) { BMInterpreter(state, kind, batch); });
				for (const auto weightKind : quantizedWeightKinds)
				{
					RegisterBenchmarkCase(
					    std::format("NativeQuantizedLinear/{}", QuantizedWeightKindName(weightKind)), kind, batch,
					    [=](benchmark::State& state) { BMNativeQuantizedLinearRun(state, kind, batch, weightKind); });
					RegisterBenchmarkCase(
					    std::format("DequantizedQuantizedLinearReference/{}", QuantizedWeightKindName(weightKind)),
					    kind, batch, [=](benchmark::State& state) {
						    BMDequantizedQuantizedLinearReferenceRun(state, kind, batch, weightKind);
					    });
				}
				for (const auto threadCount : kGGMLThreadCounts)
				{
					RegisterBenchmarkCase(
					    std::format("LlamaCppGGMLT{}", threadCount), kind, batch,
					    [=](benchmark::State& state) { BMLlamaCppGGML(state, kind, batch, threadCount); });
				}
#ifdef LITENN_BENCH_HAS_AOT
				RegisterBenchmarkCase("AOTRun", kind, batch,
				                      [=](benchmark::State& state) { BMAOTRun(state, kind, batch); });
				RegisterBenchmarkCase("AOTRunInto", kind, batch,
				                      [=](benchmark::State& state) { BMAOTRunTensorsInto(state, kind, batch); });
				RegisterBenchmarkCase("EGraphAOTRunInto", kind, batch,
				                      [=](benchmark::State& state) { BMEGraphAOTRunTensorsInto(state, kind, batch); });
				RegisterBenchmarkCase("AOTRunIntoT1", kind, batch,
				                      [=](benchmark::State& state) { BMAOTRunIntoT1(state, kind, batch); });
				RegisterBenchmarkCase("AOTRunIntoT16", kind, batch,
				                      [=](benchmark::State& state) { BMAOTRunIntoT16(state, kind, batch); });
#ifdef LITENN_ENABLE_CUDA
				if (cudaDeviceAvailable)
				{
					RegisterBenchmarkCase("CUDACPUFallbackRunInto", kind, batch, [=](benchmark::State& state) {
						BMCUDACPUFallbackRunTensorsInto(state, kind, batch);
					});
					RegisterBenchmarkCase("CUDANativeRunInto", kind, batch, [=](benchmark::State& state) {
						BMCUDANativeModelRunTensorsInto(state, kind, batch);
					});
					RegisterBenchmarkCase("CUDANativeGraphRunInto", kind, batch, [=](benchmark::State& state) {
						BMCUDANativeGraphModelRunTensorsInto(state, kind, batch);
					});
				}
#endif
#ifdef LITENN_ENABLE_VULKAN
				if (vulkanDeviceAvailable && kind == ModelKind::Linear)
				{
					RegisterBenchmarkCase("VulkanNativeRunInto", kind, batch, [=](benchmark::State& state) {
						BMVulkanNativeModelRunTensorsInto(state, kind, batch);
					});
				}
				if (vulkanDeviceAvailable && (kind == ModelKind::MLP128 || kind == ModelKind::MLP512))
				{
					RegisterBenchmarkCase("VulkanNativeGraphRunInto", kind, batch, [=](benchmark::State& state) {
						BMVulkanNativeGraphMLPRunTensorsInto(state, kind, batch);
					});
					RegisterBenchmarkCase("VulkanNativeGraphDeviceLocalRunInto", kind, batch,
					                      [=](benchmark::State& state) {
						                      BMVulkanNativeGraphMLPRunTensorsInto(state, kind, batch,
						                                                           VulkanBufferResidency::DeviceLocal);
					                      });
					if (kind == ModelKind::MLP128)
					{
						RegisterBenchmarkCase("VulkanNativeManualPipeline", kind, batch, [=](benchmark::State& state) {
							BMVulkanNativeManualMLP128RunTensorsInto(state, batch);
						});
					}
				}
#endif
#endif
			}
		}

		for (const auto batch : kBatchSizes)
		{
			auto* unmergedInterpreter = benchmark::RegisterBenchmark(
			    std::format("InterpreterLoRAUnmerged/LinearLoRA(784->512,r8)/batch:{}", batch),
			    [=](benchmark::State& state) { BMInterpreterLoRA(state, batch, false); });
			unmergedInterpreter->Unit(benchmark::kMicrosecond);

			auto* mergedInterpreter = benchmark::RegisterBenchmark(
			    std::format("InterpreterLoRAMerged/LinearLoRA(784->512,r8)/batch:{}", batch),
			    [=](benchmark::State& state) { BMInterpreterLoRA(state, batch, true); });
			mergedInterpreter->Unit(benchmark::kMicrosecond);
		}

#ifdef LITENN_BENCH_HAS_AOT
		for (const auto batch : kBatchSizes)
		{
			auto* unmergedLoRA = benchmark::RegisterBenchmark(
			    std::format("AOTLoRAUnmergedRunInto/LinearLoRA(784->512,r8)/batch:{}", batch),
			    [=](benchmark::State& state) { BMAOTLoRARunTensorsInto(state, batch, false); });
			unmergedLoRA->Unit(benchmark::kMicrosecond);

			auto* mergedLoRA = benchmark::RegisterBenchmark(
			    std::format("AOTLoRAMergedRunInto/LinearLoRA(784->512,r8)/batch:{}", batch),
			    [=](benchmark::State& state) { BMAOTLoRARunTensorsInto(state, batch, true); });
			mergedLoRA->Unit(benchmark::kMicrosecond);

			auto* rawCase = benchmark::RegisterBenchmark(
			    std::format("AOTRedundantRawRunInto/RedundantIdentity/batch:{}", batch),
			    [=](benchmark::State& state) { BMAOTRedundantRawRunTensorsInto(state, batch); });
			rawCase->UseRealTime()->Unit(benchmark::kMillisecond);

			auto* egraphCase = benchmark::RegisterBenchmark(
			    std::format("EGraphAOTRedundantRunInto/RedundantIdentity/batch:{}", batch),
			    [=](benchmark::State& state) { BMEGraphAOTRedundantRunTensorsInto(state, batch); });
			egraphCase->UseRealTime()->Unit(benchmark::kMillisecond);
		}

#ifdef LITENN_ENABLE_CUDA
		constexpr std::size_t nativeMatMulWidth = 128;
		constexpr std::array nativeMatMulDTypes{
			DataType::Float32,    DataType::Float16, DataType::BFloat16, DataType::Float8E4M3,
			DataType::Float8E5M2, DataType::Int8,    DataType::UInt8,
		};
		for (const auto batch : kBatchSizes)
		{
			for (const auto dtype : nativeMatMulDTypes)
			{
				if (!cudaDeviceAvailable || !SupportsCUDANativeMatMulBenchmarkDType(dtype))
				{
					continue;
				}
				auto* benchmarkCase = benchmark::RegisterBenchmark(
				    std::format("CUDANativeMatMul/{}/batch:{}/width:{}", DataTypeName(dtype), batch, nativeMatMulWidth),
				    [=](benchmark::State& state) {
					    BMCUDANativeMatMulRunTensorsInto(state, batch, nativeMatMulWidth, dtype);
				    });
				benchmarkCase->UseRealTime()->Unit(benchmark::kMillisecond);
			}
		}
#endif

#ifdef LITENN_ENABLE_VULKAN
		if (vulkanDeviceAvailable)
		{
			constexpr std::size_t vulkanNativeMatMulWidth = 128;
			constexpr std::array vulkanNativeCastDTypes{
				DataType::Float16,
				DataType::Int8,
				DataType::UInt8,
			};
			for (const auto batch : kBatchSizes)
			{
				const auto elementCount = batch * kInputWidth;
				auto* benchmarkCase = benchmark::RegisterBenchmark(
				    std::format("VulkanNativeElementwiseAddRunInto/F32/elements:{}", elementCount),
				    [=](benchmark::State& state) { BMVulkanNativeElementwiseAddRunTensorsInto(state, elementCount); });
				benchmarkCase->UseRealTime()->Unit(benchmark::kMillisecond);

				auto* unaryAbsBenchmarkCase = benchmark::RegisterBenchmark(
				    std::format("VulkanNativeUnaryAbsRunInto/F32/elements:{}", elementCount),
				    [=](benchmark::State& state) { BMVulkanNativeUnaryAbsRunTensorsInto(state, elementCount); });
				unaryAbsBenchmarkCase->UseRealTime()->Unit(benchmark::kMillisecond);

				if (SupportsVulkanNativeElementwiseBenchmarkDType(DataType::Float16))
				{
					auto* fp16BenchmarkCase = benchmark::RegisterBenchmark(
					    std::format("VulkanNativeElementwiseAddRunInto/F16/elements:{}", elementCount),
					    [=](benchmark::State& state) {
						    BMVulkanNativeElementwiseAddRunTensorsInto(state, elementCount, DataType::Float16);
					    });
					fp16BenchmarkCase->UseRealTime()->Unit(benchmark::kMillisecond);

					auto* fp16UnaryAbsBenchmarkCase = benchmark::RegisterBenchmark(
					    std::format("VulkanNativeUnaryAbsRunInto/F16/elements:{}", elementCount),
					    [=](benchmark::State& state) {
						    BMVulkanNativeUnaryAbsRunTensorsInto(state, elementCount, DataType::Float16);
					    });
					fp16UnaryAbsBenchmarkCase->UseRealTime()->Unit(benchmark::kMillisecond);
				}
				if (SupportsVulkanNativeElementwiseBenchmarkDType(DataType::Int8))
				{
					auto* int8BenchmarkCase = benchmark::RegisterBenchmark(
					    std::format("VulkanNativeElementwiseAddRunInto/Int8/elements:{}", elementCount),
					    [=](benchmark::State& state) {
						    BMVulkanNativeElementwiseAddRunTensorsInto(state, elementCount, DataType::Int8);
					    });
					int8BenchmarkCase->UseRealTime()->Unit(benchmark::kMillisecond);

					auto* uint8BenchmarkCase = benchmark::RegisterBenchmark(
					    std::format("VulkanNativeElementwiseAddRunInto/UInt8/elements:{}", elementCount),
					    [=](benchmark::State& state) {
						    BMVulkanNativeElementwiseAddRunTensorsInto(state, elementCount, DataType::UInt8);
					    });
					uint8BenchmarkCase->UseRealTime()->Unit(benchmark::kMillisecond);

					auto* int8UnaryAbsBenchmarkCase = benchmark::RegisterBenchmark(
					    std::format("VulkanNativeUnaryAbsRunInto/Int8/elements:{}", elementCount),
					    [=](benchmark::State& state) {
						    BMVulkanNativeUnaryAbsRunTensorsInto(state, elementCount, DataType::Int8);
					    });
					int8UnaryAbsBenchmarkCase->UseRealTime()->Unit(benchmark::kMillisecond);
				}

				auto* binaryChainBenchmarkCase = benchmark::RegisterBenchmark(
				    std::format("VulkanNativeBinaryChainRunInto/F32/elements:{}", elementCount),
				    [=](benchmark::State& state) { BMVulkanNativeBinaryChainRunTensorsInto(state, elementCount); });
				binaryChainBenchmarkCase->UseRealTime()->Unit(benchmark::kMillisecond);

				auto* binaryDAGBenchmarkCase = benchmark::RegisterBenchmark(
				    std::format("VulkanNativeBinaryDAGRunInto/F32/elements:{}", elementCount),
				    [=](benchmark::State& state) { BMVulkanNativeBinaryDAGRunTensorsInto(state, elementCount); });
				binaryDAGBenchmarkCase->UseRealTime()->Unit(benchmark::kMillisecond);

				auto* mixedElementwiseDAGBenchmarkCase = benchmark::RegisterBenchmark(
				    std::format("VulkanNativeMixedElementwiseDAGRunInto/F32/elements:{}", elementCount),
				    [=](benchmark::State& state) {
					    BMVulkanNativeMixedElementwiseDAGRunTensorsInto(state, elementCount);
				    });
				mixedElementwiseDAGBenchmarkCase->UseRealTime()->Unit(benchmark::kMillisecond);

				auto* branchedBinaryDAGBenchmarkCase = benchmark::RegisterBenchmark(
				    std::format("VulkanNativeBranchedBinaryDAGRunInto/F32/elements:{}", elementCount),
				    [=](benchmark::State& state) {
					    BMVulkanNativeBranchedBinaryDAGRunTensorsInto(state, elementCount);
				    });
				branchedBinaryDAGBenchmarkCase->UseRealTime()->Unit(benchmark::kMillisecond);

				for (const auto dstType : vulkanNativeCastDTypes)
				{
					if (!SupportsVulkanNativeCastBenchmarkDType(dstType))
					{
						continue;
					}
					auto* castBenchmarkCase = benchmark::RegisterBenchmark(
					    std::format("VulkanNativeCastRunInto/F32To{}/elements:{}", DataTypeName(dstType), elementCount),
					    [=](benchmark::State& state) {
						    BMVulkanNativeCastRunTensorsInto(state, dstType, elementCount);
					    });
					castBenchmarkCase->UseRealTime()->Unit(benchmark::kMillisecond);
				}

				constexpr std::array vulkanNativeReduceOps{ ReduceOp::Sum, ReduceOp::Mean, ReduceOp::Max,
					                                        ReduceOp::Min };
				for (const auto op : vulkanNativeReduceOps)
				{
					auto* reduceBenchmarkCase = benchmark::RegisterBenchmark(
					    std::format("VulkanNativeReduce/F32/{}/batch:{}/width:{}", ReduceOpBenchmarkName(op), batch,
					                vulkanNativeMatMulWidth),
					    [=](benchmark::State& state) {
						    BMVulkanNativeReduceRunTensorsInto(state, op, batch, vulkanNativeMatMulWidth);
					    });
					reduceBenchmarkCase->UseRealTime()->Unit(benchmark::kMillisecond);
				}

				auto* softmaxBenchmarkCase = benchmark::RegisterBenchmark(
				    std::format("VulkanNativeSoftmax/F32/Axis1/batch:{}/width:{}", batch, vulkanNativeMatMulWidth),
				    [=](benchmark::State& state) {
					    BMVulkanNativeSoftmaxRunTensorsInto(state, batch, vulkanNativeMatMulWidth);
				    });
				softmaxBenchmarkCase->UseRealTime()->Unit(benchmark::kMillisecond);

				constexpr std::array vulkanNativeNormalizationModes{ NormalizationMode::LayerNorm,
					                                                 NormalizationMode::RMSNorm };
				for (const auto mode : vulkanNativeNormalizationModes)
				{
					auto* normalizationBenchmarkCase = benchmark::RegisterBenchmark(
					    std::format("VulkanNativeNormalization/F32/{}/batch:{}/width:{}",
					                NormalizationModeBenchmarkName(mode), batch, vulkanNativeMatMulWidth),
					    [=](benchmark::State& state) {
						    BMVulkanNativeNormalizationRunTensorsInto(state, mode, batch, vulkanNativeMatMulWidth);
					    });
					normalizationBenchmarkCase->UseRealTime()->Unit(benchmark::kMillisecond);

					auto* affineNormalizationBenchmarkCase = benchmark::RegisterBenchmark(
					    std::format("VulkanNativeNormalizationAffine/F32/{}/batch:{}/width:{}",
					                NormalizationModeBenchmarkName(mode), batch, vulkanNativeMatMulWidth),
					    [=](benchmark::State& state) {
						    BMVulkanNativeAffineNormalizationRunTensorsInto(state, mode, batch,
						                                                    vulkanNativeMatMulWidth);
					    });
					affineNormalizationBenchmarkCase->UseRealTime()->Unit(benchmark::kMillisecond);
				}

				constexpr std::size_t vulkanNativeGroupNormGroups = 8;
				auto* groupNormBenchmarkCase = benchmark::RegisterBenchmark(
				    std::format("VulkanNativeGroupNorm/F32/groups:{}/elements:{}", vulkanNativeGroupNormGroups,
				                elementCount),
				    [=](benchmark::State& state) {
					    BMVulkanNativeGroupNormRunTensorsInto(state, elementCount, vulkanNativeGroupNormGroups);
				    });
				groupNormBenchmarkCase->UseRealTime()->Unit(benchmark::kMillisecond);

				auto* affineGroupNormBenchmarkCase = benchmark::RegisterBenchmark(
				    std::format("VulkanNativeGroupNormAffine/F32/groups:{}/elements:{}", vulkanNativeGroupNormGroups,
				                elementCount),
				    [=](benchmark::State& state) {
					    BMVulkanNativeAffineGroupNormRunTensorsInto(state, elementCount, vulkanNativeGroupNormGroups);
				    });
				affineGroupNormBenchmarkCase->UseRealTime()->Unit(benchmark::kMillisecond);

				constexpr std::size_t vulkanNativePoolChannels = 8;
				constexpr std::size_t vulkanNativePoolSpatial = 16;
				for (const auto poolMode : { PoolMode::Max, PoolMode::Average })
				{
					auto* poolBenchmarkCase = benchmark::RegisterBenchmark(
					    std::format("VulkanNativePool2D/F32/{}/batch:{}/channels:{}/spatial:{}",
					                PoolModeBenchmarkName(poolMode), batch, vulkanNativePoolChannels,
					                vulkanNativePoolSpatial),
					    [=](benchmark::State& state) {
						    BMVulkanNativePool2DRunTensorsInto(state, poolMode, batch, vulkanNativePoolChannels,
						                                       vulkanNativePoolSpatial, vulkanNativePoolSpatial);
					    });
					poolBenchmarkCase->UseRealTime()->Unit(benchmark::kMillisecond);

					auto* paddedPoolBenchmarkCase = benchmark::RegisterBenchmark(
					    std::format("VulkanNativePool2D/F32/{}Padded/batch:{}/channels:{}/spatial:{}",
					                PoolModeBenchmarkName(poolMode), batch, vulkanNativePoolChannels,
					                vulkanNativePoolSpatial),
					    [=](benchmark::State& state) {
						    BMVulkanNativePool2DRunTensorsInto(state, poolMode, batch, vulkanNativePoolChannels,
						                                       vulkanNativePoolSpatial, vulkanNativePoolSpatial,
						                                       { 1, 1 }, { 1, 1 }, false);
					    });
					paddedPoolBenchmarkCase->UseRealTime()->Unit(benchmark::kMillisecond);
				}
				auto* paddedAverageIncludePadBenchmarkCase = benchmark::RegisterBenchmark(
				    std::format("VulkanNativePool2D/F32/AveragePaddedIncludePad/batch:{}/channels:{}/spatial:{}", batch,
				                vulkanNativePoolChannels, vulkanNativePoolSpatial),
				    [=](benchmark::State& state) {
					    BMVulkanNativePool2DRunTensorsInto(state, PoolMode::Average, batch, vulkanNativePoolChannels,
					                                       vulkanNativePoolSpatial, vulkanNativePoolSpatial, { 1, 1 },
					                                       { 1, 1 }, true);
				    });
				paddedAverageIncludePadBenchmarkCase->UseRealTime()->Unit(benchmark::kMillisecond);

				auto* conv2DBenchmarkCase = benchmark::RegisterBenchmark(
				    std::format("VulkanNativeConv2D/F32/batch:{}/channels:{}/outChannels:{}/spatial:{}", batch,
				                vulkanNativePoolChannels, vulkanNativePoolChannels, vulkanNativePoolSpatial),
				    [=](benchmark::State& state) {
					    BMVulkanNativeConv2DRunTensorsInto(state, batch, vulkanNativePoolChannels,
					                                       vulkanNativePoolChannels, vulkanNativePoolSpatial,
					                                       vulkanNativePoolSpatial);
				    });
				conv2DBenchmarkCase->UseRealTime()->Unit(benchmark::kMillisecond);

				auto* nearestUpsampleBenchmarkCase = benchmark::RegisterBenchmark(
				    std::format("VulkanNativeUpsampleNearest/F32/batch:{}/channels:{}/spatial:{}x{}", batch,
				                vulkanNativePoolChannels, vulkanNativePoolSpatial, vulkanNativePoolSpatial * 2),
				    [=](benchmark::State& state) {
					    BMVulkanNativeNearestUpsampleRunTensorsInto(state, batch, vulkanNativePoolChannels,
					                                                vulkanNativePoolSpatial, vulkanNativePoolSpatial,
					                                                2);
				    });
				nearestUpsampleBenchmarkCase->UseRealTime()->Unit(benchmark::kMillisecond);

				auto* sliceBenchmarkCase = benchmark::RegisterBenchmark(
				    std::format("VulkanNativeSlice/F32/batch:{}/channels:{}to{}/spatial:{}", batch,
				                vulkanNativePoolChannels, vulkanNativePoolChannels / 2, vulkanNativePoolSpatial),
				    [=](benchmark::State& state) {
					    BMVulkanNativeSliceRunTensorsInto(state, batch, vulkanNativePoolChannels,
					                                      vulkanNativePoolSpatial, vulkanNativePoolSpatial);
				    });
				sliceBenchmarkCase->UseRealTime()->Unit(benchmark::kMillisecond);

				auto* concatBenchmarkCase = benchmark::RegisterBenchmark(
				    std::format("VulkanNativeConcat/F32/batch:{}/channels:{}plus{}/spatial:{}", batch,
				                vulkanNativePoolChannels / 2, vulkanNativePoolChannels / 2, vulkanNativePoolSpatial),
				    [=](benchmark::State& state) {
					    BMVulkanNativeConcatRunTensorsInto(state, batch, vulkanNativePoolChannels / 2,
					                                       vulkanNativePoolChannels / 2, vulkanNativePoolSpatial,
					                                       vulkanNativePoolSpatial);
				    });
				concatBenchmarkCase->UseRealTime()->Unit(benchmark::kMillisecond);

				auto* convTranspose2DBenchmarkCase = benchmark::RegisterBenchmark(
				    std::format("VulkanNativeConvTranspose2D/F32/batch:{}/channels:{}/outChannels:{}/spatial:{}x{}",
				                batch, vulkanNativePoolChannels, vulkanNativePoolChannels, vulkanNativePoolSpatial,
				                (vulkanNativePoolSpatial - 1) * 2 + 1),
				    [=](benchmark::State& state) {
					    BMVulkanNativeConvTranspose2DRunTensorsInto(state, batch, vulkanNativePoolChannels,
					                                                vulkanNativePoolChannels, vulkanNativePoolSpatial,
					                                                vulkanNativePoolSpatial);
				    });
				convTranspose2DBenchmarkCase->UseRealTime()->Unit(benchmark::kMillisecond);

				auto* matMulBenchmarkCase = benchmark::RegisterBenchmark(
				    std::format("VulkanNativeMatMul/F32/batch:{}/width:{}", batch, vulkanNativeMatMulWidth),
				    [=](benchmark::State& state) {
					    BMVulkanNativeMatMulRunTensorsInto(state, batch, vulkanNativeMatMulWidth);
				    });
				matMulBenchmarkCase->UseRealTime()->Unit(benchmark::kMillisecond);

				auto* matMulBiasBenchmarkCase = benchmark::RegisterBenchmark(
				    std::format("VulkanNativeMatMulBiasAdd/F32/batch:{}/width:{}", batch, vulkanNativeMatMulWidth),
				    [=](benchmark::State& state) {
					    BMVulkanNativeMatMulBiasAddRunTensorsInto(state, batch, vulkanNativeMatMulWidth);
				    });
				matMulBiasBenchmarkCase->UseRealTime()->Unit(benchmark::kMillisecond);

				auto* linearChainBenchmarkCase = benchmark::RegisterBenchmark(
				    std::format("VulkanNativeLinearChain/F32/layers:2/batch:{}/width:{}", batch,
				                vulkanNativeMatMulWidth),
				    [=](benchmark::State& state) {
					    BMVulkanNativeHomogeneousLinearChainRunTensorsInto(state, batch, vulkanNativeMatMulWidth);
				    });
				linearChainBenchmarkCase->UseRealTime()->Unit(benchmark::kMillisecond);
			}
		}
#endif
#endif

#ifdef LITENN_ENABLE_CUDA
		constexpr std::size_t cudaDeviceMatMulWidth = 128;
		constexpr std::array cudaDeviceMatMulDTypes{
			DataType::Float32,    DataType::Float16, DataType::BFloat16, DataType::Float8E4M3,
			DataType::Float8E5M2, DataType::Int8,    DataType::UInt8,
		};
		for (const auto batch : kBatchSizes)
		{
			for (const auto dtype : cudaDeviceMatMulDTypes)
			{
				if (!cudaDeviceAvailable || !SupportsCUDADeviceMatMulBenchmarkDType(dtype))
				{
					continue;
				}
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
