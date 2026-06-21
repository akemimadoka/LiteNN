#include <gtest/gtest.h>

#include <LiteNN.h>
#include <LiteNN/Compiler/CUDANativeCodegen.h>
#include <LiteNN/Compiler/CUDANativePayload.h>
#include <LiteNN/Compiler/CompiledModule.h>
#include <LiteNN/Pass/FusionPass.h>
#include <LiteNN/Runtime/Interpreter.h>

#ifdef LITENN_ENABLE_CUDA
#include <cuda_runtime_api.h>
#endif

#include <algorithm>
#include <array>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <initializer_list>
#include <span>
#include <stdexcept>
#include <string>
#include <string_view>
#include <utility>
#include <vector>

using namespace LiteNN;

namespace
{
	float ReadFloat(const Tensor<CPU>& t, std::size_t i)
	{
		return static_cast<const float*>(t.UnsafeRawData())[i];
	}

	std::vector<float> ReadAsFloat32(const Tensor<CPU>& tensor)
	{
		std::vector<float> values(tensor.NumElements());
		CPU cpu;
		DeviceTraits<CPU>::ConvertTo(cpu, tensor.DType(), tensor.UnsafeRawData(), tensor.NumElements(),
		                             DataType::Float32, values.data());
		return values;
	}

	std::uint32_t ReadU32LE(std::span<const std::byte> bytes, std::size_t offset)
	{
		std::uint32_t value = 0;
		for (int i = 0; i < 4; ++i)
		{
			value |= std::to_integer<std::uint32_t>(bytes[offset + i]) << (i * 8);
		}
		return value;
	}

	void ExpectTensorNear(const Tensor<CPU>& tensor, std::span<const float> expected, float tolerance = 1e-6f)
	{
		ASSERT_EQ(tensor.NumElements(), expected.size());
		const auto actual = ReadAsFloat32(tensor);
		for (std::size_t i = 0; i < expected.size(); ++i)
		{
			EXPECT_NEAR(actual[i], expected[i], tolerance);
		}
	}

	void ExpectTensorNear(const Tensor<CPU>& tensor, const Tensor<CPU>& expected, float tolerance = 1e-6f)
	{
		EXPECT_EQ(tensor.DType(), expected.DType());
		EXPECT_EQ(tensor.Shape().ToOwned(), expected.Shape().ToOwned());
		ASSERT_EQ(tensor.NumElements(), expected.NumElements());
		const auto actualValues = ReadAsFloat32(tensor);
		const auto expectedValues = ReadAsFloat32(expected);
		for (std::size_t i = 0; i < expected.NumElements(); ++i)
		{
			EXPECT_NEAR(actualValues[i], expectedValues[i], tolerance);
		}
	}

	void ExpectOutputsNear(std::span<const Tensor<CPU>> outputs, std::span<const Tensor<CPU>> expected,
	                       float tolerance = 1e-6f)
	{
		ASSERT_EQ(outputs.size(), expected.size());
		for (std::size_t i = 0; i < outputs.size(); ++i)
		{
			SCOPED_TRACE(i);
			ExpectTensorNear(outputs[i], expected[i], tolerance);
		}
	}

	std::string BytesToString(std::span<const std::byte> bytes)
	{
		return std::string(reinterpret_cast<const char*>(bytes.data()), bytes.size());
	}

	CUDA CUDAWithHostFallback()
	{
		return CUDA{ .deviceIndex = 0, .hostFallbackPolicy = CUDAHostFallbackPolicy::Allow };
	}

	Graph BuildBinaryGraph(BinaryOp op, std::span<const std::size_t> lhsShape, std::span<const std::size_t> rhsShape,
	                       std::span<const std::size_t> outputShape, std::string outputName)
	{
		Graph graph;
		Subgraph sg;
		const auto a = sg.AddParam(DataType::Float32, std::vector<std::size_t>(lhsShape.begin(), lhsShape.end()));
		const auto b = sg.AddParam(DataType::Float32, std::vector<std::size_t>(rhsShape.begin(), rhsShape.end()));
		const auto y = sg.AddNode(
		    BinaryOpNode{ op, { a, 0 }, { b, 0 } },
		    { OutputInfo{ DataType::Float32, std::vector<std::size_t>(outputShape.begin(), outputShape.end()) } });
		sg.SetResults(std::vector<NodeOutput>{ { y, 0 } });
		graph.AddSubgraph(std::move(sg));
		graph.SetForward(0);
		graph.SetInputNames({ "lhs", "rhs" });
		graph.SetOutputNames({ std::move(outputName) });
		return graph;
	}

	Graph BuildSimpleBinaryGraph(BinaryOp op, std::string outputName)
	{
		return BuildBinaryGraph(op, std::array{ 2uz, 2uz }, std::array{ 2uz, 2uz }, std::array{ 2uz, 2uz },
		                        std::move(outputName));
	}

	Graph BuildSimpleUnaryGraph(UnaryOp op, std::string outputName)
	{
		Graph graph;
		Subgraph sg;
		const auto input = sg.AddParam(DataType::Float32, { 2, 2 });
		const auto y = sg.AddNode(UnaryOpNode{ op, { input, 0 } }, { OutputInfo{ DataType::Float32, { 2, 2 } } });
		sg.SetResults({ { y, 0 } });
		graph.AddSubgraph(std::move(sg));
		graph.SetForward(0);
		graph.SetInputNames({ "input" });
		graph.SetOutputNames({ std::move(outputName) });
		return graph;
	}

	Graph BuildCastGraph(DataType srcType, DataType dstType, std::vector<std::size_t> shape, std::string outputName)
	{
		Graph graph;
		Subgraph sg;
		const auto input = sg.AddParam(srcType, shape);
		const auto y = sg.AddNode(CastNode{ { input, 0 }, dstType }, { OutputInfo{ dstType, shape } });
		sg.SetResults({ { y, 0 } });
		graph.AddSubgraph(std::move(sg));
		graph.SetForward(0);
		graph.SetInputNames({ "input" });
		graph.SetOutputNames({ std::move(outputName) });
		return graph;
	}

	Graph BuildSimpleMatMulGraph(DataType dtype = DataType::Float32, std::string outputName = "matmul")
	{
		Graph graph;
		Subgraph sg;
		const auto a = sg.AddParam(dtype, { 2, 2 });
		const auto b = sg.AddParam(dtype, { 2, 2 });
		const auto y =
		    sg.AddNode(BinaryOpNode{ BinaryOp::MatMul, { a, 0 }, { b, 0 } }, { OutputInfo{ dtype, { 2, 2 } } });
		sg.SetResults({ { y, 0 } });
		graph.AddSubgraph(std::move(sg));
		graph.SetForward(0);
		graph.SetInputNames({ "lhs", "rhs" });
		graph.SetOutputNames({ std::move(outputName) });
		return graph;
	}

	Graph BuildMatMulGraph(DataType dtype, std::vector<std::size_t> lhsShape, std::vector<std::size_t> rhsShape,
	                       std::vector<std::size_t> outputShape, std::string outputName)
	{
		Graph graph;
		Subgraph sg;
		const auto lhs = sg.AddParam(dtype, lhsShape);
		const auto rhs = sg.AddParam(dtype, rhsShape);
		const auto y =
		    sg.AddNode(BinaryOpNode{ BinaryOp::MatMul, { lhs, 0 }, { rhs, 0 } }, { OutputInfo{ dtype, outputShape } });
		sg.SetResults({ { y, 0 } });
		graph.AddSubgraph(std::move(sg));
		graph.SetForward(0);
		graph.SetInputNames({ "lhs", "rhs" });
		graph.SetOutputNames({ std::move(outputName) });
		return graph;
	}

	Graph BuildSimplePowGraph()
	{
		return BuildSimpleBinaryGraph(BinaryOp::Pow, "pow");
	}

	Graph BuildReduceGraph(ReduceOp op, std::size_t axis, std::vector<std::size_t> outputShape, std::string outputName)
	{
		Graph graph;
		Subgraph sg;
		const auto input = sg.AddParam(DataType::Float32, { 2, 3 });
		const auto y = sg.AddNode(ReduceOpNode{ op, { input, 0 }, axis },
		                          { OutputInfo{ DataType::Float32, std::move(outputShape) } });
		sg.SetResults({ { y, 0 } });
		graph.AddSubgraph(std::move(sg));
		graph.SetForward(0);
		graph.SetInputNames({ "input" });
		graph.SetOutputNames({ std::move(outputName) });
		return graph;
	}

	Graph BuildConcatGraph(std::vector<std::size_t> lhsShape, std::vector<std::size_t> rhsShape,
	                       std::vector<std::size_t> outputShape, std::size_t axis, std::string outputName)
	{
		Graph graph;
		Subgraph sg;
		const auto lhs = sg.AddParam(DataType::Float32, std::move(lhsShape));
		const auto rhs = sg.AddParam(DataType::Float32, std::move(rhsShape));
		const auto y = sg.AddNode(ConcatNode{ { { lhs, 0 }, { rhs, 0 } }, axis },
		                          { OutputInfo{ DataType::Float32, std::move(outputShape) } });
		sg.SetResults({ { y, 0 } });
		graph.AddSubgraph(std::move(sg));
		graph.SetForward(0);
		graph.SetInputNames({ "lhs", "rhs" });
		graph.SetOutputNames({ std::move(outputName) });
		return graph;
	}

	Graph BuildSliceGraph(std::size_t axis, std::size_t start, std::size_t length, std::vector<std::size_t> outputShape,
	                      std::string outputName)
	{
		Graph graph;
		Subgraph sg;
		const auto input = sg.AddParam(DataType::Float32, { 2, 5 });
		const auto y = sg.AddNode(SliceNode{ { input, 0 }, axis, start, length },
		                          { OutputInfo{ DataType::Float32, std::move(outputShape) } });
		sg.SetResults({ { y, 0 } });
		graph.AddSubgraph(std::move(sg));
		graph.SetForward(0);
		graph.SetInputNames({ "input" });
		graph.SetOutputNames({ std::move(outputName) });
		return graph;
	}

	Graph BuildMatMulBiasGraph(bool relu, DataType dtype = DataType::Float32, std::size_t m = 2, std::size_t k = 3,
	                           std::size_t n = 2)
	{
		Graph graph;
		Subgraph sg;
		const auto lhs = sg.AddParam(dtype, { m, k });
		const auto rhs = sg.AddParam(dtype, { k, n });
		const auto bias = sg.AddParam(dtype, { 1, n });
		const auto matmul =
		    sg.AddNode(BinaryOpNode{ BinaryOp::MatMul, { lhs, 0 }, { rhs, 0 } }, { OutputInfo{ dtype, { m, n } } });
		const auto add =
		    sg.AddNode(BinaryOpNode{ BinaryOp::Add, { matmul, 0 }, { bias, 0 } }, { OutputInfo{ dtype, { m, n } } });
		NodeId result = add;
		if (relu)
		{
			const auto zeroTensor = Tensor<CPU>({ 0.0 }, { 1 }, dtype);
			const auto zero = sg.AddNode(ConstantNode{ zeroTensor.CopyToDevice(PolymorphicDevice{ CPU{} }) },
			                             { OutputInfo{ dtype, { 1 } } });
			result =
			    sg.AddNode(BinaryOpNode{ BinaryOp::Max, { add, 0 }, { zero, 0 } }, { OutputInfo{ dtype, { m, n } } });
		}
		sg.SetResults({ { result, 0 } });
		graph.AddSubgraph(std::move(sg));
		graph.SetForward(0);
		graph.SetInputNames({ "lhs", "rhs", "bias" });
		graph.SetOutputNames({ relu ? "matmul_bias_relu" : "matmul_bias" });
		return graph;
	}

	std::array<std::byte, 2> EncodeF16(float value)
	{
		const Tensor<CPU> half({ static_cast<double>(value) }, { 1 }, DataType::Float16);
		std::array<std::byte, 2> bytes{};
		std::memcpy(bytes.data(), half.UnsafeRawData(), bytes.size());
		return bytes;
	}

	Graph BuildGGMLQ8_0QuantizedMatMulGraph()
	{
		constexpr std::size_t batch = 2;
		constexpr std::size_t inFeatures = 32;
		constexpr std::size_t outFeatures = 3;
		constexpr float scale = 0.25F;
		std::vector<std::byte> storage(outFeatures * 34);
		const auto scaleBytes = EncodeF16(scale);
		for (std::size_t out = 0; out < outFeatures; ++out)
		{
			const auto blockBase = out * 34;
			storage[blockBase] = scaleBytes[0];
			storage[blockBase + 1] = scaleBytes[1];
			for (std::size_t k = 0; k < inFeatures; ++k)
			{
				const auto signedValue = static_cast<std::int8_t>(static_cast<int>((out + k) % 17) - 8);
				storage[blockBase + 2 + k] = static_cast<std::byte>(signedValue);
			}
		}

		auto tensor = Tensor<CPU>(Uninitialized, { outFeatures, 34 }, DataType::UInt8);
		std::memcpy(tensor.UnsafeRawData(), storage.data(), storage.size());
		auto params =
		    BlockQuantization(QuantizedBlockFormat::GGML_Q8_0, { outFeatures, inFeatures }, DataType::Float32);

		Graph graph;
		const auto weight = graph.AddVariable(Variable::CreateFrozenQuantized(std::move(tensor), params));
		Subgraph sg;
		const auto input = sg.AddParam(DataType::Float32, { batch, inFeatures });
		const auto weightRef =
		    sg.AddNode(VariableRefNode{ weight }, { OutputInfo{ DataType::UInt8, { outFeatures, 34 } } });
		const auto output = sg.AddNode(QuantizedMatMulNode{ { input, 0 }, { weightRef, 0 }, params, true },
		                               { OutputInfo{ DataType::Float32, { batch, outFeatures } } });
		sg.SetResults({ { output, 0 } });
		graph.SetForward(graph.AddSubgraph(std::move(sg)));
		graph.SetInputNames({ "input" });
		graph.SetOutputNames({ "output" });
		return graph;
	}

	Graph BuildGGMLQ4_KQuantizedMatMulGraph()
	{
		constexpr std::size_t batch = 1;
		constexpr std::size_t inFeatures = 256;
		constexpr std::size_t outFeatures = 2;
		const auto d = EncodeF16(0.25F);
		const auto dmin = EncodeF16(0.125F);
		std::vector<std::byte> storage(outFeatures * 144);
		for (std::size_t out = 0; out < outFeatures; ++out)
		{
			const auto blockBase = out * 144;
			storage[blockBase] = d[0];
			storage[blockBase + 1] = d[1];
			storage[blockBase + 2] = dmin[0];
			storage[blockBase + 3] = dmin[1];
			for (std::size_t i = 0; i < 4; ++i)
			{
				storage[blockBase + 4 + i] = std::byte{ 1 };
				storage[blockBase + 12 + i] = std::byte{ 1 };
			}
			for (std::size_t lane = 0; lane < inFeatures; ++lane)
			{
				const auto nibble = static_cast<std::uint8_t>((out + lane) % 16);
				const auto quantOffset = (lane / 64) * 32 + lane % 32;
				auto& byte = storage[blockBase + 16 + quantOffset];
				const auto current = std::to_integer<std::uint8_t>(byte);
				byte = lane % 64 < 32 ? static_cast<std::byte>((current & 0xF0u) | nibble)
				                      : static_cast<std::byte>((current & 0x0Fu) | (nibble << 4));
			}
		}

		auto tensor = Tensor<CPU>(Uninitialized, { outFeatures, 144 }, DataType::UInt8);
		std::memcpy(tensor.UnsafeRawData(), storage.data(), storage.size());
		auto params =
		    BlockQuantization(QuantizedBlockFormat::GGML_Q4_K, { outFeatures, inFeatures }, DataType::Float32);

		Graph graph;
		const auto weight = graph.AddVariable(Variable::CreateFrozenQuantized(std::move(tensor), params));
		Subgraph sg;
		const auto input = sg.AddParam(DataType::Float32, { batch, inFeatures });
		const auto weightRef =
		    sg.AddNode(VariableRefNode{ weight }, { OutputInfo{ DataType::UInt8, { outFeatures, 144 } } });
		const auto output = sg.AddNode(QuantizedMatMulNode{ { input, 0 }, { weightRef, 0 }, params, true },
		                               { OutputInfo{ DataType::Float32, { batch, outFeatures } } });
		sg.SetResults({ { output, 0 } });
		graph.SetForward(graph.AddSubgraph(std::move(sg)));
		graph.SetInputNames({ "input" });
		graph.SetOutputNames({ "output" });
		return graph;
	}

	Graph BuildTinyMLPGraph(std::size_t batch, DataType dtype = DataType::Float32)
	{
		ModelBuilder builder;
		Graph& graph = builder.UnsafeMutableGraph();
		const auto h1 = Layer::CreateLinear(
		    builder,
		    Tensor<CPU>({ 0.5, -0.25, 0.75, 0.125, -0.5, 0.25, 1.0, -1.0, 0.375, 0.625, -0.75, 0.5 }, { 3, 4 }, dtype),
		    Tensor<CPU>({ 0.1, -0.2, 0.3, -0.4 }, { 1, 4 }, dtype));
		const auto h2 = Layer::CreateLinear(
		    builder, Tensor<CPU>({ 0.25, -0.5, 0.75, 0.5, 0.125, -0.25, -0.375, 0.625 }, { 4, 2 }, dtype),
		    Tensor<CPU>({ 0.05, -0.15 }, { 1, 2 }, dtype));

		Subgraph sg;
		const auto input = sg.AddParam(dtype, { batch, 3 });
		const auto hidden = Layer::AddReLU(sg, Layer::AddLinear(sg, h1, { input, 0 }));
		sg.SetResults({ Layer::AddLinear(sg, h2, hidden) });
		graph.SetForward(graph.AddSubgraph(std::move(sg)));
		graph.SetInputNames({ "input" });
		graph.SetOutputNames({ "logits" });
		return builder.UnsafeTakeGraph();
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

	struct TensorInputSpec
	{
		std::vector<double> values;
		std::vector<std::size_t> shape;
		DataType dtype{ DataType::Float32 };
	};

	std::vector<Tensor<CPU>> MakeCPUInputs(std::span<const TensorInputSpec> specs)
	{
		std::vector<Tensor<CPU>> inputs;
		inputs.reserve(specs.size());
		for (const auto& spec : specs)
		{
			inputs.emplace_back(std::span<const double>(spec.values.data(), spec.values.size()),
			                    ShapeView{ spec.shape }, spec.dtype);
		}
		return inputs;
	}

	std::vector<double> RepeatingValues(std::size_t count, std::initializer_list<double> pattern)
	{
		std::vector<double> values;
		values.reserve(count);
		for (auto i = 0uz; i < count; ++i)
		{
			values.push_back(*(pattern.begin() + static_cast<std::ptrdiff_t>(i % pattern.size())));
		}
		return values;
	}

	std::vector<Tensor<CUDA>> MakeCUDAInputs(std::span<const TensorInputSpec> specs)
	{
		std::vector<Tensor<CUDA>> inputs;
		inputs.reserve(specs.size());
		for (const auto& spec : specs)
		{
			auto cpuInput = Tensor<CPU>(std::span<const double>(spec.values.data(), spec.values.size()),
			                            ShapeView{ spec.shape }, spec.dtype);
			inputs.push_back(cpuInput.CopyToDevice(CUDA{}));
		}
		return inputs;
	}
} // namespace

TEST(CompiledModuleCUDATest, CompilerArtifactsExposeStableCUDANativeABI)
{
	{
		auto artifact = Compiler<CUDA>::CompileArtifact(Detail::BuildExecutablePlanFromGraph(BuildSimpleMatMulGraph()));
		EXPECT_EQ(artifact.Backend(), CompiledModuleBackend::CUDANative);
		EXPECT_EQ(CompiledModuleArtifact::CopyFromImage(artifact.Image()).Backend(), CompiledModuleBackend::CUDANative);
		ASSERT_EQ(artifact.InputSpecs().size(), 2u);
		ASSERT_EQ(artifact.OutputSpecs().size(), 1u);
		EXPECT_EQ(artifact.InputSpecs()[0].name, "lhs");
		EXPECT_EQ(artifact.InputSpecs()[1].name, "rhs");
		EXPECT_EQ(artifact.OutputSpecs()[0].name, "matmul");

		const auto payload = DeserializeCUDANativeInstructionPayload(artifact.Instructions());
		EXPECT_EQ(payload.binaryKind, CUDANativeBinaryKind::LibraryCall);
		EXPECT_EQ(payload.featureSet,
		          CUDANativeFeatureSet(CUDANativeFeature::StaticShape, CUDANativeFeature::SingleSubgraph,
		                               CUDANativeFeature::MatMulCUBLASF32));
		EXPECT_EQ(payload.target, "cublas");
		EXPECT_TRUE(payload.binary.empty());
		ASSERT_EQ(payload.scalarData.size(), 3u * sizeof(std::uint32_t));
		EXPECT_EQ(ReadU32LE(payload.scalarData, 0), 2u);
		EXPECT_EQ(ReadU32LE(payload.scalarData, 4), 2u);
		EXPECT_EQ(ReadU32LE(payload.scalarData, 8), 2u);
		ASSERT_EQ(payload.kernels.size(), 1u);
		EXPECT_EQ(payload.kernels[0].name, "litenn_cublas_matmul_f32");
		ASSERT_EQ(payload.kernels[0].arguments.size(), 3u);
		EXPECT_EQ(payload.kernels[0].arguments[0].kind, CUDANativeArgumentKind::OutputTensor);
		EXPECT_EQ(payload.kernels[0].arguments[0].byteSize, 16u);
		EXPECT_EQ(payload.kernels[0].arguments[1].kind, CUDANativeArgumentKind::InputTensor);
		EXPECT_EQ(payload.kernels[0].arguments[1].byteSize, 16u);
		EXPECT_EQ(payload.kernels[0].arguments[2].kind, CUDANativeArgumentKind::InputTensor);
		EXPECT_EQ(payload.kernels[0].arguments[2].byteSize, 16u);
	}

#ifdef LITENN_ENABLE_CUDA_DRIVER
	{
		auto artifact =
		    Compiler<CUDA>::CompileArtifact(Detail::BuildExecutablePlanFromGraph(BuildGGMLQ8_0QuantizedMatMulGraph()));
		ASSERT_EQ(artifact.Backend(), CompiledModuleBackend::CUDANative);
		ASSERT_EQ(artifact.InputSpecs().size(), 1u);
		ASSERT_EQ(artifact.OutputSpecs().size(), 1u);
		EXPECT_EQ(artifact.InputSpecs()[0].name, "input");
		EXPECT_EQ(artifact.OutputSpecs()[0].name, "output");

		const auto payload = DeserializeCUDANativeInstructionPayload(artifact.Instructions());
		EXPECT_EQ(payload.binaryKind, CUDANativeBinaryKind::PTX);
		EXPECT_TRUE(payload.featureSet.HasFeature(CUDANativeFeature::GGMLQ8_0MatMulF32));
		EXPECT_TRUE(payload.featureSet.HasFeature(CUDANativeFeature::ConstantTensor));
		EXPECT_EQ(payload.target, CUDANativeNVPTXTargetChip());
		ASSERT_FALSE(payload.binary.empty());
		ASSERT_EQ(payload.kernels.size(), 1u);
		const auto& kernel = payload.kernels[0];
		EXPECT_EQ(kernel.name, CUDANativeGGMLBlockMatMulF32KernelName(QuantizedBlockFormat::GGML_Q8_0));
		ASSERT_EQ(kernel.arguments.size(), 4u);
		EXPECT_EQ(kernel.arguments[0].kind, CUDANativeArgumentKind::OutputTensor);
		EXPECT_EQ(kernel.arguments[1].kind, CUDANativeArgumentKind::InputTensor);
		EXPECT_EQ(kernel.arguments[2].kind, CUDANativeArgumentKind::ConstantTensor);
		EXPECT_EQ(kernel.arguments[3].kind, CUDANativeArgumentKind::Scalar);
		EXPECT_EQ(payload.constantData.size(), 3u * 34u);
	}

	{
		auto artifact =
		    Compiler<CUDA>::CompileArtifact(Detail::BuildExecutablePlanFromGraph(BuildGGMLQ4_KQuantizedMatMulGraph()));
		ASSERT_EQ(artifact.Backend(), CompiledModuleBackend::CUDANative);

		const auto payload = DeserializeCUDANativeInstructionPayload(artifact.Instructions());
		EXPECT_EQ(payload.binaryKind, CUDANativeBinaryKind::PTX);
		EXPECT_TRUE(payload.featureSet.HasFeature(CUDANativeFeature::GGMLQ4_KMatMulF32));
		EXPECT_TRUE(payload.featureSet.HasFeature(CUDANativeFeature::ConstantTensor));
		ASSERT_EQ(payload.kernels.size(), 1u);
		const auto& kernel = payload.kernels[0];
		EXPECT_EQ(kernel.name, CUDANativeGGMLBlockMatMulF32KernelName(QuantizedBlockFormat::GGML_Q4_K));
		ASSERT_EQ(kernel.arguments.size(), 4u);
		EXPECT_EQ(kernel.arguments[2].kind, CUDANativeArgumentKind::ConstantTensor);
		EXPECT_EQ(payload.constantData.size(), 2u * 144u);
	}

	{
		const std::array outputShape{ 2uz, 3uz };
		auto artifact = Compiler<CUDA>::CompileArtifact(Detail::BuildExecutablePlanFromGraph(BuildBinaryGraph(
		    BinaryOp::Divide, std::array{ 2uz, 3uz }, std::array{ 1uz, 3uz }, outputShape, "broadcast_divide")));
		EXPECT_EQ(artifact.Backend(), CompiledModuleBackend::CUDANative);
		EXPECT_EQ(CompiledModuleArtifact::CopyFromImage(artifact.Image()).Backend(), CompiledModuleBackend::CUDANative);
		ASSERT_EQ(artifact.InputSpecs().size(), 2u);
		ASSERT_EQ(artifact.OutputSpecs().size(), 1u);
		EXPECT_EQ(artifact.InputSpecs()[0].name, "lhs");
		EXPECT_EQ(artifact.InputSpecs()[0].type.StaticShape(), (std::vector<std::size_t>{ 2, 3 }));
		EXPECT_EQ(artifact.InputSpecs()[1].name, "rhs");
		EXPECT_EQ(artifact.InputSpecs()[1].type.StaticShape(), (std::vector<std::size_t>{ 1, 3 }));
		EXPECT_EQ(artifact.OutputSpecs()[0].name, "broadcast_divide");
		EXPECT_EQ(artifact.OutputSpecs()[0].type.StaticShape(), (std::vector<std::size_t>{ 2, 3 }));

		const auto payload = DeserializeCUDANativeInstructionPayload(artifact.Instructions());
		EXPECT_EQ(payload.binaryKind, CUDANativeBinaryKind::PTX);
		EXPECT_EQ(payload.featureSet,
		          CUDANativeFeatureSet(CUDANativeFeature::StaticShape, CUDANativeFeature::SingleSubgraph,
		                               CUDANativeFeature::ElementwiseDivideF32,
		                               CUDANativeFeature::ElementwiseBroadcastF32));
		EXPECT_EQ(payload.target, CUDANativeNVPTXTargetChip());
		ASSERT_FALSE(payload.binary.empty());
		EXPECT_EQ(payload.binary.back(), std::byte{ 0 });
		const auto ptx = BytesToString(payload.binary);
		EXPECT_NE(ptx.find("Generated by LLVM NVPTX Back-End"), std::string::npos);
		EXPECT_NE(ptx.find(".visible .entry litenn_divide_broadcast_f32"), std::string::npos);
		ASSERT_EQ(payload.scalarData.size(), sizeof(std::uint32_t));
		EXPECT_EQ(ReadU32LE(payload.scalarData, 0), 6u);
		ASSERT_EQ(payload.kernels.size(), 1u);
		const auto& kernel = payload.kernels[0];
		EXPECT_EQ(kernel.name, "litenn_divide_broadcast_f32");
		EXPECT_EQ(kernel.grid.x, 1u);
		EXPECT_EQ(kernel.block.x, 6u);
		ASSERT_EQ(kernel.arguments.size(), 4u);
		EXPECT_EQ(kernel.arguments[0].kind, CUDANativeArgumentKind::OutputTensor);
		EXPECT_EQ(kernel.arguments[0].byteSize, 24u);
		EXPECT_EQ(kernel.arguments[1].kind, CUDANativeArgumentKind::InputTensor);
		EXPECT_EQ(kernel.arguments[1].index, 0u);
		EXPECT_EQ(kernel.arguments[1].byteSize, 24u);
		EXPECT_EQ(kernel.arguments[2].kind, CUDANativeArgumentKind::InputTensor);
		EXPECT_EQ(kernel.arguments[2].index, 1u);
		EXPECT_EQ(kernel.arguments[2].byteSize, 12u);
		EXPECT_EQ(kernel.arguments[3].kind, CUDANativeArgumentKind::Scalar);
		EXPECT_EQ(kernel.arguments[3].byteSize, sizeof(std::uint32_t));
	}
#endif
}

TEST(CompiledModuleCUDATest, RunsCPUAOTBridgeWithCUDATensors)
{
	if (!IsCUDADeviceAvailable())
	{
		GTEST_SKIP() << "CUDA device is not available";
	}

	auto graph = BuildSimplePowGraph();
	const auto bridgeDevice = CUDAWithHostFallback();
	auto compiled = Compiler<CUDA>::Compile(Detail::BuildExecutablePlanFromGraph(graph), bridgeDevice);

	EXPECT_EQ(compiled.Backend(), CompiledModuleBackend::CPUNative);
	ASSERT_GT(compiled.Rodata().size(), 0u);
	ASSERT_GT(compiled.Instructions().size(), 0u);
	EXPECT_EQ(compiled.FindInput("lhs"), 0u);
	EXPECT_EQ(compiled.FindOutput("pow"), 0u);

	auto lhs = Tensor<CPU>({ 2, 3, 4, 5 }, { 2, 2 }, DataType::Float32).CopyToDevice(bridgeDevice);
	auto rhs = Tensor<CPU>({ 1, 2, 3, 0 }, { 2, 2 }, DataType::Float32).CopyToDevice(bridgeDevice);
	std::array<Tensor<CUDA>, 2> inputs = { std::move(lhs), std::move(rhs) };

	auto outputs = compiled.RunTensors(inputs);
	ASSERT_EQ(outputs.size(), 1u);
	auto cpuOutput = outputs[0].CopyToDevice(CPU{});
	ExpectTensorNear(cpuOutput, std::array{ 2.0f, 9.0f, 64.0f, 1.0f });

	auto loaded = CompiledModule<CUDA>::Load(compiled.Image(), bridgeDevice);
	std::array<Tensor<CUDA>, 1> out = { Tensor<CUDA>(Uninitialized, { 2, 2 }, DataType::Float32, bridgeDevice) };
	loaded.RunTensorsInto(inputs, out);
	auto cpuOutInto = out[0].CopyToDevice(CPU{});
	ExpectTensorNear(cpuOutInto, std::array{ 2.0f, 9.0f, 64.0f, 1.0f });
}

#ifdef LITENN_ENABLE_CUDA_DRIVER
TEST(CompiledModuleCUDATest, RunsNativeGGMLQ8_0QuantizedMatMulOnCUDA)
{
	if (!IsCUDADeviceAvailable())
	{
		GTEST_SKIP() << "CUDA device is not available";
	}
	if (!IsCUDADriverAvailable())
	{
		GTEST_SKIP() << "CUDA driver is not available";
	}

	auto graph = BuildGGMLQ8_0QuantizedMatMulGraph();
	auto module = Compiler<CUDA>::CompileArtifact(Detail::BuildExecutablePlanFromGraph(graph)).Load(CUDA{});
	ASSERT_EQ(module.Backend(), CompiledModuleBackend::CUDANative);

	const std::array<double, 64> inputValues = {
		1.0,  -1.0, 0.5,   2.0,  -0.5, 1.5,   -2.0, 0.25, 1.25, -1.25, 0.75,  -0.75, 2.5,   -2.5,  0.0,   1.0,
		-1.5, 0.5,  -0.25, 0.25, 1.75, -1.75, 0.5,  -0.5, 2.0,  -2.0,  1.0,   -1.0,  0.25,  -0.25, 1.5,   -1.5,
		0.5,  1.0,  -1.0,  2.0,  -2.0, 0.5,   -0.5, 1.5,  -1.5, 0.25,  -0.25, 0.75,  -0.75, 1.25,  -1.25, 2.5,
		-2.5, 0.0,  1.0,   -1.0, 0.5,  -0.5,  1.5,  -1.5, 2.0,  -2.0,  0.25,  -0.25, 0.75,  -0.75, 1.25,  -1.25,
	};
	auto input = Tensor<CPU>(inputValues, { 2, 32 }, DataType::Float32);
	std::array<Tensor<CUDA>, 1> cudaInputs = { input.CopyToDevice(CUDA{}) };
	auto outputs = module.RunTensors(cudaInputs);
	ASSERT_EQ(outputs.size(), 1u);
	auto actual = outputs[0].CopyToDevice(CPU{});

	std::array<float, 6> expected{};
	for (std::size_t row = 0; row < 2; ++row)
	{
		for (std::size_t out = 0; out < 3; ++out)
		{
			float sum = 0.0F;
			for (std::size_t k = 0; k < 32; ++k)
			{
				const auto q = static_cast<float>(static_cast<int>((out + k) % 17) - 8);
				sum += ReadFloat(input, row * 32 + k) * 0.25F * q;
			}
			expected[row * 3 + out] = sum;
		}
	}
	ExpectTensorNear(actual, expected, 1.0e-3F);
}

TEST(CompiledModuleCUDATest, RunsNativeGGMLQ4_KQuantizedMatMulOnCUDA)
{
	if (!IsCUDADeviceAvailable())
	{
		GTEST_SKIP() << "CUDA device is not available";
	}
	if (!IsCUDADriverAvailable())
	{
		GTEST_SKIP() << "CUDA driver is not available";
	}

	auto graph = BuildGGMLQ4_KQuantizedMatMulGraph();
	auto module = Compiler<CUDA>::CompileArtifact(Detail::BuildExecutablePlanFromGraph(graph)).Load(CUDA{});
	ASSERT_EQ(module.Backend(), CompiledModuleBackend::CUDANative);

	std::vector<double> inputValues(256);
	for (std::size_t i = 0; i < inputValues.size(); ++i)
	{
		inputValues[i] = static_cast<double>(static_cast<int>(i % 11) - 5) * 0.25;
	}
	auto input = Tensor<CPU>(inputValues, { 1, 256 }, DataType::Float32);
	std::array<Tensor<CUDA>, 1> cudaInputs = { input.CopyToDevice(CUDA{}) };
	auto outputs = module.RunTensors(cudaInputs);
	ASSERT_EQ(outputs.size(), 1u);
	auto actual = outputs[0].CopyToDevice(CPU{});

	std::array<float, 2> expected{};
	for (std::size_t out = 0; out < 2; ++out)
	{
		float sum = 0.0F;
		for (std::size_t k = 0; k < 256; ++k)
		{
			const auto q = static_cast<float>((out + k) % 16);
			sum += ReadFloat(input, k) * 0.25F * q;
		}
		expected[out] = sum;
	}
	ExpectTensorNear(actual, expected, 1.0e-3F);
}
#endif

TEST(CompiledModuleCUDATest, ArtifactLoadsAsCUDABridge)
{
	if (!IsCUDADeviceAvailable())
	{
		GTEST_SKIP() << "CUDA device is not available";
	}

	auto graph = BuildSimplePowGraph();
	auto artifact = Compiler<CUDA>::CompileArtifact(Detail::BuildExecutablePlanFromGraph(graph));
	const auto bridgeDevice = CUDAWithHostFallback();
	auto module = artifact.Load(bridgeDevice);

	EXPECT_EQ(artifact.Backend(), CompiledModuleBackend::CPUNative);
	EXPECT_EQ(module.Backend(), CompiledModuleBackend::CPUNative);

	auto lhs = Tensor<CPU>({ 2, 3, 4, 5 }, { 2, 2 }, DataType::Float32).CopyToDevice(bridgeDevice);
	auto rhs = Tensor<CPU>({ 3, 2, 1, 0 }, { 2, 2 }, DataType::Float32).CopyToDevice(bridgeDevice);
	std::array<Tensor<CUDA>, 2> inputs = { std::move(lhs), std::move(rhs) };

	auto outputs = module.RunTensors(inputs);
	auto cpuOutput = outputs[0].CopyToDevice(CPU{});
	ExpectTensorNear(cpuOutput, std::array{ 8.0f, 9.0f, 4.0f, 1.0f });
}

TEST(CompiledModuleCUDATest, CPUBridgeRequiresExplicitFallbackPolicy)
{
	auto artifact = Compiler<CUDA>::CompileArtifact(Detail::BuildExecutablePlanFromGraph(BuildSimplePowGraph()));
	ASSERT_EQ(artifact.Backend(), CompiledModuleBackend::CPUNative);

	try
	{
		(void) artifact.Load(CUDA{});
		FAIL() << "expected strict CUDA load to reject CPU bridge";
	}
	catch (const std::runtime_error& ex)
	{
		const std::string message = ex.what();
		EXPECT_NE(message.find("CPU bridge"), std::string::npos);
		EXPECT_NE(message.find("CUDAHostFallbackPolicy::Allow"), std::string::npos);
	}

	EXPECT_NO_THROW((void) artifact.Load(CUDAWithHostFallback()));
}

TEST(CompiledModuleCUDATest, CUDABridgeLoadsCPUAOTExternalRegions)
{
	if (!IsCUDADeviceAvailable())
	{
		GTEST_SKIP() << "CUDA device is not available";
	}

	CompilerOptions options;
	options.enableCUDANativeAOT = false;
	options.cpuAOTThreadCount = 4;
	options.cpuAOTParallelMinFlops = 1;
	options.enableCPUAOTExternalRegions = true;
	constexpr std::size_t kBatch = 128;
	auto graph = BuildTinyMLPGraph(kBatch);
	FusionPass{}.Run(graph);

	std::vector<double> inputValues(kBatch * 3);
	for (std::size_t i = 0; i < inputValues.size(); ++i)
	{
		inputValues[i] = static_cast<double>(static_cast<int>(i % 11) - 5) * 0.1;
	}
	std::vector<TensorInputSpec> inputSpecs = {
		TensorInputSpec{ .values = inputValues, .shape = { kBatch, 3 } },
	};

	Runtime::Interpreter<CPU> interpreter;
	const auto expected =
	    interpreter.RunForward(Detail::BuildExecutablePlanFromGraph(graph), MakeCPUInputs(inputSpecs));
	auto artifact = Compiler<CUDA>::CompileArtifact(Detail::BuildExecutablePlanFromGraph(graph), options);
	EXPECT_EQ(artifact.Backend(), CompiledModuleBackend::CPUNative);
	auto separated = artifact.SeparateRodata();
	ASSERT_GT(separated.Constants().size(), 0u);
	ASSERT_GT(separated.Weights().size(), 0u);
	ASSERT_GE(separated.ExternalTensorInfos().size(), 4u);
	const auto weightEntryCount = std::ranges::count_if(separated.ExternalTensorInfos(),
	                                                    [](const auto& info) { return info.region == "weights"; });
	EXPECT_GE(weightEntryCount, 4);

	auto runAndCheck = [&](CompiledModule<CUDA>& module) {
		EXPECT_EQ(module.Backend(), CompiledModuleBackend::CPUNative);
		auto outputs = module.RunTensors(MakeCUDAInputs(inputSpecs));
		ASSERT_EQ(outputs.size(), expected.size());
		ExpectTensorNear(outputs[0].CopyToDevice(CPU{}), expected[0], 1e-4f);
	};

	const auto bridgeDevice = CUDAWithHostFallback();
	auto artifactLoaded = artifact.Load(bridgeDevice);
	runAndCheck(artifactLoaded);

	auto separatedLoaded = CompiledModule<CUDA>::Load(separated.Image(), bridgeDevice);
	runAndCheck(separatedLoaded);
}

TEST(CompiledModuleCUDATest, RunsNativeMatMulWithCUBLAS)
{
	if (!IsCUDADeviceAvailable())
	{
		GTEST_SKIP() << "CUDA device is not available";
	}

	auto graph = BuildSimpleMatMulGraph();
	auto artifact = Compiler<CUDA>::CompileArtifact(Detail::BuildExecutablePlanFromGraph(graph));
	ASSERT_EQ(artifact.Backend(), CompiledModuleBackend::CUDANative);
	auto module = artifact.Load(CUDA{});
	EXPECT_EQ(module.Backend(), CompiledModuleBackend::CUDANative);
	EXPECT_EQ(module.FindInput("lhs"), 0u);
	EXPECT_EQ(module.FindInput("rhs"), 1u);
	EXPECT_EQ(module.FindOutput("matmul"), 0u);

	auto lhs = Tensor<CPU>({ 1, 2, 3, 4 }, { 2, 2 }, DataType::Float32).CopyToDevice(CUDA{});
	auto rhs = Tensor<CPU>({ 10, 20, 30, 40 }, { 2, 2 }, DataType::Float32).CopyToDevice(CUDA{});
	std::array<Tensor<CUDA>, 2> inputs = { std::move(lhs), std::move(rhs) };

	auto outputs = module.RunTensors(inputs);
	ASSERT_EQ(outputs.size(), 1u);
	ExpectTensorNear(outputs[0].CopyToDevice(CPU{}), std::array{ 70.0f, 100.0f, 150.0f, 220.0f });

	std::array<Tensor<CUDA>, 1> out = { Tensor<CUDA>(Uninitialized, { 2, 2 }, DataType::Float32, CUDA{}) };
	module.RunTensorsInto(inputs, out);
	ExpectTensorNear(out[0].CopyToDevice(CPU{}), std::array{ 70.0f, 100.0f, 150.0f, 220.0f });
}

TEST(CompiledModuleCUDATest, RunsNativeElementwiseBinaryOpsWithCUDATensors)
{
	if (!IsCUDADeviceAvailable())
	{
		GTEST_SKIP() << "CUDA device is not available";
	}
	if (!IsCUDADriverAvailable())
	{
		GTEST_SKIP() << "CUDA driver is not available";
	}

	struct Case
	{
		BinaryOp op;
		std::string_view outputName;
		std::array<float, 4> expected;
	};
	const std::array cases = {
		Case{ BinaryOp::Add, "sum", { 11.0f, 22.0f, 33.0f, 44.0f } },
		Case{ BinaryOp::Subtract, "difference", { -9.0f, -18.0f, -27.0f, -36.0f } },
		Case{ BinaryOp::Multiply, "product", { 10.0f, 40.0f, 90.0f, 160.0f } },
		Case{ BinaryOp::Divide, "quotient", { 0.1f, 0.1f, 0.1f, 0.1f } },
		Case{ BinaryOp::Max, "maximum", { 10.0f, 20.0f, 30.0f, 40.0f } },
		Case{ BinaryOp::Min, "minimum", { 1.0f, 2.0f, 3.0f, 4.0f } },
	};

	for (const auto& testCase : cases)
	{
		SCOPED_TRACE(testCase.outputName);
		auto graph = BuildSimpleBinaryGraph(testCase.op, std::string(testCase.outputName));
		auto artifact = Compiler<CUDA>::CompileArtifact(Detail::BuildExecutablePlanFromGraph(graph));
		ASSERT_EQ(artifact.Backend(), CompiledModuleBackend::CUDANative);
		const auto payload = DeserializeCUDANativeInstructionPayload(artifact.Instructions());
		ASSERT_EQ(payload.binaryKind, CUDANativeBinaryKind::PTX);
		EXPECT_NE(BytesToString(payload.binary).find("Generated by LLVM NVPTX Back-End"), std::string::npos);
		auto module = artifact.Load(CUDA{});
		EXPECT_EQ(module.Backend(), CompiledModuleBackend::CUDANative);
		EXPECT_EQ(module.FindOutput(testCase.outputName), 0u);

		auto lhs = Tensor<CPU>({ 1, 2, 3, 4 }, { 2, 2 }, DataType::Float32).CopyToDevice(CUDA{});
		auto rhs = Tensor<CPU>({ 10, 20, 30, 40 }, { 2, 2 }, DataType::Float32).CopyToDevice(CUDA{});
		std::array<Tensor<CUDA>, 2> inputs = { std::move(lhs), std::move(rhs) };

		auto outputs = module.RunTensors(inputs);
		ASSERT_EQ(outputs.size(), 1u);
		auto cpuOutput = outputs[0].CopyToDevice(CPU{});
		ExpectTensorNear(cpuOutput, testCase.expected);

		std::array<Tensor<CUDA>, 1> out = { Tensor<CUDA>(Uninitialized, { 2, 2 }, DataType::Float32, CUDA{}) };
		module.RunTensorsInto(inputs, out);
		auto cpuOutInto = out[0].CopyToDevice(CPU{});
		ExpectTensorNear(cpuOutInto, testCase.expected);
	}
}

TEST(CompiledModuleCUDATest, RunsNativeElementwiseBroadcastBinaryOpsWithCUDATensors)
{
	if (!IsCUDADeviceAvailable())
	{
		GTEST_SKIP() << "CUDA device is not available";
	}
	if (!IsCUDADriverAvailable())
	{
		GTEST_SKIP() << "CUDA driver is not available";
	}

	struct Case
	{
		BinaryOp op;
		std::string_view outputName;
		std::vector<std::size_t> lhsShape;
		std::vector<std::size_t> rhsShape;
		std::vector<double> lhs;
		std::vector<double> rhs;
		std::vector<float> expected;
	};
	const std::vector cases = {
		Case{
		    .op = BinaryOp::Add,
		    .outputName = "bias_added",
		    .lhsShape = { 2, 3 },
		    .rhsShape = { 1, 3 },
		    .lhs = { 1, 2, 3, 4, 5, 6 },
		    .rhs = { 10, 20, 30 },
		    .expected = { 11.0f, 22.0f, 33.0f, 14.0f, 25.0f, 36.0f },
		},
		Case{
		    .op = BinaryOp::Subtract,
		    .outputName = "row_shifted",
		    .lhsShape = { 2, 3 },
		    .rhsShape = { 2, 1 },
		    .lhs = { 1, 2, 3, 4, 5, 6 },
		    .rhs = { 10, 20 },
		    .expected = { -9.0f, -8.0f, -7.0f, -16.0f, -15.0f, -14.0f },
		},
		Case{
		    .op = BinaryOp::Multiply,
		    .outputName = "scaled",
		    .lhsShape = { 1, 3 },
		    .rhsShape = { 2, 3 },
		    .lhs = { 2, 3, 4 },
		    .rhs = { 1, 2, 3, 4, 5, 6 },
		    .expected = { 2.0f, 6.0f, 12.0f, 8.0f, 15.0f, 24.0f },
		},
		Case{
		    .op = BinaryOp::Max,
		    .outputName = "broadcast_max",
		    .lhsShape = { 2, 3 },
		    .rhsShape = { 1, 3 },
		    .lhs = { 1, 25, 3, 40, 5, 60 },
		    .rhs = { 10, 20, 30 },
		    .expected = { 10.0f, 25.0f, 30.0f, 40.0f, 20.0f, 60.0f },
		},
		Case{
		    .op = BinaryOp::Min,
		    .outputName = "broadcast_min",
		    .lhsShape = { 2, 1 },
		    .rhsShape = { 2, 3 },
		    .lhs = { 7, 18 },
		    .rhs = { 1, 9, 8, 20, 2, 30 },
		    .expected = { 1.0f, 7.0f, 7.0f, 18.0f, 2.0f, 18.0f },
		},
	};

	for (const auto& testCase : cases)
	{
		SCOPED_TRACE(testCase.outputName);
		const std::array outputShape{ 2uz, 3uz };
		auto graph = BuildBinaryGraph(testCase.op, testCase.lhsShape, testCase.rhsShape, outputShape,
		                              std::string(testCase.outputName));
		auto artifact = Compiler<CUDA>::CompileArtifact(Detail::BuildExecutablePlanFromGraph(graph));
		ASSERT_EQ(artifact.Backend(), CompiledModuleBackend::CUDANative);
		const auto payload = DeserializeCUDANativeInstructionPayload(artifact.Instructions());
		ASSERT_EQ(payload.binaryKind, CUDANativeBinaryKind::PTX);
		EXPECT_NE(BytesToString(payload.binary).find("Generated by LLVM NVPTX Back-End"), std::string::npos);
		auto module = artifact.Load(CUDA{});
		EXPECT_EQ(module.Backend(), CompiledModuleBackend::CUDANative);
		EXPECT_EQ(module.FindOutput(testCase.outputName), 0u);

		auto lhs = Tensor<CPU>(std::span<const double>(testCase.lhs.data(), testCase.lhs.size()),
		                       ShapeView{ testCase.lhsShape }, DataType::Float32)
		               .CopyToDevice(CUDA{});
		auto rhs = Tensor<CPU>(std::span<const double>(testCase.rhs.data(), testCase.rhs.size()),
		                       ShapeView{ testCase.rhsShape }, DataType::Float32)
		               .CopyToDevice(CUDA{});
		std::array<Tensor<CUDA>, 2> inputs = { std::move(lhs), std::move(rhs) };

		auto outputs = module.RunTensors(inputs);
		ASSERT_EQ(outputs.size(), 1u);
		ExpectTensorNear(outputs[0].CopyToDevice(CPU{}), testCase.expected);

		std::array<Tensor<CUDA>, 1> out = { Tensor<CUDA>(Uninitialized, { 2, 3 }, DataType::Float32, CUDA{}) };
		module.RunTensorsInto(inputs, out);
		ExpectTensorNear(out[0].CopyToDevice(CPU{}), testCase.expected);
	}
}

TEST(CompiledModuleCUDATest, RunsNativeElementwiseUnaryOpsWithCUDATensors)
{
	if (!IsCUDADeviceAvailable())
	{
		GTEST_SKIP() << "CUDA device is not available";
	}
	if (!IsCUDADriverAvailable())
	{
		GTEST_SKIP() << "CUDA driver is not available";
	}

	struct Case
	{
		UnaryOp op;
		std::string_view outputName;
		std::array<double, 4> input;
		std::array<float, 4> expected;
		float tolerance{ 1e-6f };
	};
	const std::array cases = {
		Case{ UnaryOp::Negate, "negated", { -4.0, -1.0, 0.0, 9.0 }, { 4.0f, 1.0f, 0.0f, -9.0f } },
		Case{ UnaryOp::Abs, "absolute", { -4.0, -1.0, 0.0, 9.0 }, { 4.0f, 1.0f, 0.0f, 9.0f } },
		Case{ UnaryOp::Sqrt, "sqrt", { 4.0, 1.0, 0.0, 9.0 }, { 2.0f, 1.0f, 0.0f, 3.0f } },
		Case{
		    UnaryOp::Exp,
		    "exp",
		    { 0.0, 1.0, -1.0, 2.0 },
		    { static_cast<float>(std::exp(0.0)), static_cast<float>(std::exp(1.0)), static_cast<float>(std::exp(-1.0)),
		      static_cast<float>(std::exp(2.0)) },
		    2e-3f,
		},
		Case{
		    UnaryOp::Log,
		    "log",
		    { 1.0, 2.0, 4.0, 0.5 },
		    { static_cast<float>(std::log(1.0)), static_cast<float>(std::log(2.0)), static_cast<float>(std::log(4.0)),
		      static_cast<float>(std::log(0.5)) },
		    2e-3f,
		},
		Case{
		    UnaryOp::Sin,
		    "sin",
		    { 0.0, 0.5, -0.5, 1.0 },
		    { static_cast<float>(std::sin(0.0)), static_cast<float>(std::sin(0.5)), static_cast<float>(std::sin(-0.5)),
		      static_cast<float>(std::sin(1.0)) },
		    2e-3f,
		},
		Case{
		    UnaryOp::Cos,
		    "cos",
		    { 0.0, 0.5, -0.5, 1.0 },
		    { static_cast<float>(std::cos(0.0)), static_cast<float>(std::cos(0.5)), static_cast<float>(std::cos(-0.5)),
		      static_cast<float>(std::cos(1.0)) },
		    2e-3f,
		},
	};

	for (const auto& testCase : cases)
	{
		SCOPED_TRACE(testCase.outputName);
		auto graph = BuildSimpleUnaryGraph(testCase.op, std::string(testCase.outputName));
		auto artifact = Compiler<CUDA>::CompileArtifact(Detail::BuildExecutablePlanFromGraph(graph));
		ASSERT_EQ(artifact.Backend(), CompiledModuleBackend::CUDANative);
		const auto payload = DeserializeCUDANativeInstructionPayload(artifact.Instructions());
		ASSERT_EQ(payload.binaryKind, CUDANativeBinaryKind::PTX);
		EXPECT_NE(BytesToString(payload.binary).find("Generated by LLVM NVPTX Back-End"), std::string::npos);
		auto module = artifact.Load(CUDA{});
		EXPECT_EQ(module.Backend(), CompiledModuleBackend::CUDANative);
		EXPECT_EQ(module.FindInput("input"), 0u);
		EXPECT_EQ(module.FindOutput(testCase.outputName), 0u);

		auto input = Tensor<CPU>({ testCase.input[0], testCase.input[1], testCase.input[2], testCase.input[3] },
		                         { 2, 2 }, DataType::Float32)
		                 .CopyToDevice(CUDA{});
		std::array<Tensor<CUDA>, 1> inputs = { std::move(input) };

		auto outputs = module.RunTensors(inputs);
		ASSERT_EQ(outputs.size(), 1u);
		auto cpuOutput = outputs[0].CopyToDevice(CPU{});
		ExpectTensorNear(cpuOutput, testCase.expected, testCase.tolerance);

		std::array<Tensor<CUDA>, 1> out = { Tensor<CUDA>(Uninitialized, { 2, 2 }, DataType::Float32, CUDA{}) };
		module.RunTensorsInto(inputs, out);
		auto cpuOutInto = out[0].CopyToDevice(CPU{});
		ExpectTensorNear(cpuOutInto, testCase.expected, testCase.tolerance);
	}
}

TEST(CompiledModuleCUDATest, RunIntoHonorsExternalCUDAStreamForNativePayload)
{
	if (!IsCUDADeviceAvailable())
	{
		GTEST_SKIP() << "CUDA device is not available";
	}
	if (!IsCUDADriverAvailable())
	{
		GTEST_SKIP() << "CUDA driver is not available";
	}

	auto graph = BuildSimpleBinaryGraph(BinaryOp::Add, "sum");
	auto module = Compiler<CUDA>::Compile(Detail::BuildExecutablePlanFromGraph(graph), CUDA{});
	ASSERT_EQ(module.Backend(), CompiledModuleBackend::CUDANative);

	auto lhs = Tensor<CPU>({ 1, 2, 3, 4 }, { 2, 2 }, DataType::Float32).CopyToDevice(CUDA{});
	auto rhs = Tensor<CPU>({ 10, 20, 30, 40 }, { 2, 2 }, DataType::Float32).CopyToDevice(CUDA{});
	std::array<Tensor<CUDA>, 2> inputs = { std::move(lhs), std::move(rhs) };
	std::array<Tensor<CUDA>, 1> outputs = { Tensor<CUDA>(Uninitialized, { 2, 2 }, DataType::Float32, CUDA{}) };

	cudaStream_t stream{};
	ASSERT_EQ(cudaStreamCreate(&stream), cudaSuccess);
	module.RunTensorsInto(inputs, outputs, CompiledModuleCUDARunOptions{ .stream = stream, .synchronize = false });
	EXPECT_EQ(cudaStreamSynchronize(stream), cudaSuccess);
	EXPECT_EQ(cudaStreamDestroy(stream), cudaSuccess);

	ExpectTensorNear(outputs[0].CopyToDevice(CPU{}), std::array{ 11.0f, 22.0f, 33.0f, 44.0f });
}

TEST(CompiledModuleCUDATest, CPUBridgeRejectsAsynchronousRunOptions)
{
	if (!IsCUDADeviceAvailable())
	{
		GTEST_SKIP() << "CUDA device is not available";
	}

	const auto bridgeDevice = CUDAWithHostFallback();
	auto module = Compiler<CUDA>::Compile(Detail::BuildExecutablePlanFromGraph(BuildSimplePowGraph()), bridgeDevice);
	ASSERT_EQ(module.Backend(), CompiledModuleBackend::CPUNative);
	auto lhs = Tensor<CPU>({ 2, 3, 4, 5 }, { 2, 2 }, DataType::Float32).CopyToDevice(bridgeDevice);
	auto rhs = Tensor<CPU>({ 1, 2, 3, 0 }, { 2, 2 }, DataType::Float32).CopyToDevice(bridgeDevice);
	std::array<Tensor<CUDA>, 2> inputs = { std::move(lhs), std::move(rhs) };
	std::array<Tensor<CUDA>, 1> outputs = { Tensor<CUDA>(Uninitialized, { 2, 2 }, DataType::Float32, bridgeDevice) };

	try
	{
		module.RunTensorsInto(inputs, outputs, CompiledModuleCUDARunOptions{ .synchronize = false });
		FAIL() << "expected CPU bridge async policy validation to throw";
	}
	catch (const std::runtime_error& ex)
	{
		const std::string message = ex.what();
		EXPECT_NE(message.find("CPU bridge"), std::string::npos);
		EXPECT_NE(message.find("asynchronous"), std::string::npos);
	}
}

TEST(CompiledModuleCUDATest, CompilerArtifactsExposeP3NativePayloads)
{
#ifdef LITENN_ENABLE_CUDA_DRIVER
	{
		auto artifact = Compiler<CUDA>::CompileArtifact(
		    Detail::BuildExecutablePlanFromGraph(BuildReduceGraph(ReduceOp::Mean, 0, { 3 }, "mean_axis0")));
		ASSERT_EQ(artifact.Backend(), CompiledModuleBackend::CUDANative);
		const auto payload = DeserializeCUDANativeInstructionPayload(artifact.Instructions());
		EXPECT_EQ(payload.binaryKind, CUDANativeBinaryKind::PTX);
		EXPECT_TRUE(payload.featureSet.HasFeature(CUDANativeFeature::ReduceF32));
		EXPECT_EQ(payload.target, CUDANativeNVPTXTargetChip());
		ASSERT_EQ(payload.kernels.size(), 1u);
		EXPECT_EQ(payload.kernels[0].name, "litenn_reduce_mean_f32");
	}

	{
		auto artifact = Compiler<CUDA>::CompileArtifact(
		    Detail::BuildExecutablePlanFromGraph(BuildConcatGraph({ 2, 3 }, { 2, 2 }, { 2, 5 }, 1, "concat_axis1")));
		ASSERT_EQ(artifact.Backend(), CompiledModuleBackend::CUDANative);
		const auto payload = DeserializeCUDANativeInstructionPayload(artifact.Instructions());
		EXPECT_EQ(payload.binaryKind, CUDANativeBinaryKind::PTX);
		EXPECT_TRUE(payload.featureSet.HasFeature(CUDANativeFeature::ConcatF32));
		EXPECT_TRUE(payload.featureSet.HasFeature(CUDANativeFeature::MultiKernelLaunch));
		ASSERT_EQ(payload.kernels.size(), 2u);
		EXPECT_EQ(payload.kernels[0].name, "litenn_concat_f32_input_0");
		EXPECT_EQ(payload.kernels[1].name, "litenn_concat_f32_input_1");
	}

	{
		auto artifact = Compiler<CUDA>::CompileArtifact(
		    Detail::BuildExecutablePlanFromGraph(BuildSliceGraph(1, 1, 3, { 2, 3 }, "slice_axis1")));
		ASSERT_EQ(artifact.Backend(), CompiledModuleBackend::CUDANative);
		const auto payload = DeserializeCUDANativeInstructionPayload(artifact.Instructions());
		EXPECT_EQ(payload.binaryKind, CUDANativeBinaryKind::PTX);
		EXPECT_TRUE(payload.featureSet.HasFeature(CUDANativeFeature::SliceF32));
		ASSERT_EQ(payload.kernels.size(), 1u);
		EXPECT_EQ(payload.kernels[0].name, "litenn_slice_f32");
	}

	{
		auto artifact = Compiler<CUDA>::CompileArtifact(Detail::BuildExecutablePlanFromGraph(
		    BuildCastGraph(DataType::Float32, DataType::Float16, { 2, 2 }, "cast_f16")));
		ASSERT_EQ(artifact.Backend(), CompiledModuleBackend::CUDANative);
		const auto payload = DeserializeCUDANativeInstructionPayload(artifact.Instructions());
		EXPECT_EQ(payload.binaryKind, CUDANativeBinaryKind::PTX);
		EXPECT_TRUE(payload.featureSet.HasFeature(CUDANativeFeature::Cast));
		EXPECT_EQ(payload.target, CUDANativeNVPTXTargetChip());
		ASSERT_EQ(payload.kernels.size(), 1u);
		EXPECT_EQ(payload.kernels[0].name, CUDANativeCastKernelName(DataType::Float32, DataType::Float16));
		ASSERT_EQ(payload.kernels[0].arguments.size(), 3u);
		EXPECT_EQ(payload.kernels[0].arguments[0].kind, CUDANativeArgumentKind::OutputTensor);
		EXPECT_EQ(payload.kernels[0].arguments[0].byteSize, 8u);
		EXPECT_EQ(payload.kernels[0].arguments[1].kind, CUDANativeArgumentKind::InputTensor);
		EXPECT_EQ(payload.kernels[0].arguments[1].byteSize, 16u);
		EXPECT_EQ(payload.kernels[0].arguments[2].kind, CUDANativeArgumentKind::Scalar);
		EXPECT_EQ(payload.kernels[0].arguments[2].byteSize, sizeof(std::uint32_t));
	}

	{
		struct NativeCastArtifactCase
		{
			std::string name;
			DataType srcType{ DataType::Float32 };
			DataType dstType{ DataType::Float32 };
		};

		const std::vector<NativeCastArtifactCase> cases = {
			{ .name = "cast_bf16", .srcType = DataType::Float32, .dstType = DataType::BFloat16 },
			{ .name = "cast_f32_from_bf16", .srcType = DataType::BFloat16, .dstType = DataType::Float32 },
			{ .name = "cast_f8e4m3", .srcType = DataType::Float32, .dstType = DataType::Float8E4M3 },
			{ .name = "cast_f32_from_f8e4m3", .srcType = DataType::Float8E4M3, .dstType = DataType::Float32 },
			{ .name = "cast_f8e5m2", .srcType = DataType::Float32, .dstType = DataType::Float8E5M2 },
			{ .name = "cast_f32_from_f8e5m2", .srcType = DataType::Float8E5M2, .dstType = DataType::Float32 },
		};

		for (const auto& testCase : cases)
		{
			SCOPED_TRACE(testCase.name);
			auto artifact = Compiler<CUDA>::CompileArtifact(Detail::BuildExecutablePlanFromGraph(
			    BuildCastGraph(testCase.srcType, testCase.dstType, { 2, 2 }, testCase.name)));
			ASSERT_EQ(artifact.Backend(), CompiledModuleBackend::CUDANative);
			const auto payload = DeserializeCUDANativeInstructionPayload(artifact.Instructions());
			EXPECT_EQ(payload.binaryKind, CUDANativeBinaryKind::PTX);
			EXPECT_TRUE(payload.featureSet.HasFeature(CUDANativeFeature::Cast));
			EXPECT_EQ(payload.target, CUDANativeNVPTXTargetChip());
			ASSERT_EQ(payload.kernels.size(), 1u);
			EXPECT_EQ(payload.kernels[0].name, CUDANativeCastKernelName(testCase.srcType, testCase.dstType));
			ASSERT_EQ(payload.kernels[0].arguments.size(), 3u);
			EXPECT_EQ(payload.kernels[0].arguments[0].kind, CUDANativeArgumentKind::OutputTensor);
			EXPECT_EQ(payload.kernels[0].arguments[0].byteSize,
			          static_cast<std::uint32_t>(4u * ElementByteSize(testCase.dstType)));
			EXPECT_EQ(payload.kernels[0].arguments[1].kind, CUDANativeArgumentKind::InputTensor);
			EXPECT_EQ(payload.kernels[0].arguments[1].byteSize,
			          static_cast<std::uint32_t>(4u * ElementByteSize(testCase.srcType)));
			EXPECT_EQ(payload.kernels[0].arguments[2].kind, CUDANativeArgumentKind::Scalar);
			EXPECT_EQ(payload.kernels[0].arguments[2].byteSize, sizeof(std::uint32_t));
		}
	}

	{
		auto artifact = Compiler<CUDA>::CompileArtifact(
		    Detail::BuildExecutablePlanFromGraph(BuildSimpleMatMulGraph(DataType::Float16, "matmul_f16")));
		ASSERT_EQ(artifact.Backend(), CompiledModuleBackend::CUDANative);
		const auto payload = DeserializeCUDANativeInstructionPayload(artifact.Instructions());
		EXPECT_EQ(payload.binaryKind, CUDANativeBinaryKind::LibraryCall);
		EXPECT_TRUE(payload.featureSet.HasFeature(CUDANativeFeature::MatMulCUBLASLowPrecision));
		EXPECT_EQ(payload.target, "cublas");
		ASSERT_EQ(payload.kernels.size(), 1u);
		EXPECT_EQ(payload.kernels[0].name, "litenn_cublas_matmul_f16");
		EXPECT_EQ(payload.kernels[0].arguments[0].byteSize, 8u);
		EXPECT_EQ(payload.kernels[0].arguments[1].byteSize, 8u);
		EXPECT_EQ(payload.kernels[0].arguments[2].byteSize, 8u);
	}

	{
		auto graph = BuildMatMulBiasGraph(true, DataType::Float16);
		FusionPass{}.Run(graph);
		auto artifact = Compiler<CUDA>::CompileArtifact(Detail::BuildExecutablePlanFromGraph(graph));
		ASSERT_EQ(artifact.Backend(), CompiledModuleBackend::CUDANative);
		const auto payload = DeserializeCUDANativeInstructionPayload(artifact.Instructions());
		EXPECT_EQ(payload.binaryKind, CUDANativeBinaryKind::PTX);
		EXPECT_TRUE(payload.featureSet.HasFeature(CUDANativeFeature::MatMulCUBLASLowPrecision));
		EXPECT_TRUE(payload.featureSet.HasFeature(CUDANativeFeature::MatMulBiasAddReLULowPrecision));
		ASSERT_EQ(payload.kernels.size(), 2u);
		EXPECT_EQ(payload.kernels[0].name, "litenn_cublas_matmul_f16");
		EXPECT_EQ(payload.kernels[1].name, "litenn_matmul_bias_relu_epilogue_f16");
	}

	{
		auto graph = BuildMatMulBiasGraph(true);
		FusionPass{}.Run(graph);
		auto artifact = Compiler<CUDA>::CompileArtifact(Detail::BuildExecutablePlanFromGraph(graph));
		ASSERT_EQ(artifact.Backend(), CompiledModuleBackend::CUDANative);
		const auto payload = DeserializeCUDANativeInstructionPayload(artifact.Instructions());
		EXPECT_EQ(payload.binaryKind, CUDANativeBinaryKind::PTX);
		EXPECT_TRUE(payload.featureSet.HasFeature(CUDANativeFeature::MatMulCUBLASF32));
		EXPECT_TRUE(payload.featureSet.HasFeature(CUDANativeFeature::MatMulBiasAddReLUF32));
		EXPECT_TRUE(payload.featureSet.HasFeature(CUDANativeFeature::MultiKernelLaunch));
		ASSERT_EQ(payload.kernels.size(), 2u);
		EXPECT_EQ(payload.kernels[0].name, "litenn_cublas_matmul_f32");
		EXPECT_EQ(payload.kernels[1].name, "litenn_matmul_bias_relu_epilogue_f32");
	}
#else
	GTEST_SKIP() << "CUDA driver support is not enabled";
#endif
}

TEST(CompiledModuleCUDATest, CompilerArtifactsExposeLowPrecisionNativeLinearChainPayload)
{
#ifdef LITENN_ENABLE_CUDA_DRIVER
	auto graph = BuildTinyMLPGraph(2, DataType::Float16);
	FusionPass{}.Run(graph);
	auto artifact = Compiler<CUDA>::CompileArtifact(Detail::BuildExecutablePlanFromGraph(graph));
	ASSERT_EQ(artifact.Backend(), CompiledModuleBackend::CUDANative) << Debug::DumpGraph(graph);
	const auto payload = DeserializeCUDANativeInstructionPayload(artifact.Instructions());
	EXPECT_EQ(payload.binaryKind, CUDANativeBinaryKind::PTX);
	EXPECT_TRUE(payload.featureSet.HasFeature(CUDANativeFeature::MatMulCUBLASLowPrecision));
	EXPECT_TRUE(payload.featureSet.HasFeature(CUDANativeFeature::MatMulBiasAddLowPrecision));
	EXPECT_TRUE(payload.featureSet.HasFeature(CUDANativeFeature::MatMulBiasAddReLULowPrecision));
	EXPECT_TRUE(payload.featureSet.HasFeature(CUDANativeFeature::MultiKernelLaunch));
	EXPECT_TRUE(payload.featureSet.HasFeature(CUDANativeFeature::Workspace));
	EXPECT_TRUE(payload.featureSet.HasFeature(CUDANativeFeature::ConstantTensor));
	EXPECT_GT(payload.constantData.size(), 0u);
	ASSERT_EQ(payload.kernels.size(), 4u);
	EXPECT_EQ(payload.kernels[0].name, "litenn_cublas_matmul_f16");
	EXPECT_EQ(payload.kernels[1].name, "litenn_matmul_bias_relu_epilogue_f16_0");
	EXPECT_EQ(payload.kernels[2].name, "litenn_cublas_matmul_f16");
	EXPECT_EQ(payload.kernels[3].name, "litenn_matmul_bias_add_epilogue_f16_1");
#else
	GTEST_SKIP() << "CUDA driver support is not enabled";
#endif
}

TEST(CompiledModuleCUDATest, RunsNativeCastPayloadsOnCUDA)
{
	if (!IsCUDADeviceAvailable())
	{
		GTEST_SKIP() << "CUDA device is not available";
	}
	if (!IsCUDADriverAvailable())
	{
		GTEST_SKIP() << "CUDA driver is not available";
	}

	struct NativeCastCase
	{
		std::string name;
		DataType srcType{ DataType::Float32 };
		DataType dstType{ DataType::Float32 };
		std::vector<double> values;
		float tolerance{ 1e-6f };
		bool requiresNativeConversion{};
	};

	const std::vector<NativeCastCase> cases = {
		{ .name = "f32_to_f16",
		  .srcType = DataType::Float32,
		  .dstType = DataType::Float16,
		  .values = { 1.25, -2.5, 0.0625, 3.75 },
		  .tolerance = 1e-3f },
		{ .name = "f16_to_f32",
		  .srcType = DataType::Float16,
		  .dstType = DataType::Float32,
		  .values = { 1.25, -2.5, 0.0625, 3.75 },
		  .tolerance = 1e-3f },
		{ .name = "f32_to_bf16",
		  .srcType = DataType::Float32,
		  .dstType = DataType::BFloat16,
		  .values = { 1.25, -2.5, 0.0625, 3.75 },
		  .tolerance = 2e-2f,
		  .requiresNativeConversion = true },
		{ .name = "bf16_to_f32",
		  .srcType = DataType::BFloat16,
		  .dstType = DataType::Float32,
		  .values = { 1.25, -2.5, 0.0625, 3.75 },
		  .tolerance = 2e-2f,
		  .requiresNativeConversion = true },
		{ .name = "f32_to_f8e4m3",
		  .srcType = DataType::Float32,
		  .dstType = DataType::Float8E4M3,
		  .values = { 1.0, -2.0, 0.5, 4.0 },
		  .tolerance = 2.5e-1f,
		  .requiresNativeConversion = true },
		{ .name = "f8e4m3_to_f32",
		  .srcType = DataType::Float8E4M3,
		  .dstType = DataType::Float32,
		  .values = { 1.0, -2.0, 0.5, 4.0 },
		  .tolerance = 2.5e-1f,
		  .requiresNativeConversion = true },
		{ .name = "f32_to_f8e5m2",
		  .srcType = DataType::Float32,
		  .dstType = DataType::Float8E5M2,
		  .values = { 1.0, -2.0, 0.5, 4.0 },
		  .tolerance = 5e-1f,
		  .requiresNativeConversion = true },
		{ .name = "f8e5m2_to_f32",
		  .srcType = DataType::Float8E5M2,
		  .dstType = DataType::Float32,
		  .values = { 1.0, -2.0, 0.5, 4.0 },
		  .tolerance = 5e-1f,
		  .requiresNativeConversion = true },
		{ .name = "f32_to_i8",
		  .srcType = DataType::Float32,
		  .dstType = DataType::Int8,
		  .values = { 1.0, -2.0, 3.0, -4.0 } },
		{ .name = "f32_to_u8",
		  .srcType = DataType::Float32,
		  .dstType = DataType::UInt8,
		  .values = { 1.0, 2.0, 3.0, 4.0 } },
		{ .name = "i8_to_f32",
		  .srcType = DataType::Int8,
		  .dstType = DataType::Float32,
		  .values = { 1.0, -2.0, 3.0, -4.0 } },
		{ .name = "u8_to_f32",
		  .srcType = DataType::UInt8,
		  .dstType = DataType::Float32,
		  .values = { 1.0, 2.0, 3.0, 4.0 } },
	};

	for (const auto& testCase : cases)
	{
		if (testCase.requiresNativeConversion && !CUDASupportsNativeConversion(testCase.srcType, testCase.dstType))
		{
			continue;
		}

		SCOPED_TRACE(testCase.name);
		auto graph = BuildCastGraph(testCase.srcType, testCase.dstType, { 2, 2 }, testCase.name);
		auto artifact = Compiler<CUDA>::CompileArtifact(Detail::BuildExecutablePlanFromGraph(graph));
		ASSERT_EQ(artifact.Backend(), CompiledModuleBackend::CUDANative) << Debug::DumpGraph(graph);

		Runtime::Interpreter<CPU> interpreter;
		std::vector<TensorInputSpec> inputSpecs = { TensorInputSpec{
			.values = testCase.values, .shape = { 2, 2 }, .dtype = testCase.srcType } };
		auto expectedInputs = MakeCPUInputs(inputSpecs);
		const auto expected = interpreter.RunForward(Detail::BuildExecutablePlanFromGraph(graph), expectedInputs);

		auto module = artifact.Load(CUDA{});
		auto cudaInputs = MakeCUDAInputs(inputSpecs);
		auto cudaOutputs = module.RunTensors(cudaInputs);
		ASSERT_EQ(cudaOutputs.size(), 1u);
		auto actual = cudaOutputs[0].CopyToDevice(CPU{});
		ExpectTensorNear(actual, expected[0], testCase.tolerance);
	}
}

TEST(CompiledModuleCUDATest, RunsNativeLowPrecisionMatMulPayloadsOnCUDA)
{
	if (!IsCUDADeviceAvailable())
	{
		GTEST_SKIP() << "CUDA device is not available";
	}

	struct LowPrecisionMatMulCase
	{
		std::string name;
		DataType dtype{ DataType::Float16 };
		std::size_t m{ 2 };
		std::size_t k{ 3 };
		std::size_t n{ 2 };
		std::vector<double> lhsValues;
		std::vector<double> rhsValues;
		float tolerance{ 1e-6f };
	};

	const std::vector<LowPrecisionMatMulCase> cases = {
		{ .name = "matmul_f16",
		  .dtype = DataType::Float16,
		  .lhsValues = { 1.0, 2.0, 3.0, 4.0, 5.0, 6.0 },
		  .rhsValues = { 7.0, 8.0, 9.0, 10.0, 11.0, 12.0 },
		  .tolerance = 1e-3f },
		{ .name = "matmul_i8",
		  .dtype = DataType::Int8,
		  .m = 16,
		  .k = 32,
		  .n = 16,
		  .lhsValues = RepeatingValues(16 * 32, { -1.0, 0.0, 1.0, 2.0 }),
		  .rhsValues = RepeatingValues(32 * 16, { 1.0, -1.0, 0.0, 2.0 }) },
		{ .name = "matmul_u8",
		  .dtype = DataType::UInt8,
		  .m = 16,
		  .k = 32,
		  .n = 16,
		  .lhsValues = RepeatingValues(16 * 32, { 0.0, 1.0, 2.0, 3.0 }),
		  .rhsValues = RepeatingValues(32 * 16, { 1.0, 0.0, 2.0, 1.0 }) },
	};

	std::size_t executedCases = 0;
	for (const auto& testCase : cases)
	{
		if (!CUDASupportsNativeMatMul(testCase.dtype))
		{
			continue;
		}

		SCOPED_TRACE(testCase.name);
		auto graph = BuildMatMulGraph(testCase.dtype, { testCase.m, testCase.k }, { testCase.k, testCase.n },
		                              { testCase.m, testCase.n }, testCase.name);
		auto artifact = Compiler<CUDA>::CompileArtifact(Detail::BuildExecutablePlanFromGraph(graph));
		ASSERT_EQ(artifact.Backend(), CompiledModuleBackend::CUDANative) << Debug::DumpGraph(graph);

		Runtime::Interpreter<CPU> interpreter;
		std::vector<TensorInputSpec> inputSpecs = {
			TensorInputSpec{
			    .values = testCase.lhsValues, .shape = { testCase.m, testCase.k }, .dtype = testCase.dtype },
			TensorInputSpec{
			    .values = testCase.rhsValues, .shape = { testCase.k, testCase.n }, .dtype = testCase.dtype },
		};
		auto expectedInputs = MakeCPUInputs(inputSpecs);
		const auto expected = interpreter.RunForward(Detail::BuildExecutablePlanFromGraph(graph), expectedInputs);

		auto module = artifact.Load(CUDA{});
		auto cudaInputs = MakeCUDAInputs(inputSpecs);
		auto cudaOutputs = module.RunTensors(cudaInputs);
		ASSERT_EQ(cudaOutputs.size(), 1u);
		ExpectTensorNear(cudaOutputs[0].CopyToDevice(CPU{}), expected[0], testCase.tolerance);
		++executedCases;
	}

	if (executedCases == 0)
	{
		GTEST_SKIP() << "CUDA device does not report native low-precision MatMul support";
	}
}

TEST(CompiledModuleCUDATest, RunsNativeLowPrecisionMatMulBiasPayloadsOnCUDA)
{
	if (!IsCUDADeviceAvailable())
	{
		GTEST_SKIP() << "CUDA device is not available";
	}
	if (!IsCUDADriverAvailable())
	{
		GTEST_SKIP() << "CUDA driver is not available";
	}

	struct LowPrecisionMatMulBiasCase
	{
		std::string name;
		DataType dtype{ DataType::Float16 };
		bool relu{};
		std::size_t m{ 2 };
		std::size_t k{ 3 };
		std::size_t n{ 2 };
		std::vector<double> lhsValues;
		std::vector<double> rhsValues;
		std::vector<double> biasValues;
		float tolerance{ 1e-6f };
	};

	const std::vector<LowPrecisionMatMulBiasCase> cases = {
		{ .name = "matmul_bias_relu_f16",
		  .dtype = DataType::Float16,
		  .relu = true,
		  .lhsValues = { 1.0, -2.0, 3.0, -1.0, 2.0, -3.0 },
		  .rhsValues = { 1.0, 2.0, -1.0, 3.0, 2.0, -2.0 },
		  .biasValues = { 0.5, -0.75 },
		  .tolerance = 1e-3f },
		{ .name = "matmul_bias_i8",
		  .dtype = DataType::Int8,
		  .relu = false,
		  .m = 16,
		  .k = 32,
		  .n = 16,
		  .lhsValues = RepeatingValues(16 * 32, { -1.0, 0.0, 1.0, 2.0 }),
		  .rhsValues = RepeatingValues(32 * 16, { 1.0, -1.0, 0.0, 2.0 }),
		  .biasValues = RepeatingValues(16, { 1.0, -2.0 }) },
		{ .name = "matmul_bias_u8",
		  .dtype = DataType::UInt8,
		  .relu = true,
		  .m = 16,
		  .k = 32,
		  .n = 16,
		  .lhsValues = RepeatingValues(16 * 32, { 0.0, 1.0, 2.0, 3.0 }),
		  .rhsValues = RepeatingValues(32 * 16, { 1.0, 0.0, 2.0, 1.0 }),
		  .biasValues = RepeatingValues(16, { 1.0, 2.0 }) },
	};

	std::size_t executedCases = 0;
	for (const auto& testCase : cases)
	{
		if (!CUDASupportsNativeMatMul(testCase.dtype))
		{
			continue;
		}

		SCOPED_TRACE(testCase.name);
		auto graph = BuildMatMulBiasGraph(testCase.relu, testCase.dtype, testCase.m, testCase.k, testCase.n);
		FusionPass{}.Run(graph);
		auto artifact = Compiler<CUDA>::CompileArtifact(Detail::BuildExecutablePlanFromGraph(graph));
		ASSERT_EQ(artifact.Backend(), CompiledModuleBackend::CUDANative) << Debug::DumpGraph(graph);

		Runtime::Interpreter<CPU> interpreter;
		std::vector<TensorInputSpec> inputSpecs = {
			TensorInputSpec{
			    .values = testCase.lhsValues, .shape = { testCase.m, testCase.k }, .dtype = testCase.dtype },
			TensorInputSpec{
			    .values = testCase.rhsValues, .shape = { testCase.k, testCase.n }, .dtype = testCase.dtype },
			TensorInputSpec{ .values = testCase.biasValues, .shape = { 1, testCase.n }, .dtype = testCase.dtype },
		};
		auto expectedInputs = MakeCPUInputs(inputSpecs);
		const auto expected = interpreter.RunForward(Detail::BuildExecutablePlanFromGraph(graph), expectedInputs);

		auto module = artifact.Load(CUDA{});
		auto cudaInputs = MakeCUDAInputs(inputSpecs);
		auto cudaOutputs = module.RunTensors(cudaInputs);
		ASSERT_EQ(cudaOutputs.size(), 1u);
		ExpectTensorNear(cudaOutputs[0].CopyToDevice(CPU{}), expected[0], testCase.tolerance);
		++executedCases;
	}

	if (executedCases == 0)
	{
		GTEST_SKIP() << "CUDA device does not report native low-precision MatMul support";
	}
}

TEST(CompiledModuleCUDATest, CompilerArtifactsExposeNativeLinearChainPayload)
{
#ifdef LITENN_ENABLE_CUDA_DRIVER
	auto graph = BuildTinyMLPGraph(2);
	FusionPass{}.Run(graph);
	auto artifact = Compiler<CUDA>::CompileArtifact(Detail::BuildExecutablePlanFromGraph(graph));
	ASSERT_EQ(artifact.Backend(), CompiledModuleBackend::CUDANative) << Debug::DumpGraph(graph);
	const auto payload = DeserializeCUDANativeInstructionPayload(artifact.Instructions());
	EXPECT_EQ(payload.binaryKind, CUDANativeBinaryKind::PTX);
	EXPECT_TRUE(payload.featureSet.HasFeature(CUDANativeFeature::MatMulCUBLASF32));
	EXPECT_TRUE(payload.featureSet.HasFeature(CUDANativeFeature::MatMulBiasAddF32));
	EXPECT_TRUE(payload.featureSet.HasFeature(CUDANativeFeature::MatMulBiasAddReLUF32));
	EXPECT_TRUE(payload.featureSet.HasFeature(CUDANativeFeature::MultiKernelLaunch));
	EXPECT_TRUE(payload.featureSet.HasFeature(CUDANativeFeature::Workspace));
	EXPECT_TRUE(payload.featureSet.HasFeature(CUDANativeFeature::ConstantTensor));
	EXPECT_GT(payload.constantData.size(), 0u);
	ASSERT_EQ(payload.kernels.size(), 4u);
	EXPECT_EQ(payload.kernels[0].name, "litenn_cublas_matmul_f32");
	EXPECT_EQ(payload.kernels[1].name, "litenn_matmul_bias_relu_epilogue_f32_0");
	EXPECT_EQ(payload.kernels[2].name, "litenn_cublas_matmul_f32");
	EXPECT_EQ(payload.kernels[3].name, "litenn_matmul_bias_add_epilogue_f32_1");
	EXPECT_EQ(payload.kernels[0].arguments[1].kind, CUDANativeArgumentKind::InputTensor);
	EXPECT_EQ(payload.kernels[0].arguments[2].kind, CUDANativeArgumentKind::ConstantTensor);
	EXPECT_EQ(payload.kernels[2].arguments[0].kind, CUDANativeArgumentKind::OutputTensor);
#else
	GTEST_SKIP() << "CUDA driver support is not enabled";
#endif
}

TEST(CompiledModuleCUDATest, SeparatedNativeLinearChainMovesConstantsOutOfInstructions)
{
#ifdef LITENN_ENABLE_CUDA_DRIVER
	auto graph = BuildTinyMLPGraph(2);
	FusionPass{}.Run(graph);
	auto artifact = Compiler<CUDA>::CompileArtifact(Detail::BuildExecutablePlanFromGraph(graph));
	ASSERT_EQ(artifact.Backend(), CompiledModuleBackend::CUDANative) << Debug::DumpGraph(graph);
	const auto inlinePayload = DeserializeCUDANativeInstructionPayload(artifact.Instructions());
	ASSERT_GT(inlinePayload.constantData.size(), 0u);

	auto separated = artifact.SeparateRodata();
	EXPECT_GT(separated.Metadata().size(), 0u);
	EXPECT_GT(separated.Constants().size(), 0u);
	EXPECT_EQ(separated.Weights().size(), 0u);
	const auto separatedPayload = DeserializeCUDANativeInstructionPayload(separated.Instructions());
	EXPECT_EQ(separatedPayload.constantData.size(), 0u);

	auto copied = CompiledModuleSeparatedArtifact::CopyFromImage(separated.Image());
	EXPECT_EQ(copied.Backend(), CompiledModuleBackend::CUDANative);
	EXPECT_EQ(copied.Constants().size(), inlinePayload.constantData.size());

	if (!IsCUDADeviceAvailable())
	{
		GTEST_SKIP() << "CUDA device is not available";
	}
	if (!IsCUDADriverAvailable())
	{
		GTEST_SKIP() << "CUDA driver is not available";
	}

	std::vector<TensorInputSpec> inputSpecs = { TensorInputSpec{ .values = { 1.0f, -2.0f, 0.5f, -1.0f, 0.25f, 2.0f },
		                                                         .shape = { 2, 3 } } };
	Runtime::Interpreter<CPU> interpreter;
	const auto expected =
	    interpreter.RunForward(Detail::BuildExecutablePlanFromGraph(graph), MakeCPUInputs(inputSpecs));
	auto module = copied.Load(CUDA{});
	auto outputs = module.RunTensors(MakeCUDAInputs(inputSpecs));
	ASSERT_EQ(outputs.size(), expected.size());
	ExpectTensorNear(outputs[0].CopyToDevice(CPU{}), expected[0], 1e-4f);
#else
	GTEST_SKIP() << "CUDA driver support is not enabled";
#endif
}

TEST(CompiledModuleCUDATest, RunsNativeP3OpsWithCUDATensors)
{
	if (!IsCUDADeviceAvailable())
	{
		GTEST_SKIP() << "CUDA device is not available";
	}
	if (!IsCUDADriverAvailable())
	{
		GTEST_SKIP() << "CUDA driver is not available";
	}

	struct Case
	{
		std::string_view name;
		Graph graph;
		std::vector<TensorInputSpec> inputs;
		bool runCPUAOT{ true };
		bool runFusionPass{};
		float tolerance{ 1e-6f };
	};

	std::vector<Case> cases;
	cases.push_back(Case{
	    .name = "reduce_sum_axis1",
	    .graph = BuildReduceGraph(ReduceOp::Sum, 1, { 2 }, "sum_axis1"),
	    .inputs = { TensorInputSpec{ .values = { 1.0, 2.0, 3.0, 4.0, 5.0, 6.0 }, .shape = { 2, 3 } } },
	});
	cases.push_back(Case{
	    .name = "reduce_mean_axis0",
	    .graph = BuildReduceGraph(ReduceOp::Mean, 0, { 3 }, "mean_axis0"),
	    .inputs = { TensorInputSpec{ .values = { 1.0, 2.0, 3.0, 4.0, 5.0, 6.0 }, .shape = { 2, 3 } } },
	});
	cases.push_back(Case{
	    .name = "reduce_max_axis1",
	    .graph = BuildReduceGraph(ReduceOp::Max, 1, { 2 }, "max_axis1"),
	    .inputs = { TensorInputSpec{ .values = { 1.0, 7.0, 3.0, 4.0, 5.0, 6.0 }, .shape = { 2, 3 } } },
	});
	cases.push_back(Case{
	    .name = "reduce_min_axis1",
	    .graph = BuildReduceGraph(ReduceOp::Min, 1, { 2 }, "min_axis1"),
	    .inputs = { TensorInputSpec{ .values = { 1.0, 7.0, 3.0, 4.0, 5.0, 6.0 }, .shape = { 2, 3 } } },
	});
	cases.push_back(Case{
	    .name = "concat_axis0",
	    .graph = BuildConcatGraph({ 1, 3 }, { 2, 3 }, { 3, 3 }, 0, "concat_axis0"),
	    .inputs = { TensorInputSpec{ .values = { 1.0, 2.0, 3.0 }, .shape = { 1, 3 } },
	                TensorInputSpec{ .values = { 4.0, 5.0, 6.0, 7.0, 8.0, 9.0 }, .shape = { 2, 3 } } },
	    .runCPUAOT = false,
	});
	cases.push_back(Case{
	    .name = "concat_axis1",
	    .graph = BuildConcatGraph({ 2, 3 }, { 2, 2 }, { 2, 5 }, 1, "concat_axis1"),
	    .inputs = { TensorInputSpec{ .values = { 1.0, 2.0, 3.0, 4.0, 5.0, 6.0 }, .shape = { 2, 3 } },
	                TensorInputSpec{ .values = { 10.0, 20.0, 30.0, 40.0 }, .shape = { 2, 2 } } },
	    .runCPUAOT = false,
	});
	cases.push_back(Case{
	    .name = "slice_axis0",
	    .graph = BuildSliceGraph(0, 1, 1, { 1, 5 }, "slice_axis0"),
	    .inputs = { TensorInputSpec{ .values = { 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0 },
	                                 .shape = { 2, 5 } } },
	    .runCPUAOT = false,
	});
	cases.push_back(Case{
	    .name = "slice_axis1",
	    .graph = BuildSliceGraph(1, 1, 3, { 2, 3 }, "slice_axis1"),
	    .inputs = { TensorInputSpec{ .values = { 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0 },
	                                 .shape = { 2, 5 } } },
	    .runCPUAOT = false,
	});
	cases.push_back(Case{
	    .name = "matmul_bias_add",
	    .graph = BuildMatMulBiasGraph(false),
	    .inputs = { TensorInputSpec{ .values = { 1.0, -2.0, 3.0, -4.0, 5.0, -6.0 }, .shape = { 2, 3 } },
	                TensorInputSpec{ .values = { 1.0, 2.0, -3.0, 4.0, 5.0, -6.0 }, .shape = { 3, 2 } },
	                TensorInputSpec{ .values = { 10.0, -10.0 }, .shape = { 1, 2 } } },
	    .runFusionPass = true,
	});
	cases.push_back(Case{
	    .name = "matmul_bias_relu",
	    .graph = BuildMatMulBiasGraph(true),
	    .inputs = { TensorInputSpec{ .values = { 1.0, -2.0, 3.0, -4.0, 5.0, -6.0 }, .shape = { 2, 3 } },
	                TensorInputSpec{ .values = { 1.0, 2.0, -3.0, 4.0, 5.0, -6.0 }, .shape = { 3, 2 } },
	                TensorInputSpec{ .values = { 10.0, -10.0 }, .shape = { 1, 2 } } },
	    .runFusionPass = true,
	});

	for (const auto& testCase : cases)
	{
		SCOPED_TRACE(testCase.name);
		Runtime::Interpreter<CPU> interpreter;
		auto expectedInputs = MakeCPUInputs(testCase.inputs);
		const auto expected =
		    interpreter.RunForward(Detail::BuildExecutablePlanFromGraph(testCase.graph), expectedInputs);

		if (testCase.runCPUAOT)
		{
			auto cpuAOTInputs = MakeCPUInputs(testCase.inputs);
			auto cpuAOTOutputs = Compiler<CPU>::CompileArtifact(Detail::BuildExecutablePlanFromGraph(testCase.graph))
			                         .Load()
			                         .RunTensors(cpuAOTInputs);
			ExpectOutputsNear(cpuAOTOutputs, expected, testCase.tolerance);
		}

		auto cudaGraph = testCase.graph;
		if (testCase.runFusionPass)
		{
			FusionPass{}.Run(cudaGraph);
		}
		auto artifact = Compiler<CUDA>::CompileArtifact(Detail::BuildExecutablePlanFromGraph(cudaGraph));
		ASSERT_EQ(artifact.Backend(), CompiledModuleBackend::CUDANative);
		auto module = artifact.Load(CUDA{});
		ASSERT_EQ(module.Backend(), CompiledModuleBackend::CUDANative);

		auto cudaInputs = MakeCUDAInputs(testCase.inputs);
		auto cudaOutputs = module.RunTensors(cudaInputs);
		std::vector<Tensor<CPU>> cudaCPUOutputs;
		cudaCPUOutputs.reserve(cudaOutputs.size());
		for (const auto& output : cudaOutputs)
		{
			cudaCPUOutputs.push_back(output.CopyToDevice(CPU{}));
		}
		ExpectOutputsNear(cudaCPUOutputs, expected, testCase.tolerance);
	}
}

TEST(CompiledModuleCUDATest, RunsNativeLinearChainWithConstantsAndWorkspace)
{
	if (!IsCUDADeviceAvailable())
	{
		GTEST_SKIP() << "CUDA device is not available";
	}
	if (!IsCUDADriverAvailable())
	{
		GTEST_SKIP() << "CUDA driver is not available";
	}

	auto graph = BuildTinyMLPGraph(2);
	std::vector<TensorInputSpec> inputSpecs = { TensorInputSpec{ .values = { 1.0f, -2.0f, 0.5f, -1.0f, 0.25f, 2.0f },
		                                                         .shape = { 2, 3 } } };
	auto expectedInputs = MakeCPUInputs(inputSpecs);
	Runtime::Interpreter<CPU> interpreter;
	const auto expected = interpreter.RunForward(Detail::BuildExecutablePlanFromGraph(graph), expectedInputs);

	auto cudaGraph = graph;
	FusionPass{}.Run(cudaGraph);
	auto artifact = Compiler<CUDA>::CompileArtifact(Detail::BuildExecutablePlanFromGraph(cudaGraph));
	ASSERT_EQ(artifact.Backend(), CompiledModuleBackend::CUDANative) << Debug::DumpGraph(cudaGraph);
	auto module = artifact.Load(CUDA{});
	ASSERT_EQ(module.Backend(), CompiledModuleBackend::CUDANative);

	auto cudaInputs = MakeCUDAInputs(inputSpecs);
	auto cudaOutputs = module.RunTensors(cudaInputs);
	std::vector<Tensor<CPU>> cudaCPUOutputs;
	for (const auto& output : cudaOutputs)
	{
		cudaCPUOutputs.push_back(output.CopyToDevice(CPU{}));
	}
	ExpectOutputsNear(cudaCPUOutputs, expected, 1e-4f);
}

TEST(CompiledModuleCUDATest, RunsNativeLowPrecisionLinearChainWithConstantsAndWorkspace)
{
	if (!IsCUDADeviceAvailable())
	{
		GTEST_SKIP() << "CUDA device is not available";
	}
	if (!IsCUDADriverAvailable())
	{
		GTEST_SKIP() << "CUDA driver is not available";
	}
	if (!CUDASupportsNativeMatMul(DataType::Float16))
	{
		GTEST_SKIP() << "CUDA device does not report native float16 MatMul support";
	}

	auto graph = BuildTinyMLPGraph(2, DataType::Float16);
	std::vector<TensorInputSpec> inputSpecs = { TensorInputSpec{
		.values = { 1.0f, -2.0f, 0.5f, -1.0f, 0.25f, 2.0f }, .shape = { 2, 3 }, .dtype = DataType::Float16 } };
	auto expectedInputs = MakeCPUInputs(inputSpecs);
	Runtime::Interpreter<CPU> interpreter;
	const auto expected = interpreter.RunForward(Detail::BuildExecutablePlanFromGraph(graph), expectedInputs);

	auto cudaGraph = graph;
	FusionPass{}.Run(cudaGraph);
	auto artifact = Compiler<CUDA>::CompileArtifact(Detail::BuildExecutablePlanFromGraph(cudaGraph));
	ASSERT_EQ(artifact.Backend(), CompiledModuleBackend::CUDANative) << Debug::DumpGraph(cudaGraph);
	auto module = artifact.Load(CUDA{});
	ASSERT_EQ(module.Backend(), CompiledModuleBackend::CUDANative);

	auto cudaInputs = MakeCUDAInputs(inputSpecs);
	auto cudaOutputs = module.RunTensors(cudaInputs);
	std::vector<Tensor<CPU>> cudaCPUOutputs;
	for (const auto& output : cudaOutputs)
	{
		cudaCPUOutputs.push_back(output.CopyToDevice(CPU{}));
	}
	ExpectOutputsNear(cudaCPUOutputs, expected, 1e-3f);
}

TEST(CompiledModuleCUDATest, RunsNativeLinearChainWithCUDAGraphReplay)
{
	if (!IsCUDADeviceAvailable())
	{
		GTEST_SKIP() << "CUDA device is not available";
	}
	if (!IsCUDADriverAvailable())
	{
		GTEST_SKIP() << "CUDA driver is not available";
	}

	auto graph = BuildTinyMLPGraph(2);
	std::vector<TensorInputSpec> inputSpecs = { TensorInputSpec{ .values = { 1.0f, -2.0f, 0.5f, -1.0f, 0.25f, 2.0f },
		                                                         .shape = { 2, 3 } } };
	auto expectedInputs = MakeCPUInputs(inputSpecs);
	Runtime::Interpreter<CPU> interpreter;
	const auto expected = interpreter.RunForward(Detail::BuildExecutablePlanFromGraph(graph), expectedInputs);

	auto cudaGraph = graph;
	FusionPass{}.Run(cudaGraph);
	auto module = Compiler<CUDA>::Compile(Detail::BuildExecutablePlanFromGraph(cudaGraph), CUDA{});
	ASSERT_EQ(module.Backend(), CompiledModuleBackend::CUDANative);

	auto cudaInputs = MakeCUDAInputs(inputSpecs);
	std::vector<Tensor<CUDA>> cudaOutputs;
	for (const auto& spec : module.OutputSpecs())
	{
		cudaOutputs.emplace_back(Uninitialized, ShapeView{ spec.type.StaticShape() }, spec.type.dtype, CUDA{});
	}

	module.RunTensorsInto(cudaInputs, cudaOutputs, CompiledModuleCUDARunOptions::GraphReplay());
	module.RunTensorsInto(cudaInputs, cudaOutputs, CompiledModuleCUDARunOptions::GraphReplay());

	std::vector<Tensor<CPU>> cudaCPUOutputs;
	for (const auto& output : cudaOutputs)
	{
		cudaCPUOutputs.push_back(output.CopyToDevice(CPU{}));
	}
	ExpectOutputsNear(cudaCPUOutputs, expected, 1e-4f);
}

TEST(CompiledModuleCUDATest, CUDAGraphReplayRejectsExternalStream)
{
	if (!IsCUDADeviceAvailable())
	{
		GTEST_SKIP() << "CUDA device is not available";
	}
	if (!IsCUDADriverAvailable())
	{
		GTEST_SKIP() << "CUDA driver is not available";
	}

	auto graph = BuildSimpleBinaryGraph(BinaryOp::Add, "sum");
	auto module = Compiler<CUDA>::Compile(Detail::BuildExecutablePlanFromGraph(graph), CUDA{});
	ASSERT_EQ(module.Backend(), CompiledModuleBackend::CUDANative);

	auto lhs = Tensor<CPU>({ 1, 2, 3, 4 }, { 2, 2 }, DataType::Float32).CopyToDevice(CUDA{});
	auto rhs = Tensor<CPU>({ 10, 20, 30, 40 }, { 2, 2 }, DataType::Float32).CopyToDevice(CUDA{});
	std::array<Tensor<CUDA>, 2> inputs = { std::move(lhs), std::move(rhs) };
	std::array<Tensor<CUDA>, 1> outputs = { Tensor<CUDA>(Uninitialized, { 2, 2 }, DataType::Float32, CUDA{}) };

	cudaStream_t stream{};
	ASSERT_EQ(cudaStreamCreate(&stream), cudaSuccess);
	try
	{
		module.RunTensorsInto(
		    inputs, outputs,
		    CompiledModuleCUDARunOptions{ .stream = stream, .graphReplay = CUDAGraphReplayMode::Enabled });
		FAIL() << "expected graph replay stream validation to throw";
	}
	catch (const std::runtime_error& ex)
	{
		const std::string message = ex.what();
		EXPECT_NE(message.find("CUDA graph replay"), std::string::npos);
		EXPECT_NE(message.find("default stream"), std::string::npos);
	}
	EXPECT_EQ(cudaStreamDestroy(stream), cudaSuccess);
}

TEST(CompiledModuleCUDATest, LoadsCUDANativeArtifactFromExportedSymbolAddresses)
{
	auto artifact = Compiler<CUDA>::CompileArtifact(Detail::BuildExecutablePlanFromGraph(BuildSimpleMatMulGraph()));
	ASSERT_EQ(artifact.Backend(), CompiledModuleBackend::CUDANative);

	const std::uint64_t rodataSize = artifact.Rodata().size();
	const std::uint64_t instructionSize = artifact.Instructions().size();
	auto exportedArtifact = CompiledModuleArtifact::FromExportedSymbols({
	    .rodata = artifact.Rodata().data(),
	    .rodataSize = &rodataSize,
	    .instructions = artifact.Instructions().data(),
	    .instructionSize = &instructionSize,
	});

	EXPECT_EQ(exportedArtifact.Backend(), CompiledModuleBackend::CUDANative);
	ASSERT_EQ(exportedArtifact.InputSpecs().size(), 2u);
	ASSERT_EQ(exportedArtifact.OutputSpecs().size(), 1u);
	EXPECT_EQ(exportedArtifact.InputSpecs()[0].name, "lhs");
	EXPECT_EQ(exportedArtifact.OutputSpecs()[0].name, "matmul");

	if (!IsCUDADeviceAvailable())
	{
		GTEST_SKIP() << "CUDA device is not available";
	}

	auto module = exportedArtifact.Load(CUDA{});
	EXPECT_EQ(module.Backend(), CompiledModuleBackend::CUDANative);
	auto lhs = Tensor<CPU>({ 1, 2, 3, 4 }, { 2, 2 }, DataType::Float32).CopyToDevice(CUDA{});
	auto rhs = Tensor<CPU>({ 10, 20, 30, 40 }, { 2, 2 }, DataType::Float32).CopyToDevice(CUDA{});
	std::array<Tensor<CUDA>, 2> inputs = { std::move(lhs), std::move(rhs) };
	auto outputs = module.RunTensors(inputs);
	ASSERT_EQ(outputs.size(), 1u);
	ExpectTensorNear(outputs[0].CopyToDevice(CPU{}), std::array{ 70.0f, 100.0f, 150.0f, 220.0f });
}

TEST(CompiledModuleCUDATest, RejectsInvalidExplicitCUDANativeTarget)
{
#ifdef LITENN_ENABLE_CUDA_DRIVER
	EXPECT_THROW((void) CUDANativeNVPTXTargetChip("compute_75"), std::runtime_error);
	EXPECT_EQ(CUDANativeNVPTXTargetChip("sm_75"), "sm_75");
#else
	GTEST_SKIP() << "CUDA driver support is not enabled";
#endif
}

TEST(CompiledModuleCUDATest, MatchesCPUInterpreterAndAOTAcrossNumericalMatrix)
{
	if (!IsCUDADeviceAvailable())
	{
		GTEST_SKIP() << "CUDA device is not available";
	}
	if (!IsCUDADriverAvailable())
	{
		GTEST_SKIP() << "CUDA driver is not available";
	}

	struct Case
	{
		std::string_view name;
		Graph graph;
		std::vector<TensorInputSpec> inputs;
		CompiledModuleBackend expectedCUDABackend{ CompiledModuleBackend::CUDANative };
		float tolerance{ 1e-6f };
	};

	std::vector<Case> cases;
	cases.reserve(22);

	const std::array unaryCases = {
		std::pair{ UnaryOp::Negate, std::string_view{ "unary_negate" } },
		std::pair{ UnaryOp::Abs, std::string_view{ "unary_abs" } },
		std::pair{ UnaryOp::Sqrt, std::string_view{ "unary_sqrt" } },
		std::pair{ UnaryOp::Exp, std::string_view{ "unary_exp" } },
		std::pair{ UnaryOp::Log, std::string_view{ "unary_log" } },
		std::pair{ UnaryOp::Sin, std::string_view{ "unary_sin" } },
		std::pair{ UnaryOp::Cos, std::string_view{ "unary_cos" } },
	};
	for (const auto& [op, name] : unaryCases)
	{
		cases.push_back(Case{
		    .name = name,
		    .graph = BuildSimpleUnaryGraph(op, std::string(name)),
		    .inputs = { TensorInputSpec{ .values = { 0.5, 1.0, 2.0, 4.0 }, .shape = { 2, 2 } } },
		    .tolerance = 2e-3f,
		});
	}

	const std::array binaryCases = {
		std::pair{ BinaryOp::Add, std::string_view{ "binary_add" } },
		std::pair{ BinaryOp::Subtract, std::string_view{ "binary_subtract" } },
		std::pair{ BinaryOp::Multiply, std::string_view{ "binary_multiply" } },
		std::pair{ BinaryOp::Divide, std::string_view{ "binary_divide" } },
		std::pair{ BinaryOp::Max, std::string_view{ "binary_max" } },
		std::pair{ BinaryOp::Min, std::string_view{ "binary_min" } },
	};
	for (const auto& [op, name] : binaryCases)
	{
		cases.push_back(Case{
		    .name = name,
		    .graph = BuildSimpleBinaryGraph(op, std::string(name)),
		    .inputs = { TensorInputSpec{ .values = { 1.0, 2.0, 3.0, 4.0 }, .shape = { 2, 2 } },
		                TensorInputSpec{ .values = { 4.0, 3.0, 2.0, 1.0 }, .shape = { 2, 2 } } },
		});
	}

	const std::array outputShape{ 2uz, 3uz };
	cases.push_back(Case{
	    .name = "broadcast_add",
	    .graph = BuildBinaryGraph(BinaryOp::Add, std::array{ 2uz, 3uz }, std::array{ 1uz, 3uz }, outputShape,
	                              "broadcast_add"),
	    .inputs = { TensorInputSpec{ .values = { 1.0, 2.0, 3.0, 4.0, 5.0, 6.0 }, .shape = { 2, 3 } },
	                TensorInputSpec{ .values = { 10.0, 20.0, 30.0 }, .shape = { 1, 3 } } },
	});
	cases.push_back(Case{
	    .name = "broadcast_subtract",
	    .graph = BuildBinaryGraph(BinaryOp::Subtract, std::array{ 2uz, 3uz }, std::array{ 2uz, 1uz }, outputShape,
	                              "broadcast_subtract"),
	    .inputs = { TensorInputSpec{ .values = { 1.0, 2.0, 3.0, 4.0, 5.0, 6.0 }, .shape = { 2, 3 } },
	                TensorInputSpec{ .values = { 10.0, 20.0 }, .shape = { 2, 1 } } },
	});
	cases.push_back(Case{
	    .name = "broadcast_multiply",
	    .graph = BuildBinaryGraph(BinaryOp::Multiply, std::array{ 1uz, 3uz }, std::array{ 2uz, 3uz }, outputShape,
	                              "broadcast_multiply"),
	    .inputs = { TensorInputSpec{ .values = { 2.0, 3.0, 4.0 }, .shape = { 1, 3 } },
	                TensorInputSpec{ .values = { 1.0, 2.0, 3.0, 4.0, 5.0, 6.0 }, .shape = { 2, 3 } } },
	});
	cases.push_back(Case{
	    .name = "broadcast_divide",
	    .graph = BuildBinaryGraph(BinaryOp::Divide, std::array{ 2uz, 3uz }, std::array{ 1uz, 3uz }, outputShape,
	                              "broadcast_divide"),
	    .inputs = { TensorInputSpec{ .values = { 8.0, 18.0, 32.0, 20.0, 45.0, 80.0 }, .shape = { 2, 3 } },
	                TensorInputSpec{ .values = { 2.0, 3.0, 4.0 }, .shape = { 1, 3 } } },
	});
	cases.push_back(Case{
	    .name = "broadcast_max",
	    .graph = BuildBinaryGraph(BinaryOp::Max, std::array{ 2uz, 3uz }, std::array{ 1uz, 3uz }, outputShape,
	                              "broadcast_max"),
	    .inputs = { TensorInputSpec{ .values = { 1.0, 25.0, 3.0, 40.0, 5.0, 60.0 }, .shape = { 2, 3 } },
	                TensorInputSpec{ .values = { 10.0, 20.0, 30.0 }, .shape = { 1, 3 } } },
	});
	cases.push_back(Case{
	    .name = "broadcast_min",
	    .graph = BuildBinaryGraph(BinaryOp::Min, std::array{ 2uz, 1uz }, std::array{ 2uz, 3uz }, outputShape,
	                              "broadcast_min"),
	    .inputs = { TensorInputSpec{ .values = { 7.0, 18.0 }, .shape = { 2, 1 } },
	                TensorInputSpec{ .values = { 1.0, 9.0, 8.0, 20.0, 2.0, 30.0 }, .shape = { 2, 3 } } },
	});
	cases.push_back(Case{
	    .name = "matmul_cublas",
	    .graph = BuildSimpleMatMulGraph(),
	    .inputs = { TensorInputSpec{ .values = { 1.0, 2.0, 3.0, 4.0 }, .shape = { 2, 2 } },
	                TensorInputSpec{ .values = { 10.0, 20.0, 30.0, 40.0 }, .shape = { 2, 2 } } },
	});
	cases.push_back(Case{
	    .name = "pow_bridge_fallback",
	    .graph = BuildSimplePowGraph(),
	    .inputs = { TensorInputSpec{ .values = { 2.0, 3.0, 4.0, 5.0 }, .shape = { 2, 2 } },
	                TensorInputSpec{ .values = { 3.0, 2.0, 1.0, 0.0 }, .shape = { 2, 2 } } },
	    .expectedCUDABackend = CompiledModuleBackend::CPUNative,
	});

	for (const auto& testCase : cases)
	{
		SCOPED_TRACE(testCase.name);

		auto interpreterInputs = MakeCPUInputs(testCase.inputs);
		Runtime::Interpreter<CPU> interpreter;
		const auto expected =
		    interpreter.RunForward(Detail::BuildExecutablePlanFromGraph(testCase.graph), interpreterInputs);

		auto cpuAOTInputs = MakeCPUInputs(testCase.inputs);
		auto cpuAOTModule = Compiler<CPU>::CompileArtifact(Detail::BuildExecutablePlanFromGraph(testCase.graph)).Load();
		auto cpuAOTOutputs = cpuAOTModule.RunTensors(cpuAOTInputs);
		ExpectOutputsNear(cpuAOTOutputs, expected, testCase.tolerance);

		auto cudaArtifact = Compiler<CUDA>::CompileArtifact(Detail::BuildExecutablePlanFromGraph(testCase.graph));
		EXPECT_EQ(cudaArtifact.Backend(), testCase.expectedCUDABackend);
		const auto cudaDevice =
		    testCase.expectedCUDABackend == CompiledModuleBackend::CPUNative ? CUDAWithHostFallback() : CUDA{};
		auto cudaModule = cudaArtifact.Load(cudaDevice);
		EXPECT_EQ(cudaModule.Backend(), testCase.expectedCUDABackend);
		auto cudaInputs = MakeCUDAInputs(testCase.inputs);
		auto cudaOutputs = cudaModule.RunTensors(cudaInputs);

		std::vector<Tensor<CPU>> cudaCPUOutputs;
		cudaCPUOutputs.reserve(cudaOutputs.size());
		for (const auto& output : cudaOutputs)
		{
			cudaCPUOutputs.push_back(output.CopyToDevice(CPU{}));
		}
		ExpectOutputsNear(cudaCPUOutputs, expected, testCase.tolerance);
	}
}
