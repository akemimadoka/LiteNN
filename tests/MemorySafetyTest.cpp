#include <gtest/gtest.h>

#include <LiteNN.h>

#include <array>
#include <cstddef>
#include <vector>

#if defined(LITENN_ENABLE_MLIR)
#include <LiteNN/Compiler/CompiledModule.h>
#endif

#if defined(_MSC_VER) && defined(_DEBUG)
#include <crtdbg.h>
#endif

using namespace LiteNN;

namespace
{
	constexpr int kTensorStressIterations = 256;
	constexpr int kCompiledModuleStressIterations = 64;

	float ReadFloat(const Tensor<CPU>& tensor, std::size_t index)
	{
		return static_cast<const float*>(tensor.RawData())[index];
	}

#if defined(_MSC_VER) && defined(_DEBUG)
	class DebugHeapScope
	{
	public:
		DebugHeapScope()
		{
			_CrtMemCheckpoint(&before_);
		}

		testing::AssertionResult ExpectNoLeaks() const
		{
			_CrtMemState after{};
			_CrtMemState diff{};
			_CrtMemCheckpoint(&after);
			if (_CrtMemDifference(&diff, &before_, &after) != 0)
			{
				return testing::AssertionFailure()
				       << "detected net debug heap growth during memory-safety stress loop";
			}
			return testing::AssertionSuccess();
		}

	private:
		_CrtMemState before_{};
	};
#endif

#if defined(LITENN_ENABLE_MLIR)
	Graph BuildSimpleAddGraph()
	{
		Graph graph;
		Subgraph sg;
		const auto a = sg.AddParam(DataType::Float32, { 2, 2 });
		const auto b = sg.AddParam(DataType::Float32, { 2, 2 });
		const auto y = sg.AddNode(BinaryOpNode{ BinaryOp::Add, { a, 0 }, { b, 0 } },
		                          { OutputInfo{ DataType::Float32, { 2, 2 } } });
		sg.SetResults({ { y, 0 } });
		graph.AddSubgraph(std::move(sg));
		graph.SetForward(0);
		return graph;
	}
#endif
} // namespace

TEST(MemorySafetyTest, TensorViewsSurviveRepeatedAssignmentAndDestruction)
{
#if defined(_MSC_VER) && defined(_DEBUG)
	DebugHeapScope heapScope;
#endif

	for (int iteration = 0; iteration < kTensorStressIterations; ++iteration)
	{
		SCOPED_TRACE(iteration);
		Tensor<CPU> tensor({ 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12 }, { 4, 3 }, DataType::Float32);
		auto row = tensor[1];
		Tensor<CPU> replacement({ static_cast<double>(iteration + 1), static_cast<double>(iteration + 2),
		                         static_cast<double>(iteration + 3) },
		                        { 3 }, DataType::Float32);

		row = replacement;
		EXPECT_FLOAT_EQ(ReadFloat(tensor, 3), static_cast<float>(iteration + 1));
		EXPECT_FLOAT_EQ(ReadFloat(tensor, 4), static_cast<float>(iteration + 2));
		EXPECT_FLOAT_EQ(ReadFloat(tensor, 5), static_cast<float>(iteration + 3));

		auto detachedCopy = row;
		EXPECT_FLOAT_EQ(ReadFloat(detachedCopy, 0), static_cast<float>(iteration + 1));
		EXPECT_FLOAT_EQ(ReadFloat(detachedCopy, 1), static_cast<float>(iteration + 2));
		EXPECT_FLOAT_EQ(ReadFloat(detachedCopy, 2), static_cast<float>(iteration + 3));

		auto transposed = tensor.Transpose();
		EXPECT_EQ(transposed.NumElements(), tensor.NumElements());
	}

#if defined(_MSC_VER) && defined(_DEBUG)
	EXPECT_TRUE(heapScope.ExpectNoLeaks());
#endif
}

TEST(MemorySafetyTest, PolymorphicDeviceCopiesRoundTripInStressLoop)
{
	const Tensor<CPU> seed({ 1, 2, 3, 4 }, { 2, 2 }, DataType::Float32);

#if defined(_MSC_VER) && defined(_DEBUG)
	DebugHeapScope heapScope;
#endif

	for (int iteration = 0; iteration < kTensorStressIterations; ++iteration)
	{
		SCOPED_TRACE(iteration);
		PolymorphicDevice device{ CPU{} };
		PolymorphicDevice clone = device;

		EXPECT_TRUE(device.IsSameDevice(clone));
		EXPECT_TRUE(device.IsSameDeviceType(clone));

		auto polyTensor = seed.CopyToDevice(device);
		auto mirrored = polyTensor.CopyToDevice(clone);
		auto roundTrip = mirrored.CopyToDevice(CPU{});
		EXPECT_FLOAT_EQ(ReadFloat(roundTrip, 0), 1.0f);
		EXPECT_FLOAT_EQ(ReadFloat(roundTrip, 3), 4.0f);

		Tensor<PolymorphicDevice> workspace(polyTensor);
		auto workspaceCpu = workspace.CopyToDevice(CPU{});
		EXPECT_FLOAT_EQ(ReadFloat(workspaceCpu, 1), 2.0f);
		EXPECT_FLOAT_EQ(ReadFloat(workspaceCpu, 2), 3.0f);
	}

#if defined(_MSC_VER) && defined(_DEBUG)
	EXPECT_TRUE(heapScope.ExpectNoLeaks());
#endif
}

#if defined(LITENN_ENABLE_MLIR)
TEST(MemorySafetyTest, CompiledModuleLoadCopiesImageAcrossLifetimeCycles)
{
	// Linux/WSL LeakSanitizer currently reports an external 80-byte-per-load leak
	// in the LLVM MCJIT/RuntimeDyld loader used by CompiledModule::Load. This test
	// still validates LiteNN's image copy/lifetime behavior, but that specific
	// LSan report should not be interpreted as evidence that LiteNN retained the
	// copied rodata or instruction buffers.
	auto compiled = Compiler<CPU>::Compile(BuildSimpleAddGraph());

	for (int iteration = 0; iteration < kCompiledModuleStressIterations; ++iteration)
	{
		SCOPED_TRACE(iteration);
		CompiledModule<CPU> loaded;
		{
			std::vector<std::byte> rodata(compiled.Rodata().begin(), compiled.Rodata().end());
			std::vector<std::byte> instructions(compiled.Instructions().begin(), compiled.Instructions().end());
			loaded = CompiledModule<CPU>::Load({
			    .rodata = rodata.data(),
			    .rodataSize = rodata.size(),
			    .instructions = instructions.data(),
			    .instructionSize = instructions.size(),
			});
		}

		std::array<Tensor<CPU>, 2> inputs = {
		    Tensor<CPU>({ static_cast<double>(iteration + 1), static_cast<double>(iteration + 2),
		                  static_cast<double>(iteration + 3), static_cast<double>(iteration + 4) },
		                 { 2, 2 }, DataType::Float32),
		    Tensor<CPU>({ 10, 20, 30, 40 }, { 2, 2 }, DataType::Float32),
		};

		auto outputs = loaded.Run(inputs);
		ASSERT_EQ(outputs.size(), 1u);
		EXPECT_FLOAT_EQ(ReadFloat(outputs[0], 0), static_cast<float>(iteration + 11));
		EXPECT_FLOAT_EQ(ReadFloat(outputs[0], 1), static_cast<float>(iteration + 22));
		EXPECT_FLOAT_EQ(ReadFloat(outputs[0], 2), static_cast<float>(iteration + 33));
		EXPECT_FLOAT_EQ(ReadFloat(outputs[0], 3), static_cast<float>(iteration + 44));
	}
}
#endif