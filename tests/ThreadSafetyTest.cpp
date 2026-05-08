#include <gtest/gtest.h>

#include <LiteNN.h>
#include <LiteNN/Pass/AutogradPass.h>
#include <LiteNN/Runtime/Interpreter.h>

#include <cmath>
#include <format>
#include <future>
#include <string>
#include <vector>

using namespace LiteNN;

namespace
{
	struct WorkerResult
	{
		bool ok{};
		std::string message;
	};

	float ReadFloat(const Tensor<CPU>& tensor, std::size_t index)
	{
		const auto cpuTensor = tensor.CopyToDevice(CPU{});
		return static_cast<const float*>(cpuTensor.RawData())[index];
	}

	bool NearlyEqual(float lhs, float rhs)
	{
		return std::fabs(lhs - rhs) < 1e-5f;
	}

	Graph BuildSquareGraphWithBackward()
	{
		Graph graph;
		Subgraph sg;
		const auto x = sg.AddParam(DataType::Float32, { 1 });
		const auto y = sg.AddNode(BinaryOpNode{ BinaryOp::Multiply, { x, 0 }, { x, 0 } },
		                          { OutputInfo{ DataType::Float32, { 1 } } });
		sg.SetResults({ { y, 0 } });
		graph.SetForward(graph.AddSubgraph(std::move(sg)));

		AutogradPass autograd;
		autograd.Run(graph);
		return graph;
	}

	WorkerResult RunInterpreterWorker(const Graph& graph, int workerId)
	{
		try
		{
			Runtime::Interpreter<CPU> interpreter;
			for (int iteration = 0; iteration < 32; ++iteration)
			{
				const auto x = static_cast<float>(workerId * 32 + iteration + 1);
				const auto dy = static_cast<float>(iteration % 3 + 1);

				std::vector<Tensor<CPU>> fwdInputs;
				fwdInputs.emplace_back(Tensor<CPU>({ x }, { 1 }));
				auto forward = interpreter.RunForward(graph, fwdInputs);
				if (forward.size() != 1 || !NearlyEqual(ReadFloat(forward[0], 0), x * x))
				{
					return { false, std::format("worker {} forward mismatch at iteration {}", workerId, iteration) };
				}

				std::vector<Tensor<CPU>> bwdInputs;
				bwdInputs.emplace_back(Tensor<CPU>({ x }, { 1 }));
				bwdInputs.emplace_back(Tensor<CPU>({ dy }, { 1 }));
				auto gradients = interpreter.RunBackward(graph, bwdInputs);
				if (gradients.size() != 1 || !NearlyEqual(ReadFloat(gradients[0], 0), 2.0f * x * dy))
				{
					return { false, std::format("worker {} backward mismatch at iteration {}", workerId, iteration) };
				}
			}
		}
		catch (const std::exception& ex)
		{
			return { false, std::format("worker {} threw: {}", workerId, ex.what()) };
		}

		return { true, {} };
	}
} // namespace

TEST(ThreadSafety, ReadOnlyGraphCanRunConcurrentlyWithSeparateInterpreters)
{
	auto graph = BuildSquareGraphWithBackward();

	std::vector<std::future<WorkerResult>> futures;
	for (int workerId = 0; workerId < 4; ++workerId)
	{
		futures.push_back(std::async(std::launch::async, [&graph, workerId] {
			return RunInterpreterWorker(graph, workerId);
		}));
	}

	for (auto& future : futures)
	{
		auto result = future.get();
		EXPECT_TRUE(result.ok) << result.message;
	}
}
