#include <gtest/gtest.h>

#include <LiteNN.h>
#include <LiteNN/Debug/Dump.h>
#include <LiteNN/Optimizer/GraphOps.h>
#include <LiteNN/Runtime/Interpreter.h>
#include <LiteNN/Serialization/ModelIO.h>
#include <LiteNN/Validation/GraphValidator.h>

#include <filesystem>
#include <initializer_list>
#include <optional>
#include <ranges>
#include <vector>

using namespace LiteNN;

namespace
{
	float ReadFloat(const Tensor<CPU>& tensor, std::size_t index)
	{
		return static_cast<const float*>(tensor.RawData())[index];
	}

	void ExpectShape(ShapeView shape, std::initializer_list<std::size_t> expected)
	{
		EXPECT_TRUE(std::ranges::equal(shape.Dims, expected));
	}

	void ExpectTensorNear(const Tensor<CPU>& tensor, std::initializer_list<float> expected, float tolerance = 1e-5F)
	{
		ASSERT_EQ(tensor.NumElements(), expected.size());
		std::size_t index = 0;
		for (const auto value : expected)
		{
			EXPECT_NEAR(ReadFloat(tensor, index), value, tolerance) << "index=" << index;
			++index;
		}
	}

	std::vector<Tensor<CPU>> RunGraph(Graph& graph, std::vector<Tensor<CPU>> inputs)
	{
		Runtime::Interpreter<CPU> interpreter;
		return interpreter.RunForward(graph, inputs);
	}

	Graph BuildOptimizerStepGraph()
	{
		Graph graph;
		Subgraph subgraph;
		const auto parameter = subgraph.AddParam(DataType::Float32, { 2 });
		const auto gradient = subgraph.AddParam(DataType::Float32, { 2 });
		const auto velocity = subgraph.AddParam(DataType::Float32, { 2 });
		const auto firstMoment = subgraph.AddParam(DataType::Float32, { 2 });
		const auto secondMoment = subgraph.AddParam(DataType::Float32, { 2 });

		const auto sgd = Optimizer::AddSGDStep(subgraph, { parameter, 0 }, { gradient, 0 },
		                                       std::nullopt, 0.5, 0.0, 0.1);
		const auto sgdMomentum = Optimizer::AddSGDStep(subgraph, { parameter, 0 }, { gradient, 0 },
		                                               NodeOutput{ velocity, 0 }, 0.1, 0.9);
		const auto adamw = Optimizer::AddAdamWStep(subgraph, { parameter, 0 }, { gradient, 0 },
		                                           { firstMoment, 0 }, { secondMoment, 0 },
		                                           0.1, 0.9, 0.999, 1e-8, 0.01, 1);

		subgraph.SetResults({ sgd[0], sgdMomentum[0], sgdMomentum[1], adamw[0], adamw[1], adamw[2] });
		graph.SetForward(graph.AddSubgraph(std::move(subgraph)));
		return graph;
	}

	std::vector<Tensor<CPU>> MakeOptimizerStepInputs()
	{
		std::vector<Tensor<CPU>> inputs;
		inputs.emplace_back(Tensor<CPU>({ 1.0F, -2.0F }, { 2 }));
		inputs.emplace_back(Tensor<CPU>({ 0.1F, -0.2F }, { 2 }));
		inputs.emplace_back(Tensor<CPU>({ 0.5F, -0.5F }, { 2 }));
		inputs.emplace_back(Tensor<CPU>({ 0.0F, 0.0F }, { 2 }));
		inputs.emplace_back(Tensor<CPU>({ 0.0F, 0.0F }, { 2 }));
		return inputs;
	}
} // namespace

TEST(OptimizerGraphOp, ExecutesSGDAndAdamWSteps)
{
	auto graph = BuildOptimizerStepGraph();
	Validation::ValidateGraph(graph);

	const auto outputs = RunGraph(graph, MakeOptimizerStepInputs());
	ASSERT_EQ(outputs.size(), 6u);
	for (const auto& output : outputs)
	{
		ExpectShape(output.Shape(), { 2 });
	}
	ExpectTensorNear(outputs[0], { 0.9F, -1.8F });
	ExpectTensorNear(outputs[1], { 0.945F, -1.935F });
	ExpectTensorNear(outputs[2], { 0.55F, -0.65F });
	ExpectTensorNear(outputs[3], { 0.899F, -1.898F }, 1e-4F);
	ExpectTensorNear(outputs[4], { 0.01F, -0.02F });
	ExpectTensorNear(outputs[5], { 0.00001F, 0.00004F });
}

TEST(OptimizerGraphOp, SerializationRoundTripPreservesOptimizerStepNodes)
{
	auto graph = BuildOptimizerStepGraph();
	Validation::ValidateGraph(graph);

	const auto path = std::filesystem::path("litenn_optimizer_step_nodes_roundtrip_test.ltnn");
	std::filesystem::remove(path);
	Serialization::SaveModel(graph, path);
	auto loaded = Serialization::LoadModel(path);
	std::filesystem::remove(path);

	auto expected = RunGraph(graph, MakeOptimizerStepInputs());
	auto actual = RunGraph(loaded, MakeOptimizerStepInputs());
	ASSERT_EQ(actual.size(), expected.size());
	for (auto output = 0uz; output < actual.size(); ++output)
	{
		ASSERT_EQ(actual[output].NumElements(), expected[output].NumElements());
		for (auto index = 0uz; index < actual[output].NumElements(); ++index)
		{
			EXPECT_NEAR(ReadFloat(actual[output], index), ReadFloat(expected[output], index), 1e-5F)
			    << "output=" << output << ", index=" << index;
		}
	}
}

TEST(OptimizerGraphOp, DumpIncludesOptimizerStepNodeKinds)
{
	auto graph = BuildOptimizerStepGraph();
	const auto dump = Debug::DumpGraph(graph);
	EXPECT_NE(dump.find("SGDStepNode"), std::string::npos);
	EXPECT_NE(dump.find("AdamWStepNode"), std::string::npos);
}
