#include <gtest/gtest.h>

#include <LiteNN.h>
#include <LiteNN/Pass/AutogradPass.h>
#include <LiteNN/Pass/ForwardOnlyPass.h>
#include <LiteNN/Runtime/Interpreter.h>

#include <cmath>
#include <variant>
#include <vector>

using namespace LiteNN;

namespace
{
	float ReadFloat(const Tensor<CPU>& tensor, std::size_t index)
	{
		return static_cast<const float*>(tensor.RawData())[index];
	}

	bool HasActivationNode(const Graph& graph)
	{
		for (SubgraphId subgraphId = 0; subgraphId < graph.SubgraphCount(); ++subgraphId)
		{
			for (const auto& entry : graph.GetSubgraph(subgraphId).Nodes())
			{
				if (std::holds_alternative<SaveActivationNode>(entry.node) ||
				    std::holds_alternative<TapeSaveActivationNode>(entry.node) ||
				    std::holds_alternative<LoadActivationNode>(entry.node) ||
				    std::holds_alternative<TapeLoadActivationNode>(entry.node))
				{
					return true;
				}
			}
		}
		return false;
	}
} // namespace

TEST(ForwardOnlyPass, ExtractsInferenceGraphFromAutogradGraph)
{
	Graph graph;
	const auto weightIndex = graph.AddVariable(Variable::Create(Tensor<CPU>({ 3.0f }, { 1 })));

	Subgraph sg;
	const auto x = sg.AddParam(DataType::Float32, { 1 });
	const auto weight = sg.AddNode(VariableRefNode{ weightIndex }, { OutputInfo{ DataType::Float32, { 1 } } });
	const auto product = sg.AddNode(BinaryOpNode{ BinaryOp::Multiply, { x, 0 }, { weight, 0 } },
	                                { OutputInfo{ DataType::Float32, { 1 } } });
	const auto y = sg.AddNode(UnaryOpNode{ UnaryOp::Sqrt, { product, 0 } },
	                          { OutputInfo{ DataType::Float32, { 1 } } });
	sg.SetResults({ { y, 0 } });
	graph.SetForward(graph.AddSubgraph(std::move(sg)));
	graph.SetInputNames({ "x" });
	graph.SetOutputNames({ "y" });

	AutogradPass autograd;
	autograd.Run(graph);
	ASSERT_TRUE(graph.Backward().has_value());
	ASSERT_TRUE(HasActivationNode(graph));

	Runtime::Interpreter<CPU> interpreter;
	std::vector<Tensor<CPU>> inputs;
	inputs.emplace_back(Tensor<CPU>({ 2.0f }, { 1 }));
	auto trainingForward = interpreter.RunForward(graph, inputs);

	auto forwardOnly = ExtractForwardOnlyGraph(graph);
	EXPECT_FALSE(forwardOnly.Backward().has_value());
	EXPECT_EQ(forwardOnly.ActivationSlotCount(), 0u);
	EXPECT_EQ(forwardOnly.TapeSlotCount(), 0u);
	EXPECT_FALSE(HasActivationNode(forwardOnly));
	ASSERT_EQ(forwardOnly.InputSignature().size(), 1u);
	ASSERT_EQ(forwardOnly.OutputSignature().size(), 1u);
	EXPECT_EQ(forwardOnly.InputSignature()[0].name, "x");
	EXPECT_EQ(forwardOnly.OutputSignature()[0].name, "y");

	Runtime::Interpreter<CPU> forwardOnlyInterpreter;
	auto inferenceForward = forwardOnlyInterpreter.RunForward(forwardOnly, inputs);
	ASSERT_EQ(trainingForward.size(), 1u);
	ASSERT_EQ(inferenceForward.size(), 1u);
	EXPECT_FLOAT_EQ(ReadFloat(inferenceForward[0], 0), ReadFloat(trainingForward[0], 0));
	EXPECT_FLOAT_EQ(ReadFloat(inferenceForward[0], 0), std::sqrt(6.0f));
}
