#include <gtest/gtest.h>

#include <LiteNN.h>
#include <LiteNN/Pass/AutogradPass.h>
#include <LiteNN/Runtime/Interpreter.h>

#include <vector>

using namespace LiteNN;

namespace
{
	float ReadFloat(const Tensor<CPU>& tensor, std::size_t index)
	{
		const auto cpuTensor = tensor.CopyToDevice(CPU{});
		return static_cast<const float*>(cpuTensor.RawData())[index];
	}

	std::vector<std::size_t> ShapeOf(const Tensor<CPU>& tensor)
	{
		return { tensor.Shape().Dims.begin(), tensor.Shape().Dims.end() };
	}
} // namespace

TEST(AutogradRegression, BroadcastAddReducesParamGradientToOriginalShape)
{
	Graph graph;
	Subgraph sg;

	const auto x = sg.AddParam(DataType::Float32, { 2, 3 });
	const auto bias = sg.AddParam(DataType::Float32, { 1, 3 });
	const auto y = sg.AddNode(BinaryOpNode{ BinaryOp::Add, { x, 0 }, { bias, 0 } },
	                          { OutputInfo{ DataType::Float32, { 2, 3 } } });
	sg.SetResults({ { y, 0 } });
	graph.SetForward(graph.AddSubgraph(std::move(sg)));

	AutogradPass autograd;
	autograd.Run(graph);

	Runtime::Interpreter<CPU> interpreter;
	std::vector<Tensor<CPU>> fwdInputs;
	fwdInputs.emplace_back(Tensor<CPU>({ 1, 2, 3, 4, 5, 6 }, { 2, 3 }));
	fwdInputs.emplace_back(Tensor<CPU>({ 10, 20, 30 }, { 1, 3 }));
	(void)interpreter.RunForward(graph, fwdInputs);

	std::vector<Tensor<CPU>> bwdInputs;
	bwdInputs.emplace_back(Tensor<CPU>({ 1, 2, 3, 4, 5, 6 }, { 2, 3 }));
	bwdInputs.emplace_back(Tensor<CPU>({ 10, 20, 30 }, { 1, 3 }));
	bwdInputs.emplace_back(Tensor<CPU>({ 1, 1, 1, 1, 1, 1 }, { 2, 3 }));

	auto gradients = interpreter.RunBackward(graph, bwdInputs);

	ASSERT_EQ(gradients.size(), 2);
	EXPECT_EQ(ShapeOf(gradients[0]), (std::vector<std::size_t>{ 2, 3 }));
	EXPECT_EQ(ShapeOf(gradients[1]), (std::vector<std::size_t>{ 1, 3 }));

	for (std::size_t i = 0; i < gradients[0].NumElements(); ++i)
	{
		EXPECT_FLOAT_EQ(ReadFloat(gradients[0], i), 1.0f);
	}
	EXPECT_FLOAT_EQ(ReadFloat(gradients[1], 0), 2.0f);
	EXPECT_FLOAT_EQ(ReadFloat(gradients[1], 1), 2.0f);
	EXPECT_FLOAT_EQ(ReadFloat(gradients[1], 2), 2.0f);
}

TEST(AutogradRegression, VariableGradientsAreReturnedAfterInputGradientsInVariableIndexOrder)
{
	Graph graph;
	auto w0 = Variable::Create(Tensor<CPU>({ 7 }, { 1 }));
	auto w1 = Variable::Create(Tensor<CPU>({ 5 }, { 1 }));
	const auto w0Idx = graph.AddVariable(std::move(w0));
	const auto w1Idx = graph.AddVariable(std::move(w1));

	Subgraph sg;
	const auto x = sg.AddParam(DataType::Float32, { 1 });
	const auto w1Ref = sg.AddNode(VariableRefNode{ w1Idx }, { OutputInfo{ DataType::Float32, { 1 } } });
	const auto term1 = sg.AddNode(BinaryOpNode{ BinaryOp::Multiply, { x, 0 }, { w1Ref, 0 } },
	                              { OutputInfo{ DataType::Float32, { 1 } } });
	const auto w0Ref = sg.AddNode(VariableRefNode{ w0Idx }, { OutputInfo{ DataType::Float32, { 1 } } });
	const auto base0 = sg.AddNode(BinaryOpNode{ BinaryOp::Multiply, { x, 0 }, { w0Ref, 0 } },
	                              { OutputInfo{ DataType::Float32, { 1 } } });
	const auto two = sg.AddNode(
	    ConstantNode{ Tensor<CPU>({ 2 }, { 1 }).CopyToDevice(PolymorphicDevice{ CPU{} }) },
	    { OutputInfo{ DataType::Float32, { 1 } } });
	const auto term0 = sg.AddNode(BinaryOpNode{ BinaryOp::Multiply, { two, 0 }, { base0, 0 } },
	                              { OutputInfo{ DataType::Float32, { 1 } } });
	const auto y = sg.AddNode(BinaryOpNode{ BinaryOp::Add, { term1, 0 }, { term0, 0 } },
	                          { OutputInfo{ DataType::Float32, { 1 } } });
	sg.SetResults({ { y, 0 } });
	graph.SetForward(graph.AddSubgraph(std::move(sg)));

	AutogradPass autograd;
	autograd.Run(graph);

	Runtime::Interpreter<CPU> interpreter;
	std::vector<Tensor<CPU>> fwdInputs;
	fwdInputs.emplace_back(Tensor<CPU>({ 3 }, { 1 }));
	(void)interpreter.RunForward(graph, fwdInputs);

	std::vector<Tensor<CPU>> bwdInputs;
	bwdInputs.emplace_back(Tensor<CPU>({ 3 }, { 1 }));
	bwdInputs.emplace_back(Tensor<CPU>({ 1 }, { 1 }));

	auto gradients = interpreter.RunBackward(graph, bwdInputs);

	ASSERT_EQ(gradients.size(), 3);
	EXPECT_FLOAT_EQ(ReadFloat(gradients[0], 0), 19.0f);
	EXPECT_FLOAT_EQ(ReadFloat(gradients[1], 0), 6.0f);
	EXPECT_FLOAT_EQ(ReadFloat(gradients[2], 0), 3.0f);
}

TEST(AutogradRegression, MultipleOutputGradientsAccumulateIntoSharedInput)
{
	Graph graph;
	Subgraph sg;

	const auto x = sg.AddParam(DataType::Float32, { 1 });
	const auto square = sg.AddNode(BinaryOpNode{ BinaryOp::Multiply, { x, 0 }, { x, 0 } },
	                               { OutputInfo{ DataType::Float32, { 1 } } });
	const auto three = sg.AddNode(
	    ConstantNode{ Tensor<CPU>({ 3 }, { 1 }).CopyToDevice(PolymorphicDevice{ CPU{} }) },
	    { OutputInfo{ DataType::Float32, { 1 } } });
	const auto shifted = sg.AddNode(BinaryOpNode{ BinaryOp::Add, { x, 0 }, { three, 0 } },
	                                { OutputInfo{ DataType::Float32, { 1 } } });
	sg.SetResults({ { square, 0 }, { shifted, 0 } });
	graph.SetForward(graph.AddSubgraph(std::move(sg)));

	AutogradPass autograd;
	autograd.Run(graph);

	Runtime::Interpreter<CPU> interpreter;
	std::vector<Tensor<CPU>> fwdInputs;
	fwdInputs.emplace_back(Tensor<CPU>({ 4 }, { 1 }));
	auto forward = interpreter.RunForward(graph, fwdInputs);

	ASSERT_EQ(forward.size(), 2);
	EXPECT_FLOAT_EQ(ReadFloat(forward[0], 0), 16.0f);
	EXPECT_FLOAT_EQ(ReadFloat(forward[1], 0), 7.0f);

	std::vector<Tensor<CPU>> bwdInputs;
	bwdInputs.emplace_back(Tensor<CPU>({ 4 }, { 1 }));
	bwdInputs.emplace_back(Tensor<CPU>({ 2 }, { 1 }));
	bwdInputs.emplace_back(Tensor<CPU>({ 3 }, { 1 }));

	auto gradients = interpreter.RunBackward(graph, bwdInputs);

	ASSERT_EQ(gradients.size(), 1);
	EXPECT_FLOAT_EQ(ReadFloat(gradients[0], 0), 19.0f);
}
