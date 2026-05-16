#include <gtest/gtest.h>

#include <LiteNN.h>
#include <LiteNN/Pass/AutogradPass.h>
#include <LiteNN/Runtime/Interpreter.h>

#include <filesystem>
#include <vector>

using namespace LiteNN;

namespace
{
	float ReadFloat(const Tensor<CPU>& tensor, std::size_t index)
	{
		const auto cpuTensor = tensor.CopyToDevice(CPU{});
		return static_cast<const float*>(cpuTensor.RawData())[index];
	}

	float ReadAsFloat(const Tensor<CPU>& tensor, std::size_t index)
	{
		auto cpuTensor = tensor.CopyToDevice(CPU{});
		Tensor<CPU> converted(Uninitialized, cpuTensor.Shape(), DataType::Float32);
		CPU cpu;
		DeviceTraits<CPU>::ConvertTo(cpu, cpuTensor.DType(), cpuTensor.RawData(), cpuTensor.NumElements(),
		                             DataType::Float32, converted.RawData());
		return static_cast<const float*>(converted.RawData())[index];
	}

	float ReadVariableDataFloat(const Graph& graph, std::size_t variableIndex, std::size_t elementIndex)
	{
		const auto tensor = graph.GetVariable(variableIndex)->Data().CopyToDevice(CPU{});
		return ReadFloat(tensor, elementIndex);
	}

	Graph BuildLinearGraph()
	{
		Graph graph;
		const auto weightIndex = graph.AddVariable(
		    Variable::Create(Tensor<CPU>({ 1.0f, 2.0f, 3.0f, 4.0f }, { 2, 2 })));
		const auto biasIndex = graph.AddVariable(Variable::Create(Tensor<CPU>({ 5.0f, 6.0f }, { 1, 2 })));

		Subgraph sg;
		const auto x = sg.AddParam(DataType::Float32, { 1, 2 });
		const auto weight = sg.AddNode(VariableRefNode{ weightIndex }, { OutputInfo{ DataType::Float32, { 2, 2 } } });
		const auto bias = sg.AddNode(VariableRefNode{ biasIndex }, { OutputInfo{ DataType::Float32, { 1, 2 } } });
		const auto matmul = sg.AddNode(BinaryOpNode{ BinaryOp::MatMul, { x, 0 }, { weight, 0 } },
		                               { OutputInfo{ DataType::Float32, { 1, 2 } } });
		const auto y = sg.AddNode(BinaryOpNode{ BinaryOp::Add, { matmul, 0 }, { bias, 0 } },
		                          { OutputInfo{ DataType::Float32, { 1, 2 } } });
		sg.SetResults({ { y, 0 } });
		graph.SetForward(graph.AddSubgraph(std::move(sg)));
		graph.SetInputNames({ "features" });
		graph.SetOutputNames({ "logits" });
		return graph;
	}
} // namespace

TEST(ModelIO, SaveLoadPreservesForwardBackwardAndVariables)
{
	auto graph = BuildLinearGraph();
	AutogradPass autograd;
	autograd.Run(graph);

	const auto path = std::filesystem::path("litenn_modelio_roundtrip_test.ltnn");
	std::filesystem::remove(path);
	Serialization::SaveModel(graph, path);

	auto loaded = Serialization::LoadModel(path);
	std::filesystem::remove(path);

	ASSERT_EQ(loaded.InputSignature().size(), 1);
	ASSERT_EQ(loaded.OutputSignature().size(), 1);
	EXPECT_EQ(loaded.InputSignature()[0].name, "features");
	EXPECT_EQ(loaded.OutputSignature()[0].name, "logits");
	ASSERT_EQ(loaded.VariableCount(), 2);
	EXPECT_FLOAT_EQ(ReadVariableDataFloat(loaded, 0, 0), 1.0f);
	EXPECT_FLOAT_EQ(ReadVariableDataFloat(loaded, 0, 1), 2.0f);
	EXPECT_FLOAT_EQ(ReadVariableDataFloat(loaded, 0, 2), 3.0f);
	EXPECT_FLOAT_EQ(ReadVariableDataFloat(loaded, 0, 3), 4.0f);
	EXPECT_FLOAT_EQ(ReadVariableDataFloat(loaded, 1, 0), 5.0f);
	EXPECT_FLOAT_EQ(ReadVariableDataFloat(loaded, 1, 1), 6.0f);

	Runtime::Interpreter<CPU> interpreter;
	std::vector<Tensor<CPU>> inputs;
	inputs.emplace_back(Tensor<CPU>({ 2.0f, 3.0f }, { 1, 2 }));

	auto outputs = interpreter.RunForward(loaded, inputs);
	ASSERT_EQ(outputs.size(), 1);
	EXPECT_FLOAT_EQ(ReadFloat(outputs[0], 0), 16.0f);
	EXPECT_FLOAT_EQ(ReadFloat(outputs[0], 1), 22.0f);

	std::vector<Tensor<CPU>> backwardInputs;
	backwardInputs.emplace_back(Tensor<CPU>({ 2.0f, 3.0f }, { 1, 2 }));
	backwardInputs.emplace_back(Tensor<CPU>({ 1.0f, 1.0f }, { 1, 2 }));

	auto gradients = interpreter.RunBackward(loaded, backwardInputs);
	ASSERT_EQ(gradients.size(), 3);
	EXPECT_FLOAT_EQ(ReadFloat(gradients[0], 0), 3.0f);
	EXPECT_FLOAT_EQ(ReadFloat(gradients[0], 1), 7.0f);
	EXPECT_FLOAT_EQ(ReadFloat(gradients[1], 0), 2.0f);
	EXPECT_FLOAT_EQ(ReadFloat(gradients[1], 1), 2.0f);
	EXPECT_FLOAT_EQ(ReadFloat(gradients[1], 2), 3.0f);
	EXPECT_FLOAT_EQ(ReadFloat(gradients[1], 3), 3.0f);
	EXPECT_FLOAT_EQ(ReadFloat(gradients[2], 0), 1.0f);
	EXPECT_FLOAT_EQ(ReadFloat(gradients[2], 1), 1.0f);
}

TEST(ModelIO, SaveLoadPreservesLowPrecisionScalarDTypes)
{
	Graph graph;
	const auto weightIndex =
	    graph.AddVariable(Variable::Create(Tensor<CPU>({ 1.0, 2.0 }, { 2 }, DataType::Float16)));

	Subgraph sg;
	const auto x = sg.AddParam(DataType::Float16, { 2 });
	const auto weight = sg.AddNode(VariableRefNode{ weightIndex }, { OutputInfo{ DataType::Float16, { 2 } } });
	const auto y = sg.AddNode(BinaryOpNode{ BinaryOp::Add, { x, 0 }, { weight, 0 } },
	                          { OutputInfo{ DataType::Float16, { 2 } } });
	sg.SetResults({ { y, 0 } });
	graph.SetForward(graph.AddSubgraph(std::move(sg)));
	graph.SetInputNames({ "x" });
	graph.SetOutputNames({ "y" });

	const auto path = std::filesystem::path("litenn_modelio_low_precision_roundtrip_test.ltnn");
	std::filesystem::remove(path);
	Serialization::SaveModel(graph, path);

	auto loaded = Serialization::LoadModel(path);
	std::filesystem::remove(path);

	ASSERT_EQ(loaded.VariableCount(), 1);
	EXPECT_EQ(loaded.GetVariable(0)->Data().DType(), DataType::Float16);
	ASSERT_EQ(loaded.InputSignature().size(), 1);
	ASSERT_EQ(loaded.OutputSignature().size(), 1);
	EXPECT_EQ(loaded.InputSignature()[0].dtype, DataType::Float16);
	EXPECT_EQ(loaded.OutputSignature()[0].dtype, DataType::Float16);

	Runtime::Interpreter<CPU> interpreter;
	std::vector<Tensor<CPU>> inputs;
	inputs.emplace_back(Tensor<CPU>({ 3.0, 4.0 }, { 2 }, DataType::Float16));
	auto outputs = interpreter.RunForward(loaded, inputs);
	ASSERT_EQ(outputs.size(), 1);
	EXPECT_EQ(outputs[0].DType(), DataType::Float16);
	EXPECT_NEAR(ReadAsFloat(outputs[0], 0), 4.0F, 1e-3F);
	EXPECT_NEAR(ReadAsFloat(outputs[0], 1), 6.0F, 1e-3F);
}
