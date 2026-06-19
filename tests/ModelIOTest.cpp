#include <gtest/gtest.h>

#include <LiteNN.h>
#include <LiteNN/Pass/AutogradPass.h>
#include <LiteNN/Runtime/Interpreter.h>
#include <LiteNN/Serialization/ModelIO.h>

#include <filesystem>
#include <fstream>
#include <vector>

using namespace LiteNN;

namespace
{
	float ReadFloat(const Tensor<CPU>& tensor, std::size_t index)
	{
		const auto cpuTensor = tensor.CopyToDevice(CPU{});
		return static_cast<const float*>(cpuTensor.UnsafeRawData())[index];
	}

	float ReadAsFloat(const Tensor<CPU>& tensor, std::size_t index)
	{
		auto cpuTensor = tensor.CopyToDevice(CPU{});
		Tensor<CPU> converted(Uninitialized, cpuTensor.Shape(), DataType::Float32);
		CPU cpu;
		DeviceTraits<CPU>::ConvertTo(cpu, cpuTensor.DType(), cpuTensor.UnsafeRawData(), cpuTensor.NumElements(),
		                             DataType::Float32, converted.UnsafeRawData());
		return static_cast<const float*>(converted.UnsafeRawData())[index];
	}

	float ReadVariableDataFloat(const Graph& graph, std::size_t variableIndex, std::size_t elementIndex)
	{
		const auto tensor = graph.GetVariable(variableIndex)->Data().CopyToDevice(CPU{});
		return ReadFloat(tensor, elementIndex);
	}

	Graph BuildLinearGraph()
	{
		Graph graph;
		const auto weightIndex = graph.AddVariable(Variable::Create(Tensor<CPU>({ 1.0f, 2.0f, 3.0f, 4.0f }, { 2, 2 })));
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
