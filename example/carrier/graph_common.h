#pragma once

#include <LiteNN.h>
#include <LiteNN/Compiler/CompiledModule.h>

#include <array>
#include <cstddef>
#include <format>
#include <iostream>
#include <stdexcept>
#include <string_view>

namespace LiteNN::Examples::Carrier
{
	inline float ReadFloat(const Tensor<CPU>& tensor, std::size_t index)
	{
		return static_cast<const float*>(tensor.RawData())[index];
	}

	inline Graph BuildCarrierExampleGraph()
	{
		Graph graph;
		Subgraph sg;
		const auto lhs = sg.AddParam(DataType::Float32, { 2, 2 });
		const auto rhs = sg.AddParam(DataType::Float32, { 2, 2 });
		const auto sum = sg.AddNode(BinaryOpNode{ BinaryOp::Add, { lhs, 0 }, { rhs, 0 } },
		                           { OutputInfo{ DataType::Float32, { 2, 2 } } });
		sg.SetResults({ { sum, 0 } });
		graph.AddSubgraph(std::move(sg));
		graph.SetForward(0);
		graph.SetInputNames({ "lhs", "rhs" });
		graph.SetOutputNames({ "sum" });
		return graph;
	}

	inline std::array<Tensor<CPU>, 2> MakeSampleInputs()
	{
		return {
			Tensor<CPU>({ 1, 2, 3, 4 }, { 2, 2 }, DataType::Float32),
			Tensor<CPU>({ 10, 20, 30, 40 }, { 2, 2 }, DataType::Float32),
		};
	}

	inline void VerifySampleOutputs(std::span<const Tensor<CPU>> outputs)
	{
		if (outputs.size() != 1 || outputs[0].NumElements() != 4)
		{
			throw std::runtime_error("Carrier example output metadata mismatch");
		}
		constexpr std::array<float, 4> expected = { 11.0f, 22.0f, 33.0f, 44.0f };
		for (std::size_t i = 0; i < expected.size(); ++i)
		{
			if (ReadFloat(outputs[0], i) != expected[i])
			{
				throw std::runtime_error(std::format("Carrier example output {} mismatch", i));
			}
		}
	}

	inline void PrintRunSummary(std::string_view mode, const CompiledModule<CPU>& module,
	                            std::span<const Tensor<CPU>> outputs)
	{
		std::cout << std::format(
		    "{} carrier load succeeded: inputs=[{}, {}], output {} = [{}, {}, {}, {}]\n",
		    mode,
		    module.InputSpecs().empty() ? "?" : module.InputSpecs()[0].name,
		    module.InputSpecs().size() < 2 ? "?" : module.InputSpecs()[1].name,
		    module.OutputSpecs().empty() ? "?" : module.OutputSpecs()[0].name,
		    ReadFloat(outputs[0], 0),
		    ReadFloat(outputs[0], 1),
		    ReadFloat(outputs[0], 2),
		    ReadFloat(outputs[0], 3));
	}
} // namespace LiteNN::Examples::Carrier