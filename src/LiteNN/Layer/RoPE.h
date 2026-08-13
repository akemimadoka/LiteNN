#include <LiteNN/Graph.h>
#include <LiteNN/ModelBuilder.h>

#include <cmath>
#include <stdexcept>

#ifndef LITENN_LAYER_ROPE_H
#define LITENN_LAYER_ROPE_H

namespace LiteNN::Layer
{
	// Rotary Position Embedding helper for 2D [sequenceLength, featureSize] inputs.
	inline NodeOutput AddRoPE(Subgraph& subgraph, NodeOutput input, RoPELayout layout, double base = 10000.0,
	                          std::size_t positionOffset = 0, double frequencyScale = 1.0)
	{
		const auto info = subgraph.GetOutputInfo(input); // copy
		if (info.shape.size() != 2)
		{
			throw std::runtime_error("RoPE input must be 2D with shape [sequenceLength, featureSize]");
		}
		if (!IsFloatingDataType(info.dtype))
		{
			throw std::runtime_error("RoPE input dtype must be floating-point");
		}
		if ((info.shape[1] % 2) != 0)
		{
			throw std::runtime_error("RoPE featureSize must be even");
		}
		if (!(std::isfinite(base) && base > 0.0))
		{
			throw std::runtime_error("RoPE base must be finite and greater than zero");
		}
		if (!(std::isfinite(frequencyScale) && frequencyScale > 0.0))
		{
			throw std::runtime_error("RoPE frequencyScale must be finite and greater than zero");
		}

		const auto output = subgraph.AddNode(RoPENode{ .input = input,
		                                               .positions = std::nullopt,
		                                               .layout = layout,
		                                               .base = base,
		                                               .frequencyScale = frequencyScale,
		                                               .positionOffset = positionOffset },
		                                     { info });
		return { output, 0 };
	}

	/// Applies RoPE using one runtime position per input row.
	inline NodeOutput AddRoPEAtPositions(Subgraph& subgraph, NodeOutput input, NodeOutput positions, RoPELayout layout,
	                                     double base = 10000.0, double frequencyScale = 1.0)
	{
		const auto info = subgraph.GetOutputInfo(input);
		const auto positionInfo = subgraph.GetOutputInfo(positions);
		if (info.shape.size() != 2 || (info.shape[1] % 2) != 0 || !IsFloatingDataType(info.dtype))
		{
			throw std::runtime_error("Dynamic RoPE input must be floating-point [sequenceLength, evenFeatureSize]");
		}
		if ((positionInfo.dtype != DataType::Int32 && positionInfo.dtype != DataType::Int64) ||
		    positionInfo.shape != std::vector<std::size_t>{ info.shape[0] })
		{
			throw std::runtime_error("Dynamic RoPE positions must be Int32/Int64 [sequenceLength]");
		}
		if (!(std::isfinite(base) && base > 0.0 && std::isfinite(frequencyScale) && frequencyScale > 0.0))
		{
			throw std::runtime_error("Dynamic RoPE base and frequencyScale must be finite and greater than zero");
		}
		const auto output = subgraph.AddNode(RoPENode{ .input = input,
		                                               .positions = positions,
		                                               .layout = layout,
		                                               .base = base,
		                                               .frequencyScale = frequencyScale,
		                                               .positionOffset = 0 },
		                                     { info });
		return { output, 0 };
	}

	inline SubgraphId BuildRoPE(ModelBuilder& builder, DataType dtype, ShapeView shape, RoPELayout layout,
	                            double base = 10000.0, std::size_t positionOffset = 0, double frequencyScale = 1.0)
	{
		Subgraph subgraph;
		const auto input = subgraph.AddParam(dtype, shape.ToOwned());
		const auto result = AddRoPE(subgraph, { input, 0 }, layout, base, positionOffset, frequencyScale);
		subgraph.SetResults({ result });
		return builder.AddSubgraph(std::move(subgraph));
	}
} // namespace LiteNN::Layer

#endif
