#ifndef LITENN_LAYER_NORMALIZATION_H
#define LITENN_LAYER_NORMALIZATION_H

#include <LiteNN/ComputePrimitives.h>
#include <LiteNN/Graph.h>

#include <cmath>
#include <optional>
#include <stdexcept>

namespace LiteNN::Layer
{
	inline NodeOutput AddNormalization(Subgraph& subgraph, NodeOutput input, NormalizationMode mode,
	                                   std::size_t axis, double epsilon,
	                                   std::optional<NodeOutput> scale = std::nullopt,
	                                   std::optional<NodeOutput> bias = std::nullopt,
	                                   std::size_t groupCount = 1)
	{
		const auto info = subgraph.GetOutputInfo(input);
		if (!IsFloatingDataType(info.dtype))
		{
			throw std::runtime_error("Normalization requires a floating-point input tensor");
		}
		if (!std::isfinite(epsilon) || epsilon <= 0.0)
		{
			throw std::runtime_error("Normalization epsilon must be finite and positive");
		}
		if (scale)
		{
			const auto scaleInfo = subgraph.GetOutputInfo(*scale);
			if (!IsFloatingDataType(scaleInfo.dtype))
			{
				throw std::runtime_error("Normalization scale must be floating-point");
			}
			(void)::LiteNN::Detail::BroadcastToShape(scaleInfo.shape, info.shape);
		}
		if (bias)
		{
			const auto biasInfo = subgraph.GetOutputInfo(*bias);
			if (!IsFloatingDataType(biasInfo.dtype))
			{
				throw std::runtime_error("Normalization bias must be floating-point");
			}
			(void)::LiteNN::Detail::BroadcastToShape(biasInfo.shape, info.shape);
		}
		if (mode == NormalizationMode::LayerNorm || mode == NormalizationMode::RMSNorm)
		{
			if (info.shape.empty() || axis >= info.shape.size())
			{
				throw std::runtime_error("Normalization axis out of range");
			}
		}
		else if (mode == NormalizationMode::GroupNorm)
		{
			if (groupCount == 0)
			{
				throw std::runtime_error("GroupNorm requires groupCount > 0");
			}
			if (info.shape.empty() || info.shape.size() > 4)
			{
				throw std::runtime_error("GroupNorm input rank must be between 1 and 4");
			}
		}
		else
		{
			throw std::runtime_error("Invalid normalization mode");
		}

		const auto result = subgraph.AddNode(
		    NormalizationNode{ input, scale, bias, mode, axis, groupCount, epsilon },
		    { OutputInfo{ info.dtype, info.shape } });
		return { result, 0 };
	}
} // namespace LiteNN::Layer

#endif
