#ifndef LITENN_LAYER_SSMSCAN_H
#define LITENN_LAYER_SSMSCAN_H

#include <LiteNN/ComputePrimitives.h>
#include <LiteNN/Graph.h>

#include <optional>
#include <stdexcept>

namespace LiteNN::Layer
{
	inline NodeOutput AddSSMScan(Subgraph& subgraph, NodeOutput state, NodeOutput dt, NodeOutput a,
	                             NodeOutput b, NodeOutput c, std::optional<NodeOutput> d = std::nullopt)
	{
		const auto stateInfo = subgraph.GetOutputInfo(state);
		if (stateInfo.shape.size() != 2 || !IsFloatingDataType(stateInfo.dtype))
		{
			throw std::runtime_error("SSMScan state must be floating-point rank-2 [steps, channels]");
		}
		for (const auto input : { dt, a, b, c })
		{
			const auto info = subgraph.GetOutputInfo(input);
			if (!IsFloatingDataType(info.dtype))
			{
				throw std::runtime_error("SSMScan inputs must be floating-point tensors");
			}
			(void)::LiteNN::Detail::BroadcastToShape(info.shape, stateInfo.shape);
		}
		if (d)
		{
			const auto info = subgraph.GetOutputInfo(*d);
			if (!IsFloatingDataType(info.dtype))
			{
				throw std::runtime_error("SSMScan D input must be floating-point");
			}
			(void)::LiteNN::Detail::BroadcastToShape(info.shape, stateInfo.shape);
		}
		const auto result = subgraph.AddNode(SSMScanNode{ state, dt, a, b, c, d },
		                                     { OutputInfo{ stateInfo.dtype, stateInfo.shape } });
		return { result, 0 };
	}
} // namespace LiteNN::Layer

#endif
