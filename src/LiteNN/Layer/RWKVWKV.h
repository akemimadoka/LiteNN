#ifndef LITENN_LAYER_RWKVWKV_H
#define LITENN_LAYER_RWKVWKV_H

#include <LiteNN/ComputePrimitives.h>
#include <LiteNN/Graph.h>

#include <ranges>
#include <stdexcept>

namespace LiteNN::Layer
{
	inline NodeOutput AddRWKVWKV(Subgraph& subgraph, NodeOutput key, NodeOutput value, NodeOutput receptance,
	                             NodeOutput timeDecay, NodeOutput timeFirst)
	{
		const auto keyInfo = subgraph.GetOutputInfo(key);
		const auto valueInfo = subgraph.GetOutputInfo(value);
		const auto receptanceInfo = subgraph.GetOutputInfo(receptance);
		if (keyInfo.shape.size() != 2 || !IsFloatingDataType(keyInfo.dtype))
		{
			throw std::runtime_error("RWKVWKV key must be floating-point rank-2 [steps, channels]");
		}
		if (valueInfo.dtype != keyInfo.dtype || receptanceInfo.dtype != keyInfo.dtype ||
		    !std::ranges::equal(valueInfo.shape, keyInfo.shape) ||
		    !std::ranges::equal(receptanceInfo.shape, keyInfo.shape))
		{
			throw std::runtime_error("RWKVWKV key/value/receptance metadata must match");
		}
		for (const auto input : { timeDecay, timeFirst })
		{
			const auto info = subgraph.GetOutputInfo(input);
			if (!IsFloatingDataType(info.dtype))
			{
				throw std::runtime_error("RWKVWKV time parameters must be floating-point tensors");
			}
			(void)::LiteNN::Detail::BroadcastToShape(info.shape, keyInfo.shape);
		}
		const auto result = subgraph.AddNode(RWKVWKVNode{ key, value, receptance, timeDecay, timeFirst },
		                                     { OutputInfo{ keyInfo.dtype, keyInfo.shape } });
		return { result, 0 };
	}
} // namespace LiteNN::Layer

#endif
