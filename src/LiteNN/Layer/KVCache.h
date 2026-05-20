#include <LiteNN/Graph.h>

#include <stdexcept>

#ifndef LITENN_LAYER_KVCACHE_H
#define LITENN_LAYER_KVCACHE_H

namespace LiteNN::Layer
{
	struct KVCachePair
	{
		NodeOutput keys;
		NodeOutput values;
	};

	namespace Detail
	{
		inline OutputInfo ConcatInfo(const OutputInfo& lhs, const OutputInfo& rhs, std::size_t axis,
		                            std::string_view label)
		{
			if (lhs.dtype != rhs.dtype)
			{
				throw std::runtime_error(std::format("{} tensors must share the same dtype", label));
			}
			if (lhs.shape.size() != rhs.shape.size())
			{
				throw std::runtime_error(std::format("{} tensors must share the same rank", label));
			}
			if (axis >= lhs.shape.size())
			{
				throw std::runtime_error(std::format("{} axis out of range", label));
			}

			auto shape = lhs.shape;
			for (std::size_t dim = 0; dim < shape.size(); ++dim)
			{
				if (dim == axis)
				{
					continue;
				}
				if (lhs.shape[dim] != rhs.shape[dim])
				{
					throw std::runtime_error(std::format("{} tensors mismatch at dim {}", label, dim));
				}
			}
			shape[axis] += rhs.shape[axis];
			return { lhs.dtype, std::move(shape) };
		}

		inline OutputInfo SliceInfo(const OutputInfo& input, std::size_t axis, std::size_t start, std::size_t length,
		                           std::string_view label)
		{
			if (axis >= input.shape.size())
			{
				throw std::runtime_error(std::format("{} axis out of range", label));
			}
			if (length == 0 || start > input.shape[axis] || length > input.shape[axis] - start)
			{
				throw std::runtime_error(std::format("{} slice range is out of bounds", label));
			}

			auto shape = input.shape;
			shape[axis] = length;
			return { input.dtype, std::move(shape) };
		}

		inline void ValidateKVPair(const OutputInfo& keys, const OutputInfo& values, std::size_t axis,
		                          std::string_view label)
		{
			if (keys.shape.size() != values.shape.size())
			{
				throw std::runtime_error(std::format("{} key/value tensors must share the same rank", label));
			}
			if (axis >= keys.shape.size())
			{
				throw std::runtime_error(std::format("{} axis out of range", label));
			}
			if (keys.shape[axis] != values.shape[axis])
			{
				throw std::runtime_error(std::format("{} key/value sequence lengths must match", label));
			}
		}
	} // namespace Detail

	inline KVCachePair AddKVCacheAppend(Subgraph& subgraph, KVCachePair cache, KVCachePair appended,
	                                  std::size_t axis = 0)
	{
		const auto cacheKeys = subgraph.GetOutputInfo(cache.keys);
		const auto cacheValues = subgraph.GetOutputInfo(cache.values);
		const auto appendKeys = subgraph.GetOutputInfo(appended.keys);
		const auto appendValues = subgraph.GetOutputInfo(appended.values);

		Detail::ValidateKVPair(cacheKeys, cacheValues, axis, "KV cache");
		Detail::ValidateKVPair(appendKeys, appendValues, axis, "KV append");

		const auto keyInfo = Detail::ConcatInfo(cacheKeys, appendKeys, axis, "KV cache keys");
		const auto valueInfo = Detail::ConcatInfo(cacheValues, appendValues, axis, "KV cache values");

		const auto keys = subgraph.AddNode(ConcatNode{ { cache.keys, appended.keys }, axis }, { keyInfo });
		const auto values = subgraph.AddNode(ConcatNode{ { cache.values, appended.values }, axis }, { valueInfo });
		return { { keys, 0 }, { values, 0 } };
	}

	inline KVCachePair AddKVCacheView(Subgraph& subgraph, KVCachePair cache, std::size_t start, std::size_t length,
	                                std::size_t axis = 0)
	{
		const auto keyInfo = subgraph.GetOutputInfo(cache.keys);
		const auto valueInfo = subgraph.GetOutputInfo(cache.values);
		Detail::ValidateKVPair(keyInfo, valueInfo, axis, "KV cache view");

		const auto slicedKeyInfo = Detail::SliceInfo(keyInfo, axis, start, length, "KV cache key view");
		const auto slicedValueInfo = Detail::SliceInfo(valueInfo, axis, start, length, "KV cache value view");

		const auto keys =
		    subgraph.AddNode(SliceNode{ cache.keys, axis, start, length }, { std::move(slicedKeyInfo) });
		const auto values =
		    subgraph.AddNode(SliceNode{ cache.values, axis, start, length }, { std::move(slicedValueInfo) });
		return { { keys, 0 }, { values, 0 } };
	}
	
} // namespace LiteNN::Layer

#endif