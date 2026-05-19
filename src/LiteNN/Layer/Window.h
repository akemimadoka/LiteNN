#ifndef LITENN_LAYER_WINDOW_H
#define LITENN_LAYER_WINDOW_H

#include <LiteNN/Layer/Pad.h>
#include <LiteNN/Layer/Permute.h>
#include <LiteNN/Layer/Reshape.h>

#include <format>
#include <stdexcept>
#include <utility>
#include <vector>

namespace LiteNN::Layer
{
	namespace Detail
	{
		inline std::size_t PaddingToMultiple(std::size_t extent, std::size_t multiple)
		{
			return (multiple - extent % multiple) % multiple;
		}

		inline NodeOutput AddSlice(Subgraph& subgraph, NodeOutput input, std::size_t axis,
		                           std::size_t start, std::size_t length)
		{
			const auto info = subgraph.GetOutputInfo(input);
			if (axis >= info.shape.size() || start + length > info.shape[axis])
			{
				throw std::runtime_error("Slice range is out of bounds");
			}
			auto outputShape = info.shape;
			outputShape[axis] = length;
			const auto result = subgraph.AddNode(SliceNode{ input, axis, start, length },
			                                     { OutputInfo{ info.dtype, std::move(outputShape) } });
			return { result, 0 };
		}
	} // namespace Detail

	inline NodeOutput AddWindowPartition(Subgraph& subgraph, NodeOutput input, std::size_t windowSize)
	{
		const auto info = subgraph.GetOutputInfo(input);
		if (info.dtype != DataType::Float32)
		{
			throw std::runtime_error("WindowPartition currently follows ggml and requires Float32 input");
		}
		if (info.shape.size() != 4 || windowSize == 0)
		{
			throw std::runtime_error("WindowPartition expects [channels, width, height, batch] and windowSize > 0");
		}

		const auto channels = info.shape[0];
		const auto width = info.shape[1];
		const auto height = info.shape[2];
		const auto batch = info.shape[3];
		const auto padWidth = Detail::PaddingToMultiple(width, windowSize);
		const auto padHeight = Detail::PaddingToMultiple(height, windowSize);
		const auto paddedWidth = width + padWidth;
		const auto paddedHeight = height + padHeight;
		const auto windowsWide = paddedWidth / windowSize;
		const auto windowsHigh = paddedHeight / windowSize;

		const std::vector<std::size_t> lowPads{ 0uz, 0uz, 0uz, 0uz };
		const std::vector<std::size_t> highPads{ 0uz, padWidth, padHeight, 0uz };
		const auto padded = AddPad(subgraph, input, lowPads, highPads, PadMode::Constant, 0.0);
		const std::vector<std::size_t> splitShape{ channels, windowsWide, windowSize, windowsHigh, windowSize, batch };
		const auto split = AddReshape(subgraph, padded, splitShape);
		const auto partitioned = AddPermute(subgraph, split, { 0uz, 2uz, 4uz, 1uz, 3uz, 5uz });
		const std::vector<std::size_t> outputShape{ channels, windowSize, windowSize, windowsWide * windowsHigh * batch };
		return AddReshape(subgraph, partitioned, outputShape);
	}

	inline NodeOutput AddWindowUnpartition(Subgraph& subgraph, NodeOutput input, std::size_t width,
	                                      std::size_t height, std::size_t windowSize)
	{
		const auto info = subgraph.GetOutputInfo(input);
		if (info.dtype != DataType::Float32)
		{
			throw std::runtime_error("WindowUnpartition currently follows ggml and requires Float32 input");
		}
		if (info.shape.size() != 4 || windowSize == 0 || width == 0 || height == 0)
		{
			throw std::runtime_error("WindowUnpartition expects [channels, window, window, windows*batch]");
		}
		if (info.shape[1] != windowSize || info.shape[2] != windowSize)
		{
			throw std::runtime_error("WindowUnpartition input window dimensions must match windowSize");
		}

		const auto channels = info.shape[0];
		const auto padWidth = Detail::PaddingToMultiple(width, windowSize);
		const auto padHeight = Detail::PaddingToMultiple(height, windowSize);
		const auto paddedWidth = width + padWidth;
		const auto paddedHeight = height + padHeight;
		const auto windowsWide = paddedWidth / windowSize;
		const auto windowsHigh = paddedHeight / windowSize;
		const auto windowCount = windowsWide * windowsHigh;
		if (info.shape[3] % windowCount != 0)
		{
			throw std::runtime_error(std::format("WindowUnpartition input has {} windows, not divisible by {}",
			                               info.shape[3], windowCount));
		}
		const auto batch = info.shape[3] / windowCount;

		const std::vector<std::size_t> splitShape{ channels, windowSize, windowSize, windowsWide, windowsHigh, batch };
		const auto split = AddReshape(subgraph, input, splitShape);
		const auto unpermuted = AddPermute(subgraph, split, { 0uz, 3uz, 1uz, 4uz, 2uz, 5uz });
		const std::vector<std::size_t> paddedShape{ channels, paddedWidth, paddedHeight, batch };
		auto current = AddReshape(subgraph, unpermuted, paddedShape);
		if (paddedWidth != width)
		{
			current = Detail::AddSlice(subgraph, current, 1uz, 0uz, width);
		}
		if (paddedHeight != height)
		{
			current = Detail::AddSlice(subgraph, current, 2uz, 0uz, height);
		}
		return current;
	}
} // namespace LiteNN::Layer

#endif