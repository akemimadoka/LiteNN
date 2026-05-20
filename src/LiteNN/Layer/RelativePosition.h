#ifndef LITENN_LAYER_RELATIVEPOSITION_H
#define LITENN_LAYER_RELATIVEPOSITION_H

#include <LiteNN/Layer/BroadcastTo.h>
#include <LiteNN/Layer/Gather.h>
#include <LiteNN/Layer/LayerUtils.h>
#include <LiteNN/Layer/Reshape.h>

#include <cstdint>
#include <ranges>
#include <span>
#include <stdexcept>
#include <vector>

namespace LiteNN::Layer
{
	namespace Detail
	{
		inline Tensor<CPU> MakeRelativePositionIndexTensor(std::size_t querySize, std::size_t keySize)
		{
			Tensor<CPU> indices(Uninitialized, { querySize, keySize }, DataType::Int32);
			auto* data = static_cast<std::int32_t*>(indices.RawData());
			for (auto query = 0uz; query < querySize; ++query)
			{
				for (auto key = 0uz; key < keySize; ++key)
				{
					data[query * keySize + key] = static_cast<std::int32_t>((keySize - key - 1uz) + query);
				}
			}
			return indices;
		}
	} // namespace Detail

	inline NodeOutput AddGetRelativePosition(Subgraph& subgraph, NodeOutput relativePosition,
	                                         std::size_t querySize, std::size_t keySize)
	{
		const auto info = subgraph.GetOutputInfo(relativePosition);
		if (info.shape.size() != 2 || querySize == 0 || keySize == 0)
		{
			throw std::runtime_error("GetRelativePosition expects [length, channels] and positive query/key sizes");
		}
		if (querySize != keySize)
		{
			throw std::runtime_error("GetRelativePosition currently follows ggml and requires querySize == keySize");
		}
		if (info.shape[0] != 2 * querySize - 1)
		{
			throw std::runtime_error("GetRelativePosition length must equal 2 * querySize - 1");
		}
		if (info.dtype == DataType::Bool)
		{
			throw std::runtime_error("GetRelativePosition does not support Bool tensors");
		}

		const auto indices = Detail::MakeRelativePositionIndexTensor(querySize, keySize);
		const auto indexNode = Detail::AddConstant(subgraph, indices);
		return AddGather(subgraph, relativePosition, { indexNode, 0 }, 0uz);
	}

	inline NodeOutput AddRelativePositionBias2D(Subgraph& subgraph, NodeOutput scores,
	                                           NodeOutput widthBias, NodeOutput heightBias)
	{
		const auto scoreInfo = subgraph.GetOutputInfo(scores);
		const auto widthInfo = subgraph.GetOutputInfo(widthBias);
		const auto heightInfo = subgraph.GetOutputInfo(heightBias);
		if (scoreInfo.shape.size() != 5 || widthInfo.shape.size() != 3 || heightInfo.shape.size() != 3)
		{
			throw std::runtime_error(
			    "RelativePositionBias2D expects scores [qH, qW, kH, kW, heads], width [qW, kW, heads], height [qH, kH, heads]");
		}
		if (scoreInfo.dtype != widthInfo.dtype || scoreInfo.dtype != heightInfo.dtype || scoreInfo.dtype == DataType::Bool)
		{
			throw std::runtime_error("RelativePositionBias2D requires matching non-Bool dtypes");
		}

		const auto queryHeight = scoreInfo.shape[0];
		const auto queryWidth = scoreInfo.shape[1];
		const auto keyHeight = scoreInfo.shape[2];
		const auto keyWidth = scoreInfo.shape[3];
		const auto heads = scoreInfo.shape[4];
		if (!std::ranges::equal(widthInfo.shape, std::vector<std::size_t>{ queryWidth, keyWidth, heads }) ||
		    !std::ranges::equal(heightInfo.shape, std::vector<std::size_t>{ queryHeight, keyHeight, heads }))
		{
			throw std::runtime_error("RelativePositionBias2D bias shapes do not match score dimensions");
		}

		const std::vector<std::size_t> scoreShape = scoreInfo.shape;
		const auto widthReshaped = AddReshape(subgraph, widthBias, { 1uz, queryWidth, 1uz, keyWidth, heads });
		const auto widthExpanded = AddBroadcastTo(subgraph, widthReshaped, scoreShape);
		const auto heightReshaped = AddReshape(subgraph, heightBias, { queryHeight, 1uz, keyHeight, 1uz, heads });
		const auto heightExpanded = AddBroadcastTo(subgraph, heightReshaped, scoreShape);

		const auto withWidth = subgraph.AddNode(BinaryOpNode{ BinaryOp::Add, scores, widthExpanded },
		                                      { OutputInfo{ scoreInfo.dtype, scoreShape } });
		const auto withHeight = subgraph.AddNode(BinaryOpNode{ BinaryOp::Add, { withWidth, 0 }, heightExpanded },
		                                       { OutputInfo{ scoreInfo.dtype, scoreShape } });
		return { withHeight, 0 };
	}
} // namespace LiteNN::Layer

#endif