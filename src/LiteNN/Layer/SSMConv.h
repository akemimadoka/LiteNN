#ifndef LITENN_LAYER_SSMCONV_H
#define LITENN_LAYER_SSMCONV_H

#include <LiteNN/Layer/Conv2D.h>
#include <LiteNN/Layer/Permute.h>
#include <LiteNN/Layer/Reshape.h>

#include <stdexcept>
#include <vector>

namespace LiteNN::Layer
{
	inline NodeOutput AddSSMConv(Subgraph& subgraph, NodeOutput convInput, NodeOutput weight)
	{
		const auto inputInfo = subgraph.GetOutputInfo(convInput);
		const auto weightInfo = subgraph.GetOutputInfo(weight);
		if (inputInfo.shape.size() != 3 || weightInfo.shape.size() != 2)
		{
			throw std::runtime_error("SSMConv expects convInput [kernel - 1 + tokens, channels, batch] and weight [kernel, channels]");
		}
		if (inputInfo.dtype != DataType::Float32 || weightInfo.dtype != DataType::Float32)
		{
			throw std::runtime_error("SSMConv currently follows ggml and requires Float32 tensors");
		}

		const auto inputLength = inputInfo.shape[0];
		const auto channels = inputInfo.shape[1];
		const auto batch = inputInfo.shape[2];
		const auto kernel = weightInfo.shape[0];
		if (weightInfo.shape[1] != channels || kernel == 0 || inputLength < kernel)
		{
			throw std::runtime_error("SSMConv weight shape must be [kernel, channels] with input length >= kernel");
		}
		const auto tokens = inputLength - kernel + 1uz;

		const auto inputBCL = AddPermute(subgraph, convInput, { 2uz, 1uz, 0uz });
		const auto inputNCHW = AddReshape(subgraph, inputBCL, { batch, channels, 1uz, inputLength });
		const auto weightCK = AddPermute(subgraph, weight, { 1uz, 0uz });
		const auto weightGrouped = AddReshape(subgraph, weightCK, { channels, 1uz, 1uz, kernel });
		const auto convolved = AddConv2D(subgraph, inputNCHW, weightGrouped, std::nullopt,
		                                { 1uz, 1uz }, { 1uz, 1uz }, { 0uz, 0uz }, { 0uz, 0uz }, channels);
		const auto convolvedBCT = AddReshape(subgraph, convolved, { batch, channels, tokens });
		return AddPermute(subgraph, convolvedBCT, { 1uz, 2uz, 0uz });
	}
} // namespace LiteNN::Layer

#endif