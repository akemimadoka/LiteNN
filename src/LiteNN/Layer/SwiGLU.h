#include <LiteNN/Graph.h>
#include <LiteNN/Layer/Activation.h>
#include <LiteNN/Layer/Linear.h>
#include <LiteNN/ModelBuilder.h>

#include <stdexcept>

#ifndef LITENN_LAYER_SWIGLU_H
#define LITENN_LAYER_SWIGLU_H

namespace LiteNN::Layer
{
	struct SwiGLUMLPLayer
	{
		LinearLayer gateProjection;
		LinearLayer upProjection;
		LinearLayer downProjection;
	};

	inline void ValidateSwiGLUMLP(const LinearLayer& gateProjection, const LinearLayer& upProjection,
	                              const LinearLayer& downProjection)
	{
		if (gateProjection.dtype != upProjection.dtype || gateProjection.dtype != downProjection.dtype)
		{
			throw std::runtime_error("SwiGLU projections must share the same dtype");
		}
		if (gateProjection.inFeatures != upProjection.inFeatures)
		{
			throw std::runtime_error("SwiGLU gate/up projections must share the same input width");
		}
		if (gateProjection.outFeatures != upProjection.outFeatures)
		{
			throw std::runtime_error("SwiGLU gate/up projections must share the same hidden width");
		}
		if (downProjection.inFeatures != gateProjection.outFeatures)
		{
			throw std::runtime_error("SwiGLU down projection input width must match gate/up hidden width");
		}
	}

	namespace Detail
	{
		inline SwiGLUMLPLayer CreateSwiGLUMLPImpl(Graph& graph, Tensor<CPU> gateWeight, Tensor<CPU> upWeight,
		                                          Tensor<CPU> downWeight)
		{
			const auto gateProjection = Detail::MakeLinearLayerImpl(graph, std::move(gateWeight));
			const auto upProjection = Detail::MakeLinearLayerImpl(graph, std::move(upWeight));
			const auto downProjection = Detail::MakeLinearLayerImpl(graph, std::move(downWeight));
			ValidateSwiGLUMLP(gateProjection, upProjection, downProjection);
			return {
				.gateProjection = gateProjection,
				.upProjection = upProjection,
				.downProjection = downProjection,
			};
		}
	} // namespace Detail

	inline SwiGLUMLPLayer CreateSwiGLUMLP(ModelBuilder& builder, Tensor<CPU> gateWeight, Tensor<CPU> upWeight,
	                                      Tensor<CPU> downWeight)
	{
		return Detail::CreateSwiGLUMLPImpl(builder.UnsafeMutableGraph(), std::move(gateWeight), std::move(upWeight),
		                                   std::move(downWeight));
	}

	inline NodeOutput AddSwiGLUMLP(Subgraph& subgraph, const SwiGLUMLPLayer& layer, NodeOutput input)
	{
		ValidateSwiGLUMLP(layer.gateProjection, layer.upProjection, layer.downProjection);
		const auto gate = AddLinear(subgraph, layer.gateProjection, input);
		const auto gateActivated = AddSiLU(subgraph, gate);
		const auto up = AddLinear(subgraph, layer.upProjection, input);
		const auto gated = subgraph.AddNode(
		    BinaryOpNode{ BinaryOp::Multiply, gateActivated, up },
		    { OutputInfo{ layer.gateProjection.dtype, { subgraph.GetOutputInfo(gateActivated).shape } } });
		return AddLinear(subgraph, layer.downProjection, { gated, 0 });
	}

	namespace Detail
	{
		inline SubgraphId BuildSwiGLUMLPImpl(Graph& graph, const SwiGLUMLPLayer& layer, std::size_t batchSize)
		{
			ValidateSwiGLUMLP(layer.gateProjection, layer.upProjection, layer.downProjection);
			Subgraph subgraph;
			const auto input =
			    subgraph.AddParam(layer.gateProjection.dtype, { batchSize, layer.gateProjection.inFeatures });
			const auto result = AddSwiGLUMLP(subgraph, layer, { input, 0 });
			subgraph.SetResults({ result });
			return graph.AddSubgraph(std::move(subgraph));
		}
	} // namespace Detail

	inline SubgraphId BuildSwiGLUMLP(ModelBuilder& builder, const SwiGLUMLPLayer& layer, std::size_t batchSize = 1)
	{
		return Detail::BuildSwiGLUMLPImpl(builder.UnsafeMutableGraph(), layer, batchSize);
	}
} // namespace LiteNN::Layer

#endif
