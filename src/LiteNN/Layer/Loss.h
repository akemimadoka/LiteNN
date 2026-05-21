#include <LiteNN/Graph.h>

#include <stdexcept>
#include <utility>
#include <vector>

#ifndef LITENN_LAYER_LOSS_H
#define LITENN_LAYER_LOSS_H

namespace LiteNN::Layer
{
	inline NodeOutput AddCrossEntropyLoss(Subgraph& subgraph, NodeOutput logits, NodeOutput labels)
	{
		const auto logitsInfo = subgraph.GetOutputInfo(logits);
		const auto labelsInfo = subgraph.GetOutputInfo(labels);
		if (logitsInfo.dtype != DataType::Float32 || labelsInfo.dtype != DataType::Float32)
		{
			throw std::runtime_error("CrossEntropyLoss requires Float32 logits and labels");
		}
		if (logitsInfo.shape.empty() || logitsInfo.shape.back() == 0)
		{
			throw std::runtime_error("CrossEntropyLoss expects shape [..., classes]");
		}
		if (logitsInfo.shape != labelsInfo.shape)
		{
			throw std::runtime_error("CrossEntropyLoss logits and labels shapes must match");
		}
		const auto result = subgraph.AddNode(CrossEntropyLossNode{ logits, labels },
		                                    { OutputInfo{ DataType::Float32, { 1 } } });
		return { result, 0 };
	}

	inline NodeOutput AddCrossEntropyLossBackward(Subgraph& subgraph, NodeOutput grad,
	                                              NodeOutput logits, NodeOutput labels)
	{
		const auto gradInfo = subgraph.GetOutputInfo(grad);
		const auto logitsInfo = subgraph.GetOutputInfo(logits);
		const auto labelsInfo = subgraph.GetOutputInfo(labels);
		if (gradInfo.dtype != DataType::Float32 || gradInfo.shape != std::vector<std::size_t>{ 1 })
		{
			throw std::runtime_error("CrossEntropyLossBackward grad must be Float32 [1]");
		}
		if (logitsInfo.dtype != DataType::Float32 || labelsInfo.dtype != DataType::Float32)
		{
			throw std::runtime_error("CrossEntropyLossBackward requires Float32 logits and labels");
		}
		if (logitsInfo.shape.empty() || logitsInfo.shape.back() == 0)
		{
			throw std::runtime_error("CrossEntropyLossBackward expects shape [..., classes]");
		}
		if (logitsInfo.shape != labelsInfo.shape)
		{
			throw std::runtime_error("CrossEntropyLossBackward logits and labels shapes must match");
		}
		const auto result = subgraph.AddNode(CrossEntropyLossBackwardNode{ grad, logits, labels },
		                                    { OutputInfo{ DataType::Float32, logitsInfo.shape } });
		return { result, 0 };
	}

	inline SubgraphId BuildCrossEntropyLoss(Graph& graph, ShapeView logitsShape)
	{
		Subgraph subgraph;
		const auto logits = subgraph.AddParam(DataType::Float32, logitsShape.ToOwned());
		const auto labels = subgraph.AddParam(DataType::Float32, logitsShape.ToOwned());
		const auto result = AddCrossEntropyLoss(subgraph, { logits, 0 }, { labels, 0 });
		subgraph.SetResults({ result });
		return graph.AddSubgraph(std::move(subgraph));
	}
} // namespace LiteNN::Layer

#endif
