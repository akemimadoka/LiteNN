#ifndef LITENN_MODEL_BUILDER_H
#define LITENN_MODEL_BUILDER_H

#include <LiteNN/ExecutablePlan.h>

#include <memory>
#include <utility>

namespace LiteNN
{
	class ModelBuilder
	{
	public:
		ModelBuilder() = default;

		explicit ModelBuilder(ModelGraph model) : model_(std::move(model))
		{
		}

		ModelGraph& Model() noexcept
		{
			return model_;
		}

		const ModelGraph& Model() const noexcept
		{
			return model_;
		}

		Graph& UnsafeMutableGraph() noexcept
		{
			return model_.UnsafeMutableGraph();
		}

		const Graph& UnsafeGraphView() const noexcept
		{
			return model_.UnsafeGraphView();
		}

		Graph UnsafeTakeGraph() noexcept
		{
			return model_.UnsafeTakeGraph();
		}

		ExecutablePlan BuildExecutablePlan(const OpSchemaRegistry& registry = DefaultOpSchemaRegistry()) const
		{
			return LiteNN::BuildExecutablePlan(model_, registry);
		}

		std::size_t AddVariable(std::shared_ptr<Variable> variable)
		{
			return UnsafeMutableGraph().AddVariable(std::move(variable));
		}

		void SetVariableName(std::size_t variableIndex, std::string name)
		{
			UnsafeMutableGraph().SetVariableName(variableIndex, std::move(name));
		}

		SubgraphId AddSubgraph(Subgraph subgraph)
		{
			return UnsafeMutableGraph().AddSubgraph(std::move(subgraph));
		}

		void SetForward(SubgraphId subgraph)
		{
			UnsafeMutableGraph().SetForward(subgraph);
		}

		void SetBackward(SubgraphId subgraph)
		{
			UnsafeMutableGraph().SetBackward(subgraph);
		}

	private:
		ModelGraph model_;
	};
} // namespace LiteNN

#endif
