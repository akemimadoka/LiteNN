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

		explicit ModelBuilder(ModelGraph model) : model_(std::move(model)) {}

		ModelGraph& Model() noexcept
		{
			return model_;
		}

		const ModelGraph& Model() const noexcept
		{
			return model_;
		}

		Graph& MutableGraph() noexcept
		{
			return model_.MutableGraph();
		}

		const Graph& GraphView() const noexcept
		{
			return model_.GraphView();
		}

		Graph TakeGraph() noexcept
		{
			return model_.TakeGraph();
		}

		std::size_t AddVariable(std::shared_ptr<Variable> variable)
		{
			return MutableGraph().AddVariable(std::move(variable));
		}

		void SetVariableName(std::size_t variableIndex, std::string name)
		{
			MutableGraph().SetVariableName(variableIndex, std::move(name));
		}

		SubgraphId AddSubgraph(Subgraph subgraph)
		{
			return MutableGraph().AddSubgraph(std::move(subgraph));
		}

		void SetForward(SubgraphId subgraph)
		{
			MutableGraph().SetForward(subgraph);
		}

		void SetBackward(SubgraphId subgraph)
		{
			MutableGraph().SetBackward(subgraph);
		}

	private:
		ModelGraph model_;
	};
} // namespace LiteNN

#endif
