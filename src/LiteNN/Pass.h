#ifndef LITENN_PASS_H
#define LITENN_PASS_H

namespace LiteNN
{
	class Graph;

	// Graph → Graph 变换的基类
	struct Pass
	{
		virtual ~Pass() = default;
		virtual void Run(Graph& graph) = 0;
	};
} // namespace LiteNN

#endif
