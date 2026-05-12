#include <LiteNN/Compiler/CompiledModule.h>
#include <LiteNN/Graph.h>

#include <string>

#ifndef LITENN_COMPILER_DUMP_H
#define LITENN_COMPILER_DUMP_H

namespace LiteNN::Debug
{
	enum class MLIRDumpStage
	{
		InputDialect,
		AfterLowering,
		AfterBufferization,
		AfterLLVMCodegen,
	};

	std::string DumpMLIR(const Graph& graph, MLIRDumpStage stage = MLIRDumpStage::InputDialect);
	std::string DumpCompiledModuleMetadata(const CompiledModuleArtifact& artifact);
	std::string DumpCompiledModuleMetadata(const CompiledModule<CPU>& module);
#ifdef LITENN_ENABLE_CUDA
	std::string DumpCompiledModuleMetadata(const CompiledModule<CUDA>& module);
#endif
} // namespace LiteNN::Debug

#endif
