#include "graph_common.h"

#include <cstdint>
#include <exception>
#include <iostream>

extern "C"
{
	extern const unsigned char litenn_example_module_rodata[];
	extern const std::uint64_t litenn_example_module_rodata_size;
	extern const unsigned char litenn_example_module_instructions[];
	extern const std::uint64_t litenn_example_module_instructions_size;
}

int main()
{
	using namespace LiteNN;
	using namespace LiteNN::Examples::Carrier;

	try
	{
		auto artifact = CompiledModuleArtifact::FromExportedSymbols({
		    .rodata = litenn_example_module_rodata,
		    .rodataSize = &litenn_example_module_rodata_size,
		    .instructions = litenn_example_module_instructions,
		    .instructionSize = &litenn_example_module_instructions_size,
		});
		auto module = artifact.Load();
		auto inputs = MakeSampleInputs();
		auto outputs = module.Run(inputs);
		VerifySampleOutputs(outputs);
		PrintRunSummary("Static", module, outputs);
		return 0;
	}
	catch (const std::exception& ex)
	{
		std::cerr << "error: " << ex.what() << '\n';
		return 1;
	}
}