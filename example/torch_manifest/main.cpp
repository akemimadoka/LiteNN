#include <LiteNN.h>
#include <LiteNN/Runtime/Interpreter.h>
#include <LiteNN/Serialization/TorchManifest.h>

#ifdef LITENN_ENABLE_MLIR
#include <LiteNN/Compiler/CompiledModule.h>
#endif

#include <cstddef>
#include <exception>
#include <filesystem>
#include <iostream>
#include <span>
#include <string_view>
#include <vector>

namespace
{
	float ReadFloat(const LiteNN::Tensor<LiteNN::CPU>& tensor, std::size_t index)
	{
		return static_cast<const float*>(tensor.RawData())[index];
	}

	void PrintTensor(std::string_view label, const LiteNN::Tensor<LiteNN::CPU>& tensor)
	{
		std::cout << label << ": [";
		for (std::size_t i = 0; i < tensor.NumElements(); ++i)
		{
			if (i != 0)
			{
				std::cout << ", ";
			}
			std::cout << ReadFloat(tensor, i);
		}
		std::cout << "]\n";
	}

	void PrintReport(const LiteNN::Serialization::TorchManifestReport& report)
	{
		const auto printGroup = [](std::string_view title, const std::vector<std::string>& entries) {
			if (entries.empty())
			{
				return;
			}
			std::cout << title << ":\n";
			for (const auto& entry : entries)
			{
				std::cout << "  - " << entry << '\n';
			}
		};
		printGroup("Imported tensors", report.importedTensors);
		printGroup("Lowered ops", report.loweredOps);
		printGroup("Folded constants", report.foldedConstants);
	}

	void PrintUsage(const char* argv0)
	{
		std::cerr << "Usage: " << argv0 << " <manifest.json> <weights.safetensors>\n"
		          << "Generate the fixture first with: python311 export_fixture.py\n";
	}
} // namespace

int main(int argc, char** argv)
{
	try
	{
		if (argc != 3)
		{
			PrintUsage(argv[0]);
			return 1;
		}

		const std::filesystem::path manifestPath = argv[1];
		const std::filesystem::path weightsPath = argv[2];
		auto imported = LiteNN::Serialization::LoadTorchManifest(manifestPath, weightsPath);
		PrintReport(imported.report);

		std::vector<LiteNN::Tensor<LiteNN::CPU>> inputs;
		inputs.emplace_back(LiteNN::Tensor<LiteNN::CPU>({
		                                                    1.0, -2.0, 0.5,
		                                                    0.0, 3.0, -1.0,
		                                                },
		                                                { 2, 3 }));

		LiteNN::Runtime::Interpreter<LiteNN::CPU> interpreter;
		const auto interpreted = interpreter.RunForward(imported.graph, inputs);
		PrintTensor("Interpreter output", interpreted[0]);

#ifdef LITENN_ENABLE_MLIR
		auto compiled = LiteNN::Compiler<LiteNN::CPU>::Compile(
		    imported.graph, LiteNN::CompilerOptions::FromEnvironment());
		const auto compiledOutputs = compiled.Run(std::span<const LiteNN::Tensor<LiteNN::CPU>>(inputs));
		PrintTensor("CPU AOT output", compiledOutputs[0]);
#else
		std::cout << "CPU AOT output skipped because LiteNNCompiler is not available in this build.\n";
#endif
		return 0;
	}
	catch (const std::exception& ex)
	{
		std::cerr << "litenn_torch_manifest_example: " << ex.what() << '\n';
		return 1;
	}
}
