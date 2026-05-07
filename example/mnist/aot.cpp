#include "mnist_common.h"

#include <LiteNN/Compiler/CompiledModule.h>

#include <array>
#include <cstdlib>
#include <exception>
#include <filesystem>
#include <format>
#include <iostream>
#include <string_view>

namespace
{
	using namespace LiteNN;
	using namespace LiteNN::Examples::Mnist;

	struct AotOptions
	{
		Options mnist;
		std::filesystem::path objectPath;
		bool writeObject{};
	};

	void PrintUsage(std::string_view exe)
	{
		std::cout << std::format(
		    "Usage: {} [--data DIR] [--train-limit N] [--test-limit N] [--epochs N] [--learning-rate X] [--seed N] [--object PATH]\n"
		    "\n"
		    "Trains with LiteNN Backward/SGD, then compiles the trained graph with AOT and reloads it.\n"
		    "Default data directory: {}\n\n",
		    exe, std::filesystem::path(LITENN_MNIST_DEFAULT_DATA_DIR).string());
		PrintCommonOptions();
		std::cout <<
		    "AOT options:\n"
		    "  --object <path>       Also write a carrier object with rodata/instruction symbols.\n";
	}

	AotOptions ParseArgs(int argc, char** argv)
	{
		AotOptions options;
		for (int i = 1; i < argc; ++i)
		{
			const std::string_view arg = argv[i];
			if (arg == "--help" || arg == "-h")
			{
				PrintUsage(argv[0]);
				std::exit(0);
			}
			if (arg == "--object")
			{
				options.objectPath = std::string(RequireValue(i, argc, argv, arg));
				options.writeObject = true;
				continue;
			}
			if (!ParseCommonOption(arg, i, argc, argv, options.mnist))
			{
				throw std::runtime_error(std::format("Unknown argument: {}", arg));
			}
		}
		return options;
	}

	CompiledModule<CPU> CompileAndLoadFromImage(const Graph& graph, const AotOptions& options)
	{
		std::cout << "Compiling graph with LiteNN AOT\n";
		auto compiled = Compiler<CPU>::Compile(graph);

		if (options.writeObject)
		{
			compiled.WriteObjectFile(options.objectPath, "litenn_mnist_module");
			std::cout << std::format("Wrote carrier object to {}\n", options.objectPath.string());
		}

		const auto image = compiled.Image();
		std::cout << std::format("Loading compiled module from rodata={} bytes, instructions={} bytes\n",
		                         image.rodataSize, image.instructionSize);

		return CompiledModule<CPU>::Load({
		    .rodata = image.rodata,
		    .rodataSize = image.rodataSize,
		    .instructions = image.instructions,
		    .instructionSize = image.instructionSize,
		});
	}

	int Run(const AotOptions& options)
	{
		std::cout << std::format("Loading MNIST from {}\n", options.mnist.dataDir.string());
		const auto train = LoadTrainSplit(options.mnist);
		const auto test = LoadTestSplit(options.mnist);

		std::cout << std::format("Training linear softmax classifier with {} images\n", train.Count());
		auto trainingGraph = BuildTrainableMnistGraph(options.mnist.seed);
		TrainMnistGraph(trainingGraph, train, options.mnist);

		auto inferenceGraph = BuildInferenceGraphFromTrainedVariables(trainingGraph);
		auto module = CompileAndLoadFromImage(inferenceGraph, options);

		const auto correct = Evaluate(test, options.mnist.showSamples, [&](Tensor<CPU> image) {
			std::array<Tensor<CPU>, 1> inputs = { std::move(image) };
			return module.Run(inputs);
		});

		PrintAccuracy(correct, test.Count());
		return 0;
	}
} // namespace

int main(int argc, char** argv)
{
	try
	{
		return Run(ParseArgs(argc, argv));
	}
	catch (const std::exception& ex)
	{
		std::cerr << "error: " << ex.what() << "\n\n";
		PrintUsage(argv[0]);
		return 1;
	}
}
