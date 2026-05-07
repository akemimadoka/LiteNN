#include "mnist_common.h"

#include <LiteNN/Runtime/Interpreter.h>

#include <array>
#include <cstdlib>
#include <exception>
#include <format>
#include <iostream>
#include <string_view>

namespace
{
	using namespace LiteNN;
	using namespace LiteNN::Examples::Mnist;

	void PrintUsage(std::string_view exe)
	{
		std::cout << std::format(
		    "Usage: {} [--data DIR] [--train-limit N] [--test-limit N] [--epochs N] [--learning-rate X] [--seed N]\n"
		    "\n"
		    "Trains MNIST classification with LiteNN Runtime::Interpreter Backward/SGD, then evaluates it.\n"
		    "Default data directory: {}\n\n",
		    exe, std::filesystem::path(LITENN_MNIST_DEFAULT_DATA_DIR).string());
		PrintCommonOptions();
	}

	Options ParseArgs(int argc, char** argv)
	{
		Options options;
		for (int i = 1; i < argc; ++i)
		{
			const std::string_view arg = argv[i];
			if (arg == "--help" || arg == "-h")
			{
				PrintUsage(argv[0]);
				std::exit(0);
			}
			if (!ParseCommonOption(arg, i, argc, argv, options))
			{
				throw std::runtime_error(std::format("Unknown argument: {}", arg));
			}
		}
		return options;
	}

	int Run(const Options& options)
	{
		std::cout << std::format("Loading MNIST from {}\n", options.dataDir.string());
		const auto train = LoadTrainSplit(options);
		const auto test = LoadTestSplit(options);

		std::cout << std::format("Training linear softmax classifier with {} images\n", train.Count());
		auto trainingGraph = BuildTrainableMnistGraph(options.seed);
		TrainMnistGraph(trainingGraph, train, options);

		auto inferenceGraph = BuildInferenceGraphFromTrainedVariables(trainingGraph);
		Runtime::Interpreter<CPU> interpreter;

		const auto correct = Evaluate(test, options.showSamples, [&](Tensor<CPU> image) {
			std::array<Tensor<CPU>, 1> inputs = { std::move(image) };
			return interpreter.RunForward(inferenceGraph, inputs);
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
