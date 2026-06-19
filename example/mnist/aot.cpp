#include "mnist_common.h"

#include <LiteNN/Compiler/CompiledModule.h>

#include <array>
#include <cstdlib>
#include <exception>
#include <filesystem>
#include <format>
#include <iostream>
#include <optional>
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
		bool compareInterpreter{};
	};

	void PrintUsage(std::string_view exe)
	{
		std::cout << std::format(
		    "Usage: {} [--data DIR] [--train-limit N] [--test-limit N] [--epochs N] [--learning-rate X] [--seed N] "
		    "[--object PATH] [--compare-interpreter]\n"
		    "\n"
		    "Trains with LiteNN CPU AOT forward/backward/SGD, then compiles and reloads the inference graph.\n"
		    "Default data directory: {}\n\n",
		    exe, std::filesystem::path(LITENN_MNIST_DEFAULT_DATA_DIR).string());
		PrintCommonOptions();
		std::cout << "AOT options:\n"
		             "  --object <path>       Also write a carrier object with rodata/instruction symbols.\n"
		             "  --compare-interpreter Train an interpreter reference and report numerical drift.\n";
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
			if (arg == "--compare-interpreter")
			{
				options.compareInterpreter = true;
				continue;
			}
			if (!ParseCommonOption(arg, i, argc, argv, options.mnist))
			{
				throw std::runtime_error(std::format("Unknown argument: {}", arg));
			}
		}
		return options;
	}

	CompiledModule<CPU> CompileAndLoadFromArtifact(const Graph& graph, const AotOptions& options)
	{
		std::cout << "Compiling graph with LiteNN AOT\n";
		auto artifact =
		    Compiler<CPU>::CompileArtifact(Detail::BuildExecutablePlanFromGraph(graph), CompilerOptions::Defaults());

		if (options.writeObject)
		{
			artifact.WriteObjectFile(options.objectPath, "litenn_mnist_module");
			std::cout << std::format("Wrote carrier object to {}\n", options.objectPath.string());
		}

		const auto image = artifact.Image();
		std::cout << std::format("Loading compiled module from rodata={} bytes, instructions={} bytes\n",
		                         image.rodataSize, image.instructionSize);

		return artifact.Load();
	}

	int Run(const AotOptions& options)
	{
		std::cout << std::format("Loading MNIST from {}\n", options.mnist.dataDir.string());
		const auto train = LoadTrainSplit(options.mnist);
		const auto test = LoadTestSplit(options.mnist);

		std::optional<Graph> interpreterGraph;
		std::optional<TrainingSummary> interpreterSummary;
		if (options.compareInterpreter)
		{
			std::cout << std::format("Training interpreter reference using {} images\n", train.Count());
			interpreterGraph.emplace(BuildTrainableMnistGraph(options.mnist.seed));
			interpreterSummary = TrainMnistGraph(*interpreterGraph, train, options.mnist);
		}

		std::cout << std::format("Training linear softmax classifier with CPU AOT using {} images\n", train.Count());
		auto trainingGraph = BuildTrainableMnistGraph(options.mnist.seed);
		const auto aotSummary =
		    TrainMnistGraph(trainingGraph, train, options.mnist, Training::TrainExecutionPolicy::AOT);
		if (interpreterGraph && interpreterSummary)
		{
			std::cout << std::format("parity: loss_drift={:.8g}, accuracy_drift={:.8g}, max_weight_drift={:.8g}\n",
			                         std::abs(aotSummary.averageLoss - interpreterSummary->averageLoss),
			                         std::abs(aotSummary.accuracy - interpreterSummary->accuracy),
			                         MaxVariableDifference(trainingGraph, *interpreterGraph));
		}

		auto inferenceGraph = BuildInferenceGraphFromTrainedVariables(trainingGraph);
		if (options.mnist.saveModelPath)
		{
			SaveMnistModelPackage(trainingGraph, *options.mnist.saveModelPath);
		}
		auto module = CompileAndLoadFromArtifact(inferenceGraph, options);

		const auto correct = Evaluate(test, options.mnist.showSamples, [&](Tensor<CPU> image) {
			std::array<Tensor<CPU>, 1> inputs = { std::move(image) };
			return module.RunTensors(inputs);
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
