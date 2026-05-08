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
		    "       {} [--hidden-size N] [--save PATH] [--load PATH]\n"
		    "\n"
		    "Trains MNIST classification with LiteNN Runtime::Interpreter Backward/SGD, then evaluates it.\n"
		    "  --hidden-size N  Use a 2-layer MLP (input->N->10) instead of the default linear model.\n"
		    "  --save PATH      After training, save the inference model to PATH.\n"
		    "  --load PATH      Skip training; load a previously-saved inference model from PATH.\n"
		    "Default data directory: {}\n\n",
		    exe, exe, std::filesystem::path(LITENN_MNIST_DEFAULT_DATA_DIR).string());
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
		Runtime::Interpreter<CPU> interpreter;
		Graph inferenceGraph;

		if (options.loadModelPath)
		{
			// 跳过训练，直接从文件加载推理图
			inferenceGraph = LoadMnistInferenceModel(*options.loadModelPath);
		}
		else
		{
			const auto useMlp = options.hiddenSize > 0;
			const auto hiddenSize = useMlp ? options.hiddenSize : std::size_t{0};

			std::cout << std::format("Loading MNIST from {}\n", options.dataDir.string());
			const auto train = LoadTrainSplit(options);
			const auto test = LoadTestSplit(options);

			if (useMlp)
			{
				std::cout << std::format("Training 2-layer MLP (784->{}->10) with {} images\n",
				                         hiddenSize, train.Count());
				auto trainingGraph = BuildTrainableMlpGraph(options.seed, hiddenSize);
				TrainMnistGraph(trainingGraph, train, options);

				if (options.saveModelPath)
				{
					SaveMnistModel(trainingGraph, *options.saveModelPath);
				}
				inferenceGraph = BuildInferenceGraphFromTrainedVariables(trainingGraph);
			}
			else
			{
				std::cout << std::format("Training linear softmax classifier with {} images\n", train.Count());
				auto trainingGraph = BuildTrainableMnistGraph(options.seed);
				TrainMnistGraph(trainingGraph, train, options);

				if (options.saveModelPath)
				{
					SaveMnistModel(trainingGraph, *options.saveModelPath);
				}
				inferenceGraph = BuildInferenceGraphFromTrainedVariables(trainingGraph);
			}

			const auto correct = Evaluate(test, options.showSamples, [&](Tensor<CPU> image) {
				std::array<Tensor<CPU>, 1> inputs = { std::move(image) };
				return interpreter.RunForward(inferenceGraph, inputs);
			});
			PrintAccuracy(correct, test.Count());
			return 0;
		}

		// load 路径：需要单独加载测试集
		std::cout << std::format("Loading MNIST test split from {}\n", options.dataDir.string());
		const auto test = LoadTestSplit(options);
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
