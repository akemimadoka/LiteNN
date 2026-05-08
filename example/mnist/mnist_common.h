#pragma once

#include <LiteNN.h>
#include <LiteNN/Initializer/Initializer.h>
#include <LiteNN/Layer/Linear.h>
#include <LiteNN/Optimizer/Loss.h>
#include <LiteNN/Optimizer/SGD.h>
#include <LiteNN/Pass/AutogradPass.h>
#include <LiteNN/Pass/ForwardOnlyPass.h>
#include <LiteNN/Runtime/Interpreter.h>
#include <LiteNN/Training/Trainer.h>

#include <algorithm>
#include <array>
#include <cstdint>
#include <filesystem>
#include <format>
#include <fstream>
#include <initializer_list>
#include <iostream>
#include <limits>
#include <random>
#include <span>
#include <stdexcept>
#include <string>
#include <string_view>
#include <utility>
#include <vector>

#ifndef LITENN_MNIST_DEFAULT_DATA_DIR
#define LITENN_MNIST_DEFAULT_DATA_DIR "example/mnist/data"
#endif

namespace LiteNN::Examples::Mnist
{
	constexpr std::size_t kDigitCount = 10;
	constexpr std::size_t kMnistRows = 28;
	constexpr std::size_t kMnistCols = 28;
	constexpr std::size_t kMnistPixels = kMnistRows * kMnistCols;
	constexpr std::size_t kWeightVariableIndex = 0;
	constexpr std::size_t kBiasVariableIndex = 1;

	struct Options
	{
		std::filesystem::path dataDir = LITENN_MNIST_DEFAULT_DATA_DIR;
		std::size_t trainLimit = 1000;
		std::size_t testLimit = 1000;
		std::size_t showSamples = 5;
		std::size_t epochs = 3;
		float learningRate = 0.05f;
		std::uint32_t seed = 42;
	};

	struct MnistSplit
	{
		std::vector<float> images;
		std::vector<std::uint8_t> labels;
		std::size_t rows{};
		std::size_t cols{};

		std::size_t Count() const
		{
			return labels.size();
		}

		std::span<const float> Image(std::size_t index) const
		{
			const auto imageSize = rows * cols;
			return { images.data() + index * imageSize, imageSize };
		}
	};

	inline std::uint32_t ReadBigEndianU32(std::istream& in)
	{
		std::array<unsigned char, 4> bytes{};
		in.read(reinterpret_cast<char*>(bytes.data()), bytes.size());
		if (!in)
		{
			throw std::runtime_error("Unexpected end of IDX file");
		}
		return (static_cast<std::uint32_t>(bytes[0]) << 24) |
		       (static_cast<std::uint32_t>(bytes[1]) << 16) |
		       (static_cast<std::uint32_t>(bytes[2]) << 8) |
		       static_cast<std::uint32_t>(bytes[3]);
	}

	inline std::filesystem::path ResolveMnistFile(const std::filesystem::path& dataDir,
	                                              std::initializer_list<std::string_view> names)
	{
		for (const auto name : names)
		{
			auto path = dataDir / std::string(name);
			if (std::filesystem::exists(path))
			{
				return path;
			}
		}
		return dataDir / std::string(*names.begin());
	}

	inline std::ifstream OpenBinary(const std::filesystem::path& path)
	{
		std::ifstream file(path, std::ios::binary);
		if (!file)
		{
			throw std::runtime_error(std::format("Failed to open {}", path.string()));
		}
		return file;
	}

	inline std::vector<float> LoadImages(const std::filesystem::path& path, std::size_t& count,
	                                     std::size_t& rows, std::size_t& cols, std::size_t limit)
	{
		auto file = OpenBinary(path);
		const auto magic = ReadBigEndianU32(file);
		if (magic != 2051)
		{
			throw std::runtime_error(std::format("{} is not an IDX image file", path.string()));
		}

		count = ReadBigEndianU32(file);
		rows = ReadBigEndianU32(file);
		cols = ReadBigEndianU32(file);
		count = std::min(count, limit);

		const auto imageSize = rows * cols;
		std::vector<unsigned char> raw(count * imageSize);
		file.read(reinterpret_cast<char*>(raw.data()), raw.size());
		if (!file)
		{
			throw std::runtime_error(std::format("Failed to read {} images from {}", count, path.string()));
		}

		std::vector<float> images(raw.size());
		std::ranges::transform(raw, images.begin(), [](unsigned char pixel) {
			return static_cast<float>(pixel) / 255.0f;
		});
		return images;
	}

	inline std::vector<std::uint8_t> LoadLabels(const std::filesystem::path& path, std::size_t limit)
	{
		auto file = OpenBinary(path);
		const auto magic = ReadBigEndianU32(file);
		if (magic != 2049)
		{
			throw std::runtime_error(std::format("{} is not an IDX label file", path.string()));
		}

		auto count = static_cast<std::size_t>(ReadBigEndianU32(file));
		count = std::min(count, limit);
		std::vector<std::uint8_t> labels(count);
		file.read(reinterpret_cast<char*>(labels.data()), labels.size());
		if (!file)
		{
			throw std::runtime_error(std::format("Failed to read {} labels from {}", count, path.string()));
		}
		return labels;
	}

	inline MnistSplit LoadSplit(const std::filesystem::path& dataDir,
	                            std::initializer_list<std::string_view> imageNames,
	                            std::initializer_list<std::string_view> labelNames, std::size_t limit)
	{
		MnistSplit split;
		std::size_t imageCount{};
		split.images = LoadImages(ResolveMnistFile(dataDir, imageNames), imageCount, split.rows, split.cols, limit);
		split.labels = LoadLabels(ResolveMnistFile(dataDir, labelNames), limit);
		if (split.labels.size() != imageCount)
		{
			throw std::runtime_error("MNIST image and label counts do not match");
		}
		if (split.rows != kMnistRows || split.cols != kMnistCols)
		{
			throw std::runtime_error("This example expects 28x28 MNIST images");
		}
		if (split.Count() == 0)
		{
			throw std::runtime_error("MNIST split is empty");
		}
		return split;
	}

	inline MnistSplit LoadTrainSplit(const Options& options)
	{
		return LoadSplit(options.dataDir, { "train-images.idx3-ubyte", "train-images-idx3-ubyte" },
		                 { "train-labels.idx1-ubyte", "train-labels-idx1-ubyte" }, options.trainLimit);
	}

	inline MnistSplit LoadTestSplit(const Options& options)
	{
		return LoadSplit(options.dataDir, { "t10k-images.idx3-ubyte", "t10k-images-idx3-ubyte" },
		                 { "t10k-labels.idx1-ubyte", "t10k-labels-idx1-ubyte" }, options.testLimit);
	}

	inline Tensor<CPU> MakeFloatTensor(std::span<const float> values, ShapeView shape)
	{
		return Optimizer::MakeFloatTensor(values, shape);
	}

	inline Graph BuildMnistGraphWithParameters(Tensor<CPU> weightTensor, Tensor<CPU> biasTensor)
	{
		if (weightTensor.DType() != DataType::Float32 || biasTensor.DType() != DataType::Float32 ||
		    weightTensor.Shape().NumDim() != 2 || weightTensor.Shape()[0] != kMnistPixels ||
		    weightTensor.Shape()[1] != kDigitCount || biasTensor.Shape().NumDim() != 2 || biasTensor.Shape()[0] != 1 ||
		    biasTensor.Shape()[1] != kDigitCount)
		{
			throw std::runtime_error("Unexpected MNIST parameter tensor shape or dtype");
		}

		Graph graph;
		const auto classifier = Layer::CreateLinear(graph, std::move(weightTensor), std::move(biasTensor));
		if (classifier.weightVariable != kWeightVariableIndex || !classifier.biasVariable ||
		    *classifier.biasVariable != kBiasVariableIndex)
		{
			throw std::runtime_error("Unexpected MNIST linear variable layout");
		}

		Subgraph forward;
		const auto image = forward.AddParam(DataType::Float32, { 1, kMnistPixels });
		const auto logits = Layer::AddLinear(forward, classifier, { image, 0 });
		forward.SetResults({ logits });

		const auto forwardId = graph.AddSubgraph(std::move(forward));
		graph.SetForward(forwardId);
		graph.SetInputNames({ "image" });
		graph.SetOutputNames({ "logits" });
		return graph;
	}

	inline Graph BuildTrainableMnistGraph(std::uint32_t seed = 42)
	{
		std::mt19937 rng(seed);
		return BuildMnistGraphWithParameters(Initializer::XavierUniform({ kMnistPixels, kDigitCount }, rng),
		                                     Initializer::Zeros({ 1, kDigitCount }));
	}

	inline Graph BuildInferenceGraphFromTrainedVariables(const Graph& trainedGraph)
	{
		return ExtractForwardOnlyGraph(trainedGraph);
	}

	inline const float* FloatData(const Tensor<CPU>& tensor)
	{
		return static_cast<const float*>(tensor.RawData());
	}

	inline int ArgMaxLogits(const Tensor<CPU>& logits)
	{
		const auto* data = FloatData(logits);
		return static_cast<int>(std::max_element(data, data + kDigitCount) - data);
	}

	inline void TrainMnistGraph(Graph& graph, const MnistSplit& train, const Options& options)
	{
		Training::CPUTrainer<Optimizer::SGD> trainer(
		    graph, Optimizer::SGD(Optimizer::SGDOptions{ .learningRate = options.learningRate }));

		for (std::size_t epoch = 0; epoch < options.epochs; ++epoch)
		{
			double totalLoss = 0.0;
			std::size_t correct = 0;

			for (std::size_t i = 0; i < train.Count(); ++i)
			{
				const auto image = train.Image(i);
				const auto label = train.labels[i];

				auto forwardImage = MakeFloatTensor(image, { 1, kMnistPixels });
				std::array<Tensor<CPU>, 1> forwardInputs = { std::move(forwardImage) };
				auto step = trainer.StepSoftmaxCrossEntropy(forwardInputs, label);
				if (step.outputs.size() != 1)
				{
					throw std::runtime_error("MNIST graph returned an unexpected output count");
				}

				correct += ArgMaxLogits(step.outputs[0]) == static_cast<int>(label) ? 1 : 0;
				totalLoss += step.loss;
			}

			const auto accuracy = 100.0 * static_cast<double>(correct) / static_cast<double>(train.Count());
			const auto averageLoss = totalLoss / static_cast<double>(train.Count());
			std::cout << std::format("epoch {}/{}: loss={:.4f}, train_accuracy={:.2f}%\n",
			                         epoch + 1, options.epochs, averageLoss, accuracy);
		}
	}

	inline void PrintLogits(const Tensor<CPU>& logits)
	{
		const auto* data = FloatData(logits);
		for (std::size_t i = 0; i < kDigitCount; ++i)
		{
			std::cout << std::format("{}{:.3f}", i == 0 ? "" : ", ", data[i]);
		}
	}

	inline std::size_t ParseSize(std::string_view value, std::string_view option)
	{
		std::size_t parsed{};
		std::size_t pos{};
		const auto text = std::string(value);
		parsed = std::stoull(text, &pos);
		if (pos != text.size())
		{
			throw std::runtime_error(std::format("Invalid value for {}", option));
		}
		return parsed;
	}

	inline float ParseFloat(std::string_view value, std::string_view option)
	{
		std::size_t pos{};
		const auto text = std::string(value);
		const auto parsed = std::stof(text, &pos);
		if (pos != text.size())
		{
			throw std::runtime_error(std::format("Invalid value for {}", option));
		}
		return parsed;
	}

	inline std::string_view RequireValue(int& index, int argc, char** argv, std::string_view option)
	{
		if (index + 1 >= argc)
		{
			throw std::runtime_error(std::format("Missing value for {}", option));
		}
		return argv[++index];
	}

	inline bool ParseCommonOption(std::string_view arg, int& index, int argc, char** argv, Options& options)
	{
		if (arg == "--data")
		{
			options.dataDir = std::string(RequireValue(index, argc, argv, arg));
			return true;
		}
		if (arg == "--train-limit")
		{
			options.trainLimit = ParseSize(RequireValue(index, argc, argv, arg), arg);
			return true;
		}
		if (arg == "--test-limit")
		{
			options.testLimit = ParseSize(RequireValue(index, argc, argv, arg), arg);
			return true;
		}
		if (arg == "--show-samples")
		{
			options.showSamples = ParseSize(RequireValue(index, argc, argv, arg), arg);
			return true;
		}
		if (arg == "--epochs")
		{
			options.epochs = ParseSize(RequireValue(index, argc, argv, arg), arg);
			if (options.epochs == 0)
			{
				throw std::runtime_error("--epochs must be greater than zero");
			}
			return true;
		}
		if (arg == "--learning-rate")
		{
			options.learningRate = ParseFloat(RequireValue(index, argc, argv, arg), arg);
			if (!(options.learningRate > 0.0f))
			{
				throw std::runtime_error("--learning-rate must be greater than zero");
			}
			return true;
		}
		if (arg == "--seed")
		{
			const auto seed = ParseSize(RequireValue(index, argc, argv, arg), arg);
			if (seed > std::numeric_limits<std::uint32_t>::max())
			{
				throw std::runtime_error("--seed is out of range for uint32");
			}
			options.seed = static_cast<std::uint32_t>(seed);
			return true;
		}
		if (!arg.starts_with("-"))
		{
			options.dataDir = std::string(arg);
			return true;
		}
		return false;
	}

	inline void PrintCommonOptions()
	{
		std::cout <<
		    "Common options:\n"
		    "  --data <dir>          Directory containing MNIST IDX files.\n"
		    "  --train-limit <n>     Maximum training images used with Backward/SGD. Default: 1000.\n"
		    "  --test-limit <n>      Maximum test images evaluated. Default: 1000.\n"
		    "  --epochs <n>          Training epochs. Default: 3.\n"
		    "  --learning-rate <x>   SGD learning rate. Default: 0.05.\n"
		    "  --seed <n>            Parameter initializer seed. Default: 42.\n"
		    "  --show-samples <n>    Print the first n predictions and logits.\n";
	}

	inline void PrintAccuracy(std::size_t correct, std::size_t total)
	{
		const auto accuracy = 100.0 * static_cast<double>(correct) / static_cast<double>(total);
		std::cout << std::format("accuracy: {}/{} = {:.2f}%\n", correct, total, accuracy);
	}

	template <typename Runner>
	std::size_t Evaluate(const MnistSplit& test, std::size_t showSamples, Runner&& runner)
	{
		std::size_t correct = 0;
		for (std::size_t i = 0; i < test.Count(); ++i)
		{
			auto outputs = runner(MakeFloatTensor(test.Image(i), { 1, kMnistPixels }));
			if (outputs.size() != 1)
			{
				throw std::runtime_error("MNIST graph returned an unexpected output count");
			}

			const auto prediction = ArgMaxLogits(outputs[0]);
			const auto label = static_cast<int>(test.labels[i]);
			correct += prediction == label ? 1 : 0;

			if (i < showSamples)
			{
				std::cout << std::format("sample {:>4}: predicted={}, label={}, logits=[", i, prediction, label);
				PrintLogits(outputs[0]);
				std::cout << "]\n";
			}
		}
		return correct;
	}
} // namespace LiteNN::Examples::Mnist
