# LiteNN MNIST Example

This example contains two MNIST classifiers that share the same data loading,
training, and graph construction code:

- `litenn_mnist_interpreter`: trains with `Runtime::Interpreter::RunBackward`
  and SGD, then evaluates with `Runtime::Interpreter`.
- `litenn_mnist_aot`: trains the same graph with `RunBackward`, copies the
  trained parameters into a forward-only inference graph, then compiles that
  graph with `Compiler<CPU>`, reloads it from rodata/instruction addresses with
  `CompiledModule<CPU>::Load`, and evaluates the loaded module.

The LiteNN graph is a trainable linear classifier:

```text
logits = image @ weight + bias
```

Graph construction uses `LiteNN::Layer::CreateLinear` / `AddLinear`. Training
uses `LiteNN::Training::CPUTrainer`, `Optimizer::SoftmaxCrossEntropyWithLogits`,
and `Optimizer::SGD`.
Weight parameters are initialized with `LiteNN::Initializer::XavierUniform`,
and bias parameters use `Initializer::Zeros`. The loss helper computes
`dLoss/dLogits`, then `CPUTrainer` passes the gradient to the LiteNN backward graph:

```text
CPUTrainer::StepSoftmaxCrossEntropy([image], label)
  -> RunForward(graph, image)
  -> RunBackward(graph, [image, grad_logits])
  -> StoreVariableGradients
  -> Optimizer::SGD::Step
```

After training, both examples call `ExtractForwardOnlyGraph` so inference and
AOT compilation use a forward-only graph with the trained weights and without
backward/activation-save nodes. The graph exposes named signatures:
`image -> logits`.

## Dataset

The repository example expects MNIST IDX files under `example/mnist/data`:

```text
train-images.idx3-ubyte
train-labels.idx1-ubyte
t10k-images.idx3-ubyte
t10k-labels.idx1-ubyte
```

The loader also accepts the common dash-separated file names
`train-images-idx3-ubyte`, `train-labels-idx1-ubyte`,
`t10k-images-idx3-ubyte`, and `t10k-labels-idx1-ubyte`.

## Build

From the repository root:

```powershell
cmake -S . -B build -DLITENN_ENABLE_MLIR=ON
cmake --build build --target litenn_mnist
```

`litenn_mnist_aot` is only built when `LiteNNCompiler` is available, which
requires `LITENN_ENABLE_MLIR=ON`. Without MLIR, the interpreter executable is
still available.

## Run

Interpreter training and inference:

```powershell
build\example\mnist\litenn_mnist_interpreter.exe --epochs 3 --train-limit 1000 --test-limit 1000
```

AOT compile/load after training:

```powershell
build\example\mnist\litenn_mnist_aot.exe --epochs 3 --train-limit 1000 --test-limit 1000
```

Write a carrier object while running the AOT example:

```powershell
build\example\mnist\litenn_mnist_aot.exe --epochs 3 --train-limit 1000 --test-limit 1000 --object build\mnist_module.o
```

Common options:

```text
--data <dir>          Directory containing MNIST IDX files.
--train-limit <n>     Maximum training images used with Backward/SGD. Default: 1000.
--test-limit <n>      Maximum test images evaluated. Default: 1000.
--epochs <n>          Training epochs. Default: 3.
--learning-rate <x>   SGD learning rate. Default: 0.05.
--seed <n>            Parameter initializer seed. Default: 42.
--show-samples <n>    Print the first n predictions and logits.
```
