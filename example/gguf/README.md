# LiteNN GGUF Conversion Example

This example demonstrates the two supported GGUF stages:

- `import`: read a GGUF file into a LiteNN weight archive.
- `lower-llama`: import and lower a LLaMA-family GGUF archive into an executable LiteNN graph for fixed-length prefill.

Build:

```powershell
cmake --build build --parallel
```

Run the self-contained C++ example:

```powershell
build\example\gguf\litenn_gguf_conversion_example.exe
```

The program creates a tiny GGUF fixture, imports it, lowers it, saves both `.ltnn` files, and runs the lowered graph with the CPU interpreter.

Command-line conversion for a real file:

```powershell
build\tools\gguf\litenn_gguf_convert.exe --import model.gguf model.archive.ltnn
build\tools\gguf\litenn_gguf_convert.exe --lower-llama model.gguf model.prefill.ltnn 16
build\tools\gguf\litenn_gguf_convert.exe --lower-llama model.gguf model.segment.ltnn 4 16
build\tools\gguf\litenn_gguf_convert.exe --lower-llama-decode model.gguf model.decode.ltnn 1 16
```

When the MLIR compiler is enabled, converted or lowered `.ltnn` graphs can be
emitted as carrier objects with exported rodata/instruction symbols:

```powershell
build\tools\gguf\litenn_gguf_convert.exe --compile-cpu model.decode.ltnn model.decode.cpu.o litenn_llama_decode
build\tools\gguf\litenn_gguf_convert.exe --compile-cuda model.decode.ltnn model.decode.cuda.o litenn_llama_decode
```

Current scope: decode graphs expose static-shape KV cache inputs and updated-cache outputs. Dynamic cache growth and llama.cpp golden-logit validation are still tracked in `docs/Roadmap.md`.
