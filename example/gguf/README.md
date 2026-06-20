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

The program creates a tiny GGUF fixture, imports it, lowers it, writes vNext
package manifests for the imported/prefill/decode graphs, and runs the lowered
graph with the CPU interpreter.

Command-line conversion for a real file:

```powershell
build\tools\gguf\litenn_gguf_convert.exe --import model.gguf model.archive.ltnn
build\tools\gguf\litenn_gguf_convert.exe --lower-llama model.gguf model.prefill.ltnn 16
build\tools\gguf\litenn_gguf_convert.exe --lower-llama-quantized model.gguf model.prefill.quantized.ltnn model.prefill.quantized.weights.bin 16
build\tools\gguf\litenn_gguf_convert.exe --lower-llama model.gguf model.segment.ltnn 4 16
build\tools\gguf\litenn_gguf_convert.exe --lower-llama-decode model.gguf model.decode.ltnn 1 16
build\tools\gguf\litenn_gguf_convert.exe --run-llama-token-ids model.gguf 1,2,3,4
build\tools\gguf\litenn_gguf_convert.exe --run-llama-prompt model.gguf "hello"
build\tools\gguf\litenn_gguf_convert.exe --run-llama-package-token-ids model.prefill.quantized.ltnn 1,2,3,4
build\tools\gguf\litenn_gguf_convert.exe --run-llama-decode-loop-token-id model.gguf 1 8 generated_token_ids.txt
build\tools\gguf\litenn_gguf_convert.exe --run-llama-decode-loop-token-id model.gguf 1 8 --sample random --temperature 0.7 --top-k 40 --top-p 0.9 --repeat-penalty 1.1 --seed 42 --output generated_token_ids.txt
```

`--run-llama-token-ids` is a token-id level smoke path. It imports the GGUF file,
lowers a fixed-length prefill graph with quantized weights preserved, executes
with the CPU interpreter plus the GGML quantized MatMul adapter, and prints the
logits shape plus the greedy next token. Production tokenizer parity and
full prompt-to-decode generation are tracked in `docs/Roadmap.md`.
`--run-llama-prompt` uses a deliberately limited exact-vocabulary tokenizer
bridge over `tokenizer.ggml.tokens`; it is useful for fixtures and diagnostics
but does not replace GPT2/BPE or llama.cpp tokenizer parity.
`--run-llama-package-token-ids` runs an already lowered `.ltnn` package and is
useful for validating conversion artifacts without re-importing the GGUF file.
`--run-llama-decode-loop-token-id` is a decode-loop smoke path: it prebuilds the
static-shape decode plans for each cache length, carries updated KV-cache tensors
between steps, and prints build/run timing, generated token ids, and tokenizer
pieces when `tokenizer.ggml.tokens` is available. If an output path is supplied,
it writes both lists to that file. The decode loop defaults to greedy sampling;
`--sample random` enables seedable temperature/top-k/top-p sampling with an
optional repeat penalty.

When the MLIR compiler is enabled, converted or lowered `.ltnn` graphs can be
emitted as carrier objects with exported rodata/instruction symbols:

```powershell
build\tools\gguf\litenn_gguf_convert.exe --compile-cpu model.decode.ltnn model.decode.cpu.o litenn_llama_decode
build\tools\gguf\litenn_gguf_convert.exe --compile-cuda model.decode.ltnn model.decode.cuda.o litenn_llama_decode
```

For large models or packaging flows that should keep metadata, constants,
weights, and instructions in separate regions, emit split carrier objects:

```powershell
build\tools\gguf\litenn_gguf_convert.exe --compile-cpu-separated model.decode.ltnn model.decode.cpu.parts litenn_llama_decode
build\tools\gguf\litenn_gguf_convert.exe --compile-cuda-separated model.decode.ltnn model.decode.cuda.parts litenn_llama_decode
```

The separated form writes `<prefix>_metadata.o`, `<prefix>_constants.o`,
`<prefix>_weights.o`, and `<prefix>_instructions.o`. This layout avoids a
single large PE/COFF section and lets embedders bind each region from a static
library, shared library, memory-mapped file, or in-memory span. Applications
that prefer raw files can use `CompiledModuleSeparatedArtifact::WriteRegionFiles`
after compiling through the C++ API.

Current scope: decode graphs expose static-shape KV cache inputs and updated-cache outputs. Dynamic cache growth and llama.cpp golden-logit validation are still tracked in `docs/Roadmap.md`.
