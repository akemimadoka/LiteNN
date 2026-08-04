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
build\tools\gguf\litenn_gguf_convert.exe --dump-llama-token-id-logits model.gguf 1,2,3,4 litenn_last_logits.txt
build\tools\gguf\litenn_gguf_convert.exe --run-llama-prompt model.gguf "hello"
build\tools\gguf\litenn_gguf_convert.exe --tokenize-llama-prompt model.gguf "hello" tokens.json --chat-template
build\tools\gguf\litenn_gguf_convert.exe --run-llama-package-token-ids model.prefill.quantized.ltnn 1,2,3,4
build\tools\gguf\litenn_gguf_convert.exe --run-llama-decode-loop-token-id model.gguf 1 8 generated_token_ids.txt
build\tools\gguf\litenn_gguf_convert.exe --run-llama-decode-loop-token-ids model.gguf 1,2,3,4 8 generated_token_ids.txt
build\tools\gguf\litenn_gguf_convert.exe --run-llama-decode-loop-token-id model.gguf 1 8 --sample random --temperature 0.7 --top-k 40 --top-p 0.9 --repeat-penalty 1.1 --seed 42 --output generated_token_ids.txt --logits-output final_decode_logits.txt
build\tools\gguf\litenn_gguf_convert.exe --run-llama-decode-loop-token-id model.gguf 1 8 --ignore-eos
build\tools\gguf\litenn_gguf_convert.exe --run-llama-prompt-decode-loop model.gguf "hello" 8 --sample random --seed 42 --output generated_token_ids.txt
build\tools\gguf\litenn_gguf_convert.exe --run-llama-prompt-decode-loop model.gguf "hello" 8 --chat-template --output generated_token_ids.txt
```

`--run-llama-token-ids` is a token-id level smoke path. It imports the GGUF file,
lowers a fixed-length prefill graph with quantized weights preserved, executes
with the CPU interpreter plus the GGML quantized MatMul adapter, and prints the
logits shape plus the greedy next token. Production tokenizer parity and
full prompt-to-decode generation are tracked in `docs/Roadmap.md`.
`--dump-llama-token-id-logits` runs the same fixed-length prefill path and writes
the last-token logits as `index: value` lines, matching llama-debug text output
for golden comparison tooling.
`--run-llama-prompt` uses a deliberately limited exact-vocabulary tokenizer
bridge over `tokenizer.ggml.tokens`; it is useful for fixtures and diagnostics
but does not replace GPT2/BPE or llama.cpp tokenizer parity.
When configured with `-DLITENN_ENABLE_LLAMA_CPP_TOKENIZER=ON`,
`litenn_gguf_convert` links the isolated llama.cpp tokenizer adapter and the
prompt commands can tokenize through the model's real GGUF tokenizer in-process.
`--tokenize-llama-prompt` writes the same `litenn.llamacpp_tokens.v1` JSON schema
as the standalone adapter, and `--chat-template` applies the model's default
single-user chat template before tokenization.
`--run-llama-package-token-ids` runs an already lowered `.ltnn` package and is
useful for validating conversion artifacts without re-importing the GGUF file.
`--run-llama-decode-loop-token-id` is a decode-loop smoke path: it prebuilds the
single max-capacity decode plan, compiles it once through CPU AOT with external
weight regions, advances an Int64 runtime position, carries
full-capacity KV-cache tensors between steps, and prints build/run timing,
generated token ids, and tokenizer pieces when `tokenizer.ggml.tokens` is
available. If an output path is supplied,
it writes both lists to that file. The decode loop defaults to greedy sampling;
`--sample random` enables seedable temperature/top-k/top-p sampling with an
optional repeat penalty. Generation stops early on `tokenizer.ggml.eos_token_id`
when present; pass `--ignore-eos` to force the requested number of generated
tokens by suppressing EOS during sampling, matching llama.cpp. The loop also
rejects prompts whose prompt-plus-generated length exceeds
the model context length.
`--logits-output` can be added to any decode-loop command to write the final
step's last-token logits as `index: value` lines, matching the prefill dump
format used by golden-comparison tooling.
`--logits-output-dir` writes the full-prompt position and every later decode
position as `position-NNNNNN.txt`; prompt-intermediate positions are omitted to
avoid enormous full-vocabulary dumps. The replay harness classifies the
full-prompt position as `prefill` and later positions as one-based `decodeStep`
entries in `litenn_decode_manifest.json`.
`--run-llama-decode-loop-token-ids` accepts an externally tokenized prompt as
comma-separated token ids, which is the preferred bridge for real tokenizer
parity work until the optional llama.cpp tokenizer adapter is wired in.
`--run-llama-prompt-decode-loop` feeds the exact-vocabulary prompt tokens through
the same static decode loop first, then generates the requested number of new
tokens. All GGUF model execution commands require an MLIR-enabled build and use
CPU AOT; the Interpreter remains a correctness-test reference and is not a CLI
model runtime. Large K-quant graphs currently have high first-compile latency
because quantized MatMul is expanded into generic MLIR. Set
`LITENN_COMPILE_DIAGNOSTICS=1` to print coarse CPU AOT compile phases; for large
LLM experiments, `LITENN_CPU_AOT_LLVM_OPT_LEVEL=0` is useful when validating
pipeline correctness before paying for optimized object emission.

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

Build a stateful decode package and separated CPU/CUDA artifacts with explicit
fallback policy:

```powershell
python311 example\gguf\build_stateful_artifacts.py `
  --model model.gguf `
  --litenn build-cuda\tools\gguf\litenn_gguf_convert.exe `
  --out-dir build\qwen_stateful `
  --past-length 0 --max-cache-length 4096 `
  --cuda-policy native-required
```

`artifact_manifest.json` records the actual compiler backend and whether the
CUDA request produced a CPU bridge. `native-required` rejects bridges;
`bridge-allowed` permits but exposes them; `optional` also records unavailable
CUDA builds; `disabled` emits CPU artifacts only.
The stateful package uses the same capacity ABI: `token_ids`,
`current_position`, and one full key/value plane per layer are inputs; logits,
`next_position`, and updated planes are outputs. A caller may therefore reuse
one compiled artifact for every decode position below `max-cache-length`.

Capture llama.cpp golden artifacts for a fixed prompt:

```powershell
python311 scripts\gguf_capture_llamacpp_golden.py `
  --model model.gguf `
  --prompt "hello" `
  --out-dir build\gguf_golden\hello `
  --llama-debug third_party\llama.cpp\build\bin\llama-debug.exe `
  --llama-cli third_party\llama.cpp\build\bin\llama-cli.exe `
  --predict 16 --seed 42
```

The capture script writes `manifest.json`, llama.cpp stdout/stderr logs,
`llama-debug --save-logits` prompt/logit files, and optional fixed-seed
`llama-cli` generated text. Those artifacts are the external acceptance input
for future LiteNN-vs-llama.cpp parity checks; generated captures should stay in
build/output directories rather than source control.

Replay the captured llama.cpp prompt token ids through LiteNN:

```powershell
python311 scripts\gguf_run_litenn_from_golden.py `
  --manifest build\gguf_golden\hello\manifest.json `
  --litenn build-release\tools\gguf\litenn_gguf_convert.exe `
  --steps 16 --sample greedy
```

The replay script reads the llama-debug `*-prompt.txt` artifact, extracts the
`token ids:` line, runs `--run-llama-decode-loop-token-ids`, and writes
`litenn_decode_manifest.json` plus LiteNN stdout/stderr/output and per-position
logits metadata next to the golden capture. Add `--capture-decode-logits` (or an
explicit `--logits-output-dir`) when numerical decode comparison is needed;
full-vocabulary dumps are opt-in because they can be large.

Compare LiteNN last-token prefill logits against the llama-debug logits text
artifact:

```powershell
python311 scripts\gguf_compare_llamacpp_logits.py `
  --manifest build\gguf_golden\hello\manifest.json `
  --litenn build-release\tools\gguf\litenn_gguf_convert.exe `
  --abs-tol 1e-4 --rel-tol 1e-4
```

The comparison script replays the captured prompt token ids through
`--dump-llama-token-id-logits`, reads llama-debug `index: value` logits, and
writes `logits_compare.json` with max absolute/relative errors and the largest
mismatches.

For exact first-token and multi-token decode comparison, build the isolated
llama.cpp API helper. This is intentionally a separate CMake project so the
LiteNN build does not acquire a runtime or link dependency on llama.cpp:

```powershell
cmake -S tools\llamacpp-adapter -B build-llamacpp-adapter -DCMAKE_BUILD_TYPE=Release
cmake --build build-llamacpp-adapter --target litenn_llamacpp_adapter --parallel
```

Capture reference logits for a fixed token stream and compare every common
decode step:

```powershell
python311 scripts\gguf_capture_llamacpp_decode_logits.py `
  --tool build-llamacpp-adapter\litenn_llamacpp_adapter.exe `
  --model model.gguf --prompt-token-ids 1,2,3 --generated-token-ids 4,5,6 `
  --out-dir build\gguf_golden\hello\llamacpp_decode_logits
python311 scripts\gguf_compare_llamacpp_decode_logits.py `
  --reference-manifest build\gguf_golden\hello\llamacpp_decode_logits\manifest.json `
  --replay-manifest build\gguf_golden\hello\litenn_decode_manifest.json
```

The comparison rejects mismatched prompt ids and generated-token prefixes
before reading logits. `qwen_smoke.py --llamacpp-decode-golden-tool <path>`
automates capture and comparison after its regular replay.

Production support is an evidence gate rather than a model-name allowlist:

```powershell
python311 scripts\gguf_production_gate.py `
  --smoke-report build\qwen_smoke\qwen_smoke_report.json `
  --artifact-manifest build\qwen_stateful\artifact_manifest.json `
  --require-prefill --require-decode --require-text
```

The default gate requires `cuda-native`, successful compatibility analysis,
explicitly absent fallback, and requested golden comparisons. It writes
`production_gate.json` and exits non-zero when evidence is missing or failed.

Create an honest decode comparison table from available evidence:

```powershell
python311 benchmark\gguf_decode_compare.py `
  --litenn-smoke-report build\qwen_smoke\qwen_smoke_report.json `
  --llama-bench-json build\llama_bench.json `
  --pytorch-json build\pytorch_decode.json `
  --output-dir build\qwen_decode_compare
```

The collector emits JSON, CSV, and Markdown rows with `ms/token`, token/s,
fallback state, and percentage differences against same-device-class
llama.cpp and PyTorch/HF baselines. Missing backends remain absent rather than
being represented by bridge or synthetic measurements.

Run a long-context LiteNN matrix with the same smoke driver:

```powershell
python311 benchmark\gguf_context_matrix.py `
  --model model.gguf `
  --litenn build-tools\tools\gguf\litenn_gguf_convert.exe `
  --targets 2k,32k,128k,1m `
  --token-ids 1,2,3 `
  --steps 1 `
  --paged-reference-decode `
  --paged-resident-pages 64 `
  --aot-cache-dir build\qwen_aot_cache `
  --out-dir build\qwen_context_matrix
```

Use `--dry-run` first to inspect the exact commands. The matrix writes
`gguf_context_matrix.json` and `.md` with build/run/token metrics when rows
complete; private model paths stay in the ignored output directory, not in the
repo.

The same isolated adapter provides production tokenizer parity without linking
llama.cpp into LiteNN. Tokenize and detokenize through the manifest-backed
driver:

```powershell
python311 scripts\gguf_tokenizer_adapter.py tokenize `
  --tool build-llamacpp-adapter\litenn_llamacpp_adapter.exe `
  --model model.gguf --text "hello" `
  --workdir build\gguf_tokenizer --output build\gguf_tokenizer\tokens.json
python311 scripts\gguf_tokenizer_adapter.py detokenize `
  --tool build-llamacpp-adapter\litenn_llamacpp_adapter.exe `
  --model model.gguf --token-ids 14990 `
  --workdir build\gguf_tokenizer --output build\gguf_tokenizer\text.bin
```

`tokens.json` records the exact token ids plus the model's BOS and special-token
policy. Detokenized text is binary-safe so byte-fallback output is not damaged
by an intermediate locale conversion. The adapter also exposes `chat-template`
for a single user turn and appends the assistant-generation marker using the
GGUF model's default template. The Qwen smoke driver can run the full
prompt/template/token boundary directly:

```powershell
python311 example\gguf\qwen_smoke.py `
  --model model.gguf --prompt "hello" `
  --llamacpp-tokenizer-tool build-llamacpp-adapter\litenn_llamacpp_adapter.exe `
  --steps 1 --output build\qwen_smoke\tokens.txt `
  --text-output build\qwen_smoke\generated.txt
```

With `--llamacpp-tokenizer-tool`, `qwen_smoke.py` treats `--prompt` as an
instruct-model user turn by default and applies the model's chat template before
tokenization. Add `--raw-prompt` only when deliberate continuation-style text
completion is desired.

The smoke driver defaults LiteNN decode to `LITENN_COMPILE_DIAGNOSTICS=1` and
`LITENN_CPU_AOT_LLVM_OPT_LEVEL=0`, because large GGUF first-run CPU AOT
compilation is still measured in minutes. Increase `--steps` after the first
single-token run succeeds; use `--llvm-opt-level 3` only when benchmarking
steady-state optimized CPU AOT behavior.
An experimental separated-artifact cache can be enabled with
`--aot-cache-dir build\qwen_smoke\aot_cache`, but it is intentionally opt-in:
the current 14B CPU AOT instruction object is large enough that first-run cache
population is not yet a good default user experience.

Prepared Q4_K/Q6_K experiments can select `--cpu-aot-ggml-prepacked-weight-policy all`
with `--cpu-aot-ggml-prepacked-weight-layout expanded-v1`, `compact-v3`, or `field-interleaved-v4`.
The layout is part of both the decode artifact key and shared-weight key, so the
same cache root can safely hold both variants. For a controlled matrix, use
`benchmark/gguf_decode_thread_matrix.py --cpu-aot-ggml-prepacked-weight-layouts
expanded-v1,compact-v3,field-interleaved-v4`; it combines the layout axis with policy and thread axes.

Compare generated text after replay:

```powershell
python311 scripts\gguf_compare_generation_text.py `
  --manifest build\gguf_golden\hello\manifest.json `
  --replay-manifest build\gguf_golden\hello\litenn_decode_manifest.json
```

This comparison reads llama-cli stdout from the capture manifest and joins the
generated token pieces from LiteNN's replay output. It is useful as an automated
acceptance harness, while full tokenizer parity remains a separate requirement.

Run a Qwen/Qwen2.5-style end-to-end smoke sequence from a GGUF path:

```powershell
python311 example\gguf\qwen_smoke.py `
  --model model.gguf `
  --litenn build-release\tools\gguf\litenn_gguf_convert.exe `
  --token-ids 1,2,3,4 `
  --backend-policy cpu-aot `
  --max-tokens 16 `
  --output build\gguf_qwen_smoke\generated_token_ids.txt `
  --workdir build\gguf_qwen_smoke
```

For prompt-level validation, let llama.cpp provide the tokenizer/golden side and
ask the smoke driver to replay those token ids through LiteNN:

```powershell
python311 example\gguf\qwen_smoke.py `
  --model model.gguf `
  --litenn build-release\tools\gguf\litenn_gguf_convert.exe `
  --prompt "hello" `
  --capture-llamacpp `
  --compare-logits `
  --compare-text `
  --llama-debug third_party\llama.cpp\build\bin\llama-debug.exe `
  --llama-cli third_party\llama.cpp\build\bin\llama-cli.exe `
  --max-tokens 16
```

`qwen_smoke.py` first runs the LiteNN LLM compatibility analyzer, then executes
the selected token-id or llama.cpp-capture path and writes
`qwen_smoke_report.json` with command, stdout, stderr, and return-code evidence.
The current backend policy is intentionally limited to `cpu-aot`; CUDA native
and bridge decode policies are tracked as follow-up runtime integration work.

Current scope: decode graphs expose static-shape KV cache inputs and updated-cache outputs. Dynamic cache growth and llama.cpp golden-logit validation are still tracked in `docs/Roadmap.md`.
