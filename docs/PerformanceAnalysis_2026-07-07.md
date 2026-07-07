# LiteNN GGUF Decode vs llama.cpp Kernel Analysis - 2026-07-07

This report extends `PerformanceAnalysis_2026-07-05.md` with a phase split, a cache-hit policy matrix, and a
source-level audit of the relevant llama.cpp CPU quantized decode path. The private local model path is intentionally
not recorded here; all evidence paths below are workspace-relative.

## Evidence

LiteNN evidence:

- `build/qwen_profile_policy_20260707/combined_compare/gguf_decode_compare.json`
- `build/qwen_profile_policy_cachehit_20260707/gguf_decode_thread_matrix.md`
- `build/qwen_profile_policy_cachehit_20260707/llama_compare/gguf_decode_compare.md`
- `build/qwen_profile_policy_cachehit_20260707/policy_disabled_threads_8_profile_bundle/gguf_decode_summary.json`
- `build/qwen_profile_policy_cachehit_20260707/policy_profitable_threads_8_profile_bundle/gguf_decode_summary.json`
- `build/qwen_profile_policy_cachehit_20260707/policy_all_threads_8_profile_bundle/gguf_decode_summary.json`

llama.cpp CPU-only control evidence:

- `build/qwen_profile_policy_20260707/llama_cpp_t8.json`

The LiteNN cache-hit run used `python311 benchmark/gguf_decode_thread_matrix.py` with `--require-aot-cache-hit`,
`--no-aot-cache-write`, `--threads 8`, `--llvm-opt-level 0`, and the prepared-weight policy matrix
`disabled,profitable,all`. The llama.cpp control used a CPU-only `llama-bench` row with `ngl=0`, `flash_attn=0`,
and `T8`.

## Phase Profile

The current decode harness splits the user-visible workflow into analyze, chat-template application, prompt
tokenization, LiteNN decode, and detokenization. Tokenizer and chat-template work are delegated through the optional
llama.cpp adapter and are not the steady decode kernel bottleneck.

### First-Run / Cache-Population Policy Matrix

| Prepared policy | Analyze ms | Template ms | Tokenize ms | Decode command ms | Detok ms | Build/setup ms | Runtime ms | Gen ms/token | Gen tok/s |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `disabled` | `23631.3` | `4787.3` | `4346.1` | `75853.5` | `4793.8` | `48974.9` | `19599.4` | `1232.2` | `0.812` |
| `profitable` | `5701.3` | `4758.6` | `4619.3` | `78209.6` | `4018.0` | `54824.8` | `16992.8` | `1063.1` | `0.941` |
| `all` | `4976.1` | `3969.7` | `4630.5` | `110004.5` | `9357.2` | `89571.6` | `14121.3` | `861.6` | `1.161` |
| llama.cpp CPU-only T8 | n/a | n/a | n/a | n/a | n/a | n/a | `1880.8` | `235.1` | `4.254` |

The `all` prepared policy is a real full-decode win, but it also expands the shared weight cache from the raw
`~8.98 GB` layout to about `17.93 GB`. It is therefore a speed/memory policy, not an obvious default.

### Cache-Hit Policy Matrix

| Prepared policy | Analyze ms | Template ms | Tokenize ms | Decode command ms | Detok ms | Build/setup ms | Runtime ms | Gen tokens | Gen ms/token | Gen tok/s | Notes |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| `disabled` | `5342.8` | `4074.3` | `3822.1` | `73492.5` | `4387.1` | `48703.9` | `19020.6` | `8` | `1188.6` | `0.841` | complete |
| `profitable` | `5241.0` | `3816.3` | `3942.0` | `78009.4` | `4062.4` | `54856.0` | `16745.2` | `8` | `1033.0` | `0.968` | complete |
| `all` | `5254.1` | `3888.7` | `3744.1` | `107632.0` | `3936.4` | n/a | n/a | `5` | `816.9` | `1.224` | short/interrupted sample |
| llama.cpp CPU-only T8 | n/a | n/a | n/a | n/a | n/a | n/a | `1880.8` | `8` | `235.1` | `4.254` | `llama-bench` |

Even under the best observed LiteNN cache-hit sample, generated-token throughput is about `28.8%` of the llama.cpp
CPU-only T8 control. The completed `disabled` and `profitable` rows are about `19.8%` and `22.8%` of llama.cpp
throughput respectively.

The cache-hit run still spends tens of seconds in the decode command's build/setup bucket. That remains a load-time
usability issue, but it is not the reason for the per-token gap: the generated-token module body is already dominated by
projection helpers.

## Stronger and Weaker Phases

### Strong / Not the Current Bottleneck

| Phase or bucket | Evidence | Interpretation |
| --- | --- | --- |
| Host shell overhead | `0.009-0.023 ms` per generation token in cache-hit profile rows | CLI/runtime wrapper overhead is effectively zero compared with the step. |
| Module non-helper generated code | about `20-25 ms` per token in the last cache-hit steps | AOT entry execution outside helpers is visible but far smaller than projection helper time. |
| Active-prefix attention | `14.7-19.4 ms` total over the short matrix runs | Attention is not the current 2K-context bottleneck after active-prefix lowering. |
| KV append | below `1 ms` total in the short matrix runs | State update is not driving the current gap. |
| Interpreter avoidance | `fallback_count=0` in completed rows | The slow path is not accidental Interpreter execution. |

### Weak / Behind llama.cpp

| Phase or bucket | Evidence | Interpretation |
| --- | --- | --- |
| Q4_K/Q6_K projection helpers | `helper_total_ms` is `7983.9 ms` for the profitable generation phase and `3926.2 ms` for the short all-prepacked generation sample | Quantized projection is still the step-latency owner. |
| Q4_K gate/up | top cache-hit row remains `projection/ffn_gate_up_grouped` or `projection/ffn_gate_or_up`, `41-47%` of generation time | The largest model-level duplicated hidden-vector scan is still not llama.cpp-class. |
| Prepared-weight memory policy | `all` improves speed but expands shared weights to `~17.93 GB` | The current prepared layout is too expanded to be a production default. |
| Build/setup/load phase | cache-hit rows still report `48-55 s` setup before runtime | Cache-hit loading is not yet interactive even though it is separate from token latency. |

## llama.cpp Source Audit

The relevant llama.cpp CPU path is not just "use Q8_K". It combines activation quantization, compact/repacked weight
layouts, architecture-specific vector dots, and GEMV tiling:

- `third_party/llama.cpp/ggml/src/ggml-cpu/ggml-cpu.c:303` wires `GGML_TYPE_Q4_K` to
  `ggml_vec_dot_q4_K_q8_K`.
- `third_party/llama.cpp/ggml/src/ggml-cpu/ggml-cpu.c:319` wires `GGML_TYPE_Q6_K` to
  `ggml_vec_dot_q6_K_q8_K`.
- `third_party/llama.cpp/ggml/src/ggml-cpu/ggml-cpu.c:383` wires `GGML_TYPE_Q8_K` to
  `quantize_row_q8_K`.
- `third_party/llama.cpp/ggml/src/ggml-cpu/arch/x86/quants.c:505` implements x86 `quantize_row_q8_K`.
- `third_party/llama.cpp/ggml/src/ggml-cpu/arch/x86/quants.c:1900` implements x86
  `ggml_vec_dot_q4_K_q8_K`.
- `third_party/llama.cpp/ggml/src/ggml-cpu/arch/x86/quants.c:2288` implements x86
  `ggml_vec_dot_q6_K_q8_K`.
- `third_party/llama.cpp/ggml/src/ggml-cpu/repack.cpp:887` and `:958` implement generic Q4_K repacked GEMV entry
  points for `8x4` and `8x8` layouts.
- `third_party/llama.cpp/ggml/src/ggml-cpu/repack.cpp:1114` and `:1118` implement generic Q6_K repacked GEMV entry
  points.
- `third_party/llama.cpp/ggml/src/ggml-cpu/repack.cpp:2836`, `:2913`, and `:3092` build compact interleaved
  `block_q4_Kx8`, `block_q4_Kx16`, and `block_q6_Kx8` layouts.
- `third_party/llama.cpp/ggml/src/ggml-cpu/arch/x86/repack.cpp:1464` provides an x86 Q4_K `8x8` repacked GEMV path.

This explains why LiteNN's existing direct/prepacked helpers do not close the gap. The current LiteNN prepacked
sidecar improves decode by reorganizing weights, but it still computes against Float32 activations and uses an expanded
prepared layout. llama.cpp's hot path changes the arithmetic contract: Float32 activations are staged as Q8_K, and the
weight format is compactly interleaved for low-thread GEMV/vec-dot kernels.

## Applicability Matrix

| llama.cpp optimization | Applicable to LiteNN? | Priority | Notes |
| --- | --- | --- | --- |
| Q8_K activation staging | Yes, but not as the current per-helper prototype | P0 | Quantize each normalized hidden vector once per decode step and reuse it across Q/K/V/O, gate/up/down, and logits where numeric tolerance permits. |
| Q4_K/Q6_K x Q8_K vec-dot kernels | Yes | P0 | This is the highest-yield arithmetic change. The current direct Float32 x GGML helper is the measured bottleneck. |
| Compact interleaved repack layouts | Yes | P0 | Replace the expanded `all` prepared layout with versioned compact layouts similar in spirit to `block_q4_Kx8/x16` and `block_q6_Kx8`. |
| Repacked GEMV tiling | Yes | P0 | Target `batch=1` decode rows first: gate/up, hidden/output, FFN-down, KV, and logits. |
| Thread/grain retuning | Partially | P1 after P0 kernels | Existing evidence shows T4/T8 retuning alone does not close the gap and can regress LiteNN before the low-thread kernels improve. |
| Prompt replay logits skip | Yes | P1 | Helps prefill/replay latency, not steady generated-token TPS. |
| mmap/borrowed source weight regions | Yes | P1 | Reduces setup memory and cache-hit load cost. It does not directly solve per-token helper time. |
| AMX/KleidiAI/ISA-specific backends | Yes, later | P2 | Valuable after the portable compact repack + Q8_K contract is stable. |
| CUDA/Vulkan decode kernels | Yes, separate backend project | P2 | Do not use GPU work to hide the CPU parity gap; keep CPU projection parity as the current control target. |

## Updated Conclusion

The best current explanation for the gap is now concrete rather than speculative:

1. LiteNN's generated-token path is not spending meaningful time in the Interpreter, host wrapper, active-prefix
   attention, or KV append.
2. The remaining gap is concentrated in Q4_K/Q6_K projection helpers, especially Q4_K gate/up and hidden/output rows.
3. The prepared-weight path proves layout matters, but its expanded representation is not the llama.cpp strategy and is
   too memory-heavy as a default.
4. The next high-yield implementation should be a compact repacked Q4_K/Q6_K x Q8_K decode GEMV family with
   step-level activation staging, not another blanket thread-policy tweak.

Acceptance target for the next tranche: bring the real Qwen2.5-Coder 14B Q4_K_M stateful CPU AOT generation row below
`300 ms/token` on the local CPU-only control setup without increasing fallback count or hiding a larger setup-memory
regression.
