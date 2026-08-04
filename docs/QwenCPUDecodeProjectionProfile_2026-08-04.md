# Qwen CPU Decode Projection Phase Profile - 2026-08-04

This document records the first production-model profile that separates Q8_K activation preparation from CPU thread
pool execution for field-interleaved-v4 Q4_K/Q6_K projections. It is a focused evidence record for the FFN-Down
closure work; the broader LiteNN/llama.cpp comparison remains in `QwenCPUDecodePerformanceEvidence_2026-08-04.md`.

## Scope and Method

- Model class: Qwen2.5 Coder 14B, Q4_K_M GGUF.
- Backend: separated CPU AOT, cache hit required, field-interleaved-v4 prepared weights.
- Runtime policy: 8 requested CPU AOT threads, adaptive worker wait, LLVM optimization level 0.
- Instrumentation: `--profile-helpers --stream-stats`, with structured projection and worker events.
- Stable sample: generated-token steps 10 through 24, 15 observations.
- Exclusions: the first module run, prompt replay, and the first logits-bearing generation step were excluded because
  they contained page-fault and first-touch costs.
- Correctness gate: no runtime fallback and identical generated token sequence within the measured run.

The private model path is intentionally omitted. The equivalent command shape is:

```powershell
python311 example/gguf/qwen_smoke.py `
  --model <model.gguf> `
  --litenn build-release/tools/gguf/litenn_gguf_convert.exe `
  --llamacpp-tokenizer-tool build-release/tools/llamacpp-adapter/litenn_llamacpp_adapter.exe `
  --prompt "hello" --stateful --max-tokens 16 `
  --workdir <profile-workdir> --aot-cache-dir <existing-cache> `
  --require-aot-cache-hit --profile-helpers --stream-stats --ignore-eos `
  --llvm-opt-level 0 --cpu-aot-threads 8 --cpu-aot-worker-wait adaptive `
  --cpu-aot-ggml-prepacked-weight-policy all `
  --cpu-aot-ggml-prepacked-weight-layout field-interleaved-v4
```

The reused AOT artifact predates the direct SwiGLU-to-Down fusion. This does not affect the Q8_K preparation or
field-v4 worker measurements below, but the run must not be used as post-fusion end-to-end evidence.

## Stable Decode Results

The stable profiled token took `266.887 ms` on average. Compiled-module execution accounted for `266.609 ms`, helper
time for `256.184 ms`, and module work outside helper timers for `10.425 ms`.

| Phase | Average per step | Minimum | Maximum | Share of step |
| --- | ---: | ---: | ---: | ---: |
| Activation cache lookup | `0.0076 ms` | `0.0050 ms` | `0.0120 ms` | `<0.01%` |
| Float32 activation copy | `0.1726 ms` | `0.1460 ms` | `0.2010 ms` | `0.06%` |
| Q8_K activation quantization | `45.2007 ms` | `44.2250 ms` | `46.1670 ms` | `16.94%` |
| Thread-pool lock wait | `0.0075 ms` | `0.0070 ms` | `0.0080 ms` | `<0.01%` |
| Worker dispatch | `14.9425 ms` | `12.9780 ms` | `17.1430 ms` | `5.60%` |
| Parallel wall time | `77.7852 ms` | `74.1780 ms` | `83.7080 ms` | `29.15%` |
| Final barrier wait | `3.8463 ms` | `2.7030 ms` | `4.8940 ms` | `1.44%` |

`dispatch` and `barrier wait` are contained within `parallel wall time`; they must not be added to it. Activation
lookup, copy, and quantization occur before the parallel interval and are additive. The profiled single-projection path
therefore owns approximately `45.381 + 77.785 = 123.166 ms/step`, of which Q8_K preparation is `36.7%`.

The caller contributed `58.943 ms` of useful work while worker useful time summed to `362.076 ms`. The worker sum is
CPU time accumulated across participants, not wall time.

Dynamic task claims were uneven on individual Q4_K calls. Averaged across the stable steps, the minimum/maximum claim
extrema were `0.53/11.53` for Q4_K Down and `0.67/9.47` for Q4_K hidden projections; Q6_K Down was more even at
`1.80/8.07`. Some Q4_K participants therefore arrived after the caller or other workers had claimed most tasks.
However, the corresponding full-step barrier totals remained only `0.937`, `0.736`, and `1.432 ms`. This makes worker
wake-up/dispatch batching a valid secondary target, but does not support treating final-barrier imbalance as the main
bottleneck.

## Shape Attribution

| Format and shape | Calls/step | Threads | Quantize | Dispatch | Parallel wall | Barrier |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Q4_K `13824 -> 5120` | 24 | 8 | `16.495 ms` | `5.872 ms` | `19.553 ms` | `0.937 ms` |
| Q6_K `13824 -> 5120` | 24 | 8 | `16.398 ms` | `5.737 ms` | `29.087 ms` | `1.432 ms` |
| Q4_K `5120 -> 5120` | 48 | 4 | `12.056 ms` | `3.107 ms` | `17.102 ms` | `0.736 ms` |
| Q6_K `5120 -> 152064` | 1 | 8 | `0.251 ms` | `0.226 ms` | `12.044 ms` | `0.741 ms` |

FFN-Down activation quantization alone costs `32.893 ms/step`. Hidden/output projections add `12.056 ms/step`, and
the vocabulary projection adds `0.251 ms/step`. Lookup and copy are negligible for every shape.

The structured events currently cover the 97 ordinary projection calls. Grouped Gate/Up helpers are not included in
this phase table, so `45.2007 ms/step` is a measured lower bound for all Q8_K activation preparation in the complete
decode schedule, not an estimate of its total cost.

## Microbenchmark Correlation

The dedicated Release benchmark used the same quantizer and exact Qwen activation sizes, with five repetitions:

| Benchmark | Median | Throughput | Blocks |
| --- | ---: | ---: | ---: |
| `GGMLQ8KActivationPrepare/qwen_hidden/batch:1/in:5120` | `235 us` | `82.97 MiB/s` | 20 |
| `GGMLQ8KActivationPrepare/qwen_ffn_down/batch:1/in:13824` | `640 us` | `82.17 MiB/s` | 54 |

Production-profile averages were about `251 us` for 5120 elements and `685 us` for 13824 elements, only `6.8-7.1%`
above the isolated benchmark. The benchmark is therefore representative enough to gate quantizer changes before an
expensive full-model rerun.

## Conclusions

1. Q8_K activation quantization is the largest newly isolated fixed owner: at least `45.20 ms/step`, including
   `32.89 ms/step` in FFN-Down. It is the first implementation target.
2. The current scalar quantizer reaches only about `82 MiB/s` on tiny contiguous inputs. Its `std::nearbyint`-based
   per-element rounding and scalar reductions should be replaced with parity-tested nearest-integer and SIMD paths.
3. Activation identity lookup and Float32 copying are not performance problems. Removing or complicating the cache for
   these phases cannot provide meaningful decode benefit.
4. Thread-pool mutex contention is effectively zero. Final barrier cost is measurable but secondary. Affinity and
   barrier-only tuning should not precede quantizer work.
5. Dispatch costs `14.94 ms/step` across 97 calls and is the next fixed-cost target after quantization. Reducing helper
   count or batching dispatch is more promising than optimizing the uncontended lock.
6. Q6_K Down parallel execution remains slower than Q4_K Down (`29.09` versus `19.55 ms/step`) after activation
   preparation is excluded. Kernel bandwidth/instruction work remains a separate P0 after quantization.
7. A full-model conclusion requires a fresh post-change artifact and alternating no-profile LiteNN/llama.cpp controls.
   The phase profile identifies owners; it does not by itself claim end-to-end speedup.

## Acceptance Gates

- Preserve byte-exact Q8_K scales, quantized values, and block sums against the scalar reference, including ties,
  signed maxima, tails, and non-contiguous source views.
- Require a stable improvement on both exact-size Q8_K preparation rows before running the private production model.
- Require unchanged generated tokens, no fallback, and a lower stable no-profile token median before retaining a
  full-model optimization.
- Keep phase profiling opt-in; the default runtime path must not pay per-task timestamps or profile aggregation.

## Q8_K Nearest-Integer Optimization Result

The scalar `std::nearbyint` conversion prevented the compiler from optimizing the otherwise contiguous Q8_K
preparation loop. Replacing it with the bounded IEEE-754 nearest-integer construction used by ggml/llama.cpp retained
the existing signed-max, clamp, and block-sum semantics.

Correctness evidence:

- A byte-level scalar-reference test covers positive and negative signed maxima, half-integer rounding boundaries,
  multiple blocks, an all-zero block, and non-contiguous source strides.
- All 19 `GGUFLLaMAQuantizedExecution.*` tests pass.
- The old and new real-model runs generated the same 16-token sequence with no fallback.

Exact-shape Release benchmark medians improved as follows:

| Shape | Before | After | Speedup | Reduction |
| --- | ---: | ---: | ---: | ---: |
| 5120 elements | `235 us` | `5.59 us` | `42.0x` | `97.62%` |
| 13824 elements | `640 us` | `15.3 us` | `41.8x` | `97.61%` |

The same cache-hit production profile and stable-step window measured:

| Metric | Before | After | Change |
| --- | ---: | ---: | ---: |
| Profiled token latency | `266.887 ms` | `184.275 ms` | `-82.612 ms` (`-30.95%`) |
| Helper time | `256.184 ms` | `172.525 ms` | `-83.659 ms` (`-32.66%`) |
| Ordinary-projection Q8_K quantization | `45.201 ms` | `1.187 ms` | `-44.014 ms` (`-97.37%`) |
| Ordinary-projection parallel wall | `77.785 ms` | `69.177 ms` | `-8.608 ms` (`-11.07%`) |
| Final barrier wait | `3.846 ms` | `3.662 ms` | `-0.184 ms` (`-4.78%`) |

The end-to-end improvement is larger than the previously visible `45.2 ms` quantization lower bound because grouped
Gate/Up helpers use the same quantizer but were not included in the structured ordinary-projection phase table.

Three cache-hit runs without helper profiling produced identical tokens. Their stable generated-step averages were
`197.748`, `195.874`, and `200.156 ms/token`, for `5.057`, `5.105`, and `4.996 tokens/s`. The median stable latency is
`197.748 ms/token`, `22.94%` below the previous adjacent `256.616 ms/token` baseline. Profiled and unprofiled absolute
latencies must not be compared directly because host frequency and first-touch state differed between the run batches.

Manual AVX2/AVX-512 activation quantization is no longer a P0 item: the remaining structured Q8_K preparation is only
about `1.19 ms/step`. The next profile decision must instead compare the now-dominant grouped Gate/Up and Q6_K Down
work against the same-run CPU-only llama.cpp stage control, then target the largest remaining cross-runtime gap.

## Fresh CPU-Only Control Re-Ranking

Two locally built llama.cpp CPU-only `llama-bench` binaries were measured after the Q8_K change. Both used commit
`b81c2cdd7`, Release, `GGML_NATIVE=ON`, `-march=native`, no CUDA, no BLAS, and flash attention disabled. Their best
16-token results were at two threads:

| Control | T2 | T4 | T8 | T16 | T32 |
| --- | ---: | ---: | ---: | ---: | ---: |
| Primary `llama-bench` | `4.565 t/s` | `4.399 t/s` | `3.973 t/s` | `2.921 t/s` | `1.946 t/s` |
| Alternate `llama-bench` | `4.717 t/s` | `4.407 t/s` | `3.920 t/s` | not run | not run |

A separately built `llama-completion` from the same Release tree then measured the actual prompt/decode path over 32
generated tokens: T2 was `198.91 ms/token` (`5.03 t/s`) and T8 was `242.77 ms/token` (`4.12 t/s`). The LiteNN
three-run stable median of `197.748 ms/token` (`5.057 t/s`) is effectively tied with this strongest current local
control and is faster than both `llama-bench` controls. This closes the old local `~54 ms/token` gap; it does not
invalidate the user's independently observed `6.85 t/s`, which remains the stronger target until its exact build,
thread, affinity, polling, and command configuration is reproduced.

| Actual decode control | Threads | Latency | Throughput | LiteNN throughput difference |
| --- | ---: | ---: | ---: | ---: |
| LiteNN cache-hit decode | 8 | `197.748 ms/token` | `5.057 t/s` | baseline |
| llama.cpp `llama-completion` | 2 | `198.91 ms/token` | `5.03 t/s` | LiteNN `+0.54%` |
| llama.cpp `llama-completion` | 8 | `242.77 ms/token` | `4.12 t/s` | LiteNN `+22.74%` |

The runtimes use their own best locally measured thread counts in the primary row. The T8 row is retained as a
same-requested-thread diagnostic, not as the headline comparison. `llama-bench` is excluded from this percentage table
because its execution boundary is not identical to the prompt/completion tools.

The fresh low-boundary llama.cpp T2 profiler measured:

| Stage | llama.cpp T2 |
| --- | ---: |
| Attention block | `65.158 ms/token` |
| FFN prefix and Gate Q4_K | `45.099 ms/token` |
| FFN Up Q4_K | `43.206 ms/token` |
| Activation and Down Q4_K | `22.169 ms/token` |
| Activation and Down Q6_K | `35.223 ms/token` |
| Complete FFN block, coarser profile | `140.013 ms/token` |
| Final logits | `12.893 ms/token` |

The corresponding LiteNN helper totals were approximately `38.9 ms` for QKV/output projection plus attention/RoPE/KV
update, `122.1 ms` for grouped Gate/Up plus SwiGLU plus Q4_K/Q6_K Down, and `11.49 ms` for logits. These boundaries are
not identical enough for a percentage claim, but they provide no evidence that grouped Gate/Up or Down is currently
slower than the bundled control. Absolute LiteNN helper rank is therefore no longer a sufficient reason to change a
kernel.

Two follow-up experiments were rejected:

- Grouped Q4_K x16 reused one activation scan across adjacent output groups, but the Qwen Gate/Up median changed only
  from `1.14` to `1.11 ms` with about `4%` run noise and slightly worse CPU time. The uncommitted path was removed.
- Replacing per-worker binary semaphores with `atomic::wait/notify_one` reduced measured ordinary-projection dispatch
  from `12.930` to `10.129 ms/step` (`-21.7%`), but parallel wall time increased `1.213 ms`, barrier wait increased
  `0.580 ms`, and total profiled latency regressed `0.54%`. The uncommitted path was removed.

The next performance gate is to reproduce the stronger `6.85 t/s` external control configuration. Until then, retain
the current kernels and semaphore worker path, and require a fresh stage difference before another CPU P0 change.
