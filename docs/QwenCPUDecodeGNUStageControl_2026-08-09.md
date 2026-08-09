# Qwen CPU Decode GNU Stage Control - 2026-08-09

## Scope

This report closes the GNU/no-OpenMP half of the low-overhead reference stage matrix. It repeats the accepted
non-synchronizing aggregate-counter method over the same exact Qwen prompt/decode window used by the Clang/no-OpenMP
activation/Down split. Raw process artifacts remain in the ignored
`build/qwen_stage_gnu_activation_control_20260809_fresh` directory.

The counters live only in a detached benchmark worktree. LiteNN production targets do not link llama.cpp.

## Measurement Identity

- Workload: Qwen2.5-Coder-14B-Instruct Q4_K_M, `8,988,110,272`-byte GGUF.
- Host: AMD Ryzen 9 9950X, Windows, balanced power policy.
- Reference source: llama.cpp `b81c2cdd748dc2704d5989cf03936325554c12d3`.
- Compiler/runtime: GCC `16.1.0`, Release `-O3`, native ISA, OpenMP disabled, 8 threads.
- Window: 9 fixed prompt tokens followed by 15 fixed decode tokens; saved metadata contains only counts and SHA-256
  digests.
- Pairing: five repetitions with alternating clean/instrumented order.
- Clean profiler SHA-256: `e582fb62c985771889051a0c86e44e0563b698ca0676da50e7deb7d48968f189`.
- Instrumented profiler SHA-256: `2e97fe7d05ac7864402e4b19e6b7fdb9706fba66bcf79390297b955b3adc8046`.

The clean and instrumented llama/ggml libraries were regenerated together with the same compiler and CMake options.
This is material: an earlier reused clean profiler had a different binary identity and produced an invalid `-6.65%`
median instrumentation delta.

## Gate Result

All strict gates passed:

| Gate | Result | Limit |
| --- | ---: | ---: |
| Clean whole-run CV | `1.46%` | `<= 3%` |
| Instrumented whole-run CV | `2.13%` | `<= 3%` |
| Median instrumentation delta | `-1.00%` | absolute value `<= 3%` |
| Median aggregate coverage | `98.64%` | `95-102%` |
| Maximum normalized-stage CV | `4.17%` | `<= 15%` |

The clean and instrumented medians were `164.022` and `161.811 ms/token`. Their median weighted actual frequencies
were `5061` and `5037 MHz`, respectively.

## Accepted GNU Stages

| Normalized stage | Median ms/token | CV | Calls/token |
| --- | ---: | ---: | ---: |
| Attention | `36.338` | `1.69%` | `48` |
| FFN Gate/Up | `70.564` | `1.53%` | `96` |
| FFN activation | `0.196` | `4.17%` | `48` |
| FFN Down | `43.128` | `1.73%` | `48` |
| Final logits | `11.739` | `1.60%` | `1` |

## Compiler Cross-Check

The accepted Clang/no-OpenMP campaign measured `165.299 ms/token` clean, with Attention/Gate-Up/activation/Down/logits
at `36.846/70.916/0.210/43.045/11.920 ms/token`. The non-adjacent GNU-minus-Clang differences are:

| Component | Absolute difference | Relative difference |
| --- | ---: | ---: |
| Clean whole token | `-1.277 ms` | `-0.77%` |
| Attention | `-0.508 ms` | `-1.38%` |
| FFN Gate/Up | `-0.352 ms` | `-0.50%` |
| FFN activation | `-0.014 ms` | `-6.67%` |
| FFN Down | `+0.083 ms` | `+0.19%` |
| Final logits | `-0.181 ms` | `-1.52%` |

These are separate campaigns and the GNU run had about `1.9%` higher measured frequency, so the table does not rank
compiler throughput. It does establish that compiler choice does not materially reorder the reference stages. The
activation percentage looks larger only because the stage is about `0.2 ms`; its absolute difference is `0.014 ms`.

## Conclusions

1. The GNU/no-OpenMP aggregate profile is accepted. This closes the compiler diagnostic requested by the CPU P0
   evidence gate.
2. GNU and Clang both place reference SwiGLU activation near `0.2 ms/token` and Down near `43 ms/token`. The confirmed
   LiteNN activation deficit is therefore not a Clang-specific artifact, while no LiteNN Down deficit is established.
3. The bounded vector-activation policy remains the only measured CPU implementation P0. Compiler switching,
   projection rewrites, or more Down tuning do not explain the `10.833 ms/token` LiteNN activation gap.
4. Clean and instrumented controls must be regenerated as one build identity. A matching source revision and visible
   CMake options are insufficient when reused binary hashes differ and the measured perturbation fails the gate.

## Reproduction

Prepare clean and instrumented llama.cpp builds with the same GCC, native-ISA, and no-OpenMP configuration as described
in `benchmark/llama_cpp_stage_profile/README.md`, configure one profiler against each build, then run:

```powershell
python311 benchmark\run_llama_cpp_stage_control.py `
  --model <model.gguf> `
  --baseline-binary gnu-noopenmp=<clean-profiler.exe> `
  --binary gnu-noopenmp=<instrumented-profiler.exe> `
  --mode aggregate --threads 8 --warmup 0 --steps 15 --repetitions 5 `
  --prefill-token-ids <nine-fixed-token-ids> `
  --decode-token-ids <fifteen-fixed-token-ids> `
  --overhead-threshold-percent 3 --stage-variance-threshold-percent 15 `
  --output-json build\qwen-stage-gnu\control.json `
  --output-md build\qwen-stage-gnu\control.md
```
