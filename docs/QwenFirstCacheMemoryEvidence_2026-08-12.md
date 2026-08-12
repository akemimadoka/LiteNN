# Qwen First-Cache Memory Evidence - 2026-08-12

## Scope

This report isolates the memory and publication cost of the first CPU AOT cache build. It uses a local, path-redacted
Qwen2.5-Coder-14B Q4_K_M GGUF, stateful decode, CPU AOT O0, eight helper threads, compact prepared weights, and a
100 ms process-memory sampling interval. The model path and machine-private cache roots are intentionally excluded.

The measurements answer three separate questions:

1. Why did first-cache publication approach 30 GB for an approximately 9 GB archive?
2. Does the fix reduce the continuous peak, rather than only one allocation counter?
3. Are source GGUF tensor payloads released before steady decode?

## Measurement Support

`benchmark/process_memory.py` now provides a dependency-free process sampler for:

- Windows: working set, peak working set, private commit, peak private commit, and virtual size through
  `GetProcessMemoryInfo`.
- Linux: RSS, peak RSS, anonymous RSS, file-backed RSS, and virtual size through `/proc/<pid>/status`.
- macOS: resident and virtual size through `libproc`.

`example/gguf/qwen_smoke.py` samples only the decode subprocess, associates samples with timed LiteNN stages, writes a
`<step>.memory.json` artifact, adds memory counters to the Chrome trace, and includes peak rows in the waterfall report.
The Windows and Linux paths have focused automated coverage; the Linux implementation was also exercised under WSL.

## Root Cause

GGUF import created inference parameters with `Variable::Create` and `Variable::CreateQuantized`. Those APIs allocate
same-shaped gradient storage by design. During first-cache publication, three model-sized regions therefore overlapped:

| Region | Approximate size | Required for inference build |
| --- | ---: | --- |
| Imported GGUF tensor payloads | 9 GB | Yes, until external-weight preparation finishes |
| Automatically allocated gradients | 9 GB | No |
| Prepared external CPU AOT weights | 9.160 GB | Yes |

The importer now uses frozen variable factories, and regression coverage verifies that ordinary and quantized imported
weights have no gradient storage. After module construction, the GGUF tool also clears the imported variables and the
decode-plan variable owners, so source payloads do not remain live throughout generation.

## Results

### Continuous first-build peak

| Measurement | Before fix | After frozen import | Change |
| --- | ---: | ---: | ---: |
| Observed working set / sampled RSS | about 27.37 GB, single sample | 18.566 GB continuous peak | about -8.80 GB |
| Observed private bytes | about 27.49 GB, single sample | 18.679 GB continuous peak | about -8.81 GB |
| Prepared shared-weight output | 9.160 GB | 9.160 GB | unchanged |

The new peak occurred during `cpu-aot emit object file`, not during the final weight-file write. The approximately one
model-sized reduction matches the removed gradient allocation. Because the baseline was a single observation while the
new result is continuously sampled, the percentage is directional; the eliminated allocation and its size are proven
independently by the ownership audit and tests.

The same fresh-cache run recorded these publication times:

| Stage | Earlier run | Frozen-import run |
| --- | ---: | ---: |
| CPU AOT artifact compile | 27.667 s | 20.547 s |
| Object emission | 9.929 s | 6.402 s |
| Metadata build | 7.793 s | 6.911 s |
| Shared-weight write | 20.452 s | 17.875 s |
| Compile-only total | 61.979 s | 48.862 s |

These are single-run timings and are not a latency acceptance result. They show that removing gradients did not trade
memory for a slower publication path.

### Source-weight lifetime

A separate fresh compile followed by prompt replay and one generated token showed:

- peak sampled RSS/private: `18.567/18.681 GB`, again during object emission;
- after module construction and source-owner release: RSS fell to `9.435 GB` within about 215 ms;
- while nine decode steps ran: RSS/private grew from `9.435/9.525 GB` to `9.933/10.028 GB` as mapped weight pages were
  touched;
- final shutdown samples are excluded from steady-state claims because the working set was already being torn down.

This confirms that steady decode no longer retains both imported and prepared model-sized payloads. The approximately
10 GB active footprint is consistent with the 9.160 GB mapped prepared-weight region plus runtime code, state, and
process overhead.

### Cache-hit control

A compile-only, required-cache-hit control completed module loading in `16.1 ms`. Its six short-lived process samples
reached `182.8 MB` RSS and `282.9 MB` private bytes; prepared weight pages were mapped but not evaluated. This row
validates cache lookup and mapping overhead only and must not be presented as steady decode memory.

## Conclusions

1. The roughly 27.5 GB first-build footprint was not an inherent requirement of 14B quantized inference. An unused
   inference-gradient copy accounted for approximately 9 GB and has been removed.
2. Source-weight ownership is now bounded to module construction. Steady decode uses the prepared mapped weights rather
   than retaining a second imported tensor archive.
3. First-build peak remains approximately twice the model payload because imported source weights and prepared output
   coexist while the compiler externalizes and emits the artifact. This is now a distinct streaming/publication problem,
   not a gradient or lifetime leak.
4. The next memory optimization should stream or map source GGUF payloads and publish prepared regions incrementally.
   It should be accepted only if continuous peak private bytes fall below `1.5x` the combined required model payload
   without regressing cache identity, failure atomicity, or decode parity.
5. Long-context memory must be measured separately. KV-cache, allocator fragmentation, and paged-attention residency are
   not represented by this short `max_cache_length=137` control.

