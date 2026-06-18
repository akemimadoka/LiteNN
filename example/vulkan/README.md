# LiteNN Vulkan Backend Selection Example

This example shows the current Vulkan AOT backend selection boundary:

- `Add(lhs, rhs)` is supported by the Vulkan-native SPIR-V path. The example prints the native support report, loads the
  normal artifact, loads the separated metadata/instruction regions, runs both modules on `Tensor<Vulkan>` inputs, and
  prints CPU-side Vulkan profile events.
- `LinearExternalWeights(input)` is supported by the fused Vulkan-native MatMulBiasAddReLU path. It loads the separated
  image through borrowed regions and demonstrates a non-empty `weights` region for mobile asset packaging.
- `Add(Add(lhs, rhs), tail)` and `Multiply(Add(lhs, rhs), tail)` are supported by the first Vulkan-native binary chain
  path. Normal runs batch the recorded native command buffers into one synchronized queue submit; profile runs keep
  per-kernel synchronized dispatch so CPU/GPU timings remain attributable.
- A diamond-shaped binary graph exercises the current Vulkan-native elementwise DAG path. Older builds used this case to
  demonstrate explicit CPU bridge fallback; current builds should compile it as a native multi-kernel DAG.

```powershell
cmake -S . -B build -DLITENN_ENABLE_MLIR=ON -DLITENN_ENABLE_VULKAN=ON -DLITENN_BUILD_EXAMPLES=ON
cmake --build build --target litenn_vulkan_add_example --parallel
.\build\example\vulkan\litenn_vulkan_add_example.exe
```

Expected output on a Vulkan compute device:

```text
Add native support: yes (<capability>)
Add artifact backend: vulkan_native
Vulkan Add result: 11 22 33 44
Vulkan Add separated result: 11 22 33 44
Profile kernel[0] entry=main ... gpu_ms=<value-or-n/a>
Separated regions: metadata=<bytes> constants=0 weights=0 instructions=<bytes>
LinearExternalWeights native support: yes (<capability>)
LinearExternalWeights artifact backend: vulkan_native
LinearExternalWeights separated regions: metadata=<bytes> constants=0 weights=<bytes> instructions=<bytes> external_tensors=2
Vulkan LinearExternalWeights separated result: 39 0 53 0
TwoAdd native support: yes (<capability>)
TwoAdd artifact backend: vulkan_native
Vulkan TwoAdd chain result: 111 222 333 444
MixedChain native support: yes (<capability>)
MixedChain artifact backend: vulkan_native
Vulkan MixedChain result: 1100 4400 9900 17600
Diamond native support: yes (<capability>)
Diamond artifact backend: vulkan_native
Diamond native result: 112 224 336 448
```

The program exits successfully with a skip message when no Vulkan compute device is available.
