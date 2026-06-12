# LiteNN Vulkan Backend Selection Example

This example shows the current Vulkan AOT backend selection boundary:

- `Add(lhs, rhs)` is supported by the Vulkan-native SPIR-V path. The example prints the native support report, loads the
  normal artifact, loads the separated metadata/instruction regions, runs both modules on `Tensor<Vulkan>` inputs, and
  prints CPU-side Vulkan profile events.
- `Add(Add(lhs, rhs), tail)` is supported by the first Vulkan-native same-op binary chain path and runs as two native
  kernels.
- `Multiply(Add(lhs, rhs), tail)` intentionally exceeds the current same-op chain matcher. The compiler emits a
  CPU-native bridge artifact, strict Vulkan loading rejects it, and the example only runs it after explicitly setting
  `VulkanHostFallbackPolicy::Allow`.

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
TwoAdd native support: yes (<capability>)
TwoAdd artifact backend: vulkan_native
Vulkan TwoAdd chain result: 111 222 333 444
MixedChain native support: no (<reason>)
MixedChain artifact backend: cpu_native
Strict Vulkan load rejected CPU bridge: <reason>
Explicit CPU bridge MixedChain result: 1100 4400 9900 17600
```

The program exits successfully with a skip message when no Vulkan compute device is available.
