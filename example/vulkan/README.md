# LiteNN Vulkan Add Example

This example compiles a same-shape Float32 add graph to a Vulkan-native SPIR-V payload, loads it through both
`CompiledModule<Vulkan>` and separated metadata/instruction regions, runs it on `Tensor<Vulkan>` inputs, and copies the
result back to CPU for printing.

```powershell
cmake -S . -B build -DLITENN_ENABLE_MLIR=ON -DLITENN_ENABLE_VULKAN=ON -DLITENN_BUILD_EXAMPLES=ON
cmake --build build --target litenn_vulkan_add_example --parallel
.\build\example\vulkan\litenn_vulkan_add_example.exe
```

Expected output on a Vulkan compute device:

```text
Vulkan Add result: 11 22 33 44
Vulkan Add separated result: 11 22 33 44
Separated regions: metadata=<bytes> constants=0 weights=0 instructions=<bytes>
```

The program exits successfully with a skip message when no Vulkan compute device is available.
