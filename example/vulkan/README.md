# LiteNN Vulkan Add Example

This example compiles a same-shape Float32 add graph to a Vulkan-native SPIR-V payload, loads it through
`CompiledModule<Vulkan>`, runs it on `Tensor<Vulkan>` inputs, and copies the result back to CPU for printing.

```powershell
cmake -S . -B build -DLITENN_ENABLE_MLIR=ON -DLITENN_ENABLE_VULKAN=ON -DLITENN_BUILD_EXAMPLES=ON
cmake --build build --target litenn_vulkan_add_example --parallel
.\build\example\vulkan\litenn_vulkan_add_example.exe
```

The program exits successfully with a skip message when no Vulkan compute device is available.
