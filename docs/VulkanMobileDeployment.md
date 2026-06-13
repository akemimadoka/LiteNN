# LiteNN Vulkan Mobile Deployment

This note defines the current mobile packaging contract for the Vulkan-native backend. It is intentionally narrower than
desktop CPU/CUDA AOT: Vulkan instructions are SPIR-V words in the instruction region, not object files, and the runtime
owns descriptor binding and dispatch.

## Supported Build Shape

Recommended Android profile:

```powershell
cmake -S . -B build-android-vulkan `
  -DCMAKE_TOOLCHAIN_FILE=%ANDROID_NDK_HOME%\build\cmake\android.toolchain.cmake `
  -DANDROID_ABI=arm64-v8a `
  -DANDROID_PLATFORM=android-26 `
  -DLITENN_ENABLE_VULKAN=ON `
  -DLITENN_ENABLE_MLIR=OFF `
  -DLITENN_BUILD_TESTS=OFF `
  -DLITENN_BUILD_EXAMPLES=OFF
cmake --build build-android-vulkan --parallel
```

The production mobile runtime should link `LiteNNCore` and `LiteNNVulkanRuntime`. Keep `LiteNNCompiler` out of the
application package unless the app explicitly compiles graphs on device. The normal mobile flow is:

1. Build/import/optimize the graph on a host machine.
2. Compile a `CompiledModuleArtifact` for `CompiledModuleBackend::VulkanNative`.
3. Store separated metadata/constants/weights/instructions in the app asset bundle or memory-mapped package.
4. Load the separated image with `CompiledModule<Vulkan>::Load` or `LoadBorrowedExternalRegions`.

## Loader Requirements

- Android devices must provide a system Vulkan loader and at least one compute-capable queue family.
- `LiteNNVulkanRuntime` creates its own `VkInstance` and logical device for the selected `Vulkan::deviceIndex`.
- Runtime buffers currently require host-visible coherent storage-buffer memory. This is simple and portable for the
  first backend slice, but it is not the final high-performance mobile memory model.
- When the selected device reports and can legally enable optional low-precision features, `LiteNNVulkanRuntime` adds the
  matching feature structs and required KHR device extensions to `vkCreateDevice`. `QueryVulkanDeviceCapabilities`
  reports both physical availability and logical-device enablement.
- The current descriptor ABI uses descriptor set `0` and storage-buffer bindings matching `VulkanNativeArgumentSpec`.
  Payloads may contain one SPIR-V module with multiple compute entry points; each `VulkanNativeKernelSpec` names the
  entry point it dispatches.
- The instruction region stores serialized `VulkanNativeInstructionPayload`, including SPIR-V words, feature flags,
  entry point, descriptor bindings, byte ranges, and dispatch dimensions.
- Payload version 2 also stores per-kernel requirements: descriptor ABI version, local workgroup layout, required device
  feature bits, optional subgroup-size requirement, and storage-buffer offset-alignment requirement. These requirements
  are validated before creating shader modules or pipelines.
- Static same-shape elementwise/cast kernels currently use `LocalSize = 64` and `ceil(numel / 64)` dispatch groups. The
  generated shader guards tail threads with `global_id < numel`, so tensor element counts do not need to be multiples of
  the workgroup size.
- `CompiledModule<Vulkan>::Load` performs the first native payload/device compatibility gate before shader module or
  pipeline creation. It checks the `vulkan1.1` target, API version, per-kernel local workgroup and dispatch-group
  limits, descriptor ABI, storage-buffer descriptor count limits, storage-buffer range/alignment, subgroup requirements,
  low-precision feature requirements, and future descriptor-indexing/runtime-descriptor-array requirement bits. The
  capability snapshot distinguishes physical feature availability from LiteNN logical-device feature enablement.

## Validation Layers

Validation layers are a development aid, not a runtime dependency. Mobile release builds must not require
`VK_LAYER_KHRONOS_validation`.

Recommended debug policy:

- Enable platform validation layers in the application or test harness, not inside LiteNN core.
- Validate generated SPIR-V before packaging. LiteNN already rejects modules with unsupported addressing/memory models,
  unsupported globals, missing compute entry points, or invalid `LocalSize`.
- Treat validation-layer warnings as release blockers when they mention descriptor layout, storage-buffer alignment,
  queue synchronization, or memory lifetime.

## Artifact Layout

The Vulkan backend uses the same separated package regions as CPU/CUDA AOT:

- metadata: tensor signatures, backend tag, version, target information
- constants: immutable scalar/table data when present
- weights: external model tensors when present
- instructions: Vulkan-native payload with SPIR-V words

Do not package Vulkan instructions as PE/COFF/ELF/Mach-O objects. Static/shared-library embedding can still expose the
four regions as symbols, but the instruction bytes remain SPIR-V payload bytes.

`LoadBorrowedExternalRegions` is the preferred path for memory-mapped mobile assets. The caller owns the mapped package
lifetime and must keep the regions alive while the module is loaded. For current Vulkan-native kernels, external
constants/weights are validated from the separated metadata table and uploaded into runtime-owned Vulkan tensors during
load; the mapped host region does not need to stay readable after that upload for native execution. CPU-bridge artifacts
still follow the CPU borrowed-region lifetime contract.

## Unsupported Desktop Assumptions

The mobile Vulkan runtime must not depend on:

- JIT loading of CPU object files from the instruction region
- CUDA driver APIs, CUDA graph replay, PTX, cubin, or fatbin payloads
- desktop-only validation layer installation paths
- mutable environment-variable configuration as the only policy surface
- implicit CPU fallback when a Vulkan-native graph cannot be compiled

Unsupported kernels must remain explicit: either fail compilation for `VulkanNative` or use a caller-selected bridge or
fallback policy. Benchmarks should report the selected backend rather than silently mixing CPU and Vulkan execution.
Callers can use `Compiler<Vulkan>::QueryNativeSupport(plan)` before compiling to check whether the current native Vulkan
slice can cover a plan and to surface the reason a graph would otherwise use CPU bridge fallback.

## Profiling

`CompiledModuleVulkanRunOptions::profileEvents` can point to a caller-owned vector to collect per-kernel profile events
during synchronized Vulkan-native execution. Events include kernel index, entry point, dispatch groups, local workgroup
layout, descriptor count, module creation wall time, CPU-side dispatch wall time, and optional GPU timestamp-query elapsed
time. GPU timestamp fields are populated only when the selected compute queue reports timestamp support; otherwise the
event keeps `gpuTimestampAvailable=false`. `litenn_profile` also reports one-shot input upload and output download time
for Vulkan rows. Persisted multi-backend Vulkan profile tables are still follow-on work.

## Current Coverage

The current native Vulkan slice supports static-shape, single-subgraph kernels for:

- same-shape `Float32` binary Add/Subtract/Multiply/Divide/Max/Min
- same-shape `Float32` binary chains composed of supported binary ops, for example
  `Multiply(Add(lhs, rhs), tail)`, executed as multiple synchronized Vulkan-native kernels that reuse the public output
  buffer as the chain accumulator
- rank-2 static `Float32` MatMul using one shader invocation per output element; this is a correctness and benchmark
  baseline, not the final tiled/shared-memory production GEMM kernel
- fused rank-2 static `Float32` MatMulBiasAdd/MatMulBiasAddReLU with `[1,N]` or `[M,N]` bias rows; this uses the same
  baseline one-output-element-per-invocation kernel shape
- static-axis `Float32` Reduce Sum/Mean/Max using one shader invocation per output element and a scalar loop over the
  reduced axis; this is a correctness baseline before workgroup/subgroup reductions
- static-axis `Float32` Softmax using max-subtracted scalar loops along the softmax axis; this is a correctness baseline
  before workgroup/subgroup softmax reductions
- static-axis non-affine `Float32` LayerNorm/RMSNorm using scalar loops along the normalized axis; affine scale/bias,
  GroupNorm, and workgroup/subgroup normalization reductions remain follow-on work
- fused rank-2 static `Float32` MatMulBiasAdd/MatMulBiasAddReLU where weight and bias are graph variables/constants in
  separated constants/weights regions; the first model-shaped benchmark is `VulkanNativeRunInto/Linear(784->10)`
- same-shape `Float32` unary Negate/Abs/Sqrt/Exp/Log/Sin/Cos
- same-shape 32-bit casts: `Float32 -> Int32` and `Int32 -> Float32`
- same-shape low-precision cast SPIR-V generation for `Float16`, `Int8`, and `UInt8` storage types; runtime execution
  requires target-device feature enablement for matching 8-bit or 16-bit storage-buffer access. Current builds enable the
  optional feature chain when the selected device supports it, reject unsupported artifacts at load time with a capability
  diagnostic, and register `VulkanNativeCastRunInto/F32ToFloat16|Int8|UInt8` benchmark rows only when execution is legal.
- `benchmark/bench.cpp` registers `VulkanNativeElementwiseAddRunInto`, `VulkanNativeReduce/F32/SumAxis1|MeanAxis1|
  MaxAxis1`, `VulkanNativeSoftmax/F32/Axis1`,
  `VulkanNativeNormalization/F32/LayerNormAxis1|RMSNormAxis1`, `VulkanNativeMatMul/F32`, and
  `VulkanNativeMatMulBiasAdd/F32` rows only when a Vulkan compute device exists. It also registers model-level
  `VulkanNativeRunInto` rows for the single-Linear model once external weight binding is available. Multi-layer MLP rows
  remain deferred until Vulkan has workspace/multi-kernel linear-chain scheduling.

Low-precision arithmetic beyond simple casts, production tiled reductions/softmax/normalization/matmul/multi-layer
linear chains, affine normalization, GroupNorm, convolution, device-local memory, tiled/shared-memory kernels, and async
queue integration remain follow-on production GPU-backend work rather than part of the current bootstrap.
