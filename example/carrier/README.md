# LiteNN Carrier Example

This example demonstrates the production-oriented AOT path in three layers:

1. `Compiler<CPU>::CompileArtifact` produces an owning artifact with rodata and native object bytes.
2. `CompiledModuleArtifact::WriteObjectFile` emits a carrier object that exports
   `<prefix>_rodata`, `<prefix>_rodata_size`, `<prefix>_instructions`, and
   `<prefix>_instructions_size`.
3. Runtime code reconstructs an artifact from exported symbol addresses with
   `CompiledModuleArtifact::FromExportedSymbols`, then loads a runnable
   `CompiledModule<CPU>` with `artifact.Load()`.

The build generates a fixed add graph carrier object and exposes it through both:

- `litenn_carrier_static`: links the carrier object through a static library and loads it via `extern` symbol addresses.
- `litenn_carrier_shared_loader`: opens the generated shared library at runtime, resolves exported symbols with `GetProcAddress` or `dlsym`, and loads the module from those addresses.

## Build

From the repository root:

```powershell
cmake -S . -B build -DLITENN_ENABLE_MLIR=ON
cmake --build build --target litenn_carrier
```

## Run

Static library path:

```powershell
build\example\carrier\litenn_carrier_static.exe
```

Shared library path with the default sibling DLL or shared object:

```powershell
build\example\carrier\litenn_carrier_shared_loader.exe
```

You can also point the shared loader at an explicit library path:

```powershell
build\example\carrier\litenn_carrier_shared_loader.exe build\example\carrier\litenn_carrier_image_shared.dll
```