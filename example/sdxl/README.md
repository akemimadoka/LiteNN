# LiteNN SDXL Import Experiment

This example is an integration harness for experimenting with Stable Diffusion XL weights in LiteNN.

Important limitation: an SDXL `.safetensors` file stores weights, not the executable network. A real port also needs the model architecture/configuration and fixed execution contract.

## Required SDXL Information

- Diffusers or original SDXL configs: `model_index.json`, UNet config, VAE config, both text encoder configs, tokenizer vocab/merges, and scheduler config.
- Target subgraph: usually start with one fixed-shape UNet denoise step before attempting text encoders, VAE decode, or the full scheduler loop.
- Fixed tensor shapes: batch size, latent height/width, token length, pooled embedding shape, timestep dtype/shape, and output sample shape.
- Weight naming convention: original Stability checkpoint names such as `model.diffusion_model.*` / `conditioner.*` or diffusers names such as `unet.*`, `vae.*`, `text_encoder.*`, `text_encoder_2.*`.
- Precision and layout policy: fp32/fp16/bf16 storage, whether to materialize PyTorch conv/linear/norm layouts at import time, and whether unused tensors should fail conversion.

## Workflow

Inspect a checkpoint:

```sh
litenn_sdxl_example --inspect path/to/sdxl.safetensors
```

Probe a Stability-AI `generative-models` SDXL config against a checkpoint and emit a small manifest:

```sh
python311 example/sdxl/sdxl_manifest_probe.py \
  --config path/to/generative-models/configs/inference/sd_xl_base.yaml \
  --safetensors path/to/sdxl.safetensors \
  --probe unet-euler-smoke \
  --height 64 \
  --width 64 \
  --emit-probe-manifest build/sdxl_unet_euler_smoke_manifest.json
```

The probe reads only the safetensors header for compatibility checks. The generated manifest references real checkpoint
tensor names and can be imported by the C++ example. Supported probes are `unet-stem`, `unet-resblock`,
`unet-euler-smoke`, `unet-conditioning-smoke`, `unet-full-fixed`, `spatial-transformer-smoke`, `spatial-transformer-2d-smoke`,
`vae-decode-stem`, and `vae-decode-full`; `unet-resblock` covers the stem Conv2D plus the first SDXL UNet ResBlock (`input_blocks.1.0`)
with PyTorch-style GroupNorm, SiLU, timestep projection, two Conv2D layers, and residual add.
`unet-euler-smoke` emits a 4-channel `noise_pred` tensor from real SDXL stem, `time_embed`, first ResBlock, and output
weights so the Euler sampler can run a complete latent-update loop before full UNet generation is available.
`unet-conditioning-smoke` adds the SDXL label/vector conditioning MLP, `spatial-transformer-smoke` emits a fixed token
self/cross-attention block from `middle_block.1`, `spatial-transformer-2d-smoke` wraps the same block in the NCHW
SpatialTransformer path with GroupNorm/proj_in/proj_out/flatten/restore, and `vae-decode-full` walks the VAE decoder ResBlocks/Upsamples/final
projection for fixed shapes. `unet-full-fixed` walks the SDXL UNet input/middle/output blocks for batch=1 fixed shapes,
including skip-stack channel concat, ResBlocks, downsample/upsample, and all discovered SpatialTransformer blocks while
still expecting externally supplied `context` and `vector_cond` tensors. For the CPU AOT smoke path, these probes materialize F16 checkpoint tensors as F32
constants with manifest `target_dtype`. Pass `--compute-dtype F16` or `--compute-dtype BF16` to keep generated
UNet/VAE probe inputs and imported weights in low precision; non-F32 timestep embeddings insert an explicit cast after
the sinusoidal F32 embedding boundary.

`vae-decode-full` controls dense VAE mid-attention with `--vae-mid-attention-policy auto|force|skip` and
`--vae-attention-max-mib`. The default `auto` policy emits exact dense attention for small fixed-shape smoke tests and
skips it for large 1024x1024-style decodes when the estimated score/probability workspace exceeds the limit. The
manifest metadata records the policy, estimated bytes, status, and reason so image-quality or memory regressions are
visible in generated artifacts. Use `force` only when validating exact attention on a machine with enough memory.

The probe can also write a discovered block traversal plan:

```sh
python311 example/sdxl/sdxl_manifest_probe.py \
  --config path/to/generative-models/configs/inference/sd_xl_base.yaml \
  --safetensors path/to/sdxl.safetensors \
  --emit-skeleton-plan build/sdxl_skeleton_plan.json
```

Import a LiteNN Torch manifest plus SDXL weights into a serialized graph:

```sh
litenn_sdxl_example --import sdxl_unet_manifest.json path/to/sdxl.safetensors sdxl_unet.ltnn --allow-extra-tensors
```

Write an input-binding safetensors file for a smoke manifest:

```sh
python311 example/sdxl/sdxl_write_inputs.py \
  --probe unet-full-fixed \
  --height 64 \
  --width 64 \
  --context-tokens 4 \
  --output build/sdxl_unet_inputs.safetensors
```

The helper writes F32 tensors whose names match the manifest inputs. It is intended for AOT/DLL binding tests and
for bridging externally exported conditioning tensors into LiteNN while the native text encoder path is still being
built. Custom bindings can also be added with `--tensor name=dimxdim`.

Export a real Stability-AI SpatialTransformer parity fixture and compare LiteNN output against PyTorch:

```sh
python311 example/sdxl/sdxl_manifest_probe.py \
  --config path/to/generative-models/configs/inference/sd_xl_base.yaml \
  --safetensors path/to/sdxl.safetensors \
  --probe spatial-transformer-2d-smoke \
  --height 64 \
  --width 64 \
  --context-tokens 4 \
  --emit-probe-manifest build/sdxl_spatial_2d_manifest.json
litenn_sdxl_example --import build/sdxl_spatial_2d_manifest.json path/to/sdxl.safetensors \
  build/sdxl_spatial_2d.ltnn --allow-extra-tensors
python311 example/sdxl/sdxl_export_spatial_transformer_fixture.py \
  --generative-models path/to/generative-models \
  --config path/to/generative-models/configs/inference/sd_xl_base.yaml \
  --checkpoint path/to/sdxl.safetensors \
  --height 64 \
  --width 64 \
  --context-tokens 4 \
  --inputs-output build/sdxl_spatial_2d_inputs.safetensors \
  --reference-output build/sdxl_spatial_2d_reference.safetensors
litenn_sdxl_example --run-model-with-inputs build/sdxl_spatial_2d.ltnn \
  build/sdxl_spatial_2d_inputs.safetensors --output build/sdxl_spatial_2d_actual.safetensors
python311 example/sdxl/sdxl_compare_safetensors.py \
  --actual build/sdxl_spatial_2d_actual.safetensors \
  --expected build/sdxl_spatial_2d_reference.safetensors
```

This parity flow is checkpoint-gated and requires the external generative-models Python environment, so it is kept as
an example harness rather than a default CTest.

Export prompt conditioning through Stability-AI/generative-models for LiteNN input binding:

```sh
python311 example/sdxl/sdxl_export_conditioning.py \
  --generative-models path/to/generative-models \
  --config path/to/generative-models/configs/inference/sd_xl_base.yaml \
  --checkpoint path/to/sdxl.safetensors \
  --prompt "1girl" \
  --height 1024 \
  --width 1024 \
  --output build/sdxl_conditioning.safetensors
```

The conditioning export writes LiteNN-friendly `context` and `vector_cond` tensors when the original runtime produces
`crossattn` and `vector` conditioning, plus `negative_*`, `*_cfg`, and raw `cond.*` / `uncond.*` tensors for debugging.
By default this helper instantiates only the conditioner and loads `conditioner.*` tensors from the checkpoint, avoiding
the cost of loading the full UNet/VAE just to compute text embeddings. Add `--full-model` only when debugging parity
against the original runtime's complete model construction path. Use `--output-dtype F16` or `--output-dtype BF16`
when binding conditioning into a low-precision manifest.

Compile the serialized graph into a carrier object:

```sh
litenn_sdxl_example --compile-object sdxl_unet.ltnn sdxl_unet.obj litenn_sdxl_module
```

For diagnostics, `--run-model` compiles and runs a serialized graph in-process, and `--compile-raw-object` writes the
instruction object that is normally embedded into the carrier object:

```sh
litenn_sdxl_example --run-model sdxl_unet.ltnn
litenn_sdxl_example --run-model-with-inputs sdxl_unet.ltnn build/sdxl_spatial_2d_inputs.safetensors
litenn_sdxl_example --run-model-with-inputs sdxl_unet.ltnn build/sdxl_spatial_2d_inputs.safetensors \
  --output build/sdxl_step_output.safetensors
litenn_sdxl_example --benchmark-model-with-inputs sdxl_unet.ltnn build/sdxl_unet_inputs.safetensors \
  --device cpu --warmup 1 --iterations 5 --json build/sdxl_cpu_aot_step.json
litenn_sdxl_example --benchmark-model-with-inputs sdxl_unet.ltnn build/sdxl_unet_inputs.safetensors \
  --device cuda --warmup 1 --iterations 5 --json build/sdxl_cuda_aot_step.json
litenn_sdxl_example --compile-raw-object sdxl_unet.ltnn sdxl_unet.raw.obj
```

`--benchmark-model-with-inputs` reports compile/load/input-bind/upload/run timing separately and records rodata,
instruction, input, and output byte sizes. The CUDA row also reports the actual compiled backend, so unsupported graphs
show up as `cpu_native` bridge/fallback instead of being mistaken for native CUDA execution.

On Windows, the command also writes `litenn_sdxl_module_exports.def`. Link the object into a DLL with the generated def file, then load and run either a zero-input or bound-input smoke test:

```sh
g++ -shared sdxl_unet.obj litenn_sdxl_module_exports.def -o sdxl_unet.dll
litenn_sdxl_example --load-dll sdxl_unet.dll litenn_sdxl_module
litenn_sdxl_example --load-dll-with-inputs sdxl_unet.dll build/sdxl_spatial_2d_inputs.safetensors litenn_sdxl_module
litenn_sdxl_example --load-dll-with-inputs sdxl_unet.dll build/sdxl_spatial_2d_inputs.safetensors \
  litenn_sdxl_module --output build/sdxl_step_output.safetensors
```

Run a minimal Euler sampler loop over a denoiser-shaped carrier DLL:

```sh
litenn_sdxl_example --sample-euler sdxl_unet.dll litenn_sdxl_module --steps 4 --seed 1234
litenn_sdxl_example --sample-euler sdxl_unet.dll litenn_sdxl_module --steps 4 --seed 1234 --inputs conditioning.safetensors
litenn_sdxl_example --sample-euler sdxl_unet.dll litenn_sdxl_module \
  --scheduler edm --sigma-max 14.6146 --sigma-min 0.0292 --rho 3 \
  --steps 30 --seed 1234 --inputs conditioning.safetensors --output-latent build/final_latent.safetensors
```

Run a VAE decode carrier over a saved latent and convert the output tensor to PNG:

```sh
litenn_sdxl_example --load-dll-with-inputs sdxl_vae.dll build/final_latent.safetensors \
  litenn_sdxl_vae --output build/decoded_image.safetensors
python311 example/sdxl/sdxl_tensor_to_png.py \
  --input build/decoded_image.safetensors \
  --output build/decoded_image.png
```

Measure the current staged LiteNN smoke path:

```sh
python311 example/sdxl/sdxl_bench.py \
  --exe build/example/sdxl/litenn_sdxl_example.exe \
  --config path/to/generative-models/configs/inference/sd_xl_base.yaml \
  --safetensors path/to/sdxl.safetensors \
  --probe unet-full-fixed \
  --benchmark-devices cpu,cuda \
  --workdir build/sdxl_bench \
  --markdown-out build/sdxl_bench/results.md \
  --json-out build/sdxl_bench/results.json
```

The benchmark table includes `cpu-aot-denoise-step` and `cuda-aot-denoise-step` rows with backend, compile/load/run
latency, artifact MiB, and bound input/output MiB. Use smaller probes such as `unet-conditioning-smoke` for quick
sanity checks and `unet-full-fixed` when the machine has enough memory for the complete fixed-shape UNet artifact.
When benchmarking `vae-decode-full`, the script forwards `--vae-mid-attention-policy` and
`--vae-attention-max-mib` to the manifest generator so 1024x1024 artifacts include the same memory diagnostics as
manual probe runs.

Generate a reference image through Stability-AI/generative-models after its python311 dependencies are installed:

```sh
python311 example/sdxl/sdxl_generate_reference.py \
  --generative-models path/to/generative-models \
  --config path/to/generative-models/configs/inference/sd_xl_base.yaml \
  --checkpoint path/to/sdxl.safetensors \
  --prompt "1girl" \
  --height 1024 \
  --width 1024 \
  --output build/sdxl_reference_1girl.png
```

## Prompt-To-Image Command Sequence

The checked-in one-shot harness runs the current LiteNN prompt-to-image path: export prompt conditioning, emit/import
UNet and VAE manifests, compile carrier DLL/SO artifacts or separated image regions, run Euler denoising, decode the
latent, and write PNG.

On Windows, first build the example and make MinGW available to the linker:

```bat
cmd /c "set PATH=C:\msys64\mingw64\bin;%PATH% && cmake --build build --parallel"
```

Then run the validated 64x64 smoke path for prompt `1girl`:

```bat
set LITENN_EXE=build\example\sdxl\litenn_sdxl_example.exe
set GM=<path-to-generative-models>
set CONFIG=%GM%\configs\inference\sd_xl_base.yaml
set CKPT=<path-to-sdxl.safetensors>
set OUT=build\sdxl_1girl_smoke
set CXX=C:\msys64\mingw64\bin\g++.exe

python311 example\sdxl\sdxl_prompt_to_image.py ^
  --exe %LITENN_EXE% ^
  --generative-models %GM% ^
  --config %CONFIG% ^
  --checkpoint %CKPT% ^
  --prompt "1girl" ^
  --height 64 ^
  --width 64 ^
  --steps 2 ^
  --workdir %OUT% ^
  --output-png %OUT%\1girl_smoke.png ^
  --cxx %CXX% ^
  --aot-load-mode image-regions
```

The smoke path uses `unet-conditioning-smoke`, so it validates the LiteNN SDXL plumbing and AOT carrier loading rather
than full SDXL image quality. It still uses real prompt conditioning, real SDXL checkpoint tensors, LiteNN AOT for the
denoiser probe, LiteNN AOT for VAE decode, and `sdxl_tensor_to_png.py` for the final PNG. The `image-regions` mode
loads rodata and instruction bytes directly from files, avoiding the shared-library link/load step; use
`--aot-load-mode dll` when specifically validating exported carrier symbols.

The same harness can target the full fixed-shape UNet and 1024x1024 output:

```bat
python311 example\sdxl\sdxl_prompt_to_image.py ^
  --exe %LITENN_EXE% ^
  --generative-models %GM% ^
  --config %CONFIG% ^
  --checkpoint %CKPT% ^
  --prompt "1girl" ^
  --height 1024 ^
  --width 1024 ^
  --steps 30 ^
  --unet-probe unet-full-fixed ^
  --vae-mid-attention-policy auto ^
  --workdir build\sdxl_1girl_full ^
  --output-png build\sdxl_1girl_full\1girl.png ^
  --cxx %CXX%
```

Current status: the full fixed-shape UNet manifest imports correctly, but the materialized F32 `.ltnn` graph is very
large and CPU AOT compilation is not yet interactive. For large imported graphs, enable separated CPU AOT weights with
`LITENN_CPU_AOT_EXTERNAL_REGIONS=1`; `--compile-object` will then write a separated carrier object, and
`--compile-image-regions` will write metadata to the existing `.rodata.bin` path plus sibling `.constants.bin` and
`.weights.bin` files. A low-precision manifest can reduce serialized weight size roughly in half, but full-graph CPU AOT
still needs more compile-time/codegen work before it is practical. Use the smoke command above for a fully runnable
pipeline while that work continues.

The expanded command sequence is:

```bat
python311 example\sdxl\sdxl_export_conditioning.py ^
  --generative-models %GM% --config %CONFIG% --checkpoint %CKPT% ^
  --prompt "1girl" --height 64 --width 64 --output %OUT%\conditioning.safetensors

python311 example\sdxl\sdxl_manifest_probe.py ^
  --config %CONFIG% --safetensors %CKPT% --probe unet-conditioning-smoke ^
  --height 64 --width 64 --emit-probe-manifest %OUT%\unet_manifest.json
%LITENN_EXE% --import %OUT%\unet_manifest.json %CKPT% %OUT%\unet.ltnn --allow-extra-tensors
set LITENN_CPU_AOT_EXTERNAL_REGIONS=1
%LITENN_EXE% --compile-object %OUT%\unet.ltnn %OUT%\unet.obj litenn_sdxl_unet
%CXX% -shared %OUT%\unet.obj %OUT%\litenn_sdxl_unet_exports.def -o %OUT%\unet.dll

%LITENN_EXE% --denoise-latent %OUT%\unet.dll %OUT%\conditioning.safetensors %OUT%\final_latent.safetensors ^
  litenn_sdxl_unet --steps 2 --seed 1234 --scheduler edm --sigma-max 14.6146 --sigma-min 0.0292 ^
  --rho 3 --denoiser-contract sgm-edm --cfg-mode dual --cfg-scale 6

python311 example\sdxl\sdxl_manifest_probe.py ^
  --config %CONFIG% --safetensors %CKPT% --probe vae-decode-full ^
  --height 64 --width 64 --vae-mid-attention-policy skip --emit-probe-manifest %OUT%\vae_manifest.json
%LITENN_EXE% --import %OUT%\vae_manifest.json %CKPT% %OUT%\vae.ltnn --allow-extra-tensors
set LITENN_CPU_AOT_EXTERNAL_REGIONS=1
%LITENN_EXE% --compile-object %OUT%\vae.ltnn %OUT%\vae.obj litenn_sdxl_vae
%CXX% -shared %OUT%\vae.obj %OUT%\litenn_sdxl_vae_exports.def -o %OUT%\vae.dll

%LITENN_EXE% --load-dll-with-inputs %OUT%\vae.dll %OUT%\final_latent.safetensors litenn_sdxl_vae ^
  --output %OUT%\decoded_image.safetensors
python311 example\sdxl\sdxl_tensor_to_png.py ^
  --input %OUT%\decoded_image.safetensors --output %OUT%\1girl_smoke.png
```

The same expanded flow can avoid DLL/shared-object linking by writing separated image regions:

```bat
%LITENN_EXE% --compile-image-regions %OUT%\unet.ltnn %OUT%\unet_regions litenn_sdxl_unet
%LITENN_EXE% --denoise-latent-image ^
  %OUT%\unet_regions\litenn_sdxl_unet.rodata.bin ^
  %OUT%\unet_regions\litenn_sdxl_unet.instructions.obj ^
  %OUT%\conditioning.safetensors %OUT%\final_latent.safetensors ^
  --steps 2 --seed 1234 --scheduler edm --sigma-max 14.6146 --sigma-min 0.0292 ^
  --rho 3 --denoiser-contract sgm-edm --cfg-mode dual --cfg-scale 6

%LITENN_EXE% --compile-image-regions %OUT%\vae.ltnn %OUT%\vae_regions litenn_sdxl_vae
%LITENN_EXE% --run-image-with-inputs ^
  %OUT%\vae_regions\litenn_sdxl_vae.rodata.bin ^
  %OUT%\vae_regions\litenn_sdxl_vae.instructions.obj ^
  %OUT%\final_latent.safetensors --output %OUT%\decoded_image.safetensors
```

On ELF platforms, link the object as a shared object:

```sh
g++ -shared sdxl_unet.o -o libsdxl_unet.so
litenn_sdxl_example --load-dll ./libsdxl_unet.so litenn_sdxl_module
```

The `--load-dll` path is a carrier ABI smoke test. It creates zero tensors from the compiled input signature.
`--run-model-with-inputs` and `--load-dll-with-inputs` require every compiled input to exist in a safetensors file with
matching name, dtype, and shape, and can write a single output tensor with `--output`. The `--sample-euler` path
initializes the `latent` input, fills a `timestep`/`timesteps` input when present, optionally binds other inputs from
`--inputs`, requires a `noise_pred` output (or a single output), and applies an Euler epsilon-prediction update across
the configured sigma range. It supports the original linear smoke schedule and an EDM rho schedule, and can write the
final latent as a safetensors file for VAE decode experiments. For SDXL-style runtime parity work, it also supports
explicit denoiser contracts (`epsilon`, `denoised`, `sgm-edm`, `sgm-eps`, `sgm-v`), timestep modes
(`auto`, `legacy`, `sigma`, `edm-log`, `zero`), latent input scaling, and dual-pass classifier-free guidance via
`--cfg-mode dual --cfg-scale X`. Dual CFG binds `negative_*` or `uncond.*` safetensors entries for conditioning inputs.

`--denoise-latent` is a positional wrapper around the sampler for pipeline scripts:

```sh
litenn_sdxl_example --denoise-latent sdxl_unet.dll conditioning.safetensors build/final_latent.safetensors \
  litenn_sdxl_module --steps 30 --scheduler edm --sigma-max 14.6146 --sigma-min 0.0292 \
  --denoiser-contract sgm-edm --cfg-mode dual --cfg-scale 6
```

`--denoise-latent-image` is the equivalent positional wrapper for rodata/instruction image-region files.

## Current Coverage

This example reuses the G12 Torch manifest importer. The importer can lower Linear/Embedding/Conv2D/LayerNorm/RMSNorm/GroupNorm/timestep embedding/activation/Softmax/Pad/Upsample/Reshape/Transpose style graphs today.

The importer also provides SDXL-oriented composite manifest ops for fixed-shape experiments: `residual_block`, `feed_forward`, `geglu_feed_forward`, `attention_block`, `spatial_transformer_2d`, `concat`, and `vae_decode`. These templates lower to existing LiteNN graph nodes, so they are meant as import-time assembly helpers rather than new core Graph concepts.

## Denoise Runtime Contract

The denoise loop is intentionally outside the compiled graph for now. A compiled UNet step should receive already-bound latents, timestep embedding input, text/context embeddings, and any additive attention masks.

- The scheduler owns timestep order, sigma/alpha values, and the latent update equation. The example currently includes
  a linear smoke scheduler and an EDM rho scheduler for denoiser-shaped smoke graphs.
- Classifier-free guidance is a runtime binding policy: either run conditional/unconditional batches together or invoke the compiled step twice and combine outputs outside the graph.
- Latent scaling is explicit: manifests may model per-graph input/output scale, but scheduler-specific scaling before and after each step belongs to the runtime harness. `sgm-*` contracts follow the Stability-AI `Denoiser` shape: latent input is multiplied by `c_in`, the timestep input receives `c_noise` in `auto` mode, and raw network output is converted to a denoised prediction before the Euler derivative is computed.
- Benchmarks should report import time, serialization time, AOT compile time, DLL/shared-object load time, and one denoise-step invocation separately.

Full production SDXL still needs native tokenizer/text-encoder execution inside LiteNN, full fixed-shape UNet AOT
compile-time/weight-size reduction, and broader 1024x1024 parity/benchmark coverage.
These are tracked in `docs/Roadmap.md` under G12.5 and the longer-term model-parity queues.
