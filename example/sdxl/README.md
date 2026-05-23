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
`unet-euler-smoke`, `unet-conditioning-smoke`, `spatial-transformer-smoke`, `spatial-transformer-2d-smoke`,
`vae-decode-stem`, and `vae-decode-full`; `unet-resblock` covers the stem Conv2D plus the first SDXL UNet ResBlock (`input_blocks.1.0`)
with PyTorch-style GroupNorm, SiLU, timestep projection, two Conv2D layers, and residual add.
`unet-euler-smoke` emits a 4-channel `noise_pred` tensor from real SDXL stem, `time_embed`, first ResBlock, and output
weights so the Euler sampler can run a complete latent-update loop before full UNet generation is available.
`unet-conditioning-smoke` adds the SDXL label/vector conditioning MLP, `spatial-transformer-smoke` emits a fixed token
self/cross-attention block from `middle_block.1`, `spatial-transformer-2d-smoke` wraps the same block in the NCHW
SpatialTransformer path with GroupNorm/proj_in/proj_out/flatten/restore, and `vae-decode-full` walks the VAE decoder ResBlocks/Upsamples/final
projection for fixed shapes. For the CPU AOT smoke path, these probes materialize F16 checkpoint tensors as F32
constants with manifest `target_dtype`.

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

Compile the serialized graph into a carrier object:

```sh
litenn_sdxl_example --compile-object sdxl_unet.ltnn sdxl_unet.obj litenn_sdxl_module
```

For diagnostics, `--run-model` compiles and runs a serialized graph in-process, and `--compile-raw-object` writes the
instruction object that is normally embedded into the carrier object:

```sh
litenn_sdxl_example --run-model sdxl_unet.ltnn
litenn_sdxl_example --compile-raw-object sdxl_unet.ltnn sdxl_unet.raw.obj
```

On Windows, the command also writes `litenn_sdxl_module_exports.def`. Link the object into a DLL with the generated def file, then load and run a zero-input smoke test:

```sh
g++ -shared sdxl_unet.obj litenn_sdxl_module_exports.def -o sdxl_unet.dll
litenn_sdxl_example --load-dll sdxl_unet.dll litenn_sdxl_module
```

Run a minimal Euler sampler loop over a denoiser-shaped carrier DLL:

```sh
litenn_sdxl_example --sample-euler sdxl_unet.dll litenn_sdxl_module --steps 4 --seed 1234
```

Measure the current staged LiteNN smoke path:

```sh
python311 example/sdxl/sdxl_bench.py \
  --exe build/example/sdxl/litenn_sdxl_example.exe \
  --config path/to/generative-models/configs/inference/sd_xl_base.yaml \
  --safetensors path/to/sdxl.safetensors \
  --workdir build/sdxl_bench \
  --markdown-out build/sdxl_bench/results.md \
  --json-out build/sdxl_bench/results.json
```

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

On ELF platforms, link the object as a shared object:

```sh
g++ -shared sdxl_unet.o -o libsdxl_unet.so
litenn_sdxl_example --load-dll ./libsdxl_unet.so litenn_sdxl_module
```

The `--load-dll` path is a carrier ABI smoke test. It creates zero tensors from the compiled input signature. The
`--sample-euler` path initializes the `latent` input, fills a `timestep`/`timesteps` input when present, zero-fills any
other inputs, requires a `noise_pred` output (or a single output), and applies an Euler epsilon-prediction update across
the configured sigma range.

## Current Coverage

This example reuses the G12 Torch manifest importer. The importer can lower Linear/Embedding/Conv2D/LayerNorm/RMSNorm/GroupNorm/timestep embedding/activation/Softmax/Pad/Upsample/Reshape/Transpose style graphs today.

The importer also provides SDXL-oriented composite manifest ops for fixed-shape experiments: `residual_block`, `feed_forward`, `geglu_feed_forward`, `attention_block`, `spatial_transformer_2d`, `concat`, and `vae_decode`. These templates lower to existing LiteNN graph nodes, so they are meant as import-time assembly helpers rather than new core Graph concepts.

## Denoise Runtime Contract

The denoise loop is intentionally outside the compiled graph for now. A compiled UNet step should receive already-bound latents, timestep embedding input, text/context embeddings, and any additive attention masks.

- The scheduler owns timestep order, sigma/alpha values, and the latent update equation. The example currently includes
  a simple Euler sampler for denoiser-shaped smoke graphs.
- Classifier-free guidance is a runtime binding policy: either run conditional/unconditional batches together or invoke the compiled step twice and combine outputs outside the graph.
- Latent scaling is explicit: manifests may model per-graph input/output scale, but scheduler-specific scaling before and after each step belongs to the runtime harness.
- Benchmarks should report import time, serialization time, AOT compile time, DLL/shared-object load time, and one denoise-step invocation separately.

Full production SDXL still needs tokenizer/text-encoder execution inside LiteNN, full UNet traversal generation,
memory-aware 1024x1024 VAE attention handling, and broader parity/benchmark coverage.
These are tracked in `docs/Roadmap.md` under G12.5 and the longer-term model-parity queues.
