#!/usr/bin/env python3
"""Probe Stability-AI generative-models SDXL configs and emit LiteNN manifests.

This script intentionally avoids importing the generative-models Python package.
It only needs PyYAML and a safetensors file header, so it can run before the
full SDXL inference environment is installed.
"""

from __future__ import annotations

import argparse
import json
import re
import struct
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml


TENSOR_COMPONENT_PREFIXES: tuple[tuple[str, str], ...] = (
    ("model.diffusion_model.", "unet"),
    ("first_stage_model.", "vae"),
    ("conditioner.embedders.0.", "text_encoder"),
    ("conditioner.embedders.1.", "text_encoder_2"),
)

DEFAULT_VAE_ATTENTION_MAX_MIB = 512


@dataclass(frozen=True)
class TensorInfo:
    name: str
    dtype: str
    shape: list[int]
    data_offsets: tuple[int, int]

    @property
    def bytes(self) -> int:
        return self.data_offsets[1] - self.data_offsets[0]


def load_yaml(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as stream:
        value = yaml.safe_load(stream)
    if not isinstance(value, dict):
        raise ValueError("SDXL config root must be a YAML mapping")
    return value


def load_safetensors_header(path: Path) -> dict[str, Any]:
    with path.open("rb") as stream:
        header_size_bytes = stream.read(8)
        if len(header_size_bytes) != 8:
            raise ValueError("safetensors file is too small to contain a header length")
        header_size = struct.unpack("<Q", header_size_bytes)[0]
        header_bytes = stream.read(header_size)
        if len(header_bytes) != header_size:
            raise ValueError("safetensors file is truncated before the full header")
    value = json.loads(header_bytes)
    if not isinstance(value, dict):
        raise ValueError("safetensors header root must be a JSON object")
    return value


def parse_tensors(header: dict[str, Any]) -> dict[str, TensorInfo]:
    tensors: dict[str, TensorInfo] = {}
    for name, raw in header.items():
        if name == "__metadata__":
            continue
        if not isinstance(raw, dict):
            raise ValueError(f"tensor {name!r} metadata must be an object")
        offsets = raw.get("data_offsets")
        if not isinstance(offsets, list) or len(offsets) != 2:
            raise ValueError(f"tensor {name!r} must have two data_offsets")
        shape = raw.get("shape")
        if not isinstance(shape, list) or not all(isinstance(dim, int) for dim in shape):
            raise ValueError(f"tensor {name!r} must have an integer shape")
        dtype = raw.get("dtype")
        if not isinstance(dtype, str):
            raise ValueError(f"tensor {name!r} must have a string dtype")
        tensors[name] = TensorInfo(name=name, dtype=dtype, shape=shape, data_offsets=(offsets[0], offsets[1]))
    return tensors


def tensor_component(name: str) -> str:
    for prefix, component in TENSOR_COMPONENT_PREFIXES:
        if name.startswith(prefix):
            return component
    return "unknown"


def summarize_tensors(tensors: dict[str, TensorInfo]) -> dict[str, dict[str, int]]:
    summary: dict[str, dict[str, int]] = {}
    for tensor in tensors.values():
        component = tensor_component(tensor.name)
        bucket = summary.setdefault(component, {"count": 0, "bytes": 0})
        bucket["count"] += 1
        bucket["bytes"] += tensor.bytes
    return dict(sorted(summary.items()))


def model_params(config: dict[str, Any]) -> dict[str, Any]:
    model = config.get("model", {})
    params = model.get("params", {}) if isinstance(model, dict) else {}
    if not isinstance(params, dict):
        raise ValueError("config model.params must be a mapping")
    return params


def nested_mapping(root: dict[str, Any], *keys: str) -> dict[str, Any]:
    value: Any = root
    for key in keys:
        if not isinstance(value, dict):
            return {}
        value = value.get(key, {})
    return value if isinstance(value, dict) else {}


def network_params(config: dict[str, Any]) -> dict[str, Any]:
    return nested_mapping(model_params(config), "network_config", "params")


def first_stage_params(config: dict[str, Any]) -> dict[str, Any]:
    return nested_mapping(model_params(config), "first_stage_config", "params")


def conditioner_embedders(config: dict[str, Any]) -> list[dict[str, Any]]:
    embedders = nested_mapping(model_params(config), "conditioner_config", "params").get("emb_models", [])
    return embedders if isinstance(embedders, list) else []


def dtype_to_torch(dtype: str) -> str:
    mapping = {
        "F64": "F64",
        "F32": "F32",
        "F16": "F16",
        "BF16": "BF16",
        "F8_E4M3": "F8_E4M3",
        "F8_E5M2": "F8_E5M2",
        "I64": "I64",
        "I32": "I32",
        "I8": "I8",
        "U8": "U8",
        "BOOL": "BOOL",
    }
    if dtype not in mapping:
        raise ValueError(f"unsupported safetensors dtype {dtype!r}")
    return mapping[dtype]


def require_tensor(tensors: dict[str, TensorInfo], name: str) -> TensorInfo:
    tensor = tensors.get(name)
    if tensor is None:
        raise KeyError(f"required tensor is missing from checkpoint: {name}")
    return tensor


def expected_sdxl_keys() -> list[str]:
    return [
        "model.diffusion_model.input_blocks.0.0.weight",
        "model.diffusion_model.input_blocks.0.0.bias",
        "model.diffusion_model.input_blocks.1.0.in_layers.0.weight",
        "model.diffusion_model.input_blocks.1.0.in_layers.0.bias",
        "model.diffusion_model.input_blocks.1.0.in_layers.2.weight",
        "model.diffusion_model.input_blocks.1.0.in_layers.2.bias",
        "model.diffusion_model.input_blocks.1.0.emb_layers.1.weight",
        "model.diffusion_model.input_blocks.1.0.emb_layers.1.bias",
        "model.diffusion_model.input_blocks.1.0.out_layers.0.weight",
        "model.diffusion_model.input_blocks.1.0.out_layers.0.bias",
        "model.diffusion_model.input_blocks.1.0.out_layers.3.weight",
        "model.diffusion_model.input_blocks.1.0.out_layers.3.bias",
        "model.diffusion_model.out.0.weight",
        "model.diffusion_model.out.0.bias",
        "model.diffusion_model.time_embed.0.weight",
        "model.diffusion_model.time_embed.0.bias",
        "model.diffusion_model.time_embed.2.weight",
        "model.diffusion_model.time_embed.2.bias",
        "model.diffusion_model.label_emb.0.0.weight",
        "model.diffusion_model.out.2.weight",
        "first_stage_model.decoder.conv_in.weight",
        "first_stage_model.decoder.conv_out.weight",
        "conditioner.embedders.0.transformer.text_model.embeddings.token_embedding.weight",
        "conditioner.embedders.1.model.token_embedding.weight",
    ]


def compatibility_report(config: dict[str, Any], tensors: dict[str, TensorInfo]) -> dict[str, Any]:
    net = network_params(config)
    vae = first_stage_params(config).get("ddconfig", {})
    if not isinstance(vae, dict):
        vae = {}

    missing = [name for name in expected_sdxl_keys() if name not in tensors]
    checks: list[dict[str, Any]] = []

    def add_shape_check(name: str, expected: list[int]) -> None:
        tensor = tensors.get(name)
        checks.append(
            {
                "name": name,
                "expected_shape": expected,
                "actual_shape": tensor.shape if tensor is not None else None,
                "ok": tensor is not None and tensor.shape == expected,
            }
        )

    model_channels = int(net.get("model_channels", 320))
    in_channels = int(net.get("in_channels", 4))
    out_channels = int(net.get("out_channels", 4))
    adm_in_channels = int(net.get("adm_in_channels", 2816))
    vae_ch = int(vae.get("ch", 128))
    vae_z_channels = int(vae.get("z_channels", 4))
    vae_ch_mult = vae.get("ch_mult", [1, 2, 4, 4])
    vae_decoder_in_channels = vae_ch * int(vae_ch_mult[-1])
    add_shape_check("model.diffusion_model.input_blocks.0.0.weight", [model_channels, in_channels, 3, 3])
    add_shape_check("model.diffusion_model.input_blocks.1.0.in_layers.0.weight", [model_channels])
    add_shape_check(
        "model.diffusion_model.input_blocks.1.0.in_layers.2.weight",
        [model_channels, model_channels, 3, 3],
    )
    add_shape_check(
        "model.diffusion_model.input_blocks.1.0.emb_layers.1.weight",
        [model_channels, model_channels * 4],
    )
    add_shape_check("model.diffusion_model.input_blocks.1.0.out_layers.0.weight", [model_channels])
    add_shape_check(
        "model.diffusion_model.input_blocks.1.0.out_layers.3.weight",
        [model_channels, model_channels, 3, 3],
    )
    add_shape_check("model.diffusion_model.out.0.weight", [model_channels])
    add_shape_check("model.diffusion_model.out.2.weight", [out_channels, model_channels, 3, 3])
    add_shape_check("model.diffusion_model.time_embed.0.weight", [model_channels * 4, model_channels])
    add_shape_check("model.diffusion_model.time_embed.0.bias", [model_channels * 4])
    add_shape_check("model.diffusion_model.time_embed.2.weight", [model_channels * 4, model_channels * 4])
    add_shape_check("model.diffusion_model.time_embed.2.bias", [model_channels * 4])
    add_shape_check("model.diffusion_model.label_emb.0.0.weight", [model_channels * 4, adm_in_channels])
    add_shape_check("first_stage_model.decoder.conv_in.weight", [vae_decoder_in_channels, vae_z_channels, 3, 3])
    add_shape_check("first_stage_model.decoder.conv_out.weight", [int(vae.get("out_ch", 3)), vae_ch, 3, 3])

    return {
        "components": summarize_tensors(tensors),
        "network": {
            "target": nested_mapping(model_params(config), "network_config").get("target"),
            "in_channels": in_channels,
            "out_channels": out_channels,
            "model_channels": model_channels,
            "channel_mult": net.get("channel_mult"),
            "num_res_blocks": net.get("num_res_blocks"),
            "attention_resolutions": net.get("attention_resolutions"),
            "transformer_depth": net.get("transformer_depth"),
            "context_dim": net.get("context_dim"),
            "adm_in_channels": adm_in_channels,
            "num_head_channels": net.get("num_head_channels"),
        },
        "vae": {
            "target": nested_mapping(model_params(config), "first_stage_config").get("target"),
            "z_channels": vae_z_channels,
            "ch": vae_ch,
            "ch_mult": vae_ch_mult,
            "out_ch": vae.get("out_ch"),
        },
        "conditioner": [
            {
                "input_key": item.get("input_key"),
                "target": item.get("target"),
                "params": item.get("params", {}),
            }
            for item in conditioner_embedders(config)
            if isinstance(item, dict)
        ],
        "missing_required_keys": missing,
        "shape_checks": checks,
        "compatible": not missing and all(check["ok"] for check in checks),
    }


def manifest_tensor(
    name: str,
    tensor: TensorInfo,
    layout: str = "identity",
    shape: list[int] | None = None,
    target_dtype: str | None = None,
) -> dict[str, Any]:
    result = {
        "name": name,
        "source": tensor.name,
        "dtype": dtype_to_torch(tensor.dtype),
        "shape": shape if shape is not None else tensor.shape,
        "layout": layout,
    }
    if target_dtype is not None:
        result["target_dtype"] = target_dtype
    return result


def unet_tensor(tensors: dict[str, TensorInfo], suffix: str) -> TensorInfo:
    return require_tensor(tensors, f"model.diffusion_model.{suffix}")


def manifest_unet_tensor(
    tensors: dict[str, TensorInfo],
    manifest_name: str,
    source_suffix: str,
    layout: str = "identity",
    shape: list[int] | None = None,
    target_dtype: str | None = None,
) -> dict[str, Any]:
    return manifest_tensor(manifest_name, unet_tensor(tensors, source_suffix), layout, shape, target_dtype)


def torch_groupnorm_shape(tensor: TensorInfo) -> list[int]:
    if len(tensor.shape) != 1:
        raise ValueError(f"groupnorm affine tensor {tensor.name!r} must be rank-1")
    return [1, tensor.shape[0], 1, 1]


def torch_linear_weight_shape(tensor: TensorInfo) -> list[int]:
    if len(tensor.shape) != 2:
        raise ValueError(f"linear weight tensor {tensor.name!r} must be rank-2")
    return [tensor.shape[1], tensor.shape[0]]


def torch_bias_1d_shape(tensor: TensorInfo) -> list[int]:
    if len(tensor.shape) != 1:
        raise ValueError(f"linear bias tensor {tensor.name!r} must be rank-1")
    return [1, tensor.shape[0]]


def vae_tensor(tensors: dict[str, TensorInfo], suffix: str) -> TensorInfo:
    return require_tensor(tensors, f"first_stage_model.{suffix}")


def manifest_vae_tensor(
    tensors: dict[str, TensorInfo],
    manifest_name: str,
    source_suffix: str,
    layout: str = "identity",
    shape: list[int] | None = None,
    target_dtype: str | None = None,
) -> dict[str, Any]:
    return manifest_tensor(manifest_name, vae_tensor(tensors, source_suffix), layout, shape, target_dtype)


def maybe_tensor(tensors: dict[str, TensorInfo], name: str) -> TensorInfo | None:
    return tensors.get(name)


def linear_spec(weight: str, bias: str | None = None) -> dict[str, str]:
    result = {"weight": weight}
    if bias is not None:
        result["bias"] = bias
    return result


def sdxl_unet_heads(channel_count: int) -> int:
    # SDXL base uses num_head_channels=64. Keep the smoke manifest robust for
    # unusual checkpoints by falling back to one head when the division fails.
    return channel_count // 64 if channel_count >= 64 and channel_count % 64 == 0 else 1


def node_name(prefix: str) -> str:
    return re.sub(r"[^A-Za-z0-9]+", "_", prefix).strip("_")


def emit_unet_stem_manifest(tensors: dict[str, TensorInfo], *, batch: int, height: int, width: int) -> dict[str, Any]:
    latent_h = height // 8
    latent_w = width // 8
    weight = require_tensor(tensors, "model.diffusion_model.input_blocks.0.0.weight")
    bias = require_tensor(tensors, "model.diffusion_model.input_blocks.0.0.bias")
    return {
        "format": "litenn.torch_manifest.v1",
        "inputs": [
            {"name": "latent", "dtype": "torch.float16", "shape": [batch, weight.shape[1], latent_h, latent_w]},
        ],
        "tensors": [
            manifest_tensor("unet.input_blocks.0.0.weight", weight, "torch_conv2d_weight"),
            manifest_tensor("unet.input_blocks.0.0.bias", bias, "identity"),
        ],
        "nodes": [
            {
                "name": "unet_stem_conv",
                "op": "conv2d",
                "input": "latent",
                "weight": "unet.input_blocks.0.0.weight",
                "bias": "unet.input_blocks.0.0.bias",
                "padding": [1, 1],
                "output": "stem",
            }
        ],
        "outputs": [{"name": "stem", "source": "stem"}],
    }


def emit_unet_resblock_manifest(tensors: dict[str, TensorInfo], *, batch: int, height: int, width: int) -> dict[str, Any]:
    latent_h = height // 8
    latent_w = width // 8
    res_prefix = "input_blocks.1.0"
    stem_weight = unet_tensor(tensors, "input_blocks.0.0.weight")
    stem_bias = unet_tensor(tensors, "input_blocks.0.0.bias")
    norm1_weight = unet_tensor(tensors, f"{res_prefix}.in_layers.0.weight")
    norm1_bias = unet_tensor(tensors, f"{res_prefix}.in_layers.0.bias")
    conv1_weight = unet_tensor(tensors, f"{res_prefix}.in_layers.2.weight")
    conv1_bias = unet_tensor(tensors, f"{res_prefix}.in_layers.2.bias")
    temb_weight = unet_tensor(tensors, f"{res_prefix}.emb_layers.1.weight")
    temb_bias = unet_tensor(tensors, f"{res_prefix}.emb_layers.1.bias")
    norm2_weight = unet_tensor(tensors, f"{res_prefix}.out_layers.0.weight")
    norm2_bias = unet_tensor(tensors, f"{res_prefix}.out_layers.0.bias")
    conv2_weight = unet_tensor(tensors, f"{res_prefix}.out_layers.3.weight")
    conv2_bias = unet_tensor(tensors, f"{res_prefix}.out_layers.3.bias")
    channels = stem_weight.shape[0]
    emb_channels = channels * 4
    return {
        "format": "litenn.torch_manifest.v1",
        "inputs": [
            {"name": "latent", "dtype": "torch.float16", "shape": [batch, stem_weight.shape[1], latent_h, latent_w]},
            {"name": "temb", "dtype": "torch.float16", "shape": [batch, emb_channels]},
        ],
        "tensors": [
            manifest_tensor("unet.input_blocks.0.0.weight", stem_weight, "torch_conv2d_weight"),
            manifest_tensor("unet.input_blocks.0.0.bias", stem_bias, "identity"),
            manifest_unet_tensor(
                tensors,
                "unet.input_blocks.1.0.in_layers.0.weight",
                f"{res_prefix}.in_layers.0.weight",
                "torch_groupnorm_weight",
                torch_groupnorm_shape(norm1_weight),
            ),
            manifest_unet_tensor(
                tensors,
                "unet.input_blocks.1.0.in_layers.0.bias",
                f"{res_prefix}.in_layers.0.bias",
                "torch_groupnorm_bias",
                torch_groupnorm_shape(norm1_bias),
            ),
            manifest_unet_tensor(
                tensors,
                "unet.input_blocks.1.0.in_layers.2.weight",
                f"{res_prefix}.in_layers.2.weight",
                "torch_conv2d_weight",
                conv1_weight.shape,
            ),
            manifest_unet_tensor(
                tensors,
                "unet.input_blocks.1.0.in_layers.2.bias",
                f"{res_prefix}.in_layers.2.bias",
                "identity",
                conv1_bias.shape,
            ),
            manifest_unet_tensor(
                tensors,
                "unet.input_blocks.1.0.emb_layers.1.weight",
                f"{res_prefix}.emb_layers.1.weight",
                "torch_linear_weight",
                torch_linear_weight_shape(temb_weight),
            ),
            manifest_unet_tensor(
                tensors,
                "unet.input_blocks.1.0.emb_layers.1.bias",
                f"{res_prefix}.emb_layers.1.bias",
                "torch_bias_1d",
                torch_bias_1d_shape(temb_bias),
            ),
            manifest_unet_tensor(
                tensors,
                "unet.input_blocks.1.0.out_layers.0.weight",
                f"{res_prefix}.out_layers.0.weight",
                "torch_groupnorm_weight",
                torch_groupnorm_shape(norm2_weight),
            ),
            manifest_unet_tensor(
                tensors,
                "unet.input_blocks.1.0.out_layers.0.bias",
                f"{res_prefix}.out_layers.0.bias",
                "torch_groupnorm_bias",
                torch_groupnorm_shape(norm2_bias),
            ),
            manifest_unet_tensor(
                tensors,
                "unet.input_blocks.1.0.out_layers.3.weight",
                f"{res_prefix}.out_layers.3.weight",
                "torch_conv2d_weight",
                conv2_weight.shape,
            ),
            manifest_unet_tensor(
                tensors,
                "unet.input_blocks.1.0.out_layers.3.bias",
                f"{res_prefix}.out_layers.3.bias",
                "identity",
                conv2_bias.shape,
            ),
        ],
        "nodes": [
            {
                "name": "unet_stem_conv",
                "op": "conv2d",
                "input": "latent",
                "weight": "unet.input_blocks.0.0.weight",
                "bias": "unet.input_blocks.0.0.bias",
                "padding": [1, 1],
                "output": "stem",
            },
            {
                "name": "unet_input_blocks_1_0",
                "op": "residual_block",
                "input": "stem",
                "temb": "temb",
                "activation": "silu",
                "norm1": {
                    "weight": "unet.input_blocks.1.0.in_layers.0.weight",
                    "bias": "unet.input_blocks.1.0.in_layers.0.bias",
                    "num_groups": 32,
                    "eps": 1e-5,
                    "layout": "pytorch",
                },
                "conv1": {
                    "weight": "unet.input_blocks.1.0.in_layers.2.weight",
                    "bias": "unet.input_blocks.1.0.in_layers.2.bias",
                    "padding": [1, 1],
                },
                "temb_projection": {
                    "weight": "unet.input_blocks.1.0.emb_layers.1.weight",
                    "bias": "unet.input_blocks.1.0.emb_layers.1.bias",
                },
                "norm2": {
                    "weight": "unet.input_blocks.1.0.out_layers.0.weight",
                    "bias": "unet.input_blocks.1.0.out_layers.0.bias",
                    "num_groups": 32,
                    "eps": 1e-5,
                    "layout": "pytorch",
                },
                "conv2": {
                    "weight": "unet.input_blocks.1.0.out_layers.3.weight",
                    "bias": "unet.input_blocks.1.0.out_layers.3.bias",
                    "padding": [1, 1],
                },
                "output": "resblock",
            },
        ],
        "outputs": [{"name": "resblock", "source": "resblock"}],
    }


def emit_unet_euler_smoke_manifest(
    tensors: dict[str, TensorInfo],
    *,
    batch: int,
    height: int,
    width: int,
) -> dict[str, Any]:
    latent_h = height // 8
    latent_w = width // 8
    res_prefix = "input_blocks.1.0"
    stem_weight = unet_tensor(tensors, "input_blocks.0.0.weight")
    stem_bias = unet_tensor(tensors, "input_blocks.0.0.bias")
    time0_weight = unet_tensor(tensors, "time_embed.0.weight")
    time0_bias = unet_tensor(tensors, "time_embed.0.bias")
    time2_weight = unet_tensor(tensors, "time_embed.2.weight")
    time2_bias = unet_tensor(tensors, "time_embed.2.bias")
    norm1_weight = unet_tensor(tensors, f"{res_prefix}.in_layers.0.weight")
    norm1_bias = unet_tensor(tensors, f"{res_prefix}.in_layers.0.bias")
    conv1_weight = unet_tensor(tensors, f"{res_prefix}.in_layers.2.weight")
    conv1_bias = unet_tensor(tensors, f"{res_prefix}.in_layers.2.bias")
    temb_weight = unet_tensor(tensors, f"{res_prefix}.emb_layers.1.weight")
    temb_bias = unet_tensor(tensors, f"{res_prefix}.emb_layers.1.bias")
    norm2_weight = unet_tensor(tensors, f"{res_prefix}.out_layers.0.weight")
    norm2_bias = unet_tensor(tensors, f"{res_prefix}.out_layers.0.bias")
    conv2_weight = unet_tensor(tensors, f"{res_prefix}.out_layers.3.weight")
    conv2_bias = unet_tensor(tensors, f"{res_prefix}.out_layers.3.bias")
    out_norm_weight = unet_tensor(tensors, "out.0.weight")
    out_norm_bias = unet_tensor(tensors, "out.0.bias")
    out_weight = unet_tensor(tensors, "out.2.weight")
    out_bias = unet_tensor(tensors, "out.2.bias")
    channels = stem_weight.shape[0]
    emb_channels = channels * 4
    compute_dtype = "F32"
    return {
        "format": "litenn.torch_manifest.v1",
        "inputs": [
            {"name": "latent", "dtype": "torch.float32", "shape": [batch, stem_weight.shape[1], latent_h, latent_w]},
            {"name": "timestep", "dtype": "torch.float32", "shape": [batch]},
        ],
        "tensors": [
            manifest_tensor("unet.input_blocks.0.0.weight", stem_weight, "torch_conv2d_weight", target_dtype=compute_dtype),
            manifest_tensor("unet.input_blocks.0.0.bias", stem_bias, "identity", target_dtype=compute_dtype),
            manifest_tensor(
                "unet.time_embed.0.weight",
                time0_weight,
                "torch_linear_weight",
                torch_linear_weight_shape(time0_weight),
                target_dtype=compute_dtype,
            ),
            manifest_tensor(
                "unet.time_embed.0.bias",
                time0_bias,
                "torch_bias_1d",
                torch_bias_1d_shape(time0_bias),
                target_dtype=compute_dtype,
            ),
            manifest_tensor(
                "unet.time_embed.2.weight",
                time2_weight,
                "torch_linear_weight",
                torch_linear_weight_shape(time2_weight),
                target_dtype=compute_dtype,
            ),
            manifest_tensor(
                "unet.time_embed.2.bias",
                time2_bias,
                "torch_bias_1d",
                torch_bias_1d_shape(time2_bias),
                target_dtype=compute_dtype,
            ),
            manifest_unet_tensor(
                tensors,
                "unet.input_blocks.1.0.in_layers.0.weight",
                f"{res_prefix}.in_layers.0.weight",
                "torch_groupnorm_weight",
                torch_groupnorm_shape(norm1_weight),
                target_dtype=compute_dtype,
            ),
            manifest_unet_tensor(
                tensors,
                "unet.input_blocks.1.0.in_layers.0.bias",
                f"{res_prefix}.in_layers.0.bias",
                "torch_groupnorm_bias",
                torch_groupnorm_shape(norm1_bias),
                target_dtype=compute_dtype,
            ),
            manifest_unet_tensor(
                tensors,
                "unet.input_blocks.1.0.in_layers.2.weight",
                f"{res_prefix}.in_layers.2.weight",
                "torch_conv2d_weight",
                conv1_weight.shape,
                target_dtype=compute_dtype,
            ),
            manifest_unet_tensor(
                tensors,
                "unet.input_blocks.1.0.in_layers.2.bias",
                f"{res_prefix}.in_layers.2.bias",
                "identity",
                conv1_bias.shape,
                target_dtype=compute_dtype,
            ),
            manifest_unet_tensor(
                tensors,
                "unet.input_blocks.1.0.emb_layers.1.weight",
                f"{res_prefix}.emb_layers.1.weight",
                "torch_linear_weight",
                torch_linear_weight_shape(temb_weight),
                target_dtype=compute_dtype,
            ),
            manifest_unet_tensor(
                tensors,
                "unet.input_blocks.1.0.emb_layers.1.bias",
                f"{res_prefix}.emb_layers.1.bias",
                "torch_bias_1d",
                torch_bias_1d_shape(temb_bias),
                target_dtype=compute_dtype,
            ),
            manifest_unet_tensor(
                tensors,
                "unet.input_blocks.1.0.out_layers.0.weight",
                f"{res_prefix}.out_layers.0.weight",
                "torch_groupnorm_weight",
                torch_groupnorm_shape(norm2_weight),
                target_dtype=compute_dtype,
            ),
            manifest_unet_tensor(
                tensors,
                "unet.input_blocks.1.0.out_layers.0.bias",
                f"{res_prefix}.out_layers.0.bias",
                "torch_groupnorm_bias",
                torch_groupnorm_shape(norm2_bias),
                target_dtype=compute_dtype,
            ),
            manifest_unet_tensor(
                tensors,
                "unet.input_blocks.1.0.out_layers.3.weight",
                f"{res_prefix}.out_layers.3.weight",
                "torch_conv2d_weight",
                conv2_weight.shape,
                target_dtype=compute_dtype,
            ),
            manifest_unet_tensor(
                tensors,
                "unet.input_blocks.1.0.out_layers.3.bias",
                f"{res_prefix}.out_layers.3.bias",
                "identity",
                conv2_bias.shape,
                target_dtype=compute_dtype,
            ),
            manifest_tensor(
                "unet.out.0.weight",
                out_norm_weight,
                "torch_groupnorm_weight",
                torch_groupnorm_shape(out_norm_weight),
                target_dtype=compute_dtype,
            ),
            manifest_tensor(
                "unet.out.0.bias",
                out_norm_bias,
                "torch_groupnorm_bias",
                torch_groupnorm_shape(out_norm_bias),
                target_dtype=compute_dtype,
            ),
            manifest_tensor("unet.out.2.weight", out_weight, "torch_conv2d_weight", target_dtype=compute_dtype),
            manifest_tensor("unet.out.2.bias", out_bias, "identity", target_dtype=compute_dtype),
        ],
        "nodes": [
            {
                "name": "unet_timestep_sinusoidal",
                "op": "timestep_embedding",
                "timesteps": "timestep",
                "dim": channels,
                "max_period": 10000,
                "output": "time_sinusoidal",
            },
            {
                "name": "unet_time_embed_0",
                "op": "linear",
                "input": "time_sinusoidal",
                "weight": "unet.time_embed.0.weight",
                "bias": "unet.time_embed.0.bias",
                "output": "time_hidden",
            },
            {
                "name": "unet_time_embed_act",
                "op": "silu",
                "input": "time_hidden",
                "output": "time_hidden_act",
            },
            {
                "name": "unet_time_embed_2",
                "op": "linear",
                "input": "time_hidden_act",
                "weight": "unet.time_embed.2.weight",
                "bias": "unet.time_embed.2.bias",
                "output": "temb",
            },
            {
                "name": "unet_stem_conv",
                "op": "conv2d",
                "input": "latent",
                "weight": "unet.input_blocks.0.0.weight",
                "bias": "unet.input_blocks.0.0.bias",
                "padding": [1, 1],
                "output": "stem",
            },
            {
                "name": "unet_input_blocks_1_0",
                "op": "residual_block",
                "input": "stem",
                "temb": "temb",
                "activation": "silu",
                "norm1": {
                    "weight": "unet.input_blocks.1.0.in_layers.0.weight",
                    "bias": "unet.input_blocks.1.0.in_layers.0.bias",
                    "num_groups": 32,
                    "eps": 1e-5,
                    "layout": "pytorch",
                },
                "conv1": {
                    "weight": "unet.input_blocks.1.0.in_layers.2.weight",
                    "bias": "unet.input_blocks.1.0.in_layers.2.bias",
                    "padding": [1, 1],
                },
                "temb_projection": {
                    "weight": "unet.input_blocks.1.0.emb_layers.1.weight",
                    "bias": "unet.input_blocks.1.0.emb_layers.1.bias",
                },
                "norm2": {
                    "weight": "unet.input_blocks.1.0.out_layers.0.weight",
                    "bias": "unet.input_blocks.1.0.out_layers.0.bias",
                    "num_groups": 32,
                    "eps": 1e-5,
                    "layout": "pytorch",
                },
                "conv2": {
                    "weight": "unet.input_blocks.1.0.out_layers.3.weight",
                    "bias": "unet.input_blocks.1.0.out_layers.3.bias",
                    "padding": [1, 1],
                },
                "output": "resblock",
            },
            {
                "name": "unet_out_norm",
                "op": "group_norm",
                "input": "resblock",
                "weight": "unet.out.0.weight",
                "bias": "unet.out.0.bias",
                "num_groups": 32,
                "eps": 1e-5,
                "layout": "pytorch",
                "output": "out_norm",
            },
            {
                "name": "unet_out_silu",
                "op": "silu",
                "input": "out_norm",
                "output": "out_act",
            },
            {
                "name": "unet_noise_pred",
                "op": "conv2d",
                "input": "out_act",
                "weight": "unet.out.2.weight",
                "bias": "unet.out.2.bias",
                "padding": [1, 1],
                "output": "noise_pred",
            },
        ],
        "outputs": [{"name": "noise_pred", "source": "noise_pred"}],
    }


def emit_unet_conditioning_smoke_manifest(
    tensors: dict[str, TensorInfo],
    *,
    batch: int,
    height: int,
    width: int,
) -> dict[str, Any]:
    manifest = emit_unet_euler_smoke_manifest(tensors, batch=batch, height=height, width=width)
    label0_weight = unet_tensor(tensors, "label_emb.0.0.weight")
    label0_bias = unet_tensor(tensors, "label_emb.0.0.bias")
    label2_weight = unet_tensor(tensors, "label_emb.0.2.weight")
    label2_bias = unet_tensor(tensors, "label_emb.0.2.bias")
    compute_dtype = "F32"

    manifest["inputs"].append(
        {"name": "vector_cond", "dtype": "torch.float32", "shape": [batch, label0_weight.shape[1]]}
    )
    manifest["tensors"].extend(
        [
            manifest_tensor(
                "unet.label_emb.0.0.weight",
                label0_weight,
                "torch_linear_weight",
                torch_linear_weight_shape(label0_weight),
                target_dtype=compute_dtype,
            ),
            manifest_tensor(
                "unet.label_emb.0.0.bias",
                label0_bias,
                "torch_bias_1d",
                torch_bias_1d_shape(label0_bias),
                target_dtype=compute_dtype,
            ),
            manifest_tensor(
                "unet.label_emb.0.2.weight",
                label2_weight,
                "torch_linear_weight",
                torch_linear_weight_shape(label2_weight),
                target_dtype=compute_dtype,
            ),
            manifest_tensor(
                "unet.label_emb.0.2.bias",
                label2_bias,
                "torch_bias_1d",
                torch_bias_1d_shape(label2_bias),
                target_dtype=compute_dtype,
            ),
        ]
    )

    nodes = manifest["nodes"]
    time2_index = next(i for i, node in enumerate(nodes) if node["name"] == "unet_time_embed_2")
    nodes[time2_index]["output"] = "time_emb"
    nodes[time2_index + 1 : time2_index + 1] = [
        {
            "name": "unet_label_emb_0",
            "op": "linear",
            "input": "vector_cond",
            "weight": "unet.label_emb.0.0.weight",
            "bias": "unet.label_emb.0.0.bias",
            "output": "label_hidden",
        },
        {
            "name": "unet_label_emb_act",
            "op": "silu",
            "input": "label_hidden",
            "output": "label_hidden_act",
        },
        {
            "name": "unet_label_emb_2",
            "op": "linear",
            "input": "label_hidden_act",
            "weight": "unet.label_emb.0.2.weight",
            "bias": "unet.label_emb.0.2.bias",
            "output": "label_emb",
        },
        {
            "name": "unet_conditioning_add",
            "op": "add",
            "lhs": "time_emb",
            "rhs": "label_emb",
            "output": "temb",
        },
    ]
    manifest["metadata"] = {
        "probe": "unet-conditioning-smoke",
        "description": "time_embed plus SDXL label/vector conditioning prefix feeding the first UNet ResBlock",
    }
    return manifest


def emit_spatial_transformer_smoke_manifest(
    tensors: dict[str, TensorInfo],
    *,
    tokens: int,
    context_tokens: int,
) -> dict[str, Any]:
    prefix = "middle_block.1.transformer_blocks.0"
    channel_count = unet_tensor(tensors, f"{prefix}.attn1.to_q.weight").shape[0]
    context_width = unet_tensor(tensors, f"{prefix}.attn2.to_k.weight").shape[1]
    compute_dtype = "F32"

    def wt(name: str, suffix: str, layout: str, shape: list[int] | None = None) -> dict[str, Any]:
        return manifest_unet_tensor(tensors, name, f"{prefix}.{suffix}", layout, shape, target_dtype=compute_dtype)

    def maybe_bias(name: str, suffix: str) -> dict[str, str]:
        spec = {"weight": name}
        bias_source = f"model.diffusion_model.{prefix}.{suffix}"
        if bias_source in tensors:
            spec["bias"] = name.replace(".weight", ".bias")
        return spec

    tensor_entries = [
        wt("unet.middle_block.1.transformer_blocks.0.norm1.weight", "norm1.weight", "torch_norm_weight",
           torch_bias_1d_shape(unet_tensor(tensors, f"{prefix}.norm1.weight"))),
        wt("unet.middle_block.1.transformer_blocks.0.norm1.bias", "norm1.bias", "torch_norm_bias",
           torch_bias_1d_shape(unet_tensor(tensors, f"{prefix}.norm1.bias"))),
        wt("unet.middle_block.1.transformer_blocks.0.norm2.weight", "norm2.weight", "torch_norm_weight",
           torch_bias_1d_shape(unet_tensor(tensors, f"{prefix}.norm2.weight"))),
        wt("unet.middle_block.1.transformer_blocks.0.norm2.bias", "norm2.bias", "torch_norm_bias",
           torch_bias_1d_shape(unet_tensor(tensors, f"{prefix}.norm2.bias"))),
        wt("unet.middle_block.1.transformer_blocks.0.norm3.weight", "norm3.weight", "torch_norm_weight",
           torch_bias_1d_shape(unet_tensor(tensors, f"{prefix}.norm3.weight"))),
        wt("unet.middle_block.1.transformer_blocks.0.norm3.bias", "norm3.bias", "torch_norm_bias",
           torch_bias_1d_shape(unet_tensor(tensors, f"{prefix}.norm3.bias"))),
        wt("unet.middle_block.1.transformer_blocks.0.attn1.to_q.weight", "attn1.to_q.weight",
           "torch_linear_weight", torch_linear_weight_shape(unet_tensor(tensors, f"{prefix}.attn1.to_q.weight"))),
        wt("unet.middle_block.1.transformer_blocks.0.attn1.to_k.weight", "attn1.to_k.weight",
           "torch_linear_weight", torch_linear_weight_shape(unet_tensor(tensors, f"{prefix}.attn1.to_k.weight"))),
        wt("unet.middle_block.1.transformer_blocks.0.attn1.to_v.weight", "attn1.to_v.weight",
           "torch_linear_weight", torch_linear_weight_shape(unet_tensor(tensors, f"{prefix}.attn1.to_v.weight"))),
        wt("unet.middle_block.1.transformer_blocks.0.attn1.to_out.0.weight", "attn1.to_out.0.weight",
           "torch_linear_weight", torch_linear_weight_shape(unet_tensor(tensors, f"{prefix}.attn1.to_out.0.weight"))),
        wt("unet.middle_block.1.transformer_blocks.0.attn1.to_out.0.bias", "attn1.to_out.0.bias",
           "torch_bias_1d", torch_bias_1d_shape(unet_tensor(tensors, f"{prefix}.attn1.to_out.0.bias"))),
        wt("unet.middle_block.1.transformer_blocks.0.attn2.to_q.weight", "attn2.to_q.weight",
           "torch_linear_weight", torch_linear_weight_shape(unet_tensor(tensors, f"{prefix}.attn2.to_q.weight"))),
        wt("unet.middle_block.1.transformer_blocks.0.attn2.to_k.weight", "attn2.to_k.weight",
           "torch_linear_weight", torch_linear_weight_shape(unet_tensor(tensors, f"{prefix}.attn2.to_k.weight"))),
        wt("unet.middle_block.1.transformer_blocks.0.attn2.to_v.weight", "attn2.to_v.weight",
           "torch_linear_weight", torch_linear_weight_shape(unet_tensor(tensors, f"{prefix}.attn2.to_v.weight"))),
        wt("unet.middle_block.1.transformer_blocks.0.attn2.to_out.0.weight", "attn2.to_out.0.weight",
           "torch_linear_weight", torch_linear_weight_shape(unet_tensor(tensors, f"{prefix}.attn2.to_out.0.weight"))),
        wt("unet.middle_block.1.transformer_blocks.0.attn2.to_out.0.bias", "attn2.to_out.0.bias",
           "torch_bias_1d", torch_bias_1d_shape(unet_tensor(tensors, f"{prefix}.attn2.to_out.0.bias"))),
        wt("unet.middle_block.1.transformer_blocks.0.ff.net.0.proj.weight", "ff.net.0.proj.weight",
           "torch_linear_weight", torch_linear_weight_shape(unet_tensor(tensors, f"{prefix}.ff.net.0.proj.weight"))),
        wt("unet.middle_block.1.transformer_blocks.0.ff.net.0.proj.bias", "ff.net.0.proj.bias",
           "torch_bias_1d", torch_bias_1d_shape(unet_tensor(tensors, f"{prefix}.ff.net.0.proj.bias"))),
        wt("unet.middle_block.1.transformer_blocks.0.ff.net.2.weight", "ff.net.2.weight",
           "torch_linear_weight", torch_linear_weight_shape(unet_tensor(tensors, f"{prefix}.ff.net.2.weight"))),
        wt("unet.middle_block.1.transformer_blocks.0.ff.net.2.bias", "ff.net.2.bias",
           "torch_bias_1d", torch_bias_1d_shape(unet_tensor(tensors, f"{prefix}.ff.net.2.bias"))),
    ]
    for bias_suffix in (
        "attn1.to_q.bias",
        "attn1.to_k.bias",
        "attn1.to_v.bias",
        "attn2.to_q.bias",
        "attn2.to_k.bias",
        "attn2.to_v.bias",
    ):
        full_name = f"model.diffusion_model.{prefix}.{bias_suffix}"
        if full_name in tensors:
            tensor_entries.append(
                wt(
                    f"unet.middle_block.1.transformer_blocks.0.{bias_suffix}",
                    bias_suffix,
                    "torch_bias_1d",
                    torch_bias_1d_shape(tensors[full_name]),
                )
            )
    return {
        "format": "litenn.torch_manifest.v1",
        "metadata": {
            "probe": "spatial-transformer-smoke",
            "description": "fixed-shape SDXL transformer self-attention plus cross-attention over token tensors",
            "limitations": [
                "4D spatial flatten/unflatten is intentionally outside this smoke manifest",
            ],
        },
        "inputs": [
            {"name": "tokens", "dtype": "torch.float32", "shape": [tokens, channel_count]},
            {"name": "context", "dtype": "torch.float32", "shape": [context_tokens, context_width]},
        ],
        "tensors": tensor_entries,
        "nodes": [
            {
                "name": "spatial_norm1",
                "op": "layer_norm",
                "input": "tokens",
                "weight": "unet.middle_block.1.transformer_blocks.0.norm1.weight",
                "bias": "unet.middle_block.1.transformer_blocks.0.norm1.bias",
                "axis": 1,
                "eps": 1e-5,
                "output": "norm1",
            },
            {
                "name": "spatial_attn1",
                "op": "attention_block",
                "input": "norm1",
                "heads": sdxl_unet_heads(channel_count),
                "q": maybe_bias("unet.middle_block.1.transformer_blocks.0.attn1.to_q.weight", "attn1.to_q.bias"),
                "k": maybe_bias("unet.middle_block.1.transformer_blocks.0.attn1.to_k.weight", "attn1.to_k.bias"),
                "v": maybe_bias("unet.middle_block.1.transformer_blocks.0.attn1.to_v.weight", "attn1.to_v.bias"),
                "out": linear_spec(
                    "unet.middle_block.1.transformer_blocks.0.attn1.to_out.0.weight",
                    "unet.middle_block.1.transformer_blocks.0.attn1.to_out.0.bias",
                ),
                "residual": False,
                "output": "attn1_delta",
            },
            {"name": "spatial_attn1_residual", "op": "add", "lhs": "tokens", "rhs": "attn1_delta", "output": "attn1"},
            {
                "name": "spatial_norm2",
                "op": "layer_norm",
                "input": "attn1",
                "weight": "unet.middle_block.1.transformer_blocks.0.norm2.weight",
                "bias": "unet.middle_block.1.transformer_blocks.0.norm2.bias",
                "axis": 1,
                "eps": 1e-5,
                "output": "norm2",
            },
            {
                "name": "spatial_attn2",
                "op": "attention_block",
                "input": "norm2",
                "context": "context",
                "heads": sdxl_unet_heads(channel_count),
                "q": maybe_bias("unet.middle_block.1.transformer_blocks.0.attn2.to_q.weight", "attn2.to_q.bias"),
                "k": maybe_bias("unet.middle_block.1.transformer_blocks.0.attn2.to_k.weight", "attn2.to_k.bias"),
                "v": maybe_bias("unet.middle_block.1.transformer_blocks.0.attn2.to_v.weight", "attn2.to_v.bias"),
                "out": linear_spec(
                    "unet.middle_block.1.transformer_blocks.0.attn2.to_out.0.weight",
                    "unet.middle_block.1.transformer_blocks.0.attn2.to_out.0.bias",
                ),
                "residual": False,
                "output": "attn2_delta",
            },
            {"name": "spatial_attn2_residual", "op": "add", "lhs": "attn1", "rhs": "attn2_delta", "output": "attn2"},
            {
                "name": "spatial_norm3",
                "op": "layer_norm",
                "input": "attn2",
                "weight": "unet.middle_block.1.transformer_blocks.0.norm3.weight",
                "bias": "unet.middle_block.1.transformer_blocks.0.norm3.bias",
                "axis": 1,
                "eps": 1e-5,
                "output": "norm3",
            },
            {
                "name": "spatial_ff",
                "op": "geglu_feed_forward",
                "input": "norm3",
                "proj": {
                    "weight": "unet.middle_block.1.transformer_blocks.0.ff.net.0.proj.weight",
                    "bias": "unet.middle_block.1.transformer_blocks.0.ff.net.0.proj.bias",
                },
                "down": {
                    "weight": "unet.middle_block.1.transformer_blocks.0.ff.net.2.weight",
                    "bias": "unet.middle_block.1.transformer_blocks.0.ff.net.2.bias",
                },
                "residual": False,
                "output": "ff_delta",
            },
            {"name": "spatial_ff_residual", "op": "add", "lhs": "attn2", "rhs": "ff_delta", "output": "tokens_out"},
        ],
        "outputs": [{"name": "tokens_out", "source": "tokens_out"}],
    }


def emit_spatial_transformer_2d_smoke_manifest(
    tensors: dict[str, TensorInfo],
    *,
    batch: int,
    height: int,
    width: int,
    context_tokens: int,
) -> dict[str, Any]:
    if batch != 1:
        raise ValueError("spatial-transformer-2d-smoke currently requires --batch 1")
    st_prefix = "middle_block.1"
    block_prefix = f"{st_prefix}.transformer_blocks.0"
    channel_count = unet_tensor(tensors, f"{st_prefix}.norm.weight").shape[0]
    context_width = unet_tensor(tensors, f"{block_prefix}.attn2.to_k.weight").shape[1]
    feature_h = max(height // 32, 1)
    feature_w = max(width // 32, 1)
    compute_dtype = "F32"

    def st_tensor(name: str, suffix: str, layout: str, shape: list[int] | None = None) -> dict[str, Any]:
        return manifest_unet_tensor(tensors, name, f"{st_prefix}.{suffix}", layout, shape, target_dtype=compute_dtype)

    def block_tensor(name: str, suffix: str, layout: str, shape: list[int] | None = None) -> dict[str, Any]:
        return manifest_unet_tensor(tensors, name, f"{block_prefix}.{suffix}", layout, shape, target_dtype=compute_dtype)

    def maybe_bias(name: str, suffix: str) -> dict[str, str]:
        spec = {"weight": name}
        bias_source = f"model.diffusion_model.{block_prefix}.{suffix}"
        if bias_source in tensors:
            spec["bias"] = name.replace(".weight", ".bias")
        return spec

    tensors_out = [
        st_tensor("unet.middle_block.1.norm.weight", "norm.weight", "torch_groupnorm_weight",
                  torch_groupnorm_shape(unet_tensor(tensors, f"{st_prefix}.norm.weight"))),
        st_tensor("unet.middle_block.1.norm.bias", "norm.bias", "torch_groupnorm_bias",
                  torch_groupnorm_shape(unet_tensor(tensors, f"{st_prefix}.norm.bias"))),
        st_tensor("unet.middle_block.1.proj_in.weight", "proj_in.weight", "torch_linear_weight",
                  torch_linear_weight_shape(unet_tensor(tensors, f"{st_prefix}.proj_in.weight"))),
        st_tensor("unet.middle_block.1.proj_in.bias", "proj_in.bias", "torch_bias_1d",
                  torch_bias_1d_shape(unet_tensor(tensors, f"{st_prefix}.proj_in.bias"))),
        st_tensor("unet.middle_block.1.proj_out.weight", "proj_out.weight", "torch_linear_weight",
                  torch_linear_weight_shape(unet_tensor(tensors, f"{st_prefix}.proj_out.weight"))),
        st_tensor("unet.middle_block.1.proj_out.bias", "proj_out.bias", "torch_bias_1d",
                  torch_bias_1d_shape(unet_tensor(tensors, f"{st_prefix}.proj_out.bias"))),
        block_tensor("unet.middle_block.1.transformer_blocks.0.norm1.weight", "norm1.weight", "torch_norm_weight",
                     torch_bias_1d_shape(unet_tensor(tensors, f"{block_prefix}.norm1.weight"))),
        block_tensor("unet.middle_block.1.transformer_blocks.0.norm1.bias", "norm1.bias", "torch_norm_bias",
                     torch_bias_1d_shape(unet_tensor(tensors, f"{block_prefix}.norm1.bias"))),
        block_tensor("unet.middle_block.1.transformer_blocks.0.norm2.weight", "norm2.weight", "torch_norm_weight",
                     torch_bias_1d_shape(unet_tensor(tensors, f"{block_prefix}.norm2.weight"))),
        block_tensor("unet.middle_block.1.transformer_blocks.0.norm2.bias", "norm2.bias", "torch_norm_bias",
                     torch_bias_1d_shape(unet_tensor(tensors, f"{block_prefix}.norm2.bias"))),
        block_tensor("unet.middle_block.1.transformer_blocks.0.norm3.weight", "norm3.weight", "torch_norm_weight",
                     torch_bias_1d_shape(unet_tensor(tensors, f"{block_prefix}.norm3.weight"))),
        block_tensor("unet.middle_block.1.transformer_blocks.0.norm3.bias", "norm3.bias", "torch_norm_bias",
                     torch_bias_1d_shape(unet_tensor(tensors, f"{block_prefix}.norm3.bias"))),
        block_tensor("unet.middle_block.1.transformer_blocks.0.attn1.to_q.weight", "attn1.to_q.weight",
                     "torch_linear_weight", torch_linear_weight_shape(unet_tensor(tensors, f"{block_prefix}.attn1.to_q.weight"))),
        block_tensor("unet.middle_block.1.transformer_blocks.0.attn1.to_k.weight", "attn1.to_k.weight",
                     "torch_linear_weight", torch_linear_weight_shape(unet_tensor(tensors, f"{block_prefix}.attn1.to_k.weight"))),
        block_tensor("unet.middle_block.1.transformer_blocks.0.attn1.to_v.weight", "attn1.to_v.weight",
                     "torch_linear_weight", torch_linear_weight_shape(unet_tensor(tensors, f"{block_prefix}.attn1.to_v.weight"))),
        block_tensor("unet.middle_block.1.transformer_blocks.0.attn1.to_out.0.weight", "attn1.to_out.0.weight",
                     "torch_linear_weight", torch_linear_weight_shape(unet_tensor(tensors, f"{block_prefix}.attn1.to_out.0.weight"))),
        block_tensor("unet.middle_block.1.transformer_blocks.0.attn1.to_out.0.bias", "attn1.to_out.0.bias",
                     "torch_bias_1d", torch_bias_1d_shape(unet_tensor(tensors, f"{block_prefix}.attn1.to_out.0.bias"))),
        block_tensor("unet.middle_block.1.transformer_blocks.0.attn2.to_q.weight", "attn2.to_q.weight",
                     "torch_linear_weight", torch_linear_weight_shape(unet_tensor(tensors, f"{block_prefix}.attn2.to_q.weight"))),
        block_tensor("unet.middle_block.1.transformer_blocks.0.attn2.to_k.weight", "attn2.to_k.weight",
                     "torch_linear_weight", torch_linear_weight_shape(unet_tensor(tensors, f"{block_prefix}.attn2.to_k.weight"))),
        block_tensor("unet.middle_block.1.transformer_blocks.0.attn2.to_v.weight", "attn2.to_v.weight",
                     "torch_linear_weight", torch_linear_weight_shape(unet_tensor(tensors, f"{block_prefix}.attn2.to_v.weight"))),
        block_tensor("unet.middle_block.1.transformer_blocks.0.attn2.to_out.0.weight", "attn2.to_out.0.weight",
                     "torch_linear_weight", torch_linear_weight_shape(unet_tensor(tensors, f"{block_prefix}.attn2.to_out.0.weight"))),
        block_tensor("unet.middle_block.1.transformer_blocks.0.attn2.to_out.0.bias", "attn2.to_out.0.bias",
                     "torch_bias_1d", torch_bias_1d_shape(unet_tensor(tensors, f"{block_prefix}.attn2.to_out.0.bias"))),
        block_tensor("unet.middle_block.1.transformer_blocks.0.ff.net.0.proj.weight", "ff.net.0.proj.weight",
                     "torch_linear_weight", torch_linear_weight_shape(unet_tensor(tensors, f"{block_prefix}.ff.net.0.proj.weight"))),
        block_tensor("unet.middle_block.1.transformer_blocks.0.ff.net.0.proj.bias", "ff.net.0.proj.bias",
                     "torch_bias_1d", torch_bias_1d_shape(unet_tensor(tensors, f"{block_prefix}.ff.net.0.proj.bias"))),
        block_tensor("unet.middle_block.1.transformer_blocks.0.ff.net.2.weight", "ff.net.2.weight",
                     "torch_linear_weight", torch_linear_weight_shape(unet_tensor(tensors, f"{block_prefix}.ff.net.2.weight"))),
        block_tensor("unet.middle_block.1.transformer_blocks.0.ff.net.2.bias", "ff.net.2.bias",
                     "torch_bias_1d", torch_bias_1d_shape(unet_tensor(tensors, f"{block_prefix}.ff.net.2.bias"))),
    ]
    for bias_suffix in (
        "attn1.to_q.bias",
        "attn1.to_k.bias",
        "attn1.to_v.bias",
        "attn2.to_q.bias",
        "attn2.to_k.bias",
        "attn2.to_v.bias",
    ):
        full_name = f"model.diffusion_model.{block_prefix}.{bias_suffix}"
        if full_name in tensors:
            tensors_out.append(
                block_tensor(
                    f"unet.middle_block.1.transformer_blocks.0.{bias_suffix}",
                    bias_suffix,
                    "torch_bias_1d",
                    torch_bias_1d_shape(tensors[full_name]),
                )
            )

    return {
        "format": "litenn.torch_manifest.v1",
        "metadata": {
            "probe": "spatial-transformer-2d-smoke",
            "description": "fixed-shape batch=1 SDXL middle-block SpatialTransformer over NCHW features",
            "feature_height": feature_h,
            "feature_width": feature_w,
        },
        "inputs": [
            {"name": "features", "dtype": "torch.float32", "shape": [batch, channel_count, feature_h, feature_w]},
            {"name": "context", "dtype": "torch.float32", "shape": [context_tokens, context_width]},
        ],
        "tensors": tensors_out,
        "nodes": [
            {
                "name": "middle_spatial_transformer",
                "op": "spatial_transformer_2d",
                "input": "features",
                "context": "context",
                "use_linear": True,
                "norm": {
                    "weight": "unet.middle_block.1.norm.weight",
                    "bias": "unet.middle_block.1.norm.bias",
                    "num_groups": 32,
                    "eps": 1e-6,
                    "layout": "pytorch",
                },
                "proj_in": {
                    "weight": "unet.middle_block.1.proj_in.weight",
                    "bias": "unet.middle_block.1.proj_in.bias",
                },
                "blocks": [
                    {
                        "norm1": {
                            "weight": "unet.middle_block.1.transformer_blocks.0.norm1.weight",
                            "bias": "unet.middle_block.1.transformer_blocks.0.norm1.bias",
                            "axis": 1,
                            "eps": 1e-5,
                        },
                        "attn1": {
                            "heads": sdxl_unet_heads(channel_count),
                            "q": maybe_bias("unet.middle_block.1.transformer_blocks.0.attn1.to_q.weight", "attn1.to_q.bias"),
                            "k": maybe_bias("unet.middle_block.1.transformer_blocks.0.attn1.to_k.weight", "attn1.to_k.bias"),
                            "v": maybe_bias("unet.middle_block.1.transformer_blocks.0.attn1.to_v.weight", "attn1.to_v.bias"),
                            "out": linear_spec(
                                "unet.middle_block.1.transformer_blocks.0.attn1.to_out.0.weight",
                                "unet.middle_block.1.transformer_blocks.0.attn1.to_out.0.bias",
                            ),
                            "residual": False,
                        },
                        "norm2": {
                            "weight": "unet.middle_block.1.transformer_blocks.0.norm2.weight",
                            "bias": "unet.middle_block.1.transformer_blocks.0.norm2.bias",
                            "axis": 1,
                            "eps": 1e-5,
                        },
                        "attn2": {
                            "heads": sdxl_unet_heads(channel_count),
                            "q": maybe_bias("unet.middle_block.1.transformer_blocks.0.attn2.to_q.weight", "attn2.to_q.bias"),
                            "k": maybe_bias("unet.middle_block.1.transformer_blocks.0.attn2.to_k.weight", "attn2.to_k.bias"),
                            "v": maybe_bias("unet.middle_block.1.transformer_blocks.0.attn2.to_v.weight", "attn2.to_v.bias"),
                            "out": linear_spec(
                                "unet.middle_block.1.transformer_blocks.0.attn2.to_out.0.weight",
                                "unet.middle_block.1.transformer_blocks.0.attn2.to_out.0.bias",
                            ),
                            "residual": False,
                        },
                        "norm3": {
                            "weight": "unet.middle_block.1.transformer_blocks.0.norm3.weight",
                            "bias": "unet.middle_block.1.transformer_blocks.0.norm3.bias",
                            "axis": 1,
                            "eps": 1e-5,
                        },
                        "ff": {
                            "proj": {
                                "weight": "unet.middle_block.1.transformer_blocks.0.ff.net.0.proj.weight",
                                "bias": "unet.middle_block.1.transformer_blocks.0.ff.net.0.proj.bias",
                            },
                            "down": {
                                "weight": "unet.middle_block.1.transformer_blocks.0.ff.net.2.weight",
                                "bias": "unet.middle_block.1.transformer_blocks.0.ff.net.2.bias",
                            },
                            "residual": False,
                        },
                    }
                ],
                "proj_out": {
                    "weight": "unet.middle_block.1.proj_out.weight",
                    "bias": "unet.middle_block.1.proj_out.bias",
                },
                "output": "features_out",
            }
        ],
        "outputs": [{"name": "features_out", "source": "features_out"}],
    }


def emit_unet_full_fixed_manifest(
    tensors: dict[str, TensorInfo],
    *,
    batch: int,
    height: int,
    width: int,
    context_tokens: int,
) -> dict[str, Any]:
    if batch != 1:
        raise ValueError("unet-full-fixed currently emits batch=1 SpatialTransformer blocks")
    latent_h = height // 8
    latent_w = width // 8
    compute_dtype = "F32"
    tensor_entries: list[dict[str, Any]] = []
    seen_tensors: set[str] = set()
    nodes: list[dict[str, Any]] = []
    unsupported: list[str] = []

    def has(source_suffix: str) -> bool:
        return f"model.diffusion_model.{source_suffix}" in tensors

    def wt(source_suffix: str) -> TensorInfo:
        return unet_tensor(tensors, source_suffix)

    def add_tensor(
        manifest_name: str,
        source_suffix: str,
        layout: str = "identity",
        shape: list[int] | None = None,
    ) -> None:
        if manifest_name in seen_tensors:
            return
        seen_tensors.add(manifest_name)
        tensor_entries.append(
            manifest_unet_tensor(tensors, manifest_name, source_suffix, layout, shape, target_dtype=compute_dtype)
        )

    def add_groupnorm_tensors(source_prefix: str, manifest_prefix: str, norm_name: str) -> None:
        weight = wt(f"{source_prefix}.{norm_name}.weight")
        bias = wt(f"{source_prefix}.{norm_name}.bias")
        add_tensor(f"{manifest_prefix}.{norm_name}.weight", f"{source_prefix}.{norm_name}.weight",
                   "torch_groupnorm_weight", torch_groupnorm_shape(weight))
        add_tensor(f"{manifest_prefix}.{norm_name}.bias", f"{source_prefix}.{norm_name}.bias",
                   "torch_groupnorm_bias", torch_groupnorm_shape(bias))

    def add_conv_tensors(source_prefix: str, manifest_prefix: str, conv_name: str) -> None:
        source_name = f"{source_prefix}.{conv_name}" if source_prefix else conv_name
        weight = wt(f"{source_name}.weight")
        bias = wt(f"{source_name}.bias")
        add_tensor(f"{manifest_prefix}.{conv_name}.weight", f"{source_name}.weight",
                   "torch_conv2d_weight", weight.shape)
        add_tensor(f"{manifest_prefix}.{conv_name}.bias", f"{source_name}.bias", "identity", bias.shape)

    def add_linear_tensors(source_prefix: str, manifest_prefix: str, linear_name: str) -> None:
        source_name = f"{source_prefix}.{linear_name}" if source_prefix else linear_name
        weight = wt(f"{source_name}.weight")
        add_tensor(f"{manifest_prefix}.{linear_name}.weight", f"{source_name}.weight",
                   "torch_linear_weight", torch_linear_weight_shape(weight))
        if has(f"{source_name}.bias"):
            bias = wt(f"{source_name}.bias")
            add_tensor(f"{manifest_prefix}.{linear_name}.bias", f"{source_name}.bias",
                       "torch_bias_1d", torch_bias_1d_shape(bias))

    def add_1d_affine_tensors(source_prefix: str, manifest_prefix: str, name: str) -> None:
        weight = wt(f"{source_prefix}.{name}.weight")
        bias = wt(f"{source_prefix}.{name}.bias")
        add_tensor(f"{manifest_prefix}.{name}.weight", f"{source_prefix}.{name}.weight",
                   "torch_norm_weight", torch_bias_1d_shape(weight))
        add_tensor(f"{manifest_prefix}.{name}.bias", f"{source_prefix}.{name}.bias",
                   "torch_norm_bias", torch_bias_1d_shape(bias))

    def linear_or_bias_spec(manifest_prefix: str, source_prefix: str, linear_name: str) -> dict[str, str]:
        spec = {"weight": f"{manifest_prefix}.{linear_name}.weight"}
        if has(f"{source_prefix}.{linear_name}.bias"):
            spec["bias"] = f"{manifest_prefix}.{linear_name}.bias"
        return spec

    def has_resblock(source_prefix: str) -> bool:
        return has(f"{source_prefix}.in_layers.0.weight") and has(f"{source_prefix}.out_layers.3.weight")

    def has_downsample(source_prefix: str) -> bool:
        return has(f"{source_prefix}.op.weight")

    def has_upsample(source_prefix: str) -> bool:
        return has(f"{source_prefix}.conv.weight")

    def has_spatial_transformer(source_prefix: str) -> bool:
        return has(f"{source_prefix}.norm.weight") and has(f"{source_prefix}.proj_in.weight")

    def transformer_block_indices(source_prefix: str) -> list[int]:
        result = {
            int(match.group(1))
            for name in tensors
            if (match := re.match(
                rf"model\.diffusion_model\.{re.escape(source_prefix)}\.transformer_blocks\.(\d+)\.",
                name,
            ))
        }
        return sorted(result)

    def add_resblock(source_prefix: str, current: str) -> str:
        if has(f"{source_prefix}.h_upd.in_layers.0.weight") or has(f"{source_prefix}.x_upd.op.weight"):
            unsupported.append(f"{source_prefix}: ResBlock up/down variant is not supported by residual_block")
        manifest_prefix = "unet." + source_prefix
        add_groupnorm_tensors(source_prefix, manifest_prefix, "in_layers.0")
        add_conv_tensors(source_prefix, manifest_prefix, "in_layers.2")
        add_linear_tensors(source_prefix, manifest_prefix, "emb_layers.1")
        add_groupnorm_tensors(source_prefix, manifest_prefix, "out_layers.0")
        add_conv_tensors(source_prefix, manifest_prefix, "out_layers.3")

        block: dict[str, Any] = {
            "name": node_name(source_prefix),
            "op": "residual_block",
            "input": current,
            "temb": "temb",
            "activation": "silu",
            "norm1": {
                "weight": f"{manifest_prefix}.in_layers.0.weight",
                "bias": f"{manifest_prefix}.in_layers.0.bias",
                "num_groups": 32,
                "eps": 1e-5,
                "layout": "pytorch",
            },
            "conv1": {
                "weight": f"{manifest_prefix}.in_layers.2.weight",
                "bias": f"{manifest_prefix}.in_layers.2.bias",
                "padding": [1, 1],
            },
            "temb_projection": {
                "weight": f"{manifest_prefix}.emb_layers.1.weight",
                "bias": f"{manifest_prefix}.emb_layers.1.bias",
            },
            "norm2": {
                "weight": f"{manifest_prefix}.out_layers.0.weight",
                "bias": f"{manifest_prefix}.out_layers.0.bias",
                "num_groups": 32,
                "eps": 1e-5,
                "layout": "pytorch",
            },
            "conv2": {
                "weight": f"{manifest_prefix}.out_layers.3.weight",
                "bias": f"{manifest_prefix}.out_layers.3.bias",
                "padding": [1, 1],
            },
            "output": node_name(source_prefix + ".out"),
        }
        if has(f"{source_prefix}.skip_connection.weight"):
            add_conv_tensors(source_prefix, manifest_prefix, "skip_connection")
            block["skip"] = {
                "weight": f"{manifest_prefix}.skip_connection.weight",
                "bias": f"{manifest_prefix}.skip_connection.bias",
                "padding": [0, 0],
            }
        nodes.append(block)
        return block["output"]

    def add_downsample(source_prefix: str, current: str) -> str:
        manifest_prefix = "unet." + source_prefix
        add_conv_tensors(source_prefix, manifest_prefix, "op")
        output = node_name(source_prefix + ".down")
        nodes.append(
            {
                "name": node_name(source_prefix),
                "op": "conv2d",
                "input": current,
                "weight": f"{manifest_prefix}.op.weight",
                "bias": f"{manifest_prefix}.op.bias",
                "stride": [2, 2],
                "padding": [1, 1],
                "output": output,
            }
        )
        return output

    def add_upsample(source_prefix: str, current: str, next_h: int, next_w: int) -> str:
        manifest_prefix = "unet." + source_prefix
        resize_output = node_name(source_prefix + ".resize")
        nodes.append(
            {
                "name": node_name(source_prefix + ".resize"),
                "op": "upsample",
                "input": current,
                "mode": "nearest",
                "output_spatial_shape": [next_h, next_w],
                "output": resize_output,
            }
        )
        add_conv_tensors(source_prefix, manifest_prefix, "conv")
        output = node_name(source_prefix + ".out")
        nodes.append(
            {
                "name": node_name(source_prefix + ".conv"),
                "op": "conv2d",
                "input": resize_output,
                "weight": f"{manifest_prefix}.conv.weight",
                "bias": f"{manifest_prefix}.conv.bias",
                "padding": [1, 1],
                "output": output,
            }
        )
        return output

    def add_spatial_transformer(source_prefix: str, current: str) -> str:
        manifest_prefix = "unet." + source_prefix
        add_groupnorm_tensors(source_prefix, manifest_prefix, "norm")
        proj_in_weight = wt(f"{source_prefix}.proj_in.weight")
        proj_out_weight = wt(f"{source_prefix}.proj_out.weight")
        use_linear = len(proj_in_weight.shape) == 2
        if use_linear:
            add_linear_tensors(source_prefix, manifest_prefix, "proj_in")
            add_linear_tensors(source_prefix, manifest_prefix, "proj_out")
        elif len(proj_in_weight.shape) == 4 and len(proj_out_weight.shape) == 4:
            add_conv_tensors(source_prefix, manifest_prefix, "proj_in")
            add_conv_tensors(source_prefix, manifest_prefix, "proj_out")
        else:
            unsupported.append(f"{source_prefix}: unsupported SpatialTransformer proj_in/proj_out ranks")

        blocks: list[dict[str, Any]] = []
        for block_index in transformer_block_indices(source_prefix):
            block_source = f"{source_prefix}.transformer_blocks.{block_index}"
            block_manifest = f"{manifest_prefix}.transformer_blocks.{block_index}"
            for norm_name in ("norm1", "norm2", "norm3"):
                add_1d_affine_tensors(block_source, block_manifest, norm_name)
            for linear_name in (
                "attn1.to_q",
                "attn1.to_k",
                "attn1.to_v",
                "attn1.to_out.0",
                "attn2.to_q",
                "attn2.to_k",
                "attn2.to_v",
                "attn2.to_out.0",
                "ff.net.0.proj",
                "ff.net.2",
            ):
                add_linear_tensors(block_source, block_manifest, linear_name)
            width = wt(f"{block_source}.attn1.to_q.weight").shape[0]
            blocks.append(
                {
                    "norm1": {
                        "weight": f"{block_manifest}.norm1.weight",
                        "bias": f"{block_manifest}.norm1.bias",
                        "axis": 1,
                        "eps": 1e-5,
                    },
                    "attn1": {
                        "heads": sdxl_unet_heads(width),
                        "q": linear_or_bias_spec(block_manifest, block_source, "attn1.to_q"),
                        "k": linear_or_bias_spec(block_manifest, block_source, "attn1.to_k"),
                        "v": linear_or_bias_spec(block_manifest, block_source, "attn1.to_v"),
                        "out": linear_or_bias_spec(block_manifest, block_source, "attn1.to_out.0"),
                        "residual": False,
                    },
                    "norm2": {
                        "weight": f"{block_manifest}.norm2.weight",
                        "bias": f"{block_manifest}.norm2.bias",
                        "axis": 1,
                        "eps": 1e-5,
                    },
                    "attn2": {
                        "heads": sdxl_unet_heads(width),
                        "q": linear_or_bias_spec(block_manifest, block_source, "attn2.to_q"),
                        "k": linear_or_bias_spec(block_manifest, block_source, "attn2.to_k"),
                        "v": linear_or_bias_spec(block_manifest, block_source, "attn2.to_v"),
                        "out": linear_or_bias_spec(block_manifest, block_source, "attn2.to_out.0"),
                        "residual": False,
                    },
                    "norm3": {
                        "weight": f"{block_manifest}.norm3.weight",
                        "bias": f"{block_manifest}.norm3.bias",
                        "axis": 1,
                        "eps": 1e-5,
                    },
                    "ff": {
                        "proj": linear_or_bias_spec(block_manifest, block_source, "ff.net.0.proj"),
                        "down": linear_or_bias_spec(block_manifest, block_source, "ff.net.2"),
                        "residual": False,
                    },
                }
            )
        if not blocks:
            unsupported.append(f"{source_prefix}: SpatialTransformer has no transformer_blocks")

        output = node_name(source_prefix + ".out")
        node: dict[str, Any] = {
            "name": node_name(source_prefix),
            "op": "spatial_transformer_2d",
            "input": current,
            "context": "context",
            "use_linear": use_linear,
            "norm": {
                "weight": f"{manifest_prefix}.norm.weight",
                "bias": f"{manifest_prefix}.norm.bias",
                "num_groups": 32,
                "eps": 1e-6,
                "layout": "pytorch",
            },
            "proj_in": {
                "weight": f"{manifest_prefix}.proj_in.weight",
                "bias": f"{manifest_prefix}.proj_in.bias",
            },
            "blocks": blocks,
            "proj_out": {
                "weight": f"{manifest_prefix}.proj_out.weight",
                "bias": f"{manifest_prefix}.proj_out.bias",
            },
            "output": output,
        }
        nodes.append(node)
        return output

    def process_module(source_prefix: str, current: str, h: int, w: int) -> tuple[str, int, int]:
        if has_resblock(source_prefix):
            current = add_resblock(source_prefix, current)
            return current, h, w
        if has_spatial_transformer(source_prefix):
            current = add_spatial_transformer(source_prefix, current)
            return current, h, w
        if has_downsample(source_prefix):
            current = add_downsample(source_prefix, current)
            return current, (h + 1) // 2, (w + 1) // 2
        if has_upsample(source_prefix):
            next_h = h * 2
            next_w = w * 2
            current = add_upsample(source_prefix, current, next_h, next_w)
            return current, next_h, next_w
        unsupported.append(f"{source_prefix}: no supported module tensor pattern")
        return current, h, w

    stem_weight = wt("input_blocks.0.0.weight")
    add_tensor("unet.input_blocks.0.0.weight", "input_blocks.0.0.weight", "torch_conv2d_weight", stem_weight.shape)
    add_tensor("unet.input_blocks.0.0.bias", "input_blocks.0.0.bias", "identity",
               wt("input_blocks.0.0.bias").shape)
    for linear_name in ("time_embed.0", "time_embed.2", "label_emb.0.0", "label_emb.0.2"):
        add_linear_tensors("", "unet", linear_name)
    out_norm_weight = wt("out.0.weight")
    out_norm_bias = wt("out.0.bias")
    add_tensor("unet.out.0.weight", "out.0.weight", "torch_groupnorm_weight", torch_groupnorm_shape(out_norm_weight))
    add_tensor("unet.out.0.bias", "out.0.bias", "torch_groupnorm_bias", torch_groupnorm_shape(out_norm_bias))
    add_tensor("unet.out.2.weight", "out.2.weight", "torch_conv2d_weight", wt("out.2.weight").shape)
    add_tensor("unet.out.2.bias", "out.2.bias", "identity", wt("out.2.bias").shape)

    model_channels = stem_weight.shape[0]
    context_width = wt("middle_block.1.transformer_blocks.0.attn2.to_k.weight").shape[1]
    vector_width = wt("label_emb.0.0.weight").shape[1]
    nodes.extend(
        [
            {
                "name": "unet_timestep_sinusoidal",
                "op": "timestep_embedding",
                "timesteps": "timestep",
                "dim": model_channels,
                "max_period": 10000,
                "output": "time_sinusoidal",
            },
            {
                "name": "unet_time_embed_0",
                "op": "linear",
                "input": "time_sinusoidal",
                "weight": "unet.time_embed.0.weight",
                "bias": "unet.time_embed.0.bias",
                "output": "time_hidden",
            },
            {"name": "unet_time_embed_act", "op": "silu", "input": "time_hidden", "output": "time_hidden_act"},
            {
                "name": "unet_time_embed_2",
                "op": "linear",
                "input": "time_hidden_act",
                "weight": "unet.time_embed.2.weight",
                "bias": "unet.time_embed.2.bias",
                "output": "time_emb",
            },
            {
                "name": "unet_label_emb_0",
                "op": "linear",
                "input": "vector_cond",
                "weight": "unet.label_emb.0.0.weight",
                "bias": "unet.label_emb.0.0.bias",
                "output": "label_hidden",
            },
            {"name": "unet_label_emb_act", "op": "silu", "input": "label_hidden", "output": "label_hidden_act"},
            {
                "name": "unet_label_emb_2",
                "op": "linear",
                "input": "label_hidden_act",
                "weight": "unet.label_emb.0.2.weight",
                "bias": "unet.label_emb.0.2.bias",
                "output": "label_emb",
            },
            {"name": "unet_conditioning_add", "op": "add", "lhs": "time_emb", "rhs": "label_emb", "output": "temb"},
            {
                "name": "unet_stem_conv",
                "op": "conv2d",
                "input": "latent",
                "weight": "unet.input_blocks.0.0.weight",
                "bias": "unet.input_blocks.0.0.bias",
                "padding": [1, 1],
                "output": "input_blocks_0_out",
            },
        ]
    )

    input_block_indices = sorted(
        {
            int(match.group(1))
            for name in tensors
            if (match := re.match(r"model\.diffusion_model\.input_blocks\.(\d+)\.", name))
        }
    )
    output_block_indices = sorted(
        {
            int(match.group(1))
            for name in tensors
            if (match := re.match(r"model\.diffusion_model\.output_blocks\.(\d+)\.", name))
        }
    )
    if not input_block_indices or input_block_indices[0] != 0:
        raise ValueError("unet-full-fixed requires input_blocks.0 stem tensors")

    current = "input_blocks_0_out"
    current_h = latent_h
    current_w = latent_w
    skip_stack: list[tuple[str, int, int]] = [(current, current_h, current_w)]
    input_block_summaries: list[dict[str, Any]] = [
        {"index": 0, "output": current, "height": current_h, "width": current_w}
    ]
    for block_index in input_block_indices[1:]:
        submodules = sorted(
            {
                int(match.group(1))
                for name in tensors
                if (match := re.match(rf"model\.diffusion_model\.input_blocks\.{block_index}\.(\d+)\.", name))
            }
        )
        for submodule in submodules:
            current, current_h, current_w = process_module(
                f"input_blocks.{block_index}.{submodule}", current, current_h, current_w
            )
        skip_stack.append((current, current_h, current_w))
        input_block_summaries.append(
            {"index": block_index, "output": current, "height": current_h, "width": current_w}
        )

    middle_summaries: list[dict[str, Any]] = []
    middle_indices = sorted(
        {
            int(match.group(1))
            for name in tensors
            if (match := re.match(r"model\.diffusion_model\.middle_block\.(\d+)\.", name))
        }
    )
    for middle_index in middle_indices:
        current, current_h, current_w = process_module(f"middle_block.{middle_index}", current, current_h, current_w)
        middle_summaries.append(
            {"index": middle_index, "output": current, "height": current_h, "width": current_w}
        )

    output_block_summaries: list[dict[str, Any]] = []
    for block_index in output_block_indices:
        if not skip_stack:
            raise ValueError("unet-full-fixed output block traversal exhausted the skip stack")
        skip_value, skip_h, skip_w = skip_stack.pop()
        if (skip_h, skip_w) != (current_h, current_w):
            unsupported.append(
                f"output_blocks.{block_index}: skip spatial {skip_h}x{skip_w} does not match current {current_h}x{current_w}"
            )
        concat_output = f"output_blocks_{block_index}_skip_concat"
        nodes.append(
            {
                "name": f"unet_output_blocks_{block_index}_skip_concat",
                "op": "concat",
                "inputs": [current, skip_value],
                "axis": 1,
                "output": concat_output,
            }
        )
        current = concat_output
        submodules = sorted(
            {
                int(match.group(1))
                for name in tensors
                if (match := re.match(rf"model\.diffusion_model\.output_blocks\.{block_index}\.(\d+)\.", name))
            }
        )
        for submodule in submodules:
            current, current_h, current_w = process_module(
                f"output_blocks.{block_index}.{submodule}", current, current_h, current_w
            )
        output_block_summaries.append(
            {"index": block_index, "output": current, "height": current_h, "width": current_w}
        )

    if skip_stack:
        unsupported.append(f"UNet traversal left {len(skip_stack)} unused skip value(s)")
    if unsupported:
        raise ValueError("unet-full-fixed cannot emit an importable manifest:\n  - " + "\n  - ".join(unsupported))

    nodes.extend(
        [
            {
                "name": "unet_out_norm",
                "op": "group_norm",
                "input": current,
                "weight": "unet.out.0.weight",
                "bias": "unet.out.0.bias",
                "num_groups": 32,
                "eps": 1e-5,
                "layout": "pytorch",
                "output": "out_norm",
            },
            {"name": "unet_out_silu", "op": "silu", "input": "out_norm", "output": "out_act"},
            {
                "name": "unet_noise_pred",
                "op": "conv2d",
                "input": "out_act",
                "weight": "unet.out.2.weight",
                "bias": "unet.out.2.bias",
                "padding": [1, 1],
                "output": "noise_pred",
            },
        ]
    )

    return {
        "format": "litenn.torch_manifest.v1",
        "metadata": {
            "probe": "unet-full-fixed",
            "description": "fixed-shape SDXL UNet traversal from Stability checkpoint layout",
            "height": height,
            "width": width,
            "latent_height": latent_h,
            "latent_width": latent_w,
            "context_tokens": context_tokens,
            "input_blocks": input_block_summaries,
            "middle_blocks": middle_summaries,
            "output_blocks": output_block_summaries,
            "limitations": [
                "batch=1 SpatialTransformer lowering",
                "Stability-AI SDXL base-style resblock_updown=False topology",
                "external tokenizer/text-encoder conditioning inputs",
            ],
        },
        "inputs": [
            {"name": "latent", "dtype": "torch.float32", "shape": [batch, stem_weight.shape[1], latent_h, latent_w]},
            {"name": "timestep", "dtype": "torch.float32", "shape": [batch]},
            {"name": "context", "dtype": "torch.float32", "shape": [context_tokens, context_width]},
            {"name": "vector_cond", "dtype": "torch.float32", "shape": [batch, vector_width]},
        ],
        "tensors": tensor_entries,
        "nodes": nodes,
        "outputs": [{"name": "noise_pred", "source": "noise_pred"}],
    }


def emit_vae_decode_stem_manifest(tensors: dict[str, TensorInfo], *, batch: int, height: int, width: int) -> dict[str, Any]:
    latent_h = height // 8
    latent_w = width // 8
    weight = require_tensor(tensors, "first_stage_model.decoder.conv_in.weight")
    bias = require_tensor(tensors, "first_stage_model.decoder.conv_in.bias")
    return {
        "format": "litenn.torch_manifest.v1",
        "inputs": [
            {"name": "latent", "dtype": "torch.float16", "shape": [batch, weight.shape[1], latent_h, latent_w]},
        ],
        "tensors": [
            manifest_tensor("vae.decoder.conv_in.weight", weight, "torch_conv2d_weight"),
            manifest_tensor("vae.decoder.conv_in.bias", bias, "identity"),
        ],
        "nodes": [
            {
                "name": "vae_decoder_conv_in",
                "op": "scale",
                "input": "latent",
                "factor": 1.0 / 0.13025,
                "output": "scaled_latent",
            },
            {
                "name": "vae_decoder_conv",
                "op": "conv2d",
                "input": "scaled_latent",
                "weight": "vae.decoder.conv_in.weight",
                "bias": "vae.decoder.conv_in.bias",
                "padding": [1, 1],
                "output": "decoder_hidden",
            },
        ],
        "outputs": [{"name": "decoder_hidden", "source": "decoder_hidden"}],
    }


def emit_vae_decode_full_manifest(
    tensors: dict[str, TensorInfo],
    *,
    batch: int,
    height: int,
    width: int,
    vae_mid_attention_policy: str = "auto",
    vae_attention_max_mib: int = DEFAULT_VAE_ATTENTION_MAX_MIB,
) -> dict[str, Any]:
    latent_h = height // 8
    latent_w = width // 8
    compute_dtype = "F32"
    tensor_entries: list[dict[str, Any]] = []
    seen_tensors: set[str] = set()
    nodes: list[dict[str, Any]] = []
    vae_mid_attention_report: dict[str, Any] = {
        "status": "not_evaluated",
        "policy": vae_mid_attention_policy,
        "max_workspace_mib": vae_attention_max_mib,
        "max_workspace_bytes": vae_attention_max_mib * 1024 * 1024,
        "feature_height": latent_h,
        "feature_width": latent_w,
        "batch": batch,
        "dtype": compute_dtype,
    }

    def add_tensor(manifest_name: str, source_suffix: str, layout: str, shape: list[int] | None = None) -> None:
        if manifest_name in seen_tensors:
            return
        seen_tensors.add(manifest_name)
        tensor_entries.append(
            manifest_vae_tensor(tensors, manifest_name, source_suffix, layout, shape, target_dtype=compute_dtype)
        )

    def node_name(prefix: str) -> str:
        return re.sub(r"[^A-Za-z0-9]+", "_", prefix).strip("_")

    def add_groupnorm_tensors(source_prefix: str, manifest_prefix: str, norm_name: str) -> None:
        weight = vae_tensor(tensors, f"{source_prefix}.{norm_name}.weight")
        bias = vae_tensor(tensors, f"{source_prefix}.{norm_name}.bias")
        add_tensor(f"{manifest_prefix}.{norm_name}.weight", f"{source_prefix}.{norm_name}.weight",
                   "torch_groupnorm_weight", torch_groupnorm_shape(weight))
        add_tensor(f"{manifest_prefix}.{norm_name}.bias", f"{source_prefix}.{norm_name}.bias",
                   "torch_groupnorm_bias", torch_groupnorm_shape(bias))

    def add_conv_tensors(source_prefix: str, manifest_prefix: str, conv_name: str) -> None:
        weight = vae_tensor(tensors, f"{source_prefix}.{conv_name}.weight")
        bias = vae_tensor(tensors, f"{source_prefix}.{conv_name}.bias")
        add_tensor(f"{manifest_prefix}.{conv_name}.weight", f"{source_prefix}.{conv_name}.weight",
                   "torch_conv2d_weight", weight.shape)
        add_tensor(f"{manifest_prefix}.{conv_name}.bias", f"{source_prefix}.{conv_name}.bias", "identity", bias.shape)

    def add_resblock(source_prefix: str, current: str) -> str:
        manifest_prefix = "vae." + source_prefix
        add_groupnorm_tensors(source_prefix, manifest_prefix, "norm1")
        add_groupnorm_tensors(source_prefix, manifest_prefix, "norm2")
        add_conv_tensors(source_prefix, manifest_prefix, "conv1")
        add_conv_tensors(source_prefix, manifest_prefix, "conv2")

        block: dict[str, Any] = {
            "name": node_name(source_prefix),
            "op": "residual_block",
            "input": current,
            "activation": "silu",
            "norm1": {
                "weight": f"{manifest_prefix}.norm1.weight",
                "bias": f"{manifest_prefix}.norm1.bias",
                "num_groups": 32,
                "eps": 1e-6,
                "layout": "pytorch",
            },
            "conv1": {
                "weight": f"{manifest_prefix}.conv1.weight",
                "bias": f"{manifest_prefix}.conv1.bias",
                "padding": [1, 1],
            },
            "norm2": {
                "weight": f"{manifest_prefix}.norm2.weight",
                "bias": f"{manifest_prefix}.norm2.bias",
                "num_groups": 32,
                "eps": 1e-6,
                "layout": "pytorch",
            },
            "conv2": {
                "weight": f"{manifest_prefix}.conv2.weight",
                "bias": f"{manifest_prefix}.conv2.bias",
                "padding": [1, 1],
            },
            "output": node_name(source_prefix + ".out"),
        }

        skip_source = f"first_stage_model.{source_prefix}.nin_shortcut.weight"
        if skip_source in tensors:
            add_conv_tensors(source_prefix, manifest_prefix, "nin_shortcut")
            block["skip"] = {
                "weight": f"{manifest_prefix}.nin_shortcut.weight",
                "bias": f"{manifest_prefix}.nin_shortcut.bias",
                "padding": [0, 0],
            }
        nodes.append(block)
        return block["output"]

    def add_mid_attention(current: str, h: int, w: int) -> str:
        nonlocal vae_mid_attention_report
        source_prefix = "decoder.mid.attn_1"
        if f"first_stage_model.{source_prefix}.q.weight" not in tensors:
            vae_mid_attention_report = {
                **vae_mid_attention_report,
                "status": "absent",
                "reason": "checkpoint does not contain decoder.mid.attn_1.q.weight",
            }
            return current

        channels = vae_tensor(tensors, f"{source_prefix}.q.weight").shape[0]
        hw = h * w
        bytes_per_element = 4
        score_bytes = hw * hw * bytes_per_element
        activation_bytes = batch * channels * h * w * bytes_per_element
        estimated_workspace_bytes = score_bytes * 2 + activation_bytes * 5
        base_report = {
            **vae_mid_attention_report,
            "feature_height": h,
            "feature_width": w,
            "channels": channels,
            "tokens": hw,
            "score_bytes": score_bytes,
            "probability_bytes": score_bytes,
            "activation_bytes": activation_bytes,
            "estimated_workspace_bytes": estimated_workspace_bytes,
        }
        if batch != 1:
            vae_mid_attention_report = {
                **base_report,
                "status": "skipped",
                "reason": "only batch=1 VAE mid-attention is currently emitted",
            }
            return current
        if vae_mid_attention_policy == "skip":
            vae_mid_attention_report = {
                **base_report,
                "status": "skipped",
                "reason": "policy requested skip",
            }
            return current
        if (
            vae_mid_attention_policy == "auto"
            and estimated_workspace_bytes > vae_mid_attention_report["max_workspace_bytes"]
        ):
            vae_mid_attention_report = {
                **base_report,
                "status": "skipped",
                "reason": "estimated workspace exceeds --vae-attention-max-mib",
            }
            return current

        manifest_prefix = "vae." + source_prefix
        add_groupnorm_tensors(source_prefix, manifest_prefix, "norm")
        for conv_name in ("q", "k", "v", "proj_out"):
            add_conv_tensors(source_prefix, manifest_prefix, conv_name)

        nodes.extend(
            [
                {
                    "name": "vae_mid_attn_norm",
                    "op": "group_norm",
                    "input": current,
                    "weight": f"{manifest_prefix}.norm.weight",
                    "bias": f"{manifest_prefix}.norm.bias",
                    "num_groups": 32,
                    "eps": 1e-6,
                    "layout": "pytorch",
                    "output": "vae_mid_attn_norm",
                },
                {
                    "name": "vae_mid_attn_q",
                    "op": "conv2d",
                    "input": "vae_mid_attn_norm",
                    "weight": f"{manifest_prefix}.q.weight",
                    "bias": f"{manifest_prefix}.q.bias",
                    "padding": [0, 0],
                    "output": "vae_mid_q",
                },
                {
                    "name": "vae_mid_attn_k",
                    "op": "conv2d",
                    "input": "vae_mid_attn_norm",
                    "weight": f"{manifest_prefix}.k.weight",
                    "bias": f"{manifest_prefix}.k.bias",
                    "padding": [0, 0],
                    "output": "vae_mid_k",
                },
                {
                    "name": "vae_mid_attn_v",
                    "op": "conv2d",
                    "input": "vae_mid_attn_norm",
                    "weight": f"{manifest_prefix}.v.weight",
                    "bias": f"{manifest_prefix}.v.bias",
                    "padding": [0, 0],
                    "output": "vae_mid_v",
                },
                {"name": "vae_mid_q_flat", "op": "reshape", "input": "vae_mid_q", "shape": [channels, hw], "output": "vae_mid_q_flat"},
                {"name": "vae_mid_q_tokens", "op": "transpose", "input": "vae_mid_q_flat", "output": "vae_mid_q_tokens"},
                {"name": "vae_mid_k_flat", "op": "reshape", "input": "vae_mid_k", "shape": [channels, hw], "output": "vae_mid_k_flat"},
                {"name": "vae_mid_scores", "op": "matmul", "lhs": "vae_mid_q_tokens", "rhs": "vae_mid_k_flat", "output": "vae_mid_scores"},
                {"name": "vae_mid_scores_scale", "op": "scale", "input": "vae_mid_scores", "factor": channels ** -0.5, "output": "vae_mid_scaled"},
                {"name": "vae_mid_probs", "op": "softmax", "input": "vae_mid_scaled", "axis": 1, "output": "vae_mid_probs"},
                {"name": "vae_mid_v_flat", "op": "reshape", "input": "vae_mid_v", "shape": [channels, hw], "output": "vae_mid_v_flat"},
                {"name": "vae_mid_v_tokens", "op": "transpose", "input": "vae_mid_v_flat", "output": "vae_mid_v_tokens"},
                {"name": "vae_mid_attended", "op": "matmul", "lhs": "vae_mid_probs", "rhs": "vae_mid_v_tokens", "output": "vae_mid_attended"},
                {"name": "vae_mid_attended_c_hw", "op": "transpose", "input": "vae_mid_attended", "output": "vae_mid_attended_c_hw"},
                {
                    "name": "vae_mid_attended_nchw",
                    "op": "reshape",
                    "input": "vae_mid_attended_c_hw",
                    "shape": [batch, channels, h, w],
                    "output": "vae_mid_attended_nchw",
                },
                {
                    "name": "vae_mid_attn_proj_out",
                    "op": "conv2d",
                    "input": "vae_mid_attended_nchw",
                    "weight": f"{manifest_prefix}.proj_out.weight",
                    "bias": f"{manifest_prefix}.proj_out.bias",
                    "padding": [0, 0],
                    "output": "vae_mid_attn_proj",
                },
                {"name": "vae_mid_attn_residual", "op": "add", "lhs": current, "rhs": "vae_mid_attn_proj", "output": "vae_mid_attn_out"},
            ]
        )
        vae_mid_attention_report = {
            **base_report,
            "status": "emitted",
            "reason": "exact dense attention emitted",
        }
        return "vae_mid_attn_out"

    conv_in_weight = vae_tensor(tensors, "decoder.conv_in.weight")
    add_tensor("vae.decoder.conv_in.weight", "decoder.conv_in.weight", "torch_conv2d_weight", conv_in_weight.shape)
    add_tensor("vae.decoder.conv_in.bias", "decoder.conv_in.bias", "identity", vae_tensor(tensors, "decoder.conv_in.bias").shape)
    current = "scaled_latent"
    nodes.append({"name": "vae_latent_scale", "op": "scale", "input": "latent", "factor": 1.0 / 0.13025, "output": current})
    nodes.append(
        {
            "name": "vae_decoder_conv_in",
            "op": "conv2d",
            "input": current,
            "weight": "vae.decoder.conv_in.weight",
            "bias": "vae.decoder.conv_in.bias",
            "padding": [1, 1],
            "output": "vae_hidden",
        }
    )
    current = "vae_hidden"

    current = add_resblock("decoder.mid.block_1", current)
    current = add_mid_attention(current, latent_h, latent_w)
    current = add_resblock("decoder.mid.block_2", current)

    up_levels = sorted(
        {
            int(match.group(1))
            for name in tensors
            if (match := re.match(r"first_stage_model\.decoder\.up\.(\d+)\.block\.\d+\.", name))
        },
        reverse=True,
    )
    current_h = latent_h
    current_w = latent_w
    for level in up_levels:
        block_ids = sorted(
            {
                int(match.group(1))
                for name in tensors
                if (match := re.match(rf"first_stage_model\.decoder\.up\.{level}\.block\.(\d+)\.", name))
            }
        )
        for block_id in block_ids:
            current = add_resblock(f"decoder.up.{level}.block.{block_id}", current)
        upsample_prefix = f"decoder.up.{level}.upsample.conv"
        if f"first_stage_model.{upsample_prefix}.weight" in tensors:
            current_h *= 2
            current_w *= 2
            resize_output = f"vae_up_{level}_resized"
            nodes.append(
                {
                    "name": f"vae_up_{level}_upsample_resize",
                    "op": "upsample",
                    "input": current,
                    "mode": "nearest",
                    "output_spatial_shape": [current_h, current_w],
                    "output": resize_output,
                }
            )
            add_tensor(f"vae.{upsample_prefix}.weight", f"{upsample_prefix}.weight", "torch_conv2d_weight",
                       vae_tensor(tensors, f"{upsample_prefix}.weight").shape)
            add_tensor(f"vae.{upsample_prefix}.bias", f"{upsample_prefix}.bias", "identity",
                       vae_tensor(tensors, f"{upsample_prefix}.bias").shape)
            nodes.append(
                {
                    "name": f"vae_up_{level}_upsample_conv",
                    "op": "conv2d",
                    "input": resize_output,
                    "weight": f"vae.{upsample_prefix}.weight",
                    "bias": f"vae.{upsample_prefix}.bias",
                    "padding": [1, 1],
                    "output": f"vae_up_{level}_upsampled",
                }
            )
            current = f"vae_up_{level}_upsampled"

    add_groupnorm_tensors("decoder", "vae.decoder", "norm_out")
    add_tensor("vae.decoder.conv_out.weight", "decoder.conv_out.weight", "torch_conv2d_weight",
               vae_tensor(tensors, "decoder.conv_out.weight").shape)
    add_tensor("vae.decoder.conv_out.bias", "decoder.conv_out.bias", "identity",
               vae_tensor(tensors, "decoder.conv_out.bias").shape)
    nodes.extend(
        [
            {
                "name": "vae_decoder_norm_out",
                "op": "group_norm",
                "input": current,
                "weight": "vae.decoder.norm_out.weight",
                "bias": "vae.decoder.norm_out.bias",
                "num_groups": 32,
                "eps": 1e-6,
                "layout": "pytorch",
                "output": "vae_norm_out",
            },
            {"name": "vae_decoder_silu", "op": "silu", "input": "vae_norm_out", "output": "vae_out_act"},
            {
                "name": "vae_decoder_conv_out",
                "op": "conv2d",
                "input": "vae_out_act",
                "weight": "vae.decoder.conv_out.weight",
                "bias": "vae.decoder.conv_out.bias",
                "padding": [1, 1],
                "output": "decoded",
            },
            {"name": "vae_decoder_to_image", "op": "scale", "input": "decoded", "factor": 0.5, "bias": 0.5, "output": "image_unclamped"},
            {"name": "vae_decoder_clamp", "op": "clamp", "input": "image_unclamped", "min": 0.0, "max": 1.0, "output": "image"},
        ]
    )

    return {
        "format": "litenn.torch_manifest.v1",
        "metadata": {
            "probe": "vae-decode-full",
            "description": "fixed-shape SDXL VAE decoder traversal with memory-aware mid attention policy",
            "latent_height": latent_h,
            "latent_width": latent_w,
            "vae_mid_attention": vae_mid_attention_report,
        },
        "inputs": [{"name": "latent", "dtype": "torch.float32", "shape": [batch, conv_in_weight.shape[1], latent_h, latent_w]}],
        "tensors": tensor_entries,
        "nodes": nodes,
        "outputs": [{"name": "image", "source": "image"}],
    }


def emit_manifest(args: argparse.Namespace, tensors: dict[str, TensorInfo]) -> dict[str, Any]:
    if args.probe == "unet-stem":
        return emit_unet_stem_manifest(tensors, batch=args.batch, height=args.height, width=args.width)
    if args.probe == "unet-resblock":
        return emit_unet_resblock_manifest(tensors, batch=args.batch, height=args.height, width=args.width)
    if args.probe == "unet-euler-smoke":
        return emit_unet_euler_smoke_manifest(tensors, batch=args.batch, height=args.height, width=args.width)
    if args.probe == "unet-conditioning-smoke":
        return emit_unet_conditioning_smoke_manifest(tensors, batch=args.batch, height=args.height, width=args.width)
    if args.probe == "unet-full-fixed":
        return emit_unet_full_fixed_manifest(
            tensors,
            batch=args.batch,
            height=args.height,
            width=args.width,
            context_tokens=args.context_tokens,
        )
    if args.probe == "spatial-transformer-smoke":
        return emit_spatial_transformer_smoke_manifest(
            tensors, tokens=args.tokens, context_tokens=args.context_tokens
        )
    if args.probe == "spatial-transformer-2d-smoke":
        return emit_spatial_transformer_2d_smoke_manifest(
            tensors, batch=args.batch, height=args.height, width=args.width, context_tokens=args.context_tokens
        )
    if args.probe == "vae-decode-stem":
        return emit_vae_decode_stem_manifest(tensors, batch=args.batch, height=args.height, width=args.width)
    if args.probe == "vae-decode-full":
        return emit_vae_decode_full_manifest(
            tensors,
            batch=args.batch,
            height=args.height,
            width=args.width,
            vae_mid_attention_policy=args.vae_mid_attention_policy,
            vae_attention_max_mib=args.vae_attention_max_mib,
        )
    raise ValueError(f"unsupported probe manifest kind {args.probe!r}")


def build_skeleton_plan(config: dict[str, Any], tensors: dict[str, TensorInfo]) -> dict[str, Any]:
    net = network_params(config)
    input_blocks: dict[int, set[str]] = {}
    output_blocks: dict[int, set[str]] = {}
    middle_parts: set[str] = set()
    transformer_blocks: dict[str, int] = {}
    vae_up_blocks: dict[int, set[int]] = {}

    for name in tensors:
        if match := re.match(r"model\.diffusion_model\.input_blocks\.(\d+)\.(\d+)\.", name):
            input_blocks.setdefault(int(match.group(1)), set()).add(match.group(2))
        if match := re.match(r"model\.diffusion_model\.output_blocks\.(\d+)\.(\d+)\.", name):
            output_blocks.setdefault(int(match.group(1)), set()).add(match.group(2))
        if match := re.match(r"model\.diffusion_model\.middle_block\.(\d+)\.", name):
            middle_parts.add(match.group(1))
        if match := re.match(
            r"model\.diffusion_model\.(input_blocks\.\d+\.\d+|middle_block\.\d+|output_blocks\.\d+\.\d+)"
            r"\.transformer_blocks\.(\d+)\.",
            name,
        ):
            scope = match.group(1)
            transformer_blocks[scope] = max(transformer_blocks.get(scope, 0), int(match.group(2)) + 1)
        if match := re.match(r"first_stage_model\.decoder\.up\.(\d+)\.block\.(\d+)\.", name):
            vae_up_blocks.setdefault(int(match.group(1)), set()).add(int(match.group(2)))

    return {
        "unet": {
            "model_channels": net.get("model_channels"),
            "channel_mult": net.get("channel_mult"),
            "input_blocks": [
                {"index": index, "submodules": sorted(parts)} for index, parts in sorted(input_blocks.items())
            ],
            "middle_parts": sorted(middle_parts),
            "output_blocks": [
                {"index": index, "submodules": sorted(parts)} for index, parts in sorted(output_blocks.items())
            ],
            "transformer_block_counts": dict(sorted(transformer_blocks.items())),
            "skip_join_requirement": "concat along channel axis; manifest concat now lowers to ConcatNode",
        },
        "vae": {
            "decoder_up_blocks": [
                {"level": level, "blocks": sorted(blocks)} for level, blocks in sorted(vae_up_blocks.items(), reverse=True)
            ],
            "mid_attention": (
                "vae-decode-full emits exact dense attention for batch=1 when the configured memory policy allows it; "
                "large 1024x1024 decodes default to a recorded skip fallback unless --vae-mid-attention-policy force is used"
            ),
        },
        "runtime": {
            "required_bindings": [
                "latent",
                "timestep",
                "text/context embeddings",
                "SDXL vector conditioning",
                "scheduler sigmas",
                "classifier-free-guidance batching or two-pass combine",
            ],
            "deferred_production_items": [
                "tokenizer/text encoder execution inside LiteNN",
                "exact tiled/chunked VAE attention if quality parity requires it beyond the current memory-policy fallback",
                "image encoder/refiner variants",
            ],
            "full_unet_manifest": "unet-full-fixed emits ResBlock/SpatialTransformer/downsample/upsample/skip traversal for fixed batch=1 shapes",
        },
    }


def print_summary(report: dict[str, Any]) -> None:
    print("SDXL config/checkpoint probe")
    print(f"  compatible: {report['compatible']}")
    print("  components:")
    for component, info in report["components"].items():
        print(f"    {component}: {info['count']} tensor(s), {info['bytes']} byte(s)")
    print("  network:")
    for key, value in report["network"].items():
        print(f"    {key}: {value}")
    print("  VAE:")
    for key, value in report["vae"].items():
        print(f"    {key}: {value}")
    print("  conditioner:")
    for item in report["conditioner"]:
        print(f"    {item['input_key']}: {item['target']}")
    if report["missing_required_keys"]:
        print("  missing required keys:")
        for key in report["missing_required_keys"]:
            print(f"    {key}")
    print("  shape checks:")
    for check in report["shape_checks"]:
        state = "ok" if check["ok"] else "mismatch"
        print(f"    {state}: {check['name']} expected={check['expected_shape']} actual={check['actual_shape']}")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True, type=Path, help="Path to generative-models SDXL YAML config")
    parser.add_argument("--safetensors", required=True, type=Path, help="Path to SDXL safetensors checkpoint")
    parser.add_argument("--json", action="store_true", help="Print the probe report as JSON")
    parser.add_argument("--emit-probe-manifest", type=Path, help="Write a small LiteNN manifest for one probe subgraph")
    parser.add_argument("--emit-skeleton-plan", type=Path, help="Write the discovered SDXL block traversal plan as JSON")
    parser.add_argument(
        "--probe",
        choices=[
            "unet-stem",
            "unet-resblock",
            "unet-euler-smoke",
            "unet-conditioning-smoke",
            "unet-full-fixed",
            "spatial-transformer-smoke",
            "spatial-transformer-2d-smoke",
            "vae-decode-stem",
            "vae-decode-full",
        ],
        default="unet-stem",
    )
    parser.add_argument("--batch", type=int, default=1)
    parser.add_argument("--height", type=int, default=1024)
    parser.add_argument("--width", type=int, default=1024)
    parser.add_argument("--tokens", type=int, default=64, help="Token count for spatial-transformer-smoke")
    parser.add_argument("--context-tokens", type=int, default=77, help="Context token count for spatial-transformer-smoke")
    parser.add_argument(
        "--vae-mid-attention-policy",
        choices=["auto", "force", "skip"],
        default="auto",
        help="Policy for dense VAE mid-attention in vae-decode-full",
    )
    parser.add_argument(
        "--vae-attention-max-mib",
        type=int,
        default=DEFAULT_VAE_ATTENTION_MAX_MIB,
        help="Auto-skip dense VAE mid-attention when estimated workspace exceeds this MiB limit",
    )
    return parser


def main() -> int:
    parser = build_parser()
    args = parser.parse_args()
    if args.height % 8 != 0 or args.width % 8 != 0:
        parser.error("--height and --width must be divisible by 8 for SDXL latent shapes")
    if args.batch <= 0:
        parser.error("--batch must be positive")
    if args.height <= 0 or args.width <= 0:
        parser.error("--height and --width must be positive")
    if args.tokens <= 0 or args.context_tokens <= 0:
        parser.error("--tokens and --context-tokens must be positive")
    if args.vae_attention_max_mib <= 0:
        parser.error("--vae-attention-max-mib must be positive")

    config = load_yaml(args.config)
    tensors = parse_tensors(load_safetensors_header(args.safetensors))
    report = compatibility_report(config, tensors)
    report["execution_skeleton"] = build_skeleton_plan(config, tensors)

    if args.emit_probe_manifest is not None:
        manifest = emit_manifest(args, tensors)
        args.emit_probe_manifest.parent.mkdir(parents=True, exist_ok=True)
        args.emit_probe_manifest.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    if args.emit_skeleton_plan is not None:
        args.emit_skeleton_plan.parent.mkdir(parents=True, exist_ok=True)
        args.emit_skeleton_plan.write_text(
            json.dumps(report["execution_skeleton"], indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )

    if args.json:
        print(json.dumps(report, indent=2, sort_keys=True))
    else:
        print_summary(report)
        if args.emit_probe_manifest is not None:
            print(f"  wrote probe manifest: {args.emit_probe_manifest}")
        if args.emit_skeleton_plan is not None:
            print(f"  wrote skeleton plan: {args.emit_skeleton_plan}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
