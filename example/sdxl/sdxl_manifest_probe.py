#!/usr/bin/env python3
"""Probe Stability-AI generative-models SDXL configs and emit LiteNN manifests.

This script intentionally avoids importing the generative-models Python package.
It only needs PyYAML and a safetensors file header, so it can run before the
full SDXL inference environment is installed.
"""

from __future__ import annotations

import argparse
import json
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


def emit_manifest(args: argparse.Namespace, tensors: dict[str, TensorInfo]) -> dict[str, Any]:
    if args.probe == "unet-stem":
        return emit_unet_stem_manifest(tensors, batch=args.batch, height=args.height, width=args.width)
    if args.probe == "unet-resblock":
        return emit_unet_resblock_manifest(tensors, batch=args.batch, height=args.height, width=args.width)
    if args.probe == "unet-euler-smoke":
        return emit_unet_euler_smoke_manifest(tensors, batch=args.batch, height=args.height, width=args.width)
    if args.probe == "vae-decode-stem":
        return emit_vae_decode_stem_manifest(tensors, batch=args.batch, height=args.height, width=args.width)
    raise ValueError(f"unsupported probe manifest kind {args.probe!r}")


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
    parser.add_argument(
        "--probe",
        choices=["unet-stem", "unet-resblock", "unet-euler-smoke", "vae-decode-stem"],
        default="unet-stem",
    )
    parser.add_argument("--batch", type=int, default=1)
    parser.add_argument("--height", type=int, default=1024)
    parser.add_argument("--width", type=int, default=1024)
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

    config = load_yaml(args.config)
    tensors = parse_tensors(load_safetensors_header(args.safetensors))
    report = compatibility_report(config, tensors)

    if args.emit_probe_manifest is not None:
        manifest = emit_manifest(args, tensors)
        args.emit_probe_manifest.parent.mkdir(parents=True, exist_ok=True)
        args.emit_probe_manifest.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")

    if args.json:
        print(json.dumps(report, indent=2, sort_keys=True))
    else:
        print_summary(report)
        if args.emit_probe_manifest is not None:
            print(f"  wrote probe manifest: {args.emit_probe_manifest}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
