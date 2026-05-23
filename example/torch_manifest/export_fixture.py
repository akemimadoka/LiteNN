import json
import struct
from pathlib import Path

import torch


def write_safetensors(path: Path, tensors):
    payload = bytearray()
    header = {}
    offset = 0
    for name, tensor in tensors.items():
        tensor = tensor.detach().cpu().contiguous()
        if tensor.dtype != torch.float32:
            raise ValueError(f"{name}: only torch.float32 is supported by this tiny fixture writer")
        values = [float(value) for value in tensor.reshape(-1).tolist()]
        data = struct.pack("<" + "f" * len(values), *values)
        header[name] = {
            "dtype": "F32",
            "shape": list(tensor.shape),
            "data_offsets": [offset, offset + len(data)],
        }
        payload.extend(data)
        offset += len(data)

    header_bytes = json.dumps(header, separators=(",", ":")).encode("utf-8")
    path.write_bytes(struct.pack("<Q", len(header_bytes)) + header_bytes + payload)


def main():
    out_dir = Path(__file__).resolve().parent
    weight = torch.tensor(
        [
            [0.25, 2.0, -0.75],
            [-1.0, 0.5, 1.5],
        ],
        dtype=torch.float32,
    )
    bias = torch.tensor([0.1, -0.2], dtype=torch.float32)
    x = torch.tensor(
        [
            [1.0, -2.0, 0.5],
            [0.0, 3.0, -1.0],
        ],
        dtype=torch.float32,
    )
    golden = torch.relu(torch.nn.functional.linear(x, weight, bias))

    write_safetensors(out_dir / "linear_relu.safetensors", {
        "linear.weight": weight,
        "linear.bias": bias,
    })
    (out_dir / "golden.txt").write_text(str(golden.tolist()) + "\n", encoding="utf-8")
    print(f"Wrote {out_dir / 'linear_relu.safetensors'}")
    print(f"Golden output: {golden.tolist()}")


if __name__ == "__main__":
    main()
