"""Fetch only the pinned official Qwen3.5-4B layer-31 causal tail."""

# Copyright (C) 2026 vibecodingtoolmaker
# SPDX-License-Identifier: AGPL-3.0-only

from __future__ import annotations

import argparse
import hashlib
import json
import os
import struct
from pathlib import Path
import sys


SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from prepare_qwen35_lm_norm import (
    EXPECTED_DTYPE,
    SOURCE_REPOSITORY,
    SOURCE_REVISION,
    _read_exact_range,
)


TAIL_TENSORS = {
    "model.language_model.layers.31.post_attention_layernorm.weight": {
        "shard": "model.safetensors-00002-of-00002.safetensors",
        "shape": [2560],
        "sha256": "7202d29a8a2fc7f7bc8da8fe838339ef241d9009140940eaf8d4ca4cf94d9c8a",
    },
    "model.language_model.layers.31.mlp.gate_proj.weight": {
        "shard": "model.safetensors-00001-of-00002.safetensors",
        "shape": [9216, 2560],
        "sha256": "0780377faf6e4e3aa63292db8d6df1c57372758d14d93a6b59943e6f49a6a7fd",
    },
    "model.language_model.layers.31.mlp.up_proj.weight": {
        "shard": "model.safetensors-00001-of-00002.safetensors",
        "shape": [9216, 2560],
        "sha256": "342d44e2e8808432f87098f65be1d2933d8531ab5e0454e49aac2e4f0af59792",
    },
    "model.language_model.layers.31.mlp.down_proj.weight": {
        "shard": "model.safetensors-00001-of-00002.safetensors",
        "shape": [2560, 9216],
        "sha256": "4ced0b87e3d533c8554006fd8e1d7431e08b0b3d94c2379dd3584f3d8553aef9",
    },
}
EXPECTED_TOTAL_TENSOR_BYTES = 141_562_880


def _default_destination() -> Path:
    forge_root = Path(__file__).resolve().parents[3]
    return forge_root / "models" / "text_encoder" / "qwen35_4b_causal_tail.safetensors"


def _source_url(shard: str) -> str:
    return (
        f"https://huggingface.co/{SOURCE_REPOSITORY}/resolve/{SOURCE_REVISION}/{shard}"
    )


def _read_header(shard: str) -> tuple[int, dict]:
    url = _source_url(shard)
    header_length = struct.unpack("<Q", _read_exact_range(url, 0, 7))[0]
    if header_length < 2 or header_length > 128 * 1024 * 1024:
        raise RuntimeError(f"Unsafe safetensors header length: {header_length}.")
    header_bytes = _read_exact_range(url, 8, 8 + header_length - 1)
    return header_length, json.loads(header_bytes.rstrip(b" ").decode("utf-8"))


def _tensor_byte_count(shape: list[int]) -> int:
    count = 2
    for dimension in shape:
        count *= dimension
    return count


def _fetch_tail_tensors() -> dict[str, bytes]:
    headers = {
        shard: _read_header(shard)
        for shard in sorted({str(spec["shard"]) for spec in TAIL_TENSORS.values()})
    }
    payloads: dict[str, bytes] = {}
    for name, spec in TAIL_TENSORS.items():
        shard = str(spec["shard"])
        header_length, header = headers[shard]
        tensor = header.get(name)
        if not isinstance(tensor, dict):
            raise RuntimeError(f"Pinned source tensor {name!r} was not found.")
        expected_shape = spec["shape"]
        if (
            tensor.get("dtype") != EXPECTED_DTYPE
            or tensor.get("shape") != expected_shape
        ):
            raise RuntimeError(
                f"Pinned tensor contract changed for {name!r}: "
                f"dtype={tensor.get('dtype')!r}, shape={tensor.get('shape')!r}."
            )
        offsets = tensor.get("data_offsets")
        expected_bytes = _tensor_byte_count(expected_shape)
        if (
            not isinstance(offsets, list)
            or len(offsets) != 2
            or not all(isinstance(value, int) for value in offsets)
            or offsets[1] - offsets[0] != expected_bytes
        ):
            raise RuntimeError(f"Unsafe source offsets for {name!r}: {offsets!r}.")
        data_start = 8 + header_length
        payload = _read_exact_range(
            _source_url(shard),
            data_start + offsets[0],
            data_start + offsets[1] - 1,
        )
        digest = hashlib.sha256(payload).hexdigest()
        expected_digest = str(spec["sha256"])
        if expected_digest and digest != expected_digest:
            raise RuntimeError(
                f"SHA-256 mismatch for {name!r}: expected {expected_digest}, "
                f"received {digest}."
            )
        payloads[name] = payload
        print(f"{name}: {len(payload)} bytes, SHA-256 {digest}")  # noqa: T201
    if sum(map(len, payloads.values())) != EXPECTED_TOTAL_TENSOR_BYTES:
        raise RuntimeError("The extracted causal-tail byte count is incorrect.")
    return payloads


def _build_local_safetensors(payloads: dict[str, bytes]) -> bytes:
    data_offset = 0
    header: dict[str, object] = {
        "__metadata__": {
            "source_repository": SOURCE_REPOSITORY,
            "source_revision": SOURCE_REVISION,
            "license": "Apache-2.0",
            "purpose": "Qwen3.5-4B layer-31 causal tail for explicit Forgy use",
        }
    }
    ordered_names = sorted(payloads)
    for name in ordered_names:
        payload = payloads[name]
        header[name] = {
            "dtype": EXPECTED_DTYPE,
            "shape": TAIL_TENSORS[name]["shape"],
            "data_offsets": [data_offset, data_offset + len(payload)],
        }
        data_offset += len(payload)
    encoded = json.dumps(header, separators=(",", ":"), sort_keys=True).encode("utf-8")
    encoded += b" " * (-len(encoded) % 8)
    return (
        struct.pack("<Q", len(encoded))
        + encoded
        + b"".join(payloads[name] for name in ordered_names)
    )


def main() -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Extract the four official Qwen3.5-4B layer-31 causal-tail tensors "
            "with bounded HTTP ranges; neither complete model shard is downloaded."
        )
    )
    parser.add_argument(
        "--destination",
        type=Path,
        default=_default_destination(),
        help="Visible output path for the supplemental safetensors file.",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Replace an existing destination only after all checks pass.",
    )
    args = parser.parse_args()
    destination = args.destination.expanduser().resolve()
    print(f"Destination: {destination}")  # noqa: T201
    print(f"Pinned source: {SOURCE_REPOSITORY}@{SOURCE_REVISION}")  # noqa: T201
    if destination.exists() and not args.force:
        raise FileExistsError(
            f"Destination already exists: {destination}. Use --force to replace it."
        )

    payloads = _fetch_tail_tensors()
    destination.parent.mkdir(parents=True, exist_ok=True)
    temp_path = destination.with_name(f"{destination.name}.{os.getpid()}.tmp")
    try:
        temp_path.write_bytes(_build_local_safetensors(payloads))
        os.replace(temp_path, destination)
    finally:
        temp_path.unlink(missing_ok=True)
    print(f"Wrote {destination.stat().st_size} bytes.")  # noqa: T201
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
