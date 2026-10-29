"""Fetch only the pinned official Qwen3.5-4B final language-model RMSNorm."""

# Copyright (C) 2026 vibecodingtoolmaker
# SPDX-License-Identifier: AGPL-3.0-only

from __future__ import annotations

import argparse
import hashlib
import json
import os
import struct
import urllib.request
from pathlib import Path


SOURCE_REPOSITORY = "Qwen/Qwen3.5-4B"
SOURCE_REVISION = "851bf6e806efd8d0a36b00ddf55e13ccb7b8cd0a"
SOURCE_SHARD = "model.safetensors-00002-of-00002.safetensors"
SOURCE_TENSOR = "model.language_model.norm.weight"
LOCAL_TENSOR = "qwen35_4b_lm_norm.weight"
EXPECTED_DTYPE = "BF16"
EXPECTED_SHAPE = [2560]
EXPECTED_BYTE_COUNT = 5120
EXPECTED_TENSOR_SHA256 = (
    "96294c653d4e8ab158cdae0b7e794559654ffdc62db880725ed714d47f708621"
)


def _default_destination() -> Path:
    forge_root = Path(__file__).resolve().parents[3]
    return forge_root / "models" / "text_encoder" / "qwen35_4b_lm_norm.safetensors"


def _source_url() -> str:
    return (
        f"https://huggingface.co/{SOURCE_REPOSITORY}/resolve/"
        f"{SOURCE_REVISION}/{SOURCE_SHARD}"
    )


def _read_exact_range(url: str, start: int, end: int) -> bytes:
    request = urllib.request.Request(
        url,
        headers={
            "Range": f"bytes={start}-{end}",
            "User-Agent": "Forgy-Qwen35-Norm-Extractor/1.0",
        },
    )
    with urllib.request.urlopen(request, timeout=60) as response:
        if response.status != 206:
            raise RuntimeError(
                "The source server did not honor the bounded byte-range request; "
                "refusing to download the complete model shard."
            )
        expected_content_range = f"bytes {start}-{end}/"
        content_range = str(response.headers.get("Content-Range", ""))
        if not content_range.startswith(expected_content_range):
            raise RuntimeError(
                f"Unexpected Content-Range {content_range!r}; expected a range "
                f"starting with {expected_content_range!r}."
            )
        payload = response.read(end - start + 1)
    if len(payload) != end - start + 1:
        raise RuntimeError(
            f"Range response contained {len(payload)} bytes instead of "
            f"{end - start + 1}."
        )
    return payload


def _fetch_source_tensor() -> bytes:
    url = _source_url()
    header_length = struct.unpack("<Q", _read_exact_range(url, 0, 7))[0]
    if header_length < 2 or header_length > 128 * 1024 * 1024:
        raise RuntimeError(f"Unsafe safetensors header length: {header_length}.")
    header_bytes = _read_exact_range(url, 8, 8 + header_length - 1)
    header = json.loads(header_bytes.rstrip(b" ").decode("utf-8"))
    tensor = header.get(SOURCE_TENSOR)
    if not isinstance(tensor, dict):
        raise RuntimeError(f"Pinned source tensor {SOURCE_TENSOR!r} was not found.")
    if tensor.get("dtype") != EXPECTED_DTYPE or tensor.get("shape") != EXPECTED_SHAPE:
        raise RuntimeError(
            "Pinned source tensor contract changed: "
            f"dtype={tensor.get('dtype')!r}, shape={tensor.get('shape')!r}."
        )
    offsets = tensor.get("data_offsets")
    if (
        not isinstance(offsets, list)
        or len(offsets) != 2
        or not all(isinstance(value, int) for value in offsets)
        or offsets[1] - offsets[0] != EXPECTED_BYTE_COUNT
    ):
        raise RuntimeError(f"Unsafe source tensor offsets: {offsets!r}.")
    data_start = 8 + header_length
    return _read_exact_range(
        url,
        data_start + offsets[0],
        data_start + offsets[1] - 1,
    )


def _build_local_safetensors(payload: bytes, tensor_sha256: str) -> bytes:
    metadata = {
        "source_repository": SOURCE_REPOSITORY,
        "source_revision": SOURCE_REVISION,
        "source_shard": SOURCE_SHARD,
        "source_tensor": SOURCE_TENSOR,
        "tensor_sha256": tensor_sha256,
        "license": "Apache-2.0",
    }
    header = {
        "__metadata__": metadata,
        LOCAL_TENSOR: {
            "dtype": EXPECTED_DTYPE,
            "shape": EXPECTED_SHAPE,
            "data_offsets": [0, len(payload)],
        },
    }
    encoded = json.dumps(header, separators=(",", ":"), sort_keys=True).encode("utf-8")
    encoded += b" " * (-len(encoded) % 8)
    return struct.pack("<Q", len(encoded)) + encoded + payload


def main() -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Extract only the official Qwen3.5-4B final RMSNorm with bounded "
            "HTTP range requests; the multi-gigabyte model shard is never downloaded."
        )
    )
    parser.add_argument(
        "--destination",
        type=Path,
        default=_default_destination(),
        help="Visible output path for the small safetensors file.",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Replace an existing destination only after all checks pass.",
    )
    args = parser.parse_args()
    destination = args.destination.expanduser().resolve()
    print(f"Destination: {destination}")  # noqa: T201
    print(  # noqa: T201
        f"Pinned source: {SOURCE_REPOSITORY}@{SOURCE_REVISION}/{SOURCE_SHARD}"
    )
    print(f"Tensor: {SOURCE_TENSOR}")  # noqa: T201
    if destination.exists() and not args.force:
        raise FileExistsError(
            f"Destination already exists: {destination}. Use --force to replace it."
        )

    payload = _fetch_source_tensor()
    tensor_sha256 = hashlib.sha256(payload).hexdigest()
    if EXPECTED_TENSOR_SHA256 and tensor_sha256 != EXPECTED_TENSOR_SHA256:
        raise RuntimeError(
            f"Tensor SHA-256 mismatch: expected {EXPECTED_TENSOR_SHA256}, "
            f"received {tensor_sha256}."
        )
    destination.parent.mkdir(parents=True, exist_ok=True)
    temp_path = destination.with_name(f"{destination.name}.{os.getpid()}.tmp")
    try:
        temp_path.write_bytes(_build_local_safetensors(payload, tensor_sha256))
        os.replace(temp_path, destination)
    finally:
        temp_path.unlink(missing_ok=True)
    print(f"Tensor SHA-256: {tensor_sha256}")  # noqa: T201
    print(f"Wrote {destination.stat().st_size} bytes.")  # noqa: T201
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
