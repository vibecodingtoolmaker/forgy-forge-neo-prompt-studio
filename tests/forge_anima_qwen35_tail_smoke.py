"""Bounded diagnostic for speech through the cached Qwen3.5 Anima conversion."""

# Copyright (C) 2026 vibecodingtoolmaker
# SPDX-License-Identifier: AGPL-3.0-only

from __future__ import annotations

import argparse
import os
from pathlib import Path
import sys


EXTENSION_ROOT = Path(__file__).resolve().parents[1]
FORGE_ROOT = EXTENSION_ROOT.parents[1]
QWEN35_EXTENSION_ROOT = FORGE_ROOT / "extensions" / "sd-forge-qwen-35-encoder"
LM_NORM_PATH = FORGE_ROOT / "models" / "text_encoder" / "qwen35_4b_lm_norm.safetensors"
CAUSAL_TAIL_PATH = (
    FORGE_ROOT / "models" / "text_encoder" / "qwen35_4b_causal_tail.safetensors"
)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="qwen35_4b.safetensors")
    parser.add_argument("--max-tokens", type=int, default=4)
    return parser.parse_args()


def main() -> int:
    args = _parse_args()
    os.environ["IGNORE_CMD_ARGS_ERRORS"] = "1"
    for path in (
        EXTENSION_ROOT,
        FORGE_ROOT,
        FORGE_ROOT / "modules_forge" / "packages",
        QWEN35_EXTENSION_ROOT,
        QWEN35_EXTENSION_ROOT / "scripts",
    ):
        if str(path) not in sys.path:
            sys.path.insert(0, str(path))

    from modules_forge.initialization import initialize_forge

    initialize_forge()

    from modules import initialize

    initialize.imports()

    import torch
    from backend import memory_management
    from backend.patcher.clip import CLIP
    from forgy.adapters.qwen35_runtime import (
        Qwen35CausalLMRuntime,
        load_qwen35_tail_weights,
    )
    from forge_qwen35_encoder import Qwen35EncoderForForge
    from modules_forge import main_entry

    main_entry.refresh_models()
    provider = Qwen35EncoderForForge()
    model, tokenizer = provider._load_model(
        args.model,
        use_calibration=False,
        use_alignment=False,
        alignment_strength=1.0,
        output_scale=1.0,
    )
    provider._qwen35_clip = CLIP(
        model_dict={"qwen35_4b": model},
        tokenizer_dict={},
    )
    try:
        memory_management.load_model_gpu(provider._qwen35_clip.patcher)
        device = torch.device(provider._qwen35_clip.patcher.load_device)
        embedding = model.get_input_embeddings()
        weight = embedding.weight.data
        tail = load_qwen35_tail_weights(
            LM_NORM_PATH,
            CAUSAL_TAIL_PATH,
            device=device,
            dtype=weight.dtype,
        )
        runtime = Qwen35CausalLMRuntime(model, tail)

        rendered = (
            "<|im_start|>system\nReturn only one concise image prompt."
            "<|im_end|>\n<|im_start|>user\nA red fox beside an alpine lake."
            "<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n"
        )
        token_ids = tokenizer.encode(rendered, add_special_tokens=False)
        input_count = len(token_ids)
        generated_ids: list[int] = []
        with torch.no_grad():
            input_ids = torch.tensor([token_ids], device=device, dtype=torch.long)
            hidden_states = runtime.prefill(input_ids)
            for _ in range(max(1, min(args.max_tokens, 8))):
                logits = runtime.logits(hidden_states)
                next_token = int(torch.argmax(logits, dim=-1).item())
                generated_ids.append(next_token)
                if next_token == tokenizer.eos_token_id:
                    break
                hidden_states = runtime.decode(next_token)

        decoded = tokenizer.decode(generated_ids, skip_special_tokens=False)
        escaped = decoded.encode("unicode_escape").decode("ascii")
        print("QWEN35_TAIL_DIAGNOSTIC_OK")  # noqa: T201
        print("TOKENS", input_count, len(generated_ids))  # noqa: T201
        print("TOKEN_IDS", generated_ids)  # noqa: T201
        print("DECODED_ESCAPED", escaped)  # noqa: T201
        return 0
    finally:
        memory_management.unload_all_models()


if __name__ == "__main__":
    raise SystemExit(main())
