"""Manually probe Forge-owned Anima Qwen3-0.6B full-context decoding.

This is an isolated developer diagnostic, not an automated test or a supported
Forgy adapter. It deliberately reuses only the active Forge model, tokenizer,
encoder, and ModelPatcher, generates at most 32 tokens, then unloads the model.
"""

# Copyright (C) 2026 vibecodingtoolmaker
# SPDX-License-Identifier: AGPL-3.0-only

from __future__ import annotations

import argparse
import os
from pathlib import Path
import runpy
import sys


EXTENSION_ROOT = Path(__file__).resolve().parents[1]
FORGE_ROOT = EXTENSION_ROOT.parents[1]
ANIMA_MODEL_CLASS = "backend.diffusion_engine.anima.Anima"
ANIMA_ENGINE_CLASS = "backend.text_processing.anima_engine.AnimaTextProcessingEngine"
ANIMA_ENCODER_CLASS = "backend.nn.llm.llama.Qwen3_06B"
ANIMA_HIDDEN_SIZE = 1024
ANIMA_VOCAB_SIZE = 151936
DEFAULT_PROMPT = (
    "A polished natural-language image prompt: A red fox resting beside a "
    "quiet alpine lake at sunrise,"
)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Manually validate tied-embedding text generation with Forge's "
            "active Anima Qwen3-0.6B stack. This does not enable an Anima "
            "Forgy adapter."
        )
    )
    parser.add_argument(
        "--checkpoint", required=True, help="Forge-visible Anima checkpoint name"
    )
    parser.add_argument(
        "--module",
        action="append",
        dest="modules",
        default=[],
        help="Forge-visible additional-module name; repeat for encoder and VAE",
    )
    parser.add_argument(
        "--max-tokens",
        type=int,
        default=16,
        help="Number of new tokens; clamped to the diagnostic ceiling of 32",
    )
    parser.add_argument("--prompt", default=DEFAULT_PROMPT)
    return parser.parse_args()


def _class_path(value) -> str:
    cls = type(value)
    return f"{cls.__module__}.{cls.__qualname__}"


def main() -> int:
    args = _parse_args()
    max_tokens = max(1, min(int(args.max_tokens), 32))
    os.environ["IGNORE_CMD_ARGS_ERRORS"] = "1"
    for path in (
        FORGE_ROOT,
        EXTENSION_ROOT,
        FORGE_ROOT / "modules_forge" / "packages",
    ):
        if str(path) not in sys.path:
            sys.path.insert(0, str(path))

    from modules_forge.initialization import initialize_forge

    initialize_forge()

    from modules import initialize

    initialize.imports()

    import torch

    from backend import memory_management
    from modules import sd_models, shared
    from modules_forge import main_entry

    sd_models.setup_model()
    main_entry.refresh_models()
    checkpoint_info = sd_models.get_closet_checkpoint_match(args.checkpoint)
    if checkpoint_info is None:
        raise FileNotFoundError(f"Forge did not resolve {args.checkpoint!r}")
    missing_modules = [
        name for name in args.modules if name not in main_entry.module_list
    ]
    if missing_modules:
        raise FileNotFoundError(
            "Forge did not resolve additional modules: " + ", ".join(missing_modules)
        )
    module_paths = sorted(main_entry.module_list[name] for name in args.modules)

    shared.opts.set("sd_model_checkpoint", checkpoint_info.title, run_callbacks=False)
    shared.opts.set("forge_additional_modules", module_paths, run_callbacks=False)
    main_entry.refresh_model_loading_parameters(refresh=True)

    namespace = None
    generated_ids: list[int] = []
    try:
        sd_model, _ = sd_models.forge_model_reload()
        if _class_path(sd_model) != ANIMA_MODEL_CLASS:
            raise RuntimeError(
                f"Expected {ANIMA_MODEL_CLASS}, received {_class_path(sd_model)}"
            )

        namespace = runpy.run_path(
            str(EXTENSION_ROOT / "scripts" / "forge_krea_prompt_assistant.py"),
            run_name="forgy_anima_qwen06_full_context_smoke",
        )

        forge_objects = getattr(sd_model, "forge_objects", None)
        clip = getattr(forge_objects, "clip", None)
        cond_stage_model = getattr(clip, "cond_stage_model", None)
        tokenizer_container = getattr(clip, "tokenizer", None)
        encoder = getattr(cond_stage_model, "qwen3_06b", None)
        tokenizer = getattr(tokenizer_container, "qwen3_06b", None)
        engine = getattr(sd_model, "text_processing_engine_anima", None)
        if clip is None or encoder is None or tokenizer is None or engine is None:
            raise RuntimeError("Forge did not expose the complete Anima text stack")
        if _class_path(engine) != ANIMA_ENGINE_CLASS:
            raise RuntimeError(f"Unexpected Anima engine: {_class_path(engine)}")
        if _class_path(encoder) != ANIMA_ENCODER_CLASS:
            raise RuntimeError(f"Unexpected Anima encoder: {_class_path(encoder)}")
        if getattr(engine, "text_encoder", None) is not encoder:
            raise RuntimeError("Anima engine and CLIP expose different encoders")
        if getattr(engine, "qwen_tokenizer", None) is not tokenizer:
            raise RuntimeError("Anima engine and CLIP expose different tokenizers")

        core_model = getattr(encoder, "model", None)
        config = getattr(core_model, "config", None)
        embedding = getattr(core_model, "embed_tokens", None)
        final_norm = getattr(core_model, "norm", None)
        if (
            getattr(config, "hidden_size", None) != ANIMA_HIDDEN_SIZE
            or getattr(config, "vocab_size", None) != ANIMA_VOCAB_SIZE
        ):
            raise RuntimeError("The active Anima encoder contract is incompatible")
        if embedding is None or final_norm is None:
            raise RuntimeError(
                "The active Anima encoder lacks tied embeddings or final LM norm"
            )

        prompt_ids = namespace["_tokenize"](tokenizer, str(args.prompt))
        device = torch.device(clip.patcher.load_device)
        memory_management.load_model_gpu(clip.patcher)

        weight = getattr(embedding, "weight", None)
        expected_shape = (ANIMA_VOCAB_SIZE, ANIMA_HIDDEN_SIZE)
        if (
            not isinstance(weight, torch.Tensor)
            or tuple(weight.shape) != expected_shape
        ):
            raise RuntimeError("The active Anima embedding matrix is incompatible")
        if weight.device != device:
            raise RuntimeError("Forge did not place the Anima embedding on its device")
        # Forge loaded this Parameter under inference_mode. ``.data`` creates a
        # normal TensorImpl that shares the same storage but owns a version
        # counter, which PyTorch 2.11's output projection requires. The alias is
        # read-only and avoids copying roughly 311 MB of embedding weights.
        projection_weight = weight.data
        if projection_weight.data_ptr() != weight.data_ptr() or torch.is_inference(
            projection_weight
        ):
            raise RuntimeError("Could not create a zero-copy projection alias")

        token_ids = list(prompt_ids)
        stop_tokens = {
            int(token_id)
            for token_id in (
                getattr(tokenizer, "eos_token_id", None),
                151645,
            )
            if token_id is not None
        }
        finish_reason = "max_tokens"

        # The Qwen3-0.6B path uses Forge's manual-cast Linear layers. With the
        # current PyTorch 2.11 runtime those layers reject inference tensors,
        # while no_grad still provides the required graph-free execution.
        with torch.no_grad():
            for _ in range(max_tokens):
                input_ids = torch.tensor([token_ids], device=device, dtype=torch.long)
                embeds = embedding(input_ids)
                output = core_model(None, embeds=embeds, attention_mask=None)
                if not isinstance(output, tuple) or len(output) != 2:
                    raise RuntimeError(
                        "Anima Qwen3 did not return full-context hidden states"
                    )
                hidden_states, _intermediate = output
                last_hidden = hidden_states[:, -1:, :].data.to(
                    dtype=projection_weight.dtype
                )
                logits = torch.nn.functional.linear(
                    last_hidden, projection_weight, bias=None
                )[:, -1, :]
                next_token = int(torch.argmax(logits, dim=-1).item())
                generated_ids.append(next_token)
                if next_token in stop_tokens:
                    finish_reason = "eos"
                    break
                token_ids.append(next_token)

        visible = namespace["_decode_generated_text"](tokenizer, generated_ids)
        if not visible:
            raise RuntimeError("Anima Qwen3 produced no visible text")

        print("ANIMA_QWEN06_FULL_CONTEXT_OK")  # noqa: T201
        print("ENCODER", _class_path(encoder))  # noqa: T201
        print("TOKENS", len(prompt_ids), len(generated_ids), finish_reason)  # noqa: T201
        print("VISIBLE_CHARS", len(visible))  # noqa: T201
        print("VISIBLE", visible)  # noqa: T201
        return 0
    finally:
        generated_ids.clear()
        sd_models.unload_model_weights()


if __name__ == "__main__":
    raise SystemExit(main())
