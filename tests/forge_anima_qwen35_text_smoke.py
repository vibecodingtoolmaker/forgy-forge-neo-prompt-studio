"""Run one bounded Forgy Anima generation through a cached Qwen3.5 provider."""

# Copyright (C) 2026 vibecodingtoolmaker
# SPDX-License-Identifier: AGPL-3.0-only

from __future__ import annotations

import argparse
import os
from pathlib import Path
import runpy
import sys
from types import SimpleNamespace


EXTENSION_ROOT = Path(__file__).resolve().parents[1]
FORGE_ROOT = EXTENSION_ROOT.parents[1]
QWEN35_EXTENSION_ROOT = FORGE_ROOT / "extensions" / "sd-forge-qwen-35-encoder"


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--module", action="append", dest="modules", default=[])
    parser.add_argument("--qwen35-model", default="qwen35_4b.safetensors")
    parser.add_argument("--max-tokens", type=int, default=8)
    return parser.parse_args()


def main() -> int:
    args = _parse_args()
    os.environ["IGNORE_CMD_ARGS_ERRORS"] = "1"
    for path in (
        FORGE_ROOT,
        EXTENSION_ROOT,
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

    from backend import memory_management
    from backend.patcher.clip import CLIP
    from forge_qwen35_encoder import Qwen35EncoderForForge
    from modules import scripts as forge_scripts
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

    generation_id = None
    namespace = None
    original_txt2img = getattr(forge_scripts, "scripts_txt2img", None)
    original_img2img = getattr(forge_scripts, "scripts_img2img", None)
    try:
        sd_model, _ = sd_models.forge_model_reload()
        provider = Qwen35EncoderForForge()
        model, _tokenizer = provider._load_model(
            args.qwen35_model,
            use_calibration=False,
            use_alignment=False,
            alignment_strength=1.0,
            output_scale=1.0,
        )
        provider._qwen35_clip = CLIP(
            model_dict={"qwen35_4b": model},
            tokenizer_dict={},
        )
        forge_scripts.scripts_txt2img = SimpleNamespace(alwayson_scripts=[provider])
        forge_scripts.scripts_img2img = SimpleNamespace(alwayson_scripts=[])

        namespace = runpy.run_path(
            str(EXTENSION_ROOT / "scripts" / "forge_krea_prompt_assistant.py"),
            run_name="forgy_anima_qwen35_text_smoke",
        )
        context = namespace["MODEL_MANAGER"].resolve(sd_model)
        assert context.identity.adapter_id == "anima-qwen3-0.6b"

        generation_id = namespace["_begin_generation_request"]()
        result = context.adapter.generate_text(
            model_context=context,
            idea="A red fox resting beside a quiet alpine lake at sunrise.",
            instruction="Use a wide cinematic composition and natural detail.",
            system_prompt=namespace["DEFAULT_PERSONA_PROMPT"],
            vram_profile="16 GB",
            max_tokens=max(1, min(args.max_tokens, 16)),
            do_sample=False,
            temperature=0.7,
            top_k=64,
            top_p=0.95,
            min_p=0.05,
            repetition_penalty=1.05,
            seed=0,
            generation_id=generation_id,
            anima_text_provider=namespace["ANIMA_TEXT_PROVIDER_QWEN35"],
            workflow=namespace["Workflow"].IDEA_TO_PROMPT,
        )
        text, input_count, output_count, finish_reason, *_rest = result
        visible, _thinking_hidden = context.adapter.strip_model_thinking(text)
        if not visible:
            raise RuntimeError("Anima Qwen3.5 provider produced no visible text")
        print("ANIMA_QWEN35_TEXT_GENERATION_OK")  # noqa: T201
        print("ADAPTER", context.identity.display_name)  # noqa: T201
        print("PROVIDER", namespace["ANIMA_TEXT_PROVIDER_QWEN35"])  # noqa: T201
        print("TOKENS", input_count, output_count, finish_reason)  # noqa: T201
        print("VISIBLE_CHARS", len(visible))  # noqa: T201
        print("VISIBLE", visible)  # noqa: T201
        return 0
    finally:
        if generation_id and namespace is not None:
            namespace["_clear_generation_request"](generation_id)
        forge_scripts.scripts_txt2img = original_txt2img
        forge_scripts.scripts_img2img = original_img2img
        memory_management.unload_all_models()
        sd_models.unload_model_weights()


if __name__ == "__main__":
    raise SystemExit(main())
