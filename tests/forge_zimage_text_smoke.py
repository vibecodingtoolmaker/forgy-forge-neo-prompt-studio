"""Run one bounded Forgy text generation with a real Forge-owned Z-Image stack."""

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


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--module", action="append", dest="modules", default=[])
    parser.add_argument("--max-tokens", type=int, default=64)
    return parser.parse_args()


def main() -> int:
    args = _parse_args()
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
    try:
        sd_model, _ = sd_models.forge_model_reload()
        namespace = runpy.run_path(
            str(EXTENSION_ROOT / "scripts" / "forge_krea_prompt_assistant.py"),
            run_name="forgy_zimage_text_smoke",
        )
        context = namespace["MODEL_MANAGER"].resolve(sd_model)
        assert context.identity.adapter_id == "z-image-qwen3-4b"
        assert context.capabilities.text_generation is True
        assert context.capabilities.vision_input is False

        generation_id = namespace["_begin_generation_request"]()
        result = context.adapter.generate_text(
            model_context=context,
            idea="A red fox resting beside a quiet alpine lake at sunrise.",
            instruction="Use a wide cinematic composition and natural detail.",
            system_prompt=namespace["DEFAULT_PERSONA_PROMPT"],
            vram_profile="8 GB",
            max_tokens=max(1, min(args.max_tokens, 256)),
            do_sample=False,
            temperature=0.7,
            top_k=64,
            top_p=0.95,
            min_p=0.05,
            repetition_penalty=1.05,
            seed=0,
            generation_id=generation_id,
            workflow=namespace["Workflow"].IDEA_TO_PROMPT,
        )
        text, input_count, output_count, finish_reason, *_rest = result
        visible, _thinking_hidden = context.adapter.strip_model_thinking(text)
        if not visible:
            raise RuntimeError("Z-Image Qwen3 produced no visible text")
        print("ZIMAGE_TEXT_GENERATION_OK")  # noqa: T201
        print("ADAPTER", context.identity.display_name)  # noqa: T201
        print("TOKENS", input_count, output_count, finish_reason)  # noqa: T201
        print("VISIBLE_CHARS", len(visible))  # noqa: T201
        return 0
    finally:
        if generation_id:
            namespace["_clear_generation_request"](generation_id)
        sd_models.unload_model_weights()


if __name__ == "__main__":
    raise SystemExit(main())
