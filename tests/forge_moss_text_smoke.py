"""Run bounded Forgy text generation with Forge's active MOSS Qwen3.5-9B."""

# Copyright (C) 2026 vibecodingtoolmaker
# SPDX-License-Identifier: AGPL-3.0-only

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import runpy
import statistics
import sys
import time

from PIL import Image, ImageDraw


EXTENSION_ROOT = Path(__file__).resolve().parents[1]
FORGE_ROOT = EXTENSION_ROOT.parents[1]


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--module", action="append", dest="modules", default=[])
    parser.add_argument("--max-tokens", type=int, default=16)
    parser.add_argument(
        "--persona-name",
        help="Use one named local Idea-to-Prompt persona for the text smoke.",
    )
    parser.add_argument(
        "--vision",
        action="store_true",
        help="Exercise MOSS Image-to-Prompt with a synthetic reference image.",
    )
    parser.add_argument(
        "--profile",
        action="store_true",
        help="Print synchronized MOSS generation-stage timings.",
    )
    parser.add_argument(
        "--profile-detail",
        action="store_true",
        help=(
            "Add CUDA-event layer-family totals, per-call latency statistics, "
            "and peak VRAM to the synchronized stage profile."
        ),
    )
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
    namespace = None
    runtime_globals = None
    original_runtime = None
    original_head_loader = None
    original_model_placement = None
    timings: dict[str, float] = {}
    counters: dict[str, int] = {}
    call_timings: dict[str, list[float]] = {}
    cuda_events: dict[str, list[tuple[object, object]]] = {}
    try:
        model_reload_started = time.perf_counter()
        sd_model, _ = sd_models.forge_model_reload()
        timings["model_reload"] = time.perf_counter() - model_reload_started
        counters["model_reload"] = 1
        namespace = runpy.run_path(
            str(EXTENSION_ROOT / "scripts" / "forge_krea_prompt_assistant.py"),
            run_name="forgy_moss_text_smoke",
        )
        if args.profile or args.profile_detail:
            import torch

            generation_function = namespace["_generate_with_active_moss"]
            runtime_globals = getattr(
                generation_function, "__wrapped__", generation_function
            ).__globals__
            original_runtime = runtime_globals["ForgeQwen359BCausalLMRuntime"]
            original_head_loader = runtime_globals["load_moss_lm_head"]
            runtime_memory_management = runtime_globals["memory_management"]
            original_model_placement = runtime_memory_management.load_models_gpu

            def synchronize() -> None:
                if torch.cuda.is_available():
                    torch.cuda.synchronize()

            def timed(name, operation, *operation_args, **operation_kwargs):
                synchronize()
                started = time.perf_counter()
                try:
                    return operation(*operation_args, **operation_kwargs)
                finally:
                    synchronize()
                    elapsed = time.perf_counter() - started
                    timings[name] = timings.get(name, 0.0) + elapsed
                    counters[name] = counters.get(name, 0) + 1
                    call_timings.setdefault(name, []).append(elapsed)

            def cuda_timed(name, operation, *operation_args, **operation_kwargs):
                if not args.profile_detail or not torch.cuda.is_available():
                    return operation(*operation_args, **operation_kwargs)
                start = torch.cuda.Event(enable_timing=True)
                end = torch.cuda.Event(enable_timing=True)
                start.record()
                try:
                    return operation(*operation_args, **operation_kwargs)
                finally:
                    end.record()
                    cuda_events.setdefault(name, []).append((start, end))

            def profiled_head_loader(*loader_args, **loader_kwargs):
                return timed(
                    "head_load", original_head_loader, *loader_args, **loader_kwargs
                )

            def profiled_model_placement(*placement_args, **placement_kwargs):
                return timed(
                    "model_placement",
                    original_model_placement,
                    *placement_args,
                    **placement_kwargs,
                )

            class ProfiledMossRuntime(original_runtime):
                def __init__(self, *runtime_args, **runtime_kwargs):
                    self._profile_phase = "runtime_init"
                    timed(
                        "runtime_init",
                        super().__init__,
                        *runtime_args,
                        **runtime_kwargs,
                    )
                    if args.profile_detail:
                        weighted_modules = [
                            module
                            for module in self.model.modules()
                            if isinstance(getattr(module, "weight", None), torch.Tensor)
                        ]
                        manual_cast_modules = [
                            module
                            for module in weighted_modules
                            if bool(getattr(module, "parameters_manual_cast", False))
                        ]
                        patched_modules = [
                            module
                            for module in weighted_modules
                            if getattr(module, "weight_function", ())
                            or getattr(module, "bias_function", ())
                        ]
                        reference_weight = self.embedding_weight
                        off_device_modules = [
                            module
                            for module in weighted_modules
                            if module.weight.device != reference_weight.device
                        ]
                        mixed_dtype_modules = [
                            module
                            for module in weighted_modules
                            if module.weight.is_floating_point()
                            and module.weight.dtype != reference_weight.dtype
                        ]
                        print(  # noqa: T201
                            "PROFILE_MODULES",
                            f"weighted={len(weighted_modules)}",
                            f"manual_cast={len(manual_cast_modules)}",
                            f"patched={len(patched_modules)}",
                            f"off_device={len(off_device_modules)}",
                            f"mixed_dtype={len(mixed_dtype_modules)}",
                            "manual_classes="
                            + ",".join(
                                sorted(
                                    {
                                        type(module).__name__
                                        for module in manual_cast_modules
                                    }
                                )
                            ),
                        )

                def prefill(self, input_ids):
                    self._profile_phase = "prefill"
                    try:
                        return timed("prefill", super().prefill, input_ids)
                    finally:
                        self._profile_phase = "idle"

                def prefill_embeddings(self, embeddings, position_ids):
                    self._profile_phase = "prefill"
                    try:
                        return timed(
                            "prefill",
                            super().prefill_embeddings,
                            embeddings,
                            position_ids,
                        )
                    finally:
                        self._profile_phase = "idle"

                def logits(self, hidden_states):
                    return timed("logits", super().logits, hidden_states)

                def decode(self, token_id):
                    self._profile_phase = "decode"
                    try:
                        return timed("decode", super().decode, token_id)
                    finally:
                        self._profile_phase = "idle"

                def _linear_attention(self, *runtime_args, **runtime_kwargs):
                    return cuda_timed(
                        f"{self._profile_phase}.linear_attention",
                        super()._linear_attention,
                        *runtime_args,
                        **runtime_kwargs,
                    )

                def _full_attention(self, *runtime_args, **runtime_kwargs):
                    return cuda_timed(
                        f"{self._profile_phase}.full_attention",
                        super()._full_attention,
                        *runtime_args,
                        **runtime_kwargs,
                    )

                def _mlp(self, *runtime_args, **runtime_kwargs):
                    return cuda_timed(
                        f"{self._profile_phase}.mlp",
                        super()._mlp,
                        *runtime_args,
                        **runtime_kwargs,
                    )

            runtime_globals["load_moss_lm_head"] = profiled_head_loader
            runtime_globals["ForgeQwen359BCausalLMRuntime"] = ProfiledMossRuntime
            runtime_memory_management.load_models_gpu = profiled_model_placement

        context = namespace["MODEL_MANAGER"].resolve(sd_model)
        assert context.identity.adapter_id == "moss-image-qwen35-9b"
        assert context.capabilities.text_generation is True
        assert context.capabilities.vision_input is True

        persona_prompt = namespace["DEFAULT_PERSONA_PROMPT"]
        if args.persona_name:
            persona_store = json.loads(
                (EXTENSION_ROOT / "personas.json").read_text(encoding="utf-8")
            )
            try:
                persona_prompt = persona_store["personas"][args.persona_name]
            except KeyError as exc:
                raise KeyError(
                    f"Unknown local Idea-to-Prompt persona: {args.persona_name}"
                ) from exc

        generation_id = namespace["_begin_generation_request"]()
        memory_before = None
        if args.profile_detail and torch.cuda.is_available():
            torch.cuda.synchronize()
            torch.cuda.reset_peak_memory_stats()
            free_bytes, total_bytes = torch.cuda.mem_get_info()
            memory_before = {
                "allocated": torch.cuda.memory_allocated(),
                "reserved": torch.cuda.memory_reserved(),
                "free": free_bytes,
                "total": total_bytes,
            }
        generation_started = time.perf_counter()
        generation_options = {
            "system_prompt": persona_prompt,
            "vram_profile": "8 GB",
            "max_tokens": max(1, min(args.max_tokens, 64)),
            "do_sample": False,
            "temperature": 0.7,
            "top_k": 64,
            "top_p": 0.95,
            "min_p": 0.05,
            "repetition_penalty": 1.05,
            "seed": 0,
            "generation_id": generation_id,
        }
        if args.vision:
            image = Image.new("RGB", (384, 256), "#5d83b3")
            draw = ImageDraw.Draw(image)
            draw.ellipse((128, 64, 256, 192), fill="#d45731")
            draw.rectangle((155, 95, 229, 169), fill="#f1d8a8")
            result = context.adapter.generate_image(
                image=image,
                instruction="Describe the simple geometric picture faithfully.",
                show_model_thinking=False,
                **generation_options,
            )
        else:
            result = context.adapter.generate_text(
                model_context=context,
                idea="A red fox resting beside a quiet alpine lake at sunrise.",
                instruction="Use a wide cinematic composition and natural detail.",
                workflow=namespace["Workflow"].IDEA_TO_PROMPT,
                **generation_options,
            )
        timings["generation_total"] = time.perf_counter() - generation_started
        text, input_count, output_count, finish_reason, *_rest = result
        visible, _thinking_hidden = context.adapter.strip_model_thinking(text)
        if not visible:
            raise RuntimeError("MOSS Qwen3.5-9B produced no visible text")
        print(  # noqa: T201
            "MOSS_VISION_GENERATION_OK" if args.vision else "MOSS_TEXT_GENERATION_OK"
        )
        print("ADAPTER", context.identity.display_name)  # noqa: T201
        print("PERSONA", args.persona_name or "Default")  # noqa: T201
        print("TOKENS", input_count, output_count, finish_reason)  # noqa: T201
        print("VISIBLE", visible[:240].replace("\n", " "))  # noqa: T201
        if args.profile or args.profile_detail:
            for name in (
                "model_reload",
                "generation_total",
                "model_placement",
                "head_load",
                "runtime_init",
                "prefill",
                "logits",
                "decode",
            ):
                print(  # noqa: T201
                    "PROFILE",
                    name,
                    f"{timings.get(name, 0.0):.3f}s",
                    f"calls={counters.get(name, 1)}",
                )
        if args.profile_detail:
            synchronize()
            for name in sorted(cuda_events):
                elapsed_ms = sum(
                    start.elapsed_time(end) for start, end in cuda_events[name]
                )
                print(  # noqa: T201
                    "PROFILE_CUDA",
                    name,
                    f"{elapsed_ms / 1000.0:.3f}s",
                    f"calls={len(cuda_events[name])}",
                )
            for name in ("logits", "decode"):
                values = call_timings.get(name, [])
                if not values:
                    continue
                ordered = sorted(values)
                p95_index = round(0.95 * (len(ordered) - 1))
                print(  # noqa: T201
                    "PROFILE_CALL",
                    name,
                    f"min={ordered[0]:.3f}s",
                    f"median={statistics.median(ordered):.3f}s",
                    f"p95={ordered[p95_index]:.3f}s",
                    f"max={ordered[-1]:.3f}s",
                )
            if memory_before is not None:
                free_after, _total_after = torch.cuda.mem_get_info()
                mib = 1024 * 1024
                print(  # noqa: T201
                    "PROFILE_VRAM",
                    f"allocated_before={memory_before['allocated'] / mib:.1f}MiB",
                    f"reserved_before={memory_before['reserved'] / mib:.1f}MiB",
                    f"peak_allocated={torch.cuda.max_memory_allocated() / mib:.1f}MiB",
                    f"peak_reserved={torch.cuda.max_memory_reserved() / mib:.1f}MiB",
                    f"free_before={memory_before['free'] / mib:.1f}MiB",
                    f"free_after={free_after / mib:.1f}MiB",
                )
        return 0
    finally:
        if runtime_globals is not None:
            runtime_globals["ForgeQwen359BCausalLMRuntime"] = original_runtime
            runtime_globals["load_moss_lm_head"] = original_head_loader
            if original_model_placement is not None:
                runtime_globals[
                    "memory_management"
                ].load_models_gpu = original_model_placement
        if generation_id and namespace is not None:
            namespace["_clear_generation_request"](generation_id)
        sd_models.unload_model_weights()


if __name__ == "__main__":
    raise SystemExit(main())
