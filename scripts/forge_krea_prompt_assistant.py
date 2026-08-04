"""KREA2 prompt generation by reusing Forge Neo's active Qwen3-VL encoder.

Copyright (C) 2026 vibecodingtoolmaker
SPDX-License-Identifier: AGPL-3.0-only

This extension never loads a language model. It only operates on the KREA2
Qwen3-VL text/vision encoder and tokenizer already owned by the active Forge
diffusion engine.
"""

from __future__ import annotations

import html
import json
import logging
import math
import os
import sys
import threading
import time
import uuid
from functools import partial
from pathlib import Path

import gradio as gr
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image, ImageOps

from backend import memory_management
from modules import call_queue, script_callbacks, shared


LOGGER = logging.getLogger("forge_krea_prompt_assistant")
EXTENSION_VERSION = "0.5.0-alpha.1"
KREA2_ENCODER_MODULE = "backend.nn.llm.llama"
KREA2_ENCODER_CLASS = "Qwen3VL"
AUTO_VRAM_PROFILE = "Auto"
VRAM_PROFILES = {
    8: 4096,
    12: 5120,
    16: 6144,
    24: 8192,
    32: 12288,
}
DEFAULT_MAX_OUTPUT_TOKENS = 1024
PERSONA_STORE_VERSION = 1
DEFAULT_PERSONA_NAME = "Default"
MAX_PERSONAS = 100
MAX_PERSONA_NAME_LENGTH = 80
MAX_SYSTEM_PROMPT_LENGTH = 16_000
EXTENSION_ROOT = Path(__file__).resolve().parents[1]
PERSONAS_PATH = EXTENSION_ROOT / "personas.json"
IMAGE_PERSONAS_PATH = EXTENSION_ROOT / "image_personas.json"
PERSONA_LOCK = threading.RLock()
GENERATION_CANCEL_LOCK = threading.RLock()
GENERATION_CANCEL_EVENTS: dict[str, threading.Event] = {}
CHAT_SYSTEM_START = "<|im_start|>system\n"
CHAT_USER_BOUNDARY = "<|im_end|>\n<|im_start|>user\n"
VISION_TARGET_PIXELS = 768 * 768
VISION_PATCH_FACTOR = 32
VISION_MAX_DIMENSION = 4096
IMAGE_PREFILL_RESERVE_BYTES = 512 * 1024 * 1024
DEFAULT_IMAGE_REQUEST = (
    "Create a KREA2 image-generation prompt that faithfully reconstructs "
    "the uploaded image."
)

DEFAULT_PERSONA_PROMPT = """You are a prompt-writing assistant for the KREA2 image-generation model. Rewrite the user's image idea as one polished image prompt.

Requirements:
- Return only the finished prompt, with no introduction, notes, headings, quotation marks, or alternatives.
- Write coherent natural-language prose, not a comma-separated tag list.
- Preserve every explicit subject, action, setting, style, and constraint from the idea.
- Add useful concrete visual detail without changing the concept or inventing extra characters.
- Organize the description around subject, action, environment, color, shape, size, texture, quantity, visible text, spatial relationships, composition, lighting, materials, and camera perspective when relevant.
- Avoid empty quality slogans and avoid repeating details.
- Write the finished prompt in English."""

DEFAULT_IMAGE_PERSONA_PROMPT = """You are an image-analysis and prompt-writing assistant for the KREA2 image-generation model. Examine the uploaded image and convert its visible content into one polished prompt that can be used to recreate it.

Requirements:
- Return only the finished prompt, with no introduction, analysis, notes, headings, quotation marks, or alternatives.
- Write coherent natural-language prose, not a comma-separated tag list.
- Faithfully describe visible subjects, actions, expressions, clothing, objects, environment, colors, shapes, scale, textures, quantities, and spatial relationships.
- Preserve the image's composition, crop, camera angle, perspective, focal length cues, depth of field, lighting, shadows, palette, materials, and artistic or photographic style.
- Include legible visible text exactly as it appears and do not invent unreadable text.
- Do not identify real people or infer names, private facts, diagnoses, ethnicity, religion, sexuality, or other sensitive traits that are not visually explicit.
- Do not mention the source image, the analysis process, or uncertainty in the finished prompt.
- Avoid empty quality slogans and avoid repeating details.
- Write the finished prompt in English."""


class PromptAssistantError(RuntimeError):
    """Expected, user-facing validation or compatibility failure."""


class GenerationCancelled(PromptAssistantError):
    """Prompt generation was cancelled by the user."""


def _begin_generation_request() -> str:
    generation_id = uuid.uuid4().hex
    with GENERATION_CANCEL_LOCK:
        GENERATION_CANCEL_EVENTS[generation_id] = threading.Event()
    return generation_id


def _generation_cancelled(generation_id) -> bool:
    if not generation_id:
        return False
    with GENERATION_CANCEL_LOCK:
        cancel_event = GENERATION_CANCEL_EVENTS.get(str(generation_id))
    return bool(cancel_event and cancel_event.is_set())


def _raise_if_generation_cancelled(generation_id) -> None:
    if _generation_cancelled(generation_id):
        raise GenerationCancelled("Generation cancelled by user.")


def _request_generation_stop(generation_id):
    with GENERATION_CANCEL_LOCK:
        cancel_event = GENERATION_CANCEL_EVENTS.get(str(generation_id or ""))
        if cancel_event is not None:
            cancel_event.set()
    if cancel_event is None:
        return (
            "### Generation has already finished.",
            gr.update(value="Stop", interactive=False),
        )
    return (
        "### ⏹ Stop requested — finishing the current model step…",
        gr.update(value="⏹ Stopping…", interactive=False),
    )


def _clear_generation_request(generation_id) -> None:
    if not generation_id:
        return
    with GENERATION_CANCEL_LOCK:
        GENERATION_CANCEL_EVENTS.pop(str(generation_id), None)


def _detected_vram_gb() -> float:
    try:
        return max(float(memory_management.total_vram) / 1024.0, 0.0)
    except (AttributeError, TypeError, ValueError):
        return 0.0


def _automatic_vram_tier(vram_gb: float | None = None) -> int:
    if vram_gb is None:
        vram_gb = _detected_vram_gb()
    reported_gb = int(round(max(float(vram_gb), 0.0)))
    eligible = [tier for tier in VRAM_PROFILES if tier <= reported_gb]
    return max(eligible) if eligible else min(VRAM_PROFILES)


def _resolve_vram_profile(selection) -> tuple[int, int]:
    text = str(selection or AUTO_VRAM_PROFILE).strip()
    if text.startswith(AUTO_VRAM_PROFILE):
        tier = _automatic_vram_tier()
    else:
        try:
            tier = int(text.split()[0])
        except (TypeError, ValueError, IndexError) as exc:
            raise PromptAssistantError(f"Unknown VRAM profile: {text}") from exc
        if tier not in VRAM_PROFILES:
            raise PromptAssistantError(f"Unknown VRAM profile: {text}")
    return tier, VRAM_PROFILES[tier]


def _vram_profile_status(selection) -> str:
    tier, context_limit = _resolve_vram_profile(selection)
    detected = _detected_vram_gb()
    mode = (
        f"automatically detected from {detected:.1f} GB"
        if str(selection or AUTO_VRAM_PROFILE).startswith(AUTO_VRAM_PROFILE)
        else "manually selected"
    )
    return (
        f"**{tier} GB VRAM profile** ({mode}) · shared context window: "
        f"**{context_limit} tokens**. Actual text tokens and future image tokens "
        "are deducted; the remainder is available for generation."
    )


def _change_vram_profile(selection, current_max_tokens):
    _, context_limit = _resolve_vram_profile(selection)
    try:
        current = int(current_max_tokens)
    except (TypeError, ValueError):
        current = DEFAULT_MAX_OUTPUT_TOKENS
    current = min(max(current, 32), context_limit)
    return (
        gr.update(maximum=context_limit, value=current),
        _vram_profile_status(selection),
    )


def _validate_persona_name(name) -> str:
    if not isinstance(name, str):
        raise PromptAssistantError("The persona name must be text.")
    name = name.strip()
    if not name:
        raise PromptAssistantError("Please enter a persona name.")
    if len(name) > MAX_PERSONA_NAME_LENGTH:
        raise PromptAssistantError(
            f"The persona name may contain at most {MAX_PERSONA_NAME_LENGTH} characters."
        )
    if any(character in name for character in "\r\n\t"):
        raise PromptAssistantError(
            "The persona name may not contain line breaks or tabs."
        )
    return name


def _validate_system_prompt(system_prompt) -> str:
    if not isinstance(system_prompt, str) or not system_prompt.strip():
        raise PromptAssistantError("The persona system prompt may not be empty.")
    system_prompt = system_prompt.strip()
    if len(system_prompt) > MAX_SYSTEM_PROMPT_LENGTH:
        raise PromptAssistantError(
            "The persona system prompt is too long; the maximum is "
            f"{MAX_SYSTEM_PROMPT_LENGTH} characters."
        )
    return system_prompt


def _default_persona_store(default_prompt=DEFAULT_PERSONA_PROMPT) -> dict:
    return {
        "version": PERSONA_STORE_VERSION,
        "personas": {DEFAULT_PERSONA_NAME: default_prompt},
    }


def _read_persona_store(
    path=PERSONAS_PATH, default_prompt=DEFAULT_PERSONA_PROMPT
) -> dict:
    with PERSONA_LOCK:
        if not path.exists():
            return _default_persona_store(default_prompt)
        try:
            raw = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            raise PromptAssistantError(
                f"The persona file could not be read: {exc}"
            ) from exc

        if not isinstance(raw, dict) or raw.get("version") != PERSONA_STORE_VERSION:
            raise PromptAssistantError(
                "The persona file uses an unknown format version."
            )
        raw_personas = raw.get("personas")
        if not isinstance(raw_personas, dict):
            raise PromptAssistantError(
                "The persona file does not contain a valid persona collection."
            )
        if len(raw_personas) > MAX_PERSONAS:
            raise PromptAssistantError(
                f"The persona file contains more than {MAX_PERSONAS} personas."
            )

        personas = {}
        for name, system_prompt in raw_personas.items():
            valid_name = _validate_persona_name(name)
            personas[valid_name] = _validate_system_prompt(system_prompt)
        if DEFAULT_PERSONA_NAME not in personas:
            personas = {
                DEFAULT_PERSONA_NAME: default_prompt,
                **personas,
            }
        return {"version": PERSONA_STORE_VERSION, "personas": personas}


def _write_persona_store(personas: dict[str, str], path=PERSONAS_PATH) -> None:
    if len(personas) > MAX_PERSONAS:
        raise PromptAssistantError(f"At most {MAX_PERSONAS} personas can be stored.")
    validated = {
        _validate_persona_name(name): _validate_system_prompt(system_prompt)
        for name, system_prompt in personas.items()
    }
    if DEFAULT_PERSONA_NAME not in validated:
        raise PromptAssistantError("The Default persona may not be removed.")

    payload = {
        "version": PERSONA_STORE_VERSION,
        "personas": validated,
    }
    serialized = json.dumps(payload, ensure_ascii=False, indent=2) + "\n"
    temporary_path = path.with_name(path.name + ".tmp")
    with PERSONA_LOCK:
        try:
            temporary_path.write_text(serialized, encoding="utf-8", newline="\n")
            os.replace(temporary_path, path)
        except OSError as exc:
            raise PromptAssistantError(
                f"The persona file could not be saved: {exc}"
            ) from exc


def _initial_persona_state_for(path, default_prompt) -> tuple[list[str], str, str, str]:
    try:
        store = _read_persona_store(path, default_prompt)
        personas = store["personas"]
        selected = DEFAULT_PERSONA_NAME
        return list(personas), selected, personas[selected], ""
    except PromptAssistantError as exc:
        LOGGER.exception("Could not initialize persona store")
        return (
            [DEFAULT_PERSONA_NAME],
            DEFAULT_PERSONA_NAME,
            default_prompt,
            f"**Persona file error:** {exc}",
        )


def _initial_persona_state() -> tuple[list[str], str, str, str]:
    return _initial_persona_state_for(PERSONAS_PATH, DEFAULT_PERSONA_PROMPT)


def _initial_image_persona_state() -> tuple[list[str], str, str, str]:
    return _initial_persona_state_for(IMAGE_PERSONAS_PATH, DEFAULT_IMAGE_PERSONA_PROMPT)


def _load_persona_fields_from(name, path, default_prompt):
    try:
        name = _validate_persona_name(name)
        personas = _read_persona_store(path, default_prompt)["personas"]
        if name not in personas:
            raise PromptAssistantError(f"Persona not found: {name}")
        return name, personas[name], f"Persona **{html.escape(name)}** loaded."
    except PromptAssistantError as exc:
        return "", "", f"**Persona error:** {exc}"


def _load_persona_fields(name):
    return _load_persona_fields_from(name, PERSONAS_PATH, DEFAULT_PERSONA_PROMPT)


def _load_image_persona_fields(name):
    return _load_persona_fields_from(
        name, IMAGE_PERSONAS_PATH, DEFAULT_IMAGE_PERSONA_PROMPT
    )


def _save_persona_fields_to(name, system_prompt, path, default_prompt):
    try:
        name = _validate_persona_name(name)
        system_prompt = _validate_system_prompt(system_prompt)
        with PERSONA_LOCK:
            personas = _read_persona_store(path, default_prompt)["personas"]
            if name not in personas and len(personas) >= MAX_PERSONAS:
                raise PromptAssistantError(
                    f"At most {MAX_PERSONAS} personas can be stored."
                )
            personas[name] = system_prompt
            _write_persona_store(personas, path)
        return (
            gr.update(choices=list(personas), value=name),
            name,
            system_prompt,
            f"Persona **{html.escape(name)}** saved.",
        )
    except PromptAssistantError as exc:
        return gr.update(), name, system_prompt, f"**Not saved:** {exc}"


def _save_persona_fields(name, system_prompt):
    return _save_persona_fields_to(
        name, system_prompt, PERSONAS_PATH, DEFAULT_PERSONA_PROMPT
    )


def _save_image_persona_fields(name, system_prompt):
    return _save_persona_fields_to(
        name, system_prompt, IMAGE_PERSONAS_PATH, DEFAULT_IMAGE_PERSONA_PROMPT
    )


def _new_persona_fields():
    return (
        gr.update(value=None),
        "",
        "",
        "New persona: enter a name and system prompt, then save it.",
    )


def _delete_persona_fields_from(name, path, default_prompt):
    try:
        name = _validate_persona_name(name)
        if name == DEFAULT_PERSONA_NAME:
            raise PromptAssistantError(
                "The Default persona can be edited but not deleted."
            )
        with PERSONA_LOCK:
            personas = _read_persona_store(path, default_prompt)["personas"]
            if name not in personas:
                raise PromptAssistantError(f"Persona not found: {name}")
            del personas[name]
            _write_persona_store(personas, path)
        selected = DEFAULT_PERSONA_NAME
        return (
            gr.update(choices=list(personas), value=selected),
            selected,
            personas[selected],
            f"Persona **{html.escape(name)}** deleted; Default is active again.",
        )
    except PromptAssistantError as exc:
        return gr.update(), name, gr.update(), f"**Not deleted:** {exc}"


def _delete_persona_fields(name):
    return _delete_persona_fields_from(name, PERSONAS_PATH, DEFAULT_PERSONA_PROMPT)


def _delete_image_persona_fields(name):
    return _delete_persona_fields_from(
        name, IMAGE_PERSONAS_PATH, DEFAULT_IMAGE_PERSONA_PROMPT
    )


def _class_path(value) -> str:
    cls = type(value)
    return f"{cls.__module__}.{cls.__qualname__}"


def _forge_stack_status() -> tuple[bool, str]:
    sd_models_module = sys.modules.get("modules.sd_models")
    model_data = getattr(sd_models_module, "model_data", None)
    if model_data is None:
        return False, (
            "**Forge runtime is still initializing.** The loader will report "
            "the active KREA2 stack as soon as Forge's model manager is ready."
        )
    try:
        sd_model, _, _, _, _ = _active_krea_components(model_data.get_sd_model())
    except PromptAssistantError as exc:
        return False, (
            "**KREA2 stack not ready.** Select KREA2, its text encoder, and VAE "
            "in Forge's model controls, then click **Load current Forge "
            f"selection**. Current state: {exc}"
        )
    except Exception as exc:
        LOGGER.debug("Forge runtime readiness check failed", exc_info=True)
        return False, (
            "**Forge runtime not ready yet.** The model loader will become "
            f"available after Forge finishes initialization ({type(exc).__name__})."
        )

    checkpoint_info = getattr(sd_model, "sd_checkpoint_info", None)
    checkpoint_name = getattr(checkpoint_info, "name", None) or os.path.basename(
        getattr(sd_model, "filename", "KREA2")
    )
    configured_modules = [
        os.path.basename(path)
        for path in getattr(shared.opts, "forge_additional_modules", [])
    ]
    modules_display = (
        ", ".join(html.escape(name) for name in configured_modules)
        or "integrated modules"
    )
    return True, (
        "**KREA2 stack ready.** "
        f"Model: **{html.escape(checkpoint_name)}** · "
        f"VAE / text encoder: {modules_display}"
    )


def _load_current_forge_selection():
    from modules import sd_models
    from modules_forge import main_entry

    started = time.perf_counter()
    try:
        main_entry.refresh_model_loading_parameters(refresh=True)
        sd_models.forge_model_reload()
        ready, ready_status = _forge_stack_status()
        if not ready:
            raise PromptAssistantError(
                "Forge loaded the current selection, but it is not a compatible "
                "KREA2 stack. Select KREA2, its VAE, and its Qwen3-VL text "
                "encoder in Forge first."
            )
        elapsed = time.perf_counter() - started
        return f"{ready_status} · loaded in {elapsed:.1f} s"
    except PromptAssistantError as exc:
        LOGGER.warning("Forge current-selection load rejected: %s", exc)
        return f"**Not loaded:** {exc}"
    except torch.OutOfMemoryError:
        LOGGER.exception("CUDA OOM while Forge loaded the current selection")
        return (
            "**CUDA out of memory while loading the stack.** Forge remains in "
            "control of model memory; try a lower-bit diffusion model or free VRAM."
        )
    except Exception as exc:
        LOGGER.exception("Unexpected Forge current-selection load failure")
        return f"**Load error:** {type(exc).__name__}: {exc}"


def _active_krea_components(sd_model_override=None):
    """Resolve, but never retain, the active KREA2 runtime components."""
    sd_model = sd_model_override if sd_model_override is not None else shared.sd_model
    if sd_model is None:
        raise PromptAssistantError(
            "No diffusion model is loaded. Use 'Load current Forge selection' above."
        )

    if _class_path(sd_model) != "backend.diffusion_engine.krea.Krea2":
        raise PromptAssistantError(
            "This alpha currently supports only an active KREA2 model. "
            f"Active model: {_class_path(sd_model)}. Select a compatible stack "
            "in Forge, then use 'Load current Forge selection' above."
        )

    forge_objects = getattr(sd_model, "forge_objects", None)
    clip = getattr(forge_objects, "clip", None)
    cond_stage_model = getattr(clip, "cond_stage_model", None)
    tokenizer_container = getattr(clip, "tokenizer", None)
    encoder = getattr(cond_stage_model, "qwen3vl_4b", None)
    tokenizer = getattr(tokenizer_container, "qwen3vl_4b", None)
    engine = getattr(sd_model, "text_processing_engine_qwen", None)

    if clip is None or encoder is None or tokenizer is None or engine is None:
        raise PromptAssistantError(
            "The expected KREA2 encoder/tokenizer objects were not found. Select "
            "a complete stack in Forge, then use 'Load current Forge selection' "
            "above."
        )

    if getattr(engine, "text_encoder", None) is not encoder:
        raise PromptAssistantError(
            "Forge is using unexpectedly different KREA2 encoder objects."
        )
    if getattr(engine, "tokenizer", None) is not tokenizer:
        raise PromptAssistantError(
            "Forge is using unexpectedly different KREA2 tokenizer objects."
        )
    if (
        type(encoder).__module__ != KREA2_ENCODER_MODULE
        or type(encoder).__name__ != KREA2_ENCODER_CLASS
    ):
        raise PromptAssistantError(
            "The active KREA2 encoder class is unknown to this adapter: "
            f"{_class_path(encoder)}"
        )

    patcher = getattr(clip, "patcher", None)
    if patcher is None:
        raise PromptAssistantError("The text encoder's Forge ModelPatcher is missing.")

    return sd_model, clip, encoder, tokenizer, engine


def _render_generation_prompt(
    engine, idea: str, instruction: str, system_prompt: str
) -> str:
    template = getattr(engine, "llama_template", None)
    if not isinstance(template, str) or template.count("{}") != 1:
        raise PromptAssistantError("The KREA2 chat template is incompatible.")
    if (
        template.count(CHAT_SYSTEM_START) != 1
        or template.count(CHAT_USER_BOUNDARY) != 1
    ):
        raise PromptAssistantError(
            "The role structure of the KREA2 chat template is incompatible."
        )

    system_prompt = _validate_system_prompt(system_prompt)
    user_message = idea.strip()
    instruction = str(instruction or "").strip()
    if instruction:
        user_message += f"\n\nAdditional instruction:\n{instruction}"
    rendered = template.format(user_message)
    system_start = rendered.index(CHAT_SYSTEM_START) + len(CHAT_SYSTEM_START)
    system_end = rendered.index(CHAT_USER_BOUNDARY, system_start)
    rendered = rendered[:system_start] + system_prompt + rendered[system_end:]
    # This empty reasoning block mirrors Qwen3 text-generation convention and
    # makes the model continue directly with the requested answer.
    return rendered + "<think>\n\n</think>\n\n"


def _tokenize(tokenizer, rendered_prompt: str) -> list[int]:
    encoded = tokenizer(
        rendered_prompt,
        add_special_tokens=False,
        return_attention_mask=False,
    )
    try:
        token_ids = encoded["input_ids"]
    except (KeyError, TypeError):
        token_ids = getattr(encoded, "input_ids", None)
    if not isinstance(token_ids, list) or not token_ids:
        raise PromptAssistantError("The KREA2 tokenizer returned no token IDs.")
    if token_ids and isinstance(token_ids[0], list):
        if len(token_ids) != 1:
            raise PromptAssistantError("Only one prompt at a time is supported.")
        token_ids = token_ids[0]
    return [int(token_id) for token_id in token_ids]


def _prepare_uploaded_image(
    image,
) -> tuple[torch.Tensor, tuple[int, int], tuple[int, int], int]:
    if image is None:
        raise PromptAssistantError("Please upload an image first.")
    if not isinstance(image, Image.Image):
        raise PromptAssistantError("The uploaded image has an unsupported format.")

    image = ImageOps.exif_transpose(image).convert("RGB")
    original_width, original_height = image.size
    if original_width < 1 or original_height < 1:
        raise PromptAssistantError("The uploaded image has invalid dimensions.")

    scale = math.sqrt(VISION_TARGET_PIXELS / float(original_width * original_height))
    scale = min(
        scale,
        VISION_MAX_DIMENSION / float(original_width),
        VISION_MAX_DIMENSION / float(original_height),
    )

    def aligned_dimension(value: float) -> int:
        aligned = int(round(value / VISION_PATCH_FACTOR)) * VISION_PATCH_FACTOR
        return min(max(aligned, VISION_PATCH_FACTOR), VISION_MAX_DIMENSION)

    prepared_width = aligned_dimension(original_width * scale)
    prepared_height = aligned_dimension(original_height * scale)
    pixels = np.asarray(image, dtype=np.float32)
    image_tensor = torch.from_numpy(np.ascontiguousarray(pixels)).div_(255.0)
    image_tensor = image_tensor.movedim(-1, 0).unsqueeze(0)
    image_tensor = (
        F.interpolate(
            image_tensor,
            size=(prepared_height, prepared_width),
            mode="area",
        )
        .movedim(1, -1)
        .contiguous()
    )

    visual_tokens = prepared_width * prepared_height // (VISION_PATCH_FACTOR**2)
    return (
        image_tensor,
        (original_width, original_height),
        (prepared_width, prepared_height),
        visual_tokens,
    )


def _render_image_generation_prompt(
    engine, instruction: str, system_prompt: str
) -> str:
    vision_block = getattr(engine, "vision_block", None)
    if not isinstance(vision_block, str) or not vision_block:
        raise PromptAssistantError("The KREA2 image chat template is unavailable.")
    instruction = (
        instruction.strip()
        if isinstance(instruction, str) and instruction.strip()
        else DEFAULT_IMAGE_REQUEST
    )
    return _render_generation_prompt(
        engine,
        f"{vision_block}\n{instruction}",
        "",
        system_prompt,
    )


def _multimodal_embeds(engine, encoder, token_ids, image_tensor):
    image_token_id = getattr(engine, "id_image", None)
    if not isinstance(image_token_id, int):
        raise PromptAssistantError("The KREA2 image-token ID is unavailable.")
    if token_ids.count(image_token_id) != 1:
        raise PromptAssistantError(
            "The KREA2 image prompt did not contain exactly one image token."
        )

    tokens = [
        {
            "type": "image",
            "data": image_tensor,
            "original_type": "image",
        }
        if token_id == image_token_id
        else token_id
        for token_id in token_ids
    ]
    embeds, attention_mask, token_counts, embeds_info = engine.process_embeds([tokens])
    if embeds.ndim != 3 or embeds.shape[0] != 1:
        raise PromptAssistantError(
            "The KREA2 vision encoder returned incompatible image embeddings."
        )
    if not token_counts or int(token_counts[0]) != int(embeds.shape[1]):
        raise PromptAssistantError(
            "The KREA2 vision encoder returned inconsistent token accounting."
        )

    build_image_inputs = getattr(encoder, "build_image_inputs", None)
    if not callable(build_image_inputs):
        raise PromptAssistantError(
            "The active KREA2 encoder has no compatible multimodal input builder."
        )
    position_ids, visual_pos_masks, deepstack = build_image_inputs(embeds, embeds_info)
    if position_ids is None or visual_pos_masks is None or not deepstack:
        raise PromptAssistantError(
            "The active KREA2 encoder returned incomplete multimodal metadata."
        )
    return embeds, attention_mask, position_ids, visual_pos_masks, deepstack


def _prefill_with_deepstack(
    core_model,
    embeds,
    attention_mask,
    position_ids,
    visual_pos_masks,
    deepstack,
    past_key_values,
):
    if len(deepstack) > len(core_model.layers):
        raise PromptAssistantError(
            "The KREA2 vision encoder returned too many DeepStack features."
        )

    handles = []

    def make_hook(visual_embeds):
        def inject_visual_features(_module, args, kwargs):
            hidden_states = kwargs.get("x")
            if not isinstance(hidden_states, torch.Tensor):
                raise PromptAssistantError(
                    "The KREA2 language layer did not expose its hidden states."
                )
            mask = visual_pos_masks.to(device=hidden_states.device)
            selected = hidden_states[mask]
            visual = visual_embeds.to(
                device=hidden_states.device, dtype=hidden_states.dtype
            )
            if selected.shape != visual.shape:
                raise PromptAssistantError(
                    "The KREA2 DeepStack image features have an incompatible shape."
                )
            updated = hidden_states.clone()
            updated[mask] = selected + visual
            updated_kwargs = dict(kwargs)
            updated_kwargs["x"] = updated
            return args, updated_kwargs

        return inject_visual_features

    try:
        for layer, visual_embeds in zip(core_model.layers, deepstack):
            handles.append(
                layer.register_forward_pre_hook(
                    make_hook(visual_embeds), with_kwargs=True
                )
            )
        return core_model(
            None,
            embeds=embeds,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=past_key_values,
        )
    finally:
        for handle in handles:
            handle.remove()


def _execution_dtype(core_model, device: torch.device) -> torch.dtype:
    try:
        dtype = core_model.layers[0].self_attn.q_proj.weight.dtype
    except Exception:
        dtype = None
    if dtype in (torch.float16, torch.bfloat16):
        return dtype
    return (
        torch.bfloat16 if memory_management.should_use_bf16(device) else torch.float32
    )


def _allocate_kv_cache(core_model, batch: int, capacity: int, device, dtype):
    config = core_model.config
    return [
        (
            torch.empty(
                batch,
                config.num_key_value_heads,
                capacity,
                config.head_dim,
                device=device,
                dtype=dtype,
            ),
            torch.empty(
                batch,
                config.num_key_value_heads,
                capacity,
                config.head_dim,
                device=device,
                dtype=dtype,
            ),
            0,
        )
        for _ in range(config.num_hidden_layers)
    ]


def _kv_cache_bytes(config, batch: int, capacity: int, dtype: torch.dtype) -> int:
    element_size = torch.empty((), dtype=dtype).element_size()
    return int(
        batch
        * config.num_hidden_layers
        * 2  # key and value
        * config.num_key_value_heads
        * capacity
        * config.head_dim
        * element_size
    )


def _tied_embedding_logits(hidden_states: torch.Tensor, embedding) -> torch.Tensor:
    weight_functions = getattr(embedding, "weight_function", [])
    if weight_functions:
        raise PromptAssistantError(
            "The active KREA2 token embedding layer has online patches; safe "
            "tied-embedding projection is not implemented for this case yet."
        )

    weight = getattr(embedding, "weight", None)
    if not isinstance(weight, torch.Tensor):
        raise PromptAssistantError(
            "The existing token embedding matrix is not a directly usable tensor."
        )
    if weight.device != hidden_states.device:
        raise PromptAssistantError(
            "Forge did not place the token embedding matrix on the same device "
            "as the encoder; the extension will not create a large helper copy."
        )

    last_hidden = hidden_states[:, -1:, :].to(dtype=weight.dtype)
    return F.linear(last_hidden, weight, bias=None)[:, -1, :]


def _sample_token(
    logits: torch.Tensor,
    history: list[int],
    do_sample: bool,
    temperature: float,
    top_k: int,
    top_p: float,
    min_p: float,
    repetition_penalty: float,
    generator: torch.Generator | None,
) -> torch.Tensor:
    logits = logits.float()
    if not do_sample or temperature <= 0.0:
        return torch.argmax(logits, dim=-1, keepdim=True)

    if history and repetition_penalty != 1.0:
        token_ids = torch.tensor(
            sorted(set(history)), device=logits.device, dtype=torch.long
        )
        selected = logits[:, token_ids]
        selected = torch.where(
            selected < 0,
            selected * repetition_penalty,
            selected / repetition_penalty,
        )
        logits[:, token_ids] = selected

    logits.div_(temperature)
    top_k = min(max(int(top_k), 1), logits.shape[-1])
    candidate_logits, candidate_ids = torch.topk(logits, top_k, dim=-1)

    if min_p > 0.0:
        unfiltered_probs = torch.softmax(candidate_logits, dim=-1)
        threshold = min_p * unfiltered_probs.max(dim=-1, keepdim=True).values
        candidate_logits.masked_fill_(
            unfiltered_probs < threshold,
            torch.finfo(candidate_logits.dtype).min,
        )

    if top_p < 1.0:
        sorted_logits, sorted_indices = torch.sort(
            candidate_logits, descending=True, dim=-1
        )
        cumulative_probs = torch.cumsum(torch.softmax(sorted_logits, dim=-1), dim=-1)
        remove_sorted = cumulative_probs > top_p
        remove_sorted[..., 0] = False
        remove = torch.zeros_like(candidate_logits, dtype=torch.bool)
        remove.scatter_(1, sorted_indices, remove_sorted)
        candidate_logits.masked_fill_(remove, torch.finfo(candidate_logits.dtype).min)

    probabilities = torch.softmax(candidate_logits, dim=-1)
    sampled_index = torch.multinomial(probabilities, num_samples=1, generator=generator)
    return candidate_ids.gather(1, sampled_index)


@torch.inference_mode()
def _generate_with_active_krea(
    idea: str,
    instruction: str,
    system_prompt: str,
    vram_profile,
    max_tokens: int,
    do_sample: bool,
    temperature: float,
    top_k: int,
    top_p: float,
    min_p: float,
    repetition_penalty: float,
    seed: int,
    generation_id: str,
) -> tuple[str, int, int, str, int, int, int, int, int, bool, int]:
    if not isinstance(idea, str) or not idea.strip():
        raise PromptAssistantError("Please enter an idea first.")
    _raise_if_generation_cancelled(generation_id)

    profile_tier, context_limit = _resolve_vram_profile(vram_profile)
    requested_max_tokens = int(max_tokens)
    if requested_max_tokens < 1 or requested_max_tokens > context_limit:
        raise PromptAssistantError(
            f"The {profile_tier} GB VRAM profile allows an output limit from "
            f"1 to {context_limit} tokens."
        )
    temperature = min(max(float(temperature), 0.01), 2.0)
    top_k = min(max(int(top_k), 1), 1000)
    top_p = min(max(float(top_p), 0.05), 1.0)
    min_p = min(max(float(min_p), 0.0), 1.0)
    repetition_penalty = min(max(float(repetition_penalty), 0.1), 5.0)
    seed = int(seed) % (2**63 - 1)

    _, clip, encoder, tokenizer, engine = _active_krea_components()
    rendered_prompt = _render_generation_prompt(
        engine, idea, instruction, system_prompt
    )
    core_model = getattr(encoder, "model", None)
    embedding = getattr(core_model, "embed_tokens", None)
    config = getattr(core_model, "config", None)
    if core_model is None or embedding is None or config is None:
        raise PromptAssistantError("The internal KREA2 Qwen core is unavailable.")

    expected_shape = (config.vocab_size, config.hidden_size)
    device = torch.device(clip.patcher.load_device)
    dtype = _execution_dtype(core_model, device)
    prompt_token_ids = _tokenize(tokenizer, rendered_prompt)
    input_token_count = len(prompt_token_ids)
    profile_output_available = context_limit - input_token_count
    if profile_output_available < 1:
        raise PromptAssistantError(
            f"The persona, idea, optional instruction, and chat template use "
            f"{input_token_count} tokens and fill the {context_limit}-token "
            "context window. Shorten the idea, instruction, or persona."
        )
    max_tokens = min(requested_max_tokens, profile_output_available)
    capacity = input_token_count + max_tokens
    model_context_limit = getattr(config, "max_position_embeddings", None)
    if isinstance(model_context_limit, int) and capacity > model_context_limit:
        raise PromptAssistantError(
            f"The active KREA2 model supports at most {model_context_limit} "
            "context tokens."
        )

    kv_bytes_per_token = _kv_cache_bytes(config, batch=1, capacity=1, dtype=dtype)
    kv_cache_bytes = kv_bytes_per_token * capacity
    # This targets the exact patcher already owned by KREA2 and tells Forge how
    # much additional room the temporary cache needs before weights are placed.
    memory_management.load_models_gpu(
        [clip.patcher], memory_required=int(kv_cache_bytes * 1.10)
    )
    _raise_if_generation_cancelled(generation_id)

    # Forge has now placed/offloaded weights while honoring the requested cache
    # room. Verify the physical free memory once more because other allocations
    # and fragmentation can change between UI creation and execution.
    free_vram = int(memory_management.get_free_memory(device))
    reserved_vram = int(memory_management.extra_reserved_memory())
    usable_cache_vram = max(free_vram - reserved_vram, 0)
    vram_capacity_limit = int(usable_cache_vram / (kv_bytes_per_token * 1.10))
    vram_output_available = max(vram_capacity_limit - input_token_count, 0)
    available_output_tokens = min(profile_output_available, vram_output_available)
    vram_limited = available_output_tokens < profile_output_available
    if available_output_tokens < 1:
        raise PromptAssistantError(
            "After loading the KREA2 encoder, there is not enough free VRAM for "
            "the input KV cache. Select a smaller context profile or free more "
            "VRAM in Forge."
        )
    max_tokens = min(requested_max_tokens, available_output_tokens)
    capacity = input_token_count + max_tokens

    # Fetch the parameter after model placement; a patcher may replace a
    # Parameter object while moving or applying weights.
    weight = getattr(embedding, "weight", None)
    if not isinstance(weight, torch.Tensor) or tuple(weight.shape) != expected_shape:
        raise PromptAssistantError(
            "The KREA2 embedding matrix does not have the expected LM projection shape."
        )
    if weight.device != device:
        raise PromptAssistantError(
            "Forge did not place the KREA2 embedding matrix on the execution "
            "device after the ModelPatcher call; the extension will not create "
            "a large helper copy."
        )
    past_key_values = _allocate_kv_cache(
        core_model, batch=1, capacity=capacity, device=device, dtype=dtype
    )
    generator = torch.Generator(device=device).manual_seed(seed) if do_sample else None
    stop_tokens = {
        int(token_id)
        for token_id in (
            getattr(tokenizer, "eos_token_id", None),
            151645,  # <|im_end|> in the KREA2 Qwen tokenizer
        )
        if token_id is not None
    }

    input_ids = torch.tensor([prompt_token_ids], device=device, dtype=torch.long)
    embeds = embedding(input_ids).to(dtype=dtype)
    generated_ids: list[int] = []
    finish_reason = "max_tokens"

    for _ in range(max_tokens):
        if _generation_cancelled(generation_id):
            finish_reason = "cancelled"
            break
        output = core_model(
            None,
            embeds=embeds,
            attention_mask=None,
            past_key_values=past_key_values,
        )
        if not isinstance(output, tuple) or len(output) != 3:
            raise PromptAssistantError(
                "The KREA2 Qwen core returned no compatible KV cache."
            )
        hidden_states, _, past_key_values = output
        logits = _tied_embedding_logits(hidden_states, embedding)
        next_token = _sample_token(
            logits=logits,
            history=generated_ids,
            do_sample=bool(do_sample),
            temperature=temperature,
            top_k=top_k,
            top_p=top_p,
            min_p=min_p,
            repetition_penalty=repetition_penalty,
            generator=generator,
        )
        token_id = int(next_token[0, 0].item())
        generated_ids.append(token_id)
        if token_id in stop_tokens:
            finish_reason = "eos"
            break
        embeds = embedding(next_token).to(dtype=dtype)

    generated_text = tokenizer.decode(generated_ids, skip_special_tokens=True).strip()
    if not generated_text:
        if finish_reason == "cancelled":
            raise GenerationCancelled("Generation cancelled by user.")
        raise PromptAssistantError(
            "The active Qwen3-VL text encoder produced no visible prompt. "
            "Please try again."
        )
    return (
        generated_text,
        len(prompt_token_ids),
        len(generated_ids),
        finish_reason,
        profile_tier,
        capacity,
        context_limit,
        requested_max_tokens,
        available_output_tokens,
        vram_limited,
        seed,
    )


@torch.inference_mode()
def _generate_with_active_krea_image(
    image,
    instruction: str,
    system_prompt: str,
    vram_profile,
    max_tokens: int,
    do_sample: bool,
    temperature: float,
    top_k: int,
    top_p: float,
    min_p: float,
    repetition_penalty: float,
    seed: int,
    generation_id: str,
):
    _raise_if_generation_cancelled(generation_id)
    profile_tier, context_limit = _resolve_vram_profile(vram_profile)
    requested_max_tokens = int(max_tokens)
    if requested_max_tokens < 1 or requested_max_tokens > context_limit:
        raise PromptAssistantError(
            f"The {profile_tier} GB VRAM profile allows an output limit from "
            f"1 to {context_limit} tokens."
        )
    temperature = min(max(float(temperature), 0.01), 2.0)
    top_k = min(max(int(top_k), 1), 1000)
    top_p = min(max(float(top_p), 0.05), 1.0)
    min_p = min(max(float(min_p), 0.0), 1.0)
    repetition_penalty = min(max(float(repetition_penalty), 0.1), 5.0)
    seed = int(seed) % (2**63 - 1)

    (
        image_tensor,
        original_size,
        prepared_size,
        estimated_visual_tokens,
    ) = _prepare_uploaded_image(image)
    _, clip, encoder, tokenizer, engine = _active_krea_components()
    rendered_prompt = _render_image_generation_prompt(
        engine, instruction, system_prompt
    )
    prompt_token_ids = _tokenize(tokenizer, rendered_prompt)
    image_token_id = getattr(engine, "id_image", None)
    if prompt_token_ids.count(image_token_id) != 1:
        raise PromptAssistantError(
            "The KREA2 tokenizer did not produce exactly one image token."
        )

    estimated_input_count = len(prompt_token_ids) - 1 + estimated_visual_tokens
    profile_output_available = context_limit - estimated_input_count
    if profile_output_available < 1:
        raise PromptAssistantError(
            f"The image, persona, instruction, and chat template require about "
            f"{estimated_input_count} tokens and fill the {context_limit}-token "
            "context window. Shorten the persona or choose a larger profile."
        )

    core_model = getattr(encoder, "model", None)
    embedding = getattr(core_model, "embed_tokens", None)
    config = getattr(core_model, "config", None)
    if core_model is None or embedding is None or config is None:
        raise PromptAssistantError("The internal KREA2 Qwen core is unavailable.")

    expected_shape = (config.vocab_size, config.hidden_size)
    device = torch.device(clip.patcher.load_device)
    dtype = _execution_dtype(core_model, device)
    estimated_max_tokens = min(requested_max_tokens, profile_output_available)
    estimated_capacity = estimated_input_count + estimated_max_tokens
    kv_bytes_per_token = _kv_cache_bytes(config, batch=1, capacity=1, dtype=dtype)
    estimated_kv_bytes = kv_bytes_per_token * estimated_capacity
    memory_management.load_models_gpu(
        [clip.patcher],
        memory_required=(int(estimated_kv_bytes * 1.10) + IMAGE_PREFILL_RESERVE_BYTES),
    )
    _raise_if_generation_cancelled(generation_id)

    (
        embeds,
        attention_mask,
        position_ids,
        visual_pos_masks,
        deepstack,
    ) = _multimodal_embeds(engine, encoder, prompt_token_ids, image_tensor)
    _raise_if_generation_cancelled(generation_id)
    if embeds.device != device:
        raise PromptAssistantError(
            "Forge placed the KREA2 image embeddings on an unexpected device."
        )
    embeds = embeds.to(dtype=dtype)
    input_token_count = int(embeds.shape[1])
    visual_token_count = int(visual_pos_masks.sum().item())
    if visual_token_count != estimated_visual_tokens:
        raise PromptAssistantError(
            "The KREA2 vision token count differs from the normalized image estimate."
        )

    profile_output_available = context_limit - input_token_count
    if profile_output_available < 1:
        raise PromptAssistantError(
            f"The image, persona, instruction, and chat template use "
            f"{input_token_count} tokens and fill the {context_limit}-token "
            "context window."
        )
    max_tokens = min(requested_max_tokens, profile_output_available)
    capacity = input_token_count + max_tokens
    model_context_limit = getattr(config, "max_position_embeddings", None)
    if isinstance(model_context_limit, int) and capacity > model_context_limit:
        raise PromptAssistantError(
            f"The active KREA2 model supports at most {model_context_limit} "
            "context tokens."
        )

    free_vram = int(memory_management.get_free_memory(device))
    reserved_vram = int(memory_management.extra_reserved_memory())
    usable_cache_vram = max(free_vram - reserved_vram, 0)
    vram_capacity_limit = int(usable_cache_vram / (kv_bytes_per_token * 1.10))
    vram_output_available = max(vram_capacity_limit - input_token_count, 0)
    available_output_tokens = min(profile_output_available, vram_output_available)
    vram_limited = available_output_tokens < profile_output_available
    if available_output_tokens < 1:
        raise PromptAssistantError(
            "After encoding the uploaded image, there is not enough free VRAM "
            "for the input KV cache. Select a smaller context profile or free "
            "more VRAM in Forge."
        )
    max_tokens = min(requested_max_tokens, available_output_tokens)
    capacity = input_token_count + max_tokens

    weight = getattr(embedding, "weight", None)
    if not isinstance(weight, torch.Tensor) or tuple(weight.shape) != expected_shape:
        raise PromptAssistantError(
            "The KREA2 embedding matrix does not have the expected LM projection shape."
        )
    if weight.device != device:
        raise PromptAssistantError(
            "Forge did not place the KREA2 embedding matrix on the execution "
            "device after the ModelPatcher call; the extension will not create "
            "a large helper copy."
        )

    past_key_values = _allocate_kv_cache(
        core_model, batch=1, capacity=capacity, device=device, dtype=dtype
    )
    generator = torch.Generator(device=device).manual_seed(seed) if do_sample else None
    stop_tokens = {
        int(token_id)
        for token_id in (
            getattr(tokenizer, "eos_token_id", None),
            151645,
        )
        if token_id is not None
    }

    output = _prefill_with_deepstack(
        core_model=core_model,
        embeds=embeds,
        attention_mask=attention_mask,
        position_ids=position_ids,
        visual_pos_masks=visual_pos_masks,
        deepstack=deepstack,
        past_key_values=past_key_values,
    )
    _raise_if_generation_cancelled(generation_id)
    next_position = int(position_ids.max().item()) + 1
    generated_ids: list[int] = []
    finish_reason = "max_tokens"

    for generation_index in range(max_tokens):
        if _generation_cancelled(generation_id):
            finish_reason = "cancelled"
            break
        if not isinstance(output, tuple) or len(output) != 3:
            raise PromptAssistantError(
                "The KREA2 Qwen core returned no compatible KV cache."
            )
        hidden_states, _, past_key_values = output
        logits = _tied_embedding_logits(hidden_states, embedding)
        next_token = _sample_token(
            logits=logits,
            history=generated_ids,
            do_sample=bool(do_sample),
            temperature=temperature,
            top_k=top_k,
            top_p=top_p,
            min_p=min_p,
            repetition_penalty=repetition_penalty,
            generator=generator,
        )
        token_id = int(next_token[0, 0].item())
        generated_ids.append(token_id)
        if token_id in stop_tokens:
            finish_reason = "eos"
            break
        if generation_index + 1 >= max_tokens:
            continue
        token_embeds = embedding(next_token).to(dtype=dtype)
        decode_position_ids = torch.full(
            (3, 1),
            next_position,
            device=device,
            dtype=torch.long,
        )
        output = core_model(
            None,
            embeds=token_embeds,
            attention_mask=None,
            position_ids=decode_position_ids,
            past_key_values=past_key_values,
        )
        next_position += 1

    generated_text = tokenizer.decode(generated_ids, skip_special_tokens=True).strip()
    if not generated_text:
        if finish_reason == "cancelled":
            raise GenerationCancelled("Generation cancelled by user.")
        raise PromptAssistantError(
            "The active Qwen3-VL vision encoder produced no visible prompt. "
            "Please try again."
        )
    return (
        generated_text,
        input_token_count,
        len(generated_ids),
        finish_reason,
        profile_tier,
        capacity,
        context_limit,
        requested_max_tokens,
        available_output_tokens,
        vram_limited,
        seed,
        visual_token_count,
        original_size,
        prepared_size,
    )


def _run_generation(
    idea,
    instruction,
    persona_name,
    system_prompt,
    vram_profile,
    max_tokens,
    do_sample,
    temperature,
    top_k,
    top_p,
    min_p,
    repetition_penalty,
    seed,
    generation_id,
):
    started = time.perf_counter()
    try:
        (
            text,
            input_count,
            output_count,
            finish_reason,
            profile_tier,
            reserved_context,
            context_limit,
            requested_max_tokens,
            available_output_tokens,
            vram_limited,
            used_seed,
        ) = _generate_with_active_krea(
            idea=idea,
            instruction=instruction,
            system_prompt=system_prompt,
            vram_profile=vram_profile,
            max_tokens=max_tokens,
            do_sample=do_sample,
            temperature=temperature,
            top_k=top_k,
            top_p=top_p,
            min_p=min_p,
            repetition_penalty=repetition_penalty,
            seed=seed,
            generation_id=generation_id,
        )
    except GenerationCancelled:
        LOGGER.info("KREA2 prompt generation cancelled by user")
        return "", "**Cancelled by user.**"
    except PromptAssistantError as exc:
        LOGGER.warning("KREA2 prompt generation rejected: %s", exc)
        return "", f"**Not executed:** {exc}"
    except torch.OutOfMemoryError:
        LOGGER.exception("CUDA OOM during KREA2 prompt generation")
        return "", (
            "**CUDA out of memory.** No second model was loaded. Forge can "
            "manage memory normally again on the next job."
        )
    except Exception as exc:
        LOGGER.exception("Unexpected KREA2 prompt-generation failure")
        return "", f"**Error:** {type(exc).__name__}: {exc}"
    finally:
        _clear_generation_request(generation_id)

    elapsed = time.perf_counter() - started
    persona_display = html.escape((persona_name or "Unsaved").strip())
    effective_max_tokens = reserved_context - input_count
    limit_note = (
        f"Output limit automatically reduced {requested_max_tokens} → "
        f"{effective_max_tokens} · "
        if effective_max_tokens < requested_max_tokens
        else f"Output limit: {effective_max_tokens} · "
    )
    vram_note = " · limited by free VRAM" if vram_limited else ""
    sampling_note = " (sampling disabled)" if not do_sample else ""
    return text, (
        f"**Active Qwen3-VL text encoder** · "
        f"Persona: **{persona_display}** · "
        f"Seed: **{used_seed}**{sampling_note} · "
        f"VRAM profile: **{profile_tier} GB** · "
        f"{input_count} input tokens · {output_count} output tokens · "
        f"available output window: {available_output_tokens}{vram_note} · "
        f"{limit_note}reserved context: {reserved_context}/{context_limit} · "
        f"finish: `{finish_reason}` · {elapsed:.1f} s"
    )


def _run_image_generation(
    image,
    instruction,
    persona_name,
    system_prompt,
    vram_profile,
    max_tokens,
    do_sample,
    temperature,
    top_k,
    top_p,
    min_p,
    repetition_penalty,
    seed,
    generation_id,
):
    started = time.perf_counter()
    try:
        (
            text,
            input_count,
            output_count,
            finish_reason,
            profile_tier,
            reserved_context,
            context_limit,
            requested_max_tokens,
            available_output_tokens,
            vram_limited,
            used_seed,
            visual_token_count,
            original_size,
            prepared_size,
        ) = _generate_with_active_krea_image(
            image=image,
            instruction=instruction,
            system_prompt=system_prompt,
            vram_profile=vram_profile,
            max_tokens=max_tokens,
            do_sample=do_sample,
            temperature=temperature,
            top_k=top_k,
            top_p=top_p,
            min_p=min_p,
            repetition_penalty=repetition_penalty,
            seed=seed,
            generation_id=generation_id,
        )
    except GenerationCancelled:
        LOGGER.info("KREA2 image-to-prompt generation cancelled by user")
        return "", "**Cancelled by user.**"
    except PromptAssistantError as exc:
        LOGGER.warning("KREA2 image-to-prompt generation rejected: %s", exc)
        return "", f"**Not executed:** {exc}"
    except torch.OutOfMemoryError:
        LOGGER.exception("CUDA OOM during KREA2 image-to-prompt generation")
        return "", (
            "**CUDA out of memory.** No second model was loaded. Forge can "
            "manage memory normally again on the next job."
        )
    except Exception as exc:
        LOGGER.exception("Unexpected KREA2 image-to-prompt failure")
        return "", f"**Error:** {type(exc).__name__}: {exc}"
    finally:
        _clear_generation_request(generation_id)

    elapsed = time.perf_counter() - started
    persona_display = html.escape((persona_name or "Unsaved").strip())
    effective_max_tokens = reserved_context - input_count
    limit_note = (
        f"Output limit automatically reduced {requested_max_tokens} → "
        f"{effective_max_tokens} · "
        if effective_max_tokens < requested_max_tokens
        else f"Output limit: {effective_max_tokens} · "
    )
    vram_note = " · limited by free VRAM" if vram_limited else ""
    sampling_note = " (sampling disabled)" if not do_sample else ""
    original_width, original_height = original_size
    prepared_width, prepared_height = prepared_size
    return text, (
        f"**Active Qwen3-VL vision encoder** · "
        f"Persona: **{persona_display}** · "
        f"Seed: **{used_seed}**{sampling_note} · "
        f"Image: {original_width}×{original_height} → "
        f"{prepared_width}×{prepared_height} · "
        f"{visual_token_count} visual tokens · "
        f"VRAM profile: **{profile_tier} GB** · "
        f"{input_count} total input tokens · {output_count} output tokens · "
        f"available output window: {available_output_tokens}{vram_note} · "
        f"{limit_note}reserved context: {reserved_context}/{context_limit} · "
        f"finish: `{finish_reason}` · {elapsed:.1f} s"
    )


def _transfer_prompt(generated: str, current: str, *, mode: str, target_name: str):
    generated = (generated or "").strip()
    current = (current or "").rstrip()
    if not generated:
        return current, "**Not sent:** Generate a prompt first."
    if mode == "append":
        updated = f"{current}\n{generated}" if current else generated
        action = "Appended to"
    else:
        updated = generated
        action = "Replaced"
    return updated, f"✅ {action} the **{target_name}** positive prompt."


def _start_action_button(label: str):
    return gr.update(value=f"⏳ {label}…", interactive=False)


def _finish_action_button(status, *, normal_label: str, success_label: str):
    status_text = str(status or "").strip()
    failed = not status_text or status_text.startswith(("**Not", "**Error"))
    label = normal_label if failed else f"✓ {success_label}"
    return gr.update(value=label, interactive=True)


def _start_prompt_transfer(target_name: str):
    return (
        gr.update(value="⏳ Sending…", interactive=False),
        f"⏳ Updating the **{target_name}** positive prompt…",
    )


def _reset_prompt_transfer_feedback(labels: tuple[str, ...]):
    return tuple(gr.update(value=label, interactive=True) for label in labels) + ("",)


def _connect_prompt_buttons(
    result,
    reset_trigger,
    txt_replace,
    txt_append,
    img_replace,
    img_append,
    feedback,
):
    try:
        from modules import ui

        txt_prompt = ui.txt2img_paste_fields[0][0]
        img_prompt = ui.img2img_paste_fields[0][0]
    except Exception:
        LOGGER.exception("Could not connect Forge positive-prompt fields")
        return

    actions = (
        (
            txt_replace,
            txt_prompt,
            "replace",
            "txt2img",
            "txt2img: replace",
            "txt2img replaced",
        ),
        (
            txt_append,
            txt_prompt,
            "append",
            "txt2img",
            "txt2img: append",
            "txt2img appended",
        ),
        (
            img_replace,
            img_prompt,
            "replace",
            "img2img",
            "img2img: replace",
            "img2img replaced",
        ),
        (
            img_append,
            img_prompt,
            "append",
            "img2img",
            "img2img: append",
            "img2img appended",
        ),
    )
    for button, target, mode, target_name, normal_label, success_label in actions:
        started = button.click(
            fn=partial(_start_prompt_transfer, target_name),
            inputs=[],
            outputs=[button, feedback],
            queue=False,
            show_progress="hidden",
        )
        transferred = started.then(
            fn=partial(_transfer_prompt, mode=mode, target_name=target_name),
            inputs=[result, target],
            outputs=[target, feedback],
            queue=False,
            show_progress="hidden",
        )
        transferred.then(
            fn=partial(
                _finish_action_button,
                normal_label=normal_label,
                success_label=success_label,
            ),
            inputs=[feedback],
            outputs=[button],
            queue=False,
            show_progress="hidden",
        )

    labels = tuple(action[4] for action in actions)
    reset_trigger.click(
        fn=partial(_reset_prompt_transfer_feedback, labels),
        inputs=[],
        outputs=[txt_replace, txt_append, img_replace, img_append, feedback],
        queue=False,
        show_progress="hidden",
    )
    result.change(
        fn=partial(_reset_prompt_transfer_feedback, labels),
        inputs=[],
        outputs=[txt_replace, txt_append, img_replace, img_append, feedback],
        queue=False,
        show_progress="hidden",
    )


def _create_persona_controls(
    persona_choices, selected_persona, selected_prompt, initial_status
):
    persona_select = gr.Dropdown(
        label="Persona",
        choices=persona_choices,
        value=selected_persona,
    )
    with gr.Accordion("Edit and manage personas", open=False):
        persona_name = gr.Textbox(
            label="Persona name",
            value=selected_persona,
        )
        persona_prompt = gr.Textbox(
            label="System prompt",
            value=selected_prompt,
            lines=14,
            max_lines=30,
        )
        with gr.Row():
            persona_save = gr.Button("Save persona", variant="primary")
            persona_new = gr.Button("New persona")
            persona_delete = gr.Button("Delete persona")
        persona_status = gr.Markdown(initial_status)
    return (
        persona_select,
        persona_name,
        persona_prompt,
        persona_save,
        persona_new,
        persona_delete,
        persona_status,
    )


def _create_sampling_controls(
    automatic_profile, vram_profile_choices, initial_context_limit
):
    with gr.Accordion("Sampling", open=False):
        vram_profile = gr.Dropdown(
            label="Context profile (optional smaller fallback)",
            choices=vram_profile_choices,
            value=automatic_profile,
        )
        do_sample = gr.Checkbox(label="Enable sampling", value=True)
        max_tokens = gr.Slider(
            label="Maximum output tokens",
            minimum=32,
            maximum=initial_context_limit,
            step=1,
            value=DEFAULT_MAX_OUTPUT_TOKENS,
        )
        temperature = gr.Slider(
            label="Temperature", minimum=0.01, maximum=2.0, step=0.01, value=0.7
        )
        top_k = gr.Slider(label="Top K", minimum=1, maximum=1000, step=1, value=64)
        top_p = gr.Slider(
            label="Top P", minimum=0.05, maximum=1.0, step=0.01, value=0.95
        )
        min_p = gr.Slider(
            label="Min P", minimum=0.0, maximum=1.0, step=0.01, value=0.05
        )
        repetition_penalty = gr.Slider(
            label="Repetition Penalty",
            minimum=0.1,
            maximum=5.0,
            step=0.01,
            value=1.05,
        )
        seed = gr.Number(label="Seed", value=0, precision=0)
    return (
        vram_profile,
        max_tokens,
        do_sample,
        temperature,
        top_k,
        top_p,
        min_p,
        repetition_penalty,
        seed,
    )


def _connect_persona_controls(controls, load_fn, save_fn, delete_fn):
    (
        persona_select,
        persona_name,
        persona_prompt,
        persona_save,
        persona_new,
        persona_delete,
        persona_status,
    ) = controls
    persona_select.change(
        fn=load_fn,
        inputs=[persona_select],
        outputs=[persona_name, persona_prompt, persona_status],
        show_progress="hidden",
    )
    save_started = persona_save.click(
        fn=partial(_start_action_button, "Saving"),
        inputs=[],
        outputs=[persona_save],
        queue=False,
        show_progress="hidden",
    )
    save_event = save_started.then(
        fn=save_fn,
        inputs=[persona_name, persona_prompt],
        outputs=[persona_select, persona_name, persona_prompt, persona_status],
        show_progress="hidden",
    )
    save_event.then(
        fn=partial(
            _finish_action_button,
            normal_label="Save persona",
            success_label="Persona saved",
        ),
        inputs=[persona_status],
        outputs=[persona_save],
        queue=False,
        show_progress="hidden",
    )

    new_started = persona_new.click(
        fn=partial(_start_action_button, "Preparing"),
        inputs=[],
        outputs=[persona_new],
        queue=False,
        show_progress="hidden",
    )
    new_event = new_started.then(
        fn=_new_persona_fields,
        inputs=[],
        outputs=[persona_select, persona_name, persona_prompt, persona_status],
        show_progress="hidden",
    )
    new_event.then(
        fn=partial(
            _finish_action_button,
            normal_label="New persona",
            success_label="New persona ready",
        ),
        inputs=[persona_status],
        outputs=[persona_new],
        queue=False,
        show_progress="hidden",
    )

    delete_started = persona_delete.click(
        fn=partial(_start_action_button, "Deleting"),
        inputs=[],
        outputs=[persona_delete],
        queue=False,
        show_progress="hidden",
    )
    delete_event = delete_started.then(
        fn=delete_fn,
        inputs=[persona_select],
        outputs=[persona_select, persona_name, persona_prompt, persona_status],
        show_progress="hidden",
    )
    delete_event.then(
        fn=partial(
            _finish_action_button,
            normal_label="Delete persona",
            success_label="Persona deleted",
        ),
        inputs=[persona_status],
        outputs=[persona_delete],
        queue=False,
        show_progress="hidden",
    )


def _start_text_generation():
    generation_id = _begin_generation_request()
    return (
        "### ⏳ The Qwen3-VL text encoder is generating the prompt…",
        gr.update(value="⏳ Generating…", interactive=False),
        gr.update(value="Stop", interactive=True),
        generation_id,
    )


def _start_model_loading():
    return (
        "### ⏳ Forge is loading the current model selection…",
        gr.update(value="⏳ Loading current selection…", interactive=False),
    )


def _start_image_generation():
    generation_id = _begin_generation_request()
    return (
        "### ⏳ The Qwen3-VL vision encoder is analyzing the image and generating the prompt…",
        gr.update(value="⏳ Analyzing image…", interactive=False),
        gr.update(value="Stop", interactive=True),
        generation_id,
    )


def _finish_text_generation():
    return (
        gr.update(value="Generate prompt", interactive=True),
        gr.update(value="Stop", interactive=False),
    )


def _finish_model_loading():
    return gr.update(value="Load current Forge selection", interactive=True)


def _finish_image_generation():
    return (
        gr.update(value="Generate prompt from image", interactive=True),
        gr.update(value="Stop", interactive=False),
    )


def _on_ui_tabs():
    stack_ready, initial_loader_status = _forge_stack_status()

    persona_choices, selected_persona, selected_prompt, initial_persona_status = (
        _initial_persona_state()
    )
    (
        image_persona_choices,
        selected_image_persona,
        selected_image_prompt,
        initial_image_persona_status,
    ) = _initial_image_persona_state()
    detected_vram_gb = _detected_vram_gb()
    automatic_tier = _automatic_vram_tier(detected_vram_gb)
    automatic_profile = (
        f"{AUTO_VRAM_PROFILE} ({detected_vram_gb:.1f} GB → {automatic_tier} GB)"
    )
    available_manual_tiers = [
        tier
        for tier in VRAM_PROFILES
        if tier <= max(automatic_tier, int(round(detected_vram_gb)))
    ]
    vram_profile_choices = [automatic_profile] + [
        f"{tier} GB" for tier in available_manual_tiers
    ]
    initial_context_limit = VRAM_PROFILES[automatic_tier]
    with gr.Blocks(analytics_enabled=False) as tab:
        with gr.Row(equal_height=True):
            with gr.Column(scale=4, min_width=320):
                gr.Markdown(f"### KREA2 Prompt Assistant — Alpha {EXTENSION_VERSION}")
                loader_status = gr.Markdown(initial_loader_status)
            with gr.Column(scale=1, min_width=260):
                load_forge_stack = gr.Button(
                    "Load current Forge selection",
                    variant="primary" if not stack_ready else "secondary",
                    size="lg",
                )
        gr.Markdown(
            "Create KREA2 prompts from an idea or image with the active Qwen3-VL "
            "encoder in Forge. The selected persona is the complete system prompt."
        )

        with gr.Tabs():
            with gr.Tab("Idea to prompt"):
                text_persona_controls = _create_persona_controls(
                    persona_choices,
                    selected_persona,
                    selected_prompt,
                    initial_persona_status,
                )
                persona_name = text_persona_controls[1]
                persona_prompt = text_persona_controls[2]
                idea = gr.Textbox(
                    label="Idea",
                    placeholder="woman repairing an engine inside an old spaceship",
                    lines=5,
                    max_lines=12,
                )
                text_instruction = gr.Textbox(
                    label="Optional instruction",
                    placeholder=(
                        "For example: Use a wide cinematic composition and "
                        "emphasize warm practical lighting."
                    ),
                    lines=3,
                    max_lines=8,
                )
                vram_status = gr.Markdown(_vram_profile_status(automatic_profile))
                with gr.Row():
                    generate = gr.Button("Generate prompt", variant="primary", scale=4)
                    stop_generation = gr.Button(
                        "Stop", variant="stop", interactive=False, scale=1
                    )
                text_generation_id = gr.State("")
                result = gr.Textbox(
                    label="Generated KREA2 prompt",
                    lines=12,
                    max_lines=24,
                    show_copy_button=True,
                )
                status = gr.Markdown()
                with gr.Row():
                    txt_replace = gr.Button("txt2img: replace")
                    txt_append = gr.Button("txt2img: append")
                with gr.Row():
                    img_replace = gr.Button("img2img: replace")
                    img_append = gr.Button("img2img: append")
                prompt_action_status = gr.State("")
                text_sampling_controls = _create_sampling_controls(
                    automatic_profile,
                    vram_profile_choices,
                    initial_context_limit,
                )

            with gr.Tab("Image to prompt"):
                image_persona_controls = _create_persona_controls(
                    image_persona_choices,
                    selected_image_persona,
                    selected_image_prompt,
                    initial_image_persona_status,
                )
                image_persona_name = image_persona_controls[1]
                image_persona_prompt = image_persona_controls[2]
                gr.Markdown(
                    "The uploaded image is processed locally and is not stored "
                    "by the extension. The optional instruction is the complete "
                    "user request sent alongside the image."
                )
                source_image = gr.Image(
                    label="Reference image",
                    source="upload",
                    type="pil",
                    image_mode="RGB",
                    height=420,
                )
                image_instruction = gr.Textbox(
                    label="Optional instruction",
                    placeholder=DEFAULT_IMAGE_REQUEST,
                    lines=3,
                    max_lines=8,
                )
                image_vram_status = gr.Markdown(_vram_profile_status(automatic_profile))
                with gr.Row():
                    image_generate = gr.Button(
                        "Generate prompt from image", variant="primary", scale=4
                    )
                    image_stop_generation = gr.Button(
                        "Stop", variant="stop", interactive=False, scale=1
                    )
                image_generation_id = gr.State("")
                image_result = gr.Textbox(
                    label="Generated KREA2 prompt",
                    lines=12,
                    max_lines=24,
                    show_copy_button=True,
                )
                image_status = gr.Markdown()
                with gr.Row():
                    image_txt_replace = gr.Button("txt2img: replace")
                    image_txt_append = gr.Button("txt2img: append")
                with gr.Row():
                    image_img_replace = gr.Button("img2img: replace")
                    image_img_append = gr.Button("img2img: append")
                image_prompt_action_status = gr.State("")
                image_sampling_controls = _create_sampling_controls(
                    automatic_profile,
                    vram_profile_choices,
                    initial_context_limit,
                )

        (
            vram_profile,
            max_tokens,
            do_sample,
            temperature,
            top_k,
            top_p,
            min_p,
            repetition_penalty,
            seed,
        ) = text_sampling_controls
        model_load_event = load_forge_stack.click(
            fn=_start_model_loading,
            inputs=[],
            outputs=[loader_status, load_forge_stack],
            queue=False,
            show_progress="hidden",
        ).then(
            fn=call_queue.wrap_queued_call(_load_current_forge_selection),
            inputs=[],
            outputs=[loader_status],
            show_progress="minimal",
        )
        model_load_event.then(
            fn=_finish_model_loading,
            inputs=[],
            outputs=[load_forge_stack],
            queue=False,
            show_progress="hidden",
        )

        text_generation_event = generate.click(
            fn=_start_text_generation,
            inputs=[],
            outputs=[status, generate, stop_generation, text_generation_id],
            queue=False,
            show_progress="hidden",
        ).then(
            fn=call_queue.wrap_queued_call(_run_generation),
            inputs=[
                idea,
                text_instruction,
                persona_name,
                persona_prompt,
                vram_profile,
                max_tokens,
                do_sample,
                temperature,
                top_k,
                top_p,
                min_p,
                repetition_penalty,
                seed,
                text_generation_id,
            ],
            outputs=[result, status],
            show_progress="minimal",
        )
        text_generation_event.then(
            fn=_finish_text_generation,
            inputs=[],
            outputs=[generate, stop_generation],
            queue=False,
            show_progress="hidden",
        )
        stop_generation.click(
            fn=_request_generation_stop,
            inputs=[text_generation_id],
            outputs=[status, stop_generation],
            queue=False,
            show_progress="hidden",
        )
        vram_profile.change(
            fn=_change_vram_profile,
            inputs=[vram_profile, max_tokens],
            outputs=[max_tokens, vram_status],
            show_progress="hidden",
        )
        _connect_persona_controls(
            text_persona_controls,
            _load_persona_fields,
            _save_persona_fields,
            _delete_persona_fields,
        )
        _connect_prompt_buttons(
            result,
            generate,
            txt_replace,
            txt_append,
            img_replace,
            img_append,
            prompt_action_status,
        )

        (
            image_vram_profile,
            image_max_tokens,
            image_do_sample,
            image_temperature,
            image_top_k,
            image_top_p,
            image_min_p,
            image_repetition_penalty,
            image_seed,
        ) = image_sampling_controls
        image_generation_event = image_generate.click(
            fn=_start_image_generation,
            inputs=[],
            outputs=[
                image_status,
                image_generate,
                image_stop_generation,
                image_generation_id,
            ],
            queue=False,
            show_progress="hidden",
        ).then(
            fn=call_queue.wrap_queued_call(_run_image_generation),
            inputs=[
                source_image,
                image_instruction,
                image_persona_name,
                image_persona_prompt,
                image_vram_profile,
                image_max_tokens,
                image_do_sample,
                image_temperature,
                image_top_k,
                image_top_p,
                image_min_p,
                image_repetition_penalty,
                image_seed,
                image_generation_id,
            ],
            outputs=[image_result, image_status],
            show_progress="minimal",
        )
        image_generation_event.then(
            fn=_finish_image_generation,
            inputs=[],
            outputs=[image_generate, image_stop_generation],
            queue=False,
            show_progress="hidden",
        )
        image_stop_generation.click(
            fn=_request_generation_stop,
            inputs=[image_generation_id],
            outputs=[image_status, image_stop_generation],
            queue=False,
            show_progress="hidden",
        )
        image_vram_profile.change(
            fn=_change_vram_profile,
            inputs=[image_vram_profile, image_max_tokens],
            outputs=[image_max_tokens, image_vram_status],
            show_progress="hidden",
        )
        _connect_persona_controls(
            image_persona_controls,
            _load_image_persona_fields,
            _save_image_persona_fields,
            _delete_image_persona_fields,
        )
        _connect_prompt_buttons(
            image_result,
            image_generate,
            image_txt_replace,
            image_txt_append,
            image_img_replace,
            image_img_append,
            image_prompt_action_status,
        )

    return [(tab, "KREA2 Prompt Assistant", "forge_krea_prompt_assistant")]


script_callbacks.on_ui_tabs(_on_ui_tabs, name="forge_krea_prompt_assistant.ui")
