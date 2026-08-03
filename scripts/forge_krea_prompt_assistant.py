"""KREA2 prompt generation by reusing Forge Neo's active Qwen3-VL encoder.

Copyright (C) 2026 vibecodingtoolmaker
SPDX-License-Identifier: AGPL-3.0-only

This extension never loads a language model. It only operates on the KREA2
text encoder and tokenizer already owned by the active Forge diffusion engine.
"""

from __future__ import annotations

import html
import json
import logging
import os
import threading
import time
from pathlib import Path

import gradio as gr
import torch
import torch.nn.functional as F

from backend import memory_management
from modules import call_queue, script_callbacks, shared


LOGGER = logging.getLogger("forge_krea_prompt_assistant")
EXTENSION_VERSION = "0.3.0-alpha.1"
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
PERSONA_LOCK = threading.RLock()
CHAT_SYSTEM_START = "<|im_start|>system\n"
CHAT_USER_BOUNDARY = "<|im_end|>\n<|im_start|>user\n"

DEFAULT_PERSONA_PROMPT = """You are a prompt-writing assistant for the KREA2 image-generation model. Rewrite the user's image idea as one polished image prompt.

Requirements:
- Return only the finished prompt, with no introduction, notes, headings, quotation marks, or alternatives.
- Write coherent natural-language prose, not a comma-separated tag list.
- Preserve every explicit subject, action, setting, style, and constraint from the idea.
- Add useful concrete visual detail without changing the concept or inventing extra characters.
- Organize the description around subject, action, environment, color, shape, size, texture, quantity, visible text, spatial relationships, composition, lighting, materials, and camera perspective when relevant.
- Avoid empty quality slogans and avoid repeating details.
- Write the finished prompt in English."""


class PromptAssistantError(RuntimeError):
    """Expected, user-facing validation or compatibility failure."""


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


def _default_persona_store() -> dict:
    return {
        "version": PERSONA_STORE_VERSION,
        "personas": {DEFAULT_PERSONA_NAME: DEFAULT_PERSONA_PROMPT},
    }


def _read_persona_store() -> dict:
    with PERSONA_LOCK:
        if not PERSONAS_PATH.exists():
            return _default_persona_store()
        try:
            raw = json.loads(PERSONAS_PATH.read_text(encoding="utf-8"))
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
                DEFAULT_PERSONA_NAME: DEFAULT_PERSONA_PROMPT,
                **personas,
            }
        return {"version": PERSONA_STORE_VERSION, "personas": personas}


def _write_persona_store(personas: dict[str, str]) -> None:
    if len(personas) > MAX_PERSONAS:
        raise PromptAssistantError(
            f"At most {MAX_PERSONAS} personas can be stored."
        )
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
    temporary_path = PERSONAS_PATH.with_name(PERSONAS_PATH.name + ".tmp")
    with PERSONA_LOCK:
        try:
            temporary_path.write_text(serialized, encoding="utf-8", newline="\n")
            os.replace(temporary_path, PERSONAS_PATH)
        except OSError as exc:
            raise PromptAssistantError(
                f"The persona file could not be saved: {exc}"
            ) from exc


def _initial_persona_state() -> tuple[list[str], str, str, str]:
    try:
        store = _read_persona_store()
        personas = store["personas"]
        selected = DEFAULT_PERSONA_NAME
        return list(personas), selected, personas[selected], ""
    except PromptAssistantError as exc:
        LOGGER.exception("Could not initialize persona store")
        return (
            [DEFAULT_PERSONA_NAME],
            DEFAULT_PERSONA_NAME,
            DEFAULT_PERSONA_PROMPT,
            f"**Persona file error:** {exc}",
        )


def _load_persona_fields(name):
    try:
        name = _validate_persona_name(name)
        personas = _read_persona_store()["personas"]
        if name not in personas:
            raise PromptAssistantError(f"Persona not found: {name}")
        return name, personas[name], f"Persona **{html.escape(name)}** loaded."
    except PromptAssistantError as exc:
        return "", "", f"**Persona error:** {exc}"


def _save_persona_fields(name, system_prompt):
    try:
        name = _validate_persona_name(name)
        system_prompt = _validate_system_prompt(system_prompt)
        with PERSONA_LOCK:
            personas = _read_persona_store()["personas"]
            if name not in personas and len(personas) >= MAX_PERSONAS:
                raise PromptAssistantError(
                    f"At most {MAX_PERSONAS} personas can be stored."
                )
            personas[name] = system_prompt
            _write_persona_store(personas)
        return (
            gr.update(choices=list(personas), value=name),
            name,
            system_prompt,
            f"Persona **{html.escape(name)}** saved.",
        )
    except PromptAssistantError as exc:
        return gr.update(), name, system_prompt, f"**Not saved:** {exc}"


def _new_persona_fields():
    return (
        gr.update(value=None),
        "",
        "",
        "New persona: enter a name and system prompt, then save it.",
    )


def _delete_persona_fields(name):
    try:
        name = _validate_persona_name(name)
        if name == DEFAULT_PERSONA_NAME:
            raise PromptAssistantError(
                "The Default persona can be edited but not deleted."
            )
        with PERSONA_LOCK:
            personas = _read_persona_store()["personas"]
            if name not in personas:
                raise PromptAssistantError(f"Persona not found: {name}")
            del personas[name]
            _write_persona_store(personas)
        selected = DEFAULT_PERSONA_NAME
        return (
            gr.update(choices=list(personas), value=selected),
            selected,
            personas[selected],
            f"Persona **{html.escape(name)}** deleted; Default is active again.",
        )
    except PromptAssistantError as exc:
        return gr.update(), name, gr.update(), f"**Not deleted:** {exc}"


def _class_path(value) -> str:
    cls = type(value)
    return f"{cls.__module__}.{cls.__qualname__}"


def _active_krea_components():
    """Resolve, but never retain, the active KREA2 runtime components."""
    sd_model = shared.sd_model
    if sd_model is None:
        raise PromptAssistantError("No diffusion model is loaded.")

    if _class_path(sd_model) != "backend.diffusion_engine.krea.Krea2":
        raise PromptAssistantError(
            "This alpha currently supports only an active KREA2 model. "
            f"Active model: {_class_path(sd_model)}"
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
            "The expected KREA2 encoder/tokenizer objects were not found."
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


def _render_generation_prompt(engine, idea: str, system_prompt: str) -> str:
    template = getattr(engine, "llama_template", None)
    if not isinstance(template, str) or template.count("{}") != 1:
        raise PromptAssistantError("The KREA2 chat template is incompatible.")
    if template.count(CHAT_SYSTEM_START) != 1 or template.count(CHAT_USER_BOUNDARY) != 1:
        raise PromptAssistantError(
            "The role structure of the KREA2 chat template is incompatible."
        )

    system_prompt = _validate_system_prompt(system_prompt)
    rendered = template.format(idea.strip())
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


def _execution_dtype(core_model, device: torch.device) -> torch.dtype:
    try:
        dtype = core_model.layers[0].self_attn.q_proj.weight.dtype
    except Exception:
        dtype = None
    if dtype in (torch.float16, torch.bfloat16):
        return dtype
    return (
        torch.bfloat16
        if memory_management.should_use_bf16(device)
        else torch.float32
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
        cumulative_probs = torch.cumsum(
            torch.softmax(sorted_logits, dim=-1), dim=-1
        )
        remove_sorted = cumulative_probs > top_p
        remove_sorted[..., 0] = False
        remove = torch.zeros_like(candidate_logits, dtype=torch.bool)
        remove.scatter_(1, sorted_indices, remove_sorted)
        candidate_logits.masked_fill_(
            remove, torch.finfo(candidate_logits.dtype).min
        )

    probabilities = torch.softmax(candidate_logits, dim=-1)
    sampled_index = torch.multinomial(
        probabilities, num_samples=1, generator=generator
    )
    return candidate_ids.gather(1, sampled_index)


@torch.inference_mode()
def _generate_with_active_krea(
    idea: str,
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
) -> tuple[str, int, int, str, int, int, int, int, int, bool, int]:
    if not isinstance(idea, str) or not idea.strip():
        raise PromptAssistantError("Please enter an image idea first.")

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
    rendered_prompt = _render_generation_prompt(engine, idea, system_prompt)
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
            f"The persona, image idea, and chat template use {input_token_count} "
            f"tokens and fill the {context_limit}-token context window. Shorten "
            "the idea or persona."
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
    generator = (
        torch.Generator(device=device).manual_seed(seed) if do_sample else None
    )
    stop_tokens = {
        int(token_id)
        for token_id in (
            getattr(tokenizer, "eos_token_id", None),
            151645,  # <|im_end|> in the KREA2 Qwen tokenizer
        )
        if token_id is not None
    }

    input_ids = torch.tensor(
        [prompt_token_ids], device=device, dtype=torch.long
    )
    embeds = embedding(input_ids).to(dtype=dtype)
    generated_ids: list[int] = []
    finish_reason = "max_tokens"

    for _ in range(max_tokens):
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

    generated_text = tokenizer.decode(
        generated_ids, skip_special_tokens=True
    ).strip()
    if not generated_text:
        raise PromptAssistantError(
            "KREA2 produced no visible prompt. Please try again."
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


def _run_generation(
    idea,
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
        )
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
        f"**KREA2 / active Qwen3-VL encoder** · "
        f"Persona: **{persona_display}** · "
        f"Seed: **{used_seed}**{sampling_note} · "
        f"VRAM profile: **{profile_tier} GB** · "
        f"{input_count} input tokens · {output_count} output tokens · "
        f"available output window: {available_output_tokens}{vram_note} · "
        f"{limit_note}reserved context: {reserved_context}/{context_limit} · "
        f"finish: `{finish_reason}` · {elapsed:.1f} s"
    )


def _replace_prompt(generated: str) -> str:
    return (generated or "").strip()


def _append_prompt(generated: str, current: str) -> str:
    generated = (generated or "").strip()
    current = (current or "").rstrip()
    if not generated:
        return current
    return f"{current}\n{generated}" if current else generated


def _connect_prompt_buttons(result, txt_replace, txt_append, img_replace, img_append):
    try:
        from modules import ui

        txt_prompt = ui.txt2img_paste_fields[0][0]
        img_prompt = ui.img2img_paste_fields[0][0]
    except Exception:
        LOGGER.exception("Could not connect Forge positive-prompt fields")
        return

    txt_replace.click(
        fn=_replace_prompt, inputs=[result], outputs=[txt_prompt]
    )
    txt_append.click(
        fn=_append_prompt,
        inputs=[result, txt_prompt],
        outputs=[txt_prompt],
    )
    img_replace.click(
        fn=_replace_prompt, inputs=[result], outputs=[img_prompt]
    )
    img_append.click(
        fn=_append_prompt,
        inputs=[result, img_prompt],
        outputs=[img_prompt],
    )


def _on_ui_tabs():
    persona_choices, selected_persona, selected_prompt, initial_persona_status = (
        _initial_persona_state()
    )
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
        gr.Markdown(
            f"""
### KREA2 Prompt Assistant — Alpha {EXTENSION_VERSION}

Generates natural-language image prompts with the **KREA2 Qwen3-VL encoder
already loaded by Forge**. It does not load or retain a second language model.

The selected persona text is the complete system prompt. The extension adds no
hidden style or content instructions.
"""
        )
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
            persona_status = gr.Markdown(initial_persona_status)

        idea = gr.Textbox(
            label="Image idea",
            placeholder="woman repairing an engine inside an old spaceship",
            lines=5,
            max_lines=12,
        )
        vram_status = gr.Markdown(_vram_profile_status(automatic_profile))
        generate = gr.Button("Generate prompt", variant="primary")
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
            top_k = gr.Slider(
                label="Top K", minimum=1, maximum=1000, step=1, value=64
            )
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

        generate.click(
            fn=call_queue.wrap_queued_call(_run_generation),
            inputs=[
                idea,
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
            ],
            outputs=[result, status],
            show_progress="minimal",
        )
        vram_profile.change(
            fn=_change_vram_profile,
            inputs=[vram_profile, max_tokens],
            outputs=[max_tokens, vram_status],
            show_progress="hidden",
        )
        persona_select.change(
            fn=_load_persona_fields,
            inputs=[persona_select],
            outputs=[persona_name, persona_prompt, persona_status],
            show_progress="hidden",
        )
        persona_save.click(
            fn=_save_persona_fields,
            inputs=[persona_name, persona_prompt],
            outputs=[persona_select, persona_name, persona_prompt, persona_status],
            show_progress="hidden",
        )
        persona_new.click(
            fn=_new_persona_fields,
            inputs=[],
            outputs=[persona_select, persona_name, persona_prompt, persona_status],
            show_progress="hidden",
        )
        persona_delete.click(
            fn=_delete_persona_fields,
            inputs=[persona_select],
            outputs=[persona_select, persona_name, persona_prompt, persona_status],
            show_progress="hidden",
        )
        _connect_prompt_buttons(
            result, txt_replace, txt_append, img_replace, img_append
        )

    return [(tab, "KREA2 Prompt Assistant", "forge_krea_prompt_assistant")]


script_callbacks.on_ui_tabs(
    _on_ui_tabs, name="forge_krea_prompt_assistant.ui"
)
