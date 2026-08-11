"""KREA2/Qwen3-VL model-family detection and capability adapter."""

# Copyright (C) 2026 vibecodingtoolmaker
# SPDX-License-Identifier: AGPL-3.0-only

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable

from .base import AdapterError, AdapterProbe, ModelIdentity
from ..capabilities import (
    ModelCapabilities,
    PromptDialect,
    ReasoningMode,
    Workflow,
)
from .qwen import (
    decode_generated_text,
    repetition_loop_suffix,
    strip_model_thinking,
)


KREA2_MODEL_CLASS = "backend.diffusion_engine.krea.Krea2"
KREA2_ENCODER_MODULE = "backend.nn.llm.llama"
KREA2_ENCODER_CLASS = "Qwen3VL"
CHAT_SYSTEM_START = "<|im_start|>system\n"
CHAT_USER_BOUNDARY = "<|im_end|>\n<|im_start|>user\n"


@dataclass(frozen=True)
class Krea2Components:
    """Request-local Forge-owned objects required by the KREA2 runtime."""

    sd_model: Any
    clip: Any
    encoder: Any
    tokenizer: Any
    engine: Any
    patcher: Any


class Krea2Adapter:
    """The first explicit Forgy model-family adapter."""

    adapter_id = "krea2-qwen3-vl"
    capabilities = ModelCapabilities(
        workflows=frozenset(
            {
                Workflow.FORGY_CHAT,
                Workflow.IDEA_TO_PROMPT,
                Workflow.IMAGE_TO_PROMPT,
                Workflow.REFINE_PROMPT,
            }
        ),
        prompt_dialect=PromptDialect.NATURAL_LANGUAGE,
        text_generation=True,
        vision_input=True,
        multi_turn_chat=True,
        system_prompt=True,
        reasoning_mode=ReasoningMode.CONTROLLABLE,
        sampling_controls=frozenset(
            {
                "max_tokens",
                "temperature",
                "top_k",
                "top_p",
                "min_p",
                "repetition_penalty",
                "seed",
            }
        ),
        default_persona_family="krea2-natural-language",
    )

    def __init__(self) -> None:
        self._text_generator: Callable[..., Any] | None = None
        self._image_generator: Callable[..., Any] | None = None

    @staticmethod
    def _class_path(value: Any) -> str:
        cls = type(value)
        return f"{cls.__module__}.{cls.__qualname__}"

    def probe(self, sd_model: Any) -> AdapterProbe:
        class_path = self._class_path(sd_model)
        return AdapterProbe(
            matched=class_path == KREA2_MODEL_CLASS,
            evidence=(f"diffusion_model={class_path}",),
        )

    def resolve(
        self, sd_model: Any, probe: AdapterProbe
    ) -> tuple[ModelIdentity, Krea2Components]:
        if not probe.matched:
            raise AdapterError("KREA2 adapter received a non-KREA2 model stack.")

        forge_objects = getattr(sd_model, "forge_objects", None)
        clip = getattr(forge_objects, "clip", None)
        cond_stage_model = getattr(clip, "cond_stage_model", None)
        tokenizer_container = getattr(clip, "tokenizer", None)
        encoder = getattr(cond_stage_model, "qwen3vl_4b", None)
        tokenizer = getattr(tokenizer_container, "qwen3vl_4b", None)
        engine = getattr(sd_model, "text_processing_engine_qwen", None)

        if clip is None or encoder is None or tokenizer is None or engine is None:
            raise AdapterError(
                "The expected text encoder and tokenizer objects for the active "
                "model adapter were not found. Select a complete supported stack "
                "in Forge, then use 'Load current Forge selection' above."
            )
        if getattr(engine, "text_encoder", None) is not encoder:
            raise AdapterError(
                "Forge is using unexpectedly different text encoder objects."
            )
        if getattr(engine, "tokenizer", None) is not tokenizer:
            raise AdapterError(
                "Forge is using unexpectedly different tokenizer objects."
            )
        if (
            type(encoder).__module__ != KREA2_ENCODER_MODULE
            or type(encoder).__name__ != KREA2_ENCODER_CLASS
        ):
            raise AdapterError(
                "The active text encoder class is not supported by this adapter: "
                f"{self._class_path(encoder)}"
            )

        patcher = getattr(clip, "patcher", None)
        if patcher is None:
            raise AdapterError("The text encoder's Forge ModelPatcher is missing.")

        evidence = probe.evidence + (
            f"text_encoder={self._class_path(encoder)}",
            f"tokenizer={self._class_path(tokenizer)}",
        )
        identity = ModelIdentity(
            adapter_id=self.adapter_id,
            family="krea2",
            variant="qwen3-vl-4b",
            display_name="KREA2 · Qwen3-VL 4B",
            evidence=evidence,
        )
        return identity, Krea2Components(
            sd_model=sd_model,
            clip=clip,
            encoder=encoder,
            tokenizer=tokenizer,
            engine=engine,
            patcher=patcher,
        )

    def bind_runtime(
        self,
        *,
        text_generator: Callable[..., Any],
        image_generator: Callable[..., Any],
    ) -> None:
        """Attach implementations without retaining any live model components."""

        self._text_generator = text_generator
        self._image_generator = image_generator

    def generate_text(self, **kwargs):
        if self._text_generator is None:
            raise AdapterError("The KREA2 text runtime has not been registered.")
        return self._text_generator(**kwargs)

    def generate_image(self, **kwargs):
        if self._image_generator is None:
            raise AdapterError("The KREA2 vision runtime has not been registered.")
        return self._image_generator(**kwargs)

    @staticmethod
    def render_generation_prompt(
        engine,
        idea: str,
        instruction: str,
        system_prompt: str,
        *,
        workflow: Workflow = Workflow.IDEA_TO_PROMPT,
        validate_system_prompt: Callable[..., str],
    ) -> str:
        """Apply the exact Qwen3-VL chat template used by the KREA2 adapter."""

        del workflow  # KREA2 keeps the behavior validated by its burn test.

        template = getattr(engine, "llama_template", None)
        if not isinstance(template, str) or template.count("{}") != 1:
            raise AdapterError("The active chat template is incompatible.")
        if (
            template.count(CHAT_SYSTEM_START) != 1
            or template.count(CHAT_USER_BOUNDARY) != 1
        ):
            raise AdapterError(
                "The role structure of the active chat template is incompatible."
            )

        system_prompt = validate_system_prompt(system_prompt, allow_empty=True)
        user_message = idea.strip()
        instruction = str(instruction or "").strip()
        if instruction:
            user_message += f"\n\nAdditional instruction:\n{instruction}"
        rendered = template.format(user_message)
        system_start = rendered.index(CHAT_SYSTEM_START) + len(CHAT_SYSTEM_START)
        system_end = rendered.index(CHAT_USER_BOUNDARY, system_start)
        rendered = rendered[:system_start] + system_prompt + rendered[system_end:]
        return rendered + "<think>\n\n</think>\n\n"

    def render_image_generation_prompt(
        self,
        engine,
        instruction: str,
        system_prompt: str,
        *,
        default_image_request: str,
        validate_system_prompt: Callable[..., str],
    ) -> str:
        """Insert the KREA2 vision marker into the validated user request."""

        vision_block = getattr(engine, "vision_block", None)
        if not isinstance(vision_block, str) or not vision_block:
            raise AdapterError("The active image chat template is unavailable.")
        user_message = (
            instruction.strip()
            if isinstance(instruction, str) and instruction.strip()
            else default_image_request
        )
        return self.render_generation_prompt(
            engine,
            f"{vision_block}\n{user_message}",
            "",
            system_prompt,
            workflow=Workflow.IMAGE_TO_PROMPT,
            validate_system_prompt=validate_system_prompt,
        )

    @staticmethod
    def decode_generated_text(tokenizer, generated_ids) -> str:
        """Decode visibly while preserving reasoning markers for later filtering."""

        return decode_generated_text(tokenizer, generated_ids)

    @staticmethod
    def strip_model_thinking(text: str) -> tuple[str, bool]:
        """Remove any visible Qwen reasoning before returning model output."""

        return strip_model_thinking(text)

    @staticmethod
    def repetition_loop_suffix(token_ids: list[int]) -> tuple[int, int] | None:
        """Detect a repeated Qwen output suffix before it consumes the context."""

        return repetition_loop_suffix(token_ids)
