"""Z-Image Base/Turbo adapter for Forge's Qwen3-4B prompt backend."""

# Copyright (C) 2026 vibecodingtoolmaker
# SPDX-License-Identifier: AGPL-3.0-only

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable

from .base import AdapterError, AdapterProbe, ModelIdentity
from .qwen import (
    decode_generated_text,
    repetition_loop_suffix,
    strip_model_thinking,
)
from ..capabilities import (
    ModelCapabilities,
    PromptDialect,
    ReasoningMode,
    Workflow,
)


ZIMAGE_MODEL_CLASS = "backend.diffusion_engine.zimage.ZImage"
ZIMAGE_ENGINE_CLASS = "backend.text_processing.qwen3_engine.Qwen3TextProcessingEngine"
ZIMAGE_ENCODER_MODULE = "backend.nn.llm.llama"
ZIMAGE_ENCODER_CLASS = "Qwen3_4B"
ZIMAGE_HIDDEN_SIZE = 2560
ZIMAGE_VOCAB_SIZE = 151936
QWEN_SYSTEM_START = "<|im_start|>system\n"
QWEN_MESSAGE_END = "<|im_end|>\n"
ZIMAGE_WORKFLOW_CONTRACTS = {
    Workflow.IDEA_TO_PROMPT: """Workflow contract:
- Treat the user's image idea as the intended visual concept, not as a question to answer or prose to correct.
- Preserve every explicit subject, action, setting, style, relationship, and constraint. Do not silently replace an unclear detail with a different subject or story.
- Add only visually useful detail that supports the same concept; do not invent extra characters or unrelated objects.
- Follow the selected system persona for style and return exactly one finished image-generation prompt with no introduction, heading, quotation marks, notes, or alternatives.""",
    Workflow.REFINE_PROMPT: """Workflow contract:
- Revise the existing image prompt only according to the refinement instruction.
- Preserve every unaffected subject, action, setting, style, composition, relationship, and constraint. Do not silently replace ambiguous wording or invent unrelated content.
- Follow the selected system persona for style and return exactly one complete revised image-generation prompt with no introduction, heading, quotation marks, notes, or alternatives.""",
    Workflow.FORGY_CHAT: """Workflow contract:
- Treat the latest user message as a request about the working image prompt, not as prose to correct or a question that replaces the user's concept.
- Preserve explicit subjects, actions, relationships, and constraints unless the user asks to change them.
- Return exactly two sections using these exact uppercase markers: FORGY_REPLY: and UPDATED_PROMPT:.
- Put only a concise conversational response after FORGY_REPLY:. Put the complete updated image prompt after UPDATED_PROMPT:, or [UNCHANGED] when no prompt change is appropriate.""",
}


@dataclass(frozen=True)
class ZImageComponents:
    """Request-local Forge-owned objects required by the Z-Image text runtime."""

    sd_model: Any
    clip: Any
    encoder: Any
    tokenizer: Any
    engine: Any
    patcher: Any


class ZImageAdapter:
    """Text-only Forgy adapter for Z-Image Base and Turbo."""

    adapter_id = "z-image-qwen3-4b"
    capabilities = ModelCapabilities(
        workflows=frozenset(
            {
                Workflow.FORGY_CHAT,
                Workflow.IDEA_TO_PROMPT,
                Workflow.REFINE_PROMPT,
            }
        ),
        prompt_dialect=PromptDialect.NATURAL_LANGUAGE,
        text_generation=True,
        vision_input=False,
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
        default_persona_family="z-image-natural-language",
    )

    def __init__(self) -> None:
        self._text_generator: Callable[..., Any] | None = None

    @staticmethod
    def _class_path(value: Any) -> str:
        cls = type(value)
        return f"{cls.__module__}.{cls.__qualname__}"

    def probe(self, sd_model: Any) -> AdapterProbe:
        class_path = self._class_path(sd_model)
        return AdapterProbe(
            matched=class_path == ZIMAGE_MODEL_CLASS,
            evidence=(f"diffusion_model={class_path}",),
        )

    def resolve(
        self, sd_model: Any, probe: AdapterProbe
    ) -> tuple[ModelIdentity, ZImageComponents]:
        if not probe.matched:
            raise AdapterError("Z-Image adapter received a non-Z-Image model stack.")

        forge_objects = getattr(sd_model, "forge_objects", None)
        clip = getattr(forge_objects, "clip", None)
        cond_stage_model = getattr(clip, "cond_stage_model", None)
        tokenizer_container = getattr(clip, "tokenizer", None)
        encoder = getattr(cond_stage_model, "qwen3", None)
        tokenizer = getattr(tokenizer_container, "qwen3", None)
        engine = getattr(sd_model, "text_processing_engine_gemma", None)

        if clip is None or encoder is None or tokenizer is None or engine is None:
            raise AdapterError(
                "The expected Qwen3-4B encoder and tokenizer objects for the "
                "Z-Image adapter were not found. Select a complete Z-Image stack "
                "in Forge, then use 'Load current Forge selection' above."
            )
        if self._class_path(engine) != ZIMAGE_ENGINE_CLASS:
            raise AdapterError(
                "The active Z-Image text-processing engine is not supported: "
                f"{self._class_path(engine)}"
            )
        if getattr(engine, "text_encoder", None) is not encoder:
            raise AdapterError("Forge is using unexpectedly different Qwen3 objects.")
        if getattr(engine, "tokenizer", None) is not tokenizer:
            raise AdapterError(
                "Forge is using unexpectedly different Qwen3 tokenizer objects."
            )
        if (
            type(encoder).__module__ != ZIMAGE_ENCODER_MODULE
            or type(encoder).__name__ != ZIMAGE_ENCODER_CLASS
        ):
            raise AdapterError(
                "The active Z-Image text encoder class is not supported: "
                f"{self._class_path(encoder)}"
            )

        core_model = getattr(encoder, "model", None)
        config = getattr(core_model, "config", None)
        if (
            getattr(config, "hidden_size", None) != ZIMAGE_HIDDEN_SIZE
            or getattr(config, "vocab_size", None) != ZIMAGE_VOCAB_SIZE
        ):
            raise AdapterError(
                "The active Z-Image encoder does not expose the expected "
                f"Qwen3-4B contract ({ZIMAGE_HIDDEN_SIZE} hidden, "
                f"{ZIMAGE_VOCAB_SIZE} vocabulary)."
            )

        patcher = getattr(clip, "patcher", None)
        if patcher is None:
            raise AdapterError(
                "The Z-Image text encoder's Forge ModelPatcher is missing."
            )

        variant, checkpoint_evidence = self._variant(sd_model)
        display_variant = variant.title() if variant != "unknown" else "Base/Turbo"
        evidence = probe.evidence + (
            f"text_engine={self._class_path(engine)}",
            f"text_encoder={self._class_path(encoder)}",
            checkpoint_evidence,
        )
        return ModelIdentity(
            adapter_id=self.adapter_id,
            family="z-image",
            variant=variant,
            display_name=f"Z-Image {display_variant} · Qwen3 4B",
            evidence=evidence,
        ), ZImageComponents(
            sd_model=sd_model,
            clip=clip,
            encoder=encoder,
            tokenizer=tokenizer,
            engine=engine,
            patcher=patcher,
        )

    @staticmethod
    def _variant(sd_model: Any) -> tuple[str, str]:
        checkpoint = getattr(sd_model, "sd_checkpoint_info", None)
        values = [
            getattr(checkpoint, key, None)
            for key in ("name", "title", "model_name", "filename")
        ]
        values.append(getattr(sd_model, "filename", None))
        description = " ".join(str(value) for value in values if value)
        normalized = description.casefold()
        if "turbo" in normalized:
            variant = "turbo"
        elif "base" in normalized:
            variant = "base"
        else:
            variant = "unknown"
        checkpoint_name = Path(description).name if description else "unavailable"
        return variant, f"checkpoint={checkpoint_name}"

    def bind_runtime(self, *, text_generator: Callable[..., Any]) -> None:
        """Attach text generation without retaining any live model components."""

        self._text_generator = text_generator

    def generate_text(self, **kwargs):
        if self._text_generator is None:
            raise AdapterError("The Z-Image text runtime has not been registered.")
        return self._text_generator(**kwargs)

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
        template = getattr(engine, "llama_template", None)
        if not isinstance(template, str) or template.count("{}") != 1:
            raise AdapterError(
                "The active Z-Image Qwen3 chat template is incompatible."
            )
        if "<|im_start|>user\n" not in template or not template.endswith(
            "<|im_start|>assistant\n"
        ):
            raise AdapterError(
                "The role structure of the active Z-Image Qwen3 template is incompatible."
            )

        system_prompt = validate_system_prompt(system_prompt, allow_empty=True)
        user_message = idea.strip()
        instruction = str(instruction or "").strip()
        if instruction:
            user_message += f"\n\nAdditional instruction:\n{instruction}"
        workflow_contract = ZIMAGE_WORKFLOW_CONTRACTS.get(workflow)
        if workflow_contract:
            user_message += f"\n\n{workflow_contract}"
        rendered = template.format(user_message)
        if system_prompt:
            rendered = f"{QWEN_SYSTEM_START}{system_prompt}{QWEN_MESSAGE_END}{rendered}"
        return rendered + "<think>\n\n</think>\n\n"

    @staticmethod
    def decode_generated_text(tokenizer, generated_ids) -> str:
        return decode_generated_text(tokenizer, generated_ids)

    @staticmethod
    def strip_model_thinking(text: str) -> tuple[str, bool]:
        return strip_model_thinking(text)

    @staticmethod
    def repetition_loop_suffix(token_ids: list[int]) -> tuple[int, int] | None:
        return repetition_loop_suffix(token_ids)
