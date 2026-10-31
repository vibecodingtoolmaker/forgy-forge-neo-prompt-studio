"""Anima adapter for Forge's stock Qwen3-0.6B prompt backend."""

# Copyright (C) 2026 vibecodingtoolmaker
# SPDX-License-Identifier: AGPL-3.0-only

from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable

from .base import (
    AdapterError,
    AdapterProbe,
    ModelIdentity,
    engine_component_matches,
)
from .qwen import (
    apply_model_thinking_mode,
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


ANIMA_MODEL_CLASS = "backend.diffusion_engine.anima.Anima"
ANIMA_ENGINE_CLASSES = frozenset(
    {
        "backend.text_processing.anima_engine.AnimaTextProcessingEngine",
        "backend.text_processing.anima_engine.Qwen06Engine",
    }
)
ANIMA_ENCODER_MODULE = "backend.nn.llm.llama"
ANIMA_ENCODER_CLASS = "Qwen3_06B"
ANIMA_HIDDEN_SIZE = 1024
ANIMA_VOCAB_SIZE = 151936
ANIMA_TEXT_PROVIDER_ORIGINAL = "Original Qwen3 0.6B"
ANIMA_TEXT_PROVIDER_QWEN35 = "Qwen3.5 4B adapter cache (experimental)"
ANIMA_TEXT_PROVIDERS = (
    ANIMA_TEXT_PROVIDER_ORIGINAL,
    ANIMA_TEXT_PROVIDER_QWEN35,
)
QWEN_SYSTEM_START = "<|im_start|>system\n"
QWEN_USER_START = "<|im_start|>user\n"
QWEN_ASSISTANT_START = "<|im_start|>assistant\n"
QWEN_MESSAGE_END = "<|im_end|>\n"
ANIMA_WORKFLOW_CONTRACTS = {
    Workflow.IDEA_TO_PROMPT: """Workflow contract:
- Treat the user's image idea as the intended visual concept, not as a question to answer or prose to correct.
- Preserve every explicit subject, action, setting, style, relationship, and constraint. Do not silently replace an unclear detail with a different subject or story.
- Add only visually useful detail that supports the same concept; do not invent extra characters or unrelated objects.
- Follow the selected system persona for style and return exactly one finished natural-language image prompt with no introduction, heading, quotation marks, notes, or alternatives.""",
    Workflow.REFINE_PROMPT: """Workflow contract:
- Revise the existing image prompt only according to the refinement instruction.
- Preserve every unaffected subject, action, setting, style, composition, relationship, and constraint. Do not silently replace ambiguous wording or invent unrelated content.
- Follow the selected system persona for style and return exactly one complete revised natural-language image prompt with no introduction, heading, quotation marks, notes, or alternatives.""",
    Workflow.FORGY_CHAT: """Workflow contract:
- Treat the latest user message as a request about the working image prompt, not as prose to correct or a question that replaces the user's concept.
- Preserve explicit subjects, actions, relationships, and constraints unless the user asks to change them.
- Return exactly two sections using these exact uppercase markers: FORGY_REPLY: and UPDATED_PROMPT:.
- Put only a concise conversational response after FORGY_REPLY:. Put the complete updated natural-language image prompt after UPDATED_PROMPT:, or [UNCHANGED] when no prompt change is appropriate.""",
}


@dataclass(frozen=True)
class AnimaComponents:
    """Request-local Forge-owned objects for stock Anima text generation."""

    sd_model: Any
    clip: Any
    encoder: Any
    tokenizer: Any
    engine: Any
    patcher: Any


class AnimaAdapter:
    """Text-only Forgy adapter for Anima with its stock Qwen3-0.6B encoder."""

    adapter_id = "anima-qwen3-0.6b"
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
        default_persona_family="anima-natural-language",
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
            matched=class_path == ANIMA_MODEL_CLASS,
            evidence=(f"diffusion_model={class_path}",),
        )

    def resolve(
        self, sd_model: Any, probe: AdapterProbe
    ) -> tuple[ModelIdentity, AnimaComponents]:
        if not probe.matched:
            raise AdapterError("Anima adapter received a non-Anima model stack.")

        forge_objects = getattr(sd_model, "forge_objects", None)
        clip = getattr(forge_objects, "clip", None)
        cond_stage_model = getattr(clip, "cond_stage_model", None)
        tokenizer_container = getattr(clip, "tokenizer", None)
        encoder = getattr(cond_stage_model, "qwen3_06b", None)
        tokenizer = getattr(tokenizer_container, "qwen3_06b", None)
        engine = getattr(sd_model, "text_processing_engine_qwen", None)
        if engine is None:
            engine = getattr(sd_model, "text_processing_engine_anima", None)

        if clip is None or encoder is None or tokenizer is None or engine is None:
            raise AdapterError(
                "The expected stock Qwen3-0.6B encoder and tokenizer objects for "
                "Anima were not found. Select a complete Anima stack with the "
                "original qwen_3_06b text encoder in Forge, then use 'Load current "
                "Forge selection' above."
            )
        if self._class_path(engine) not in ANIMA_ENGINE_CLASSES:
            raise AdapterError(
                "The active Anima text-processing engine is not supported: "
                f"{self._class_path(engine)}"
            )
        if not engine_component_matches(
            getattr(engine, "text_encoder", None),
            encoder,
            wrapper_attribute="transformer",
        ):
            raise AdapterError(
                "Forge is using unexpectedly different Anima Qwen3 encoder objects."
            )
        if not engine_component_matches(
            getattr(engine, "qwen_tokenizer", None),
            tokenizer,
            wrapper_attribute="tokenizer",
        ):
            raise AdapterError(
                "Forge is using unexpectedly different Anima Qwen3 tokenizer objects."
            )
        if (
            type(encoder).__module__ != ANIMA_ENCODER_MODULE
            or type(encoder).__name__ != ANIMA_ENCODER_CLASS
        ):
            raise AdapterError(
                "The active Anima text encoder class is not supported by the "
                f"stock adapter: {self._class_path(encoder)}"
            )

        core_model = getattr(encoder, "model", None)
        config = getattr(core_model, "config", None)
        embedding = getattr(core_model, "embed_tokens", None)
        final_norm = getattr(core_model, "norm", None)
        if (
            getattr(config, "hidden_size", None) != ANIMA_HIDDEN_SIZE
            or getattr(config, "vocab_size", None) != ANIMA_VOCAB_SIZE
        ):
            raise AdapterError(
                "The active Anima encoder does not expose the expected stock "
                f"Qwen3-0.6B contract ({ANIMA_HIDDEN_SIZE} hidden, "
                f"{ANIMA_VOCAB_SIZE} vocabulary)."
            )
        if embedding is None or final_norm is None:
            raise AdapterError(
                "The stock Anima Qwen3 encoder lacks its trained tied embedding "
                "or final language-model norm."
            )

        patcher = getattr(clip, "patcher", None)
        if patcher is None:
            raise AdapterError(
                "The Anima text encoder's Forge ModelPatcher is missing."
            )

        checkpoint = getattr(sd_model, "sd_checkpoint_info", None)
        values = [
            getattr(checkpoint, key, None)
            for key in ("name", "title", "model_name", "filename")
        ]
        values.append(getattr(sd_model, "filename", None))
        description = " ".join(str(value) for value in values if value)
        checkpoint_name = Path(description).name if description else "unavailable"
        evidence = probe.evidence + (
            f"text_engine={self._class_path(engine)}",
            f"text_encoder={self._class_path(encoder)}",
            f"checkpoint={checkpoint_name}",
        )
        return ModelIdentity(
            adapter_id=self.adapter_id,
            family="anima",
            variant="qwen3-0.6b",
            display_name="Anima · Qwen3 0.6B (original TE)",
            evidence=evidence,
        ), AnimaComponents(
            sd_model=sd_model,
            clip=clip,
            encoder=encoder,
            tokenizer=tokenizer,
            engine=engine,
            patcher=patcher,
        )

    def bind_runtime(self, *, text_generator: Callable[..., Any]) -> None:
        """Attach text generation without retaining live Forge components."""

        self._text_generator = text_generator

    def generate_text(self, **kwargs):
        if self._text_generator is None:
            raise AdapterError("The Anima text runtime has not been registered.")
        kwargs["anima_text_provider"] = self.normalize_text_provider(
            kwargs.get("anima_text_provider")
        )
        return self._text_generator(**kwargs)

    @staticmethod
    def normalize_text_provider(value: Any) -> str:
        """Validate the request-local Anima speech provider selection."""

        if value in (None, "", ANIMA_TEXT_PROVIDER_ORIGINAL):
            return ANIMA_TEXT_PROVIDER_ORIGINAL
        if value == ANIMA_TEXT_PROVIDER_QWEN35:
            return ANIMA_TEXT_PROVIDER_QWEN35
        raise AdapterError(f"Unknown Anima text provider: {value}")

    @staticmethod
    def render_generation_prompt(
        engine,
        idea: str,
        instruction: str,
        system_prompt: str,
        *,
        workflow: Workflow = Workflow.IDEA_TO_PROMPT,
        validate_system_prompt: Callable[..., str],
        show_model_thinking: bool = False,
    ) -> str:
        if AnimaAdapter._class_path(engine) not in ANIMA_ENGINE_CLASSES:
            raise AdapterError("The active Anima prompt engine is incompatible.")

        system_prompt = validate_system_prompt(system_prompt, allow_empty=True)
        user_message = idea.strip()
        instruction = str(instruction or "").strip()
        if instruction:
            user_message += f"\n\nAdditional instruction:\n{instruction}"
        workflow_contract = ANIMA_WORKFLOW_CONTRACTS.get(workflow)
        if workflow_contract:
            user_message += f"\n\n{workflow_contract}"

        rendered = ""
        if system_prompt:
            rendered += f"{QWEN_SYSTEM_START}{system_prompt}{QWEN_MESSAGE_END}"
        rendered += (
            f"{QWEN_USER_START}{user_message}{QWEN_MESSAGE_END}{QWEN_ASSISTANT_START}"
        )
        return apply_model_thinking_mode(rendered, enabled=show_model_thinking)

    @staticmethod
    def decode_generated_text(tokenizer, generated_ids) -> str:
        text = decode_generated_text(tokenizer, generated_ids)
        text = re.sub(r"(?is)^\s*assistant\s*:\s*", "", text, count=1)
        next_role = re.search(
            r"(?is)(?:[\r\n]+|(?<=[.!?]))\s*(?:human|user|assistant)\s*:",
            text,
        )
        if next_role is not None:
            text = text[: next_role.start()]
        return text.strip()

    @staticmethod
    def strip_model_thinking(
        text: str,
        *,
        prefilled_thinking: bool = False,
    ) -> tuple[str, bool]:
        return strip_model_thinking(
            text,
            prefilled_thinking=prefilled_thinking,
        )

    @staticmethod
    def repetition_loop_suffix(token_ids: list[int]) -> tuple[int, int] | None:
        return repetition_loop_suffix(token_ids)
