"""MOSS-Image adapter for Forge's Qwen3.5-9B text encoder."""

# Copyright (C) 2026 vibecodingtoolmaker
# SPDX-License-Identifier: AGPL-3.0-only

from __future__ import annotations

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
    QWEN_THINKING_SUFFIX,
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


MOSS_MODEL_CLASS = "backend.diffusion_engine.moss_image.MOSSImage"
MOSS_ENGINE_CLASS = "backend.text_processing.moss_image_engine.Qwen359BEngine"
MOSS_ENCODER_MODULE = "backend.nn.llm.llama"
MOSS_ENCODER_CLASS = "Qwen35_9B"
MOSS_HIDDEN_SIZE = 4096
MOSS_VOCAB_SIZE = 248320
QWEN_SYSTEM_START = "<|im_start|>system\n"
QWEN_MESSAGE_END = "<|im_end|>\n"
MOSS_WORKFLOW_CONTRACTS = {
    Workflow.IDEA_TO_PROMPT: """Workflow contract:
- Treat the user's image idea as the intended visual concept, not as a question to answer or prose to correct.
- Preserve every explicit subject, count, action, setting, style, relationship, visible text, and constraint.
- Add only concrete visible detail that supports the same concept; do not invent unrelated people, objects, or copy.
- Follow the selected system persona's requested prompt dialect. A cohesive caption and a comma-separated tag sequence are both valid MOSS-Image prompts.
- Return exactly one finished image prompt with no introduction, heading, quotation marks, notes, or alternatives.""",
    Workflow.REFINE_PROMPT: """Workflow contract:
- Revise the existing image prompt only according to the refinement instruction.
- Preserve every unaffected subject, count, action, setting, style, composition, relationship, visible text, and constraint.
- Preserve the existing caption or tag-sequence dialect unless the user or selected system persona explicitly requests another MOSS-Image prompt format.
- Return exactly one complete revised image prompt with no introduction, heading, quotation marks, notes, or alternatives.""",
    Workflow.FORGY_CHAT: """Workflow contract:
- Treat the latest user message as a request about the working image prompt, not as prose to correct or a question that replaces the user's concept.
- Preserve explicit subjects, counts, actions, relationships, visible text, and constraints unless the user asks to change them.
- Return exactly two sections using these exact uppercase markers: FORGY_REPLY: and UPDATED_PROMPT:.
- Put only a concise conversational response after FORGY_REPLY:. Put the complete updated image prompt after UPDATED_PROMPT:, or [UNCHANGED] when no prompt change is appropriate.""",
    Workflow.IMAGE_TO_PROMPT: """Workflow contract:
- Treat the uploaded picture as the authoritative visual source.
- Describe its visible subject, count, action, setting, composition, lighting, color, materials, style, relationships, and legible text faithfully.
- Apply the user's instruction without inventing unrelated content.
- Follow the selected system persona's requested caption or tag-sequence dialect.
- Return exactly one finished image prompt with no introduction, heading, quotation marks, notes, or alternatives.""",
}


@dataclass(frozen=True)
class MossComponents:
    """Request-local Forge-owned objects required by the MOSS text runtime."""

    sd_model: Any
    clip: Any
    encoder: Any
    tokenizer: Any
    engine: Any
    patcher: Any


class MossAdapter:
    """Forgy adapter for an active MOSS-Image Qwen3.5-VL stack."""

    adapter_id = "moss-image-qwen35-9b"
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
        default_persona_family="moss-natural-language",
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
            matched=class_path == MOSS_MODEL_CLASS,
            evidence=(f"diffusion_model={class_path}",),
        )

    def resolve(
        self, sd_model: Any, probe: AdapterProbe
    ) -> tuple[ModelIdentity, MossComponents]:
        if not probe.matched:
            raise AdapterError("MOSS adapter received a non-MOSS model stack.")

        forge_objects = getattr(sd_model, "forge_objects", None)
        clip = getattr(forge_objects, "clip", None)
        cond_stage_model = getattr(clip, "cond_stage_model", None)
        tokenizer_container = getattr(clip, "tokenizer", None)
        encoder = getattr(cond_stage_model, "qwen35_9b", None)
        tokenizer = getattr(tokenizer_container, "qwen35_9b", None)
        engine = getattr(sd_model, "text_processing_engine_qwen", None)

        if clip is None or encoder is None or tokenizer is None or engine is None:
            raise AdapterError(
                "The expected Qwen3.5-9B encoder and tokenizer objects for MOSS "
                "were not found. Select a complete MOSS-Image stack in Forge, "
                "then use 'Load current Forge selection' above."
            )
        if self._class_path(engine) != MOSS_ENGINE_CLASS:
            raise AdapterError(
                "The active MOSS text-processing engine is not supported: "
                f"{self._class_path(engine)}"
            )
        if not engine_component_matches(
            getattr(engine, "text_encoder", None),
            encoder,
            wrapper_attribute="transformer",
        ):
            raise AdapterError(
                "Forge is using unexpectedly different MOSS Qwen3.5 encoder objects."
            )
        if not engine_component_matches(
            getattr(engine, "tokenizer", None),
            tokenizer,
            wrapper_attribute="tokenizer",
        ):
            raise AdapterError(
                "Forge is using unexpectedly different MOSS tokenizer objects."
            )
        if (
            type(encoder).__module__ != MOSS_ENCODER_MODULE
            or type(encoder).__name__ != MOSS_ENCODER_CLASS
        ):
            raise AdapterError(
                "The active MOSS text encoder class is not supported: "
                f"{self._class_path(encoder)}"
            )

        core_model = getattr(encoder, "model", None)
        config = getattr(core_model, "config", None)
        embedding = getattr(core_model, "embed_tokens", None)
        final_norm = getattr(core_model, "norm", None)
        if (
            getattr(config, "hidden_size", None) != MOSS_HIDDEN_SIZE
            or getattr(config, "vocab_size", None) != MOSS_VOCAB_SIZE
            or getattr(config, "num_hidden_layers", None) != 32
        ):
            raise AdapterError(
                "The active MOSS encoder does not expose the expected Qwen3.5-9B "
                f"contract ({MOSS_HIDDEN_SIZE} hidden, {MOSS_VOCAB_SIZE} vocabulary, "
                "32 layers)."
            )
        if embedding is None or final_norm is None:
            raise AdapterError(
                "The active MOSS Qwen3.5 encoder lacks its token embedding or final "
                "language-model norm."
            )

        patcher = getattr(clip, "patcher", None)
        if patcher is None:
            raise AdapterError("The MOSS text encoder's Forge ModelPatcher is missing.")

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
            family="moss-image",
            variant="qwen3.5-9b",
            display_name="MOSS-Image · Qwen3.5 9B",
            evidence=evidence,
        ), MossComponents(
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
        """Attach generation without retaining any live Forge components."""

        self._text_generator = text_generator
        self._image_generator = image_generator

    def generate_text(self, **kwargs):
        if self._text_generator is None:
            raise AdapterError("The MOSS text runtime has not been registered.")
        return self._text_generator(**kwargs)

    def generate_image(self, **kwargs):
        if self._image_generator is None:
            raise AdapterError("The MOSS vision runtime has not been registered.")
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
        show_model_thinking: bool = False,
    ) -> str:
        template = getattr(engine, "llama_template", None)
        if not isinstance(template, str) or template.count("{}") != 1:
            raise AdapterError("The active MOSS Qwen3.5 chat template is incompatible.")
        if "<|im_start|>user\n" not in template or not template.endswith(
            f"<|im_start|>assistant\n{QWEN_THINKING_SUFFIX}"
        ):
            raise AdapterError(
                "The role structure of the active MOSS Qwen3.5 template is incompatible."
            )

        system_prompt = validate_system_prompt(system_prompt, allow_empty=True)
        user_message = idea.strip()
        instruction = str(instruction or "").strip()
        if instruction:
            user_message += f"\n\nAdditional instruction:\n{instruction}"
        workflow_contract = MOSS_WORKFLOW_CONTRACTS.get(workflow)
        if workflow_contract:
            user_message += f"\n\n{workflow_contract}"

        rendered = template.format(user_message)
        rendered = rendered[: -len(QWEN_THINKING_SUFFIX)]
        if system_prompt:
            rendered = f"{QWEN_SYSTEM_START}{system_prompt}{QWEN_MESSAGE_END}{rendered}"
        return apply_model_thinking_mode(rendered, enabled=show_model_thinking)

    def render_image_generation_prompt(
        self,
        engine,
        instruction: str,
        system_prompt: str,
        *,
        default_image_request: str,
        validate_system_prompt: Callable[..., str],
        show_model_thinking: bool = False,
    ) -> str:
        """Insert Forge's exact MOSS picture marker into a validated request."""

        vision_block = getattr(engine, "vision_block", None)
        if not isinstance(vision_block, str) or not vision_block:
            raise AdapterError("The active MOSS image chat template is unavailable.")
        user_message = (
            instruction.strip()
            if isinstance(instruction, str) and instruction.strip()
            else default_image_request
        )
        return self.render_generation_prompt(
            engine,
            f"Picture 1: {vision_block}\n{user_message}",
            "",
            system_prompt,
            workflow=Workflow.IMAGE_TO_PROMPT,
            validate_system_prompt=validate_system_prompt,
            show_model_thinking=show_model_thinking,
        )

    @staticmethod
    def decode_generated_text(tokenizer, generated_ids) -> str:
        return decode_generated_text(tokenizer, generated_ids)

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
