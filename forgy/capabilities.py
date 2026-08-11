"""Declarative model capabilities used by Forgy's workflows and UI."""

# Copyright (C) 2026 vibecodingtoolmaker
# SPDX-License-Identifier: AGPL-3.0-only

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from types import MappingProxyType
from typing import Mapping, TYPE_CHECKING

if TYPE_CHECKING:
    from .model_manager import ModelContext


class Workflow(str, Enum):
    """Stable names for model-dependent Forgy workflows."""

    FORGY_CHAT = "forgy_chat"
    IDEA_TO_PROMPT = "idea_to_prompt"
    IMAGE_TO_PROMPT = "image_to_prompt"
    REFINE_PROMPT = "refine_prompt"


class PromptDialect(str, Enum):
    """Prompt syntax preferred by the target image-model family."""

    NATURAL_LANGUAGE = "natural_language"
    TAGS = "tags"
    HYBRID = "hybrid"


class ReasoningMode(str, Enum):
    """Observable reasoning behavior of a prompt-generation backend."""

    NONE = "none"
    HIDDEN = "hidden"
    VISIBLE = "visible"
    CONTROLLABLE = "controllable"


@dataclass(frozen=True)
class ModelCapabilities:
    """Static abilities declared by one validated model-family adapter.

    Runtime availability is resolved separately. For example, a workflow can be
    declared here but still be unavailable when Forge exposes an incomplete
    stack or the current memory profile is too small.
    """

    workflows: frozenset[Workflow]
    prompt_dialect: PromptDialect
    text_generation: bool
    vision_input: bool
    multi_turn_chat: bool
    system_prompt: bool
    reasoning_mode: ReasoningMode
    sampling_controls: frozenset[str]
    default_persona_family: str

    def supports(self, workflow: Workflow) -> bool:
        """Return whether the adapter declares support for ``workflow``."""

        return workflow in self.workflows


@dataclass(frozen=True)
class WorkflowAvailability:
    """Effective state and user-facing explanation for one workflow."""

    enabled: bool
    reason: str


@dataclass(frozen=True)
class EffectiveCapabilities:
    """Runtime-evaluated workflow availability for one model context."""

    adapter_id: str
    workflows: Mapping[Workflow, WorkflowAvailability]

    def for_workflow(self, workflow: Workflow) -> WorkflowAvailability:
        return self.workflows[workflow]


class CapabilityManager:
    """Combine declared adapter abilities with validated runtime availability."""

    def resolve(self, context: "ModelContext") -> EffectiveCapabilities:
        resolved = {}
        for workflow in Workflow:
            resolved[workflow] = self._workflow_availability(context, workflow)
        return EffectiveCapabilities(
            adapter_id=context.identity.adapter_id,
            workflows=MappingProxyType(resolved),
        )

    @staticmethod
    def _workflow_availability(
        context: "ModelContext", workflow: Workflow
    ) -> WorkflowAvailability:
        capabilities = context.capabilities
        display_name = context.identity.display_name
        if not capabilities.supports(workflow):
            return WorkflowAvailability(
                False,
                f"{display_name} does not support {workflow.value}.",
            )
        if not capabilities.text_generation:
            return WorkflowAvailability(
                False,
                f"{display_name} cannot generate prompt text.",
            )
        if workflow is Workflow.IMAGE_TO_PROMPT and not capabilities.vision_input:
            return WorkflowAvailability(
                False,
                f"{display_name} has no vision input.",
            )
        return WorkflowAvailability(True, f"Available through {display_name}.")
