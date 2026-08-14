"""Dependency-free contracts shared by Forgy's model-family adapters."""

# Copyright (C) 2026 vibecodingtoolmaker
# SPDX-License-Identifier: AGPL-3.0-only

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Protocol

from ..capabilities import ModelCapabilities


class AdapterError(RuntimeError):
    """Expected validation or compatibility failure from a model adapter."""


@dataclass(frozen=True)
class AdapterProbe:
    """Read-only result of asking an adapter whether a live stack belongs to it."""

    matched: bool
    evidence: tuple[str, ...] = ()


@dataclass(frozen=True)
class ModelIdentity:
    """Stable model identity exposed to shared Forgy orchestration."""

    adapter_id: str
    family: str
    variant: str
    display_name: str
    evidence: tuple[str, ...]


class ModelAdapter(Protocol):
    """Small contract implemented independently by each model family."""

    adapter_id: str
    capabilities: ModelCapabilities

    def probe(self, sd_model: Any) -> AdapterProbe:
        """Inspect without retaining Forge-owned runtime objects."""

    def resolve(self, sd_model: Any, probe: AdapterProbe) -> tuple[ModelIdentity, Any]:
        """Validate and return request-local components for the matched stack."""
