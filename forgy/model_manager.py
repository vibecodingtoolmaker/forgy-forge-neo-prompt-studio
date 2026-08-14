"""Resolve Forge's active stack to one explicit Forgy model adapter."""

# Copyright (C) 2026 vibecodingtoolmaker
# SPDX-License-Identifier: AGPL-3.0-only

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Iterable

from .adapters.base import AdapterError, ModelAdapter, ModelIdentity
from .capabilities import ModelCapabilities


class UnsupportedModelError(AdapterError):
    """No installed adapter can honestly support the active Forge stack."""


@dataclass(frozen=True)
class ModelContext:
    """One request-local adapter resolution.

    A context may contain live Forge objects, so callers must not persist it on
    the UI script instance or in module globals.
    """

    adapter: ModelAdapter
    identity: ModelIdentity
    capabilities: ModelCapabilities
    components: Any


def qualified_name(value: Any) -> str:
    """Return a stable diagnostic class path without importing model modules."""

    cls = type(value)
    return f"{cls.__module__}.{cls.__qualname__}"


class ModelManager:
    """Explicit adapter registry with deterministic, fail-closed resolution."""

    def __init__(self, adapters: Iterable[ModelAdapter]) -> None:
        registered = tuple(adapters)
        ids = [adapter.adapter_id for adapter in registered]
        if len(ids) != len(set(ids)):
            raise ValueError("model adapter IDs must be unique")
        self._adapters = registered

    @property
    def adapter_ids(self) -> tuple[str, ...]:
        return tuple(adapter.adapter_id for adapter in self._adapters)

    def resolve(self, sd_model: Any) -> ModelContext:
        if sd_model is None:
            raise UnsupportedModelError(
                "No diffusion model is loaded. Use 'Load current Forge selection' above."
            )

        matches = []
        for adapter in self._adapters:
            probe = adapter.probe(sd_model)
            if probe.matched:
                matches.append((adapter, probe))

        if not matches:
            raise UnsupportedModelError(
                "No supported model stack is active. "
                f"Active model: {qualified_name(sd_model)}. Select a supported "
                "stack in Forge, then use 'Load current Forge selection' above."
            )
        if len(matches) > 1:
            adapter_ids = ", ".join(adapter.adapter_id for adapter, _ in matches)
            raise UnsupportedModelError(
                f"The active stack ambiguously matches multiple adapters: {adapter_ids}."
            )

        adapter, probe = matches[0]
        identity, components = adapter.resolve(sd_model, probe)
        return ModelContext(
            adapter=adapter,
            identity=identity,
            capabilities=adapter.capabilities,
            components=components,
        )
