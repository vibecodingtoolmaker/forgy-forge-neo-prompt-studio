"""Explicit model-family adapters shipped with Forgy."""

# Copyright (C) 2026 vibecodingtoolmaker
# SPDX-License-Identifier: AGPL-3.0-only

from .anima import (
    ANIMA_TEXT_PROVIDER_ORIGINAL,
    ANIMA_TEXT_PROVIDER_QWEN35,
    ANIMA_TEXT_PROVIDERS,
    AnimaAdapter,
    AnimaComponents,
)
from .base import AdapterError, AdapterProbe, ModelIdentity
from .flux2_klein import Flux2KleinAdapter, Flux2KleinComponents
from .krea2 import Krea2Adapter, Krea2Components
from .moss import MossAdapter, MossComponents
from .zimage import ZImageAdapter, ZImageComponents

__all__ = [
    "AnimaAdapter",
    "AnimaComponents",
    "ANIMA_TEXT_PROVIDER_ORIGINAL",
    "ANIMA_TEXT_PROVIDER_QWEN35",
    "ANIMA_TEXT_PROVIDERS",
    "AdapterError",
    "AdapterProbe",
    "Flux2KleinAdapter",
    "Flux2KleinComponents",
    "Krea2Adapter",
    "Krea2Components",
    "MossAdapter",
    "MossComponents",
    "ModelIdentity",
    "ZImageAdapter",
    "ZImageComponents",
]
