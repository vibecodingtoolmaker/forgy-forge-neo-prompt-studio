"""Explicit model-family adapters shipped with Forgy."""

# Copyright (C) 2026 vibecodingtoolmaker
# SPDX-License-Identifier: AGPL-3.0-only

from .base import AdapterError, AdapterProbe, ModelIdentity
from .flux2_klein import Flux2KleinAdapter, Flux2KleinComponents
from .krea2 import Krea2Adapter, Krea2Components
from .zimage import ZImageAdapter, ZImageComponents

__all__ = [
    "AdapterError",
    "AdapterProbe",
    "Flux2KleinAdapter",
    "Flux2KleinComponents",
    "Krea2Adapter",
    "Krea2Components",
    "ModelIdentity",
    "ZImageAdapter",
    "ZImageComponents",
]
