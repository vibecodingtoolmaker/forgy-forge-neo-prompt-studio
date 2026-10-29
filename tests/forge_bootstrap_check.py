"""Validate Forgy's modular entry point in a fully initialized Forge runtime."""

# Copyright (C) 2026 vibecodingtoolmaker
# SPDX-License-Identifier: AGPL-3.0-only

from __future__ import annotations

import runpy
import sys
from pathlib import Path


EXTENSION_ROOT = Path(__file__).resolve().parents[1]
FORGE_ROOT = EXTENSION_ROOT.parents[1]
for path in (FORGE_ROOT, EXTENSION_ROOT, FORGE_ROOT / "modules_forge" / "packages"):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from modules import initialize


initialize.imports()

SCRIPT_PATH = EXTENSION_ROOT / "scripts" / "forge_krea_prompt_assistant.py"
namespace = runpy.run_path(str(SCRIPT_PATH), run_name="forgy_import_check")

manager = namespace["MODEL_MANAGER"]
adapter = namespace["KREA2_ADAPTER"]
klein_adapter = namespace["FLUX2_KLEIN_ADAPTER"]
anima_adapter = namespace["ANIMA_ADAPTER"]
moss_adapter = namespace["MOSS_ADAPTER"]
capability_manager = namespace["CAPABILITY_MANAGER"]

assert manager.adapter_ids == (
    "krea2-qwen3-vl",
    "z-image-qwen3-4b",
    "flux2-klein-qwen3-4b",
    "anima-qwen3-0.6b",
    "moss-image-qwen35-9b",
)
assert adapter.adapter_id == "krea2-qwen3-vl"
assert adapter.capabilities.text_generation is True
assert adapter.capabilities.vision_input is True
assert klein_adapter.adapter_id == "flux2-klein-qwen3-4b"
assert klein_adapter.capabilities.text_generation is True
assert klein_adapter.capabilities.vision_input is False
assert anima_adapter.adapter_id == "anima-qwen3-0.6b"
assert anima_adapter.capabilities.text_generation is True
assert anima_adapter.capabilities.vision_input is False
assert moss_adapter.adapter_id == "moss-image-qwen35-9b"
assert moss_adapter.capabilities.text_generation is True
assert moss_adapter.capabilities.vision_input is True
assert capability_manager is not None
assert adapter.__dict__.get("sd_model") is None
assert adapter.__dict__.get("components") is None
assert klein_adapter.__dict__.get("sd_model") is None
assert klein_adapter.__dict__.get("components") is None
assert anima_adapter.__dict__.get("sd_model") is None
assert anima_adapter.__dict__.get("components") is None
assert moss_adapter.__dict__.get("sd_model") is None
assert moss_adapter.__dict__.get("components") is None

print("FORGE_IMPORT_OK")  # noqa: T201
print("MODEL_MANAGER_OK", manager.adapter_ids)  # noqa: T201
print("KREA2_CAPABILITIES_OK")  # noqa: T201
print("FLUX2_KLEIN_CAPABILITIES_OK")  # noqa: T201
print("ANIMA_CAPABILITIES_OK")  # noqa: T201
print("MOSS_CAPABILITIES_OK")  # noqa: T201
print("NO_PERSISTENT_MODEL_CONTEXT_OK")  # noqa: T201
