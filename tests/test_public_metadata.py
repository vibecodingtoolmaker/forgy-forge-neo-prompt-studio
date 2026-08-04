# Copyright (C) 2026 vibecodingtoolmaker
# SPDX-License-Identifier: AGPL-3.0-only

import ast
import json
from pathlib import Path
import sys
import threading
import time
from types import ModuleType, SimpleNamespace
import unittest
from unittest.mock import patch
import uuid


ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts" / "forge_krea_prompt_assistant.py"
PERSONAS_EXAMPLE = ROOT / "personas.example.json"
IMAGE_PERSONAS_EXAMPLE = ROOT / "image_personas.example.json"
ROADMAP = ROOT / "ROADMAP.md"


def public_constants(source: str) -> dict[str, object]:
    tree = ast.parse(source, filename=str(SCRIPT))
    wanted = {
        "EXTENSION_VERSION",
        "DEFAULT_PERSONA_PROMPT",
        "DEFAULT_IMAGE_PERSONA_PROMPT",
        "VRAM_PROFILES",
    }
    values: dict[str, object] = {}

    for node in tree.body:
        if not isinstance(node, ast.Assign) or len(node.targets) != 1:
            continue
        target = node.targets[0]
        if isinstance(target, ast.Name) and target.id in wanted:
            values[target.id] = ast.literal_eval(node.value)

    return values


def isolated_function(source: str, name: str, namespace: dict[str, object]):
    tree = ast.parse(source, filename=str(SCRIPT))
    function = next(
        node
        for node in tree.body
        if isinstance(node, ast.FunctionDef) and node.name == name
    )
    module = ast.fix_missing_locations(ast.Module(body=[function], type_ignores=[]))
    exec(compile(module, str(SCRIPT), "exec"), namespace)  # noqa: S102
    return namespace[name]


class PublicMetadataTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.source = SCRIPT.read_text(encoding="utf-8")
        cls.constants = public_constants(cls.source)
        cls.personas = json.loads(PERSONAS_EXAMPLE.read_text(encoding="utf-8"))
        cls.image_personas = json.loads(
            IMAGE_PERSONAS_EXAMPLE.read_text(encoding="utf-8")
        )
        cls.roadmap = ROADMAP.read_text(encoding="utf-8")

    def test_extension_script_compiles(self) -> None:
        compile(self.source, str(SCRIPT), "exec")

    def test_expected_constants_are_literal_and_present(self) -> None:
        self.assertEqual(
            set(self.constants),
            {
                "EXTENSION_VERSION",
                "DEFAULT_PERSONA_PROMPT",
                "DEFAULT_IMAGE_PERSONA_PROMPT",
                "VRAM_PROFILES",
            },
        )

    def test_public_alpha_version(self) -> None:
        self.assertEqual(self.constants["EXTENSION_VERSION"], "0.5.0-alpha.1")

    def test_example_default_matches_built_in_default(self) -> None:
        self.assertEqual(
            self.personas["personas"]["Default"],
            self.constants["DEFAULT_PERSONA_PROMPT"],
        )

    def test_image_example_default_matches_built_in_default(self) -> None:
        self.assertEqual(
            self.image_personas["personas"]["Default"],
            self.constants["DEFAULT_IMAGE_PERSONA_PROMPT"],
        )

    def test_vram_profiles(self) -> None:
        self.assertEqual(
            self.constants["VRAM_PROFILES"],
            {8: 4096, 12: 5120, 16: 6144, 24: 8192, 32: 12288},
        )

    def test_spdx_header_is_present(self) -> None:
        self.assertIn("SPDX-License-Identifier: AGPL-3.0-only", self.source[:300])

    def test_model_loading_is_delegated_to_forge(self) -> None:
        self.assertIn("main_entry.refresh_model_loading_parameters(", self.source)
        self.assertIn("sd_models.forge_model_reload()", self.source)
        self.assertNotIn("main_entry.checkpoint_change(", self.source)
        self.assertNotIn("main_entry.modules_change(", self.source)
        self.assertNotIn("from_pretrained(", self.source)

    def test_current_forge_selection_loader_only_refreshes_and_reloads(self) -> None:
        calls = []
        sd_models = SimpleNamespace(forge_model_reload=lambda: calls.append("reload"))
        main_entry = SimpleNamespace(
            refresh_model_loading_parameters=lambda **kwargs: calls.append(
                ("refresh", kwargs)
            )
        )
        modules = ModuleType("modules")
        modules.sd_models = sd_models
        modules_forge = ModuleType("modules_forge")
        modules_forge.main_entry = main_entry
        logger = SimpleNamespace(
            warning=lambda *args, **kwargs: None,
            exception=lambda *args, **kwargs: None,
        )
        error_type = type("PromptAssistantError", (Exception,), {})
        loader = isolated_function(
            self.source,
            "_load_current_forge_selection",
            {
                "LOGGER": logger,
                "PromptAssistantError": error_type,
                "_forge_stack_status": lambda: (True, "**KREA2 stack ready.**"),
                "time": time,
                "torch": SimpleNamespace(OutOfMemoryError=RuntimeError),
            },
        )

        with patch.dict(
            sys.modules,
            {"modules": modules, "modules_forge": modules_forge},
        ):
            result = loader()

        self.assertEqual(calls, [("refresh", {"refresh": True}), "reload"])
        self.assertIn("KREA2 stack ready", result)

    def test_generation_cancel_request_lifecycle(self) -> None:
        namespace = {
            "GENERATION_CANCEL_LOCK": threading.RLock(),
            "GENERATION_CANCEL_EVENTS": {},
            "gr": SimpleNamespace(update=lambda **kwargs: kwargs),
            "threading": threading,
            "uuid": uuid,
        }
        for name in (
            "_begin_generation_request",
            "_generation_cancelled",
            "_request_generation_stop",
            "_clear_generation_request",
        ):
            isolated_function(self.source, name, namespace)

        generation_id = namespace["_begin_generation_request"]()
        self.assertFalse(namespace["_generation_cancelled"](generation_id))

        status, button_update = namespace["_request_generation_stop"](generation_id)
        self.assertIn("Stop requested", status)
        self.assertFalse(button_update["interactive"])
        self.assertTrue(namespace["_generation_cancelled"](generation_id))

        namespace["_clear_generation_request"](generation_id)
        self.assertFalse(namespace["_generation_cancelled"](generation_id))

    def test_both_generation_loops_check_for_cancellation(self) -> None:
        self.assertGreaterEqual(
            self.source.count("if _generation_cancelled(generation_id):"), 2
        )
        self.assertIn('variant="stop"', self.source)

    def test_text_instruction_is_added_to_the_user_message(self) -> None:
        error_type = type("PromptAssistantError", (Exception,), {})
        namespace = {
            "CHAT_SYSTEM_START": "<|im_start|>system\n",
            "CHAT_USER_BOUNDARY": "<|im_end|>\n<|im_start|>user\n",
            "DEFAULT_IMAGE_REQUEST": "Describe the uploaded image.",
            "PromptAssistantError": error_type,
            "_validate_system_prompt": lambda value: value,
        }
        render = isolated_function(self.source, "_render_generation_prompt", namespace)
        render_image = isolated_function(
            self.source, "_render_image_generation_prompt", namespace
        )
        engine = SimpleNamespace(
            llama_template=(
                "<|im_start|>system\noriginal<|im_end|>\n<|im_start|>user\n{}"
            ),
            vision_block="<|vision_start|><|image_pad|><|vision_end|>",
        )

        rendered = render(
            engine,
            "a glass observatory",
            "Use moonlight and a wide composition.",
            "custom persona",
        )

        self.assertIn("<|im_start|>system\ncustom persona", rendered)
        self.assertIn("a glass observatory", rendered)
        self.assertIn(
            "Additional instruction:\nUse moonlight and a wide composition.",
            rendered,
        )
        image_rendered = render_image(engine, "", "custom image persona")
        self.assertIn(engine.vision_block, image_rendered)
        self.assertIn("Describe the uploaded image.", image_rendered)

    def test_prompt_transfer_reports_target_and_action(self) -> None:
        transfer = isolated_function(self.source, "_transfer_prompt", {})

        updated, status = transfer(
            "new prompt", "existing prompt", mode="append", target_name="txt2img"
        )

        self.assertEqual(updated, "existing prompt\nnew prompt")
        self.assertIn("Appended", status)
        self.assertIn("txt2img", status)

    def test_requested_roadmap_areas_are_recorded(self) -> None:
        for heading in (
            "Local prompt history",
            "Refine this prompt",
            "Diagnostic report",
            "Z-Image adapter",
        ):
            self.assertIn(f"### {heading}", self.roadmap)


if __name__ == "__main__":
    unittest.main()
