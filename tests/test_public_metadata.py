# Copyright (C) 2026 vibecodingtoolmaker
# SPDX-License-Identifier: AGPL-3.0-only

import ast
import hashlib
import json
import os
from pathlib import Path
import re
import sys
from tempfile import TemporaryDirectory
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
REFINEMENT_PERSONAS_EXAMPLE = ROOT / "refinement_personas.example.json"
AGENT_PERSONAS_EXAMPLE = ROOT / "agent_personas.example.json"
ROADMAP = ROOT / "ROADMAP.md"
STYLE = ROOT / "style.css"
README = ROOT / "README.md"
METADATA = ROOT / "metadata.ini"
SECURITY = ROOT / "SECURITY.md"
LICENSE = ROOT / "LICENSE"


def public_constants(source: str) -> dict[str, object]:
    tree = ast.parse(source, filename=str(SCRIPT))
    wanted = {
        "EXTENSION_NAME",
        "EXTENSION_VERSION",
        "GHOST_PERSONA_NAME",
        "DEFAULT_PERSONA_PROMPT",
        "DEFAULT_IMAGE_PERSONA_PROMPT",
        "DEFAULT_REFINEMENT_PERSONA_PROMPT",
        "DEFAULT_AGENT_PERSONA_PROMPT",
        "FORGY_MAX_OUTPUT_TOKENS",
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
        cls.refinement_personas = json.loads(
            REFINEMENT_PERSONAS_EXAMPLE.read_text(encoding="utf-8")
        )
        cls.agent_personas = json.loads(
            AGENT_PERSONAS_EXAMPLE.read_text(encoding="utf-8")
        )
        cls.roadmap = ROADMAP.read_text(encoding="utf-8")
        cls.style = STYLE.read_text(encoding="utf-8")
        cls.readme = README.read_text(encoding="utf-8")
        cls.metadata = METADATA.read_text(encoding="utf-8")
        cls.security = SECURITY.read_text(encoding="utf-8")
        cls.license = LICENSE.read_text(encoding="utf-8")

    def test_extension_script_compiles(self) -> None:
        compile(self.source, str(SCRIPT), "exec")

    def test_expected_constants_are_literal_and_present(self) -> None:
        self.assertEqual(
            set(self.constants),
            {
                "EXTENSION_NAME",
                "EXTENSION_VERSION",
                "GHOST_PERSONA_NAME",
                "DEFAULT_PERSONA_PROMPT",
                "DEFAULT_IMAGE_PERSONA_PROMPT",
                "DEFAULT_REFINEMENT_PERSONA_PROMPT",
                "DEFAULT_AGENT_PERSONA_PROMPT",
                "FORGY_MAX_OUTPUT_TOKENS",
                "VRAM_PROFILES",
            },
        )

    def test_public_beta_branding_and_version(self) -> None:
        self.assertEqual(
            self.constants["EXTENSION_NAME"],
            "Forgy — Forge Neo Prompt Studio",
        )
        self.assertEqual(self.constants["EXTENSION_VERSION"], "0.5.0-beta.1")
        self.assertIn("# Forgy — Forge Neo Prompt Studio", self.readme)
        self.assertIn("**Beta 0.5.0-beta.1**", self.readme)
        self.assertIn("Name = Forgy — Forge Neo Prompt Studio", self.metadata)
        self.assertIn("Forgy — Forge Neo Prompt Studio", self.security)
        self.assertTrue(self.license.startswith("Forgy — Forge Neo Prompt Studio\n"))
        self.assertIn(
            "https://github.com/vibecodingtoolmaker/forgy-forge-neo-prompt-studio",
            self.readme,
        )
        self.assertIn(
            'gr.Markdown(f"### {EXTENSION_NAME} — {EXTENSION_VERSION}")',
            self.source,
        )
        self.assertIn(
            'return [(tab, EXTENSION_NAME, "forge_krea_prompt_assistant")]',
            self.source,
        )

    def test_general_workflow_language_is_model_family_neutral(self) -> None:
        self.assertIn(
            "Create, refine, and discuss your prompts with the active text encoder",
            self.source,
        )
        self.assertIn(
            'f"**{EXTENSION_NAME} is not ready.** Select a supported model family, "',
            self.source,
        )
        self.assertIn("No supported model stack is active.", self.source)
        self.assertIn("Text encoder / VAE:", self.source)
        for outdated_text in (
            "KREA2 Prompt Assistant",
            "KREA2 stack ready",
            "KREA2 stack not ready",
            "Generated KREA2 prompt",
            "Create, refine, and discuss KREA2 prompts",
            "The Qwen3-VL text encoder is generating",
            "The Qwen3-VL vision encoder is analyzing",
        ):
            self.assertNotIn(outdated_text, self.source)
        for prompt in (
            self.constants["DEFAULT_PERSONA_PROMPT"],
            self.constants["DEFAULT_IMAGE_PERSONA_PROMPT"],
            self.constants["DEFAULT_REFINEMENT_PERSONA_PROMPT"],
            self.constants["DEFAULT_AGENT_PERSONA_PROMPT"],
        ):
            self.assertNotIn("KREA2", prompt)

    def test_forgy_allows_long_iterative_prompt_turns(self) -> None:
        self.assertEqual(self.constants["FORGY_MAX_OUTPUT_TOKENS"], 4096)
        self.assertIn(
            "forgy_sampling_controls[1].value = min(\n"
            "                    initial_context_limit, FORGY_MAX_OUTPUT_TOKENS",
            self.source,
        )

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

    def test_refinement_example_default_matches_built_in_default(self) -> None:
        self.assertEqual(
            self.refinement_personas["personas"]["Default"],
            self.constants["DEFAULT_REFINEMENT_PERSONA_PROMPT"],
        )

    def test_agent_example_default_matches_built_in_default(self) -> None:
        self.assertEqual(
            self.agent_personas["personas"]["Default"],
            self.constants["DEFAULT_AGENT_PERSONA_PROMPT"],
        )

    def test_unchanged_legacy_defaults_migrate_without_touching_custom_personas(
        self,
    ) -> None:
        legacy_default = "Legacy built-in prompt"
        neutral_default = "Neutral built-in prompt"
        legacy_hash = hashlib.sha256(legacy_default.encode("utf-8")).hexdigest()
        error_type = type("PromptAssistantError", (Exception,), {})
        namespace = {
            "DEFAULT_PERSONA_NAME": "Default",
            "GHOST_PERSONA_NAME": "Ghost",
            "PERSONA_STORE_VERSION": 1,
            "MAX_PERSONAS": 100,
            "MAX_PERSONA_NAME_LENGTH": 80,
            "MAX_SYSTEM_PROMPT_LENGTH": 16_000,
            "PERSONAS_PATH": Path("unused-personas.json"),
            "DEFAULT_PERSONA_PROMPT": neutral_default,
            "LEGACY_DEFAULT_PROMPT_MIGRATIONS": {legacy_hash: neutral_default},
            "PERSONA_LOCK": threading.RLock(),
            "PromptAssistantError": error_type,
            "LOGGER": SimpleNamespace(info=lambda *args, **kwargs: None),
            "hashlib": hashlib,
            "json": json,
            "os": os,
        }
        for function_name in (
            "_validate_persona_name",
            "_validate_system_prompt",
            "_default_persona_store",
            "_write_persona_store",
            "_read_persona_store",
        ):
            isolated_function(self.source, function_name, namespace)

        with TemporaryDirectory() as temporary_directory:
            path = Path(temporary_directory) / "personas.json"
            path.write_text(
                json.dumps(
                    {
                        "version": 1,
                        "personas": {
                            "Default": legacy_default,
                            "Ghost": "",
                            "User style": "Keep this custom prompt.",
                        },
                    }
                ),
                encoding="utf-8",
            )
            migrated = namespace["_read_persona_store"](path, neutral_default)
            persisted = json.loads(path.read_text(encoding="utf-8"))

        self.assertEqual(migrated["personas"]["Default"], neutral_default)
        self.assertEqual(migrated["personas"]["User style"], "Keep this custom prompt.")
        self.assertEqual(persisted, migrated)

    def test_all_workflows_include_the_empty_ghost_persona(self) -> None:
        self.assertEqual(self.constants["GHOST_PERSONA_NAME"], "Ghost")
        for personas in (
            self.personas,
            self.image_personas,
            self.refinement_personas,
            self.agent_personas,
        ):
            self.assertIn("Ghost", personas["personas"])
            self.assertEqual(personas["personas"]["Ghost"], "")

    def test_only_explicit_runtime_paths_may_accept_an_empty_system_prompt(
        self,
    ) -> None:
        error_type = type("PromptAssistantError", (Exception,), {})
        validate = isolated_function(
            self.source,
            "_validate_system_prompt",
            {
                "MAX_SYSTEM_PROMPT_LENGTH": 16_000,
                "PromptAssistantError": error_type,
            },
        )
        with self.assertRaises(error_type):
            validate("")
        self.assertEqual(validate("", allow_empty=True), "")

        system_start = "<|im_start|>system\n"
        user_boundary = "<|im_end|>\n<|im_start|>user\n"
        render = isolated_function(
            self.source,
            "_render_generation_prompt",
            {
                "CHAT_SYSTEM_START": system_start,
                "CHAT_USER_BOUNDARY": user_boundary,
                "PromptAssistantError": error_type,
                "_validate_system_prompt": validate,
            },
        )
        engine = SimpleNamespace(
            llama_template=(
                f"{system_start}Built-in instructions{user_boundary}"
                "{}<|im_end|>\n<|im_start|>assistant\n"
            )
        )
        rendered = render(engine, "A quiet lake", "", "")
        self.assertIn(f"{system_start}{user_boundary}A quiet lake", rendered)

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
                "_forge_stack_status": lambda: (
                    True,
                    "**Forgy — Forge Neo Prompt Studio is ready.**",
                ),
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
        self.assertIn("Forgy — Forge Neo Prompt Studio is ready", result)

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
            "_validate_system_prompt": lambda value, **_kwargs: value,
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

        updated, status = transfer(
            "new prompt",
            "existing prompt",
            mode="replace",
            target_name="img2img",
            start_generation=True,
        )
        self.assertEqual(updated, "new prompt")
        self.assertTrue(status.startswith("✅"))
        self.assertIn("Starting image generation", status)

    def test_generate_after_transfer_uses_forge_native_generate_buttons(self) -> None:
        build_js = isolated_function(self.source, "_forge_generate_click_js", {})
        txt2img_js = build_js("txt2img_generate", "txt2img")
        img2img_js = build_js("img2img_generate", "img2img")

        self.assertIn("async (generatedPrompt)", txt2img_js)
        self.assertIn("generatedPrompt.trim()", txt2img_js)
        self.assertNotIn("status.startsWith", txt2img_js)
        self.assertIn('root.querySelector("#txt2img_generate")', txt2img_js)
        self.assertIn('root.querySelector("#img2img_generate")', img2img_js)
        self.assertIn('root.querySelector("#txt2img_interrupt")', txt2img_js)
        self.assertIn('root.querySelector("#txt2img_skip")', txt2img_js)
        self.assertIn('root.querySelector("#txt2img_interrupting")', txt2img_js)
        self.assertIn("gradioApp()", txt2img_js)
        self.assertIn("requestAnimationFrame", txt2img_js)
        self.assertIn("generateButton.click()", txt2img_js)
        self.assertIn("waitFor(isGenerating, 15000)", txt2img_js)
        self.assertIn("waitFor(() => !isGenerating(), 43200000)", txt2img_js)
        self.assertIn("gallerySignature() !== previousGallery", txt2img_js)
        self.assertIn("Forge created a new **txt2img** gallery image", txt2img_js)
        self.assertIn("inputs=[result]", self.source)
        self.assertIn("outputs=[feedback, button]", self.source)
        self.assertIn("_finish_forge_generation", self.source)
        self.assertIn('"txt2img image generated"', self.source)
        self.assertIn('"img2img image generated"', self.source)
        self.assertEqual(
            self.source.count('"txt2img: replace + generate"'),
            5,
        )
        self.assertEqual(
            self.source.count('"img2img: replace + generate"'),
            5,
        )

    def test_wide_workflows_group_transfer_and_generate_buttons_in_four_columns(
        self,
    ) -> None:
        rows = []
        for node in ast.walk(ast.parse(self.source, filename=str(SCRIPT))):
            if not isinstance(node, ast.With):
                continue
            if not any(
                isinstance(item.context_expr, ast.Call)
                and isinstance(item.context_expr.func, ast.Attribute)
                and item.context_expr.func.attr == "Row"
                for item in node.items
            ):
                continue
            names = []
            for statement in node.body:
                if (
                    isinstance(statement, ast.Assign)
                    and len(statement.targets) == 1
                    and isinstance(statement.targets[0], ast.Name)
                    and isinstance(statement.value, ast.Call)
                    and isinstance(statement.value.func, ast.Attribute)
                    and statement.value.func.attr == "Button"
                ):
                    names.append(statement.targets[0].id)
            if names:
                rows.append(tuple(names))

        self.assertIn(
            (
                "txt_replace",
                "txt_replace_generate",
                "txt_append",
                "txt_append_generate",
            ),
            rows,
        )
        self.assertIn(
            (
                "img_replace",
                "img_replace_generate",
                "img_append",
                "img_append_generate",
            ),
            rows,
        )
        self.assertIn(
            (
                "image_txt_replace",
                "image_txt_replace_generate",
                "image_txt_append",
                "image_txt_append_generate",
            ),
            rows,
        )
        self.assertIn(
            (
                "image_img_replace",
                "image_img_replace_generate",
                "image_img_append",
                "image_img_append_generate",
            ),
            rows,
        )
        self.assertIn(
            (
                "refinement_txt_replace",
                "refinement_txt_replace_generate",
                "refinement_txt_append",
                "refinement_txt_append_generate",
            ),
            rows,
        )
        self.assertIn(
            (
                "refinement_img_replace",
                "refinement_img_replace_generate",
                "refinement_img_append",
                "refinement_img_append_generate",
            ),
            rows,
        )
        self.assertIn(
            (
                "forgy_txt_replace",
                "forgy_txt_replace_generate",
                "forgy_txt_append",
                "forgy_txt_append_generate",
            ),
            rows,
        )
        self.assertIn(
            (
                "forgy_img_replace",
                "forgy_img_replace_generate",
                "forgy_img_append",
                "forgy_img_append_generate",
            ),
            rows,
        )
        self.assertGreaterEqual(self.source.count("min_width=0"), 8)

    def test_primary_workflows_put_independent_persona_management_below_actions(
        self,
    ) -> None:
        idea_start = self.source.index('with gr.Tab("Idea to prompt"):')
        image_start = self.source.index('with gr.Tab("Image to prompt"):')
        refine_start = self.source.index('with gr.Tab("Refine prompt"):')
        idea_block = self.source[idea_start:image_start]
        image_block = self.source[image_start:refine_start]

        self.assertLess(
            idea_block.index("_create_active_persona_controls("),
            idea_block.index("idea = gr.Textbox("),
        )
        self.assertLess(
            idea_block.index('prompt_action_status = gr.State("")'),
            idea_block.index("_create_persona_manager("),
        )
        self.assertLess(
            image_block.index("_create_active_persona_controls("),
            image_block.index("source_image = gr.Image("),
        )
        self.assertLess(
            image_block.index('image_prompt_action_status = gr.State("")'),
            image_block.index("_create_persona_manager("),
        )
        self.assertIn('label="Persona to edit"', self.source)
        self.assertIn("_connect_separate_persona_controls(", self.source)
        self.assertIn("_refresh_active_persona", self.source)

    def test_image_instruction_shares_a_compact_row_with_reference_image(self) -> None:
        image_start = self.source.index('with gr.Tab("Image to prompt"):')
        refine_start = self.source.index('with gr.Tab("Refine prompt"):')
        image_block = self.source[image_start:refine_start]

        row_start = image_block.index('elem_classes=["forge-krea-image-input-row"]')
        image_position = image_block.index("source_image = gr.Image(", row_start)
        instruction_position = image_block.index(
            "image_instruction = gr.Textbox(", image_position
        )
        generation_position = image_block.index(
            "image_generate = gr.Button(", instruction_position
        )

        self.assertLess(image_position, instruction_position)
        self.assertLess(instruction_position, generation_position)
        self.assertIn("equal_height=True", image_block)
        self.assertIn("height=380", image_block)
        self.assertIn("lines=12", image_block)

    def test_refinement_uses_compact_sources_and_bottom_persona_management(
        self,
    ) -> None:
        refine_start = self.source.index('with gr.Tab("Refine prompt"):')
        event_wiring_start = self.source.index(
            "\n        (\n            vram_profile,", refine_start
        )
        refine_block = self.source[refine_start:event_wiring_start]

        self.assertLess(
            refine_block.index("_create_active_persona_controls("),
            refine_block.index("prompt_to_refine = gr.Textbox("),
        )
        self.assertLess(
            refine_block.index("prompt_to_refine = gr.Textbox("),
            refine_block.index("load_text_for_refinement = gr.Button("),
        )
        self.assertLess(
            refine_block.index('refinement_prompt_action_status = gr.State("")'),
            refine_block.index("_create_persona_manager("),
        )
        self.assertIn("with gr.Column(scale=4, min_width=360):", refine_block)
        self.assertIn("with gr.Column(scale=1, min_width=190):", refine_block)
        self.assertIn("refinement_active_persona_controls", self.source)
        self.assertIn("_refresh_active_refinement_persona", self.source)

    def test_navigation_buttons_share_compact_side_columns_with_prompt_fields(
        self,
    ) -> None:
        idea_start = self.source.index('with gr.Tab("Idea to prompt"):')
        image_start = self.source.index('with gr.Tab("Image to prompt"):')
        refine_start = self.source.index('with gr.Tab("Refine prompt"):')
        event_wiring_start = self.source.index(
            "\n        (\n            vram_profile,", refine_start
        )
        blocks = {
            "idea": self.source[idea_start:image_start],
            "image": self.source[image_start:refine_start],
            "refine": self.source[refine_start:event_wiring_start],
        }

        expected = {
            "idea": (
                "result = gr.Textbox(",
                "text_send_to_refinement = gr.Button(",
                "text_send_to_forgy = gr.Button(",
                "status = gr.Markdown()",
            ),
            "image": (
                "image_result = gr.Textbox(",
                "image_send_to_refinement = gr.Button(",
                "image_send_to_forgy = gr.Button(",
                "image_status = gr.Markdown()",
            ),
            "refine": (
                "prompt_to_refine = gr.Textbox(",
                "load_text_for_refinement = gr.Button(",
                "refinement_send_to_forgy = gr.Button(",
                "refinement_load_status = gr.Markdown()",
            ),
        }
        for name, markers in expected.items():
            block = blocks[name]
            positions = [block.index(marker) for marker in markers]
            self.assertEqual(positions, sorted(positions), name)
            compact_region = block[positions[0] : positions[-1]]
            self.assertIn("with gr.Column(scale=1, min_width=190):", compact_region)

    def test_friendly_input_labels_and_bottom_vram_information(self) -> None:
        forgy_start = self.source.index('with gr.Tab("Forgy Chat"):')
        idea_start = self.source.index('with gr.Tab("Idea to prompt"):')
        image_start = self.source.index('with gr.Tab("Image to prompt"):')
        refine_start = self.source.index('with gr.Tab("Refine prompt"):')
        event_wiring_start = self.source.index(
            "\n        (\n            vram_profile,", refine_start
        )
        blocks = {
            "forgy": self.source[forgy_start:idea_start],
            "idea": self.source[idea_start:image_start],
            "image": self.source[image_start:refine_start],
            "refine": self.source[refine_start:event_wiring_start],
        }
        vram_markers = {
            "forgy": "forgy_vram_status = gr.Markdown(",
            "idea": "vram_status = gr.Markdown(",
            "image": "image_vram_status = gr.Markdown(",
            "refine": "refinement_vram_status = gr.Markdown(",
        }

        self.assertIn(
            'label="Write down your ideas for a picture and let Forgy do its magic"',
            blocks["idea"],
        )
        friendly_instruction = '"Optional instructions to refine your vision alongside'
        self.assertIn(friendly_instruction, blocks["idea"])
        self.assertIn(friendly_instruction, blocks["image"])
        for name, block in blocks.items():
            sampling_position = block.index("_create_sampling_controls(")
            vram_position = block.index(vram_markers[name])
            self.assertLess(sampling_position, vram_position, name)
            self.assertNotIn("gr.Button(", block[vram_position:], name)

    def test_refinement_request_is_explicit_and_requires_both_fields(self) -> None:
        error_type = type("PromptAssistantError", (Exception,), {})
        refinement_message = isolated_function(
            self.source,
            "_refinement_user_message",
            {"PromptAssistantError": error_type},
        )

        message = refinement_message(
            "A portrait under soft window light.",
            "Change the lighting to dramatic moonlight.",
        )

        self.assertIn("Existing prompt:\nA portrait", message)
        self.assertIn("Refinement instruction:\nChange the lighting", message)
        with self.assertRaises(error_type):
            refinement_message("", "Make it cinematic.")
        with self.assertRaises(error_type):
            refinement_message("A cinematic portrait.", "")

    def test_refinement_has_an_independent_ui_and_persona_store(self) -> None:
        self.assertIn('with gr.Tab("Refine prompt"):', self.source)
        self.assertIn("REFINEMENT_PERSONAS_PATH", self.source)
        self.assertIn("_run_refinement", self.source)
        button_labels = []
        for node in ast.walk(ast.parse(self.source, filename=str(SCRIPT))):
            if (
                isinstance(node, ast.Call)
                and isinstance(node.func, ast.Attribute)
                and node.func.attr == "Button"
                and node.args
                and isinstance(node.args[0], ast.Constant)
            ):
                button_labels.append(node.args[0].value)
        self.assertEqual(button_labels.count("Send to prompt refinement"), 2)

    def test_refinement_preserves_the_existing_prompt_after_failure(self) -> None:
        error_type = type("PromptAssistantError", (Exception,), {})
        run_refinement = isolated_function(
            self.source,
            "_run_refinement",
            {
                "PromptAssistantError": error_type,
                "_clear_generation_request": lambda generation_id: None,
                "_refinement_user_message": lambda prompt, instruction: "request",
                "_run_generation": lambda *args: ("", "**Error:** simulated"),
            },
        )

        prompt, status = run_refinement(
            "Original prompt",
            "Change the lighting",
            "Default",
            "system prompt",
            "Auto",
            1024,
            True,
            0.7,
            64,
            0.95,
            0.05,
            1.05,
            0,
            "generation-id",
        )

        self.assertEqual(prompt, "Original prompt")
        self.assertIn("simulated", status)

    def test_forgy_request_contains_working_state_and_recent_chat(self) -> None:
        namespace = {
            "FORGY_MAX_CONTEXT_TURNS": 2,
            "FORGY_MAX_STORED_TURNS": 50,
            "QWEN_NO_THINK_DIRECTIVE": "/no_think",
            "PromptAssistantError": type("PromptAssistantError", (Exception,), {}),
        }
        normalize = isolated_function(
            self.source, "_normalize_forgy_history", namespace
        )
        namespace["_normalize_forgy_history"] = normalize
        build = isolated_function(self.source, "_build_forgy_request", namespace)
        history = [
            {"role": "user", "content": "first"},
            {"role": "assistant", "content": "second"},
            {"role": "user", "content": "third"},
            {"role": "assistant", "content": "fourth"},
            {"role": "user", "content": "fifth"},
            {"role": "assistant", "content": "sixth"},
        ]

        request, omitted_turns = build(
            "Move the camera farther away.", "Current portrait prompt.", history
        )

        self.assertIn("Current portrait prompt.", request)
        self.assertIn("Move the camera farther away.", request)
        self.assertNotIn("first", request)
        self.assertIn("sixth", request)
        self.assertTrue(request.rstrip().endswith("/no_think"))
        self.assertEqual(omitted_turns, 1)

    def test_internal_thinking_is_removed_before_visible_output(self) -> None:
        strip_thinking = isolated_function(
            self.source,
            "_strip_model_thinking",
            {"re": re},
        )

        cleaned, removed = strip_thinking(
            "<think>private chain of thought</think>\n"
            "FORGY_REPLY:\nVisible answer.\n\n"
            "UPDATED_PROMPT:\n[UNCHANGED]"
        )
        self.assertTrue(removed)
        self.assertNotIn("private chain of thought", cleaned)
        self.assertIn("FORGY_REPLY:\nVisible answer.", cleaned)

        cleaned, removed = strip_thinking("<analysis>unfinished private reasoning")
        self.assertTrue(removed)
        self.assertEqual(cleaned, "")

    def test_controlled_decode_keeps_reasoning_markers_for_filtering(self) -> None:
        class FakeTokenizer:
            all_special_tokens = ["<|im_end|>", "<think>", "</think>"]

            @staticmethod
            def decode(token_ids, *, skip_special_tokens):
                del token_ids
                self.assertFalse(skip_special_tokens)
                return "<think>private</think>Visible<|im_end|>"

        decode = isolated_function(
            self.source,
            "_decode_generated_text",
            {"re": re},
        )
        decoded = decode(FakeTokenizer(), [1, 2, 3])

        self.assertEqual(decoded, "<think>private</think>Visible")

    def test_repetition_loop_detection_stops_repeated_suffixes(self) -> None:
        detect = isolated_function(
            self.source,
            "_repetition_loop_suffix",
            {},
        )
        block = list(range(32))

        self.assertEqual(detect([999, *block, *block]), (32, 2))
        self.assertIsNone(detect(list(range(100))))

    def test_forgy_response_parser_preserves_or_updates_prompt_safely(self) -> None:
        namespace = {
            "FORGY_REPLY_MARKER": "FORGY_REPLY:",
            "FORGY_PROMPT_MARKER": "UPDATED_PROMPT:",
            "FORGY_UNCHANGED_MARKER": "[UNCHANGED]",
        }
        strip_fence = isolated_function(self.source, "_strip_forgy_fence", namespace)
        namespace["_strip_forgy_fence"] = strip_fence
        parse = isolated_function(self.source, "_parse_forgy_response", namespace)

        reply, prompt, parsed = parse(
            "FORGY_REPLY:\nI widened the shot.\n\nUPDATED_PROMPT:\nA wide shot.",
            "A portrait.",
        )
        self.assertEqual(reply, "I widened the shot.")
        self.assertEqual(prompt, "A wide shot.")
        self.assertTrue(parsed)

        _, unchanged, parsed = parse(
            "FORGY_REPLY:\nNo change needed.\n\nUPDATED_PROMPT:\n[UNCHANGED]",
            "A portrait.",
        )
        self.assertEqual(unchanged, "A portrait.")
        self.assertTrue(parsed)

        raw, preserved, parsed = parse("Unstructured reply", "A portrait.")
        self.assertEqual(raw, "Unstructured reply")
        self.assertEqual(preserved, "A portrait.")
        self.assertFalse(parsed)

    def test_forgy_prompt_versions_ignore_empty_values_and_stay_bounded(self) -> None:
        normalize = isolated_function(
            self.source,
            "_normalize_prompt_versions",
            {"FORGY_MAX_PROMPT_VERSIONS": 2},
        )

        self.assertEqual(
            normalize([None, "", " first ", "second", "third"]),
            ["second", "third"],
        )

    def test_forgy_ui_and_session_controls_are_present(self) -> None:
        forgy_start = self.source.index('with gr.Tab("Forgy Chat"):')
        idea_start = self.source.index('with gr.Tab("Idea to prompt"):')
        forgy_block = self.source[forgy_start:idea_start]

        self.assertLess(forgy_start, idea_start)
        self.assertIn('type="messages"', self.source)
        self.assertIn("AGENT_PERSONAS_PATH", self.source)
        self.assertIn("_run_forgy_turn", self.source)
        self.assertIn("forgy_undo", self.source)
        self.assertIn("forgy_clear_chat", self.source)
        self.assertLess(
            forgy_block.index("forgy_send = gr.Button("),
            forgy_block.index("forgy_undo = gr.Button("),
        )
        self.assertLess(
            forgy_block.index("forgy_clear_chat = gr.Button("),
            forgy_block.index('forgy_generation_id = gr.State("")'),
        )
        self.assertIn('"Grab last generated image"', self.source)
        self.assertIn("_capture_forge_gallery_component", self.source)
        self.assertIn("script_callbacks.on_after_component(", self.source)
        self.assertIn("max_output_cap=FORGY_MAX_OUTPUT_TOKENS", self.source)
        self.assertIn("agent_active_persona_controls", forgy_block)
        self.assertIn('label="Forgy\'s prompt output and working input"', forgy_block)
        self.assertIn("Paste an existing prompt here", forgy_block)
        self.assertNotIn('gr.Accordion("Optional image attachment"', forgy_block)
        self.assertLess(
            forgy_block.index('"**Optional image attachment**'),
            forgy_block.index("forgy_image = gr.Image("),
        )
        self.assertLess(
            forgy_block.index('forgy_prompt_action_status = gr.State("")'),
            forgy_block.index("agent_persona_controls = _create_persona_manager("),
        )
        self.assertIn("**Prompt history location:**", forgy_block)
        self.assertIn("not written to a local file", forgy_block)
        self.assertIn("_refresh_active_agent_persona", self.source)
        self.assertGreaterEqual(
            self.source.count('finish_reason = "repetition_stop"'), 2
        )

    def test_forgy_chat_scrolls_to_the_latest_message(self) -> None:
        build_js = isolated_function(self.source, "_forgy_scroll_to_bottom_js", {})
        scroll_js = build_js("forge_krea_forgy_chat")

        self.assertIn('getElementById("forge_krea_forgy_chat")', scroll_js)
        self.assertIn('[role="log"][aria-label="chatbot conversation"]', scroll_js)
        self.assertIn("conversation.scrollTop = conversation.scrollHeight", scroll_js)
        self.assertIn("requestAnimationFrame", scroll_js)
        self.assertIn("setTimeout(scrollToBottom, 600)", scroll_js)
        self.assertIn("forgy_chat.change(", self.source)
        self.assertIn("elem_id=FORGY_CHAT_ELEMENT_ID", self.source)

    def test_workflow_defaults_are_isolated_from_forge_ui_config_collisions(
        self,
    ) -> None:
        self.assertIn("persona_select.do_not_save_to_config = True", self.source)
        self.assertIn("persona_name.do_not_save_to_config = True", self.source)
        self.assertIn("persona_prompt.do_not_save_to_config = True", self.source)
        self.assertIn(
            "forgy_sampling_controls[1].do_not_save_to_config = True",
            self.source,
        )
        self.assertIn(
            "forgy_sampling_controls[7].do_not_save_to_config = True",
            self.source,
        )

    def test_forgy_grabs_the_last_image_from_the_latest_forge_gallery(self) -> None:
        class FakeImage:
            def __init__(self, color):
                self.color = color

            def convert(self, _mode):
                return self

            def copy(self):
                return FakeImage(self.color)

            def getpixel(self, _position):
                return self.color

        logger = SimpleNamespace(exception=lambda *args, **kwargs: None)
        temp_checks = []
        namespace = {
            "Image": SimpleNamespace(Image=FakeImage),
            "ImageOps": SimpleNamespace(exif_transpose=lambda image: image),
            "LOGGER": logger,
            "Path": Path,
            "np": SimpleNamespace(ndarray=type("FakeArray", (), {})),
            "_ensure_forge_temp_directory": lambda: temp_checks.append(True),
        }
        latest = isolated_function(self.source, "_latest_gallery_image", namespace)
        namespace["_latest_gallery_image"] = latest
        skip_marker = object()
        namespace["gr"] = SimpleNamespace(skip=lambda: skip_marker)
        grab = isolated_function(self.source, "_grab_last_forge_image", namespace)

        red = FakeImage((255, 0, 0))
        green = FakeImage((0, 255, 0))
        blue = FakeImage((0, 0, 255))
        image, status = grab(
            "img2img",
            [(red, None), (green, None)],
            [(blue, None)],
        )

        self.assertEqual(image.getpixel((0, 0)), (0, 0, 255))
        self.assertIn("img2img", status)
        self.assertIsNot(image, blue)
        self.assertEqual(temp_checks, [True])

        missing, status = grab("", None, [])
        self.assertIs(missing, skip_marker)
        self.assertIn("Generate an image", status)

        namespace["_ensure_forge_temp_directory"] = lambda: "**Not loaded:** temp"
        blocked, status = grab("txt2img", [(red, None)], [])
        self.assertIs(blocked, skip_marker)
        self.assertIn("temp", status)

    def test_forgy_creates_the_configured_forge_temp_directory(self) -> None:
        logger = SimpleNamespace(exception=lambda *args, **kwargs: None)
        with TemporaryDirectory() as root:
            temp_dir = Path(root) / "missing" / "forge-temp"
            shared = SimpleNamespace(opts=SimpleNamespace(temp_dir=str(temp_dir)))
            ensure_temp = isolated_function(
                self.source,
                "_ensure_forge_temp_directory",
                {"LOGGER": logger, "Path": Path, "shared": shared},
            )

            self.assertIsNone(ensure_temp())
            self.assertTrue(temp_dir.is_dir())

            blocker = Path(root) / "not-a-directory"
            blocker.write_text("blocked", encoding="utf-8")
            shared.opts.temp_dir = str(blocker / "tmp")
            self.assertIn("Not loaded", ensure_temp())

    def test_persona_dropdowns_have_a_visible_scroll_area(self) -> None:
        self.assertIn('elem_classes=["forge-krea-persona-dropdown"]', self.source)
        self.assertIn(".forge-krea-persona-dropdown ul.options", self.style)
        self.assertIn("overflow-y: scroll", self.style)
        self.assertIn("scrollbar-gutter: stable", self.style)

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
