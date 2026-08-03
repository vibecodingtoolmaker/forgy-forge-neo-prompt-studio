# Copyright (C) 2026 vibecodingtoolmaker
# SPDX-License-Identifier: AGPL-3.0-only

import ast
import json
from pathlib import Path
import unittest


ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts" / "forge_krea_prompt_assistant.py"
PERSONAS_EXAMPLE = ROOT / "personas.example.json"


def public_constants(source: str) -> dict[str, object]:
    tree = ast.parse(source, filename=str(SCRIPT))
    wanted = {"EXTENSION_VERSION", "DEFAULT_PERSONA_PROMPT", "VRAM_PROFILES"}
    values: dict[str, object] = {}

    for node in tree.body:
        if not isinstance(node, ast.Assign) or len(node.targets) != 1:
            continue
        target = node.targets[0]
        if isinstance(target, ast.Name) and target.id in wanted:
            values[target.id] = ast.literal_eval(node.value)

    return values


class PublicMetadataTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.source = SCRIPT.read_text(encoding="utf-8")
        cls.constants = public_constants(cls.source)
        cls.personas = json.loads(PERSONAS_EXAMPLE.read_text(encoding="utf-8"))

    def test_extension_script_compiles(self) -> None:
        compile(self.source, str(SCRIPT), "exec")

    def test_expected_constants_are_literal_and_present(self) -> None:
        self.assertEqual(
            set(self.constants),
            {"EXTENSION_VERSION", "DEFAULT_PERSONA_PROMPT", "VRAM_PROFILES"},
        )

    def test_public_alpha_version(self) -> None:
        self.assertEqual(self.constants["EXTENSION_VERSION"], "0.3.0-alpha.1")

    def test_example_default_matches_built_in_default(self) -> None:
        self.assertEqual(
            self.personas["personas"]["Default"],
            self.constants["DEFAULT_PERSONA_PROMPT"],
        )

    def test_vram_profiles(self) -> None:
        self.assertEqual(
            self.constants["VRAM_PROFILES"],
            {8: 4096, 12: 5120, 16: 6144, 24: 8192, 32: 12288},
        )

    def test_spdx_header_is_present(self) -> None:
        self.assertIn("SPDX-License-Identifier: AGPL-3.0-only", self.source[:300])


if __name__ == "__main__":
    unittest.main()
