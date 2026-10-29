"""Tests for MOSS-specific deterministic negative-prompt suggestions."""

# Copyright (C) 2026 vibecodingtoolmaker
# SPDX-License-Identifier: AGPL-3.0-only

import unittest

from forgy.adapters.moss_negatives import (
    MOSS_DOCUMENTED_NEGATIVE_EXAMPLE,
    MOSS_SAFE_NEGATIVE_CORE,
    suggest_moss_negative_prompt,
)


class MossNegativePromptTests(unittest.TestCase):
    def test_safe_core_is_a_subset_of_the_documented_example(self) -> None:
        self.assertEqual(
            MOSS_DOCUMENTED_NEGATIVE_EXAMPLE,
            (
                "nsfw",
                "worst quality",
                "bad anatomy",
                "watermark",
                "jpeg artifact",
            ),
        )
        self.assertTrue(
            set(MOSS_SAFE_NEGATIVE_CORE).issubset(MOSS_DOCUMENTED_NEGATIVE_EXAMPLE)
        )

    def test_penguins_get_animal_anatomy_but_no_human_hands_or_feet(self) -> None:
        result = suggest_moss_negative_prompt(
            "A colony of penguins crossing a windswept Antarctic ice shelf."
        )

        self.assertEqual(
            result,
            "worst quality, watermark, jpeg artifact, bad anatomy, extra limbs",
        )
        self.assertNotIn("hands", result)
        self.assertNotIn("feet", result)
        self.assertNotIn("fingers", result)

    def test_visible_human_anatomy_receives_specific_terms(self) -> None:
        result = suggest_moss_negative_prompt(
            "Full-body photo of a woman standing barefoot and holding a lantern, "
            "her face clearly visible."
        )

        for term in (
            "bad anatomy",
            "deformed hands",
            "extra fingers",
            "deformed feet",
            "extra toes",
            "deformed face",
            "asymmetrical eyes",
        ):
            self.assertIn(term, result)

    def test_illustrious_style_numbered_human_tags_are_detected(self) -> None:
        result = suggest_moss_negative_prompt(
            "1girl, solo, full_body, standing, detailed hands"
        )

        self.assertIn("bad anatomy", result)
        self.assertIn("deformed hands", result)
        self.assertIn("deformed feet", result)

    def test_requested_typography_gets_text_specific_terms(self) -> None:
        result = suggest_moss_negative_prompt(
            'A poster whose title reads "MOSS IS HERE" in bold typography.'
        )

        self.assertIn("garbled text", result)
        self.assertIn("misspelled text", result)
        self.assertIn("illegible text", result)

    def test_style_exclusions_are_not_added_for_a_hybrid_style(self) -> None:
        photo = suggest_moss_negative_prompt("A photorealistic wildlife photograph")
        illustration = suggest_moss_negative_prompt(
            "A watercolor illustration of a fox"
        )
        hybrid = suggest_moss_negative_prompt("A photorealistic anime illustration")

        self.assertIn("cartoon", photo)
        self.assertIn("photorealistic", illustration)
        self.assertNotIn("cartoon", hybrid)
        self.assertNotIn("photographic", hybrid)

    def test_nsfw_is_opt_in_only_for_an_explicit_safe_content_request(self) -> None:
        adult_art = suggest_moss_negative_prompt("An artistic nude adult woman")
        family_safe = suggest_moss_negative_prompt(
            "A family-friendly, fully clothed portrait of a woman"
        )

        self.assertNotIn("nsfw", adult_art)
        self.assertIn("nsfw", family_safe)
        self.assertIn("nudity", family_safe)

    def test_explicitly_requested_watermark_is_not_negated(self) -> None:
        result = suggest_moss_negative_prompt(
            "A visible watermark used as the central graphic element"
        )

        self.assertNotIn("watermark", result)
        self.assertIn("worst quality", result)

        exclusion = suggest_moss_negative_prompt("A clean poster without a watermark")
        self.assertIn("watermark", exclusion)

    def test_empty_input_has_no_suggestion(self) -> None:
        self.assertEqual(suggest_moss_negative_prompt("", None), "")


if __name__ == "__main__":
    unittest.main()
