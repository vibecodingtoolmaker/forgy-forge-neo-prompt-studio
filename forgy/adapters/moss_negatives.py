"""Deterministic, subject-aware negative-prompt suggestions for MOSS-Image."""

# Copyright (C) 2026 vibecodingtoolmaker
# SPDX-License-Identifier: AGPL-3.0-only

from __future__ import annotations

import re
from collections.abc import Iterable


# The official MOSS user guide lists these terms as an effective example.  The
# content- and anatomy-specific entries are deliberately added only when the
# request calls for them; otherwise they can contradict the requested image.
MOSS_DOCUMENTED_NEGATIVE_EXAMPLE = (
    "nsfw",
    "worst quality",
    "bad anatomy",
    "watermark",
    "jpeg artifact",
)
MOSS_SAFE_NEGATIVE_CORE = ("worst quality", "watermark", "jpeg artifact")

_HUMAN_RE = re.compile(
    r"(?:\b\d+\s*(?:girl|girls|boy|boys|woman|women|man|men)\b|"
    r"\b(?:person|people|human|man|men|woman|women|boy|boys|girl|girls|"
    r"child|children|baby|babies|adult|adults|couple|crowd|family|"
    r"frau|frauen|mann|m[aä]nner|mensch|menschen|junge|m[aä]dchen|kind|kinder)\b)",
    re.IGNORECASE,
)
_ANIMAL_RE = re.compile(
    r"\b(?:animal|animals|wildlife|creature|creatures|bird|birds|penguin|"
    r"penguins|cat|cats|kitten|dog|dogs|puppy|horse|horses|fox|foxes|wolf|"
    r"wolves|bear|bears|lion|lions|tiger|tigers|elephant|elephants|deer|"
    r"rabbit|rabbits|fish|whale|whales|dolphin|dolphins|reptile|reptiles|"
    r"dragon|dragons|tier|tiere|wildtier|wildtiere|vogel|v[oö]gel|pinguin|"
    r"pinguine|katze|katzen|hund|hunde|pferd|pferde|fuchs|f[uü]chse|wolf|"
    r"w[oö]lfe|b[aä]r|b[aä]ren|l[oö]we|l[oö]wen|tiger|elefant|elefanten)\b",
    re.IGNORECASE,
)
_HANDS_RE = re.compile(
    r"\b(?:hand|hands|finger|fingers|holding|gripping|gesturing|"
    r"full[_ -]?body|standing|sitting|dancing|running|hand|h[aä]nde|finger|"
    r"haltend|greifend|ganzk[oö]rper|stehend|sitzend|tanzend|laufend)\b",
    re.IGNORECASE,
)
_FEET_RE = re.compile(
    r"\b(?:foot|feet|toe|toes|barefoot|full[_ -]?body|walking|standing|running|"
    r"fu[ßs]|f[uü][ßs]e|zehe|zehen|barfu[ßs]|ganzk[oö]rper|gehend|stehend|laufend)\b",
    re.IGNORECASE,
)
_FACE_RE = re.compile(
    r"\b(?:face|facial|eyes?|close[_ -]?up|headshot|gesicht|augen?|nahaufnahme)\b",
    re.IGNORECASE,
)
_TEXT_RE = re.compile(
    r"\b(?:text|typography|lettering|word|words|phrase|caption|title|logo|"
    r"signage|inscription|schrift|typografie|wort|w[oö]rter|satz|titel|logo|"
    r"beschriftung)\b|[\"“”„][^\"“”„]{1,80}[\"“”„]",
    re.IGNORECASE,
)
_PHOTO_RE = re.compile(
    r"\b(?:photo|photograph|photography|photorealistic|photo-realistic|"
    r"cinematic still|documentary still|dslr|fotograf(?:ie|isch)?|foto|"
    r"fotorealistisch)\b",
    re.IGNORECASE,
)
_ILLUSTRATION_RE = re.compile(
    r"\b(?:anime|manga|illustration|illustrated|drawing|painting|watercolou?r|"
    r"comic|cartoon|cel[ -]?shaded|line art|sketch|anime|manga|illustration|"
    r"zeichnung|gem[aä]lde|aquarell|comic|karikatur|skizze)\b",
    re.IGNORECASE,
)
_NO_PEOPLE_RE = re.compile(
    r"\b(?:no[_ ]people|without people|no[_ ]humans?|without humans?|uninhabited|"
    r"empty of people|menschenleer|ohne menschen|keine menschen|unbewohnt)\b",
    re.IGNORECASE,
)
_SAFE_CONTENT_RE = re.compile(
    r"\b(?:safe for work|safe content|rating[:_ ]safe|suitable for children|"
    r"family[_ -]?friendly|no nudity|"
    r"non[ -]?sexual|fully clothed|jugendfrei|familienfreundlich|keine nacktheit|"
    r"nicht sexuell|vollst[aä]ndig bekleidet)\b",
    re.IGNORECASE,
)
_WATERMARK_WANTED_RE = re.compile(
    r"\b(?:(?:visible|prominent|intentional|include|including|with|featuring)\s+"
    r"(?:an?\s+)?watermark|watermark\s+(?:as|used as|logo|motif|element))\b",
    re.IGNORECASE,
)
_JPEG_ARTIFACT_WANTED_RE = re.compile(
    r"\b(?:(?:intentional|visible|with|featuring)\s+jpeg artifacts?|"
    r"jpeg artifacts?\s+(?:as|used as|aesthetic|effect))\b",
    re.IGNORECASE,
)


def _append_unique(target: list[str], values: Iterable[str]) -> None:
    existing = {value.casefold() for value in target}
    for value in values:
        normalized = value.strip()
        key = normalized.casefold()
        if normalized and key not in existing:
            target.append(normalized)
            existing.add(key)


def suggest_moss_negative_prompt(*request_parts: object) -> str:
    """Build a stable MOSS negative prompt without asking the language model.

    The final positive prompt and the user's request can both be supplied.  The
    function never mutates or rewrites either one and intentionally stays
    conservative when it cannot classify a subject.
    """

    text = "\n".join(str(part or "").strip() for part in request_parts).strip()
    if not text:
        return ""

    suggestions: list[str] = []
    core = list(MOSS_SAFE_NEGATIVE_CORE)
    if _WATERMARK_WANTED_RE.search(text):
        core.remove("watermark")
    if _JPEG_ARTIFACT_WANTED_RE.search(text):
        core.remove("jpeg artifact")
    _append_unique(suggestions, core)

    has_human = bool(_HUMAN_RE.search(text))
    has_animal = bool(_ANIMAL_RE.search(text))
    if has_human or has_animal:
        _append_unique(suggestions, ("bad anatomy", "extra limbs"))
    if has_human and _HANDS_RE.search(text):
        _append_unique(
            suggestions,
            ("deformed hands", "extra fingers", "missing fingers", "fused fingers"),
        )
    if has_human and _FEET_RE.search(text):
        _append_unique(suggestions, ("deformed feet", "extra toes", "missing toes"))
    if has_human and _FACE_RE.search(text):
        _append_unique(suggestions, ("deformed face", "asymmetrical eyes"))

    if _TEXT_RE.search(text):
        _append_unique(
            suggestions,
            ("garbled text", "misspelled text", "illegible text"),
        )

    has_photo_style = bool(_PHOTO_RE.search(text))
    has_illustrated_style = bool(_ILLUSTRATION_RE.search(text))
    if has_photo_style and not has_illustrated_style:
        _append_unique(suggestions, ("cartoon", "illustration", "anime", "CGI"))
    elif has_illustrated_style and not has_photo_style:
        _append_unique(suggestions, ("photorealistic", "photographic"))

    if _NO_PEOPLE_RE.search(text):
        _append_unique(suggestions, ("people", "person", "human"))
    if _SAFE_CONTENT_RE.search(text):
        _append_unique(suggestions, ("nsfw", "nudity", "explicit sexual content"))

    return ", ".join(suggestions)


__all__ = [
    "MOSS_DOCUMENTED_NEGATIVE_EXAMPLE",
    "MOSS_SAFE_NEGATIVE_CORE",
    "suggest_moss_negative_prompt",
]
