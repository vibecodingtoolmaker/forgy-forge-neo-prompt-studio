"""Shared, dependency-free visible-output helpers for Qwen prompt backends."""

# Copyright (C) 2026 vibecodingtoolmaker
# SPDX-License-Identifier: AGPL-3.0-only

from __future__ import annotations

import re


def decode_generated_text(tokenizer, generated_ids) -> str:
    """Decode visibly while preserving reasoning markers for later filtering."""

    text = tokenizer.decode(generated_ids, skip_special_tokens=False)
    special_tokens = {
        str(token)
        for token in (getattr(tokenizer, "all_special_tokens", []) or [])
        if token
    }
    special_tokens.update({"<|endoftext|>", "<|im_start|>", "<|im_end|>"})
    for token in sorted(special_tokens, key=len, reverse=True):
        if re.fullmatch(r"</?(?:think|analysis)\s*>", str(token).strip(), re.I):
            continue
        text = text.replace(str(token), "")
    return text.strip()


def strip_model_thinking(text: str) -> tuple[str, bool]:
    """Remove visible Qwen reasoning before returning model output."""

    text = str(text or "")
    removed = False
    complete_block = re.compile(
        r"<(think|analysis)(?:\s[^>]*)?>.*?</\1\s*>",
        re.IGNORECASE | re.DOTALL,
    )
    while True:
        text, count = complete_block.subn("", text)
        if not count:
            break
        removed = True

    unmatched_open = re.search(
        r"<(?:think|analysis)(?:\s[^>]*)?>",
        text,
        re.IGNORECASE,
    )
    if unmatched_open:
        text = text[: unmatched_open.start()]
        removed = True

    text, dangling_closes = re.subn(
        r"</(?:think|analysis)\s*>",
        "",
        text,
        flags=re.IGNORECASE,
    )
    return text.strip(), removed or bool(dangling_closes)


def repetition_loop_suffix(token_ids: list[int]) -> tuple[int, int] | None:
    """Detect a repeated Qwen output suffix before it consumes the context."""

    token_count = len(token_ids)
    max_block_size = min(128, token_count // 2)
    for block_size in range(max_block_size, 7, -1):
        repeat_count = 2 if block_size >= 32 else 3
        repeated_size = block_size * repeat_count
        if token_count < repeated_size:
            continue
        block = token_ids[-block_size:]
        if all(
            token_ids[-(index + 1) * block_size : -index * block_size or None] == block
            for index in range(1, repeat_count)
        ):
            return block_size, repeat_count
    return None
