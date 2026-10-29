"""Shared, dependency-free visible-output helpers for Qwen prompt backends."""

# Copyright (C) 2026 vibecodingtoolmaker
# SPDX-License-Identifier: AGPL-3.0-only

from __future__ import annotations

import re


QWEN_THINKING_SUFFIX = "<think>\n"
QWEN_NO_THINKING_SUFFIX = "<think>\n\n</think>\n\n"


def apply_model_thinking_mode(rendered_prompt: str, *, enabled: bool) -> str:
    """Select Qwen's thinking or no-thinking assistant-prefix contract."""

    rendered_prompt = str(rendered_prompt)
    if rendered_prompt.endswith(QWEN_NO_THINKING_SUFFIX):
        rendered_prompt = rendered_prompt[: -len(QWEN_NO_THINKING_SUFFIX)]
    return rendered_prompt + (
        QWEN_THINKING_SUFFIX if bool(enabled) else QWEN_NO_THINKING_SUFFIX
    )


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

    # In Qwen's thinking template the opening ``<think>`` is part of the input
    # prefix. Generated tokens therefore commonly contain reasoning followed by
    # only ``</think>`` and the visible answer.
    dangling_close = re.search(
        r"</(?:think|analysis)\s*>",
        text,
        flags=re.IGNORECASE,
    )
    if dangling_close:
        text = text[dangling_close.end() :]
        removed = True

    text = re.sub(
        r"</(?:think|analysis)\s*>",
        "",
        text,
        flags=re.IGNORECASE,
    )
    return text.strip(), removed


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
