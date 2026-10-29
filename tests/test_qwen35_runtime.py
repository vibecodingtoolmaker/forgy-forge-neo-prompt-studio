# Copyright 2025 The Qwen Team and The HuggingFace Inc. team.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import unittest

try:
    import torch
except ModuleNotFoundError:
    torch = None

if torch is not None:
    from forgy.adapters.qwen35_runtime import (
        Qwen35RuntimeError,
        ROPE_THETA,
        ROTARY_DIM,
        _rotary_embeddings,
        moss_backbone_compute_dtype,
        qwen35_rms_norm,
    )
else:
    Qwen35RuntimeError = None
    ROPE_THETA = None
    ROTARY_DIM = None
    _rotary_embeddings = None
    moss_backbone_compute_dtype = None
    qwen35_rms_norm = None


@unittest.skipIf(torch is None, "PyTorch is available only inside Forge.")
class Qwen35RuntimeTests(unittest.TestCase):
    def test_moss_runtime_uses_the_transformer_dtype_not_embedding_dtype(self) -> None:
        class TinyMossModel(torch.nn.Module):
            def __init__(self) -> None:
                super().__init__()
                self.embed_tokens = torch.nn.Embedding(8, 4, dtype=torch.float16)
                self.layers = torch.nn.ModuleList(
                    [torch.nn.Linear(4, 4, bias=False, dtype=torch.bfloat16)]
                )
                self.norm = torch.nn.LayerNorm(4, dtype=torch.bfloat16)

        encoder = type("Encoder", (), {"model": TinyMossModel()})()

        self.assertEqual(moss_backbone_compute_dtype(encoder), torch.bfloat16)

    def test_moss_runtime_rejects_mixed_transformer_dtypes(self) -> None:
        class MixedMossModel(torch.nn.Module):
            def __init__(self) -> None:
                super().__init__()
                self.embed_tokens = torch.nn.Embedding(8, 4, dtype=torch.float16)
                self.layers = torch.nn.ModuleList(
                    [
                        torch.nn.Linear(4, 4, bias=False, dtype=torch.bfloat16),
                        torch.nn.Linear(4, 4, bias=False, dtype=torch.float32),
                    ]
                )

        encoder = type("Encoder", (), {"model": MixedMossModel()})()

        with self.assertRaisesRegex(Qwen35RuntimeError, "mixed backbone dtypes"):
            moss_backbone_compute_dtype(encoder)

    def test_qwen35_rms_norm_uses_one_plus_checkpoint_weight(self) -> None:
        hidden_states = torch.tensor([[[3.0, 4.0]]], dtype=torch.float32)
        checkpoint_weight = torch.tensor([0.25, -0.5], dtype=torch.float32)

        actual = qwen35_rms_norm(hidden_states, checkpoint_weight)

        normalized = hidden_states * torch.rsqrt(
            hidden_states.pow(2).mean(-1, keepdim=True) + 1e-6
        )
        expected = normalized * (1.0 + checkpoint_weight)
        torch.testing.assert_close(actual, expected)

    def test_qwen35_mrope_interleaves_temporal_height_and_width_axes(self) -> None:
        position_ids = torch.tensor([[2], [7], [11]], dtype=torch.long)

        cosine, sine = _rotary_embeddings(
            0,
            1,
            device=torch.device("cpu"),
            dtype=torch.float32,
            position_ids=position_ids,
        )

        inverse_frequency = 1.0 / (
            ROPE_THETA
            ** (torch.arange(0, ROTARY_DIM, 2, dtype=torch.float32) / ROTARY_DIM)
        )
        expected_positions = (2, 7, 11, 2)
        for frequency_index, expected_position in enumerate(expected_positions):
            angle = expected_position * inverse_frequency[frequency_index]
            torch.testing.assert_close(cosine[0, 0, frequency_index], angle.cos())
            torch.testing.assert_close(sine[0, 0, frequency_index], angle.sin())

    def test_qwen35_mrope_rejects_incompatible_position_shape(self) -> None:
        with self.assertRaisesRegex(Qwen35RuntimeError, "position IDs"):
            _rotary_embeddings(
                0,
                2,
                device=torch.device("cpu"),
                dtype=torch.float32,
                position_ids=torch.zeros((2, 2), dtype=torch.long),
            )


if __name__ == "__main__":
    unittest.main()
