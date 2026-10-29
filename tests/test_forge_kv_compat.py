# Copyright (C) 2026 vibecodingtoolmaker
# SPDX-License-Identifier: AGPL-3.0-only

from pathlib import Path
import sys
from types import SimpleNamespace
import unittest

try:
    import torch
except ModuleNotFoundError:
    torch = None

if torch is not None:
    from forgy.forge_kv_compat import ForgeKVCacheError, forward_with_kv_cache
else:
    ForgeKVCacheError = None
    forward_with_kv_cache = None


EXTENSION_ROOT = Path(__file__).resolve().parents[1]


def _local_forge_root():
    candidates = (EXTENSION_ROOT.parents[1], EXTENSION_ROOT.parent)
    return next(
        (
            candidate
            for candidate in candidates
            if (candidate / "backend" / "nn" / "llm" / "llama.py").is_file()
        ),
        None,
    )


def _cache_for(core_model, capacity=8):
    config = core_model.config
    return [
        (
            torch.empty(1, config.num_key_value_heads, capacity, config.head_dim),
            torch.empty(1, config.num_key_value_heads, capacity, config.head_dim),
            0,
        )
        for _layer in core_model.layers
    ]


class NativeCacheCore:
    config = SimpleNamespace(num_key_value_heads=1, head_dim=2)
    layers = (object(),)

    def __init__(self):
        self.calls = 0

    def forward(
        self,
        _input_ids,
        *,
        embeds,
        attention_mask,
        position_ids,
        past_key_values,
    ):
        self.calls += 1
        key, value, index = past_key_values[0]
        next_cache = [(key, value, index + embeds.shape[1])]
        return embeds + 1, None, next_cache

    __call__ = forward


@unittest.skipIf(torch is None, "PyTorch is available only inside Forge.")
class ForgeKVCompatibilityTests(unittest.TestCase):
    def test_legacy_native_cache_contract_remains_authoritative(self) -> None:
        core_model = NativeCacheCore()
        cache = _cache_for(core_model)
        embeds = torch.zeros(1, 3, 4)

        output = forward_with_kv_cache(
            core_model,
            embeds=embeds,
            attention_mask=None,
            past_key_values=cache,
        )

        self.assertEqual(core_model.calls, 1)
        self.assertEqual(len(output), 3)
        self.assertEqual(output[2][0][2], 3)
        torch.testing.assert_close(output[0], embeds + 1)

    def test_current_forge_cached_decode_matches_full_context_forward(self) -> None:
        forge_root = _local_forge_root()
        if forge_root is None:
            self.skipTest(
                "A neighboring Forge checkout is required for this contract test."
            )
        for path in (forge_root, forge_root / "modules_forge" / "packages"):
            if str(path) not in sys.path:
                sys.path.insert(0, str(path))
        from backend.nn.llm.llama import Llama2_

        config = SimpleNamespace(
            vocab_size=32,
            hidden_size=8,
            intermediate_size=16,
            num_hidden_layers=2,
            num_attention_heads=2,
            num_key_value_heads=1,
            max_position_embeddings=32,
            rms_norm_eps=1e-6,
            rope_theta=10_000.0,
            transformer_type="llama",
            head_dim=4,
            rms_norm_add=False,
            mlp_activation="silu",
            qkv_bias=False,
            rope_dims=None,
            q_norm=None,
            k_norm=None,
            rope_scale=None,
            final_norm=False,
            interleaved_mrope=False,
            lm_head=False,
        )
        torch.manual_seed(1234)
        core_model = Llama2_(config).eval()
        embeds = torch.randn(1, 4, config.hidden_size)
        cache = _cache_for(core_model)

        with torch.no_grad():
            full_context, _intermediate = core_model(None, embeds=embeds)
            prefill = forward_with_kv_cache(
                core_model,
                embeds=embeds[:, :3],
                attention_mask=None,
                past_key_values=cache,
            )
            decoded = forward_with_kv_cache(
                core_model,
                embeds=embeds[:, 3:],
                attention_mask=None,
                past_key_values=prefill[2],
            )

        self.assertEqual([entry[2] for entry in prefill[2]], [3, 3])
        self.assertEqual([entry[2] for entry in decoded[2]], [4, 4])
        torch.testing.assert_close(
            decoded[0][:, -1],
            full_context[:, -1],
            rtol=1e-5,
            atol=1e-6,
        )

    def test_mismatched_cache_layer_count_fails_closed(self) -> None:
        class CurrentCacheCore(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.config = SimpleNamespace()
                self.layers = torch.nn.ModuleList([torch.nn.Identity()])

            def forward(self, _input_ids, **_kwargs):
                raise AssertionError("The incompatible native forward must not run.")

        core_model = CurrentCacheCore()

        with self.assertRaisesRegex(ForgeKVCacheError, "different layer counts"):
            forward_with_kv_cache(
                core_model,
                embeds=torch.zeros(1, 1, 1),
                attention_mask=None,
                past_key_values=[],
            )


if __name__ == "__main__":
    unittest.main()
