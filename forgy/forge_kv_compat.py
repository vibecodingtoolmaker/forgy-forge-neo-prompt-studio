"""Request-local KV-cache compatibility for Forge-managed Llama cores.

Forge Neo historically exposed ``past_key_values`` directly on ``Llama2_.forward``.
The Qwen-Image-2.1 update retained cache support in every transformer layer but
removed that top-level plumbing. Forgy uses this adapter to preserve the old
contract without patching Forge or retaining another model.
"""

from __future__ import annotations

import inspect

import torch


class ForgeKVCacheError(RuntimeError):
    """Raised when an active Forge core cannot provide safe cached decoding."""


def _has_native_kv_cache(core_model) -> bool:
    forward = getattr(core_model, "forward", None)
    if not callable(forward):
        raise ForgeKVCacheError("The active language-model core has no forward pass.")
    try:
        parameter = inspect.signature(forward).parameters.get("past_key_values")
    except (TypeError, ValueError):
        parameter = None
    return parameter is not None and parameter.kind is not inspect.Parameter.VAR_KEYWORD


def _validated_cache_length(core_model, past_key_values) -> int:
    layers = getattr(core_model, "layers", None)
    if layers is None or not isinstance(past_key_values, (list, tuple)):
        raise ForgeKVCacheError(
            "The active language-model core has no compatible request-local KV cache."
        )
    if len(past_key_values) != len(layers):
        raise ForgeKVCacheError(
            "The active language-model core and KV cache have different layer counts."
        )

    cache_length = None
    for layer_cache in past_key_values:
        if not isinstance(layer_cache, (list, tuple)) or len(layer_cache) != 3:
            raise ForgeKVCacheError(
                "The active language-model core returned an invalid KV-cache entry."
            )
        key, value, index = layer_cache
        if not isinstance(key, torch.Tensor) or not isinstance(value, torch.Tensor):
            raise ForgeKVCacheError(
                "The active language-model core returned a non-tensor KV cache."
            )
        if key.ndim != 4 or value.ndim != 4 or key.shape != value.shape:
            raise ForgeKVCacheError(
                "The active language-model core returned incompatible KV-cache tensors."
            )
        try:
            index = int(index)
        except (TypeError, ValueError, OverflowError) as exc:
            raise ForgeKVCacheError(
                "The active language-model core returned an invalid KV-cache index."
            ) from exc
        if index < 0 or index > key.shape[2]:
            raise ForgeKVCacheError(
                "The active language-model core returned an out-of-range KV-cache index."
            )
        if cache_length is None:
            cache_length = index
        elif cache_length != index:
            raise ForgeKVCacheError(
                "The active language-model core returned inconsistent KV-cache indices."
            )
    return cache_length or 0


def _forge_llama_helpers(core_model):
    forward = getattr(type(core_model), "forward", None)
    namespace = getattr(forward, "__globals__", {})
    precompute_freqs_cis = namespace.get("precompute_freqs_cis")
    attention_function = namespace.get("attention_function")
    if not callable(precompute_freqs_cis) or not callable(attention_function):
        raise ForgeKVCacheError(
            "The active language-model core does not expose Forge's cache-compatible "
            "attention helpers."
        )
    return precompute_freqs_cis, attention_function


def _precompute_rope(core_model, precompute_freqs_cis, position_ids, device):
    config = core_model.config
    kwargs = {"device": device}
    try:
        supports_interleaved = (
            "interleaved_mrope" in inspect.signature(precompute_freqs_cis).parameters
        )
    except (TypeError, ValueError):
        supports_interleaved = False
    if supports_interleaved:
        kwargs["interleaved_mrope"] = getattr(config, "interleaved_mrope", False)
    return precompute_freqs_cis(
        config.head_dim,
        position_ids,
        config.rope_theta,
        config.rope_scale,
        config.rope_dims,
        **kwargs,
    )


def _compat_forward_with_kv_cache(
    core_model,
    *,
    embeds: torch.Tensor,
    attention_mask,
    position_ids,
    past_key_values,
):
    layers = getattr(core_model, "layers", None)
    config = getattr(core_model, "config", None)
    if layers is None or config is None or not isinstance(embeds, torch.Tensor):
        raise ForgeKVCacheError(
            "The active language-model core cannot use Forgy's Forge KV adapter."
        )

    past_len = _validated_cache_length(core_model, past_key_values)
    precompute_freqs_cis, attention_function = _forge_llama_helpers(core_model)

    hidden_states = embeds
    if getattr(core_model, "normalize_in", False):
        hidden_states = hidden_states * (config.hidden_size**0.5)

    sequence_length = hidden_states.shape[1]
    if position_ids is None:
        position_ids = torch.arange(
            past_len,
            past_len + sequence_length,
            device=hidden_states.device,
        ).unsqueeze(0)
    freqs_cis = _precompute_rope(
        core_model,
        precompute_freqs_cis,
        position_ids,
        hidden_states.device,
    )

    mask = None
    if attention_mask is not None:
        mask = 1.0 - attention_mask.to(hidden_states.dtype).reshape(
            (attention_mask.shape[0], 1, -1, attention_mask.shape[-1])
        ).expand(
            attention_mask.shape[0],
            1,
            sequence_length,
            attention_mask.shape[-1],
        )
        mask = mask.masked_fill(
            mask.to(torch.bool), torch.finfo(hidden_states.dtype).min / 4
        )

    if sequence_length > 1:
        causal_mask = (
            torch.empty(
                past_len + sequence_length,
                past_len + sequence_length,
                dtype=hidden_states.dtype,
                device=hidden_states.device,
            )
            .fill_(torch.finfo(hidden_states.dtype).min / 4)
            .triu_(1)
        )
        mask = causal_mask if mask is None else mask + causal_mask

    next_key_values = []
    for layer, layer_cache in zip(layers, past_key_values):
        hidden_states, current_cache = layer(
            x=hidden_states,
            attention_mask=mask,
            freqs_cis=freqs_cis,
            optimized_attention=attention_function,
            past_key_value=layer_cache,
        )
        if current_cache is None:
            raise ForgeKVCacheError(
                "A Forge language-model layer did not return its updated KV cache."
            )
        next_key_values.append(current_cache)

    norm = getattr(core_model, "norm", None)
    if norm is not None:
        hidden_states = norm(hidden_states)
    return hidden_states, None, next_key_values


def forward_with_kv_cache(
    core_model,
    *,
    embeds: torch.Tensor,
    attention_mask,
    past_key_values,
    position_ids=None,
):
    """Run one cached step against either Forge Llama forward contract."""

    if _has_native_kv_cache(core_model):
        output = core_model(
            None,
            embeds=embeds,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=past_key_values,
        )
        if not isinstance(output, tuple) or len(output) != 3:
            raise ForgeKVCacheError(
                "The active language-model core returned no compatible KV cache."
            )
        _validated_cache_length(core_model, output[2])
        return output

    return _compat_forward_with_kv_cache(
        core_model,
        embeds=embeds,
        attention_mask=attention_mask,
        position_ids=position_ids,
        past_key_values=past_key_values,
    )
