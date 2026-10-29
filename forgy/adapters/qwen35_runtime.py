"""Request-local official Qwen3.5 speech over the cached Anima encoder.

This module deliberately owns no model parameters.  It applies the official
Qwen3.5 computation to tensors already held by ``sd-forge-qwen-35-encoder`` and
adds only the separately verified final layer-31 MLP and language-model norm.

Adapted from Hugging Face Transformers commit
83d46aa2a2c47bef8350580f30981c974add0983:
``src/transformers/models/qwen3_5/modeling_qwen3_5.py``.
"""

# Copyright 2025 The Qwen Team and The HuggingFace Inc. team.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import torch
import torch.nn.functional as functional
from safetensors import safe_open


HIDDEN_SIZE = 2560
INTERMEDIATE_SIZE = 9216
VOCAB_SIZE = 248320
NUM_LAYERS = 32
FULL_ATTENTION_LAYERS = frozenset({3, 7, 11, 15, 19, 23, 27, 31})
LINEAR_KEY_HEADS = 16
LINEAR_VALUE_HEADS = 32
LINEAR_HEAD_DIM = 128
ATTENTION_HEADS = 16
ATTENTION_KV_HEADS = 4
ATTENTION_HEAD_DIM = 256
ROTARY_DIM = 64
ROPE_THETA = 10_000_000.0
MROPE_SECTION = (11, 11, 10)
RMS_EPSILON = 1e-6
CHUNK_SIZE = 64
SOURCE_REPOSITORY = "Qwen/Qwen3.5-4B"
SOURCE_REVISION = "851bf6e806efd8d0a36b00ddf55e13ccb7b8cd0a"
FINAL_NORM_SHA256 = "96294c653d4e8ab158cdae0b7e794559654ffdc62db880725ed714d47f708621"
MOSS_HIDDEN_SIZE = 4096
MOSS_INTERMEDIATE_SIZE = 12288
MOSS_LM_HEAD_KEYS = ("lm_head.weight", "model.lm_head.weight")
MOSS_BACKBONE_KEYS = frozenset(
    {
        "model.language_model.embed_tokens.weight",
        "model.language_model.norm.weight",
    }
)


class Qwen35RuntimeError(RuntimeError):
    """The cached provider cannot safely satisfy the official runtime contract."""


@dataclass(frozen=True)
class Qwen35TailWeights:
    """The five verified tensors absent from the converted conditioning model."""

    final_norm: torch.Tensor
    post_attention_norm: torch.Tensor
    gate_projection: torch.Tensor
    up_projection: torch.Tensor
    down_projection: torch.Tensor


@dataclass
class _LinearCache:
    conv_state: torch.Tensor
    recurrent_state: torch.Tensor


@dataclass
class _AttentionCache:
    key: torch.Tensor
    value: torch.Tensor


@dataclass
class _RequestCache:
    sequence_length: int = 0
    next_rope_position: int = 0
    linear: dict[int, _LinearCache] = field(default_factory=dict)
    attention: dict[int, _AttentionCache] = field(default_factory=dict)


def _shape(tensor: torch.Tensor) -> tuple[int, ...]:
    return tuple(tensor.shape)


def _require_shape(name: str, tensor: torch.Tensor, expected: tuple[int, ...]) -> None:
    if _shape(tensor) != expected:
        raise Qwen35RuntimeError(
            f"{name} has shape {_shape(tensor)}, expected {expected}."
        )


def _require_metadata(
    name: str,
    metadata: dict[str, str] | None,
    expected: dict[str, str],
) -> None:
    metadata = metadata or {}
    mismatches = [key for key, value in expected.items() if metadata.get(key) != value]
    if mismatches:
        raise Qwen35RuntimeError(
            f"{name} has incompatible provenance metadata: {', '.join(mismatches)}."
        )


def moss_backbone_compute_dtype(encoder: Any) -> torch.dtype:
    """Return the single floating dtype used by MOSS projection weights.

    Forge may keep the token embedding in FP16 while the 32 transformer layers
    are BF16.  The causal runtime must follow the backbone dtype; otherwise every
    manual-cast Linear materializes a converted copy of its weight on each token.
    """

    model = getattr(encoder, "model", None)
    embedding = getattr(model, "embed_tokens", None)
    embedding_weight = getattr(embedding, "weight", None)
    if not isinstance(model, torch.nn.Module) or not isinstance(
        embedding_weight, torch.Tensor
    ):
        raise Qwen35RuntimeError(
            "The active MOSS encoder exposes no inspectable transformer weights."
        )

    weights = []
    for name, module in model.named_modules():
        if name == "embed_tokens":
            continue
        weight = getattr(module, "weight", None)
        if (
            isinstance(weight, torch.Tensor)
            and weight.is_floating_point()
            and weight.ndim >= 2
        ):
            weights.append(weight)
    if not weights:
        raise Qwen35RuntimeError(
            "The active MOSS transformer exposes no floating backbone weights."
        )

    dtypes = {weight.dtype for weight in weights}
    if len(dtypes) != 1:
        names = ", ".join(sorted(str(dtype) for dtype in dtypes))
        raise Qwen35RuntimeError(
            f"The active MOSS transformer uses mixed backbone dtypes: {names}."
        )
    dtype = next(iter(dtypes))
    if dtype not in (torch.float16, torch.bfloat16, torch.float32):
        raise Qwen35RuntimeError(
            f"The active MOSS transformer uses unsupported dtype {dtype}."
        )
    return dtype


def load_qwen35_tail_weights(
    lm_norm_path: Path,
    causal_tail_path: Path,
    *,
    device: torch.device,
    dtype: torch.dtype,
) -> Qwen35TailWeights:
    """Load and validate only the missing official tensors for one request."""

    if not lm_norm_path.is_file():
        raise Qwen35RuntimeError(f"Qwen3.5 LM norm is missing: {lm_norm_path}")
    if not causal_tail_path.is_file():
        raise Qwen35RuntimeError(f"Qwen3.5 causal tail is missing: {causal_tail_path}")

    with safe_open(lm_norm_path, framework="pt", device="cpu") as norm_file:
        _require_metadata(
            "Qwen3.5 LM norm",
            norm_file.metadata(),
            {
                "source_repository": SOURCE_REPOSITORY,
                "source_revision": SOURCE_REVISION,
                "license": "Apache-2.0",
                "source_tensor": "model.language_model.norm.weight",
                "tensor_sha256": FINAL_NORM_SHA256,
            },
        )
        final_norm = norm_file.get_tensor("qwen35_4b_lm_norm.weight")
    with safe_open(causal_tail_path, framework="pt", device="cpu") as tail_file:
        _require_metadata(
            "Qwen3.5 causal tail",
            tail_file.metadata(),
            {
                "source_repository": SOURCE_REPOSITORY,
                "source_revision": SOURCE_REVISION,
                "license": "Apache-2.0",
                "purpose": "Qwen3.5-4B layer-31 causal tail for explicit Forgy use",
            },
        )
        post_attention_norm = tail_file.get_tensor(
            "model.language_model.layers.31.post_attention_layernorm.weight"
        )
        gate_projection = tail_file.get_tensor(
            "model.language_model.layers.31.mlp.gate_proj.weight"
        )
        up_projection = tail_file.get_tensor(
            "model.language_model.layers.31.mlp.up_proj.weight"
        )
        down_projection = tail_file.get_tensor(
            "model.language_model.layers.31.mlp.down_proj.weight"
        )

    _require_shape("final LM norm", final_norm, (HIDDEN_SIZE,))
    _require_shape("layer-31 post-attention norm", post_attention_norm, (HIDDEN_SIZE,))
    _require_shape(
        "layer-31 gate projection",
        gate_projection,
        (INTERMEDIATE_SIZE, HIDDEN_SIZE),
    )
    _require_shape(
        "layer-31 up projection",
        up_projection,
        (INTERMEDIATE_SIZE, HIDDEN_SIZE),
    )
    _require_shape(
        "layer-31 down projection",
        down_projection,
        (HIDDEN_SIZE, INTERMEDIATE_SIZE),
    )

    return Qwen35TailWeights(
        final_norm=final_norm.to(device=device, dtype=dtype),
        post_attention_norm=post_attention_norm.to(device=device, dtype=dtype),
        gate_projection=gate_projection.to(device=device, dtype=dtype),
        up_projection=up_projection.to(device=device, dtype=dtype),
        down_projection=down_projection.to(device=device, dtype=dtype),
    )


def qwen35_rms_norm(
    hidden_states: torch.Tensor,
    weight: torch.Tensor,
    *,
    epsilon: float = RMS_EPSILON,
) -> torch.Tensor:
    """Official Qwen3.5 RMSNorm, whose checkpoint scale is ``1 + weight``."""

    normalized = hidden_states.float()
    normalized = normalized * torch.rsqrt(
        normalized.pow(2).mean(-1, keepdim=True) + epsilon
    )
    normalized = normalized * (1.0 + weight.float())
    return normalized.type_as(hidden_states)


def _l2_norm(tensor: torch.Tensor) -> torch.Tensor:
    return tensor * torch.rsqrt(
        (tensor * tensor).sum(dim=-1, keepdim=True) + RMS_EPSILON
    )


def _causal_convolution(mixed_qkv: torch.Tensor, weight: torch.Tensor) -> torch.Tensor:
    sequence_length = mixed_qkv.shape[-1]
    result = functional.conv1d(
        mixed_qkv.to(weight.dtype),
        weight=weight,
        bias=None,
        padding=weight.shape[-1] - 1,
        groups=mixed_qkv.shape[1],
    )
    return functional.silu(result[:, :, :sequence_length]).to(mixed_qkv.dtype)


def _update_causal_convolution(
    mixed_qkv: torch.Tensor,
    conv_state: torch.Tensor,
    weight: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    state_length = conv_state.shape[-1]
    combined = torch.cat([conv_state, mixed_qkv], dim=-1).to(weight.dtype)
    updated_state = combined[:, :, -state_length:]
    result = functional.conv1d(
        combined,
        weight=weight,
        bias=None,
        padding=0,
        groups=mixed_qkv.shape[1],
    )
    result = functional.silu(result[:, :, -mixed_qkv.shape[-1] :])
    return result.to(mixed_qkv.dtype), updated_state


def _chunk_gated_delta_rule(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    decay: torch.Tensor,
    beta: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Official Torch chunk fallback used for Qwen3.5 prefill."""

    initial_dtype = query.dtype
    batch_size, sequence_length, _, key_head_dim = key.shape
    value_heads, value_head_dim = value.shape[-2:]

    query, key, value, beta, decay = [
        tensor.transpose(1, 2).to(torch.float32, memory_format=torch.contiguous_format)
        for tensor in (query, key, value, beta, decay)
    ]
    query = _l2_norm(query)
    key = _l2_norm(key)
    query = query * (query.shape[-1] ** -0.5)

    padding = (CHUNK_SIZE - sequence_length % CHUNK_SIZE) % CHUNK_SIZE
    query, key, value = (
        functional.pad(tensor, (0, 0, 0, padding)) for tensor in (query, key, value)
    )
    beta, decay = (functional.pad(tensor, (0, padding)) for tensor in (beta, decay))
    total_length = sequence_length + padding
    chunk_count = total_length // CHUNK_SIZE

    value_beta = value * beta.unsqueeze(-1)
    key_beta = key * beta.unsqueeze(-1)
    query, key, key_beta, value_beta = [
        tensor.reshape(
            tensor.shape[0], tensor.shape[1], -1, CHUNK_SIZE, tensor.shape[-1]
        )
        for tensor in (query, key, key_beta, value_beta)
    ]
    decay = decay.reshape(decay.shape[0], decay.shape[1], -1, CHUNK_SIZE)

    upper_mask = torch.ones(
        CHUNK_SIZE, CHUNK_SIZE, dtype=torch.bool, device=query.device
    ).triu(1)
    cumulative_decay = decay.cumsum(dim=3)
    pairwise_decay = cumulative_decay.unsqueeze(4) - cumulative_decay.unsqueeze(3)
    pairwise_decay = pairwise_decay.masked_fill(upper_mask, float("-inf")).exp()

    transform_system = (key_beta @ key.transpose(-1, -2)) * pairwise_decay
    intra_chunk_attention = (query @ key.transpose(-1, -2)) * pairwise_decay
    decayed_key_beta = key_beta * cumulative_decay.exp().unsqueeze(-1)
    new_values = torch.linalg.solve_triangular(
        transform_system,
        value_beta,
        upper=False,
        unitriangular=True,
    )
    cumulative_keys = torch.linalg.solve_triangular(
        transform_system,
        decayed_key_beta,
        upper=False,
        unitriangular=True,
    )

    recurrent_state = torch.zeros(
        (batch_size, value_heads, key_head_dim, value_head_dim),
        dtype=new_values.dtype,
        device=new_values.device,
    )
    output = torch.zeros_like(new_values)
    query = query * cumulative_decay.exp().unsqueeze(-1)
    key = key * (cumulative_decay[..., -1:] - cumulative_decay).exp().unsqueeze(-1)
    chunk_decay = cumulative_decay[..., -1].exp()[..., None, None]

    for index in range(chunk_count):
        corrected_values = (
            new_values[:, :, index] - cumulative_keys[:, :, index] @ recurrent_state
        )
        inter_chunk_attention = query[:, :, index] @ recurrent_state
        output[:, :, index] = (
            inter_chunk_attention
            + intra_chunk_attention[:, :, index] @ corrected_values
        )
        recurrent_state = (
            recurrent_state * chunk_decay[:, :, index]
            + key[:, :, index].transpose(-1, -2) @ corrected_values
        )

    output = output.reshape(batch_size, value_heads, -1, value_head_dim)
    output = output[:, :, :sequence_length]
    output = output.transpose(1, 2).to(
        initial_dtype, memory_format=torch.contiguous_format
    )
    return output, recurrent_state


def _recurrent_gated_delta_rule(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    decay: torch.Tensor,
    beta: torch.Tensor,
    recurrent_state: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Official Torch recurrent fallback used for cached single-token decode."""

    initial_dtype = query.dtype
    query, key, value, beta, decay = [
        tensor.transpose(1, 2).to(torch.float32, memory_format=torch.contiguous_format)
        for tensor in (query, key, value, beta, decay)
    ]
    query = _l2_norm(query) / (query.shape[-1] ** 0.5)
    key = _l2_norm(key)
    recurrent_state = recurrent_state.to(value)
    output = torch.zeros_like(value)

    for index in range(query.shape[2]):
        query_token = query[:, :, index]
        key_token = key[:, :, index]
        value_token = value[:, :, index]
        recurrent_state = recurrent_state * decay[:, :, index].exp()[..., None, None]
        beta_token = beta[:, :, index].unsqueeze(-1)
        remembered_value = (recurrent_state * key_token.unsqueeze(-1)).sum(dim=-2)
        delta = (value_token - remembered_value) * beta_token
        recurrent_state = recurrent_state + key_token.unsqueeze(-1) * delta.unsqueeze(
            -2
        )
        output[:, :, index] = (recurrent_state * query_token.unsqueeze(-1)).sum(dim=-2)

    return (
        output.transpose(1, 2).contiguous().to(initial_dtype),
        recurrent_state,
    )


def _gated_rms_norm(
    hidden_states: torch.Tensor,
    gate: torch.Tensor,
    weight: torch.Tensor,
) -> torch.Tensor:
    input_dtype = hidden_states.dtype
    normalized = hidden_states.float()
    normalized = normalized * torch.rsqrt(
        normalized.pow(2).mean(-1, keepdim=True) + RMS_EPSILON
    )
    normalized = weight * normalized.to(input_dtype)
    normalized = normalized * functional.silu(gate.float())
    return normalized.to(input_dtype)


def _rotary_embeddings(
    position: int,
    sequence_length: int,
    *,
    device: torch.device,
    dtype: torch.dtype,
    position_ids: torch.Tensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    inverse_frequency = 1.0 / (
        ROPE_THETA
        ** (
            torch.arange(0, ROTARY_DIM, 2, device=device, dtype=torch.float32)
            / ROTARY_DIM
        )
    )
    if position_ids is None:
        position_ids = torch.arange(
            position,
            position + sequence_length,
            device=device,
            dtype=torch.long,
        ).unsqueeze(0)
    else:
        if position_ids.ndim == 1:
            position_ids = position_ids.unsqueeze(0)
        if tuple(position_ids.shape[1:]) != (sequence_length,) or position_ids.shape[
            0
        ] not in (1, 3):
            raise Qwen35RuntimeError(
                "Qwen3.5 position IDs must have shape [1, sequence] or [3, sequence]."
            )
        position_ids = position_ids.to(device=device, dtype=torch.long)

    frequencies = position_ids.float().unsqueeze(-1) * inverse_frequency.reshape(
        1, 1, -1
    )
    if frequencies.shape[0] == 3:
        interleaved = frequencies[0].clone()
        for axis, offset in ((1, 1), (2, 2)):
            length = MROPE_SECTION[axis] * 3
            interleaved[:, offset:length:3] = frequencies[axis, :, offset:length:3]
        frequencies = interleaved
    else:
        frequencies = frequencies[0]
    embeddings = torch.cat((frequencies, frequencies), dim=-1).unsqueeze(0)
    return embeddings.cos().to(dtype), embeddings.sin().to(dtype)


def _rotate_half(tensor: torch.Tensor) -> torch.Tensor:
    first = tensor[..., : tensor.shape[-1] // 2]
    second = tensor[..., tensor.shape[-1] // 2 :]
    return torch.cat((-second, first), dim=-1)


def _apply_rotary(
    query: torch.Tensor,
    key: torch.Tensor,
    cosine: torch.Tensor,
    sine: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    cosine = cosine.unsqueeze(1)
    sine = sine.unsqueeze(1)
    query_rotary, query_pass = (
        query[..., :ROTARY_DIM],
        query[..., ROTARY_DIM:],
    )
    key_rotary, key_pass = key[..., :ROTARY_DIM], key[..., ROTARY_DIM:]
    query_rotary = query_rotary * cosine + _rotate_half(query_rotary) * sine
    key_rotary = key_rotary * cosine + _rotate_half(key_rotary) * sine
    return (
        torch.cat((query_rotary, query_pass), dim=-1),
        torch.cat((key_rotary, key_pass), dim=-1),
    )


def _repeat_key_values(tensor: torch.Tensor) -> torch.Tensor:
    batch, heads, sequence_length, head_dim = tensor.shape
    tensor = tensor[:, :, None, :, :].expand(
        batch,
        ATTENTION_KV_HEADS,
        ATTENTION_HEADS // ATTENTION_KV_HEADS,
        sequence_length,
        head_dim,
    )
    return tensor.reshape(batch, ATTENTION_HEADS, sequence_length, head_dim)


class Qwen35CausalLMRuntime:
    """One request-local cached causal runtime over a Forge-managed backbone."""

    def __init__(self, model: Any, tail: Qwen35TailWeights) -> None:
        self.model = model
        self.tail = tail
        self.cache = _RequestCache()
        self._validate_contract()

    @property
    def embedding_weight(self) -> torch.Tensor:
        return self.model.get_input_embeddings().weight

    def _validate_contract(self) -> None:
        class_path = f"{type(self.model).__module__}.{type(self.model).__qualname__}"
        if class_path != "lib_qwen35.model.Qwen35HybridModel":
            raise Qwen35RuntimeError(
                f"Unsupported cached Qwen3.5 provider model: {class_path}."
            )
        if len(getattr(self.model, "layers", ())) != NUM_LAYERS:
            raise Qwen35RuntimeError("The cached Qwen3.5 provider is not 32 layers.")
        _require_shape(
            "Qwen3.5 tied embedding",
            self.embedding_weight,
            (VOCAB_SIZE, HIDDEN_SIZE),
        )
        for index, layer in enumerate(self.model.layers):
            mixer_name = (
                "self_attn" if index in FULL_ATTENTION_LAYERS else "linear_attn"
            )
            if not hasattr(layer, mixer_name):
                raise Qwen35RuntimeError(f"Qwen3.5 layer {index} lacks {mixer_name}.")
            if index < NUM_LAYERS - 1 and not hasattr(layer, "mlp"):
                raise Qwen35RuntimeError(f"Qwen3.5 layer {index} lacks its MLP.")
        for tensor in self.tail.__dict__.values():
            if tensor.device != self.embedding_weight.device:
                raise Qwen35RuntimeError(
                    "Qwen3.5 tail and backbone use different devices."
                )
            if tensor.dtype != self.embedding_weight.dtype:
                raise Qwen35RuntimeError(
                    "Qwen3.5 tail and backbone use different dtypes."
                )

    def _linear_attention(
        self,
        index: int,
        mixer: Any,
        hidden_states: torch.Tensor,
        *,
        decode: bool,
    ) -> torch.Tensor:
        batch_size, sequence_length, _ = hidden_states.shape
        mixed_qkv = mixer.in_proj_qkv(hidden_states).transpose(1, 2)
        gate = mixer.in_proj_z(hidden_states).reshape(
            batch_size, sequence_length, LINEAR_VALUE_HEADS, LINEAR_HEAD_DIM
        )
        beta = mixer.in_proj_b(hidden_states).sigmoid()
        decay = -mixer.A_log.float().exp() * functional.softplus(
            mixer.in_proj_a(hidden_states).float() + mixer.dt_bias
        )

        if decode:
            cached = self.cache.linear[index]
            mixed_qkv, conv_state = _update_causal_convolution(
                mixed_qkv,
                cached.conv_state,
                mixer.conv1d.weight,
            )
        else:
            kernel_size = mixer.conv1d.weight.shape[-1]
            conv_state = mixed_qkv[:, :, -kernel_size:]
            if conv_state.shape[-1] < kernel_size:
                conv_state = functional.pad(
                    conv_state, (kernel_size - conv_state.shape[-1], 0)
                )
            mixed_qkv = _causal_convolution(mixed_qkv, mixer.conv1d.weight)

        mixed_qkv = mixed_qkv.transpose(1, 2)
        query, key, value = torch.split(mixed_qkv, (2048, 2048, 4096), dim=-1)
        query = query.reshape(
            batch_size, sequence_length, LINEAR_KEY_HEADS, LINEAR_HEAD_DIM
        )
        key = key.reshape(
            batch_size, sequence_length, LINEAR_KEY_HEADS, LINEAR_HEAD_DIM
        )
        value = value.reshape(
            batch_size, sequence_length, LINEAR_VALUE_HEADS, LINEAR_HEAD_DIM
        )
        query = query.repeat_interleave(2, dim=2)
        key = key.repeat_interleave(2, dim=2)

        if decode:
            output, recurrent_state = _recurrent_gated_delta_rule(
                query,
                key,
                value,
                decay,
                beta,
                cached.recurrent_state,
            )
        else:
            output, recurrent_state = _chunk_gated_delta_rule(
                query, key, value, decay, beta
            )
        self.cache.linear[index] = _LinearCache(
            conv_state=conv_state,
            recurrent_state=recurrent_state,
        )

        output = output.reshape(-1, LINEAR_HEAD_DIM)
        gate = gate.reshape(-1, LINEAR_HEAD_DIM)
        output = _gated_rms_norm(output, gate, mixer.norm.weight)
        output = output.reshape(batch_size, sequence_length, -1)
        return mixer.out_proj(output)

    def _full_attention(
        self,
        index: int,
        mixer: Any,
        hidden_states: torch.Tensor,
        *,
        position: int,
        decode: bool,
        position_ids: torch.Tensor | None = None,
    ) -> torch.Tensor:
        batch_size, sequence_length, _ = hidden_states.shape
        query_and_gate = mixer.q_proj(hidden_states).view(
            batch_size, sequence_length, ATTENTION_HEADS, ATTENTION_HEAD_DIM * 2
        )
        query, gate = torch.chunk(query_and_gate, 2, dim=-1)
        gate = gate.reshape(batch_size, sequence_length, -1)
        query = qwen35_rms_norm(query, mixer.q_norm.weight).transpose(1, 2)
        key = qwen35_rms_norm(
            mixer.k_proj(hidden_states).view(
                batch_size,
                sequence_length,
                ATTENTION_KV_HEADS,
                ATTENTION_HEAD_DIM,
            ),
            mixer.k_norm.weight,
        ).transpose(1, 2)
        value = (
            mixer.v_proj(hidden_states)
            .view(
                batch_size,
                sequence_length,
                ATTENTION_KV_HEADS,
                ATTENTION_HEAD_DIM,
            )
            .transpose(1, 2)
        )

        cosine, sine = _rotary_embeddings(
            position,
            sequence_length,
            device=hidden_states.device,
            dtype=hidden_states.dtype,
            position_ids=position_ids,
        )
        query, key = _apply_rotary(query, key, cosine, sine)
        if decode:
            cached = self.cache.attention[index]
            key = torch.cat((cached.key, key), dim=2)
            value = torch.cat((cached.value, value), dim=2)
        self.cache.attention[index] = _AttentionCache(key=key, value=value)

        repeated_key = _repeat_key_values(key)
        repeated_value = _repeat_key_values(value)
        scores = torch.matmul(query, repeated_key.transpose(2, 3)) * (
            ATTENTION_HEAD_DIM**-0.5
        )
        if not decode and sequence_length > 1:
            causal_mask = torch.full(
                (sequence_length, sequence_length),
                torch.finfo(scores.dtype).min,
                dtype=scores.dtype,
                device=scores.device,
            ).triu(1)
            scores = scores + causal_mask
        attention = functional.softmax(scores, dim=-1, dtype=torch.float32).to(
            query.dtype
        )
        output = torch.matmul(attention, repeated_value)
        output = (
            output.transpose(1, 2).contiguous().reshape(batch_size, sequence_length, -1)
        )
        output = output * torch.sigmoid(gate)
        return mixer.o_proj(output)

    def _mlp(self, index: int, hidden_states: torch.Tensor) -> torch.Tensor:
        if index == NUM_LAYERS - 1:
            normalized = qwen35_rms_norm(hidden_states, self.tail.post_attention_norm)
            return functional.linear(
                functional.silu(
                    functional.linear(normalized, self.tail.gate_projection, bias=None)
                )
                * functional.linear(normalized, self.tail.up_projection, bias=None),
                self.tail.down_projection,
                bias=None,
            )

        layer = self.model.layers[index]
        normalized = qwen35_rms_norm(
            hidden_states, layer.post_attention_layernorm.weight
        )
        return layer.mlp.down_proj(
            functional.silu(layer.mlp.gate_proj(normalized))
            * layer.mlp.up_proj(normalized)
        )

    def _forward(self, embeddings: torch.Tensor, *, decode: bool) -> torch.Tensor:
        if embeddings.ndim != 3 or embeddings.shape[-1] != HIDDEN_SIZE:
            raise Qwen35RuntimeError(
                f"Qwen3.5 embeddings have incompatible shape {_shape(embeddings)}."
            )
        if decode and embeddings.shape[1] != 1:
            raise Qwen35RuntimeError("Cached Qwen3.5 decode accepts exactly one token.")
        if decode and self.cache.sequence_length < 1:
            raise Qwen35RuntimeError("Qwen3.5 decode was requested before prefill.")
        if not decode and self.cache.sequence_length:
            raise Qwen35RuntimeError("A Qwen3.5 request may only be prefixed once.")

        position = self.cache.sequence_length if decode else 0
        hidden_states = embeddings
        for index, layer in enumerate(self.model.layers):
            residual = hidden_states
            normalized = qwen35_rms_norm(hidden_states, layer.input_layernorm.weight)
            if index in FULL_ATTENTION_LAYERS:
                mixed = self._full_attention(
                    index,
                    layer.self_attn,
                    normalized,
                    position=position,
                    decode=decode,
                )
            else:
                mixed = self._linear_attention(
                    index,
                    layer.linear_attn,
                    normalized,
                    decode=decode,
                )
            hidden_states = residual + mixed
            hidden_states = hidden_states + self._mlp(index, hidden_states)

        self.cache.sequence_length += embeddings.shape[1]
        return qwen35_rms_norm(hidden_states, self.tail.final_norm)

    def prefill(self, input_ids: torch.Tensor) -> torch.Tensor:
        """Prefill one prompt and return its official normalized hidden states."""

        if input_ids.ndim != 2 or input_ids.shape[0] != 1:
            raise Qwen35RuntimeError(
                "Qwen3.5 prefill currently supports batch size one."
            )
        embeddings = self.model.get_input_embeddings()(input_ids)
        return self._forward(embeddings, decode=False)

    def decode(self, token_id: int) -> torch.Tensor:
        """Advance one token through the request-local recurrent and KV caches."""

        input_ids = torch.tensor(
            [[token_id]], device=self.embedding_weight.device, dtype=torch.long
        )
        embeddings = self.model.get_input_embeddings()(input_ids)
        return self._forward(embeddings, decode=True)

    def logits(self, hidden_states: torch.Tensor) -> torch.Tensor:
        """Project the last hidden state through the model's tied embedding."""

        return functional.linear(
            hidden_states[:, -1, :], self.embedding_weight, bias=None
        )


def inspect_moss_lm_head(path: Path) -> tuple[str, str]:
    """Validate one selected dense MOSS encoder file without loading its head."""

    path = Path(path)
    if not path.is_file():
        raise Qwen35RuntimeError(f"Selected MOSS text encoder is missing: {path}")

    try:
        with safe_open(path, framework="pt", device="cpu") as source:
            keys = set(source.keys())
            if not MOSS_BACKBONE_KEYS.issubset(keys):
                raise Qwen35RuntimeError(
                    f"{path.name} is not a complete MOSS Qwen3.5-9B encoder."
                )
            head_keys = [key for key in MOSS_LM_HEAD_KEYS if key in keys]
            if len(head_keys) != 1:
                raise Qwen35RuntimeError(
                    f"{path.name} exposes {len(head_keys)} supported MOSS LM heads; "
                    "expected exactly one."
                )
            head_key = head_keys[0]
            head = source.get_slice(head_key)
            shape = tuple(head.get_shape())
            dtype = str(head.get_dtype())
    except Qwen35RuntimeError:
        raise
    except Exception as exc:
        raise Qwen35RuntimeError(
            f"Could not inspect selected MOSS text encoder {path.name}: "
            f"{type(exc).__name__}: {exc}"
        ) from exc

    if shape != (VOCAB_SIZE, MOSS_HIDDEN_SIZE):
        raise Qwen35RuntimeError(
            f"{path.name} LM head has shape {shape}, expected "
            f"({VOCAB_SIZE}, {MOSS_HIDDEN_SIZE})."
        )
    if dtype not in {"BF16", "F16", "F32"}:
        raise Qwen35RuntimeError(
            f"{path.name} uses a {dtype} LM head. Forgy's first MOSS provider "
            "supports a dense BF16, FP16, or FP32 head only."
        )
    return head_key, dtype


def load_moss_lm_head(
    path: Path,
    *,
    device: torch.device,
    dtype: torch.dtype,
) -> torch.Tensor:
    """Load only the active MOSS checkpoint's untied head for one request."""

    head_key, _source_dtype = inspect_moss_lm_head(path)
    try:
        with safe_open(path, framework="pt", device="cpu") as source:
            head = source.get_tensor(head_key)
    except Exception as exc:
        raise Qwen35RuntimeError(
            f"Could not load the selected MOSS LM head from {path.name}: "
            f"{type(exc).__name__}: {exc}"
        ) from exc

    _require_shape("MOSS LM head", head, (VOCAB_SIZE, MOSS_HIDDEN_SIZE))
    if not head.is_floating_point():
        raise Qwen35RuntimeError("The selected MOSS LM head is not floating point.")
    return head.to(device=device, dtype=dtype)


class ForgeQwen359BCausalLMRuntime(Qwen35CausalLMRuntime):
    """Cached causal speech over Forge's active MOSS Qwen3.5-9B backbone."""

    def __init__(self, encoder: Any, lm_head: torch.Tensor) -> None:
        self.encoder = encoder
        self.model = getattr(encoder, "model", None)
        self.lm_head = lm_head
        self.compute_dtype = moss_backbone_compute_dtype(encoder)
        self.tail = None
        self.cache = _RequestCache()
        self._validate_contract()

    @property
    def embedding_weight(self) -> torch.Tensor:
        embedding = getattr(self.model, "embed_tokens", None)
        return getattr(embedding, "weight", None)

    def _validate_contract(self) -> None:
        encoder_path = (
            f"{type(self.encoder).__module__}.{type(self.encoder).__qualname__}"
        )
        if encoder_path != "backend.nn.llm.llama.Qwen35_9B":
            raise Qwen35RuntimeError(
                f"Unsupported Forge MOSS Qwen3.5 encoder: {encoder_path}."
            )
        config = getattr(self.model, "config", None)
        expected_config = {
            "hidden_size": MOSS_HIDDEN_SIZE,
            "intermediate_size": MOSS_INTERMEDIATE_SIZE,
            "vocab_size": VOCAB_SIZE,
            "num_hidden_layers": NUM_LAYERS,
            "num_attention_heads": ATTENTION_HEADS,
            "num_key_value_heads": ATTENTION_KV_HEADS,
            "linear_num_key_heads": LINEAR_KEY_HEADS,
            "linear_num_value_heads": LINEAR_VALUE_HEADS,
            "linear_key_head_dim": LINEAR_HEAD_DIM,
            "linear_value_head_dim": LINEAR_HEAD_DIM,
        }
        mismatches = [
            name
            for name, expected in expected_config.items()
            if getattr(config, name, None) != expected
        ]
        if mismatches:
            raise Qwen35RuntimeError(
                "The active MOSS Qwen3.5-9B configuration is incompatible: "
                + ", ".join(mismatches)
                + "."
            )
        if len(getattr(self.model, "layers", ())) != NUM_LAYERS:
            raise Qwen35RuntimeError(
                "The active MOSS Qwen3.5 encoder is not 32 layers."
            )

        embedding_weight = self.embedding_weight
        if not isinstance(embedding_weight, torch.Tensor):
            raise Qwen35RuntimeError(
                "The active MOSS token embedding is not a directly usable tensor."
            )
        _require_shape(
            "MOSS token embedding",
            embedding_weight,
            (VOCAB_SIZE, MOSS_HIDDEN_SIZE),
        )
        _require_shape(
            "MOSS LM head",
            self.lm_head,
            (VOCAB_SIZE, MOSS_HIDDEN_SIZE),
        )
        final_norm = getattr(self.model, "norm", None)
        if final_norm is None:
            raise Qwen35RuntimeError("The active MOSS encoder lacks its final norm.")
        _require_shape(
            "MOSS final norm",
            final_norm.weight,
            (MOSS_HIDDEN_SIZE,),
        )
        if self.lm_head.device != embedding_weight.device:
            raise Qwen35RuntimeError(
                "The MOSS LM head and active Forge encoder use different devices."
            )
        if self.lm_head.dtype != self.compute_dtype:
            raise Qwen35RuntimeError(
                "The MOSS LM head and active Forge transformer use different dtypes."
            )
        misplaced = []
        for name, module in self.model.named_modules():
            weight = getattr(module, "weight", None)
            if (
                isinstance(weight, torch.Tensor)
                and weight.is_floating_point()
                and weight.device != embedding_weight.device
            ):
                misplaced.append(name or type(module).__name__)
        if misplaced:
            raise Qwen35RuntimeError(
                "The active MOSS transformer is not fully colocated with its token "
                f"embedding: {', '.join(misplaced[:5])}."
            )

        for index, layer in enumerate(self.model.layers):
            mixer_name = (
                "self_attn" if index in FULL_ATTENTION_LAYERS else "linear_attn"
            )
            if not hasattr(layer, mixer_name):
                raise Qwen35RuntimeError(f"MOSS layer {index} lacks {mixer_name}.")
            if not hasattr(layer, "mlp"):
                raise Qwen35RuntimeError(f"MOSS layer {index} lacks its MLP.")

    def _mlp(self, index: int, hidden_states: torch.Tensor) -> torch.Tensor:
        layer = self.model.layers[index]
        normalized = qwen35_rms_norm(
            hidden_states, layer.post_attention_layernorm.weight
        )
        return layer.mlp.down_proj(
            functional.silu(layer.mlp.gate_proj(normalized))
            * layer.mlp.up_proj(normalized)
        )

    def _forward(
        self,
        embeddings: torch.Tensor,
        *,
        decode: bool,
        position_ids: torch.Tensor | None = None,
    ) -> torch.Tensor:
        if embeddings.ndim != 3 or embeddings.shape[-1] != MOSS_HIDDEN_SIZE:
            raise Qwen35RuntimeError(
                f"MOSS embeddings have incompatible shape {_shape(embeddings)}."
            )
        if decode and embeddings.shape[1] != 1:
            raise Qwen35RuntimeError("Cached MOSS decode accepts exactly one token.")
        if decode and self.cache.sequence_length < 1:
            raise Qwen35RuntimeError("MOSS decode was requested before prefill.")
        if not decode and self.cache.sequence_length:
            raise Qwen35RuntimeError("A MOSS request may only be prefixed once.")

        position = self.cache.next_rope_position if decode else 0
        hidden_states = embeddings
        for index, layer in enumerate(self.model.layers):
            residual = hidden_states
            normalized = qwen35_rms_norm(hidden_states, layer.input_layernorm.weight)
            if index in FULL_ATTENTION_LAYERS:
                mixed = self._full_attention(
                    index,
                    layer.self_attn,
                    normalized,
                    position=position,
                    decode=decode,
                    position_ids=position_ids,
                )
            else:
                mixed = self._linear_attention(
                    index,
                    layer.linear_attn,
                    normalized,
                    decode=decode,
                )
            hidden_states = residual + mixed
            hidden_states = hidden_states + self._mlp(index, hidden_states)

        self.cache.sequence_length += embeddings.shape[1]
        if position_ids is None:
            self.cache.next_rope_position += embeddings.shape[1]
        else:
            self.cache.next_rope_position = int(position_ids[:, -1].max().item()) + 1
        return qwen35_rms_norm(hidden_states, self.model.norm.weight)

    def prefill(self, input_ids: torch.Tensor) -> torch.Tensor:
        if input_ids.ndim != 2 or input_ids.shape[0] != 1:
            raise Qwen35RuntimeError("MOSS prefill currently supports batch size one.")
        embeddings = self.encoder.get_input_embeddings()(input_ids).to(
            dtype=self.compute_dtype
        )
        return self._forward(embeddings, decode=False)

    def prefill_embeddings(
        self,
        embeddings: torch.Tensor,
        position_ids: torch.Tensor,
    ) -> torch.Tensor:
        """Prefill one Forge-produced multimodal sequence with Qwen3.5 MRoPE."""

        if embeddings.ndim != 3 or embeddings.shape[0] != 1:
            raise Qwen35RuntimeError(
                "MOSS multimodal prefill currently supports batch size one."
            )
        return self._forward(
            embeddings.to(dtype=self.compute_dtype),
            decode=False,
            position_ids=position_ids,
        )

    def decode(self, token_id: int) -> torch.Tensor:
        input_ids = torch.tensor(
            [[token_id]], device=self.embedding_weight.device, dtype=torch.long
        )
        embeddings = self.encoder.get_input_embeddings()(input_ids).to(
            dtype=self.compute_dtype
        )
        return self._forward(embeddings, decode=True)

    def logits(self, hidden_states: torch.Tensor) -> torch.Tensor:
        return functional.linear(hidden_states[:, -1, :], self.lm_head, bias=None)
