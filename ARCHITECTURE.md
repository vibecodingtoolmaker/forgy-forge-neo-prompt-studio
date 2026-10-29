# Forgy model architecture

Forgy separates shared product behavior from model-family behavior. The purpose
is to add or repair one model family without scattering conditional branches
through personas, workflows, Forge actions, and UI code.

## Runtime flow

```text
Forge active selection
  -> ModelManager.resolve()
  -> request-local ModelContext
  -> CapabilityManager.resolve()
  -> workflow guard and UI updates
  -> selected adapter operation
```

`ModelManager` keeps only the explicit adapter registry. It never stores the
active diffusion model, encoder, tokenizer, `ModelPatcher`, CUDA tensors, or KV
cache. Every resolution returns a new request-local `ModelContext`.

`CapabilityManager` combines a validated adapter with its declared operations.
The Python workflow guard remains authoritative; Gradio `interactive` updates
are user guidance and are refreshed after Forge loads the current selection.

## Package boundaries

```text
forgy/
  capabilities.py       typed capabilities and effective workflow state
  model_manager.py       deterministic, fail-closed adapter resolution
  adapters/
    base.py              dependency-free adapter contracts
    anima.py             Anima stock Qwen3-0.6B text-only adapter
    flux2_klein.py       FLUX.2 Klein 4B text-only adapter
    krea2.py             KREA2 detection, components, prompt protocol, and dispatch
    moss.py              MOSS-Image Qwen3.5-9B text/vision adapter
    qwen.py              shared visible-output and repetition safeguards
    qwen35_runtime.py    request-local official Qwen3.5 causal computation
    zimage.py            Z-Image Base/Turbo text-only adapter
scripts/
  forge_krea_prompt_assistant.py
                         compatibility entry point, shared workflows, UI, and Forge actions
```

The existing script filename, callback names, element IDs, CSS classes, and
installation-directory compatibility are intentionally unchanged.

## Current KREA2 adapter

The first adapter identifies the exact KREA2 diffusion-engine class and then
validates the Qwen3-VL encoder, tokenizer, processing engine, and Forge
`ModelPatcher` by object identity and class contract. It declares all four
current workflows, natural-language prompts, vision input, multi-turn chat,
controllable reasoning, and supported sampling controls.

KREA2-specific chat-template rendering, image marker insertion, controlled
decode, reasoning removal, repetition-loop detection, component resolution,
and runtime dispatch live on the KREA2 adapter. The established low-level token
and vision loops remain behavior-preserving runtime functions registered with
that adapter; shared workflow runners no longer call a KREA2 implementation
directly.

## Initial Z-Image adapter

Z-Image Base and Turbo share one adapter because Forge exposes both through the
same `ZImage` engine and Qwen3-4B text contract. The adapter validates the exact
engine and encoder classes, object identity, hidden width, vocabulary size, and
`ModelPatcher`. It enables text-only Forgy Chat, Idea-to-Prompt, and Refine.
Vision is declared unavailable, so Image-to-Prompt and image attachment are
disabled in both the UI and backend.

Z-Image also owns its workflow-specific user contracts. These reinforce the
Idea-to-Prompt, Refine, and structured Forgy Chat task boundaries for the
smaller Qwen3 encoder without changing the selected persona, which remains the
complete system prompt. KREA2 keeps its previously validated request rendering
unchanged.

The shared autoregressive runtime reconstructs logits only through the existing
tied input embedding matrix; it does not load an LM head or a second model.

## FLUX.2 Klein 4B adapter

FLUX.2 Klein 4B and Base 4B use an explicit third adapter. It matches Forge's
exact `Flux2` diffusion-engine class and validates the Klein processing engine,
Qwen3-4B encoder class, object identity, hidden width, vocabulary, tokenizer,
and `ModelPatcher`. Like Z-Image, it enables text-only Forgy Chat,
Idea-to-Prompt, and Refine while capability-gating Image-to-Prompt and Forgy
image attachment.

The adapter preserves Forge's existing Klein chat template, including its
single empty thinking block, and owns separate workflow contracts for
natural-language FLUX prompts. It reuses the same request-local tied-embedding
generation runtime without changing the frozen Z-Image adapter.

Klein 9B intentionally fails closed. Forge exposes its Qwen3-8B transformer but
does not retain the checkpoint's separate, untied LM head. Forgy therefore
cannot reconstruct correct output logits from the input embedding matrix and
does not guess, load an extra head, or modify Forge core.

## Anima Qwen3-0.6B adapter and optional Qwen3.5 provider

Anima uses a separate text-only adapter when Forge exposes the original
`Qwen3_06B`, its matching tokenizer, `AnimaTextProcessingEngine`, final
1024-wide language-model norm, tied 151936-token embedding, and `ModelPatcher`.
It enables Forgy Chat, Idea-to-Prompt, and Refine and disables both vision
workflows.

The current PyTorch 2.11 Forge stack rejects the Anima manual-cast layers when
their inputs are inference tensors. The adapter therefore preserves the
mechanically validated `torch.no_grad()` boundary, but now uses Forge's
preallocated request-local KV-cache contract instead of recomputing the complete
prompt for every output token. Q/K/V compute in FP32 under these manual-cast
layers, so the temporary cache is explicitly FP32 and bounded to 256 output
tokens. The final normalized 1024-wide state is projected through a zero-copy
alias of Forge's tied input embedding. No model object or CUDA cache is retained.

The stock 0.6B encoder is not reliable at reproducing Forgy's full agent response
schema. Only that provider therefore uses a compact Forgy mode: a new message is
normalized into a direct Idea-to-Prompt request, while an existing working prompt
uses the refinement contract. Forgy wraps a valid result in its UI protocol.
Instruction copies, replacement characters, JSON-like or repeated garbage are
rejected; a failed new expansion falls back visibly to the cleaned user request,
and a failed refinement preserves the current prompt. Qwen3.5 and other model
families keep the normal agent protocol.

The optional Qwen3.5 Anima extension has a different backend contract. Its local
converted checkpoint contains a 248320-row tied embedding and 2560-wide hybrid
backbone, followed by a learned 2560-to-1024 Anima conditioning projection. It
omits the causal LM's final RMSNorm and layer-31 post-attention norm/MLP. Forgy
therefore never substitutes the conditioning projection, an identity norm, or
invented weights.

The non-persistent `Anima text provider` selector defaults to the original 0.6B
path. Its experimental Qwen3.5 choice resolves exactly one model already cached
by `sd-forge-qwen-35-encoder`; it never calls that extension's `_load_model()`.
No cache, multiple distinct caches, another filename, missing provenance, or an
incompatible tensor/model shape fails closed without falling back to 0.6B. The
third-party extension remains unmodified and owns the 4B backbone and Forge
`ModelPatcher`. Forge loads extension scripts with a basename that retains the
`.py` suffix, so provider discovery normalizes that suffix before matching the
exact `forge_qwen35_encoder.Qwen35EncoderForForge` type.

`qwen35_runtime.py` applies the pinned official Apache-2.0 computation to those
existing tensors: `(1 + weight)` Qwen3.5 RMSNorm, the 16-to-32-head Gated
DeltaNet layout with L2-normalized queries/keys, partial 64-wide text RoPE at
theta 10,000,000, sigmoid attention gating, all 32 MLPs, chunked prefill,
request-local recurrent/KV decode caches, final RMSNorm, and the tied embedding
projection. The five absent BF16 tensors are bounded range extracts from pinned
Qwen/Qwen3.5-4B revision
`851bf6e806efd8d0a36b00ddf55e13ccb7b8cd0a`; their provenance is validated and
the tensors live only for the request. The norm-only and locally converted
backbone produced non-language, while the corrected isolated smoke began
`A close-up of` and the full Anima/provider smoke began
`A wide cinematic composition featuring a red`.

## Experimental MOSS-Image Qwen3.5-9B adapter

MOSS-Image exposes a complete Forge-owned `Qwen35_9B` backbone, vision encoder,
tokenizer, final norm, and `ModelPatcher`. The adapter validates those exact live
objects and enables Forgy Chat, Idea-to-Prompt, Image-to-Prompt, and Refine.
MOSS workflow contracts preserve the selected persona's caption or tag-sequence
dialect instead of imposing the natural-language default on every request.
Image-to-Prompt replaces exactly one MOSS image token through Forge's active
`SDClipModel`, consumes Forge's `embeds_info`, and applies Qwen3.5's interleaved
three-axis MRoPE in the same request-local causal runtime. No second vision or
language model is loaded.

Unlike Forgy's Qwen3-4B routes, the official MOSS Qwen3.5 configuration does not
tie its token embedding and LM head. Forge currently omits `lm_head` and MTP when
assembling its conditioning encoder. Forgy therefore resolves exactly one dense
BF16, FP16, or FP32 Qwen3.5-9B safetensors file from Forge's active additional
module selection and loads only that file's 248320-by-4096 `lm_head` for one
request. It does not scan for alternatives, cache the head, create another
backbone, or change Forge core. Quantized or ambiguous heads fail closed.

`ForgeQwen359BCausalLMRuntime` reuses the existing Qwen3.5 recurrence and KV
implementation with MOSS's 4096-wide backbone, all 32 resident Forge layers,
the active final norm, and the request-local untied head. Full single-device
placement is required because the direct cached runtime cannot safely mix Forge
low-VRAM CPU and CUDA layers. Real RTX 5090 smokes produced clean visible text
from a 311-token text request and a synthetic 576-visual-token image request.
Four output tokens took 15.1 to 20.5 seconds for text across two profiled runs
and 34.9 seconds for vision inside generation. The selected local BF16 encoder
has no MTP tensors, so MTP and
quantized-head acceleration remain outside this compatibility scope.

## Adding another model family

1. Add one adapter module with an exact, read-only `probe()`.
2. Validate every required live Forge object in `resolve()` and return a fresh
   components object.
3. Declare capabilities, prompt dialect, sampling controls, and the future
   default-persona family explicitly.
4. Implement only the operations the family actually supports.
5. Register the adapter explicitly with `ModelManager`; do not use guessed
   fallbacks or automatic module discovery.
6. Add dependency-free contract tests and a focused live Forge smoke test.

Unknown, incomplete, or ambiguous stacks fail closed. UI state never replaces
backend validation.

## Future prompt backends and representation bridges

A target image-model adapter is not automatically a generative prompt backend.
Future SDXL/Pony-style support may use an optional user-confirmed prompt model,
but its resource and loading lifecycle must remain separate from target-model
detection. The research-only Embedding Bridge is also a separate conditioning
integration and is not a Forgy runtime dependency.
