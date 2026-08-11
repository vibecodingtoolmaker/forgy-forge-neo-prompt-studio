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
    krea2.py             KREA2 detection, components, prompt protocol, and dispatch
    qwen.py              shared visible-output and repetition safeguards
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
