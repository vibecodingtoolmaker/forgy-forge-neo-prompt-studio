# Changelog

All notable changes to this project will be documented in this file.

## Unreleased

## 0.5.2-alpha.1 - 2026-08-13

- Added an isolated text-only FLUX.2 Klein 4B/Base 4B adapter using Forge's
  active tied Qwen3-4B encoder, tokenizer, and `ModelPatcher` for Forgy Chat,
  Idea-to-Prompt, and Refine.
- Added Klein-specific stack validation and workflow contracts without changing
  the released Z-Image adapter or shared personas.
- Made Klein 9B fail closed with an explicit explanation because Forge discards
  its separate untied Qwen3-8B LM head.
- Added dependency-free Klein adapter coverage and a bounded live Forge smoke
  harness. A real local Klein 4B stack completed deterministic text generation
  with 64 output tokens, 306 visible characters, and clean weight unloading.
- Made the Forgy image attachment clickable: an uploaded or explicitly grabbed
  image opens in a full-screen lightbox and closes again on image, backdrop,
  close-button, Escape, or Enter activation.
- Added an undoable `×` control below the working prompt's copy control and kept
  conversation state unchanged when only the working prompt is cleared.
- Added `Grab last generated prompt`. Forgy records the positive prompt when the
  latest txt2img or img2img gallery changes, restores that exact generation
  prompt on explicit request, and keeps the previous working prompt available to
  Undo.
- Reset successful image/prompt grab labels after two seconds and successful
  Undo/Clear labels after one second instead of leaving stale checkmarks visible.
- Promoted the combined idea, image-analysis, and refinement behavior to Forgy
  Chat's built-in `Default` persona. The former built-in prompt remains available
  as `Default Legacy`; only its exact unchanged hash migrates, so edited defaults
  and custom personas remain untouched.

## 0.5.1-alpha.1 - 2026-08-11

- Renamed the public product to `Forgy — Forge Neo Prompt Studio` across the UI,
  documentation, support files, and repository metadata.
- Updated installation, workflow badge, issue, release, and comparison links
  for `https://github.com/vibecodingtoolmaker/forgy-forge-neo-prompt-studio`.
- Kept compatibility-sensitive Python names, callback and tab IDs, CSS classes,
  and existing installation directories unchanged.
- Added a dependency-free model adapter package with explicit model identity,
  request-local component resolution, typed capabilities, and fail-closed model
  selection.
- Moved KREA2 stack validation, Qwen3-VL prompt protocol, visible-output
  filtering, and runtime dispatch behind the first explicit adapter while
  preserving the established generation loops and compatibility entry point.
- Added capability-driven backend guards and dynamic Gradio workflow/image
  control updates after Forge loads the user's current selection.
- Added focused adapter, capability, no-persistent-model-state, and full Forge
  bootstrap validation.
- Added the initial Z-Image Base/Turbo adapter using Forge's active Qwen3-4B
  encoder and the shared tied-embedding autoregressive text runtime. Forgy Chat,
  Idea-to-Prompt, and Refine are enabled; vision workflows are explicitly
  disabled.
- Live-validated a bounded 64-token Z-Image Base/Qwen3-4B text request in an
  isolated Forge process with no image generation, no saved settings, and model
  unload afterward.
- Made Forgy Chat accept harmless `FORGY reply:` / `UPDATED PROMPT:` marker
  variations while still preserving the working prompt for unstructured model
  output.
- Added Z-Image-only workflow contracts for Idea-to-Prompt, Refine, and Forgy
  Chat so the smaller Qwen treats user text as visual source content, preserves
  explicit concepts, and follows each workflow's output shape. KREA2 prompt
  rendering remains unchanged.
- Added a visible capability explanation beside both disabled image inputs when
  the active Forgy adapter is text-only.

## 0.5.0-beta.1 - 2026-08-05

- Replaced the extension's original KREA2-oriented public identity, promoted
  the version marker to `0.5.0-beta.1`, and made its general UI, status,
  persona, metadata, and support language model-family neutral. The README now
  documents KREA2/Qwen3-VL only as the first concrete backend adapter rather
  than as the product identity.
- Made the workflow guidance friendlier and less formal: Idea-to-Prompt now
  invites users to let Forgy develop their picture idea, and both primary input
  workflows describe their optional instruction as a refinement alongside the
  selected persona. The active VRAM-profile summary now sits at the very bottom
  of all four workflow tabs, after persona management and sampling controls.
- Promoted `Forgy Chat` to the first workflow tab and reorganized it as the
  studio entry point. Its working-prompt label and guidance are clearer, the
  image attachment is permanently visible below the prompt, Forge action rows
  now sit below the two-column workspace, independent persona management sits
  below those actions, and the UI explains that its 50 undo prompt versions are
  session-only rather than stored in a local file. `Undo prompt change` and
  `Clear conversation` now sit directly below the chat send/stop controls.
- Unified cross-workflow navigation controls. Idea-to-Prompt and
  Image-to-Prompt now place `Send to prompt refinement` and `Send to Forgy`
  beside their generated prompt, while Prompt Refinement places `Send to Forgy`
  with its two source-result actions beside `Prompt to refine`.
- Reorganized Prompt Refinement to match the primary workflows: its two source
  result buttons now sit in a narrow column beside `Prompt to refine`, txt2img
  and img2img actions each use one ordered four-button row, and independent
  persona management now lives below the result actions.
- Reorganized Idea-to-Prompt and Image-to-Prompt without changing generation
  behavior. Persona management now sits below each workflow's result actions
  and has an editor-only persona selector independent of the active generation
  persona. Image-to-Prompt now places its optional instruction beside the
  reference image in a compact two-column row.
- Made every `replace/append + generate` action track Forge's native render
  lifecycle. The clicked button now stays disabled from prompt transfer through
  sampling and gallery delivery, then changes to a generated-success check and
  re-enables. Missing controls, failed starts, very long waits, and completed
  runs without a new gallery image return a recoverable error state.
- Fixed `Grab last generated image` failing after a Forge generation when the
  configured temporary-image directory did not yet exist. Forgy now prepares
  that directory before Gradio post-processes the copied gallery image and
  reports a local UI error instead of raising a route traceback if creation
  fails.
- Raised Forgy's output ceiling and initial maximum-output setting from 1024 to
  4096 tokens for testing long, iterative prompt-building sessions. The shared
  VRAM-aware context limit and existing repetition/cancellation guards remain
  active.
- Fixed `replace + generate` and `append + generate` stopping after the prompt
  transfer because their browser step received the previous feedback value.
  The native Forge Generate button is now resolved through Forge's Gradio root
  and clicked after the transferred prompt has rendered.
- Fixed Forgy's conversation viewport jumping to the first message after a
  state update. New chat content now scrolls to the latest message again after
  the immediate DOM update and delayed Markdown rendering.
- Added `replace + generate` and `append + generate` actions for both txt2img
  and img2img in every prompt-producing workflow. A successful transfer now
  triggers Forge's existing native Generate button without changing tabs or
  bypassing Forge's current generation settings and queue.
- Compacted the wide Idea-to-Prompt and Image-to-Prompt transfer controls into
  one four-button row per Forge target, placing each normal action directly
  beside its matching `+ generate` action. Forgy and Prompt Refinement now use
  the same action order for a consistent studio-wide layout.
- Prevented Qwen `<think>` and `<analysis>` blocks from reaching any visible
  workflow output by preserving their markers during decoding and removing the
  complete private-reasoning block before display.
- Added Forgy-specific guardrails for the Qwen3-VL 4B encoder: the `/no_think`
  directive, a configurable bounded output ceiling, a safer 1.15 repetition
  penalty, and early termination with `repetition_stop` when an exact token
  sequence begins looping.
- Isolated persona fields and Forgy's safety-critical sampling defaults from
  Forge's label-based `ui-config.json` persistence so identically labelled
  controls in other workflow tabs can no longer overwrite them.
- Added a Forgy `Grab last generated image` action that tracks the most recently
  updated txt2img or img2img Forge gallery and copies its final image into the
  local Forgy attachment field only on explicit user request.
- Added the reserved `Ghost` persona with a deliberately empty system prompt to
  Idea-to-Prompt, Image-to-Prompt, Prompt Refinement, and Forgy while retaining
  non-empty validation for every normal user persona.
- Made all persona dropdowns visibly scrollable for longer user-created lists.
- Added the first Forgy creative-copilot workflow with local text and image
  conversation, a visible working prompt, and user-managed agent personas.
- Added a visible two-section Forgy response protocol that keeps the working
  prompt unchanged when the protocol cannot be parsed safely.
- Added session-only conversation context, bounded prompt-version history, and
  one-step prompt Undo without storing uploaded images or chat logs on disk.
- Added direct `Send to Forgy` actions from Idea-to-Prompt, Image-to-Prompt, and
  Prompt Refinement, plus Forge prompt-transfer actions inside Forgy.
- Added the same VRAM-aware sampling, seed reporting, busy state, multimodal
  Qwen3-VL path, and cooperative Stop control to Forgy turns.
- Added an iterative `Refine prompt` workflow that revises an existing prompt
  with a focused user instruction while preserving the original prompt after a
  validation error, cancellation without output, or runtime failure.
- Added independent user-managed refinement personas whose selected text is the
  complete system prompt.
- Added one-click loading of the latest Idea-to-Prompt or Image-to-Prompt result
  from within refinement plus direct `Send to prompt refinement` actions in both
  generation tabs.
- Added the same VRAM-aware sampling, deterministic seed reporting, busy state,
  cooperative Stop control, and Forge prompt-transfer actions to refinement.
- Kept the public metadata test suite dependency-free so GitHub's minimal
  Python validation can exercise gallery selection without installing Forge's
  runtime NumPy and Pillow packages.

## 0.5.0-alpha.1 - 2026-08-04

- Added local image-to-prompt generation with the active KREA2 Qwen3-VL vision
  encoder, multimodal rotary positions, and DeepStack prefill features.
- Added independent user-managed image personas and a built-in Default image
  persona.
- Added aspect-preserving image normalization, visual-token accounting, and a
  second free-VRAM check after vision encoding.
- Added separate Idea-to-Prompt and Image-to-Prompt UI tabs.
- Added an optional instruction field to Idea-to-Prompt so both workflows use a
  consistent primary-input plus instruction pattern.
- Added prominent working indicators and disabled busy buttons for both prompt
  generation workflows.
- Corrected runtime wording to identify the active Qwen3-VL text or vision
  encoder as the component performing prompt generation.
- Added pending and success feedback for persona and Forge prompt-transfer
  buttons, with transfer feedback kept on the button and reset for each new
  generation.
- Added cooperative Stop controls for idea-to-prompt and image-to-prompt jobs,
  including partial prompt recovery when tokens have already been generated.
- Added a one-click Forge-native loader for the model, VAE, and text encoder
  already selected in Forge, with readiness and progress reporting.
- Added a security policy and structured bug/feature issue forms.
- Added a read-only GitHub validation workflow with pinned action revisions.
- Added public-release contribution and security-reporting guidance.

## 0.3.0-alpha.1

- Added user-managed personas as complete system prompts.
- Added automatic VRAM detection and shared context profiles.
- Added runtime free-VRAM validation and dynamic output-window reduction.
- Increased the default requested output length to 1024 tokens.
- Added the normalized generation seed to the result status.
- Converted the complete extension UI and user-facing diagnostics to English.
- Prepared local persona storage for safe exclusion from public repositories.
- Added public repository metadata, AGPL licensing, and AI-development
  transparency documentation.

[Unreleased]: https://github.com/vibecodingtoolmaker/forgy-forge-neo-prompt-studio/compare/v0.5.2-alpha.1...HEAD
[0.5.2-alpha.1]: https://github.com/vibecodingtoolmaker/forgy-forge-neo-prompt-studio/compare/v0.5.1-alpha.1...v0.5.2-alpha.1
[0.5.1-alpha.1]: https://github.com/vibecodingtoolmaker/forgy-forge-neo-prompt-studio/compare/v0.5.0-beta.1...v0.5.1-alpha.1
[0.5.0-beta.1]: https://github.com/vibecodingtoolmaker/forgy-forge-neo-prompt-studio/compare/v0.5.0-alpha.1...v0.5.0-beta.1
[0.5.0-alpha.1]: https://github.com/vibecodingtoolmaker/forgy-forge-neo-prompt-studio/compare/v0.3.0-alpha.1...v0.5.0-alpha.1
[0.3.0-alpha.1]: https://github.com/vibecodingtoolmaker/forgy-forge-neo-prompt-studio/releases/tag/v0.3.0-alpha.1
