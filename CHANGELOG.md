# Changelog

All notable changes to this project will be documented in this file.

## Unreleased

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

[Unreleased]: https://github.com/vibecodingtoolmaker/forge-krea-prompt-assistant/compare/v0.5.0-alpha.1...HEAD
[0.5.0-alpha.1]: https://github.com/vibecodingtoolmaker/forge-krea-prompt-assistant/compare/v0.3.0-alpha.1...v0.5.0-alpha.1
[0.3.0-alpha.1]: https://github.com/vibecodingtoolmaker/forge-krea-prompt-assistant/releases/tag/v0.3.0-alpha.1
