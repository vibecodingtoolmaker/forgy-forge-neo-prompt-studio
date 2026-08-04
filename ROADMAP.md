# Roadmap

The KREA2 idea-to-prompt and image-to-prompt foundation is considered complete
for the 0.5 alpha series. The following items are recorded for later exploration
after broader testing of the current release.

## Planned areas

### Local prompt history

Keep a small, local history of generated prompts together with useful metadata
such as persona, seed, input type, and sampling settings. Uploaded images should
not be archived automatically.

### Refine this prompt

Allow an existing generated prompt to be revised with a focused follow-up
instruction while preserving the user's selected persona and control over the
request.

### Diagnostic report

Provide a copyable, privacy-conscious support report containing extension and
Forge versions, model component classes, VRAM/context information, and relevant
errors. Prompts, personas, images, secrets, and personal paths must be excluded
by default.

### Z-Image adapter

Add Z-Image as a separate backend adapter after the KREA2 path has been tested
and stabilized. Backend-specific behavior should remain isolated instead of
adding model-family conditionals throughout the KREA2 implementation.

These roadmap entries describe direction, not release commitments or schedules.
