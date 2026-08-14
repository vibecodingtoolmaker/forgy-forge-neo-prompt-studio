# Roadmap

The initial Idea-to-Prompt, Image-to-Prompt, Prompt Refinement, and Forgy Chat
foundation is considered complete for the 0.5 beta series. The remaining items
are recorded for later exploration.

## Implemented areas

### Refine this prompt

Revise an existing generated or pasted prompt with a focused follow-up
instruction while preserving the user's selected refinement persona and control
over the request. Refinement supports iterative revisions, VRAM-aware sampling,
seed reporting, cooperative cancellation, and direct Forge prompt transfer.

## Planned areas

### Local prompt history

Keep a small, local history of generated prompts together with useful metadata
such as persona, seed, input type, and sampling settings. Uploaded images should
not be archived automatically.

### Diagnostic report

Provide a copyable, privacy-conscious support report containing extension and
Forge versions, model component classes, VRAM/context information, and relevant
errors. Prompts, personas, images, secrets, and personal paths must be excluded
by default.

### Z-Image adapter

The first separate Z-Image Base/Turbo adapter is included in the current beta.
It reuses Forge's active Qwen3-4B encoder for text-only Forgy Chat,
Idea-to-Prompt, and Refine while capability-gating all vision paths. Remaining
work includes broader Base/Turbo prompt-quality testing, family-specific Default
personas, UI smoke testing after model switches, and bounded output tuning.

### FLUX.2 Klein adapter

The initial FLUX.2 Klein 4B/Base 4B adapter is included in the current beta.
It reuses Forge's active tied Qwen3-4B encoder for text-only Forgy Chat,
Idea-to-Prompt, and Refine while leaving Klein reference-image handling in
Forge. Automated contract coverage and a bounded real-model text smoke are
complete. Normal Forge image generation plus broader Forgy prompt-quality and
workflow testing remain part of beta validation. Klein 9B is explicitly out of
scope unless Forge exposes its separate untied LM head without a second model
loader.

### Forgy guided generation loop

Explore returning an image to the conversation after the user explicitly starts
it with an existing `replace + generate` or `append + generate` action. Any
guided loop must keep Forge's current generation settings and must never start
another render without user confirmation.

These roadmap entries describe direction, not release commitments or schedules.
