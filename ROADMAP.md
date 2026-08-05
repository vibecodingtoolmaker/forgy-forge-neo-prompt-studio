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

Add Z-Image as a separate backend adapter after the current adapter has been
tested and stabilized. Backend-specific behavior should remain isolated instead
of adding model-family conditionals throughout an existing implementation.

### Forgy guided generation loop

Explore returning an image to the conversation after the user explicitly starts
it with an existing `replace + generate` or `append + generate` action. Any
guided loop must keep Forge's current generation settings and must never start
another render without user confirmation.

These roadmap entries describe direction, not release commitments or schedules.
