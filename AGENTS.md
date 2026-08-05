# Development Rules for Forgy Prompt Studio

Read `PROJECT_STATUS.md` before working on this repository.

## Scope and model ownership

- Keep implementation changes inside this extension. Do not modify Forge Neo
  core files to make this extension work.
- Reuse the model, text/vision encoder, tokenizer, and model patcher already
  owned by Forge. Never download or instantiate a second language model and do
  not add `from_pretrained()`.
- Forge is the source of truth for the selected checkpoint, text encoder, VAE,
  sampler, seed, dimensions, extensions, queue, VRAM placement, and offloading.
- The optional loader may refresh and reload only Forge's current user
  selection. Opening the extension must not allocate a model automatically.
- Keep encoder, tokenizer, CUDA tensors, and KV caches request-local. Do not add
  persistent global references to Forge-owned model objects.

## Architecture and compatibility

- Keep shared UI, personas, statuses, and documentation model-family neutral.
- Isolate model-family behavior in explicit adapters. The current
  KREA2/Qwen3-VL functions are the first adapter; unsupported stacks must fail
  clearly instead of using a guessed fallback.
- Preserve the internal ID `forge_krea_prompt_assistant`, repository/folder and
  script names, callback names, existing element IDs, and `forge-krea-*` CSS
  classes unless the task explicitly includes a compatibility migration.
- Do not add a `requirements.txt` for modules already supplied by Python or
  Forge. Introduce a dependency only when the feature truly requires it and
  document why Forge's environment does not already provide it.

## User control, safety, and privacy

- The selected persona is the complete system prompt. Do not append hidden
  style, content, safety, NSFW, or product-specific instructions.
- `Ghost` is the only allowed empty persona. Preserve `Default`, preserve custom
  personas, and migrate a legacy default only when its exact known hash matches.
- Never start model loading or image rendering autonomously. Native Forge
  generation requires an explicit `replace + generate` or `append + generate`
  click from the user.
- Never trigger generation from an empty result or overwrite a valid working
  prompt after an error, cancellation without output, or malformed Forgy
  response.
- Keep persistent persona files extension-local, atomic, and ignored by Git.
  Keep Forgy chat, images, and undo history session-only unless the user
  explicitly requests a reviewed persistence design.
- Exclude prompts, personas, images, secrets, and personal filesystem paths from
  diagnostics and public artifacts by default.

## UI and Forge integration

- Use Forge's existing queue and native controls rather than duplicating
  generation settings or bypassing the queue.
- Treat DOM-dependent `+ generate`, gallery capture, and autoscroll code as
  version-sensitive. Verify actual live behavior after changing it; static
  string tests alone are insufficient.
- Keep the workflow order `Forgy Chat`, `Idea to prompt`, `Image to prompt`,
  `Refine prompt` unless the task explicitly changes the product flow.
- Keep the standard Forge action order per target: `replace`, `replace +
  generate`, `append`, `append + generate`.
- Keep active persona selection near the workflow input, persona management
  below result actions, and VRAM information at the bottom of each workflow.

## Verification and repository hygiene

- Preserve unrelated user changes in this working tree. Inspect `git status`
  before and after editing.
- Run the extension unit suite, Ruff lint, Ruff format check, and
  `git diff --check` after code changes and before release work.
- For UI/runtime changes, restart an isolated Forge instance when practical and
  verify the real interaction without stopping or altering the user's live
  instance.
- Record separately what passed automated tests, what was observed live, and
  what remains untested.
- Never stage local persona JSON, images, logs, generated output, temporary
  files, secrets, or personal paths.
- Do not commit, tag, push, merge, publish a release, or change GitHub settings
  unless the user explicitly requests that external mutation.
