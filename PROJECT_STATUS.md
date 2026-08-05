# Project Status: Forgy Prompt Studio

Last updated: 2026-08-05

This file is the handover document for continuing development in a new Codex
chat. Read `AGENTS.md` before changing the project.

## Quick orientation

- Workspace: a Forge Neo checkout containing this repository under
  `extensions/forge-krea-prompt-assistant`
- Extension repository: the directory containing this status file
- Public repository:
  `https://github.com/vibecodingtoolmaker/forge-krea-prompt-assistant`
- Development branch: `develop`; public release branch: `main`
- Current published prerelease: `v0.5.0-beta.1` (2026-08-05)
- Previous release: `v0.5.0-alpha.1` / commit `6a0762a`
- The extension's public name is now **Forgy Prompt Studio**. The repository
  directory, Python filename, Gradio element IDs, callback names, and internal
  Forge tab ID still contain `forge_krea_prompt_assistant` or `forge-krea` on
  purpose. Renaming those identifiers is a separate compatibility migration.

The Beta 1 release includes the main script, tests, public documentation,
metadata, issue forms, all four persona examples, license header, security
policy, roadmap, changelog, `.gitignore`, `style.css`, this status file, and
`AGENTS.md`. In a fresh checkout these files must all be tracked. Local persona
stores and generated/runtime files must remain ignored.

## Current development state

The original KREA2 prompt helper has become a local prompt-studio suite with
four workflows:

1. **Forgy Chat** is the first/default tab. It supports text or text-plus-image
   conversation, a visible/editable working prompt, prompt-version Undo,
   conversation clearing, explicit attachment of the latest Forge gallery
   image, and transfer to txt2img/img2img.
2. **Idea to prompt** converts an idea plus an optional instruction into a
   polished prompt.
3. **Image to prompt** analyzes one uploaded image plus an optional instruction
   with the active vision encoder and produces text only.
4. **Refine prompt** iteratively revises an existing prompt according to a
   required refinement instruction.

All workflows have independent, user-managed persona stores. The selected
persona is the complete system prompt; the extension adds no hidden style or
content policy. `Default` is protected, and the reserved `Ghost` persona is the
only persona allowed to have an empty system prompt. Persona editor selection
is independent from the persona currently used for generation.

The current backend implementation still supports only the KREA2 model family
with its Forge-owned Qwen3-VL 4B text/vision encoder and compatible VAE. The UI,
status text, product metadata, and built-in personas are model-family neutral so
future families can be added as separate adapters.

The extension never downloads or creates a second language model. It reuses
Forge's active model patcher, encoder, tokenizer, and vision module. The compact
runtime bar can explicitly ask Forge to load the model, text encoder, and VAE
already selected in Forge's normal controls; it does not change those
selections.

## Changes implemented on 2026-08-05

### Forgy Chat and prompt refinement

- Added the full Forgy Chat workflow with session-only conversation and up to
  50 prompt undo versions.
- Added a structured Forgy response protocol using `FORGY_REPLY:` and
  `UPDATED_PROMPT:`. A missing or malformed protocol leaves the working prompt
  unchanged instead of guessing.
- Added `/no_think`, removal of visible `<think>`/`<analysis>` blocks, a 1.15
  default repetition penalty, and exact repeated-token-suffix detection for the
  chat-oriented use of Qwen3-VL 4B.
- Raised Forgy's selectable and initial output ceiling to 4096 tokens. This is
  intentionally experimental; the effective output is still limited by the
  shared VRAM-aware context window.
- Added prompt refinement with its own personas, sampling controls, Stop
  behavior, source-result import, and direct Forge actions.
- Added `Send to Forgy` and `Send to prompt refinement` paths between the
  specialized workflows.

### Forge actions and runtime behavior

- Added `replace + generate` and `append + generate` for txt2img and img2img in
  every prompt-producing workflow.
- Fixed these actions so they use the newly transferred result, resolve Forge's
  native Generate button through the Gradio root, and start Forge's existing
  queue without changing the user's generation settings.
- Added browser-side render lifecycle tracking. The clicked button remains
  disabled while Forge is active and is released after Forge returns to idle
  and a new gallery result is observed; errors and timeouts are recoverable.
- Added visible pending/success feedback to all action buttons.
- Added cooperative Stop controls to all encoder-generation workflows.
- Added `Grab last generated image` for Forgy. It follows the most recently
  updated txt2img/img2img gallery and explicitly copies its final image into the
  Forgy attachment.
- Fixed the gallery grab traceback caused by a missing Forge `output\tmp`
  directory by creating Forge's configured temporary directory before Gradio
  caches the copied PIL image.
- Fixed Forgy chat autoscroll so the conversation moves to the newest message
  after both the immediate update and delayed Markdown rendering.

### Personas, layout, and product identity

- Added separate refinement and Forgy persona stores and distributable example
  files.
- Added `Ghost` to all four stores, while keeping normal empty-persona saves
  invalid.
- Added scrollable persona dropdown styling in `style.css`.
- Added a hash-based migration for only the exact, unchanged legacy Idea and
  Image `Default` prompts. User-modified defaults and custom personas are not
  overwritten. The local ignored persona files in this checkout have already
  been migrated.
- Moved persona management below the work/result actions in all tabs and added
  an independent `Persona to edit` selector.
- Made Image-to-Prompt's image and optional instruction a compact two-column
  input row.
- Standardized each Forge target as one four-button row ordered `replace`,
  `replace + generate`, `append`, `append + generate`.
- Moved cross-workflow buttons beside the prompt field they act on and moved
  the VRAM-profile information to the bottom of every workflow.
- Promoted Forgy Chat to the first tab and moved its image attachment and Forge
  actions into the main visible flow.
- Renamed the product to **Forgy Prompt Studio**, changed the version marker to
  `0.5.0-beta.1`, and neutralized general KREA2-specific UI, persona, metadata,
  issue-form, security, and README wording.
- Updated documentation to describe KREA2/Qwen3-VL as the first backend adapter
  rather than the product identity.

The detailed chronological list is also in `CHANGELOG.md` under
`0.5.0-beta.1`.

## Important technical decisions

### Forge owns model loading and memory

- Do not edit Forge Neo core files for this extension.
- Do not use `from_pretrained()`, download weights, or retain a second model.
- `_active_krea_components()` resolves only the currently active Forge objects.
- `_load_current_forge_selection()` may only call Forge's refresh and reload
  path for the selection the user already made.
- Do not keep global references to the encoder, tokenizer, CUDA tensors, or KV
  cache. Let Forge's memory manager place/offload the active objects, reserve
  calculated cache memory, and release request-local state after use.

### Model-family neutrality means adapters, not generic guesses

- Product UI and shared workflow language must remain neutral.
- Backend-specific architecture checks and multimodal code may be explicit and
  honest. The current `_active_krea_components()`,
  `_generate_with_active_krea()`, and `_generate_with_active_krea_image()` are
  the first adapter, not a generic implementation.
- Add Z-Image or another family as an isolated adapter. Do not scatter
  model-family `if` statements through shared persona, sampling, and UI code.
- Unsupported or unknown stacks must fail clearly rather than attempting a
  speculative fallback.

### The user remains in control

- A persona is the full system prompt. Do not silently append style, content,
  NSFW, safety, or brand-specific instructions.
- Forge remains the source of truth for checkpoint, VAE, text encoder, sampler,
  seed, dimensions, extensions, and queue behavior.
- Loading and rendering require explicit user actions. Forgy may discuss or
  prepare a prompt but must never autonomously start another image render.
- An empty or malformed model result must never overwrite a useful prompt or
  trigger generation.

### Local state and privacy

- Persistent persona files are extension-local, atomically written, and ignored
  by Git.
- Forgy chat messages, uploaded images, and undo versions are session-only and
  must not be added to extension persistence implicitly.
- Diagnostic features must exclude prompts, personas, images, secrets, and
  personal paths by default.
- No additional `requirements.txt` is needed currently: runtime imports come
  from Python's standard library or packages already required and supplied by
  Forge Neo.

### Compatibility identifiers

Keep `forge_krea_prompt_assistant`, existing `forge-krea-*` element classes, the
script filename, and repository directory unchanged until an explicit migration
plan accounts for Forge `ui-config.json`, callback names, CSS selectors, user
installations, and GitHub URLs.

## Known bugs, risks, and open problems

No reproducible blocker is currently known after the latest fixes, but the
following risks remain:

- The `+ generate` lifecycle uses Forge/Gradio DOM structure, native element
  IDs, activity controls, and gallery changes. A Forge UI update can break this
  integration even while Python tests continue to pass.
- The 4096-token Forgy setting has only been used as a large test ceiling. The
  Qwen3-VL 4B encoder can still hallucinate, become repetitive, or produce a
  malformed Forgy protocol. Current guards bound common failures but do not turn
  the diffusion encoder into a general chat model.
- With `Ghost`, the Forgy protocol is normally absent; the raw reply is shown
  and the working prompt is intentionally preserved. This can look like a
  failed prompt update but is expected behavior.
- Only the KREA2/Qwen3-VL adapter exists. Z-Image and tag-oriented model-family
  adapters are not implemented.
- Forgy's history is not persistent. `Clear conversation` intentionally keeps
  the current working prompt and its undo versions; page reload or Forge restart
  discards the session history.
- Image-to-Prompt analyzes an image to produce text. It does not implement
  img2img conditioning, reference-image styling, identity editing, or automatic
  parameter changes.
- The repository/folder/internal IDs still use the old KREA2-oriented slug.
  This is deliberate for compatibility but should eventually be reviewed as a
  separately planned migration.
- Visual polish and responsive layout are not final. Very narrow browser widths,
  high UI scaling, and unusually long translated button labels may need more
  CSS work.

## Areas not yet fully tested

Automated tests cover static structure and isolated behavior, but they do not
replace these runtime checks:

- No broad GPU matrix has been run for the automatic 8/12/16/24/32 GB VRAM
  profiles. Initial development used a 32 GB-class Windows system.
- Linux, non-default temporary directories, read-only persona directories, and
  multiple Forge Neo versions have not been tested end-to-end.
- The final 4096-token Forgy path has not been stress-tested across different
  conversation lengths, attached-image sizes, VRAM tiers, and Stop timing.
- After the latest lifecycle fix, all eight txt2img/img2img
  replace/append-plus-generate paths should be manually rechecked for button
  unlock, success state, timeout recovery, and no-result recovery.
- The missing-temp-directory fix has unit coverage but should be reproduced once
  more in a fresh live Forge session by removing only the exact disposable temp
  directory and then using `Grab last generated image`.
- Chat autoscroll, persona dropdown scrollbars, bottom persona managers, compact
  layouts, and narrow-window behavior need a final browser smoke test after the
  user's normal Forge instance is restarted.
- Image-to-Prompt and Forgy-with-image need further live tests with portrait,
  landscape, alpha-channel, EXIF-rotated, and very large images.
- Cancellation has unit/structural coverage, but cancellation during vision
  prefill, late decoding, and native Forge image sampling should be manually
  distinguished. The extension Stop button cancels encoder generation; Forge's
  own interrupt control remains responsible for image sampling.
- GitHub Actions must be checked on the published release commit. Local success
  does not prove that the public Linux/Python 3.11 validation run completed.

Today's isolated Forge UI check on port 7862 confirmed the new tab/heading,
neutral not-ready and descriptive text, `Generated prompt` labels, friendly
Idea labels, neutral built-in personas, and absence of the old product wording.
That isolated instance was stopped. The user's normal Forge instance on port
7860 was not stopped or modified and must be restarted to load the latest code.

## Automated validation status

Before this handover, the following checks passed for the beta implementation:

- 43 `unittest` tests;
- Ruff lint check;
- Ruff format check;
- `git diff --check` (apart from Git's existing LF-to-CRLF warning for
  `.gitignore`);
- parsing of `metadata.ini` and the GitHub YAML files;
- an isolated live Forge UI smoke test on port 7862.

Re-run the automated checks after any further change and immediately before a
commit or release:

```powershell
cd <Forge Neo checkout>\extensions\forge-krea-prompt-assistant
& ..\..\venv\Scripts\python.exe -m unittest discover -s tests -v
& ..\..\venv\Scripts\ruff.exe check scripts\forge_krea_prompt_assistant.py tests\test_public_metadata.py
& ..\..\venv\Scripts\ruff.exe format --check scripts\forge_krea_prompt_assistant.py tests\test_public_metadata.py
git diff --check
```

## Recommended next steps

1. Confirm that tag/release `v0.5.0-beta.1`, the `main` branch commit, and the
   GitHub Actions validation all refer to the same reviewed source state.
2. Restart the user's Forge test instance and run one focused KREA2/Qwen3-VL
   smoke pass through all four workflows.
3. Specifically verify each `+ generate` button remains locked only until the
   new image reaches the correct gallery, then re-enables with a success check.
4. Re-test `Grab last generated image`, Forgy autoscroll, Stop, Undo, Clear,
   persona create/edit/delete, `Ghost`, and legacy-persona preservation.
5. Inspect `git status`, the complete diff, and ignored local persona data.
   Ensure no `personas.json`, `image_personas.json`,
   `refinement_personas.json`, `agent_personas.json`, generated image, log, or
   personal path is staged.
6. Run the four automated commands above before any follow-up release.
7. Triage Beta 1 feedback without mixing unrelated fixes into one release.
8. After the beta is stable, choose the next bounded feature from `ROADMAP.md`:
   privacy-conscious diagnostics, local prompt history, or a separate Z-Image
   adapter. Keep the guided image loop explicitly user-triggered.

## Relevant files and functions

### Runtime implementation

- `scripts/forge_krea_prompt_assistant.py`
  - Product/config constants: `EXTENSION_NAME`, `EXTENSION_VERSION`,
    `VRAM_PROFILES`, persona paths, Forgy limits, and built-in personas.
  - Persona persistence and migration: `_read_persona_store()`,
    `_write_persona_store()`, `_save_persona_fields_to()`,
    `_delete_persona_fields_from()`.
  - Runtime readiness/loading: `_forge_stack_status()`,
    `_load_current_forge_selection()`, `_active_krea_components()`.
  - Prompt construction: `_render_generation_prompt()`,
    `_render_image_generation_prompt()`, `_refinement_user_message()`,
    `_build_forgy_request()`.
  - Image preparation/multimodal prefill: `_prepare_uploaded_image()`,
    `_multimodal_embeds()`, `_prefill_with_deepstack()`.
  - Token generation: `_generate_with_active_krea()`,
    `_generate_with_active_krea_image()`, `_sample_token()`,
    `_allocate_kv_cache()`.
  - Visible-output safeguards: `_decode_generated_text()`,
    `_strip_model_thinking()`, `_repetition_loop_suffix()`,
    `_parse_forgy_response()`.
  - Workflow runners: `_run_generation()`, `_run_image_generation()`,
    `_run_refinement()`, `_run_forgy_turn()`.
  - Forge actions and UI JavaScript: `_transfer_prompt()`,
    `_forge_generate_click_js()`, `_connect_prompt_buttons()`,
    `_forgy_scroll_to_bottom_js()`.
  - Latest-gallery handling: `_capture_forge_gallery_component()`,
    `_remember_forge_gallery_source()`, `_latest_gallery_image()`,
    `_ensure_forge_temp_directory()`, `_grab_last_forge_image()`.
  - UI construction and event wiring: `_create_active_persona_controls()`,
    `_create_persona_manager()`, `_create_sampling_controls()`, `_on_ui_tabs()`.

### Tests, style, data, and public files

- `tests/test_public_metadata.py`: 43 unit/static contract tests for branding,
  persona migration, VRAM profiles, Forge-owned loading, cancellation, prompt
  transfer, native generation hooks, layout, refinement, Forgy parsing and
  safeguards, gallery grabbing, temp-directory handling, CSS, and roadmap.
- `style.css`: visible scrolling for long persona dropdown lists.
- `personas.example.json`: distributable Idea-to-Prompt personas.
- `image_personas.example.json`: distributable Image-to-Prompt personas.
- `refinement_personas.example.json`: distributable refinement personas.
- `agent_personas.example.json`: distributable Forgy personas.
- `README.md`: public behavior, installation, privacy, limitations, and current
  first-adapter documentation.
- `CHANGELOG.md`: all Beta 1 release changes and later unreleased work.
- `ROADMAP.md`: implemented refinement and planned history, diagnostics,
  Z-Image adapter, and explicitly guided generation loop.
- `metadata.ini`, `LICENSE`, `SECURITY.md`, `.github/`: public identity,
  licensing, disclosure policy, issue forms, and read-only CI validation.
- `.gitignore`: excludes local personas, temporary files, outputs, environments,
  and editor/runtime artifacts.
