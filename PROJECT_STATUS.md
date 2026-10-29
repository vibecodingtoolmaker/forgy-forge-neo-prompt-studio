# Project Status: Forgy — Forge Neo Prompt Studio

Last updated: 2026-10-29

This file is the handover document for continuing development in a new Codex
chat. Read `AGENTS.md` before changing the project.

## Tagesfortschritt 2026-10-29 — experimental MOSS-Image text/vision adapter

### Outcome

- Added an explicit `moss-image-qwen35-9b` adapter for Forge Neo's
  current `backend.diffusion_engine.moss_image.MOSSImage` stack. It validates
  the exact Qwen3.5-9B encoder, tokenizer, processing engine, 4096-wide/248320
  vocabulary/32-layer contract, vision encoder, and Forge `ModelPatcher` before
  enabling Forgy Chat, Idea-to-Prompt, Image-to-Prompt, and Refine.
- MOSS's official config declares an untied LM head. Forge retains the complete
  conditioning backbone and final norm but intentionally omits `lm_head` and MTP
  when it assembles the text encoder. Forgy therefore resolves exactly one dense
  BF16, FP16, or FP32 MOSS encoder from Forge's selected additional modules and
  loads only its 248320-by-4096 `lm_head` for the request. It never scans for a
  fallback, creates a second encoder, or persists the head.
- Extended the request-local Qwen3.5 causal runtime for Forge's native 9B
  backbone. The path reuses all active Forge layers, final norm, tokenizer, and
  patcher with recurrent Gated DeltaNet state plus full-attention KV caches. A
  1024-token ceiling bounds the experimental provider.
- Added interleaved three-axis Qwen3.5 MRoPE and a multimodal prefill that accepts
  exactly one Forge-produced image embedding sequence. MOSS Image-to-Prompt uses
  the active Forge vision module and tokenizer; it does not load or retain a
  second tagger, vision model, or language model.
- Confirmed from the official MOSS guide that SFT accepts natural-language
  captions, tag sequences, and JSON layouts. MOSS workflow contracts now defer
  caption-versus-tag formatting to the selected persona and preserve the current
  dialect during refinement. Local ignored persona stores contain separate MOSS
  Caption, Illustrious-style Tag, image-tagging, refinement, and Forgy variants
  for manual comparison; existing personas remain unchanged.
- Added a separate MOSS negative-prompt channel to all four workflows. The
  deterministic suggestion starts with `worst quality, watermark, jpeg artifact`
  and adds only request-relevant anatomy, typography, style, no-people, or
  safe-content terms. Illustrious-style numbered human tags such as `1girl` are
  recognized, while animal-only prompts such as penguins never receive human
  hand or foot defects. The visible field is editable, and an explicit checkbox
  controls whether Forge transfer actions replace or merge it into Forge's
  native negative-prompt field.
- Fixed the main MOSS performance fault without changing Forge. The selected
  encoder stores its token embedding in FP16 but 377 remaining weighted modules
  in BF16. Forgy previously used the embedding dtype for hidden states and the
  LM head, forcing Forge `manual_cast` to convert dense 9B backbone weights
  repeatedly. The runtime now validates the transformer projection dtype,
  converts embeddings once, and loads the request-local head in that dtype.
- Extended the real MOSS smoke profiler with synchronized model placement and
  stage timings, CUDA-event totals for MLP/full/linear attention, per-call
  latency distributions, module dtype/placement census, and peak VRAM.
- The first real smoke exposed that Forge's normal low-VRAM placement left
  862 MiB of the encoder on CPU, which is incompatible with this direct cached
  runtime. The final implementation requests a full single-device load and also
  fails clearly if Forge still reports partial placement. No implicit mixed
  CPU/CUDA execution remains.

### Real and automated evidence

- A real RTX 5090 smoke used local
  `mossRefinedRealism_v108steps_bf16.safetensors`,
  `qwen3.5_9b_uncensored_bf16.safetensors`, and `flux2-vae.safetensors`.
  Forge loaded the complete 17.95 GiB active encoder on CUDA. Forgy completed
  311 input tokens plus 16 greedy output tokens and returned the clean visible
  prefix `A red fox resting beside a quiet alpine lake at sunrise, captured in a`.
- An initial pre-optimization four-token profile measured 20.5 seconds inside generation:
  1.9 seconds for head loading, 9.6 seconds for 311-token prefill, 4.7 seconds for
  four decodes, and 0.04 seconds for LM-head projection. The Qwen3.5 execution,
  not the untied head, is the optimization boundary.
- The post-MRoPE text regression smoke returned the same visible prefix and
  completed in 15.1 seconds inside generation, including 7.0 seconds of prefill
  and 4.4 seconds for four decodes. Timing therefore varies materially with the
  warmed local stack; both runs preserve the same functional result.
- A real synthetic-image smoke completed 857 input tokens, including 576 visual
  tokens, plus four greedy output tokens. It returned the visible prefix
  `A flat, minimalist` in 34.9 seconds inside generation; multimodal prefill took
  13.2 seconds and four decodes took 3.8 seconds.
- The controlled detailed A/B used the same MOSS checkpoint, encoder, persona,
  369 input tokens, and 16 greedy output tokens. Generation fell from 43.52 to
  8.73 seconds, prefill from 11.64 to 1.26 seconds, and decode from 27.74 to
  1.48 seconds. Median decode latency fell from 1.706 to 0.056 seconds per token,
  while the visible output prefix remained identical. Peak allocated VRAM fell
  from 32.16 to 29.90 GiB.
- A longer optimized run produced 64 visible tokens in 8.94 seconds total, with
  1.22 seconds of prefill and 3.79 seconds of decode. The optimized synthetic
  vision smoke completed 871 input plus four output tokens in 6.49 seconds;
  multimodal prefill took 1.14 seconds and decode 0.33 seconds.
- A following non-profile production-path smoke also passed with the selected
  local persona: 369 input tokens, 16 output tokens, the expected visible fox
  prompt prefix, and `MOSS_TEXT_GENERATION_OK`.
- `unittest discover -s tests -v`: 101 tests passed. Ruff lint and format checks
  passed across 30 Python files. The Forge bootstrap registered all five adapters
  and printed `NO_PERSISTENT_MODEL_CONTEXT_OK`. A model-free Gradio build smoke
  constructed exactly four MOSS negative fields and four transfer checkboxes and
  printed `FORGY_MOSS_NEGATIVE_UI_BUILD_OK 4`. `git diff --check` reported no
  whitespace errors, only the pre-existing `style.css` LF-to-CRLF warning.

### Boundaries and repository state

- Dense BF16/FP16/FP32 LM heads are supported. Common quantized MOSS head formats
  are intentionally rejected until their Forge quantization metadata and output
  projection can be reused without materializing a guessed dense copy.
- The selected local BF16 encoder contains 760 tensors and no MTP tensors. A
  speculative MTP port would therefore provide no benefit for this real stack.
- Visible post-restart Gradio use, broader sampling quality, cancellation,
  lower-VRAM hardware, real-photo MOSS vision quality, and a following native
  image generation in the same long-lived Forge process remain untested.
- Work remains uncommitted in the intentionally dirty nested `develop` checkout,
  which is still one commit behind `origin/develop`. No Forge core file, selected
  model option, persona file, model weight, commit, tag, push, or release was
  changed.

## Tagesabschluss 2026-10-08 — Forge Neo 2.29.2 KV hotfix

### Outcome

- The current Forge Neo `neo` line was verified at `831d242d`, matching its
  remote at the time of diagnosis. Commit `83391b92` (`Qwen-Image-2.1`) removed
  the top-level `past_key_values` input/output plumbing from the shared
  `Llama2_` forward path while leaving the per-layer Attention and Transformer
  cache implementation intact. That exact contract change caused Forgy's
  visible `returned no compatible KV cache` failure after model loading.
- Added `forgy/forge_kv_compat.py`. It keeps Forge's earlier native three-value
  forward contract authoritative and, for the current Forge contract, reconnects
  only the existing cache-capable transformer layers. Cache tensors, layer
  count, indices, and cross-layer consistency fail closed. No Forge core file,
  second model, persistent model reference, or no-cache decode fallback was
  introduced.
- Routed the shared KREA2/Z-Image/FLUX.2 Klein text loop and KREA2 multimodal
  prefill/decode path through that request-local compatibility boundary. The
  unreleased stock-Anima path in this dirty development tree uses the same
  boundary; the independent Qwen3.5 provider retains its own request-local
  runtime.
- After a Forge restart, the maintainer confirmed the original KREA2
  Idea-to-Prompt reproduction now completes successfully. This is live proof
  for that workflow and active model only, not a live qualification of every
  text family, KREA2 vision, long output, cancellation, or lower-VRAM offload.

### Published hotfix

- Published the isolated public prerelease
  `v0.5.2-beta.3` on 2026-10-08:
  `https://github.com/vibecodingtoolmaker/forgy-forge-neo-prompt-studio/releases/tag/v0.5.2-beta.3`.
- Public `develop` points to
  `b1c3bcd1115eeeb792994e81e10d1dcee81266b3` (`Release Forgy
  v0.5.2-beta.3 hotfix`). Public `main` and the annotated release tag resolve to
  merge commit `4846ab67a475615b9adf474b20f5072169413c03`.
- Both public GitHub Actions runs completed successfully: develop run
  `37703885872` and main run `37703974818`.
- The release was built in a clean temporary worktree directly on
  `v0.5.2-beta.2`. It contains only the KV compatibility code, focused tests,
  version metadata, and release documentation. The separate Anima/Qwen3.5
  development work was intentionally excluded. The temporary worktree and
  helper branch were removed after the verified push; all release content is
  recoverable from `develop`, `main`, and `v0.5.2-beta.3`.

### Exact local repository state

- Continue working only in the nested repository
  `H:\DEVELOPMENT-sd-webui-forge-neo\extensions\forge-krea-prompt-assistant`.
  The surrounding Forge root remains reference-only and its unrelated dirty
  files were not changed.
- The original checkout remains intentionally dirty on local `develop` at
  `e4d470ed12e4d17c347998f407ba836859726e2b`. It is now one commit behind
  `origin/develop` because the isolated hotfix commit was pushed without
  rewriting or stashing this worktree.
- This dirty tree still contains the complete unreleased Anima/Qwen3.5 work and
  its documentation, plus the locally proven KV compatibility module and test.
  Its product version marker remains `0.5.2-beta.2`; the clean public release
  tree carries `0.5.2-beta.3`. Do not run a blind pull, reset, checkout, merge,
  or cherry-pick: the remote hotfix overlaps the locally modified main script
  and the untracked KV compatibility files.
- No local persona JSON, image, prompt, model weight, supplemental tensor,
  secret, log, or generated output was staged or included in the release.

### Verification boundary

- The complete dirty development tree passed 84 local `unittest` cases, Ruff
  lint, Ruff format check, and Forge bootstrap with all four adapters plus
  `NO_PERSISTENT_MODEL_CONTEXT_OK` after the compatibility fix.
- The exact clean public release tree passed all 69 local test cases: 66
  dependency-free public checks and three Forge/PyTorch KV contract checks. The
  latter include a real current-Forge `Llama2_` comparison proving cached
  prefill-plus-decode matches the equivalent full-context final hidden state.
- A `python -S` simulation confirmed that the three Forge/PyTorch-only cases
  skip cleanly when the public CI environment has no Torch installation. Ruff,
  format, diff checks, and the three-adapter Forge bootstrap passed on the exact
  `main` merge tree. Both public Linux/Python 3.11 Actions then passed.
- Automated and bootstrap checks prove import, cache, dispatch, and lifecycle
  contracts. The maintainer's restarted Idea-to-Prompt run is the only new live
  workflow evidence from this hotfix session.

### Resume point

1. Read `AGENTS.md` and this section, then inspect `git status --short --branch`
   and the complete dirty diff. Preserve all existing Anima/Qwen3.5 work.
2. Reconcile `origin/develop`/`b1c3bcd` into the dirty development state only
   after comparing the overlapping script, `forgy/forge_kv_compat.py`, tests,
   version metadata, and documentation. Do not assume a clean cherry-pick.
3. Keep `v0.5.2-beta.3` immutable and isolated. Any Anima/Qwen3.5 publication
   needs its own reviewed release scope after the known selected-persona
   contract is resolved and the outstanding live gates are revisited.
4. After reconciliation or any further code edit, rerun the full unit suite,
   Ruff check, Ruff format check, Forge bootstrap, and `git diff --check` before
   staging or release work.

## Abschluss 2026-09-05 — Anima resume point

### Exact repository state

- Work only in the nested repository
  `H:\DEVELOPMENT-sd-webui-forge-neo\extensions\forge-krea-prompt-assistant`.
  Forge root remains reference-read-only.
- The nested repository is on `develop` at
  `977ab80f1413a87015b30a2fe18eff03c0e4d9cd`, matching `origin/develop`.
  The complete Anima Parts I/II implementation is intentionally uncommitted in
  the current dirty worktree; preserve and review the full diff before any
  release work.
- Nothing from this continuation was staged, committed, tagged, pushed, merged,
  or released. The third-party `sd-forge-qwen-35-encoder` repository is still
  clean on `main` and was not modified.

### Proven current behavior

- Stock Anima/Qwen3-0.6B resolves through Forge-owned components and performs
  bounded request-local decoding. Real new-prompt and refinement smokes each
  completed in about 5.2 seconds with 128 output tokens; the former 537-second
  full-context path is no longer used. These are mechanical/runtime and narrow
  prompt-quality proofs, not broad semantic qualification.
- The Qwen3.5 provider resolver now recognizes Forge's actual
  `forge_qwen35_encoder.py.Qwen35EncoderForForge` ScriptRunner registration. A
  live registry probe reached the intended no-cache diagnostic instead of the
  false adapter-not-found error.
- The complete cache-backed Anima/Qwen3.5 RTX 5090 smoke passed again after that
  resolver fix: 320 input tokens, 8 output tokens, visible prefix
  `A wide cinematic composition featuring a red`.
- Current automated checks are green: 81 dependency-free tests, Ruff check,
  Ruff format check across 24 Python files, Forge bootstrap with all four
  adapters and no persistent model context, and `git diff --check`. The latter
  reports only Git's existing `style.css` LF-to-CRLF working-copy warning.

### Open runtime and product gates

- Restart Forge before the next visible test so the provider-discovery fix is
  actually loaded. For Qwen3.5, first enable `Qwen3.5 Text Encoder (Anima)` in
  txt2img or img2img, select `qwen35_4b.safetensors`, complete one normal Forge
  generation, and only then choose `Qwen3.5 4B adapter cache (experimental)` in
  Forgy. Selecting the Forgy provider alone intentionally does not load 4B.
- Run the first stock-0.6B quality baseline with `Enable and show Qwen model
  thinking` disabled. The last supplied screenshot had it enabled; `/think`
  changes the request and consumes the same small output-token budget.
- Known Persona contract issue: stock-0.6B Forgy Chat currently replaces the
  visibly selected persona with `DEFAULT_PERSONA_PROMPT` or
  `DEFAULT_REFINEMENT_PERSONA_PROMPT` inside its compact path. This conflicts
  with the repository rule that the selected persona is the complete system
  prompt. Do not paper over it with another hidden instruction. Capture one
  current non-thinking 0.6B output, then make the compact Anima behavior visible
  and user-controlled while keeping family-specific logic in the Anima adapter.
- Visible post-restart Gradio confirmation, broad 0.6B and Qwen3.5 prompt-quality
  testing, longer Qwen3.5 output, lower-VRAM/offload behavior, and normal Anima
  image generation remain unproven.

### Recommended first actions in the new chat

1. Read `AGENTS.md` and this top handoff section, then inspect `git status` and
   the complete dirty diff without resetting it.
2. Restart Forge and record one stock-0.6B Forgy Chat response with thinking off,
   including elapsed time, visible reply, updated prompt, and any fallback note.
3. Resolve the Persona contract based on that evidence. Keep the selected
   persona authoritative; if 0.6B needs a compact persona, expose it explicitly
   rather than silently substituting it.
4. Populate the third-party Qwen3.5 cache through one user-triggered normal Forge
   generation, then run the visible Forgy Qwen3.5 request and confirm the status
   no longer reports adapter-not-found.
5. Re-run unit, Ruff, format, bootstrap, and diff checks after any code change.

## Tagesfortschritt 2026-08-28 — Anima Parts I and II

### Part I: stock Anima/Qwen3-0.6B adapter

- Added the isolated `anima-qwen3-0.6b` adapter. It recognizes only Forge's
  exact `Anima` engine with `AnimaTextProcessingEngine`, the original
  `Qwen3_06B` encoder and tokenizer, 1024-wide/151936-vocabulary contract,
  trained final LM norm, tied input/output embedding, and `ModelPatcher`.
- The adapter is text-only and enables Forgy Chat, Idea-to-Prompt, and Refine.
  Image-to-Prompt and Forgy image attachment remain disabled.
- Added Anima-owned Qwen role rendering and workflow contracts without changing
  KREA2, Z-Image, or FLUX.2 Klein behavior.
- Added an Anima-specific visible-output boundary that removes a leading plain
  `Assistant:` label and stops at a later generated `Human:`, `User:`, or
  `Assistant:` role transition. This bounds the role leakage observed in the
  first real 32-token adapter smoke without rewriting the model's prompt text.
- Added a dedicated runtime using the already proven `torch.no_grad()` boundary,
  Forge's preallocated request-local KV-cache contract, an FP32 cache matching
  the manual-cast Q/K/V computation, and a zero-copy tied output projection. A
  256-token provider ceiling bounds runtime and cache size. No second model or
  persistent Forge object is introduced.
- A visible Forgy test exposed the former full-context decoder: 537 seconds for
  a response that copied the `UPDATED_PROMPT` schema placeholder instead of
  producing a prompt. The parser now rejects that exact template echo.
- Added a stock-0.6B-only compact Forgy mode. New messages use a normalized
  Idea-to-Prompt request; an existing working prompt uses the refinement
  contract. Valid prompt text is wrapped into Forgy's UI protocol by the
  application. Obvious Unicode, JSON-like, repeated, or instruction-copy garbage
  is rejected; new prompts fall back visibly to the cleaned user request and
  failed refinements preserve the current prompt.
- Added `tests/forge_anima_text_smoke.py` to exercise the real registered adapter
  rather than only the earlier feasibility loop.

### Part II: explicit cached Qwen3.5 provider

- The existing `sd-forge-qwen-35-encoder` checkout remains clean and unmodified.
  It owns and Forge-manages the separately loaded Qwen3.5 4B conditioning model;
  Forgy never calls its `_load_model()` or retains another 4B model reference.
- Header inspection of the local `qwen35_4b.safetensors` found the 248320 x 2560
  BF16 tied embedding and hybrid backbone, but no `lm_head` and no trained
  2560-wide causal LM final norm. Its only final `norm.*` tensors implement the
  Anima conditioning projection `2560 -> 1024`.
- The official Qwen3.5-4B configuration declares tied word embeddings. The exact
  trained 2560-wide final RMSNorm was therefore range-extracted, with explicit
  user approval, from pinned official revision
  `851bf6e806efd8d0a36b00ddf55e13ccb7b8cd0a`. The 5120-byte tensor has SHA-256
  `96294c653d4e8ab158cdae0b7e794559654ffdc62db880725ed714d47f708621` and
  is stored locally as `models/text_encoder/qwen35_4b_lm_norm.safetensors` in a
  5552-byte provenance-carrying safetensors file. The source shard itself was
  not downloaded.
- `scripts/prepare_qwen35_lm_norm.py` makes that extraction reproducible using
  bounded HTTP ranges, exact source revision/tensor/dtype/shape/hash checks, a
  visible editable destination, refusal to overwrite without `--force`, and an
  atomic final replace. Forgy still performs no automatic model or tensor
  download.
- Norm-only speech was falsified by a real four-token RTX 5090 diagnostic. The
  cached 4.138B-parameter conversion loaded through its own Forge
  `ModelPatcher`, but decoded token IDs `[3791, 177281, 103216, 246375]` as the
  escaped non-language fragment `ylevere\\u7574 pono`.
- `scripts/prepare_qwen35_causal_tail.py` range-extracted the four exact missing
  layer-31 tensors after explicit user approval: post-attention norm plus
  9216-wide gate/up/down MLP projections, 141,562,880 raw BF16 bytes (about
  135 MiB). The local provenance-carrying result is
  `models/text_encoder/qwen35_4b_causal_tail.safetensors`; all shapes, dtypes,
  and per-tensor SHA-256 values were verified without downloading a full shard.
- Supplying the official tail to the local converted forward still produced the
  identical non-language tokens. Source comparison then proved the deeper
  mismatch: the conversion interpreted Qwen3.5 Gated DeltaNet as Mamba-like
  recurrence, used the wrong Q/K/V head split, raw rather than `(1 + weight)`
  Qwen3.5 RMSNorm, full RoPE at theta 1,000,000 instead of partial 64-wide RoPE
  at theta 10,000,000, and SiLU instead of sigmoid attention gating.
- Added `forgy/adapters/qwen35_runtime.py`, adapted with Apache-2.0 attribution
  from pinned Transformers commit
  `83d46aa2a2c47bef8350580f30981c974add0983`. It applies the official tensor-free
  computation to the existing cached backbone, including chunked prefill and
  request-local recurrent/KV decode caches, all layer MLPs, final norm, and tied
  output projection. The five supplemental tensors are provenance-checked,
  loaded only for the request, and released with its cache.
- Added an off-by-default, non-persistent `Anima text provider` selector under
  model diagnostics. `Original Qwen3 0.6B` remains the default. `Qwen3.5 4B
  adapter cache (experimental)` requires exactly one cached
  `qwen35_4b.safetensors` provider from `sd-forge-qwen-35-encoder`; zero,
  multiple, renamed, incomplete, or incompatible providers fail closed without
  silently falling back.
- Fixed live provider discovery after a Forge ScriptRunner probe showed that the
  installed class is registered as
  `forge_qwen35_encoder.py.Qwen35EncoderForForge`, while direct imports use the
  same module name without `.py`. The matcher now accepts both exact forms. The
  next live registry probe resolved the installed provider and correctly
  reported only that its 4B cache had not yet been populated.
- Fixed the causal-tail preparation utility's sibling import so Forge can scan
  the extension's script directory without emitting a startup import error.
- Added `tests/forge_anima_qwen35_text_smoke.py` for the complete registered
  Anima adapter/provider path. The earlier tail diagnostic now exercises the
  same corrected official causal runtime in isolation.

### Current verification boundary

- 78 dependency-free tests passed, including Forge's `.py`-suffixed live provider
  registration, Anima provider selection,
  fail-closed dispatch, exact Qwen3.5 `(1 + weight)` norm semantics, capability,
  prompt-contract, role-boundary, and no-persistent-state checks.
- The Forge bootstrap check passed with all four adapters registered and no
  persistent live model context.
- The corrected isolated Qwen3.5 runtime loaded the real 4.138B-parameter local
  backbone via its Forge `ModelPatcher` and greedily decoded four tokens as
  `[32, 3160, 5072, 314]` / `A close-up of`.
- The complete Anima/Qwen3.5 provider smoke passed on the RTX 5090 with the real
  Anima checkpoint, original 0.6B Anima component, WAN VAE, cached Qwen3.5 4B
  provider, and supplemental tail: 320 input tokens, 8 output tokens, clean
  prefix `A wide cinematic composition featuring a red`. The same complete smoke
  passed again after the live-provider discovery fix.
- A fresh default-provider regression smoke also passed: 305 input tokens,
  8 output tokens, clean prefix `A red fox resting beside a`. The new selector
  therefore did not replace or regress the original 0.6B route.
- The corrected original-provider Forgy smoke completed a 308-input/128-output
  new-prompt request in 5.215 seconds and returned a 535-character polished fox
  prompt. A second 268-input/128-output refinement completed in 5.227 seconds,
  preserved the subject and sunrise, and applied the requested camera change.
  A deliberately malformed run exercised the transparent `quality_fallback`
  instead of accepting garbage.
- Ruff check and format check passed across all 24 Python files, and
  `git diff --check` reported no whitespace error (only the existing `style.css`
  LF-to-CRLF warning). A normal visible Gradio request, broad prompt-quality
  qualification, post-fix visible Gradio confirmation, longer Qwen3.5 outputs,
  lower-VRAM/offload behavior, and Anima image generation remain unproven.
- Nothing was staged, committed, tagged, pushed, merged, or released.

## Tagesabschluss 2026-08-23 — VL tagging baseline and resume point

### Exact repository and runtime state

- The nested extension repository is on `develop` at
  `977ab80f1413a87015b30a2fe18eff03c0e4d9cd`, matching `origin/develop` before
  today's handoff edit. Nothing was staged, committed, tagged, pushed, merged,
  or released today.
- The worktree was already intentionally dirty. It contains an unreleased Anima
  feasibility diagnostic plus pending passive model-detection/UI work in
  `PROJECT_STATUS.md`, `scripts/forge_krea_prompt_assistant.py`, `style.css`,
  `tests/test_public_metadata.py`, and the untracked
  `tests/forge_anima_qwen06_full_context_smoke.py`. Preserve and review this work
  as one local continuation state; do not reset it.
- A local ignored image-persona file contains the experimental tagging persona.
  It is private local test data and must not be staged or copied into a public
  example file. No source image or generated tag output was added to Git.
- The user's normal Forge instance supplied the live observations below. It was
  not stopped, restarted, or otherwise managed during this handoff.

### What today's tagging run actually proved

- The current KREA2/Qwen3-VL image-to-prompt path accepted an image and a local
  one-line tagging persona and returned visible text. This proves the mechanical
  vision/generation route for that run only.
- An initial response mixed a prose caption with a plausible short tag line.
  After the persona was tightened to require one unique comma-separated line,
  generic handling of ambiguous material, and a hard textual limit, subsequent
  responses expanded into very long invented tag families and semantic
  enumeration loops. Persona wording alone therefore did not enforce the
  requested tag count or stop condition.
- Do not interpret these runs as a general verdict on every local Qwen3-VL
  checkpoint: the exact checkpoint variant and full sampling settings were not
  captured in this handoff. They do show that the tested configuration is not a
  trustworthy default tagger without independent validation and application-side
  output limits.
- The existing token-loop guard detects exact repeated suffix blocks only, with
  a maximum block size of 128 tokens. It can stop short literal token cycles but
  does not detect a long semantic enumeration whose terms keep changing. The
  observed failure is therefore compatible with the current implementation and
  is not evidence that the guard ran incorrectly.
- A future tagger path should validate and truncate structured tag output at the
  application boundary. A prompt-stated maximum is guidance to the model, not a
  hard guarantee. Dataset-specific canonical vocabularies for SDXL/Illustrious,
  Pony, and NAI remain a separate future design decision and were not established
  today.

### Open live UI regression

- With no text encoder ready, controls below the top status area were observed
  moving downward rhythmically. After a model became ready, an image input was
  observed shifting downward. This is unresolved live behavior, not a cosmetic
  fix that can be declared complete.
- The pending local implementation polls readiness through a one-second
  `gr.Timer`, routes the polling result through hidden state, disables the timer
  after readiness, and reserves minimum height for the affected Markdown areas.
  Its isolated/static tests do not reproduce browser layout. The timer/state
  event path and the real rendered heights are the first suspects to isolate,
  but no root cause has been proven.
- Reproduce this in a fresh isolated Forge browser session before changing more
  CSS. Record the layout both while no encoder is ready and across the transition
  to a loaded encoder; verify that the timer itself does not cause recurring DOM
  reflow and that status/capability messages have stable measured dimensions.

### Current VL integration boundary

- `krea2-qwen3-vl` and the experimental `moss-image-qwen35-9b` adapter expose
  vision input and Image-to-Prompt. Z-Image/Qwen3-4B and FLUX.2 Klein/Qwen3-4B
  are intentionally text-only.
- Multiple locally available Qwen3-VL-4B weight variants still exercise the
  same Forgy adapter and model-family contract. They are useful checkpoint
  comparisons, but they are not independent VL architectures.
- No second VL-family adapter and no explicit SDXL/Illustrious/Pony/NAI tagging
  sideload currently exists. Opening Forgy must continue to avoid loading a
  second model automatically. Any sideload proposal must require an explicit
  user action, show the additional VRAM cost, provide a reliable unload path,
  keep Forge-owned objects request-local, and fail closed for unknown stacks.

### Immediate next-session plan

1. Read this section and `AGENTS.md`, then inspect the complete dirty diff before
   editing. Preserve the ignored persona and all unrelated local work.
2. Reproduce and isolate the rhythmic UI movement in an isolated browser session.
   Keep this as a small correctness fix separate from the VL architecture work.
3. Inventory locally available vision/tagger models and choose two or three
   genuinely different candidate architectures. Do not download or load a model
   merely while scanning the inventory.
4. Before implementation, propose the explicit sideload lifecycle and adapter
   boundary: selection, architecture validation, vision preprocessing/chat
   template, VRAM warning, request-local ownership, cancellation, unload, and
   failure behavior.
5. Define one fixed local comparison set and record the exact checkpoint,
   sampling settings, persona version, output-token limit, latency, and VRAM for
   every run. Keep private images and raw outputs out of Git.
6. Score candidates with application-side checks for one-line format, maximum
   tag count, uniqueness, repetition families, unsupported anatomy/relations,
   and visible-fact precision. Only then select a default TE and later add
   model-family vocabulary profiles.

### Handoff verification

- The complete current automatic suite passed: 66 dependency-free tests.
- Ruff lint and Ruff format check passed for the script, `forgy`, and `tests`.
- `git diff --check` passed, with only Git's existing LF-to-CRLF warning for
  `style.css`.
- No fresh browser/runtime smoke was performed by Codex for this handoff. The
  tagging behavior and layout movement above are user-observed live evidence;
  their exact checkpoint/settings and browser-level cause remain unverified.

Resume from the nested repository with:

```powershell
git status --short --branch
git diff -- PROJECT_STATUS.md scripts/forge_krea_prompt_assistant.py style.css tests/test_public_metadata.py
rg -n "vision_input|IMAGE_TO_PROMPT|MODEL_MANAGER|gr.Timer" forgy scripts tests
& ..\..\venv\Scripts\python.exe -m unittest discover -s tests -v
& ..\..\venv\Scripts\ruff.exe check scripts\forge_krea_prompt_assistant.py forgy tests
& ..\..\venv\Scripts\ruff.exe format --check scripts\forge_krea_prompt_assistant.py forgy tests
git diff --check
```

## Quick orientation

- Workspace: a Forge Neo checkout containing this extension repository. An
  existing working copy may retain its pre-rename directory name.
- Extension repository: the directory containing this status file
- Public repository:
  `https://github.com/vibecodingtoolmaker/forgy-forge-neo-prompt-studio`
- Development branch: `develop`; stable public release branch: `main`;
  development prereleases may be tagged from `develop`
- Current main prerelease: `v0.5.2-beta.2` (2026-10-06)
- Previous main prerelease: `v0.5.2-beta.1` (2026-08-14)
- Previous development prerelease: `v0.5.2-alpha.1` (2026-08-13)
- The extension's public name is now **Forgy — Forge Neo Prompt Studio**. The
  Python filename, Gradio element IDs, callback names, and internal Forge tab ID
  still contain `forge_krea_prompt_assistant` or `forge-krea` on purpose.
  Existing installations may also retain their original directory name.
  Renaming those identifiers is a separate compatibility migration.

The Beta 0.5.2 main prerelease includes the main script, modular
`forgy` package, architecture document, tests, public documentation, metadata,
issue forms, all four persona examples, license header, security policy,
roadmap, changelog, `.gitignore`, `style.css`, this status file, and `AGENTS.md`.
In a fresh checkout these files must all be tracked. Local persona stores and
generated/runtime files must remain ignored.

## Current development state

The original KREA2 prompt helper has become a local prompt-studio suite with
four workflows:

1. **Forgy Chat** is the first/default tab. It supports text or text-plus-image
   conversation, a visible/editable working prompt, prompt-version Undo,
   conversation clearing, an undoable prompt-only clear control, explicit
   attachment and enlargement of the latest Forge gallery image, restoration of
   the latest recorded generation prompt, and transfer to txt2img/img2img.
2. **Idea to prompt** converts an idea plus an optional instruction into a
   polished prompt.
3. **Image to prompt** analyzes one uploaded image plus an optional instruction
   with the active vision encoder and produces text only.
4. **Refine prompt** iteratively revises an existing prompt according to a
   required refinement instruction.

All workflows have independent, user-managed persona stores. The selected
persona is the complete system prompt. A model adapter may add a narrow,
documented workflow contract to the user request when its encoder needs clearer
task boundaries; it does not replace the persona or add a hidden style persona.
`Default` is protected, and the reserved `Ghost` persona is the only persona
allowed to have an empty system prompt. Forgy Chat's combined specialist is now
its built-in `Default`; the former unchanged default is retained as `Default
Legacy`, with exact-hash migration that preserves edited and custom personas.
Persona editor selection is independent from the persona currently used for
generation.

The development branch supports KREA2 with its Forge-owned Qwen3-VL 4B
text/vision encoder plus text-only Z-Image Base/Turbo, FLUX.2 Klein 4B/Base 4B,
and stock Anima/Qwen3-0.6B adapters using Forge's active encoders. All resolve through an
explicit model manager, request-local model context, typed capability manager,
and isolated adapters. The UI, status text, product metadata, and built-in
personas remain model-family neutral.

The extension never downloads or creates a second language model. It reuses
Forge's active model patcher, encoder, tokenizer, and vision module. The compact
runtime bar can explicitly ask Forge to load the model, text encoder, and VAE
already selected in Forge's normal controls; it does not change those
selections.

### Anima runtime evidence and remaining validation

`tests/forge_anima_qwen06_full_context_smoke.py` is the historical manual,
isolated feasibility diagnostic for the Forge-owned Anima Qwen3-0.6B text stack.
With the real local Anima checkpoint, Qwen3-0.6B encoder, and VAE, bounded greedy
full-context runs completed with both 16 and 32 generated tokens and unloaded
the model cleanly. The registered runtime no longer uses that quadratic loop;
it uses the request-local KV-cache path documented in the current handoff above.
The diagnostic validates exact Forge object identity, the 1024-wide/151936-vocab
encoder contract, its final norm, and the tied input/output embedding without
copying or retaining a second language model. It is intentionally capped at 32
tokens and is not part of the automatic test suite.

That earlier result was the runtime-mechanics feasibility proof. The registered
adapter harness now also passed a fresh 32-token real-stack run with 305 input
tokens and a clean 110-character visible prompt after its explicit role-boundary
cleanup. Normal Forge/Gradio workflow, broader prompt-quality, and image-generation
validation remain open. The separate Qwen3.5-4B conditioning conversion still is
not a drop-in speech backend by itself; the experimental Forgy provider works
only through the explicitly selected, provenance-checked official causal runtime
and tail described in the current 2026-08-28 handoff above.

## Changes in v0.5.2-beta.2

- Restored KREA2, Z-Image, and FLUX.2 Klein stack detection for Forge's current
  text-processing wrappers while retaining compatibility with the earlier
  direct engine objects.
- Kept component validation tied to the underlying Forge-owned encoder and
  tokenizer identities so mismatched wrappers still fail closed.
- Updated KREA2 Image-to-Prompt for Forge's current `SDClipModel`
  `process_tokens` path and image-token contract while retaining the legacy
  `process_embeds` path.
- Included the post-beta public Forgy persona wording cleanup from `develop`.

## Changes in v0.5.2-beta.1

- Promoted the complete `v0.5.2-alpha.1` multi-adapter implementation to the
  public `main` release line.
- Added prominent README guidance that Forgy's prompt quality and instruction
  following depend strongly on the active text encoder.
- Documented support for alternative compatible text-encoder weights when they
  work with the selected model and retain the architecture, tokenizer,
  dimensions, and output-head contract required by its Forgy adapter.

## Changes in v0.5.2-alpha.1

- Added an isolated `Flux2KleinAdapter` for FLUX.2 Klein 4B and Base 4B. It
  validates Forge's exact `Flux2` and `KleinTextProcessingEngine` classes,
  Qwen3-4B identity and dimensions, tokenizer identity, and `ModelPatcher`.
- Enabled text-only Forgy Chat, Idea-to-Prompt, and Refine through the shared
  request-local tied-embedding runtime. Image-to-Prompt and Forgy image
  attachment stay disabled; Klein reference images remain native Forge inputs.
- Preserved the released Z-Image adapter unchanged and added separate
  Klein-owned natural-language workflow contracts.
- Klein 9B fails closed because Forge discards its separate untied Qwen3-8B LM
  head. Forgy does not load an extra output head or modify Forge core.
- Added dependency-free adapter tests and
  `tests/forge_flux2_klein_text_smoke.py`. On 2026-08-13, a real local Klein 4B
  FP8 checkpoint, FLUX.2 VAE, and Qwen3-4B text encoder completed the bounded
  deterministic smoke: 305 input tokens, 64 output tokens, 306 visible
  characters, `max_tokens` finish, and clean weight unloading. Normal Klein
  image generation and broader Forgy prompt-quality testing remain pending.
- Added click-to-enlarge behavior for uploaded and explicitly grabbed Forgy
  images, with local click/keyboard close behavior and no extra image storage.
- Added an undoable prompt-only `×` control beneath the copy control and a `Grab
  last generated prompt` action that restores the positive prompt recorded with
  the latest gallery generation.
- Added transient button-state timers: two seconds for image/prompt grabs and
  one second for Undo/Clear feedback.
- Promoted the tested combined Forgy persona to the built-in `Default`, retained
  the former built-in as `Default Legacy`, and limited migration to the exact
  unchanged legacy hash.

## Changes implemented on 2026-08-10

### Modular model and capability foundation

- Added the dependency-free `forgy` package with adapter contracts, typed model
  capabilities, effective workflow availability, and deterministic model
  resolution.
- Added the first explicit `Krea2Adapter`, including exact diffusion-engine and
  Qwen3-VL validation, request-local Forge components, prompt-template behavior,
  visible-output filtering, repetition protection, and runtime dispatch.
- Kept adapter objects free of the active Forge model, encoder, tokenizer,
  `ModelPatcher`, CUDA tensors, and KV cache. Each resolution produces a fresh
  request-local `ModelContext`.
- Routed shared text and image workflow runners through the selected adapter and
  added authoritative capability guards before every operation.
- Added capability-driven Gradio updates after Forge loads the current user
  selection. Unsupported workflows and image inputs are disabled while backend
  validation remains authoritative.
- Preserved the script filename, callback names, tab and element IDs, CSS
  classes, persona stores, and existing user settings.
- Added `ARCHITECTURE.md`, dependency-free adapter tests, and a full Forge import
  bootstrap check.
- Added the first Z-Image Base/Turbo adapter. It enables Forgy Chat,
  Idea-to-Prompt, and Refine while disabling standalone and attached-image
  analysis because the active Z-Image Qwen3 stack has no vision module.
- Generalized the tied-embedding Qwen text loop so KREA2 and Z-Image can share
  runtime mechanics while retaining separate detection and prompt protocols.
- Live-validated Z-Image Base with the real local Qwen3-4B and AE modules: a
  bounded 64-token request returned 324 visible characters, generated no image,
  saved no Forge settings, and unloaded the weights afterward.
- User burn-testing confirmed KREA2 rejection of mismatched SDXL/preset stacks,
  correct KREA2 loading, Chat, Refine, text-to-image transfer/generation, and
  Idea-to-Image. Z-Image model switching, adapter/encoder recognition, Forgy
  text generation, Idea-to-Prompt, and normal Forge image generation also ran.
- Fixed the Z-Image chat result from the burn test: harmless marker variations
  such as `FORGY reply:` are now parsed, so a valid `UPDATED_PROMPT:` is copied
  into Forgy's working prompt instead of leaving the raw protocol in chat.
- Added Z-Image-specific Idea, Refine, and Chat request contracts. They remind
  Qwen that source text is an image concept rather than prose to correct, require
  preservation of explicit subjects and relationships, and reinforce the exact
  output shape without changing KREA2's validated prompt renderer.
- Replaced the silent disabled image controls with an explicit text-only adapter
  notice in Forgy Chat and Image-to-Prompt.

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
- Renamed the product to **Forgy — Forge Neo Prompt Studio**, retained the
  `0.5.0-beta.1` version marker, and neutralized general KREA2-specific UI,
  persona, metadata, issue-form, security, and README wording.
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
- Backend-specific architecture checks, prompt contracts, and multimodal code
  may be explicit and honest. KREA2 and Z-Image keep their validation and
  request rendering in separate adapter modules; low-level Qwen runtime
  mechanics may be shared only after both contracts are validated.
- Add another family as an isolated adapter. Do not scatter
  model-family `if` statements through shared persona, sampling, and UI code.
- Unsupported or unknown stacks must fail clearly rather than attempting a
  speculative fallback.

### The user remains in control

- A persona is the full system prompt. Do not silently append style personas,
  model-content policies, NSFW, safety, or brand-specific instructions. A
  narrow workflow contract may be added to the user request only when it is
  adapter-owned, documented, and tested.
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
script filename, and existing installation directories unchanged until an
explicit migration plan accounts for Forge `ui-config.json`, callback names,
CSS selectors, and user installations. The public GitHub repository URL is the
new `forgy-forge-neo-prompt-studio` location.

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
- Z-Image is currently text-only and has had one bounded Base live smoke. Turbo
  shares the validated static runtime contract but has not received a separate
  Forgy text-generation smoke in this change. Tag-oriented model-family
  adapters are not implemented.
- FLUX.2 Klein 4B is currently text-only in Forgy and has automated contract
  coverage plus one bounded real-model text smoke. Broader prompt-quality and
  workflow testing remain; Klein 9B is deliberately unsupported while Forge
  omits its untied LM head.
- Forgy's history is not persistent. `Clear conversation` intentionally keeps
  the current working prompt and its undo versions; page reload or Forge restart
  discards the session history.
- Image-to-Prompt analyzes an image to produce text. It does not implement
  img2img conditioning, reference-image styling, identity editing, or automatic
  parameter changes.
- The Python/script, callback, Gradio, and CSS identifiers still use the older
  KREA-oriented naming. This is deliberate for compatibility but should
  eventually be reviewed as a separately planned migration.
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
  layouts, image lightbox behavior with a real attachment, and narrow-window
  behavior need a final browser smoke test after the user's normal Forge
  instance is restarted.
- Image-to-Prompt and Forgy-with-image need further live tests with portrait,
  landscape, alpha-channel, EXIF-rotated, and very large images.
- Cancellation has unit/structural coverage, but cancellation during vision
  prefill, late decoding, and native Forge image sampling should be manually
  distinguished. The extension Stop button cancels encoder generation; Forge's
  own interrupt control remains responsible for image sampling.
- GitHub Actions must be checked on the published release commit. Local success
  does not prove that the public Linux/Python 3.11 validation run completed.

Today's isolated Forge UI check confirmed the prompt-clear control's placement,
prompt clear/Undo behavior, one-second transient Undo/Clear resets, the new
Forgy default/legacy persona state, successful tab construction, and installation
of the image-lightbox handler. The isolated UI had no generated attachment, so
the actual enlarged-image interaction remains a manual smoke item. That instance
was stopped; the user's normal Forge instance was not stopped or modified and
must be restarted to load the latest code.

## Automated validation status

Before this handover, the following checks passed for the current development
implementation:

- 81 dependency-free `unittest` tests;
- Ruff lint check;
- Ruff format check;
- `git diff --check` (apart from Git's existing LF-to-CRLF warning for
  `style.css`);
- parsing of `metadata.ini` and the GitHub YAML files;
- full Forge import/bootstrap validation for all four registered adapter IDs;
- an isolated real Z-Image Base/Qwen3-4B text-generation smoke;
- an isolated real FLUX.2 Klein 4B/Qwen3-4B text-generation smoke with 64
  visible output tokens and clean weight unloading;
- an isolated real-DOM Forgy UI smoke covering prompt clear/Undo, transient
  button resets, persona state, tab construction, and lightbox-handler setup.
  Capability switching and the lightbox with a real image still need a fresh
  browser smoke after restarting Forge.

Re-run the automated checks after any further change and immediately before a
commit or release:

```powershell
cd <extension repository>
& ..\..\venv\Scripts\python.exe -m unittest discover -s tests -v
& ..\..\venv\Scripts\ruff.exe check scripts\forge_krea_prompt_assistant.py forgy tests
& ..\..\venv\Scripts\ruff.exe format --check scripts\forge_krea_prompt_assistant.py forgy tests
git diff --check
```

## Recommended next steps

1. Confirm that tag/release `v0.5.2-beta.2`, the `main` branch commit, and the
   GitHub Actions validation all refer to the same reviewed source state.
2. Restart the user's Forge test instance and run one focused KREA2/Qwen3-VL
   smoke pass through all four workflows.
3. Specifically verify each `+ generate` button remains locked only until the
   new image reaches the correct gallery, then re-enables with a success check.
4. Re-test the image lightbox, both Grab actions, prompt-only clear, transient
   label resets, Forgy autoscroll, Stop, Undo, conversation clear, persona
   create/edit/delete, `Ghost`, and `Default Legacy` preservation.
5. Inspect `git status`, the complete diff, and ignored local persona data.
   Ensure no `personas.json`, `image_personas.json`,
   `refinement_personas.json`, `agent_personas.json`, generated image, log, or
   personal path is staged.
6. Run the four automated commands above before any follow-up release.
7. Triage Beta 0.5.2 feedback without mixing unrelated fixes into one release.
8. Restart an isolated Forge UI, switch between KREA2, Z-Image, Klein 4B, and
   stock Anima, and verify
   capability-driven buttons, image inputs, status text, and Default persona
   behavior. Then run a separate bounded Turbo text smoke.
9. Record a normal Klein image-generation smoke and broaden Forgy prompt-quality
   testing without changing the released Z-Image adapter.

## Relevant files and functions

### Runtime implementation

- `forgy/capabilities.py`: typed static capabilities and effective workflow
  availability.
- `forgy/model_manager.py`: explicit registry and request-local, fail-closed
  model resolution.
- `forgy/adapters/krea2.py`: KREA2/Qwen3-VL component validation, prompt
  protocol, visible-output safeguards, and runtime dispatch.
- `forgy/adapters/flux2_klein.py`: FLUX.2 Klein 4B/Base 4B Qwen3-4B
  validation, text-workflow capabilities, prompt protocol, and runtime dispatch.
- `forgy/adapters/zimage.py`: Z-Image Base/Turbo Qwen3-4B validation,
  text-workflow capabilities, prompt protocol, and runtime dispatch.
- `forgy/adapters/anima.py`: stock Anima/Qwen3-0.6B validation, text-workflow
  capabilities, prompt protocol, and runtime dispatch.
- `forgy/adapters/qwen.py`: shared Qwen decode, thinking removal, and repetition
  protection.
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
  - Token generation: `_generate_with_active_text_adapter()`, compatibility
    `_generate_with_active_krea()`,
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

- `tests/test_public_metadata.py`: 52 unit/static contract tests for branding,
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
- `CHANGELOG.md`: chronological prerelease changes and comparison links.
- `ROADMAP.md`: implemented refinement and planned history, diagnostics,
  Z-Image adapter, and explicitly guided generation loop.
- `metadata.ini`, `LICENSE`, `SECURITY.md`, `.github/`: public identity,
  licensing, disclosure policy, issue forms, and read-only CI validation.
- `.gitignore`: excludes local personas, temporary files, outputs, environments,
  and editor/runtime artifacts.
