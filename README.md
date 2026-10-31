# Forgy — Forge Neo Prompt Studio

**Alpha 0.6.0-alpha.3 (development prerelease)**

[![Validate](https://github.com/vibecodingtoolmaker/forgy-forge-neo-prompt-studio/actions/workflows/validate.yml/badge.svg?branch=develop)](https://github.com/vibecodingtoolmaker/forgy-forge-neo-prompt-studio/actions/workflows/validate.yml)

A Forge Neo extension that turns a short image idea or an uploaded reference
image into a polished image-generation prompt and can iteratively refine an
existing prompt. Forgy, the included local creative copilot, can discuss an idea
or uploaded image while maintaining a visible working prompt. The extension
reuses the text and vision encoder already owned by Forge. It does not download,
instantiate, or retain a second language model. MOSS generation loads only
the selected encoder checkpoint's untied output head for the duration of one
request. The optional experimental
Anima/Qwen3.5 speech path reuses one model already cached by the separately
installed Qwen3.5 Anima adapter.

The highlight of this alpha is the experimental MOSS-Image/Qwen3.5-9B adapter.
It brings the active MOSS text and vision encoder to Forgy Chat,
Idea-to-Prompt, Image-to-Prompt, and Prompt Refinement, including editable,
prompt-aware negative suggestions. The release also includes text-only Z-Image
Base/Turbo, FLUX.2 Klein 4B, and stock Anima/Qwen3-0.6B adapters. It has not yet
been tested across a broad range of GPUs and Forge Neo configurations.

Forgy supports both Forge's earlier top-level KV-cache return and the current
Forge Neo 2.29.2 per-layer cache path. No Forge core changes or additional
durable model loads are required.

Alpha.3 is a focused hotfix for KREA2 Idea-to-Prompt and Image-to-Prompt output
that could appear empty after the MOSS thinking-diagnostics update. The shared
cleanup now preserves a valid visible prompt while continuing to hide private
reasoning for every supported Qwen-family adapter.

## Welcome

Hello and welcome! I hope Forgy makes it easier to turn rough ideas into useful
image-generation prompts while keeping the prompt-writing behavior entirely
under user control through personas. Feedback, careful testing, and
constructive contributions are very welcome.

## AI-assisted development transparency

Forgy — Forge Neo Prompt Studio is created and maintained by
**vibecodingtoolmaker** with substantial AI-assisted development support from
**OpenAI Codex**. Code, documentation, reviews, and test ideas have been
developed and refined through this collaborative workflow — also known, with
some affection, as *vibecoding*.

The project direction, feature decisions, hands-on Forge testing, final review,
and release responsibility remain with the human maintainer. Codex is a
development tool, not a runtime dependency: the installed extension does not
contact OpenAI or send prompts, personas, images, model information, or other
user data to Codex. This project is not affiliated with or endorsed by OpenAI.

## Requirements

- Forge Neo with a supported model, its matching text encoder, and compatible
  VAE files available in Forge's configured model directories;
- a CUDA-capable GPU supported by the Forge installation.

Development and current testing target Forge Neo 2.29.2.

The KREA2 adapter supports its Qwen3-VL 4B text/vision encoder and a compatible
VAE across all four workflows. The initial Z-Image Base/Turbo adapter reuses
Forge's active Qwen3-4B encoder for Forgy Chat, Idea-to-Prompt, and Refine;
Z-Image has no vision input, so Image-to-Prompt and Forgy image attachment are
disabled. The FLUX.2 Klein adapter provides the same three text workflows for
Klein 4B and Base 4B through Forge's tied Qwen3-4B encoder. Klein reference
images remain Forge generation inputs rather than Forgy vision inputs. Klein
9B is rejected because Forge does not retain the separate untied Qwen3-8B LM
head needed for text generation. The development branch also supports Anima
through its original Forge-owned Qwen3-0.6B encoder for the three text
workflows. That small encoder uses a request-local FP32 KV cache, a 256-token
runtime ceiling, and a compact Forgy prompt/refinement mode. Obvious template
copies and malformed base-model output are rejected; Forgy preserves the current
prompt or falls back transparently to the cleaned new request. An explicit,
off-by-default Qwen3.5 provider can instead reuse one unambiguous
`sd-forge-qwen-35-encoder` cache together with the exact pinned official causal
tail; it fails closed when the cache or verified local tail assets are missing.
The experimental MOSS-Image adapter provides all four workflows from Forge's
active Qwen3.5-9B text/vision backbone. MOSS accepts both cohesive captions and
ordered tag sequences; the selected persona controls that prompt dialect, and
refinement preserves it unless the user explicitly requests a change. Because
the official MOSS configuration uses
an untied output head, Forgy accepts exactly one dense BF16, FP16, or FP32 MOSS
encoder among Forge's selected additional modules, loads only its `lm_head` for
the request, and requires Forge to place the complete backbone on one device.
Quantized LM heads, ambiguous module selections, and partial CPU/GPU placement
fail closed. Image-to-Prompt uses Forge's active Qwen3.5 vision encoder and exact
three-axis MRoPE metadata; no image model is sideloaded. RTX 5090 profiling then
exposed an important Forge storage detail: the token embedding can be FP16 while
the complete transformer is BF16. Following the embedding dtype caused
`manual_cast` to convert backbone weights repeatedly. Forgy now validates and
follows the dense transformer projection dtype, converts embeddings once, and
loads the request-local output head in that dtype. In an identical
369-input/16-output A/B smoke, generation fell from 43.52 to 8.73 seconds,
prefill from 11.64 to 1.26 seconds, and decode from 27.74 to 1.48 seconds. A
64-token prompt completed in 8.94 seconds with a 56 ms median decode; a
synthetic 871-token vision request completed in 6.49 seconds with a 1.14-second
prefill. Forge model reload time is separate from those generation timings.

Each workflow also exposes a separate, editable `Suggested negative prompt
(MOSS)` field. The official SFT model card uses CFG 3 with an empty negative
prompt by default, while the official user guide documents `nsfw, worst quality,
bad anatomy, watermark, jpeg artifact` as an effective example. Forgy therefore
uses only `worst quality, watermark, jpeg artifact` as the non-contradictory
core. It adds anatomy defects only for detected people or animals, hand/foot
terms only when relevant human anatomy is actually requested, typography defects
only for requested text, and style exclusions only for an unambiguous photo or
illustration dialect. `nsfw` is added only when the request explicitly asks for
safe-for-work or family-friendly content. This deterministic post-processing
does not spend language-model tokens and never inserts negative terms into the
positive prompt. The adjacent checkbox controls whether normal Forge transfer
actions replace or merge the visible suggestion into Forge's negative-prompt
field; disabling it preserves the existing Forge negative prompt. See the
[official SFT model card](https://huggingface.co/OpenMOSS-Team/MOSS-Image-SFT)
and [MOSS user guide](https://github.com/OpenMOSS/MOSS-Image/blob/main/USERGUIDE.md).

Model detection, typed capabilities,
request-local component resolution, and family prompt behavior live behind
explicit adapter contracts. Unsupported stacks fail closed, and model-dependent
workflow and image controls refresh after Forge loads the current selection. See
[`ARCHITECTURE.md`](ARCHITECTURE.md) for the extension contract.

### Text encoder choice and output quality

Forgy's replies and prompts are generated by Forge's active text encoder, so
their quality depends strongly on the encoder weights selected. Instruction
following, concept preservation, language fluency, detail, consistency, and
adherence to Forgy's response markers can all improve or degrade when the text
encoder changes.

The matching encoder recommended for each supported model is the tested
baseline, not a requirement to use one exact weight file.
Alternative compatible text encoders may be selected in Forge as long as they
work with the active model and still expose the architecture, tokenizer,
dimensions, and — where
required — output-head contract expected by the corresponding Forgy adapter.
Forge or Forgy will reject a structurally incompatible stack. A compatible
encoder that loads successfully can still produce noticeably different prompt
quality, so compare results and keep the encoder that works best for the model
and workflow.

## Installation

Clone the current main release into the Forge Neo `extensions` directory:

```powershell
cd sd-webui-forge-neo/extensions
git clone https://github.com/vibecodingtoolmaker/forgy-forge-neo-prompt-studio.git
```

```text
sd-webui-forge-neo/
└── extensions/
    └── forgy-forge-neo-prompt-studio/
```

Restart Forge Neo. The `Forgy — Forge Neo Prompt Studio` tab will appear in the
main UI.
No additional Python packages are required.

## Usage

1. Select a supported model, its matching text encoder, and VAE in Forge's
   normal model controls.
2. Open `Forgy — Forge Neo Prompt Studio`. If the selection has not been loaded yet,
   click `Load current Forge selection` in the compact runtime bar at the top.
3. Start in `Forgy Chat`, or open `Idea to prompt`, `Image to prompt`, or
   `Refine prompt` for a focused workflow.
4. Select the active persona at the top, provide the requested input and
   instruction, and choose the desired maximum output length. Persona editing
   is available separately below the result and action controls.
5. Click the corresponding generate button.
6. Copy the result or send it directly to the positive `txt2img` or `img2img`
   prompt field.

The system status below the generated prompt shows the active persona, the
actual normalized seed, input/output token counts, available output context,
finish reason, and elapsed time. While the encoder is working, the status area
shows a prominent hourglass message naming the active text or vision encoder,
and the active generation button is temporarily disabled to prevent
duplicate requests. A `Stop` button beside each generation button requests a
controlled cancellation. The encoder stops after the current model step; if
visible tokens already exist, the partial prompt is returned with the finish
reason `cancelled`.

Action buttons provide explicit feedback: their label changes as soon as the
click is accepted. Successful `txt2img` and `img2img` transfers show a checkmark
directly on the clicked button. Transfer-button labels reset when a new prompt
generation starts and when a new result arrives.

Each prompt-producing workflow also offers `replace + generate` and `append +
generate` for txt2img and img2img. The extension first completes the positive
prompt transfer and only then clicks Forge's existing Generate button. Rendering
therefore starts while the user returns to the generation tab, using the model,
VAE, sampler, seed, dimensions, extensions, and queue behavior already selected
in Forge. An empty assistant result never starts a generation. The clicked
transfer-and-generate button remains disabled while Forge is rendering. It is
released with a success check only after Forge's native activity controls return
to idle and a new gallery image appears; start failures, timeouts, and runs that
produce no gallery image restore the normal button instead.

In the full-width Idea-to-Prompt and Image-to-Prompt tabs, each Forge target
uses one compact four-button row ordered as `replace`, `replace + generate`,
`append`, and `append + generate`. Forgy Chat uses the same full-width action
rows below its chat and prompt workspace. Prompt
Refinement now uses the same four-button target rows and places its two compact
source-result buttons beside the prompt being refined.
Idea-to-Prompt and Image-to-Prompt keep the active persona selector near the
input while placing the independent `Persona to edit` selector and management
fields below the result actions. In Image-to-Prompt, the optional instruction
shares one compact row with the reference image.
Prompt Refinement follows the same persona pattern: its active persona stays at
the top and its independent management section sits below the result actions.
Forgy Chat is the first and default workflow tab and follows the same pattern,
with its independent persona management below the Forge action rows.
All four workflows place the active VRAM-profile summary at the very bottom,
after their persona-management and Sampling sections, so this informational
status does not interrupt the working flow.

For local testing, the collapsed `Model output diagnostics (testing)` section
offers an `Anima text provider` selector and an `Enable and show Qwen model
thinking` checkbox. Both are request-local and are not saved to Forge's UI
configuration. The Anima provider defaults to `Original Qwen3 0.6B`. Its
Forgy Chat path deliberately uses compact prompt/refinement semantics rather
than asking the base encoder to reproduce the full two-section agent protocol;
the status identifies this mode. The selected Qwen3.5 provider and stronger
model families retain their normal agent protocol. The
experimental `Qwen3.5 4B adapter cache` choice requires exactly one model already
cached by `sd-forge-qwen-35-encoder`; Forgy never calls that extension's model
loader and reports a clear failure instead of falling back silently. The
required UI order is: enable `Qwen3.5 Text Encoder (Anima)` in txt2img or img2img,
select `qwen35_4b.safetensors`, complete one normal Forge generation, and only
then select the Qwen3.5 provider in Forgy. Forge registers that extension's
script module with a trailing `.py`; Forgy accepts both that live name and the
normal direct-import name. The
thinking option is off by default,
is not saved to Forge's UI configuration, and applies only to subsequent
requests while selected. Enabling it switches the supported Qwen adapter to its
thinking prefix and shows the complete decoded generated text in a separate raw
output field for the active workflow. The normal result stays separately
filtered and usable for prompt transfer. Thinking consumes the same output-token
budget, can increase latency substantially, and may reveal prompt or persona
content; review the raw field before sharing it in a bug report.

Idea-to-Prompt and Image-to-Prompt place `Send to prompt refinement` and `Send
to Forgy` in a narrow column beside the generated prompt. Prompt Refinement
uses that same side-column pattern for both source-result actions and `Send to
Forgy`, keeping cross-workflow navigation close to the prompt it acts on.

## Forge model loader

Forge remains the single source of truth for the selected checkpoint, text
encoder, and VAE. The compact runtime bar at the top reports whether a complete
supported stack is active and offers one action: `Load current Forge selection`.

The button refreshes Forge's current loading parameters and invokes Forge's own
queued model reload. It does not change or save selections, add duplicate model
lists, scan model directories, download files, or instantiate weights itself.
Forge remains responsible for architecture detection, component assembly,
offloading, and memory management. The extension accepts the result only when
the selected adapter can resolve the active model, encoder, tokenizer, and
ModelPatcher. The current development implementation includes KREA2/Qwen3-VL,
text-only Z-Image Base/Turbo, FLUX.2 Klein 4B, stock Anima/Qwen3-0.6B, and an
experimental MOSS-Image/Qwen3.5-9B text/vision adapter, plus Anima's explicitly
selected experimental Qwen3.5 cached provider.

Loading stays explicit so opening the assistant cannot unexpectedly allocate
VRAM or unload another active model.

## Personas

The selected persona text is the complete system prompt. In `Idea to prompt`,
the idea and its optional additional instruction form the user message. In
`Image to prompt`, the uploaded image and the optional instruction form the user
request. When the image instruction is empty, the UI uses the visible default
request shown in its placeholder. In `Refine prompt`, the current prompt and the
  required refinement instruction form the user message. The extension adds no
  hidden style persona or model-content policy. A family adapter may append a
  documented, task-only workflow contract to the user request when its encoder
  needs clearer output boundaries; the Z-Image and FLUX.2 Klein contracts are
  described in `ARCHITECTURE.md`.

The focused Idea-to-Prompt, Image-to-Prompt, and Refine stores keep their
workflow-specific `Default` prompts. Forgy Chat's `Default` combines idea
expansion, image analysis, and prompt refinement in one interactive persona. A
fresh installation receives only these four workflow-specific defaults; no
additional testing or legacy persona is seeded. An unchanged old Default may be
migrated by exact known hash, while edited defaults and every user-created local
persona are preserved. `Default` can be edited but not deleted. Users can create,
edit, and delete additional personas in the UI.

Idea personas are stored locally in `personas.json`; image personas use
`image_personas.json`; refinement personas use `refinement_personas.json`; Forgy
personas use `agent_personas.json`. All four use this format:

```json
{
  "version": 1,
  "personas": {
    "Default": "Complete system prompt ...",
    "My Persona": "Another complete system prompt ..."
  }
}
```

The files are written atomically. Personal persona data is excluded from Git by
default. `personas.example.json`, `image_personas.example.json`,
`refinement_personas.example.json`, and `agent_personas.example.json` contain the
four distributable defaults. When a local file does not exist, its built-in
Default persona and any built-in backup are used automatically.

## Image to prompt

The image workflow accepts one uploaded image. EXIF orientation is applied,
alpha is composited by converting to RGB, and the image is normalized locally
to approximately 768 x 768 pixels of visual area while preserving its aspect
ratio. This normally produces about 576 Qwen3-VL visual tokens.

Those visual tokens replace the single image placeholder in the chat template.
The extension also applies Qwen3-VL multidimensional rotary positions and the
three DeepStack vision features during the language-model prefill. The uploaded
image is not passed to the active VAE and is not used as img2img conditioning;
it is analyzed only to produce text.

## Refine prompt

The refinement workflow accepts a pasted prompt, can load the most recent result
from within the refinement tab, or can receive it through `Send to prompt
refinement` in either generation tab. A required, visible refinement instruction
tells the encoder exactly what to revise. The selected refinement persona
remains the complete system prompt.

A successful revision replaces the prompt in the refinement textbox so another
focused instruction can be applied immediately. Validation errors, cancellation
before any output token exists, CUDA out-of-memory errors, and unexpected
runtime failures preserve the previous prompt. Refinement uses the text encoder
only; it does not retain or reprocess an uploaded image from the image workflow.

## Forgy Chat creative copilot

Forgy Chat is the extension's first and default workflow. Forgy is a local,
interactive prompt-building assistant using Forge's active text or vision
encoder. A turn can contain text alone or text plus one uploaded image.
`Forgy's prompt output and working input` is always visible and editable beside
the chat. The optional image attachment is permanently visible directly below
that field and above the full-width Forge action rows.
`Grab last generated image` attaches the final image from whichever Forge
gallery — txt2img or img2img — most recently received a result. If that source
is unavailable, Forgy falls back to the other non-empty gallery. The image is
copied into Forgy's normal local attachment field only after the explicit
button click. Before Gradio post-processes that copy, the extension ensures
Forge's configured temporary-image directory exists; it does not introduce a
separate storage location. Clicking an uploaded or grabbed Forgy image opens a
large local lightbox; clicking the enlarged image or backdrop closes it again.

`Grab last generated prompt` records the positive prompt when a txt2img or
img2img gallery receives its latest image and explicitly loads that exact prompt
into Forgy's working field. If no recorded generation is available, it falls
back only to a non-empty prompt belonging to an available gallery. The small
`×` below the working field's copy control clears only that prompt and adds the
previous value to Undo. Successful grab labels reset after two seconds;
successful Undo and conversation-clear labels reset after one second.

The built-in Forgy persona asks the model to return a concise conversational
reply and either a complete updated prompt or the visible `[UNCHANGED]` marker.
If that response protocol is missing or malformed, the raw assistant reply is
shown but the working prompt is preserved. Users can edit the complete Forgy
system prompt in the persona controls; keeping its visible output markers is
required for automatic prompt updates.

Forgy is deliberately bounded for the currently supported diffusion-owned Qwen
encoders, which are not replacements for a dedicated full-size chat model.
Forgy uses each validated family template, allows up to 4096 output tokens for
long iterative prompt sessions, and defaults to a 1.15 repetition penalty.
Input and output still share the selected VRAM-aware context window, so the
effective output limit can be lower after the persona, conversation, current
prompt, and optional image tokens are counted. If an exact token sequence starts
looping, generation stops early with the finish reason
`repetition_stop` and the duplicate tail is discarded. Any `<think>` or
`<analysis>` content is removed from the normal workflow result. It is shown
only in the explicitly enabled, request-local testing diagnostic described
above. These guardrails keep a weak turn bounded; they cannot make the 4B
encoder reason like a larger conversational model.

Prompt changes — including the small prompt-clear action and an explicit prompt
grab — are kept in a bounded, session-only version list so `Undo prompt change`
can restore the previous prompt. `Clear conversation` removes chat
messages but deliberately preserves the current prompt and its versions. Chat
messages, prompt versions, and uploaded images are not written to extension
storage. The Forgy Chat UI states this location explicitly: up to 50 previous
working prompts exist only in the current Forge UI session and disappear on a
page reload or Forge restart. An attached image is included with each turn until
the user clears or replaces it. Whenever the conversation value changes, the Forgy viewport
scrolls to the newest message after both the initial UI update and delayed
Markdown rendering.

Idea-to-Prompt, Image-to-Prompt, and Prompt Refinement each provide `Send to
Forgy`. Forgy's current prompt can then be sent to the normal positive `txt2img`
or `img2img` prompt. Forgy never starts image generation autonomously; rendering
begins only when the user explicitly clicks a `replace + generate` or `append +
generate` action.

## VRAM and shared context

Forge's detected physical VRAM selects a conservative context profile
automatically. A smaller manual fallback is available under `Sampling`.

| VRAM profile | Shared context | Full-context KV cache* |
|---:|---:|---:|
| 8 GB | 4096 tokens | about 576 MiB |
| 12 GB | 5120 tokens | about 720 MiB |
| 16 GB | 6144 tokens | about 864 MiB |
| 24 GB | 8192 tokens | about 1152 MiB |
| 32 GB | 12288 tokens | about 1728 MiB |

\* For the currently supported 2560-wide Qwen3 4B adapters with an FP16/BF16 KV
cache. Runtime allocation uses the active encoder configuration and dtype.

Input and output share one window:

```text
available output = shared context - actual input tokens
```

The input includes the persona, user request, active chat template, and — in the
image workflow — the actual visual tokens. A refinement request also includes
the complete current prompt. If the requested output limit is
too large, the extension reduces it and reports the effective value. After
Forge places or offloads encoder weights and completes image encoding, the
extension checks currently free VRAM again before allocating the cache.
The original Anima 0.6B provider additionally caps generated output at 256
tokens and uses an FP32 cache because Forge's manual-cast attention computes its
queries, keys, and values in FP32.

## Seed and reproducibility

When sampling is enabled, generation uses the seed shown beneath the result.
The displayed value is the normalized seed actually passed to the PyTorch
generator. With sampling disabled, the status explicitly marks the seed as
inactive.

Matching the idea, persona, sampling settings, seed, model, and software stack
should reproduce the same prompt, subject to the usual limits of GPU and
framework determinism.

## Safety and privacy properties

- no model download and no `from_pretrained()` call;
- no second language model or weight copy;
- optional stack loading delegates to Forge's existing loader and queue;
- uses only the active adapter's `clip.patcher`, encoder, tokenizer, and vision
  module, except for the explicitly selected Anima/Qwen3.5 path, which reuses
  exactly one existing Qwen3.5 adapter cache;
- runs inside the Forge queue;
- asks Forge's memory manager to reserve the calculated KV-cache space;
- keeps no global encoder, tokenizer, or CUDA tensor reference;
- discards temporary KV/recurrent caches and Qwen3.5 tail tensors after each
  request;
- stores personas only in the local extension directory;
- keeps Forgy conversation and prompt versions in the current UI session only;
- does not write uploaded images to extension storage; Forge/Gradio may still
  use its normal temporary-upload handling;
- modifies no Forge core file;
- rejects unsupported model families and unknown encoder classes.

## Current limitations

- this development branch contains KREA2 plus text-only Z-Image Base/Turbo,
  FLUX.2 Klein 4B, and stock Anima/Qwen3-0.6B adapters; Z-Image Turbo still
  needs a separate live text smoke, while Klein 4B and Anima each have bounded
  real-model text smokes but need broader prompt-quality and workflow testing;
- the original Anima 0.6B encoder is a small base text encoder, so Forgy Chat is
  intentionally prompt-focused: new-prompt and current-prompt refinement are
  supported, while richer conversational behavior belongs to the Qwen3.5 option;
- Anima's experimental Qwen3.5 provider has bounded isolated and integrated
  RTX 5090 text smokes, but still needs visible Gradio, longer-output,
  lower-VRAM/offload, and broad prompt-quality testing;
- one uploaded image per image-to-prompt request;
- image upload generates text only and does not configure img2img or
  reference-image conditioning;
- Z-Image has no Image-to-Prompt or Forgy image analysis; its first adapter is
  text-only and still needs broader live prompt-quality testing;
- FLUX.2 Klein 4B is text-only inside Forgy; Klein reference images remain
  native Forge generation inputs, normal Klein image generation still needs a
  recorded smoke, and Klein 9B is not supported without its discarded untied
  LM head;
- MOSS text, 64-token, synthetic-image, visible Idea-to-Prompt, and visible
  Image-to-Prompt smokes pass on an RTX 5090. The maintainer also confirmed all
  locally configured Idea personas and strong prose-prompt quality. Broader
  sampling, cancellation, diverse real-image inputs, and lower-VRAM/offload
  behavior still need live testing; the selected local encoder contains no MTP
  tensors, so speculative decoding is not available for that stack;
- no persistent local generation history; Forgy keeps only bounded session undo versions;
- Forgy never starts a Forge image generation autonomously; the user must click
  an explicit `replace + generate` or `append + generate` action;
- no dedicated Pony, Illustrious, or NoobAI tag-prompt adapters.

Future experiments are tracked in the project [roadmap](ROADMAP.md). The next
areas being considered include model-family default personas, broader Z-Image
and FLUX.2 Klein validation, local prompt history, and privacy-conscious
diagnostic reports.

## Feedback and security

Please use the repository's structured
[issue forms](https://github.com/vibecodingtoolmaker/forgy-forge-neo-prompt-studio/issues/new/choose)
for reproducible bug reports and focused feature requests. Remove personal
prompts, personas, images, local paths, access tokens, and unrelated log data
before submitting diagnostic information.

Do not disclose suspected vulnerabilities in a public issue. Follow the
[security policy](SECURITY.md) and use GitHub's private vulnerability reporting
flow from the repository's Security tab.

## License

AGPL-3.0-only. See `LICENSE`.
