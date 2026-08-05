# Forgy — Forge Neo Prompt Studio

**Beta 0.5.0-beta.1**

[![Validate](https://github.com/vibecodingtoolmaker/forgy-forge-neo-prompt-studio/actions/workflows/validate.yml/badge.svg?branch=main)](https://github.com/vibecodingtoolmaker/forgy-forge-neo-prompt-studio/actions/workflows/validate.yml)

A Forge Neo extension that turns a short image idea or an uploaded reference
image into a polished image-generation prompt and can iteratively refine an
existing prompt. Forgy, the included local creative copilot, can discuss an idea
or uploaded image while maintaining a visible working prompt. The extension
reuses the text and vision encoder already owned by Forge. It does not download
or load a second language model.

This first beta establishes the prompt-studio workflow with Forgy Chat,
Idea-to-Prompt, Image-to-Prompt, and Prompt Refinement. It has not yet been
tested across a broad range of GPUs and Forge Neo configurations.

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

Development and initial testing target Forge Neo 2.28.

The first adapter supports the KREA2 model family with its Qwen3-VL 4B text and
vision encoder and a compatible VAE. Future model families are intended to use
separate adapters while sharing the studio's neutral workflows, personas, and
controls.

## Installation

Clone or copy this repository into the Forge Neo `extensions` directory:

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
ModelPatcher. The current beta includes the KREA2/Qwen3-VL adapter.

Loading stays explicit so opening the assistant cannot unexpectedly allocate
VRAM or unload another active model.

## Personas

The selected persona text is the complete system prompt. In `Idea to prompt`,
the idea and its optional additional instruction form the user message. In
`Image to prompt`, the uploaded image and the optional instruction form the user
request. When the image instruction is empty, the UI uses the visible default
request shown in its placeholder. In `Refine prompt`, the current prompt and the
required refinement instruction form the user message. The extension adds no
hidden style or content instruction.

`Default` provides the original natural-language prompt-writing behavior. It
can be edited but not deleted. The built-in `Ghost` persona has an
intentionally empty system prompt for direct, unguided encoder experiments and
cannot be edited into a non-empty persona or deleted. Other personas still
require a non-empty system prompt. Users can create, edit, and delete additional
personas in the UI.

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
Default persona is used automatically.

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
separate storage location.

The built-in Forgy persona asks the model to return a concise conversational
reply and either a complete updated prompt or the visible `[UNCHANGED]` marker.
If that response protocol is missing or malformed, the raw assistant reply is
shown but the working prompt is preserved. Users can edit the complete Forgy
system prompt in the persona controls; keeping its visible output markers is
required for automatic prompt updates.

Forgy is deliberately bounded for the currently supported Qwen3-VL 4B encoder,
which is primarily a diffusion text/vision encoder rather than a full-size chat
model. Forgy appends Qwen's `/no_think` directive, allows up to 4096 output
tokens for long iterative prompt sessions, and defaults to a 1.15 repetition
penalty. Input and output still share the selected VRAM-aware context window, so
the effective output limit can be lower after the persona, conversation, current
prompt, and optional image tokens are counted. If an exact token sequence starts
looping, generation stops early with the finish reason
`repetition_stop` and the duplicate tail is discarded. Any `<think>` or
`<analysis>` block produced despite the directive is removed before visible
output. These guardrails keep a weak turn bounded; they cannot make the 4B
encoder reason like a larger conversational model.

Selecting `Ghost` in Forgy removes those system-level response instructions.
The raw reply remains visible, but Forgy's automatic working-prompt update will
normally preserve the current prompt because the expected response markers may
be absent.

Prompt changes are kept in a bounded, session-only version list so `Undo prompt
change` can restore the previous prompt. `Clear conversation` removes chat
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

\* For the currently supported Qwen3-VL 4B adapter with an FP16/BF16
KV cache. Runtime allocation uses the active encoder configuration and dtype.

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
  module;
- runs inside the Forge queue;
- asks Forge's memory manager to reserve the calculated KV-cache space;
- keeps no global encoder, tokenizer, or CUDA tensor reference;
- discards the temporary KV cache after each request;
- stores personas only in the local extension directory;
- keeps Forgy conversation and prompt versions in the current UI session only;
- does not write uploaded images to extension storage; Forge/Gradio may still
  use its normal temporary-upload handling;
- modifies no Forge core file;
- rejects unsupported model families and unknown encoder classes.

## Current limitations

- one model-family adapter in this beta: KREA2 with Qwen3-VL;
- one uploaded image per image-to-prompt request;
- image upload generates text only and does not configure img2img or
  reference-image conditioning;
- no Z-Image backend yet;
- no persistent local generation history; Forgy keeps only bounded session undo versions;
- Forgy never starts a Forge image generation autonomously; the user must click
  an explicit `replace + generate` or `append + generate` action;
- no dedicated Pony, Illustrious, or NoobAI tag-prompt adapters.

Future experiments are tracked in the project [roadmap](ROADMAP.md). The next
areas being considered are local prompt history, privacy-conscious diagnostic
reports, and a separate Z-Image adapter.

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
