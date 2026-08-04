# Forge KREA2 Prompt Assistant

**Alpha 0.5.0-alpha.1**

[![Validate](https://github.com/vibecodingtoolmaker/forge-krea-prompt-assistant/actions/workflows/validate.yml/badge.svg?branch=main)](https://github.com/vibecodingtoolmaker/forge-krea-prompt-assistant/actions/workflows/validate.yml)

A Forge Neo extension that turns a short image idea or an uploaded reference
image into a polished, natural-language KREA2 prompt. It reuses the Qwen3-VL
text and vision encoder already owned by the active KREA2 diffusion engine. It
does not download or load a second language model.

This alpha establishes the intended KREA2 feature foundation for idea-to-prompt
and image-to-prompt workflows. Image-to-prompt remains new, and the extension
has not yet been tested across a broad range of GPUs and Forge Neo
configurations.

## Welcome

Hello and welcome! I hope this assistant makes it easier to turn rough ideas
into useful KREA2 prompts while keeping the prompt-writing behavior entirely
under user control through personas. Feedback, careful testing, and
constructive contributions are very welcome.

## AI-assisted development transparency

Forge KREA2 Prompt Assistant is created and maintained by
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

- Forge Neo with KREA2 model, Qwen3-VL text encoder, and compatible VAE files
  available in Forge's configured model directories;
- a CUDA-capable GPU supported by the Forge installation.

Development and initial testing target Forge Neo 2.28.

## Installation

Clone or copy this repository into the Forge Neo `extensions` directory:

```powershell
cd sd-webui-forge-neo/extensions
git clone https://github.com/vibecodingtoolmaker/forge-krea-prompt-assistant.git
```

```text
sd-webui-forge-neo/
└── extensions/
    └── forge-krea-prompt-assistant/
```

Restart Forge Neo. The `KREA2 Prompt Assistant` tab will appear in the main UI.
No additional Python packages are required.

## Usage

1. Select the KREA2 model, Qwen3-VL text encoder, and VAE in Forge's normal model
   controls.
2. Open `KREA2 Prompt Assistant`. If the selection has not been loaded yet,
   click `Load current Forge selection` in the compact runtime bar at the top.
3. Open `Idea to prompt` or `Image to prompt`.
4. Select or edit a persona, provide the idea or image, optionally add a
   separate instruction, and choose the desired maximum output length.
5. Click the corresponding generate button.
6. Copy the result or send it directly to the positive `txt2img` or `img2img`
   prompt field.

The system status below the generated prompt shows the active persona, the
actual normalized seed, input/output token counts, available output context,
finish reason, and elapsed time. While the encoder is working, the status area
shows a prominent hourglass message naming the active Qwen3-VL text or vision
encoder, and the active generate button is temporarily disabled to prevent
duplicate requests. A `Stop` button beside each generate button requests a
controlled cancellation. The encoder stops after the current model step; if
visible tokens already exist, the partial prompt is returned with the finish
reason `cancelled`.

Action buttons provide explicit feedback: their label changes as soon as the
click is accepted. Successful `txt2img` and `img2img` transfers show a checkmark
directly on the clicked button. Transfer-button labels reset when a new prompt
generation starts and when a new result arrives.

## Forge model loader

Forge remains the single source of truth for the selected checkpoint, text
encoder, and VAE. The compact runtime bar at the top reports whether a complete
KREA2 stack is active and offers one action: `Load current Forge selection`.

The button refreshes Forge's current loading parameters and invokes Forge's own
queued model reload. It does not change or save selections, add duplicate model
lists, scan model directories, download files, or instantiate weights itself.
Forge remains responsible for architecture detection, component assembly,
offloading, and memory management. The extension accepts the result only when
Forge exposes the expected KREA2 model, Qwen3-VL encoder/tokenizer, and
ModelPatcher.

Loading stays explicit so opening the assistant cannot unexpectedly allocate
VRAM or unload another active model.

## Personas

The selected persona text is the complete system prompt. In `Idea to prompt`,
the idea and its optional additional instruction form the user message. In
`Image to prompt`, the uploaded image and the optional instruction form the user
request. When the image instruction is empty, the UI uses the visible default
request shown in its placeholder. The extension adds no hidden style or content
instruction.

`Default` reproduces the original natural-language KREA2 prompt-writing
behavior. It can be edited but not deleted. Users can create, edit, and delete
additional personas in the UI.

Idea personas are stored locally in `personas.json`; image personas use the
separate `image_personas.json` file. Both use this format:

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
default. `personas.example.json` and `image_personas.example.json` contain the
two distributable defaults. When a local file does not exist, its built-in
Default persona is used automatically.

## Image to prompt

The image workflow accepts one uploaded image. EXIF orientation is applied,
alpha is composited by converting to RGB, and the image is normalized locally
to approximately 768 x 768 pixels of visual area while preserving its aspect
ratio. This normally produces about 576 Qwen3-VL visual tokens.

Those visual tokens replace the single image placeholder in the chat template.
The extension also applies Qwen3-VL multidimensional rotary positions and the
three DeepStack vision features during the language-model prefill. The uploaded
image is not passed to the KREA2 VAE and is not used as img2img conditioning;
it is analyzed only to produce text.

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

\* For the currently supported KREA2 Qwen3-VL 4B encoder with an FP16/BF16
KV cache. Runtime allocation uses the active encoder configuration and dtype.

Input and output share one window:

```text
available output = shared context - actual input tokens
```

The input includes the persona, user instruction, KREA2 chat template, and — in
the image workflow — the actual visual tokens. If the requested output limit is
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
- uses only the active KREA2 `clip.patcher`, encoder, tokenizer, and vision
  module;
- runs inside the Forge queue;
- asks Forge's memory manager to reserve the calculated KV-cache space;
- keeps no global encoder, tokenizer, or CUDA tensor reference;
- discards the temporary KV cache after each request;
- stores personas only in the local extension directory;
- does not write uploaded images to extension storage; Forge/Gradio may still
  use its normal temporary-upload handling;
- modifies no Forge core file;
- rejects unsupported model families and unknown KREA2 encoder classes.

## Current limitations

- KREA2 only;
- one uploaded image per image-to-prompt request;
- image upload generates text only and does not configure img2img or KREA2
  reference-image conditioning;
- no Z-Image backend yet;
- no prompt history;
- no dedicated Pony, Illustrious, or NoobAI tag-prompt adapters.

Future experiments are tracked in the project [roadmap](ROADMAP.md). The next
areas being considered are local prompt history, iterative prompt refinement,
privacy-conscious diagnostic reports, and a separate Z-Image adapter.

## Feedback and security

Please use the repository's structured
[issue forms](https://github.com/vibecodingtoolmaker/forge-krea-prompt-assistant/issues/new/choose)
for reproducible bug reports and focused feature requests. Remove personal
prompts, personas, images, local paths, access tokens, and unrelated log data
before submitting diagnostic information.

Do not disclose suspected vulnerabilities in a public issue. Follow the
[security policy](SECURITY.md) and use GitHub's private vulnerability reporting
flow from the repository's Security tab.

## License

AGPL-3.0-only. See `LICENSE`.
