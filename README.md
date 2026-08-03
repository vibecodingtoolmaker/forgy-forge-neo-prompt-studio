# Forge KREA2 Prompt Assistant

**Alpha 0.3.0-alpha.1**

A Forge Neo extension that turns a short image idea into a polished,
natural-language KREA2 prompt. It reuses the Qwen3-VL encoder and tokenizer
already owned by the active KREA2 diffusion engine. It does not download or
load a second language model.

This is an early alpha. The KREA2 text-generation path is functional, but the
extension has not yet been tested across a broad range of GPUs and Forge Neo
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

- Forge Neo with an active KREA2 model;
- the KREA2 Qwen3-VL text encoder loaded through Forge;
- a CUDA-capable GPU supported by the Forge installation.

Development and initial testing target Forge Neo 2.28.

## Installation

Clone or copy this repository into the Forge Neo `extensions` directory:

```powershell
cd sd-webui-forge-neo/extensions
git clone https://github.com/vibecodingtoolmaker/Forge-KREA2-Prompt-Assistant.git
```

```text
sd-webui-forge-neo/
└── extensions/
    └── Forge-KREA2-Prompt-Assistant/
```

Restart Forge Neo. The `KREA2 Prompt Assistant` tab will appear in the main UI.
No additional Python packages are required.

## Usage

1. Load KREA2 and its Qwen3-VL text encoder.
2. Run one normal image generation so Forge has initialized the known KREA2
   path.
3. Open `KREA2 Prompt Assistant`.
4. Select or edit a persona, enter an image idea, and choose the desired
   maximum output length.
5. Click `Generate prompt`.
6. Copy the result or send it directly to the positive `txt2img` or `img2img`
   prompt field.

The system status below the generated prompt shows the active persona, the
actual normalized seed, input/output token counts, available output context,
finish reason, and elapsed time.

## Personas

The selected persona text is the complete system prompt. The image idea is the
user message. The extension adds no hidden style or content instruction.

`Default` reproduces the original natural-language KREA2 prompt-writing
behavior. It can be edited but not deleted. Users can create, edit, and delete
additional personas in the UI.

Personas are stored locally in `personas.json` using this format:

```json
{
  "version": 1,
  "personas": {
    "Default": "Complete system prompt ...",
    "My Persona": "Another complete system prompt ..."
  }
}
```

The file is written atomically. Personal `personas.json` data is excluded from
Git by default; `personas.example.json` contains the distributable default.
When no local file exists, the built-in Default persona is used automatically.

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

The input includes the persona, image idea, and KREA2 chat template. If the
requested output limit is too large, the extension reduces it and reports the
effective value. After Forge places or offloads encoder weights, the extension
also checks currently free VRAM before allocating the cache.

Future reference-image support can use the same accounting. A KREA2 reference
image normalized to roughly 768 x 768 pixels typically adds about 576 visual
tokens plus vision control tokens.

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
- uses only the active KREA2 `clip.patcher`, encoder, and tokenizer;
- runs inside the Forge queue;
- asks Forge's memory manager to reserve the calculated KV-cache space;
- keeps no global encoder, tokenizer, or CUDA tensor reference;
- discards the temporary KV cache after each request;
- stores personas only in the local extension directory;
- modifies no Forge core file;
- rejects unsupported model families and unknown KREA2 encoder classes.

## Current limitations

- KREA2 only;
- no reference-image input in the assistant UI yet;
- no Z-Image backend yet;
- no prompt history;
- no dedicated Pony, Illustrious, or NoobAI tag-prompt adapters.

## License

AGPL-3.0-only. See `LICENSE`.
