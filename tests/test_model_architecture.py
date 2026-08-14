# Copyright (C) 2026 vibecodingtoolmaker
# SPDX-License-Identifier: AGPL-3.0-only

from pathlib import Path
from types import SimpleNamespace
import unittest

from forgy.adapters import (
    AdapterError,
    Flux2KleinAdapter,
    Krea2Adapter,
    ZImageAdapter,
)
from forgy.capabilities import (
    CapabilityManager,
    ModelCapabilities,
    PromptDialect,
    ReasoningMode,
    Workflow,
)
from forgy.model_manager import ModelManager, UnsupportedModelError


ROOT = Path(__file__).resolve().parents[1]


def _runtime_class(name: str, module: str):
    value = type(name, (), {})
    value.__module__ = module
    return value


def _complete_krea2_stack():
    encoder_type = _runtime_class("Qwen3VL", "backend.nn.llm.llama")
    tokenizer_type = _runtime_class("Qwen3VLTokenizer", "test_runtime")
    model_type = _runtime_class("Krea2", "backend.diffusion_engine.krea")

    encoder = encoder_type()
    tokenizer = tokenizer_type()
    patcher = object()
    clip = SimpleNamespace(
        cond_stage_model=SimpleNamespace(qwen3vl_4b=encoder),
        tokenizer=SimpleNamespace(qwen3vl_4b=tokenizer),
        patcher=patcher,
    )
    engine = SimpleNamespace(text_encoder=encoder, tokenizer=tokenizer)
    model = model_type()
    model.forge_objects = SimpleNamespace(clip=clip)
    model.text_processing_engine_qwen = engine
    return model, encoder, tokenizer, patcher


def _complete_zimage_stack(*, checkpoint_name="Z-Image-Turbo.safetensors"):
    encoder_type = _runtime_class("Qwen3_4B", "backend.nn.llm.llama")
    tokenizer_type = _runtime_class("Qwen2Tokenizer", "test_runtime")
    engine_type = _runtime_class(
        "Qwen3TextProcessingEngine", "backend.text_processing.qwen3_engine"
    )
    model_type = _runtime_class("ZImage", "backend.diffusion_engine.zimage")

    encoder = encoder_type()
    encoder.model = SimpleNamespace(
        config=SimpleNamespace(hidden_size=2560, vocab_size=151936)
    )
    tokenizer = tokenizer_type()
    patcher = object()
    clip = SimpleNamespace(
        cond_stage_model=SimpleNamespace(qwen3=encoder),
        tokenizer=SimpleNamespace(qwen3=tokenizer),
        patcher=patcher,
    )
    engine = engine_type()
    engine.text_encoder = encoder
    engine.tokenizer = tokenizer
    engine.llama_template = "<|im_start|>user\n{}<|im_end|>\n<|im_start|>assistant\n"
    model = model_type()
    model.forge_objects = SimpleNamespace(clip=clip)
    model.text_processing_engine_gemma = engine
    model.sd_checkpoint_info = SimpleNamespace(name=checkpoint_name)
    return model, encoder, tokenizer, patcher, engine


def _complete_flux2_klein_stack(
    *,
    checkpoint_name="flux-2-klein-4b.safetensors",
    encoder_name="Qwen3_4B",
    hidden_size=2560,
):
    encoder_type = _runtime_class(encoder_name, "backend.nn.llm.llama")
    tokenizer_type = _runtime_class("Qwen2TokenizerFast", "test_runtime")
    engine_type = _runtime_class(
        "KleinTextProcessingEngine", "backend.text_processing.klein_engine"
    )
    model_type = _runtime_class("Flux2", "backend.diffusion_engine.flux2")

    encoder = encoder_type()
    encoder.model = SimpleNamespace(
        config=SimpleNamespace(hidden_size=hidden_size, vocab_size=151936)
    )
    tokenizer = tokenizer_type()
    patcher = object()
    clip = SimpleNamespace(
        cond_stage_model=SimpleNamespace(qwen3=encoder),
        tokenizer=SimpleNamespace(qwen3=tokenizer),
        patcher=patcher,
    )
    engine = engine_type()
    engine.text_encoder = encoder
    engine.tokenizer = tokenizer
    engine.llama_template = (
        "<|im_start|>user\n{}<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n"
    )
    model = model_type()
    model.forge_objects = SimpleNamespace(clip=clip)
    model.text_processing_engine_gemma = engine
    model.sd_checkpoint_info = SimpleNamespace(name=checkpoint_name)
    return model, encoder, tokenizer, patcher, engine


class ModelArchitectureTests(unittest.TestCase):
    def test_every_new_module_compiles_without_forge_dependencies(self) -> None:
        for path in (ROOT / "forgy").rglob("*.py"):
            source = path.read_text(encoding="utf-8")
            compile(source, str(path), "exec")
            self.assertNotIn("from_pretrained(", source)

    def test_krea2_capabilities_are_explicit_and_model_neutral(self) -> None:
        capabilities = Krea2Adapter.capabilities
        self.assertEqual(capabilities.prompt_dialect, PromptDialect.NATURAL_LANGUAGE)
        self.assertEqual(capabilities.reasoning_mode, ReasoningMode.CONTROLLABLE)
        self.assertTrue(capabilities.text_generation)
        self.assertTrue(capabilities.vision_input)
        self.assertTrue(capabilities.multi_turn_chat)
        self.assertTrue(capabilities.system_prompt)
        self.assertEqual(
            capabilities.workflows,
            frozenset(
                {
                    Workflow.FORGY_CHAT,
                    Workflow.IDEA_TO_PROMPT,
                    Workflow.IMAGE_TO_PROMPT,
                    Workflow.REFINE_PROMPT,
                }
            ),
        )

    def test_manager_resolves_request_local_krea2_context(self) -> None:
        model, encoder, tokenizer, patcher = _complete_krea2_stack()
        adapter = Krea2Adapter()
        manager = ModelManager((adapter,))

        first = manager.resolve(model)
        second = manager.resolve(model)

        self.assertEqual(first.identity.adapter_id, "krea2-qwen3-vl")
        self.assertEqual(first.identity.family, "krea2")
        self.assertEqual(first.identity.variant, "qwen3-vl-4b")
        self.assertIs(first.components.encoder, encoder)
        self.assertIs(first.components.tokenizer, tokenizer)
        self.assertIs(first.components.patcher, patcher)
        self.assertIsNot(first, second)
        self.assertIsNot(first.components, second.components)
        self.assertNotIn("components", adapter.__dict__)
        self.assertNotIn("sd_model", adapter.__dict__)

    def test_manager_fails_closed_for_unknown_or_incomplete_stacks(self) -> None:
        adapter = Krea2Adapter()
        manager = ModelManager((adapter,))

        with self.assertRaisesRegex(UnsupportedModelError, "No diffusion model"):
            manager.resolve(None)
        with self.assertRaisesRegex(UnsupportedModelError, "No supported model"):
            manager.resolve(object())

        model_type = _runtime_class("Krea2", "backend.diffusion_engine.krea")
        incomplete = model_type()
        incomplete.forge_objects = SimpleNamespace(clip=None)
        incomplete.text_processing_engine_qwen = None
        with self.assertRaisesRegex(AdapterError, "expected text encoder"):
            manager.resolve(incomplete)

    def test_zimage_adapter_resolves_base_and_turbo_without_vision(self) -> None:
        adapter = ZImageAdapter()
        manager = ModelManager((Krea2Adapter(), adapter))
        model, encoder, tokenizer, patcher, _engine = _complete_zimage_stack()

        context = manager.resolve(model)

        self.assertEqual(context.identity.adapter_id, "z-image-qwen3-4b")
        self.assertEqual(context.identity.family, "z-image")
        self.assertEqual(context.identity.variant, "turbo")
        self.assertIs(context.components.encoder, encoder)
        self.assertIs(context.components.tokenizer, tokenizer)
        self.assertIs(context.components.patcher, patcher)
        self.assertTrue(context.capabilities.text_generation)
        self.assertFalse(context.capabilities.vision_input)
        self.assertFalse(context.capabilities.supports(Workflow.IMAGE_TO_PROMPT))
        self.assertTrue(context.capabilities.supports(Workflow.FORGY_CHAT))

        base_model, *_ = _complete_zimage_stack(
            checkpoint_name="Z-Image-Base.safetensors"
        )
        self.assertEqual(manager.resolve(base_model).identity.variant, "base")

    def test_zimage_adapter_builds_qwen_system_and_user_roles(self) -> None:
        adapter = ZImageAdapter()
        _model, _encoder, _tokenizer, _patcher, engine = _complete_zimage_stack()

        rendered = adapter.render_generation_prompt(
            engine,
            "a red fox",
            "use dramatic backlight",
            "Return one polished natural-language prompt.",
            workflow=Workflow.IDEA_TO_PROMPT,
            validate_system_prompt=lambda value, **_kwargs: value,
        )

        self.assertTrue(rendered.startswith("<|im_start|>system\n"))
        self.assertIn("<|im_start|>user\na red fox", rendered)
        self.assertIn("Additional instruction:\nuse dramatic backlight", rendered)
        self.assertIn("Treat the user's image idea", rendered)
        self.assertIn("return exactly one finished image-generation prompt", rendered)
        self.assertTrue(rendered.endswith("<think>\n\n</think>\n\n"))

    def test_zimage_workflow_contracts_are_task_specific(self) -> None:
        adapter = ZImageAdapter()
        _model, _encoder, _tokenizer, _patcher, engine = _complete_zimage_stack()
        validate = lambda value, **_kwargs: value

        refinement = adapter.render_generation_prompt(
            engine,
            "Existing prompt:\\nA portrait.",
            "",
            "custom persona",
            workflow=Workflow.REFINE_PROMPT,
            validate_system_prompt=validate,
        )
        chat = adapter.render_generation_prompt(
            engine,
            "Latest user message:\\nWiden the shot.",
            "",
            "custom persona",
            workflow=Workflow.FORGY_CHAT,
            validate_system_prompt=validate,
        )

        self.assertIn("Revise the existing image prompt only", refinement)
        self.assertIn("FORGY_REPLY: and UPDATED_PROMPT:", chat)

    def test_zimage_capabilities_disable_image_workflow_only(self) -> None:
        model, *_ = _complete_zimage_stack()
        context = ModelManager((ZImageAdapter(),)).resolve(model)

        effective = CapabilityManager().resolve(context)

        self.assertTrue(effective.for_workflow(Workflow.IDEA_TO_PROMPT).enabled)
        self.assertTrue(effective.for_workflow(Workflow.REFINE_PROMPT).enabled)
        self.assertTrue(effective.for_workflow(Workflow.FORGY_CHAT).enabled)
        image_state = effective.for_workflow(Workflow.IMAGE_TO_PROMPT)
        self.assertFalse(image_state.enabled)
        self.assertIn("does not support", image_state.reason)

    def test_flux2_klein_4b_resolves_as_isolated_text_adapter(self) -> None:
        adapter = Flux2KleinAdapter()
        manager = ModelManager((Krea2Adapter(), ZImageAdapter(), adapter))
        model, encoder, tokenizer, patcher, _engine = _complete_flux2_klein_stack()

        context = manager.resolve(model)

        self.assertEqual(context.identity.adapter_id, "flux2-klein-qwen3-4b")
        self.assertEqual(context.identity.family, "flux2-klein")
        self.assertEqual(context.identity.variant, "4b")
        self.assertEqual(
            context.identity.display_name,
            "FLUX.2 Klein 4B · Qwen3 4B",
        )
        self.assertIs(context.components.encoder, encoder)
        self.assertIs(context.components.tokenizer, tokenizer)
        self.assertIs(context.components.patcher, patcher)
        self.assertTrue(context.capabilities.text_generation)
        self.assertFalse(context.capabilities.vision_input)
        self.assertFalse(context.capabilities.supports(Workflow.IMAGE_TO_PROMPT))
        self.assertTrue(context.capabilities.supports(Workflow.FORGY_CHAT))
        self.assertNotIn("components", adapter.__dict__)
        self.assertNotIn("sd_model", adapter.__dict__)

        base_model, *_ = _complete_flux2_klein_stack(
            checkpoint_name="FLUX.2-klein-base-4B.safetensors"
        )
        base_context = manager.resolve(base_model)
        self.assertEqual(base_context.identity.variant, "base-4b")
        self.assertIn("Base 4B", base_context.identity.display_name)

    def test_flux2_klein_rejects_9b_without_untied_lm_head(self) -> None:
        model, *_ = _complete_flux2_klein_stack(
            checkpoint_name="flux-2-klein-9b.safetensors",
            encoder_name="Qwen3_8B",
            hidden_size=4096,
        )

        with self.assertRaisesRegex(AdapterError, "untied LM head"):
            ModelManager((Flux2KleinAdapter(),)).resolve(model)

    def test_flux2_klein_builds_its_existing_no_think_template_once(self) -> None:
        adapter = Flux2KleinAdapter()
        _model, _encoder, _tokenizer, _patcher, engine = _complete_flux2_klein_stack()

        rendered = adapter.render_generation_prompt(
            engine,
            "a glass greenhouse in winter",
            "use soft morning light",
            "Return one polished natural-language prompt.",
            workflow=Workflow.IDEA_TO_PROMPT,
            validate_system_prompt=lambda value, **_kwargs: value,
        )

        self.assertTrue(rendered.startswith("<|im_start|>system\n"))
        self.assertIn("<|im_start|>user\na glass greenhouse", rendered)
        self.assertIn("Additional instruction:\nuse soft morning light", rendered)
        self.assertIn("Treat the user's image idea", rendered)
        self.assertIn("finished natural-language image prompt", rendered)
        self.assertEqual(rendered.count("<think>"), 1)
        self.assertTrue(rendered.endswith("<think>\n\n</think>\n\n"))

    def test_runtime_binding_keeps_dispatch_inside_the_selected_adapter(self) -> None:
        calls = []
        adapter = Krea2Adapter()
        adapter.bind_runtime(
            text_generator=lambda **kwargs: calls.append(("text", kwargs)) or "text",
            image_generator=lambda **kwargs: calls.append(("image", kwargs)) or "image",
        )

        self.assertEqual(adapter.generate_text(value=1), "text")
        self.assertEqual(adapter.generate_image(value=2), "image")
        self.assertEqual(
            calls,
            [("text", {"value": 1}), ("image", {"value": 2})],
        )

    def test_capability_manager_disables_only_unsupported_workflows(self) -> None:
        capabilities = ModelCapabilities(
            workflows=frozenset(Workflow),
            prompt_dialect=PromptDialect.TAGS,
            text_generation=True,
            vision_input=False,
            multi_turn_chat=True,
            system_prompt=True,
            reasoning_mode=ReasoningMode.HIDDEN,
            sampling_controls=frozenset({"temperature"}),
            default_persona_family="test-tags",
        )
        context = SimpleNamespace(
            identity=SimpleNamespace(
                adapter_id="test-adapter",
                display_name="Test adapter",
            ),
            capabilities=capabilities,
        )

        effective = CapabilityManager().resolve(context)

        self.assertTrue(effective.for_workflow(Workflow.FORGY_CHAT).enabled)
        image_state = effective.for_workflow(Workflow.IMAGE_TO_PROMPT)
        self.assertFalse(image_state.enabled)
        self.assertIn("no vision input", image_state.reason)
        with self.assertRaises(TypeError):
            effective.workflows[Workflow.FORGY_CHAT] = image_state

    def test_unbound_adapter_runtime_fails_clearly(self) -> None:
        adapter = Krea2Adapter()
        with self.assertRaisesRegex(AdapterError, "text runtime"):
            adapter.generate_text()
        with self.assertRaisesRegex(AdapterError, "vision runtime"):
            adapter.generate_image()


if __name__ == "__main__":
    unittest.main()
