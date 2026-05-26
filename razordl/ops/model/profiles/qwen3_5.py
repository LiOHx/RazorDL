"""Profile for Qwen3.5 (model_type: ``qwen3_5``).

Qwen3.5 checkpoints carry a *multimodal* top-level config with nested
``text_config`` / ``vision_config`` subconfigs.  transformers' own
``Qwen3_5ForCausalLM`` has ``config_class = Qwen3_5TextConfig`` and relies on
the ``base_config_key = "text_config"`` unwrap path in
``PretrainedConfig.from_pretrained`` to descend into ``text_config`` when
loading.

vLLM 0.19+ does ``AutoConfig.register("qwen3_5", vllm.Qwen3_5Config,
exist_ok=True)`` on first engine touch, replacing transformers' registration
with vLLM's top-level config class.  When that happens, the ``base_config_key``
unwrap doesn't fire and ``Qwen3_5ForCausalLM`` receives the top-level config,
crashing inside ``Qwen3_5TextModel.__init__`` with
``AttributeError: 'Qwen3_5Config' object has no attribute 'vocab_size'``.

We pre-resolve the text-only sub-config here so ``from_pretrained(config=...)``
gets the right object regardless of who edited ``AutoConfig`` last.
"""
from razordl.ops.model.profiles.registry import ModelProfile, register


@register
class Qwen35Profile(ModelProfile):
    model_type = "qwen3_5"

    def prepare_config(self, cfg):
        if hasattr(cfg, "text_config") and not hasattr(cfg, "vocab_size"):
            return cfg.text_config
        return cfg
