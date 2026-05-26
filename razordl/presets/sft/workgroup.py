from razordl.core.base import logging
from razordl.core.engine.single_model.lm_workgroup import LMWorkGroup
from razordl.core.engine.single_model.workgroup import ModelGroup as _ModelGroup
from razordl.ops.model.huggingface import build_causal_lm, build_left_padding_tokenizer

logger = logging.getLogger(__name__)


class SFTModelGroup(_ModelGroup):
    """Standard HuggingFace CausalLM ModelGroup for SFT-style presets."""

    def build_processor(self):
        return build_left_padding_tokenizer(
            self.model_group_config.processor_config.processor_path,
            self.model_group_config.model_config.model_path,
        )

    def build_model(self):
        return build_causal_lm(
            self.model_group_config.model_config.model_path,
            device=self.device,
            use_bf16=self.model_group_config.model_config.use_bf16,
            local_rank=self.local_rank,
            logger=logger,
        )


class SFTWorkGroup(LMWorkGroup):
    """SFT preset: standard CausalLM model plus default next-token CE loss."""

    model_group_class = SFTModelGroup
