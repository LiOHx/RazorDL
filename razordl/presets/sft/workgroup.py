import torch
from tensordict.tensordict import TensorDict

from razordl.core.base import logging
from razordl.core.engine.single_model.workgroup import ModelGroup as _ModelGroup, WorkGroup
from razordl.ops.loss.distributed import DistCrossEntropyLoss
from razordl.ops.loss.fused_linear_ce import fused_linear_cross_entropy
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
            precision=self.model_group_config.model_config.precision,
            trainable=self.is_trainable,
            local_rank=self.local_rank,
            logger=logger,
        )


class SFTWorkGroup(WorkGroup):
    """SFT preset: standard CausalLM model plus default next-token CE loss.

    The loss streams through FusedLinearCrossEntropy (see
    razordl/ops/loss/fused_linear_ce.py): the forward runs with
    ``logits_to_keep=1`` so the [B, L, V] logits are never materialized,
    and presets that need a different per-token loss (DFT's confidence
    weighting) only swap ``self.criterion`` -- the reduction contract is
    ``criterion.reduce_per_token_ce(ce_per_token, labels)``.
    """

    model_group_class = SFTModelGroup

    def __init__(self, config):
        super().__init__(config)
        self.model_group = self.model_group_class(config)
        mc = config.worker_group_config.model_group_config.model_config
        self.sp_size = getattr(mc, "sp_size", 1)
        self.fused_linear_tile_size = getattr(mc, "fused_linear_tile_size", 2048)
        self.criterion = DistCrossEntropyLoss(ignore_index=-100)

    def update_step(self, input_dict: TensorDict, step: int) -> dict:
        if self.sp_size > 1:
            from razordl.ops.parallel.sequence_parallel import split_for_sp

            sp_input = split_for_sp(
                input_dict["input_ids"],
                input_dict["attention_mask"],
                input_dict.get("labels"),
            )
            batch_size = getattr(input_dict, "batch_size", [sp_input["input_ids"].shape[0]])
            input_dict = TensorDict(dict(sp_input), batch_size=batch_size)

        labels = input_dict.pop("labels")
        model = self.model_group.model

        # split_for_sp hands back labels already shifted to next-token
        # targets (see its docstring); only the non-SP path rolls here.
        loss = self._compute_loss(model, input_dict, labels, shifted=self.sp_size > 1)
        raw_loss = loss.detach()
        self._backward_loss(loss, self.model_group)
        return {"loss": raw_loss.item()}

    def _compute_loss(self, model, input_dict, labels, shifted: bool = False):
        """``shifted`` means *labels* already hold next-token targets aligned
        with the unshifted hidden states (the SP split does this); otherwise
        the targets are rolled here and the garbage column is sliced off
        before the reduction (gradient-free, pinned by per_token_logp's
        gradient-parity test)."""
        softcap = getattr(model.config, "final_logit_softcapping", None)
        output = model(
            **input_dict,
            output_hidden_states=True,
            logits_to_keep=1,
        )
        hidden = output.hidden_states[-1]              # [B, L, D], post final norm
        raw_model = model.module if hasattr(model, "module") else model
        weight = raw_model.get_output_embeddings().weight

        if shifted:
            ce = fused_linear_cross_entropy(
                hidden, weight, labels, self.fused_linear_tile_size,
                softcap=softcap, ignore_index=-100,
            )
            target_labels = labels
        else:
            rolled = torch.cat([labels[:, 1:], labels[:, :1]], dim=1)
            nll = fused_linear_cross_entropy(
                hidden, weight, rolled, self.fused_linear_tile_size,
                softcap=softcap, ignore_index=-100,
            )
            ce = nll[:, :-1]
            target_labels = labels[:, 1:]
        return self.criterion.reduce_per_token_ce(ce, target_labels)
