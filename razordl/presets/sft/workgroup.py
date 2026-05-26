import types

import torch
import torch.utils.checkpoint
from tensordict.tensordict import TensorDict

from razordl.core.base import logging
from razordl.core.engine.single_model.workgroup import ModelGroup as _ModelGroup, WorkGroup
from razordl.ops.distributed.utils import all_gather_object
from razordl.ops.loss.distributed import DistCrossEntropyLoss
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


class SFTWorkGroup(WorkGroup):
    """SFT preset: standard CausalLM model plus default next-token CE loss."""

    model_group_class = SFTModelGroup

    def __init__(self, config):
        super().__init__(config)
        self.model_group = self.model_group_class(config)
        mc = config.worker_group_config.model_group_config.model_config
        self.sp_size = getattr(mc, "sp_size", 1)
        self.criterion = DistCrossEntropyLoss(ignore_index=-100)
        self.chunked_loss = getattr(mc, "chunked_loss", False)
        self.chunk_size = getattr(mc, "chunk_size", 2048)

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

        loss = self._compute_loss(model, input_dict, labels)
        loss.backward()
        return {"loss": loss.item()}

    def _compute_loss(self, model, input_dict, labels):
        if self.chunked_loss:
            return self._chunked_loss_compute(model, input_dict, labels)
        return self._simple_loss_compute(model, input_dict, labels)

    def _simple_loss_compute(self, model, input_dict, labels):
        output = model(**input_dict)
        logits = output.logits
        return self.criterion(
            logits[:, :-1, :].reshape(-1, logits.size(-1)),
            labels[:, 1:].reshape(-1),
        )

    def _chunked_loss_compute(self, model, input_dict, labels):
        """Monkey-patch forward to skip lm_head, then compute loss in chunks."""
        lm_head = model.lm_head
        text_model = self._find_text_model(model)
        softcap = getattr(model.config, "final_logit_softcapping", None)
        chunk_size = self.chunk_size
        original_forward = model.forward

        def _chunked_forward(
            self_m,
            input_ids=None,
            attention_mask=None,
            position_ids=None,
            labels=None,
            **kwargs,
        ):
            kwargs.pop("logits_to_keep", None)
            outputs = text_model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                position_ids=position_ids,
                **kwargs,
            )
            hidden_states = outputs.last_hidden_state

            loss = None
            if labels is not None:
                hs = hidden_states[:, :-1, :].contiguous()
                target = labels[:, 1:].contiguous()
                seq_len = hs.shape[1]
                total_loss = torch.zeros(1, device=hs.device, dtype=torch.float32)

                for start in range(0, seq_len, chunk_size):
                    end = min(start + chunk_size, seq_len)
                    c_hs = hs[:, start:end, :]
                    c_tgt = target[:, start:end].reshape(-1)

                    def _chunk_fn(h, t):
                        logits = lm_head(h)
                        if softcap is not None:
                            logits = logits / softcap
                            logits = torch.tanh(logits) * softcap
                        return torch.nn.functional.cross_entropy(
                            logits.reshape(-1, logits.size(-1)).float(),
                            t,
                            ignore_index=-100,
                            reduction="sum",
                        ).unsqueeze(0)

                    ce = torch.utils.checkpoint.checkpoint(
                        _chunk_fn,
                        c_hs,
                        c_tgt,
                        use_reentrant=False,
                    )
                    total_loss = total_loss + ce.squeeze(0)

                valid_local = (target != -100).sum().item()
                batch_valid = sum(all_gather_object(valid_local))
                loss = total_loss / max(batch_valid, 1)

            from transformers.modeling_outputs import CausalLMOutputWithPast

            return CausalLMOutputWithPast(loss=loss, logits=None)

        model.forward = types.MethodType(_chunked_forward, model)
        try:
            input_dict["labels"] = labels
            output = model(**input_dict)
            return output.loss
        finally:
            model.forward = original_forward

    @staticmethod
    def _find_text_model(model):
        t = model
        for attr in ("model", "model", "language_model"):
            nxt = getattr(t, attr, None)
            if nxt is None:
                continue
            if not hasattr(nxt, "lm_head"):
                return nxt
            t = nxt
        return t
