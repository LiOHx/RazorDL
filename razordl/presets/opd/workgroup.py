"""PG On-Policy Distillation (OPD) workgroup.

Student samples its own rollouts via vLLM; teacher provides per-token
log-probs that become the (negative) advantage in a PPO clipped objective.
See ``razordl/presets/opd/README.md`` and the project plan for the math.
"""
import contextlib
import copy
import os

import torch

from razordl.core.base import logging
from razordl.core.base.metrics import DistStats
from razordl.core.engine.on_policy_single_model.config import Config
from razordl.core.engine.on_policy_single_model.modelgroup import ModelGroup as _ModelGroup
from razordl.core.engine.on_policy_single_model.workgroup import WorkGroup as _WorkGroup
from razordl.ops.distributed.utils import all_gather_object
from razordl.ops.model.huggingface import build_causal_lm, build_left_padding_tokenizer
from razordl.ops.model.per_token_logp import compute_per_token_log_probs

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Shared CausalLM model group
# ---------------------------------------------------------------------------

class OPDCausalLMModelGroup(_ModelGroup):
    """Shared HuggingFace CausalLM loader for student (policy) and teacher."""

    def build_processor(self):
        return build_left_padding_tokenizer(
            self.model_group_config.processor_config.processor_path,
            self.model_group_config.model_config.model_path,
            ensure_pad_token=True,
        )

    def build_model(self):
        return build_causal_lm(
            self.model_group_config.model_config.model_path,
            device=self.device,
            use_bf16=self.model_group_config.model_config.use_bf16,
            local_rank=self.local_rank,
            logger=logger,
        )


# ---------------------------------------------------------------------------
# Policy (student) model group — trainable + vLLM rollout
# ---------------------------------------------------------------------------

class OPDPolicyModelGroup(OPDCausalLMModelGroup):
    """Trainable student model with optional vLLM rollout engine.

    OPD uses ``n=1`` sampling (one rollout per prompt); the vLLM init mirrors
    GRPO's but pins ``SamplingParams.n=1``.
    """

    def __init__(self, config: Config):
        super().__init__(config)
        self._vllm_engine = None
        self._init_vllm_engine()

    def _init_vllm_engine(self):
        if os.environ.get("SKIP_VLLM"):
            logger.info("SKIP_VLLM set, using HF generate fallback")
            return
        try:
            from razordl.ops.model.vllm_rollout import (
                GRPOVLLMRollout, InferenceConfig, SamplingParamsConfig,
            )
        except ImportError as e:
            logger.warning("vLLM not available: %s", e)
            return
        try:
            model_cfg = self.model_group_config.model_config
            adapter_cfg = model_cfg.adapter_config
            proc_cfg = self.model_group_config.processor_config

            model_path = model_cfg.model_path
            max_prompt_len = getattr(proc_cfg, "max_length", 512)
            data_cfg = self.config.data_config
            max_completion_len = getattr(data_cfg, "max_completion_length", 256)
            max_model_len = max_prompt_len + max_completion_len
            use_lora = adapter_cfg.use_adapter
            lora_r = getattr(adapter_cfg, "lora_r", 8)

            inf_cfg = InferenceConfig(
                model=model_path,
                enable_sleep_mode=True,
                tensor_parallel_size=1,
                distributed_executor_backend="external_launcher",
                dtype="bfloat16" if model_cfg.use_bf16 else "float16",
                enforce_eager=False,
                gpu_memory_utilization=0.25,
                disable_custom_all_reduce=True,
                skip_tokenizer_init=False,
                max_model_len=max_model_len,
                max_num_seqs=64,
                load_format="auto",
                disable_log_stats=True,
                max_num_batched_tokens=max_model_len * 4,
                enable_chunked_prefill=True,
                enable_prefix_caching=True,
                trust_remote_code=True,
                enable_lora=use_lora,
                max_lora_rank=opd_vllm_max_lora_rank(lora_r) if use_lora else 64,
                seed=self.config.trainer_config.seed,
            )

            sp_cfg = SamplingParamsConfig(
                n=1,
                temperature=getattr(data_cfg, "temperature", 0.7),
                top_p=getattr(data_cfg, "top_p", 0.9),
                top_k=getattr(data_cfg, "top_k", 50),
                max_tokens=max_completion_len,
                detokenize=False,
            )

            self._vllm_engine = GRPOVLLMRollout(inf_cfg, sp_cfg, use_lora, lora_r)
            self._sync_weights_to_vllm()
        except Exception as e:
            logger.warning("vLLM init failed: %s", e)
            self._vllm_engine = None

    def _sync_weights_to_vllm(self):
        engine = self._vllm_engine
        if engine is None:
            return
        use_lora = self.model_group_config.model_config.adapter_config.use_adapter
        if use_lora:
            _sync_lora_weights(self.model, engine)
        else:
            _sync_full_weights(self.model, engine)

    @contextlib.contextmanager
    def vllm_rollout_context(self):
        self._sync_weights_to_vllm()
        try:
            yield self._vllm_engine
        finally:
            pass


def opd_vllm_max_lora_rank(lora_rank: int) -> int:
    for r in (8, 16, 32, 64, 128, 256, 320, 512):
        if lora_rank <= r:
            return r
    raise ValueError(f"lora_rank too large: {lora_rank}")


def _sync_lora_weights(model, engine):
    peft_config = getattr(model, "peft_config", {}).get("default", None)
    if peft_config is None:
        return

    from torch.distributed.tensor import DTensor

    params = model.base_model.model.state_dict()
    lora_params = ((k, v) for k, v in params.items() if "lora" in k)

    def _to_full(kv):
        name, tensor = kv
        if isinstance(tensor, DTensor):
            tensor = tensor.full_tensor()
        return name, tensor.cpu()

    weights_iter = (_to_full(p) for p in lora_params)
    engine.update_weights(weights_iter, peft_config=asdict_peft(peft_config))


def _sync_full_weights(model, engine):
    from torch.distributed.tensor import DTensor

    params = model.state_dict()
    weights_iter = (
        (k, v.full_tensor().cpu() if isinstance(v, DTensor) else v.cpu())
        for k, v in params.items()
    )
    engine.update_weights(weights_iter)


def asdict_peft(peft_config) -> dict:
    if not peft_config:
        return {}
    return {
        "r": getattr(peft_config, "r", 8),
        "lora_alpha": getattr(peft_config, "lora_alpha", 16),
        "target_modules": list(getattr(peft_config, "target_modules", [])),
        "bias": getattr(peft_config, "bias", "none"),
        "task_type": str(getattr(peft_config, "task_type", "CAUSAL_LM")),
        "lora_dropout": getattr(peft_config, "lora_dropout", 0.0),
    }


# ---------------------------------------------------------------------------
# Teacher model group — frozen, never saved to checkpoint
# ---------------------------------------------------------------------------

class OPDTeacherModelGroup(OPDCausalLMModelGroup):
    """Frozen teacher.

    ``FSDPModelGroup.save_*`` already early-returns for ``is_trainable=False``
    after the Phase-1 fix, but we also override here as defense in depth so
    the teacher cannot accidentally be persisted by a future refactor.
    """

    def save_model_and_processor(self, checkpoint_dir: str):
        return

    def save_checkpoint(self, checkpoint_dir: str):
        return


# ---------------------------------------------------------------------------
# OPD WorkGroup
# ---------------------------------------------------------------------------

class OPDWorkGroup(_WorkGroup):
    """PG On-Policy Distillation WorkGroup.

    Rollouts come from the student; the teacher provides per-token logπ which
    is fed into a PPO clip objective as ``advantage = -kl_penalty(...).detach()``.
    No reference-KL penalty term is added (the teacher signal already lives
    in the advantage).
    """

    def __init__(self, config: Config):
        super().__init__(config)
        self.config = config
        dc = config.data_config
        if not dc.teacher_model:
            raise ValueError(
                "OPD requires data_config.teacher_model (set `teacher_model:` in YAML)."
            )

        self.clip_eps = float(dc.clip_eps)
        self.loss_mode = str(dc.loss_mode)
        self.loss_max_clamp = float(dc.loss_max_clamp)
        self.logp_min_clamp = float(dc.log_prob_min_clamp)
        self.temperature = float(dc.temperature)
        self.top_p = float(dc.top_p)
        self.top_k = int(dc.top_k)
        self._last_metrics = {}

        policy_config = copy.deepcopy(config)
        policy_config.worker_group_config.model_group_config.model_group_name = "policy_model_group"
        self.policy_model_group = OPDPolicyModelGroup(policy_config)

        teacher_config = copy.deepcopy(config)
        tmgc = teacher_config.worker_group_config.model_group_config
        tmgc.model_group_name = "reference_model_group"
        tmgc.model_config.model_path = dc.teacher_model
        tmgc.processor_config.processor_path = (
            dc.teacher_processor_path or tmgc.model_config.model_path
        )
        tmgc.model_config.is_trainable = False
        tmgc.model_config.adapter_config.use_adapter = False
        tmgc.optimizer_config.learning_rate = 0.0
        self.reference_model_group = OPDTeacherModelGroup(teacher_config)

        self._validate_tokenizer_compat()

    def _validate_tokenizer_compat(self):
        s = self.policy_model_group.processor
        t = self.reference_model_group.processor
        if s.vocab_size != t.vocab_size:
            raise ValueError(
                f"OPD: student vocab_size={s.vocab_size}, teacher vocab_size={t.vocab_size}. "
                "Student and teacher must share the same tokenizer."
            )
        if s.get_vocab() != t.get_vocab():
            raise ValueError("OPD: student and teacher get_vocab() differ")
        for attr in ("pad_token_id", "eos_token_id", "bos_token_id"):
            sv, tv = getattr(s, attr, None), getattr(t, attr, None)
            if sv != tv:
                raise ValueError(f"OPD: tokenizer.{attr} mismatch (student={sv}, teacher={tv})")

    # ------------------------------------------------------------------
    # Rollout
    # ------------------------------------------------------------------

    def rollout(self, input_dict: dict, step: int) -> dict:
        prompt_ids = input_dict["prompt_ids"]
        prompt_mask = input_dict["prompt_attention_mask"]
        batch_size = prompt_ids.size(0)
        device = prompt_ids.device
        processor = self.policy_model_group.processor

        vllm_seed = self.config.trainer_config.seed + step
        prompt_token_ids = _remove_left_padding_batch(prompt_ids, processor.pad_token_id)

        vllm = self.policy_model_group._vllm_engine
        use_vllm = vllm is not None

        if use_vllm:
            with self.policy_model_group.vllm_rollout_context() as vllm_engine:
                gen_output = vllm_engine.generate(prompt_token_ids, seed=vllm_seed)
            response_token_ids = gen_output["response_token_ids"]
            response_masks_list = gen_output["response_mask"]

            all_input_ids = []
            all_attention_masks = []
            all_response_masks = []

            for i in range(batch_size):
                pt_ids = list(prompt_token_ids[i])
                rsp_ids = response_token_ids[i]
                full = pt_ids + rsp_ids
                all_input_ids.append(full)
                all_attention_masks.append([1] * len(full))
                rm = [0] * len(pt_ids) + response_masks_list[i]
                all_response_masks.append(rm)
        else:
            max_new = getattr(self.config.data_config, "max_completion_length", 256)
            with torch.no_grad():
                generated = self.policy_model_group.model.generate(
                    input_ids=prompt_ids,
                    attention_mask=prompt_mask,
                    max_new_tokens=max_new,
                    do_sample=True,
                    temperature=self.temperature,
                    top_p=self.top_p,
                    top_k=self.top_k,
                    pad_token_id=processor.pad_token_id,
                    eos_token_id=processor.eos_token_id,
                )

            prompt_len = prompt_ids.size(1)
            attention_mask = (generated != processor.pad_token_id).long()
            response_mask = torch.zeros_like(attention_mask)
            response_mask[:, prompt_len:] = 1
            response_mask = response_mask * attention_mask

            all_input_ids = [generated[i].tolist() for i in range(generated.size(0))]
            all_attention_masks = [attention_mask[i].tolist() for i in range(attention_mask.size(0))]
            all_response_masks = [response_mask[i].tolist() for i in range(response_mask.size(0))]

        max_len = max(len(ids) for ids in all_input_ids)
        pad_id = processor.pad_token_id

        for i in range(len(all_input_ids)):
            pad_len = max_len - len(all_input_ids[i])
            all_input_ids[i] = [pad_id] * pad_len + all_input_ids[i]
            all_attention_masks[i] = [0] * pad_len + all_attention_masks[i]
            all_response_masks[i] = [0] * pad_len + all_response_masks[i]

        return {
            "input_ids": torch.tensor(all_input_ids, dtype=torch.long, device=device),
            "attention_mask": torch.tensor(all_attention_masks, dtype=torch.long, device=device),
            "response_mask": torch.tensor(all_response_masks, dtype=torch.long, device=device),
        }

    # ------------------------------------------------------------------
    # Engine-hook stubs (unused — OPD overrides _run_update_step)
    # ------------------------------------------------------------------

    def compute_reward(self, rollout_output: dict) -> torch.Tensor:
        raise NotImplementedError("OPD computes per-token advantage inside _run_update_step")

    def compute_advantage(self, rewards: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError("OPD computes per-token advantage inside _run_update_step")

    def compute_loss(self, rollout_output: dict, advantages: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError("OPD uses _compute_loss_chunk inside _run_update_step")

    # ------------------------------------------------------------------
    # Core: chunked PPO PG loss with no ref-KL
    # ------------------------------------------------------------------

    def _compute_loss_chunk(self, mini: dict, advantages: torch.Tensor):
        """Pure-tensor PPO-clipped policy-gradient loss for one chunk.

        Does NOT open ``trainer_context()`` / ``inference_context()`` — the
        caller (``_run_update_step``) is responsible for those.
        """
        input_ids = mini["input_ids"]
        attention_mask = mini["attention_mask"]
        response_mask = mini["response_mask"]
        old_log_probs = mini["old_log_probs"]

        log_probs = compute_per_token_log_probs(
            self.policy_model_group.model,
            input_ids,
            attention_mask,
            logp_min_clamp=self.logp_min_clamp,
            no_grad=False,
        )
        log_ratio = log_probs - old_log_probs
        ratio = log_ratio.exp()
        eps = self.clip_eps
        ratio_clipped = ratio.clamp(1.0 - eps, 1.0 + eps)
        pg_loss = -torch.min(ratio * advantages, ratio_clipped * advantages)

        response_mask_shifted = response_mask[:, 1:]
        per_token_loss = pg_loss * response_mask_shifted

        clipped = (torch.abs(ratio - 1.0) > eps).float()
        masked_tokens = response_mask_shifted > 0
        metrics = {
            "clipped_count": clipped[masked_tokens].sum().item() if masked_tokens.any() else 0.0,
            "ratio_sum": ratio[masked_tokens].sum().item() if masked_tokens.any() else 0.0,
            "token_count": response_mask_shifted.sum().item(),
        }

        local_bs = input_ids.size(0)
        global_bs = sum(all_gather_object(local_bs))
        loss_sum = per_token_loss.sum() * local_bs / global_bs
        return loss_sum, metrics

    def _run_update_step(self, input_dict: dict, step: int) -> dict:
        policy_group = self._get_policy_model_group()
        teacher_group = self._get_reference_model_group()

        # ---- 1) Student rollout -------------------------------------------
        with policy_group.inference_context():
            rollout_output = self.rollout(input_dict, step)

        # ---- 2) old_log_probs from policy (no-grad) -----------------------
        with policy_group.inference_context():
            old_log_probs = compute_per_token_log_probs(
                policy_group.model,
                rollout_output["input_ids"],
                rollout_output["attention_mask"],
                logp_min_clamp=self.logp_min_clamp,
                no_grad=True,
            )
        rollout_output["old_log_probs"] = old_log_probs

        # ---- 3) Teacher forward → per-token advantage ---------------------
        with teacher_group.inference_context():
            teacher_log_probs = compute_per_token_log_probs(
                teacher_group.model,
                rollout_output["input_ids"],
                rollout_output["attention_mask"],
                logp_min_clamp=self.logp_min_clamp,
                no_grad=True,
            )
        distill_loss = kl_penalty(old_log_probs, teacher_log_probs, self.loss_mode)
        distill_loss = distill_loss.clamp(-self.loss_max_clamp, self.loss_max_clamp)
        advantages = -distill_loss.detach()  # [B, L-1]

        # ---- 4) Global normalization (chunked-loss pattern from GRPO) ------
        response_mask_shifted = rollout_output["response_mask"][:, 1:]
        local_token_count = response_mask_shifted.sum().item()
        global_token_count = max(sum(all_gather_object(local_token_count)), 1e-8)

        total_samples = rollout_output["input_ids"].size(0)
        mini_batch_size = input_dict["prompt_ids"].size(0)

        total_loss_sum = 0.0
        total_clipped = 0.0
        total_ratio_sum = 0.0
        total_tokens = 0.0
        num_chunks = 0

        # ---- 5) Chunked PPO PG loss + backward ----------------------------
        for i in range(0, total_samples, mini_batch_size):
            end = min(i + mini_batch_size, total_samples)
            mini = {
                "input_ids": rollout_output["input_ids"][i:end],
                "attention_mask": rollout_output["attention_mask"][i:end],
                "response_mask": rollout_output["response_mask"][i:end],
                "old_log_probs": old_log_probs[i:end],
            }
            mini_adv = advantages[i:end]

            with policy_group.trainer_context():
                loss_sum, metrics = self._compute_loss_chunk(mini, mini_adv)
                loss = loss_sum / global_token_count
                loss.backward()

            total_loss_sum += loss_sum.item()
            total_clipped += metrics["clipped_count"]
            total_ratio_sum += metrics["ratio_sum"]
            total_tokens += metrics["token_count"]
            num_chunks += 1

        valid = response_mask_shifted.bool()
        self._last_metrics = {
            "clip_fraction": total_clipped / (total_tokens + 1e-8),
            "mean_ratio": total_ratio_sum / (total_tokens + 1e-8),
        }

        return dict(
            loss=total_loss_sum / max(num_chunks * global_token_count, 1e-8),
            advantage=DistStats.from_tensor(advantages[valid]),
            distill_loss=DistStats.from_tensor(distill_loss[valid]),
            opd=self._last_metrics,
        )


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def kl_penalty(student_logp: torch.Tensor, teacher_logp: torch.Tensor, mode: str) -> torch.Tensor:
    """Per-token KL estimator between student and teacher.

    Ported from ``verl/trainer/ppo/core_algos.py::kl_penalty_forward``.  Inputs
    are the log-probs of the *same* tokens under student and teacher.
    """
    if mode in ("kl", "k1"):
        return student_logp - teacher_logp
    if mode == "abs":
        return (student_logp - teacher_logp).abs()
    if mode in ("mse", "k2"):
        return 0.5 * (student_logp - teacher_logp).square()
    if mode in ("low_var_kl", "k3"):
        kl = teacher_logp - student_logp
        kl = torch.clamp(kl, min=-20, max=20)
        kld = (kl.exp() - kl - 1).contiguous()
        return torch.clamp(kld, min=-10, max=10)
    raise ValueError(f"Unsupported loss_mode: {mode!r}")


def _remove_left_padding_batch(prompt_ids: torch.Tensor, pad_id: int) -> list[list[int]]:
    result = []
    for i in range(prompt_ids.size(0)):
        ids = prompt_ids[i]
        non_pad = (ids != pad_id).nonzero(as_tuple=False)
        if len(non_pad) > 0:
            result.append(ids[non_pad[0].item():].tolist())
        else:
            result.append(ids.tolist())
    return result
