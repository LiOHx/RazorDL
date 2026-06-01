import contextlib
import os
import re
import torch

from razordl.core.engine.on_policy_single_model.workgroup import WorkGroup as _WorkGroup
from razordl.core.engine.on_policy_single_model.modelgroup import ModelGroup as _ModelGroup
from razordl.core.engine.on_policy_single_model.config import Config
from razordl.core.base import logging
from razordl.core.base.metrics import DistStats
from razordl.ops.distributed.utils import all_gather_object
from razordl.ops.model.huggingface import build_causal_lm, build_left_padding_tokenizer
from razordl.ops.model.per_token_logp import compute_per_token_log_probs

logger = logging.getLogger(__name__)


class GRPOCausalLMModelGroup(_ModelGroup):
    """Shared HuggingFace CausalLM loader for policy and reference models."""

    def build_processor(self):
        return build_left_padding_tokenizer(
            self.model_group_config.processor_config.processor_path,
            self.model_group_config.model_config.model_path,
            ensure_pad_token=True,
        )

    def build_model(self):
        return build_causal_lm(
            self.model_group_config.model_config.model_path,
            use_bf16=self.model_group_config.model_config.use_bf16,
            local_rank=self.local_rank,
            logger=logger,
        )


# ---------------------------------------------------------------------------
# Policy model group (trainable + vLLM rollout)
# ---------------------------------------------------------------------------

class GRPOPolicyModelGroup(GRPOCausalLMModelGroup):
    """Trainable policy model with optional vLLM rollout engine."""

    def __init__(self, config: Config):
        super().__init__(config)
        self._vllm_engine = None
        self._init_vllm_engine()

    # ---- vLLM engine ----------------------------------------------

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
            max_completion_len = getattr(data_cfg, "max_completion_length", 2048)
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
                max_lora_rank=vllm_max_lora_rank(lora_r) if use_lora else 64,
                seed=self.config.trainer_config.seed,
            )

            data_cfg = self.config.data_config
            sp_cfg = SamplingParamsConfig(
                n=getattr(data_cfg, "group_size", 1),
                temperature=getattr(data_cfg, "temperature", 0.7),
                top_p=getattr(data_cfg, "top_p", 0.7),
                top_k=getattr(data_cfg, "top_k", 50),
                max_tokens=getattr(data_cfg, "max_completion_length", 256),
                detokenize=False,
            )

            self._vllm_engine = GRPOVLLMRollout(inf_cfg, sp_cfg, use_lora, lora_r)

            # Pre-load weights into vLLM
            self._sync_weights_to_vllm()
        except Exception as e:
            logger.warning("vLLM init failed: %s", e)
            self._vllm_engine = None

    def _sync_weights_to_vllm(self):
        """Extract weights from FSDP2 model and load into vLLM."""
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
        """Context that syncs policy weights to vLLM before generate."""
        self._sync_weights_to_vllm()
        try:
            yield self._vllm_engine
        finally:
            pass  # vLLM stays resident; release happens via sleep mode


def vllm_max_lora_rank(lora_rank: int) -> int:
    for r in (8, 16, 32, 64, 128, 256, 320, 512):
        if lora_rank <= r:
            return r
    raise ValueError(f"lora_rank too large: {lora_rank}")


def _sync_lora_weights(model, engine):
    """Extract LoRA params from (potentially FSDP2) model → vLLM."""
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
    """Extract full model weights → vLLM (first step only)."""
    from torch.distributed.tensor import DTensor

    params = model.state_dict()
    weights_iter = (
        (k, v.full_tensor().cpu() if isinstance(v, DTensor) else v.cpu())
        for k, v in params.items()
    )
    engine.update_weights(weights_iter)


def asdict_peft(peft_config) -> dict:
    """Convert a PEFT LoraConfig to a plain dict (compatible with vLLM)."""
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
# Reference model group (frozen, no optimizer)
# ---------------------------------------------------------------------------

class GRPOReferenceModelGroup(GRPOCausalLMModelGroup):
    pass


# ---------------------------------------------------------------------------
# GRPO WorkGroup
# ---------------------------------------------------------------------------

class GRPOWorkGroup(_WorkGroup):
    """GRPO WorkGroup — vLLM rollout + reward + advantage + loss."""

    def __init__(self, config: Config):
        super().__init__(config)
        self.config = config
        self.group_size = getattr(config.data_config, "group_size", 4)
        self.kl_coef = getattr(config.data_config, "kl_coef", 0.01)
        self.temperature = getattr(config.data_config, "temperature", 0.7)
        self.top_p = getattr(config.data_config, "top_p", 0.7)
        self.top_k = getattr(config.data_config, "top_k", 50)
        self.clip_eps = getattr(config.data_config, "clip_eps", 0.2)
        self._last_advantage_info = {}
        self._last_loss_info = {}

        import copy
        policy_config = copy.deepcopy(config)
        policy_config.worker_group_config.model_group_config.model_group_name = "policy_model_group"
        self.policy_model_group = GRPOPolicyModelGroup(policy_config)

        ref_config = copy.deepcopy(config)
        ref_config.worker_group_config.model_group_config.model_group_name = "reference_model_group"
        ref_config.worker_group_config.model_group_config.model_config.is_trainable = False
        ref_config.worker_group_config.model_group_config.model_config.adapter_config.use_adapter = False
        ref_config.worker_group_config.model_group_config.optimizer_config.learning_rate = 0.0
        self.reference_model_group = GRPOReferenceModelGroup(ref_config)

    # ------------------------------------------------------------------
    # Rollout
    # ------------------------------------------------------------------

    def rollout(self, input_dict: dict, step: int) -> dict:
        prompt_ids = input_dict["prompt_ids"]  # [B, S]
        prompt_mask = input_dict["prompt_attention_mask"]
        answers = input_dict["answers"]
        batch_size = prompt_ids.size(0)
        device = prompt_ids.device
        processor = self.policy_model_group.processor

        vllm_seed = self.config.trainer_config.seed + step

        # Remove left padding for vLLM
        prompt_token_ids = _remove_left_padding_batch(prompt_ids, processor.pad_token_id)

        vllm = self.policy_model_group._vllm_engine
        use_vllm = vllm is not None

        if use_vllm:
            # --- vLLM path -------------------------------------------------
            with self.policy_model_group.vllm_rollout_context() as vllm_engine:
                gen_output = vllm_engine.generate(prompt_token_ids, seed=vllm_seed)
            response_token_ids = gen_output["response_token_ids"]
            response_masks_list = gen_output["response_mask"]

            # Build full sequences
            all_input_ids = []
            all_attention_masks = []
            all_response_masks = []
            all_responses = []

            for i in range(batch_size):
                pt_ids = list(prompt_token_ids[i])  # original (no pad)
                ans = answers[i]
                for g in range(self.group_size):
                    idx = i * self.group_size + g
                    rsp_ids = response_token_ids[idx]
                    full = pt_ids + rsp_ids
                    all_input_ids.append(full)
                    all_attention_masks.append([1] * len(full))
                    rm = [0] * len(pt_ids) + response_masks_list[idx]
                    all_response_masks.append(rm)
                    all_responses.append(processor.decode(rsp_ids, skip_special_tokens=True))
        else:
            # --- HF generate fallback --------------------------------------
            prompt_ids_repeated = prompt_ids.repeat_interleave(self.group_size, dim=0)
            prompt_mask_repeated = prompt_mask.repeat_interleave(self.group_size, dim=0)
            max_new = getattr(self.config.data_config, "max_completion_length", 64)

            with torch.no_grad():
                generated = self.policy_model_group.model.generate(
                    input_ids=prompt_ids_repeated,
                    attention_mask=prompt_mask_repeated,
                    max_new_tokens=max_new,
                    do_sample=True,
                    temperature=self.temperature,
                    top_p=self.top_p,
                    top_k=self.top_k,
                    pad_token_id=processor.pad_token_id,
                    eos_token_id=processor.eos_token_id,
                )

            prompt_len = prompt_ids_repeated.size(1)
            total_len = generated.size(1)
            attention_mask = (generated != processor.pad_token_id).long()
            response_mask = torch.zeros_like(attention_mask)
            response_mask[:, prompt_len:] = 1
            response_mask = response_mask * attention_mask

            responses = processor.batch_decode(generated[:, prompt_len:], skip_special_tokens=True)

            # Convert to lists
            all_input_ids = [generated[i].tolist() for i in range(generated.size(0))]
            all_attention_masks = [attention_mask[i].tolist() for i in range(attention_mask.size(0))]
            all_response_masks = [response_mask[i].tolist() for i in range(response_mask.size(0))]
            all_responses = list(responses)

        # Pad to longest
        max_len = max(len(ids) for ids in all_input_ids)
        pad_id = processor.pad_token_id

        for i in range(len(all_input_ids)):
            pad_len = max_len - len(all_input_ids[i])
            all_input_ids[i] = [pad_id] * pad_len + all_input_ids[i]
            all_attention_masks[i] = [0] * pad_len + all_attention_masks[i]
            all_response_masks[i] = [0] * pad_len + all_response_masks[i]

        # Repeat answers to match group_size responses per prompt
        answers_repeated = [a for a in answers for _ in range(self.group_size)]

        return {
            "input_ids": torch.tensor(all_input_ids, dtype=torch.long, device=device),
            "attention_mask": torch.tensor(all_attention_masks, dtype=torch.long, device=device),
            "response_mask": torch.tensor(all_response_masks, dtype=torch.long, device=device),
            "answers": answers_repeated,
            "responses": all_responses,
        }

    # ------------------------------------------------------------------
    # Reward / Advantage / Loss
    # ------------------------------------------------------------------

    def compute_reward(self, rollout_output: dict) -> torch.Tensor:
        responses = rollout_output["responses"]
        answers = rollout_output["answers"]
        rewards = _extract_rewards(responses, answers)

        for i in range(min(3, len(responses))):
            fmt_ok = "</think>" in responses[i]
            logger.info("[GRPO sample] ans=%s fmt=%s reward=%.1f | rsp=%s",
                        answers[i], fmt_ok, rewards[i],
                        repr(responses[i][-80:]))  # tail of response shows answer

        device = rollout_output["input_ids"].device
        return torch.tensor(rewards, dtype=torch.float32, device=device)

    def compute_advantage(self, rewards: torch.Tensor) -> torch.Tensor:
        batch_size = rewards.size(0) // self.group_size
        rewards_per_group = rewards.view(batch_size, self.group_size)
        mean = rewards_per_group.mean(dim=1, keepdim=True)
        raw_std = rewards_per_group.std(dim=1, keepdim=True)
        std = raw_std + 1e-8
        advantages = (rewards_per_group - mean) / std
        nonzero_groups = (raw_std.squeeze(1) > 1e-8).float()
        self._last_advantage_info = {
            "nonzero_group_fraction": nonzero_groups.mean().item(),
            "zero_group_fraction": 1.0 - nonzero_groups.mean().item(),
            "group_reward_mean": mean.mean().item(),
            "group_reward_std_mean": raw_std.mean().item(),
        }
        return advantages.view(-1)

    def _compute_loss_chunk(self, rollout_output: dict, advantages: torch.Tensor):
        """Compute loss for a mini-batch chunk, returning raw sums for global normalization.

        Returns
        -------
        loss_sum : torch.Tensor
            Unnormalized sum of per-token losses (scaled for FSDP2).
        metrics : dict
            Raw sums and counts for global metric aggregation across chunks.
        """
        input_ids = rollout_output["input_ids"]
        attention_mask = rollout_output["attention_mask"]
        response_mask = rollout_output["response_mask"]

        # The PPO/GRPO ratio is policy / old_policy.  Since this preset does
        # one update per rollout batch, the current policy *is* the rollout
        # policy.  Single forward + detach avoids a second forward that would
        # overflow the FSDP2 activation-offload group counter.
        #
        # NOTE: With single-update GRPO, old_log_probs == policy_log_probs
        # (ratio == 1.0), so clip_fraction is always 0.  The clip only becomes
        # meaningful if multi-epoch / mini-batch replay is added later.
        policy_log_probs = compute_per_token_log_probs(
            self.policy_model_group.model, input_ids, attention_mask
        )
        old_log_probs = policy_log_probs.detach()
        ref_log_probs = compute_per_token_log_probs(
            self.reference_model_group.model, input_ids, attention_mask, no_grad=True
        )

        log_ratio = policy_log_probs - old_log_probs
        ratio = torch.exp(log_ratio)
        advantages_expanded = advantages.unsqueeze(1)

        # PPO-style ratio clipping, matching the GRPO objective used by veRL/TRL.
        ratio_clipped = torch.clamp(ratio, 1.0 - self.clip_eps, 1.0 + self.clip_eps)
        pg_loss = -torch.min(
            ratio * advantages_expanded,
            ratio_clipped * advantages_expanded,
        )

        # k1 KL estimator (Schulman 2017, TRL, DeepSeekMath):
        #   exp(ref_logp - policy_logp) - (ref_logp - policy_logp) - 1 >= 0.
        ref_log_ratio = ref_log_probs - policy_log_probs
        kl = torch.exp(ref_log_ratio) - ref_log_ratio - 1
        per_token_loss = pg_loss + self.kl_coef * kl
        response_mask = response_mask[:, 1:]  # align with shifted log_probs
        per_token_loss = per_token_loss * response_mask

        # Raw sums (not normalized) so that multiple chunk backward() calls
        # can be scaled by a *single* global denominator.
        token_count = response_mask.sum()
        loss_sum = per_token_loss.sum()

        # Token-weighted metric sums for global aggregation across chunks.
        clipped = (torch.abs(ratio - 1.0) > self.clip_eps).float()
        masked_tokens = response_mask > 0
        metrics = {
            "pg_loss_sum": (pg_loss * response_mask).sum().item(),
            "kl_sum": (kl * response_mask).sum().item(),
            "clipped_count": clipped[masked_tokens].sum().item() if masked_tokens.any() else 0.0,
            "ratio_sum": ratio[masked_tokens].sum().item() if masked_tokens.any() else 0.0,
            "token_count": token_count.item(),
        }

        # FSDP2 scaling: keep the same logic as the original compute_loss.
        local_bs = input_ids.size(0)
        global_bs = sum(all_gather_object(local_bs))
        loss_sum = loss_sum * local_bs / global_bs

        return loss_sum, metrics

    def compute_loss(self, rollout_output: dict, advantages: torch.Tensor) -> torch.Tensor:
        """Public API: compute mean loss over a (possibly full) batch.

        Backward-compatible wrapper around _compute_loss_chunk.
        """
        loss_sum, metrics = self._compute_loss_chunk(rollout_output, advantages)
        self._last_loss_info = {
            "pg_loss": metrics["pg_loss_sum"] / (metrics["token_count"] + 1e-8),
            "kl": metrics["kl_sum"] / (metrics["token_count"] + 1e-8),
            "clip_fraction": metrics["clipped_count"] / (metrics["token_count"] + 1e-8),
            "mean_ratio": metrics["ratio_sum"] / (metrics["token_count"] + 1e-8),
        }
        return loss_sum / (metrics["token_count"] + 1e-8)

    def _run_update_step(self, input_dict: dict, step: int) -> dict:
        """Run one GRPO update and report GRPO-specific diagnostics.

        After rollout all data is static, so loss computation is chunked into
        mini-batches.  Each chunk does its own forward + backward, but the
        backward uses the *global* token count as denominator so that the total
        gradient is exactly equivalent to a single full-batch backward.
        """
        policy_group = self._get_policy_model_group()
        with policy_group.inference_context():
            rollout_output = self.rollout(input_dict, step)

        rewards = self.compute_reward(rollout_output)
        advantages = self.compute_advantage(rewards)

        ref_group = self._get_reference_model_group()

        # Global token count for unified normalization across all chunks.
        global_response_mask = rollout_output["response_mask"][:, 1:]
        local_token_count = global_response_mask.sum().item()
        global_token_count = sum(all_gather_object(local_token_count))
        global_token_count = max(global_token_count, 1e-8)

        # Chunked loss computation.
        total_samples = rollout_output["input_ids"].size(0)
        mini_batch_size = input_dict["prompt_ids"].size(0)

        total_pg_loss_sum = 0.0
        total_kl_sum = 0.0
        total_clipped = 0.0
        total_ratio_sum = 0.0
        total_tokens = 0.0
        total_loss_sum = 0.0
        num_chunks = 0

        for i in range(0, total_samples, mini_batch_size):
            end = min(i + mini_batch_size, total_samples)
            mini_output = {
                "input_ids": rollout_output["input_ids"][i:end],
                "attention_mask": rollout_output["attention_mask"][i:end],
                "response_mask": rollout_output["response_mask"][i:end],
            }
            mini_advantages = advantages[i:end]

            with policy_group.trainer_context():
                if ref_group is not None:
                    with ref_group.inference_context():
                        loss_sum, metrics = self._compute_loss_chunk(mini_output, mini_advantages)
                else:
                    loss_sum, metrics = self._compute_loss_chunk(mini_output, mini_advantages)
                # Divide by global token count so that N chunk backward() calls
                # produce the same total gradient as one full-batch backward().
                loss = loss_sum / global_token_count
                loss.backward()

            total_loss_sum += loss_sum.item()
            total_pg_loss_sum += metrics["pg_loss_sum"]
            total_kl_sum += metrics["kl_sum"]
            total_clipped += metrics["clipped_count"]
            total_ratio_sum += metrics["ratio_sum"]
            total_tokens += metrics["token_count"]
            num_chunks += 1

        # Token-weighted global metrics across all chunks.
        self._last_loss_info = {
            "pg_loss": total_pg_loss_sum / (total_tokens + 1e-8),
            "kl": total_kl_sum / (total_tokens + 1e-8),
            "clip_fraction": total_clipped / (total_tokens + 1e-8),
            "mean_ratio": total_ratio_sum / (total_tokens + 1e-8),
        }

        avg_loss = total_loss_sum / (num_chunks * global_token_count)

        return dict(
            loss=avg_loss,
            reward=DistStats.from_tensor(rewards),
            advantage=DistStats.from_tensor(advantages),
            grpo={
                **self._last_advantage_info,
                **self._last_loss_info,
            },
        )


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

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


def _extract_rewards(responses: list[str], answers: list[str]) -> list[float]:
    """Standard GRPO reward: format (0.5) + correctness (1.0).

    Three reward levels avoid the binary all-0 or all-1 problem where group
    advantage collapses to zero.
    """
    rewards = []
    for rsp, ans in zip(responses, answers):
        rewards.append(_grpo_reward(rsp, ans))
    return rewards


def _grpo_reward(response: str, ground_truth: str) -> float:
    """Compute GRPO reward for a single response.

    The required output format is::

        <think>reasoning</think>
        \\boxed{answer}

    - 0.0:  wrong format (no ``</think>``) → zero reward, period
    - 0.5:  correct format, wrong answer   → partial credit for format
    - 1.5:  correct format, correct answer → full credit
    """
    format_ok = "</think>" in response
    if not format_ok:
        return 0.0

    # Only look AFTER </think> — the system prompt tells the model to put
    # \\boxed{answer} there.  Answers buried inside <think> get no credit.
    post_think = response.split("</think>")[-1]
    extracted = _extract_answer_gsm8k(post_think)
    correct = _check_answer(extracted, ground_truth)

    return 1.5 if correct else 0.5


def _extract_answer_gsm8k(text: str) -> str:
    """Extract final answer from text.

    Supports multiple answer formats commonly used in math RL:
    ``\\boxed{...}``, ``#### ...``, or bare trailing number.
    """
    # \\boxed{...} format (standard in DeepSeek-R1 / openR1)
    boxed = re.search(r"\\boxed\{([^}]+)\}", text)
    if boxed:
        return boxed.group(1).strip()

    # #### delimiter (VERL / gsm8k convention)
    if "####" in text:
        text = text.split("####")[-1]

    # Fallback: last number in the text
    numbers = re.findall(r"-?\d+\.?\d*", text.replace(",", ""))
    return numbers[-1] if numbers else ""


def _check_answer(predicted: str, ground_truth: str) -> bool:
    """Compare predicted answer with ground truth.

    Uses float equivalence so ``5``, ``5.0``, and ``5.`` are treated equal.
    Strips commas from ground truth (gsm8k has answers like ``1,080``).
    """
    try:
        return abs(float(predicted) - float(ground_truth.replace(",", ""))) < 1e-6
    except ValueError:
        return predicted.strip().lower() == ground_truth.strip().lower()


def _masked_mean(values: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    return (values * mask).sum() / (mask.sum() + 1e-8)
