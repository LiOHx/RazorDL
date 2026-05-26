"""vLLM-based rollout engine for GRPO training.

Adapted from ``custom_work/deep_emb_research/vllm_rollout.py``.
Provides fast batched generation with weight synchronisation from the
FSDP2 training model.

.. note::

   vLLM >=0.16 uses the V1 engine which runs the model in a separate
   EngineCore process.  Weight synchronisation from an in-process FSDP2
   model is not yet ported to the V1 API.  Use ``SKIP_VLLM=1`` to fall
   back to HuggingFace ``model.generate()`` for now.
"""

import copy
import os
import tempfile
from dataclasses import dataclass, asdict
import torch

try:
    from vllm import LLM, SamplingParams
    from vllm.lora.request import LoRARequest
    HAS_VLLM = True
except ImportError:
    HAS_VLLM = False


def _load_weights_from_file(worker_self, file_path: str):
    """Load weights from a safetensors file on each vLLM worker.

    Runs inside the EngineCore process via ``LLM.collective_rpc()``.
    """
    from safetensors.torch import load_file
    from vllm.model_executor.model_loader.utils import process_weights_after_loading

    model = worker_self.model_runner.model
    state_dict = load_file(file_path)
    model.load_weights(state_dict.items())
    model_config = worker_self.model_runner.vllm_config.model_config
    device = next(model.parameters()).device
    process_weights_after_loading(model, model_config, device)
    return len(state_dict)


def get_vllm_max_lora_rank(lora_rank: int) -> int:
    valid = [8, 16, 32, 64, 128, 256, 320, 512]
    for r in valid:
        if lora_rank <= r:
            return r
    raise ValueError(f"lora_rank must be <= {valid[-1]}, got {lora_rank}")


@dataclass
class InferenceConfig:
    model: str = ""
    enable_sleep_mode: bool = False
    tensor_parallel_size: int = 1
    distributed_executor_backend: str = "external_launcher"
    dtype: str = "bfloat16"
    enforce_eager: bool = False
    gpu_memory_utilization: float = 0.60
    disable_custom_all_reduce: bool = True
    skip_tokenizer_init: bool = False
    max_model_len: int = 4096
    max_num_seqs: int = 64
    load_format: str = "auto"
    disable_log_stats: bool = True
    max_num_batched_tokens: int = 16384
    enable_chunked_prefill: bool = True
    enable_prefix_caching: bool = True
    trust_remote_code: bool = True
    enable_lora: bool = False
    max_lora_rank: int = 64
    seed: int = 42

    def to_dict(self):
        return asdict(self)


@dataclass
class SamplingParamsConfig:
    n: int = 1
    temperature: float = 1.0
    top_p: float = 0.7
    top_k: int = 50
    max_tokens: int = 256
    detokenize: bool = False
    seed: int | None = None

    def to_dict(self):
        return asdict(self)


class GRPOVLLMRollout:
    """vLLM rollout engine for GRPO training.

    Weight sync from FSDP2 uses ``collective_rpc`` with a safetensors
    file to transfer weights into the EngineCore process (vLLM >=0.16 V1).
    """

    def __init__(
        self,
        inference_config: InferenceConfig,
        sampling_params_config: SamplingParamsConfig,
        use_lora: bool = False,
        lora_rank: int = 8,
    ):
        if not HAS_VLLM:
            raise ImportError("vLLM is not installed. Install with: pip install vllm")

        # Required for collective_rpc with callables (vLLM V1)
        os.environ.setdefault("VLLM_ALLOW_INSECURE_SERIALIZATION", "1")

        self.inference_config = inference_config
        self.use_lora = use_lora
        self.lora_rank = lora_rank

        ic = inference_config.to_dict()
        self.engine = LLM(**ic)
        self.tokenizer = self.engine.get_tokenizer()
        self.tokenizer.padding_side = "left"

        sp = sampling_params_config.to_dict()
        kwargs = {"detokenize": False}
        for k in sp:
            if hasattr(SamplingParams(), k):
                kwargs[k] = sp[k]
        self.sampling_params = SamplingParams(**kwargs)

    @torch.no_grad()
    def update_weights(self, weights_iter, peft_config=None):
        """Sync training weights into vLLM.

        Full-model: saves weights to a temp safetensors file, then uses
        ``collective_rpc`` to load them inside the EngineCore process.
        LoRA: not yet ported to V1.
        """
        if peft_config:
            raise NotImplementedError(
                "LoRA weight sync is not yet implemented for vLLM V1."
            )
        self._update_weights_full(weights_iter)

    def _update_weights_full(self, weights_iter):
        """Full-model weight sync via temp safetensors + collective_rpc."""
        from safetensors.torch import save_file

        state_dict = {}
        for name, tensor in weights_iter:
            state_dict[name] = tensor.cpu() if tensor.is_cuda else tensor

        fd, tmp_path = tempfile.mkstemp(suffix=".safetensors", prefix="vllm_sync_")
        os.close(fd)
        try:
            save_file(state_dict, tmp_path)
            self.engine.collective_rpc(
                _load_weights_from_file,
                kwargs={"file_path": tmp_path},
            )
        finally:
            os.unlink(tmp_path)

    @torch.no_grad()
    def generate(self, prompt_token_ids: list[list[int]], seed: int | None = None) -> dict:
        """Generate with vLLM.  Returns response token ids and masks.

        Args:
            prompt_token_ids: list of prompt token id sequences.
            seed: if provided, overrides the SamplingParams seed for this call
                  (per-step deterministic seeding).
        """
        from vllm.lora.request import LoRARequest

        batch_size = len(prompt_token_ids)
        lora_requests = None
        if self.use_lora:
            active = list(self.engine.llm_engine.list_loras())
            if active:
                lora_requests = [
                    LoRARequest(lora_name="grpo", lora_int_id=active[0], lora_path="/stub")
                ] * batch_size

        sp = self.sampling_params
        if seed is not None:
            sp = copy.copy(sp)
            sp.seed = seed

        from vllm.inputs import TokensPrompt
        prompts = [TokensPrompt(prompt_token_ids=list(ids)) for ids in prompt_token_ids]
        outputs = self.engine.generate(
            prompts=prompts,
            sampling_params=sp,
            lora_request=lora_requests,
            use_tqdm=False,
        )

        response_ids = []
        response_masks = []
        for o in outputs:
            # SamplingParams.n>1 -> multiple sampled completions per prompt
            for completion in o.outputs:
                ids = list(completion.token_ids)
                response_ids.append(ids)
                response_masks.append([1] * len(ids))

        return dict(
            response_token_ids=response_ids,
            response_mask=response_masks,
        )
