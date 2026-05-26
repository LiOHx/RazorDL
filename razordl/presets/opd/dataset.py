import json
import os
import torch
from torch.utils.data import Dataset
from transformers import AutoTokenizer

from razordl.core.base import logging
from razordl.core.engine.on_policy_single_model.config import Config

logger = logging.getLogger(__name__)

OPD_SYSTEM_PROMPT = (
    "You are a helpful math assistant. Solve the problem step by step. "
    "Output your final numerical answer using the format: \\boxed{answer}."
)


class OPDDataset(Dataset):
    """Dataset for PG On-Policy Distillation.

    Data format: OpenAI chat-completion messages, optionally with an
    ``answer`` field (ignored — OPD does not need ground truth).  The
    student samples its own completions; the teacher supervises those
    completions via per-token log-probabilities.

    .. code-block:: jsonl

        {"messages": [{"role": "user", "content": "What is 2 + 3?"}], "answer": "5"}
        {"messages": [{"role": "user", "content": "Explain backprop."}]}

    A short math-tutor system prompt is prepended so the student's rollouts
    resemble the kind of reasoning the teacher produces.
    """

    def __init__(self, config: Config):
        self.config = config
        self.data_config = config.data_config
        self.max_length = getattr(self.data_config, "max_length", 1024)

        model_path = config.worker_group_config.model_group_config.model_config.model_path
        self.processor = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
        self.processor.padding_side = "left"
        if self.processor.pad_token is None:
            self.processor.pad_token = self.processor.eos_token
        self.dataset_processor = self.processor

        self.raw_data = self._load_data()

        self.data = []
        from tqdm.auto import tqdm
        for item in tqdm(self.raw_data, desc="Tokenizing prompts", unit="samples"):
            self.data.append(self._tokenize_prompt(item))

        logger.info(f"[OPD Dataset] Loaded {len(self.data)} prompts")

    def _load_data(self) -> list:
        data_path = self.data_config.train_data_path
        if not data_path or not os.path.exists(data_path):
            raise FileNotFoundError(f"Data path not found: {data_path}")

        candidates = [
            os.path.join(data_path, "train.jsonl"),
            os.path.join(data_path, "train.json"),
        ]
        data_file = None
        for c in candidates:
            if os.path.exists(c):
                data_file = c
                break
        if data_file is None:
            raise FileNotFoundError(f"No data file found in {data_path}")

        if data_file.endswith(".jsonl"):
            with open(data_file, "r", encoding="utf-8") as f:
                return [json.loads(line) for line in f]
        with open(data_file, "r", encoding="utf-8") as f:
            data = json.load(f)
            return data if isinstance(data, list) else data.get("data", [data])

    def _tokenize_prompt(self, item: dict) -> dict:
        messages = item.get("messages", [])
        # Drop trailing assistant turn — OPD generates its own.
        if messages and messages[-1]["role"] == "assistant":
            messages = messages[:-1]
        # Prepend math system prompt unless the data already supplies one.
        if not messages or messages[0]["role"] != "system":
            messages = [{"role": "system", "content": OPD_SYSTEM_PROMPT}] + messages

        prompt_text = self.processor.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
            enable_thinking=False,
        )
        prompt_ids = self.processor.encode(
            prompt_text,
            add_special_tokens=False,
            truncation=True,
            max_length=self.max_length,
        )
        return {"prompt_ids": prompt_ids}

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return {"prompt_ids": self.data[idx]["prompt_ids"]}


class OPDCollator:
    """Left-pad prompt token ids for vLLM rollout."""

    def __init__(self, processor, sp_size: int = 1):
        self.processor = processor
        self.pad_token_id = processor.pad_token_id
        self.sp_size = sp_size

    def __call__(self, batch):
        max_len = max(len(item["prompt_ids"]) for item in batch)
        if self.sp_size > 1:
            max_len = ((max_len + self.sp_size - 1) // self.sp_size) * self.sp_size

        prompt_ids = []
        attention_mask = []
        for item in batch:
            seq_len = len(item["prompt_ids"])
            pad_len = max_len - seq_len
            prompt_ids.append([self.pad_token_id] * pad_len + item["prompt_ids"])
            attention_mask.append([0] * pad_len + [1] * seq_len)

        return {
            "prompt_ids": torch.tensor(prompt_ids, dtype=torch.long),
            "prompt_attention_mask": torch.tensor(attention_mask, dtype=torch.long),
        }
