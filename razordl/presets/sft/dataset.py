import hashlib
import json
import math
import os
import statistics
from typing import Any

import torch
from torch.utils.data import Dataset
from transformers import AutoTokenizer

from razordl.core.base import logging

logger = logging.getLogger(__name__)


class SFTDataset(Dataset):
    """Base dataset for supervised fine-tuning.

    Supports both **offline** (default) and **lazy/online** tokenization:

    - ``lazy_tokenize=False`` (default): all samples are tokenized once in
      ``__init__``.  Training loop only reads pre-computed ``input_ids``,
      ``labels``, and ``attention_mask``.  Fast, but uses more RAM.
    - ``lazy_tokenize=True``: raw records are stored in memory and tokenized
      on-the-fly in ``__getitem__``.  Saves RAM at the cost of CPU work
      every epoch.

    Override ``format_item()`` to convert a raw data record into
    OpenAI chat-completion format::

        {
            "messages": [
                {"role": "user", "content": "..."},
                {"role": "assistant", "content": "..."},
            ],
            "tools": [...],  # optional
        }
    """

    IGNORE_INDEX = -100

    def __init__(self, config):
        self.config = config
        self.data_config = config.data_config
        self.max_length = getattr(self.data_config, "max_length", 1024)
        self.sp_size = getattr(self.data_config, "sp_size", 1)
        self.lazy_tokenize = getattr(
            config.worker_group_config.model_group_config.processor_config,
            "lazy_tokenize",
            False,
        )

        model_path = config.worker_group_config.model_group_config.model_config.model_path
        self.processor = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
        self.processor.padding_side = "left"
        if self.processor.pad_token is None:
            self.processor.pad_token = self.processor.eos_token

        self.tokenizer = self.processor
        self.dataset_processor = self.processor

        # Load raw records
        self.raw_data = self.load_data()

        # Offline tokenization (default)
        if not self.lazy_tokenize:
            cache_path = self._cache_path()
            if cache_path is not None and os.path.exists(cache_path):
                logger.info(f"[DATASET] Loading tokenized cache: {cache_path}")
                self.data = torch.load(cache_path, weights_only=False)
                self._log_stats()
                return

            logger.info(
                f"[DATASET] Pre-tokenizing {len(self.raw_data)} samples "
                f"(lazy_tokenize=False)..."
            )
            from tqdm.auto import tqdm

            self.data = []
            for item in tqdm(self.raw_data, desc="Tokenizing", unit="samples"):
                self.data.append(self._tokenize_item(item))
            # Free raw data to save memory
            self.raw_data = None
            self._log_stats()

            # Save cache (atomic per-worker: unique tmp name to avoid race)
            if cache_path is not None:
                os.makedirs(os.path.dirname(cache_path), exist_ok=True)
                if not os.path.exists(cache_path):
                    tmp_path = f"{cache_path}.tmp.{os.getpid()}"
                    torch.save(self.data, tmp_path)
                    try:
                        os.rename(tmp_path, cache_path)
                    except OSError:
                        # Another worker beat us — that's fine, remove our tmp
                        if os.path.exists(tmp_path):
                            os.remove(tmp_path)
                    logger.info(f"[DATASET] Cache saved: {cache_path}")
                else:
                    logger.info(f"[DATASET] Cache already exists (other worker): {cache_path}")
        else:
            self.data = self.raw_data
            logger.info(
                f"[DATASET] Loaded {len(self.data)} raw samples "
                f"(lazy_tokenize=True, tokenize on-the-fly)"
            )

    def load_data(self) -> list:
        """Load raw data. Auto-detects JSON / JSONL files in the data path.

        Override this for custom data sources (parquet, HuggingFace, DB, etc.).
        """
        data_path = self.data_config.train_data_path
        if not data_path:
            raise ValueError("data_path is not set in config.yaml")

        candidates = [
            os.path.join(data_path, "train.jsonl"),
            os.path.join(data_path, "train.json"),
            os.path.join(data_path, "data.jsonl"),
            os.path.join(data_path, "data.json"),
        ]
        data_file = None
        for c in candidates:
            if os.path.exists(c):
                data_file = c
                break

        if data_file is None:
            raise FileNotFoundError(
                f"No data file found in {data_path}. "
                f"Expected one of: {[os.path.basename(c) for c in candidates]}"
            )

        if data_file.endswith(".jsonl"):
            with open(data_file, "r", encoding="utf-8") as f:
                return [json.loads(line) for line in f]
        else:
            with open(data_file, "r", encoding="utf-8") as f:
                data = json.load(f)
                if isinstance(data, list):
                    return data
                return data.get("data", [data])

    def format_item(self, item: Any) -> dict:
        """Convert a raw record to OpenAI chat-completion format.

        Expected return::

            {
                "messages": [
                    {"role": "user", "content": "..."},
                    {"role": "assistant", "content": "..."},
                ],
                "tools": [...],   # optional
            }
        """
        return {
            "messages": item.get("messages", []),
            "tools": item.get("tools", None),
        }

    def _cache_hash(self) -> str:
        """Compute a hash from model path + max_length + data file mtimes.

        Returns None if no data files are found (skip caching).
        """
        data_dir = self.data_config.train_data_path
        if not data_dir or not os.path.isdir(data_dir):
            return None

        # Collect data files (json, jsonl, parquet)
        data_files = sorted(
            f for f in os.listdir(data_dir)
            if f.endswith((".json", ".jsonl", ".parquet"))
        )
        if not data_files:
            return None

        hasher = hashlib.sha256()
        model_path = self.config.worker_group_config.model_group_config.model_config.model_path
        hasher.update(model_path.encode())
        hasher.update(str(self.max_length).encode())
        for fname in data_files:
            fpath = os.path.join(data_dir, fname)
            hasher.update(fname.encode())
            hasher.update(str(os.path.getmtime(fpath)).encode())
        return hasher.hexdigest()[:12]

    def _cache_path(self) -> str | None:
        """Return the cache file path, or None if caching is not applicable."""
        h = self._cache_hash()
        if h is None:
            return None
        return os.path.join(
            self.data_config.train_data_path,
            ".cache",
            f"sft_cache_{h}_{self.max_length}.pt",
        )

    def _tokenize_item(self, item: Any) -> dict:
        """Tokenize a single formatted record."""
        formatted = self.format_item(item)
        messages = formatted["messages"]
        tools = formatted.get("tools", None)

        # Build full text (all messages)
        full_text = self.tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            tools=tools,
            add_generation_prompt=False,
        )
        input_ids = self.tokenizer.encode(full_text, add_special_tokens=False)
        labels = [self.IGNORE_INDEX] * len(input_ids)

        # Mask prompt tokens: for each assistant message, compute its token span.
        # Avoid re-encoding the full conversation for the last assistant message
        # since its end_pos is just len(input_ids).
        for i, msg in enumerate(messages):
            if msg["role"] == "assistant":
                prefix_messages = messages[:i]
                prefix_text = self.tokenizer.apply_chat_template(
                    prefix_messages,
                    tokenize=False,
                    tools=tools,
                    add_generation_prompt=True,
                )
                prefix_ids = self.tokenizer.encode(prefix_text, add_special_tokens=False)
                start_pos = len(prefix_ids)

                if i == len(messages) - 1:
                    end_pos = len(input_ids)
                else:
                    current_messages = messages[:i + 1]
                    current_text = self.tokenizer.apply_chat_template(
                        current_messages,
                        tokenize=False,
                        tools=tools,
                        add_generation_prompt=False,
                    )
                    current_ids = self.tokenizer.encode(current_text, add_special_tokens=False)
                    end_pos = len(current_ids)

                if end_pos <= len(input_ids):
                    labels[start_pos:end_pos] = input_ids[start_pos:end_pos]

        original_length = len(input_ids)
        if len(input_ids) > self.max_length:
            input_ids = input_ids[:self.max_length]
            labels = labels[:self.max_length]

        return {
            "input_ids": input_ids,
            "labels": labels,
            "attention_mask": [1] * len(input_ids),
            "_original_length": original_length,
        }

    def _log_stats(self):
        """Print token-length distribution after offline pre-tokenization."""
        num_samples = len(self.data)
        if num_samples == 0:
            return

        lengths = [len(d["input_ids"]) for d in self.data]
        original_lengths = [d.get("_original_length", l) for d, l in zip(self.data, lengths)]
        total_tokens = sum(lengths)

        # Length stats
        avg_len = statistics.mean(lengths)
        median_len = statistics.median(lengths)
        min_len = min(lengths)
        max_len = max(lengths)
        std_len = statistics.stdev(lengths) if num_samples > 1 else 0.0

        # Percentiles
        def _percentile(data, p):
            s = sorted(data)
            k = (len(s) - 1) * p / 100.0
            f = math.floor(k)
            c = math.ceil(k)
            if f == c:
                return s[int(k)]
            return s[f] * (c - k) + s[c] * (k - f)

        p50 = _percentile(lengths, 50)
        p90 = _percentile(lengths, 90)
        p95 = _percentile(lengths, 95)
        p99 = _percentile(lengths, 99)

        # Truncation
        truncated = sum(1 for o, l in zip(original_lengths, lengths) if o > l)
        truncation_rate = truncated / num_samples

        # Valid / prompt token split
        valid_tokens = [sum(1 for lab in d["labels"] if lab != self.IGNORE_INDEX) for d in self.data]
        prompt_tokens = [sum(1 for lab in d["labels"] if lab == self.IGNORE_INDEX) for d in self.data]
        avg_valid = statistics.mean(valid_tokens)
        avg_prompt = statistics.mean(prompt_tokens)
        valid_ratio = sum(valid_tokens) / total_tokens if total_tokens > 0 else 0.0

        logger.info("=" * 60)
        logger.info("[DATASET STATS] Token-length distribution")
        logger.info(f"  Samples:               {num_samples}")
        logger.info(f"  Total tokens:          {total_tokens}")
        logger.info(f"  Avg / Median length:   {avg_len:.1f} / {median_len:.1f}")
        logger.info(f"  Min / Max length:      {min_len} / {max_len}")
        logger.info(f"  Std dev:               {std_len:.1f}")
        logger.info(f"  P50 / P90 / P95 / P99: {p50:.0f} / {p90:.0f} / {p95:.0f} / {p99:.0f}")
        logger.info(f"  Truncation rate:       {truncated} / {num_samples} ({truncation_rate*100:.1f}%)")
        logger.info(f"  Avg prompt tokens:     {avg_prompt:.1f}  (masked, not learned)")
        logger.info(f"  Avg response tokens:   {avg_valid:.1f}  (learned)")
        logger.info(f"  Learned token ratio:   {valid_ratio*100:.1f}%")
        logger.info("=" * 60)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        if self.lazy_tokenize:
            return self._tokenize_item(self.data[idx])
        # Strip internal key before returning
        item = dict(self.data[idx])
        item.pop("_original_length", None)
        return item


class SFTCollator:
    """Collator that pads sequences, with optional SP (sequence-parallel) alignment."""

    def __init__(self, tokenizer, sp_size: int = 1):
        self.tokenizer = tokenizer
        self.pad_token_id = tokenizer.pad_token_id
        self.sp_size = sp_size

    def __call__(self, features):
        max_len = max(len(f["input_ids"]) for f in features)

        # Align to sp_size multiple when SP is enabled
        if self.sp_size > 1:
            max_len = ((max_len + self.sp_size - 1) // self.sp_size) * self.sp_size

        batch = {"input_ids": [], "attention_mask": [], "labels": []}
        for f in features:
            seq_len = len(f["input_ids"])
            pad_len = max_len - seq_len
            batch["input_ids"].append(
                [self.pad_token_id] * pad_len + f["input_ids"]
            )
            batch["attention_mask"].append(
                [0] * pad_len + f["attention_mask"]
            )
            batch["labels"].append(
                [-100] * pad_len + f["labels"]
            )

        batch = {k: torch.tensor(v, dtype=torch.long) for k, v in batch.items()}
        return batch
