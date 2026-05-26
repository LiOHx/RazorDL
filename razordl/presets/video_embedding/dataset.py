import json
import math
import os
import random

import numpy as np
import torch
from torch.utils.data import Dataset
from transformers import AutoProcessor

from razordl.core.base import logging
from razordl.ops.multimodal.video import video_path_to_base64_image_lst
from razordl.presets.video_embedding.config import VideoEmbeddingConfig

logger = logging.getLogger(__name__)


def _get_query_messages(system_prompt: str, query: str) -> list[dict]:
    return [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": [{"type": "text", "text": query}]},
    ]


def _get_video_messages(
    system_prompt: str,
    video_file_path: str,
    nframes: int = None,
    sample_fps: float = None,
    load_video_base64: bool = False,
) -> list[dict]:
    if load_video_base64:
        with open(video_file_path, "r") as f:
            video_base64_lst = json.load(f)
    else:
        video_base64_lst = video_path_to_base64_image_lst(
            video_file_path, nframes=nframes, sample_fps=sample_fps
        )
    return [
        {"role": "system", "content": [{"type": "text", "text": system_prompt}]},
        {
            "role": "user",
            "content": [
                {"type": "image", "image": f"data:image/png;base64,{b64}"}
                for b64 in video_base64_lst
            ],
        },
    ]


def _get_video_input_dict(
    processor: AutoProcessor,
    system_prompt: str,
    video_path_lst: list[str],
    nframes: int = 8,
    sample_fps: float = None,
    max_length: int = 32768,
    load_video_base64: bool = False,
) -> dict:
    from qwen_vl_utils import process_vision_info

    video_messages_lst = []
    for video_path in video_path_lst:
        video_messages_lst.append(
            _get_video_messages(
                system_prompt, video_path, nframes=nframes,
                sample_fps=sample_fps, load_video_base64=load_video_base64,
            )
        )

    input_texts, processed_images, processed_videos = [], [], []
    for messages in video_messages_lst:
        text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=False)
        image_inputs, video_inputs = process_vision_info(messages)
        input_texts.append(text)
        processed_images.append(image_inputs)
        processed_videos.append(video_inputs)

    # multi_image mode: images=frames, videos=None
    # video mode: images=None, videos=frames
    has_images = any(p is not None for p in processed_images)
    has_videos = any(p is not None for p in processed_videos)

    kwargs = {
        "text": input_texts,
        "padding": True,
        "truncation": True,
        "max_length": max_length,
        "return_tensors": "pt",
    }
    if has_images:
        kwargs["images"] = [p for p in processed_images if p is not None]
    if has_videos:
        kwargs["videos"] = [p for p in processed_videos if p is not None]

    return processor(**kwargs)


def _get_query_input_dict(
    processor: AutoProcessor,
    system_prompt: str,
    queries: list[str],
    max_length: int = 1024,
) -> dict:
    messages_lst = [_get_query_messages(system_prompt, q) for q in queries]
    input_texts = []
    for messages in messages_lst:
        text = processor.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=False
        )
        input_texts.append(text)
    return processor(
        text=input_texts,
        padding=True,
        truncation=True,
        max_length=max_length,
        return_tensors="pt",
    )


class VideoEmbeddingDataset(Dataset):
    """Dataset for video-text contrastive learning."""

    def __init__(self, config: VideoEmbeddingConfig):
        import datasets

        self.config = config
        self.data_config = config.data_config
        self.nframes = getattr(self.data_config, "nframes", 8)
        self.sample_fps = None

        # Dataset paths
        data_path = self.data_config.train_data_path
        self.video_base_path = os.path.join(data_path, "video")
        # Check for new directory structure: video_base64/nframes_{n}/
        base64_dir = os.path.join(data_path, "video_base64", f"nframes_{self.nframes}")
        legacy_base64_dir = os.path.join(data_path, "video_base64")
        if os.path.exists(base64_dir):
            self.video_base64_path = base64_dir
            self.load_video_base64 = True
        elif os.path.exists(legacy_base64_dir) and any(f.endswith(".json") for f in os.listdir(legacy_base64_dir)):
            self.video_base64_path = legacy_base64_dir
            self.load_video_base64 = True
        else:
            self.video_base64_path = None
            self.load_video_base64 = False
            logger.info("[DATASET] video_base64 not found, loading from video files")

        train_file = os.path.join(data_path, "train.jsonl")
        datasets.builder.has_sufficient_disk_space = lambda needed_bytes, directory=".": True
        self.dataset = datasets.load_dataset("json", data_files=train_file, split="train")
        self.total_len = len(self.dataset)

        # Processor
        model_path = config.worker_group_config.model_group_config.model_config.model_path
        self.dataset_processor = AutoProcessor.from_pretrained(
            model_path, trust_remote_code=True
        )
        self.dataset_processor.tokenizer.padding_side = "left"

        assert self.data_config.document_group_size >= 2

    def __len__(self):
        return self.total_len

    def __getitem__(self, item_id):
        item = self.dataset[item_id]
        query = item["query"]
        pos = item["pos_video_ids"]
        neg = item["neg_video_ids"]

        group_size = self.data_config.document_group_size
        paragraphs = pos + neg
        is_supporting = [True] * len(pos) + [False] * len(neg)

        # Sample 1 positive + 1 negative + (group_size-2) others
        sample_ids = [random.choice(range(len(pos))), len(pos) + random.choice(range(len(neg)))]
        if len(paragraphs) < group_size - 2:
            num = math.ceil(group_size / len(paragraphs))
            other_ids = random.sample(list(range(len(paragraphs))) * num, group_size - 2)
        else:
            other_candidates = [i for i in range(len(paragraphs)) if i not in sample_ids]
            other_ids = random.sample(other_candidates, group_size - 2)
        sample_ids.extend(other_ids)
        sample_ids = sorted(sample_ids)

        sample_video_ids = [paragraphs[i] for i in sample_ids]
        labels = torch.tensor([[int(is_supporting[i]) for i in sample_ids]])

        # Query input
        query_input_dict = _get_query_input_dict(
            self.dataset_processor,
            self.data_config.query_system_prompt,
            [query],
            max_length=self.data_config.query_max_length,
        )

        # Video input
        if self.load_video_base64:
            # New structure: video_base64/nframes_8/000000.json
            # Legacy structure: video_base64/000000_8_None.json
            if os.path.basename(self.video_base64_path).startswith("nframes_"):
                video_paths = [
                    os.path.join(self.video_base64_path, vid.split(".")[0] + ".json")
                    for vid in sample_video_ids
                ]
            else:
                video_paths = [
                    os.path.join(
                        self.video_base64_path,
                        vid.split(".")[0] + f"_{self.nframes}_{self.sample_fps}.json",
                    )
                    for vid in sample_video_ids
                ]
        else:
            video_paths = [os.path.join(self.video_base_path, vid) for vid in sample_video_ids]

        video_input_dict = _get_video_input_dict(
            self.dataset_processor,
            self.data_config.video_system_prompt,
            video_paths,
            nframes=self.nframes,
            sample_fps=self.sample_fps,
            max_length=self.data_config.video_max_length,
            load_video_base64=self.load_video_base64,
        )

        return {
            "query_input_dict": query_input_dict,
            "video_input_dict": video_input_dict,
            "labels": labels,
            "sample_video_ids": np.array([sample_video_ids]),
        }


class VideoEmbeddingCollator:
    """Collator that pads query and video inputs to max length in batch."""

    def __init__(self, processor: AutoProcessor, sp_size: int = 1):
        self.processor = processor
        self.pad_token_id = processor.tokenizer.pad_token_id
        self.sp_size = sp_size

    def __call__(self, features: list[dict]) -> dict:
        max_query_len = max(f["query_input_dict"]["input_ids"].size(1) for f in features)
        max_video_len = max(f["video_input_dict"]["input_ids"].size(1) for f in features)

        # Pad each feature
        for feature in features:
            self._pad_dict(feature["query_input_dict"], max_query_len)
            self._pad_dict(feature["video_input_dict"], max_video_len)

        # Concatenate into batch
        batch = {
            "query_input_dict": self._concat_dicts([f["query_input_dict"] for f in features]),
            "video_input_dict": self._concat_dicts([f["video_input_dict"] for f in features]),
            "labels": torch.concat([f["labels"] for f in features], dim=0),
            "sample_video_ids": np.concatenate([f["sample_video_ids"] for f in features], axis=0),
        }

        # Handle vision tensors
        self._maybe_concat_vision(batch["video_input_dict"], features, "pixel_values", "image_grid_thw")
        self._maybe_concat_vision(batch["video_input_dict"], features, "pixel_values_videos", "video_grid_thw")

        return batch

    def _pad_dict(self, d: dict, max_len: int):
        """Pad input_ids and attention_mask to max_len (left padding)."""
        cur_len = d["input_ids"].size(1)
        if cur_len >= max_len:
            return
        pad_len = max_len - cur_len
        d["input_ids"] = torch.concat([
            torch.full((d["input_ids"].size(0), pad_len), self.pad_token_id, dtype=d["input_ids"].dtype),
            d["input_ids"],
        ], dim=1)
        d["attention_mask"] = torch.concat([
            torch.zeros((d["attention_mask"].size(0), pad_len), dtype=d["attention_mask"].dtype),
            d["attention_mask"],
        ], dim=1)

    def _concat_dicts(self, dicts: list[dict]) -> dict:
        """Concatenate a list of single-sample dicts into a batch dict."""
        keys = [k for k in dicts[0].keys() if k not in (
            "pixel_values", "image_grid_thw", "pixel_values_videos", "video_grid_thw"
        )]
        return {k: torch.concat([d[k] for d in dicts], dim=0) for k in keys}

    def _maybe_concat_vision(self, batch_dict: dict, features: list[dict], pixel_key: str, grid_key: str):
        """Concatenate vision tensors if present."""
        first = features[0]["video_input_dict"]
        if pixel_key in first and first[pixel_key] is not None:
            batch_dict[pixel_key] = torch.concat([f["video_input_dict"][pixel_key] for f in features], dim=0)
        if grid_key in first and first[grid_key] is not None:
            batch_dict[grid_key] = torch.concat([f["video_input_dict"][grid_key] for f in features], dim=0)
