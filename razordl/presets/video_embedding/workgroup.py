import os

import torch
import torch.nn as nn
from tensordict.tensordict import TensorDict

from razordl.core.base import logging
from razordl.core.engine.single_model.workgroup import (
    ModelGroup as _ModelGroup,
    WorkGroup as _WorkGroup,
)
from razordl.ops.model.huggingface import enforce_model_profile
from razordl.ops.multimodal import split_multi_modal_input_dict
from razordl.presets.video_embedding.config import VideoEmbeddingConfig

logger = logging.getLogger(__name__)


class VideoEmbeddingModelGroup(_ModelGroup):
    """ModelGroup for multimodal video-text embedding models."""

    def build_processor(self):
        from transformers import AutoProcessor

        processor_path = self.model_group_config.processor_config.processor_path
        if processor_path is None:
            processor_path = self.model_group_config.model_config.model_path

        processor = AutoProcessor.from_pretrained(
            processor_path,
            min_pixels=256 * 28 * 28,
            max_pixels=1280 * 28 * 28,
            trust_remote_code=True,
        )
        processor.tokenizer.padding_side = "left"
        return processor

    def build_model(self):
        from transformers import AutoModelForImageTextToText

        use_bf16 = torch.cuda.is_available() and torch.cuda.is_bf16_supported()
        model_path = self.model_group_config.model_config.model_path
        cfg = enforce_model_profile(model_path)
        model = AutoModelForImageTextToText.from_pretrained(
            model_path,
            config=cfg,
            torch_dtype=torch.bfloat16 if use_bf16 else torch.float16,
            attn_implementation="flash_attention_2",
            trust_remote_code=True,
        ).to(self.device)
        model.config.use_cache = False
        return model


class UnifiedContrastiveLoss(nn.Module):
    """Contrastive loss for query-video matching."""

    def __init__(self, gamma=1, margin=0):
        super().__init__()
        self.gamma = gamma
        self.margin = margin

    def forward(self, x: torch.Tensor, labels: torch.Tensor):
        neg_loss = torch.exp(self.gamma * (x + self.margin) - labels * 1e6).sum(-1)
        pos_loss = torch.exp(self.gamma * (-x) - (1 - labels) * 1e6).sum(-1)
        loss = torch.log(1 + neg_loss * pos_loss)
        return loss.mean()


def _pooling(last_hidden_state):
    reps = last_hidden_state[:, -1, :]
    reps = torch.nn.functional.normalize(reps, p=2, dim=-1)
    return reps


class VideoEmbeddingWorkGroup(_WorkGroup):
    """WorkGroup for video-text contrastive learning.

    update_step computes the contrastive loss and calls backward().
    The engine's WorkGroup wrapper handles optimizer.step() / zero_grad()
    and gradient clipping via _post_update_step.
    """

    def __init__(self, config: VideoEmbeddingConfig):
        super().__init__(config)
        self.config = config
        self.model_group = VideoEmbeddingModelGroup(config)
        self.criterion = UnifiedContrastiveLoss()

    def _get_embeddings(self, input_dict: dict, batch_size=1, display_progress=False):
        """Run forward to get pooled embeddings."""
        import tqdm

        model = self.model_group.model
        processor = self.model_group.processor
        input_dict_lst = split_multi_modal_input_dict(input_dict, processor, batch_size=batch_size)
        batch_embeddings_lst = []
        loader = tqdm.tqdm(input_dict_lst) if display_progress else input_dict_lst
        for inp in loader:
            hidden_states = model(**inp, return_dict=True, output_hidden_states=True)
            hidden_states = hidden_states.hidden_states[-1]
            batch_embeddings = _pooling(hidden_states)
            batch_embeddings_lst.append(batch_embeddings)
        embeddings = torch.cat(batch_embeddings_lst, dim=0)
        return embeddings

    def _pre_process_batch(self, input_dict: dict, batch_size: int = 1) -> dict:
        """Pre-compute query and video embeddings (no_grad)."""
        with torch.no_grad():
            self.model_group.model.eval()
            query_emb = self._get_embeddings(
                input_dict["query_input_dict"], batch_size=batch_size, display_progress=False
            ).detach()
            video_emb = self._get_embeddings(
                input_dict["video_input_dict"], batch_size=batch_size, display_progress=False
            ).detach()
        return {"query_embeddings": query_emb, "video_embeddings": video_emb}

    def _compute_loss(self, input_dict: dict, query_emb: torch.Tensor, video_emb: torch.Tensor):
        """Compute contrastive loss with micro-batching."""
        processor = self.model_group.processor
        model = self.model_group.model
        temperature = self.config.data_config.temperature
        micro_batch = self.model_group.model_group_config.model_config.micro_batch_size_per_gpu
        labels = input_dict["labels"]
        query_input = input_dict["query_input_dict"]
        video_input = input_dict["video_input_dict"]

        model.train()
        similarity_lst = []

        # Query-side micro-batch
        query_list = split_multi_modal_input_dict(query_input, processor, batch_size=micro_batch)
        for i, q_inp in enumerate(query_list):
            micro_query = self._get_embeddings(q_inp, batch_size=micro_batch, display_progress=False)
            tmp_query = torch.clone(query_emb).detach()
            tmp_query[i * micro_batch : (i + 1) * micro_batch] = micro_query
            tmp_video = video_emb.view(query_emb.size(0), -1, video_emb.size(-1))
            similarity = torch.bmm(
                tmp_query.unsqueeze(1), tmp_video.transpose(1, 2)
            ).squeeze(1)
            loss = self.criterion(similarity / temperature, labels)
            loss.backward()
            similarity_lst.append(similarity.detach().cpu())

        # Video-side micro-batch
        video_list = split_multi_modal_input_dict(video_input, processor, batch_size=micro_batch)
        for i, v_inp in enumerate(video_list):
            micro_video = self._get_embeddings(v_inp, batch_size=micro_batch, display_progress=False)
            tmp_video = torch.clone(video_emb).detach()
            tmp_video[i * micro_batch : (i + 1) * micro_batch] = micro_video
            tmp_video = tmp_video.view(query_emb.size(0), -1, video_emb.size(-1))
            similarity = torch.bmm(
                query_emb.unsqueeze(1), tmp_video.transpose(1, 2)
            ).squeeze(1)
            loss = self.criterion(similarity / temperature, labels)
            loss.backward()
            similarity_lst.append(similarity.detach().cpu())
            del micro_video, tmp_video

        batch_loss = loss.item()
        del similarity_lst

        # Compute mAP
        sorted_score, sorted_indices = torch.sort(similarity, dim=-1, descending=True)
        labels_dev = labels.to(sorted_indices.device)
        sorted_labels = torch.gather(labels_dev, dim=-1, index=sorted_indices)
        sorted_labels = sorted_labels.to("cpu").tolist()

        def _map(labels: list[int]) -> float:
            ap_lst = []
            for j, lab in enumerate(labels):
                if lab == 1:
                    rank = j + 1
                    ap = sum(labels[:rank]) / rank
                    ap_lst.append(ap)
            return sum(ap_lst) / len(ap_lst) if ap_lst else 0.0

        mAP = sum(_map(sl) for sl in sorted_labels) / len(sorted_labels)

        return {"loss": batch_loss, "mAP": mAP}

    def update_step(self, input_dict: TensorDict, step: int) -> dict:
        """Compute contrastive loss.  Engine wrapper handles optimizer step."""
        batch_data = self._pre_process_batch(input_dict)
        return self._compute_loss(
            input_dict,
            batch_data["query_embeddings"],
            batch_data["video_embeddings"],
        )
