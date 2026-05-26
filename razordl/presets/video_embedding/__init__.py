from razordl.presets.video_embedding.config import VideoEmbeddingConfig
from razordl.presets.video_embedding.workgroup import VideoEmbeddingWorkGroup
from razordl.presets.video_embedding.dataset import (
    VideoEmbeddingDataset,
    VideoEmbeddingCollator,
)

DATASET = "video_demo"

# Aliases for CLI convention: {preset.upper()}Config / {preset.upper()}WorkGroup etc.
VIDEO_EMBEDDINGConfig = VideoEmbeddingConfig
VIDEO_EMBEDDINGWorkGroup = VideoEmbeddingWorkGroup
VIDEO_EMBEDDINGDataset = VideoEmbeddingDataset
VIDEO_EMBEDDINGCollator = VideoEmbeddingCollator

__all__ = [
    "VideoEmbeddingConfig",
    "VideoEmbeddingWorkGroup",
    "VideoEmbeddingDataset",
    "VideoEmbeddingCollator",
    "VIDEO_EMBEDDINGConfig",
    "VIDEO_EMBEDDINGWorkGroup",
    "VIDEO_EMBEDDINGDataset",
    "VIDEO_EMBEDDINGCollator",
]
