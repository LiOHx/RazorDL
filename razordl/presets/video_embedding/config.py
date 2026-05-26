from razordl.core.engine.common.flat_config import build_single_model_config_dict
from razordl.core.engine.single_model.config import Config as _Config, DataConfig as _DataConfig, dataclass, field


@dataclass
class VideoEmbeddingDataConfig(_DataConfig):
    """Data configuration for video-text contrastive learning."""

    train_data_path: str = ""
    document_group_size: int = 8
    query_system_prompt: str = ""
    video_system_prompt: str = ""
    query_max_length: int = 1024
    video_max_length: int = 32768
    dataset_processor_path: str = ""
    temperature: float = 0.05
    nframes: int = 8
    sp_size: int = 1


@dataclass
class VideoEmbeddingConfig(_Config):
    """Top-level configuration for video-embedding preset."""

    data_config: VideoEmbeddingDataConfig = field(default_factory=VideoEmbeddingDataConfig)

    @classmethod
    def from_flat_dict(cls, d: dict) -> "VideoEmbeddingConfig":
        model_default = "OpenSearch-AI/Ops-MM-embedding-v1-2B"
        query_max_length = d.get("query_max_length", 1024)
        video_max_length = d.get("video_max_length", 32768)
        data_config = {
            "train_data_path": d.get("data_path", ""),
            "document_group_size": d.get("document_group_size", 8),
            "query_system_prompt": d.get("query_system_prompt", ""),
            "video_system_prompt": d.get("video_system_prompt", ""),
            "query_max_length": query_max_length,
            "video_max_length": video_max_length,
            "temperature": d.get("temperature", 0.05),
            "nframes": d.get("nframes", 8),
            "sp_size": d.get("sp_size", 1),
            "dataset_processor_path": d.get("dataset_processor_path", ""),
        }
        config_dict = build_single_model_config_dict(
            d,
            data_config=data_config,
            model_default=model_default,
            processor_max_length=video_max_length,
            lr_default=1e-4,
            grad_accum_default=8,
            num_epochs_default=1,
            log_steps_default=10,
            task_type_default="FEATURE_EXTRACTION",
        )
        return cls.from_dict(config_dict)
