from razordl.core.engine.common.flat_config import build_single_model_config_dict
from razordl.core.engine.on_policy_single_model.config import Config as _Config, DataConfig as _DataConfig, dataclass, field


@dataclass
class GRPODataConfig(_DataConfig):
    """Data configuration for GRPO training."""

    train_data_path: str = ""
    max_length: int = 1024
    dataset_processor_path: str = ""
    sp_size: int = 1
    group_size: int = 4
    kl_coef: float = 0.01
    clip_eps: float = 0.2
    max_completion_length: int = 256
    temperature: float = 0.7
    top_p: float = 0.7
    top_k: int = 50


@dataclass
class GRPOConfig(_Config):
    """Top-level configuration for GRPO preset."""

    data_config: GRPODataConfig = field(default_factory=GRPODataConfig)

    @classmethod
    def from_flat_dict(cls, d: dict) -> "GRPOConfig":
        model_default = "Qwen/Qwen3-4B-Instruct"
        max_length = d.get("max_length", 1024)
        data_config = {
            "train_data_path": d.get("data_path", ""),
            "max_length": max_length,
            "sp_size": d.get("sp_size", 1),
            "dataset_processor_path": d.get("dataset_processor_path", ""),
            "group_size": d.get("group_size", 4),
            "kl_coef": d.get("kl_coef", 0.01),
            "clip_eps": d.get("clip_eps", 0.2),
            "max_completion_length": d.get("max_completion_length", 256),
            "temperature": d.get("temperature", 0.7),
            "top_p": d.get("top_p", 0.7),
            "top_k": d.get("top_k", 50),
        }
        config_dict = build_single_model_config_dict(
            d,
            data_config=data_config,
            model_default=model_default,
            processor_max_length=max_length,
            lr_default=5e-5,
            grad_accum_default=2,
            num_epochs_default=3,
            log_steps_default=1,
            task_type_default="CAUSAL_LM",
        )
        return cls.from_dict(config_dict)
