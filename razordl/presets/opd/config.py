from razordl.core.engine.common.flat_config import build_single_model_config_dict
from razordl.core.engine.on_policy_single_model.config import (
    Config as _Config,
    DataConfig as _DataConfig,
    dataclass,
    field,
)


@dataclass
class OPDDataConfig(_DataConfig):
    """Data configuration for PG On-Policy Distillation.

    The ``teacher_model`` / ``teacher_processor_path`` keys live on the data
    config rather than introducing a second ``model_group_config`` slot.
    ``OPDWorkGroup.__init__`` deep-copies the single model_group_config and
    overrides the teacher fields, reusing the engine's existing
    ``reference_model_group`` slot.
    """

    train_data_path: str = ""
    max_length: int = 1024
    dataset_processor_path: str = ""
    sp_size: int = 1

    # PG OPD knobs
    loss_mode: str = "k1"           # k1 / k2 / k3 / abs / mse / low_var_kl
    loss_max_clamp: float = 10.0
    log_prob_min_clamp: float = -20.0
    clip_eps: float = 0.2
    max_completion_length: int = 2048
    temperature: float = 0.7
    top_p: float = 0.9
    top_k: int = 50

    # Teacher placement — REQUIRED at runtime
    teacher_model: str = ""
    teacher_processor_path: str = ""  # empty = use student tokenizer


@dataclass
class OPDConfig(_Config):
    """Top-level configuration for OPD preset."""

    data_config: OPDDataConfig = field(default_factory=OPDDataConfig)

    @classmethod
    def from_flat_dict(cls, d: dict) -> "OPDConfig":
        model_default = "Qwen/Qwen3-0.6B"
        max_length = d.get("max_length", 1024)
        data_config = {
            "train_data_path": d.get("data_path", ""),
            "max_length": max_length,
            "sp_size": d.get("sp_size", 1),
            "dataset_processor_path": d.get("dataset_processor_path", ""),
            "loss_mode": d.get("loss_mode", "k1"),
            "loss_max_clamp": d.get("loss_max_clamp", 10.0),
            "log_prob_min_clamp": d.get("log_prob_min_clamp", -20.0),
            "clip_eps": d.get("clip_eps", 0.2),
            "max_completion_length": d.get("max_completion_length", 2048),
            "temperature": d.get("temperature", 0.7),
            "top_p": d.get("top_p", 0.9),
            "top_k": d.get("top_k", 50),
            "teacher_model": d.get("teacher_model", ""),
            "teacher_processor_path": d.get("teacher_processor_path", ""),
        }
        config_dict = build_single_model_config_dict(
            d,
            data_config=data_config,
            model_default=model_default,
            processor_max_length=max_length,
            lr_default=5e-6,
            grad_accum_default=2,
            num_epochs_default=3,
            log_steps_default=1,
            task_type_default="CAUSAL_LM",
        )
        return cls.from_dict(config_dict)
