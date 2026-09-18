"""NEW preset config — [改] 把 NEW 换成你的 preset 名。

Shape copied from presets/dft/config.py: own DataConfig with the preset's
extra keys, and from_flat_dict delegating everything shared to
build_single_model_config_dict (never copy the trainer/model/optimizer
mapping here — see presets/CLAUDE.md).
"""

from razordl.core.engine.common.flat_config import build_single_model_config_dict
from razordl.core.engine.single_model.config import Config as _Config, DataConfig as _DataConfig, dataclass, field


@dataclass
class NEWDataConfig(_DataConfig):
    """Data configuration for NEW training."""

    train_data_path: str = ""
    max_length: int = 1024
    dataset_processor_path: str = ""
    sp_size: int = 1
    new_param: float = 0.0  # [改] 你的 preset 特有参数（flat key 同名）


@dataclass
class NEWConfig(_Config):
    """Top-level configuration for NEW preset."""

    data_config: NEWDataConfig = field(default_factory=NEWDataConfig)

    @classmethod
    def from_flat_dict(cls, d: dict) -> "NEWConfig":
        model_path = d.get("model", "Qwen/Qwen3-4B-Instruct")
        max_length = d.get("max_length", 1024)
        data_config = {
            "train_data_path": d.get("data_path", ""),
            "max_length": max_length,
            "sp_size": d.get("sp_size", 1),
            "dataset_processor_path": d.get("dataset_processor_path", ""),
            "new_param": d.get("new_param", 0.0),  # [改]
        }
        config_dict = build_single_model_config_dict(
            d,
            data_config=data_config,
            model_default=model_path,
            processor_max_length=max_length,
            lr_default=5e-5,
            grad_accum_default=2,
            num_epochs_default=3,
            log_steps_default=10,
            task_type_default=None,
        )
        return cls.from_dict(config_dict)
