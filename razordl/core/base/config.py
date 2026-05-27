from dataclasses import dataclass, asdict, field, fields, is_dataclass
from typing import Any, Type, TypeVar, Dict
from abc import ABC
from pydantic import Field

T = TypeVar("T", bound="DictSerializable")

@dataclass
class DictSerializable:
    """
    Base class for configuration dataclasses providing automatic
    dict <-> config conversion.
    """
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert the config object to a dictionary."""
        return asdict(self)

    @classmethod
    def from_dict(cls: Type[T], data: Dict[str, Any]) -> T:
        """Create a config object from a dictionary."""
        if data is None:
            return None
        
        if not isinstance(data, dict):
            # Fallback if data is not a dict (e.g. passing the object itself)
            if isinstance(data, cls):
                return data
            # Allow passing None if the parameter is optional but data wasn't None
            return data

        init_kwargs = {}
        for f in fields(cls):
            # Skip if field is not in data (rely on default values)
            if f.name not in data:
                continue
                
            value = data[f.name]
            
            # If value is None, just pass it (unless type checking is strict, but for now allow it)
            if value is None:
                init_kwargs[f.name] = None
                continue
            
            field_type = f.type
            
            # Handle nested dataclasses
            if is_dataclass(field_type) and isinstance(value, dict):
                if issubclass(field_type, DictSerializable):
                    init_kwargs[f.name] = field_type.from_dict(value)
                else:
                    init_kwargs[f.name] = field_type(**value)
            else:
                init_kwargs[f.name] = value
                
        return cls(**init_kwargs)


@dataclass
class BaseDataConfig(DictSerializable):
    train_data_path: str
    dataset_processor_path: str



@dataclass
class BaseProcessorConfig(DictSerializable):
    processor_path: str = None
    max_length: int = 8192
    lazy_tokenize: bool = False  # True = on-the-fly in __getitem__ (saves RAM)




@dataclass
class BaseAdapterConfig(DictSerializable):
    use_adapter: bool = False
    adapter_name: str = "default"
    adapter_path: str = None
    task_type: str = None  # PEFT task type, e.g. "CAUSAL_LM", "FEATURE_EXTRACTION"



@dataclass
class BaseModelConfig(DictSerializable):
    model_path: str
    micro_batch_size_per_gpu: int = 1
    _is_offload_param: bool = False
    _is_offload_optimizer: bool = False
    enable_gradient_checkpointing: bool = False
    sp_size: int = 1
    use_bf16: bool = True       # bfloat16 compute dtype; auto-detected from GPU if not set
    chunked_loss: bool = False  # compute loss in chunks to avoid giant logits tensor
    chunk_size: int = 2048      # tokens per chunk when chunked_loss=True
    adapter_config: BaseAdapterConfig = field(default_factory=BaseAdapterConfig)



@dataclass
class BaseOptimizerConfig(DictSerializable):
    optimizer_path: str = None
    learning_rate: float = 1e-4
    weight_decay: float = 1e-2
    max_grad_norm: float | None = 1.0
    accumulate_grad_steps: int = 1



@dataclass
class BaseSchedulerConfig(DictSerializable):
    scheduler_path: str = None



@dataclass
class BaseModelGroupConfig(DictSerializable):
    model_group_name: str = None
    processor_config: BaseProcessorConfig = field(default_factory=BaseProcessorConfig)
    model_config: BaseModelConfig = field(default_factory=BaseModelConfig)
    optimizer_config: BaseOptimizerConfig = field(default_factory=BaseOptimizerConfig)
    scheduler_config: BaseSchedulerConfig = field(default_factory=BaseSchedulerConfig)



@dataclass
class BaseWorkerGroupConfig(DictSerializable):
    worker_group_name: str = None



@dataclass
class BaseTrainerConfig(DictSerializable):
    ray_kwargs: dict = field(default_factory=dict)
    step_batch_size: int = 1
    data_loader_num_workers: int = 2
    num_epochs: int = 1
    # --- experiment management ---
    outputs_dir: str = "./outputs"          # parent directory for all experiments
    resume_mode: str = "auto"              # "auto" (detect+hash-check) | "manual" (require resume_from)
    resume_from: str = None                # path to experiment dir to resume (manual mode)
    init_from: str = None                  # path to checkpoint dir to fork from (new exp, weights only)
    # --- internal (set by framework, not user-configurable) ---
    output_dir: str = None                 # actual experiment dir, set by main() before Ray starts
    resume_checkpoint_dir: str = None
    log_info_steps: int = 10
    save_model_steps: int = -1
    save_checkpoint_steps: int = -1
    seed: int = 42
    compute_checksums: bool = False


@dataclass
class BaseConfig(ABC, DictSerializable):
    data_config: BaseDataConfig = Field(default_factory=BaseDataConfig)
    trainer_config: BaseTrainerConfig = Field(default_factory=BaseTrainerConfig)
