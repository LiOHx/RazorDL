from razordl.core.base.config import (
    BaseModelGroupConfig,
    BaseProcessorConfig,
    BaseAdapterConfig,
    BaseModelConfig,
    BaseOptimizerConfig,
    BaseSchedulerConfig,
    BaseWorkerGroupConfig,
    BaseTrainerConfig,
    BaseConfig,
    BaseDataConfig,
    dataclass,
    field
)

@dataclass
class DataConfig(BaseDataConfig):
    pass


@dataclass
class ProcessorConfig(BaseProcessorConfig):
    pass


@dataclass
class AdapterConfig(BaseAdapterConfig):
    lora_r: int = 8
    lora_alpha: int = 16
    lora_dropout: float = 0.05
    lora_target_modules: list = field(default_factory=lambda: ["q_proj", "v_proj", "k_proj", "o_proj"])
    modules_to_save: list[str] | None = None
    task_type: str = None


@dataclass
class ModelConfig(BaseModelConfig):
    is_trainable: bool = True  # False = frozen model, no optimizer (e.g. reference in GRPO)
    adapter_config: AdapterConfig = field(default_factory=AdapterConfig)


@dataclass
class OptimizerConfig(BaseOptimizerConfig):
    pass


@dataclass
class SchedulerConfig(BaseSchedulerConfig):
    pass


@dataclass
class ModelGroupConfig(BaseModelGroupConfig):
    model_group_name: str = None
    processor_config: ProcessorConfig = field(default_factory=ProcessorConfig)
    model_config: ModelConfig = field(default_factory=ModelConfig)
    optimizer_config: OptimizerConfig = field(default_factory=OptimizerConfig)
    scheduler_config: SchedulerConfig = field(default_factory=SchedulerConfig)


@dataclass
class WorkerGroupConfig(BaseWorkerGroupConfig):
    worker_group_name: str = None
    model_group_config: ModelGroupConfig = field(default_factory=ModelGroupConfig)


@dataclass
class TrainerConfig(BaseTrainerConfig):
    pass


@dataclass
class Config(BaseConfig):
    data_config: DataConfig = field(default_factory=DataConfig)
    trainer_config: TrainerConfig = field(default_factory=TrainerConfig)
    worker_group_config: WorkerGroupConfig = field(default_factory=WorkerGroupConfig)
