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

# All shared fields (LoRA, precision, parallel_backend, is_trainable, experiment
# management, ...) live on the Base* dataclasses in core/base/config.py.  These
# subclasses only re-type the nested defaults so from_dict builds the variant
# tree; presets subclass them to add task-specific keys.


@dataclass
class DataConfig(BaseDataConfig):
    pass


@dataclass
class ProcessorConfig(BaseProcessorConfig):
    pass


@dataclass
class AdapterConfig(BaseAdapterConfig):
    pass


@dataclass
class ModelConfig(BaseModelConfig):
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
