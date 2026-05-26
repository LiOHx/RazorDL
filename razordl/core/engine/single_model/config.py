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
    # train_data_path: str = None
    pass



@dataclass
class ProcessorConfig(BaseProcessorConfig):
    # processor_path: str = None
    pass



@dataclass
class AdapterConfig(BaseAdapterConfig):
    lora_r: int = 8
    lora_alpha: int = 16
    lora_dropout: float = 0.05
    lora_target_modules: list = field(default_factory=lambda: ["q_proj", "v_proj", "k_proj", "o_proj"])
    modules_to_save: list[str] | None = None
    task_type: str = None  # e.g. "CAUSAL_LM", "FEATURE_EXTRACTION".  None = PEFT auto-detect



@dataclass
class ModelConfig(BaseModelConfig):
    # model_path: str
    # micro_batch_size_per_gpu: int = 1
    # _is_offload_param: bool = False
    # _is_offload_optimizer: bool = False
    # enable_gradient_checkpointing: bool = False
    adapter_config: AdapterConfig = field(default_factory=AdapterConfig)



@dataclass
class OptimizerConfig(BaseOptimizerConfig):
    # optimizer_path: str = None
    # learning_rate: float = 1e-4
    # weight_decay: float = 1e-2
    # max_grad_norm: float | None = 1.0
    # accumulate_grad_steps: int = 1
    pass



@dataclass
class SchedulerConfig(BaseSchedulerConfig):
    # scheduler_path: str = None
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
    # ray_kwargs: dict = field(default_factory=dict)
    # step_batch_size: int = 1
    # data_loader_num_workers: int = 2
    # num_epochs: int = 1
    # output_dir: str = None
    # resume_checkpoint_dir: str = None
    # log_info_steps: int = 10
    # save_model_steps: int = -1
    # save_checkpoint_steps: int = -1  # 保存checkpoint的步数间隔，-1表示不保存checkpoint    
    # auto_resume: bool = True  # 是否自动从最新的checkpoint恢复训练（防止中断）
    # seed: int = 42



@dataclass
class Config(BaseConfig):
    data_config: DataConfig = field(default_factory=DataConfig)
    trainer_config: TrainerConfig = field(default_factory=TrainerConfig)
    worker_group_config: WorkerGroupConfig = field(default_factory=WorkerGroupConfig)

