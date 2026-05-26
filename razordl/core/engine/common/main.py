import os
from functools import partial

import ray
import torch
from omegaconf import OmegaConf
from ray.train.torch import TorchTrainer
from torch.utils.data import Dataset

try:
    from ray.train import ScalingConfig
except ImportError:
    from ray.air.config import ScalingConfig

from razordl.core.base import logging
from razordl.core.base.workgroup import BaseWorkGroup
from razordl.core.engine.common.trainer import EngineTrainer
from razordl.ops.hardware.device import check_device_compatibility


def train_loop_per_worker(
    config,
    workgroup_class,
    train_dataset_class,
    train_collator_class,
    trainer_class=EngineTrainer,
    worker_setup=None,
):
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        handlers=[logging.StreamHandler()],
        force=True,
    )
    if worker_setup is not None:
        worker_setup(config)

    trainer = trainer_class(config, workgroup_class, train_dataset_class, train_collator_class)
    trainer.run_training_loop()


def main(
    config,
    workgroup_class: BaseWorkGroup,
    train_dataset_class: Dataset,
    train_collator_class: type,
    *,
    trainer_class=EngineTrainer,
    worker_setup=None,
):
    check_device_compatibility()

    config.trainer_config.output_dir = os.path.abspath(config.trainer_config.output_dir)
    config.data_config.train_data_path = os.path.abspath(config.data_config.train_data_path)

    logger = logging.getLogger(__name__)
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        handlers=[logging.StreamHandler()],
    )

    logger.info("*" * 100)
    logger.info("Config:")
    logger.info(config)
    logger.info("*" * 100)

    if ray.is_initialized():
        ray.shutdown()
    if not ray.is_initialized():
        default_runtime_env = {
            "env_vars": {
                "TOKENIZERS_PARALLELISM": "true",
                "NCCL_DEBUG": "WARN",
                "VLLM_LOGGING_LEVEL": "WARN",
                "PYTORCH_CUDA_ALLOC_CONF": "",
            }
        }
        ray_init_kwargs = config.trainer_config.ray_kwargs.get("ray_init", {})
        runtime_env_kwargs = ray_init_kwargs.get("runtime_env", {})
        runtime_env = OmegaConf.merge(default_runtime_env, runtime_env_kwargs)
        ray_init_kwargs = OmegaConf.create({**ray_init_kwargs, "runtime_env": runtime_env})
        print(f"ray init kwargs: {ray_init_kwargs}")
        ray.init(**OmegaConf.to_container(ray_init_kwargs))

    try:
        num_gpus = torch.cuda.device_count()
        print(f"num_gpus: {num_gpus}")
        train_loop = partial(
            train_loop_per_worker,
            workgroup_class=workgroup_class,
            train_dataset_class=train_dataset_class,
            train_collator_class=train_collator_class,
            trainer_class=trainer_class,
            worker_setup=worker_setup,
        )
        trainer = TorchTrainer(
            train_loop_per_worker=train_loop,
            train_loop_config=config,
            scaling_config=ScalingConfig(num_workers=num_gpus, use_gpu=(num_gpus > 0)),
        )
        result = trainer.fit()
        print("Final metrics:", result.metrics)
    finally:
        ray.shutdown()
