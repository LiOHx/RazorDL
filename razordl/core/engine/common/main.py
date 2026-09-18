import os
from datetime import datetime
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
from razordl.core.base import checkpoint_info as ckpt_info
from razordl.core.base.workgroup import BaseWorkGroup
from razordl.core.engine.common.trainer import EngineTrainer
from razordl.ops.hardware.device import check_device_compatibility
from razordl.ops.snapshot import (
    _razordl_git_info,
    compute_code_hash,
    get_latest_experiment,
    get_provenance,
    is_experiment_completed,
    snapshot_code,
)


def _log_hardware_capabilities(config, logger) -> None:
    """Log detected accelerator capabilities and the precision they resolve to.

    Printed before Ray starts so "why is this slow?" is answerable from the
    first few lines of the log rather than by bisecting the config.
    """
    from razordl.ops.hardware.device import describe
    from razordl.ops.hardware.precision import resolve_precision

    info = describe()
    logger.info(
        "[HW] %s x%s | %s | %.1f GB | native bf16: %s | flash-attn 2: %s",
        info.get("name") or info["device"],
        info["count"],
        info.get("compute_capability") or "n/a",
        info["memory_gb"],
        info["native_bf16"],
        info["flash_attention_2"],
    )

    requested = config.worker_group_config.model_group_config.model_config.precision
    resolved = resolve_precision(requested)
    logger.info("[HW] precision: %s -> %s", requested, resolved)
    if resolved == "fp16":
        logger.info(
            "[HW] fp16 path active: fp32 master weights for trainable params "
            "+ GradScaler loss scaling + autocast forward."
        )
    elif resolved == "bf16":
        logger.info(
            "[HW] bf16 path active: fp32 master weights for trainable params "
            "+ autocast forward%s.",
            "" if info["native_bf16"] else
            " (bf16 is emulated in software on this GPU; see the measured "
            "trade-off against fp16 in ops/hardware/precision.py)",
        )


def _exclude_paths(outputs_dir: str) -> frozenset[str]:
    """Absolute dirs the code snapshot / hash must never descend into.

    ``outputs_dir`` is user-configurable, so the name-based EXCLUDE_DIRS in
    ops/snapshot.py cannot cover it.  Including it would copy every previous
    experiment into each new snapshot and, worse, hash their step_info.jsonl /
    checkpoint_info.json so the code hash changes every run and auto-resume
    never matches.
    """
    return frozenset({os.path.abspath(outputs_dir)})


def _create_experiment_dir(outputs_dir: str) -> str:
    ts = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    exp_dir = os.path.join(outputs_dir, ts)
    os.makedirs(exp_dir, exist_ok=True)
    return exp_dir


def _resolve_experiment(outputs_dir, resume_mode, resume_from, project_dir, init_from=None):
    """Determine the experiment directory with snapshot/hash logic.

    Returns ``(exp_dir, is_new)``.
    """
    logger = logging.getLogger(__name__)
    outputs_dir = os.path.abspath(outputs_dir)

    if init_from:
        # A fork is always a fresh experiment: weights come from init_from,
        # step counter and optimizer start at zero, so resuming the latest
        # experiment here would silently turn the fork into a resume.
        exp_dir = _create_experiment_dir(outputs_dir)
        logger.info("[EXP] New experiment (init_from fork): %s", exp_dir)
        return exp_dir, True

    if resume_mode == "manual":
        if resume_from:
            exp_dir = os.path.abspath(resume_from)
            if not os.path.isdir(exp_dir):
                raise FileNotFoundError(f"resume_from path does not exist: {exp_dir}")
            logger.info("[EXP] Manual resume from: %s", exp_dir)
            return exp_dir, False
        # manual without resume_from → always new
        exp_dir = _create_experiment_dir(outputs_dir)
        logger.info("[EXP] New experiment (manual, no resume_from): %s", exp_dir)
        return exp_dir, True

    # resume_mode == "auto"
    latest = get_latest_experiment(outputs_dir)

    if latest is None:
        exp_dir = _create_experiment_dir(outputs_dir)
        logger.info("[EXP] New experiment (no history): %s", exp_dir)
        return exp_dir, True

    if is_experiment_completed(latest):
        exp_dir = _create_experiment_dir(outputs_dir)
        logger.info("[EXP] New experiment (latest %s is complete): %s", os.path.basename(latest), exp_dir)
        return exp_dir, True

    # Latest experiment is incomplete — check code/config hash
    provenance = get_provenance(latest)
    if provenance is None:
        # No provenance.json — old experiment, can't verify, start fresh
        exp_dir = _create_experiment_dir(outputs_dir)
        logger.info("[EXP] New experiment (no provenance in %s): %s", os.path.basename(latest), exp_dir)
        return exp_dir, True

    current_code_hash = compute_code_hash(project_dir, exclude_paths=_exclude_paths(outputs_dir))
    razordl_info = _razordl_git_info()
    current_razordl_id = razordl_info.get("razordl_git_commit") or razordl_info.get("razordl_version", "")
    saved_razordl_id = provenance.get("razordl_git_commit") or provenance.get("razordl_version", "")

    if current_code_hash != provenance.get("code_hash") or current_razordl_id != saved_razordl_id:
        exp_dir = _create_experiment_dir(outputs_dir)
        if current_code_hash != provenance.get("code_hash"):
            logger.warning("[EXP] Project code changed since %s", os.path.basename(latest))
        else:
            logger.warning(
                "[EXP] RazorDL changed (%s → %s) since %s",
                saved_razordl_id, current_razordl_id, os.path.basename(latest),
            )
        logger.warning("[EXP] Creating new experiment: %s", exp_dir)
        return exp_dir, True

    # Hash matches — resume
    logger.info("[EXP] Auto-resuming from: %s (code + razordl hash match)", latest)
    return latest, False


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


    # --- experiment management (before Ray starts) ---
    logger = logging.getLogger(__name__)
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        handlers=[logging.StreamHandler()],
    )

    _log_hardware_capabilities(config, logger)

    project_dir = os.getcwd()
    outputs_dir = config.trainer_config.outputs_dir
    resume_mode = config.trainer_config.resume_mode
    resume_from = config.trainer_config.resume_from
    init_from = config.trainer_config.init_from

    # If output_dir is already set (e.g. from a code snapshot), use it directly
    if config.trainer_config.output_dir:
        exp_dir = os.path.abspath(config.trainer_config.output_dir)
        is_new = not os.path.isdir(exp_dir)
        if is_new:
            logger.info("[EXP] output_dir %s does not exist — creating new experiment", exp_dir)
            os.makedirs(exp_dir, exist_ok=True)
            snapshot_code(exp_dir, project_dir, exclude_paths=_exclude_paths(outputs_dir))
        else:
            logger.info("[EXP] Using pre-set output_dir: %s", exp_dir)
    else:
        exp_dir, is_new = _resolve_experiment(
            outputs_dir, resume_mode, resume_from, project_dir, init_from=init_from
        )
        if is_new:
            provenance = snapshot_code(exp_dir, project_dir, exclude_paths=_exclude_paths(outputs_dir))
            logger.info("[EXP] Code snapshot saved to %s/code/", exp_dir)
            logger.info("[EXP] Code hash: %s", provenance["code_hash"])
            if provenance.get("git_commit"):
                logger.info("[EXP] Git commit: %s (dirty=%s)", provenance["git_commit"], provenance["git_dirty"])

    config.trainer_config.output_dir = os.path.abspath(exp_dir)
    config.data_config.train_data_path = os.path.abspath(config.data_config.train_data_path)

    # init_from: fork from a checkpoint — weights load, step/optimizer reset.
    # The checkpoint path is resolved into resume_checkpoint_dir by BaseTrainer
    # on the workers (get_resume_checkpoint_dir); main() only decides the
    # experiment dir.  Validating here as well fails fast, before Ray starts.
    if init_from:
        init_from = os.path.abspath(init_from)
        if not os.path.isdir(init_from):
            raise FileNotFoundError(f"[INIT_FROM] checkpoint dir does not exist: {init_from}")
        if not ckpt_info.has_model_weights(init_from):
            raise ValueError(
                f"[INIT_FROM] no model.safetensors / adapter_model.safetensors under {init_from}"
            )
        logger.info("[EXP] Forking model weights from: %s", init_from)
    # --- experiment management end ---

    logger.info("*" * 100)
    logger.info("Experiment: %s", exp_dir)
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
                # Inherit the driver's allocator settings (e.g. expandable_segments:True,
                # the anti-fragmentation switch a 22 GB single card needs most). Blanking
                # it here dropped every user-set value; ray_kwargs.ray_init.runtime_env
                # .env_vars still overrides.
                "PYTORCH_CUDA_ALLOC_CONF": os.environ.get("PYTORCH_CUDA_ALLOC_CONF", ""),
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
