import os
import json
import random
import shutil
import time
from abc import abstractmethod

import numpy as np
import torch
from ray.train.torch import prepare_data_loader
from tensordict.tensordict import TensorDict
from torch.utils.data import Dataset

from razordl.core.base.dataloader import ResumeState, TrainingDataLoader
from razordl.core.base.workgroup import BaseWorkGroup
from razordl.core.base.config import BaseConfig
from razordl.core.base import logging
from razordl.core.base import checkpoint_info as ckpt_info
from razordl.core.base.metrics import Reducible
from razordl.ops.distributed.utils import all_gather_object
logger = logging.getLogger(__name__)

_WARNED_KEYS = set()


def _summarize_step_info(info, key_path=""):
    """Aggregate per-rank gathered values into per-sample statistics.

    After ``all_gather_object``, numeric leaf values from each rank are
    collected into lists.  This function walks the dict tree and reduces
    each numeric list to its mean (the per-sample expectation).  Nested
    dicts are recursed.  Everything else passes through unchanged, with a
    one-time warning per key path.
    """
    if isinstance(info, dict):
        return {k: _summarize_step_info(v, f"{key_path}.{k}" if key_path else str(k))
                for k, v in info.items()}
    if (
        isinstance(info, (list, tuple))
        and info
        and all(isinstance(x, Reducible) and type(x) is type(info[0]) for x in info)
    ):
        merged = info[0]
        for x in info[1:]:
            merged = merged.merge(x)
        return merged.to_logged()
    if isinstance(info, Reducible):
        return info.to_logged()
    if isinstance(info, (list, tuple)):
        if info and all(isinstance(x, (int, float)) for x in info):
            return _safe_mean(info)
        if info and all(isinstance(x, bool) for x in info):
            return _safe_mean(info)
        # Non-numeric list — pass through, warn once
        if key_path and key_path not in _WARNED_KEYS:
            _WARNED_KEYS.add(key_path)
            logger.warning(
                f"[step_info] {key_path} is a list of non-numeric values "
                f"({type(info[0]).__name__ if info else 'empty'}). "
                f"Passing through unchanged — consider using a numeric type."
            )
        return info
    if isinstance(info, torch.Tensor):
        if info.numel() == 1:
            return info.item()
        return _safe_mean(info.flatten().tolist())
    # Already a scalar or other type — pass through
    return info


def _safe_mean(values):
    if not values:
        return 0.0
    return sum(values) / len(values)


def set_seed(seed: int):
    """Seed all random generators for deterministic training.

    Call this before model initialization and at the start of every
    training step.  Uses the same seed across Python, NumPy, and PyTorch
    (CPU + CUDA).
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


class BaseTrainer():

    def __init__(self, config: BaseConfig):

        self.config = config
        self.trainer_config = config.trainer_config
        self.output_dir = self.config.trainer_config.output_dir
        assert self.output_dir is not None, "output_dir is not set"
        if int(os.environ.get("LOCAL_RANK", "0")) == 0:
            os.makedirs(self.output_dir, exist_ok=True)
        if torch.distributed.is_available() and torch.distributed.is_initialized():
            torch.distributed.barrier()

        self._training_start_time = time.time()
        self._resumed_from: str | None = None
        self._last_step_info: dict = {}

        self.get_resume_checkpoint_dir()

        # Seed before model init for deterministic weight initialization
        set_seed(self.trainer_config.seed)

        self.train_dataset: Dataset =  None
        self.train_collator = None
        self.prepare_data_and_workgroup()
        self.data_loader = self.build_data_loader()
    
    
    def _is_complete_checkpoint(self, ckpt_dir: str, require_complete_marker: bool = True) -> bool:
        return ckpt_info.is_complete(ckpt_dir, require_marker=require_complete_marker)


    def get_resume_checkpoint_dir(self):
        output_dir = self.config.trainer_config.output_dir
        max_resume_step = -1

        resume_checkpoint_dir = None
        if not os.path.isdir(output_dir):
            return
        for dir_name in sorted(os.listdir(output_dir)):
            dir_path = os.path.join(output_dir, dir_name)
            if not dir_name.startswith("checkpoint_") or not os.path.isdir(dir_path):
                continue
            if dir_name.endswith(".tmp"):
                continue

            resume_step = int(dir_name.split("_")[1])
            if self._is_complete_checkpoint(dir_path) and resume_step > max_resume_step:
                max_resume_step = resume_step
                resume_checkpoint_dir = dir_path

        logger.info(f"[RESUME] Found checkpoint at step {max_resume_step}: {resume_checkpoint_dir}")
        self.config.trainer_config.resume_checkpoint_dir = resume_checkpoint_dir


    def get_resume_state(self):
        if getattr(self.config.trainer_config, 'init_from', None):
            logger.info("[INIT_FROM] Starting from step 0 (forked from checkpoint)")
            logger.info("[INIT_FROM] Seed: %s", self.trainer_config.seed)
            return ResumeState.from_seed(self.trainer_config.seed)
        if self.config.trainer_config.resume_checkpoint_dir:
            if not self._is_complete_checkpoint(self.config.trainer_config.resume_checkpoint_dir):
                raise ValueError(
                    "[RESUME] Invalid checkpoint: "
                    f"{self.config.trainer_config.resume_checkpoint_dir} is missing "
                    "model/adapter or optimizer state."
                )
            logger.info(f"[RESUME] Resuming from {self.config.trainer_config.resume_checkpoint_dir}")
            self._resumed_from = os.path.basename(
                self.config.trainer_config.resume_checkpoint_dir.rstrip(os.sep)
            )
            self._check_topology_compat(self.config.trainer_config.resume_checkpoint_dir)
            return ResumeState.from_checkpoint_dir(self.config.trainer_config.resume_checkpoint_dir)
        else:
            logger.info(f"[RESUME] No checkpoint found, starting from scratch")
            logger.info(f"[RESUME] Seed: {self.trainer_config.seed}")
            return ResumeState.from_seed(self.trainer_config.seed)


    def _check_topology_compat(self, ckpt_dir: str) -> None:
        """Warn (do not raise) if resuming on different world_size or sp_size."""
        if int(os.environ.get("LOCAL_RANK", "0")) != 0:
            return
        info = ckpt_info.read_info(ckpt_dir)
        if info is None:
            return
        saved_topology = info.get("topology") or {}
        current = {
            "world_size": int(os.environ.get("WORLD_SIZE", "1")),
            "sp_size": getattr(self.config.data_config, "sp_size", 1) or 1,
        }
        for key, current_value in current.items():
            saved_value = saved_topology.get(key)
            if saved_value is not None and saved_value != current_value:
                logger.warning(
                    f"[RESUME WARNING] {key} mismatch: checkpoint saved with "
                    f"{key}={saved_value}, resuming with {key}={current_value}"
                )
    

    def build_data_loader(self) -> TrainingDataLoader:
        resume_state = self.get_resume_state()

        dp_rank = None
        dp_size = None
        sp_size = getattr(self.config.data_config, "sp_size", 1)
        if sp_size > 1:
            import torch.distributed as dist
            from razordl.ops.parallel.sequence_parallel import get_sp_data_parallel_info
            dp_rank, dp_size = get_sp_data_parallel_info(
                dist.get_rank(), dist.get_world_size(), sp_size
            )

        data_loader = TrainingDataLoader(
            self.train_dataset,
            collate_fn=self.train_collator,
            batch_size=self.config.trainer_config.step_batch_size,
            num_epochs=self.config.trainer_config.num_epochs,
            resume_state=resume_state,
            num_workers=self.config.trainer_config.data_loader_num_workers,
            persistent_workers=True if self.config.trainer_config.data_loader_num_workers > 0 else False,
            prefetch_factor=1 if self.config.trainer_config.data_loader_num_workers > 0 else None,
            pin_memory=torch.cuda.is_available(),
            dp_rank=dp_rank,
            dp_size=dp_size,
            seed=self.trainer_config.seed,
            )
        loader = prepare_data_loader(
            data_loader, add_dist_sampler=False, move_to_device=False
        )
        return loader


    def save_model_and_processor(self, checkpoint_dir: str):

        workgroup_fields = [
            name
            for name in dir(self)
            if not name.startswith('__') and
                isinstance(getattr(self, name), BaseWorkGroup)
        ]
        for workgroup_field in workgroup_fields:
            workgroup:BaseWorkGroup = getattr(self, workgroup_field)
            workgroup_checkpoint_dir = os.path.join(checkpoint_dir, workgroup_field)
            workgroup.save_model_and_processor(workgroup_checkpoint_dir)
        if int(os.environ.get("LOCAL_RANK", "0")) == 0:
            info = ckpt_info.build_info(
                completed_step=getattr(self, "completed_step", 0),
                config=self.config,
                elapsed_seconds=time.time() - self._training_start_time,
                last_step_info=self._last_step_info,
                resumed_from=self._resumed_from,
                ckpt_dir=checkpoint_dir,
                kind="model_only",
                compute_checksums=getattr(self.trainer_config, "compute_checksums", False),
            )
            ckpt_info.write_info(checkpoint_dir, info)


    def save_checkpoint(self, checkpoint_dir: str):
        workgroup_fields = [
            name 
            for name in dir(self) 
            if not name.startswith('__') and 
                isinstance(getattr(self, name), BaseWorkGroup)
        ]
        for workgroup_field in workgroup_fields:
            workgroup:BaseWorkGroup = getattr(self, workgroup_field)
            workgroup_checkpoint_dir = os.path.join(checkpoint_dir, workgroup_field)
            workgroup.save_checkpoint(workgroup_checkpoint_dir)


    def _save_checkpoint_atomic(self, checkpoint_dir: str, completed_step: int):
        """Save full checkpoint atomically.

        Writes to a .tmp directory first, then renames after all files
        are written.  Only rank 0 writes files and performs the atomic
        rename; other ranks only participate in the collective FSDP2 save.
        """
        local_rank = int(os.environ.get("LOCAL_RANK", "0"))
        tmp_dir = checkpoint_dir + ".tmp"
        if local_rank == 0:
            if os.path.exists(tmp_dir):
                shutil.rmtree(tmp_dir)
            os.makedirs(tmp_dir, exist_ok=True)
        # All ranks wait for rank 0 to finish creating the directory
        if torch.distributed.is_available() and torch.distributed.is_initialized():
            torch.distributed.barrier()
        self.save_checkpoint(tmp_dir)
        if local_rank == 0:
            if not self._is_complete_checkpoint(tmp_dir, require_complete_marker=False):
                raise RuntimeError(
                    f"Refusing to mark incomplete checkpoint as complete: {tmp_dir}"
                )
            info = ckpt_info.build_info(
                completed_step=completed_step,
                config=self.config,
                elapsed_seconds=time.time() - self._training_start_time,
                last_step_info=self._last_step_info,
                resumed_from=self._resumed_from,
                ckpt_dir=tmp_dir,
                kind="checkpoint",
                compute_checksums=getattr(self.trainer_config, "compute_checksums", False),
            )
            ckpt_info.write_info(tmp_dir, info)
            if os.path.exists(checkpoint_dir):
                shutil.rmtree(checkpoint_dir)
            os.rename(tmp_dir, checkpoint_dir)

    def run_training_loop(self):
        data_loader = self.data_loader
        self.completed_step = self.data_loader.resume_state.completed_step
        if self.completed_step > 0:
            logger.info(f"[TRAINER] Resuming from completed_step={self.completed_step}")
        else:
            logger.info(f"[TRAINER] Training starting from scratch")

        local_rank = int(os.environ.get("LOCAL_RANK", "0"))
        from tqdm import tqdm
        if local_rank == 0:
            tqdm_loader = tqdm(
                data_loader,
                total=len(data_loader) + self.completed_step,
                initial=self.completed_step,
            )
        else:
            tqdm_loader = data_loader

        device = torch.device(f"cuda:{int(os.environ.get('LOCAL_RANK', '0'))}") if torch.cuda.is_available() else torch.device("cpu")

        for batch_data in tqdm_loader:
            batch_data = self._move_batch_to_device(batch_data, device)

            current_step = self.completed_step + 1

            step_start_time = time.time()
            step_info = self.update_step(batch_data, current_step)
            step_info = all_gather_object(step_info, float_mean=True)
            step_info = _summarize_step_info(step_info)
            step_time = round(time.time() - step_start_time, 2)
            step_info["step"] = current_step
            step_info["step_time"] = step_time

            self.completed_step = current_step
            self._last_step_info = dict(step_info)

            if current_step % self.config.trainer_config.log_info_steps == 0:
                logger.info(f"[TRAINER] Step {current_step} completed, cost time: {step_time}s, step_info: {step_info}")
                if int(os.environ.get("LOCAL_RANK", "0")) == 0:
                    step_info_path = os.path.join(self.output_dir, "step_info.jsonl")
                    with open(step_info_path, "a") as f:
                        f.write(json.dumps(step_info, ensure_ascii=False) + "\n")

            if (
                self.config.trainer_config.save_model_steps > 0
                and current_step % self.config.trainer_config.save_model_steps == 0
            ):
                checkpoint_dir = os.path.join(self.output_dir, f"checkpoint_{self.completed_step:06d}")
                if local_rank == 0:
                    os.makedirs(checkpoint_dir, exist_ok=True)
                if torch.distributed.is_available() and torch.distributed.is_initialized():
                    torch.distributed.barrier()
                self.save_model_and_processor(checkpoint_dir)

            if (
                self.config.trainer_config.save_checkpoint_steps > 0
                and current_step % self.config.trainer_config.save_checkpoint_steps == 0
            ):
                checkpoint_dir = os.path.join(self.output_dir, f"checkpoint_{self.completed_step:06d}")
                self._save_checkpoint_atomic(checkpoint_dir, self.completed_step)

        logger.info(f"[TRAINER] Training completed")
        checkpoint_dir = self.output_dir
        if local_rank == 0:
            os.makedirs(checkpoint_dir, exist_ok=True)
        if torch.distributed.is_available() and torch.distributed.is_initialized():
            torch.distributed.barrier()
        self.save_model_and_processor(checkpoint_dir)


    def _move_batch_to_device(self, batch_data, device):
        """Move all tensors in a batch to the target device (recursively)."""
        if isinstance(batch_data, dict):
            return {
                k: self._move_batch_to_device(v, device)
                for k, v in batch_data.items()
            }
        elif isinstance(batch_data, torch.Tensor):
            return batch_data.to(device)
        elif hasattr(batch_data, "to"):
            return batch_data.to(device)
        return batch_data

    @abstractmethod
    def prepare_data_and_workgroup(self):
        pass
        
    @abstractmethod
    def update_step(self, input_dict: TensorDict, step:int) -> dict:
        pass
