from torch.utils.data import DataLoader, Sampler
import torch
import torch.distributed as dist
import math
import random
import numpy as np
from typing import Optional
from dataclasses import dataclass
import os
from razordl.core.base import logging
logger = logging.getLogger(__name__)


def _set_seed(seed: int):
    """Set all global RNG sources to the given seed."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


@dataclass
class ResumeState:
    completed_step: int = 0

    @classmethod
    def from_checkpoint_dir(cls, checkpoint_dir: str):
        step = int(checkpoint_dir.split("/")[-1].split("_")[1])
        return cls(completed_step=step)

    @classmethod
    def from_seed(cls, seed: int):
        _set_seed(seed)
        return cls(completed_step=0)


class InfiniteBatchSampler(Sampler):
    def __init__(self, dataset, batch_size, rank=0, world_size=1, start_step=0, shuffle=True, drop_last=False, seed=0, num_epochs=1):
        self.dataset = dataset
        self.batch_size = batch_size
        self.rank = rank
        self.world_size = world_size
        self.shuffle = shuffle
        self.drop_last = drop_last
        self.seed = seed
        self.num_epochs = num_epochs
        
        # Calculate batches per epoch for progress tracking
        num_samples = math.ceil(len(self.dataset) / self.world_size)
        if self.drop_last:
            self.batches_per_epoch = num_samples // self.batch_size
        else:
            self.batches_per_epoch = math.ceil(num_samples / self.batch_size)
            
        self.set_progress(start_step)

    def set_progress(self, step):
        self.step = step
        if self.batches_per_epoch > 0:
            self.epoch = step // self.batches_per_epoch
            self.start_batch_in_epoch = step % self.batches_per_epoch
        else:
            self.epoch = 0
            self.start_batch_in_epoch = 0

    def __iter__(self):
        curr_epoch = self.epoch
        start_batch = self.start_batch_in_epoch
        
        while curr_epoch < self.num_epochs:
            g = torch.Generator()
            g.manual_seed(self.seed + curr_epoch)
            
            if self.shuffle:
                indices = torch.randperm(len(self.dataset), generator=g).tolist()
            else:
                indices = list(range(len(self.dataset)))
            
            # Add padding to make it evenly divisible by world_size
            if self.world_size > 1:
                num_samples = math.ceil(len(self.dataset) / self.world_size)
                total_size = num_samples * self.world_size
                indices += indices[:(total_size - len(indices))]
                
                # Subsample for this rank
                indices = indices[self.rank:total_size:self.world_size]
            
            # Yield batches
            batch = []
            batch_idx = 0
            for idx in indices:
                batch.append(idx)
                if len(batch) == self.batch_size:
                    if batch_idx >= start_batch:
                        yield batch
                        self.step += 1 # Update step
                    batch = []
                    batch_idx += 1
            
            if len(batch) > 0 and not self.drop_last:
                if batch_idx >= start_batch:
                    yield batch
                    self.step += 1 # Update step
                batch_idx += 1
            
            curr_epoch += 1
            start_batch = 0
            self.epoch = curr_epoch # Update internal state
            
    def __len__(self):
        total_steps = self.batches_per_epoch * self.num_epochs
        return max(0, total_steps - self.step)



class TrainingDataLoader(DataLoader):
    def __init__(self,
        dataset,
        collate_fn,
        batch_size: int,
        num_epochs: int = 1,
        resume_state: Optional[ResumeState] = None,
        num_workers: int = 0,
        persistent_workers: bool = False,
        prefetch_factor: Optional[int] = None,
        pin_memory: bool = False,
        dp_rank: Optional[int] = None,
        dp_size: Optional[int] = None,
        seed: int = 42,
    ) -> None:


        self.resume_state = resume_state if resume_state is not None else ResumeState()

        rank = int(os.environ.get("LOCAL_RANK", "0"))
        world_size = int(os.environ.get("WORLD_SIZE", "1"))

        if dist.is_available() and dist.is_initialized():
            rank = dist.get_rank()
            world_size = dist.get_world_size()

        if dp_rank is not None:
            rank = dp_rank
        if dp_size is not None:
            world_size = dp_size

        self.batch_sampler_instance = InfiniteBatchSampler(
            dataset,
            batch_size=batch_size,
            rank=rank,
            world_size=world_size,
            start_step=self.resume_state.completed_step,
            shuffle=True,
            drop_last=False,
            seed=seed,
            num_epochs=num_epochs
        )

        super().__init__(
            dataset,
            batch_sampler=self.batch_sampler_instance,
            num_workers=num_workers,
            collate_fn=collate_fn,
            persistent_workers=persistent_workers,
            prefetch_factor=prefetch_factor,
            pin_memory=pin_memory,
        )

    def get_consumed_batches(self) -> int:
        return self.batch_sampler_instance.step
