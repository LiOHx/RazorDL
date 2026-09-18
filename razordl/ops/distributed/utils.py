import os
from typing import Any

import numpy as np
import torch
import torch.distributed as dist


def get_global_rank() -> int:
    """Global rank: the process group when initialised, else ``RANK``, else ``LOCAL_RANK``, else 0."""
    if dist.is_available() and dist.is_initialized():
        return dist.get_rank()
    return int(os.environ.get("RANK", os.environ.get("LOCAL_RANK", "0")))


def is_global_rank0() -> bool:
    """Gate for writes that must happen exactly once per job.

    ``LOCAL_RANK == 0`` is true on *every node*, so files gated on it were
    written once per node: on a shared filesystem that is N concurrent
    writers of the same checkpoint_info.json / tokenizer / scaler.pt, and the
    atomic rename in the trainer raced with itself.  Ray Train sets ``RANK``
    before the process group exists, so the env fallback keeps early-startup
    writes (output dir creation) single-writer too.
    """
    return get_global_rank() == 0


def _gather_across_ranks(obj):
    if dist.is_available() and dist.is_initialized():
        world_size = dist.get_world_size()
        gathered_list = [None for _ in range(world_size)]
        dist.all_gather_object(gathered_list, obj)
        return gathered_list
    return [obj]


def all_gather_object(x: torch.Tensor|np.ndarray|list|dict[str, torch.Tensor|np.ndarray|list]|Any, float_mean: bool = False):


    def all_gather_tensor(tensor: torch.Tensor):
        local_tensor = tensor.detach()
        world_size = dist.get_world_size()
        gathered_list = [None for _ in range(world_size)]
        dist.all_gather_object(gathered_list, local_tensor.cpu())
        return torch.cat(gathered_list, dim=0).to(local_tensor.device, dtype=local_tensor.dtype)
    

    def all_gather_array(array: np.ndarray):
        local_array = array
        world_size = dist.get_world_size()
        gathered_list = [None for _ in range(world_size)]
        dist.all_gather_object(gathered_list, local_array)
        return np.concatenate(gathered_list, axis=0)
    

    def _all_gather_dict(dict: dict[str, torch.Tensor|np.ndarray|list|Any]):
        local_dict = dict
        local_device_dict = {}
        local_dtype_dict = {}
        for k in local_dict:
            if isinstance(local_dict[k], torch.Tensor):
                local_device_dict[k] = local_dict[k].device
                local_dtype_dict[k] = local_dict[k].dtype
                local_dict[k] = local_dict[k].cpu()
        world_size = dist.get_world_size()
        gathered_list = [None for _ in range(world_size)]
        dist.all_gather_object(gathered_list, local_dict)
        gathered_dict = {}
        for k in local_dict:
            gathered_dict[k] = []
        
        for item in gathered_list:
            for k in gathered_dict:
                gathered_dict[k].append(item[k])

        for k, v in gathered_dict.items():
            if isinstance(v[0], torch.Tensor):
                gathered_dict[k] = torch.cat(gathered_dict[k], dim=0).to(local_device_dict[k], dtype=local_dtype_dict[k])
            elif isinstance(v[0], np.ndarray):
                gathered_dict[k] = np.concatenate(gathered_dict[k], axis=0)
        return gathered_dict


    def all_gather_dict(dict: dict[str, torch.Tensor|np.ndarray|list|Any]):
        local_dict = dict
        gathered_dict = {}
        for k, v in local_dict.items():
            gathered_dict[k] = all_gather_object(v, float_mean=float_mean)
        return gathered_dict

    if dist.is_available() and dist.is_initialized():
        if isinstance(x, torch.Tensor):
            return all_gather_tensor(x)
        elif isinstance(x, np.ndarray):
            return all_gather_array(x)
        elif isinstance(x, dict):
            return all_gather_dict(x)
        elif isinstance(x, list):
            return sum(_gather_across_ranks(x), [])
        else:
            if isinstance(x, float) and float_mean:
                gathered_list = _gather_across_ranks(x)
                return sum(gathered_list) / len(gathered_list)
            return _gather_across_ranks(x)


    return [x]

