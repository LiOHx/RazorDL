import numpy as np
from typing import Any
import torch.distributed as dist
import torch
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

