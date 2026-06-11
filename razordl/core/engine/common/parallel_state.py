from __future__ import annotations

import json
import os
from typing import Iterator

import torch


def unwrap_model(model):
    return model.module if hasattr(model, "module") else model


def materialize_tensor(tensor):
    try:
        from torch.distributed.tensor import DTensor
    except Exception:
        DTensor = None

    if DTensor is not None and isinstance(tensor, DTensor):
        try:
            tensor = tensor.full_tensor()
        except Exception:
            try:
                tensor = tensor.to_local()
            except Exception:
                tensor = getattr(tensor, "_local_tensor", tensor)
    if hasattr(tensor, "is_cuda") and tensor.is_cuda:
        tensor = tensor.cpu()
    return tensor


def iter_model_weights_for_sync(model, *, lora_only: bool) -> Iterator[tuple[str, torch.Tensor]]:
    model = unwrap_model(model)
    if lora_only:
        peft_config = getattr(model, "peft_config", {}).get("default", None)
        if peft_config is None:
            return
        params = model.base_model.model.state_dict()
        items = ((k, v) for k, v in params.items() if "lora" in k)
    else:
        items = model.state_dict().items()

    for name, tensor in items:
        yield name, materialize_tensor(tensor).cpu()


def _is_peft_model(model) -> bool:
    try:
        from peft import PeftModel

        return isinstance(unwrap_model(model), PeftModel)
    except Exception:
        return False


def _save_adapter_config(model, adapter_dir: str) -> None:
    model = unwrap_model(model)
    if not hasattr(model, "peft_config"):
        return

    adapter_config_to_save = {}
    is_single_adapter = len(model.peft_config) == 1
    if is_single_adapter:
        config = list(model.peft_config.values())[0]
        config_dict = config.to_dict()
        for k, v in config_dict.items():
            if isinstance(v, (str, int, float, bool, list, dict, type(None))):
                adapter_config_to_save[k] = v
            elif isinstance(v, (set, tuple)):
                adapter_config_to_save[k] = list(v)
            else:
                adapter_config_to_save[k] = str(v)
    else:
        for key, config in model.peft_config.items():
            config_dict = config.to_dict()
            cleaned = {}
            for k, v in config_dict.items():
                if isinstance(v, (str, int, float, bool, list, dict, type(None))):
                    cleaned[k] = v
                elif isinstance(v, (set, tuple)):
                    cleaned[k] = list(v)
                else:
                    cleaned[k] = str(v)
            adapter_config_to_save[key] = cleaned

    with open(os.path.join(adapter_dir, "adapter_config.json"), "w") as f:
        json.dump(adapter_config_to_save, f, indent=2, ensure_ascii=False)


def save_full_or_adapter_model(
    model,
    save_dir: str,
    *,
    save_lora_separately: bool = True,
    save_full_model: bool = True,
) -> None:
    model = unwrap_model(model)
    os.makedirs(save_dir, exist_ok=True)
    is_peft_model = _is_peft_model(model)
    state_dict = {k: materialize_tensor(v) for k, v in model.state_dict().items()}

    if is_peft_model and save_lora_separately:
        adapter_dir = os.path.join(save_dir, "adapter")
        os.makedirs(adapter_dir, exist_ok=True)
        adapter_state_dict = {}
        for key, value in state_dict.items():
            if "lora_" in key or "adapter" in key:
                new_key = key.replace("base_model.model.", "").replace(".default", "")
                adapter_state_dict[new_key] = value
        if adapter_state_dict:
            from safetensors.torch import save_file

            save_file(adapter_state_dict, os.path.join(adapter_dir, "adapter_model.safetensors"))
            _save_adapter_config(model, adapter_dir)
        elif not save_full_model:
            raise RuntimeError(f"No LoRA parameters found while saving adapter to {adapter_dir}")

    if not is_peft_model and not save_full_model:
        save_full_model = True

    if save_full_model:
        from safetensors.torch import save_file

        try:
            model.config.save_pretrained(save_dir)
        except Exception:
            with open(os.path.join(save_dir, "config.json"), "w") as f:
                f.write(model.config.to_json_string())
        tmp_path = os.path.join(save_dir, "model.safetensors.tmp")
        final_path = os.path.join(save_dir, "model.safetensors")
        save_file(state_dict, tmp_path)
        os.replace(tmp_path, final_path)
    else:
        try:
            model.config.save_pretrained(save_dir)
        except Exception:
            with open(os.path.join(save_dir, "config.json"), "w") as f:
                f.write(model.config.to_json_string())
