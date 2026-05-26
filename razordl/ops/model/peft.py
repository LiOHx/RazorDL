
import os
import logging

from safetensors.torch import load_file

logger = logging.getLogger(__name__)

def get_adapter_state_dict(adapter_model_path):
    def _normalize_lora_key(key: str) -> str:
        if not isinstance(key, str):
            return key
        # 兼容 vllm 的 key 清洗逻辑
        normalized = key.replace("base_model.model.", "")
        normalized = normalized.replace(".default.", ".")
        if normalized.endswith(".default"):
            normalized = normalized[: -len(".default")]
        return normalized

    def _prepare_lora_state_dict_for_loading(state_dict):
        if not isinstance(state_dict, dict):
            return state_dict
        prepared = {}
        for key, value in state_dict.items():
            prepared[_normalize_lora_key(key)] = value
        return prepared

    adapter_state_dict_raw = load_file(adapter_model_path)
    adapter_state_dict = _prepare_lora_state_dict_for_loading(adapter_state_dict_raw)
    return adapter_state_dict

def set_adapter_state_dict(model, adapter_state_dict):
    try:
        from peft.utils import set_peft_model_state_dict
    except ImportError:
        try:
            from peft.utils.save_and_load import set_peft_model_state_dict
        except ImportError:
            set_peft_model_state_dict = None

    # save_fsdp2 strips base_model.model. prefix and .default. suffix from
    # LoRA keys for vLLM compatibility.  Restore base_model.model. prefix.
    model_keys = list(model.state_dict().keys())
    model_has_prefix = any(k.startswith("base_model.model.") for k in model_keys)
    adapter_has_prefix = any(k.startswith("base_model.model.") for k in adapter_state_dict)

    sd = {}
    if model_has_prefix and not adapter_has_prefix:
        for k, v in adapter_state_dict.items():
            sd[f"base_model.model.{k}"] = v
    elif not model_has_prefix and adapter_has_prefix:
        for k, v in adapter_state_dict.items():
            sd[k.replace("base_model.model.", "")] = v
    else:
        sd = dict(adapter_state_dict)

    # set_peft_model_state_dict handles .default. internally — do NOT add
    # it here or it will double.  Only restore .default. when falling back
    # to plain load_state_dict.
    if set_peft_model_state_dict is not None:
        try:
            result = set_peft_model_state_dict(model, sd, adapter_name="default")
            return getattr(result, "missing_keys", []), getattr(result, "unexpected_keys", [])
        except Exception:
            pass

    # load_state_dict fallback — must restore .default. ourselves
    _lora_model_keys = [k for k in model_keys if "lora_" in k.lower()]
    if any(".default." in k for k in _lora_model_keys):
        _key_map = {}
        for mk in _lora_model_keys:
            nk = mk.replace("base_model.model.", "").replace(".default.", ".")
            _key_map[nk] = mk
        _fixed = {}
        for k, v in sd.items():
            nk = k.replace("base_model.model.", "")
            k = _key_map.get(nk, k)
            _fixed[k] = v
        sd = _fixed

    missing, unexpected = model.load_state_dict(sd, strict=False)
    return missing, unexpected


def load_lora_adapter_compatible(model, adapter_path, adapter_config, local_rank: int):
    """
    兼容两种 LoRA adapter 加载方式：
    1) 标准 PEFT `save_pretrained()` 产物：`PeftModel.from_pretrained(model, adapter_dir)`
    2) 为兼容 vLLM/RazorDL 做过 key 清洗的 adapter：先 `get_peft_model(LoraConfig)` 建结构，再用
       `get_adapter_state_dict()` + `set_adapter_state_dict()` 加载 `adapter_model.safetensors`
    """
    import warnings
    from peft import LoraConfig, PeftModel, get_peft_model

    def _resolve_lora_targets(target_modules):
        """Auto-fix target modules for models that wrap nn.Linear in a
        custom container (e.g. Gemma4ClippableLinear.linear).

        Scans ALL named modules. If every instance of a target name is a
        non-standard wrapper with an inner ``.linear`` child, rewrites the
        target to ``<target>.linear``.
        """
        import torch.nn as nn
        PEFT_SUPPORTED = (nn.Linear, nn.Embedding, nn.Conv1d, nn.Conv2d, nn.Conv3d)

        resolved = list(target_modules)
        for i, target in enumerate(resolved):
            needs_unwrap = False
            found_any = False
            for name, module in model.named_modules():
                if name.endswith(f".{target}") or name == target:
                    found_any = True
                    if isinstance(module, PEFT_SUPPORTED):
                        needs_unwrap = False
                        break
                    if hasattr(module, "linear") and isinstance(module.linear, nn.Linear):
                        needs_unwrap = True
            if found_any and needs_unwrap:
                resolved[i] = f"{target}.linear"

        if resolved != list(target_modules) and local_rank == 0:
            logger.info(
                f"[ADAPTER] Auto-resolved LoRA targets for custom linear wrappers: "
                f"{list(target_modules)} -> {resolved}"
            )
        return resolved

    def _build_lora_config(target_modules):
        kwargs = dict(
            r=adapter_config.lora_r,
            lora_alpha=adapter_config.lora_alpha,
            target_modules=target_modules,
            lora_dropout=adapter_config.lora_dropout,
            modules_to_save=adapter_config.modules_to_save,
            bias="none",
        )
        if adapter_config.task_type is not None:
            kwargs["task_type"] = adapter_config.task_type
        return LoraConfig(**kwargs)

    def _build_lora_config_from_train_cfg():
        resolved = _resolve_lora_targets(adapter_config.lora_target_modules)
        return _build_lora_config(resolved)

    def _with_default_adapter_key(key: str) -> str:
        """
        将清洗过的 LoRA key（无 `.default.`）转换为 PEFT 常见的 default adapter key。
        e.g. `...lora_A.weight` -> `...lora_A.default.weight`
        """
        if not isinstance(key, str):
            return key
        if ".default." in key:
            return key
        # Linear LoRA
        if key.endswith(".lora_A.weight"):
            return key[: -len(".lora_A.weight")] + ".lora_A.default.weight"
        if key.endswith(".lora_B.weight"):
            return key[: -len(".lora_B.weight")] + ".lora_B.default.weight"
        # Embedding LoRA (部分模型/版本会出现)
        if key.endswith(".lora_embedding_A.weight"):
            return key[: -len(".lora_embedding_A.weight")] + ".lora_embedding_A.default.weight"
        if key.endswith(".lora_embedding_B.weight"):
            return key[: -len(".lora_embedding_B.weight")] + ".lora_embedding_B.default.weight"
        return key

    def _load_cleaned_adapter_weights_into(peft_wrapped_model, adapter_dir: str):
        """
        从 adapter_model.safetensors 加载（可能被清洗过 key 的）LoRA 权重到 PEFT 模型结构中。
        """
        adapter_model_path = os.path.join(adapter_dir, "adapter_model.safetensors")
        if not os.path.exists(adapter_model_path):
            raise FileNotFoundError(f"adapter_model.safetensors not found under: {adapter_dir}")

        from razordl.ops.model.peft import get_adapter_state_dict, set_adapter_state_dict

        adapter_state_dict = get_adapter_state_dict(adapter_model_path)
        adapter_state_dict = {_with_default_adapter_key(k): v for k, v in adapter_state_dict.items()}
        missing, unexpected = set_adapter_state_dict(peft_wrapped_model, adapter_state_dict)

        if local_rank == 0:
            if missing:
                logger.warning(f"[ADAPTER WARNING] Missing keys when loading adapter: {missing}")
            if unexpected:
                logger.warning(f"[ADAPTER WARNING] Unexpected keys when loading adapter: {unexpected}")
            logger.info(f"[ADAPTER] LoRA adapter loaded by compat weights ({len(adapter_state_dict)} tensors)")

        return peft_wrapped_model

    def _resolve_adapter_dir(path):
        if not path or not os.path.exists(path):
            return None
        # 支持传入 checkpoint 根目录（里面有 adapter/ 子目录）
        if os.path.isdir(path):
            direct = path
            nested = os.path.join(path, "adapter")
            if os.path.exists(os.path.join(direct, "adapter_model.safetensors")) or os.path.exists(
                os.path.join(direct, "adapter_config.json")
            ):
                return direct
            if os.path.exists(os.path.join(nested, "adapter_model.safetensors")) or os.path.exists(
                os.path.join(nested, "adapter_config.json")
            ):
                return nested
        return path

    adapter_dir = _resolve_adapter_dir(adapter_path)

    # 没有现成 adapter：创建一个新的可训练 LoRA 结构
    if not adapter_dir:
        lora_cfg = _build_lora_config_from_train_cfg()
        try:
            return get_peft_model(model, lora_cfg)
        except ValueError as e:
            if "is not supported" not in str(e):
                raise
            # Fallback: append .linear to each target for models using
            # custom linear wrappers (e.g. Gemma4ClippableLinear).
            fallback_targets = [f"{t}.linear" for t in adapter_config.lora_target_modules]
            if local_rank == 0:
                logger.warning(
                    f"[ADAPTER] PEFT rejected target modules, retrying with "
                    f"inner .linear targets: {fallback_targets}"
                )
            return get_peft_model(model, _build_lora_config(fallback_targets))

    if local_rank == 0:
        logger.info(f"Loading LoRA adapter from {adapter_dir}")

    # 方式1：标准 PEFT adapter（save_pretrained 产物）
    # 若发现 missing adapter keys（常见于为兼容 vLLM 做过 key 清洗的 adapter），自动回退到方式2。
    try:
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            peft_model = PeftModel.from_pretrained(model, adapter_dir, is_trainable=True)

        has_missing_adapter_keys = any(
            ("missing adapter keys" in str(w.message).lower()) or ("found missing adapter keys" in str(w.message).lower())
            for w in caught
        )
        if has_missing_adapter_keys:
            # 注意：from_pretrained 可能已经把 LoRA 结构注入到了模型里。
            # 这里不要再二次 get_peft_model，否则会重复注入导致显存/参数量暴涨。
            if local_rank == 0:
                logger.warning("[ADAPTER] PEFT reported missing adapter keys; trying compat weights load into same model.")
            return _load_cleaned_adapter_weights_into(peft_model, adapter_dir)
        return peft_model
    except Exception as e:
        # 方式2：from_pretrained 彻底失败时，兼容 vLLM / RazorDL 清洗过 key 的 adapter
        if local_rank == 0:
            logger.warning(f"[ADAPTER] Standard PEFT load failed, fallback to compat loader: {e}")

        # 2.1 先用 adapter_config.json（若存在）构建 PEFT 结构；否则用训练配置创建
        try:
            lora_cfg = LoraConfig.from_pretrained(adapter_dir)
            lora_cfg.inference_mode = False
        except Exception:
            lora_cfg = _build_lora_config_from_train_cfg()

        peft_model = get_peft_model(model, lora_cfg)
        return _load_cleaned_adapter_weights_into(peft_model, adapter_dir)
