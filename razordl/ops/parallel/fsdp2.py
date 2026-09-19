from contextlib import contextmanager, nullcontext
from abc import ABC
import torch.distributed as dist
# Compatibility: Shard moved in PyTorch 2.5+
try:
    from torch.distributed.tensor import Shard, DTensor
except ImportError:
    try:
        from torch.distributed.tensor.placement_types import Shard
        from torch.distributed.tensor import DTensor
    except ImportError:
        from torch.distributed._tensor import Shard, DTensor
from torch.distributed.device_mesh import init_device_mesh
from torch.distributed.fsdp import CPUOffloadPolicy, FSDPModule, MixedPrecisionPolicy, fully_shard
import torch
import torch.nn as nn
import os
from torch.distributed.fsdp import FullyShardedDataParallel as FSDP
import logging

from razordl.ops.distributed.utils import is_global_rank0
from razordl.ops.hardware.device import get_available_device, get_device_id

logger = logging.getLogger(__name__)
fully_shard_module = torch.distributed.fsdp._fully_shard._fully_shard


@contextmanager
def maybe_patch_fsdp_module(model):
    if fully_shard_module is None:
        yield
        return

    orig_fsdp_module = fully_shard_module.FSDPModule

    class FSDPModuleABC(ABC, orig_fsdp_module):
        pass

    try:
        if isinstance(model, ABC):
            fully_shard_module.FSDPModule = FSDPModuleABC
        yield
    finally:
        fully_shard_module.FSDPModule = orig_fsdp_module


def get_shard_placement_fn(fsdp_size):
    """Choose the dimension that can divide fsdp_size to avoid padding"""

    def shard_placement_fn(param):
        shape = list(param.shape)
        for i in range(len(shape)):
            if shape[i] % fsdp_size == 0:
                return Shard(i)
        return Shard(0)

    return shard_placement_fn


def create_device_mesh(world_size, fsdp_size):
    device_name = get_available_device()
    if fsdp_size < 0 or fsdp_size >= world_size:
        device_mesh = init_device_mesh(device_name, mesh_shape=(world_size,), mesh_dim_names=["fsdp"])
    else:
        device_mesh = init_device_mesh(
            device_name, mesh_shape=(world_size // fsdp_size, fsdp_size), mesh_dim_names=["ddp", "fsdp"]
        )
    return device_mesh


def _mark_fsdp_managed(model) -> None:
    """Flag the root so ``transformers`` knows the params are sharded.

    ``GenerationMixin.generate`` sets ``synced_gpus`` only when
    ``is_fsdp_managed_module(self)`` (transformers 5.x looks at exactly
    this attribute for FSDP2). Without it, a rank whose batch finishes
    early leaves the generate loop while the others still all-gather its
    shards, and the HF-generate rollout of GRPO/OPD deadlocks on >1 GPU.
    A PEFT root forwards ``generate`` to the wrapped HF model, so that
    module is the ``self`` transformers inspects and gets the flag too.
    """
    model._is_fsdp_managed_module = True
    get_base_model = getattr(model, "get_base_model", None)
    if callable(get_base_model):
        try:
            get_base_model()._is_fsdp_managed_module = True
        except Exception:  # not a PeftModel after all; the root flag stands
            pass


def apply_fsdp2(model, fsdp_kwargs, config):
    """model: AutoModelForCausalLM"""
    assert CPUOffloadPolicy is not None, "PyTorch version >= 2.4 is required for using fully_shard API (FSDP2)"

    default_transformer_cls_names_to_wrap = getattr(model, "_no_split_modules", None)
    fsdp_transformer_layer_cls_to_wrap = config.get("wrap_policy", {}).get(
        "transformer_layer_cls_to_wrap", default_transformer_cls_names_to_wrap
    )

    if isinstance(fsdp_transformer_layer_cls_to_wrap, set):
        fsdp_transformer_layer_cls_to_wrap = list(fsdp_transformer_layer_cls_to_wrap)
    elif isinstance(fsdp_transformer_layer_cls_to_wrap, str):
        fsdp_transformer_layer_cls_to_wrap = [fsdp_transformer_layer_cls_to_wrap]

    assert len(fsdp_transformer_layer_cls_to_wrap) > 0 and fsdp_transformer_layer_cls_to_wrap[0] is not None

    modules = []
    for name, module in model.named_modules():
        if module.__class__.__name__ in fsdp_transformer_layer_cls_to_wrap or (
            isinstance(module, nn.Embedding) and not model.config.tie_word_embeddings
        ):
            modules.append(module)

    for idx, module in enumerate(modules):
        # if torch.distributed.is_initialized() and torch.distributed.get_rank() == 0:
        #     print(f"wrap module {module.__class__.__name__}")
        with maybe_patch_fsdp_module(module):
            fully_shard(module, **fsdp_kwargs)

    # if torch.distributed.is_initialized() and torch.distributed.get_rank() == 0:
    #     print(f"wrap module {model.__class__.__name__}")
    with maybe_patch_fsdp_module(model):
        fully_shard(model, **fsdp_kwargs)  # fsdp2 will not reshard_after_forward for root module
    _mark_fsdp_managed(model)


def fsdp2_load_full_state_dict(model: torch.nn.Module, full_state: dict, device_mesh=None, cpu_offload=None):
    """
    Loads the full state dict (could be only on rank 0) into the sharded model. This is done by broadcasting the
    parameters from rank 0 to all other ranks. This function modifies the model in-place.

    Args:
        model (`torch.nn.Module`): The model to load the state dict into
        full_state (`dict`): The full state dict to load, can only be on rank 0
    """

    from torch.distributed.checkpoint.state_dict import StateDictOptions, set_model_state_dict

    # To broadcast, it needs to be instantiated in the GPU.
    if dist.get_rank() == 0:
        model = model.to(device=get_device_id(), non_blocking=True)
    else:
        model = model.to_empty(device=get_device_id())

    cpu_offload = cpu_offload is not None
    options = StateDictOptions(full_state_dict=True, cpu_offload=cpu_offload, broadcast_from_rank0=True)
    set_model_state_dict(model, full_state, options=options)

    # rotary_emb is not in state_dict, so we need to broadcast it manually
    for name, buf in model.named_buffers():
        dist.broadcast(buf, src=0)

    if cpu_offload:
        model.to("cpu", non_blocking=True)
        for buf in model.buffers():
            buf.data = buf.data.to(get_device_id())


def _reshard_after_forward(fsdp_mesh) -> bool:
    """Reshard after forward only when there is more than one shard.

    With a single rank there is nothing to gather or free, so resharding just
    adds per-layer bookkeeping.  Keeping fully_shard itself (rather than
    skipping FSDP2 at world_size == 1) preserves the MixedPrecisionPolicy
    cast and the checkpoint / offload code paths.
    """
    return int(fsdp_mesh.shape[-1]) > 1


def model_to_fsdp2(model, device_mesh, mp_policy) -> None:
    fsdp_mesh = device_mesh
    cpu_offload = None 

    fsdp_kwargs = {
        "mesh": fsdp_mesh,
        "mp_policy": mp_policy,
        "offload_policy": cpu_offload ,
        "reshard_after_forward": _reshard_after_forward(fsdp_mesh),
        "shard_placement_fn": get_shard_placement_fn(fsdp_size=fsdp_mesh.shape[-1]),
    }
    full_state = model.state_dict()
    apply_fsdp2(model, fsdp_kwargs, config={})
    fsdp2_load_full_state_dict(model, full_state, fsdp_mesh, cpu_offload)


def model_to_fsdp2_with_lora(model, device_mesh, mp_policy) -> None:
    """
    将 PEFT LoRA 模型包装为 FSDP2，避免 state_dict 加载破坏 LoRA 结构
    
    Args:
        model: PEFT wrapped model (PeftModel)
        device_mesh: FSDP device mesh
        mp_policy: Mixed precision policy
    """
    import os
    import torch.nn as nn
    
    # 打印模型结构以便调试
    if int(os.environ.get("LOCAL_RANK", "0")) == 0:
        print(f"\n=== Model Structure ===")
        print(f"Model type: {type(model)}")
        print(f"Has base_model: {hasattr(model, 'base_model')}")
        
        # 获取 _no_split_modules，可能在不同层级
        no_split_modules = None
        if hasattr(model, '_no_split_modules'):
            no_split_modules = model._no_split_modules
        elif hasattr(model, 'base_model') and hasattr(model.base_model, '_no_split_modules'):
            no_split_modules = model.base_model._no_split_modules
        elif hasattr(model, 'base_model') and hasattr(model.base_model, 'model') and hasattr(model.base_model.model, '_no_split_modules'):
            no_split_modules = model.base_model.model._no_split_modules
        
        print(f"_no_split_modules: {no_split_modules}")
    
    # 准备 FSDP2 参数
    fsdp_mesh = device_mesh
    fsdp_kwargs = {
        "mesh": fsdp_mesh,
        "mp_policy": mp_policy,
        "offload_policy": None,
        "reshard_after_forward": _reshard_after_forward(fsdp_mesh),
        "shard_placement_fn": get_shard_placement_fn(fsdp_size=fsdp_mesh.shape[-1]),
    }
    
    # 对于 PEFT 模型，我们需要手动包装正确的层
    # 获取 base model 的 _no_split_modules
    base_model = model.base_model.model if hasattr(model, 'base_model') else model
    default_transformer_cls_names_to_wrap = getattr(base_model, "_no_split_modules", None)
    
    if default_transformer_cls_names_to_wrap is None:
        raise ValueError("Cannot find _no_split_modules in model")
    
    if isinstance(default_transformer_cls_names_to_wrap, str):
        default_transformer_cls_names_to_wrap = [default_transformer_cls_names_to_wrap]
    
    if int(os.environ.get("LOCAL_RANK", "0")) == 0:
        print(f"Wrapping modules: {default_transformer_cls_names_to_wrap}")
    
    # 查找并包装 transformer 层
    modules_to_wrap = []
    for name, module in model.named_modules():
        if module.__class__.__name__ in default_transformer_cls_names_to_wrap:
            modules_to_wrap.append((name, module))
    
    if int(os.environ.get("LOCAL_RANK", "0")) == 0:
        print(f"Found {len(modules_to_wrap)} modules to wrap")
    
    # 包装每个 transformer 层
    for name, module in modules_to_wrap:
        with maybe_patch_fsdp_module(module):
            fully_shard(module, **fsdp_kwargs)
    
    # 包装根模块
    with maybe_patch_fsdp_module(model):
        fully_shard(model, **fsdp_kwargs)
    _mark_fsdp_managed(model)
    
    # 确保 LoRA 参数仍然可训练
    for name, param in model.named_parameters():
        if "lora_" in name:
            param.requires_grad = True
            
    if int(os.environ.get("LOCAL_RANK", "0")) == 0:
        print("✓ FSDP2 applied for LoRA training")
    
    # 统计可训练参数
    trainable_params = 0
    all_params = 0
    for name, param in model.named_parameters():
        all_params += param.numel()
        if param.requires_grad:
            trainable_params += param.numel()
    
    if int(os.environ.get("LOCAL_RANK", "0")) == 0:
        print(f"After FSDP2: trainable params: {trainable_params:,} || all params: {all_params:,} || trainable%: {100 * trainable_params / all_params:.4f}")


@torch.no_grad()
def offload_fsdp2_model_to_cpu(model, empty_cache: bool = True):
    assert isinstance(model, FSDPModule) or isinstance(model, nn.Module)
    # FSDP2 offload robustness fix:
    # When sequence lengths are very long, FSDP might internally shard/unshard in ways 
    # that leave parameters in states where direct .cpu() calls fail on _local_tensor access.
    # We try the standard way first, but fallback to a forceful move if it fails.
    try:
        model.cpu()
    except AttributeError as e:
        if "'Parameter' object has no attribute '_local_tensor'" in str(e):
            import logging
            logger = logging.getLogger(__name__)
            logger.warning(f"FSDP2 offload warning: {e}. Falling back to manual parameter offload.")
            
            # Manually move all parameters and buffers to CPU
            # This bypasses FSDP's internal state checks which might be confused by long sequences
            for p in model.parameters():
                with torch.no_grad():
                    if hasattr(p, 'data'):
                        p.data = p.data.cpu()
                    if p.grad is not None:
                        p.grad.data = p.grad.data.cpu()
            
            for b in model.buffers():
                with torch.no_grad():
                    if hasattr(b, 'data'):
                        b.data = b.data.cpu()
        else:
            raise e




@torch.no_grad()
def load_fsdp2_model_to_gpu(model):
    assert isinstance(model, FSDPModule) or isinstance(model, nn.Module)
    if torch.cuda.is_available():
        device = torch.device(f"cuda:{get_device_id()}")
    else:
        device = torch.device("cpu")
    model.to(device)




@torch.no_grad()
def offload_fsdp_optimizer(optimizer):
    if not optimizer.state:
        return
    for param_group in optimizer.param_groups:
        for param in param_group["params"]:
            state = optimizer.state[param]
            for key, value in state.items():
                if isinstance(value, torch.Tensor):
                    state[key] = value.to("cpu", non_blocking=True)


@torch.no_grad()
def load_fsdp_optimizer(optimizer, device_id):
    if not optimizer.state:
        return
    if torch.cuda.is_available():
        device = torch.device(f"cuda:{device_id}")
    else:
        device = torch.device("cpu")
    for param_group in optimizer.param_groups:
        for param in param_group["params"]:
            state = optimizer.state[param]
            for key, value in state.items():
                if isinstance(value, torch.Tensor):
                    state[key] = value.to(device, non_blocking=True)



import os as _os
import torch
import torch.distributed as dist

def save_fsdp2(
    model: FSDPModule, 
    save_dir: str, 
    save_lora_separately: bool = True, 
    save_full_model: bool = True
) -> None:
    """Save a HF model/processor by materializing a full CPU state_dict from FSDP.
    
    兼容 FSDP1 (classic) 和 FSDP2 (composable)，支持 LoRA/PEFT 模型。
    所有rank都必须调用此函数（FSDP集体操作要求），但只有rank0写文件。
    
    Args:
        model: FSDP wrapped model (可能是 PEFT 模型)
        processor: processor or tokenizer to save (自动检测类型)
        save_dir: 保存目录
        save_lora_separately: 如果是 LoRA 模型，是否单独保存 adapter (默认 True)
        save_full_model: 是否保存完整模型 (默认 True)
                        - 如果是 LoRA 模型且 save_full_model=False，只保存 adapter (推荐)
                        - 如果不是 LoRA 模型，此参数被忽略（总是保存完整模型）
    
    Examples:
        # 只保存 LoRA adapter (推荐，节省空间)
        save_fsdp2(model, save_dir, 
                                       save_lora_separately=True, 
                                       save_full_model=False)
        
        # 同时保存 adapter 和完整模型
        save_fsdp2(model, save_dir, 
                                       save_lora_separately=True, 
                                       save_full_model=True)
    """
    import os as _os
    from contextlib import nullcontext as _nullctx
    try:
        from torch.distributed.tensor import DTensor as _DTensor
    except Exception:
        _DTensor = None

    # 检测是否是 PEFT/LoRA 模型
    is_peft_model = False
    try:
        from peft import PeftModel
        # 获取未包装的模型来检查
        unwrapped_model = model
        while hasattr(unwrapped_model, '_fsdp_wrapped_module'):
            unwrapped_model = unwrapped_model._fsdp_wrapped_module
        is_peft_model = isinstance(unwrapped_model, PeftModel)
    except Exception:
        pass

    # Who writes: the SAME source of truth as save_optimizer_fsdp2 and
    # save_processor_fsdp2 (is_global_rank0).  This function used to probe
    # Ray context -> RANK env -> dist.get_rank() in its own order, which can
    # disagree with the other two helpers about who writes when a save runs
    # under an out-of-framework launcher.
    rank = 0 if is_global_rank0() else 1

    # rank0创建目录
    if rank == 0:
        _os.makedirs(save_dir, exist_ok=True)
        if is_peft_model and save_lora_separately:
            _os.makedirs(_os.path.join(save_dir, "adapter"), exist_ok=True)
    
    # 同步
    if dist.is_available() and dist.is_initialized():
        dist.barrier()

    # 使用 FSDP2 (composable fully_shard) 的保存方式
    state_dict = {}
    
    # 方法1：使用 PyTorch DCP state_dict API。相比 summon_full_params，
    # 这个路径在 composable FSDP2 下更稳定，也会保留完整的 FQN key。
    try:
        from torch.distributed.checkpoint.state_dict import get_model_state_dict, StateDictOptions
        
        state_dict = get_model_state_dict(
            model,
            options=StateDictOptions(
                full_state_dict=True,
                cpu_offload=True,
            )
        )
        
        if rank == 0:
            print("[SAVE] Using FSDP2 get_model_state_dict API")
            if is_peft_model:
                print(f"[SAVE] Detected PEFT/LoRA model")
                print(f"[SAVE]   - save_lora_separately={save_lora_separately}")
                print(f"[SAVE]   - save_full_model={save_full_model}")
                if not save_full_model and not save_lora_separately:
                    print("[SAVE WARNING] save_full_model=False and save_lora_separately=False")
                    print("[SAVE WARNING] Nothing will be saved! Setting save_lora_separately=True")
                    save_lora_separately = True
    except Exception as e1:
        # 方法2：使用 summon_full_params
        try:
            from torch.distributed._composable.fsdp import summon_full_params
            
            # 所有rank都参与 summon_full_params（这是集体操作）
            # rank0_only=True 让只有rank0获得完整参数，其他rank得到空字典
            with summon_full_params(model, writeback=False, offload_to_cpu=True, rank0_only=True):
                state_dict = model.state_dict()
            
            if rank == 0:
                print("[SAVE] Using FSDP2 summon_full_params API")
        except Exception as e2:
            # 方法3：回退到 FSDP1 (classic) API
            try:
                from torch.distributed.fsdp import StateDictType, FullStateDictConfig
                
                save_policy = FullStateDictConfig(offload_to_cpu=True, rank0_only=True)
                
                with FSDP.state_dict_type(model, StateDictType.FULL_STATE_DICT, save_policy):
                    state_dict = model.state_dict()
                
                if rank == 0:
                    print("[SAVE] Using FSDP1 classic API")
            except Exception as e3:
                # 最后的回退：直接获取 state_dict（会是分片的！）
                if rank == 0:
                    print(f"[SAVE ERROR] All FSDP APIs failed!")
                    print(f"  get_model_state_dict: {e1}")
                    print(f"  summon_full_params: {e2}")
                    print(f"  FSDP1 API: {e3}")
                    print("[SAVE WARNING] Falling back to direct state_dict - THIS WILL BE SHARDED!")
                state_dict = model.state_dict()

    if rank == 0 and not state_dict:
        raise RuntimeError(
            f"Refusing to write incomplete checkpoint at {save_dir}: "
            "model state_dict is empty."
        )
    
    # 【关键】只有rank0有完整的state_dict并执行保存
    if rank == 0:
        # Unwrap any remaining DTensors
        for _k in list(state_dict.keys()):
            _v = state_dict[_k]
            if _DTensor is not None and isinstance(_v, _DTensor):
                try:
                    _v = _v.full_tensor()
                except Exception:
                    try:
                        _v = _v.to_local()
                    except Exception:
                        _v = getattr(_v, "_local_tensor", _v)
            # 确保在CPU上
            if hasattr(_v, "is_cuda") and _v.is_cuda:
                _v = _v.cpu()
            state_dict[_k] = _v
        
        # 如果是 PEFT 模型且需要单独保存 adapter
        if is_peft_model and save_lora_separately:
            try:
                # 1. 保存 LoRA adapter（只有几MB）
                adapter_dir = _os.path.join(save_dir, "adapter")
                
                # LoRA matrices + modules_to_save copies, in PEFT's saved key layout
                from razordl.ops.model.peft import adapter_state_dict_for_saving

                adapter_state_dict = adapter_state_dict_for_saving(state_dict)

                # 保存 adapter
                if adapter_state_dict:
                    from safetensors.torch import save_file as _save_safetensors
                    adapter_path = _os.path.join(adapter_dir, "adapter_model.safetensors")
                    _save_safetensors(adapter_state_dict, adapter_path)
                    print(f"[SAVE] LoRA adapter saved: {adapter_path} ({len(adapter_state_dict)} tensors)")
                    
                    # 计算 adapter 大小
                    adapter_size_mb = sum(v.numel() * v.element_size() for v in adapter_state_dict.values()) / (1024 * 1024)
                    print(f"[SAVE] LoRA adapter size: {adapter_size_mb:.2f} MB")
                    
                    # 保存 adapter config (如果模型有的话)
                    try:
                        unwrapped = model
                        while hasattr(unwrapped, '_fsdp_wrapped_module'):
                            unwrapped = unwrapped._fsdp_wrapped_module
                        if hasattr(unwrapped, 'peft_config'):
                            import json
                            # 兼容 vllm: 如果只有一个 adapter，直接保存其 config 内容
                            adapter_config_to_save = {}
                            is_single_adapter = len(unwrapped.peft_config) == 1
                            
                            if is_single_adapter:
                                # 直接取第一个 adapter 的配置
                                config = list(unwrapped.peft_config.values())[0]
                                config_dict = config.to_dict()
                                for k, v in config_dict.items():
                                    if isinstance(v, (str, int, float, bool, list, dict, type(None))):
                                        adapter_config_to_save[k] = v
                                    elif isinstance(v, (set, tuple)):
                                        adapter_config_to_save[k] = list(v)
                                    else:
                                        try:
                                            adapter_config_to_save[k] = str(v)
                                        except Exception:
                                            pass
                            else:
                                # 原有逻辑：保留 adapter name 嵌套
                                for key, config in unwrapped.peft_config.items():
                                    config_dict = config.to_dict()
                                    cleaned_config = {}
                                    for k, v in config_dict.items():
                                        if isinstance(v, (str, int, float, bool, list, dict, type(None))):
                                            cleaned_config[k] = v
                                        elif isinstance(v, (set, tuple)):
                                            cleaned_config[k] = list(v)
                                        else:
                                            try:
                                                cleaned_config[k] = str(v)
                                            except Exception:
                                                pass
                                    adapter_config_to_save[key] = cleaned_config

                            # 保存到文件
                            config_path = _os.path.join(adapter_dir, "adapter_config.json")
                            with open(config_path, "w") as f:
                                json.dump(adapter_config_to_save, f, indent=2, ensure_ascii=False)
                            print(f"[SAVE] LoRA config saved to {config_path}")
                    except Exception as e:
                        print(f"[SAVE WARNING] Could not save adapter config: {e}")
                        import traceback
                        traceback.print_exc()
                else:
                    print(f"[SAVE WARNING] No LoRA parameters found in state_dict!")
                
            except Exception as e:
                print(f"[SAVE WARNING] Failed to save LoRA adapter separately: {e}")
                print("[SAVE] Will try to save full model only")
                # 如果保存 adapter 失败，强制保存完整模型
                if not save_full_model:
                    print("[SAVE WARNING] Forcing save_full_model=True due to adapter save failure")
                    save_full_model = True
        
        # 如果不是 LoRA 模型，必须保存完整模型
        if not is_peft_model and not save_full_model:
            print("[SAVE WARNING] Not a PEFT model, ignoring save_full_model=False")
            save_full_model = True
        
        # 调试信息
        if "embed_tokens.weight" in state_dict:
            print(f"[SAVE DEBUG] to: {save_dir}, embed_tokens.weight: {tuple(state_dict['embed_tokens.weight'].shape)}")
        
        # 保存完整模型（如果需要）
        if save_full_model:
            try:
                from safetensors.torch import save_file as _save_safetensors
                # 保存config
                try:
                    model.config.save_pretrained(save_dir)
                except Exception:
                    with open(_os.path.join(save_dir, "config.json"), "w") as _f:
                        _f.write(model.config.to_json_string())
                # 原子写入
                _tmp_path = _os.path.join(save_dir, "model.safetensors.tmp")
                _final_path = _os.path.join(save_dir, "model.safetensors")
                _save_safetensors(state_dict, _tmp_path)
                _os.replace(_tmp_path, _final_path)
                
                # 计算完整模型大小
                full_size_mb = sum(v.numel() * v.element_size() for v in state_dict.values()) / (1024 * 1024)
                print(f"[SAVE] Full model saved: {_final_path} ({full_size_mb:.2f} MB)")
                
                # 验证
                try:
                    from safetensors.torch import load_file as _load_safetensors
                    _loaded = _load_safetensors(_final_path)
                    if "embed_tokens.weight" in _loaded:
                        print(f"[SAVE VERIFY] from: {save_dir}, embed_tokens.weight: {tuple(_loaded['embed_tokens.weight'].shape)}")
                except Exception:
                    if _os.path.exists(_final_path):
                        _os.unlink(_final_path)
                    raise RuntimeError(
                        f"Saved model file is unreadable: {_final_path}. "
                        f"Check disk space and permissions."
                    )
            except Exception as _e:
                print(f"[SAVE ERROR] {_e}, falling back to save_pretrained")
                model.save_pretrained(save_dir, state_dict=state_dict, safe_serialization=True)
        else:
            # 只保存 adapter，但仍需保存 config
            print(f"[SAVE] Skipping full model save (save_full_model=False)")
            try:
                model.config.save_pretrained(save_dir)
            except Exception:
                with open(_os.path.join(save_dir, "config.json"), "w") as _f:
                    _f.write(model.config.to_json_string())
            print(f"[SAVE] Config saved to {save_dir}")
        
    
    # 最终同步
    if dist.is_available() and dist.is_initialized():
        dist.barrier()


def save_processor_fsdp2(processor, save_dir: str) -> None:

    # 同步
    if dist.is_available() and dist.is_initialized():
        dist.barrier()
    if is_global_rank0():
        # 保存 processor 或 tokenizer (自动检测)
        try:
            processor.save_pretrained(save_dir)
            # 检测是 processor 还是 tokenizer
            if hasattr(processor, 'image_processor') or hasattr(processor, 'feature_extractor'):
                print(f"[SAVE] Processor saved to {save_dir}")
            else:
                print(f"[SAVE] Tokenizer saved to {save_dir}")
        except Exception as e:
            print(f"[SAVE ERROR] Failed to save processor/tokenizer: {e}")
    
    # 最终同步
    if dist.is_available() and dist.is_initialized():
        dist.barrier()


def save_optimizer_fsdp2(model, optimizer=None, save_dir: str | None = None):
    if save_dir is None and isinstance(optimizer, (str, _os.PathLike)):
        save_dir = optimizer
        optimizer = model
        model = None
    elif optimizer is None:
        optimizer = model
        model = None

    rank0 = is_global_rank0()
    # global rank 0 创建目录
    if rank0:
        _os.makedirs(save_dir, exist_ok=True)
    
    # 同步
    if dist.is_available() and dist.is_initialized():
        dist.barrier()
    
    # 获取优化器 state_dict
    try:
        # FSDP2 优化器可能需要特殊处理
        from torch.distributed.checkpoint.state_dict import get_optimizer_state_dict, StateDictOptions
        if model is None:
            raise ValueError("model is required for FSDP2 optimizer state_dict")
        
        optimizer_state = get_optimizer_state_dict(
            model,
            optimizer,
            options=StateDictOptions(
                full_state_dict=True,
                cpu_offload=True,
            )
        )
        
        if rank0:
            print("[SAVE] Using FSDP2 get_optimizer_state_dict API")
    except Exception as e:
        if model is not None:
            raise RuntimeError("Failed to save FSDP2 optimizer state_dict") from e
        # 回退到标准方法
        optimizer_state = optimizer.state_dict()
        
        if rank0:
            print(f"[SAVE] Using standard optimizer.state_dict()")
            # print(f"[SAVE DEBUG] FSDP2 API not available: {e}")
    
    # 只有 rank0 保存文件
    if rank0:
        # 保存优化器状态
        optimizer_path = _os.path.join(save_dir, "optimizer.pt")
        torch.save(optimizer_state, optimizer_path)
        print(f"[SAVE] Optimizer saved to {optimizer_path}")
    
    # 最终同步
    if dist.is_available() and dist.is_initialized():
        dist.barrier()
