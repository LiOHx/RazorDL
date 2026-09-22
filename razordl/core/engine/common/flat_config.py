def _resolve_precision_key(d: dict) -> str:
    """Read the `precision` flat key, honouring the deprecated `use_bf16` boolean.

    `precision` wins when both are present -- an explicit new-style key is never
    overridden by a stale one left in the same file.
    """
    import warnings

    from razordl.ops.hardware.precision import (
        PRECISION_CHOICES,
        legacy_use_bf16_to_precision,
    )

    precision = d.get("precision", None)
    use_bf16 = d.get("use_bf16", None)

    if precision is not None and use_bf16 is not None:
        warnings.warn(
            f"Both 'precision' and the deprecated 'use_bf16' are set; using "
            f"precision={precision!r} and ignoring use_bf16={use_bf16!r}. "
            f"Remove 'use_bf16' from your config.",
            DeprecationWarning,
            stacklevel=2,
        )
    elif precision is None and use_bf16 is not None:
        precision = legacy_use_bf16_to_precision(use_bf16)
        warnings.warn(
            f"'use_bf16' is deprecated; use 'precision: auto|bf16|fp16|fp32'. "
            f"Mapped use_bf16={use_bf16!r} to precision={precision!r}. Note that "
            f"use_bf16=False maps to fp16, while 'auto' would pick bf16 on any "
            f"CUDA GPU and fp32 without one.",
            DeprecationWarning,
            stacklevel=2,
        )
    elif precision is None:
        precision = "auto"

    precision = str(precision).lower()
    if precision not in PRECISION_CHOICES:
        raise ValueError(
            f"precision must be one of {list(PRECISION_CHOICES)}, got {precision!r}"
        )
    return precision


def build_single_model_config_dict(
    d: dict,
    *,
    data_config: dict,
    model_default: str,
    processor_max_length: int,
    lr_default: float = 5e-5,
    weight_decay_default: float = 0.01,
    max_grad_norm_default: float | None = 1.0,
    grad_accum_default: int = 2,
    batch_size_default: int = 1,
    num_epochs_default: int = 3,
    num_workers_default: int = 2,
    log_steps_default: int = 10,
    save_steps_default: int = 100,
    save_ckpt_steps_default: int = 500,
    task_type_default: str | None = None,
    lazy_tokenize_default: bool = False,
) -> dict:
    """Build the shared nested config dict from the public flat YAML schema."""

    model_path = d.get("model", model_default)
    outputs_dir = d.get("outputs_dir", "./outputs")
    resume_mode = d.get("resume_mode", "auto")
    resume_from = d.get("resume_from", None)
    init_from = d.get("init_from", None)

    # Legacy support: auto-migrate old output_dir → outputs_dir
    output_dir = None
    if "output_dir" in d and "outputs_dir" not in d:
        import warnings
        warnings.warn(
            "config.yaml uses deprecated 'output_dir'; please rename to 'outputs_dir' "
            "(the framework now auto-creates timestamped experiment dirs under outputs_dir). "
            "Your old output_dir value has been migrated to outputs_dir.",
            DeprecationWarning,
        )
        outputs_dir = d["output_dir"]
    elif "output_dir" in d:
        # output_dir NEXT TO outputs_dir is only ever written by
        # ops/snapshot.py::snapshot_code into <exp>/code/config.yaml.  It pins
        # the run to the original experiment dir so a copied code/ snapshot
        # resumes those checkpoints instead of starting a new experiment.
        output_dir = d["output_dir"]

    # Legacy support: auto_resume → resume_mode
    if "auto_resume" in d and "resume_mode" not in d:
        resume_mode = "auto" if d["auto_resume"] else "manual"

    use_adapter = d.get("use_adapter", True)
    lora_r = d.get("lora_r", 8)
    lora_alpha = d.get("lora_alpha", 16)
    lora_dropout = d.get("lora_dropout", 0.05)
    lora_target_modules = d.get("lora_target_modules", ["q_proj", "v_proj", "k_proj", "o_proj"])
    adapter_name = d.get("adapter_name", "default")
    adapter_path = d.get("adapter_path", None)
    modules_to_save = d.get("modules_to_save", None)
    task_type = d.get("task_type", task_type_default)

    lr = d.get("lr", lr_default)
    weight_decay = d.get("weight_decay", weight_decay_default)
    max_grad_norm = d.get("max_grad_norm", max_grad_norm_default)
    grad_accum = d.get("grad_accum", grad_accum_default)
    optimizer_path = d.get("optimizer_path", None)
    scheduler_path = d.get("scheduler_path", None)

    batch_size = d.get("batch_size", batch_size_default)
    num_epochs = d.get("num_epochs", num_epochs_default)
    num_workers = d.get("num_workers", num_workers_default)
    seed = d.get("seed", 42)
    enable_gradient_checkpointing = d.get("enable_gradient_checkpointing", False)
    enable_activation_offload = d.get("enable_activation_offload", False)

    processor_path = d.get("processor_path", model_path)
    sp_size = d.get("sp_size", data_config.get("sp_size", 1))
    lazy_tokenize = d.get("lazy_tokenize", lazy_tokenize_default)

    offload_param = d.get("offload_param", False)
    offload_optimizer = d.get("offload_optimizer", False)

    log_steps = d.get("log_steps", log_steps_default)
    save_steps = d.get("save_steps", save_steps_default)
    save_ckpt_steps = d.get("save_ckpt_steps", save_ckpt_steps_default)
    resume_checkpoint_dir = d.get("resume_checkpoint_dir", None)
    compute_checksums = d.get("compute_checksums", False)

    precision = _resolve_precision_key(d)
    parallel_backend = d.get("parallel_backend", "fsdp2")
    fused_linear_tile_size = d.get("fused_linear_tile_size", d.get("chunk_size", 2048))
    if "chunk_size" in d or "chunked_loss" in d:
        import warnings
        warnings.warn(
            "'chunked_loss'/'chunk_size' retired: the lm_head CE always streams "
            "through FusedLinearCrossEntropy now.  Use "
            f"'fused_linear_tile_size' (mapped chunk_size={fused_linear_tile_size} "
            "for this run); the 'chunked_loss' flag is a no-op.",
            DeprecationWarning,
            stacklevel=2,
        )
    ray_kwargs = d.get("ray_kwargs", {})

    model_group_name = d.get("model_group_name", None)
    worker_group_name = d.get("worker_group_name", None)

    data_config = {**data_config, "sp_size": sp_size}
    return {
        "data_config": data_config,
        "worker_group_config": {
            "worker_group_name": worker_group_name,
            "model_group_config": {
                "model_group_name": model_group_name,
                "processor_config": {
                    "processor_path": processor_path,
                    "max_length": processor_max_length,
                    "lazy_tokenize": lazy_tokenize,
                },
                "model_config": {
                    "model_path": model_path,
                    "micro_batch_size_per_gpu": batch_size,
                    "enable_gradient_checkpointing": enable_gradient_checkpointing,
                    "enable_activation_offload": enable_activation_offload,
                    "sp_size": sp_size,
                    "precision": precision,
                    "parallel_backend": parallel_backend,
                    "fused_linear_tile_size": fused_linear_tile_size,
                    "_is_offload_param": offload_param,
                    "_is_offload_optimizer": offload_optimizer,
                    "adapter_config": {
                        "use_adapter": use_adapter,
                        "adapter_name": adapter_name,
                        "adapter_path": adapter_path,
                        "lora_r": lora_r,
                        "lora_alpha": lora_alpha,
                        "lora_dropout": lora_dropout,
                        "lora_target_modules": lora_target_modules,
                        "modules_to_save": modules_to_save,
                        "task_type": task_type,
                    },
                },
                "optimizer_config": {
                    "optimizer_path": optimizer_path,
                    "learning_rate": lr,
                    "weight_decay": weight_decay,
                    "max_grad_norm": max_grad_norm,
                    "accumulate_grad_steps": grad_accum,
                },
                "scheduler_config": {
                    "scheduler_path": scheduler_path,
                },
            },
        },
        "trainer_config": {
            "ray_kwargs": ray_kwargs,
            "step_batch_size": batch_size,
            "data_loader_num_workers": num_workers,
            "num_epochs": num_epochs,
            "outputs_dir": outputs_dir,
            "resume_mode": resume_mode,
            "resume_from": resume_from,
            "init_from": init_from,
            "output_dir": output_dir,  # None unless pinned by a code snapshot; else set by main()
            "resume_checkpoint_dir": resume_checkpoint_dir,
            "log_info_steps": log_steps * grad_accum,
            "save_model_steps": save_steps * grad_accum,
            "save_checkpoint_steps": save_ckpt_steps * grad_accum,
            "seed": seed,
            "compute_checksums": compute_checksums,
        },
    }
