"""Normalise LoRA target-module settings from ``config.yaml``.

``lora_target_modules`` accepts three spellings so a config copied from
LLaMA Factory keeps working:

- a list of module names (``[q_proj, v_proj]``)
- a comma-separated string (``"q_proj,v_proj"``)
- ``all`` / ``all-linear`` -- hand PEFT its native ``"all-linear"`` wildcard,
  which expands to every ``nn.Linear``/``Conv1D`` layer except the output
  head.  This is what reaches Qwen3.5's MLP and hybrid linear-attention
  projections, which the default q/v/k/o list never touches.

The canonical form stored on ``BaseAdapterConfig`` is a ``list[str]`` or the
single string ``"all-linear"``.
"""

ALL_LINEAR = "all-linear"
_ALL_ALIASES = {"all", "all-linear"}
DEFAULT_TARGETS = ("q_proj", "v_proj", "k_proj", "o_proj")


def normalize_lora_targets(value, *, default=DEFAULT_TARGETS):
    """Return a list of module names, or the ``"all-linear"`` wildcard."""
    if value is None:
        return list(default)

    if isinstance(value, str):
        text = value.strip()
        if not text:
            raise ValueError("lora_target_modules must not be empty")
        if text.lower() in _ALL_ALIASES:
            return ALL_LINEAR
        names = [part.strip() for part in text.split(",") if part.strip()]
        if not names:
            raise ValueError("lora_target_modules must not be empty")
    elif isinstance(value, (list, tuple, set)):
        names = [str(item).strip() for item in value if str(item).strip()]
        if not names:
            raise ValueError("lora_target_modules must not be empty")
    else:
        raise ValueError(
            "lora_target_modules must be a list of module names or a string "
            f"(e.g. all-linear), got {type(value).__name__}"
        )

    all_like = [name for name in names if name.lower() in _ALL_ALIASES]
    if all_like:
        if len(names) != 1:
            raise ValueError(
                "'all'/'all-linear' must be the only lora_target_modules entry; "
                f"got {names}"
            )
        return ALL_LINEAR
    return names


def normalize_lora_exclude_modules(value):
    """``None``, a regex string, or a list of module names/suffixes."""
    if value is None:
        return None
    if isinstance(value, str):
        text = value.strip()
        return text or None
    if isinstance(value, (list, tuple, set)):
        names = [str(item).strip() for item in value if str(item).strip()]
        return names or None
    raise ValueError(
        "lora_exclude_modules must be a regex string or a list of module "
        f"names, got {type(value).__name__}"
    )
