"""razordl-independent utilities (distributed, parallel, model, loss, hardware, ...).

Intentionally empty: import from the submodules. Re-exporting peft / FSDP2
/ sequence-parallel helpers here made ``import razordl.ops.anything`` (and
thus ``razordl --help``) import torch and peft.
"""
