from razordl.presets.grpo.config import GRPOConfig
from razordl.presets.grpo.workgroup import GRPOWorkGroup
from razordl.presets.grpo.dataset import GRPODataset, GRPOCollator

# Engine declaration — CLI uses this to determine which engine to load.
# "single_model" | "on_policy_single_model"
ENGINE = "on_policy_single_model"

# Dataset declaration — CLI copies this dataset from razordl/datasets/ to the project.
DATASET = "gsm8k"

# CLI convention: {CamelCase}Config (e.g. GrpoConfig from "grpo")
GrpoConfig = GRPOConfig
GrpoWorkGroup = GRPOWorkGroup
GrpoDataset = GRPODataset
GrpoCollator = GRPOCollator

__all__ = [
    "GRPOConfig",
    "GRPOWorkGroup",
    "GRPODataset",
    "GRPOCollator",
    "GrpoConfig",
    "GrpoWorkGroup",
    "GrpoDataset",
    "GrpoCollator",
    "ENGINE",
    "DATASET",
]
