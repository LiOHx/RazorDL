from abc import abstractmethod

import torch

from razordl.core.base.metrics import DistStats
from razordl.core.base.workgroup import BaseModelGroup
from razordl.core.engine.common.workgroup import EngineWorkGroup
from razordl.core.engine.on_policy_single_model.config import Config, WorkerGroupConfig


class WorkGroup(EngineWorkGroup):
    """WorkGroup template for on-policy single-model training.

    Presets provide rollout/reward/advantage/loss hooks.  The shared engine
    wrapper handles seeding, offload boundaries, gradient clipping, and
    optimizer stepping.
    """

    config: Config
    worker_group_config: WorkerGroupConfig

    @abstractmethod
    def rollout(self, input_dict: dict, step: int) -> dict:
        pass

    @abstractmethod
    def compute_reward(self, rollout_output: dict) -> torch.Tensor:
        pass

    @abstractmethod
    def compute_advantage(self, rewards: torch.Tensor) -> torch.Tensor:
        pass

    @abstractmethod
    def compute_loss(self, rollout_output: dict, advantages: torch.Tensor) -> torch.Tensor:
        pass

    def _run_update_step(self, input_dict: dict, step: int) -> dict:
        policy_group = self._get_policy_model_group()
        with policy_group.inference_context():
            rollout_output = self.rollout(input_dict, step)

        rewards = self.compute_reward(rollout_output)
        advantages = self.compute_advantage(rewards)

        ref_group = self._get_reference_model_group()
        with policy_group.trainer_context():
            if ref_group is not None:
                with ref_group.inference_context():
                    loss = self.compute_loss(rollout_output, advantages)
            else:
                loss = self.compute_loss(rollout_output, advantages)
            loss.backward()

        return {
            "loss": loss.item(),
            "reward": DistStats.from_tensor(rewards),
            "advantage": DistStats.from_tensor(advantages),
        }

    def _get_policy_model_group(self) -> BaseModelGroup:
        for name in ("policy_model_group", "policy"):
            if hasattr(self, name):
                return getattr(self, name)
        for _name, obj in self.__dict__.items():
            if isinstance(obj, BaseModelGroup) and getattr(obj, "optimizer", None) is not None:
                return obj
        raise RuntimeError("No trainable policy ModelGroup found in WorkGroup")

    def _get_reference_model_group(self) -> BaseModelGroup | None:
        for name in ("reference_model_group", "reference"):
            if hasattr(self, name):
                return getattr(self, name)
        return None
