"""Once-per-job writes are gated on the *global* rank.

`LOCAL_RANK == 0` holds on every node, so on a multi-node run each node's
first worker wrote checkpoint_info.json, the tokenizer, scaler.pt and raced
the trainer's atomic rename. Dist is never initialised here; the env
fallback (`RANK`, then `LOCAL_RANK`) is what the test exercises.
"""

import os
from types import SimpleNamespace

import pytest
import torch.distributed as dist

from razordl.core.base.trainer import BaseTrainer
from razordl.core.engine.common.modelgroup import ParallelModelGroup
from razordl.core.engine.common.parallel_backend import DDPBackend
from razordl.ops.distributed import utils as dist_utils
from razordl.ops.parallel.fsdp2 import save_processor_fsdp2


@pytest.fixture
def second_node_first_gpu(monkeypatch):
    monkeypatch.setenv("RANK", "1")
    monkeypatch.setenv("LOCAL_RANK", "0")


def test_global_rank_precedence(monkeypatch):
    monkeypatch.delenv("RANK", raising=False)
    monkeypatch.delenv("LOCAL_RANK", raising=False)
    assert dist_utils.get_global_rank() == 0
    monkeypatch.setenv("LOCAL_RANK", "3")
    assert dist_utils.get_global_rank() == 3  # single node: LOCAL_RANK is the global rank
    monkeypatch.setenv("RANK", "5")
    assert dist_utils.get_global_rank() == 5
    monkeypatch.setattr(dist, "is_initialized", lambda: True)
    monkeypatch.setattr(dist, "get_rank", lambda: 7)
    assert dist_utils.get_global_rank() == 7  # the process group wins once it exists
    assert not dist_utils.is_global_rank0()


def test_processor_not_saved_off_rank0(second_node_first_gpu, tmp_path):
    calls = []
    processor = SimpleNamespace(save_pretrained=lambda d: calls.append(d))
    save_processor_fsdp2(processor, str(tmp_path / "proc"))
    assert calls == []


def test_processor_saved_on_rank0(monkeypatch, tmp_path):
    monkeypatch.setenv("RANK", "0")
    monkeypatch.setenv("LOCAL_RANK", "0")
    calls = []
    processor = SimpleNamespace(save_pretrained=lambda d: calls.append(d))
    save_processor_fsdp2(processor, str(tmp_path / "proc"))
    assert calls == [str(tmp_path / "proc")]


def test_grad_scaler_not_saved_off_rank0(second_node_first_gpu, tmp_path):
    mg = SimpleNamespace(scaler=SimpleNamespace(state_dict=lambda: {"scale": 1.0}), local_rank=0)
    ParallelModelGroup.save_grad_scaler(mg, str(tmp_path / "wg"))
    assert not (tmp_path / "wg").exists()


def test_ddp_backend_rank_is_global(second_node_first_gpu):
    backend = DDPBackend.__new__(DDPBackend)
    assert backend._rank() == 1


def test_atomic_checkpoint_off_rank0_touches_nothing(second_node_first_gpu, tmp_path):
    trainer = BaseTrainer.__new__(BaseTrainer)
    saved = []
    trainer.save_checkpoint = lambda d: saved.append(d)  # the collective part every rank runs
    ckpt = tmp_path / "checkpoint_000010"
    BaseTrainer._save_checkpoint_atomic(trainer, str(ckpt), completed_step=10)
    assert saved == [str(ckpt) + ".tmp"]
    assert sorted(os.listdir(tmp_path)) == []  # no .tmp dir, no rename, no info file
