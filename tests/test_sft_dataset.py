"""SFTDataset contracts that do not need a model download."""

import json

from razordl.presets.sft.dataset import SFTDataset


class _Tok:
    chat_template = "{{ messages }}"
    pad_token = "<pad>"
    eos_token = "<eos>"
    padding_side = "left"


def _dataset(cls, tmp_path, monkeypatch):
    data = tmp_path / "train.jsonl"
    if not data.exists():  # keep the mtime stable across calls
        data.write_text(json.dumps({"messages": []}) + "\n")
    ds = cls.__new__(cls)
    ds.data_config = type("D", (), {"train_data_path": str(tmp_path)})()
    ds.config = type("C", (), {})()
    ds.config.worker_group_config = type("W", (), {})()
    ds.config.worker_group_config.model_group_config = type("M", (), {})()
    ds.config.worker_group_config.model_group_config.model_config = type("MC", (), {"model_path": "m"})()
    ds.max_length = 16
    ds.tokenizer = _Tok()
    return ds


def test_cache_key_changes_with_format_item_and_chat_template(tmp_path, monkeypatch):
    class Custom(SFTDataset):
        def format_item(self, item):
            return {"messages": item["conv"], "tools": None}

    base = _dataset(SFTDataset, tmp_path, monkeypatch)
    custom = _dataset(Custom, tmp_path, monkeypatch)
    assert base._cache_path() != custom._cache_path()

    other_template = _dataset(SFTDataset, tmp_path, monkeypatch)
    other_template.tokenizer = type("T2", (_Tok,), {"chat_template": "{{ other }}"})()
    assert base._cache_path() != other_template._cache_path()

    # Same inputs -> same key (the hash is deterministic, not id-based).
    assert base._cache_path() == _dataset(SFTDataset, tmp_path, monkeypatch)._cache_path()
