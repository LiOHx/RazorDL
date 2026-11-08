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


def test_cache_disabled_returns_no_cache_path(tmp_path, monkeypatch):
    ds = _dataset(SFTDataset, tmp_path, monkeypatch)
    assert ds._cache_path() is not None
    ds.cache_tokenized = False
    assert ds._cache_path() is None


def test_cache_key_changes_with_processor_path(tmp_path, monkeypatch):
    def with_processor(path):
        ds = _dataset(SFTDataset, tmp_path, monkeypatch)
        ds.config.worker_group_config.model_group_config.processor_config = type(
            "P", (), {"processor_path": path}
        )()
        return ds

    assert with_processor("tokenizer-a")._cache_path() != with_processor("tokenizer-b")._cache_path()


def test_dataset_loads_tokenizer_from_resolved_processor_path(tmp_path, monkeypatch):
    import razordl.presets.sft.dataset as dataset_module

    data = tmp_path / "train.jsonl"
    data.write_text(json.dumps({"messages": []}) + "\n")

    loaded_paths = []

    class FakeTokenizer:
        chat_template = "{{ messages }}"
        pad_token = "<pad>"
        eos_token = "<eos>"
        padding_side = "left"

        @classmethod
        def from_pretrained(cls, path, **kwargs):
            loaded_paths.append(path)
            return cls()

    monkeypatch.setattr(dataset_module.AutoTokenizer, "from_pretrained", FakeTokenizer.from_pretrained)

    config = type("C", (), {})()
    config.data_config = type(
        "D",
        (),
        {
            "train_data_path": str(tmp_path),
            "max_length": 16,
            "sp_size": 1,
            "dataset_processor_path": "dataset-tokenizer",
        },
    )()
    config.worker_group_config = type("W", (), {})()
    config.worker_group_config.model_group_config = type("M", (), {})()
    config.worker_group_config.model_group_config.model_config = type(
        "MC", (), {"model_path": "base-model"}
    )()
    config.worker_group_config.model_group_config.processor_config = type(
        "PC", (), {"processor_path": "shared-tokenizer", "lazy_tokenize": True}
    )()

    SFTDataset(config)

    assert loaded_paths == ["shared-tokenizer"]
