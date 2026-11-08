"""SFTDataset contracts that do not need a model download."""

import json

import pytest

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


class _StreamingTokenizer:
    chat_template = "{{ messages }}"
    pad_token = "<pad>"
    eos_token = "<eos>"
    padding_side = "left"

    def apply_chat_template(
        self, messages, tokenize=False, tools=None, add_generation_prompt=False
    ):
        return " ".join(message["content"] for message in messages)

    def encode(self, text, add_special_tokens=False):
        return [1] * len(text.split())


def _streaming_config(data_dir, streaming=True):
    config = type("C", (), {})()
    config.data_config = type(
        "D",
        (),
        {
            "train_data_path": str(data_dir),
            "max_length": 16,
            "sp_size": 1,
            "dataset_processor_path": "tokenizer",
            "streaming": streaming,
        },
    )()
    config.worker_group_config = type("W", (), {})()
    config.worker_group_config.model_group_config = type("M", (), {})()
    config.worker_group_config.model_group_config.model_config = type(
        "MC", (), {"model_path": "base-model"}
    )()
    config.worker_group_config.model_group_config.processor_config = type(
        "PC",
        (),
        {
            "processor_path": "shared-tokenizer",
            "lazy_tokenize": False,
            "cache_tokenized": True,
        },
    )()
    return config


def test_streaming_jsonl_keeps_offsets_and_tokenizes_on_demand(tmp_path, monkeypatch):
    import razordl.presets.sft.dataset as dataset_module

    data_dir = tmp_path / "data"
    data_dir.mkdir()
    rows = [
        {"messages": [{"role": "user", "content": "hello"}, {"role": "assistant", "content": "world"}]},
        {"messages": [{"role": "user", "content": "what"}, {"role": "assistant", "content": "is"}]},
        {"messages": [{"role": "user", "content": "two"}, {"role": "assistant", "content": "four"}]},
    ]
    with open(data_dir / "train.jsonl", "w") as f:
        for row in rows:
            f.write(json.dumps(row) + "\n")
        f.write("\n")  # blank lines are not samples

    monkeypatch.setattr(
        dataset_module.AutoTokenizer,
        "from_pretrained",
        lambda *args, **kwargs: _StreamingTokenizer(),
    )

    dataset = SFTDataset(_streaming_config(data_dir))

    assert dataset.streaming
    assert dataset.raw_data is None
    assert len(dataset) == 3
    assert not (data_dir / ".cache").exists()

    item = dataset[1]
    assert item["input_ids"]
    assert len(item["input_ids"]) == len(item["attention_mask"])
    assert len(item["labels"]) == len(item["input_ids"])


def test_streaming_rejects_non_jsonl(tmp_path, monkeypatch):
    import razordl.presets.sft.dataset as dataset_module

    data_dir = tmp_path / "data"
    data_dir.mkdir()
    (data_dir / "train.json").write_text(json.dumps([]))

    monkeypatch.setattr(
        dataset_module.AutoTokenizer,
        "from_pretrained",
        lambda *args, **kwargs: _StreamingTokenizer(),
    )

    with pytest.raises(ValueError, match="requires a .jsonl"):
        SFTDataset(_streaming_config(data_dir))


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
