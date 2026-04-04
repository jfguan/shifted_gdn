from __future__ import annotations

import os
from collections.abc import Callable, Iterator
from dataclasses import dataclass
from enum import Enum
from functools import partial


class DatasetName(str, Enum):
    PG19 = "pg19"
    THE_STACK = "the_stack"


from datasets import load_dataset as hf_load, interleave_datasets
import numpy as np
import numpy.typing as npt
import torch
from tokenizers import Tokenizer, models, pre_tokenizers, trainers


@dataclass
class DatasetConfig:
    cache_dir: str
    vocab_size: int

    bpe_train_chars: int

    train_chars: int
    val_chars: int

    # callable args: char_target and split
    stream_train: Callable[[int], str]
    stream_val: Callable[[int], str]


def _stream_pg19(char_target: int, hf_split: str) -> str:
    ds = hf_load("emozilla/pg19", split=hf_split, streaming=True)
    texts = (row["text"] for row in ds)

    return _collect_chunks(texts, char_target)


def _stream_stack(char_target: int, seed: int) -> str:
    streams = []

    languages = ["python", "javascript", "typescript", "java", "c", "cpp", "rust", "go"]
    for lang in languages:
        ds = hf_load(
            "bigcode/the-stack-dedup",
            data_dir=f"data/{lang}",
            split="train",
            streaming=True,
        )
        streams.append(ds.select_columns(["content"]))

    combined = interleave_datasets(streams, seed=seed)

    # Filter for only files above 32_000 character length
    texts = (row["content"] for row in combined if len(row["content"]) >= 32_000)
    return _collect_chunks(texts, char_target)


def _collect_chunks(texts: Iterator[str], char_target: int) -> str:
    """Accumulate text chunks from an iterator until char_target is reached."""
    chunks, char_total = [], 0
    while char_total < char_target:
        text = next(texts)

        chunks.append(text)
        char_total += len(text)

        if len(chunks) % 100 == 0:
            print(f"  {len(chunks)} items, {char_total:,} chars...", flush=True)

    print(f"  {len(chunks)} items, {char_total:,} chars (done)")
    return "\n\n".join(chunks)


DATA_DIR = os.path.dirname(os.path.abspath(__file__))

DATASETS: dict[str, DatasetConfig] = {
    "pg19": DatasetConfig(
        cache_dir=os.path.join(DATA_DIR, "pg19"),
        vocab_size=1024,
        train_chars=80_000_000,
        val_chars=4_000_000,
        bpe_train_chars=5_000_000,
        stream_train=partial(_stream_pg19, hf_split="train"),
        stream_val=partial(_stream_pg19, hf_split="validation"),
    ),
    "the_stack": DatasetConfig(
        cache_dir=os.path.join(DATA_DIR, "the_stack"),
        vocab_size=1024,
        train_chars=2_000_000_000,
        val_chars=20_000_000,
        bpe_train_chars=10_000_000,
        stream_train=partial(_stream_stack, seed=42),
        stream_val=partial(_stream_stack, seed=1337),
    ),
}


class Dataset:
    """Result of load_dataset: holds token arrays, vocab size, and encode/decode methods."""

    def __init__(
        self,
        train: npt.NDArray[np.uint16],
        val: npt.NDArray[np.uint16],
        vocab_size: int,
        tokenizer: Tokenizer,
    ) -> None:
        self.train = train
        self.val = val
        self.vocab_size = vocab_size
        self._tokenizer = tokenizer

    def encode(self, text: str) -> list[int]:
        return self._tokenizer.encode(text).ids

    def decode(self, ids: list[int]) -> str:
        return self._tokenizer.decode(ids)


def load_dataset(name: str = "pg19") -> Dataset:
    """Load (or download + tokenize + cache) a dataset."""
    if name not in DATASETS:
        raise ValueError(f"Unknown dataset: {name!r}. Choose from {list(DATASETS)}")

    cfg = DATASETS[name]
    os.makedirs(cfg.cache_dir, exist_ok=True)

    tokenizer_path = os.path.join(cfg.cache_dir, "tokenizer.json")
    train_path = os.path.join(cfg.cache_dir, "train_tokens.npy")
    val_path = os.path.join(cfg.cache_dir, "val_tokens.npy")

    # Try cache first
    if (
        os.path.exists(tokenizer_path)
        and os.path.exists(train_path)
        and os.path.exists(val_path)
    ):
        tokenizer = Tokenizer.from_file(tokenizer_path)
        if tokenizer.get_vocab_size() == cfg.vocab_size:
            return Dataset(
                np.load(train_path, mmap_mode="r"), np.load(val_path, mmap_mode="r"), cfg.vocab_size, tokenizer
            )

    # Load or train tokenizer
    if os.path.exists(tokenizer_path):
        print(f"Loading existing tokenizer from {tokenizer_path}")
        tokenizer = Tokenizer.from_file(tokenizer_path)
    else:
        bpe_text = cfg.stream_train(cfg.bpe_train_chars)
        tokenizer = _train_tokenizer(bpe_text, cfg.vocab_size)
        tokenizer.save(tokenizer_path)
        del bpe_text

    # Download data
    train_text = cfg.stream_train(cfg.train_chars)
    val_text = cfg.stream_val(cfg.val_chars)

    # Tokenize data
    train_data = _tokenize(tokenizer, train_text, "train")
    np.save(train_path, train_data)

    val_data = _tokenize(tokenizer, val_text, "val")
    np.save(val_path, val_data)

    print(f"Cached to {cfg.cache_dir}")
    return Dataset(train_data, val_data, cfg.vocab_size, tokenizer)


def _train_tokenizer(text: str, vocab_size: int) -> Tokenizer:
    print(f"Training BPE tokenizer (vocab_size={vocab_size}) on {len(text):,} chars...")

    tokenizer = Tokenizer(models.BPE())
    tokenizer.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=False)

    trainer = trainers.BpeTrainer(vocab_size=vocab_size, special_tokens=[])
    tokenizer.train_from_iterator(text.splitlines(), trainer=trainer)

    return tokenizer


def _tokenize(tokenizer: Tokenizer, text: str, label: str, chunk_size: int = 10_000_000) -> npt.NDArray[np.uint16]:
    from tqdm import tqdm

    n_chars = len(text)
    all_ids = []

    chunks = range(0, n_chars, chunk_size)
    for start in tqdm(chunks, desc=f"Tokenizing {label}", unit="chunk"):
        end = min(start + chunk_size, n_chars)
        all_ids.extend(tokenizer.encode(text[start:end]).ids)

    data = np.array(all_ids, dtype=np.uint16)
    ratio = n_chars / len(data)
    print(f"  {n_chars:,} chars -> {len(data):,} tokens ({ratio:.1f}x compression)")
    return data


class DataLoader:
    """Simple random-batch data loader from a flat token array."""

    def __init__(
        self, data: npt.NDArray[np.uint16], batch_size: int, seq_len: int
    ) -> None:
        self.data = data
        self.B = batch_size
        self.T = seq_len

    def batch(self) -> tuple[torch.Tensor, torch.Tensor]:
        ix = torch.randint(len(self.data) - self.T, (self.B,))
        x = torch.stack(
            [torch.from_numpy(self.data[i : i + self.T].copy().astype(int)) for i in ix]
        )
        y = torch.stack(
            [
                torch.from_numpy(self.data[i + 1 : i + 1 + self.T].copy().astype(int))
                for i in ix
            ]
        )
        return x, y
