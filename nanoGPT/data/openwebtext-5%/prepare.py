"""Prepare a 5% OpenWebText subset using streaming memmap writes."""

import os
from pathlib import Path

import numpy as np
import tiktoken
from datasets import load_dataset
from tqdm import tqdm

SAMPLE_RATIO = 0.05
VAL_RATIO = 0.1
RNG_SEED = 42
NUM_PROC = int(os.environ.get("OWT_NUM_PROC", 8))


def _tokenize_dataset(dset, encoder, desc):
    def process(example):
        ids = encoder.encode_ordinary(example["text"])
        ids.append(encoder.eot_token)
        return {"ids": ids, "len": len(ids)}

    return dset.map(
        process,
        remove_columns=["text"],
        desc=desc,
        num_proc=NUM_PROC,
    )


def _write_memmap(split_name, dset, out_path):
    arr_len = int(np.sum(dset["len"], dtype=np.uint64))
    if arr_len == 0:
        raise SystemExit(f"{split_name}: nothing to write (no tokens)")

    arr = np.memmap(out_path, dtype=np.uint16, mode="w+", shape=(arr_len,))
    idx = 0
    total_docs = len(dset)
    num_shards = max(1, min(total_docs, 1024))

    for shard_idx in tqdm(range(num_shards), desc=f"writing {out_path.name}"):
        shard = dset.shard(num_shards=num_shards, index=shard_idx, contiguous=True)
        if len(shard) == 0:
            continue
        shard = shard.with_format("numpy")
        arr_batch = np.concatenate(shard["ids"])
        next_idx = idx + len(arr_batch)
        arr[idx:next_idx] = arr_batch
        idx = next_idx

    arr.flush()


def main():
    out_dir = Path(__file__).parent
    train_path = out_dir / "train.bin"
    val_path = out_dir / "val.bin"

    print("Loading OpenWebText train split...")
    train_split = load_dataset("openwebtext", split="train", num_proc=NUM_PROC)
    n_total = len(train_split)
    n_sample = max(1, int(n_total * SAMPLE_RATIO))
    print(f"Total documents: {n_total}. Sampling {n_sample} (~{SAMPLE_RATIO*100:.1f}%).")

    sampled = train_split.shuffle(seed=RNG_SEED).select(range(n_sample))
    split_dataset = sampled.train_test_split(test_size=VAL_RATIO, seed=RNG_SEED)
    split_dataset["val"] = split_dataset.pop("test")

    encoder = tiktoken.get_encoding("gpt2")
    tokenized = _tokenize_dataset(split_dataset, encoder, "tokenizing 5% subset")

    _write_memmap("train", tokenized["train"], train_path)
    _write_memmap("val", tokenized["val"], val_path)

    print(f"Saved train tokens to {train_path}")
    print(f"Saved val tokens to {val_path}")
    print("✅ 5% OpenWebText subset ready (train.bin / val.bin).")


if __name__ == "__main__":
    main()
