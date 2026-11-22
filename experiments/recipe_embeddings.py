"""Encode the entire recipe catalogue with the frozen TESMR sentence encoder."""

import argparse
import json
from ast import literal_eval
from hashlib import sha256
from time import perf_counter

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from recipe_embeddings_download import CHECKPOINT, MODEL_ID, REVISION, ROOT
from tokenizers import Tokenizer
from transformers.models.bert.modeling_bert import BertModel

CACHE = ROOT / "data/cache/recipe_embeddings"
REPORT = ROOT / "results/recipe_embeddings"
RAW = ROOT / "data/RAW_recipes.csv"
CHUNK = 8192
BATCH = 192
LENGTH = 64
FORMAT_VERSION = 1
GENERIC_TAGS = {
    "course",
    "preparation",
    "main-ingredient",
    "time-to-make",
    "occasion",
    "dietary",
    "taste-mood",
    "cuisine",
    "equipment",
    "number-of-servings",
}


def file_digest(path):
    with path.open("rb") as handle:
        return sha256(handle.read()).hexdigest()


def as_text(row, field):
    title = row.name.strip() or "untitled recipe"
    if field == "title":
        return title
    ingredients = ", ".join(literal_eval(row.ingredients))
    tags = [tag.replace("-", " ") for tag in literal_eval(row.tags) if tag not in GENERIC_TAGS]
    return f"{title}. Ingredients: {ingredients}. Tags: {', '.join(tags[:6])}."


def write_json(path, contents):
    pending = path.with_suffix(".tmp")
    pending.write_text(json.dumps(contents, indent=2) + "\n")
    pending.replace(path)


def encode_batch(model, encodings, perm, dest, start, device):
    width = (max(len(encodings[i].ids) for i in perm) + 7) // 8 * 8
    ids = np.zeros((len(perm), width), dtype=np.int64)
    mask = np.zeros_like(ids)
    for j, idx in enumerate(perm):
        tokens = encodings[idx].ids
        ids[j, : len(tokens)] = tokens
        mask[j, : len(tokens)] = 1
    tokens = torch.from_numpy(ids).to(device)
    attention = torch.from_numpy(mask).to(device)
    hidden = model(input_ids=tokens, attention_mask=attention).last_hidden_state
    valid = attention.unsqueeze(-1).to(hidden.dtype)
    pooled = (hidden * valid).sum(dim=1).float() / attention.sum(dim=1, keepdim=True)
    normalized = F.normalize(pooled, dim=-1).half().cpu().numpy()
    dest[start + perm] = normalized


def encode_field(catalogue, field, tokenizer, model, device, signature):
    filename = CACHE / f"{field}.npy"
    interim = CACHE / f"{field}.partial.npy"
    checkpoint = CACHE / f"{field}.progress.json"
    manifest = REPORT / f"{field}.json"
    if filename.exists() and manifest.exists():
        info = json.loads(manifest.read_text())
        if signature != info["signature"] or info["sha256"] != file_digest(filename):
            raise ValueError(
                f"Existing {field} embeddings differ from the requested recipe or model revision"
            )
        print(f"{field} already verified")
        return
    if interim.exists() and checkpoint.exists():
        state = json.loads(checkpoint.read_text())
        if state["signature"] != signature:
            raise ValueError("Partial encoding used a different checkpoint or catalogue")
        dest = np.load(interim, mmap_mode="r+")
    else:
        state = dict(signature=signature, rows_done=0, truncated=0, tokens=0, wall_seconds=0.0)
        dest = np.lib.format.open_memmap(
            interim, mode="w+", dtype=np.float16, shape=(len(catalogue), 384)
        )
        write_json(checkpoint, state)
    for start in range(state["rows_done"], len(catalogue), CHUNK):
        begin = perf_counter()
        rows = catalogue.iloc[start : start + CHUNK]
        texts = [as_text(row, field) for row in rows.itertuples(index=False)]
        encodings = tokenizer.encode_batch(texts)
        lengths = np.array([len(item.ids) for item in encodings], dtype=int)
        order = np.argsort(lengths, kind="stable")
        with torch.inference_mode():
            for j in range(0, len(rows), BATCH):
                encode_batch(model, encodings, order[j : j + BATCH], dest, start, device)
        dest.flush()
        state["tokens"] += int(lengths.sum())
        state["truncated"] += sum(bool(item.overflowing) for item in encodings)
        state["rows_done"] += len(rows)
        state["wall_seconds"] += perf_counter() - begin
        write_json(checkpoint, state)
        print(
            f"{field}: {state['rows_done']:,}/{len(catalogue):,}, {state['wall_seconds']:.1f} s",
            flush=True,
        )
    minimum, maximum = np.inf, 0.0
    for start in range(0, len(catalogue), CHUNK):
        norms = np.linalg.norm(dest[start : start + CHUNK].astype(np.float32), axis=1)
        if not np.all(np.isfinite(norms)) or np.max(np.abs(norms - 1)) > 0.001:
            raise ValueError("Invalid embedding norms")
        minimum, maximum = min(minimum, float(norms.min())), max(maximum, float(norms.max()))
    del dest
    interim.replace(filename)
    write_json(
        manifest,
        {
            **state,
            "field": field,
            "device": device,
            "dtype": "float16",
            "min_l2_norm": minimum,
            "max_l2_norm": maximum,
            "sha256": file_digest(filename),
            "pytorch": torch.__version__,
        },
    )
    checkpoint.unlink()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--fields", nargs="+", choices=["card", "title"], default=["card", "title"])
    parser.add_argument(
        "--device",
        choices=["cpu", "mps"],
        default="mps" if torch.backends.mps.is_available() else "cpu",
    )
    args = parser.parse_args()
    if args.device == "mps" and not torch.backends.mps.is_available():
        raise RuntimeError("Apple Metal device is unavailable in this runtime")
    CACHE.mkdir(exist_ok=True, parents=True)
    REPORT.mkdir(exist_ok=True, parents=True)
    catalogue = pd.read_csv(
        RAW, usecols=["id", "name", "ingredients", "tags"], keep_default_na=False
    )
    catalogue = catalogue.sort_values("id").reset_index(drop=True)
    ids = catalogue.id.to_numpy(np.int64)
    if not np.all(ids[1:] > ids[:-1]):
        raise ValueError("Recipe IDs must be unique and sorted")
    np.save(CACHE / "ids.npy", ids)
    tokenizer = Tokenizer.from_file(str(CHECKPOINT / "tokenizer.json"))
    tokenizer.no_padding()
    tokenizer.enable_truncation(max_length=LENGTH)
    model = BertModel.from_pretrained(
        CHECKPOINT, local_files_only=True, attn_implementation="sdpa"
    ).eval()
    if args.device == "mps":
        model.half()
    model.to(args.device)
    signature = dict(
        format_version=FORMAT_VERSION,
        model=MODEL_ID,
        revision=REVISION,
        token_limit=LENGTH,
        id_sha256=file_digest(CACHE / "ids.npy"),
        source_sha256=file_digest(RAW),
        safetensor_sha256=file_digest(CHECKPOINT / "model.safetensors"),
        tokenizer_sha256=file_digest(CHECKPOINT / "tokenizer.json"),
        backend=args.device,
    )
    for field in args.fields:
        encode_field(catalogue, field, tokenizer, model, args.device, signature)


if __name__ == "__main__":
    main()
