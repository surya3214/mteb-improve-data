"""Parquet / JSON IO helpers and grouped train/validation splitting."""

from __future__ import annotations

import hashlib
import json
import random
from pathlib import Path
from typing import Any, Iterable, Sequence

import pyarrow as pa
import pyarrow.parquet as pq


def write_json(path: Path, obj: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def write_parquet(path: Path, rows: Sequence[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        table = pa.table({})
    else:
        # Normalize nested lists/dicts to JSON strings for robust parquet round-trips.
        normalized = [_normalize_row(r) for r in rows]
        table = pa.Table.from_pylist(normalized)
    pq.write_table(table, path)


def read_parquet(path: Path) -> list[dict[str, Any]]:
    table = pq.read_table(path)
    return table.to_pylist()


_FORCE_STRING_KEYS = {
    "original_id",
    "query_id",
    "positive_id",
    "document_id",
    "preferred_id",
    "rejected_id",
    "license",
    "task",
    "config",
    "language",
    "dataset",
    "revision",
    "source_split",
    "class_namespace",
    "cluster_namespace",
    "experiment_id",
    "positive_language",
    "negative_language",
    "class_id",
    "anchor",
    "positive",
    "negative",
    "sentence1",
    "sentence2",
}


def _normalize_row(row: dict[str, Any]) -> dict[str, Any]:
    """Coerce Hub-heterogeneous values into a parquet-friendly row.

    Hugging Face rows often mix int/string IDs across datasets; forcing identity
    and provenance fields to strings avoids Arrow type unification errors.
    """
    out: dict[str, Any] = {}
    for k, v in row.items():
        if isinstance(v, (list, dict)):
            out[k] = json.dumps(v, ensure_ascii=False)
        elif v is None:
            out[k] = None
        elif k in _FORCE_STRING_KEYS:
            out[k] = str(v)
        elif isinstance(v, bool):
            out[k] = v
        elif isinstance(v, (int, float)):
            out[k] = v
        else:
            out[k] = str(v)
    return out


def hash_rows(rows: Sequence[dict[str, Any]]) -> str:
    payload = json.dumps(rows, sort_keys=True, ensure_ascii=False, default=str)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def grouped_split(
    rows: Sequence[dict[str, Any]],
    *,
    group_key: str,
    val_ratio: float = 0.05,
    seed: int = 42,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """Split by group to avoid related examples crossing train/validation."""
    rng = random.Random(seed)
    groups: dict[str, list[dict[str, Any]]] = {}
    for row in rows:
        gid = str(row.get(group_key) or row.get("original_id") or id(row))
        groups.setdefault(gid, []).append(row)
    keys = sorted(groups)
    rng.shuffle(keys)
    n_val = int(len(keys) * val_ratio)
    val_keys = set(keys[:n_val])
    train, val = [], []
    for key, items in groups.items():
        (val if key in val_keys else train).extend(items)
    return train, val


def ensure_build_dir(root: Path, build_id: str) -> Path:
    path = root / build_id
    path.mkdir(parents=True, exist_ok=True)
    return path
