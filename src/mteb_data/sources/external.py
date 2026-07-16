"""External source adapters (streaming / capped)."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterator

import yaml


@dataclass
class ExternalSource:
    name: str
    dataset: str
    family: str
    priority: int
    description: str
    languages: list[str]
    shape: str
    license: str
    overlap_risk: str
    default_cap: int
    streaming: bool = True
    config: str | None = None
    split: str = "train"
    notes: str | None = None


def default_external_path() -> Path:
    from mteb_data.paths import repo_root

    return repo_root() / "configs" / "external_sources.yaml"


def load_external_sources(path: str | Path | None = None) -> list[ExternalSource]:
    p = Path(path) if path else default_external_path()
    raw = yaml.safe_load(p.read_text(encoding="utf-8"))
    return [ExternalSource(**item) for item in raw["sources"]]


def stream_external_pairs(
    source: ExternalSource,
    *,
    max_rows: int | None = None,
) -> Iterator[dict[str, Any]]:
    """Yield lightweight pair/triplet dicts from an external Hub dataset.

    This intentionally keeps transformations conservative and source-specific.
    """
    from datasets import load_dataset

    cap = max_rows if max_rows is not None else source.default_cap
    kwargs: dict[str, Any] = {
        "path": source.dataset,
        "split": source.split,
        "streaming": source.streaming,
    }
    if source.config:
        kwargs["name"] = source.config

    ds = load_dataset(**kwargs)
    count = 0
    for row in ds:
        item = _map_external_row(source, dict(row))
        if item is None:
            continue
        yield item
        count += 1
        if count >= cap:
            break


def _map_external_row(source: ExternalSource, row: dict[str, Any]) -> dict[str, Any] | None:
    name = source.name
    if name in {"nllb", "ccmatrix"}:
        # Common bitext columns vary; try several.
        src = row.get("source") or row.get("src") or row.get("sentence1")
        tgt = row.get("target") or row.get("tgt") or row.get("sentence2")
        if "translation" in row and isinstance(row["translation"], dict):
            vals = list(row["translation"].values())
            if len(vals) >= 2:
                src, tgt = vals[0], vals[1]
        if not src or not tgt:
            return None
        return {
            "family": "bitext",
            "anchor": str(src),
            "positive": str(tgt),
            "source": source.name,
            "dataset": source.dataset,
            "license": source.license,
            "overlap_risk": source.overlap_risk,
        }
    if name == "xp3x":
        prompt = row.get("inputs") or row.get("prompt") or row.get("input")
        answer = row.get("targets") or row.get("target") or row.get("output")
        if not prompt or not answer:
            return None
        return {
            "family": "instruction",
            "anchor": str(prompt),
            "positive": str(answer),
            "source": source.name,
            "dataset": source.dataset,
            "license": source.license,
            "overlap_risk": source.overlap_risk,
        }
    if name == "s2orc":
        title = row.get("title")
        abstract = row.get("abstract") or row.get("paperAbstract")
        if not title or not abstract:
            return None
        return {
            "family": "retrieval",
            "anchor": str(title),
            "positive": str(abstract),
            "source": source.name,
            "dataset": source.dataset,
            "license": source.license,
            "overlap_risk": source.overlap_risk,
        }
    if name == "embed-nemotron":
        query = row.get("query") or row.get("anchor")
        positive = row.get("positive") or row.get("document") or row.get("passage")
        if not query or not positive:
            return None
        return {
            "family": "retrieval",
            "anchor": str(query),
            "positive": str(positive if isinstance(positive, str) else positive[0]),
            "negatives": row.get("negatives") or row.get("hard_negatives") or [],
            "source": source.name,
            "dataset": source.dataset,
            "license": source.license,
            "overlap_risk": source.overlap_risk,
        }
    return None
