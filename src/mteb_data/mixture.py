"""Mixture weight computation."""

from __future__ import annotations

from collections import defaultdict
from typing import Any, Iterable


FAMILIES = ["sts", "classification", "clustering", "retrieval", "reranking"]


def temperature_weight(count: int, temperature: float = 0.4) -> float:
    return float(count) ** temperature if count > 0 else 0.0


def compute_mixture_weights(
    counts: dict[str, dict[str, int]],
    *,
    family_mass: float = 1.0,
    temperature: float = 0.4,
) -> dict[str, Any]:
    """Compute hierarchical mixture weights.

    counts structure: {family: {language_or_task: n}}
    Returns equal family mass, then language-balanced mass, then temperature-smoothed
    dataset/task mass within each language bucket when provided as task keys.
    """
    present = [f for f in FAMILIES if counts.get(f)]
    if not present:
        return {"families": {}, "tasks": {}}

    family_weight = family_mass / len(present)
    families = {f: family_weight for f in present}
    task_weights: dict[str, float] = {}

    for family in present:
        lang_counts = counts[family]
        # Group by language if keys look like "lang::task", else treat as tasks.
        by_lang: dict[str, dict[str, int]] = defaultdict(dict)
        for key, n in lang_counts.items():
            if "::" in key:
                lang, task = key.split("::", 1)
            else:
                lang, task = "all", key
            by_lang[lang][task] = n

        langs = [lang for lang, tasks in by_lang.items() if tasks]
        lang_mass = family_weight / max(len(langs), 1)
        for lang in langs:
            tasks = by_lang[lang]
            raw = {task: temperature_weight(n, temperature) for task, n in tasks.items()}
            total = sum(raw.values()) or 1.0
            for task, w in raw.items():
                task_weights[f"{family}::{lang}::{task}"] = lang_mass * (w / total)

    return {
        "families": families,
        "tasks": task_weights,
        "temperature": temperature,
        "notes": "Equal family mass; equal language mass within family; dataset ~ n^temperature.",
    }


def counts_from_rows(rows: Iterable[dict[str, Any]], family: str) -> dict[str, int]:
    out: dict[str, int] = defaultdict(int)
    for row in rows:
        lang = row.get("language", "unknown")
        task = row.get("task", "unknown")
        out[f"{lang}::{task}"] += 1
    return dict(out)
