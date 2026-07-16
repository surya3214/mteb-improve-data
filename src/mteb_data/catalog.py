"""Catalog loading for pinned MTEB Multilingual v2 task metadata."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable

import yaml

from mteb_data.schema import TARGET_LANGUAGES, normalize_language


@dataclass
class TaskConfig:
    name: str
    family: str
    dataset: str
    revision: str
    eval_splits: list[str]
    license: str | None = None
    training_eligible: bool = False
    ineligibility_reason: str | None = None
    configs: list[dict[str, Any]] = field(default_factory=list)
    score_range: list[float] | None = None
    notes: str | None = None

    @property
    def usable_splits(self) -> list[str]:
        splits: list[str] = []
        for cfg in self.configs:
            for split in cfg.get("available_splits", []):
                if split not in self.eval_splits and split not in splits:
                    splits.append(split)
        return splits


@dataclass
class Catalog:
    benchmark: str
    mteb_commit: str
    target_languages: dict[str, str]
    tasks: list[TaskConfig]

    def eligible_tasks(self, families: Iterable[str] | None = None) -> list[TaskConfig]:
        fams = set(families) if families else None
        out = []
        for task in self.tasks:
            if not task.training_eligible:
                continue
            if fams and task.family not in fams:
                continue
            out.append(task)
        return out

    def by_name(self, name: str) -> TaskConfig:
        for task in self.tasks:
            if task.name == name:
                return task
        raise KeyError(name)


def default_catalog_path() -> Path:
    from mteb_data.paths import repo_root

    return repo_root() / "configs" / "mmteb_multilingual_v2.yaml"


def load_catalog(path: str | Path | None = None) -> Catalog:
    catalog_path = Path(path) if path else default_catalog_path()
    raw = yaml.safe_load(catalog_path.read_text(encoding="utf-8"))
    tasks = [TaskConfig(**task) for task in raw["tasks"]]
    target = raw.get("target_languages") or TARGET_LANGUAGES
    # Ensure canonical values.
    target = {k: normalize_language(v) or v for k, v in target.items()}
    return Catalog(
        benchmark=raw["benchmark"],
        mteb_commit=raw["mteb_commit"],
        target_languages=target,
        tasks=tasks,
    )


def audit_catalog(catalog: Catalog) -> dict[str, Any]:
    eligible = catalog.eligible_tasks()
    ineligible = [t for t in catalog.tasks if not t.training_eligible]
    by_family: dict[str, dict[str, int]] = {}
    for task in catalog.tasks:
        bucket = by_family.setdefault(task.family, {"eligible": 0, "ineligible": 0})
        key = "eligible" if task.training_eligible else "ineligible"
        bucket[key] += 1
    languages = sorted(set(catalog.target_languages.values()))
    return {
        "benchmark": catalog.benchmark,
        "mteb_commit": catalog.mteb_commit,
        "target_languages": languages,
        "task_counts": {
            "total": len(catalog.tasks),
            "eligible": len(eligible),
            "ineligible": len(ineligible),
            "by_family": by_family,
        },
        "eligible_tasks": [
            {
                "name": t.name,
                "family": t.family,
                "dataset": t.dataset,
                "usable_splits": t.usable_splits,
                "configs": [c.get("name") for c in t.configs],
            }
            for t in eligible
        ],
        "ineligible_tasks": [
            {
                "name": t.name,
                "family": t.family,
                "reason": t.ineligibility_reason,
                "eval_splits": t.eval_splits,
            }
            for t in ineligible
        ],
    }
