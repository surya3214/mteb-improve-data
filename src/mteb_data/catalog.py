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
    def has_hub_train(self) -> bool:
        return any("train" in cfg.get("available_splits", []) for cfg in self.configs)

    @property
    def usable_splits(self) -> list[str]:
        splits: list[str] = []
        for cfg in self.configs:
            for split in cfg.get("available_splits", []):
                if split not in self.eval_splits and split not in splits:
                    splits.append(split)
        return splits

    def selectable_splits(self, *, include_hub_train: bool = False) -> list[str]:
        """Splits that may be loaded under the current policy."""
        if self.training_eligible:
            return self.usable_splits
        if include_hub_train and self.has_hub_train:
            # Score-max override: only the Hub train split.
            return ["train"]
        return []

    def is_buildable(self, *, include_hub_train: bool = False) -> bool:
        if self.training_eligible:
            return True
        return include_hub_train and self.has_hub_train


@dataclass
class Catalog:
    benchmark: str
    mteb_commit: str
    target_languages: dict[str, str]
    tasks: list[TaskConfig]

    def eligible_tasks(
        self,
        families: Iterable[str] | None = None,
        *,
        include_hub_train: bool = False,
    ) -> list[TaskConfig]:
        fams = set(families) if families else None
        out = []
        for task in self.tasks:
            if not task.is_buildable(include_hub_train=include_hub_train):
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


def audit_catalog(catalog: Catalog, *, include_hub_train: bool = False) -> dict[str, Any]:
    eligible = catalog.eligible_tasks(include_hub_train=include_hub_train)
    ineligible = [t for t in catalog.tasks if not t.is_buildable(include_hub_train=include_hub_train)]
    hub_train_overrides = [
        t
        for t in catalog.tasks
        if include_hub_train and t.has_hub_train and not t.training_eligible
    ]
    by_family: dict[str, dict[str, int]] = {}
    for task in catalog.tasks:
        bucket = by_family.setdefault(task.family, {"eligible": 0, "ineligible": 0})
        key = "eligible" if task.is_buildable(include_hub_train=include_hub_train) else "ineligible"
        bucket[key] += 1
    languages = sorted(set(catalog.target_languages.values()))
    return {
        "benchmark": catalog.benchmark,
        "mteb_commit": catalog.mteb_commit,
        "include_hub_train": include_hub_train,
        "target_languages": languages,
        "task_counts": {
            "total": len(catalog.tasks),
            "eligible": len(eligible),
            "ineligible": len(ineligible),
            "hub_train_overrides": len(hub_train_overrides),
            "by_family": by_family,
        },
        "eligible_tasks": [
            {
                "name": t.name,
                "family": t.family,
                "dataset": t.dataset,
                "usable_splits": t.selectable_splits(include_hub_train=include_hub_train),
                "configs": [c.get("name") for c in t.configs],
                "catalog_training_eligible": t.training_eligible,
                "hub_train_override": include_hub_train and t.has_hub_train and not t.training_eligible,
            }
            for t in eligible
        ],
        "hub_train_override_tasks": [
            {
                "name": t.name,
                "family": t.family,
                "reason": t.ineligibility_reason,
                "eval_splits": t.eval_splits,
                "note": "Included only because --include-hub-train was set.",
            }
            for t in hub_train_overrides
        ],
        "ineligible_tasks": [
            {
                "name": t.name,
                "family": t.family,
                "reason": t.ineligibility_reason,
                "eval_splits": t.eval_splits,
                "has_hub_train": t.has_hub_train,
            }
            for t in ineligible
        ],
    }
