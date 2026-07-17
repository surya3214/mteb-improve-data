"""MTEB Hub loading with evaluation-split guards."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Iterator

from mteb_data.adapters.classification import adapt_classification_rows
from mteb_data.adapters.clustering import adapt_clustering_rows
from mteb_data.adapters.sts import adapt_sts_rows
from mteb_data.catalog import Catalog, TaskConfig
from mteb_data.schema import CanonicalRecord, normalize_language


@dataclass
class LoadResult:
    task: str
    config: str
    split: str
    records: list[CanonicalRecord]
    skipped_reason: str | None = None
    contamination: bool = False
    contamination_reason: str | None = None


class EvaluationSplitError(ValueError):
    """Raised when a requested split is an official MTEB evaluation split."""


def assert_trainable_split(
    task: TaskConfig,
    split: str,
    *,
    include_hub_train: bool = False,
) -> dict[str, Any]:
    """Validate whether a split may be loaded.

    Returns contamination metadata when --include-hub-train overrides safety rules.
    """
    hub_train_override = include_hub_train and split == "train" and task.has_hub_train

    if not task.training_eligible and not hub_train_override:
        raise EvaluationSplitError(
            f"Refusing to load {task.name}: training_eligible=false ({task.ineligibility_reason})"
        )

    if split in task.eval_splits and not hub_train_override:
        raise EvaluationSplitError(
            f"Refusing to load {task.name} split={split!r}: listed in eval_splits={task.eval_splits}. "
            "This prevents direct evaluation leakage."
        )

    if hub_train_override and (not task.training_eligible or split in task.eval_splits):
        return {
            "contamination": True,
            "contamination_reason": (
                task.ineligibility_reason
                or f"Hub train split overlaps MTEB eval_splits={task.eval_splits}"
            ),
        }
    return {"contamination": False, "contamination_reason": None}


def _config_in_target_languages(catalog: Catalog, language: str | None) -> bool:
    if not language:
        return True
    if "|" in str(language):
        return True
    normalized = normalize_language(language) or language
    if normalized in catalog.target_languages.values():
        return True
    if language in catalog.target_languages:
        return True
    return False


def iter_task_loads(
    catalog: Catalog,
    *,
    families: list[str] | None = None,
    task_names: list[str] | None = None,
    include_hub_train: bool = False,
    max_rows_per_split: int | None = None,
) -> Iterator[tuple[TaskConfig, dict[str, Any], str]]:
    del max_rows_per_split  # selection only; row caps applied at load time
    names = set(task_names) if task_names else None
    fams = set(families) if families else None
    for task in catalog.tasks:
        if names and task.name not in names:
            continue
        if fams and task.family not in fams:
            continue
        if not task.is_buildable(include_hub_train=include_hub_train):
            continue
        for cfg in task.configs:
            language = cfg.get("language")
            if not _config_in_target_languages(catalog, language):
                continue
            available = set(cfg.get("available_splits", []))
            for split in task.selectable_splits(include_hub_train=include_hub_train):
                if split not in available:
                    continue
                yield task, cfg, split


def load_split_rows(
    dataset: str,
    *,
    config: str,
    split: str,
    revision: str,
    max_rows: int | None = None,
) -> list[dict[str, Any]]:
    from datasets import load_dataset

    kwargs: dict[str, Any] = {"path": dataset, "split": split, "revision": revision}
    # Hub datasets use config name via `name=` / `config_name`.
    if config and config != "default":
        kwargs["name"] = config
    ds = load_dataset(**kwargs)
    if max_rows is not None:
        ds = ds.select(range(min(max_rows, len(ds))))
    return [dict(row) for row in ds]


def normalize_task_split(
    task: TaskConfig,
    cfg: dict[str, Any],
    split: str,
    *,
    max_rows: int | None = None,
    include_hub_train: bool = False,
) -> LoadResult:
    meta = assert_trainable_split(task, split, include_hub_train=include_hub_train)
    config_name = cfg.get("name", "default")
    language = cfg.get("language", "unknown")
    rows = load_split_rows(
        task.dataset,
        config=config_name,
        split=split,
        revision=task.revision,
        max_rows=max_rows,
    )
    if task.family == "sts":
        records = adapt_sts_rows(
            rows, task=task, config_name=config_name, language=language, source_split=split
        )
    elif task.family == "classification":
        records = adapt_classification_rows(
            rows, task=task, config_name=config_name, language=language, source_split=split
        )
    elif task.family == "clustering":
        records = adapt_clustering_rows(
            rows, task=task, config_name=config_name, language=language, source_split=split
        )
    else:
        return LoadResult(
            task=task.name,
            config=config_name,
            split=split,
            records=[],
            skipped_reason=f"Family {task.family} requires specialized table loaders.",
            contamination=meta["contamination"],
            contamination_reason=meta["contamination_reason"],
        )
    return LoadResult(
        task=task.name,
        config=config_name,
        split=split,
        records=records,
        contamination=meta["contamination"],
        contamination_reason=meta["contamination_reason"],
    )
