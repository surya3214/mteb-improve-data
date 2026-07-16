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


class EvaluationSplitError(ValueError):
    """Raised when a requested split is an official MTEB evaluation split."""


def assert_trainable_split(task: TaskConfig, split: str) -> None:
    if split in task.eval_splits:
        raise EvaluationSplitError(
            f"Refusing to load {task.name} split={split!r}: listed in eval_splits={task.eval_splits}. "
            "This prevents direct evaluation leakage."
        )
    if not task.training_eligible:
        raise EvaluationSplitError(
            f"Refusing to load {task.name}: training_eligible=false ({task.ineligibility_reason})"
        )


def iter_task_loads(
    catalog: Catalog,
    *,
    families: list[str] | None = None,
    task_names: list[str] | None = None,
    max_rows_per_split: int | None = None,
) -> Iterator[tuple[TaskConfig, dict[str, Any], str]]:
    names = set(task_names) if task_names else None
    fams = set(families) if families else None
    for task in catalog.tasks:
        if names and task.name not in names:
            continue
        if fams and task.family not in fams:
            continue
        if not task.training_eligible:
            continue
        for cfg in task.configs:
            language = normalize_language(cfg.get("language")) or cfg.get("language")
            # Skip non-target languages when language is a concrete BCP tag.
            if language and language not in catalog.target_languages.values() and "|" not in str(language):
                # still allow if it is one of the requested short codes mapped elsewhere
                if language not in catalog.target_languages:
                    # Keep multilingual configs that are explicitly listed for target langs only.
                    continue
            for split in cfg.get("available_splits", []):
                if split in task.eval_splits:
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
) -> LoadResult:
    assert_trainable_split(task, split)
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
        )
    return LoadResult(task=task.name, config=config_name, split=split, records=records)
