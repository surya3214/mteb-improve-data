"""Classification adapter: text + namespaced label."""

from __future__ import annotations

from typing import Any, Iterable

from mteb_data.adapters import make_provenance, row_id
from mteb_data.catalog import TaskConfig
from mteb_data.schema import CanonicalRecord, namespace_label


def adapt_classification_rows(
    rows: Iterable[dict[str, Any]],
    *,
    task: TaskConfig,
    config_name: str,
    language: str,
    source_split: str,
) -> list[CanonicalRecord]:
    records: list[CanonicalRecord] = []
    for i, row in enumerate(rows):
        text = row.get("text")
        if text is None:
            text = row.get("sentence") or row.get("content")
        if text is None:
            continue
        label = row.get("label")
        if label is None:
            label = row.get("labels")
        if label is None:
            continue
        oid = row_id(row, i)
        records.append(
            CanonicalRecord(
                provenance=make_provenance(
                    task=task.name,
                    family="classification",
                    dataset=task.dataset,
                    revision=task.revision,
                    config=config_name,
                    source_split=source_split,
                    language=language,
                    license=task.license,
                    original_id=oid,
                ),
                family="classification",
                inputs={"anchor": str(text)},
                target={
                    "class_id": label,
                    "class_namespace": namespace_label(task.name, config_name, language, label),
                },
                group={"class_namespace": namespace_label(task.name, config_name, language, label)},
            )
        )
    return records
