"""Shared adapter helpers."""

from __future__ import annotations

from typing import Any

from mteb_data.schema import Provenance, normalize_language


def make_provenance(
    *,
    task: str,
    family: str,
    dataset: str,
    revision: str,
    config: str,
    source_split: str,
    language: str,
    license: str | None,
    original_id: str | None,
) -> Provenance:
    return Provenance(
        task=task,
        family=family,
        dataset=dataset,
        revision=revision,
        config=config,
        source_split=source_split,
        language=normalize_language(language) or language,
        license=license,
        original_id=original_id,
    )


def row_id(row: dict[str, Any], index: int) -> str:
    for key in ("id", "_id", "idx", "guid"):
        if key in row and row[key] is not None:
            return str(row[key])
    return str(index)
