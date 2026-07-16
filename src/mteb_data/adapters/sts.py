"""STS adapter: sentence pairs with raw + normalized scores."""

from __future__ import annotations

from typing import Any, Iterable

from mteb_data.adapters import make_provenance, row_id
from mteb_data.catalog import TaskConfig
from mteb_data.schema import CanonicalRecord, normalize_score


def adapt_sts_rows(
    rows: Iterable[dict[str, Any]],
    *,
    task: TaskConfig,
    config_name: str,
    language: str,
    source_split: str,
) -> list[CanonicalRecord]:
    score_min, score_max = (task.score_range or [0.0, 5.0])
    records: list[CanonicalRecord] = []
    for i, row in enumerate(rows):
        s1 = row.get("sentence1")
        s2 = row.get("sentence2")
        score = row.get("score")
        if s1 is None or s2 is None or score is None:
            continue
        oid = row_id(row, i)
        records.append(
            CanonicalRecord(
                provenance=make_provenance(
                    task=task.name,
                    family="sts",
                    dataset=task.dataset,
                    revision=task.revision,
                    config=config_name,
                    source_split=source_split,
                    language=language,
                    license=task.license,
                    original_id=oid,
                ),
                family="sts",
                inputs={"anchor": str(s1), "positive": str(s2)},
                target={
                    "raw_similarity": float(score),
                    "raw_range": [float(score_min), float(score_max)],
                    "normalized_similarity": normalize_score(float(score), float(score_min), float(score_max)),
                },
                group={"pair_id": oid},
            )
        )
    return records
