"""STS scored-pair generation."""

from __future__ import annotations

from typing import Any

from mteb_data.schema import CanonicalRecord


def generate_sts_pairs(records: list[CanonicalRecord]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for rec in records:
        if rec.family != "sts":
            continue
        rows.append(
            {
                "sentence1": rec.inputs["anchor"],
                "sentence2": rec.inputs["positive"],
                "score": rec.target["normalized_similarity"],
                "raw_score": rec.target["raw_similarity"],
                "task": rec.provenance.task,
                "config": rec.provenance.config,
                "language": rec.provenance.language,
                "source_split": rec.provenance.source_split,
                "dataset": rec.provenance.dataset,
                "revision": rec.provenance.revision,
                "original_id": rec.provenance.original_id,
                "license": rec.provenance.license,
            }
        )
    return rows
