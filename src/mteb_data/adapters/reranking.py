"""Reranking adapter for query + candidate lists."""

from __future__ import annotations

from typing import Any, Iterable, Mapping

from mteb_data.adapters import make_provenance
from mteb_data.catalog import TaskConfig
from mteb_data.schema import CanonicalRecord


def adapt_reranking_rows(
    rows: Iterable[dict[str, Any]],
    *,
    task: TaskConfig,
    config_name: str,
    language: str,
    source_split: str,
    corpus: Mapping[str, dict[str, Any]] | None = None,
) -> list[CanonicalRecord]:
    """Accept either modern retrieval-style rows or legacy positive/negative lists."""
    records: list[CanonicalRecord] = []
    corpus = corpus or {}
    for i, row in enumerate(rows):
        query = row.get("query") or row.get("text") or row.get("anchor")
        qid = str(row.get("query-id") or row.get("query_id") or row.get("id") or i)
        if query is None:
            continue

        candidates: list[dict[str, Any]] = []
        scores: list[float] = []

        if "positive" in row or "negative" in row:
            for doc in row.get("positive") or []:
                text = doc if isinstance(doc, str) else doc.get("text", "")
                did = None if isinstance(doc, str) else doc.get("id")
                candidates.append({"id": str(did or f"pos-{len(candidates)}"), "text": str(text)})
                scores.append(1.0)
            for doc in row.get("negative") or []:
                text = doc if isinstance(doc, str) else doc.get("text", "")
                did = None if isinstance(doc, str) else doc.get("id")
                candidates.append({"id": str(did or f"neg-{len(candidates)}"), "text": str(text)})
                scores.append(0.0)
        elif "corpus-ids" in row or "corpus_ids" in row:
            ids = row.get("corpus-ids") or row.get("corpus_ids") or []
            rels = row.get("scores") or row.get("relevance_scores") or [0.0] * len(ids)
            for did, score in zip(ids, rels):
                text = corpus.get(str(did), {}).get("text", "")
                candidates.append({"id": str(did), "text": text})
                scores.append(float(score))
        else:
            continue

        records.append(
            CanonicalRecord(
                provenance=make_provenance(
                    task=task.name,
                    family="reranking",
                    dataset=task.dataset,
                    revision=task.revision,
                    config=config_name,
                    source_split=source_split,
                    language=language,
                    license=task.license,
                    original_id=qid,
                ),
                family="reranking",
                inputs={"anchor": str(query), "candidates": candidates},
                target={"relevance_scores": scores},
                group={"query_id": qid},
            )
        )
    return records
