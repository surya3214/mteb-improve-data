"""Retrieval adapter for corpus / queries / qrels tables."""

from __future__ import annotations

from collections import defaultdict
from typing import Any, Iterable, Mapping

from mteb_data.adapters import make_provenance
from mteb_data.catalog import TaskConfig
from mteb_data.schema import CanonicalRecord


def adapt_retrieval_tables(
    *,
    corpus: Iterable[dict[str, Any]],
    queries: Iterable[dict[str, Any]],
    qrels: Iterable[dict[str, Any]],
    task: TaskConfig,
    config_name: str,
    language: str,
    source_split: str,
    min_relevance: float = 1.0,
) -> list[CanonicalRecord]:
    corpus_map: dict[str, dict[str, Any]] = {}
    for row in corpus:
        doc_id = str(row.get("id") or row.get("_id"))
        text = row.get("text") or ""
        title = row.get("title") or ""
        corpus_map[doc_id] = {
            "id": doc_id,
            "text": text if not title else f"{title} {text}".strip(),
            "title": title,
        }

    query_map: dict[str, str] = {}
    for row in queries:
        qid = str(row.get("id") or row.get("_id"))
        query_map[qid] = str(row.get("text") or "")

    positives: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in qrels:
        qid = str(row.get("query-id") or row.get("query_id"))
        cid = str(row.get("corpus-id") or row.get("corpus_id"))
        score = float(row.get("score") or row.get("relevance") or 0)
        if score < min_relevance:
            continue
        if cid not in corpus_map or qid not in query_map:
            continue
        positives[qid].append({"id": cid, "text": corpus_map[cid]["text"], "relevance": score})

    records: list[CanonicalRecord] = []
    for qid, pos_docs in positives.items():
        records.append(
            CanonicalRecord(
                provenance=make_provenance(
                    task=task.name,
                    family="retrieval",
                    dataset=task.dataset,
                    revision=task.revision,
                    config=config_name,
                    source_split=source_split,
                    language=language,
                    license=task.license,
                    original_id=qid,
                ),
                family="retrieval",
                inputs={
                    "anchor": query_map[qid],
                    "positives": pos_docs,
                },
                target={"positive_ids": [d["id"] for d in pos_docs]},
                group={"query_id": qid},
            )
        )
    return records


def corpus_id_set(corpus: Iterable[Mapping[str, Any]]) -> set[str]:
    return {str(row.get("id") or row.get("_id")) for row in corpus}
