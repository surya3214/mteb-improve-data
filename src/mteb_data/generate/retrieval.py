"""Retrieval triple generation with optional mined negatives."""

from __future__ import annotations

import random
from typing import Any, Mapping, Sequence

from mteb_data.schema import CanonicalRecord


def generate_retrieval_triples(
    records: list[CanonicalRecord],
    *,
    mined_negatives: Mapping[str, Sequence[dict[str, Any]]] | None = None,
    corpus_pool: Sequence[dict[str, Any]] | None = None,
    num_negatives: int = 4,
    seed: int = 42,
) -> list[dict[str, Any]]:
    """Emit one row per query-positive with negatives.

    Negatives prefer mined candidates keyed by query_id. Known positive IDs are
    always excluded. If mining is unavailable, sample from corpus_pool.
    """
    rng = random.Random(seed)
    mined_negatives = mined_negatives or {}
    corpus_pool = list(corpus_pool or [])
    out: list[dict[str, Any]] = []

    for rec in records:
        if rec.family != "retrieval":
            continue
        qid = rec.group["query_id"]
        positive_ids = set(rec.target.get("positive_ids") or [])
        for pos in rec.inputs.get("positives") or []:
            negs: list[dict[str, Any]] = []
            candidates = list(mined_negatives.get(qid, []))
            for cand in candidates:
                cid = str(cand.get("id"))
                if cid in positive_ids:
                    continue
                negs.append({"id": cid, "text": cand.get("text", "")})
                if len(negs) >= num_negatives:
                    break
            if len(negs) < num_negatives and corpus_pool:
                pool = [c for c in corpus_pool if str(c.get("id")) not in positive_ids]
                rng.shuffle(pool)
                for cand in pool:
                    cid = str(cand.get("id"))
                    if any(n["id"] == cid for n in negs):
                        continue
                    negs.append({"id": cid, "text": cand.get("text", "")})
                    if len(negs) >= num_negatives:
                        break

            out.append(
                {
                    "query": rec.inputs["anchor"],
                    "positive": pos.get("text", ""),
                    "positive_id": pos.get("id"),
                    "negatives": [n["text"] for n in negs],
                    "negative_ids": [n["id"] for n in negs],
                    "query_id": qid,
                    "task": rec.provenance.task,
                    "config": rec.provenance.config,
                    "language": rec.provenance.language,
                    "dataset": rec.provenance.dataset,
                    "revision": rec.provenance.revision,
                    "source_split": rec.provenance.source_split,
                    "license": rec.provenance.license,
                }
            )
    return out
