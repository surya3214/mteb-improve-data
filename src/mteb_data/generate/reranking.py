"""Reranking view generation: pointwise, pairwise, listwise."""

from __future__ import annotations

import random
from typing import Any

from mteb_data.schema import CanonicalRecord


def generate_reranking_views(
    records: list[CanonicalRecord],
    *,
    seed: int = 42,
    max_pairwise_per_query: int = 8,
) -> dict[str, list[dict[str, Any]]]:
    rng = random.Random(seed)
    pointwise: list[dict[str, Any]] = []
    pairwise: list[dict[str, Any]] = []
    listwise: list[dict[str, Any]] = []

    for rec in records:
        if rec.family != "reranking":
            continue
        query = rec.inputs["anchor"]
        candidates = rec.inputs.get("candidates") or []
        scores = rec.target.get("relevance_scores") or []
        qid = rec.group.get("query_id")
        base = {
            "query_id": qid,
            "task": rec.provenance.task,
            "config": rec.provenance.config,
            "language": rec.provenance.language,
            "dataset": rec.provenance.dataset,
            "revision": rec.provenance.revision,
            "source_split": rec.provenance.source_split,
            "license": rec.provenance.license,
        }

        listwise.append(
            {
                **base,
                "query": query,
                "documents": [c.get("text", "") for c in candidates],
                "document_ids": [c.get("id") for c in candidates],
                "relevance_scores": list(scores),
            }
        )

        for cand, score in zip(candidates, scores):
            pointwise.append(
                {
                    **base,
                    "query": query,
                    "document": cand.get("text", ""),
                    "document_id": cand.get("id"),
                    "label": 1 if float(score) > 0 else 0,
                    "relevance": float(score),
                }
            )

        ranked = sorted(
            zip(candidates, scores),
            key=lambda x: float(x[1]),
            reverse=True,
        )
        preferred = [(c, s) for c, s in ranked if float(s) > 0]
        rejected = [(c, s) for c, s in ranked if float(s) <= 0]
        if not preferred:
            preferred = ranked[:1]
            rejected = ranked[1:]
        pairs = 0
        for pref, _ in preferred:
            for rej, _ in rejected:
                pairwise.append(
                    {
                        **base,
                        "query": query,
                        "preferred": pref.get("text", ""),
                        "rejected": rej.get("text", ""),
                        "preferred_id": pref.get("id"),
                        "rejected_id": rej.get("id"),
                    }
                )
                pairs += 1
                if pairs >= max_pairwise_per_query:
                    break
            if pairs >= max_pairwise_per_query:
                break
        # deterministic shuffle for variety across queries
        if pairs == 0 and len(ranked) >= 2:
            a, b = ranked[0][0], ranked[1][0]
            pairwise.append(
                {
                    **base,
                    "query": query,
                    "preferred": a.get("text", ""),
                    "rejected": b.get("text", ""),
                    "preferred_id": a.get("id"),
                    "rejected_id": b.get("id"),
                }
            )
        rng.shuffle(pairwise[-max_pairwise_per_query:] if pairwise else pairwise)

    return {"pointwise": pointwise, "pairwise": pairwise, "listwise": listwise}
