"""Clustering metric-learning triplet generation."""

from __future__ import annotations

import random
from collections import defaultdict
from typing import Any

from mteb_data.schema import CanonicalRecord


def generate_clustering_triplets(
    records: list[CanonicalRecord],
    *,
    seed: int = 42,
    pairs_per_anchor: int = 1,
) -> list[dict[str, Any]]:
    rng = random.Random(seed)
    by_ns: dict[str, list[CanonicalRecord]] = defaultdict(list)
    by_experiment: dict[tuple[str, str], list[CanonicalRecord]] = defaultdict(list)

    cluster_records = [r for r in records if r.family == "clustering"]
    for rec in cluster_records:
        ns = rec.group["cluster_namespace"]
        by_ns[ns].append(rec)
        by_experiment[(rec.provenance.task, rec.group.get("experiment_id", "0"))].append(rec)

    # Only keep namespaces with >=2 members so positives exist.
    valid_ns = {ns for ns, group in by_ns.items() if len(group) >= 2}
    out: list[dict[str, Any]] = []
    for rec in cluster_records:
        ns = rec.group["cluster_namespace"]
        if ns not in valid_ns:
            continue
        positives = [p for p in by_ns[ns] if p.provenance.original_id != rec.provenance.original_id]
        if not positives:
            continue
        exp_key = (rec.provenance.task, rec.group.get("experiment_id", "0"))
        negatives = [
            n
            for n in by_experiment[exp_key]
            if n.group["cluster_namespace"] != ns
        ]
        if not negatives:
            continue
        for _ in range(pairs_per_anchor):
            pos = rng.choice(positives)
            neg = rng.choice(negatives)
            out.append(
                {
                    "anchor": rec.inputs["anchor"],
                    "positive": pos.inputs["anchor"],
                    "negative": neg.inputs["anchor"],
                    "task": rec.provenance.task,
                    "config": rec.provenance.config,
                    "language": rec.provenance.language,
                    "cluster_namespace": ns,
                    "experiment_id": rec.group.get("experiment_id"),
                    "dataset": rec.provenance.dataset,
                    "revision": rec.provenance.revision,
                    "source_split": rec.provenance.source_split,
                    "original_id": rec.provenance.original_id,
                    "license": rec.provenance.license,
                }
            )
    return out
