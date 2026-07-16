"""End-to-end build orchestration."""

from __future__ import annotations

from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from mteb_data.catalog import Catalog, audit_catalog, load_catalog
from mteb_data.generate.classification import generate_classification_triplets
from mteb_data.generate.clustering import generate_clustering_triplets
from mteb_data.generate.sts import generate_sts_pairs
from mteb_data.io import ensure_build_dir, grouped_split, hash_rows, write_json, write_parquet
from mteb_data.mixture import compute_mixture_weights, counts_from_rows
from mteb_data.schema import CanonicalRecord
from mteb_data.sources.external import load_external_sources
from mteb_data.sources.mteb import EvaluationSplitError, iter_task_loads, normalize_task_split


def build_dataset(
    *,
    output_root: Path,
    catalog_path: Path | None = None,
    families: list[str] | None = None,
    task_names: list[str] | None = None,
    max_rows_per_split: int | None = None,
    seed: int = 42,
    pairs_per_anchor: int = 1,
    allow_cross_language_classification: bool = False,
    include_external_recommendations: bool = True,
) -> dict[str, Any]:
    catalog = load_catalog(catalog_path)
    build_id = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    out_dir = ensure_build_dir(output_root, build_id)

    canonical: list[CanonicalRecord] = []
    load_log: list[dict[str, Any]] = []
    exclusions: list[dict[str, Any]] = []

    for task in catalog.tasks:
        if not task.training_eligible:
            exclusions.append(
                {
                    "task": task.name,
                    "family": task.family,
                    "reason": task.ineligibility_reason,
                    "eval_splits": task.eval_splits,
                }
            )

    for task, cfg, split in iter_task_loads(
        catalog, families=families, task_names=task_names, max_rows_per_split=max_rows_per_split
    ):
        try:
            result = normalize_task_split(task, cfg, split, max_rows=max_rows_per_split)
        except EvaluationSplitError as exc:
            exclusions.append({"task": task.name, "config": cfg.get("name"), "split": split, "reason": str(exc)})
            continue
        except Exception as exc:  # Hub / network / schema issues should not abort the whole build.
            load_log.append(
                {
                    "task": task.name,
                    "config": cfg.get("name"),
                    "split": split,
                    "status": "error",
                    "error": str(exc),
                }
            )
            continue

        canonical.extend(result.records)
        load_log.append(
            {
                "task": result.task,
                "config": result.config,
                "split": result.split,
                "status": "ok" if not result.skipped_reason else "skipped",
                "n_records": len(result.records),
                "skipped_reason": result.skipped_reason,
            }
        )

    # Persist canonical JSONL-ish parquet via flattened dicts.
    canonical_rows = [r.to_dict() for r in canonical]
    write_json(out_dir / "canonical_sample.json", canonical_rows[:20])

    sts_rows = generate_sts_pairs(canonical)
    cls_rows = generate_classification_triplets(
        canonical,
        seed=seed,
        pairs_per_anchor=pairs_per_anchor,
        allow_cross_language=allow_cross_language_classification,
    )
    cluster_rows = generate_clustering_triplets(
        canonical, seed=seed, pairs_per_anchor=pairs_per_anchor
    )

    sts_train, sts_val = grouped_split(sts_rows, group_key="original_id", seed=seed)
    cls_train, cls_val = grouped_split(cls_rows, group_key="original_id", seed=seed)
    cluster_train, cluster_val = grouped_split(cluster_rows, group_key="original_id", seed=seed)

    write_parquet(out_dir / "sts_pairs.train.parquet", sts_train)
    write_parquet(out_dir / "sts_pairs.val.parquet", sts_val)
    write_parquet(out_dir / "metric_triplets.classification.train.parquet", cls_train)
    write_parquet(out_dir / "metric_triplets.classification.val.parquet", cls_val)
    write_parquet(out_dir / "metric_triplets.clustering.train.parquet", cluster_train)
    write_parquet(out_dir / "metric_triplets.clustering.val.parquet", cluster_val)

    counts: dict[str, dict[str, int]] = {
        "sts": counts_from_rows(sts_train, "sts"),
        "classification": counts_from_rows(cls_train, "classification"),
        "clustering": counts_from_rows(cluster_train, "clustering"),
        "retrieval": {},
        "reranking": {},
    }
    mixture = compute_mixture_weights(counts)

    overlap_tasks = sorted({r.provenance.task for r in canonical})
    external = []
    if include_external_recommendations:
        external = [
            {
                "name": s.name,
                "dataset": s.dataset,
                "priority": s.priority,
                "overlap_risk": s.overlap_risk,
                "default_cap": s.default_cap,
                "license": s.license,
                "notes": s.notes,
            }
            for s in load_external_sources()
        ]

    manifest = {
        "build_id": build_id,
        "benchmark": catalog.benchmark,
        "mteb_commit": catalog.mteb_commit,
        "seed": seed,
        "max_rows_per_split": max_rows_per_split,
        "benchmark_adapted": True,
        "zero_shot": False,
        "warning": (
            "Scores from models trained on these MTEB train splits are benchmark-adapted, "
            "not zero-shot. Disclose overlap via ModelMeta.training_datasets."
        ),
        "training_datasets_for_mteb_meta": overlap_tasks,
        "counts": {
            "canonical_records": len(canonical),
            "sts_train": len(sts_train),
            "sts_val": len(sts_val),
            "classification_train": len(cls_train),
            "classification_val": len(cls_val),
            "clustering_train": len(cluster_train),
            "clustering_val": len(cluster_val),
        },
        "hashes": {
            "sts_train": hash_rows(sts_train[:1000]),
            "classification_train": hash_rows(cls_train[:1000]),
            "clustering_train": hash_rows(cluster_train[:1000]),
        },
        "mixture": mixture,
        "loads": load_log,
        "exclusions": exclusions,
        "external_sources": external,
        "audit": audit_catalog(catalog),
    }
    write_json(out_dir / "manifest.json", manifest)
    write_json(out_dir / "mixture.json", mixture)
    return manifest
