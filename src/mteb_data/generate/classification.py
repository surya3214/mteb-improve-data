"""Classification triplet / pair generation."""

from __future__ import annotations

import random
from collections import defaultdict
from typing import Any

from mteb_data.schema import CanonicalRecord


def generate_classification_triplets(
    records: list[CanonicalRecord],
    *,
    seed: int = 42,
    pairs_per_anchor: int = 1,
    allow_cross_language: bool = False,
) -> list[dict[str, Any]]:
    """Sample same-label positives and different-label negatives.

    When allow_cross_language is True, positives may come from another language if
    they share the same task-local class_id and task name (shared ontology).
    """
    rng = random.Random(seed)
    by_task_label: dict[tuple[str, Any], list[CanonicalRecord]] = defaultdict(list)
    by_task_label_lang: dict[tuple[str, Any, str], list[CanonicalRecord]] = defaultdict(list)

    cls_records = [r for r in records if r.family == "classification"]
    for rec in cls_records:
        key = (rec.provenance.task, rec.target["class_id"])
        by_task_label[key].append(rec)
        by_task_label_lang[(rec.provenance.task, rec.target["class_id"], rec.provenance.language)].append(rec)

    out: list[dict[str, Any]] = []
    for rec in cls_records:
        task = rec.provenance.task
        label = rec.target["class_id"]
        lang = rec.provenance.language

        if allow_cross_language:
            positives = [p for p in by_task_label[(task, label)] if p.provenance.original_id != rec.provenance.original_id]
        else:
            positives = [
                p
                for p in by_task_label_lang[(task, label, lang)]
                if p.provenance.original_id != rec.provenance.original_id
            ]
        if not positives:
            continue

        # Negatives: same language preferred, different label, same task.
        negatives = [
            n
            for (t, lab, lng), group in by_task_label_lang.items()
            if t == task and lng == lang and lab != label
            for n in group
        ]
        if not negatives:
            negatives = [
                n
                for (t, lab), group in by_task_label.items()
                if t == task and lab != label
                for n in group
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
                    "task": task,
                    "config": rec.provenance.config,
                    "language": lang,
                    "class_id": label,
                    "class_namespace": rec.target["class_namespace"],
                    "positive_language": pos.provenance.language,
                    "negative_language": neg.provenance.language,
                    "dataset": rec.provenance.dataset,
                    "revision": rec.provenance.revision,
                    "source_split": rec.provenance.source_split,
                    "original_id": rec.provenance.original_id,
                    "license": rec.provenance.license,
                }
            )
    return out
