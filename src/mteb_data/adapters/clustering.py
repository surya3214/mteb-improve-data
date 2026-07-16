"""Clustering adapter with nested-experiment flattening."""

from __future__ import annotations

from typing import Any, Iterable

from mteb_data.adapters import make_provenance, row_id
from mteb_data.catalog import TaskConfig
from mteb_data.schema import CanonicalRecord, namespace_label


def _is_nested(sentences: Any, labels: Any) -> bool:
    return (
        isinstance(sentences, list)
        and sentences
        and isinstance(sentences[0], (list, tuple))
        and isinstance(labels, list)
        and labels
        and isinstance(labels[0], (list, tuple))
    )


def adapt_clustering_rows(
    rows: Iterable[dict[str, Any]],
    *,
    task: TaskConfig,
    config_name: str,
    language: str,
    source_split: str,
) -> list[CanonicalRecord]:
    """Flatten clustering rows into one record per sentence.

    Nested MTEB legacy rows become multiple experiments with distinct namespaces.
    """
    records: list[CanonicalRecord] = []
    for i, row in enumerate(rows):
        sentences = row.get("sentences") or row.get("text")
        labels = row.get("labels") or row.get("label") or row.get("category")
        oid = row_id(row, i)
        if sentences is None or labels is None:
            continue

        if isinstance(sentences, str):
            # Flat single-text row (e.g. SIB200-style before transform).
            label = labels
            ns = namespace_label(task.name, config_name, language, label)
            records.append(
                CanonicalRecord(
                    provenance=make_provenance(
                        task=task.name,
                        family="clustering",
                        dataset=task.dataset,
                        revision=task.revision,
                        config=config_name,
                        source_split=source_split,
                        language=language,
                        license=task.license,
                        original_id=oid,
                    ),
                    family="clustering",
                    inputs={"anchor": sentences},
                    target={"cluster_id": label, "cluster_namespace": ns},
                    group={"cluster_namespace": ns, "experiment_id": "0"},
                )
            )
            continue

        if _is_nested(sentences, labels):
            experiments = zip(sentences, labels)
            for exp_idx, (exp_sents, exp_labels) in enumerate(experiments):
                for j, (sent, label) in enumerate(zip(exp_sents, exp_labels)):
                    ns = f"{task.name}/{config_name}/{language}/exp{exp_idx}/{label}"
                    records.append(
                        CanonicalRecord(
                            provenance=make_provenance(
                                task=task.name,
                                family="clustering",
                                dataset=task.dataset,
                                revision=task.revision,
                                config=config_name,
                                source_split=source_split,
                                language=language,
                                license=task.license,
                                original_id=f"{oid}:exp{exp_idx}:{j}",
                            ),
                            family="clustering",
                            inputs={"anchor": str(sent)},
                            target={"cluster_id": label, "cluster_namespace": ns},
                            group={"cluster_namespace": ns, "experiment_id": str(exp_idx)},
                        )
                    )
            continue

        # Flat list experiment.
        for j, (sent, label) in enumerate(zip(sentences, labels)):
            ns = namespace_label(task.name, config_name, language, label)
            records.append(
                CanonicalRecord(
                    provenance=make_provenance(
                        task=task.name,
                        family="clustering",
                        dataset=task.dataset,
                        revision=task.revision,
                        config=config_name,
                        source_split=source_split,
                        language=language,
                        license=task.license,
                        original_id=f"{oid}:{j}",
                    ),
                    family="clustering",
                    inputs={"anchor": str(sent)},
                    target={"cluster_id": label, "cluster_namespace": ns},
                    group={"cluster_namespace": ns, "experiment_id": "0"},
                )
            )
    return records
