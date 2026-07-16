"""Hard-negative mining helpers."""

from __future__ import annotations

from typing import Any, Iterable, Mapping, Sequence


def filter_false_negatives(
    candidates: Sequence[Mapping[str, Any]],
    *,
    positive_ids: set[str],
    teacher_scores: Mapping[str, float] | None = None,
    max_teacher_score: float | None = None,
) -> list[dict[str, Any]]:
    """Drop known positives and optionally high-scoring teacher false negatives."""
    out: list[dict[str, Any]] = []
    teacher_scores = teacher_scores or {}
    for cand in candidates:
        cid = str(cand.get("id"))
        if cid in positive_ids:
            continue
        if max_teacher_score is not None and cid in teacher_scores:
            if float(teacher_scores[cid]) >= max_teacher_score:
                continue
        out.append(dict(cand))
    return out


def select_rank_window(
    ranked_candidates: Sequence[Mapping[str, Any]],
    *,
    positive_ids: set[str],
    num_negatives: int = 8,
    skip_top_k: int = 0,
    teacher_scores: Mapping[str, float] | None = None,
    max_teacher_score: float | None = None,
) -> list[dict[str, Any]]:
    """Keep a rank-window of negatives after excluding positives / false negatives."""
    filtered = filter_false_negatives(
        ranked_candidates[skip_top_k:],
        positive_ids=positive_ids,
        teacher_scores=teacher_scores,
        max_teacher_score=max_teacher_score,
    )
    return filtered[:num_negatives]


def attach_mined_negatives(
    query_to_ranked: Mapping[str, Sequence[Mapping[str, Any]]],
    query_to_positive_ids: Mapping[str, Iterable[str]],
    *,
    num_negatives: int = 8,
    skip_top_k: int = 0,
    teacher_scores_by_query: Mapping[str, Mapping[str, float]] | None = None,
    max_teacher_score: float | None = None,
) -> dict[str, list[dict[str, Any]]]:
    teacher_scores_by_query = teacher_scores_by_query or {}
    out: dict[str, list[dict[str, Any]]] = {}
    for qid, ranked in query_to_ranked.items():
        out[qid] = select_rank_window(
            ranked,
            positive_ids=set(map(str, query_to_positive_ids.get(qid, []))),
            num_negatives=num_negatives,
            skip_top_k=skip_top_k,
            teacher_scores=teacher_scores_by_query.get(qid),
            max_teacher_score=max_teacher_score,
        )
    return out
