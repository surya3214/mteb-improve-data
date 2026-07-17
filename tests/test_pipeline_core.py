"""Unit tests for schemas, adapters, generators, and split guards."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from mteb_data.adapters.classification import adapt_classification_rows
from mteb_data.adapters.clustering import adapt_clustering_rows
from mteb_data.adapters.reranking import adapt_reranking_rows
from mteb_data.adapters.retrieval import adapt_retrieval_tables
from mteb_data.adapters.sts import adapt_sts_rows
from mteb_data.catalog import load_catalog
from mteb_data.generate.classification import generate_classification_triplets
from mteb_data.generate.clustering import generate_clustering_triplets
from mteb_data.generate.reranking import generate_reranking_views
from mteb_data.generate.retrieval import generate_retrieval_triples
from mteb_data.generate.sts import generate_sts_pairs
from mteb_data.io import grouped_split, hash_rows, read_parquet, write_parquet
from mteb_data.mining import attach_mined_negatives, select_rank_window
from mteb_data.mixture import compute_mixture_weights
from mteb_data.schema import normalize_score
from mteb_data.sources.mteb import EvaluationSplitError, assert_trainable_split


@pytest.fixture
def catalog():
    return load_catalog()


def _task(catalog, name: str):
    return catalog.by_name(name)


def test_audit_counts(catalog):
    eligible = catalog.eligible_tasks()
    assert any(t.name == "MassiveIntentClassification" for t in eligible)
    assert any(t.name == "STSBenchmark" for t in eligible)
    assert catalog.by_name("FinancialPhrasebankClassification").training_eligible is False
    assert catalog.by_name("SIB200ClusteringS2S").training_eligible is False
    assert catalog.by_name("MIRACLRetrievalHardNegatives").training_eligible is False


def test_refuse_eval_split_named_train(catalog):
    task = _task(catalog, "FinancialPhrasebankClassification")
    with pytest.raises(EvaluationSplitError):
        assert_trainable_split(task, "train")


def test_refuse_ineligible_clustering_train(catalog):
    task = _task(catalog, "SIB200ClusteringS2S")
    with pytest.raises(EvaluationSplitError):
        assert_trainable_split(task, "train")


def test_include_hub_train_override(catalog):
    fin = _task(catalog, "FinancialPhrasebankClassification")
    sib = _task(catalog, "SIB200ClusteringS2S")
    meta = assert_trainable_split(fin, "train", include_hub_train=True)
    assert meta["contamination"] is True
    meta2 = assert_trainable_split(sib, "train", include_hub_train=True)
    assert meta2["contamination"] is True

    default_names = {t.name for t in catalog.eligible_tasks()}
    override_names = {t.name for t in catalog.eligible_tasks(include_hub_train=True)}
    assert "FinancialPhrasebankClassification" not in default_names
    assert "SIB200ClusteringS2S" not in default_names
    assert "FinancialPhrasebankClassification" in override_names
    assert "SIB200ClusteringS2S" in override_names
    assert sib.selectable_splits(include_hub_train=True) == ["train"]
    # Tasks without a Hub train stay ineligible.
    assert catalog.by_name("MIRACLRetrievalHardNegatives").is_buildable(include_hub_train=True) is False


def test_normalize_score():
    assert normalize_score(2.5, 0, 5) == 0.5
    assert normalize_score(1, 1, 4) == 0.0
    assert normalize_score(4, 1, 4) == 1.0


def test_sts_adapter_and_pairs(catalog):
    task = _task(catalog, "STSBenchmark")
    rows = [
        {"id": "1", "sentence1": "a cat", "sentence2": "a feline", "score": 4.0},
        {"id": "2", "sentence1": "car", "sentence2": "banana", "score": 0.5},
    ]
    records = adapt_sts_rows(
        rows, task=task, config_name="default", language="eng-Latn", source_split="train"
    )
    assert records[0].target["normalized_similarity"] == 0.8
    pairs = generate_sts_pairs(records)
    assert pairs[0]["score"] == 0.8
    assert pairs[0]["task"] == "STSBenchmark"
    assert pairs[0]["revision"] == task.revision


def test_classification_triplets_same_label(catalog):
    task = _task(catalog, "MassiveIntentClassification")
    rows = [
        {"id": "a1", "text": "cancel transfer", "label": 1},
        {"id": "a2", "text": "stop payment", "label": 1},
        {"id": "b1", "text": "check balance", "label": 2},
        {"id": "b2", "text": "show funds", "label": 2},
    ]
    records = adapt_classification_rows(
        rows, task=task, config_name="en", language="eng-Latn", source_split="train"
    )
    triples = generate_classification_triplets(records, seed=0, pairs_per_anchor=1)
    assert triples
    for t in triples:
        assert t["anchor"] != t["positive"]
        assert t["negative"] != t["positive"]
        assert "class_namespace" in t


def test_classification_cross_language_positives(catalog):
    task = _task(catalog, "MassiveIntentClassification")
    en = adapt_classification_rows(
        [{"id": "e1", "text": "cancel", "label": 7}, {"id": "e2", "text": "abort", "label": 8}],
        task=task,
        config_name="en",
        language="eng-Latn",
        source_split="train",
    )
    ja = adapt_classification_rows(
        [{"id": "j1", "text": "キャンセル", "label": 7}, {"id": "j2", "text": "残高", "label": 8}],
        task=task,
        config_name="ja",
        language="jpn-Jpan",
        source_split="train",
    )
    triples = generate_classification_triplets(en + ja, seed=1, allow_cross_language=True)
    assert any(t["language"] != t["positive_language"] for t in triples)


def test_nested_clustering_flatten_and_triplets(catalog):
    task = _task(catalog, "WikiCitiesClustering")
    rows = [
        {
            "id": "row0",
            "sentences": [["football", "soccer", "law"], ["cancer", "gpu"]],
            "labels": [[0, 0, 1], [4, 9]],
        }
    ]
    records = adapt_clustering_rows(
        rows, task=task, config_name="default", language="eng-Latn", source_split="train"
    )
    # experiment 0 has two members for cluster 0; experiment 1 has singleton clusters only
    assert len(records) == 5
    assert {r.group["experiment_id"] for r in records} == {"0", "1"}
    triples = generate_clustering_triplets(records, seed=0)
    assert triples
    assert all(t["experiment_id"] == "0" for t in triples)


def test_retrieval_positive_exclusion(catalog):
    task = _task(catalog, "ArguAna")
    corpus = [
        {"id": "d1", "text": "relevant doc"},
        {"id": "d2", "text": "hard negative"},
        {"id": "d3", "text": "other"},
    ]
    queries = [{"id": "q1", "text": "question"}]
    qrels = [{"query-id": "q1", "corpus-id": "d1", "score": 1}]
    records = adapt_retrieval_tables(
        corpus=corpus,
        queries=queries,
        qrels=qrels,
        task=task,
        config_name="default",
        language="eng-Latn",
        source_split="train",
    )
    mined = {"q1": [{"id": "d1", "text": "relevant doc"}, {"id": "d2", "text": "hard negative"}]}
    triples = generate_retrieval_triples(
        records, mined_negatives=mined, corpus_pool=corpus, num_negatives=2, seed=0
    )
    assert triples[0]["negative_ids"] == ["d2", "d3"]
    assert "d1" not in triples[0]["negative_ids"]


def test_reranking_views(catalog):
    task = _task(catalog, "T2Reranking")
    rows = [
        {
            "query-id": "q1",
            "query": "cancel transfer",
            "positive": ["how to cancel"],
            "negative": ["check balance", "fx fees"],
        }
    ]
    records = adapt_reranking_rows(
        rows, task=task, config_name="default", language="zho-Hans", source_split="dev"
    )
    views = generate_reranking_views(records, seed=0)
    assert views["listwise"][0]["relevance_scores"] == [1.0, 0.0, 0.0]
    assert views["pointwise"]
    assert views["pairwise"]


def test_mining_rank_window_and_teacher_filter():
    ranked = [{"id": str(i), "text": f"d{i}"} for i in range(10)]
    selected = select_rank_window(
        ranked,
        positive_ids={"1", "2"},
        num_negatives=3,
        skip_top_k=0,
        teacher_scores={"3": 0.95, "4": 0.1},
        max_teacher_score=0.9,
    )
    ids = [c["id"] for c in selected]
    assert "1" not in ids and "2" not in ids and "3" not in ids
    assert ids[0] == "0"
    attached = attach_mined_negatives(
        {"q": ranked},
        {"q": ["1"]},
        num_negatives=2,
    )
    assert len(attached["q"]) == 2


def test_grouped_split_no_leakage():
    rows = [
        {"original_id": "a", "text": "1"},
        {"original_id": "a", "text": "2"},
        {"original_id": "b", "text": "3"},
        {"original_id": "c", "text": "4"},
        {"original_id": "d", "text": "5"},
        {"original_id": "e", "text": "6"},
    ]
    train, val = grouped_split(rows, group_key="original_id", val_ratio=0.4, seed=0)
    train_ids = {r["original_id"] for r in train}
    val_ids = {r["original_id"] for r in val}
    assert train_ids.isdisjoint(val_ids)
    # group a stays together
    a_rows = [r for r in train + val if r["original_id"] == "a"]
    assert len(a_rows) == 2
    assert all(r in train for r in a_rows) or all(r in val for r in a_rows)


def test_parquet_roundtrip(tmp_path: Path):
    rows = [{"a": 1, "b": "x", "negatives": ["n1", "n2"]}]
    path = tmp_path / "t.parquet"
    write_parquet(path, rows)
    loaded = read_parquet(path)
    assert loaded[0]["a"] == 1
    assert json.loads(loaded[0]["negatives"]) == ["n1", "n2"]
    assert hash_rows(rows) == hash_rows(rows)


def test_mixture_weights_balance_families():
    counts = {
        "sts": {"eng-Latn::STS12": 1000},
        "classification": {"eng-Latn::A": 10, "jpn-Jpan::A": 10},
        "clustering": {},
        "retrieval": {"eng-Latn::R": 100000},
        "reranking": {"zho-Hans::T": 50},
    }
    mix = compute_mixture_weights(counts, temperature=0.4)
    assert abs(sum(mix["families"].values()) - 1.0) < 1e-6
    assert set(mix["families"]) == {"sts", "classification", "retrieval", "reranking"}


def test_build_all_writes_external(tmp_path: Path, monkeypatch):
    from mteb_data import pipeline as pipeline_mod
    from mteb_data.sources.external import ExternalSource

    fake_source = ExternalSource(
        name="toy",
        dataset="toy/ds",
        family="bitext",
        priority=1,
        description="toy",
        languages=["en"],
        shape="pairs",
        license="mit",
        overlap_risk="low",
        default_cap=10,
    )

    def fake_loads(*args, **kwargs):
        return iter(())

    def fake_external_sources(path=None):
        return [fake_source]

    def fake_stream(source, max_rows=None):
        n = max_rows or source.default_cap
        for i in range(n):
            yield {
                "family": "bitext",
                "anchor": f"a{i}",
                "positive": f"b{i}",
                "source": source.name,
                "dataset": source.dataset,
                "license": source.license,
                "overlap_risk": source.overlap_risk,
            }

    monkeypatch.setattr(pipeline_mod, "iter_task_loads", fake_loads)
    monkeypatch.setattr(pipeline_mod, "load_external_sources", fake_external_sources)
    monkeypatch.setattr(pipeline_mod, "stream_external_pairs", fake_stream)

    manifest = pipeline_mod.build_dataset(
        output_root=tmp_path,
        build_all=True,
        external_cap=3,
        include_external_recommendations=True,
    )
    assert manifest["build_all"] is True
    assert manifest["counts"]["external_pairs"] == 3
    assert (tmp_path / manifest["build_id"] / "external_pairs.train.parquet").exists()
