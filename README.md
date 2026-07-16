# mteb-improve-data

Generate provenance-preserving training data for maximizing **MTEB(Multilingual, v2)** scores on:

`en, ko, ar, zh, fr, de, hi, id, it, ja, pt, ru, es, vi, th, pl`

across **STS, Retrieval, Classification, Clustering, Reranking**.

This repository builds datasets. It does **not** train or evaluate embedding models.

## Important

Models trained on these MTEB train splits are **benchmark-adapted**, not zero-shot. Disclose overlap with `ModelMeta.training_datasets`.

Many Multilingual v2 Retrieval / Reranking / Clustering tasks have **no usable train split**. The catalog records them as ineligible instead of silently using `dev`/`test`.

## Install

```bash
pip install -e ".[dev]"
```

## Commands

```bash
# Summarize eligible vs ineligible tasks
mteb-data audit

# Show prioritized external sources (NLLB, CCMatrix, xP3x, ...)
mteb-data external

# Refuse eval splits even if physically named train
mteb-data check-split --task FinancialPhrasebankClassification --split train

# Build training views (downloads eligible Hub splits)
mteb-data build --output data/processed --max-rows-per-split 200
mteb-data build --tasks STSBenchmark MassiveIntentClassification --max-rows-per-split 500
mteb-data build --families sts classification
```

## What the pipeline produces

Under `data/processed/<build-id>/`:

| File | Contents |
|------|----------|
| `sts_pairs.train.parquet` | Scored sentence pairs (`score` in `[0,1]`) |
| `metric_triplets.classification.train.parquet` | Anchor / positive / negative |
| `metric_triplets.clustering.train.parquet` | Cluster-aware triplets (when eligible data exists) |
| `manifest.json` | Revisions, exclusions, hashes, mixture, MTEB overlap names |
| `mixture.json` | Family/language/task sampling weights |

Retrieval and reranking adapters/generators are implemented and unit-tested. Official Multilingual v2 retrieval/reranking tasks in the catalog are currently **ineligible** (no safe train split), so use external sources for those families.

## Canonical schemas

Raw Hub rows are normalized before pair/triplet expansion:

- **STS:** `anchor`, `positive`, raw + normalized score, provenance
- **Classification:** `anchor`, namespaced `class_id`
- **Clustering:** flattened sentences with experiment-aware cluster namespaces
- **Retrieval:** query + positives + qrel IDs
- **Reranking:** query + candidates + graded relevance

Provenance on every record: task, dataset, revision, config, source split, language/script, license, original IDs.

## Suggested SentenceTransformers losses

| Family | View | Loss |
|--------|------|------|
| STS | scored pairs | `CoSENTLoss` / `AnglELoss` |
| Classification / Clustering | triplets | `MultipleNegativesRankingLoss` or batch-hard triplet |
| Retrieval | query/pos/negs | `CachedMultipleNegativesRankingLoss` |
| Reranking | pointwise / listwise | `CrossEncoder` BCE / listwise losses |

## External data (essential beyond official train splits)

Priority sources are listed in [`configs/external_sources.yaml`](configs/external_sources.yaml):

1. NLLB / CCMatrix bitext
2. NVIDIA Embed-Nemotron (filter overlap)
3. xP3x instruction pairs (filter overlap)
4. Wikipedia structural pairs / S2ORC

Use streaming + `default_cap` so first runs stay bounded.

## Tests

```bash
pytest -q
```
