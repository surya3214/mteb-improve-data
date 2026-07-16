"""Loss-specific training-view generators."""

from mteb_data.generate.classification import generate_classification_triplets
from mteb_data.generate.clustering import generate_clustering_triplets
from mteb_data.generate.reranking import generate_reranking_views
from mteb_data.generate.retrieval import generate_retrieval_triples
from mteb_data.generate.sts import generate_sts_pairs

__all__ = [
    "generate_classification_triplets",
    "generate_clustering_triplets",
    "generate_reranking_views",
    "generate_retrieval_triples",
    "generate_sts_pairs",
]
