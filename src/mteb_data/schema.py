"""Canonical schemas and provenance for MTEB training-data generation."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any


TARGET_LANGUAGES = {
    "en": "eng-Latn",
    "ko": "kor-Hang",
    "ar": "ara-Arab",
    "zh": "zho-Hans",
    "fr": "fra-Latn",
    "de": "deu-Latn",
    "hi": "hin-Deva",
    "id": "ind-Latn",
    "it": "ita-Latn",
    "ja": "jpn-Jpan",
    "pt": "por-Latn",
    "ru": "rus-Cyrl",
    "es": "spa-Latn",
    "vi": "vie-Latn",
    "th": "tha-Thai",
    "pl": "pol-Latn",
}

# Extra aliases seen in MTEB configs / Hub subsets.
LANGUAGE_ALIASES = {
    "eng": "eng-Latn",
    "kor": "kor-Hang",
    "kor-Kore": "kor-Hang",
    "ara": "ara-Arab",
    "arb": "ara-Arab",
    "arb_Arab": "ara-Arab",
    "arb_Latn": "ara-Arab",
    "zho": "zho-Hans",
    "cmn": "zho-Hans",
    "cmn-Hans": "zho-Hans",
    "zh-CN": "zho-Hans",
    "zh-TW": "zho-Hant",
    "fra": "fra-Latn",
    "deu": "deu-Latn",
    "hin": "hin-Deva",
    "ind": "ind-Latn",
    "ita": "ita-Latn",
    "jpn": "jpn-Jpan",
    "por": "por-Latn",
    "rus": "rus-Cyrl",
    "spa": "spa-Latn",
    "vie": "vie-Latn",
    "tha": "tha-Thai",
    "pol": "pol-Latn",
    "spanish": "spa-Latn",
    "english": "eng-Latn",
    "french": "fra-Latn",
}


def normalize_language(code: str | None) -> str | None:
    if code is None:
        return None
    if code in TARGET_LANGUAGES.values():
        return code
    if code in TARGET_LANGUAGES:
        return TARGET_LANGUAGES[code]
    if code in LANGUAGE_ALIASES:
        return LANGUAGE_ALIASES[code]
    # Pass through already-scripted BCP-47-like tags when known.
    for alias, canon in LANGUAGE_ALIASES.items():
        if code.startswith(alias):
            return canon
    return code


@dataclass(frozen=True)
class Provenance:
    task: str
    family: str
    dataset: str
    revision: str
    config: str
    source_split: str
    language: str
    license: str | None = None
    original_id: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class CanonicalRecord:
    provenance: Provenance
    family: str
    inputs: dict[str, Any]
    target: dict[str, Any] = field(default_factory=dict)
    group: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "provenance": self.provenance.to_dict(),
            "family": self.family,
            "inputs": self.inputs,
            "target": self.target,
            "group": self.group,
        }


def namespace_label(task: str, config: str, language: str, label: Any) -> str:
    return f"{task}/{config}/{language}/{label}"


def normalize_score(score: float, score_min: float, score_max: float) -> float:
    if score_max == score_min:
        return 0.0
    return (float(score) - score_min) / (score_max - score_min)
