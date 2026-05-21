"""Text and categorical featurizer for GNN node construction."""

from __future__ import annotations

import hashlib
import pickle
import re
from pathlib import Path

import torch
from sentence_transformers import CrossEncoder, SentenceTransformer

from src.model.data.types import (
    EVIDENCE_TYPE_TO_INT,
    MODALITY_TO_INT,
    NUM_EVIDENCE_TYPE,
    NUM_MODALITY,
    NUM_REASONING_STRATEGY,
    NUM_SOURCE_TYPE,
    REASONING_STRATEGY_TO_INT,
    SOURCE_TYPE_TO_INT,
)

_EMBED_MODEL = "all-MiniLM-L6-v2"
_NLI_MODEL = "cross-encoder/nli-deberta-v3-small"

# Strips "Question: ... Answer:" prefix from AVeriTeC Q+A evidence text,
# keeping only the answer part which is what the NLI model should score.
_QA_ANSWER_RE = re.compile(r"(?i)^.*?answer\s*:\s*", re.DOTALL)
# Answers shorter than this are likely uninformative (dates, "No answer", etc.)
_MIN_ANSWER_LEN = 15


class Featurizer:
    """Encodes text to sentence embeddings and categorical fields to one-hot/multi-hot vectors.

    Loads the sentence-transformer model once and caches embeddings to a .pkl file
    keyed by text hash so re-runs skip re-embedding expensive text.
    """

    def __init__(self, cache_path: str | Path | None = None, device: str | None = None):
        self._device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self._nli_device = "cpu"  # CrossEncoder only runs at graph-build time, not training
        self._model: SentenceTransformer | None = None
        self._nli_model: CrossEncoder | None = None
        self._cache: dict[str, list[float]] = {}
        self._cache_path = Path(cache_path) if cache_path else None
        if self._cache_path and self._cache_path.exists():
            with open(self._cache_path, "rb") as f:
                self._cache = pickle.load(f)

    def _model_(self) -> SentenceTransformer:
        if self._model is None:
            self._model = SentenceTransformer(_EMBED_MODEL, device=self._device)
        return self._model

    def _hash(self, text: str) -> str:
        return hashlib.sha256(text.encode("utf-8")).hexdigest()[:16]

    def encode_texts(self, texts: list[str]) -> torch.Tensor:
        """Encode a list of strings → float tensor [N, 384].

        Hits the cache for already-seen texts; embeds misses in batch and updates cache.
        """
        results: list[list[float]] = []
        misses: list[tuple[int, str]] = []

        for i, t in enumerate(texts):
            h = self._hash(t)
            if h in self._cache:
                results.append(self._cache[h])
            else:
                results.append([])
                misses.append((i, t))

        if misses:
            idxs, miss_texts = zip(*misses)
            embeddings = self._model_().encode(list(miss_texts), convert_to_numpy=True)
            for i, emb in zip(idxs, embeddings):
                h = self._hash(miss_texts[list(idxs).index(i)])
                self._cache[h] = emb.tolist()
                results[i] = emb.tolist()

        return torch.tensor(results, dtype=torch.float32)

    def encode_modality(self, modality: str | None) -> torch.Tensor:
        """One-hot encode evidence modality → float tensor [NUM_MODALITY=5]."""
        vec = torch.zeros(NUM_MODALITY, dtype=torch.float32)
        idx = MODALITY_TO_INT.get(modality or "", -1)
        if idx >= 0:
            vec[idx] = 1.0
        return vec

    def encode_evidence_types(self, evidence_types: list[str]) -> torch.Tensor:
        """Multi-hot encode evidence types → float tensor [NUM_EVIDENCE_TYPE=5].

        An evidence item can have multiple types (e.g. perception + non_apprehension),
        so this is multi-hot rather than one-hot. postulation_derivation is not in
        the index (excluded from training) and is silently ignored.
        """
        vec = torch.zeros(NUM_EVIDENCE_TYPE, dtype=torch.float32)
        for et in evidence_types:
            idx = EVIDENCE_TYPE_TO_INT.get(et, -1)
            if idx >= 0:
                vec[idx] = 1.0
        return vec

    def encode_reasoning_strategy(self, strategy: str | None) -> torch.Tensor:
        """One-hot encode reasoning strategy → float tensor [NUM_REASONING_STRATEGY=6]."""
        vec = torch.zeros(NUM_REASONING_STRATEGY, dtype=torch.float32)
        idx = REASONING_STRATEGY_TO_INT.get(strategy or "", -1)
        if idx >= 0:
            vec[idx] = 1.0
        return vec

    def encode_source_type(self, category: str | None) -> torch.Tensor:
        """One-hot encode source category → float tensor [NUM_SOURCE_TYPE=6].

        category is already resolved to one of the 6 categories via
        get_source_category() in types.py — this method only does the one-hot encoding.
        """
        vec = torch.zeros(NUM_SOURCE_TYPE, dtype=torch.float32)
        idx = SOURCE_TYPE_TO_INT.get(category or "unknown", -1)
        if idx >= 0:
            vec[idx] = 1.0
        return vec

    def _extract_nli_text(self, text: str) -> str:
        """Strip 'Question: ... Answer:' prefix from Q+A evidence, keeping only the answer.

        Falls back to the original text when:
        - No Q+A pattern is found (non-AVeriTeC evidence)
        - The extracted answer is too short to be informative (date, 'No answer', etc.)
        """
        m = _QA_ANSWER_RE.match(text)
        if m:
            answer = text[m.end():].strip()
            if len(answer) >= _MIN_ANSWER_LEN:
                return answer
        return text

    def encode_nli_stance(self, claim: str, ev_texts: list[str]) -> torch.Tensor:
        """Cross-encode (claim, evidence) pairs → softmax probs [N_ev, 3].

        Column order: [p_contradiction, p_entailment, p_neutral] (MNLI label order).
        For Q+A format evidence (AVeriTeC), only the answer part is scored so
        the NLI model receives a natural premise instead of a Q+A wrapper.
        Returns empty tensor [0, 3] for empty ev_texts.
        """
        if not ev_texts:
            return torch.zeros(0, 3, dtype=torch.float32)
        if self._nli_model is None:
            self._nli_model = CrossEncoder(_NLI_MODEL, device=self._nli_device)
        nli_texts = [self._extract_nli_text(t) for t in ev_texts]
        pairs = [(t, claim) for t in nli_texts]   # evidence=premise, claim=hypothesis (standard NLI)
        scores = self._nli_model.predict(pairs, apply_softmax=True)  # np [N, 3]
        return torch.tensor(scores, dtype=torch.float32)

    def save_cache(self) -> None:
        if self._cache_path:
            self._cache_path.parent.mkdir(parents=True, exist_ok=True)
            with open(self._cache_path, "wb") as f:
                pickle.dump(self._cache, f)
