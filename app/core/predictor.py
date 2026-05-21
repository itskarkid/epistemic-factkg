"""EpistemicPredictor — thin inference orchestrator (app).

Responsibilities (only):
  1. Load model checkpoint + registry.
  2. Resolve source IDs / trust values from the registry.
  3. Call ClaimGraphBuilder.build_from_items() → graph.
  4. Call model.predict() → raw output.
  5. Call model.build_prediction_payload() → app-level dict.

Payload construction and graph-record assembly live in src/.
"""

from __future__ import annotations

import re
from pathlib import Path
from urllib.parse import urlparse

import torch

from src.epistemic.registry import (
    get_source_trust,
    load_source_trust_registry,
    resolve_source_id,
)
from src.model.config import GraphConfig
from src.model.data.builder import ClaimGraphBuilder
from src.model.data.featurizer import Featurizer
from src.model.data.types import get_source_category
from src.model.models import MODELS
from src.model.models.nlihybridhgnn import NLIHybridHGNN

_ARCHIVE_RE = re.compile(r"web\.archive\.org/web/\d+[^/]*/(.+)")


def _infer_hparams(state: dict) -> tuple[int, int]:
    for key, val in state.items():
        if "att_src" in key:
            return int(val.shape[1]) * int(val.shape[2]), int(val.shape[1])
    raise ValueError("Cannot infer hidden_dim/heads: no 'att_src' key in checkpoint")


class EpistemicPredictor:
    """Loads a trained checkpoint and runs inference on free-text claim + evidence.

    Args:
        model_name:  Key from src/model/models.MODELS.
        models_root: Directory containing per-model subdirs with checkpoints/.
    """

    def __init__(
        self,
        model_name: str = "v2-hgnn",
        models_root: Path | None = None,
        registry_path: Path | None = None,
    ) -> None:
        self.model_name = model_name
        root       = models_root or Path("out/model")
        ckpt_dir   = root / model_name / "checkpoints"
        checkpoint = ckpt_dir / "best_model.pt"
        if not checkpoint.exists():
            raise FileNotFoundError(
                f"No checkpoint at {checkpoint}. Run: just train {model_name}"
            )

        reg_path = registry_path or Path("data/registry/source_trust_registry.jsonl")
        self._registry = load_source_trust_registry(reg_path)
        is_nli = MODELS.get(model_name) is NLIHybridHGNN
        self._builder = ClaimGraphBuilder(
            registry=self._registry,
            featurizer=Featurizer(),
            use_nli=is_nli,
        )

        model_cls = MODELS[model_name]
        graph_cfg = GraphConfig.v2() if is_nli else GraphConfig.v1()
        ckpt  = torch.load(str(checkpoint), map_location="cpu", weights_only=False)
        state = ckpt.get("model_state_dict", ckpt)
        ec_thr = ckpt.get("ec_threshold", 0.35) if isinstance(ckpt, dict) else 0.35
        if not (0.0 < ec_thr < 1.0):
            raise ValueError(f"Invalid ec_threshold in checkpoint: {ec_thr}. Expected (0, 1).")
        self._ec_threshold = ec_thr
        hidden_dim, heads = _infer_hparams(state)
        self._model = model_cls(
            graph_cfg, hidden_dim=hidden_dim, heads=heads,
            dropout=0.3, ec_threshold=self._ec_threshold,
        )
        self._model.load_state_dict(state)
        self._model.eval()

    def predict(self, claim: str, evidence_items: list[dict]) -> dict:
        """Run inference on a claim + evidence list.

        evidence_items: [{text, modality, source_type, source_id?, url?}]
        Returns app-level payload from model.build_prediction_payload().
        """
        resolved = self._resolve_sources(evidence_items)
        graph    = self._builder.build_from_items(claim, resolved)

        if graph is None:
            raise ValueError(
                "All evidence items were filtered. "
                "Provide at least one non-empty evidence text."
            )

        with torch.no_grad():
            out = self._model.predict(graph.data)

        result = self._model.build_prediction_payload(out, graph.data, resolved)
        result["claim_text"]  = claim
        result["_raw_out"]    = out        # for arc_block_definition parameterized mode
        result["hetero_data"] = graph.data # for pyvis interactive graph
        return result

    def predict_from_record(self, record: dict) -> dict:
        """Run prediction on a full v3.0 record (evaluation path).

        Calls build(record) directly — source IDs, triples, and inference_strength
        are already set in the record, so source resolution is not needed.
        """
        graph = self._builder.build(record)
        if graph is None:
            raise ValueError("Record produced an empty graph after filtering.")

        with torch.no_grad():
            out = self._model.predict(graph.data)

        resolved_items = [
            {
                "text":        ev.get("text", ""),
                "modality":    ev.get("modality", "web_text"),
                "source_type": get_source_category(ev.get("source_id", ""), self._registry),
                "source_id":   ev.get("source_id", ""),
            }
            for ev in record.get("evidence", [])
        ]
        result = self._model.build_prediction_payload(out, graph.data, resolved_items)
        result["claim_triples"] = record.get("claim_triples") or []
        result["claim_text"]    = record.get("claim", "")
        result["_raw_out"]      = out
        result["hetero_data"]   = graph.data
        return result

    # ── Self-describing passthroughs ─────────────────────────────────────────

    def arc_block_definition(self, inference_out=None):
        return self._model.arc_block_definition(inference_out)

    def result_dot(self, result: dict) -> str:
        return self._model.result_dot(result)

    def decision_path_info(self, result: dict) -> dict:
        return self._model.decision_path_info(result)

    def evidence_table(self, result: dict) -> list[dict]:
        return self._model.evidence_table(result)

    # ── Private: source resolution ────────────────────────────────────────────

    def _resolve_sources(self, evidence_items: list[dict]) -> list[dict]:
        result = []
        for ev in evidence_items:
            pre_sid = ev.get("source_id", "")
            if pre_sid and pre_sid in self._registry:
                trust = get_source_trust(pre_sid, self._registry)
                result.append({**ev, "source_id": pre_sid, "resolved_trust": trust})
            else:
                modality  = ev.get("modality", "web_text")
                source_id, trust = self._resolve_source(
                    ev.get("url"), ev.get("source_type", "unknown"), modality
                )
                result.append({**ev, "source_id": source_id, "resolved_trust": trust})
        return result

    def _resolve_source(self, url: str | None, source_type: str, modality: str) -> tuple[str, float]:
        if url:
            url = url.strip()
            m = _ARCHIVE_RE.search(url)
            if m:
                embedded = m.group(1)
                if not embedded.startswith(("http://", "https://")):
                    embedded = "https://" + embedded
                url = embedded
            parsed = urlparse(url if "://" in url else "https://" + url)
            domain = parsed.netloc or parsed.path.split("/")[0]
            domain = domain.lower().removeprefix("www.")
            if domain:
                url_modality = "pdf" if url.lower().endswith(".pdf") else "web_text"
                source_id    = resolve_source_id(domain, url_modality, self._registry)
                trust        = get_source_trust(source_id, self._registry)
                return source_id, trust
        # No URL: derive from source_type + modality — always in registry
        source_id = f"{source_type}_{modality}"
        return source_id, get_source_trust(source_id, self._registry)
