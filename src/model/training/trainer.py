"""Training loop for EpistemicHGNN V1 (neuro-symbolic, ADR-013)."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import torch
import torch.nn as nn
from torch_geometric.loader import DataLoader

from src.model.data.types import NodeType
from src.model.training.config import TrainConfig


@dataclass
class EpochResult:
    loss: float
    stance_loss: float
    is_loss: float
    verdict_loss: float
    stance_acc: float  # fraction of evidence items with correct stance
    verdict_acc: float  # fraction of claims with correct verdict


class Trainer:
    """Trains EpistemicHGNN with H1 (stance) + H2 (IS regression) + verdict losses."""

    def __init__(
        self,
        model: nn.Module,
        config: TrainConfig,
        stance_class_weights: torch.Tensor | None = None,
        verdict_class_weights: torch.Tensor | None = None,
    ):
        self.model = model
        self.config = config
        self.device = torch.device(config.device)
        self.model.to(self.device)

        self.stance_criterion = nn.CrossEntropyLoss(
            weight=stance_class_weights.to(self.device)
            if stance_class_weights is not None
            else None
        )
        self.is_criterion = nn.MSELoss()
        self.verdict_criterion = nn.CrossEntropyLoss(
            weight=verdict_class_weights.to(self.device)
            if verdict_class_weights is not None
            else None
        )
        self.optimizer = torch.optim.Adam(
            model.parameters(), lr=config.lr, weight_decay=config.weight_decay
        )
        self.scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer, mode="min", patience=2, factor=0.5
        )

        self._best_val_loss = float("inf")
        self._best_val_acc = 0.0
        self._epochs_no_improve = 0
        ckpt_dir = Path(config.checkpoint_dir)
        ckpt_dir.mkdir(parents=True, exist_ok=True)
        self._ckpt_path = ckpt_dir / "best_model.pt"

    def _run_epoch(self, loader: DataLoader, train: bool) -> EpochResult:
        self.model.train(train)
        total_loss = total_stance = total_is = total_verdict = 0
        correct_stance = correct_verdict = n_ev = n_claims = 0

        with torch.set_grad_enabled(train):
            for batch in loader:
                batch = batch.to(self.device)
                out = self.model(batch)

                stance_y = batch[NodeType.EVIDENCE].stance_y.view(-1)  # [N_ev]
                is_y = batch[NodeType.EVIDENCE].is_y.view(-1, 1)  # [N_ev, 1]
                verdict_y = batch[NodeType.CLAIM].y.view(-1)  # [N_claims]

                # Stance loss is optional — v3-nli has no StanceHead
                if "stance_logits" in out:
                    s_loss = self.stance_criterion(out["stance_logits"], stance_y)
                    correct_stance += (
                        (out["stance_logits"].argmax(-1) == stance_y).sum().item()
                    )
                else:
                    s_loss = torch.tensor(0.0, device=self.device)

                i_loss = self.is_criterion(out["is_pred"], is_y)
                v_loss = self.verdict_criterion(out["verdict_logits"], verdict_y)
                loss = (
                    s_loss
                    + self.config.is_loss_weight * i_loss
                    + self.config.verdict_loss_weight * v_loss
                )

                if train:
                    self.optimizer.zero_grad()
                    loss.backward()
                    self.optimizer.step()

                nev = stance_y.size(0)
                nc = verdict_y.size(0)
                total_loss += loss.item() * nev
                total_stance += s_loss.item() * nev
                total_is += i_loss.item() * nev
                total_verdict += v_loss.item() * nc
                correct_verdict += (
                    (out["verdict_logits"].argmax(-1) == verdict_y).sum().item()
                )
                n_ev += nev
                n_claims += nc

        dev = max(n_ev, 1)
        dc = max(n_claims, 1)
        return EpochResult(
            loss=total_loss / dev,
            stance_loss=total_stance / dev,
            is_loss=total_is / dev,
            verdict_loss=total_verdict / dc,
            stance_acc=correct_stance / dev,
            verdict_acc=correct_verdict / dc,
        )

    def fit(
        self,
        train_loader: DataLoader,
        val_loader: DataLoader,
        verbose: bool = True,
    ) -> list[dict]:
        history = []
        for epoch in range(1, self.config.epochs + 1):
            tr = self._run_epoch(train_loader, train=True)
            val = self._run_epoch(val_loader, train=False)
            self.scheduler.step(val.loss)

            history.append(
                {
                    "epoch": epoch,
                    "train_loss": round(tr.loss, 4),
                    "train_s": round(tr.stance_loss, 4),
                    "train_is": round(tr.is_loss, 4),
                    "train_v": round(tr.verdict_loss, 4),
                    "train_s_acc": round(tr.stance_acc, 4),
                    "train_v_acc": round(tr.verdict_acc, 4),
                    "val_loss": round(val.loss, 4),
                    "val_s": round(val.stance_loss, 4),
                    "val_is": round(val.is_loss, 4),
                    "val_v": round(val.verdict_loss, 4),
                    "val_s_acc": round(val.stance_acc, 4),
                    "val_v_acc": round(val.verdict_acc, 4),
                }
            )

            if verbose:
                print(
                    f"Epoch {epoch:3d} | "
                    f"loss {tr.loss:.4f} "
                    f"(s={tr.stance_loss:.3f} is={tr.is_loss:.3f} v={tr.verdict_loss:.3f}) "
                    f"s_acc {tr.stance_acc:.3f} v_acc {tr.verdict_acc:.3f} | "
                    f"val loss {val.loss:.4f} "
                    f"(s={val.stance_loss:.3f} is={val.is_loss:.3f} v={val.verdict_loss:.3f}) "
                    f"s_acc {val.stance_acc:.3f} v_acc {val.verdict_acc:.3f}"
                )

            ckpt_payload = {
                "model_state_dict": self.model.state_dict(),
                "ec_threshold": self.config.ec_threshold,
            }
            # Checkpoint on acc; early stopping on loss — deliberately independent.
            if val.verdict_acc >= self._best_val_acc:
                self._best_val_acc = val.verdict_acc
                torch.save(ckpt_payload, self._ckpt_path)

            if val.loss < self._best_val_loss:
                self._best_val_loss = val.loss
                self._epochs_no_improve = 0
            else:
                self._epochs_no_improve += 1
                if self._epochs_no_improve >= self.config.patience:
                    if verbose:
                        print(f"Early stopping at epoch {epoch}")
                    break

        return history

    def load_best(self) -> None:
        ckpt = torch.load(self._ckpt_path, map_location=self.device, weights_only=False)
        state = ckpt.get("model_state_dict", ckpt) if isinstance(ckpt, dict) else ckpt
        self.model.load_state_dict(state)

    @torch.no_grad()
    def evaluate(self, loader: DataLoader) -> EpochResult:
        return self._run_epoch(loader, train=False)
