from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


class MLPBackbone(nn.Module):
    def __init__(self, n_genes: int, n_classes: int, hidden: int = 128, dropout: float = 0.2, evidential: bool = False):
        super().__init__()
        self.evidential = evidential
        self.net = nn.Sequential(
            nn.Linear(n_genes, hidden),
            nn.LayerNorm(hidden),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden, max(2, hidden // 2)),
            nn.LayerNorm(max(2, hidden // 2)),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(max(2, hidden // 2), n_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self.net(x)
        if self.evidential:
            return F.softplus(out) + 1e-4
        return out


def soft_cross_entropy(logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
    return -(targets * F.log_softmax(logits, dim=1)).sum(dim=1).mean()


def evidential_loss(evidence: torch.Tensor, targets: torch.Tensor, kl_weight: float) -> torch.Tensor:
    alpha = evidence + 1.0
    strength = alpha.sum(dim=1, keepdim=True)
    nll = (targets * (torch.digamma(strength) - torch.digamma(alpha))).sum(dim=1).mean()

    beta = torch.ones_like(alpha)
    sum_beta = beta.sum(dim=1, keepdim=True)
    ln_b_alpha = torch.lgamma(alpha).sum(dim=1, keepdim=True) - torch.lgamma(strength)
    ln_b_beta = torch.lgamma(beta).sum(dim=1, keepdim=True) - torch.lgamma(sum_beta)
    kl = ((alpha - beta) * (torch.digamma(alpha) - torch.digamma(strength))).sum(dim=1, keepdim=True)
    kl = (kl + ln_b_beta - ln_b_alpha).mean()
    return nll + kl_weight * kl


def train_mlp(
    model: nn.Module,
    x_train: np.ndarray,
    y_train: np.ndarray,
    *,
    evidential: bool,
    epochs: int,
    batch_size: int,
    lr: float,
    seed: int,
    device: str,
) -> list[float]:
    torch.manual_seed(seed)
    model.to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    x_tensor = torch.tensor(x_train, dtype=torch.float32)
    y_tensor = torch.tensor(y_train, dtype=torch.float32)
    rng = np.random.default_rng(seed)
    losses = []
    for epoch in range(epochs):
        order = rng.permutation(x_train.shape[0])
        epoch_losses = []
        model.train()
        for start in range(0, len(order), batch_size):
            idx = order[start : start + batch_size]
            xb = x_tensor[idx].to(device)
            yb = y_tensor[idx].to(device)
            out = model(xb)
            if evidential:
                loss = evidential_loss(out, yb, kl_weight=min(0.02, (epoch + 1) / max(epochs, 1) * 0.02))
            else:
                loss = soft_cross_entropy(out, yb)
            opt.zero_grad()
            loss.backward()
            opt.step()
            epoch_losses.append(float(loss.detach().cpu()))
        losses.append(float(np.mean(epoch_losses)))
    return losses


def predict_models(
    models: list[nn.Module],
    x: np.ndarray,
    *,
    evidential: bool,
    device: str,
    batch_size: int = 4096,
) -> tuple[np.ndarray, np.ndarray, np.ndarray | None]:
    mean_chunks = []
    disagreement_chunks = []
    strength_chunks = []
    with torch.no_grad():
        for start in range(0, x.shape[0], batch_size):
            stop = min(start + batch_size, x.shape[0])
            xb = torch.tensor(x[start:stop], dtype=torch.float32).to(device)
            probs = []
            strengths = []
            for model in models:
                model.eval()
                out = model(xb)
                if evidential:
                    alpha = out + 1.0
                    probs.append((alpha / alpha.sum(dim=1, keepdim=True)).cpu().numpy())
                    strengths.append(alpha.sum(dim=1).cpu().numpy())
                else:
                    probs.append(torch.softmax(out, dim=1).cpu().numpy())
            stack = np.stack(probs, axis=0)
            mean_chunks.append(stack.mean(axis=0))
            disagreement_chunks.append(stack.var(axis=0).mean(axis=1))
            if strengths:
                strength_chunks.append(np.mean(np.stack(strengths, axis=0), axis=0))
            del xb
    mean_prob = np.concatenate(mean_chunks, axis=0)
    disagreement = np.concatenate(disagreement_chunks, axis=0)
    strength = np.concatenate(strength_chunks, axis=0) if strength_chunks else None
    return mean_prob, disagreement, strength

