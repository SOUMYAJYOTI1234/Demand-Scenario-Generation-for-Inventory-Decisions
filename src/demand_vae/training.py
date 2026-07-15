"""Phase 7 training engine for the conditional VAE (roadmap Phase 7).

The loop implements the disciplined-training checklist: shuffled batching,
KL annealing (:func:`get_beta`), free bits, per-dimension KL monitoring with
the collapse detector after every epoch, early stopping on validation ELBO,
best + last checkpoints (config stored inside), and a CSV log so every run
is reconstructable. ``scripts/train.py`` is the thin CLI over this module.

Conventions: validation ELBO uses the raw (unfloored) KL and a fixed noise
seed per pass, so epochs are compared on the same estimator. "Best" means
highest validation ELBO (the tightest bound on held-out log-likelihood).
"""

from __future__ import annotations

import csv
import math
import time
from pathlib import Path

import torch

from demand_vae.models.cvae import ConditionalVAE
from demand_vae.models.vae import (
    collapse_warning,
    gaussian_kl_per_dim,
    get_beta,
    nb_vae_loss,
    reparameterize,
    vae_loss,
)

VAL_NOISE_SEED = 12345  # fixed val-pass noise: epochs compared on one estimator
ACTIVE_DIM_THRESHOLD = 0.1  # nats

LOG_COLUMNS = [
    "epoch",
    "beta",
    "train_recon",
    "train_kl",
    "train_elbo",
    "val_elbo",
    "kl_dim_min",
    "kl_dim_mean",
    "kl_dim_max",
    "active_dims",
    "seconds",
]


class EarlyStopper:
    """Stop when validation ELBO has not improved for ``patience`` epochs."""

    def __init__(self, patience: int):
        if patience < 1:
            raise ValueError(f"patience must be >= 1, got {patience}")
        self.patience = patience
        self.best_elbo = -math.inf
        self.epochs_without_improvement = 0
        self.improved = False  # did the last update() set a new best?

    def update(self, val_elbo: float) -> bool:
        """Record an epoch's validation ELBO; returns True when training should stop."""
        self.improved = val_elbo > self.best_elbo
        if self.improved:
            self.best_elbo = val_elbo
            self.epochs_without_improvement = 0
        else:
            self.epochs_without_improvement += 1
        return self.epochs_without_improvement >= self.patience


def _step_loss(model: ConditionalVAE, x, cat, cont, beta, lambda_fb, generator=None):
    ctx = model.embed_context(cat, cont)
    mu_z, log_sigma_sq = model.encode(x, ctx)
    z = reparameterize(mu_z, log_sigma_sq, generator)
    decoded = model.decode(z, ctx)
    if model.decoder_likelihood == "nb":
        mu, r = decoded
        return nb_vae_loss(mu, r, x, mu_z, log_sigma_sq, beta=beta, lambda_fb=lambda_fb)
    return vae_loss(decoded, x, mu_z, log_sigma_sq, beta=beta, lambda_fb=lambda_fb)


@torch.no_grad()
def evaluate_elbo(
    model: ConditionalVAE,
    data: tuple[torch.Tensor, torch.Tensor, torch.Tensor],
    batch_size: int = 8192,
) -> float:
    """Mean per-window ELBO (raw KL, single fixed-seed noise draw per window)."""
    x, cat, cont = data
    generator = torch.Generator().manual_seed(VAL_NOISE_SEED)
    total, n = 0.0, 0
    for start in range(0, len(x), batch_size):
        stop = min(start + batch_size, len(x))
        out = _step_loss(
            model, x[start:stop], cat[start:stop], cont[start:stop], 1.0, 0.0, generator
        )
        total += out["elbo"].item() * (stop - start)
        n += stop - start
    return total / n


@torch.no_grad()
def per_dimension_kl(
    model: ConditionalVAE,
    data: tuple[torch.Tensor, torch.Tensor, torch.Tensor],
    max_windows: int = 20000,
) -> torch.Tensor:
    """(K,) batch-mean per-dimension KL on (a subsample of) the given split."""
    x, cat, cont = (t[:max_windows] for t in data)
    ctx = model.embed_context(cat, cont)
    mu_z, log_sigma_sq = model.encode(x, ctx)
    return gaussian_kl_per_dim(mu_z, log_sigma_sq).mean(dim=0)


def save_checkpoint(
    path: Path,
    model: ConditionalVAE,
    optimizer: torch.optim.Optimizer,
    epoch: int,
    val_elbo: float,
    config_dict: dict,
    model_kwargs: dict,
) -> None:
    """Checkpoint = weights + optimizer + epoch + val ELBO + config + ctor kwargs.

    ``model_kwargs`` lets :meth:`CVAESampler.from_checkpoint` rebuild the
    architecture without re-deriving it from the data pipeline.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "epoch": epoch,
            "val_elbo": val_elbo,
            "config": config_dict,
            "model_kwargs": model_kwargs,
        },
        path,
    )


def train_cvae(
    model: ConditionalVAE,
    train_data: tuple[torch.Tensor, torch.Tensor, torch.Tensor],
    val_data: tuple[torch.Tensor, torch.Tensor, torch.Tensor],
    *,
    n_epochs: int,
    batch_size: int,
    lr: float,
    n_anneal_epochs: int,
    lambda_fb: float,
    patience: int = 5,
    seed: int = 0,
    checkpoint_dir: str | Path = "checkpoints",
    log_path: str | Path | None = None,
    config_dict: dict | None = None,
    model_kwargs: dict | None = None,
    verbose: bool = True,
) -> list[dict]:
    """Train the CVAE with annealing, free bits, early stopping, checkpoints.

    Batching is a per-epoch random permutation with contiguous tensor slices —
    equivalent to a shuffling DataLoader but several times faster on CPU
    because it avoids per-sample collation. Returns the per-epoch history
    (the rows written to ``log_path``).
    """
    torch.manual_seed(seed)
    checkpoint_dir = Path(checkpoint_dir)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    stopper = EarlyStopper(patience)
    history: list[dict] = []
    x, cat, cont = train_data
    config_dict = config_dict or {}
    model_kwargs = model_kwargs or {}

    if log_path is not None:
        log_path = Path(log_path)
        log_path.parent.mkdir(parents=True, exist_ok=True)
        with open(log_path, "w", newline="") as f:
            csv.writer(f).writerow(LOG_COLUMNS)

    for epoch in range(1, n_epochs + 1):
        t0 = time.time()
        beta = get_beta(epoch, n_anneal_epochs)
        model.train()
        perm = torch.randperm(len(x))
        sums = {"recon": 0.0, "kl": 0.0}
        n_batches = 0
        for start in range(0, len(x), batch_size):
            idx = perm[start : start + batch_size]
            optimizer.zero_grad()
            out = _step_loss(model, x[idx], cat[idx], cont[idx], beta, lambda_fb)
            if not torch.isfinite(out["loss"]):
                raise RuntimeError(f"Non-finite training loss at epoch {epoch}")
            out["loss"].backward()
            optimizer.step()
            sums["recon"] += out["recon"].item()
            sums["kl"] += out["kl"].item()
            n_batches += 1

        model.eval()
        val_elbo = evaluate_elbo(model, val_data)
        kl_per_dim = per_dimension_kl(model, val_data)
        train_recon = sums["recon"] / n_batches
        train_kl = sums["kl"] / n_batches
        row = {
            "epoch": epoch,
            "beta": round(beta, 4),
            "train_recon": round(train_recon, 4),
            "train_kl": round(train_kl, 4),
            "train_elbo": round(-(train_recon + train_kl), 4),
            "val_elbo": round(val_elbo, 4),
            "kl_dim_min": round(float(kl_per_dim.min()), 4),
            "kl_dim_mean": round(float(kl_per_dim.mean()), 4),
            "kl_dim_max": round(float(kl_per_dim.max()), 4),
            "active_dims": int((kl_per_dim > ACTIVE_DIM_THRESHOLD).sum()),
            "seconds": round(time.time() - t0, 1),
        }
        history.append(row)
        if verbose:
            print(
                f"epoch {row['epoch']:>3}/{n_epochs}  beta={row['beta']:.2f}  "
                f"train recon={row['train_recon']:.3f} KL={row['train_kl']:.3f}  "
                f"val ELBO={row['val_elbo']:.4f}  "
                f"dimKL[min/mean/max]={row['kl_dim_min']:.3f}/{row['kl_dim_mean']:.3f}/"
                f"{row['kl_dim_max']:.3f}  active={row['active_dims']}  "
                f"[{row['seconds']:.0f}s]",
                flush=True,
            )
        warning = collapse_warning(kl_per_dim, beta)
        if warning and verbose:
            print(f"WARNING: {warning}", flush=True)
        if log_path is not None:
            with open(log_path, "a", newline="") as f:
                csv.writer(f).writerow([row[c] for c in LOG_COLUMNS])

        should_stop = stopper.update(val_elbo)
        if stopper.improved:
            save_checkpoint(
                checkpoint_dir / "best.pt",
                model,
                optimizer,
                epoch,
                val_elbo,
                config_dict,
                model_kwargs,
            )
        save_checkpoint(
            checkpoint_dir / "last.pt",
            model,
            optimizer,
            epoch,
            val_elbo,
            config_dict,
            model_kwargs,
        )
        if should_stop:
            if verbose:
                print(
                    f"Early stopping at epoch {epoch}: no val-ELBO improvement for "
                    f"{patience} epochs (best {stopper.best_elbo:.4f}).",
                    flush=True,
                )
            break
    return history
