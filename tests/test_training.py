"""Tests for Phase 7: training engine, early stopping, checkpoints, CVAESampler."""

import numpy as np
import pytest
import torch

from demand_vae.models.cvae import ConditionalVAE
from demand_vae.models.sampler import CVAESampler
from demand_vae.training import EarlyStopper, train_cvae

H, LATENT, N_ITEMS, N_STORES, N_CONT = 4, 8, 6, 3, 9

MODEL_KWARGS = {
    "horizon": H,
    "latent_dim": LATENT,
    "n_items": N_ITEMS,
    "n_stores": N_STORES,
    "n_continuous": N_CONT,
    "encoder_hidden": [32, 16],
    "decoder_hidden": [16, 32],
    "item_embedding_dim": 4,
    "store_embedding_dim": 2,
    "decoder_likelihood": "nb",
}


def make_model() -> ConditionalVAE:
    torch.manual_seed(0)
    return ConditionalVAE(**MODEL_KWARGS)


def make_data(n: int):
    gen = torch.Generator().manual_seed(1)
    x = torch.randint(0, 30, (n, H), generator=gen).float()
    cat = torch.stack(
        [
            torch.randint(0, N_ITEMS, (n,), generator=gen),
            torch.randint(0, N_STORES, (n,), generator=gen),
        ],
        dim=1,
    )
    cont = torch.randn(n, N_CONT, generator=gen)
    return x, cat, cont


class TestEarlyStopper:
    def test_stops_after_patience_non_improvements(self):
        stopper = EarlyStopper(patience=5)
        assert stopper.update(-10.0) is False  # first value is an improvement
        stops = [stopper.update(-11.0) for _ in range(5)]  # 5 worse epochs
        assert stops == [False, False, False, False, True]

    def test_improvement_resets_the_counter(self):
        stopper = EarlyStopper(patience=2)
        stopper.update(-10.0)
        assert stopper.update(-12.0) is False  # 1 bad
        assert stopper.update(-9.0) is False  # improvement -> reset, new best
        assert stopper.improved
        assert stopper.update(-9.5) is False  # 1 bad
        assert stopper.update(-9.5) is True  # 2 bad -> stop
        assert stopper.best_elbo == -9.0


class TestTrainingLoop:
    def test_two_epoch_smoke_run(self, tmp_path):
        model = make_model()
        history = train_cvae(
            model,
            make_data(500),
            make_data(100),
            n_epochs=2,
            batch_size=128,
            lr=1e-3,
            n_anneal_epochs=10,
            lambda_fb=0.5,
            patience=5,
            seed=0,
            checkpoint_dir=tmp_path / "ckpt",
            log_path=tmp_path / "log.csv",
            verbose=False,
        )
        assert len(history) == 2
        assert all(np.isfinite(row["val_elbo"]) for row in history)
        assert history[0]["beta"] == pytest.approx(0.1)
        assert history[1]["beta"] == pytest.approx(0.2)
        assert (tmp_path / "ckpt" / "best.pt").exists()
        assert (tmp_path / "ckpt" / "last.pt").exists()
        log_lines = (tmp_path / "log.csv").read_text().strip().splitlines()
        assert len(log_lines) == 3  # header + 2 epochs

    def test_checkpoint_roundtrip_identical_outputs(self, tmp_path):
        model = make_model()
        train_cvae(
            model,
            make_data(200),
            make_data(50),
            n_epochs=1,
            batch_size=64,
            lr=1e-3,
            n_anneal_epochs=10,
            lambda_fb=0.5,
            patience=5,
            seed=0,
            checkpoint_dir=tmp_path,
            model_kwargs=MODEL_KWARGS,
            verbose=False,
        )
        checkpoint = torch.load(tmp_path / "best.pt", map_location="cpu", weights_only=False)
        assert {
            "model_state_dict",
            "optimizer_state_dict",
            "epoch",
            "val_elbo",
            "config",
            "model_kwargs",
        } <= set(checkpoint)
        reloaded = ConditionalVAE(**checkpoint["model_kwargs"])
        reloaded.load_state_dict(checkpoint["model_state_dict"])
        reloaded.eval()
        model.eval()
        x, cat, cont = make_data(16)
        with torch.no_grad():
            ctx_a = model.embed_context(cat, cont)
            ctx_b = reloaded.embed_context(cat, cont)
            torch.testing.assert_close(ctx_a, ctx_b)
            mu_a, ls_a = model.encode(x, ctx_a)
            mu_b, ls_b = reloaded.encode(x, ctx_b)
        torch.testing.assert_close(mu_a, mu_b)
        torch.testing.assert_close(ls_a, ls_b)


class TestCVAESampler:
    def test_sample_shape(self):
        sampler = CVAESampler(make_model(), seed=0)
        context = {
            "context_cat": np.array([1, 2]),
            "context_cont": np.zeros(N_CONT, dtype=np.float32),
        }
        draws = sampler.sample(context, 64)
        assert draws.shape == (64, H)

    def test_samples_are_nonnegative_integers(self):
        sampler = CVAESampler(make_model(), seed=0)
        batch = {
            "context_cat": np.array([[0, 0], [3, 1], [5, 2]]),
            "context_cont": np.random.default_rng(2).normal(size=(3, N_CONT)).astype(np.float32),
        }
        draws = sampler.sample_batch(batch, 200)
        assert draws.shape == (3, 200, H)
        assert (draws >= 0).all()
        np.testing.assert_array_equal(draws, np.rint(draws))  # integers only

    def test_seed_reproducibility(self):
        context = {
            "context_cat": np.array([1, 1]),
            "context_cont": np.zeros(N_CONT, dtype=np.float32),
        }
        a = CVAESampler(make_model(), seed=7).sample(context, 50)
        b = CVAESampler(make_model(), seed=7).sample(context, 50)
        np.testing.assert_array_equal(a, b)
