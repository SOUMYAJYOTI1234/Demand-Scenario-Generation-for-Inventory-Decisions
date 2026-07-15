"""Tests for CVAE Milestone 4: the Negative Binomial decoder.

The NB log-likelihood is checked against scipy before any training touches it
(design doc §8: unit-test the NB log-likelihood against a reference
implementation before training).
"""

import numpy as np
import pytest
import torch
from scipy import stats

from demand_vae.models.cvae import ConditionalVAE
from demand_vae.models.distributions import MU_FLOOR, R_FLOOR, nb_log_likelihood
from demand_vae.models.vae import nb_vae_loss, vae_loss

H, LATENT, N_ITEMS, N_STORES, N_CONT = 4, 16, 10, 3, 9
BATCH = 8


def make_model(decoder_likelihood: str = "nb") -> ConditionalVAE:
    torch.manual_seed(0)
    return ConditionalVAE(
        horizon=H,
        latent_dim=LATENT,
        n_items=N_ITEMS,
        n_stores=N_STORES,
        n_continuous=N_CONT,
        encoder_hidden=[128, 64],
        decoder_hidden=[64, 128],
        decoder_likelihood=decoder_likelihood,
    )


def make_batch(batch: int = BATCH):
    gen = torch.Generator().manual_seed(1)
    x = torch.randint(0, 50, (batch, H), generator=gen).float()
    cat = torch.stack(
        [
            torch.randint(0, N_ITEMS, (batch,), generator=gen),
            torch.randint(0, N_STORES, (batch,), generator=gen),
        ],
        dim=1,
    )
    cont = torch.randn(batch, N_CONT, generator=gen)
    return x, cat, cont


class TestNBLogLikelihood:
    def test_matches_scipy_nbinom(self):
        rng = np.random.default_rng(2)
        mu = rng.uniform(0.05, 30.0, size=(64, H))
        r = rng.uniform(0.2, 15.0, size=(64, H))
        x = rng.integers(0, 60, size=(64, H))
        ours = nb_log_likelihood(
            torch.from_numpy(x).double(), torch.from_numpy(mu), torch.from_numpy(r)
        ).numpy()
        reference = stats.nbinom.logpmf(x, n=r, p=r / (r + mu))
        np.testing.assert_allclose(ours, reference, atol=1e-4)

    def test_finite_at_zero_demand(self):
        # Zero weeks are common in M5; the likelihood must not blow up there.
        mu = torch.full((5, H), MU_FLOOR)  # worst case: mean at its floor
        r = torch.full((5, H), R_FLOOR)
        ll = nb_log_likelihood(torch.zeros(5, H), mu, r)
        assert torch.isfinite(ll).all()
        assert (ll <= 0).all()  # log-pmf of a proper distribution

    def test_beats_poisson_on_overdispersed_data(self):
        # Draw from a heavily over-dispersed NB (var = mu + mu^2/r >> mu) and
        # score both families at their method-of-moments fits: NB must win.
        rng = np.random.default_rng(3)
        true_r, true_mu = 2.0, 10.0
        x_np = rng.negative_binomial(true_r, true_r / (true_r + true_mu), size=5000)
        assert x_np.var() > 2 * x_np.mean()  # confirm over-dispersion
        x = torch.from_numpy(x_np).double()
        m, v = x_np.mean(), x_np.var()
        r_mom = m**2 / (v - m)
        nb_total = nb_log_likelihood(
            x, torch.full_like(x, m, dtype=torch.double), torch.full_like(x, r_mom)
        ).sum()
        poisson_total = (
            x * np.log(m) - m - torch.lgamma(x + 1.0)
        ).sum()  # Poisson log-pmf at lam = mean
        assert nb_total.item() > poisson_total.item()


class TestNBDecoderHeads:
    def test_floors_hold_on_random_inputs(self):
        model = make_model("nb")
        gen = torch.Generator().manual_seed(4)
        # Extreme latent/context values push the heads far negative; the
        # softplus + floor must still keep parameters strictly positive.
        z = torch.randn(256, LATENT, generator=gen) * 50
        ctx = torch.randn(256, model.context_encoder.output_dim, generator=gen) * 50
        mu, r = model.decode(z, ctx)
        assert (mu >= MU_FLOOR).all()
        assert (r >= R_FLOOR).all()
        assert torch.isfinite(mu).all() and torch.isfinite(r).all()

    def test_forward_shapes_nb(self):
        model = make_model("nb")
        x, cat, cont = make_batch()
        (mu, r), mu_z, log_sigma_sq = model(x, cat, cont)
        assert mu.shape == (BATCH, H)
        assert r.shape == (BATCH, H)
        assert mu_z.shape == (BATCH, LATENT)
        assert log_sigma_sq.shape == (BATCH, LATENT)

    def test_nb_loss_finite_and_one_step_improves(self):
        model = make_model("nb")
        x, cat, cont = make_batch(64)
        opt = torch.optim.Adam(model.parameters(), lr=1e-2)
        gen = torch.Generator().manual_seed(5)
        (mu, r), mu_z, lsq = model(x, cat, cont, generator=gen)
        out0 = nb_vae_loss(mu, r, x, mu_z, lsq, beta=1.0)
        assert torch.isfinite(out0["loss"])
        out0["loss"].backward()
        opt.step()
        gen2 = torch.Generator().manual_seed(5)
        (mu, r), mu_z, lsq = model(x, cat, cont, generator=gen2)
        out1 = nb_vae_loss(mu, r, x, mu_z, lsq, beta=1.0)
        assert out1["loss"].item() < out0["loss"].item()


class TestDecoderSwitch:
    def test_gaussian_mode_still_runs(self):
        model = make_model("gaussian")
        x, cat, cont = make_batch()
        recon, mu_z, log_sigma_sq = model(x, cat, cont)
        assert recon.shape == (BATCH, H)  # single tensor, not a tuple
        out = vae_loss(recon, x, mu_z, log_sigma_sq)
        assert torch.isfinite(out["loss"])

    def test_unknown_likelihood_rejected(self):
        with pytest.raises(ValueError, match="decoder_likelihood"):
            make_model("poisson")
