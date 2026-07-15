"""Tests for CVAE Milestone 3: conditioning both encoder and decoder.

The synthetic data here makes the context THE signal (item 0 sells ~0,
item 1 sells ~30) so the zeroed-context comparison is a sharp test: a model
that ignores y cannot tell the two regimes apart.
"""

import torch

from demand_vae.models.cvae import ConditionalVAE
from demand_vae.models.vae import reparameterize, vae_loss

H, LATENT, N_ITEMS, N_STORES, N_CONT = 4, 16, 10, 3, 9
BATCH = 32


def make_model() -> ConditionalVAE:
    torch.manual_seed(0)
    # Milestone-3 tests exercise the conditioning wiring under the Gaussian
    # decoder; the NB decoder has its own suite (test_nb_decoder.py).
    return ConditionalVAE(
        horizon=H,
        latent_dim=LATENT,
        n_items=N_ITEMS,
        n_stores=N_STORES,
        n_continuous=N_CONT,
        encoder_hidden=[128, 64],
        decoder_hidden=[64, 128],
        decoder_likelihood="gaussian",
    )


def make_batch(batch: int = BATCH):
    """Context-determined demand: item 0 -> ~0 units/week, item 1 -> ~30."""
    gen = torch.Generator().manual_seed(1)
    items = torch.arange(batch) % 2
    x = torch.where(
        items[:, None].bool(),
        30.0 + torch.randint(0, 5, (batch, H), generator=gen).float(),
        torch.randint(0, 2, (batch, H), generator=gen).float(),
    )
    cat = torch.stack([items, torch.zeros(batch, dtype=torch.long)], dim=1)
    cont = torch.randn(batch, N_CONT, generator=gen) * 0.1
    return x, cat, cont


def loss_with_context(model, x, cat, cont, gen, zero_context=False):
    ctx = model.embed_context(cat, cont)
    if zero_context:
        ctx = torch.zeros_like(ctx)
    mu, log_sigma_sq = model.encode(x, ctx)
    z = reparameterize(mu, log_sigma_sq, gen)
    recon = model.decode(z, ctx)
    return vae_loss(recon, x, mu, log_sigma_sq)


class TestConditioningWiring:
    def test_forward_shapes_with_context(self):
        model = make_model()
        x, cat, cont = make_batch()
        recon, mu, log_sigma_sq = model(x, cat, cont)
        assert recon.shape == (BATCH, H)
        assert mu.shape == (BATCH, LATENT)
        assert log_sigma_sq.shape == (BATCH, LATENT)

    def test_encoder_sees_context(self):
        # Same window, different item index -> different posterior mean.
        model = make_model()
        x, cat, cont = make_batch(2)
        x[1] = x[0]  # identical windows
        cont[1] = cont[0]
        cat = torch.tensor([[0, 0], [1, 0]])  # different items
        ctx = model.embed_context(cat, cont)
        mu, _ = model.encode(x, ctx)
        assert not torch.allclose(mu[0], mu[1])

    def test_decoder_sees_context(self):
        # Same z, different context -> different reconstruction.
        model = make_model()
        _, cat, cont = make_batch(2)
        cont[1] = cont[0]
        cat = torch.tensor([[0, 0], [1, 0]])
        ctx = model.embed_context(cat, cont)
        z = torch.zeros(2, LATENT)
        recon = model.decode(z, ctx)
        assert not torch.allclose(recon[0], recon[1])


class TestConditionalTraining:
    def test_elbo_finite_and_one_step_improves(self):
        model = make_model()
        x, cat, cont = make_batch()
        opt = torch.optim.Adam(model.parameters(), lr=1e-2)
        out0 = loss_with_context(model, x, cat, cont, torch.Generator().manual_seed(2))
        assert torch.isfinite(out0["elbo"])
        out0["loss"].backward()
        opt.step()
        out1 = loss_with_context(model, x, cat, cont, torch.Generator().manual_seed(2))
        assert out1["loss"].item() < out0["loss"].item()

    def test_zeroed_context_is_worse_after_training(self):
        # Train 10 steps with real context on context-determined data, then
        # score the same batch with real vs zeroed context (same noise):
        # a conditioned model must lose accuracy when y is removed.
        model = make_model()
        x, cat, cont = make_batch(64)
        opt = torch.optim.Adam(model.parameters(), lr=1e-2)
        for step in range(10):
            opt.zero_grad()
            out = loss_with_context(model, x, cat, cont, torch.Generator().manual_seed(step))
            out["loss"].backward()
            opt.step()
        gen_real = torch.Generator().manual_seed(99)
        gen_zero = torch.Generator().manual_seed(99)
        with torch.no_grad():
            real = loss_with_context(model, x, cat, cont, gen_real)
            zeroed = loss_with_context(model, x, cat, cont, gen_zero, zero_context=True)
        assert zeroed["elbo"].item() < real["elbo"].item()  # zeroed ELBO is worse (lower)
