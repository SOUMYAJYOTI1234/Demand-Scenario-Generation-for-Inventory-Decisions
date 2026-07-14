"""Tests for CVAE Milestone 1: the plain deterministic autoencoder."""

import torch

from demand_vae.models.autoencoder import ContextEncoder, DemandAutoencoder, reconstruction_loss

H, LATENT, N_ITEMS, N_STORES, N_CONT = 4, 16, 10, 3, 9
BATCH = 8


def make_batch(batch: int = BATCH) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    gen = torch.Generator().manual_seed(0)
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


def make_model() -> DemandAutoencoder:
    torch.manual_seed(0)
    return DemandAutoencoder(
        horizon=H,
        latent_dim=LATENT,
        n_items=N_ITEMS,
        n_stores=N_STORES,
        n_continuous=N_CONT,
        encoder_hidden=[128, 64],
        decoder_hidden=[64, 128],
    )


class TestShapes:
    def test_forward_reconstructs_window_shape(self):
        model = make_model()
        x, cat, cont = make_batch()
        recon = model(x, cat, cont)
        assert recon.shape == (BATCH, H)

    def test_encoder_produces_bottleneck_shape(self):
        model = make_model()
        x, cat, cont = make_batch()
        z = model.encode(x, cat, cont)
        assert z.shape == (BATCH, LATENT)

    def test_context_encoder_output_dim(self):
        ctx = ContextEncoder(
            N_ITEMS, N_STORES, N_CONT, item_embedding_dim=32, store_embedding_dim=4
        )
        _, cat, cont = make_batch()
        out = ctx(cat, cont)
        assert out.shape == (BATCH, ctx.output_dim) == (BATCH, 32 + 4 + N_CONT)


class TestLoss:
    def test_zero_when_reconstruction_equals_target(self):
        x, _, _ = make_batch()
        perfect = torch.log1p(x)  # identity case: reconstruction == log1p target
        assert reconstruction_loss(perfect, x).item() == 0.0

    def test_positive_otherwise(self):
        x, cat, cont = make_batch()
        recon = make_model()(x, cat, cont)
        assert reconstruction_loss(recon, x).item() > 0.0

    def test_one_gradient_step_reduces_loss(self):
        model = make_model()
        x, cat, cont = make_batch()
        opt = torch.optim.Adam(model.parameters(), lr=1e-2)
        loss0 = reconstruction_loss(model(x, cat, cont), x)
        loss0.backward()
        opt.step()
        loss1 = reconstruction_loss(model(x, cat, cont), x)
        assert loss1.item() < loss0.item()
