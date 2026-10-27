import numpy as np
import pytest
import torch

from vla_policy.baseline import LABEL_SIZES, BaselineConfig, BaselinePolicy, loss_fn

N = 4


def _batch(B=6, H=64, seed=0):
    g = torch.Generator().manual_seed(seed)
    return dict(
        side=torch.randint(0, 256, (B, H, H, 3), dtype=torch.uint8, generator=g),
        wrist=torch.randint(0, 256, (B, H, H, 3), dtype=torch.uint8, generator=g),
        state=torch.randn(B, 19, generator=g),
        labels=torch.stack([torch.randint(0, k, (B,), generator=g) for k in LABEL_SIZES], 1),
    )


def _model():
    torch.manual_seed(0)
    return BaselinePolicy(BaselineConfig(n_action_steps=N))


def _np(b):
    return b["side"].numpy(), b["wrist"].numpy(), b["state"].numpy(), b["labels"].numpy()


def test_param_count():
    n = sum(p.numel() for p in BaselinePolicy().parameters())
    assert 0.5e6 < n < 1.5e6, n


@pytest.mark.parametrize("H", [64, 128, 224])
def test_shapes_any_size(H):
    out = _model().predict(*_np(_batch(3, H)))
    assert out.shape == (3, N, 5) and out.dtype == np.float32


def test_label_embedding_ranges():
    m = _model()
    assert [e.num_embeddings for e in m.emb] == [3, 4, 2, 13]
    b = _batch(2)
    b["labels"][:, 3] = 13
    with pytest.raises((IndexError, RuntimeError)):
        m(b["side"], b["wrist"], b["state"], b["labels"])


def test_labels_matter():
    m = _model().eval()
    b = _batch(1)
    l2 = b["labels"].clone()
    l2[0, 0] = (l2[0, 0] + 1) % 3
    with torch.no_grad():
        a1 = m(b["side"], b["wrist"], b["state"], b["labels"])[0]
        a2 = m(b["side"], b["wrist"], b["state"], l2)[0]
    assert not torch.allclose(a1, a2)


def test_claw_binary_and_arm_clipped():
    m = _model()
    m.set_norm(torch.zeros(19), torch.ones(19), torch.zeros(4), torch.full((4,), 100.0))
    out = m.predict(*_np(_batch(5)))
    assert set(np.unique(out[..., 4])) <= {0.0, 1.0}
    assert np.abs(out[..., :4]).max() <= 1.0


def test_save_load_roundtrip(tmp_path):
    m = _model()
    m.set_norm(torch.randn(19), torch.rand(19) + 0.5, torch.randn(4), torch.rand(4) + 0.5)
    args = _np(_batch(4))
    m.save(tmp_path / "ck")
    assert (tmp_path / "ck" / "config.json").exists()
    m2 = BaselinePolicy.load(tmp_path / "ck")
    np.testing.assert_array_equal(m.predict(*args), m2.predict(*args))


def test_tiny_overfit():
    m = _model()
    b = _batch(8, 32, seed=1)
    g = torch.Generator().manual_seed(2)
    arm = torch.rand(8, N, 4, generator=g) * 2 - 1
    claw = (torch.rand(8, N, 1, generator=g) > 0.5).float()
    b["action"] = torch.cat([arm, claw], -1)
    b["pad"] = torch.zeros(8, N, dtype=torch.bool)
    opt = torch.optim.Adam(m.parameters(), lr=1e-3)
    for _ in range(300):
        loss, _ = loss_fn(m, b)
        opt.zero_grad()
        loss.backward()
        opt.step()
    assert loss.item() < 0.05, loss.item()
    out = m.predict(*_np(b))
    assert (out[..., 4] == claw[..., 0].numpy()).all()


def test_pad_mask():
    m = _model().eval()
    b = _batch(2, 32)
    b["action"] = torch.zeros(2, N, 5)
    b["pad"] = torch.zeros(2, N, dtype=torch.bool)
    b["pad"][:, 2:] = True
    l0, _ = loss_fn(m, b)
    b["action"][:, 2:] = 1e3
    l1, _ = loss_fn(m, b)
    assert torch.equal(l0, l1)
