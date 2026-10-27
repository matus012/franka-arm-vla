"""3.3 attempt-1 bug: normalization stats inherited from fixed-target stages had std 1e-3 on the target dims."""

import torch

from vision_lift.student import StudentPolicy
from vision_lift.train import refresh_degenerate_norm


def test_only_degenerate_dims_are_refreshed():
    m = StudentPolicy()
    mean, std = torch.zeros(27), torch.full((27,), 0.5)
    std[19:22] = 1e-3  # target constant in the parent's data
    mean[19:22] = torch.tensor([0.5, 0.15, 0.35])
    m.set_norm(mean, std)
    new_mean, new_std = torch.full((27,), 0.1), torch.full((27,), 0.3)
    new_std[19:22] = torch.tensor([0.06, 0.15, 0.07])
    dims = refresh_degenerate_norm(m, new_mean, new_std)
    assert dims == [19, 20, 21]
    assert torch.allclose(m.p_std[19:22], new_std[19:22]) and torch.allclose(m.p_mean[19:22], new_mean[19:22])
    assert torch.allclose(m.p_std[:19], std[:19]) and torch.allclose(m.p_mean[:19], mean[:19])  # others untouched


def test_constant_in_both_stays():
    m = StudentPolicy()
    m.set_norm(torch.zeros(27), torch.full((27,), 1e-3))
    assert refresh_degenerate_norm(m, torch.zeros(27), torch.full((27,), 1e-3)) == []
