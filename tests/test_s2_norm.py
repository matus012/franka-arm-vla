"""scripts/s2_norm.py: the pure analysis function + stats reuse (no dataset, no GPU)."""
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))
import s2_norm  # noqa: E402
from vla_policy.train_smolvla import norm_stats  # noqa: E402


class _Hf:
    def __init__(self, cols): self.cols = cols
    def with_format(self, _): return self.cols


class _Ds:
    def __init__(self, cols): self.hf_dataset, self.n = _Hf(cols), len(cols["action"])
    def __len__(self): return self.n


def test_analyze_ratio_constant_and_beyond_flags():
    rng = np.random.default_rng(0)
    x = np.stack([rng.normal(1, 1, 2000), rng.normal(0, 4, 2000), np.full(2000, 3.0), rng.normal(0, 1, 2000)], 1)
    x[:5, 3] = 50.0  # 0.25 % of rows far outside
    ds = _Ds({"observation.state": x, "action": x})
    st, _ = norm_stats(ds)
    s2 = {s: st["action"][s].astype(np.float64) for s in ("mean", "std")}
    parent = {"mean": np.array([1.0, 0, 3, 0]), "std": np.array([1.0, 1.0, 1.0, 1.0])}  # dim 1 ratio ~4
    r = s2_norm.analyze(x, parent, s2)
    assert [d["ratio_outside_band"] for d in r] == [False, True, False, True]  # dim 2 floored std 1 -> ratio 1; dim 3 std inflated by the outliers
    assert r[2]["near_constant_s2"] and not r[0]["near_constant_s2"]
    assert r[3]["frac_beyond_10"] == 5 / 2000 and r[3]["instrument_finding"]
    assert not r[0]["instrument_finding"] and not r[1]["instrument_finding"]
    assert abs(r[1]["std_ratio"] - 4.0) < 0.3
    assert r[0]["norm_min"] < 0 < r[0]["norm_max"]


def test_near_constant_parent_flag():
    x = np.random.default_rng(1).normal(size=(100, 1))
    r = s2_norm.analyze(x, {"mean": np.zeros(1), "std": np.array([1e-9])}, {"mean": np.zeros(1), "std": np.ones(1)})
    assert r[0]["near_constant_parent"] and r[0]["ratio_outside_band"]
