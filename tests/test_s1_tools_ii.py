"""scripts/s1_tools.py a1post instrument (ii): check_ii on synthetic state dicts (no model, no GPU) -- arm U (default, unchanged)
and arm VT (--expect-trainable vision+text), with mutants."""

from __future__ import annotations

import importlib.util
from pathlib import Path

import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("safetensors")

SPEC = importlib.util.spec_from_file_location("s1_tools", Path(__file__).resolve().parents[1] / "scripts" / "s1_tools.py")
S = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(S)

V = "model.vlm_with_expert.vlm.model.vision_model.encoder.layers.{}.w"
TX = "model.vlm_with_expert.vlm.model.text_model.{}"
LAYERS = 3  # text layers 0..2; LeRobot freezes the last (2), text_model.norm.weight and lm_head
CONN = "model.vlm_with_expert.vlm.model.connector.modality_projection.proj.weight"
HEAD = "model.vlm_with_expert.vlm.lm_head.weight"
EXPERT = "model.vlm_with_expert.lm_expert.layers.0.w"


def init_sd() -> dict:
    g = torch.Generator().manual_seed(0)
    r = lambda dt: torch.randn(4, generator=g).to(dt)  # noqa: E731
    sd = {V.format(i): r(torch.float32) for i in range(10)}  # a vision tower trained in fp32 (a Stage 1 checkpoint)
    sd |= {TX.format("embed_tokens.weight"): r(torch.bfloat16), TX.format("norm.weight"): r(torch.bfloat16)}
    for i in range(LAYERS):
        sd |= {TX.format(f"layers.{i}.self_attn.q_proj.weight"): r(torch.bfloat16),
               TX.format(f"layers.{i}.mlp.up_proj.weight"): r(torch.bfloat16)}
    return sd | {CONN: r(torch.bfloat16), HEAD: r(torch.bfloat16), EXPERT: r(torch.float32)}


def final(base: dict, move: set[str]) -> dict:
    """The run's final checkpoint: `base` with the tensors whose key contains one of `move` perturbed (fp32 masters)."""
    return {k: (v.float() + 1e-3 if any(m in k for m in move) else v.clone()) for k, v in base.items()}


VISION, TEXT_TRAIN = ".vision_model.", (".text_model.embed_tokens", ".text_model.layers.0.", ".text_model.layers.1.")
CONNECTOR, FROZEN_TEXT, LM_HEAD = ".connector.", (".text_model.norm.weight", ".text_model.layers.2."), ".lm_head."


def test_vt_passes_when_exactly_lerobots_trainable_set_moved():
    base = init_sd()
    r = S.check_ii(base, final(base, {VISION, CONNECTOR, EXPERT, *TEXT_TRAIN}), "vision+text")
    assert r["pass"] and r["text_last_layer_frozen"] == 2
    p = r["parts"]
    assert p["text_trainable"] == {"tensors": 5, "moved": 5, "bit_identical": 0}  # embed + layers 0, 1 (q, up)
    assert p["text_frozen_by_lerobot"] == {"tensors": 3, "moved": 0, "bit_identical": 3}  # norm + layer 2 (q, up)
    assert p["lm_head"]["bit_identical"] == 1 and r["connector_all_moved"] and r["vision_moved_frac"] == 1.0


def test_mutant_u_arm_checked_as_vt_fails():
    """A U-arm run (text model bit-identical) must FAIL --expect-trainable vision+text, and still pass the default."""
    base = init_sd()
    u = final(base, {VISION, EXPERT})
    vt = S.check_ii(base, u, "vision+text")
    assert not vt["pass"] and vt["text_trainable_moved_frac"] == 0.0 and not vt["connector_all_moved"]
    assert vt["frozen_parts_bit_identical"]  # the failure is the missing text / connector movement
    d = S.check_ii(base, u)
    assert d["pass"] and d["frozen_parts_bit_identical"] and d["vision_moved_frac"] == 1.0
    assert set(d) == {"parts", "frozen_parts_bit_identical", "vision_moved_frac", "pass"}  # default output unchanged
    assert not S.check_ii(base, final(base, {VISION, CONNECTOR, EXPERT, *TEXT_TRAIN}), "vision")["pass"]  # VT is not U


@pytest.mark.parametrize("what,key", [("lm_head", LM_HEAD), ("norm", ".text_model.norm.weight"),
                                      ("last_layer", ".text_model.layers.2.self_attn.q_proj")])
def test_vt_fails_when_a_lerobot_frozen_tensor_changed(what, key):
    base = init_sd()
    r = S.check_ii(base, final(base, {VISION, CONNECTOR, EXPERT, *TEXT_TRAIN, key}), "vision+text")
    assert not r["frozen_parts_bit_identical"] and not r["pass"], what


@pytest.mark.parametrize("missing", ["vision_half", "text_half", "connector"])
def test_vt_thresholds(missing):
    base = init_sd()
    move = {VISION, CONNECTOR, EXPERT, *TEXT_TRAIN}
    sd = final(base, move)
    if missing == "vision_half":  # 4 of 10 vision tensors reset to the init: 40 % < 90 %
        for i in range(4):
            sd[V.format(i)] = base[V.format(i)].clone()
    elif missing == "text_half":  # 3 of 5 trainable text tensors reset
        for k in [k for k in base if any(t in k for t in TEXT_TRAIN)][:3]:
            sd[k] = base[k].clone()
    else:
        sd[CONN] = base[CONN].clone()
    assert not S.check_ii(base, sd, "vision+text")["pass"]
    sd = final(base, move)  # 1 of 10 vision tensors unmoved = 90 % -> still passes (>= 90 %)
    sd[V.format(0)] = base[V.format(0)].clone()
    assert S.check_ii(base, sd, "vision+text")["pass"]


def test_load_ref_from_checkpoint_dir(tmp_path):
    from safetensors.torch import save_file
    base = init_sd()
    (tmp_path / "ck" / "pretrained_model").mkdir(parents=True)
    save_file(base, str(tmp_path / "ck" / "pretrained_model" / "model.safetensors"))
    for d in (tmp_path / "ck", tmp_path / "ck" / "pretrained_model"):
        got = S.load_ref(str(d))
        assert got.keys() == base.keys() and all(torch.equal(got[k], base[k]) for k in base)
