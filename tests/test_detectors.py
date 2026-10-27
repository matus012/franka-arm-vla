"""Detector unit tests (GATES.md conventions), crafted states, no simulator. Each detector has a positive and a negative
case; the GATES mutants m1-m5 are killed by the tests marked in their docstrings (scripts/detector_mutants.py)."""

from __future__ import annotations

import math

import torch

from lift_rl.metrics import Thresholds
from vla_tabletop.detectors import (PICK, PLACE, STACK, Consts113, Detectors, Geometry, StepState, in_pad, on_top,
                                    tilt_deg, yaw_err_mod90_deg)

E = torch.tensor([0.048, 0.060] * 4)  # ALL_CUBES order: red s/l, green s/l, blue s/l, yellow s/l
TZ = -0.003
GEO = Geometry(edge=E, z_rest=TZ + E / 2, h_pad=0.002, l_pad=0.10)
RS, RL, GS, GL, BS, BL = 0, 1, 2, 3, 4, 5
K = Consts113()


def zq(yaw: float = 0.0) -> torch.Tensor:
    return torch.tensor([0.0, 0.0, math.sin(yaw / 2), math.cos(yaw / 2)])


def base() -> dict:
    """One env: every cube parked far away, pads on the table, robot high and open, hand vertical."""
    cp = torch.zeros(1, 8, 3)
    cp[0, :, 0], cp[0, :, 1], cp[0, :, 2] = 15.0, torch.arange(8) * 0.2, -1.05 + E / 2
    pads = torch.tensor([[[0.40, -0.20, TZ + 0.001], [0.40, 0.20, TZ + 0.001], [0.60, -0.20, TZ + 0.001],
                          [0.60, 0.20, TZ + 0.001]]])
    R = torch.tensor([[1.0, 0, 0], [0, -1.0, 0], [0, 0, -1.0]])[None]
    z8 = torch.zeros(1, 8)
    return dict(cube_pos=cp, cube_quat=zq().repeat(1, 8, 1), cube_vel=torch.zeros(1, 8, 3), pad_pos=pads,
                pad_quat=zq().repeat(1, 4, 1), f_l=z8.clone(), f_r=z8.clone(), f_p=z8.clone(), f_a=z8.clone(),
                f_ht=torch.zeros(1), f_ct=z8.clone(), f_robot_table=torch.zeros(1), tcp=torch.tensor([[0.5, 0.0, 0.4]]),
                hand_rot=R, claw=torch.zeros(1))


def put(d: dict, c: int, x: float, y: float, z: float | None = None, yaw: float = 0.0) -> dict:
    d["cube_pos"][0, c] = torch.tensor([x, y, GEO.z_rest[c] if z is None else z])
    d["cube_quat"][0, c] = zq(yaw)
    return d


def hold(d: dict, c: int, f: float = 6.0) -> dict:
    d["f_l"][0, c], d["f_r"][0, c] = f, f
    return d


def st(d: dict) -> StepState:
    return StepState(**{k: v.clone() for k, v in d.items()})


def run(seq: list[dict], task: int, target: int, dest: int = 0, on_table: list[int] | None = None,
        grasp_dz: float = 0.0) -> dict:
    ot = torch.zeros(1, 8, dtype=torch.bool)
    for c in on_table if on_table is not None else [target]:
        ot[0, c] = True
    det = Detectors(torch.tensor([task]), torch.tensor([target]), torch.tensor([dest]), ot, GEO,
                    grasp_dz=torch.full((8,), grasp_dz))
    for d in seq:
        det.update(st(d))
    return {k: bool(v[0]) if v.dtype == torch.bool else v[0].item() for k, v in det.outcome().items()}


# --------------------------------------------------------------------------- geometry helpers
def test_tilt_and_yaw_helpers():
    assert tilt_deg(zq(1.0)[None]).item() < 1e-3
    q = torch.tensor([math.sin(math.radians(15) / 2), 0, 0, math.cos(math.radians(15) / 2)])
    assert abs(tilt_deg(q[None]).item() - 15.0) < 1e-3
    q90 = torch.tensor([math.sin(math.pi / 4), 0, 0, math.cos(math.pi / 4)])  # lying on a face: still flat
    assert tilt_deg(q90[None]).item() < 1e-3
    R = torch.tensor([[[1.0, 0, 0], [0, -1.0, 0], [0, 0, -1.0]]])
    assert yaw_err_mod90_deg(R, zq(math.radians(90))[None]).item() < 1e-3
    assert abs(yaw_err_mod90_deg(R, zq(math.radians(30))[None]).item() - 30.0) < 1e-3


def test_in_pad_uses_pad_frame():
    """m5: a pad rotated 45 deg; points inside / outside its square only in the pad frame."""
    pad = torch.tensor([[[0.5, 0.0, 0.0]]])
    pq = zq(math.pi / 4)[None, None]
    inside = torch.tensor([[[0.56, 0.0, 0.03]]])  # pad frame (0.042, -0.042): inside
    outside = torch.tensor([[[0.545, 0.045, 0.03]]])  # pad frame (0.064, 0.0): outside
    assert bool(in_pad(inside, pad, pq, 0.10)[0, 0, 0])
    assert not bool(in_pad(outside, pad, pq, 0.10)[0, 0, 0])


def test_on_top_body_frame_and_order():
    p = torch.zeros(1, 8, 3)
    p[0, :, 0] = torch.arange(8) * 1.0
    p[0, RS] = torch.tensor([0.5, 0.0, 0.1])
    p[0, BL] = torch.tensor([0.5 + 0.035, 0.0, 0.03])  # 3.5 cm off in world x
    q = zq().repeat(1, 8, 1)
    assert not bool(on_top(p, q, E)[0, RS, BL])  # |x| 3.5 cm > e_b/2 = 3 cm
    q[0, BL] = zq(math.pi / 4)  # rotated bottom: body-frame |x|, |y| = 2.5 cm
    assert bool(on_top(p, q, E)[0, RS, BL])
    assert not bool(on_top(p, q, E)[0, BL, RS])  # z order
    assert not bool(on_top(p, q, E)[0, RS, RS])


# --------------------------------------------------------------------------- pick
def _lifted(c: int, dz: float, n: int = 60) -> list[dict]:
    seq = []
    for _ in range(n):
        d = put(base(), c, 0.5, 0.0, GEO.z_rest[c].item() + dz)
        seq.append(hold(d, c))
    return seq


def test_pick_success_and_negative():
    assert run(_lifted(RS, 0.12), PICK, RS)["success"]
    assert run(_lifted(RS, 0.12), PICK, RS)["pick_hold_last50"]
    assert not run(_lifted(RS, 0.08), PICK, RS)["success"]  # too low
    seq = _lifted(RS, 0.12)
    seq[-1]["f_r"][0, RS] = 0.0  # only one finger at the end: not held
    assert not run(seq, PICK, RS)["success"]


def test_pick_uses_per_size_rest_height():
    """m1: a large cube 9.7 cm above its own rest height is not 10 cm up (the small rest height would say 10.3)."""
    assert not run(_lifted(RL, 0.097), PICK, RL)["success"]
    assert run(_lifted(RL, 0.101), PICK, RL)["success"]


def test_pick_hold_last50_negative():
    seq = _lifted(RS, 0.12, 60)
    seq[5] = put(base(), RS, 0.5, 0.0)  # was down at step 5 -> last 54 still fine
    assert run(seq, PICK, RS)["pick_hold_last50"]
    seq[30] = put(base(), RS, 0.5, 0.0)  # down at step 30 -> only 29 good steps at the end
    r = run(seq, PICK, RS)
    assert r["success"] and not r["pick_hold_last50"]


# --------------------------------------------------------------------------- place
def _placed(c: int, pad: int, n: int, x_off: float = 0.0, y_off: float = 0.0, tcp_up: float = 0.2, z_extra: float = 0.0):
    seq = []
    for _ in range(n):
        d = base()
        px, py = d["pad_pos"][0, pad, :2].tolist()
        put(d, c, px + x_off, py + y_off, GEO.z_rest[c].item() + GEO.h_pad + z_extra)
        d["tcp"][0] = torch.tensor([px, py, tcp_up])
        seq.append(d)
    return seq


def test_place_success_and_negatives():
    assert run(_placed(RS, 2, 12), PLACE, RS, 2)["success"]
    assert not run(_placed(RS, 2, 12), PLACE, RS, 1)["success"]  # wrong pad
    assert run(_placed(RS, 2, 12), PLACE, RS, 1)["wrong_dest"]
    assert not run(_placed(RS, 2, 12), PLACE, RS, 2)["wrong_dest"]
    assert not run(_placed(RS, 2, 8), PLACE, RS, 2)["success"]  # only 8 < 10 steps
    assert not run(_placed(RS, 2, 12, tcp_up=0.06), PLACE, RS, 2)["success"]  # gripper not clear
    assert not run(_placed(RS, 2, 12, x_off=0.06), PLACE, RS, 2)["success"]  # off the pad
    assert not run(_placed(RS, 2, 12, z_extra=0.008), PLACE, RS, 2)["success"]  # not resting (8 mm up)


def test_place_requires_both_fingers_released():
    """m2: the right finger still touching the cube -> not released."""
    seq = _placed(RS, 2, 12)
    for d in seq:
        d["f_r"][0, RS] = 3.0
    assert not run(seq, PLACE, RS, 2)["success"]
    seq = _placed(RS, 2, 12)
    for d in seq:
        d["f_a"][0, RS] = 3.0
    assert not run(seq, PLACE, RS, 2)["success"]


def test_place_large_cube_rest_height():
    """m1: a large cube resting on the pad at its own rest height succeeds."""
    assert run(_placed(RL, 0, 12), PLACE, RL, 0)["success"]


def test_place_on_rotated_pad():
    """m5: the cube is inside a 45-deg pad only in the pad frame."""
    seq = _placed(RS, 2, 12, x_off=0.06)
    for d in seq:
        d["pad_quat"][0, 2] = zq(math.pi / 4)
    assert run(seq, PLACE, RS, 2)["success"]
    seq = _placed(RS, 2, 12, x_off=0.045, y_off=0.045)
    for d in seq:
        d["pad_quat"][0, 2] = zq(math.pi / 4)
    assert not run(seq, PLACE, RS, 2)["success"]


def test_place_tilted_or_moving_fails():
    seq = _placed(RS, 2, 12)
    for d in seq:
        d["cube_vel"][0, RS, 0] = 0.05
    assert not run(seq, PLACE, RS, 2)["success"]
    seq = _placed(RS, 2, 12)
    q = torch.tensor([math.sin(math.radians(15) / 2), 0, 0, math.cos(math.radians(15) / 2)])
    for d in seq:
        d["cube_quat"][0, RS] = q
    assert not run(seq, PLACE, RS, 2)["success"]


# --------------------------------------------------------------------------- stack
def _stacked(top: int, bot: int, n: int, dz: float = 0.0, f_bot: float = 0.0):
    seq = []
    for _ in range(n):
        d = base()
        put(d, bot, 0.5, 0.1)
        put(d, top, 0.5, 0.1, GEO.z_rest[bot].item() + E[bot].item() / 2 + E[top].item() / 2 + dz)
        d["tcp"][0] = torch.tensor([0.5, 0.1, 0.3])
        d["f_l"][0, bot] = f_bot
        seq.append(d)
    return seq


def test_stack_success_and_negatives():
    assert run(_stacked(RS, BL, 55), STACK, RS, BL, [RS, BL])["success"]
    assert run(_stacked(RL, BS, 55), STACK, RL, BS, [RL, BS])["success"]  # large on small: heights per size
    assert not run(_stacked(RS, BL, 55, dz=0.008), STACK, RS, BL, [RS, BL])["success"]
    assert not run(_stacked(RS, BL, 55, f_bot=3.0), STACK, RS, BL, [RS, BL])["success"]  # bottom touched
    r = run(_stacked(RS, BL, 55), STACK, RS, GS, [RS, BL, GS])
    assert not r["success"] and r["wrong_dest"]


def test_stack_window_is_50_steps():
    """m3: resting for only the last 30 steps is not stable for 1 s."""
    seq = _placed(RS, 2, 25) + _stacked(RS, BL, 30)
    for d in seq[:25]:
        put(d, BL, 0.5, 0.1)
    r = run(seq, STACK, RS, BL, [RS, BL])
    assert not r["success"]
    assert r["completion_step"] == -1
    r = run(_placed(RS, 2, 5) + _stacked(RS, BL, 50), STACK, RS, BL, [RS, BL])
    assert r["success"] and r["completion_step"] == 54


def test_stack_stability_and_release_failure():
    seq = []
    for i in range(10):  # carried in the destination zone, held
        d = _stacked(RS, BL, 1, dz=0.01)[0]
        hold(d, RS)
        d["tcp"][0, 2] = d["cube_pos"][0, RS, 2]
        seq.append(d)
    seq += _stacked(RS, BL, 20)  # released, rests 20 steps, then falls off
    for _ in range(40):
        d = base()
        put(d, BL, 0.5, 0.1)
        put(d, RS, 0.58, 0.1)
        d["tcp"][0] = torch.tensor([0.5, 0.1, 0.3])
        seq.append(d)
    r = run(seq, STACK, RS, BL, [RS, BL])
    assert not r["success"]
    assert r["release_failure"] and r["release_fail_off_destination"] and r["release_fail_unstable"]
    assert r["stack_release_rest_seen"] and not r["stack_stable_after_release"]
    r = run(seq[:10] + _stacked(RS, BL, 60), STACK, RS, BL, [RS, BL])
    assert r["success"] and not r["release_failure"] and r["stack_stable_after_release"]


# --------------------------------------------------------------------------- failure detectors
def test_slam_reads_hand_table_force_only():
    """m4: a 30 N hand/finger-vs-table force is a slam; a 30 N cube-vs-table force alone is not."""
    seq = _lifted(RS, 0.12, 5)
    seq[2]["f_ht"][0] = 30.0
    r = run(seq, PICK, RS)
    assert r["slam"] and r["slam_10N"] and not r["slam_50N"]
    seq = _lifted(RS, 0.12, 5)
    seq[2]["f_ct"][0, RS] = 30.0
    assert not run(seq, PICK, RS)["slam"]


def test_jam_palm_and_scoop():
    seq = _lifted(RS, 0.12, 12)
    for d in seq:
        d["f_p"][0, RS] = 3.0
    assert run(seq, PICK, RS)["jam"]
    assert not run(seq[:9], PICK, RS)["jam"]  # 9 < 10 steps
    seq = []
    for _ in range(12):  # one-finger scoop: elevated, touched by one finger only
        d = put(base(), RS, 0.5, 0.0, GEO.z_rest[RS].item() + 0.05)
        d["f_l"][0, RS] = 3.0
        seq.append(d)
    assert run(seq, PICK, RS)["jam"]
    assert not run(_lifted(RS, 0.12, 12), PICK, RS)["jam"]  # a proper two-finger hold
    seq = _stacked(RS, BL, 12)  # resting on another cube, brushed by a finger: not elevated -> no jam
    for d in seq:
        d["f_l"][0, RS] = 3.0
    assert not run(seq, STACK, RS, BL, [RS, BL])["jam"]


def _wg_seq(held_from: dict[int, int], n: int = 12) -> list[dict]:
    """RS target at 0.4, GS / BS / BL distractors on the table; cube c is held from step held_from[c] on."""
    seq = []
    for t in range(n):
        d = put(base(), RS, 0.4, 0.0)
        for c, x in ((GS, 0.5), (BS, 0.6), (BL, 0.7)):
            put(d, c, x, 0.1)
        for c, t0 in held_from.items():
            if t >= t0:
                hold(d, c)
        seq.append(d)
    return seq


def test_wrong_grasp_cube_index():
    on = [RS, GS, BS, BL]
    r = run(_wg_seq({BL: 0}), PICK, RS, on_table=on)
    assert r["wrong_grasp"] and r["wrong_grasp_cube"] == BL  # the ALL_CUBES index of the triggering cube
    assert run(_wg_seq({BS: 2}), PICK, RS, on_table=on)["wrong_grasp_cube"] == BS
    # first trigger wins: GS triggers at step 4 (held from 0), BL only at step 8
    assert run(_wg_seq({BL: 4, GS: 0}), PICK, RS, on_table=on)["wrong_grasp_cube"] == GS
    assert run(_wg_seq({GS: 4, BL: 0}), PICK, RS, on_table=on)["wrong_grasp_cube"] == BL
    assert run(_wg_seq({BS: 0, GS: 0}), PICK, RS, on_table=on)["wrong_grasp_cube"] == GS  # same step: lowest index
    r = run(_wg_seq({}), PICK, RS, on_table=on)
    assert not r["wrong_grasp"] and r["wrong_grasp_cube"] == -1
    assert run(_wg_seq({RS: 0}), PICK, RS, on_table=on)["wrong_grasp_cube"] == -1  # holding the target is not wrong
    assert run(_wg_seq({BL: 0}), PICK, RS, on_table=[RS, GS])["wrong_grasp_cube"] == -1  # off-table cube is ignored


def test_wrong_grasp():
    seq = []
    for _ in range(6):
        d = put(base(), RS, 0.4, 0.0)
        put(d, GS, 0.5, 0.0)
        hold(d, GS)
        seq.append(d)
    assert run(seq, PICK, RS, on_table=[RS, GS])["wrong_grasp"]
    assert not run(seq[:4], PICK, RS, on_table=[RS, GS])["wrong_grasp"]  # 4 < 5 steps
    d = put(base(), RS, 0.4, 0.0)
    put(d, GS, 0.5, 0.0, GEO.z_rest[GS].item() + 0.05)  # elevated distractor touched by the arm
    d["f_a"][0, GS] = 2.0
    assert run([d], PICK, RS, on_table=[RS, GS])["wrong_grasp"]
    assert not run(_lifted(RS, 0.12, 10), PICK, RS, on_table=[RS, GS])["wrong_grasp"]


def test_early_close_band():
    d0 = put(base(), RS, 0.5, 0.0)
    d1 = put(base(), RS, 0.5, 0.0)
    d1["claw"][0] = 1.0
    d1["tcp"][0] = torch.tensor([0.5, 0.0, GEO.z_rest[RS].item()])
    assert not run([d0, d1], PICK, RS)["early_close"]
    d1b = {k: v.clone() for k, v in d1.items()}
    d1b["tcp"][0, 2] += 0.03  # closing 3 cm too high
    assert run([d0, d1b], PICK, RS)["early_close"]
    d1c = {k: v.clone() for k, v in d1.items()}
    d1c["cube_quat"][0, RS] = zq(math.radians(30))  # misaligned yaw
    assert run([d0, d1c], PICK, RS)["early_close"]
    assert run([d1], PICK, RS, grasp_dz=0.0)["early_close"] is False
    assert run([d1b], PICK, RS)["early_close"]  # closed at step 0, out of band


def test_drop_after_confirmed_grasp():
    seq = _lifted(RS, 0.12, 5)
    for _ in range(7):
        seq.append(put(base(), RS, 0.5, 0.0, GEO.z_rest[RS].item() + 0.05))  # lost in the air
    r = run(seq, PICK, RS)
    assert r["grasp_confirmed"] and r["drop"]
    assert not run(seq[:10], PICK, RS)["drop"]  # 5 steps lost: not > 5
    seq = _lifted(RS, 0.12, 5) + _placed(RS, 2, 12)  # released in the destination zone: not a drop
    assert not run(seq, PLACE, RS, 2)["drop"]


def test_distractor_disturbed_and_clean_grasp():
    a = put(base(), RS, 0.4, 0.0)
    put(a, GS, 0.6, 0.0)
    b = put(base(), RS, 0.4, 0.0)
    put(b, GS, 0.63, 0.0)
    assert run([a, b], PICK, RS, on_table=[RS, GS])["distractor_disturbed"]
    assert not run([a, a], PICK, RS, on_table=[RS, GS])["distractor_disturbed"]
    r = run(_lifted(RS, 0.12, 5), PICK, RS)
    assert r["clean"]
    seq = _lifted(RS, 0.12, 5)
    seq[1]["f_robot_table"][0] = 25.0
    assert not run(seq, PICK, RS)["clean"]


def test_invalid_is_failure():
    seq = _lifted(RS, 0.12, 5)
    seq[1]["invalid"] = torch.tensor([True])
    for d in seq:
        d.setdefault("invalid", torch.tensor([False]))
    r = run(seq, PICK, RS)
    assert r["invalid"] and not r["success"]


def test_thresholds_are_the_111_ones():
    th = Thresholds()
    assert (th.contact_n, th.slam_n, th.jam_steps, th.off_table_dz) == (1.0, 20.0, 10, 0.02)
    assert (K.k_place, K.k_stack, K.eps_z, K.wrong_grasp_steps, K.drop_steps) == (10, 50, 0.005, 5, 5)


# --------------------------------------------------------------------------- rest_cube / rest_dest (s3 probe)
def _from_afar(seq: list[dict]) -> list[dict]:
    return [base()] + seq  # step 0: every cube parked away, so the placed cube has moved


def test_rest_dest_place_and_stack():
    """The latest settled released cube and its pad / bottom cube are recorded for ANY cube, not only the target."""
    r = run(_from_afar(_placed(RS, 2, 12)), PLACE, RS, 1, [RS])  # asked for pad 1, settled on pad 2
    assert (r["rest_cube"], r["rest_dest"]) == (RS, 2) and not r["success"]
    r = run(_from_afar(_placed(RS, 2, 12)), PLACE, GS, 2, [RS, GS])  # another cube than the target
    assert (r["rest_cube"], r["rest_dest"]) == (RS, 2)
    assert run(_from_afar(_placed(RS, 2, 5)), PLACE, RS, 2)["rest_dest"] == -1  # not settled for k_place steps
    assert run(_placed(RS, 2, 12), PLACE, RS, 2)["rest_dest"] == -1  # never moved
    r = run(_from_afar(_stacked(RS, BL, 55)), STACK, RS, GS, [RS, BL, GS])
    assert (r["rest_cube"], r["rest_dest"]) == (RS, BL)
    assert (run(_lifted(RS, 0.12), PICK, RS)["rest_cube"], run(_lifted(RS, 0.12), PICK, RS)["rest_dest"]) == (-1, -1)
