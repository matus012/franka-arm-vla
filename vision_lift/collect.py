"""Rollouts in the student env with cameras: teacher-driven data collection, student evaluation, DAgger (mixed).

One `Runner.run_batch()` = one batch of N parallel 400-step episodes from a common reset (lift_rl.evaluation.rollout,
the 111 evaluator). Per control step t:
  - on camera ticks (t % PERIOD == 0) both cameras are rendered; the frame is written to disk (if recording) and
    pushed into the latency buffer; the policy sees the delayed frame (vision_lift.frames) — `Recorder`;
  - labels: teacher arm action on the 48-dim privileged observation of the TRUE state, teacher claw label =
    phase machine >= CLOSE (the reflex's own closing state);
  - executed action: per env, the teacher (arm label + N(0, action_sigma) noise, claw label) or the student
    (deterministic arm deltas + claw command), chosen by `driver` / `beta`;
  - privileged per-step records (phase, in_band, drop, jam step) feed the claw metrics, never the student.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import torch

from lift_rl.evaluation import contact_readings, cube_and_target, rollout
from lift_rl.metrics import EpisodeTracker, Thresholds, summarize
from lift_rl.phases import CLOSE
from lift_rl.phased_cfg import cube_state, tcp_and_rot

from . import cameras as C
from . import claw_metrics as CM
from .dataset import EpisodeWriter
from .frames import LATENCY, PERIOD, DelayedFrames, n_frames


def _np(x: torch.Tensor) -> np.ndarray:
    return x.detach().cpu().numpy()


class Recorder:
    """Camera ticks, latency buffer and (optional) disk writer — the single place that decides which frame the
    policy sees at step t and which frame index is stored for step t."""

    def __init__(self, writer: EpisodeWriter | None, period: int = PERIOD, latency: int = LATENCY):
        self.writer, self.period = writer, period
        self.delayed = DelayedFrames(period, latency)
        self.frame_of_step: list[int] = []

    def frame(self, t: int, capture_fn) -> tuple[int, dict]:
        if self.delayed.is_tick(t):
            fr = capture_fn()
            if self.writer is not None:
                self.writer.add_frame(t // self.period, _np(fr["side_rgb"]), _np(fr["wrist_rgb"]),
                                      _np(fr["side_depth"]), _np(fr["wrist_depth"]), _np(fr["proprio"]),
                                      priv={k: _np(v) for k, v in fr.get("priv", {}).items()})
            self.delayed.capture(t, fr)
        k, fr = self.delayed.current()
        self.frame_of_step.append(k)
        return k, fr

    def step(self, data: dict, priv: dict) -> None:
        if self.writer is not None:
            self.writer.add_step({k: _np(v) for k, v in data.items()}, {k: _np(v) for k, v in priv.items()})

    def close(self, episodes: dict) -> None:
        if self.writer is not None:
            self.writer.close(np.array(self.frame_of_step), {k: _np(v) if torch.is_tensor(v) else v
                                                              for k, v in episodes.items()})


def student_input(fr: dict, proprio: torch.Tensor, depth: bool) -> dict:
    inp = {"side_rgb": fr["side_rgb"].permute(0, 3, 1, 2), "wrist_rgb": fr["wrist_rgb"].permute(0, 3, 1, 2),
           "proprio": proprio}
    if depth:
        inp["side_depth"] = fr["side_depth"][:, None]
        inp["wrist_depth"] = fr["wrist_depth"][:, None]
    return inp


class Runner:
    def __init__(self, env, teacher, student=None, depth: bool = False, driver: str = "teacher", beta: float = 1.0,
                 action_sigma: float = 0.0, seed: int = 0, th: Thresholds | None = None, relabel: str = "stateful",
                 visual=None):
        """driver: 'teacher' | 'student' | 'mixed' (per-episode: each env is teacher-driven with prob. beta)."""
        assert driver in ("teacher", "student", "mixed")
        assert relabel in ("stateful", "resync", "catchup")  # vision_lift.relabel; labels of STUDENT-driven envs only
        self.relabel = relabel
        self.visual = visual  # vision_lift.visual_rand.VisualRandomizer (Stage 4) or None
        assert driver == "teacher" or student is not None
        self.env, self.u = env, env.unwrapped
        self.teacher, self.student, self.depth = teacher, student, depth
        self.driver, self.beta, self.sigma = driver, beta, action_sigma
        self.th = th or Thresholds()
        self.gen = torch.Generator(device=self.u.device)
        self.gen.manual_seed(seed)
        self.cube_ids = C.cube_shape_ids(self.u)
        self.aimed = False
        self.T = int(self.u.cfg.eval_steps)

    def _geom(self) -> dict:
        """Privileged geometry for diagnostics only: hand-cube relative yaw mod 90 deg in (-45, 45], TCP xy offset from
        the cube centre and TCP height above it [m]."""
        import math

        u = self.u
        tcp, R = tcp_and_rot(u)
        cube, yaw = cube_state(u)
        psi = torch.atan2(R[:, 1, 1], R[:, 0, 1])
        d = torch.remainder(psi - yaw + math.pi / 4, math.pi / 2) - math.pi / 4
        return {"rel_yaw_deg": torch.rad2deg(d).to(torch.float16),
                "tcp_xy_err": torch.linalg.norm(tcp[:, :2] - cube[:, :2], dim=-1).to(torch.float16),
                "tcp_dz": (tcp[:, 2] - cube[:, 2]).to(torch.float16)}

    def capture(self, obs) -> dict:
        u = self.u
        pos, yaw = cube_state(u)
        return {"side_rgb": C.rgb(u, "side_cam").to(torch.uint8).clone(),
                "wrist_rgb": C.rgb(u, "wrist_cam").to(torch.uint8).clone(),
                "side_depth": C.depth(u, "side_cam").clone(), "wrist_depth": C.depth(u, "wrist_cam").clone(),
                "proprio": obs["student"].clone(),
                "priv": {"cube_pos": pos.clone(), "cube_yaw": yaw.clone(),
                         "side_cube_px": C.cube_pixels(u, "side_cam", self.cube_ids),
                         "wrist_cube_px": C.cube_pixels(u, "wrist_cam", self.cube_ids)}}

    @torch.inference_mode()
    def run_batch(self, out_path: Path | None = None, attrs: dict | None = None, on_step_extra=None) -> dict:
        u, N = self.u, self.u.num_envs
        writer = None
        if out_path is not None:
            res = u.scene["side_cam"].cfg.width
            writer = EpisodeWriter(out_path, N, self.T, n_frames(self.T), res, obs_dim(u), attrs or {})
        rec = Recorder(writer)
        if self.driver == "teacher":
            teacher_env = torch.ones(N, dtype=torch.bool, device=u.device)
        elif self.driver == "student":
            teacher_env = torch.zeros(N, dtype=torch.bool, device=u.device)
        else:
            teacher_env = torch.rand(N, generator=self.gen, device=u.device) < self.beta
        st = {"t": 0}
        priv_steps: dict[str, list] = {k: [] for k in ("phase", "in_band", "drop", "jam_step", "cmd", "yaw_cmd", "rel_yaw",
                                                         "cu_phase", "cu_drop")}
        from .baseline import machine_inputs as _mi
        from .relabel import CatchUpMachine as _CU

        ev = {"cm": _CU(N, u.device), "prev": None}  # evaluator-side catch-up machine: phase-based METRICS only
        tracker_jam = EpisodeTracker(N, u.device, self.th)

        def policy(obs):
            t = st["t"]
            if self.visual is not None:
                if t == 0:  # after this batch's reset, before its first render
                    st["visual"] = self.visual.apply()
            elif not self.aimed:
                C.aim_side(u)
                self.aimed = True
            k, fr = rec.frame(t, lambda: self.capture(obs))
            self.last_fed = (k, fr)
            pm = u._phase_machine
            t_arm = self.teacher(obs["policy"]).clamp(-1.0, 1.0)  # the IK clips to [-1, 1]: label = effective action
            claw_label = (pm.phase >= CLOSE).float()
            t_arm_exec = t_arm
            if self.relabel in ("resync", "catchup"):
                from .baseline import machine_inputs
                from .relabel import CatchUpMachine, resync_teacher_obs, stateless_phase

                cube, yaw = cube_state(u)
                if t == 0 or st.get("prev_tcp") is None:
                    st["prev_tcp"] = tcp_and_rot(u)[0].clone()
                tcp, inp = machine_inputs(u, cube, yaw, st["prev_tcp"])
                st["prev_tcp"] = tcp.clone()
                if self.relabel == "resync":
                    ph_rs, gr_rs = stateless_phase(*inp[:9])
                else:
                    if t == 0:
                        st["cm"] = CatchUpMachine(N, u.device)  # step 0 = reset, like the env's machine
                    else:
                        st["cm"].step(*inp)
                    ph_rs, gr_rs = st["cm"].phase.clone(), st["cm"].grasp.clone()
                lab_arm = self.teacher(resync_teacher_obs(obs["policy"], ph_rs, gr_rs)).clamp(-1.0, 1.0)
                t_arm = torch.where(teacher_env[:, None], t_arm, lab_arm)
                claw_label = torch.where(teacher_env, claw_label, (ph_rs >= CLOSE).float())
            t_exec = torch.cat([(t_arm_exec + torch.randn(t_arm.shape, generator=self.gen, device=u.device) * self.sigma)
                                .clamp(-1.0, 1.0), (pm.phase >= CLOSE).float()[:, None]], 1)
            if self.student is not None and not bool(teacher_env.all()):
                s_exec = self.student.act(student_input(fr, obs["student"], self.depth))
                a = torch.where(teacher_env[:, None], t_exec, s_exec)
            else:
                a = t_exec
            in_band = pm.out["in_band"] if (t > 0 and "in_band" in pm.out) else torch.zeros(N, dtype=torch.bool,
                                                                                               device=u.device)
            rec.step({"proprio": obs["student"], "teacher_arm": t_arm, "claw_label": claw_label.to(torch.uint8),
                      "exec_action": a, "driver": teacher_env.to(torch.uint8).expand(N)},
                     {"phase": pm.phase.to(torch.int8), "in_band": in_band, **self._geom(),
                      **({"label_phase": st["cm"].phase.to(torch.int8)} if "cm" in st else {})})
            priv_steps["phase"].append(pm.phase.clone())
            priv_steps["in_band"].append(in_band.clone())
            priv_steps["cmd"].append(a[:, 4].clone())
            priv_steps["yaw_cmd"].append(a[:, 3].clone())
            priv_steps["cu_phase"].append(ev["cm"].phase.clone())
            priv_steps["rel_yaw"].append(self._geom()["rel_yaw_deg"].float())
            st["t"] += 1
            return a

        def on_step(t, uu):
            pm = uu._phase_machine
            c = contact_readings(uu)
            cube, _ = cube_and_target(uu)
            priv_steps["drop"].append(pm.out["drop"].bool().clone())
            cube_t, yaw_t = cube_state(uu)
            if ev["prev"] is None:
                ev["prev"] = tcp_and_rot(uu)[0].clone()
            tcp_t, inp_t = _mi(uu, cube_t, yaw_t, ev["prev"])
            ev["prev"] = tcp_t.clone()
            ev["cm"].step(*inp_t)
            priv_steps["cu_drop"].append(ev["cm"].m.out["drop"].bool().clone())
            priv_steps["jam_step"].append(tracker_jam.jam_step(cube[:, 2], c["palm"], c["lf"], c["rf"], c["arm"]))
            if on_step_extra is not None:
                on_step_extra(t, uu)

        res = rollout(self.env, policy, self.th, on_step=on_step)
        S = {k: torch.stack(v, 1) for k, v in priv_steps.items()}  # (N, T)
        res["early_close"] = CM.early_close(S["cmd"], S["in_band"])
        res["drop_before_hold"] = CM.drop_before_hold(S["drop"], S["phase"])
        res["jam_phase"] = CM.first_event_phase(CM.jam_onset(S["jam_step"], self.th.jam_steps), S["phase"])
        res["drop_phase"] = CM.first_event_phase(S["drop"], S["phase"])
        # the same phase-based metrics from the evaluator's catch-up machine (the stateful machine keeps its waypoint
        # memory and under-counts phases for students that skip the waypoint; consultant, 3.3 audit)
        res["cu_drop_before_hold"] = CM.drop_before_hold(S["cu_drop"], S["cu_phase"])
        res["cu_jam_phase"] = CM.first_event_phase(CM.jam_onset(S["jam_step"], self.th.jam_steps), S["cu_phase"])
        res["cu_drop_phase"] = CM.first_event_phase(S["cu_drop"], S["cu_phase"])
        for k in range(6):
            res[f"cu_reached_{k}"] = (S["cu_phase"] == k).any(1) | (ev["cm"].m.entered[:, k])
        res["teacher_driven"] = teacher_env.clone()
        # diagnostic (3.2): per episode, pre-grasp steps (HOVER/DESCEND) with |relative yaw| > 10 deg and a yaw command,
        # fraction rotating toward the nearer cube face; NaN when the episode has no such step
        pre = (S["phase"] <= 1) & (S["rel_yaw"].abs() > 10) & (S["yaw_cmd"].abs() > 0.05)
        good = (torch.sign(S["yaw_cmd"]) == -torch.sign(S["rel_yaw"])) & pre
        res["yaw_toward_nearer_face"] = torch.where(pre.sum(1) > 0, good.sum(1) / pre.sum(1).clamp_min(1),
                                                    torch.full((N,), float("nan"), device=u.device))
        res["cube_mass"], res["cube_mu"] = cube_physics(u)
        if "cm" in st:  # catch-up relabeler: per-episode R1 / R2 fire counts (audit)
            res["label_r1"], res["label_r2"] = st["cm"].r1.clone(), st["cm"].r2.clone()
        rec.close({k: v for k, v in res.items() if torch.is_tensor(v)})
        return res


def cube_physics(u) -> tuple[torch.Tensor, torch.Tensor]:
    """(N,) cube mass [kg] and grasp friction as the simulator holds them (set at this episode's reset)."""
    import warp as wp
    from isaaclab_newton.physics import NewtonManager

    obj = u.scene["object"]
    m = obj.data.body_mass
    m = (m.torch if hasattr(m, "torch") else m).reshape(u.num_envs, -1).sum(1)
    try:
        mu = wp.to_torch(obj._root_view.get_attribute("shape_material_mu", NewtonManager.get_model())[:, 0])
        mu = mu.reshape(u.num_envs, -1)[:, 0].to(m.device)
    except Exception:  # noqa: BLE001
        mu = torch.full_like(m, float("nan"))
    return m.clone(), mu.clone()


MASS_BINS = (0.1, 0.2, 0.3, 0.4, 0.5001)
MU_BINS = (0.5, 0.75, 1.0, 1.2501)


def by_physics_bin(outcomes: dict) -> dict:
    """Success / drop-before-HOLD / any-drop counts per (mass, friction) bin (consultant request, Stage 3.4 / 5)."""
    m, mu = outcomes["cube_mass"], outcomes["cube_mu"]
    out = {}
    for i in range(len(MASS_BINS) - 1):
        for j in range(len(MU_BINS) - 1):
            sel = (m >= MASS_BINS[i]) & (m < MASS_BINS[i + 1]) & (mu >= MU_BINS[j]) & (mu < MU_BINS[j + 1])
            n = int(sel.sum())
            if n == 0:
                continue
            key = f"mass_{MASS_BINS[i]:.1f}-{min(MASS_BINS[i + 1], 0.5):.1f}_mu_{MU_BINS[j]:.2f}-{min(MU_BINS[j + 1], 1.25):.2f}"
            out[key] = {"n": n, "success": int(outcomes["success"][sel].sum()),
                        "drop_before_hold": int(outcomes["drop_before_hold"][sel].sum()),
                        "any_drop": int((outcomes["drops"][sel] > 0).sum())}
    return out


def obs_dim(u) -> int:
    return int(u.observation_manager.group_obs_dim["student"][0])


def summarize_112(outcomes: dict) -> dict:
    """111 summary + the claw metrics + jam/drop by phase (consultant request, from Stage 2 on)."""
    s = summarize({k: v for k, v in outcomes.items() if k not in ("early_close", "drop_before_hold", "jam_phase",
                                                                   "drop_phase", "teacher_driven", "cube_mass",
                                                                   "cube_mu", "label_r1", "label_r2",
                                                                   "yaw_toward_nearer_face")
                                   and not k.startswith("cu_")})
    n = int(outcomes["success"].numel())
    from lift_rl.metrics import wilson_ci

    for key in ("early_close", "drop_before_hold"):
        k = int(outcomes[key].sum())
        s[key] = {"count": k, "rate": k / n, "ci95": list(wilson_ci(k, n))}
    s["jam_by_phase"] = CM.by_phase(outcomes["jam_phase"])
    if "cu_drop_before_hold" in outcomes:
        from lift_rl.phases import PHASES

        k = int(outcomes["cu_drop_before_hold"].sum())
        s["catchup_metrics"] = {
            "drop_before_hold": {"count": k, "rate": k / n},
            "jam_by_phase": CM.by_phase(outcomes["cu_jam_phase"]),
            "drop_by_phase": CM.by_phase(outcomes["cu_drop_phase"]),
            "reached": {PHASES[j]: float(outcomes[f"cu_reached_{j}"].float().mean()) for j in range(6)}}
    if "yaw_toward_nearer_face" in outcomes:
        y = outcomes["yaw_toward_nearer_face"].float()
        s["yaw_toward_nearer_face_mean"] = float(y[~torch.isnan(y)].mean()) if bool((~torch.isnan(y)).any()) else None
    s["drop_by_phase"] = CM.by_phase(outcomes["drop_phase"])
    if "cube_mass" in outcomes:
        s["by_mass_friction"] = by_physics_bin(outcomes)
        s["cube_mass_mean"] = float(outcomes["cube_mass"].mean())
        s["cube_mu_mean"] = float(outcomes["cube_mu"].float().nanmean())
    return s

