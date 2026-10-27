"""Task 005 (GATES "Stage 2R — instruction-contrastive flow loss"), policy venv.

For each training sample (scene o, true instruction l, chunk a, flow time t, noise eps, x_t):
  L_true_i  = per-sample flow loss with l  (LeRobot's own SmolVLAPolicy.forward, mean over valid entries)
  L_other_i = per-sample flow loss with l' (SAME eps, t, x_t, o; l' names another valid cube of the same layout)
  L = L_true + lam * mean_i max(0, m - (L_other_i - L_true_i))
where L_true is LeRobot's scalar loss (sum / valid entries), so lam = 0 is the plain trainer bit for bit.

l' = the true instruction with its colour word replaced by the colour of another member of the episode's layout group
(vla_common.guidance.other_instructions: same template, same optional size word, so l' is a train-template instruction
naming exactly that cube; s2a scenes have one size and distinct colours). The other members ARE the layout's valid train
targets (collect.py --paired-targets rolls a layout out once per valid target), so l' never names a held-out cube.
s2 (2b amended: paired s2 scenes, colours AND sizes; stage from the sidecar's file_attrs): a target is a (colour, size)
cube, and l' = the true instruction with its target noun phrase replaced by the other member's
(vla_common.guidance.other_instruction: size word iff the original has one or the other colour is on the table in two
sizes, so l' names exactly that cube; the rest of the template unchanged). s2a records and l' are unchanged.
s3a / s3b (task 007: paired place / stack scenes, a member target is a (cube, pad) / (top, bottom) pair): l' = the
instruction text of another member of the same group (its own template; it names that member's pair exactly).
s3a1 / s3b1 (task 009b, one cube size per scene): exactly as s3a / s3b.
Union sets (task 007: a stage's paired set + the 2b paired s2 set, two streams): a group is (stage, data seed,
layout_id) -- record field "group" -- so layout ids of different streams never merge; "layout" stays the layout_id.
Choice: uniform over the others, rng [seed, step, global row] (a pure function of the global batch: identical for any
world size, exact on resume, and it never touches torch's global RNG).

The noise and time are drawn with VLAFlowMatching's own samplers in its own order (noise, then time) before the two
forwards, so the RNG stream equals the plain trainer's. Both branches' x_t / t are recorded at embed_suffix and asserted
equal. `share_vision`: the second branch reuses the first branch's image embeddings (embed_image is a function of the
images alone; the inputs are asserted equal), so the vision tower runs once; gradients reach it from both branches.
"""

from __future__ import annotations

import json
from contextlib import contextmanager
from pathlib import Path

import numpy as np
import torch

from vla_common.guidance import COLOUR_RE, named_cubes, other_instruction, other_instructions

LANG_KEYS = ("observation.language.tokens", "observation.language.attention_mask")


def _stage(s: dict) -> str:
    """The episode's collect stage (exported sidecar: file_attrs; a row without one = s2a, the legacy paired format)."""
    return s.get("file_attrs", {}).get("stage", s.get("stage", "s2a"))


def _seed(s: dict):
    """The episode's data seed (sidecar `seed`, else file_attrs; None in legacy synthetic rows)."""
    return s.get("seed", s.get("file_attrs", {}).get("seed"))


def _pair(sp: dict) -> tuple:
    """s3a / s3b member target: (cube, pad colour) / (top, bottom)."""
    d = sp["dest"]
    return tuple(sp["target"]), tuple(d) if isinstance(d, list) else d


def layout_table(sidecar: list[dict]) -> dict[int, dict]:
    """lerobot episode -> {"instruction", "target", "others" (other members' targets, member order), "layout"
    (layout_id), "group" ((stage, data seed, layout_id))}. Groups by (stage, seed, layout_id); one (seed, layout_id) with
    members of two stages is an error. Every group's targets must be distinct. s2a: targets are colours (one size per
    scene). s2: targets are (colour, size) tuples and the record also holds "cubes" (the table, spec order); the
    instruction must name exactly its target among the table cubes. s3a / s3b: targets are (cube, dest) pairs, the
    record also holds "other_instructions" / "other_eps" (the other members' instruction / lerobot episode, member
    order); the instruction must name exactly its cube and destination."""
    from vla_common.tasks import TaskSpec, matches, matches_dest

    groups: dict[tuple, list[dict]] = {}
    for s in sidecar:
        if not s.get("paired"):
            raise ValueError(f"episode {s.get('lerobot_episode_index')}: not a paired-scene episode")
        groups.setdefault((_stage(s), _seed(s), int(s["layout_id"])), []).append(s)
    st: dict[tuple, set] = {}
    for stage, seed, g in groups:
        st.setdefault((seed, g), set()).add(stage)
    for (seed, g), v in st.items():
        if len(v) != 1:
            raise ValueError(f"layout {g}: members of different stages {v}")
    out: dict[int, dict] = {}
    for key, mem in groups.items():
        stage, g = key[0], key[2]
        mem = sorted(mem, key=lambda s: int(s["member"]))
        specs = [json.loads(s["spec_json"]) if isinstance(s["spec_json"], str) else s["spec_json"] for s in mem]
        s2, s3 = stage == "s2", stage in ("s3a", "s3b", "s3a1", "s3b1")
        tgt = [_pair(sp) if s3 else tuple(sp["target"]) if s2 else sp["target"][0] for sp in specs]
        if len(mem) < 2 or len(set(tgt)) != len(tgt):
            raise ValueError(f"layout {g}: needs >= 2 members with distinct targets, got {tgt}")
        cubes = {tuple(c) for c in specs[0]["cubes"]}
        for s, sp in zip(mem, specs):
            if {tuple(c) for c in sp["cubes"]} != cubes or tuple(sp["target"]) not in cubes:
                raise ValueError(f"layout {g}: member {s['member']} has another scene or an absent target")
        for s, sp, t in zip(mem, specs, tgt):
            out[int(s["lerobot_episode_index"])] = {"instruction": s["instruction"], "target": t, "layout": g,
                                                    "others": [c for c in tgt if c != t], "group": key}
            if s3:
                ts = TaskSpec.from_json(json.dumps(sp))
                if matches(ts, s["instruction"]) != [t[0]] or matches_dest(ts, s["instruction"]) != [t[1]]:
                    raise ValueError(f"layout {g}: {s['instruction']!r} does not name exactly its target {t}")
                out[int(s["lerobot_episode_index"])] |= {
                    "other_instructions": [o["instruction"] for o in mem if o is not s],
                    "other_eps": [int(o["lerobot_episode_index"]) for o in mem if o is not s]}
            if s2:
                tab = [tuple(c) for c in specs[0]["cubes"]]
                if named_cubes(s["instruction"], tab) != [[t]]:
                    raise ValueError(f"layout {g}: {s['instruction']!r} does not name exactly its target {t}")
                out[int(s["lerobot_episode_index"])]["cubes"] = tab
    return out


def read_table(dataset: Path) -> dict[int, dict]:
    rows = [json.loads(x) for x in (Path(dataset) / "meta" / "episodes_113.jsonl").read_text().splitlines()]
    return layout_table(rows)


def pick_other(table: dict[int, dict], episodes: list[int], tasks: list[str], rows: list[int], seed: int,
               step: int) -> list[str]:
    """l' per sample (see the module doc). Asserts the batch's instruction is the sidecar's (episode alignment), l' != l,
    and l' names a member target of the same layout."""
    return pick_other_eps(table, episodes, tasks, rows, seed, step)[0]


def pick_other_eps(table: dict[int, dict], episodes: list[int], tasks: list[str], rows: list[int], seed: int,
                   step: int) -> tuple[list[str], list[int]]:
    """pick_other + the lerobot episode of the member l' names (the same draw; for the cross-target candidate)."""
    out, eps = [], []
    for e, task, r in zip(episodes, tasks, rows):
        rec = table[int(e)]
        if task.strip() != rec["instruction"].strip():
            raise AssertionError(f"episode {e}: batch task {task!r} != sidecar instruction {rec['instruction']!r}")
        if "other_instructions" in rec:  # s3a / s3b: another member's own instruction, member order
            cands = rec["other_instructions"]
        elif "cubes" in rec:  # s2: one l' per other member target, member order
            cands = [other_instruction(rec["instruction"], rec["target"], o, rec["cubes"]) for o in rec["others"]]
        else:
            cands = other_instructions(rec["instruction"], rec["target"], [rec["target"], *rec["others"]])
        k = int(np.random.default_rng([seed, step, int(r)]).integers(len(cands)))
        out.append(cands[k])
        if "other_eps" in rec:
            eps.append(rec["other_eps"][k])
            continue
        col = rec["others"][k]  # other_instructions keeps the order of rec["others"] (cube_colours minus the target)
        (o,) = [x for x, y in table.items() if y["group"] == rec["group"] and y["target"] == col]
        eps.append(o)
    check_others(table, episodes, tasks, out)
    return out, eps


def check_others(table: dict[int, dict], episodes: list[int], tasks: list[str], others: list[str]) -> None:
    """Spec item (ii), asserted on every batch: l' is never l and always names another member target of the layout."""
    for e, t, o in zip(episodes, tasks, others):
        rec = table[int(e)]
        if o.strip() == t.strip():
            raise AssertionError(f"episode {e}: l' == l ({o!r}) -- the contrast is void")
        if "other_instructions" in rec:  # s3a / s3b: l' is one of the other members' instructions
            ok = o in rec["other_instructions"]
        elif "cubes" in rec:  # s2: exactly one cube noun phrase, resolving (colour + size) to exactly one other member
            named = named_cubes(o, rec["cubes"])
            ok = len(named) == 1 and len(named[0]) == 1 and named[0][0] in rec["others"]
        else:
            named = COLOUR_RE.findall(o)
            ok = len(named) == 1 and named[0] in rec["others"]
        if not ok:
            raise AssertionError(f"episode {e}: l' {o!r} does not name exactly one other cube of layout {rec['layout']}")


@contextmanager
def _patched(obj, name: str, fn):
    """Instance-attribute patch; restores what was there (an outer patch, or nothing = the class method)."""
    prev = obj.__dict__.get(name, _MISSING)
    setattr(obj, name, fn)
    try:
        yield
    finally:
        if prev is _MISSING:
            delattr(obj, name)
        else:
            setattr(obj, name, prev)


_MISSING = object()


def flow_pair(policy, b_true: dict, b_other: dict, share_vision: bool = True) -> dict:
    """Both branches with one noise / time draw. Returns l_true (LeRobot's scalar), per-sample lt / lo (B,), and the
    recorded x_t / t per branch."""
    model = policy.model
    cfg = policy.config
    act = b_true["action"]
    B = act.shape[0]
    for k, v in b_true.items():  # same o, a, pad: only the language differs
        if k in LANG_KEYS or not torch.is_tensor(v):
            continue
        if not torch.equal(v, b_other[k]):
            raise AssertionError(f"branch inputs differ at {k}")
    noise = model.sample_noise((B, cfg.chunk_size, cfg.max_action_dim), act.device)  # VLAFlowMatching.forward's order
    time = model.sample_time(B, act.device)
    elem, xs, vs, memo = [], [], [], {"mode": "record", "k": 0, "rec": []}
    f_fwd, f_suf, f_img = model.forward, model.embed_suffix, model.vlm_with_expert.embed_image

    def fwd(*a, **k):
        out = f_fwd(*a, **k)
        elem.append(out)
        return out

    def suf(noisy_actions, timestep):
        xs.append((noisy_actions, timestep))
        return f_suf(noisy_actions, timestep)

    def img(x):
        if memo["mode"] == "record":
            out = f_img(x)
            memo["rec"].append((x, out))
            return out
        x0, out = memo["rec"][memo["k"]]
        memo["k"] += 1
        if not torch.equal(x, x0):
            raise AssertionError("shared vision: second-branch image differs from the first")
        return out

    hook = model.action_out_proj.register_forward_hook(lambda m, i, o: vs.append(o))  # v_t (forward: v_t = this output)
    with _patched(model, "forward", fwd), _patched(model, "embed_suffix", suf):
        if share_vision:
            with _patched(model.vlm_with_expert, "embed_image", img):
                l_true, _ = policy.forward(b_true, noise=noise, time=time)
                memo["mode"] = "replay"
                policy.forward(b_other, noise=noise, time=time)
            assert memo["k"] == len(memo["rec"]), (memo["k"], len(memo["rec"]))
        else:
            l_true, _ = policy.forward(b_true, noise=noise, time=time)
            policy.forward(b_other, noise=noise, time=time)
    hook.remove()
    assert len(elem) == 2 and len(xs) == 2 and len(vs) == 2
    if not (torch.equal(xs[0][0], xs[1][0]) and torch.equal(xs[0][1], xs[1][1])):  # spec item (iv)
        raise AssertionError("branches saw different x_t / t")
    A = cfg.action_feature.shape[0]
    pad = b_true.get("action_is_pad")
    per = [per_sample_loss(e, A, pad) for e in elem]
    return {"l_true": l_true, "lt": per[0], "lo": per[1], "x_t": xs[0][0], "t": xs[0][1], "v_true": vs[0],
            "v_other": vs[1]}


def cross_terms(v_other: torch.Tensor, x_t: torch.Tensor, t: torch.Tensor, a_other: torch.Tensor, action_dim: int,
                rows: torch.Tensor) -> dict:
    """Fix candidate (consultant 113, 2026-10-25; NOT registered, not used by the Stage 2R run): the cross-instruction
    flow target. With x_t = t eps + (1 - t) a_true and the other member's chunk a_other, the exact velocity under l'
    (deterministic actions given (o, l')) is u' = (x_t - a_other) / t; the term is the t^2-weighted
    ||t v(x_t, t | o, l') - (x_t - a_other)||^2, per sample (LeRobot's reduction, no padded steps: rows are frame-0
    windows), averaged over `rows` (bool (B,): where a_other is the target for THIS observation -- only frame 0, the
    t = 0 state shared by the layout's members; data gate P1 / P2). Returns the per-sample values and the row count."""
    r = t[:, None, None] * v_other - (x_t - a_other)
    per = per_sample_loss(r.pow(2), action_dim, None)
    m = rows.to(per.dtype)
    return {"per": per, "sum": (per * m).sum(), "n": m.sum()}


def per_sample_loss(elem: torch.Tensor, action_dim: int, pad: torch.Tensor | None) -> torch.Tensor:
    """(B,) per-sample flow loss from VLAFlowMatching.forward's elementwise losses (B, chunk, max_action_dim): LeRobot's
    own reduction (SmolVLAPolicy.forward: truncate to the action dims, zero the padded steps, sum / valid entries) per
    row. The hinge's L_true_i / L_other_i and the margin cross-check (tests/test_s2r_tools.py) both go through here."""
    valid = torch.ones(elem.shape[:2], dtype=torch.bool, device=elem.device) if pad is None else ~pad
    nv = (valid.sum(1) * action_dim).clamp_min(1)
    return (elem[:, :, :action_dim] * valid.unsqueeze(-1)).sum((1, 2)) / nv


def hinge_terms(lt: torch.Tensor, lo: torch.Tensor, margin: float) -> dict:
    h = torch.relu(margin - (lo - lt))
    return {"hinge": h.mean(), "gap": (lo - lt).mean().detach(), "active": (h > 0).float().mean().detach(),
            "lo": lo.mean().detach()}


def step_flags(rows: list[dict], margin: float, tol: float = 0.0) -> dict:
    """Instrument read of steps.jsonl (spec item (iii)): a void contrast (l' == l) gives gap == 0 and hinge == m on
    every step. Flagged when |gap| <= tol and |hinge - m| <= tol on all logged steps."""
    gaps = np.array([r["gap"] for r in rows], float)
    hin = np.array([r["hinge"] for r in rows], float)
    void = bool(len(rows) and np.all(np.abs(gaps) <= tol) and np.all(np.abs(hin - margin) <= tol))
    return {"steps": len(rows), "gap_mean": float(gaps.mean()) if len(rows) else None,
            "hinge_mean": float(hin.mean()) if len(rows) else None, "void_contrast_flag": void}
