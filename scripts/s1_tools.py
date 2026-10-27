"""GATES Stage 1 helpers (policy venv, PERUN `pol_exec`).

  python scripts/s1_tools.py arms --dataset /data/datasets/0c_s1_pick --out /runs/s1_arms/arms_build.json --device cpu
      -> arm faithfulness (i): trainable names of F, U and the mutant U' (only freeze_vision_encoder=false) built on
         lerobot/smolvla_base; names(U) - names(F) == the vision_model names, names(F) < names(U), counts, dtypes;
         the mutant must fail the same check.
  python scripts/s1_tools.py post --f-run /runs/s1_frozen --u-run /runs/s1_unfrozen --out /runs/s1_arms/arms_post.json
      -> (ii) final checkpoints vs smolvla_base: F vision_model, both arms' connector / text model / lm_head
         bit-identical; U vision_model differs (max |d| > 0) in >= 90 % of its tensors. (iii) both steps.jsonl read
         the same row indices at every step.
  python scripts/s1_tools.py vloss --dataset ... --ckpt <pretrained_model> --out <json>
      -> offline flow-matching loss on the `val` episodes (early abort (c)): one row per camera-frame window
         (WindowSampler seed 0, epoch 0, val rows only), the checkpoint served as in the closed loop (strict fp32),
         noise and times drawn after torch.manual_seed(1000 + batch index) -> identical for every checkpoint.
  python scripts/s1_tools.py taskdump --dataset <dir> --ckpt <pretrained_model> --out <json> [--n 10]
      -> the trainer's first batch (CPU): task strings vs the sidecar instruction (asserted, 64 rows), decoded tokens.
  python scripts/s1_tools.py mix --dataset <union set> --u-run <run> --out results/s3a_mix.json [--mutant first-dataset-only]
      -> GATES 3A / 3B instrument (i), CPU: from the run's steps.jsonl per epoch the windows drawn from non-pick vs pick
         episodes (expect 347,700 / 220,400), no `val` row, every train window exactly once, steps per epoch.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

PARTS = {"vision_model": ".vlm.model.vision_model.", "connector": ".vlm.model.connector.",
         "text_model": ".vlm.model.text_model.", "lm_head": ".vlm.lm_head."}


def arms(a: argparse.Namespace) -> None:
    import torch

    from vla_policy import train_smolvla as T

    ds = T.open_dataset(T.resolve_dataset(a.dataset, None), "torchcodec")
    is_val, _, _ = T.split_episodes(ds)
    stats, _ = T.norm_stats(ds, ~is_val)
    built = {}
    for k, (fve, teo, rf) in {"F": (True, True, False), "U": (False, False, True),
                              "mutant_fve_only": (False, True, False)}.items():
        pol = T.build_policy("lerobot/smolvla_base", ds, stats, 20, fve, teo, a.device, rf, True)[0]
        names = {n for n, p in pol.named_parameters() if p.requires_grad}
        vis = {n for n, _ in pol.named_parameters() if PARTS["vision_model"] in n}
        rep = T.param_report(pol) if k != "mutant_fve_only" else {
            "params_trainable": sum(p.numel() for p in pol.parameters() if p.requires_grad)}
        built[k] = {"names": names, "vis": vis, "report": rep,
                    "trainable_dtypes": sorted({str(p.dtype) for p in pol.parameters() if p.requires_grad})}
        del pol
        torch.cuda.empty_cache() if torch.cuda.is_available() else None
    f, vis = built["F"]["names"], built["F"]["vis"]

    def check(u: set[str]) -> bool:
        return f < u and u - f == vis

    res = {"check_i_U": check(built["U"]["names"]), "check_i_mutant_fve_only": check(built["mutant_fve_only"]["names"]),
           "n_vision_tensors": len(vis), "counts": {k: v["report"]["params_trainable"] for k, v in built.items()},
           "expected": {"F": 99_880_992, "U": 186_314_016},
           "reports": {k: v["report"] for k, v in built.items()},
           "trainable_dtypes": {k: v["trainable_dtypes"] for k, v in built.items()}, "git_sha": T.git_sha()}
    res["pass"] = (res["check_i_U"] and not res["check_i_mutant_fve_only"]
                   and res["counts"]["F"] == 99_880_992 and res["counts"]["U"] == 186_314_016)
    a.out.parent.mkdir(parents=True, exist_ok=True)
    a.out.write_text(json.dumps(res, indent=1, default=str))
    print("ARMS=" + json.dumps({k: res[k] for k in ("check_i_U", "check_i_mutant_fve_only", "counts", "pass")}))


def post(a: argparse.Namespace) -> None:
    import torch
    from huggingface_hub import hf_hub_download
    from safetensors.torch import load_file

    base = load_file(hf_hub_download("lerobot/smolvla_base", "model.safetensors"))

    def final(run: Path) -> Path:
        return run / "checkpoints" / (run / "checkpoints" / "last.txt").read_text().strip() / "pretrained_model"

    res: dict = {}
    for arm, run in (("F", a.f_run), ("U", a.u_run)):
        sd = load_file(str(final(run) / "model.safetensors"))
        per = {}
        for part, key in PARTS.items():
            ks = [k for k in base if key in k]
            assert ks and all(k in sd for k in ks), (arm, part, len(ks))
            moved = [float((sd[k].float() - base[k].float()).abs().max()) > 0 for k in ks]
            same = [torch.equal(sd[k].to(base[k].dtype), base[k]) for k in ks]
            per[part] = {"tensors": len(ks), "moved": sum(moved), "bit_identical": sum(same)}
        res[arm] = {"ckpt": str(final(run)), "parts": per}
    frozen_parts_ok = all(res[arm]["parts"][p]["bit_identical"] == res[arm]["parts"][p]["tensors"]
                          for arm in ("F", "U") for p in ("connector", "text_model", "lm_head"))
    fv = res["F"]["parts"]["vision_model"]
    uv = res["U"]["parts"]["vision_model"]
    res["check_ii"] = {"frozen_parts_bit_identical": frozen_parts_ok,
                       "F_vision_bit_identical": fv["bit_identical"] == fv["tensors"],
                       "U_vision_moved_frac": uv["moved"] / uv["tensors"]}
    res["check_ii"]["pass"] = (frozen_parts_ok and res["check_ii"]["F_vision_bit_identical"]
                               and res["check_ii"]["U_vision_moved_frac"] >= 0.90)
    n, diff = 0, []
    with (a.f_run / "steps.jsonl").open() as ff, (a.u_run / "steps.jsonl").open() as fu:
        for lf, lu in zip(ff, fu, strict=True):
            jf, ju = json.loads(lf), json.loads(lu)
            n += 1
            if jf["step"] != ju["step"] or jf["index"] != ju["index"]:
                diff.append(jf["step"])
    res["check_iii"] = {"steps_compared": n, "steps_differing": len(diff), "first_diff": diff[:5], "pass": not diff}
    res["pass"] = res["check_ii"]["pass"] and res["check_iii"]["pass"]
    a.out.parent.mkdir(parents=True, exist_ok=True)
    a.out.write_text(json.dumps(res, indent=1))
    print("ARMS_POST=" + json.dumps({k: res[k] for k in ("check_ii", "check_iii", "pass")}))


def load_ref(ref: str) -> dict:
    """model.safetensors of the init the run started from: a local pretrained_model dir (or a checkpoint dir holding one)
    or a Hub id (default lerobot/smolvla_base)."""
    from safetensors.torch import load_file
    d = Path(ref)
    if d.is_dir():
        d = d / "pretrained_model" if (d / "pretrained_model").is_dir() else d
        return load_file(str(d / "model.safetensors"))
    from huggingface_hub import hf_hub_download
    return load_file(hf_hub_download(ref, "model.safetensors"))


def tensor_stats(base: dict, sd: dict, ks: list[str]) -> dict:
    import torch
    assert ks and all(k in sd for k in ks), len(ks)
    moved = [float((sd[k].float() - base[k].float()).abs().max()) > 0 for k in ks]
    same = [torch.equal(sd[k].to(base[k].dtype), base[k]) for k in ks]
    return {"tensors": len(ks), "moved": sum(moved), "bit_identical": sum(same)}


def check_ii(base: dict, sd: dict, expect: str = "vision") -> dict:
    """Instrument (ii): the final checkpoint `sd` against the init `base`. expect=vision (arm U): connector / text model /
    lm_head bit-identical, vision_model moved in >= 90 % of its tensors. expect=vision+text (arm VT, LeRobot's own
    freezing of lm_head, text_model.norm.weight and the last text layer): vision_model and the text_model tensors LeRobot
    leaves trainable moved in >= 90 % of their tensors, every connector tensor moved, the LeRobot-frozen tensors
    bit-identical to the init (both cast to the init dtype)."""
    per = {part: tensor_stats(base, sd, [k for k in base if key in k]) for part, key in PARTS.items()}
    frac = lambda x: x["moved"] / x["tensors"]  # noqa: E731
    if expect == "vision":
        ii = {"parts": per, "frozen_parts_bit_identical": all(per[p]["bit_identical"] == per[p]["tensors"]
                                                             for p in ("connector", "text_model", "lm_head")),
              "vision_moved_frac": frac(per["vision_model"])}
        ii["pass"] = ii["frozen_parts_bit_identical"] and ii["vision_moved_frac"] >= 0.90
        return ii
    assert expect == "vision+text", expect
    text = [k for k in base if PARTS["text_model"] in k]
    last = max(int(k.split(".text_model.layers.")[1].split(".")[0]) for k in text if ".text_model.layers." in k)
    frozen_text = [k for k in text if ".text_model.norm.weight" in k or f".text_model.layers.{last}." in k]
    train_text = [k for k in text if k not in frozen_text]
    parts = {"vision_model": per["vision_model"], "connector": per["connector"],
             "text_trainable": tensor_stats(base, sd, train_text), "text_frozen_by_lerobot": tensor_stats(base, sd, frozen_text),
             "lm_head": per["lm_head"]}
    frozen = [parts[p] for p in ("text_frozen_by_lerobot", "lm_head")]
    ii = {"expect_trainable": expect, "parts": parts, "text_last_layer_frozen": last,
          "frozen_parts_bit_identical": all(x["bit_identical"] == x["tensors"] for x in frozen),
          "vision_moved_frac": frac(parts["vision_model"]), "text_trainable_moved_frac": frac(parts["text_trainable"]),
          "connector_all_moved": parts["connector"]["moved"] == parts["connector"]["tensors"]}
    ii["pass"] = (ii["frozen_parts_bit_identical"] and ii["vision_moved_frac"] >= 0.90
                  and ii["text_trainable_moved_frac"] >= 0.90 and ii["connector_all_moved"])
    return ii


def a1post(a: argparse.Namespace) -> None:
    """GATES Stage 1 attempt 1 instrument on the U run (--u-run) vs attempt-0 U (--f-run = the reference run):
    (ii) final checkpoint vs smolvla_base: connector / text model / lm_head bit-identical, vision_model moved in >= 90 %
    of its tensors; (iii) steps.jsonl row indices equal the reference's at every step; (iv) for 3 logged steps (first,
    middle, last) every batch row's action chunk, read through the trainer's own dataset path (open_dataset, chunk 50),
    equals r = s1_relabel.relabel(HDF5 action, action_exec) on the chunk's rows (clamped at the episode end as LeRobot
    does); run_key.action_override absent / null in every checkpoint's training_state.pt."""
    import h5py
    import numpy as np
    import torch
    from safetensors.torch import load_file

    from s1_relabel import relabel
    from vla_policy import train_smolvla as T

    base = load_ref(a.ref_init)
    last = (a.u_run / "checkpoints" / "last.txt").read_text().strip()
    sd = load_file(str(a.u_run / "checkpoints" / last / "pretrained_model" / "model.safetensors"))
    ii = check_ii(base, sd, a.expect_trainable)
    n, diff, logged = 0, [], {}
    d = T.resolve_dataset(a.dataset, None)
    ds = T.open_dataset(d, "torchcodec")
    ep = T.column(ds, "episode_index").reshape(-1)
    if a.sampler_epochs:  # Stage 1 redesign (iii): row indices = WindowSampler(seed 0).order(epoch), epochs 1..E
        is_val, _, _ = T.split_episodes(ds)
        smp = T.WindowSampler(ep, T.column(ds, "frame_idx"), a.sampler_seed, keep=~is_val)
        spe = -(-len(smp) // 64)
        orders = {e: smp.order(e) for e in range(a.sampler_epochs)}
        epochs_seen = set()
        with (a.u_run / "steps.jsonl").open() as fu:
            for lu in fu:
                ju = json.loads(lu)
                n += 1
                e, k = (ju["step"] - 1) // spe, (ju["step"] - 1) % spe
                epochs_seen.add(e)
                if e not in orders or ju["index"] != orders[e][64 * k:64 * k + 64].tolist():
                    diff.append(ju["step"])
                if ju["step"] in a.check_steps:
                    logged[ju["step"]] = ju["index"]
        iii = {"mode": f"sampler = f(seed {a.sampler_seed}, epoch)", "epochs_checked": len(epochs_seen), "steps_per_epoch": spe,
               "steps_compared": n, "steps_differing": len(diff), "first_diff": diff[:5],
               "pass": not diff and n == spe * a.sampler_epochs and len(epochs_seen) == a.sampler_epochs}
    else:
        with (a.f_run / "steps.jsonl").open() as fr, (a.u_run / "steps.jsonl").open() as fu:
            for lr, lu in zip(fr, fu, strict=True):
                jr, ju = json.loads(lr), json.loads(lu)
                n += 1
                if jr["step"] != ju["step"] or jr["index"] != ju["index"]:
                    diff.append(ju["step"])
                if ju["step"] in a.check_steps:
                    logged[ju["step"]] = ju["index"]
        iii = {"steps_compared": n, "steps_differing": len(diff), "first_diff": diff[:5], "pass": not diff and n > 0}
    side = {s["lerobot_episode_index"]: s for s in
            (json.loads(x) for x in (d / "meta" / "episodes_113.jsonl").read_text().splitlines())}
    cache: dict = {}
    iv = {"steps": {}, "rows_checked": 0, "rows_bad": 0}
    for st in a.check_steps:
        bad = 0
        for i in logged[st]:
            e = int(ep[i])
            if e not in cache:
                s = side[e]
                with h5py.File(s["source"], "r") as h:
                    g = h[s["group"]]
                    r = relabel(g["action"][()].astype(np.float32), g["action_exec"][()].astype(np.float32))
                cache[e] = (r, int(np.flatnonzero(ep == e)[0]))
            r, lo = cache[e]
            t = i - lo
            want = r[np.minimum(t + np.arange(T.CHUNK), len(r) - 1)]
            got = ds[i]["action"].numpy().astype(np.float32)
            bad += not np.array_equal(got, want)
        iv["steps"][st] = {"rows": len(logged[st]), "rows_bad": bad}
        iv["rows_checked"] += len(logged[st])
        iv["rows_bad"] += bad
    iv["pass"] = iv["rows_bad"] == 0 and len(iv["steps"]) == len(a.check_steps)
    rk = {}
    for c in sorted((a.u_run / "checkpoints").glob("0*")):
        ts = torch.load(c / "training_state.pt", map_location="cpu", weights_only=False)
        rk[c.name] = {"action_override": ts["run_key"].get("action_override"), "dataset": ts["run_key"]["dataset"]}
    rko = {"per_ckpt": rk, "pass": bool(rk) and all(v["action_override"] is None for v in rk.values())}
    res = {"u_run": str(a.u_run), "ref_run": str(a.f_run), "final_ckpt": last, "check_ii": ii, "check_iii": iii,
           "check_iv": iv, "run_key_action_override_null": rko, "git_sha": T.git_sha()}
    if a.ref_init != "lerobot/smolvla_base":
        res["ref_init"] = a.ref_init
    res["pass"] = ii["pass"] and iii["pass"] and iv["pass"] and rko["pass"]
    a.out.parent.mkdir(parents=True, exist_ok=True)
    a.out.write_text(json.dumps(res, indent=1))
    print("A1POST=" + json.dumps({k: (v["pass"] if isinstance(v, dict) and "pass" in v else v)
                                  for k, v in res.items()}))


def load_sidecar(root: Path) -> dict[int, dict]:
    """LeRobot episode index -> its meta/episodes_113.jsonl row (instruction, spec_json, ...)."""
    rows = [json.loads(x) for x in (Path(root) / "meta" / "episodes_113.jsonl").read_text().splitlines()]
    return {r.get("lerobot_episode_index", i): r for i, r in enumerate(rows)}


def paired_bootstrap(d, groups=None, n: int = 10_000, seed: int = 0) -> list[float]:
    """95 % CI of mean(d): resample samples (groups=None) or whole groups (episodes) with replacement."""
    import numpy as np

    d, rng = np.asarray(d, np.float64), np.random.default_rng(seed)
    if groups is not None:
        u, inv = np.unique(groups, return_inverse=True)
        s, c = np.bincount(inv, d, len(u)), np.bincount(inv, minlength=len(u)).astype(np.float64)
        d, w = s, c
    else:
        w = np.ones(len(d))
    m = np.empty(n)
    for k in range(0, n, 500):
        ix = rng.integers(0, len(d), (min(500, n - k), len(d)))
        m[k:k + len(ix)] = d[ix].sum(1) / w[ix].sum(1)
    return [float(x) for x in np.percentile(m, [2.5, 97.5])]


def sample_losses(pol, batch: dict):
    """Per-sample flow-matching loss: the mean over (chunk, action dims) of pol.forward's masked loss tensor, so
    the batch mean equals pol.forward's scalar. Same noise / time draws as pol.forward for the same torch seed."""
    from lerobot.utils.constants import OBS_LANGUAGE_ATTENTION_MASK, OBS_LANGUAGE_TOKENS

    images, masks = pol.prepare_images(batch)
    ls = pol.model.forward(images, masks, batch[OBS_LANGUAGE_TOKENS], batch[OBS_LANGUAGE_ATTENTION_MASK],
                           pol.prepare_state(batch), pol.prepare_action(batch), None, None)
    ls = ls[:, :, :pol.config.action_feature.shape[0]] * (~batch["action_is_pad"])[..., None]
    return ls.mean((1, 2))


def vloss(a: argparse.Namespace) -> None:
    import numpy as np
    import torch

    from vla_common.swap import donors, swap_report
    from vla_policy import train_smolvla as T
    from vla_policy.server import SmolVLAServer

    ds = T.open_dataset(T.resolve_dataset(a.dataset, None), "torchcodec")
    is_val, _, val_eps = T.split_episodes(ds)
    rows = T.WindowSampler(T.column(ds, "episode_index"), T.column(ds, "frame_idx"), 0, keep=is_val).order(0)
    srv = SmolVLAServer(a.ckpt, 20, a.device)
    ovr = T.ActionOverride(a.action_override, ds) if a.action_override else None
    sub = torch.utils.data.Subset(ds, rows.tolist())
    dl = torch.utils.data.DataLoader(sub, batch_size=64, shuffle=False, num_workers=8)
    swap = a.task_swap
    modes = ["none", "shuffle", "blank"] if swap == "compare" else [swap]
    new_task, srep = {}, None
    if "shuffle" in modes:
        side = load_sidecar(ds.root)
        instr = [side[e]["instruction"] for e in val_eps]
        spec = [json.loads(side[e]["spec_json"]) for e in val_eps]
        d = donors(instr, [s["target"] for s in spec])
        new_task = {e: instr[d[k]] for k, e in enumerate(val_eps)}
        srep = swap_report(instr, [s["target"] for s in spec], d)
        diff_t = {e: tuple(spec[k]["target"]) != tuple(spec[d[k]]["target"]) for k, e in enumerate(val_eps)}
    per = a.per_sample or swap != "none"
    tot, cnt = {m: 0.0 for m in modes}, 0
    cols: dict[str, list] = {"index": [], "episode_index": [], "frame_index": [], **{m: [] for m in modes}}
    with torch.no_grad():
        for b, batch in enumerate(dl):
            k = len(batch["index"])
            batch = T.to_float_images(batch if ovr is None else ovr(batch))
            ep = batch["episode_index"].reshape(-1).tolist()
            if new_task:
                assert list(batch["task"]) == [instr[val_eps.index(e)] for e in ep], "dataset task != sidecar instruction"
            for m in modes:
                task = {"none": list(batch["task"]), "blank": [""] * k, "shuffle": [new_task.get(e) for e in ep]}[m]
                pb = srv.pre({**batch, "task": task})  # the server's / trainer's path: `task` strings -> tokenizer step
                torch.manual_seed(1000 + b)
                if per:
                    ps = sample_losses(srv.policy, pb)
                    tot[m] += float(ps.sum())
                    cols[m] += ps.tolist()
                else:
                    loss, _ = srv.policy.forward(pb)
                    tot[m] += float(loss) * k
            for c, key in (("index", "index"), ("episode_index", "episode_index"), ("frame_index", "frame_index")):
                cols[c] += batch[key].reshape(-1).tolist()
            cnt += k
    res = {"ckpt": a.ckpt, "ckpt_sha": srv.ckpt_sha, "val_episodes": len(val_eps), "samples": cnt,
           "action_override": str(a.action_override) if a.action_override else None,
           "task_swap": swap, "val_loss": tot[modes[0]] / cnt, "dtype": "float32 (served)", "git_sha": T.git_sha()}
    if swap == "blank" or swap == "compare":
        res["blank_task"] = ""  # tokenizer path accepts it: the NewLine step makes it one token (198)
    if srep is not None:
        res["shuffle_donors"] = srep
        res["shuffle_frac_samples_different_target"] = float(np.mean([diff_t[e] for e in cols["episode_index"]]))
    if swap == "compare":
        res.pop("val_loss")
        res["loss"] = {m: tot[m] / cnt for m in modes}
        fi, epi = np.asarray(cols["frame_index"]), np.asarray(cols["episode_index"])
        for name, sel in (("all", np.ones(cnt, bool)), (f"approach_t<={a.approach_steps}", fi <= a.approach_steps)):
            L = {m: np.asarray(cols[m])[sel] for m in modes}
            blk = {"samples": int(sel.sum()), "loss": {m: float(L[m].mean()) for m in modes}}
            for m in ("shuffle", "blank"):
                dd = L[m] - L["none"]
                blk[f"{m}_minus_none"] = {"mean": float(dd.mean()), "ci95_sample_bootstrap": paired_bootstrap(dd),
                                          "ci95_episode_bootstrap": paired_bootstrap(dd, epi[sel])}
            res["all" if name == "all" else "approach"] = blk
        res["approach_definition"] = f"frame_index (control step of the sampled row) <= {a.approach_steps}"
        assert np.isfinite(list(res["loss"].values())).all()
    else:
        assert np.isfinite(res["val_loss"])
    if a.per_sample:
        res["per_sample"] = cols
    a.out.parent.mkdir(parents=True, exist_ok=True)
    a.out.write_text(json.dumps(res, indent=1))
    print("VLOSS=" + json.dumps({k: v for k, v in res.items() if k != "per_sample"}))


def vdetail(a: argparse.Namespace) -> None:
    """Diagnosis (validation episodes only): the `vloss` batches and seeds, per chunk position (50) and per action
    dim (5) mean flow-matching loss, and the same with the noise / time draws repeated K = 8 times per row."""
    import torch

    from vla_policy import train_smolvla as T
    from vla_policy.server import SmolVLAServer

    ds = T.open_dataset(T.resolve_dataset(a.dataset, None), "torchcodec")
    is_val, _, _ = T.split_episodes(ds)
    rows = T.WindowSampler(T.column(ds, "episode_index"), T.column(ds, "frame_idx"), 0, keep=is_val).order(0)
    srv = SmolVLAServer(a.ckpt, 20, a.device)
    pol = srv.policy
    dl = torch.utils.data.DataLoader(torch.utils.data.Subset(ds, rows.tolist()), batch_size=64, shuffle=False,
                                     num_workers=8)
    s = torch.zeros(50, 5, dtype=torch.float64)
    n = torch.zeros(50, 1, dtype=torch.float64)
    with torch.no_grad():
        for b, batch in enumerate(dl):
            batch = srv.pre(T.to_float_images(batch))
            images, masks = pol.prepare_images(batch)
            state, actions = pol.prepare_state(batch), pol.prepare_action(batch)
            tok, tmask = batch["observation.language.tokens"], batch["observation.language.attention_mask"]
            valid = (~batch["action_is_pad"]).double().cpu()
            for k in range(8):
                torch.manual_seed(1000 + 100 * b + k)
                ls = pol.model.forward(images, masks, tok, tmask, state, actions, None, None)[:, :, :5].double().cpu()
                s += (ls * valid[..., None]).sum(0)
                n += valid.sum(0)[:, None]
    per = (s / n)
    res = {"ckpt": a.ckpt, "draws_per_row": 8, "loss_per_pos": per.mean(1).tolist(), "loss_per_dim": (s.sum(0) / n.sum()).tolist(),
           "loss_mean": float(s.sum() / (n.sum() * 5)), "git_sha": T.git_sha()}
    a.out.parent.mkdir(parents=True, exist_ok=True)
    a.out.write_text(json.dumps(res, indent=1))
    print("VDETAIL=" + json.dumps({k: res[k] for k in ("loss_mean", "loss_per_dim")}) + " pos0..49 "
          + " ".join(f"{x:.3f}" for x in res["loss_per_pos"]))


def taskdump(a: argparse.Namespace) -> None:
    """CPU, no training step: the trainer's first batch (WindowSampler seed 0 on the train split, batch 64, the same
    dataset object) through the trainer's preprocessor (train_smolvla.build_policy's overrides, up to and including the
    tokenizer step -- the whole `pre` is run, as train_step does). Asserts task == sidecar instruction on all 64 rows."""
    import numpy as np
    import torch
    from lerobot.configs import FeatureType, PolicyFeature, PreTrainedConfig
    from lerobot.policies import make_pre_post_processors
    from lerobot.processor import TokenizerProcessorStep
    from lerobot.utils.constants import OBS_LANGUAGE_ATTENTION_MASK, OBS_LANGUAGE_TOKENS

    from vla_policy import train_smolvla as T

    ds = T.open_dataset(T.resolve_dataset(a.dataset, None), a.video_backend)
    is_val, _, _ = T.split_episodes(ds)
    stats, _ = T.norm_stats(ds, ~is_val)
    sampler = T.WindowSampler(T.column(ds, "episode_index"), T.column(ds, "frame_idx"), 0, keep=~is_val)
    dl = torch.utils.data.DataLoader(ds, batch_size=64, sampler=sampler, num_workers=0, drop_last=False,
                                     generator=torch.Generator().manual_seed(0))
    cfg = PreTrainedConfig.from_pretrained(a.ckpt)
    f = ds.meta.features
    img = lambda k: PolicyFeature(type=FeatureType.VISUAL, shape=(3, *f[k]["shape"][:2]))  # noqa: E731
    cfg.input_features = {T.RENAME_MAP[k]: img(k) for k in T.RENAME_MAP}
    cfg.input_features["observation.state"] = PolicyFeature(type=FeatureType.STATE, shape=tuple(f["observation.state"]["shape"]))
    cfg.output_features = {"action": PolicyFeature(type=FeatureType.ACTION, shape=tuple(f["action"]["shape"]))}
    feats = {**cfg.input_features, **cfg.output_features}
    pre, _ = make_pre_post_processors(
        cfg, pretrained_path=a.ckpt,
        preprocessor_overrides={"device_processor": {"device": "cpu"},
                                "rename_observations_processor": {"rename_map": T.RENAME_MAP},
                                "normalizer_processor": {"features": feats, "norm_map": cfg.normalization_mapping, "stats": stats}})
    tokz = next(s for s in pre.steps if isinstance(s, TokenizerProcessorStep)).input_tokenizer
    batch = next(iter(dl))
    side = load_sidecar(ds.root)
    ep = batch["episode_index"].reshape(-1).tolist()
    task = list(batch["task"])
    want = [side[e]["instruction"] for e in ep]
    assert len(task) == 64 and task == want, [(e, t, w) for e, t, w in zip(ep, task, want) if t != w][:3]
    out = pre(T.to_float_images(batch))
    tok, msk = out[OBS_LANGUAGE_TOKENS], out[OBS_LANGUAGE_ATTENTION_MASK]
    rows = []
    for k in range(a.n):
        e = ep[k]
        rows.append({"row": int(batch["index"][k]), "episode": e, "task": task[k],
                     "decoded_tokens": tokz.decode(tok[k][msk[k].bool()].tolist()),
                     "sidecar_instruction": side[e]["instruction"],
                     "sidecar_target": json.loads(side[e]["spec_json"])["target"]})
        print(json.dumps(rows[-1]))
    res = {"dataset": str(ds.root), "batch_rows": len(task), "task_equals_sidecar_all_rows": True,
           "distinct_tasks_in_batch": len(set(task)), "tokens_shape": list(tok.shape), "rows": rows,
           "task_index_distinct_in_batch": len(set(batch["task_index"].reshape(-1).tolist())), "git_sha": T.git_sha()}
    a.out.parent.mkdir(parents=True, exist_ok=True)
    a.out.write_text(json.dumps(res, indent=1))
    print(f"TASKDUMP task==sidecar on all 64 rows; distinct tasks {res['distinct_tasks_in_batch']}; tokens {res['tokens_shape']}")


def read_cols(d: Path) -> dict:
    """episode_index, frame_idx, split per dataset row (position = `index`), parquet only (no videos)."""
    import numpy as np
    import pyarrow.parquet as pq

    t = [pq.read_table(f, columns=["index", "episode_index", "frame_idx", "split"]) for f in sorted((d / "data").rglob("*.parquet"))]
    c = {k: np.concatenate([np.asarray(x.column(k).to_pylist()) for x in t]).reshape(-1) for k in ("index", "episode_index", "frame_idx", "split")}
    o = np.argsort(c["index"], kind="stable")
    assert np.array_equal(c["index"][o], np.arange(len(o))), "dataset `index` is not 0..n-1"
    return {k: v[o] for k, v in c.items()}


def mix_check(steps: list[dict], ep, fi, is_val, is_pick, epochs: int, expect_main: int, expect_pick: int,
              batch: int = 64, restrict=None) -> dict:
    """Pure. steps: steps.jsonl rows (step, index[, epoch]); per row ep / fi / is_val / is_pick. Epoch of step s =
    (s - 1) // steps_per_epoch (the trainer's). `restrict` (bool per row): the mutant keeps only these drawn rows."""
    import math

    import numpy as np

    key = ep.astype(np.int64) * (int(fi.max()) + 1) + fi.astype(np.int64)
    train = ~is_val
    win = np.unique(key[train])
    spe = math.ceil(len(win) / batch)
    ds_main, ds_pick = (len(np.unique(key[train & m])) for m in (~is_pick, is_pick))
    drawn: dict[int, list] = {e: [] for e in range(epochs)}
    nsteps = {e: 0 for e in range(epochs)}
    epoch_field_bad = 0
    for r in steps:
        e = (r["step"] - 1) // spe
        epoch_field_bad += "epoch" in r and r["epoch"] != e
        idx = np.asarray(r["index"], np.int64)
        if restrict is not None:
            idx = idx[restrict[idx]]
        if e in drawn:
            drawn[e].append(idx)
            nsteps[e] += 1
    per = {}
    for e in range(epochs):
        rows = np.concatenate(drawn[e]) if drawn[e] else np.zeros(0, np.int64)
        w = key[rows]
        per[e] = {"steps": nsteps[e], "rows": len(rows), "val_rows": int(is_val[rows].sum()),
                  "distinct_windows": len(np.unique(w)), "duplicate_windows": len(w) - len(np.unique(w)),
                  "missing_windows": len(win) - int(np.isin(win, w).sum()),
                  "main_windows": int((~is_pick[rows]).sum()), "pick_windows": int(is_pick[rows].sum())}
        per[e]["pass"] = (per[e]["steps"] == spe and per[e]["rows"] == len(win) and per[e]["val_rows"] == 0
                          and per[e]["duplicate_windows"] == 0 and per[e]["missing_windows"] == 0
                          and per[e]["main_windows"] == expect_main and per[e]["pick_windows"] == expect_pick)
    res = {"epochs": epochs, "steps_per_epoch": spe, "train_windows": len(win), "dataset_windows_main_pick": [ds_main, ds_pick],
           "expect_main_pick": [expect_main, expect_pick], "steps_logged": len(steps), "epoch_field_mismatch": epoch_field_bad,
           "steps_beyond_epochs": len(steps) - sum(nsteps.values()) if restrict is None else None, "per_epoch": per}
    res["pass"] = (all(v["pass"] for v in per.values()) and [ds_main, ds_pick] == [expect_main, expect_pick]
                   and epoch_field_bad == 0 and len(steps) == spe * epochs)
    return res


def mix(a: argparse.Namespace) -> None:
    import numpy as np

    d = (Path(a.dataset) if Path(a.dataset).is_dir() else ROOT / "datasets" / a.dataset).resolve()
    c = read_cols(d)
    side = [json.loads(x) for x in (d / "meta" / "episodes_113.jsonl").read_text().splitlines()]
    pick_ep = {s["lerobot_episode_index"] for s in side if s["task"] == "pick"}
    ep = c["episode_index"].astype(np.int64)
    is_pick = np.isin(ep, list(pick_ep))
    is_val = c["split"].astype(str) == "val"
    restrict = None
    if a.mutant:  # a sampler that reads only the first stream (stage, seed of lerobot episode 0)
        k0 = (side[0]["file_attrs"]["stage"], int(side[0]["seed"]))
        restrict = np.isin(ep, [s["lerobot_episode_index"] for s in side if (s["file_attrs"]["stage"], int(s["seed"])) == k0])
    steps = [json.loads(x) for x in (a.u_run / "steps.jsonl").read_text().splitlines()]
    res = mix_check(steps, ep, c["frame_idx"], is_val, is_pick, a.epochs, a.expect_main, a.expect_pick, a.batch_size, restrict)
    res |= {"dataset": str(d), "run": str(a.u_run), "mutant": a.mutant or None,
            "pick_episodes": len(pick_ep), "episodes": len(side)}
    a.out.parent.mkdir(parents=True, exist_ok=True)
    a.out.write_text(json.dumps(res, indent=1))
    print("MIX=" + json.dumps({k: res[k] for k in ("pass", "mutant", "steps_per_epoch", "dataset_windows_main_pick")}
                              | {"per_epoch": {e: [v["main_windows"], v["pick_windows"], v["val_rows"], v["pass"]] for e, v in res["per_epoch"].items()}}))
    if not res["pass"]:
        sys.exit(1)


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("mode", choices=("arms", "post", "a1post", "vloss", "vdetail", "mix", "taskdump"))
    p.add_argument("--check-steps", type=int, nargs="+", default=[1, 10332, 20664], help="a1post: (iv) logged steps")
    p.add_argument("--expect-trainable", choices=("vision", "vision+text"), default="vision",
                   help="a1post (ii): vision = arm U (text bit-identical); vision+text = arm VT (LeRobot freezes lm_head, "
                        "text_model.norm, the last text layer; the rest of the text model must have moved)")
    p.add_argument("--ref-init", default="lerobot/smolvla_base", help="a1post (ii): Hub id or checkpoint dir the run started from")
    p.add_argument("--sampler-seed", type=int, default=0, help="a1post (iii): 0 = the training seed; 1 = the mutant")
    p.add_argument("--sampler-epochs", type=int, default=0, help="a1post: (iii) against f(seed, epoch), not --f-run")
    p.add_argument("--batch-size", type=int, default=64, help="mix: the run's batch size")
    p.add_argument("--epochs", type=int, default=4, help="mix: epochs expected in steps.jsonl")
    p.add_argument("--expect-main", type=int, default=347_700, help="mix: place / stack windows per epoch")
    p.add_argument("--expect-pick", type=int, default=220_400, help="mix: pick windows per epoch")
    p.add_argument("--mutant", choices=("first-dataset-only",), default=None, help="mix: simulate a sampler over the first stream only")
    p.add_argument("--dataset", default="/data/datasets/0c_s1_pick")
    p.add_argument("--ckpt", default="")
    p.add_argument("--f-run", type=Path, default=Path("/runs/s1_frozen"))
    p.add_argument("--u-run", type=Path, default=Path("/runs/s1_unfrozen"))
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--device", default="cuda")
    p.add_argument("--action-override", type=Path, default=None, help="vloss: npz of relabelled actions")
    p.add_argument("--task-swap", choices=("none", "shuffle", "blank", "compare"), default="none",
                   help="vloss: instruction given to the model (none = the dataset's; shuffle = another val episode's, "
                        "different target where possible; blank = ''; compare = all three on the same samples / noise)")
    p.add_argument("--n", type=int, default=10, help="taskdump: rows printed")
    p.add_argument("--video-backend", default="pyav", help="taskdump: the trainer's default")
    p.add_argument("--per-sample", action="store_true", help="vloss: also write the per-sample losses")
    p.add_argument("--approach-steps", type=int, default=30, help="vloss compare: approach chunks = frame_index <= this")
    a = p.parse_args()
    print(f"[s1_tools] {a.mode} device={a.device}", flush=True)
    {"arms": arms, "post": post, "a1post": a1post, "vloss": vloss, "vdetail": vdetail, "mix": mix, "taskdump": taskdump}[a.mode](a)


if __name__ == "__main__":
    main()
