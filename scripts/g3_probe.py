"""Task 003 (GATES "Post-kill grounding diagnosis (003)"), policy venv. Reported only, not an attempt.

  extract (GPU): one checkpoint, the dumps of scripts/g3_dump_t0.py (sample list = S_p then S_u, as given) ->
      <out>_feats.npz  per condition (true / shuffled) and layer L in 4 / 8 / 12 / 16: lang, img, kv, kv_img (N, d)
      <out>_div.json   step 2 action divergence on the first 100 layout groups of the paired dump
  divonly --guidance-w w (task 004, GATES "Inference-time instruction-guidance probe (004)", reported only): step 2
      with every chunk from vla_policy.guidance's guided sampler at weight w (each row's others = its instruction with
      the colour word replaced by each other colour of the layout); <out>_div.json carries guidance_w.
  probe (CPU): the feats files -> 5-fold group-CV logistic-regression accuracies, readings and the verdict rule.

  python scripts/g3_probe.py extract --ckpt /runs/s2a1_VT/checkpoints/034440/pretrained_model \
      --dumps /runs/g3_diag/t0_p21645.npz /runs/g3_diag/t0_u21640.npz --out /runs/g3_diag/c034440
  python scripts/g3_probe.py probe --feats results/g3_diag/c034440_feats.npz --div results/g3_diag/c034440_div.json \
      --out results/g3_diag/probe_034440.json

Features (GATES 003): the checkpoint loaded exactly as vla_policy.server.SmolVLAServer (fp32, TF32 off, its saved
processors); prefix = embed_prefix; prefill = vlm_with_expert.forward(inputs_embeds=[prefix, None], use_cache=True).
Layer L = the output of text layer L - 1 (residual stream before the final norm: the input of layer L's
input_layernorm, of the final norm for L = 16). lang = mean over language tokens with mask 1; img = mean over both
cameras' image tokens; kv = the expert's k_proj / v_proj of layer L - 1 (a cross-attention layer) applied to the cached
VLM K / V of that layer, mean over valid prefix positions, K | V; kv_img = the same over image positions only.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

LAYERS = (4, 8, 12, 16)
FEATS = ("lang", "img", "kv", "kv_img")
CONDS = ("true", "shuffled")
N_DIV_LAYOUTS = 100
DEC_ACC, DEC_GAP, NOT_GAP = 0.90, 0.40, 0.10  # task 003 readings, verbatim
EPS = 1e-9  # accuracies are k / 2000: a gap of exactly 0.10 / 0.40 reads as written, not by float rounding


# --------------------------------------------------------------------------- pure helpers (tested on CPU)
def load_samples(paths: list[Path]) -> dict:
    """Concatenate g3_dump_t0 npz files in the order given (the GATES sample list)."""
    ds = [dict(np.load(p, allow_pickle=False)) for p in paths]
    keys = [k for k in ds[0] if np.ndim(ds[0][k]) > 0]
    out = {k: np.concatenate([d[k] for d in ds]) for k in keys}
    out["subset"] = np.concatenate([np.full(len(d["index"]), "p" if bool(d["paired"]) else "u") for d in ds])
    return out


def shuffled_instructions(instr: np.ndarray) -> np.ndarray:
    """GATES 003: the instruction of sample pi(i), pi = default_rng(0).permutation(N)."""
    return instr[np.random.default_rng(0).permutation(len(instr))]


def group_folds(groups: np.ndarray, k: int = 5) -> np.ndarray:
    """Fold of each sample: unique groups sorted by id, shuffled by default_rng(0), assigned round-robin."""
    ug = np.unique(groups)
    rank = {g: r for r, g in enumerate(ug[np.random.default_rng(0).permutation(len(ug))].tolist())}
    return np.array([rank[g] % k for g in groups.tolist()], np.int64)


def fit_logreg(X: torch.Tensor, y: torch.Tensor, n_classes: int, C: float = 1.0, iters: int = 300):
    """Multinomial logistic regression, sklearn's objective / C (mean CE + ||W||^2 / (2 C n)), L-BFGS."""
    n, d = X.shape
    W = torch.zeros(d, n_classes, dtype=torch.float64, requires_grad=True)
    b = torch.zeros(n_classes, dtype=torch.float64, requires_grad=True)
    opt = torch.optim.LBFGS([W, b], lr=1.0, max_iter=iters, tolerance_grad=1e-7, tolerance_change=1e-10,
                            history_size=20, line_search_fn="strong_wolfe")

    def closure():
        opt.zero_grad()
        loss = torch.nn.functional.cross_entropy(X @ W + b, y) + (W ** 2).sum() / (2 * C * n)
        loss.backward()
        return loss

    opt.step(closure)
    return W.detach(), b.detach()


def cv_predict(X: np.ndarray, y: np.ndarray, folds: np.ndarray, n_classes: int = 4) -> np.ndarray:
    """Out-of-fold predictions; features standardised on each training fold."""
    pred = np.full(len(y), -1, np.int64)
    for f in np.unique(folds):
        tr, te = folds != f, folds == f
        Xt = torch.from_numpy(X[tr].astype(np.float64))
        mu, sd = Xt.mean(0), Xt.std(0).clamp_min(1e-6)
        W, b = fit_logreg((Xt - mu) / sd, torch.from_numpy(y[tr]), n_classes)
        Xe = (torch.from_numpy(X[te].astype(np.float64)) - mu) / sd
        pred[te] = (Xe @ W + b).argmax(1).numpy()
    assert (pred >= 0).all()
    return pred


def reading(acc: float, acc_shuf: float) -> str:
    gap = acc - acc_shuf
    if acc >= DEC_ACC - EPS and gap >= DEC_GAP - EPS:
        return "decodable"
    if gap <= NOT_GAP + EPS:
        return "not decodable"
    return "partial"


def family(readings: list[str]) -> str:
    """Over the 4 layers: decodable at >= 1 -> decodable; not decodable at all -> not decodable; else partial."""
    if "decodable" in readings:
        return "decodable"
    return "not decodable" if all(r == "not decodable" for r in readings) else "partial"


def verdict(lang: str, kv: str, ignores_language: bool) -> str:
    """GATES 003 verdict rule, per checkpoint, literal: any partial family (lang or kv) -> mixed, checked first
    (GATES clarification 2026-10-25 ~00:50 UTC, lead review F1)."""
    if "partial" in (lang, kv):
        return "mixed"
    if lang == "not decodable":
        return "R1 text"
    if lang == "decodable" and kv == "not decodable":
        return "R2 mixing"
    if kv == "decodable":
        return "R3 data/loss" if ignores_language else "none of R1-R3 at the first query"
    return "mixed"


def divergence(chunks_instr: list[np.ndarray], noise_pair: list[tuple[np.ndarray, np.ndarray]],
               steps: int | None = None) -> dict:
    """chunks_instr[l] (k_l, T, 4) arm dims, normalized, one start noise; noise_pair[l] = two chunks, one instruction.
    Instruction effect = mean over layouts of the mean pairwise Frobenius distance; noise effect = mean distance."""
    sl = slice(None, steps)
    inst, noi = [], []
    for c, (n0, n1) in zip(chunks_instr, noise_pair):
        c = c[:, sl]
        d = [np.linalg.norm(c[i] - c[j]) for i in range(len(c)) for j in range(i + 1, len(c))]
        inst.append(float(np.mean(d)))
        noi.append(float(np.linalg.norm(n0[sl] - n1[sl])))
    inst_m, noi_m = float(np.mean(inst)), float(np.mean(noi))
    ratio = inst_m / noi_m
    return {"n_layouts": len(inst), "instruction_effect": inst_m, "noise_effect": noi_m, "ratio": ratio,
            "per_layout_ratio_median": float(np.median(np.array(inst) / np.array(noi))),
            "ignores_language": bool(ratio <= 1.5)}


# --------------------------------------------------------------------------- extract (GPU)
class Extractor:
    def __init__(self, ckpt: str, device: str = "cuda"):
        from vla_policy.server import SmolVLAServer

        self.srv = SmolVLAServer(ckpt, n_action_steps=10, device=device)
        self.pol, self.dev = self.srv.policy, device
        self.model = self.pol.model
        self.vwe = self.model.vlm_with_expert
        tm = self.vwe.get_vlm_model().text_model
        assert len(tm.layers) == 16 and len(self.vwe.lm_expert.layers) == 16, (len(tm.layers),
                                                                                len(self.vwe.lm_expert.layers))
        assert self.vwe.attention_mode == "cross_attn" and self.vwe.self_attn_every_n_layers == 2
        self.cap: dict[int, torch.Tensor] = {}
        for L in LAYERS:
            mod = tm.norm if L == 16 else tm.layers[L].input_layernorm
            mod.register_forward_pre_hook(lambda m, inp, L=L: self.cap.__setitem__(L, inp[0].detach()))

    @staticmethod
    def raw(side: np.ndarray, wrist: np.ndarray, state: np.ndarray, instr: list[str]) -> dict:
        """The server's own batch construction (SmolVLAServer.act), before its preprocessor."""
        return {"observation.images.side": torch.from_numpy(side).permute(0, 3, 1, 2).float() / 255.0,
                "observation.images.wrist": torch.from_numpy(wrist).permute(0, 3, 1, 2).float() / 255.0,
                "observation.state": torch.from_numpy(state), "task": list(instr)}

    def obs(self, side: np.ndarray, wrist: np.ndarray, state: np.ndarray, instr: list[str]) -> dict:
        """The server's own batch construction (SmolVLAServer.act) through its saved preprocessor."""
        return self.srv.pre(self.raw(side, wrist, state, instr))

    @torch.no_grad()
    def features(self, obs: dict) -> dict[str, np.ndarray]:
        from lerobot.policies.common.vla_utils import make_att_2d_masks
        from lerobot.utils.constants import OBS_LANGUAGE_ATTENTION_MASK, OBS_LANGUAGE_TOKENS

        images, img_masks = self.pol.prepare_images(obs)
        state = self.pol.prepare_state(obs)
        lt, lm = obs[OBS_LANGUAGE_TOKENS], obs[OBS_LANGUAGE_ATTENTION_MASK].bool()
        embs, pad, att = self.model.embed_prefix(images, img_masks, lt, lm, state=state)
        B, P = pad.shape
        n_lang = lt.shape[1]
        n_img = P - n_lang - 1
        assert len(images) == 2 and n_img % 2 == 0 and n_lang == 48, (len(images), n_img, n_lang)
        self.cap.clear()
        _, kv = self.vwe.forward(attention_mask=make_att_2d_masks(pad, att), position_ids=torch.cumsum(pad, 1) - 1,
                                 past_key_values=None, inputs_embeds=[embs, None], use_cache=True)
        img_m = torch.zeros(B, P, dtype=torch.bool, device=pad.device)
        img_m[:, :n_img] = True
        lang_m = torch.zeros_like(img_m)
        lang_m[:, n_img:n_img + n_lang] = lm
        assert pad[:, :n_img].all() and pad[:, -1].all() and torch.equal(pad[:, n_img:n_img + n_lang], lm)

        def mean(x, m):
            m = m.to(x.dtype)[..., None]
            return ((x * m).sum(1) / m.sum(1)).float().cpu().numpy()

        out = {}
        for L in LAYERS:
            h = self.cap[L]
            assert h.shape[:2] == (B, P), h.shape
            out[f"lang_L{L}"], out[f"img_L{L}"] = mean(h, lang_m), mean(h, img_m)
            li = L - 1
            el = self.vwe.lm_expert.layers[li].self_attn
            K = kv.layers[li].keys.transpose(1, 2).reshape(B, P, -1)
            V = kv.layers[li].values.transpose(1, 2).reshape(B, P, -1)
            assert li % 2 == 1 and el.k_proj.in_features == K.shape[-1], (li, el.k_proj.in_features, K.shape)
            ekv = torch.cat([el.k_proj(K.to(el.k_proj.weight.dtype)), el.v_proj(V.to(el.v_proj.weight.dtype))], -1)
            out[f"kv_L{L}"], out[f"kv_img_L{L}"] = mean(ekv, pad), mean(ekv, img_m)
        return out

    @torch.no_grad()
    def chunk(self, obs: dict, noise: torch.Tensor) -> np.ndarray:
        """Normalized chunk (B, 50, 5), as SmolVLAServer.act before un-normalising."""
        return self.pol.predict_action_chunk(dict(obs), noise=noise.to(self.dev)).float().cpu().numpy()


def extract(a) -> None:
    S = load_samples(a.dumps)
    N = len(S["index"])
    instr = {"true": S["instruction"], "shuffled": shuffled_instructions(S["instruction"])}
    t0 = time.time()
    ex = Extractor(a.ckpt)
    print(f"[g3] {a.ckpt}: {N} samples, model up {time.time() - t0:.0f} s, sha {ex.srv.ckpt_sha[:12]}", flush=True)
    if a.cmd == "divonly":
        div_step(a, ex, S, t0, None)
        return
    feats: dict[str, list] = {}
    tokens_example = None
    for cond in CONDS:
        for b0 in range(0, N, a.batch):
            sl = slice(b0, b0 + a.batch)
            obs = ex.obs(S["side"][sl], S["wrist"][sl], S["state"][sl], instr[cond][sl].tolist())
            if tokens_example is None:
                tokens_example = ex.srv.decode_tokens(obs)[:3]
            for k, v in ex.features(obs).items():
                feats.setdefault(f"{cond}/{k}", []).append(v)
        print(f"[g3] features {cond} done ({time.time() - t0:.0f} s)", flush=True)
    np.savez(f"{a.out}_feats.npz", **{k: np.concatenate(v) for k, v in feats.items()}, label=S["label"],
             group=S["group"], subset=S["subset"], index=S["index"], instruction=S["instruction"],
             instruction_shuffled=instr["shuffled"], ckpt_sha=np.array(ex.srv.ckpt_sha))
    div_step(a, ex, S, t0, tokens_example)


def div_step(a, ex, S: dict, t0: float, tokens_example) -> None:
    """Step 2. divonly (supplementary, GATES 003 'Supplementary step 2 view'): also saves every chunk, normalized and
    un-normalized (the server's postprocessor), to <out>_divchunks.npz."""
    from vla_policy.server import NoiseBank

    N = len(S["index"])
    # step 2: the first N_DIV_LAYOUTS layout groups of the paired samples, member 0's observation
    pm = S["subset"] == "p"
    lids = list(dict.fromkeys(S["layout_id"][pm].tolist()))[:N_DIV_LAYOUTS]
    assert len(lids) == N_DIV_LAYOUTS, len(lids)
    bank = NoiseBank((ex.pol.config.chunk_size, ex.pol.config.max_action_dim))
    rows, ci, npair, saved = [], [], [], {}
    for lid in lids:
        idx = np.flatnonzero(pm & (S["layout_id"] == lid))
        m0 = idx[S["member"][idx].argmin()]
        assert S["member"][m0] == 0 and len(idx) >= 2 and idx[0] == m0  # npair below pairs ch[0] = member 0
        bank.reset(np.array([0]), np.array([S["noise_seed"][m0]]))
        n_a, n_b = bank.draw(np.array([0])), bank.draw(np.array([0]))  # q = 0, q = 1
        texts = [str(S["instruction"][i]) for i in idx] + [str(S["instruction"][m0])]
        k = len(idx)
        rep = lambda x: np.repeat(x[m0:m0 + 1], k + 1, 0)  # noqa: E731 - one observation for every instruction
        noise = torch.cat([n_a.expand(k, -1, -1), n_b])
        if getattr(a, "guidance_w", None):  # task 004: each row's others from the layout's cube colours
            from vla_common.catalog import COLORS
            from vla_common.guidance import other_instructions
            from vla_policy.guidance import guided_chunk

            cols = [c[0] for c in json.loads(str(S["spec_json"][m0]))["cubes"]]
            tgt = [COLORS[int(S["label"][i])] for i in idx] + [COLORS[int(S["label"][m0])]]
            others = [other_instructions(t, c, cols) for t, c in zip(texts, tgt)]
            raw = ex.raw(rep(S["side"]), rep(S["wrist"]), rep(S["state"]), texts)
            full = guided_chunk(ex.pol, ex.srv.pre, raw, others, noise.to(ex.dev), a.guidance_w).float().cpu().numpy()
        else:
            full = ex.chunk(ex.obs(rep(S["side"]), rep(S["wrist"]), rep(S["state"]), texts), noise)
        ch = full[..., :4]  # arm dims
        if a.cmd == "divonly":
            saved.setdefault("norm", []).append(full)
            saved.setdefault("raw", []).append(ex.srv.post(torch.from_numpy(full).to(ex.dev)).float().cpu().numpy())
        ci.append(ch[:k])
        npair.append((ch[0], ch[k]))  # member 0's instruction: noise q = 0 vs q = 1
        rows.append({"layout_id": int(lid), "members": [int(S["index"][i]) for i in idx], "instructions": texts[:k],
                     "targets": [int(S["label"][i]) for i in idx]})
    div = {"full_chunk": divergence(ci, npair), "first_10": divergence(ci, npair, 10)}
    res = {"ckpt": a.ckpt, "ckpt_sha": ex.srv.ckpt_sha, "n_samples": N, "tokens_example": tokens_example,
           "guidance_w": getattr(a, "guidance_w", None),
           "divergence": div, "layouts": rows, "wall_s": round(time.time() - t0, 1)}
    Path(f"{a.out}_div.json").write_text(json.dumps(res, indent=1))
    if saved:  # ragged per layout (k + 1 chunks): flattened with the layout id of each row; last row of a layout = noise q = 1
        lay = np.concatenate([np.full(len(c), r["layout_id"]) for c, r in zip(saved["norm"], rows)])
        np.savez(f"{a.out}_divchunks.npz", norm=np.concatenate(saved["norm"]), raw=np.concatenate(saved["raw"]),
                 layout_id=lay)
    print("G3_EXTRACT=" + json.dumps({"ckpt": a.ckpt, "guidance_w": res["guidance_w"], "div": div,
                                      "wall_s": res["wall_s"]}), flush=True)


# --------------------------------------------------------------------------- probe (CPU)
def probe(a) -> None:
    F = np.load(a.feats, allow_pickle=False)
    y, groups, subset = F["label"], F["group"], F["subset"]
    folds = group_folds(groups)
    for f in range(5):  # no layout group straddles folds
        assert not set(groups[folds == f].tolist()) & set(groups[folds != f].tolist())
    table = []
    for feat in FEATS:
        for L in LAYERS:
            r = {"feature": feat, "layer": L}
            for cond in CONDS:
                pred = cv_predict(F[f"{cond}/{feat}_L{L}"], y, folds)
                r[cond] = float((pred == y).mean())
                for s in ("p", "u"):
                    r[f"{cond}_{s}"] = float((pred[subset == s] == y[subset == s]).mean())
            r["gap"] = r["true"] - r["shuffled"]
            r["reading"] = reading(r["true"], r["shuffled"])
            table.append(r)
            print(f"[probe] {feat:6s} L{L:2d} true {r['true']:.3f} shuffled {r['shuffled']:.3f} -> {r['reading']}",
                  flush=True)
    fam = {feat: family([r["reading"] for r in table if r["feature"] == feat]) for feat in FEATS}
    div = json.loads(Path(a.div).read_text())["divergence"] if a.div else None
    ign = div["full_chunk"]["ignores_language"] if div else None
    res = {"feats": str(a.feats), "ckpt_sha": str(F["ckpt_sha"]), "n": int(len(y)),
           "class_counts": np.bincount(y, minlength=4).tolist(), "majority_rate": float(np.bincount(y).max() / len(y)),
           "n_groups": int(len(np.unique(groups))), "table": table, "family": fam, "divergence": div,
           "verdict": verdict(fam["lang"], fam["kv"], ign) if ign is not None else None}
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    Path(a.out).write_text(json.dumps(res, indent=1))
    print("G3_PROBE=" + json.dumps({"family": fam, "verdict": res["verdict"]}), flush=True)


def main() -> None:
    p = argparse.ArgumentParser()
    sub = p.add_subparsers(dest="cmd", required=True)
    e = sub.add_parser("extract")
    e.add_argument("--ckpt", required=True)
    e.add_argument("--dumps", type=Path, nargs="+", required=True)
    e.add_argument("--out", required=True, help="output stem")
    e.add_argument("--batch", type=int, default=50)
    d = sub.add_parser("divonly")
    d.add_argument("--ckpt", required=True)
    d.add_argument("--dumps", type=Path, nargs="+", required=True)
    d.add_argument("--out", required=True, help="output stem")
    d.add_argument("--guidance-w", type=float, default=None, help="task 004: guided sampler at weight w (>= 1)")
    q = sub.add_parser("probe")
    q.add_argument("--feats", type=Path, required=True)
    q.add_argument("--div", type=Path, default=None)
    q.add_argument("--out", type=Path, required=True)
    a = p.parse_args()
    torch.manual_seed(0)
    probe(a) if a.cmd == "probe" else extract(a)


if __name__ == "__main__":
    main()
