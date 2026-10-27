"""Task 011: compose the demo video from the --record renders (vla_tabletop.demo_rec) and media/cards.json.

  python scripts/demo_compose.py --rec results/demo/d11 [--only s1,2m]   -> media/clips/NN_*.mp4, media/113_demo.mp4,
                                                                          media/113_demo_short.gif, media/grids/*.json

Honesty rules (task 011): grids = episodes 0..15 of each declared demo stream in stream order, verdicts = the
evaluator's records (dumped beside each grid as media/grids/<key>.json); the full-size episode = the first episode of
the sample (PASS chapters) or its first failure (FAIL chapters and the 'where it breaks' clips), as declared in
hpc/JOBS.md before rendering; all card numbers come from media/cards.json (scripts/demo_facts.py). The viewer camera is
not the policy input; the policy's own 224 px frames are shown as an inset.
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import textwrap
from pathlib import Path

import imageio.v2 as iio
import numpy as np
from PIL import Image, ImageDraw, ImageFont

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
MEDIA = ROOT / "media"
W, H, FPS, SRC_FPS = 1280, 720, 30, 50 / 3
BG, FG, MUTED, ACC = (17, 24, 39), (243, 244, 246), (156, 163, 175), (96, 165, 250)
GOOD, BAD, WARN = (34, 197, 94), (239, 68, 68), (250, 204, 21)
CROP = (120, 60, 840, 600)  # tiles: the viewer's centre 720x540 (the look point projects to the image centre)
COLOR = {"red": (239, 68, 68), "blue": (96, 165, 250), "yellow": (250, 204, 21), "green": (34, 197, 94)}

# Declared streams (hpc/JOBS.md 011 entry): chapter key -> eval JSON stem in --rec.
RUNS = {"expert_pick": "d11_expert_pick_s51001", "expert_place": "d11_expert_place_s51002",
        "expert_stack": "d11_expert_stack_s51003", "s1": "d11_s1_s51010", "2m": "d11_2m_s51020",
        "sizes": "d11_sizes_s51030", "3aplace": "d11_3aplace_s51040", "3bstack": "d11_3bstack_s51050",
        "ground_plain": "d11_ground_plain_s51060", "ground_final": "d11_ground_2m_s51060",
        "2m_b": "d11_2m_rand_s51070", "2m_c": "d11_2m_heldout_s51080", "3apick_d": "d11_3apick_para_s51090"}
# One word per evaluator failure flag, in priority order (eval_policy.FAIL_TYPES).
REASON = [("wrong_object", "wrong cube"), ("wrong_dest", "wrong place"), ("drop", "dropped"),
          ("release_failure", "bad release"), ("slam", "hit table"), ("jam", "jammed"),
          ("early_close", "missed grasp"), ("invalid", "sim error")]


def font(size: int, bold: bool = False) -> ImageFont.FreeTypeFont:
    import matplotlib

    d = Path(matplotlib.get_data_path()) / "fonts" / "ttf"
    return ImageFont.truetype(str(d / ("DejaVuSans-Bold.ttf" if bold else "DejaVuSans.ttf")), size)


def reason(rec: dict) -> str:
    if rec["success"]:
        return "success"
    for k, w in REASON:
        if rec.get(k):
            return w
    return {"pick": "not lifted in time", "place": "not placed in time", "stack": "not stacked in time"}[rec["task"]]


# --------------------------------------------------------------------------- loading
class Ep:
    def __init__(self, d: Path):
        self.dir = d
        self.sig = json.loads((d / "signals.json").read_text())
        self.ticks = self.sig["ticks"]
        self.verdict = reason(self.sig)
        self._cache: dict = {}

    def frames(self, name: str, size: tuple[int, int] | None = None, crop: tuple | None = None) -> list[np.ndarray]:
        key = (name, size, crop)
        if key not in self._cache:
            fr = [f for f in iio.get_reader(self.dir / f"{name}.mp4")]
            if size is not None:
                fr = [np.asarray((Image.fromarray(f).crop(crop) if crop else Image.fromarray(f)).resize(size, Image.BILINEAR))
                      for f in fr]
            self._cache[key] = fr
        return self._cache[key]

    @property
    def n(self) -> int:
        return len(self.ticks)

    def target_color(self) -> str:
        return json.loads(self.sig["spec"])["target"][0] if isinstance(self.sig["spec"], str) else self.sig["spec"]["target"][0]


def load_run(rec_dir: Path, stem: str) -> tuple[dict, list[Ep]]:
    ev = json.loads((rec_dir / f"{stem}.json").read_text())
    eps = []
    for r in sorted(ev["episodes"], key=lambda r: r["index"]):
        e = Ep(rec_dir / "demo_rec" / stem / f"ep{r['index']}")
        assert e.sig["index"] == r["index"] and bool(e.sig["success"]) == bool(r["success"]), (stem, r["index"])
        eps.append(e)
    return ev, eps


# --------------------------------------------------------------------------- drawing
def canvas() -> Image.Image:
    return Image.new("RGB", (W, H), BG)


def wrap(dr: ImageDraw.ImageDraw, xy, text: str, f, fill, width_px: int, gap: int = 6) -> int:
    """Draw word-wrapped text; returns the y below it."""
    x, y = xy
    avg = max(1, int(f.getlength("abcdefghijklmnopqrstuvwxyz") / 26))
    for para in text.split("\n"):
        for line in textwrap.wrap(para, max(10, width_px // avg)) or [""]:
            dr.text((x, y), line, font=f, fill=fill)
            y += f.size + gap
    return y


def badge(dr, xy, ok: bool, text: str, f) -> None:
    x, y = xy
    s = ("✓ " if ok else "✗ ") + text
    w = f.getlength(s)
    dr.rounded_rectangle([x, y, x + w + 12, y + f.size + 8], 5, fill=(GOOD if ok else BAD) + (235,))
    dr.text((x + 6, y + 3), s, font=f, fill=(255, 255, 255))


def ring(dr, uv, r: float, col, width: int = 3) -> None:
    u, v = uv[0], uv[1]
    dr.ellipse([u - r, v - r, u + r, v + r], outline=col, width=width)


def card(title: str, sections: list[tuple[str, str]], kicker: str = "") -> np.ndarray:
    im = canvas()
    dr = ImageDraw.Draw(im)
    y = 56
    if kicker:
        dr.text((80, y), kicker.upper(), font=font(20, True), fill=ACC)
        y += 34
    y = wrap(dr, (80, y), title, font(36, True), FG, W - 160, 6) + 14
    for head, body in sections:
        if head:
            dr.text((80, y), head, font=font(19, True), fill=ACC)
            y += 27
        y = wrap(dr, (80, y), body, font(21), FG, W - 160, 5) + 12
    assert y < H - 20, f"card overflows: {title}"
    return np.asarray(im)


def status_text(ep: Ep, k: int) -> tuple[str, tuple]:
    """Live status from the evaluator's own per-tick signals."""
    t = ep.ticks[min(k, ep.n - 1)]
    task = ep.sig["task"]
    if t["wrong_grasp"]:
        return "touched / lifted a wrong cube", BAD
    if t["drop"]:
        return "dropped the cube", BAD
    if t["slam"]:
        return "hit the table or a pad (> 20 N)", BAD
    if t["success_now"]:
        return "success condition met", GOOD
    if t["held"] and t["lift_m"] > 0.02:
        return f"holding it, {100 * t['lift_m']:.0f} cm up", FG
    if t["held"] or t["confirmed"] and t["claw_closed"]:
        return "grasped", FG
    if task != "pick" and t["confirmed"] and not t["claw_closed"]:
        return "released", FG
    return "reaching", MUTED


# --------------------------------------------------------------------------- scenes (each returns a list of frames)
def hold(img: np.ndarray, secs: float) -> list[np.ndarray]:
    return [img] * int(round(secs * FPS))


def src_index(k_out: int) -> int:
    return int(k_out * SRC_FPS / FPS)


def episode_full(ep: Ep, title: str, tag: str, show_dest: bool, end_secs: float = 2.5) -> list[np.ndarray]:
    """Full-size episode: viewer 960x720 (not the policy input) + the policy's own 224 px frames + live status."""
    view = ep.frames("viewer")
    side = ep.frames("side", (288, 288))
    wrist = ep.frames("wrist", (288, 288))
    n_out = int(ep.n * FPS / SRC_FPS)
    T = ep.sig["T"] / 50
    fT, fS, fB = font(22, True), font(17), font(16, True)
    out = []
    for k in range(n_out):
        i = min(src_index(k), ep.n - 1)
        im = canvas()
        im.paste(Image.fromarray(view[i]), (0, 0))
        dr = ImageDraw.Draw(im, "RGBA")
        tk = ep.ticks[i]
        col = COLOR.get(ep.target_color(), WARN)
        ring(dr, tk["px_target"], 26, (255, 255, 255, 230), 4)
        ring(dr, tk["px_target"], 30, col + (255,), 3)
        dr.text((tk["px_target"][0] + 32, tk["px_target"][1] - 30), "target", font=fB, fill=(255, 255, 255))
        if show_dest:
            ring(dr, tk["px_dest"], 34, (255, 255, 255, 200), 3)
            dr.text((tk["px_dest"][0] + 36, tk["px_dest"][1] + 4), "destination", font=fB, fill=(255, 255, 255))
        dr.rectangle([0, 0, 960, 46], fill=(0, 0, 0, 170))
        dr.text((14, 10), f"told: “{ep.sig['instruction']}”", font=fT, fill=(255, 255, 255))
        dr.rectangle([0, 690, 960, 720], fill=(0, 0, 0, 150))
        dr.text((12, 696), "viewer camera (for you; the policy does not see it)", font=fS, fill=(220, 220, 220))
        dr.text((960 - 190, 696), f"t = {tk['t'] / 50:4.1f} s / {T:.0f} s", font=fS, fill=(220, 220, 220))
        # right panel
        x0 = 976
        dr.text((x0, 8), tag.upper(), font=font(15, True), fill=ACC)
        wrap(dr, (x0, 28), title, font(17, True), FG, 290, 3)
        dr.text((x0, 76), "what the policy sees (224 px):", font=font(14), fill=MUTED)
        im.paste(Image.fromarray(side[i]), (x0, 96))
        im.paste(Image.fromarray(wrist[i]), (x0, 96 + 292))
        dr.text((x0 + 6, 100), "side camera", font=font(13, True), fill=(255, 255, 255))
        dr.text((x0 + 6, 392), "wrist camera", font=font(13, True), fill=(255, 255, 255))
        s, c = status_text(ep, i)
        dr.text((x0, 670), "evaluator signal now:", font=font(14), fill=MUTED)
        dr.text((x0, 690), s, font=font(17, True), fill=c)
        out.append(np.asarray(im))
    # verdict
    last = Image.fromarray(out[-1])
    dr = ImageDraw.Draw(last, "RGBA")
    dr.rectangle([180, 270, 780, 450], fill=(0, 0, 0, 200))
    ok = bool(ep.sig["success"])
    dr.text((210, 290), ("✓ SUCCESS" if ok else "✗ FAILURE"), font=font(44, True), fill=GOOD if ok else BAD)
    wrap(dr, (212, 356), "the simulator's success rule held at the end" if ok else
         "evaluator: " + ep.verdict, font(22), FG, 540)
    return out + hold(np.asarray(last), end_secs)


def grid(eps: list[Ep], title: str, caption: str, cols: int = 4, end_secs: float = 3.0) -> list[np.ndarray]:
    """First 16 episodes, real time, evaluator verdict badges at the end."""
    tw, th_ = 960 // cols, 720 // ((len(eps) + cols - 1) // cols)
    th_ = min(th_, tw * 3 // 4)
    tiles = [e.frames("viewer", (tw, th_), CROP) for e in eps]
    n_src = max(e.n for e in eps)
    n_out = int(n_src * FPS / SRC_FPS)
    fB = font(13, True)
    sx, sy = tw / (CROP[2] - CROP[0]), th_ / (CROP[3] - CROP[1])
    ok_n = sum(bool(e.sig["success"]) for e in eps)

    def frame(k: int, final: bool) -> np.ndarray:
        im = canvas()
        dr = ImageDraw.Draw(im, "RGBA")
        for j, (e, fr) in enumerate(zip(eps, tiles)):
            i = min(src_index(k), e.n - 1)
            x, y = (j % cols) * tw, (j // cols) * th_
            im.paste(Image.fromarray(fr[i]), (x, y))
            u, v = e.ticks[i]["px_target"][0] - CROP[0], e.ticks[i]["px_target"][1] - CROP[1]
            ring(dr, (x + u * sx, y + v * sy), 11, COLOR.get(e.target_color(), WARN) + (230,), 2)
            dr.rectangle([x, y, x + tw - 1, y + th_ - 1], outline=(17, 24, 39), width=2)
            dr.text((x + 5, y + th_ - 18), f"#{j + 1}", font=fB, fill=(255, 255, 255))
            if final:
                badge(dr, (x + 4, y + 4), bool(e.sig["success"]), e.verdict, fB)
        x0 = 980
        dr.text((x0, 20), "HONEST SAMPLE", font=font(15, True), fill=ACC)
        y = wrap(dr, (x0, 44), title, font(19, True), FG, 280, 4) + 10
        y = wrap(dr, (x0, y), caption, font(15), MUTED, 280, 4) + 12
        y = wrap(dr, (x0, y), "ring = the cube the instruction names", font(15), FG, 280, 4) + 12
        if final:
            dr.text((x0, y + 6), f"{ok_n} / {len(eps)} succeeded", font=font(24, True), fill=FG)
            wrap(dr, (x0, y + 44), "(a small sample; the rate on the card is from 1,000 episodes)", font(14), MUTED, 280, 3)
        return np.asarray(im)

    return [frame(k, False) for k in range(n_out)] + hold(frame(n_out - 1, True), end_secs)


def side_by_side(eps: list[Ep], labels: list[str], rows: int, title: str, row_labels: list[str],
                 end_secs: float = 3.0) -> list[np.ndarray]:
    """rows x cols tiles (384x288), real time; badges at the end. eps row-major."""
    cols = len(eps) // rows
    tw, th_ = 384, 288
    lx, top = 128, 64
    tiles = [e.frames("viewer", (tw, th_), CROP) for e in eps]
    n_out = int(max(e.n for e in eps) * FPS / SRC_FPS)
    sx, sy = tw / (CROP[2] - CROP[0]), th_ / (CROP[3] - CROP[1])

    def frame(k: int, final: bool) -> np.ndarray:
        im = canvas()
        dr = ImageDraw.Draw(im, "RGBA")
        dr.text((20, 14), title, font=font(22, True), fill=FG)
        for c in range(cols):
            f, lab = font(15, True), labels[c]
            while f.getlength(lab) > tw - 12 and len(lab) > 4:
                lab = lab[:-2] + "…"
            dr.text((lx + c * tw + 6, top - 26 + 4), lab, font=f, fill=FG)
        for r in range(rows):
            wrap(dr, (12, top + r * th_ + 100), row_labels[r], font(16, True), ACC, lx - 20, 3)
        for j, (e, fr) in enumerate(zip(eps, tiles)):
            i = min(src_index(k), e.n - 1)
            x, y = lx + (j % cols) * tw, top + (j // cols) * th_
            im.paste(Image.fromarray(fr[i]), (x, y))
            u, v = e.ticks[i]["px_target"][0] - CROP[0], e.ticks[i]["px_target"][1] - CROP[1]
            ring(dr, (x + u * sx, y + v * sy), 14, COLOR.get(e.target_color(), WARN) + (230,), 3)
            dr.rectangle([x, y, x + tw - 1, y + th_ - 1], outline=BG, width=3)
            if final:
                badge(dr, (x + 6, y + 6), bool(e.sig["success"]), e.verdict, font(15, True))
        if top + rows * th_ + 40 < H:
            dr.text((lx, top + rows * th_ + 14), "ring = the cube the sentence names; badges = the simulator's evaluator "
                    "verdict at the end", font=font(16), fill=MUTED)
        return np.asarray(im)

    return [frame(k, False) for k in range(n_out)] + hold(frame(n_out - 1, True), end_secs)


# --------------------------------------------------------------------------- chapters
def write_clip(frames: list[np.ndarray], path: Path) -> float:
    path.parent.mkdir(parents=True, exist_ok=True)
    w = iio.get_writer(path, fps=FPS, codec="libx264", quality=None, macro_block_size=1,
                       pixelformat="yuv420p", ffmpeg_params=["-crf", "22", "-preset", "medium", "-tune", "animation"])
    for f in frames:
        w.append_data(f)
    w.close()
    return len(frames) / FPS


def first_shown(eps: list[Ep], verdict: str) -> Ep:
    """Declared rule: PASS chapter -> the sample's first episode; otherwise its first failure (else the first)."""
    if verdict == "PASS":
        return eps[0]
    return next((e for e in eps if not e.sig["success"]), eps[0])


def dump_grid(key: str, stem: str, ev: dict, eps: list[Ep], shown: Ep | None) -> None:
    d = MEDIA / "grids"
    d.mkdir(parents=True, exist_ok=True)
    (d / f"{key}.json").write_text(json.dumps({
        "stream": stem, "seed": ev["config"]["seed"], "ckpt": ev["ckpt"], "ckpt_sha": ev["ckpt_sha"],
        "episodes": [{"index": e.sig["index"], "instruction": e.sig["instruction"], "success": bool(e.sig["success"]),
                      "shown_verdict": e.verdict} for e in eps],
        "full_size_episode": None if shown is None else shown.sig["index"]}, indent=1))


def chapters(rec: Path, cards: dict, only: set[str] | None):
    """Yields (clip name, frames) one chapter at a time (a whole video in memory would be ~20 GB)."""

    def want(k):
        return only is None or k in only

    if want("setup"):
        yield "00_setup", hold(card("A robot arm that follows typed instructions, in simulation", [
            ("Setup", "Simulated Franka arm (Isaac Lab, physics simulation; no real robot). Cubes in 4 colours and "
                      "2 sizes, 4 coloured pads."),
            ("Policy", "SmolVLA, a pretrained ~450 M-parameter vision-language-action model, fine-tuned in this project on "
                       "demonstrations from a scripted expert (one model per skill stage). Trained on 1-4 NVIDIA H200 GPUs."),
            ("It gets", "a side camera and a wrist camera (224 x 224 px), the arm's 19 joint readings, and one sentence."),
            ("It outputs", "a chunk of 50 small arm moves + gripper open/close; the first 10 run (0.2 s), then it looks again."),
            ("How it is judged", "every episode is scored by the simulator's own success rule (shown per chapter), never by eye."),
        ], "113 · fine-tuning a VLA to pick, place and stack cubes"), 12)
    if want("expert"):
        eps = [load_run(rec, RUNS[k])[1][0] for k in ("expert_pick", "expert_place", "expert_stack")]
        fr = hold(card("What the training data looks like", [
            ("", "A scripted expert that knows the true cube positions does each task; its episodes (camera images, "
                 "joint readings, the sentence, and its actions) are what SmolVLA is trained to imitate."),
            ("", "Next: one expert episode each for pick, place and stack (not the learned policy)."),
        ], "training data"), 5)
        fr += side_by_side(eps, [f"“{e.sig['instruction']}”" for e in eps], 1,
                           "scripted expert demos (training data), real time", ["expert"], 2.0)
        yield "01_expert_demos", fr
    for n, ch in enumerate(cards["chapters"], start=2):
        k = ch["key"]
        if not want(k):
            continue
        ev, eps = load_run(rec, RUNS[k])
        eps = eps[:16]
        assert [e.sig["index"] for e in eps] == list(range(16)), k
        ex = "; ".join(f"“{e.sig['instruction']}”" for e in eps[:2])
        fr = hold(card(ch["title"], [
            ("Told (training templates)", " / ".join(ch["templates"][:4]) + "\nexamples from this sample: " + ex),
            ("Success rule (the evaluator's code)", ch["rule"]),
            ("Model and data", f"{ch['model']}: {ch['data']}. Scenes: {ch['scene']}."),
            ("Result", "\n".join(ch["result_lines"])),
        ], f"skill {n - 1} of 5 · {ch['verdict']}"), 13)
        shown = first_shown(eps, ch["verdict"])
        fr += grid(eps, ch["title"], "first 16 episodes from unseen demo scenes, in order, not selected; real time")
        what = "first episode of the sample" if ch["verdict"] == "PASS" else "first failure in the sample"
        fr += episode_full(shown, f"{what} (#{eps.index(shown) + 1})", f"skill {n - 1} of 5 · {ch['verdict']}",
                           ch["task"] != "pick")
        dump_grid(k, RUNS[k], ev, eps, shown)
        yield f"{n:02d}_{k}", fr
    if want("grounding"):
        evp, pl = load_run(rec, RUNS["ground_plain"])
        evf, fi = load_run(rec, RUNS["ground_final"])
        g = next(gid for gid in sorted({e.sig["layout_id"] for e in fi})
                 if sum(e.sig["layout_id"] == gid for e in fi) >= 3)  # declared: the first group with >= 3 cubes
        P = [e for e in pl if e.sig["layout_id"] == g][:3]
        F = [e for e in fi if e.sig["layout_id"] == g][:3]
        assert [e.sig["instruction"] for e in P] == [e.sig["instruction"] for e in F]
        gr = cards["grounding"]
        fr = hold(card("Does it listen to the words? Same scene, three instructions", [
            ("", f"Plain fine-tuning on colour scenes ignored the colour word: it picked a cube about at random "
                 f"({100 * gr['plain_rate']:.1f} % success, wrong cube {100 * gr['plain_wrong']:.1f} % on 1,000 test scenes)."),
            ("The fix", "counterfactual paired scenes (every scene demonstrated once per cube that can be named) + an "
                        "instruction-contrastive loss (the motion must fit the true sentence clearly better than another "
                        f"cube's sentence). Final colour model: {100 * gr['final_rate']:.1f} %."),
            ("Next", "the same scene (first demo scene with 3 cubes, not selected), each model told three different cubes."),
        ], "the core result"), 10)
        fr += side_by_side(P + F, [f"“{e.sig['instruction']}”" for e in F], 2,
                           "same scene, same start; only the sentence changes (real time)",
                           ["plain fine-tune (before the fix)", "final colour model"], 3.0)
        (MEDIA / "grids").mkdir(parents=True, exist_ok=True)
        (MEDIA / "grids" / "grounding.json").write_text(json.dumps({"layout_id": g, "plain": [
            {"index": e.sig["index"], "instruction": e.sig["instruction"], "success": bool(e.sig["success"]),
             "verdict": e.verdict} for e in P], "final": [
            {"index": e.sig["index"], "instruction": e.sig["instruction"], "success": bool(e.sig["success"]),
             "verdict": e.verdict} for e in F]}, indent=1))
        yield "07_grounding", fr
        export_grounding(P, F, g)
    if want("breaks"):
        fr = hold(card("Where it breaks", [
            ("", "Three checks on scenes unlike training, measured on 1,000 fresh scenes each (reported only, no pass bar). "
                 "One short clip each: the first failure among 16 demo episodes."),
        ], "limits"), 5)
        for b in cards["breaks"]:
            ev, eps = load_run(rec, RUNS[b["key"]])
            eps = eps[:16]
            shown = first_shown(eps, "FAIL")
            ok = sum(bool(e.sig["success"]) for e in eps)
            fr += hold(card(b["title"], [("", b["what"]), ("Result", b["line"]),
                                         ("", f"In the 16-episode demo sample: {ok} / 16 succeeded. Clip: first failure.")],
                            "where it breaks"), 7)
            fr += episode_full(shown, f"first failure in the sample (#{eps.index(shown) + 1})", "where it breaks", False, 2.0)
            dump_grid(b["key"], RUNS[b["key"]], ev, eps, shown)
        yield "08_breaks", fr
    if want("end"):
        rows = [(c["title"], c["gate"], c["verdict"]) for c in cards["chapters"]]
        im = canvas()
        dr = ImageDraw.Draw(im)
        dr.text((80, 50), "Results (simulation; each stage judged once on its pre-set bar)", font=font(30, True), fill=FG)
        y = 120
        for h, x in (("stage", 80), ("success", 720), ("bar", 900), ("verdict", 1030)):
            dr.text((x, y), h, font=font(20, True), fill=ACC)
        y += 40
        for t, g, v in rows:
            dr.text((80, y), t[:52], font=font(22), fill=FG)
            dr.text((720, y), f"{100 * g['rate']:.1f} %", font=font(22, True), fill=FG)
            dr.text((900, y), f">= {100 * g['bar']:.0f} %", font=font(22), fill=FG)
            dr.text((1030, y), v, font=font(22, True), fill=GOOD if v == "PASS" else BAD)
            y += 44
        y += 20
        y = wrap(dr, (80, y), "Stack: stopped on validation by its guard. Sizes, place, stack missed their bars; their failures "
                              "are motor precision (grasp centring, releases, drops), not choosing the wrong cube.",
                 font(19), MUTED, W - 160)
        y = wrap(dr, (80, y + 10), f"Compute: {cards['budget_h200_h']:.2f} H200 GPU-hours for training and evaluation "
                                   "(+0.40 for this video's renders). Code: github.com/matus012/franka-arm-vla "
                                   "(weights and datasets not released).", font(19), MUTED, W - 160)
        yield "09_end", hold(np.asarray(im), 10)


def export_grounding(P: list[Ep], F: list[Ep], g: int) -> None:
    """results/demo/grounding/traj.json for figure 3a (schema in figures/README.md)."""
    d = ROOT / "results" / "demo" / "grounding"
    d.mkdir(parents=True, exist_ok=True)
    Image.fromarray(F[0].frames("viewer")[1]).save(d / "scene.png")
    Image.fromarray(F[0].frames("side")[1]).save(d / "policy_side.png")
    rows = []
    for model, label, eps in (("plain", "plain fine-tune (before the fix)", P), ("final", "final colour model", F)):
        runs = []
        for e in eps:
            p = d / f"last_{model}_ep{e.sig['index']}.png"
            Image.fromarray(e.frames("viewer")[-1]).save(p)
            runs.append({"instruction": e.sig["instruction"], "target_color": e.target_color(),
                         "verdict": "success" if e.sig["success"] else "fail", "reason": e.verdict,
                         "tcp_px": [t["px_tcp"][:2] for t in e.ticks], "last_frame": str(p.relative_to(ROOT)).replace("\\", "/")})
        rows.append({"model": model, "label": label, "runs": runs})
    spec = json.loads(F[0].sig["spec"]) if isinstance(F[0].sig["spec"], str) else F[0].sig["spec"]
    note = "One scene: " + ", ".join(f"{c[1]} {c[0]}" for c in spec["cubes"]) + " cubes; demo scene group " \
           f"{g} (first with >= 3 cubes, not selected). Paths = measured fingertip positions."
    (d / "traj.json").write_text(json.dumps({"scene_note": note, "image": "results/demo/grounding/scene.png",
                                             "policy_view": "results/demo/grounding/policy_side.png", "rows": rows}, indent=1))


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--rec", type=Path, default=ROOT / "results" / "demo" / "d11")
    p.add_argument("--only", default="")
    p.add_argument("--no-concat", action="store_true")
    a = p.parse_args()
    cards = json.loads((MEDIA / "cards.json").read_text())
    only = set(a.only.split(",")) if a.only else None
    manifest_p = MEDIA / "chapters.json"
    manifest = json.loads(manifest_p.read_text()) if manifest_p.exists() else {}
    for name, frames in chapters(a.rec, cards, only):
        secs = write_clip(frames, MEDIA / "clips" / f"{name}.mp4")
        manifest[name] = round(secs, 2)
        print(f"{name}: {secs:.1f} s", flush=True)
        del frames
    manifest_p.write_text(json.dumps(dict(sorted(manifest.items())), indent=1))
    if a.no_concat:
        return
    import imageio_ffmpeg

    ff = imageio_ffmpeg.get_ffmpeg_exe()
    lst = MEDIA / "clips" / "list.txt"
    lst.write_text("".join(f"file '{n}.mp4'\n" for n in sorted(manifest)))
    subprocess.run([ff, "-y", "-loglevel", "error", "-f", "concat", "-safe", "0", "-i", str(lst), "-c", "copy",
                    "-movflags", "+faststart", str(MEDIA / "113_demo.mp4")], check=True)
    g = MEDIA / "clips" / "07_grounding.mp4"
    if g.exists():  # README hero: the grounding side-by-side, <= 20 s
        subprocess.run([ff, "-y", "-loglevel", "error", "-ss", "10", "-t", "14", "-i", str(g), "-vf",
                        "fps=10,scale=720:-1:flags=lanczos,split[a][b];[a]palettegen=max_colors=96[p];[b][p]paletteuse",
                        str(MEDIA / "113_demo_short.gif")], check=True)
    print(f"total {sum(manifest.values()):.1f} s -> {MEDIA / '113_demo.mp4'}")


if __name__ == "__main__":
    main()
