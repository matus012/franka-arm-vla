"""Writes fig1.html fig2.html fig3.html cards.html (1600 px wide, shot at 2x by shoot.js). Run after assets.py."""
import json, os, math, random

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
TRAJ = os.environ.get('TRAJ') or os.path.join(ROOT, 'results', 'demo', 'grounding', 'traj.json')  # env override only for tests with a fake file

CSS = """
:root{--ink:#1f2328;--muted:#5d6470;--line:#d9dce1;--blue:#3b6fd6;--grey:#8a8f98;--green:#1a9850;--red:#d7263d;--orange:#e07b12;--accent:#ff3b30;--purple:#7b4fc9}
*{box-sizing:border-box;margin:0;padding:0}
body{background:#fff;font-family:Inter,'Segoe UI',sans-serif;color:var(--ink);-webkit-font-smoothing:antialiased}
.fig{width:1600px;padding:30px 44px 30px;position:relative;background:#fff}
.head{display:flex;justify-content:space-between;align-items:baseline;margin-bottom:20px}
.htitle{font-size:24px;font-weight:800;letter-spacing:-.01em}
.htitle span{color:var(--muted);font-weight:500}
.badge{font-size:13px;color:#13703b;background:#edf8f1;border:1px solid #bfe3cd;border-radius:99px;padding:5px 12px;font-weight:600}
.caption{margin-top:22px;font-size:16.5px;line-height:1.55;color:#2b2f36;border-top:1px solid #eceef1;padding-top:14px}
.ptitle{font-size:18.5px;font-weight:700;margin-bottom:3px;display:flex;align-items:center}
.psub{font-size:14.5px;color:var(--muted);margin-bottom:10px;line-height:1.4}
.num{display:inline-flex;flex:none;width:25px;height:25px;border-radius:50%;background:var(--accent);color:#fff;font-size:13.5px;font-weight:700;align-items:center;justify-content:center;margin-right:9px}
.chip{display:inline-flex;align-items:center;gap:6px;font-size:13px;color:#3a3f47;margin-right:14px}
.sw{width:14px;height:14px;border-radius:3px;display:inline-block}
img{display:block}
.ph{border-radius:8px;overflow:hidden;border:1px solid #e3e5e8;background:#fff}
.ph img{width:100%}
.tagl{font-size:11.5px;color:var(--muted);font-weight:600;text-transform:uppercase;letter-spacing:.03em}
.bub{font-size:15px;padding:7px 12px;border-radius:12px;box-shadow:0 3px 10px rgba(0,0,0,.12);line-height:1.35;display:inline-block}
.bub.q{background:#fff;border:1px solid #d5d8dd}
.ill{font-size:11.5px;color:var(--muted);font-style:italic}
"""
BADGE = "<div class='badge'>final results · real 113 outputs</div>"
ARROW = "<svg width='46' height='40' viewBox='0 0 46 40' style='flex:none;margin-top:%dpx'><path d='M4 20 H34' stroke='#9aa0a8' stroke-width='4' stroke-linecap='round'/><path d='M30 9 L43 20 L30 31 Z' fill='#9aa0a8'/></svg>"

def page(body, css=''):
    return f"<!doctype html><html><head><meta charset='utf-8'><style>{CSS}{css}</style></head><body>{body}</body></html>"
def w(name, body, css=''):
    open(f'{name}.html', 'w', encoding='utf-8').write(page(body, css)); print('wrote', name)
def im(src, extra=''):
    return f"<div class='ph' style='{extra}'><img src='assets/{src}'></div>"

# ================================================== Figure 1 ===================================================
cells = ''.join(f"<div class='cell'>{'<span class=ill></span>'}</div>" for _ in range(0))
chunk = ''.join(f"<i class='{'on' if k < 10 else ''}'></i>" for k in range(50))
times = ['0 s', '2.3 s', '4.6 s', '7.0 s']
films = ''.join(f"<div class='fr'>{im(f'f1_close_{k}.png')}<div class='ft'>{times[k]}</div></div>" for k in range(4))
f1 = f"""
<div class='fig' id='fig'>
 <div class='head'><div class='htitle'>SmolVLA tabletop <span>· what it does</span></div>{BADGE}</div>
 <div style='display:flex;align-items:flex-start;gap:0'>
  <div style='width:430px'>
   <div class='ptitle'><span class='num'>1</span>It is shown a scene and told what to do</div>
   <div class='psub'>side camera and wrist camera (224 px), the arm's own 19 joint readings, and one sentence</div>
   <div style='display:flex;gap:10px'><div style='flex:1'>{im('f1_side_in.png')}<div class='tagl' style='margin-top:5px'>side camera</div></div>
        <div style='flex:1'>{im('f1_wrist_in.png')}<div class='tagl' style='margin-top:5px'>wrist camera</div></div></div>
   <div style='margin-top:12px;display:flex;align-items:center;gap:10px;flex-wrap:wrap'>
     <span class='bub q'><b>&ldquo;pick up the blue cube&rdquo;</b></span><span class='chip' style='margin:0'>+ 19 joint numbers</span></div>
  </div>
  {ARROW % 150}
  <div style='width:300px'>
   <div class='ptitle'><span class='num'>2</span>SmolVLA reads and decides</div>
   <div class='psub'>a ~450 M-parameter pretrained vision-language-action model, fine-tuned by me on scripted-expert demos</div>
   <div class='net'>
     <div class='nb g'>vision-language model<br><small>reads pictures + words</small></div>
     <div class='dn'>&#9660;</div>
     <div class='nb b'>action expert<br><small>turns what it read into motion</small></div>
   </div>
   <div class='ill' style='margin-top:6px'>schematic</div>
  </div>
  {ARROW % 150}
  <div style='width:410px'>
   <div class='ptitle'><span class='num'>3</span>It plans 50 moves, runs the first 10</div>
   <div class='psub'>each move = small arm step (x, y, z, turn) + gripper open / close; 10 moves = 0.2 s, then it looks again</div>
   <div class='chunk'>{chunk}</div>
   <div style='display:flex;justify-content:space-between;font-size:13px;margin-top:6px'><span style='color:var(--blue);font-weight:700'>run now: 10 moves</span><span style='color:var(--muted)'>planned, not run: 40</span></div>
   <div class='ill' style='margin-top:8px'>illustrative: the shape of one chunk, not real action values</div>
  </div>
 </div>
 <div style='margin-top:26px'>
  <div class='ptitle'><span class='num'>4</span>The arm does it, one chunk after another</div>
  <div class='psub'>real simulator frames (close-up camera, for viewing only), one episode; counts as a success when, at the end of the 7 s episode, the named cube is held by both fingers and lifted &ge; 10 cm above where it rested</div>
  <div style='display:flex;gap:16px;align-items:flex-start'>{films}
    <div class='res'><b>Same scene, other words</b><br>Told &ldquo;red&rdquo; or &ldquo;yellow&rdquo; instead, the same model lifts that cube (Figure 3a).<br><br>
    <b>Over 1,000 fresh scenes</b><br>colour model: 96.0 % picks (95 % interval 94.6 to 97.0).</div></div>
 </div>
 <div class='caption'><b>Figure 1. What the system does.</b> A simulated Franka arm gets a camera image, its own joint readings and a sentence (1). SmolVLA, a small pretrained vision-language-action model fine-tuned here on scripted-expert demonstrations, turns them into a chunk of 50 planned moves (2, 3); the first 10 are executed, then the cycle repeats with a fresh image (4). Frames show one real episode of the colour model; simulation only, no real robot.</div>
</div>"""
f1css = """
.net{display:flex;flex-direction:column;align-items:center;gap:4px}
.nb{width:100%;border-radius:10px;padding:12px 10px;text-align:center;font-weight:700;font-size:15.5px;line-height:1.3}
.nb small{font-weight:500;font-size:12.5px;color:var(--muted)}
.nb.g{border:2px solid var(--grey);background:#f6f7f8}.nb.b{border:2px solid var(--blue);background:#f3f6fe}
.dn{color:#9aa0a8;font-size:16px}
.chunk{display:grid;grid-template-columns:repeat(25,1fr);gap:3px}
.chunk i{height:22px;border-radius:3px;background:#e4e6ea;display:block}.chunk i.on{background:var(--blue)}
.fr{width:236px}.ft{font-size:13px;color:var(--muted);margin-top:5px;text-align:center}
.res{flex:1;font-size:14.5px;line-height:1.45;background:#f1faf4;border:2px solid var(--green);border-radius:12px;padding:12px 14px;color:#244a33;margin-left:6px}
"""
w('fig1', f1, f1css)

# ================================================== Figure 2 ===================================================
def box(cls, t, title, tag, thumb, desc, style=''):
    return f"""<div class='bx {cls}' style='{style}'><div class='bh'><span class='tn'>{t}</span>{title}</div><div class='tg'>{tag}</div>
      <div class='th'>{thumb}</div><div class='ds'>{desc}</div></div>"""
trio = lambda a, b, c: f"<div style='display:flex;gap:6px'>{im(a)}{im(b)}{im(c)}</div>"
duo = lambda a, b: f"<div style='display:flex;gap:8px'>{im(a)}{im(b)}</div>"
A_ = lambda p: "<svg width='36' height='40' viewBox='0 0 46 40' style='flex:none;align-self:center;margin-top:-30px'><path d='M4 20 H34' stroke='#9aa0a8' stroke-width='4' stroke-linecap='round'/><path d='M30 9 L43 20 L30 31 Z' fill='#9aa0a8'/></svg>"
row = (
  box('ots', 'T1', 'Simulator', 'off-the-shelf · we configure', duo('f1_side_in.png', 'f1_wrist_in.png'),
      'Isaac Lab physics with a Franka arm, a table, up to 5 of 8 cubes (4 colours, 2 sizes) and 4 coloured pads. Two cameras: side and wrist.') + A_(0) +
  box('ours', 'T2', 'Scripted expert', 'we write it · uses true cube positions', duo('expert_pick.png', 'expert_stack.png'),
      'Plain code that knows exactly where every cube is and does pick, place and stack. Its own success in sim, after one fix: pick 100 %, place 100 %, stack 97 % (first version: 99.8 / 100 / 98.0). Frames: pick and stack.') + A_(0) +
  box('gt', 'T3', 'Dataset: counterfactual pairs', 'ground truth · the expert\'s demos', duo('ds_side_t0.png', 'ds_side_end.png'),
      'Each scene is demonstrated once for every cube that could be named (&ldquo;pick up the red cube&rdquo;, then &ldquo;blue&rdquo;, &hellip;). 6,000 demos for the colour model. Frames shown: one dataset episode, start and end.') + A_(0) +
  box('mix', 'T4', 'SmolVLA fine-tune', 'off-the-shelf model + our loss, on PERUN H200 GPUs', im('train_thumb.png'),
      'Start from the pretrained SmolVLA. Train it to copy the expert with flow matching, plus an extra loss that makes the words matter. Curve: real training run (colour model, 68,880 steps).') + A_(0) +
  box('ours', 'T5', 'Evaluation in sim', 'we write it · fresh scenes only', im('g_close_end_blue.png'),
      'The trained policy drives the simulator on scenes it never saw: 300 to 1,000 episodes per test. Success is scored by the simulator, never by eye.')
)
f2 = f"""
<div class='fig' id='fig'>
 <div class='head'><div class='htitle'>SmolVLA tabletop <span>· how it was built and tested</span></div>{BADGE}</div>
 <div class='legend'>
    <div class='chip'><span class='sw' style='background:#eceef1;border:2px solid #8a8f98'></span>off-the-shelf (we configure)</div>
    <div class='chip'><span class='sw' style='background:#eef3fd;border:2px solid #3b6fd6'></span>we write the code</div>
    <div class='chip'><span class='sw' style='background:#edf8f1;border:2px solid #1a9850'></span>ground truth (the expert's demos)</div>
    <div class='chip'><span class='sw' style='border:2px dashed #8a8f98'></span>check</div>
 </div>
 <div class='row'>{row}</div>
 <div class='row2'>
  <div class='bx dash' style='flex:1.1'><div class='bh'>Staged gates, one fix each</div><div class='tg'>check · written before every run</div>
    <div class='ds'>Every stage has a pass bar fixed in advance. A miss gets one diagnosed fix and one more try, then the branch stops. Example: colours 31.5 % &rarr; wrong-cube diagnosis &rarr; contrastive loss &rarr; 67.7 % &rarr; more data &rarr; 96.7 %.</div></div>
  <div class='bx dash' style='flex:1'><div class='bh'>Diagnostics</div><div class='tg'>check · after a miss</div>
    <div class='ds'>Linear probe on SmolVLA's features, guidance test, paired probe (which cube was grasped), audits that no test scene is in the training data.</div></div>
  <div class='bx dash' style='flex:1'><div class='bh'>Transfer tests</div><div class='tg'>check · 1,000 episodes each</div>
    <div class='ds'>Randomized visuals, colour/size combinations never trained on, and reworded instructions, run on every model that exists (Figure 3c).</div></div>
  <div class='bx' style='flex:.9;border-color:#e0b0b6;background:#fdf3f4'><div class='bh'>Not usable</div><div class='tg'>baseline</div>
    <div class='ds'>Planned comparison with a small no-language network: it failed its own training-scene check (0 / 100), so no number is shown.</div></div>
 </div>
 <div class='caption'><b>Figure 2. How it was built.</b> A scripted expert that knows the true cube positions is run in the simulator to produce demonstrations (T2, T3); each scene is demonstrated once per nameable cube, so the words, not the scene, decide the action. SmolVLA is fine-tuned on them on PERUN H200 GPUs (T4) and tested on fresh scenes (T5). Dashed boxes are checks: stage by stage pass bars with one diagnosed fix each, diagnostics, and transfer tests. Thumbnails are real frames and a real training curve.</div>
</div>"""
f2css = """
.legend{margin:-6px 0 16px}
.row{display:flex;align-items:stretch;gap:10px}
.bx{border:2px solid var(--grey);background:#f6f7f8;border-radius:12px;padding:10px 11px 12px}
.row>.bx{flex:1;min-width:0}
.bx.ours{border-color:var(--blue);background:#f3f6fe}
.bx.mix{border-color:transparent;background:linear-gradient(#f5f6fb,#f5f6fb) padding-box,linear-gradient(90deg,#8a8f98,#3b6fd6) border-box}
.bx.gt{border-color:var(--green);background:#f1faf4}
.bx.dash{border-style:dashed;background:#fff}
.bh{font-size:16px;font-weight:700;display:flex;align-items:center;gap:6px;line-height:1.2}
.tn{font-size:11px;font-weight:800;background:#1f2328;color:#fff;border-radius:5px;padding:2px 5px}
.tg{font-size:11.5px;color:var(--muted);margin:3px 0 8px;font-weight:600;text-transform:uppercase;letter-spacing:.03em;min-height:28px}
.ds{font-size:14px;line-height:1.42;color:#2b2f36;margin-top:9px}
.th{min-height:100px}
.row2{display:flex;gap:16px;margin-top:22px;align-items:stretch}
.row2 .tg{min-height:0}
"""
w('fig2', f2, f2css)

# ================================================== Figure 3 ===================================================
import shutil
COL = {'red': '#d7263d', 'blue': '#2457d6', 'yellow': '#e0a800', 'green': '#1a9850'}
def panel_a():
    """3a from traj.json (schema: scene_note, image, policy_view, rows[{model,label,runs[{instruction,target_color,verdict,reason,tcp_px,last_frame}]}])."""
    if not os.path.exists(TRAJ):
        cols = ''.join(f"""<div class='gc'><div class='gi'><span class='bub q' style='font-size:14px;padding:5px 10px'>&ldquo;pick up the <b style='color:{c}'>{n}</b> cube&rdquo;</span></div>
   {im(f'g_close_end_{n}.png')}<div class='gt2'>lifts the <b>{n}</b> cube</div></div>""" for n, c in (('red', '#c0182b'), ('blue', '#1d3f86'), ('yellow', '#a47d00')))
        return f"""<div class='ptitle'>(a) Same picture, different word</div>
   <div class='psub'>One scene (red, blue and yellow cubes on the table). Only the instruction changes.</div>
   <div style='display:flex;gap:34px;align-items:flex-start'><div style='flex:1.25'>
   <div class='rowlab'><span class='sw' style='background:var(--green)'></span>Final colour model: last frame of each run (earlier replay of one layout)</div>
   <div style='display:flex;gap:12px'>{cols}</div></div><div style='flex:1'>
   <div class='rowlab'><span class='sw' style='background:var(--red)'></span>Plain fine-tune (before the fix)</div>
   <div class='todo'><b>Pending: paths of the plain model on this scene.</b> Not shown until the new same-scene episodes are rendered.<br>
     Measured instead (its own test, 1,000 scenes with 2 to 4 cubes): it grasped the wrong cube in <b>33.8 %</b>, and a shuffled instruction scored about the same as the true one (21 vs 19 of 100).</div></div></div>"""
    T = json.load(open(TRAJ, encoding='utf-8'))
    def cp(rel, name):
        shutil.copy(os.path.join(ROOT, rel), f'assets/{name}'); return name
    scene = cp(T['image'], 'tj_scene.png'); pol = cp(T['policy_view'], 'tj_policy.png')
    W, H = 960, 720
    out = f"""<div class='ptitle'>(a) Same picture, different word</div><div class='psub'>{T['scene_note']} Only the instruction changes.</div>
     <div style='display:flex;gap:14px;align-items:center;margin-bottom:12px'><div style='width:170px;flex:none'>{im(pol)}<div class='tagl' style='margin-top:4px'>what the policy sees</div></div>
     <div style='font-size:13.5px;color:var(--muted);line-height:1.45'>Below: the <b>viewer camera</b> (not the policy input). Lines are the measured fingertip path of each run, drawn in the colour of the cube it was told to pick.</div></div>"""
    cols_html = []
    for ri, row in enumerate(T['rows']):
        dot = 'var(--red)' if row['model'] == 'plain' else 'var(--green)'
        paths = ''
        for run in row['runs']:
            pts = ' '.join(f"{u:.1f},{v:.1f}" for u, v in run['tcp_px'])
            c = COL.get(run['target_color'], '#333')
            paths += f"<polyline points='{pts}' fill='none' stroke='#fff' stroke-width='9' stroke-linejoin='round' stroke-linecap='round' opacity='.85'/><polyline points='{pts}' fill='none' stroke='{c}' stroke-width='5' stroke-linejoin='round' stroke-linecap='round'/>"
        thumbs = ''
        for k, run in enumerate(row['runs']):
            ok = run['verdict'] == 'success'; c = COL.get(run['target_color'], '#333')
            lf = cp(run['last_frame'], f'tj_last_{ri}_{k}.png')
            thumbs += f"""<div class='gc'><div class='gi' style='font-size:13px'>&ldquo;{run['instruction']}&rdquo;</div>{im(lf, f'border:3px solid {c}')}
              <div class='gt2' style='color:{"#13703b" if ok else "#b3162b"};font-weight:700'>{'&#10003; success' if ok else '&#10007; fail'}{'' if ok else ': ' + run['reason']}</div></div>"""
        cols_html.append(f"""<div style='flex:1;min-width:0'><div class='rowlab'><span class='sw' style='background:{dot}'></span>{row['label']}</div>
          <div style='position:relative'><div class='ph'><img src='assets/{scene}'></div><div class='vtag'>viewer camera, not the policy input</div>
          <svg viewBox='0 0 {W} {H}' style='position:absolute;inset:0;width:100%;height:100%'>{paths}</svg></div>
          <div style='display:flex;gap:10px;margin-top:8px'>{thumbs}</div></div>""")
    out += "<div style='display:flex;gap:34px'>" + ''.join(cols_html) + "</div>"
    out += "<div class='ill' style='margin-top:8px'>Paths are measured fingertip positions (one point per camera frame, 16.7 Hz); thumbnails are each run's last frame; the plain model's runs are its real episodes on this scene.</div>"
    return out
PANEL_A = panel_a()
f3 = f"""
<div class='fig' id='fig'>
 <div class='head'><div class='htitle'>SmolVLA tabletop <span>· how it is proven</span></div>{BADGE}</div>
 <div class='g3'>
  <div class='pa'>{PANEL_A}</div>
  <div class='pb'>
   <div class='ptitle'>(b) Success per stage against its pass bar</div>
   <div class='psub'>Each bar is one test. Black line = the bar the stage had to reach. Whiskers = 95 % interval. Stack was stopped on validation scenes (its guard), so it has no test.</div>
   <img src='assets/fig3b.png' style='width:100%'>
   <div class='base'>No-language baseline: failed its own training-scene check (0 / 100), not usable, no bar shown.</div>
  </div>
  <div class='pc'>
   <div class='ptitle'>(c) Does it carry over? Fresh scenes of four kinds</div>
   <div class='psub'>Final evaluation, 1,000 episodes per bar, reported only (no bar). Randomized visuals were never trained on by any model.</div>
   <img src='assets/fig3c.png' style='width:100%'>
  </div>
 </div>
 <div class='caption'><b>Figure 3. How it is proven.</b> (a) One scene, one start, only the sentence changes: the lines are measured fingertip positions of each run, in the colour of the cube it was told to pick. The plain fine-tune misses the red cube, lifts the red one when asked for blue, and succeeds on yellow; the final colour model succeeds three times. One scene illustrates the mechanism; the rates are the 1,000-episode tests (plain 31.5 %, final 96.7 %). (b) Every stage against the pass bar written before its run: single-cube and colour picking pass; sizes, place and stack miss their bars. (c) Seen scenes match each model's test, but randomized visuals break every model, colour/size combinations never trained on make the colour model grasp the wrong cube 64 % of the time, and the place model collapses on reworded pick instructions.</div>
</div>"""
f3css = """
.g3{display:grid;grid-template-columns:1fr 1fr;column-gap:34px;row-gap:22px}
.pa{grid-column:1/3;margin-bottom:8px}
.pb{} .pc{}
.vtag{position:absolute;left:10px;top:10px;font-size:12.5px;font-weight:600;background:rgba(255,255,255,.88);border-radius:6px;padding:3px 8px;color:#1f2328}
.gc{flex:1;min-width:0}.gi{margin-bottom:6px;white-space:nowrap}
.gt2{font-size:13.5px;margin-top:5px;text-align:center}
.rowlab{font-size:13.5px;font-weight:700;margin-bottom:7px;display:flex;align-items:center;gap:7px}
.todo{border:2px dashed #e5a3ab;background:#fdf3f4;border-radius:10px;padding:14px 16px;font-size:14px;line-height:1.5;color:#5b2a31}
.base{margin-top:8px;font-size:13.5px;color:#5b2a31;border:1.5px dashed #c9ccd1;border-radius:8px;padding:7px 11px;display:inline-block}
"""
w('fig3', f3, f3css)

# ================================================== Concept cards ==============================================
def card(n, title, visual, inout, why):
    return f"""<div class='cd'><div class='ct'><span class='cn'>{n}</span>{title}</div><div class='cv'>{visual}</div>
     <div class='cr'><span class='cl'>IN &rarr; OUT</span><span>{inout}</span></div><div class='cr'><span class='cl'>WHY HERE</span><span>{why}</span></div></div>"""
random.seed(3)
xs = [i / 40 for i in range(41)]
def curve(noise):
    return [0.5 + 0.35 * math.sin(2 * math.pi * x * 0.9) * (1 - 0.3 * x) + noise * random.uniform(-1, 1) for x in xs]
def poly(ys, x0, col):
    pts = ' '.join(f"{x0 + x * 120:.1f},{20 + (1 - y) * 80:.1f}" for x, y in zip(xs, ys))
    return f"<polyline points='{pts}' fill='none' stroke='{col}' stroke-width='2.2' stroke-linejoin='round'/>"
flow = "<svg width='100%' viewBox='0 0 520 130'>" + ''.join(poly(curve(n), 10 + k * 128, c) for k, (n, c) in enumerate(((0.55, '#8a8f98'), (0.3, '#6c8fd6'), (0.12, '#4a78d6'), (0.0, '#1d4fc0')))) + \
    ''.join(f"<text x='{70 + k * 128}' y='122' font-size='12' text-anchor='middle' fill='#5d6470'>{t}</text>" for k, t in enumerate(('pure noise', 'less noise', 'almost there', 'planned moves'))) + \
    ''.join(f"<text x='{136 + k * 128}' y='62' font-size='20' text-anchor='middle' fill='#8a8f98'>&rarr;</text>" for k in range(3)) + "</svg><div class='ill'>illustrative</div>"
chunkv = f"""<div style='font-size:13px;color:var(--muted);margin-bottom:4px'>one query predicts 50 moves</div><div class='chunk'>{chunk}</div>
  <div style='display:flex;gap:10px;margin-top:10px;align-items:center;font-size:13px'><b style='color:var(--blue)'>first 10 run (0.2 s)</b><span>&rarr;</span><span>new camera image</span><span>&rarr;</span><span>plan again</span></div>
  <div class='ill' style='margin-top:6px'>50 Hz control; camera 16.7 Hz; the model is asked every 10 control steps</div>"""
pairv = f"""<div style='display:flex;align-items:center;gap:12px'><div style='width:170px;flex:none'>{im('g_side0.png')}<div class='tagl' style='margin-top:4px'>one scene</div></div>
   <div style='flex:1;display:flex;flex-direction:column;gap:8px'>
    <div style='display:flex;align-items:center;gap:8px;font-size:13px'><span style='flex:1'>&ldquo;pick up the <b style='color:#c0182b'>red</b> cube&rdquo;</span><div style='width:96px'>{im('g_close_end_red.png')}</div></div>
    <div style='display:flex;align-items:center;gap:8px;font-size:13px'><span style='flex:1'>&ldquo;pick up the <b style='color:#1d3f86'>blue</b> cube&rdquo;</span><div style='width:96px'>{im('g_close_end_blue.png')}</div></div></div></div>
   <div class='ill' style='margin-top:8px'>real replay of one scene told two ways; in training, the expert is shown doing both</div>"""
probev = """<div class='pr'><div class='pl'>Is the colour inside the model?</div>
  <div class='br'><span>true instruction</span><div style='width:100%;background:var(--green)'></div><b>1.00</b></div>
  <div class='br'><span>shuffled instruction</span><div style='width:40%;background:#b9bdc4'></div><b>0.36 to 0.43</b></div>
  <div class='pl' style='margin-top:12px'>Does the first chunk change with it?</div>
  <div class='br'><span>start noise moves it</span><div style='width:100%;background:var(--orange)'></div><b>1</b></div>
  <div class='br'><span>instruction moves it</span><div style='width:10%;background:var(--red)'></div><b>0.10</b></div></div>
  <div class='ill'>probe on the plain fine-tune (cross-validated); ratio from its first action chunk</div>"""
gatev = """<div class='gs'>
  <div class='gx'>bar first<br><b>&ge; 85 %</b></div><span>&rarr;</span>
  <div class='gx f'>run<br><b>31.5 % FAIL</b></div><span>&rarr;</span>
  <div class='gx'>diagnose,<br><b>one fix</b></div><span>&rarr;</span>
  <div class='gx f'>run<br><b>67.7 % FAIL</b></div></div>
  <div class='ill' style='margin-top:6px'>then a new owner-approved stage: more data &rarr; <b style='color:var(--green);font-style:normal'>96.7 % PASS</b></div>"""
cards = [
 card(1, 'Flow-matching action expert', flow, 'image + words + joint state + noisy chunk &rarr; one denoising step; repeated, noise becomes 50 planned moves', 'It is the part that outputs motion. Which cube it goes to was decided by its start noise until the contrastive loss (card 4).'),
 card(2, 'Action chunking', chunkv, 'one camera image + words &rarr; 50 planned moves, 10 executed', 'Fewer, longer plans keep the arm smooth; the cost is that it cannot react inside a chunk.'),
 card(3, 'Counterfactual pairs', pairv, 'one scene, every nameable cube &rarr; one expert demo per cube', 'Same picture, different word, different demo: the only way to tell the word from the scene.'),
 card(4, 'Instruction-contrastive hinge loss', "<img src='assets/card_hinge.png' style='width:100%'><div style='font-size:13px;margin-top:2px;text-align:center'><b>L = L<sub>true</sub> + &lambda; &middot; mean max(0, m &minus; (L<sub>other</sub> &minus; L<sub>true</sub>))</b>, &lambda; = 1, m &asymp; 4.21</div><div class='ill'>real run of the final colour model: the other-instruction loss stays far above the margin</div>",
      'a noisy chunk scored under the true and under another cube\'s instruction &rarr; a penalty if the true one does not fit better by at least m', 'In words: the motion must fit the true instruction at least m better than another cube\'s instruction. Plain fine-tuning never had to.'),
 card(5, 'Linear probe', probev, 'the model\'s internal features &rarr; can a straight line read the colour?', 'The colour was in the model; the start noise picked the cube. That ruled out the text path and pointed at the loss.'),
 card(6, 'Staged gates, one fix each', gatev, 'a stage, its pre-set bar, one diagnosed fix &rarr; pass or stop', 'Keeps a failed stage from being tuned until it passes; moving a bar is a logged deviation.'),
]
cf = f"""
<div class='fig' id='fig'>
 <div class='head'><div class='htitle'>SmolVLA tabletop <span>· concept cards</span></div>{BADGE}</div>
 <div class='cg'>{''.join(cards)}</div>
 <div class='caption'><b>Concept cards.</b> One card per idea the pipeline in Figure 2 depends on: what it looks like, what goes in and comes out, and why it matters here. Numbers are from the project's own runs; pictures marked &ldquo;illustrative&rdquo; are schematics.</div>
</div>"""
cfcss = """
.cg{display:grid;grid-template-columns:repeat(3,1fr);gap:26px}
.cd{border:1.5px solid #dfe2e6;border-radius:14px;padding:16px 18px;display:flex;flex-direction:column}
.ct{font-size:19px;font-weight:700;display:flex;align-items:center;margin-bottom:12px}
.cn{display:inline-flex;width:26px;height:26px;border-radius:50%;background:#1f2328;color:#fff;font-size:13.5px;align-items:center;justify-content:center;margin-right:10px}
.cv{background:#f6f7f9;border-radius:10px;padding:12px;height:250px;margin-bottom:14px;display:flex;flex-direction:column;justify-content:center}
.cr{display:flex;gap:14px;font-size:14.5px;line-height:1.45;margin-top:6px}
.cl{flex:none;width:78px;font-size:11.5px;font-weight:700;letter-spacing:.05em;color:var(--muted);padding-top:3px}
.chunk{display:grid;grid-template-columns:repeat(25,1fr);gap:3px}
.chunk i{height:22px;border-radius:3px;background:#e4e6ea;display:block}.chunk i.on{background:var(--blue)}
.pr{font-size:13px}.pl{font-weight:700;margin-bottom:5px}
.br{display:grid;grid-template-columns:130px 1fr 78px;align-items:center;gap:8px;margin:4px 0}
.br div{height:15px;border-radius:3px}.br b{font-size:13px}
.gs{display:flex;align-items:center;gap:5px;font-size:12.5px;line-height:1.35}
.gx{flex:1;border:1.5px solid #c9ccd1;background:#fff;border-radius:8px;padding:7px 6px;text-align:center}
.gx.f{border-color:#e0a0a8;background:#fdf3f4}
"""
w('cards', cf, cfcss)
