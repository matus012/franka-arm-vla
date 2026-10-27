"""Builds every image asset for the 113 figure set from real 113 outputs. Run from figures/src with the sim venv:
    ../../.venv/Scripts/python assets.py
"""
import json, os
import numpy as np, imageio.v3 as iio
from PIL import Image
import matplotlib; matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib import font_manager

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
A = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'assets'); os.makedirs(A, exist_ok=True)
for f in ('Inter', 'Segoe UI', 'DejaVu Sans'):
    if any(f == x.name for x in font_manager.fontManager.ttflist):
        plt.rcParams['font.family'] = f; break
INK, MUTED, BLUE, GREY, GREEN, RED, ORANGE = '#1f2328', '#5d6470', '#3b6fd6', '#8a8f98', '#1a9850', '#d7263d', '#e07b12'
plt.rcParams.update({'axes.spines.top': False, 'axes.spines.right': False, 'text.color': INK, 'axes.labelcolor': INK,
                     'xtick.color': INK, 'ytick.color': INK})
P = lambda *a: os.path.join(ROOT, *a)

# ---------------- video frames: panels of the final_010 triptychs (close-up 0:320 | side 324:564 | wrist 568:808) -------------
def frames(name):
    return list(iio.imiter(P('videos', 'final_010', name + '.mp4')))
def panel(fr, which):
    x0, x1 = {'close': (0, 320), 'side': (324, 564), 'wrist': (568, 808)}[which]
    return fr[:, x0:x1]
def save(arr, name): Image.fromarray(np.ascontiguousarray(arr)).save(f'{A}/{name}')

runs = {c: frames(f'02_2m_grounding_layout14_{c}_ep3{i}') for c, i in (('red', 1), ('blue', 2), ('yellow', 3))}
b = runs['blue']
save(panel(b[0], 'side'), 'f1_side_in.png'); save(panel(b[0], 'wrist'), 'f1_wrist_in.png')
for k, t in enumerate((0, 38, 76, 116)): save(panel(b[t], 'close'), f'f1_close_{k}.png')
for c, r in runs.items():
    save(panel(r[0], 'close'), f'g_close0_{c}.png'); save(panel(r[116], 'close'), f'g_close_end_{c}.png')
    save(panel(r[116], 'side'), f'g_side_end_{c}.png'); save(panel(r[116], 'wrist'), f'g_wrist_end_{c}.png')
save(panel(runs['red'][0], 'side'), 'g_side0.png')

# ---------------- scripted expert (training data): raw expert videos, close-up panel below the text header ---------------------
for task, seed in (('pick', 21700), ('place', 21701), ('stack', 21702)):
    v = list(iio.imiter(P('videos', 'raw', f'expert_{task}_{seed}.mp4')))
    h, w = v[0].shape[:2]
    save(v[len(v) * 2 // 3][100:, :w // 3 + 4], f'expert_{task}.png')   # first ~100 px = text header; first panel = close-up
    print('expert frame', task, v[0].shape)

# ---------------- dataset contact sheet panels (decoded from the LeRobot dataset) --------------------------------------------
sh = np.array(Image.open(P('results', 'frames', '0c_s2a_pair_224_y15', '0c_s2a_pair_rl224_y15.png')).convert('RGB'))
save(sh[101:276, 163:337], 'ds_side_t0.png'); save(sh[388:562, 163:337], 'ds_wrist_t0.png'); save(sh[101:276, 996:1170], 'ds_side_end.png')

# ---------------- training: real loss curves of the final colour model (2M, 68,880 steps) -------------------------------------
st = [json.loads(l) for l in open(P('results', 's2m', 'steps_noindex.jsonl'))]
step = np.array([x['step'] for x in st]); lt = np.array([x['loss'] for x in st]); lo = np.array([x['l_other'] for x in st])
act = np.array([x['hinge_active'] for x in st])
sm = lambda a, k=1500: np.convolve(a, np.ones(k) / k, mode='valid')
s2 = step[:len(sm(step))] + 750
def curves(path, size, m=None, lab=True, fs=9):
    fig, ax = plt.subplots(figsize=size, dpi=200)
    ax.semilogy(s2, sm(lo), color=ORANGE, lw=2, label="loss with another cube's instruction")
    ax.semilogy(s2, sm(lt), color=BLUE, lw=2, label='loss with the true instruction')
    if m: ax.axhline(m, color=INK, ls=':', lw=1); ax.text(s2[-1], m * 1.25, f'margin m = {m}', ha='right', fontsize=fs, color=INK)
    ax.set_xlabel('training step', fontsize=fs); ax.set_ylabel('flow loss (log)', fontsize=fs); ax.tick_params(labelsize=fs - 1)
    ax.set_xticks([0, 34440, 68880]); ax.set_xticklabels(['0', '34k', '69k'])
    ax.set_ylim(3e-3, 400)
    if lab: ax.legend(fontsize=fs - 1, frameon=False, loc='center right', bbox_to_anchor=(1, 0.3))
    fig.tight_layout(pad=0.4); fig.savefig(path, facecolor='white'); plt.close(fig)
curves(f'{A}/train_thumb.png', (3.3, 2.15), lab=False, fs=8)
curves(f'{A}/card_hinge.png', (4.8, 1.75), m=4.21, fs=8.5)
print('hinge active share, last 10k steps:', round(float(act[-10000:].mean()), 5), '| mean other-minus-true gap:', round(float(np.mean([x['gap'] for x in st[-10000:]])), 1))

# ---------------- Fig 3b: headline bars per stage ----------------------------------------------------------------------------
# (label, rate %, ci lo, ci hi, bar %, verdict, extra) - README.md results tables / GATES.md RESULT lines (see figures/README.md)
STAGES = [
    ('One cube\npick', 96.8, 95.5, 97.7, 90, 'PASS', ''),
    ('Colours,\nplain\nfine-tune', 31.5, 28.7, 34.4, 85, 'FAIL', 'wrong cube\n33.8 %'),
    ('Colours,\ncontrastive', 67.7, 64.7, 70.5, 85, 'FAIL', 'wrong cube\n3.5 %'),
    ('Colours,\ncontrastive\n+ one fix', 77.6, 74.9, 80.1, 85, 'FAIL', 'wrong cube\n3.9 %'),
    ('Colours,\nmore data\n(final)', 96.7, 95.4, 97.6, 85, 'PASS', 'wrong cube\n0.1 %'),
    ('Sizes\n(best)', 79.4, 76.8, 81.8, 85, 'FAIL', 'wrong cube\n2.6 %'),
    ('Place\non pad', 72.7, 69.9, 75.4, 80, 'FAIL', 'wrong pad\n0.2 %'),
    ('Stack\n(validation,\nstopped)', 40.7, 35.3, 46.3, 70, 'FAIL', ''),
]
fig, ax = plt.subplots(figsize=(8.9, 4.3), dpi=200)
for i, (lab, r, lo_, hi_, bar, v, ex) in enumerate(STAGES):
    col = GREEN if v == 'PASS' else RED
    ax.bar(i, r, width=0.58, color=col, alpha=0.82)
    ax.errorbar(i, r, yerr=[[r - lo_], [hi_ - r]], color=INK, capsize=3, lw=1.2)
    ax.plot([i - 0.42, i + 0.42], [bar, bar], color=INK, lw=2.2)
    ax.text(i, 4, f'{r:.1f}', ha='center', color='white', fontsize=10.5, fontweight='bold')
    ax.text(i, 104, v, ha='center', fontsize=10, fontweight='bold', color=col)
    if ex: ax.text(i, -30, ex, ha='center', va='top', fontsize=8, color=MUTED)
ax.set_xticks(range(len(STAGES))); ax.set_xticklabels([s[0] for s in STAGES], fontsize=8.5)
ax.set_ylim(0, 110); ax.set_xlim(-0.6, len(STAGES) - 0.4); ax.set_ylabel('success rate, %', fontsize=10)
ax.plot([], [], color=INK, lw=2.2, label='pass bar for that stage (set before the run)')
ax.errorbar([], [], yerr=[], color=INK, capsize=3, label='95 % interval')
ax.legend(loc='upper left', bbox_to_anchor=(0.0, 1.13), fontsize=9, frameon=False, ncol=2)
fig.subplots_adjust(bottom=0.31, top=0.88, left=0.08, right=0.99); fig.savefig(f'{A}/fig3b.png', facecolor='white'); plt.close(fig)

# ---------------- Fig 3c: transfer checks from results/final_eval.json ---------------------------------------------------------
fe = json.load(open(P('results', 'final_eval.json')))['cells']
print('final_eval cells:', list(fe))
for k, c in fe.items():
    print(k, c.get('label'), c.get('verdict'), {s: round(c['splits'][s]['success']['rate'] * 100, 1) for s in c['splits'] if 'success' in c['splits'][s]})

MODELS = [('s1', 'One cube (PASS)', GREY), ('2m', 'Colours (PASS)', GREEN), ('3apick', 'Place model asked to pick (FAIL)', ORANGE), ('3aplace', 'Place model asked to place (FAIL)', RED)]
SPLITS = [('a', 'Seen scenes'), ('b', 'Randomized visuals\n(never trained on)'), ('c', 'Held-out colour/size\ncombinations'), ('d', 'Reworded\ninstructions')]
fig, ax = plt.subplots(figsize=(8.9, 4.3), dpi=200)
w = 0.2; leg = []
from matplotlib.patches import Patch
for j, (k, lab, col) in enumerate(MODELS):
    for i, (s, _) in enumerate(SPLITS):
        x = i + (j - 1.5) * w
        c = fe[k]['splits'][s]
        if 'success' not in c:
            ax.text(x, 2, 'n/a', ha='center', fontsize=7.5, color=MUTED, rotation=90, va='bottom'); continue
        r, (lo_, hi_) = c['success']['rate'] * 100, [v * 100 for v in c['success']['ci95']]
        ax.bar(x, r, width=w * 0.9, color=col, alpha=0.9)
        ax.errorbar(x, r, yerr=[[r - lo_], [hi_ - r]], color=INK, capsize=2, lw=0.9)
        ax.text(x, hi_ + 1.6, f'{r:.0f}' if r >= 10 else f'{r:.1f}', ha='center', fontsize=8, color=INK)
    leg.append(Patch(color=col, label=lab))
ax.set_xticks(range(len(SPLITS))); ax.set_xticklabels([s[1] for s in SPLITS], fontsize=9)
ax.set_ylim(0, 112); ax.set_ylabel('success rate, %  (1,000 episodes per bar)', fontsize=9.5)
ax.legend(handles=leg, loc='upper center', bbox_to_anchor=(0.5, 1.16), ncol=2, fontsize=8.5, frameon=False)
w2m = fe['2m']['splits']['c']['wrong_object']
ax.annotate(f"wrong cube in {w2m['rate']*100:.1f} % of episodes", xy=(2 - 0.5 * w, 27), xytext=(1.55, 55), fontsize=8.5, color=INK,
            arrowprops=dict(arrowstyle='->', color=INK, lw=1), ha='center')
fig.subplots_adjust(bottom=0.14, top=0.88, left=0.08, right=0.99); fig.savefig(f'{A}/fig3c.png', facecolor='white'); plt.close(fig)
print('2M held-out wrong-object rate', w2m)
