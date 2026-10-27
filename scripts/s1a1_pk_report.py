"""Post-kill diagnosis (not a gate attempt): summarise the four paired variants -> results/s1a1_postkill_diag.json."""
import json
from pathlib import Path

import numpy as np

D = Path(__file__).resolve().parents[1] / "results"
V = {"n20": "N20 default", "n10": "N10", "n5": "N5", "n20zero": "N20 zero-noise"}
R = {k: {r["index"]: r for r in json.loads((D / f"s1a1_pk/s1a1_pk_{k}_s21600.json").read_text())["episodes"]} for k in V}
size = lambda r: r["target"].split("_")[-1]
out = {"note": "post-kill diagnosis - not a gate attempt, no decision", "seed": 21600, "episodes": "0-299", "variants": {}, "paired": {}}
ix = sorted(R["n20"])
assert all(sorted(R[k]) == ix for k in V)
for k, name in V.items():
    rs = [R[k][i] for i in ix]
    d = {"n": len(rs)}
    for m in ("success", "slam", "jam", "drop", "early_close", "wrong_object", "wrong_grasp"):
        d[m] = sum(r[m] for r in rs) / len(rs)
    d["no_flag_fail"] = sum((not r["success"]) and not any(r[m] for m in ("slam", "jam", "drop", "early_close", "wrong_object", "wrong_grasp")) for r in rs) / len(rs)
    d["by_size"] = {s: {"n": sum(size(r) == s for r in rs), "success": float(np.mean([r["success"] for r in rs if size(r) == s]))}
                    for s in sorted({size(r) for r in rs})}
    out["variants"][k] = d
a = np.array([R["n20"][i]["success"] for i in ix], int)
for k in V:
    if k == "n20":
        continue
    b = np.array([R[k][i]["success"] for i in ix], int)
    out["paired"][k + "_vs_n20"] = {"diff_points": 100 * float((b - a).mean()), "gained": int(((b == 1) & (a == 0)).sum()), "lost": int(((b == 0) & (a == 1)).sum())}
yaw = [k for k in R["n20"][0] if "yaw" in k]
out["grasp_yaw_logged"] = bool(yaw)
(D / "s1a1_postkill_diag.json").write_text(json.dumps(out, indent=1))
print(json.dumps(out, indent=1))
