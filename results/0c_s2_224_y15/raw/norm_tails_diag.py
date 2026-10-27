"""Lead diagnosis (not gate code): where do the |z| > 10 rows of the s2 norm table sit, and does the Stage 1 V15 set
have the same tails? z uses each dataset's own train-split mean / std (as the trainer). Prints JSON."""
import glob, json, sys
import numpy as np
import pandas as pd

out = {}
for name in sys.argv[1:]:
    root = f"/data/datasets/{name}"
    split = json.load(open(f"{root}/meta/split_113.json"))
    val = set(split["val"])
    fs = sorted(glob.glob(f"{root}/data/**/*.parquet", recursive=True))
    df = pd.concat([pd.read_parquet(f, columns=["episode_index", "frame_index", "action", "observation.state"]) for f in fs])
    df = df[~df["episode_index"].isin(val)]
    A = np.stack(df["action"].to_numpy()).astype(np.float64)
    S = np.stack(df["observation.state"].to_numpy()).astype(np.float64)
    t = df["frame_index"].to_numpy()
    res = {"rows": len(df)}
    for lab, X, dims in (("action", A, [3]), ("state", S, [16, 17])):
        mu, sd = X.mean(0), X.std(0)
        for d in dims:
            z = (X[:, d] - mu[d]) / sd[d]
            m = np.abs(z) > 10
            tt = t[m]
            res[f"{lab}{d}"] = {"mean": mu[d], "std": sd[d], "frac_beyond_10": float(m.mean()), "n": int(m.sum()),
                                "t_hist": {"t<10": int((tt < 10).sum()), "10<=t<50": int(((tt >= 10) & (tt < 50)).sum()),
                                           "50<=t<150": int(((tt >= 50) & (tt < 150)).sum()), "t>=150": int((tt >= 150).sum())},
                                "values_beyond": sorted(set(np.round(X[m, d], 2).tolist()))[:12],
                                "episodes_with_any": int(len(set(df["episode_index"].to_numpy()[m])))}
    out[name] = res
print("TAILS=" + json.dumps(out, default=float))
