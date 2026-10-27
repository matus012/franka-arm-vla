"""Task 003 instrument proof (lead review F4), policy venv: the g3_probe feature hooks capture the tensors the model uses.

  (1) final norm of the captured L16 residual == the prefix output of vlm_with_expert.forward (prefill);
  (2) the captured L4 residual fed through layer 4..15 is not re-checked; instead the L16 capture is checked by (1) and
      every capture is the input of a module called exactly once per prefill (call counts);
  (3) the K half of the kv feature at layer index 3 / 7 / 11 / 15 == the expert's k_proj output observed by a forward
      hook during a real predict_action_chunk on the same observation (first denoising call), mean over valid prefix.

  python scripts/g3_hook_check.py --ckpt checkpoints/s1a1_U/pretrained_model --dump <npz> --out <json>
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import g3_probe as G  # noqa: E402


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--ckpt", required=True)
    p.add_argument("--dump", type=Path, required=True)
    p.add_argument("--out", type=Path, required=True)
    a = p.parse_args()
    from lerobot.policies.common.vla_utils import make_att_2d_masks
    from lerobot.utils.constants import OBS_LANGUAGE_ATTENTION_MASK, OBS_LANGUAGE_TOKENS

    S = G.load_samples([a.dump])
    ex = G.Extractor(a.ckpt)
    sl = slice(0, 8)
    obs = ex.obs(S["side"][sl], S["wrist"][sl], S["state"][sl], S["instruction"][sl].tolist())
    tm = ex.vwe.get_vlm_model().text_model
    calls = {L: 0 for L in G.LAYERS}
    hs = [(tm.norm if L == 16 else tm.layers[L].input_layernorm).register_forward_pre_hook(
        lambda m, i, L=L: calls.__setitem__(L, calls[L] + 1)) for L in G.LAYERS]
    feats = ex.features(obs)
    for h in hs:
        h.remove()
    res = {"calls_per_prefill": calls}
    # (1) recompute the prefill output and compare with norm(captured L16)
    with torch.no_grad():
        images, img_masks = ex.pol.prepare_images(obs)
        state = ex.pol.prepare_state(obs)
        embs, pad, att = ex.model.embed_prefix(images, img_masks, obs[OBS_LANGUAGE_TOKENS],
                                               obs[OBS_LANGUAGE_ATTENTION_MASK].bool(), state=state)
        (out0, _), _ = ex.vwe.forward(attention_mask=make_att_2d_masks(pad, att), position_ids=torch.cumsum(pad, 1) - 1,
                                      past_key_values=None, inputs_embeds=[embs, None], use_cache=True)
        res["final_norm_of_L16_capture_maxabs"] = float((tm.norm(ex.cap[16]) - out0).abs().max())
    # (3) k_proj outputs during a real chunk
    seen: dict[int, torch.Tensor] = {}
    hk = [ex.vwe.lm_expert.layers[L - 1].self_attn.k_proj.register_forward_hook(
        lambda m, i, o, L=L: seen.setdefault(L, o.detach())) for L in G.LAYERS]
    noise = torch.randn(8, ex.pol.config.chunk_size, ex.pol.config.max_action_dim)
    ex.chunk(obs, noise)
    for h in hk:
        h.remove()
    padf = pad.float()[..., None]
    for L in G.LAYERS:
        k_mean = ((seen[L] * padf).sum(1) / padf.sum(1)).float().cpu().numpy()
        kdim = k_mean.shape[1]
        res[f"kv_L{L}_K_half_vs_kproj_hook_maxabs"] = float(np.abs(feats[f"kv_L{L}"][:, :kdim] - k_mean).max())
    res["pass"] = (all(v == 1 for v in calls.values()) and res["final_norm_of_L16_capture_maxabs"] < 1e-4
                   and all(res[f"kv_L{L}_K_half_vs_kproj_hook_maxabs"] < 1e-4 for L in G.LAYERS))
    a.out.write_text(json.dumps(res, indent=1))
    print("G3_HOOKS=" + json.dumps(res))


if __name__ == "__main__":
    main()
