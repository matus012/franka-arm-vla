"""Task 003 step 0 (policy venv, CPU): token count of every instruction the generator can emit vs tokenizer_max_length.

Every train / paraphrase template of every task (vla_common.tasks.TEMPLATES) filled with every target / destination
noun phrase (`the X cube`, `the S X cube`, `the X pad`) goes through the checkpoint's own saved preprocessor steps
(smolvla_new_line_processor + tokenizer_processor, config read from policy_preprocessor.json) and through the same
tokenizer untruncated. Reports max / min tokens vs `tokenizer_max_length` of the checkpoint config.json.

  python scripts/g3_prompt_len.py --config <ckpt>/config.json --preprocessor <ckpt>/policy_preprocessor.json \
      --out results/g3_diag/prompt_len.json
"""

from __future__ import annotations

import argparse
import itertools
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from vla_common.catalog import COLORS, SIZES  # noqa: E402
from vla_common.tasks import TEMPLATES  # noqa: E402


def all_prompts() -> list[dict]:
    """Every (split, task, template, filled text) the generator can produce."""
    cubes = [f"the {c} cube" for c in COLORS] + [f"the {s} {c} cube" for c in COLORS for s in SIZES]
    pads = [f"the {c} pad" for c in COLORS]
    out = []
    for split, pool in TEMPLATES.items():
        for task, tpls in pool.items():
            for tpl in tpls:
                if task == "s1":
                    fills = [("", "")]
                elif task == "pick":
                    fills = [(t, "") for t in cubes]
                elif task == "place":
                    fills = list(itertools.product(cubes, pads))
                else:
                    fills = list(itertools.product(cubes, cubes))
                for t, d in fills:
                    out.append({"split": split, "task": task, "template": tpl, "text": tpl.format(t=t, d=d)})
    return out


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--config", type=Path, required=True)
    p.add_argument("--preprocessor", type=Path, required=True)
    p.add_argument("--out", type=Path, required=True)
    a = p.parse_args()

    from lerobot.processor.newline_task_processor import NewLineTaskProcessorStep
    from lerobot.processor import TokenizerProcessorStep
    from transformers import AutoTokenizer

    max_len = int(json.loads(a.config.read_text())["tokenizer_max_length"])
    steps = {s["registry_name"]: s["config"] for s in json.loads(a.preprocessor.read_text())["steps"]}
    assert "smolvla_new_line_processor" in steps, list(steps)
    tc = steps["tokenizer_processor"]
    tok_step = TokenizerProcessorStep(tokenizer_name=tc["tokenizer_name"], max_length=tc["max_length"],
                                      task_key=tc["task_key"], padding_side=tc["padding_side"],
                                      padding=tc["padding"], truncation=tc["truncation"])
    nl = NewLineTaskProcessorStep()
    raw = AutoTokenizer.from_pretrained(tc["tokenizer_name"])
    rows = []
    for r in all_prompts():
        text = nl.complementary_data({"task": r["text"]})["task"]  # what the trainer / server tokenizes
        enc = tok_step._tokenize_text([text])
        n_proc = int(enc["attention_mask"].sum())
        n_raw = len(raw(text)["input_ids"])
        rows.append(r | {"sent": text, "tokens_pipeline": n_proc, "tokens_untruncated": n_raw})
    n = [r["tokens_untruncated"] for r in rows]
    longest = max(rows, key=lambda r: r["tokens_untruncated"])
    res = {"tokenizer_max_length_config": max_len, "tokenizer_step_config": tc, "n_prompts": len(rows),
           "max_tokens": max(n), "min_tokens": min(n), "longest": longest,
           "over_max": [r for r in rows if r["tokens_untruncated"] > max_len],
           "pipeline_equals_untruncated": all(r["tokens_pipeline"] == r["tokens_untruncated"] for r in rows),
           "per_split_task_max": {f"{s}/{t}": max(r["tokens_untruncated"] for r in rows if (r["split"], r["task"]) == (s, t))
                                  for s, t in sorted({(r["split"], r["task"]) for r in rows})},
           "example_tokens": raw.convert_ids_to_tokens(raw(longest["sent"])["input_ids"])}
    a.out.parent.mkdir(parents=True, exist_ok=True)
    a.out.write_text(json.dumps(res | {"rows": rows}, indent=1))
    print("PROMPT_LEN=" + json.dumps({k: res[k] for k in ("tokenizer_max_length_config", "n_prompts", "max_tokens",
                                                            "min_tokens", "pipeline_equals_untruncated")}
                                     | {"n_over_max": len(res["over_max"]), "longest": longest["sent"]}))


if __name__ == "__main__":
    main()
