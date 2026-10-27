"""Contract v2 h: every instruction the task generator renders fits SmolVLA's 48-token language budget (policy venv).

Counted exactly as SmolVLA's preprocessor sees it: the task string + the "\\n" its NewLineTaskProcessorStep appends,
tokenized by the SmolVLM2 tokenizer with the processor's default special tokens and NO truncation (the processor
truncates silently at 48, which would cut the destination off a long instruction). >= 2,000 sampled specs per stage
and split, plus every template filled with the longest noun phrases."""

from __future__ import annotations

import numpy as np
import pytest

transformers = pytest.importorskip("transformers")

from vla_common.tasks import TEMPLATES, instruction, sample_spec  # noqa: E402

MAX_TOKENS = 48  # smolvla_base tokenizer_max_length
VLM = "HuggingFaceTB/SmolVLM2-500M-Video-Instruct"
STAGES = ("s1", "s2", "s3a", "s3b", "s4")
SPLITS = ("train", "heldout_comp", "paraphrase")


@pytest.fixture(scope="module")
def n_tokens():
    tok = transformers.AutoTokenizer.from_pretrained(VLM)
    return lambda texts: [len(ids) for ids in tok([t + "\n" for t in texts], truncation=False)["input_ids"]]


@pytest.mark.parametrize("stage", STAGES)
@pytest.mark.parametrize("split", SPLITS)
def test_sampled_instructions_fit(n_tokens, stage, split):
    rng = np.random.default_rng(10 * STAGES.index(stage) + SPLITS.index(split))
    texts = [instruction(sample_spec(stage, split, rng), rng, split) for _ in range(2000)]
    counts = n_tokens(texts)
    worst = int(np.argmax(counts))
    assert counts[worst] <= MAX_TOKENS, (counts[worst], texts[worst])


def test_every_template_with_the_longest_phrases_fits(n_tokens):
    t, d = "the small yellow cube", "the large yellow cube"
    texts = [tpl.format(t=t, d=d) for pool in TEMPLATES.values() for tpls in pool.values() for tpl in tpls]
    counts = n_tokens(texts)
    assert max(counts) <= MAX_TOKENS, max(zip(counts, texts))
