"""Tasks, instruction templates, labels, held-out sets and the spec sampler (CONTRACT: Tasks and instructions)."""

from __future__ import annotations

import json
import re
from dataclasses import asdict, dataclass

import numpy as np

from .catalog import ALL_CUBES, COLORS, SIZES, CubeId

TASKS = ("pick", "place", "stack")
# 2 cubes never the manipulated cube in training (may be distractors / stack bottoms); different colour and size.
HELD_OUT_TARGETS: tuple[CubeId, ...] = (("green", "large"), ("yellow", "small"))
SEEN_CUBES: tuple[CubeId, ...] = tuple(c for c in ALL_CUBES if c not in HELD_OUT_TARGETS)
# (cube, pad colour) and (top, bottom) pairs among seen targets, never trained.
HELD_OUT_PLACE_PAIRS: tuple[tuple[CubeId, str], ...] = (
    (("red", "small"), "blue"), (("blue", "large"), "yellow"), (("red", "large"), "green"))
HELD_OUT_STACK_PAIRS: tuple[tuple[CubeId, CubeId], ...] = (
    (("red", "small"), ("blue", "large")), (("blue", "small"), ("red", "large")), (("yellow", "large"), ("green", "small")))
P_SIZE_UNFORCED = 0.3  # chance the size is stated although no other cube on the table shares the colour


def STACK_ALLOWED(top: CubeId, bottom: CubeId) -> bool:
    # Any size combination is allowed; revisit only if physics (0A.1) shows a large-on-small stack is unstable.
    return True


@dataclass
class TaskSpec:
    task: str  # "pick" | "place" | "stack"
    target: CubeId
    dest: str | CubeId | None  # pad colour (place), bottom cube (stack), None (pick)
    cubes: list[CubeId]
    pads: list[str]  # always all 4 pad colours (contract v2 k): the scene alone must not reveal the task

    def to_json(self) -> str:
        return json.dumps(asdict(self))

    @staticmethod
    def from_json(s: str) -> "TaskSpec":
        d = json.loads(s)
        dest = tuple(d["dest"]) if isinstance(d["dest"], list) else d["dest"]
        return TaskSpec(d["task"], tuple(d["target"]), dest, [tuple(c) for c in d["cubes"]], list(d["pads"]))


# ---- templates: {t} = target noun phrase, {d} = destination noun phrase ----
TEMPLATES: dict[str, dict[str, tuple[str, ...]]] = {
    "train": {
        "s1": ("pick up the cube", "pick the cube up", "pick up the cube from the table"),
        "pick": ("pick up {t}", "pick {t} up", "pick up {t} from the table", "please pick up {t}"),
        "place": ("put {t} on {d}", "place {t} on {d}", "move {t} onto {d}", "please put {t} on {d}"),
        "stack": ("stack {t} on {d}", "stack {t} on top of {d}", "stack {t} onto {d}", "please stack {t} on {d}"),
    },
    "paraphrase": {
        "s1": ("grab the cube", "lift the cube off the table", "get hold of the cube"),
        "pick": ("grab {t}", "lift {t} off the table", "get hold of {t}", "can you grab {t}"),
        "place": ("set {t} onto {d}", "place {t} on top of {d}", "set {t} down on {d}", "drop {t} onto {d}",
                  "carry {t} over to {d} and put it down"),
        "stack": ("put {t} on top of {d}", "set {t} on top of {d}", "pile {t} on {d}", "lift {t} and put it on {d}"),
    },
}
_FILLER = {"please", "can", "you", "the", "a", "it"}


def skeleton(template: str) -> tuple[str, ...]:
    """Normalised template skeleton: placeholders -> X, filler words dropped."""
    words = re.sub(r"\{[td]\}", "X", template.lower()).split()
    return tuple(w for w in words if w not in _FILLER)


def _np(cube: CubeId, spec: TaskSpec, rng: np.random.Generator) -> str:
    collide = sum(1 for c in spec.cubes if c[0] == cube[0]) > 1
    say_size = rng.random() < P_SIZE_UNFORCED or collide
    return f"the {cube[1]} {cube[0]} cube" if say_size else f"the {cube[0]} cube"


def instruction(spec: TaskSpec, rng: np.random.Generator, split: str = "train") -> str:
    """Template text. split in {"train", "heldout_comp"} -> train templates; "paraphrase" -> the disjoint set."""
    pool = TEMPLATES["paraphrase" if split == "paraphrase" else "train"]
    if len(spec.cubes) == 1 and spec.task == "pick":  # Stage 1: a single cube, "the cube" is unique
        ts = pool["s1"]
        return ts[int(rng.integers(len(ts)))]
    ts = pool[spec.task]
    tpl = ts[int(rng.integers(len(ts)))]
    t = _np(spec.target, spec, rng)
    if spec.task == "place":
        d = f"the {spec.dest} pad"
    elif spec.task == "stack":
        d = _np(spec.dest, spec, rng)
    else:
        d = ""
    return tpl.format(t=t, d=d)


def labels(spec: TaskSpec) -> np.ndarray:
    """(task, target colour, target size, destination): destination = pad 0-3, bottom cube 4-11, none 12."""
    if spec.task == "pick":
        dest = 12
    elif spec.task == "place":
        dest = COLORS.index(spec.dest)
    else:
        dest = 4 + ALL_CUBES.index(tuple(spec.dest))
    return np.array([TASKS.index(spec.task), COLORS.index(spec.target[0]), SIZES.index(spec.target[1]), dest], dtype=np.int64)


# ---- independent parser (tests only) ----
_NP_RE = re.compile(r"\bthe (?:(small|large) )?(?:(red|green|blue|yellow) )?(cube|pad)\b")


def _parse(instr: str) -> list[tuple[str, str, str]]:
    return _NP_RE.findall(instr.lower())  # [(size, color, kind)] in order of appearance, "" = not stated


def _resolve(spec: TaskSpec, np_: tuple[str, str, str]) -> list:
    size, color, kind = np_
    if kind == "pad":
        return [p for p in spec.pads if color in ("", p)]
    return [c for c in spec.cubes if color in ("", c[0]) and size in ("", c[1])]


def matches(spec: TaskSpec, instruction_: str) -> list[CubeId]:
    """Cubes on the table the instruction's first noun phrase refers to (the manipulated cube)."""
    nps = _parse(instruction_)
    return _resolve(spec, nps[0]) if nps and nps[0][2] == "cube" else []


def matches_dest(spec: TaskSpec, instruction_: str) -> list:
    """Destination candidates: pad colours (place), cubes (stack), [] for pick / no second noun phrase."""
    nps = _parse(instruction_)
    return _resolve(spec, nps[1]) if len(nps) > 1 else []


# ---- sampling ----
def _pick(rng: np.random.Generator, seq):
    return seq[int(rng.integers(len(seq)))]


def _fill_cubes(rng: np.random.Generator, must: list[CubeId]) -> list[CubeId]:
    n = int(rng.integers(max(2, len(must)), 6))
    rest = [c for c in ALL_CUBES if c not in must]
    out = list(must) + [rest[i] for i in rng.permutation(len(rest))[: n - len(must)]]
    return [out[i] for i in rng.permutation(len(out))]


def _sample_pick(rng: np.random.Generator, split: str) -> TaskSpec:
    t = _pick(rng, HELD_OUT_TARGETS if split == "heldout_comp" else SEEN_CUBES)
    return TaskSpec("pick", t, None, _fill_cubes(rng, [t]), list(COLORS))


def _sample_pick_colours_only(rng: np.random.Generator, split: str) -> TaskSpec:
    """s2a (Stage 2 fallback): 2-4 cubes, one size per scene, distinct colours (colour alone names the target)."""
    t = _pick(rng, HELD_OUT_TARGETS if split == "heldout_comp" else SEEN_CUBES)
    n = int(rng.integers(2, 5))
    others = [c for c in COLORS if c != t[0]]
    cubes = [t] + [(others[i], t[1]) for i in rng.permutation(len(others))[: n - 1]]
    return TaskSpec("pick", t, None, [cubes[i] for i in rng.permutation(n)], list(COLORS))


def _sample_place(rng: np.random.Generator, split: str) -> TaskSpec:
    if split == "heldout_comp":
        if rng.random() < 0.5:
            t, p = _pick(rng, HELD_OUT_TARGETS), _pick(rng, COLORS)
        else:
            t, p = _pick(rng, HELD_OUT_PLACE_PAIRS)
    else:
        while True:
            t, p = _pick(rng, SEEN_CUBES), _pick(rng, COLORS)
            if (t, p) not in HELD_OUT_PLACE_PAIRS:
                break
    return TaskSpec("place", t, p, _fill_cubes(rng, [t]), list(COLORS))


def _sample_stack(rng: np.random.Generator, split: str) -> TaskSpec:
    while True:
        if split == "heldout_comp":
            if rng.random() < 0.5:
                t, b = _pick(rng, HELD_OUT_TARGETS), _pick(rng, ALL_CUBES)
            else:
                t, b = _pick(rng, HELD_OUT_STACK_PAIRS)
        else:
            t, b = _pick(rng, SEEN_CUBES), _pick(rng, ALL_CUBES)  # bottom may be a held-out cube
            if (t, b) in HELD_OUT_STACK_PAIRS:
                continue
        if t != b and STACK_ALLOWED(t, b):
            break
    return TaskSpec("stack", t, b, _fill_cubes(rng, [t, b]), list(COLORS))


def _sample_one_size(rng: np.random.Generator, split: str, task: str) -> TaskSpec:
    """s3a1 (place) / s3b1 (stack), task 009b: _sample_place / _sample_stack with ONE cube size per scene (as s2a is to s2).
    Size U{small, large}; then the (cube, pad) / (top, bottom) pair by the s3a / s3b split-train rules restricted to that
    size (manipulated cube in SEEN_CUBES, held-out pairs rejected, top != bottom, STACK_ALLOWED; the bottom may be a
    held-out cube); then 2-4 cubes of that size with distinct colours (the s2a fill), so colour alone names every cube.
    Every HELD_OUT_STACK_PAIRS pair mixes sizes, so none can occur here; heldout_comp is not defined for these stages."""
    assert split in ("train", "paraphrase"), f"{task} one size: split train / paraphrase only, got {split}"
    size = _pick(rng, SIZES)
    seen = [c for c in SEEN_CUBES if c[1] == size]
    while True:
        t = _pick(rng, seen)
        if task == "place":
            d = _pick(rng, COLORS)
            if (t, d) not in HELD_OUT_PLACE_PAIRS:
                break
        else:
            d = _pick(rng, [c for c in ALL_CUBES if c[1] == size])
            if d != t and (t, d) not in HELD_OUT_STACK_PAIRS and STACK_ALLOWED(t, d):
                break
    must = [t] if task == "place" else [t, d]
    n = int(rng.integers(2, 5))
    others = [c for c in COLORS if c not in [m[0] for m in must]]
    cubes = must + [(others[i], size) for i in rng.permutation(len(others))[: n - len(must)]]
    return TaskSpec(task, t, d, [cubes[i] for i in rng.permutation(n)], list(COLORS))


def sample_spec(stage: str, split: str, rng: np.random.Generator,
                weights: tuple[float, float, float] = (1.0, 1.0, 1.0)) -> TaskSpec:
    """stage in {s1,s2,s2a,s3a,s3b,s3a1,s3b1,s4}; split in {train, heldout_comp, paraphrase} (paraphrase samples like train)."""
    assert split in ("train", "heldout_comp", "paraphrase"), split
    if stage == "s1":
        t = _pick(rng, HELD_OUT_TARGETS if split == "heldout_comp" else SEEN_CUBES)
        return TaskSpec("pick", t, None, [t], list(COLORS))
    if stage == "s2a":
        return _sample_pick_colours_only(rng, split)
    if stage == "s2":
        return _sample_pick(rng, split)
    if stage == "s3a":
        return _sample_place(rng, split)
    if stage == "s3b":
        return _sample_stack(rng, split)
    if stage in ("s3a1", "s3b1"):
        return _sample_one_size(rng, split, "place" if stage == "s3a1" else "stack")
    assert stage == "s4", stage
    w = np.asarray(weights, dtype=float)
    k = int(rng.choice(3, p=w / w.sum()))
    return (_sample_pick, _sample_place, _sample_stack)[k](rng, split)
