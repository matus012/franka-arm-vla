"""Task 004 (GATES "Inference-time instruction-guidance probe (004)"): the "other" instructions of an episode. stdlib only.

s2a scenes hold distinct colours of one size, so replacing the target colour word of the episode's own instruction by
another colour on the table names exactly that cube with the same template (and the same optional size word).

s2 scenes (2b amended, paired s2 data) hold colours AND sizes, so a target is a (colour, size) cube: `other_instruction`
replaces the whole target noun phrase ("the [size] colour cube", vla_common.tasks._np) by the other cube's, with the
size word iff the original phrase has one OR the other cube's colour is on the table in more than one size (so the new
phrase always names exactly that cube); the rest of the template is unchanged. `named_cubes` resolves the phrase.
"""

from __future__ import annotations

import re

from .catalog import COLORS, SIZES

COLOUR_RE = re.compile(r"\b(" + "|".join(COLORS) + r")\b")
CUBE_NP_RE = re.compile(r"\bthe (?:(" + "|".join(SIZES) + r") )?(" + "|".join(COLORS) + r") cube\b")


def other_instructions(instruction: str, target_colour: str, cube_colours: list[str]) -> list[str]:
    """The instruction with its (single) colour word replaced by each other colour on the table, in table order."""
    found = COLOUR_RE.findall(instruction)
    if found != [target_colour]:
        raise ValueError(f"expected exactly one colour word {target_colour!r} in {instruction!r}, found {found}")
    others = [c for c in dict.fromkeys(cube_colours) if c != target_colour]
    if not others:
        raise ValueError(f"no other colour on the table for {instruction!r}: {cube_colours}")
    return [COLOUR_RE.sub(c, instruction) for c in others]


def named_cubes(instruction: str, cubes: list[tuple[str, str]]) -> list[list[tuple[str, str]]]:
    """Per cube noun phrase of the instruction (in order): the table cubes it can refer to (colour, and size if stated)."""
    return [[tuple(c) for c in cubes if c[0] == col and size in ("", c[1])]
            for size, col in CUBE_NP_RE.findall(instruction)]


def other_instruction(instruction: str, target: tuple[str, str], other: tuple[str, str],
                      cubes: list[tuple[str, str]]) -> str:
    """s2: the instruction with its single target noun phrase replaced by `other`'s (module doc: size-word rule)."""
    found = CUBE_NP_RE.findall(instruction)
    if len(found) != 1 or found[0][1] != target[0] or found[0][0] not in ("", target[1]):
        raise ValueError(f"expected exactly one noun phrase naming {target} in {instruction!r}, found {found}")
    if tuple(other) == tuple(target) or tuple(other) not in [tuple(c) for c in cubes]:
        raise ValueError(f"other cube {other} is the target or not on the table {cubes}")
    say_size = bool(found[0][0]) or len({c[1] for c in cubes if c[0] == other[0]}) > 1
    return CUBE_NP_RE.sub(f"the {other[1]} {other[0]} cube" if say_size else f"the {other[0]} cube", instruction)
