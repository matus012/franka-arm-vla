"""Single source of truth for 113 timing (contract v2 d). 113 code passes these explicitly to vision_lift/frames.py.

PERIOD / LATENCY: GATES.md conventions (112's proven 16.7 Hz camera, 1-frame latency), unless 0A.3 / 0A.4 change them.
N (= SmolVLA n_action_steps, the bridge's query period): set by the 0A.3 RESULT. Until then N_PROVISIONAL is the value
GATES.md allows for development runs (0B), and N is None so nothing can silently use an unmeasured value.
"""

from __future__ import annotations

PERIOD = 3  # control steps per camera frame (50 Hz / 3 = 16.7 Hz)
LATENCY = 1  # camera frames between capture and use
N: int | None = 10  # Stage 1 redesign (owner 2026-10-02, GATES b72947f + N rule): 10 (0A.3 attempt 2 passed N 5/10/20)
N_PROVISIONAL = 10  # GATES 0B: development value before the 0A.3 RESULT
CONTROL_HZ = 50
