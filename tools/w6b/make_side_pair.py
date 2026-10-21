#!/usr/bin/env python3
"""Publish the side cell's terminal arm+finger pair from a run's report.

The r26 Track A mechanism pins the terminal to a native (arm, fingers)
pair, so the approach registration can be changed without moving the
terminal.  The side cell has no committed pair (the shipped
native_seed_pairs.json is ceiling-cell specific), so this tool extracts
the terminal's commanded state from a report and writes it in the pair
format the pipeline reads (W6B_NATIVE_PAIRS).

usage: make_side_pair.py <report.json> <out.json>
"""
import json
import sys
from pathlib import Path

src = Path(sys.argv[1])
dst = Path(sys.argv[2])
rep = json.loads(src.read_text())
pairs = []
for side in ("right", "left"):
    tc = rep["per_side"][side]["terminal_commands"]
    arm = tc["wrist_rad"]
    fing = tc["finger_rad"]
    assert len(arm) == 6 and len(fing) == 16, (len(arm), len(fing))
    pairs.append({
        "side": side,
        "arm": arm,
        "fingers": fing,
        "first_mm": rep["per_side"][side]["gate1_final_touch"]["rect_dist_mm"],
        "second_mm": rep["per_side"][side]["gate1_final_touch"][
            "executed_state_evidence"]["rect_dist_mm"],
        "fwd_deg": rep["per_side"][side]["gate1_final_touch"][
            "forward_axis_deg"],
        "through_mm": rep["per_side"][side]["gate1_final_touch"][
            "per_pad_burial"]["ff_tip_collision"]["through_mm"]
        if "ff_tip_collision" in rep["per_side"][side]["gate1_final_touch"][
            "per_pad_burial"] else None,
        "source": ("extracted from " + src.name +
                   " (the side cell's terminal commanded state)"),
    })
dst.write_text(json.dumps({"pairs": pairs, "n_pairs": len(pairs)}, indent=1))
print(f"wrote {dst} ({len(pairs)} pairs)")
