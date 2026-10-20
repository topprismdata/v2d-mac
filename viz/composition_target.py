"""Composition target for the W6d ego-view camera (Main-measured, r26 f_0252).

All coordinates are frame-normalized (u right, v down) unless noted.
Tolerances are ±5% of the frame per dimension (Main's acceptance).
"""
from __future__ import annotations

import json
import pathlib

TARGET = {
    "source_frame": 252,
    "frame_size": [1280, 720],
    "box_center_uv": [0.50, 0.38],
    "box_width_frac": 0.30,
    "forearm_entry_left_uv": [0.00, 0.55],
    "forearm_entry_right_uv": [1.00, 0.60],
    "table_front_edge_in_frame": False,
    "camera_constraints": {
        "height_z_m": [0.85, 0.90],
        "pitch_down_deg": [15, 20],
        "distance_to_box_m": 0.9,
        "fov_h_deg_start": 30.0,
    },
    "tolerance_frac": 0.05,
    "scene": "allegro_floating_ur5e.xml (W6B_SCENE=ur5e, r26 Track A)",
    "side_entry_note": ("MinkWire 侧入基座 (±0.55, 0, 0.80) 落地后，前臂入画角"
                        "自然趋近水平侧入；本表为相机侧目标，与基座改造并行"),
}

if __name__ == "__main__":
    out = pathlib.Path(__file__).resolve().parent / "composition_target.json"
    out.write_text(json.dumps(TARGET, indent=1, ensure_ascii=False))
    print(f"wrote {out}")
