"""Terminal render verification: panel meta, blender coverage, mujoco fallback.

    runs/.venv-mj/bin/python viz/verify_terminal_render.py
"""
from __future__ import annotations

import json
import pathlib

REPO = pathlib.Path(__file__).resolve().parents[1]


def main() -> None:
    panel = json.loads((REPO / "viz/panel_data_v2.json").read_text())
    meta = panel["meta"]
    rb = meta.get("renders_blender")
    rm = meta.get("renders")
    print("renders_blender:", "present" if rb else "MISSING")
    if rb:
        for k, v in rb["frames"].items():
            existing = sum(1 for p in v if (REPO / p).exists())
            print(f"  {k}: {existing}/{len(v)} files on disk "
                  f"(dir {rb['dir']})")
    if rm:
        for k, v in rm["frames"].items():
            existing = sum(1 for p in v if (REPO / p).exists())
            print(f"  mujoco {k}: {existing}/{len(v)}")
    print("gates:", [(g["id"], g["verdict"]) for g in panel["gates"]])
    print("mode:", meta["playback"]["mode"],
          "| phase note:", (meta.get("phase_notes") or {}).get("抬离", "")[:40])


if __name__ == "__main__":
    main()
