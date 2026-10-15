"""Wave 3.5 candidate builder: re-emit the MJCF at an adjudicated pedestal standoff.

Runs the UNMODIFIED generator (imported by path) with two module globals
overridden before main(): PEDESTAL_H_M (the standoff) and OUT (a gitignored
two-level-deep dir under runs/ so the relative meshdir still resolves). The
generator's measured two-pass build (moment arm / gravity / inertia sweeps +
resting sink) runs in full, so each variant carries tables measured from its
own geometry. The repo's committed XML and generator are untouched.

    python runs/w35/build_variant.py 0.10   # -> runs/w35/ped100/allegro_floating.xml
"""
from __future__ import annotations

import importlib.util
import pathlib
import sys

REPO = pathlib.Path("/Users/guohongbin/v2d-mac")


def load_builder():
    spec = importlib.util.spec_from_file_location(
        "baf_variant", REPO / "grounding/assets/build_allegro_floating.py")
    mod = importlib.util.module_from_spec(spec)
    sys.modules["baf_variant"] = mod
    spec.loader.exec_module(mod)
    return mod


def main() -> None:
    h = float(sys.argv[1])
    baf = load_builder()
    baf.PEDESTAL_H_M = baf.PEDESTAL_H = h
    out = REPO / f"runs/ped{int(round(h * 1000))}/allegro_floating.xml"
    out.parent.mkdir(parents=True, exist_ok=True)
    baf.OUT = out
    print(f"[variant] pedestal={h:.3f} m -> {out}", flush=True)
    baf.main()
    print(f"[variant] done: {out}", flush=True)


if __name__ == "__main__":
    main()
