# Smoke 6 (HaMeR MPS) — BLOCKED

**Verdict: `blocked`** (not `fail`). Precheck gate PLAN.md task 6.0 stopped this smoke before any environment was built.

## What blocks it

HaMeR's MANO decoder cannot be instantiated without the two MANO model files:

- `MANO_LEFT.pkl`
- `MANO_RIGHT.pkl`

Both are **license-gated**. They are not on HuggingFace, not on GitHub, not redistributable. They are only obtainable after a human registers at <https://mano.is.tue.mpg.de>, accepts the MANO non-commercial license, and downloads `models/MANO_LEFT.pkl` + `models/MANO_RIGHT.pkl`.

**The agent cannot do this step.** Registration requires a real email, manual acceptance of the license terms, and manual download of the archive. I will not attempt to register, accept a license, or fetch a credential-gated asset on your behalf.

## Precheck evidence

Commands actually run (all returned **zero** matches):

```
find ~ /tmp /Users/guohongbin/v2d-mac -maxdepth 6 \( -name "MANO_RIGHT.pkl" -o -name "MANO_LEFT.pkl" \) 2>/dev/null
find ~ -name "MANO_*.pkl" 2>/dev/null                      # full-depth, unlimited
# targeted: ~/Downloads ~/Documents ~/data ~/datasets /tmp /opt /usr/local/share /Users/Shared
#   for MANO_LEFT.pkl, MANO_RIGHT.pkl, mano_v1*, and any dir named 'mano'
find /opt/homebrew/Caskroom/miniconda -maxdepth 6 \( -name manopth -o -name smplx \)   # no bundled MANO
ls ~/.cache/huggingface/hub                                # no mano-related repos
ls runs/weights                                             # only sam2
```

## Input asset is fine — only the weights are missing

I inspected `runs/p0/assets/frames/f_0010.jpg`: a 1280×720 ego-centric frame showing **two clearly visible human hands** (left and right) grasping a tissue box, wrists and most fingers in frame and unobstructed. So the P0 asset genuinely exercises HaMeR; there is no data-side reason this smoke cannot pass once the weights exist. 39 frames are available (`f_0001.jpg` … `f_0039.jpg`); the plan consumes `f_0001` … `f_0030`.

This is a qualitative sanity note only. It does **not** count toward any acceptance criterion.

## Nothing was installed

Per the task 6.0 gate I stopped before task 6.1:

- no `git clone` of `otaheri/HaMeR`
- no `runs/.venv-hamer`
- no dependency install
- no model download

Reason: HaMeR's dependency stack (smplx/chumpy-era pins, detectron2-adjacent extras, custom CUDA-oriented ops) is a heavy install, and its checkpoint load path fails outright without MANO. Building an env that provably cannot produce a verdict is wasted time and risks polluting the machine before we know the license is obtainable.

## What you need to do

1. Go to <https://mano.is.tue.mpg.de>, register, accept the license.
2. Download the archive containing `models/MANO_LEFT.pkl` and `models/MANO_RIGHT.pkl`.
3. Put them on this machine, e.g.:
   ```
   /Users/guohongbin/v2d-mac/runs/p0/assets/mano/MANO_LEFT.pkl
   /Users/guohongbin/v2d-mac/runs/p0/assets/mano/MANO_RIGHT.pkl
   ```
   (`runs/` is already gitignored, so the licensed assets stay out of version control.)
4. Tell me the path, and I will resume from PLAN.md task 6.1.

## Then smoke 6 resumes as

1. **6.1** clone `otaheri/HaMeR` → `runs/upstream/hamer`; `uv venv runs/.venv-hamer`; install per README; symlink the MANO pkls to the paths HaMeR expects.
2. **6.2** write `tests/test_smoke6_hamer.py` **first** — thresholds frozen in the test before any GREEN — run to RED, then implement to GREEN. Criteria:
   - MANO params (`betas`/`poses`/`trans`) non-empty and all finite, both hands
   - keypoint confidence above a threshold fixed in the test (target median ≥ 0.20 over `f_0001..f_0030`)
   - reprojection error recorded (3D MANO joints projected vs HaMeR 2D keypoints), mean < 15 px
   - left/right consistency: both hands detected in ≥ 80% of 30 frames, handedness flips ≤ 2
   - dominant inference device == `mps`, with any op-level CPU fallback enumerated in the report
3. **6.3** two frames of hand overlays → `runs/p0/smoke6_hamer/evidence/`
4. commit `p0(6): hamer mps GREEN` — or `verdict=fail` with MPS error evidence if MPS collapses.

## Red line

No acceptance criterion was relaxed, widened, or marked passed. All six remain **`not_evaluated`** in `report.json` — not `pass`. `MPS` availability is reported as *available but unmeasured*, since no HaMeR inference ran.
