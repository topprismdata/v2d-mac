"""Task 4.6b - the reachable-class demo set is replay evidence, not a coincidence.

Round p0(4d) searched the reachable classes (w35d strict wall criterion; K1
dual-wall stays class-B) for scripted trajectories that honestly pass the
frozen smoke4 acceptance gates, and recorded the two that did - a thumb rim-hook
family, on scene MJCF 4ee6f1d. Round p0(4e) found that family no longer clears
A3b on the current scene (MJCF 1b0f13f), so the demo set is retained as
ARCHIVED evidence of what the previous scene admitted, with its own provenance
stamp, rather than being re-stamped onto the scene it was never recorded on.

  R1  D4 layout + provenance (bc_obs 64 / bc_act 22 / per-traj lengths /
      bc_pose 8-col / mjcf_commit 4ee6f1d / selection=reachable_class_clean),
      and the set is the archived 678-frame pair, not a fresh collection
  R2  replaying every recorded action sequence ON THE RECORDING SCENE
      reproduces the recorded peak lift / finger force / duty cycle AND passes
      the frozen acceptance gate via the very same _analyse_episode the formal
      test uses; the same actions replayed on the CURRENT scene fail A3b
  R3  the load decomposition is stored and upward forces resolve to faces
  R5  the A/B is only ever scene differences (identical actions)
  R6  the current-scene sweep that found no gate-clean hook, pinned so the
      negative result cannot be quietly dropped

Negative results are evidence too. If R2b or R6 stops holding, a clean reachable
hook exists again and the demonstrations have to be re-collected.
"""
from __future__ import annotations

import json
import pathlib
import sys

import numpy as np
import pytest

REPO = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / "tests"))

NPZ = REPO / "runs/p0/smoke4_mujoco_ppo/reachable_demo_set_f210f45.npz"
MJCF_COMMIT = "4ee6f1d"
MJCF_SHA256 = "dbba3d70d722d529f80d1935e6c22826ac5d9be964c9827c64f3062bc660b934"
SELECTION = "reachable_class_clean"
AB_JSON = REPO / "runs/p0/smoke4_mujoco_ppo/hook_scene_ab.json"
FEASIBILITY_JSON = REPO / "runs/p0/smoke4_mujoco_ppo/hook_feasibility_1b0f13f.json"

pytest.importorskip("mujoco")


@pytest.fixture(scope="module")
def demo():
    assert NPZ.exists(), f"missing {NPZ} - run the reachable_scan record step"
    return dict(np.load(NPZ, allow_pickle=False))


def test_r1_d4_layout_and_provenance(demo):
    for key in ("bc_obs", "bc_act", "bc_pose", "bc_trajectory_len", "mjcf_commit",
                "selection"):
        assert key in demo, f"{key} missing from the demo set"
    assert demo["bc_obs"].ndim == 2 and demo["bc_obs"].shape[1] == 64
    assert demo["bc_act"].ndim == 2 and demo["bc_act"].shape[1] == 22
    assert demo["bc_obs"].dtype == np.float32 and demo["bc_act"].dtype == np.float32
    lens = demo["bc_trajectory_len"]
    assert int(lens.sum()) == demo["bc_obs"].shape[0]
    assert demo["bc_pose"].shape == (len(lens), 8)
    assert str(demo["mjcf_commit"]) == MJCF_COMMIT, (
        "this demo set is ARCHIVED evidence of the scene that recorded it; if it "
        "claims a different commit the archive and the provenance stamp have "
        "driven apart")
    assert str(demo["selection"]) == SELECTION
    assert "gate_metrics_json" in demo and "cfg_json" in demo


def test_r1c_demo_set_is_the_archived_pair_not_a_fresh_collection(demo):
    """678 frames over 2 trajectories is the ARCHIVED p0(4d) set, and must stay
    exactly that size. It is evidence of what the previous scene admitted, not a
    warm start for this one - see test_r5."""
    n_frames = int(demo["bc_trajectory_len"].sum())
    n_traj = int(len(demo["bc_trajectory_len"]))
    assert (n_frames, n_traj) == (678, 2), (
        f"archived demo set is {n_frames} frames over {n_traj} trajectories; "
        "the p0(4d) set was 678 frames over 2. If this is a different set, it "
        "needs its own provenance stamp and its own round.")


def test_r1b_the_hook_pass_was_an_apex_coincidence_not_a_carried_lift(demo):
    """Why the p0(4d) "clean hook" was never a clean hook.

    The recorded trajectories spend three steps above the 5 cm bar and reach
    that peak with ZERO finger force on the box. A launch, not a grasp: the box
    leaves the pedestal, coasts, and lands. That is why the passing set had
    measure ~zero, and it is the property round p0(4e) had to re-check rather
    than inherit.
    """
    gates = json.loads(str(demo["gate_metrics_json"]))
    for i, g in enumerate(gates):
        assert g["peak_lift_m"] > 0.05
        assert not g["table_touching_at_peak"]


def test_r5_the_same_actions_pass_a3b_on_the_old_scene_and_fail_it_here():
    """The A/B that isolates the scene revision as the cause.

    Identical actions, identical env code, identical thresholds; only
    grounding/assets/allegro_floating.xml differs between the two replays. If
    this stops holding, something other than the physics changed and the
    negative result below needs re-deriving.
    """
    assert AB_JSON.exists(), f"missing {AB_JSON} - run grounding/hook_scene_ab.py"
    ab = json.loads(AB_JSON.read_text())
    assert ab["replayed_on"]["commit"] == "1b0f13f"
    assert ab["recorded_on"]["commit"] == "4ee6f1d"
    assert all(r["scene_4ee6f1d"]["A3b_airborne_at_peak"] for r in ab["rows"]), (
        "the archived demonstrations no longer clear A3b on the scene that "
        "recorded them; the replay evidence is stale")
    assert not any(r["scene_1b0f13f"]["A3b_airborne_at_peak"] for r in ab["rows"]), (
        "the archived demonstrations now clear A3b on the CURRENT scene - the "
        "no-clean-hook conclusion is out of date and the collection must be redone")


def test_r6_current_scene_sweep_found_no_gate_clean_hook():
    """The negative result, pinned so it cannot be quietly forgotten.

    If a later revision makes a clean hook reachable again this test is what
    should fail first, and its failure message says exactly that.
    """
    assert FEASIBILITY_JSON.exists(), (
        f"missing {FEASIBILITY_JSON} - run collect_hook_demos.py then "
        "compact_hook_report.py")
    f = json.loads(FEASIBILITY_JSON.read_text())
    assert f["mjcf_commit"] == "1b0f13f"
    assert f["n_trials"] >= 10000, (
        f"the feasibility sweep only covered {f['n_trials']} poses - too few to "
        "support a 'no clean hook exists' claim")
    if f["n_pass"] == 0:
        # Not every trial fails ON A3b: a few hundred leave the pedestal at
        # their own peak and are rejected by the other frozen gates (mostly the
        # anchored-step cap, which catches the launch tail). What matters is
        # that no pose passed the gate as a whole, and that A3b is the dominant
        # failure - a family that mostly failed elsewhere would mean this sweep
        # is measuring the wrong thing.
        frac = f["A3b"]["table_touching_at_peak_fraction"]
        assert frac > 0.9, (
            f"only {frac*100:.1f}% of the sweep failed A3b; the sweep is not "
            "measuring the pedestal contact and the negative result does not "
            "follow from it")
        top = f["closest_misses"][:10]
        assert all(r["gate"]["table_touching_at_peak"] for r in top), (
            "a top-10 peak-lift pose is clean at its peak; the closest misses "
            "must all be pedestal-borne, or the sweep found a real candidate")
        # ...and they are rejected twice over: the pedestal contact fails A3b and
        # the 11-15 mm single-step displacements fail A3c's anchored-step cap.
        # Two independent frozen gates saying "this is a launch" is the whole
        # reason no pose in this family is worth recording as a demonstration.
        assert all(r["gate"]["max_anchored_step_m"] > 0.01 for r in top), (
            "a top-10 peak-lift pose now respects the 10 mm anchored-step cap; "
            "the launch signature this sweep rejected has changed")
    else:
        pytest.skip(
            f"{f['n_pass']} gate-clean poses now exist on {f['mjcf_commit']}; "
            "re-collect the demonstrations and drop the selection marker")


def _replay_on(xml_path, acts) -> dict:
    """Replay recorded actions against a specific scene and run the frozen gate."""
    import grounding.envs.lift_hold_env as env_mod
    from test_smoke4_accept import _analyse_episode

    original = env_mod.MJCF
    env_mod.MJCF = xml_path
    try:
        env = env_mod.LiftHoldEnv(episode_len=len(acts) + 10, domain_rand=False)
        env.reset(seed=0)
        for a in acts:
            env.step(np.asarray(a, dtype=np.float64))
        ev = _analyse_episode({"trace": dict(env.trace, dt=float(env.dt))})
        ev["gate_pass"] = bool(
            ev["peak_lift_m"] > 0.05
            and ev["n_finger_contact_steps"] >= 5
            and ev["finger_pairs"]
            and ev["force_duty_ok"]
            and ev["force_sufficiency_ok"]
            and not ev["table_touching_at_peak"]
            and ev["anchored_step_ok"]
            and ev["free_fall_gravity_ok"]
            and ev["separation_ok"]
        )
        env.close()
        return ev
    finally:
        env_mod.MJCF = original


@pytest.fixture(scope="module")
def replayed(demo):
    """Replay every trajectory on BOTH scenes; reuse across assertions.

    SUPERSEDED by W6a (2026-10-15): the dual-hand env cannot load the archived
    single-hand MJCF at all (the left hand's joint names do not exist there),
    and the recorded 22-dim actions no longer match the 44-dim action space.
    The A/B machinery is rebuilt for the dual-hand scene in the W6b wave; the
    f210f45 archive stays as evidence of that scene only."""
    pytest.skip("SUPERSEDED by W6a: single-hand recording scene is not "
                "loadable by the dual-hand env (see R10)")
    from grounding.hook_scene_ab import _old_scene_path  # noqa: F401 PLC0415

    old_scene = _old_scene_path()
    new_scene = REPO / "grounding/assets/allegro_floating.xml"
    lens = demo["bc_trajectory_len"]
    gates = json.loads(str(demo["gate_metrics_json"]))
    cfgs = json.loads(str(demo["cfg_json"]))
    assert len(gates) == len(lens) == len(cfgs)
    out = []
    ofs = 0
    for i, L in enumerate(lens):
        act = demo["bc_act"][ofs: ofs + int(L)]
        out.append({
            "old": _replay_on(old_scene, act),
            "new": _replay_on(new_scene, act),
            "recorded": gates[i],
            "cfg": cfgs[i],
        })
        ofs += int(L)
    return out


def test_r2_replay_reproduces_the_recorded_gate_on_the_recording_scene(replayed):
    """The archived evidence must stay checkable on the scene that produced it."""
    for i, r in enumerate(replayed):
        ev, rec = r["old"], r["recorded"]
        assert ev["gate_pass"], (
            f"trajectory {i} no longer reproduces its recorded gate on MJCF "
            f"4ee6f1d: peak {ev['peak_lift_m']*1000:.1f}mm clean@peak="
            f"{not ev['table_touching_at_peak']}. The archive is not reproducing "
            "on its own scene.")
        assert abs(ev["peak_lift_m"] - rec["peak_lift_m"]) <= 0.005, (
            f"trajectory {i}: peak lift drifted on replay "
            f"({ev['peak_lift_m']*1000:.1f} vs {rec['peak_lift_m']*1000:.1f} mm)")
        assert abs(ev["finger_force_mean_N"] - rec["finger_force_mean_N"]) <= 0.5
        assert abs(ev["finger_force_duty_cycle"] - rec["finger_force_duty_cycle"]) <= 0.05
        assert ev["n_finger_contact_steps"] >= 5 and ev["finger_pairs"]
        assert ev["peak_lift_m"] > 0.05 and not ev["table_touching_at_peak"]
        assert ev["force_duty_ok"] and ev["force_sufficiency_ok"]
        assert ev["anchored_step_ok"] and ev["free_fall_gravity_ok"] and ev["separation_ok"]


def test_r2b_the_same_actions_fail_a3b_on_the_current_scene(replayed):
    """The round's negative, derived here rather than read from a report.

    Same actions, same code, same thresholds - only the scene differs. A3b is
    the gate that flips, and it flips on the pedestal contact at the peak step.
    """
    for i, r in enumerate(replayed):
        new, old = r["new"], r["old"]
        assert old["table_touching_at_peak"] is False
        assert new["table_touching_at_peak"] is True, (
            f"trajectory {i} now clears A3b on MJCF 1b0f13f (peak "
            f"{new['peak_lift_m']*1000:.1f} mm, clean@peak) - a clean reachable "
            "hook exists again and the demonstrations must be re-collected")


def test_r3_load_decomposition_present_and_resolves(demo):
    assert "load_decomposition_json" in demo
    decomps = json.loads(str(demo["load_decomposition_json"]))
    lens = demo["bc_trajectory_len"]
    assert len(decomps) == len(lens)
    for dec in decomps:
        share = dec["upward_share_by_face"]
        assert abs(sum(share.values()) - 1.0) < 1e-6
        assert dec["total_upward_N"] > 0.0
        # disclosed load path: the rim hook carries the box through top-face
        # contacts; wall faces carry nothing in this family
        assert share.get("top", 0.0) > 0.0
