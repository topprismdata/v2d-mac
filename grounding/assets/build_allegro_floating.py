"""Generate grounding/assets/allegro_floating.xml from MuJoCo Menagerie wonik_allegro.

Upstream (read-only, cloned into the gitignored runs/ tree):
  https://github.com/google-deepmind/mujoco_menagerie  ->  wonik_allegro/right_hand.xml
  commit d7b481bac63d3d600ab404ee74f7474c5efe8a67 (pinned in UPSTREAM_COMMIT below)

The generated file is committed; this script exists so the derivation is reproducible
and auditable (no binary meshes are vendored into git).

Design deviations from PLAN task 4.1, recorded in runs/p0/smoke4_mujoco_ppo/report.json:
  * MuJoCo forbids actuators on a <freejoint>, so the actuatable floating base of the
    hand is expressed as the freejoint's own 3 slide + 3 hinge joints (wrist_x/y/z,
    wrist_rx/ry/rz), each driven by a position actuator with a written ctrlrange and
    forcerange. The *object* body keeps the literal <freejoint/>.

Wave 3 revision (P0), driven entirely by smoke4b / smoke5 measurements. See
grounding/assets/REVISION.md for the numbers and their sources.

  R1  Finger forcerange is derived per joint from the moment arm between that
      joint's axis and its fingertip, measured from the MJCF's own geometry and
      minimised over the reachable grasp workspace:

          forcerange_j = (F_tip / 4) * moment_arm_j

      with FINGER_TIP_FORCE_LIMIT_N = 10 N, the Wonik Allegro V5 spec. The old
      model gave every joint a flat +-2.0 N*m, which the geometry turns into
      42-166 N at the fingertip; smoke4b then measured a 37.7 N mean grip, i.e. the
      policy was pressing ~3.8x harder than the real hand and learning to cheat by
      squeezing. kp is chosen against two measured constraints - it must be able to
      reach its own forcerange over the joint's full stroke, and it must be able to
      hold the finger's own weight within GRAVITY_SAG_BUDGET_RAD of sag.

  R2  wrist_z gains 0.062 m of downward travel and wrist_ry gains 0.058 rad of
      pitch, the residual smoke5 measured after its gauge fixing. wrist_rz/x/y had
      zero residual and are untouched.

  R3  box<->finger contacts get solref/solimp from the paper box's own stiffness
      rather than MuJoCo's soft default, which let a fingertip press 2.7 mm into a
      rigid box.

  R4  every finger and palm geom is explicitly named. Upstream they all come out of
      unnamed <default> classes, which once made a name-based contact detector
      report a structural zero.

  R5  the penetration tolerance is k x the measured resting sink rather than a
      hardcoded 1e-3 m, so it follows R3.

Wave 3.5-E extends the explicit-pair scope from the palm to the four finger
proximal links - the same solref mixing 3.5-B removed on the palm, on the links
that actually carry the grasp. Proximal only, and that is measured rather than
assumed: see PROXIMAL_BOX_PAIRS.

  R8  the proxy box gains a tray lip on its upper edge (W4): one flange per +-x
      wall, flush with the box top, protruding BOX_LIP_OVERHANG_M outward and
      hanging BOX_LIP_HEIGHT_M down. The underside of that overhang is a
      horizontal face over empty space, so a fingertip that can already get below
      the box top can push UP on it and get a normal force with no friction term -
      the clean bearing path the bare vertical wall can never provide. Both are
      parameters of build(), and 0/0 reproduces the bare box exactly, which is the
      control every sweep cell is measured against.

      The lip's mass is taken OUT of the box, not added to it: box_geom_mass()
      returns 0.15 kg minus the lip, so the object body still weighs exactly
      0.15 kg. Every force gate is calibrated on that number (A3a's 1.4715 N
      sufficiency bound, the pedestal-load ratios), and a lip that silently made
      the box heavier would move all of them in the same direction.

Run:  python grounding/assets/build_allegro_floating.py
"""

from __future__ import annotations

import functools
import pathlib
import xml.dom.minidom
import xml.etree.ElementTree as ET

REPO = pathlib.Path(__file__).resolve().parents[2]
UPSTREAM_DIR = REPO / "runs/upstream/menagerie/wonik_allegro"
UPSTREAM_HAND = UPSTREAM_DIR / "right_hand.xml"
OUT = pathlib.Path(__file__).resolve().parent / "allegro_floating.xml"

UPSTREAM_REPO = "https://github.com/google-deepmind/mujoco_menagerie"
UPSTREAM_COMMIT = "d7b481bac63d3d600ab404ee74f7474c5efe8a67"

# meshdir has to be re-pointed at the (gitignored) upstream checkout because this file
# lives in grounding/assets/, not in wonik_allegro/.
MESH_DIR = "../../runs/upstream/menagerie/wonik_allegro/assets"

# --- constants of the augmented scene (task 4.1) ---------------------------
# Tissue-box proxy: 0.24 x 0.12 x 0.08 m. The long axis is aligned with the palm's
# long axis (y) because all three fingers flex in the x-z plane at y = 0, +-0.0435;
# the 0.24 m extent therefore runs *under* the fingers instead of away from them.
BOX_HALF_M = (0.06, 0.12, 0.04)  # x, y, z half-extents -> 0.12 x 0.24 x 0.08 m
BOX_HALF = BOX_HALF_M  # backwards-compatible alias
BOX_MASS_KG = 0.15  # kg
BOX_MASS = BOX_MASS_KG

# --- W4 / R8: the tray lip on the box's upper edge -- REFUTED, NOT SHIPPED ----
# A tissue box with a folded-over rim, and that rim is what a finger would hook:
# a flange on the TOP lip_h of each +-x wall, protruding `overhang` outward, so
# the space UNDER the protrusion overhangs empty space a fingertip could push up
# on - a purely normal bearing path, where the bare vertical wall can only be
# held up by wall friction.
#
# IT WAS REFUTED KINEMATICALLY AND IS NOT IN THE SHIPPED SCENE. A 15-cell sweep
# (overhang 4-12 mm x lip_h 6-14 mm, runs/w4/lip_kin_scan.py) found ZERO
# reachable hook faces: to get a pad under the lip the finger has to come over
# the box top and cut back down under a 6-14 mm flange, and the palm floor
# (0.839 m) already sits level with the box top (0.840 m), so no such path
# exists in this hand's workspace. See REVISION.md R8 for the numbers.
#
# The PARAMETERISATION IS KEPT AS AN ASSET, deliberately, for two reasons:
#   * build(lip_overhang_m=0, lip_h_m=0) reproduces the bare box BYTE FOR BYTE,
#     which is what lets any future geometry variant be A/B-ed against a fixed
#     reference instead of against a hand-maintained baseline;
#   * the mass split it encodes is the correct one for ANY added box hardware -
#     added mass comes OUT of the 0.15 kg, never on top of it - and that is the
#     invariant every force gate in this repo is calibrated on.
#
# So these are 0.0, not 8 mm / 10 mm: the geometry is withdrawn, the machinery is
# not. Raising them regenerates a lip scene for a fresh experiment; it does not
# resurrect the refuted one.
BOX_LIP_OVERHANG_M = 0.0
BOX_LIP_HEIGHT_M = 0.0

# W5: a 45-degree chamfer plate along the box's two top +-x edges.  The bare
# box's top edge is a knife edge, and the one gate-clean top-hook trajectory
# the sweep found (1/15625, load 100% through the top-face contact) has a
# ZERO survival boundary -- a finger balanced on a knife edge dies under any
# perturbation.  A chamfer turns that edge into a short ramp, which is the
# physically honest version of the same object: production boxes have
# relieved lid edges.  0.0 reproduces the bare box byte for byte.
BOX_CHAMFER_M = 0.0
BOX_CHAMFER_T_M = 0.003  # plate thickness, m
BOX_CHAMFER_GEOM_NAMES = ("object_chamfer_negx", "object_chamfer_posx")
# The rim runs the full length of the box, so it is a tray rim and not two pads.
BOX_LIP_HALF_Y_M = BOX_HALF_M[1]
BOX_LIP_GEOM_NAMES = ("object_lip_negx", "object_lip_posx")
# The lip is the same cardboard as the proxy, so it carries the proxy's own
# effective density and its mass comes OUT of the 0.15 kg rather than being added
# to it: a lip that made the box heavier would silently invalidate every force
# gate, which is why box_geom_mass() exists and is asserted on.
BOX_LIP_DENSITY_KG_M3 = BOX_MASS_KG / (8.0 * BOX_HALF_M[0] * BOX_HALF_M[1] * BOX_HALF_M[2])
TABLE_TOP_Z_M = 0.70  # table slab top surface
TABLE_TOP_Z = TABLE_TOP_Z_M
# The box sits on a 60 mm pedestal rather than flush on the table. Measured with a
# scripted open-loop grasp search (~5800 samples): with the box flush on the slab
# the fingers can only hook its top edge and the grasp slips at ~3.2-3.6 cm, which
# is BELOW the 5 cm acceptance bar - the task was geometrically unsolvable, not
# merely hard for PPO. With 60 mm of clearance the fingers slide under the box and
# form a real power grasp: 169/700 sampled poses exceed 5 cm, best 11.5 cm, i.e.
# above the official evaluator's own 0.10 m lift threshold.
PEDESTAL_H_M = 0.06
PEDESTAL_H = PEDESTAL_H_M
HAND_HOME = "0 0 0.92"  # floating palm pose (x y z) - above the raised box

# --- R1: the real machine's fingertip force budget --------------------------
# Wonik Allegro V5 spec: 10 N per fingertip. This is the single number the whole
# forcerange table is derived from; nothing here is tuned to make the task pass.
FINGER_TIP_FORCE_LIMIT_N = 10.0
FINGER_PREFIXES = ("ff", "mf", "rf", "th")
FINGER_JOINTS = tuple(f"{f}j{i}" for f in FINGER_PREFIXES for i in range(4))
# Upstream names the actuators after the joint's role, not the joint: ffj0's
# actuator is "ffa0". Resolving an actuator by joint name returns -1, which silently
# indexes the LAST actuator and makes any forcerange comparison read one joint's
# numbers sixteen times - so the mapping is explicit and every lookup goes through
# it.
FINGER_ACTUATOR_OF = {f"{f}j{i}": f"{f}a{i}" for f in FINGER_PREFIXES for i in range(4)}
JOINTS_PER_FINGER = 4
# The 10 N is a budget for the whole fingertip, not per joint: all four joints of a
# finger press the same pad, and their contributions add,
#
#     F_tip = |sum_j tau_j / arm_j|  <=  sum_j tau_j / arm_j
#
# so giving each joint its own full 10 N would let one fingertip reach 40 N. The
# budget is therefore split evenly across a finger's four joints:
#
#     tau_j = (F_tip / 4) * arm_j   =>   sum_j tau_j / arm_j = F_tip exactly
#
# for any configuration, because arm_j is each joint's own minimum (the denominator
# is the smallest it can be, so tau_j / arm_j is the largest force that joint can
# contribute). Weights summing to 1 would all satisfy the bound; equal is the
# simplest and gives no joint a privileged share of the real hand's budget.
FINGER_FORCE_SHARE = 1.0 / JOINTS_PER_FINGER
# The servo must be able to hold the finger's own weight within this much sag.
GRAVITY_SAG_BUDGET_RAD = 0.05
# Servo bandwidth guard: omega * timestep must stay below this or the 2 ms
# integration rings. MuJoCo's implicitfast integrator tolerates a good deal more
# than 1.0; the margin is there because the effective inertia is configuration
# dependent and the table below quotes the *minimum* over the workspace.
SERVO_BANDWIDTH_GUARD = 1.0

# --- R2: wrist travel, from smoke5's measured residual ---------------------
WRIST_TRAVEL_ADD_M = {
    # axis: (previous low, previous high, added travel, units)
    #
    # smoke5 measured 0.062088 m / 0.058449 rad of residual. The added travel is
    # rounded UP past those to the next millimetre / milliradian (0.063 / 0.059)
    # rather than to the nominal 0.062 / 0.058: a range that ends at exactly
    # -0.162000 still clips smoke5's recorded command minimum of -0.162088, so the
    # axis would read as out of range on replay even though the nominal figure was
    # "added".
    "wrist_z": (-0.10, 0.16, 0.063),   # m   - vertical descent
    "wrist_ry": (-1.20, 1.20, 0.059),  # rad - pitch swing
}
WRIST_UNCHANGED = {
    "wrist_x": (-0.18, 0.18),
    "wrist_y": (-0.18, 0.18),
    "wrist_rx": (-1.80, 1.80),
    "wrist_rz": (-1.57, 1.57),
}
WRIST_DAMPING = {"slide": "25", "hinge": "1.2"}
WRIST_ARMATURE = {"slide": "0.03", "hinge": "0.03"}
# wrist actuator gains: slides carry the whole hand mass, hinges only the palm's
# inertia about a point inside the hand.
WRIST_ACTUATOR = {
    "wrist_ax": dict(kp="180", forcerange="-45 45"),
    "wrist_ay": dict(kp="180", forcerange="-45 45"),
    "wrist_az": dict(kp="180", forcerange="-45 45"),
    "wrist_rx": dict(kp="14", forcerange="-6 6"),
    "wrist_ry": dict(kp="14", forcerange="-6 6"),
    "wrist_rz": dict(kp="14", forcerange="-6 6"),
}
# mapping from joint name -> (actuator name, ctrlrange)
WRIST_ACTUATOR_OF = {
    "wrist_x": "wrist_ax", "wrist_y": "wrist_ay", "wrist_z": "wrist_az",
    "wrist_rx": "wrist_rx", "wrist_ry": "wrist_ry", "wrist_rz": "wrist_rz",
}
WRIST_AXIS = {
    "wrist_x": ("slide", "1 0 0"), "wrist_y": ("slide", "0 1 0"),
    "wrist_z": ("slide", "0 0 1"),
    "wrist_rx": ("hinge", "1 0 0"), "wrist_ry": ("hinge", "0 1 0"),
    "wrist_rz": ("hinge", "0 0 1"),
}

# --- R3: paper box vs fingertip contact -------------------------------------
# MuJoCo's default (timeconst 0.02 s) let a fingertip press the box 2.7 mm deep.
# 4 ms critically damped keeps the measured penetration under 10 N at ~0.21 mm and
# leaves the resting box dead quiet.
#
# SCOPE, and why it is narrow: the stiffness is written onto the box and onto the
# four fingertip pads only. A first pass stamped it on every colliding hand geom and
# that turned the palm-vs-pedestal contact - a pre-existing near-tangential graze in
# this scene - into an incompressible squeeze whose peak force rose from 3.2 kN to
# 77 kN, a 24x amplification of a contact R3 says nothing about. MuJoCo takes the
# stiffer of the two sides of a pair, so writing the box plus the fingertips is
# enough to govern every box<->finger contact, which is the evidence being fixed,
# while leaving the hand-vs-table contacts alone.
CONTACT_SOLREF = (0.004, 1.0)
CONTACT_SOLIMP = (0.95, 0.99, 0.001, 0.5, 2)
DEFAULT_SOLREF = (0.02, 1.0)

# --- Wave 3.5-B: explicit palm <-> box contact pair --------------------------
# The palm's collision box carries no R3 stamp (scope fix from the R3 lesson), so
# an implicit palm-vs-box pair mixed two sides: MuJoCo averages positive solref
# across a pair, and 10 N on the palm pressed 0.359 mm into the box - 1.7x the
# fingertip budget at the same load. One explicit <pair> for exactly this contact
# removes the mixing, and only this contact: the palm-vs-pedestal graze peak force
# is identical before and after (14.694 N both ways, runs/w35/palm_probe.json).
PALM_BOX_PAIR = ("palm_collision", "object_geom")

# W4/R8: the lip geoms join the box as first-class box surfaces, so they need the
# same explicit-pair treatment. Without it the palm-vs-lip and proximal-vs-lip
# contacts resolve implicitly, where MuJoCo averages the lip's 4 ms with the
# link's 20 ms - the exact mixing defect 3.5-B and 3.5-E removed on object_geom,
# re-introduced on the surface that now carries the grasp. Built from the same
# two constants so the pair set cannot silently miss a new box geom.
def box_pairs(*, lip_overhang_m: float = BOX_LIP_OVERHANG_M,
              lip_h_m: float = BOX_LIP_HEIGHT_M,
              chamfer_m: float = BOX_CHAMFER_M) -> list[tuple[str, str]]:
    """(hand geom, object geom) for every explicit box pair this scene emits."""
    out = [(PALM_BOX_PAIR[0], name) for name in object_geom_names(lip_overhang_m, lip_h_m, chamfer_m)]
    out += [(g1, name) for g1, _ in PROXIMAL_BOX_PAIRS
            for name in object_geom_names(lip_overhang_m, lip_h_m, chamfer_m)]
    return out

# --- Wave 3.5-E: explicit finger-proximal <-> box contact pairs --------------
# Same defect Wave 3.5-B found on the palm, on the links that carry the grasp.
# R3 stamped the four fingertip pads, 3.5-B added one pair for the palm, and every
# other finger link still resolved its box contact implicitly, where MuJoCo
# averages the two sides' solref and the link's own 20 ms timeconstant wins. The
# smoke5 rerun on 4ee6f1d then measured the deepest box penetration on
# rf_proximal_collision - 1.188 mm end-of-frame, 1.282 mm in-frame - load that the
# palm fix pushed off the palm and onto an unstiffened link (runs/w35/
# w35e_proximal.json). Four explicit pairs, one per proximal link, remove that
# mixing for exactly the links that carry box load and nothing else: penetration
# 1.188 -> 0.095 mm, every non-target contact pair's peak force bit-identical.
#
# Scope is proximal ONLY, and that is a measurement, not a guess: the medial and
# distal links carry no box force at all in this scene (0.000 N, 0.000 mm), so
# pairs for them govern nothing - the proximal-only, proximal+medial and
# all-phalanges variants are bit-for-bit identical in every recorded number.
# Widening the scope later has to be argued from a clip that loads those links.
PROXIMAL_BOX_PAIRS = tuple((f"{f}_proximal_collision", "object_geom")
                           for f in ("ff", "mf", "rf", "th"))

# --- R5: penetration tolerance derived from the resting sink -----------------
# k >= 3 by contract: the tolerance must sit clear of MuJoCo's own resting
# numerical tolerance so a real interpenetration cannot hide under it, while staying
# orders of magnitude below the 40 mm box half-height.
PENETRATION_TOLERANCE_K = 4.0
RESTING_SETTLE_STEPS = 900
RESTING_SAMPLE_STEPS = 400

# moment-arm sweep settings (deterministic seed -> reproducible table)
ARM_SWEEP_SAMPLES = 120_000
ARM_SWEEP_SEED = 20251015

# Pass 1 of the build writes a loadable model with a placeholder finger forcerange so
# the moment-arm / gravity / inertia measurements have a model to read; pass 2
# overwrites every one of them with the measured values. The placeholder is 0 - not the
# old +-2.0 N*m - because a value that looks like the pre-revision limit could
# survive into a committed XML by accident, and because nothing measures the pass-1
# model: the arm sweep is pure kinematics and the torque tables use no contact.
PLACEHOLDER_FORCERANGE_NM = 0.0

TEMPLATE_HEAD = """<mujoco model="allegro_floating">
  <!-- GENERATED by grounding/assets/build_allegro_floating.py - do not hand-edit.
       Base: MuJoCo Menagerie {repo} @ {commit} (wonik_allegro/right_hand.xml).
       Augments: actuatable 3-slide + 3-hinge floating base (6D wrist actuators),
       tissue-box proxy (freejoint), fixed cameras cam_overhead / cam_wrist.
       Wave 3 revision: finger forcerange derived from the measured moment arm to
       the fingertip at the Allegro V5 10 N/finger limit, widened wrist_z/wrist_ry
       travel, stiffened box<->finger contacts, named hand geoms, and an explicit
       stiff palm<->box contact pair (Wave 3.5-B) and stiff finger-proximal<->box
       contact pairs (Wave 3.5-E). -->
  <compiler angle="radian" meshdir="{meshdir}" autolimits="true"/>

  <option cone="elliptic" impratio="10" timestep="0.002" integrator="implicitfast"/>

{defaults}

  <asset>
    <material name="black" rgba="0.2 0.2 0.2 1"/>
    <material name="white" rgba="0.9 0.9 0.9 1"/>
    <material name="table" rgba="0.42 0.36 0.30 1"/>
    <material name="box" rgba="0.30 0.55 0.85 1"/>

{assets}
  </asset>

  <worldbody>
    <light pos="0 0 1.6" dir="0 0 -1" directional="true"/>
    <light pos="0.3 -0.4 1.4" dir="0 0 -1" directional="true"/>
    <geom name="floor" type="plane" size="0 0 0.05" rgba="0.25 0.25 0.28 1"
      friction="0.8 0.02 0.002"/>
    <geom name="table_slab" type="box" size="0.45 0.45 0.02" pos="0 0 {slab_z}"
      material="table" friction="0.9 0.02 0.002"/>
    <geom name="pedestal" type="box" size="0.30 0.30 {ped_half}" pos="0 0 {ped_z}"
      material="table" friction="0.9 0.02 0.002"/>

    <camera name="cam_overhead" mode="fixed" pos="0 0 1.55" xyaxes="1 0 0 0 1 0"
      fovy="45"/>
    <camera name="cam_wrist" mode="fixed" pos="0 0.16 0.10" xyaxes="1 0 0 0 0 -1"
      fovy="70"/>

    <body name="object" pos="0 0 {box_z}">
      <freejoint name="object_free"/>
      <geom name="object_geom" type="box" size="{bx} {by} {bz}" mass="{mass}"
        rgba="0.30 0.55 0.85 1" material="box" condim="4"
        friction="1.4 0.02 0.004" solref="{solref}" solimp="{solimp}"/>{lip}{chamfer}
    </body>

    <body name="hand_root" pos="{hand_home}">
{wrist_joints}
{palm}
    </body>
  </worldbody>

  <contact>
{excludes}
{pairs}
  </contact>

  <actuator>
{wrist_actuators}
{hand_actuators}
  </actuator>
</mujoco>
"""


# ===================================================================== W4/R8
def box_lip_specs(lip_overhang_m: float = BOX_LIP_OVERHANG_M,
                  lip_h_m: float = BOX_LIP_HEIGHT_M) -> list[dict]:
    """The two lip geoms, in the object body's frame.

    One flange per +-x side, flush with the box top (``z + bz``) and sitting on
    the outer ``lip_h`` of the wall, so its underside at ``z = bz - lip_h`` is a
    horizontal face overhanging empty space by exactly ``overhang``. That face is
    the hook target: a fingertip below it pushing up gets a normal force with no
    friction term at all.

    ``lip_overhang_m <= 0`` or ``lip_h_m <= 0`` yields an empty list, i.e. the
    bare box, which is what the parameter sweep's control cell uses.
    """
    if lip_overhang_m <= 0.0 or lip_h_m <= 0.0:
        return []
    bx, _by, bz = BOX_HALF_M
    half = (lip_overhang_m, BOX_LIP_HALF_Y_M, lip_h_m / 2.0)
    vol = 8.0 * half[0] * half[1] * half[2]
    mass = BOX_LIP_DENSITY_KG_M3 * vol
    out = []
    for name, sign in zip(BOX_LIP_GEOM_NAMES, (-1.0, 1.0)):
        out.append({
            "name": name,
            "size": half,
            # inner face flush with the wall, protruding `overhang` outward
            "pos": (sign * (bx + lip_overhang_m), 0.0, bz - lip_h_m / 2.0),
            "mass": mass,
        })
    return out


def lip_total_mass(lip_overhang_m: float = BOX_LIP_OVERHANG_M,
                   lip_h_m: float = BOX_LIP_HEIGHT_M) -> float:
    return sum(s["mass"] for s in box_lip_specs(lip_overhang_m, lip_h_m))


def box_chamfer_specs(chamfer_m: float = BOX_CHAMFER_M) -> list[dict]:
    """A 45-degree chamfer plate per top +-x edge, in the object body's frame.

    APPROXIMATION, stated rather than hidden: a true chamfer removes the
    corner, which a box primitive cannot express.  This is a thin plate lying
    on the chamfer plane (outer face exactly on the ideal 45-degree cut),
    while the box's own sharp corner stays underneath it and pokes past the
    plate by at most ``chamfer / sqrt(2)`` at the vertex.  For a hook probe
    that is the right approximation -- the finger lands on the plate's outer
    face, which is the ramp -- but it is not exact CSG, and any conclusion
    that depends on the sub-millimetre profile of that vertex has to say so.

    Mass uses the same density convention as the lip and is SUBTRACTED from
    the box (see box_geom_mass): the plate replaces edge material, it does
    not stack on top of 0.15 kg.
    """
    if chamfer_m <= 0.0:
        return []
    bx, _by, bz = BOX_HALF_M
    t = BOX_CHAMFER_T_M
    # numpy is imported lazily in this module, so use literal roots here
    face_len = chamfer_m * 2.0 ** 0.5  # from (bx-c, bz) to (bx, bz-c)
    vol = face_len * (2.0 * BOX_LIP_HALF_Y_M) * t
    mass = BOX_LIP_DENSITY_KG_M3 * vol
    inset = t / (2.0 * 2.0 ** 0.5)
    out = []
    for name, sign in zip(BOX_CHAMFER_GEOM_NAMES, (-1.0, 1.0)):
        out.append({
            "name": name,
            "size": (face_len / 2.0, BOX_LIP_HALF_Y_M, t / 2.0),
            "euler": f"0 {sign * -45.0:g} 0",
            # outer face flush with the ideal chamfer plane
            "pos": (sign * (bx - chamfer_m / 2.0 + inset), 0.0,
                    bz - chamfer_m / 2.0 + inset),
            "mass": mass,
        })
    return out


def chamfer_total_mass(chamfer_m: float = BOX_CHAMFER_M) -> float:
    return sum(s["mass"] for s in box_chamfer_specs(chamfer_m))


def box_geom_mass(lip_overhang_m: float = BOX_LIP_OVERHANG_M,
                  lip_h_m: float = BOX_LIP_HEIGHT_M,
                  chamfer_m: float = BOX_CHAMFER_M) -> float:
    """Mass of the ``object_geom`` box itself: 0.15 kg MINUS the lip.

    The 0.15 kg is the frozen number every force gate is written against
    (A3a's 1.4715 N sufficiency bound, the pedestal-load ratios), so the lip's
    mass has to be taken out of the box rather than added on top. The sum is
    checked in the emitted XML by tests/test_smoke6_mjcf.py, which is the only
    place that assertion can be trusted, because it reads the compiled model.
    """
    extra = lip_total_mass(lip_overhang_m, lip_h_m) + chamfer_total_mass(chamfer_m)
    remaining = BOX_MASS_KG - extra
    if remaining <= 0.0:
        raise ValueError(
            f"lip ({lip_overhang_m * 1e3:.1f} mm x {lip_h_m * 1e3:.1f} mm) plus "
            f"chamfer ({chamfer_m * 1e3:.1f} mm) weighs {extra:.4f} kg, at or over "
            f"the whole {BOX_MASS_KG} kg box - there is no mass left for the box itself"
        )
    return remaining


def box_object_mass(lip_overhang_m: float = BOX_LIP_OVERHANG_M,
                    lip_h_m: float = BOX_LIP_HEIGHT_M,
                    chamfer_m: float = BOX_CHAMFER_M) -> float:
    """Total object-body mass, lip and chamfer included. Must equal BOX_MASS_KG exactly."""
    return (box_geom_mass(lip_overhang_m, lip_h_m, chamfer_m)
            + lip_total_mass(lip_overhang_m, lip_h_m)
            + chamfer_total_mass(chamfer_m))


def object_geom_names(lip_overhang_m: float = BOX_LIP_OVERHANG_M,
                      lip_h_m: float = BOX_LIP_HEIGHT_M,
                      chamfer_m: float = BOX_CHAMFER_M) -> list[str]:
    """Every geom on the ``object`` body, in MJCF order."""
    return (["object_geom"]
            + [s["name"] for s in box_lip_specs(lip_overhang_m, lip_h_m)]
            + [s["name"] for s in box_chamfer_specs(chamfer_m)])


def lip_geom_lines(lip_overhang_m: float = BOX_LIP_OVERHANG_M,
                   lip_h_m: float = BOX_LIP_HEIGHT_M) -> str:
    """The lip geoms as MJCF text, indented to sit inside the object body.

    Returns a string that starts with a newline and ends with one, so the caller
    can append it directly after the ``object_geom`` line and have a zero-lip
    scene come out byte-identical to a bare one - that exactness is what makes
    ``build(0, 0)`` a usable A/B reference (see BOX_LIP_OVERHANG_M).

    They carry the SAME contact attributes as ``object_geom`` - friction, condim
    and the R3 solref/solimp - because a lip that resolved its contacts at
    MuJoCo's soft default would be a second, undocumented contact convention,
    which is exactly the defect R3 and W3.5-B/E exist to prevent. The explicit
    <pair> list is extended to the lips for the same reason.
    """
    specs = box_lip_specs(lip_overhang_m, lip_h_m)
    if not specs:
        return ""
    rows = []
    for s in specs:
        size = " ".join(f"{v:g}" for v in s["size"])
        pos = " ".join(f"{v:g}" for v in s["pos"])
        rows.append(
            f'      <geom name="{s["name"]}" type="box" size="{size}" pos="{pos}"\n'
            f'        mass="{s["mass"]:.6f}" rgba="0.30 0.55 0.85 1" material="box"\n'
            f'        condim="4" friction="1.4 0.02 0.004"\n'
            f'        solref="{_fmt_pair(CONTACT_SOLREF)}"\n'
            f'        solimp="{_fmt_pair(CONTACT_SOLIMP)}"/>'
        )
    return "\n" + "\n".join(rows) + "\n"


def chamfer_geom_lines(chamfer_m: float = BOX_CHAMFER_M) -> str:
    """The chamfer plates as MJCF text, same conventions as ``lip_geom_lines``.

    Same contact attributes as ``object_geom`` (friction, condim, R3
    solref/solimp) and the explicit ``<pair>`` list extends to them via
    ``object_geom_names`` -- a chamfer that resolved its contacts at MuJoCo's
    soft default would be a second, undocumented contact convention.  Empty
    string at ``chamfer_m <= 0``, which is what makes ``build(chamfer_m=0)``
    byte-identical to the bare box.
    """
    specs = box_chamfer_specs(chamfer_m)
    if not specs:
        return ""
    rows = []
    for s in specs:
        size = " ".join(f"{v:g}" for v in s["size"])
        pos = " ".join(f"{v:g}" for v in s["pos"])
        rows.append(
            f'      <geom name="{s["name"]}" type="box" size="{size}" pos="{pos}"\n'
            f'        euler="{s["euler"]}" mass="{s["mass"]:.6f}"'
            f' rgba="0.30 0.55 0.85 1" material="box"\n'
            f'        condim="4" friction="1.4 0.02 0.004"\n'
            f'        solref="{_fmt_pair(CONTACT_SOLREF)}"\n'
            f'        solimp="{_fmt_pair(CONTACT_SOLIMP)}"/>'
        )
    return "\n" + "\n".join(rows) + "\n"


# ===================================================================== R1/R2
# Everything below is measured from the *model*, never from a literal. The model
# is loaded from the XML this same script writes, so the numbers in REVISION.md and
# the numbers in the XML cannot drift apart: rebuild and the table is recomputed.


def _mujoco():
    import mujoco  # noqa: PLC0415  - optional at generator-import time
    return mujoco


def _load(mjcf_path: pathlib.Path | None = None):
    mujoco = _mujoco()
    return mujoco.MjModel.from_xml_path(str(mjcf_path or OUT))


def finger_forcerange(moment_arm_m: float) -> float:
    """R1. The torque limit of one finger joint, from its moment arm.

    Each joint gets an equal share (1/4) of its fingertip's 10 N budget, converted
    through that joint's own moment arm. Because the arm is a *minimum* over the
    reachable workspace, ``tau / arm`` is the largest fingertip force that joint can
    produce, so the four joints of a finger sum to at most the full 10 N budget in
    any configuration - which is what the spec is stated against.
    """
    return FINGER_FORCE_SHARE * FINGER_TIP_FORCE_LIMIT_N * moment_arm_m


def finger_qpos_and_actuator_adrs(model) -> tuple[np.ndarray, np.ndarray]:  # type: ignore[name-defined]
    """qpos / actuator row indices of the 16 finger joints, in FINGER_JOINTS order."""
    import numpy as np  # noqa: PLC0415

    mujoco = _mujoco()
    qadr = np.array(
        [model.jnt_qposadr[mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, n)]
         for n in FINGER_JOINTS], dtype=int)
    actr = np.array(
        [mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_ACTUATOR, FINGER_ACTUATOR_OF[n])
         for n in FINGER_JOINTS], dtype=int)
    if (actr < 0).any():
        missing = [FINGER_JOINTS[k] for k in np.nonzero(actr < 0)[0]]
        raise RuntimeError(f"no actuator found for joints {missing}")
    return qadr, actr


def finger_actuator_id(model, joint_name: str) -> int:
    """Actuator row driving ``joint_name``. Goes through FINGER_ACTUATOR_OF because a
    joint name is not an actuator name upstream."""
    aid = _mujoco().mj_name2id(model, _mujoco().mjtObj.mjOBJ_ACTUATOR,
                                FINGER_ACTUATOR_OF[joint_name])
    if aid < 0:
        raise KeyError(f"no actuator drives {joint_name}")
    return aid


def _tip_geom_ids(model) -> dict[str, int]:
    mujoco = _mujoco()
    out: dict[str, int] = {}
    for f in FINGER_PREFIXES:
        bid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, f"{f}_tip")
        gids = [g for g in range(model.ngeom)
                if model.geom_bodyid[g] == bid
                and model.geom_type[g] == mujoco.mjtGeom.mjGEOM_CAPSULE]
        if len(gids) != 1:
            raise RuntimeError(f"{f}_tip owns {len(gids)} capsule geoms, expected 1")
        out[f] = gids[0]
    return out


def _palm_geom_id(model) -> int:
    mujoco = _mujoco()
    palm_b = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "palm")
    gids = [g for g in range(model.ngeom)
            if model.geom_bodyid[g] == palm_b
            and model.geom_type[g] == mujoco.mjtGeom.mjGEOM_BOX]
    if len(gids) != 1:
        raise RuntimeError(f"palm owns {len(gids)} box collision geoms, expected 1")
    return gids[0]


def _self_collides_with_palm(model, data) -> bool:
    """True when a finger link is already touching the palm. A pose MuJoCo can
    integrate but where the finger is folded into the palm is not a grasp pose: the
    fingertip there cannot be pressing an external object, and its moment arms are
    not the ones a grasp exploits."""
    mujoco = _mujoco()
    palm_b = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "palm")
    for c in range(data.ncon):
        b1 = int(model.geom_bodyid[int(data.contact[c].geom1)])
        b2 = int(model.geom_bodyid[int(data.contact[c].geom2)])
        if palm_b not in (b1, b2):
            continue
        other = b2 if b1 == palm_b else b1
        name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_BODY, other) or ""
        if name.endswith(("_proximal", "_medial", "_distal", "_tip")):
            return True
    return False


def moment_arm_table(mjcf_path: pathlib.Path | None = None) -> dict[str, float]:
    """R1. Minimum moment arm from each finger joint's axis to its own fingertip.

    The moment arm is the perpendicular distance from the joint axis to the
    fingertip geom's centre. It is what converts a joint torque into a fingertip
    force (F = tau / arm), so it is the only geometric quantity that decides how
    much torque the joint may be given.

    It is *configuration dependent*, so the arm that matters is the smallest one the
    hand can actually be in while grasping: at a small arm the same torque makes a
    much larger fingertip force. The minimum is therefore taken over the reachable
    workspace, with two classes of pose excluded because they are not grasp poses:

      * poses where a finger link is already in contact with the palm (the finger is
        folded into itself; the fingertip is not pressing anything external);
      * poses where a joint axis passes within the fingertip capsule's own radius of
        the fingertip centre. There the axis pierces the fingertip pad, so the pad
        cannot be transmitting a force about that axis at all - including these
        would drive the arm toward zero and make the table meaningless.

    Deterministic: fixed seed, fixed sample count.
    """
    import numpy as np

    mujoco = _mujoco()
    model = _load(mjcf_path)
    data = mujoco.MjData(model)
    jids = [mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, n) for n in FINGER_JOINTS]
    qadr = [int(model.jnt_qposadr[j]) for j in jids]
    lo = np.array([model.jnt_range[j][0] for j in jids])
    hi = np.array([model.jnt_range[j][1] for j in jids])
    tip = _tip_geom_ids(model)
    radii = {f: float(model.geom_size[gid][0]) for f, gid in tip.items()}

    rng = np.random.default_rng(ARM_SWEEP_SEED)
    samples = rng.uniform(lo, hi, size=(ARM_SWEEP_SAMPLES, len(FINGER_JOINTS)))
    best = {n: np.inf for n in FINGER_JOINTS}
    for i in range(ARM_SWEEP_SAMPLES):
        data.qpos[:] = 0.0
        for k in range(len(FINGER_JOINTS)):
            data.qpos[qadr[k]] = samples[i, k]
        mujoco.mj_kinematics(model, data)
        mujoco.mj_forward(model, data)
        if _self_collides_with_palm(model, data):
            continue
        for ki, f in enumerate(FINGER_PREFIXES):
            p = np.array(data.geom_xpos[tip[f]])
            arms = []
            for j in range(4):
                jid = jids[4 * ki + j]
                axis = np.array(model.jnt_axis[jid])
                origin = np.array(data.xpos[model.jnt_bodyid[jid]])
                v = p - origin
                v = v - axis * float(v @ axis)
                arms.append(float(np.linalg.norm(v)))
            if min(arms) < radii[f]:
                continue  # a joint axis pierces the fingertip pad
            for j in range(4):
                name = FINGER_JOINTS[4 * ki + j]
                if arms[j] < best[name]:
                    best[name] = arms[j]
    missing = [n for n, v in best.items() if not np.isfinite(v)]
    if missing:
        raise RuntimeError(f"moment arm sweep found no usable pose for {missing}")
    return best


def gravity_torque_table(mjcf_path: pathlib.Path | None = None) -> dict[str, float]:
    """Worst-case |gravity torque| about each finger joint axis over the reachable
    joint space. This is the load kp has to be able to hold (R1's kp check)."""
    import numpy as np

    mujoco = _mujoco()
    model = _load(mjcf_path)
    saved = (model.geom_contype.copy(), model.geom_conaffinity.copy())
    model.geom_contype[:] = 0
    model.geom_conaffinity[:] = 0
    try:
        data = mujoco.MjData(model)
        jids = [mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, n) for n in FINGER_JOINTS]
        qadr = [int(model.jnt_qposadr[j]) for j in jids]
        dofs = [int(model.jnt_dofadr[j]) for j in jids]
        lo = np.array([model.jnt_range[j][0] for j in jids])
        hi = np.array([model.jnt_range[j][1] for j in jids])
        rng = np.random.default_rng(ARM_SWEEP_SEED + 1)
        samples = rng.uniform(lo, hi, size=(ARM_SWEEP_SAMPLES, len(FINGER_JOINTS)))
        worst = np.zeros(len(FINGER_JOINTS))
        for i in range(ARM_SWEEP_SAMPLES):
            data.qpos[:] = 0.0
            for k in range(len(FINGER_JOINTS)):
                data.qpos[qadr[k]] = samples[i, k]
            data.qvel[:] = 0.0
            mujoco.mj_forward(model, data)
            for k in range(len(FINGER_JOINTS)):
                worst[k] = max(worst[k], abs(float(data.qfrc_bias[dofs[k]])))
        return {n: float(worst[k]) for k, n in enumerate(FINGER_JOINTS)}
    finally:
        model.geom_contype[:], model.geom_conaffinity[:] = saved


def effective_inertia_table(mjcf_path: pathlib.Path | None = None) -> dict[str, float]:
    """Minimum diagonal effective inertia about each finger joint axis over the
    reachable joint space. R1's other kp constraint: the servo bandwidth
    sqrt(kp / I) must stay resolvable at the 2 ms timestep."""
    import numpy as np

    mujoco = _mujoco()
    model = _load(mjcf_path)
    data = mujoco.MjData(model)
    jids = [mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, n) for n in FINGER_JOINTS]
    qadr = [int(model.jnt_qposadr[j]) for j in jids]
    dofs = [int(model.jnt_dofadr[j]) for j in jids]
    lo = np.array([model.jnt_range[j][0] for j in jids])
    hi = np.array([model.jnt_range[j][1] for j in jids])
    rng = np.random.default_rng(ARM_SWEEP_SEED + 2)
    samples = rng.uniform(lo, hi, size=(ARM_SWEEP_SAMPLES, len(FINGER_JOINTS)))
    M = np.zeros((model.nv, model.nv))
    worst = np.full(len(FINGER_JOINTS), np.inf)
    for i in range(ARM_SWEEP_SAMPLES):
        data.qpos[:] = 0.0
        for k in range(len(FINGER_JOINTS)):
            data.qpos[qadr[k]] = samples[i, k]
        data.qvel[:] = 0.0
        mujoco.mj_forward(model, data)
        mujoco.mj_fullM(model, data, M)
        for k in range(len(FINGER_JOINTS)):
            worst[k] = min(worst[k], float(M[dofs[k], dofs[k]]))
    return {n: float(worst[k]) for k, n in enumerate(FINGER_JOINTS)}


def finger_kp_table(arms: dict[str, float], gravity: dict[str, float],
                    inertia: dict[str, float], spans: dict[str, float],
                    dt: float) -> dict[str, float]:
    """R1's kp. Two measured lower bounds and one measured upper bound:

      reach   kp >= forcerange / joint_span
              The servo's torque is kp * (ctrl - qpos), and the largest error the
              ctrlrange can command is the joint's own span. Below this the
              actuator saturates below its declared forcerange and the forcerange
              is a fiction the policy can never use.

      hold    kp >= gravity_torque / GRAVITY_SAG_BUDGET_RAD
              Otherwise the finger cannot hold its own weight within the sag budget.

      stable  kp <= SERVO_BANDWIDTH_GUARD^2 * I / dt^2
              sqrt(kp / I) is the servo's natural frequency; above ~1 rad per
              timestep the 2 ms integration rings. I is the *minimum* effective
              inertia over the workspace, so this is the worst case.
    """
    import math

    out: dict[str, float] = {}
    for n in FINGER_JOINTS:
        tau = finger_forcerange(arms[n])
        kp_reach = tau / spans[n]
        kp_hold = gravity[n] / GRAVITY_SAG_BUDGET_RAD
        kp_stable = SERVO_BANDWIDTH_GUARD ** 2 * inertia[n] / (dt * dt)
        kp = min(max(kp_reach, kp_hold), kp_stable)
        if kp <= 0.0:
            raise RuntimeError(f"{n}: no feasible kp (reach {kp_reach}, hold {kp_hold}, "
                               f"stable <= {kp_stable})")
        # Round UP to the written precision. Rounding to nearest can land a hair below
        # kp_hold (it is the binding bound on most joints) and silently break the
        # "must hold its own weight" guarantee the table above is claiming.
        out[n] = math.ceil(kp * 1e4) / 1e4
    return out


# ===================================================================== R5
def measure_resting_sink(mjcf_path: pathlib.Path | None = None, *,
                         solref_timeconst: float | None = None) -> float:
    """R5's baseline: the deepest the box sinks into the pedestal while resting.

    This is MuJoCo's own numerical compliance in *this* scene, so it is the floor of
    any penetration tolerance - a tolerance tighter than this would fail on a box
    that is not penetrating anything. Measured by simulation, never assumed.

    ``solref_timeconst`` overrides the contact stiffness, which is how the tests
    prove the function tracks the contact model instead of returning a constant.
    """
    import numpy as np

    mujoco = _mujoco()
    model = _load(mjcf_path)
    if solref_timeconst is not None:
        tc, ratio = CONTACT_SOLREF
        _apply_contact_solref(model, (solref_timeconst, ratio))
    data = mujoco.MjData(model)
    obj = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, "object_geom")
    for _ in range(RESTING_SETTLE_STEPS):
        mujoco.mj_step(model, data)
    del np
    return _max_support_penetration(model, data, obj)


def _max_support_penetration(model, data, obj_geom: int) -> float:
    pen = 0.0
    for c in range(data.ncon):
        con = data.contact[c]
        g1, g2 = int(con.geom1), int(con.geom2)
        if obj_geom not in (g1, g2):
            continue
        other = g2 if g1 == obj_geom else g1
        if int(model.geom_bodyid[other]) == 0:  # a world-body support geom
            pen = max(pen, -float(con.dist))
    return float(pen)


def _apply_contact_solref(model, solref) -> None:
    """Set solref/solimp on the object and on every finger collision geom.

    MuJoCo's contact solver takes the *stiffer* of the two sides of a pair, so
    writing both sides makes the pair's stiffness unambiguous rather than dependent
    on the pair ordering."""
    mujoco = _mujoco()
    obj = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, "object_geom")
    gids = [obj] + list(_tip_geom_ids(model).values())
    for g in gids:
        model.geom_solref[g] = solref
        model.geom_solimp[g] = CONTACT_SOLIMP


def penetration_tolerance(mjcf_path: pathlib.Path | None = None, *,
                          solref_timeconst: float | None = None) -> float:
    """R5. tolerance = PENETRATION_TOLERANCE_K x the measured resting sink.

    Derived, not hardcoded: smoke5's old 1e-3 m was a number typed next to a
    2.39e-4 m measurement, and it stayed put through two rounds of contact changes
    that moved the baseline. k >= 3 keeps it clear of the solver's own resting
    tolerance."""
    return PENETRATION_TOLERANCE_K * measure_resting_sink(
        mjcf_path, solref_timeconst=solref_timeconst)


# ===================================================================== R4
def _geom_name(body_name: str, role: str) -> str:
    return f"{body_name}_{role}"


def _name_hand_geoms(palm_el: ET.Element) -> None:
    """R4. Give every geom under <body name="palm"> an explicit name.

    Upstream they are all produced by unnamed <default> classes, which is exactly
    why a name-based contact detector once reported a structural zero. The role is
    read off the geom's class so the name says which of the pair is the colliding
    one - ``ff_proximal_collision`` vs ``ff_proximal_visual``."""
    for body in palm_el.iter("body"):
        body_name = body.get("name")
        if not body_name:
            continue
        for geom in body.findall("geom"):
            cls = geom.get("class") or ""
            if geom.get("name"):
                continue
            role = "visual" if "visual" in cls else "collision"
            geom.set("name", _geom_name(body_name, role))


def _annotate_hand_geoms(palm_el: ET.Element, solref, solimp) -> None:
    """R3. Stamp the stiffened contact onto the four fingertip pads only.

    See CONTACT_SOLREF's comment for why this is not "every colliding hand geom":
    widening it to the palm turns an unrelated table graze into a 24x force spike.
    The fingertips are identifiable by their body name (``*_tip``) and by their
    capsule geom class, so this cannot silently start matching other links."""
    for body in palm_el.iter("body"):
        body_name = body.get("name") or ""
        if not body_name.endswith("_tip"):
            continue
        for geom in body.findall("geom"):
            cls = geom.get("class") or ""
            if "visual" in cls or "tip" not in cls:
                continue  # the visual pad mesh is not a collision geom
            geom.set("solref", f"{solref[0]:g} {solref[1]:g}")
            geom.set("solimp", " ".join(f"{v:g}" for v in solimp))


# ===================================================================== emit
def _fmt_pair(values) -> str:
    return " ".join(f"{v:g}" for v in values)


def _wrist_joint_lines() -> list[str]:
    lines = []
    for name in ("wrist_x", "wrist_y", "wrist_z", "wrist_rx", "wrist_ry", "wrist_rz"):
        kind, axis = WRIST_AXIS[name]
        if name in WRIST_TRAVEL_ADD_M:
            prev_lo, prev_hi, add = WRIST_TRAVEL_ADD_M[name]
            lo, hi = prev_lo - add, prev_hi
        else:
            lo, hi = WRIST_UNCHANGED[name]
        damping = WRIST_DAMPING[kind]
        armature = WRIST_ARMATURE[kind]
        lines.append(
            f'      <joint name="{name}" type="{kind}" axis="{axis}" '
            f'range="{lo:g} {hi:g}"\n'
            f'        damping="{damping}" armature="{armature}"/>'
        )
    return lines


def _wrist_actuator_lines() -> list[str]:
    lines = []
    for jname in ("wrist_x", "wrist_y", "wrist_z", "wrist_rx", "wrist_ry", "wrist_rz"):
        aname = WRIST_ACTUATOR_OF[jname]
        spec = WRIST_ACTUATOR[aname]
        if jname in WRIST_TRAVEL_ADD_M:
            prev_lo, prev_hi, add = WRIST_TRAVEL_ADD_M[jname]
            lo, hi = prev_lo - add, prev_hi
        else:
            lo, hi = WRIST_UNCHANGED[jname]
        lines.append(
            f'    <position name="{aname}" joint="{jname}" kp="{spec["kp"]}" '
            f'ctrlrange="{lo:g} {hi:g}"\n'
            f'      forcerange="{spec["forcerange"]}"/>'
        )
    return lines


def _indent(elem: ET.Element, level: int) -> str:
    """Serialize one upstream subtree back to MJCF text at the requested indent."""
    pad = " " * level
    xml_text = ET.tostring(elem, encoding="unicode")
    pretty = xml.dom.minidom.parseString(xml_text).toprettyxml(indent="  ")
    lines = [
        ln.rstrip()
        for ln in pretty.splitlines()
        if ln.strip() and not ln.strip().startswith("<?xml")
    ]
    return "\n".join(pad + ln for ln in lines)


def _patch_defaults(raw: str) -> str:
    """Free-floating hand needs a stiffer posture servo than a table-mounted one.

    The upstream hand was authored for a hand bolted to a table, so joint damping is
    raised from 0.1. The <position kp> default is left at the upstream 1: R1 writes
    an explicit per-joint kp on every finger actuator, derived from the measured
    moment arm and gravity torque, so the class default never applies to the hand."""
    return raw.replace('damping=".1"', 'damping="0.5"')


def _upstream_parts() -> tuple[str, str, str, str, list]:
    """Parse upstream into (defaults, assets, palm, excludes, hand_acts)."""
    raw_text = _patch_defaults(UPSTREAM_HAND.read_text())
    root = ET.fromstring(raw_text)

    # ---- defaults: patch on raw text, then re-serialize ----------------
    defaults_root = root.find("default")
    assert defaults_root is not None
    defaults = _indent(defaults_root, 4)
    assert defaults.count("<default") >= 5, "upstream default block looks truncated"

    # ---- assets --------------------------------------------------------
    asset_el = root.find("asset")
    assert asset_el is not None
    assets_lines = []
    for mesh in asset_el.findall("mesh"):
        assets_lines.append(f'    <mesh file="{mesh.get("file")}"/>')
    assets = "\n".join(assets_lines)

    # ---- palm subtree: name + contact stiffness (R4, R3) ----------------
    palm_el = None
    for body in root.findall("./worldbody/body"):
        if body.get("name") == "palm":
            palm_el = body
    assert palm_el is not None, "palm body not found upstream"
    _name_hand_geoms(palm_el)
    _annotate_hand_geoms(palm_el, CONTACT_SOLREF, CONTACT_SOLIMP)
    palm = _indent(palm_el, 6)

    excludes = "\n".join(
        f'    <exclude body1="{ex.get("body1")}" body2="{ex.get("body2")}"/>'
        for ex in root.findall("./contact/exclude")
    )

    # ---- hand actuators: R1, two passes --------------------------------
    # Pass 1 writes the tree with placeholder finger actuators so the model can be
    # loaded and *measured*; pass 2 writes the measured numbers. The measurement is
    # therefore a property of the geometry the XML actually contains, not of a
    # hand-maintained table.
    hand_acts = [(a.get("name"), a.get("joint"), a.get("class")) for a in
                 root.findall("./actuator/position")]
    assert len(hand_acts) == 16, f"expected 16 hand joints, got {len(hand_acts)}"
    return defaults, assets, palm, excludes, hand_acts


def build(lip_overhang_m: float = BOX_LIP_OVERHANG_M,
          lip_h_m: float = BOX_LIP_HEIGHT_M,
          chamfer_m: float = BOX_CHAMFER_M,
          out: pathlib.Path | None = None,
          *, verbose: bool = False) -> pathlib.Path:
    """Write the MJCF for one lip setting. Returns the path written.

    Two passes, as before: pass 1 emits placeholder finger forceranges so the model
    can be loaded and MEASURED, pass 2 writes the measured numbers. The measurement
    is a property of the geometry the XML actually contains, not a hand-maintained
    table - which is why this is a function of ``lip_*`` rather than a constant.

    The lip settings are parameters with no default in the measurement path, so a
    sweep over ``overhang x lip_h`` calls this per cell and gets a self-consistent
    scene each time. ``lip_overhang_m <= 0`` reproduces the bare box exactly, which
    is the control every sweep cell is compared against.
    """
    if not UPSTREAM_HAND.exists():
        raise SystemExit(
            f"missing {UPSTREAM_HAND}\n"
            f"run: git clone --depth 1 {UPSTREAM_REPO} runs/upstream/menagerie"
        )
    # absolute, because MESH_DIR inside the XML is relative to the XML's own
    # directory: a relative `out` makes the mesh lookup depend on the caller's cwd
    out = pathlib.Path(out).resolve() if out is not None else OUT.resolve()
    defaults, assets, palm, excludes, hand_acts = _upstream_parts()

    placeholder = _emit(assets, defaults, palm, excludes, hand_acts, None,
                        lip_overhang_m=lip_overhang_m, lip_h_m=lip_h_m,
                        chamfer_m=chamfer_m)
    out.write_text(placeholder)
    _verify_out(placeholder, out)

    model = _load(out)
    arms = moment_arm_table(out)
    gravity = gravity_torque_table(out)
    inertia = effective_inertia_table(out)
    mujoco = _mujoco()
    spans = {}
    for n in FINGER_JOINTS:
        jid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, n)
        spans[n] = float(model.jnt_range[jid][1] - model.jnt_range[jid][0])
    kp = finger_kp_table(arms, gravity, inertia, spans, float(model.opt.timestep))

    final = _emit(assets, defaults, palm, excludes, hand_acts, (arms, kp),
                  lip_overhang_m=lip_overhang_m, lip_h_m=lip_h_m,
                  chamfer_m=chamfer_m)
    out.write_text(final)
    _verify_out(final, out)

    if verbose:
        print(f"wrote {out} ({len(final)} bytes, upstream commit {UPSTREAM_COMMIT})")
        print("R1 finger forcerange = (10 N / 4) x moment arm  [sum over a finger's 4 "
              "joints = 10 N at the fingertip]:")
        for n in FINGER_JOINTS:
            print(f"    {n:6s} arm={arms[n] * 1e3:7.3f} mm  "
                  f"tau=+/-{finger_forcerange(arms[n]):.4f}  "
                  f"kp={kp[n]:.4f}  gravity={gravity[n]:.4f} N*m  I={inertia[n]:.3e}")
        print(f"R3 contact solref={CONTACT_SOLREF} solimp={CONTACT_SOLIMP[:2]}")
        print(f"W4/R8 lip overhang={lip_overhang_m * 1e3:g} mm lip_h={lip_h_m * 1e3:g} mm  "
              f"lip mass={lip_total_mass(lip_overhang_m, lip_h_m) * 1e3:.3f} g  "
              f"box mass={box_geom_mass(lip_overhang_m, lip_h_m):.6f} kg  "
              f"total={box_object_mass(lip_overhang_m, lip_h_m, chamfer_m):.6f} kg")
        print(f"W3.5-B/E explicit pairs ({len(box_pairs(lip_overhang_m=lip_overhang_m, lip_h_m=lip_h_m, chamfer_m=chamfer_m))}): "
              f"{', '.join(f'{a}<->{b}' for a, b in box_pairs(lip_overhang_m=lip_overhang_m, lip_h_m=lip_h_m, chamfer_m=chamfer_m))}")
        sink = measure_resting_sink(out)
        print(f"R5 resting sink = {sink:.4e} m -> tolerance = {PENETRATION_TOLERANCE_K} x "
              f"sink = {PENETRATION_TOLERANCE_K * sink:.4e} m")
    return out


def main() -> None:
    build(verbose=True)


def _verify_out(xml_text: str, out: pathlib.Path | None = None) -> None:
    """Refuse to leave a broken artefact on disk."""
    import tempfile

    out = pathlib.Path(out) if out is not None else OUT
    with tempfile.NamedTemporaryFile("w", suffix=".xml", delete=False,
                                     dir=str(out.parent)) as fh:
        # meshdir is relative to the XML's own directory, so the temp file must live
        # next to the real one for the load to resolve the meshes.
        fh.write(xml_text)
        tmp = fh.name
    try:
        _load(pathlib.Path(tmp))
    finally:
        pathlib.Path(tmp).unlink(missing_ok=True)


def _emit(assets: str, defaults: str, palm: str, excludes: str,
          hand_acts, measured, *,
          lip_overhang_m: float = BOX_LIP_OVERHANG_M,
          lip_h_m: float = BOX_LIP_HEIGHT_M,
          chamfer_m: float = BOX_CHAMFER_M) -> str:
    if measured is None:
        arms = kp = None
    else:
        arms, kp = measured
    hand_actuator_lines = []
    for name, joint, cls in hand_acts:
        if arms is None:
            hand_actuator_lines.append(
                f'    <position name="{name}" joint="{joint}" class="{cls}" '
                f'forcerange="{-PLACEHOLDER_FORCERANGE_NM:g} '
                f'{PLACEHOLDER_FORCERANGE_NM:g}" kp="1"/>')
        else:
            tau = finger_forcerange(arms[joint])
            hand_actuator_lines.append(
                f'    <position name="{name}" joint="{joint}" class="{cls}" '
                f'kp="{kp[joint]:g}" forcerange="{-tau:.5f} {tau:.5f}"/>')
    def _box_pair(g1: str, g2: str) -> str:
        return (
            f'    <pair geom1="{g1}" geom2="{g2}" '
            f'solref="{_fmt_pair(CONTACT_SOLREF)}" '
            f'solimp="{_fmt_pair(CONTACT_SOLIMP)}" condim="4"/>'
        )

    # the template puts {pairs} on its own line, so the block carries no trailing
    # newline and the generated <contact> keeps exactly one line per pair
    pairs = "\n".join(_box_pair(g1, g2)
                      for g1, g2 in box_pairs(lip_overhang_m=lip_overhang_m,
                                              lip_h_m=lip_h_m))
    return TEMPLATE_HEAD.format(
        repo=UPSTREAM_REPO,
        commit=UPSTREAM_COMMIT,
        meshdir=MESH_DIR,
        defaults=defaults,
        assets=assets,
        slab_z=f"{TABLE_TOP_Z_M - 0.02:.4f}",
        ped_half=f"{PEDESTAL_H_M / 2:.4f}",
        ped_z=f"{TABLE_TOP_Z_M + PEDESTAL_H_M / 2:.4f}",
        box_z=f"{TABLE_TOP_Z_M + PEDESTAL_H_M + BOX_HALF_M[2]:.4f}",
        bx=BOX_HALF_M[0], by=BOX_HALF_M[1], bz=BOX_HALF_M[2],
        mass=f"{box_geom_mass(lip_overhang_m, lip_h_m, chamfer_m):g}",
        lip=lip_geom_lines(lip_overhang_m, lip_h_m),
        chamfer=chamfer_geom_lines(chamfer_m),
        hand_home=HAND_HOME,
        solref=_fmt_pair(CONTACT_SOLREF),
        solimp=_fmt_pair(CONTACT_SOLIMP),
        wrist_joints="\n".join(_wrist_joint_lines()),
        palm=palm,
        excludes=excludes,
        pairs=pairs,
        wrist_actuators="\n".join(_wrist_actuator_lines()),
        hand_actuators="\n".join(hand_actuator_lines),
    )


if __name__ == "__main__":
    main()
