"""Generate grounding/assets/allegro_floating_ur5e.xml (r12).

Composes the W6b scene with two REAL UR5e arms (mujoco-menagerie,
universal_robots_ur5e -- kinematics/inertia/limits/actuator parameters
zero-modified, red line) replacing the retired box arms AND the floating
wrist:

  * the box-arm trees (arm_yaw_*/upper_*/elbow_*) and the 6 arm actuators
    are removed;
  * the floating-wrist joints (wrist_x/y/z/rx/ry/rz +- _left, in the
    hand_root bodies) and their 12 actuators are removed -- under a real
    6R arm the wrist IS the arm;
  * ONE merged UR5e default class ("ur5e") + one shared asset set
    (meshes/materials prefixed ur5e_) serve both arms;
  * two renamed UR5e trees are inserted, ceiling-mounted at the existing
    shoulder flanges (+-0.235, 0, 1.000), 180 deg about x (arm hangs
    down), joints/actuators suffixed _r/_l;
  * each Allegro hand subtree (hand_root +- _left, minus its wrist
    joints) is re-parented under its UR5e wrist_3_link at the menagerie
    attachment_site transform -- the wmount stack reads as the UR adapter
    plate (real Wonik-UR adapter semantics).

The menagerie assets are CC-BY-SA 4.0 (ROS Industrial Consortium);
attribution shipped in grounding/assets/ur5e_assets/LICENSE.menagerie.
Run:  python grounding/assets/build_allegro_floating_ur5e.py
"""
from __future__ import annotations

import copy
import pathlib
import shutil
import xml.etree.ElementTree as ET

HERE = pathlib.Path(__file__).resolve().parent
SRC = HERE / "allegro_floating.xml"
MENAGERIE = (HERE.parents[1] / "runs/upstream/mujoco_menagerie/"
             "universal_robots_ur5e/ur5e.xml")
MESH_SRC = HERE / "ur5e_assets"
OUT = HERE / "allegro_floating_ur5e.xml"

CEILING_MOUNT_POS = {"r": "0.235 0 1.000", "l": "-0.235 0 1.000"}
CEILING_MOUNT_QUAT = "1 0 0 0"  # 180 deg about x: the arm hangs down
ATTACH_POS = "0 0.1 0"
ATTACH_QUAT = "-1 1 0 0"  # the menagerie attachment_site's own transform
WRIST_JOINTS = ("wrist_x", "wrist_y", "wrist_z",
                "wrist_rx", "wrist_ry", "wrist_rz")
WRIST_ACTUATORS = ("wrist_ax", "wrist_ay", "wrist_az",
                   "wrist_rx", "wrist_ry", "wrist_rz")
UR5E_ACT_NAMES = ("shoulder_pan", "shoulder_lift", "elbow",
                  "wrist_1", "wrist_2", "wrist_3")
UR5E_CLASSES = ("size3", "size3_limited", "size1",
                "visual", "collision", "eef_collision")


def tag(el: ET.Element) -> str:
    return el.tag.split("}")[-1]


def named(root: ET.Element, kind: str, name: str) -> ET.Element | None:
    for el in root.iter(kind):
        if el.get("name") == name:
            return el
    return None


def stage_meshes(meshdir: pathlib.Path) -> None:
    meshdir.mkdir(parents=True, exist_ok=True)
    for f in MESH_SRC.glob("*.obj"):
        shutil.copy2(f, meshdir / f.name)


def strip_wrist_joints(hand_root: ET.Element) -> None:
    for j in list(hand_root):
        if tag(j) == "joint" and (j.get("name") or "").replace(
                "_left", "") in WRIST_JOINTS:
            hand_root.remove(j)


def compose() -> None:
    scene = ET.parse(SRC).getroot()
    ur5e = ET.parse(MENAGERIE).getroot()
    world = scene.find("worldbody")

    # ---- 0. runtime mesh staging -----------------------------------------
    wonik_assets = (HERE.parents[1] /
                    "runs/upstream/menagerie/wonik_allegro/assets")
    stage_meshes(wonik_assets)

    # ---- 1. merge the UR5e default class + shared assets ------------------
    defaults = scene.find("default")
    root_path = defaults.get("class") or ""
    root_prefix = (root_path + "/") if root_path else ""
    ur5e_defaults = None
    for el in ur5e.iter("default"):
        if el.get("class") == "ur5e":
            ur5e_defaults = copy.deepcopy(el)
            break
    assert ur5e_defaults is not None, "ur5e default class not found"
    # unique child-class names (the scene default may already carry
    # visual/collision children) + rewrite every class reference to the
    # full path
    child_renames = {"size3": "ur5e_size3",
                     "size3_limited": "ur5e_size3_limited",
                     "size1": "ur5e_size1",
                     "visual": "ur5e_visual",
                     "collision": "ur5e_collision",
                     "eef_collision": "ur5e_eef_collision"}
    for node in ur5e_defaults.iter():
        cn = node.get("class")
        if cn in child_renames:
            node.set("class", child_renames[cn])

    defaults.append(ur5e_defaults)  # class "ur5e", uniquely-named children
    assets = scene.find("asset")
    for el in ur5e.find("asset"):
        new = copy.deepcopy(el)
        if tag(new) == "mesh":
            # nameless mesh assets are referenced by their file stem --
            # name them ur5e_<stem> to match the tree's prefixed refs
            stem = pathlib.Path(new.get("file")).stem
            new.set("name", "ur5e_" + (new.get("name") or stem))
        else:
            if new.get("name") is None:
                continue
            new.set("name", "ur5e_" + new.get("name"))
        assets.append(new)

    def prefix_attrs(node: ET.Element) -> None:
        for attr in ("mesh", "material"):
            v = node.get(attr)
            if v:
                node.set(attr, "ur5e_" + v)
        # the class= attribute is mapped by the tree block's flat map

    # ---- 2. strip the retired box arms + their actuators ------------------
    act_sec = scene.find("actuator")

    def remove_named(kind: str, name: str) -> None:
        for parent in scene.iter():
            for child in list(parent):
                if tag(child) == kind and child.get("name") == name:
                    parent.remove(child)
                    return

    def remove_actuator(name: str) -> None:
        for child in list(act_sec):
            if child.get("name") == name:
                act_sec.remove(child)
                return

    for arm in ("arm_yaw_r", "arm_yaw_l"):
        remove_named("body", arm)
    for an in ("arm_yaw_r", "arm_pitch_r", "arm_elbow_r",
               "arm_yaw_l", "arm_pitch_l", "arm_elbow_l"):
        remove_actuator(an)


    # ---- 3. strip the floating wrist joints + actuators -------------------
    for side in ("", "_left"):
        hr = named(world, "body", "hand_root" + side)
        strip_wrist_joints(hr)
    for an in WRIST_ACTUATORS:
        for cand in (an, an + "_left"):
            remove_actuator(cand)

    # ---- 4. insert two renamed UR5e trees + actuators ---------------------
    for sfx in ("_r", "_l"):
        base = copy.deepcopy(named(ur5e, "body", "base"))
        for node in base.iter():
            n = node.get("name")
            if n:
                node.set("name", "ur5e" + sfx + "_" + n)
            prefix_attrs(node)
            cls = node.get("class")
            if cls in UR5E_CLASSES:
                node.set("class", {"size3": "ur5e_size3",
                                   "size3_limited": "ur5e_size3_limited",
                                   "size1": "ur5e_size1",
                                   "visual": "ur5e_visual",
                                   "collision": "ur5e_collision",
                                   "eef_collision": "ur5e_eef_collision",
                                   }[cls])
        base.set("childclass", "ur5e")
        base.set("pos", CEILING_MOUNT_POS["r" if sfx == "_r" else "l"])
        base.set("quat", CEILING_MOUNT_QUAT)
        world.append(base)
        ur5e_act_sec = ur5e.find("actuator")
        for an in UR5E_ACT_NAMES:
            src = None
            for child in ur5e_act_sec:
                if child.get("name") == an:
                    src = child
                    break
            a = copy.deepcopy(src)
            a.set("name", "ur5e" + sfx + "_" + an)
            a.set("joint", "ur5e" + sfx + "_" + an + "_joint")
            if "class" in a.attrib:
                cls = a.get("class")
                a.set("class", {"size3": "ur5e_size3",
                                "size3_limited": "ur5e_size3_limited",
                                "size1": "ur5e_size1",
                                "visual": "ur5e_visual",
                                "collision": "ur5e_collision",
                                "eef_collision": "ur5e_eef_collision",
                                }.get(cls, cls))
            act_sec.append(a)

    # ---- 5. re-parent the hands under the UR5e flanges --------------------
    for sfx, hand_name in (("_r", "hand_root"), ("_l", "hand_root_left")):
        hand = named(world, "body", hand_name)
        world.remove(hand)
        hand.set("pos", ATTACH_POS)
        hand.set("quat", ATTACH_QUAT)
        if "euler" in hand.attrib:
            del hand.attrib["euler"]
        w3 = named(world, "body", "ur5e" + sfx + "_wrist_3_link")
        w3.append(hand)

    ET.indent(scene)
    ET.ElementTree(scene).write(OUT, encoding="unicode")
    print(f"wrote {OUT}")


if __name__ == "__main__":
    compose()
