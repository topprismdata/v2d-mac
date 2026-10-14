"""Minimal chumpy-free MANO v1.2 loader + forward kinematics (numpy only).

The official MANO ``MANO_*.pkl`` files were pickled with ``chumpy`` objects
inside, and ``chumpy`` does not import under numpy>=1.24 (it touches removed
numpy aliases).  Rather than fight that dependency we install a permissive
pickle stub for the handful of ``chumpy`` classes that appear in the file and
read the raw arrays straight out.  Everything after unpickling is plain numpy,
so the smoke has no chumpy/manopth runtime dependency.

Only forward kinematics on the joint regressor is needed here (retargeting
consumes 21 joint positions, not a mesh), so LBS weights/faces are loaded for
provenance bookkeeping but not used for vertex skinning.
"""

from __future__ import annotations

import hashlib
import importlib.abc
import importlib.machinery
import pickle
import sys
import types
from dataclasses import dataclass
from pathlib import Path

import numpy as np


class _ChumpyStub(np.ndarray):
    """Stand-in for ``chumpy.ch.Ch``: carries a state dict of raw arrays."""

    def __new__(cls, *args, **kwargs):
        return np.zeros(1, dtype=np.float64).view(cls)

    def __setstate__(self, state):
        if isinstance(state, tuple) and state:
            state = state[-1]
        if isinstance(state, dict):
            self.__dict__.update(state)

    def __reduce__(self):
        return (_ChumpyStub, ())

    def __getattr__(self, name):
        if name.startswith("__"):
            raise AttributeError(name)
        try:
            return self.__dict__[name]
        except KeyError:
            raise AttributeError(name) from None


class _ChumpyModuleStub(types.ModuleType):
    def __getattr__(self, name):
        if name.startswith("__"):
            raise AttributeError(name)
        return _ChumpyStub


class _ChumpyFinder(importlib.abc.MetaPathFinder, importlib.abc.Loader):
    """Resolve any ``chumpy[.*]`` module to a permissive stub, once."""

    def find_spec(self, fullname, path=None, target=None):
        if fullname == "chumpy" or fullname.startswith("chumpy."):
            return importlib.machinery.ModuleSpec(fullname, self)
        return None

    def create_module(self, spec):
        mod = _ChumpyModuleStub(spec.name)
        mod.__path__ = []
        mod.__all__ = ["Ch"]
        mod.Ch = _ChumpyStub
        return mod

    def exec_module(self, module):
        return None


def _ensure_chumpy_stub() -> None:
    if any(isinstance(f, _ChumpyFinder) for f in sys.meta_path):
        return
    sys.meta_path.insert(0, _ChumpyFinder())
    if "chumpy" not in sys.modules:
        sys.modules["chumpy"] = _ChumpyFinder().create_module(
            importlib.machinery.ModuleSpec("chumpy", None)
        )


def sha256_file(path: str | Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def _rodrigues(axis_angle: np.ndarray) -> np.ndarray:
    """(N, 3) axis-angle -> (N, 3, 3) rotation matrices (batched, no scipy)."""
    aa = np.asarray(axis_angle, dtype=np.float64).reshape(-1, 3)
    theta = np.linalg.norm(aa, axis=1)
    safe = np.where(theta < 1e-12, 1.0, theta)
    k = aa / safe[:, None]
    kx, ky, kz = k[:, 0], k[:, 1], k[:, 2]
    zeros = np.zeros_like(theta)
    k_skew = np.stack(
        [
            np.stack([zeros, -kz, ky], axis=-1),
            np.stack([kz, zeros, -kx], axis=-1),
            np.stack([-ky, kx, zeros], axis=-1),
        ],
        axis=-2,
    )
    eye = np.broadcast_to(np.eye(3), (aa.shape[0], 3, 3))
    s = np.sin(theta)[:, None, None]
    c = np.cos(theta)[:, None, None]
    small = theta < 1e-12
    rot = eye + s * k_skew + (1.0 - c) * (k_skew @ k_skew)
    if np.any(small):  # second-order fallback keeps the Taylor branch continuous
        rot[small] = eye[small] + k_skew[small]
    return rot


@dataclass(frozen=True)
class ManoModel:
    """Raw MANO arrays plus the rest-pose joint locations and the joint tree."""

    v_template: np.ndarray
    shapedirs: np.ndarray
    posedirs: np.ndarray
    j_regressor: np.ndarray
    weights: np.ndarray
    faces: np.ndarray
    parents: np.ndarray  # (n_joints,) parent index, -1 for the root
    n_joints: int

    @property
    def rest_joints(self) -> np.ndarray:
        """(n_joints, 3) joint locations of the zero-pose mean shape."""
        return self.j_regressor @ self.v_template


def _chumpy_dense(value, depth: int = 0) -> np.ndarray:
    """Dig the dense payload out of a chumpy object (possibly nested).

    ``_ChumpyStub`` subclasses ndarray but carries its real payload in
    ``__dict__``, so an ndarray check alone is not enough -- only a *plain*
    ndarray (or an array larger than the 1-element stub shell) terminates.
    """
    if depth > 6:
        raise ValueError("chumpy object nests too deeply")
    if type(value) is np.ndarray:
        return np.asarray(value, dtype=np.float64)
    payload = getattr(value, "__dict__", {})
    # Prefer the canonical dense leaf ('x' inside `a`, `a` on the sparse wrapper).
    for key in ("x", "a"):
        if key in payload:
            try:
                return _chumpy_dense(payload[key], depth + 1)
            except ValueError:
                continue
    candidates = [v for v in payload.values() if hasattr(v, "shape")]
    for cand in candidates:
        shape = getattr(cand, "shape", ())
        if len(shape) >= 2 and int(np.prod(shape)) > 8:
            return _chumpy_dense(cand, depth + 1)
    raise ValueError("chumpy object carries no dense array")


def _extract_shapedirs(raw) -> np.ndarray:
    """MANO ships ``shapedirs`` as a chumpy sparse object whose logical shape is
    (778, 3, 10); the payload array carries 20 channels of which only the first
    10 are the shape space.  Take the first 10.
    """
    dense = _chumpy_dense(raw["shapedirs"])
    if dense.ndim != 3 or dense.shape[-1] < 10:
        raise ValueError(f"unexpected shapedirs layout {dense.shape}")
    return np.ascontiguousarray(dense[..., :10])


def _extract_parents(raw, n_joints: int) -> np.ndarray:
    """Parent index per joint from MANO's kintree_table (row 0 == parents)."""
    if "kintree_table" not in raw:
        return np.array([-1] + list(range(n_joints - 1)), dtype=np.int64)
    parents = np.asarray(raw["kintree_table"])[0].astype(np.int64).copy()
    parents[parents == 0xFFFFFFFF] = -1
    if parents.shape[0] != n_joints:
        raise ValueError(f"kintree has {parents.shape[0]} joints, regressor {n_joints}")
    return parents


def load_mano(pkl_path: str | Path) -> tuple[ManoModel, dict]:
    """Load a MANO v1.2 pkl.  Returns the model plus a provenance record."""
    pkl_path = Path(pkl_path)
    if not pkl_path.is_file():
        raise FileNotFoundError(f"MANO pkl missing: {pkl_path}")
    _ensure_chumpy_stub()
    with open(pkl_path, "rb") as fh:
        raw = pickle.load(fh, encoding="latin1")

    v_template = np.asarray(raw["v_template"], dtype=np.float64)
    shapedirs = _extract_shapedirs(raw)
    posedirs = np.asarray(raw["posedirs"], dtype=np.float64)
    j_regressor_raw = raw["J_regressor"]  # often scipy sparse -- do not densify yet
    weights = np.asarray(raw["weights"], dtype=np.float64)

    # J_regressor ships either dense (16, 7783) or scipy-sparse.
    if hasattr(j_regressor_raw, "toarray"):
        j_regressor = np.asarray(j_regressor_raw.toarray(), dtype=np.float64)
    else:
        j_regressor = np.asarray(j_regressor_raw, dtype=np.float64)
    if j_regressor.shape[0] > j_regressor.shape[1]:
        j_regressor = j_regressor.T

    faces = np.asarray(raw["f"], dtype=np.int64) if "f" in raw else np.zeros((0, 3), np.int64)
    n_joints = int(j_regressor.shape[0])

    provenance = {
        "mano_pkl": str(pkl_path.resolve()),
        "mano_pkl_sha256": sha256_file(pkl_path),
        "mano_pkl_bytes": pkl_path.stat().st_size,
        "n_vertices": int(v_template.shape[0]),
        "n_joints_regressor": n_joints,
        "n_shapedirs": int(shapedirs.shape[-1]),
        "n_posedirs": int(posedirs.shape[-1]),
        "chumpy_stubbed": True,
    }
    model = ManoModel(
        v_template=v_template,
        shapedirs=shapedirs,
        posedirs=posedirs,
        j_regressor=j_regressor,
        weights=weights,
        faces=faces,
        parents=_extract_parents(raw, n_joints),
        n_joints=n_joints,
    )
    return model, provenance


# MANO's 16-joint regressor -> the 21-joint "openpose" hand skeleton used by
# dex-retargeting / MediaPipe.  Identical to HaMeR's MANO wrapper `joint_map`:
# the five *_TIP slots are not regressor joints at all, they are the standard
# MANO tip *vertices* (smplx.vertex_ids['mano']) and must be read off the skinned
# mesh, which is why this module runs the full LBS rather than joints-only.
MANO16_TO_21 = np.array(
    [0, 13, 14, 15, -1, 1, 2, 3, -2, 4, 5, 6, -3, 10, 11, 12, -4, 7, 8, 9, -5],
    dtype=np.int64,
)

# smplx.vertex_ids['mano'] -- thumb / index / middle / ring / pinky tip vertices.
MANO_TIP_VERTEX_IDS = (744, 320, 443, 554, 671)


def _rigid_transform(
    rot: np.ndarray, trans: np.ndarray, parents: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    """Chain per-joint transforms into global transforms.

    ``rot``/``trans`` are (n_joints, 3, 3) / (n_joints, 3) in parent-relative
    form; the returned pair places every joint in the model root frame.
    """
    n = rot.shape[0]
    glob_r = np.empty_like(rot)
    glob_t = np.empty_like(trans)
    for j in range(n):
        parent = int(parents[j])
        if parent < 0:
            glob_r[j] = rot[j]
            glob_t[j] = trans[j]
        else:
            glob_r[j] = glob_r[parent] @ rot[j]
            glob_t[j] = glob_r[parent] @ trans[j] + glob_t[parent]
    return glob_r, glob_t


def mano_forward(
    model: ManoModel,
    betas: np.ndarray,
    global_orient: np.ndarray,
    hand_pose: np.ndarray,
    transl: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Full MANO forward pass -> (joints21, joints16, vertices).

    Mirrors the reference SMPL/MANO implementation (smplx ``lbs``): shape blend,
    pose blend, then linear blend skinning along the kintree.  ``joints21`` is
    the 21-joint hand skeleton that dex-retargeting consumes, whose five tip
    entries are the skinned tip vertices.
    """
    betas = np.asarray(betas, dtype=np.float64).reshape(-1)
    n_used = min(betas.size, model.shapedirs.shape[-1])
    v_shaped = model.v_template + model.shapedirs[..., :n_used] @ betas[:n_used]

    j16_rest = (model.j_regressor @ v_shaped).reshape(model.n_joints, 3)

    full_pose = np.concatenate(
        [np.asarray(global_orient, dtype=np.float64).reshape(3),
         np.asarray(hand_pose, dtype=np.float64).reshape(-1)]
    )
    rots = _rodrigues(full_pose)

    # pose blend: (R_i - I) flattened over the 15 articulating joints
    pose_feature = (rots[1:] - np.eye(3)).reshape(-1)
    posedirs = model.posedirs.reshape(-1, model.posedirs.shape[-1])
    v_posed = v_shaped.reshape(-1, 3) + (posedirs @ pose_feature).reshape(-1, 3)

    # Relative transforms in the standard SMPL/MANO parameterisation: the root
    # sits at its regressed rest location (global orient then spins the whole
    # hand about the model origin), every other joint translates to its
    # rest-pose offset from its parent and rotates by its articulation.
    rel_r = np.tile(np.eye(3), (model.n_joints, 1, 1))
    rel_r[1:] = rots[1:]
    rel_t = np.zeros((model.n_joints, 3))
    rel_t[0] = j16_rest[0]
    for j in range(1, model.n_joints):
        parent = int(model.parents[j])
        rel_t[j] = j16_rest[j] - j16_rest[parent]
    glob_r, glob_t = _rigid_transform(rel_r, rel_t, model.parents)

    joints16 = glob_t.copy() + np.asarray(transl, dtype=np.float64).reshape(1, 3)

    # LBS: blend the per-joint global transforms with the skinning weights.
    # Each transform is expressed *relative to its rest joint*: the translation
    # column is ``T_j - R_j @ J_rest_j`` (smplx `rel_transforms`).  At rest pose
    # both terms coincide and the mesh collapses back onto ``v_posed``; keeping
    # only ``-R_j @ J_rest_j`` would offset the whole mesh by W @ J.
    weights = model.weights
    rel_t = glob_t - np.einsum("jab,jb->ja", glob_r, j16_rest)
    T = np.einsum("vj,jab->vab", weights, glob_r)
    verts = np.einsum("vab,vb->va", T, v_posed) + np.einsum(
        "vj,ja->va", weights, rel_t
    ) + np.asarray(transl, dtype=np.float64).reshape(1, 3)

    joints21 = np.empty((21, 3), dtype=np.float64)
    for k, src in enumerate(MANO16_TO_21):
        if src >= 0:
            joints21[k] = joints16[src]
        else:
            joints21[k] = verts[MANO_TIP_VERTEX_IDS[-(src + 1)]]
    return joints21, joints16, verts

