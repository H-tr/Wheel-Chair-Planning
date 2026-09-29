"""Symbolic forward-kinematics helper shared by :class:`Constraint` and
:class:`Cost`.

Both user-defined constraints and costs build their CasADi expressions
on top of robot FK.  This module provides :class:`SymbolicContext`,
which turns the subgroup's active joint symbol into symbolic link poses
using a small URDF kinematic-tree walker written directly in CasADi —
no pinocchio or urdf2casadi needed, so it works from a plain
``pip install``.

The expressions stay compact for code generation: runs of fixed joints
are pre-multiplied numerically, joint rotations about coordinate axes
only emit the non-trivial ``cos``/``sin`` terms, and link poses that
share a kinematic prefix share the same expression nodes.

The context also owns small filesystem helpers shared across the
compile pipeline:

- :func:`_jit_build_dir` — scratch dir for CasADi's temporary JIT C files.
- :func:`_cwd` — ``chdir`` context manager used by CasADi's ``generate``,
  which always emits files into the process's current working directory.
"""

from __future__ import annotations

import os
import xml.etree.ElementTree as ET
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path

import casadi as ca
import numpy as np

from wheelchair_planning.wheelchair import (
    HOME_JOINTS,
    PLANNING_SUBGROUPS,
    wheelchair_robot_config,
)

# The first three entries of the 10-DOF body vector are the planar base
# ``[x, y, theta]``, applied as a planar joint at the URDF root link.
_BASE_DIM = 3


def _jit_build_dir() -> Path:
    """Directory for CasADi temporary JIT artifacts."""
    return (Path(__file__).resolve().parents[2] / "build" / "casadi_jit").resolve()


@contextmanager
def _cwd(path: Path):
    """Temporarily chdir — CasADi's generate() always writes to cwd."""
    old = Path.cwd()
    os.chdir(path)
    try:
        yield
    finally:
        os.chdir(old)


# ── URDF kinematic tree ──────────────────────────────────────────────


@dataclass(frozen=True)
class _Joint:
    name: str
    type: str
    parent: str
    origin: np.ndarray  # (4, 4) parent-link -> joint frame
    axis: np.ndarray  # (3,) unit axis in the joint frame


def _rpy_to_matrix(roll: float, pitch: float, yaw: float) -> np.ndarray:
    """URDF fixed-axis RPY: ``R = Rz(yaw) @ Ry(pitch) @ Rx(roll)``."""
    cr, sr = np.cos(roll), np.sin(roll)
    cp, sp = np.cos(pitch), np.sin(pitch)
    cy, sy = np.cos(yaw), np.sin(yaw)
    return np.array(
        [
            [cy * cp, cy * sp * sr - sy * cr, cy * sp * cr + sy * sr],
            [sy * cp, sy * sp * sr + cy * cr, sy * sp * cr - cy * sr],
            [-sp, cp * sr, cp * cr],
        ]
    )


def _parse_vec(elem: ET.Element | None, attr: str, default: str) -> np.ndarray:
    text = default if elem is None else elem.get(attr, default)
    return np.array([float(v) for v in text.split()], dtype=np.float64)


def _parse_urdf(urdf_path: str) -> tuple[dict[str, _Joint], str]:
    """Return ``{child_link: joint}`` and the name of the root link."""
    robot = ET.parse(urdf_path).getroot()
    joints: dict[str, _Joint] = {}
    for j in robot.findall("joint"):
        jtype = j.get("type")
        if jtype not in ("fixed", "revolute", "continuous", "prismatic"):
            raise ValueError(
                f"SymbolicContext does not support URDF joint type {jtype!r} "
                f"(joint {j.get('name')!r})"
            )
        if j.find("mimic") is not None:
            raise ValueError(
                f"SymbolicContext does not support mimic joints ({j.get('name')!r})"
            )
        origin = j.find("origin")
        T = np.eye(4)
        T[:3, :3] = _rpy_to_matrix(*_parse_vec(origin, "rpy", "0 0 0"))
        T[:3, 3] = _parse_vec(origin, "xyz", "0 0 0")
        axis = _parse_vec(j.find("axis"), "xyz", "1 0 0")
        child = j.find("child").get("link")
        joints[child] = _Joint(
            name=j.get("name"),
            type=jtype,
            parent=j.find("parent").get("link"),
            origin=T,
            axis=axis / np.linalg.norm(axis),
        )

    links = [link.get("name") for link in robot.findall("link")]
    roots = [name for name in links if name not in joints]
    if len(roots) != 1:
        raise ValueError(f"URDF must have exactly one root link, found {roots}")
    return joints, roots[0]


def _lin(terms: list[tuple[float, object]]):
    """Sum ``coef * value`` terms, skipping zero and unit coefficients."""
    out = 0.0
    for coef, value in terms:
        if abs(coef) < 1e-15:
            continue
        term = value if coef == 1.0 else -value if coef == -1.0 else coef * value
        out = term if isinstance(out, float) and out == 0.0 else out + term
    return out


def _axis_rotation(axis: np.ndarray, c, s) -> ca.SX:
    """Rotation by angle ``(c, s) = (cos, sin)`` about a unit ``axis``.

    Written as ``a a^T + c (I - a a^T) + s [a]_x`` with numeric
    coefficients, so an axis-aligned joint yields exactly the classic
    sparse rotation (e.g. ``[[c, -s, 0], [s, c, 0], [0, 0, 1]]``).
    """
    aat = np.outer(axis, axis)
    skew = np.array(
        [[0.0, -axis[2], axis[1]], [axis[2], 0.0, -axis[0]], [-axis[1], axis[0], 0.0]]
    )
    R = ca.SX(3, 3)
    for i in range(3):
        for j in range(3):
            R[i, j] = _lin(
                [(aat[i, j], 1.0), (float(i == j) - aat[i, j], c), (skew[i, j], s)]
            )
    return R


def _compose(T_sym: ca.SX, T_const: np.ndarray) -> ca.SX:
    """``T_sym @ T_const`` for homogeneous transforms (constant on the right)."""
    out = ca.SX.eye(4)
    for i in range(3):
        for j in range(4):
            terms = [(T_const[k, j], T_sym[i, k]) for k in range(3)]
            if j == 3:
                terms.append((1.0, T_sym[i, 3]))
            out[i, j] = _lin(terms)
    return out


class _Pose:
    """Symbolic link pose with pinocchio-SE3-like ``translation``/``rotation``."""

    def __init__(self, T: ca.SX) -> None:
        self.homogeneous = T
        self.translation = T[:3, 3]
        self.rotation = T[:3, :3]


class _TreeFK:
    """Memoised symbolic FK over the URDF tree for one joint vector."""

    def __init__(
        self,
        joints: dict[str, _Joint],
        root: str,
        joint_values: dict[str, object],
        base: tuple[object, object, object],
    ) -> None:
        self._joints = joints
        self._values = joint_values
        x, y, theta = base
        c, s = ca.cos(theta), ca.sin(theta)
        T_root = ca.SX.eye(4)
        T_root[:3, :3] = _axis_rotation(np.array([0.0, 0.0, 1.0]), c, s)
        T_root[0, 3] = x
        T_root[1, 3] = y
        # Each entry is (symbolic prefix, pending constant suffix): chains
        # of fixed joints only grow the numeric suffix, so they cost no
        # symbolic operations until a moving joint (or a query) needs them.
        self._cache: dict[str, tuple[ca.SX, np.ndarray]] = {root: (T_root, np.eye(4))}

    def _factored(self, link: str) -> tuple[ca.SX, np.ndarray]:
        hit = self._cache.get(link)
        if hit is not None:
            return hit
        if link not in self._joints:
            raise ValueError(f"Unknown link: {link!r}")
        joint = self._joints[link]
        T_sym, T_const = self._factored(joint.parent)
        T_const = T_const @ joint.origin
        if joint.type != "fixed":
            if joint.name not in self._values:
                raise ValueError(
                    f"Joint {joint.name!r} on the chain to {link!r} is not in "
                    "the robot's planning joint list."
                )
            q = self._values[joint.name]
            T_sym = _compose(T_sym, T_const)
            T_joint = ca.SX.eye(4)
            if joint.type == "prismatic":
                for i in range(3):
                    T_joint[i, 3] = _lin([(joint.axis[i], q)])
            else:
                T_joint[:3, :3] = _axis_rotation(joint.axis, ca.cos(q), ca.sin(q))
            T_sym = ca.mtimes(T_sym, T_joint)
            T_const = np.eye(4)
        self._cache[link] = (T_sym, T_const)
        return T_sym, T_const

    def pose(self, link: str) -> _Pose:
        T_sym, T_const = self._factored(link)
        return _Pose(_compose(T_sym, T_const))


# ── Public context ───────────────────────────────────────────────────


class SymbolicContext:
    """CasADi-friendly view of the planner's active subgroup.

    Holds the CasADi symbolic joint vector ``q`` matching the subgroup's
    active dimension and builds symbolic link poses from the robot URDF.

    The context hides the planar-root encoding (the 10-DOF body vector
    uses ``[x, y, theta, j0..j6]``, with the base applied as a planar
    joint at the URDF root link) and the mapping from the active
    subspace back to the full 10-DOF body via ``base_config``.
    """

    def __init__(
        self,
        subgroup: str,
        base_config: np.ndarray | None = None,
    ) -> None:
        if base_config is None:
            base_config = HOME_JOINTS
        self.base_config = np.asarray(base_config, dtype=np.float64).copy()
        if self.base_config.shape != HOME_JOINTS.shape:
            raise ValueError(
                f"base_config must have shape {HOME_JOINTS.shape}, "
                f"got {self.base_config.shape}"
            )

        self.subgroup_name = subgroup
        full_names = list(wheelchair_robot_config.joint_names)
        if subgroup == "wheelchair":
            self.active_indices = list(range(len(full_names)))
            self.active_names = full_names
        else:
            sg = PLANNING_SUBGROUPS.get(subgroup)
            if sg is None:
                raise ValueError(f"Unknown subgroup: {subgroup!r}")
            self.active_names = list(sg["joints"])
            self.active_indices = [full_names.index(j) for j in self.active_names]

        self.q = ca.SX.sym("q", len(self.active_indices))
        self._full_names = full_names
        self._joints, self._root_link = _parse_urdf(wheelchair_robot_config.urdf_path)
        self._fk = self._build_fk(self.q)
        self._numeric_fk: dict[str, ca.Function] = {}

    def _build_full_q(self, q_active: ca.SX) -> list[ca.SX | float]:
        """Map active subgroup symbols onto the full 10-DOF joint vector."""
        full: list[ca.SX | float] = [float(v) for v in self.base_config]
        for i, idx in enumerate(self.active_indices):
            full[idx] = q_active[i]
        return full

    def _build_fk(self, q_active: ca.SX) -> _TreeFK:
        full = self._build_full_q(q_active)
        values = dict(zip(self._full_names[_BASE_DIM:], full[_BASE_DIM:]))
        return _TreeFK(self._joints, self._root_link, values, tuple(full[:_BASE_DIM]))

    def link_pose(self, link_name: str, q_active: ca.SX | None = None) -> _Pose:
        """Return the symbolic world pose of a URDF link.

        The result exposes ``translation`` (3x1) and ``rotation`` (3x3)
        CasADi expressions.  Pass ``q_active=self.q`` (or omit) to get
        an expression that depends on the active joints symbolically.
        """
        if q_active is None or q_active is self.q:
            return self._fk.pose(link_name)
        return self._build_fk(q_active).pose(link_name)

    def link_translation(self, link_name: str, q_active: ca.SX | None = None) -> ca.SX:
        """Symbolic 3-vector: link position in world frame."""
        return self.link_pose(link_name, q_active).translation

    def link_rotation(self, link_name: str, q_active: ca.SX | None = None) -> ca.SX:
        """Symbolic 3x3 rotation matrix of the link."""
        return self.link_pose(link_name, q_active).rotation

    def evaluate_link_pose(
        self, link_name: str, q_active_numeric: np.ndarray
    ) -> np.ndarray:
        """Compute a NUMERIC 4x4 link pose (handy for building targets)."""
        fn = self._numeric_fk.get(link_name)
        if fn is None:
            fn = ca.Function(
                f"fk_{link_name}", [self.q], [self.link_pose(link_name).homogeneous]
            )
            self._numeric_fk[link_name] = fn
        q = np.asarray(q_active_numeric, dtype=np.float64).reshape(-1)
        return np.asarray(fn(q), dtype=np.float64)

    def project(
        self,
        q_init: np.ndarray,
        residual: ca.SX,
        tol: float = 1e-8,
        max_iters: int = 100,
    ) -> np.ndarray:
        """Project a joint configuration onto the manifold ``residual(q) = 0``.

        Runs damped Gauss-Newton on the CasADi Jacobian — the same
        iteration OMPL's ``ProjectedStateSpace`` runs internally — so
        the returned config will pass the planner's tolerance check
        and can be used directly as a start or goal state.
        """
        res_fn = ca.Function("proj_res", [self.q], [ca.reshape(residual, -1, 1)])
        jac_fn = ca.Function("proj_jac", [self.q], [ca.jacobian(residual, self.q)])
        q = np.asarray(q_init, dtype=np.float64).copy()
        for _ in range(max_iters):
            r = np.asarray(res_fn(q)).flatten()
            if np.linalg.norm(r) < tol:
                return q
            J = np.asarray(jac_fn(q))
            JJt = J @ J.T + 1e-10 * np.eye(J.shape[0])
            q -= J.T @ np.linalg.solve(JJt, r)
        raise RuntimeError(
            f"SymbolicContext.project failed to converge: "
            f"|residual|={np.linalg.norm(r):.2e} after {max_iters} iters"
        )


__all__ = ["SymbolicContext"]
