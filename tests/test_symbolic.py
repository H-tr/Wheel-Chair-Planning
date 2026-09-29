"""Tests for :class:`SymbolicContext`'s pure-CasADi forward kinematics.

The symbolic FK is built directly from the URDF, so it is cross-checked
against pinocchio's numeric FK (with the same planar root joint) when
pinocchio is installed.
"""

from __future__ import annotations

import casadi as ca
import numpy as np
import pytest

from wheelchair_planning.planning import SymbolicContext
from wheelchair_planning.wheelchair import HOME_JOINTS, wheelchair_robot_config

LINKS = ["base_link", "link_base", "link3", "link7", "link_tcp"]


def _random_active(ctx: SymbolicContext, rng: np.random.Generator) -> np.ndarray:
    q = ctx.base_config[ctx.active_indices].copy()
    return q + rng.uniform(-0.8, 0.8, q.shape)


@pytest.mark.parametrize(
    "subgroup", ["wheelchair_arm", "wheelchair_base", "wheelchair_whole_body"]
)
def test_matches_pinocchio_fk(subgroup):
    pin = pytest.importorskip("pinocchio")

    model = pin.buildModelFromUrdf(
        wheelchair_robot_config.urdf_path, pin.JointModelPlanar()
    )
    data = model.createData()
    base_config = HOME_JOINTS.copy()
    base_config[:3] = [0.4, -0.2, 0.3]  # non-trivial parked base
    ctx = SymbolicContext(subgroup, base_config=base_config)
    rng = np.random.default_rng(0)

    for _ in range(5):
        q_active = _random_active(ctx, rng)
        full = base_config.copy()
        full[ctx.active_indices] = q_active
        q_pin = np.concatenate([full[:2], [np.cos(full[2]), np.sin(full[2])], full[3:]])
        pin.framesForwardKinematics(model, data, q_pin)
        for link in LINKS:
            expected = data.oMf[model.getFrameId(link)].homogeneous
            actual = ctx.evaluate_link_pose(link, q_active)
            np.testing.assert_allclose(actual, expected, atol=1e-9, err_msg=link)


def test_symbolic_expressions_match_numeric_pose():
    ctx = SymbolicContext("wheelchair_whole_body")
    fn = ca.Function(
        "fk",
        [ctx.q],
        [ctx.link_translation("link_tcp"), ctx.link_rotation("link_tcp")],
    )
    q = _random_active(ctx, np.random.default_rng(1))
    pos, rot = (np.asarray(v) for v in fn(q))
    pose = ctx.evaluate_link_pose("link_tcp", q)
    np.testing.assert_allclose(pos.reshape(-1), pose[:3, 3], atol=1e-12)
    np.testing.assert_allclose(rot, pose[:3, :3], atol=1e-12)
    np.testing.assert_allclose(rot @ rot.T, np.eye(3), atol=1e-12)


def test_link_pose_with_other_symbol():
    ctx = SymbolicContext("wheelchair_arm")
    other = ca.SX.sym("other", len(ctx.active_indices))
    fn = ca.Function("fk", [other], [ctx.link_translation("link_tcp", other)])
    q = _random_active(ctx, np.random.default_rng(2))
    np.testing.assert_allclose(
        np.asarray(fn(q)).reshape(-1),
        ctx.evaluate_link_pose("link_tcp", q)[:3, 3],
        atol=1e-12,
    )


def test_inactive_joints_are_constants():
    ctx = SymbolicContext("wheelchair_base")
    # Arm joints are frozen at base_config, so the tool position depends
    # only on the three base symbols.
    expr = ctx.link_translation("link_tcp")
    assert ca.depends_on(expr, ctx.q)
    assert ctx.q.shape == (3, 1)


def test_unknown_link_raises():
    ctx = SymbolicContext("wheelchair_arm")
    with pytest.raises(ValueError, match="Unknown link"):
        ctx.link_translation("no_such_link")
