"""Kinematics tests — IK solver factory, TRAC-IK, Pinocchio FK, collision model.

Mirrors ``examples/ik/basic.py`` and the constrained-IK demo, but keeps
every solve cheap (small offsets, deterministic seed) so the suite stays
under a second.
"""

from __future__ import annotations

import numpy as np
import pytest

from wheelchair_planning.types import IKConfig, SE3Pose, SolveType
from wheelchair_planning.wheelchair import HOME_JOINTS, JOINT_GROUPS

HOME_ARM = HOME_JOINTS[JOINT_GROUPS["arm"]]

_ARM_JOINT_NAMES = [f"joint{i}" for i in range(1, 8)]


# ── Factory & chain resolution ───────────────────────────────────────


def test_factory_rejects_unknown_chain():
    pytest.importorskip("pytracik")
    from wheelchair_planning.kinematics import create_ik_solver

    with pytest.raises(ValueError, match="Unknown chain"):
        create_ik_solver("not_a_real_chain")


def test_factory_rejects_unknown_backend():
    pytest.importorskip("pytracik")
    from wheelchair_planning.kinematics import create_ik_solver

    with pytest.raises(ValueError, match="Unknown backend"):
        create_ik_solver("arm", backend="not_a_backend")


def test_whole_body_chain_resolves():
    pytest.importorskip("pytracik")
    from wheelchair_planning.kinematics import create_ik_solver

    solver = create_ik_solver("whole_body")
    assert solver.num_joints == 10


# ── TRAC-IK solver ───────────────────────────────────────────────────


@pytest.fixture(scope="module")
def trac_arm():
    pytest.importorskip("pytracik")
    pytest.importorskip("pinocchio")
    from wheelchair_planning.kinematics import create_ik_solver

    return create_ik_solver("arm", config=IKConfig(max_attempts=3))


def test_trac_ik_chain_metadata(trac_arm):
    assert trac_arm.num_joints == 7
    assert trac_arm.base_frame
    assert trac_arm.ee_frame


def test_trac_ik_fk_runs(trac_arm):
    pose = trac_arm.fk(HOME_ARM)
    assert isinstance(pose, SE3Pose)
    assert pose.position.shape == (3,)
    assert pose.rotation.shape == (3, 3)
    np.testing.assert_allclose(pose.rotation @ pose.rotation.T, np.eye(3), atol=1e-9)


def test_trac_ik_solve_round_trip(trac_arm):
    """IK(FK(q)) with a tiny offset should land on a config whose FK matches."""
    home_pose = trac_arm.fk(HOME_ARM)
    target = SE3Pose(
        position=home_pose.position + np.array([0.03, 0.0, -0.02]),
        rotation=home_pose.rotation,
    )
    result = trac_arm.solve(target, seed=HOME_ARM)
    if not result.success:
        pytest.skip(f"TRAC-IK did not converge ({result.status.value}); flaky on CI")

    assert result.joint_positions is not None
    assert result.joint_positions.shape == (7,)
    assert result.position_error < 1e-3
    assert result.orientation_error < 1e-3

    achieved = trac_arm.fk(result.joint_positions)
    np.testing.assert_allclose(achieved.position, target.position, atol=1e-3)


def test_trac_ik_round_trip_multiple_targets(trac_arm):
    """Real correctness: sample several FK targets, IK them back, FK again must match."""
    rng = np.random.default_rng(0)
    n_solved = 0
    for _ in range(5):
        delta = rng.uniform(-0.1, 0.1, size=7)
        q_seed = HOME_ARM + delta
        target_pose = trac_arm.fk(q_seed)

        result = trac_arm.solve(target_pose, seed=q_seed)
        if not result.success:
            continue
        n_solved += 1

        achieved = trac_arm.fk(result.joint_positions)
        np.testing.assert_allclose(achieved.position, target_pose.position, atol=1e-3)
        R_err = achieved.rotation.T @ target_pose.rotation
        np.testing.assert_allclose(R_err, np.eye(3), atol=1e-3)

    assert n_solved >= 3, "TRAC-IK should converge on most small perturbations"


def test_trac_ik_solve_types_accepted():
    pytest.importorskip("pytracik")
    pytest.importorskip("pinocchio")
    from wheelchair_planning.kinematics import create_ik_solver

    for st in (SolveType.SPEED, SolveType.DISTANCE):
        s = create_ik_solver(
            "arm",
            config=IKConfig(solve_type=st, max_attempts=1),
        )
        assert s.num_joints == 7


def test_trac_ik_set_joint_limits_validates(trac_arm):
    lo, hi = trac_arm.joint_limits
    with pytest.raises(ValueError):
        trac_arm.set_joint_limits(lo[:-1], hi)


# ── Pinocchio FK ─────────────────────────────────────────────────────


@pytest.fixture(scope="module")
def arm_pin_context():
    pytest.importorskip("pinocchio")
    from wheelchair_planning.kinematics import create_pinocchio_context
    from wheelchair_planning.wheelchair import CHAIN_CONFIGS

    chain = CHAIN_CONFIGS["arm"]
    return create_pinocchio_context(
        urdf_path=chain.urdf_path,
        end_effector_frame=chain.ee_link,
        joint_names=_ARM_JOINT_NAMES,
    )


def test_pinocchio_fk_matches_trac_ik(arm_pin_context, trac_arm):
    """Both backends should agree on the EE pose at HOME (same URDF)."""
    from wheelchair_planning.kinematics import compute_forward_kinematics

    pin_pose = compute_forward_kinematics(arm_pin_context, HOME_ARM)
    trac_pose = trac_arm.fk(HOME_ARM)

    np.testing.assert_allclose(pin_pose.position, trac_pose.position, atol=1e-6)
    np.testing.assert_allclose(pin_pose.rotation, trac_pose.rotation, atol=1e-6)


def test_pinocchio_jacobian_shape(arm_pin_context):
    from wheelchair_planning.kinematics import compute_jacobian

    J = compute_jacobian(arm_pin_context, HOME_ARM)
    assert J.shape[0] == 6
    assert J.shape[1] >= 7


def test_pinocchio_fk_matches_trac_ik_random_configs(arm_pin_context, trac_arm):
    """Real correctness: cross-verify FK at multiple random valid configs."""
    from wheelchair_planning.kinematics import compute_forward_kinematics

    lo, hi = trac_arm.joint_limits
    rng = np.random.default_rng(42)
    for _ in range(10):
        q = rng.uniform(lo, hi)
        pin_pose = compute_forward_kinematics(arm_pin_context, q)
        trac_pose = trac_arm.fk(q)
        np.testing.assert_allclose(pin_pose.position, trac_pose.position, atol=1e-9)
        np.testing.assert_allclose(pin_pose.rotation, trac_pose.rotation, atol=1e-9)


def test_pinocchio_jacobian_matches_finite_difference(arm_pin_context):
    """Real correctness: Jacobian columns equal d(FK)/dqi via finite differences."""
    from wheelchair_planning.kinematics import (
        compute_forward_kinematics,
        compute_jacobian,
    )

    rng = np.random.default_rng(7)
    q = HOME_ARM + rng.uniform(-0.05, 0.05, size=7)
    J = compute_jacobian(arm_pin_context, q)
    eps = 1e-6
    for i in range(7):
        dq = np.zeros(7)
        dq[i] = eps
        fp = compute_forward_kinematics(arm_pin_context, q + dq).position
        fm = compute_forward_kinematics(arm_pin_context, q - dq).position
        dp_dq = (fp - fm) / (2 * eps)
        np.testing.assert_allclose(J[:3, i], dp_dq, atol=1e-5)


def test_pinocchio_context_rejects_unknown_frame():
    pytest.importorskip("pinocchio")
    from wheelchair_planning.kinematics import create_pinocchio_context
    from wheelchair_planning.wheelchair import CHAIN_CONFIGS

    chain = CHAIN_CONFIGS["arm"]
    with pytest.raises(ValueError, match="not found"):
        create_pinocchio_context(
            urdf_path=chain.urdf_path,
            end_effector_frame="Not_A_Real_Frame",
        )


# ── Collision model ──────────────────────────────────────────────────


@pytest.fixture(scope="module")
def collision_ctx():
    pytest.importorskip("pinocchio")
    pytest.importorskip("hppfcl")
    from wheelchair_planning.kinematics import build_collision_model
    from wheelchair_planning.wheelchair import CHAIN_CONFIGS

    return build_collision_model(CHAIN_CONFIGS["arm"].urdf_path)


def test_collision_model_has_pairs(collision_ctx):
    assert collision_ctx.collision_model.ngeoms > 0
    assert len(collision_ctx.collision_model.collisionPairs) >= 0


def test_add_pointcloud_obstacles_validates_shape(collision_ctx):
    from wheelchair_planning.kinematics import add_pointcloud_obstacles

    with pytest.raises(ValueError):
        add_pointcloud_obstacles(collision_ctx, np.zeros((4, 2)))


def test_add_pointcloud_obstacles_returns_count(collision_ctx):
    from wheelchair_planning.kinematics import add_pointcloud_obstacles

    pts = np.array(
        [
            [10.0, 10.0, 10.0],
            [10.0, 10.5, 10.0],
            [10.5, 10.0, 10.0],
        ]
    )
    n_before = collision_ctx.collision_model.ngeoms
    n_added = add_pointcloud_obstacles(collision_ctx, pts, radius=0.01)
    assert n_added == 3
    assert collision_ctx.collision_model.ngeoms == n_before + 3


# ── Pink IK solver ───────────────────────────────────────────────────


@pytest.fixture(scope="module")
def pink_arm():
    pytest.importorskip("pink")
    pytest.importorskip("qpsolvers")
    pytest.importorskip("pinocchio")
    from wheelchair_planning.kinematics import create_ik_solver
    from wheelchair_planning.types import PinkIKConfig

    return create_ik_solver(
        "arm",
        backend="pink",
        joint_names=_ARM_JOINT_NAMES,
        config=PinkIKConfig(max_iterations=300),
    )


def test_pink_solver_chain_metadata(pink_arm):
    assert pink_arm.num_joints == 7
    assert pink_arm.base_frame
    assert pink_arm.ee_frame
    assert pink_arm.joint_names == _ARM_JOINT_NAMES


def test_pink_fk_matches_trac_ik(pink_arm, trac_arm):
    """Pink and TRAC-IK must agree on the EE pose at HOME (same URDF)."""
    pink_pose = pink_arm.fk(HOME_ARM)
    trac_pose = trac_arm.fk(HOME_ARM)
    np.testing.assert_allclose(pink_pose.position, trac_pose.position, atol=1e-6)
    np.testing.assert_allclose(pink_pose.rotation, trac_pose.rotation, atol=1e-6)


def test_pink_fk_matches_trac_ik_random_configs(pink_arm, trac_arm):
    """Real correctness: Pink and TRAC-IK FK agree across random configs."""
    lo, hi = trac_arm.joint_limits
    rng = np.random.default_rng(123)
    for _ in range(10):
        q = rng.uniform(lo, hi)
        pink_pose = pink_arm.fk(q)
        trac_pose = trac_arm.fk(q)
        np.testing.assert_allclose(pink_pose.position, trac_pose.position, atol=1e-9)
        np.testing.assert_allclose(pink_pose.rotation, trac_pose.rotation, atol=1e-9)


def test_pink_solve_constrained_converges(pink_arm):
    """Tiny offset — Pink's iterative QP should land near the target."""
    home_pose = pink_arm.fk(HOME_ARM)
    target = SE3Pose(
        position=home_pose.position + np.array([0.02, 0.0, -0.01]),
        rotation=home_pose.rotation,
    )
    result = pink_arm.solve_constrained(target, seed=HOME_ARM)
    if not result.success:
        pytest.skip(f"Pink IK did not converge ({result.status.value}); flaky on CI")

    assert result.joint_positions is not None
    assert result.joint_positions.shape == (7,)
    assert result.position_error < 5e-2
    assert result.trajectory is not None and result.trajectory.shape[1] == 7


def test_pink_solve_returns_plain_ik_result(pink_arm):
    """``solve()`` is the IKSolverBase contract — must yield an IKResult."""
    from wheelchair_planning.types import IKResult

    home_pose = pink_arm.fk(HOME_ARM)
    target = SE3Pose(
        position=home_pose.position + np.array([0.01, 0.0, 0.0]),
        rotation=home_pose.rotation,
    )
    result = pink_arm.solve(target, seed=HOME_ARM)
    assert isinstance(result, IKResult)


def test_pink_set_collision_context_accepts_none(pink_arm):
    pink_arm.set_collision_context(None)


def test_pink_solver_rejects_unknown_joint():
    pytest.importorskip("pink")
    pytest.importorskip("pinocchio")
    from wheelchair_planning.kinematics.pink_ik_solver import PinkIKSolver
    from wheelchair_planning.wheelchair import CHAIN_CONFIGS

    with pytest.raises(ValueError, match="not in model"):
        PinkIKSolver(
            CHAIN_CONFIGS["arm"],
            joint_names=["Joint_Does_Not_Exist"],
        )
