"""Post-install smoke tests — minimal sanity check after ``pip install``.

These tests exist so CI can diagnose "did the wheel build/install
correctly?" in under a second, before spending minutes on the full
correctness suite.

Covered:

- Every public sub-package imports cleanly (types, wheelchair, planning,
  trajectory, utils).
- The native planner extension (``_ompl_vamp``) loads.
- The vendored TOPP-RA time parameterization runs.
- ``SymbolicContext`` builds symbolic FK (pure CasADi, no pinocchio).
- A single end-to-end plan + time-parameterize round-trip succeeds.
"""

from __future__ import annotations

import numpy as np
import pytest

# ── imports ──────────────────────────────────────────────────────────


def test_top_level_package_imports():
    import wheelchair_planning  # noqa: F401
    from wheelchair_planning import (  # noqa: F401
        planning,
        trajectory,
        types,
        wheelchair,
    )


def test_wheelchair_robot_config_populated():
    from wheelchair_planning.wheelchair import (
        HOME_JOINTS,
        PLANNING_SUBGROUPS,
        wheelchair_robot_config,
    )

    assert HOME_JOINTS.shape == (10,)
    assert len(PLANNING_SUBGROUPS) > 0
    assert wheelchair_robot_config.max_velocity is not None
    assert wheelchair_robot_config.max_velocity.shape == (10,)
    assert wheelchair_robot_config.max_acceleration is not None
    assert wheelchair_robot_config.max_acceleration.shape == (10,)


# ── native extensions ────────────────────────────────────────────────


def test_time_parameterizer_runs():
    """Vendored TOPP-RA (incl. its Cython extensions) must parameterize a tiny path."""
    from wheelchair_planning.trajectory import TimeOptimalParameterizer

    path = np.array([[0.0, 0.0], [0.5, 0.3], [1.0, 0.6]])
    param = TimeOptimalParameterizer(
        max_velocity=np.ones(2),
        max_acceleration=np.ones(2) * 2.0,
    )
    traj = param.parameterize(path)
    assert traj.duration > 0.0


def test_toppra_extensions_load():
    """The Cython extensions compiled into our wheel for TOPP-RA must load."""
    from toppra import _CythonUtils  # noqa: F401
    from toppra.solverwrapper import cy_seidel_solverwrapper  # noqa: F401


def test_planner_extension_loads():
    pytest.importorskip("wheelchair_planning._ompl_vamp")
    from wheelchair_planning.planning import create_planner
    from wheelchair_planning.types import PlannerConfig

    planner = create_planner(
        "wheelchair_arm",
        config=PlannerConfig(planner_name="rrtc", time_limit=1.0),
    )
    assert planner is not None


# ── symbolic FK ──────────────────────────────────────────────────────


def test_symbolic_context_builds_fk():
    """SymbolicContext needs only casadi (no pinocchio) to build symbolic FK."""
    from wheelchair_planning.planning import SymbolicContext

    ctx = SymbolicContext("wheelchair_arm")
    assert len(ctx.active_indices) == 7
    # Smoke the FK path: position of the tool link must be a 3-vector.
    pos = ctx.link_translation("link_tcp")
    assert pos.shape == (3, 1)


# ── end-to-end ───────────────────────────────────────────────────────


def test_end_to_end_plan_and_parameterize():
    """One pass through VAMP/OMPL planning + TOPP-RA timing."""
    pytest.importorskip("wheelchair_planning._ompl_vamp")
    from wheelchair_planning.planning import create_planner
    from wheelchair_planning.trajectory import TimeOptimalParameterizer
    from wheelchair_planning.types import PlannerConfig
    from wheelchair_planning.wheelchair import HOME_JOINTS

    planner = create_planner(
        "wheelchair_arm",
        config=PlannerConfig(planner_name="rrtc", time_limit=2.0),
    )
    start = planner.extract_config(HOME_JOINTS)
    assert planner.validate(start), "HOME must be collision-free"
    # Plan to a sampled *collision-free* goal. A single sample can land on a
    # collision boundary where sample_valid()'s SIMD-batch check and OMPL's
    # scalar planner check disagree under floating-point differences between
    # builds (e.g. manylinux wheels), making the smoke test flaky. Retry with
    # fresh goals until one is actually plannable — this verifies the real
    # property (planning to a reachable goal works) without depending on
    # build-specific rounding at a boundary.
    np.random.seed(0)
    result = None
    for _ in range(12):
        goal = planner.sample_valid()
        result = planner.plan(start, goal, time_limit=2.0)
        if result.success:
            break
    assert result is not None and result.success
    assert result.path is not None and result.path.shape[0] >= 2

    param = TimeOptimalParameterizer(np.full(7, 0.5), np.full(7, 0.6))
    traj = param.parameterize(result.path)
    assert traj.duration > 0.0

    times, positions, velocities, accelerations = traj.sample_uniform(0.02)
    assert positions.shape[1] == 7
    assert velocities.shape == positions.shape
    assert accelerations.shape == positions.shape
    assert abs(times[-1] - traj.duration) < 1e-6
