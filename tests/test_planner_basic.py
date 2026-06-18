"""End-to-end checks for the OMPL+VAMP planner without obstacles.

Mirrors the spirit of ``examples/planning/motion.py`` and
``examples/planning/subgroup.py`` (without the table) — small, fast, and
only exercises the bits the demos demonstrate.
"""

from __future__ import annotations

import numpy as np
import pytest

pytest.importorskip("wheelchair_planning._ompl_vamp")


def test_available_robots_includes_known_subgroups():
    from wheelchair_planning.planning import available_robots

    names = available_robots()
    assert "wheelchair" in names
    assert "wheelchair_arm" in names
    assert "wheelchair_base" in names
    assert "wheelchair_whole_body" in names


def test_planner_dimension_matches_subgroup(arm_planner):
    # The arm is the 7-DOF chain in PLANNING_SUBGROUPS.
    assert arm_planner._ndof == 7
    lo = np.asarray(arm_planner._planner.lower_bounds())
    hi = np.asarray(arm_planner._planner.upper_bounds())
    assert lo.shape == (7,) and hi.shape == (7,)
    assert np.all(hi > lo)


def test_extract_embed_round_trip(arm_planner, home_joints):
    extracted = arm_planner.extract_config(home_joints)
    embedded = arm_planner.embed_config(extracted)
    np.testing.assert_allclose(embedded, home_joints)


def test_home_is_collision_free(arm_planner, arm_start):
    assert arm_planner.validate(arm_start)


def test_sample_valid_returns_collision_free(arm_planner):
    cfg = arm_planner.sample_valid()
    assert cfg.shape == (7,)
    assert arm_planner.validate(cfg)


def test_trivial_plan_succeeds(arm_planner, arm_start):
    """Planning from a state to itself must always succeed."""
    result = arm_planner.plan(arm_start, arm_start)
    assert result.success
    assert result.path is not None and result.path.shape[1] == 7


def test_plan_to_random_valid_goal(arm_planner, arm_start):
    np.random.seed(0)
    goal = arm_planner.sample_valid()
    result = arm_planner.plan(arm_start, goal)
    assert result.status.value in {"success", "failed"}
    if result.success:
        assert result.path is not None
        np.testing.assert_allclose(result.path[0], arm_start, atol=1e-6)
        np.testing.assert_allclose(result.path[-1], goal, atol=1e-6)


def test_plan_rejects_wrong_dimension(arm_planner, arm_start):
    with pytest.raises(ValueError):
        arm_planner.plan(arm_start, np.zeros(8))


def test_simplify_and_interpolate_path(home_joints):
    """Post-hoc simplify + interpolate round-trip on a real plan."""
    from wheelchair_planning.planning import create_planner
    from wheelchair_planning.types import PlannerConfig

    raw_planner = create_planner(
        "wheelchair_arm",
        config=PlannerConfig(
            planner_name="rrtc",
            time_limit=2.0,
            simplify=False,
            interpolate=False,
        ),
    )
    start = raw_planner.extract_config(home_joints)
    np.random.seed(0)
    goal = raw_planner.sample_valid()
    result = raw_planner.plan(start, goal)
    if not result.success:
        pytest.skip("rrtc didn't find a path for this random goal")

    n_raw = result.path.shape[0]

    simp = raw_planner.simplify_path(result.path, time_limit=1.0)
    assert simp.shape[1] == 7
    assert simp.shape[0] <= n_raw
    np.testing.assert_allclose(simp[0], start, atol=1e-6)
    np.testing.assert_allclose(simp[-1], goal, atol=1e-6)

    dense_count = raw_planner.interpolate_path(simp, count=50, resolution=0.0)
    assert dense_count.shape == (50, 7)
    np.testing.assert_allclose(dense_count[0], simp[0], atol=1e-6)
    np.testing.assert_allclose(dense_count[-1], simp[-1], atol=1e-6)

    dense_res = raw_planner.interpolate_path(simp, count=0, resolution=64.0)
    assert dense_res.shape[1] == 7
    assert dense_res.shape[0] >= simp.shape[0]

    with pytest.raises(ValueError):
        raw_planner.interpolate_path(simp, count=10, resolution=64.0)


def test_simplify_path_rejects_wrong_dimension(arm_planner):
    with pytest.raises(ValueError):
        arm_planner.simplify_path(np.zeros((4, 8)))


def test_interpolate_path_rejects_wrong_dimension(arm_planner):
    with pytest.raises(ValueError):
        arm_planner.interpolate_path(np.zeros((4, 8)))


def test_validate_batch_matches_single(arm_planner, arm_start):
    """Batched SIMD check must agree with per-config calls on every sample."""
    np.random.seed(0)
    lo = np.asarray(arm_planner._planner.lower_bounds())
    hi = np.asarray(arm_planner._planner.upper_bounds())
    samples = np.random.uniform(lo, hi, size=(37, 7))
    samples[0] = arm_start

    expected = np.array([arm_planner.validate(s) for s in samples], dtype=bool)
    got = arm_planner.validate_batch(samples)
    assert got.shape == (37,) and got.dtype == bool
    np.testing.assert_array_equal(got, expected)


def test_validate_batch_empty(arm_planner):
    out = arm_planner.validate_batch(np.zeros((0, 7)))
    assert out.shape == (0,) and out.dtype == bool


def test_validate_batch_rejects_wrong_dimension(arm_planner):
    with pytest.raises(ValueError):
        arm_planner.validate_batch(np.zeros((4, 8)))


def test_validate_batch_full_body_roundtrip(home_joints):
    """Full-body (10-DOF) batched check — no subgroup expansion path."""
    from wheelchair_planning._ompl_vamp import OmplVampPlanner

    planner = OmplVampPlanner()
    home = home_joints.tolist()
    out_of_bounds = [10.0] * 10

    batch = [home, out_of_bounds, home, out_of_bounds] * 3
    got = planner.validate_batch(batch)
    expected = [planner.validate(c) for c in batch]
    assert got == expected


def test_whole_body_configs_collision_free(home_joints):
    """The 10-DOF whole-body subgroup must validate HOME and the zero config."""
    from wheelchair_planning.planning import create_planner

    planner = create_planner("wheelchair_whole_body", base_config=home_joints)
    assert planner._ndof == 10
    assert planner.validate(home_joints), "HOME should be collision-free"
    assert planner.validate(
        np.zeros(10)
    ), "all-zero 10-DOF config should be collision-free"
