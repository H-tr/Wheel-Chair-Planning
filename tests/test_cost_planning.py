"""Soft-cost planning — mirrors ``examples/planning/cost``.

Only checks that the cost JIT-compiles, the planner accepts it, and the
resulting RRT* run completes (success or clean failure — both are fine;
we're not benchmarking).
"""

from __future__ import annotations

import os

import numpy as np
import pytest

pytest.importorskip("wheelchair_planning._ompl_vamp")
ca = pytest.importorskip("casadi")
pytest.importorskip("pinocchio")


@pytest.fixture(scope="module", autouse=True)
def _isolated_cost_cache(tmp_path_factory):
    cache = tmp_path_factory.mktemp("cost_cache")
    old = os.environ.get("WHEELCHAIR_COST_CACHE_DIR")
    os.environ["WHEELCHAIR_COST_CACHE_DIR"] = str(cache)
    yield
    if old is None:
        os.environ.pop("WHEELCHAIR_COST_CACHE_DIR", None)
    else:
        os.environ["WHEELCHAIR_COST_CACHE_DIR"] = old


SUBGROUP = "wheelchair_arm"
EE_LINK = "link_tcp"


def _build_height_cost():
    from wheelchair_planning.planning import Cost, SymbolicContext
    from wheelchair_planning.wheelchair import HOME_JOINTS

    ctx = SymbolicContext(SUBGROUP)
    start = HOME_JOINTS[ctx.active_indices].copy()
    tcp = ctx.link_translation(EE_LINK)
    p0 = np.asarray(ctx.evaluate_link_pose(EE_LINK, start))[:3, 3]
    residual = tcp[2] - float(p0[2])
    cost = Cost(
        expression=ca.sumsqr(residual),
        q_sym=ctx.q,
        name="height_test",
        weight=10.0,
    )
    return ctx, start, cost


def test_cost_compiles():
    _ctx, _start, cost = _build_height_cost()
    assert cost.so_path.exists()
    assert cost.ambient_dim == 7


def test_cost_rejects_non_scalar_expression():
    from wheelchair_planning.planning import Cost

    q = ca.SX.sym("q", 7)
    with pytest.raises(ValueError):
        Cost(expression=ca.vertcat(q[0], q[1]), q_sym=q, name="bad")


def test_cost_rejects_negative_weight():
    from wheelchair_planning.planning import Cost

    q = ca.SX.sym("q", 7)
    with pytest.raises(ValueError):
        Cost(expression=q[0] * q[0], q_sym=q, name="bad", weight=-1.0)


def test_planner_accepts_cost_and_runs():
    from wheelchair_planning.planning import create_planner
    from wheelchair_planning.types import PlannerConfig

    _ctx, start, cost = _build_height_cost()
    planner = create_planner(
        SUBGROUP,
        config=PlannerConfig(
            planner_name="rrtstar",
            time_limit=1.5,
            simplify=False,
        ),
        costs=[cost],
    )
    assert planner.validate(start)
    result = planner.plan(start, start)
    assert result.success


def test_cost_planned_path_endpoints_match_and_remain_valid():
    """A real plan must go from start to goal, every waypoint collision-free."""
    from wheelchair_planning.planning import create_planner
    from wheelchair_planning.types import PlannerConfig

    _ctx, start, cost = _build_height_cost()
    planner = create_planner(
        SUBGROUP,
        config=PlannerConfig(
            planner_name="rrtstar",
            time_limit=2.0,
            simplify=False,
        ),
        costs=[cost],
    )
    np.random.seed(0)
    goal = planner.sample_valid()
    result = planner.plan(start, goal)
    if not result.success:
        pytest.skip(f"cost-driven plan did not solve ({result.status.value})")

    assert result.path is not None
    np.testing.assert_allclose(result.path[0], start, atol=1e-6)
    np.testing.assert_allclose(result.path[-1], goal, atol=1e-6)
    for q in result.path:
        assert planner.validate(q), "every waypoint must be collision-free"
    assert np.isfinite(result.path_cost)
    assert result.path_cost >= 0.0
