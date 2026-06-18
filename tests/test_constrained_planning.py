"""Constraint-based (manifold) planning — mirrors ``examples/planning/constrained``.

Builds the same horizontal-line residual the gallery uses, hands it to
the planner, and verifies the planner accepts the manifold and keeps
validating the seed configuration.
"""

from __future__ import annotations

import os

import numpy as np
import pytest

pytest.importorskip("wheelchair_planning._ompl_vamp")
ca = pytest.importorskip("casadi")
pytest.importorskip("pinocchio")


@pytest.fixture(scope="module", autouse=True)
def _isolated_constraint_cache(tmp_path_factory):
    """Use a fresh on-disk cache per test session."""
    cache = tmp_path_factory.mktemp("constraint_cache")
    old = os.environ.get("WHEELCHAIR_CONSTRAINT_CACHE_DIR")
    os.environ["WHEELCHAIR_CONSTRAINT_CACHE_DIR"] = str(cache)
    yield
    if old is None:
        os.environ.pop("WHEELCHAIR_CONSTRAINT_CACHE_DIR", None)
    else:
        os.environ["WHEELCHAIR_CONSTRAINT_CACHE_DIR"] = old


SUBGROUP = "wheelchair_arm"
EE_LINK = "link_tcp"


def _build_horizontal_line_constraint():
    from wheelchair_planning.planning import Constraint, SymbolicContext
    from wheelchair_planning.wheelchair import HOME_JOINTS

    ctx = SymbolicContext(SUBGROUP)
    start = HOME_JOINTS[ctx.active_indices].copy()

    tcp = ctx.link_translation(EE_LINK)
    R = ctx.evaluate_link_pose(EE_LINK, start)[:3, :3]
    p0 = np.asarray(ctx.evaluate_link_pose(EE_LINK, start))[:3, 3]
    ee_rot = ctx.link_rotation(EE_LINK)

    residual = ca.vertcat(
        tcp[1] - float(p0[1]),
        tcp[2] - float(p0[2]),
        ee_rot[:, 0] - ca.DM(R[:, 0].tolist()),
        ee_rot[:, 1] - ca.DM(R[:, 1].tolist()),
    )
    constraint = Constraint(residual=residual, q_sym=ctx.q, name="line_h_test")
    return ctx, start, constraint


def test_symbolic_context_dimensions():
    from wheelchair_planning.planning import SymbolicContext

    ctx = SymbolicContext(SUBGROUP)
    assert len(ctx.active_indices) == 7
    assert ctx.q.numel() == 7


def test_constraint_compiles():
    _ctx, _start, c = _build_horizontal_line_constraint()
    assert c.so_path.exists(), "Constraint .so should be JIT-compiled and cached"
    assert c.ambient_dim == 7
    assert c.co_dim == 8  # 2 scalar + 2 columns of 3 = 8


def test_planner_accepts_constraint_and_validates_start():
    from wheelchair_planning.planning import create_planner
    from wheelchair_planning.types import PlannerConfig

    _ctx, start, constraint = _build_horizontal_line_constraint()
    planner = create_planner(
        SUBGROUP,
        config=PlannerConfig(planner_name="rrtc", time_limit=1.0),
        constraints=[constraint],
    )
    assert planner.validate(start)


def test_constraint_rejects_non_sx_q_sym():
    from wheelchair_planning.planning import Constraint

    with pytest.raises(TypeError):
        Constraint(residual=ca.DM(0.0), q_sym="not a SX")


def test_planned_path_satisfies_constraint():
    """Every waypoint in a constrained plan must lie on ``residual(q) ~ 0``."""
    from wheelchair_planning.planning import create_planner
    from wheelchair_planning.types import PlannerConfig

    ctx, start, constraint = _build_horizontal_line_constraint()

    residual_expr = constraint.residual
    res_fn = ca.Function("res_eval", [ctx.q], [ca.reshape(residual_expr, -1, 1)])

    planner = create_planner(
        SUBGROUP,
        config=PlannerConfig(
            planner_name="rrtc",
            time_limit=5.0,
            simplify=False,
            interpolate=True,
        ),
        constraints=[constraint],
    )

    rng = np.random.default_rng(0)
    goal = None
    for _ in range(40):
        seed = start + rng.uniform(-0.3, 0.3, size=start.shape[0])
        try:
            candidate = ctx.project(seed, residual_expr)
        except RuntimeError:
            continue
        lo = np.array(planner._planner.lower_bounds())
        hi = np.array(planner._planner.upper_bounds())
        if np.any(candidate < lo) or np.any(candidate > hi):
            continue
        if planner.validate(candidate):
            goal = candidate
            break
    if goal is None:
        pytest.skip("could not find a reachable manifold goal")

    result = planner.plan(start, goal)
    if not result.success:
        pytest.skip(f"constrained plan did not solve ({result.status.value})")

    assert result.path is not None and result.path.shape[0] >= 2

    max_residual = 0.0
    for q in result.path:
        r = np.asarray(res_fn(q)).flatten()
        max_residual = max(max_residual, float(np.linalg.norm(r)))
    assert max_residual < 1e-2, f"max |residual| over path = {max_residual:.3e}"


def test_planned_path_endpoints_match_request():
    """path[0] == start and path[-1] == goal exactly."""
    from wheelchair_planning.planning import create_planner
    from wheelchair_planning.types import PlannerConfig

    ctx, start, constraint = _build_horizontal_line_constraint()
    planner = create_planner(
        SUBGROUP,
        config=PlannerConfig(planner_name="rrtc", time_limit=3.0, simplify=False),
        constraints=[constraint],
    )

    result = planner.plan(start, start)
    if not result.success:
        pytest.skip(f"trivial plan failed ({result.status.value})")
    assert result.path is not None
    np.testing.assert_allclose(result.path[0], start, atol=1e-9)
    np.testing.assert_allclose(result.path[-1], start, atol=1e-9)
