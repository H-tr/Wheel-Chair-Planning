"""Shared pytest fixtures.

The native ``_ompl_vamp`` extension and the URDFs it depends on are
built by the project's CMake / scikit-build pipeline.  Tests that need
the extension import it lazily through ``planner_factory`` so missing
artefacts surface as a clean *skip* rather than a collection error.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest


@pytest.fixture(scope="session")
def repo_root() -> Path:
    return Path(__file__).resolve().parent.parent


@pytest.fixture(scope="session")
def home_joints() -> np.ndarray:
    from wheelchair_planning.wheelchair import HOME_JOINTS

    return HOME_JOINTS.copy()


@pytest.fixture(scope="session")
def arm_planner():
    """Arm planner with a fast RRT-Connect config — shared across tests."""
    pytest.importorskip("wheelchair_planning._ompl_vamp")
    from wheelchair_planning.planning import create_planner
    from wheelchair_planning.types import PlannerConfig

    return create_planner(
        "wheelchair_arm",
        config=PlannerConfig(planner_name="rrtc", time_limit=2.0),
    )


@pytest.fixture(scope="session")
def arm_start(arm_planner, home_joints) -> np.ndarray:
    return arm_planner.extract_config(home_joints)


@pytest.fixture(scope="session")
def table_pointcloud(repo_root) -> np.ndarray:
    """Bundled table.ply rotated and shifted in front of the robot.

    Mirrors ``examples/planning/motion.py::load_table`` so the motion-planning
    tests exercise the same geometry the demo uses. The table is placed far
    enough forward/up to clear the (large) wheelchair chassis at HOME — the
    base is "parked" near the table and only the arm plans around it.
    """
    trimesh = pytest.importorskip("trimesh")
    import wheelchair_planning

    pkg_root = Path(wheelchair_planning.__file__).parent
    pcd = trimesh.load(str(pkg_root / "resources" / "envs" / "pcd" / "table.ply"))
    pts = np.asarray(pcd.vertices, dtype=np.float32)
    pts = pts - pts.mean(axis=0)
    rot = np.array(
        [[0.0, -1.0, 0.0], [1.0, 0.0, 0.0], [0.0, 0.0, 1.0]], dtype=np.float32
    )
    pts = pts @ rot.T
    # 1.6 m forward keeps ~0.2 m margin past the HOME collision boundary
    # (~1.375 m in the pixi build) so floating-point differences between
    # compilers/builds (e.g. manylinux wheels) cannot tip HOME into collision.
    pts[:, 0] += 1.6
    pts[:, 2] += 0.9
    return pts
