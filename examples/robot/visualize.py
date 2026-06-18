"""Interactive path visualizer for wheelchair_planner.plan_to_joints.

Plans a path for every group listed in *groups* (default: all supported
groups) and lets you step through them one by one in a PyBullet GUI.

Usage::

    pixi run python robot/visualize.py
    pixi run python robot/visualize.py --group wheelchair_arm
    pixi run python robot/visualize.py --planner rrtc --time_limit 1.0
    pixi run python robot/visualize.py --fps 30

Controls inside the viewer:
    SPACE       toggle auto-play / pause
    ← / →       step one waypoint back / forward (while paused)
    n           advance to the next group
    close       quit
"""

from __future__ import annotations

import sys
import time
from pathlib import Path

import numpy as np
from fire import Fire

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from robot.wheelchair_planner import SUPPORTED_GROUPS, WheelchairPlanner  # noqa: E402

from wheelchair_planning.envs.pybullet_env import PyBulletEnv  # noqa: E402
from wheelchair_planning.wheelchair import (  # noqa: E402
    HOME_JOINTS,
    VIZ_URDF_PATH,
    wheelchair_robot_config,
)

# ---------------------------------------------------------------------------
# Default goals — one hand-crafted 10-DOF target per supported group.
#
# Layout: [base(3: x, y, yaw), arm(7: joint1..joint7)]
#
# Only the joints owned by the group differ from the start; every frozen
# joint keeps its start value so plan_to_joints never returns "not same".
# ---------------------------------------------------------------------------

_BASE = HOME_JOINTS[0:3].tolist()  # [0, 0, 0]
_ARM = HOME_JOINTS[3:10].tolist()  # home xArm7 pose

# Goal segments (active joints moved to natural poses).
_GOAL_ARM = [0.0, -0.6, 0.4, 1.5, -0.2, 0.6, 0.4]  # arm reach
_GOAL_BASE = [0.5, 0.3, 0.4]  # base translate + turn (x, y, yaw)


def _cfg(*segments: list) -> np.ndarray:
    return np.array([v for seg in segments for v in seg])


# Each value is a full 10-DOF goal config; only the group's active DOFs differ.
_DEFAULT_GOALS: dict[str, np.ndarray] = {
    "wheelchair_arm": _cfg(_BASE, _GOAL_ARM),
    "wheelchair_base": _cfg(_GOAL_BASE, _ARM),
    "wheelchair_whole_body": _cfg(_GOAL_BASE, _GOAL_ARM),
}

# Ordered for a natural demo progression (arm → base → whole body).
_DEFAULT_GROUP_ORDER = [
    "wheelchair_arm",
    "wheelchair_base",
    "wheelchair_whole_body",
]


def _banner(group: str, result, t_ms: float) -> None:
    if result is None:
        status = "NO PATH (timeout)"
    elif isinstance(result, str):
        status = f"SKIPPED ({result})"
    else:
        status = f"OK  {result.shape[0]} waypoints"
    print(f"  [{group}]  {status}  ({t_ms:.0f} ms)")


def main(
    group: str | None = None,
    planner: str = "bitstar",
    time_limit: float = 2.0,
    fps: float = 50.0,
) -> None:
    """Visualize planned paths for one or all supported planning groups."""
    if group is not None:
        if group not in SUPPORTED_GROUPS:
            print(
                f"Unknown group {group!r}.\n"
                f"Supported: {', '.join(sorted(SUPPORTED_GROUPS))}"
            )
            sys.exit(1)
        groups = [group]
    else:
        groups = [g for g in _DEFAULT_GROUP_ORDER if g in SUPPORTED_GROUPS]

    env = PyBulletEnv(
        wheelchair_robot_config, visualize=True, viz_urdf_path=VIZ_URDF_PATH
    )

    print(
        f"\n── wheelchair path visualizer ──\n"
        f"  planner={planner}  time_limit={time_limit}s\n"
        f"  {len(groups)} group(s): {', '.join(groups)}\n"
    )

    ap = WheelchairPlanner(planner_name=planner, time_limit=time_limit)
    start = _cfg(_BASE, _ARM)
    env.set_configuration(start)

    for g in groups:
        goal = _DEFAULT_GOALS[g]

        t0 = time.perf_counter()
        result = ap.plan_to_joints(g, start, goal)
        elapsed_ms = (time.perf_counter() - t0) * 1e3
        _banner(g, result, elapsed_ms)

        if isinstance(result, np.ndarray):
            _, path = ap.time_parameterize(result)
            cont = env.animate_path(path, fps=fps, next_key="n")
        else:
            env.set_configuration(start)
            msg = (
                "  (no path found — press 'n' for next group, close to quit)"
                if result is None
                else "  (skipped: frozen joints differ — press 'n' for next)"
            )
            env.wait_key("n", msg)
            cont = True

        if not cont:
            break

    env.wait_for_close()


if __name__ == "__main__":
    Fire(main)
