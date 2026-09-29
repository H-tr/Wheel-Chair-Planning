"""Kinodynamic (FLASK) planning: drive to a table, then reach over it.

The wheelchair is a differential drive — its rear hub motors drive and the
front omni wheels roll freely — so the base can only move along its heading
or rotate in place.  :meth:`MotionPlanner.plan_kinodynamic` plans
time-parameterised trajectories that respect this and the joint / base
velocity and acceleration limits in ``wheelchair_planning/wheelchair.py``:

1. **drive** — the base subgroup steers around a pillar and parks in front
   of the table, arm tucked.
2. **reach** — the arm subgroup moves over the table from the parked pose.

``--whole_body`` plans both at once (base and arm move together); this is
a harder, higher-dimensional problem and needs a longer time limit.

    pixi run python examples/planning/kinodynamic.py
    pixi run python examples/planning/kinodynamic.py --whole_body --time_limit 5
    pixi run python examples/planning/kinodynamic.py --visualize=False
"""

from __future__ import annotations

import numpy as np
from fire import Fire
from motion import load_table

from wheelchair_planning.envs.pybullet_env import PyBulletEnv
from wheelchair_planning.planning import create_planner
from wheelchair_planning.types import KinodynamicConfig, PlannerConfig
from wheelchair_planning.wheelchair import HOME_JOINTS, wheelchair_robot_config

START_BASE = (-1.2, 0.0, 0.0)  # x, y, yaw
PARK_BASE = (1.6, 0.0, 0.0)  # footrest ~0.1 m from the table edge
REACH_ARM = np.array([0.0, 0.3, 0.0, 0.5, 0.0, 0.2, 0.0])


def pillar(center, radius=0.15, height=1.2, spacing=0.02) -> np.ndarray:
    """Point cloud of a vertical cylinder standing on the floor."""
    n = int(2 * np.pi * radius / spacing)
    ang, z = np.meshgrid(
        np.linspace(0.0, 2 * np.pi, n, endpoint=False), np.arange(0.0, height, spacing)
    )
    return np.c_[
        center[0] + radius * np.cos(ang).ravel(),
        center[1] + radius * np.sin(ang).ravel(),
        z.ravel(),
    ].astype(np.float32)


def full(base, arm=None) -> np.ndarray:
    q = HOME_JOINTS.copy()
    q[:3] = base
    if arm is not None:
        q[3:] = arm
    return q


def plan_leg(name, subgroup, cloud, start, goal, config):
    """Plan one leg; returns the 10-DOF sampled trajectory or ``None``."""
    planner = create_planner(
        subgroup,
        config=PlannerConfig(point_radius=0.012),
        base_config=start,
        pointcloud=cloud,
    )
    # Base positions are sampled inside these bounds: keep them tight.
    planner.set_base_bounds(-2.0, 3.0, -1.5, 1.5)
    result = planner.plan_kinodynamic(
        planner.extract_config(start), planner.extract_config(goal), config=config
    )
    if not result.success:
        print(f"  {name}: {result.status.value}")
        return None

    traj = result.trajectory
    times, q, qd, _ = traj.sample_uniform(dt=1.0 / 60.0)
    line = (
        f"  {name}: planned in {result.planning_time_ns / 1e6:.0f} ms, "
        f"{traj.duration:.1f} s long, {traj.num_segments} segments"
    )
    if traj.has_base:
        twist = traj.sample_base_twist(times)
        line += (
            f", {traj.segment_kinds.count('spin')} rotate-in-place, "
            f"{sum(traj.segment_gears)} in reverse, "
            f"max speed {np.abs(twist[:, 0]).max():.2f} m/s"
        )
    print(line)
    return planner.embed_path(q)


def main(
    whole_body: bool = False,
    time_limit: float = 3.0,
    seed: int = 1,
    visualize: bool = True,
) -> None:
    cloud = np.vstack([load_table(distance=3.0, height=0.9), pillar((0.6, 0.0))])
    config = KinodynamicConfig(time_limit=time_limit, seed=seed)
    start, parked, reached = (
        full(START_BASE),
        full(PARK_BASE),
        full(PARK_BASE, REACH_ARM),
    )

    print("── kinodynamic planning: drive to the table, reach over it ──")
    if whole_body:
        legs = [
            plan_leg(
                "drive + reach",
                "wheelchair_whole_body",
                cloud,
                start,
                reached,
                config,
            )
        ]
    else:
        legs = [
            plan_leg("drive", "wheelchair_base", cloud, start, parked, config),
            plan_leg("reach", "wheelchair_arm", cloud, parked, reached, config),
        ]
    if not visualize or any(leg is None for leg in legs):
        return

    env = PyBulletEnv(wheelchair_robot_config, visualize=True)
    env.add_pointcloud(cloud, pointsize=3)
    env.animate_path(np.vstack(legs), fps=60)


if __name__ == "__main__":
    Fire(main)
