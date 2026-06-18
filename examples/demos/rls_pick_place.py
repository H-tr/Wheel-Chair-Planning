"""End-to-end pick-and-place demo for the wheelchair + xArm7.

NOTE: this is a *simplified* adaptation of the reference
``Autolife-Planning`` demo, which orchestrated a 24-DOF humanoid through a
multi-room scene using height/body/torso subgroups (squat, whole-body
leg-pinned reaches, dual-arm hand-offs). None of those map onto a 7-DOF
arm on a wheeled base, so this demo keeps the *shape* of the original —
constrained IK for grasp poses, multi-segment planning, time
parameterization, PyBullet playback — but reduces it to a single arm
picking an object off a table and placing it elsewhere on the table.

Pipeline:
    1. Solve top-down grasp IK at the pick and place locations.
    2. Plan home -> pre-pick -> pick -> pre-place -> place -> home.
    3. Time-parameterize and animate, with the table point cloud as an
       obstacle the planner must avoid.

Run (clean env):
    env -u PYTHONPATH -u LD_LIBRARY_PATH PYTHONNOUSERSITE=1 \
      python examples/demos/rls_pick_place.py
"""

from __future__ import annotations

import sys
import time
from pathlib import Path

import numpy as np
from fire import Fire

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from robot.grasping import embed_arm, solve_pose_or_raise, stitch_paths, topdown_rotation
from robot.wheelchair_planner import WheelchairPlanner

import wheelchair_planning
from wheelchair_planning.envs.pybullet_env import PyBulletEnv
from wheelchair_planning.kinematics import create_ik_solver
from wheelchair_planning.types import PinkIKConfig, SE3Pose
from wheelchair_planning.wheelchair import (
    HOME_JOINTS,
    VIZ_URDF_PATH,
    wheelchair_robot_config,
)

GROUP = "wheelchair_arm"


def load_table(distance: float = 0.6, height: float = 0.7) -> np.ndarray:
    """Load the bundled table point cloud and place it in front of the robot."""
    pkg_root = Path(wheelchair_planning.__file__).parent
    import trimesh

    pcd = trimesh.load(str(pkg_root / "resources" / "envs" / "pcd" / "table.ply"))
    pts = np.asarray(pcd.vertices, dtype=np.float32)
    pts = pts - pts.mean(axis=0)
    rot = np.array(
        [[0.0, -1.0, 0.0], [1.0, 0.0, 0.0], [0.0, 0.0, 1.0]], dtype=np.float32
    )
    pts = pts @ rot.T
    pts[:, 0] += float(distance)
    pts[:, 2] += float(height)
    return pts


def _grasp_pose(xyz: np.ndarray) -> SE3Pose:
    """Top-down grasp pose at *xyz* (gripper approaching straight down)."""
    rot = topdown_rotation(np.array([0.0, 0.0, -1.0]))
    return SE3Pose(position=np.asarray(xyz, dtype=np.float64), rotation=rot)


def main(
    planner: str = "rrtc",
    time_limit: float = 3.0,
    lift: float = 0.12,
    dt: float = 0.02,
    fps: float = 50.0,
) -> None:
    home = HOME_JOINTS.copy()
    cloud = load_table()

    print("\n--- Wheelchair pick-and-place demo (simplified) ---")

    # Grasp targets just above the table surface.
    pick_xyz = np.array([0.55, 0.10, 0.78])
    place_xyz = np.array([0.55, 0.40, 0.78])

    ik_cfg = PinkIKConfig(lm_damping=1e-3, max_iterations=400)
    solver = create_ik_solver("arm", backend="pink", config=ik_cfg)
    arm_home = home[3:10]

    print("1) Solving grasp IK ...")
    pre_pick = embed_arm(
        solve_pose_or_raise(solver, _grasp_pose(pick_xyz + [0, 0, lift]), arm_home, "pre-pick"),
        home,
    )
    pick = embed_arm(
        solve_pose_or_raise(solver, _grasp_pose(pick_xyz), pre_pick[3:10], "pick"),
        pre_pick,
    )
    pre_place = embed_arm(
        solve_pose_or_raise(solver, _grasp_pose(place_xyz + [0, 0, lift]), pick[3:10], "pre-place"),
        pick,
    )
    place = embed_arm(
        solve_pose_or_raise(solver, _grasp_pose(place_xyz), pre_place[3:10], "place"),
        pre_place,
    )

    print("2) Planning segments (with the table as an obstacle) ...")
    ap = WheelchairPlanner(planner_name=planner, time_limit=time_limit)
    # Attach the obstacle cloud to the underlying planner.
    ap._planner.add_pointcloud(cloud)  # noqa: SLF001 - demo convenience

    segs = []
    waypoints = [home, pre_pick, pick, pre_place, place, home]
    labels = ["home->pre_pick", "pre_pick->pick", "pick->pre_place",
              "pre_place->place", "place->home"]
    for (a, b), label in zip(zip(waypoints[:-1], waypoints[1:]), labels):
        t0 = time.perf_counter()
        res = ap.plan_to_joints(GROUP, a, b, time_limit=time_limit)
        if not isinstance(res, np.ndarray):
            print(f"  {label}: FAILED ({res})")
            return
        print(f"  {label:<20} {res.shape[0]:>4} wp ({(time.perf_counter()-t0)*1e3:.0f} ms)")
        segs.append(res)

    full = stitch_paths(*segs)
    _, traj = ap.time_parameterize(full, dt=dt)

    print("3) Visualizing ...")
    env = PyBulletEnv(
        wheelchair_robot_config, visualize=True, viz_urdf_path=VIZ_URDF_PATH
    )
    env.set_configuration(home)
    env.add_pointcloud(cloud, pointsize=3)
    env.draw_sphere(pick_xyz, radius=0.02, color=(1.0, 0.2, 0.2, 0.85))
    env.draw_sphere(place_xyz, radius=0.02, color=(0.2, 0.6, 1.0, 0.85))
    env.animate_path(traj, fps=fps)
    env.wait_for_close()


if __name__ == "__main__":
    Fire(main)
