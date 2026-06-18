"""Grasping demo: constrained IK + multi-segment planning + timing.

Pipeline
1) Solve constrained IK for pregrasp and grasp from a world-frame grasp pose
   (a top-down grasp).
2) Plan four segments with the xArm7 planner subgroup:
   home -> pregrasp -> grasp -> wait -> pregrasp -> home
3) Time-parameterize home -> pregrasp -> grasp, and grasp -> pregrasp -> home.
4) Visualize in PyBullet.

Recommended run (clean env):
    env -u PYTHONPATH -u LD_LIBRARY_PATH PYTHONNOUSERSITE=1 \
      python examples/robot/grasping.py
"""

from __future__ import annotations

import sys
import time
from pathlib import Path

import numpy as np
from fire import Fire

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from robot.wheelchair_planner import WheelchairPlanner  # noqa: E402

from wheelchair_planning.envs.pybullet_env import PyBulletEnv  # noqa: E402
from wheelchair_planning.kinematics import create_ik_solver  # noqa: E402
from wheelchair_planning.types import IKStatus, PinkIKConfig, SE3Pose  # noqa: E402
from wheelchair_planning.wheelchair import (  # noqa: E402
    HOME_JOINTS,
    JOINT_GROUPS,
    VIZ_URDF_PATH,
    wheelchair_robot_config,
)

GROUP = "wheelchair_arm"

# The arm subgroup occupies indices [3:10] of the full 10-DOF config.
ARM_IDX = np.arange(JOINT_GROUPS["arm"].start, JOINT_GROUPS["arm"].stop)

# The xArm7 tool control point; IK solves directly for this frame.
EE_LINK = "link_tcp"


def _unit(v: np.ndarray) -> np.ndarray:
    n = float(np.linalg.norm(v))
    if n < 1e-12:
        raise ValueError("zero vector cannot be normalized")
    return v / n


def _pose_to_matrix(position: np.ndarray, rotation: np.ndarray) -> np.ndarray:
    T = np.eye(4, dtype=np.float64)
    T[:3, :3] = np.asarray(rotation, dtype=np.float64)
    T[:3, 3] = np.asarray(position, dtype=np.float64)
    return T


def _matrix_to_pose(T: np.ndarray) -> SE3Pose:
    return SE3Pose(position=T[:3, 3], rotation=T[:3, :3])


def topdown_rotation(
    approach: np.ndarray, world_x_hint: np.ndarray | None = None
) -> np.ndarray:
    """Build a gripper frame whose +Y axis points opposite to *approach*."""
    a = _unit(np.asarray(approach, dtype=np.float64))
    y = -a

    if world_x_hint is None:
        world_x_hint = np.array([1.0, 0.0, 0.0], dtype=np.float64)
    z = np.asarray(world_x_hint, dtype=np.float64)
    if abs(float(np.dot(z, y))) > 0.95:
        z = np.array([0.0, 1.0, 0.0], dtype=np.float64)

    z = z - float(np.dot(z, y)) * y
    z = _unit(z)
    x = np.cross(y, z)
    x = _unit(x)
    return np.column_stack([x, y, z])


def embed_arm(chain_q: np.ndarray, base_full: np.ndarray) -> np.ndarray:
    """Embed a 7-DOF arm IK solution into a full 10-DOF config."""
    out = np.asarray(base_full, dtype=np.float64).copy()
    q = np.asarray(chain_q, dtype=np.float64)
    if q.shape != (7,):
        raise ValueError(f"arm solution must be shape (7,), got {q.shape}")
    out[ARM_IDX] = q
    return out


def stitch_paths(*segments: np.ndarray) -> np.ndarray:
    """Concatenate waypoint segments while removing duplicated boundaries."""
    kept: list[np.ndarray] = []
    for i, seg in enumerate(segments):
        s = np.asarray(seg, dtype=np.float64)
        if s.ndim != 2 or s.shape[0] == 0:
            raise ValueError("each segment must be a non-empty (N, DOF) array")
        kept.append(s if i == 0 else s[1:])
    return np.concatenate(kept, axis=0)


def plan_segment_or_raise(
    planner: WheelchairPlanner,
    start: np.ndarray,
    goal: np.ndarray,
    label: str,
    time_limit: float,
) -> np.ndarray:
    t0 = time.perf_counter()
    result = planner.plan_to_joints(GROUP, start, goal, time_limit=time_limit)
    dt_ms = (time.perf_counter() - t0) * 1e3

    if isinstance(result, str):
        raise RuntimeError(f"{label}: planner rejected start/goal ({result})")
    if result is None:
        raise RuntimeError(f"{label}: planning failed/timed out")

    print(f"  {label:<20} {result.shape[0]:>4} waypoints  ({dt_ms:.0f} ms)")
    return result


def solve_pose_or_raise(
    solver, target: SE3Pose, seed: np.ndarray, label: str
) -> np.ndarray:
    result = solver.solve_constrained(target, seed=seed)
    print(
        f"  IK {label:<10} {result.status.value:<14} "
        f"pos={result.position_error:.4f} m  ori={result.orientation_error:.4f} rad"
    )
    if result.status != IKStatus.SUCCESS or result.joint_positions is None:
        raise RuntimeError(f"IK failed for {label}: {result.status.value}")
    return np.asarray(result.joint_positions, dtype=np.float64)


def main(
    planner: str = "bitstar",
    time_limit: float = 2.0,
    hold_seconds: float = 2.0,
    pregrasp_offset: float = 0.10,
    dt: float = 0.02,
    vel_scale: float = 1.0,
    a_scale: float = 1.0,
    fps: float = 50.0,
) -> None:
    """Run grasping demo with constrained IK, planning, timing, and playback."""
    if hold_seconds < 0:
        raise ValueError("hold_seconds must be >= 0")
    if pregrasp_offset <= 0:
        raise ValueError("pregrasp_offset must be > 0")

    home = HOME_JOINTS.copy()

    print("\n--- Grasping demo ---")
    print(f"planner={planner}, time_limit={time_limit}s, dt={dt}s")

    print("\n1) Solving constrained IK (arm, pink backend) ...")
    ik_cfg = PinkIKConfig(
        dt=0.05,
        convergence_thresh=4e-3,
        lm_damping=1e-3,
        orientation_cost=0.15,
        camera_frame="camera_color_optical_frame",
        camera_cost=0.02,
        max_iterations=500,
    )
    solver = create_ik_solver("arm", backend="pink", config=ik_cfg)

    home_chain = home[ARM_IDX]

    approach = np.array([0.0, 0.0, -1.0], dtype=np.float64)
    grasp_rot = topdown_rotation(approach)

    # Desired TCP targets in world (IK solves directly for link_tcp). Tune the
    # grasp position to the xArm7 workspace (mounted high on the chassis).
    grasp_pos = np.array([0.6, 0.3, 0.7], dtype=np.float64)
    pregrasp_pos = grasp_pos - pregrasp_offset * _unit(approach)

    grasp_pose = _matrix_to_pose(_pose_to_matrix(grasp_pos, grasp_rot))
    pregrasp_pose = _matrix_to_pose(_pose_to_matrix(pregrasp_pos, grasp_rot))

    pregrasp_chain = solve_pose_or_raise(solver, pregrasp_pose, home_chain, "pregrasp")
    grasp_chain = solve_pose_or_raise(solver, grasp_pose, pregrasp_chain, "grasp")

    q_pregrasp = embed_arm(pregrasp_chain, home)
    q_grasp = embed_arm(grasp_chain, q_pregrasp)

    print("\n2) Planning segments (arm subgroup) ...")
    ap = WheelchairPlanner(planner_name=planner, time_limit=time_limit)

    s1 = plan_segment_or_raise(ap, home, q_pregrasp, "home -> pregrasp", time_limit)
    s2 = plan_segment_or_raise(ap, q_pregrasp, q_grasp, "pregrasp -> grasp", time_limit)
    s3 = plan_segment_or_raise(ap, q_grasp, q_pregrasp, "grasp -> pregrasp", time_limit)
    s4 = plan_segment_or_raise(ap, q_pregrasp, home, "pregrasp -> home", time_limit)

    print("\n3) Time-parameterizing outbound and inbound chunks ...")
    outbound = stitch_paths(s1, s2)
    inbound = stitch_paths(s3, s4)

    t_out, q_out = ap.time_parameterize(
        outbound, dt=dt, vel_scale=vel_scale, a_scale=a_scale
    )
    t_in, q_in = ap.time_parameterize(
        inbound, dt=dt, vel_scale=vel_scale, a_scale=a_scale
    )

    hold_steps = int(round(hold_seconds / dt))
    hold = np.repeat(q_grasp[None, :], hold_steps, axis=0)

    playback = np.concatenate(
        [
            q_out,
            hold,
            q_in[1:] if len(q_in) > 1 else q_in,
        ],
        axis=0,
    )

    print(
        "  outbound: "
        f"{len(q_out)} pts ({float(t_out[-1]) if len(t_out) else 0.0:.2f}s), "
        f"hold: {hold_seconds:.2f}s ({hold_steps} pts), "
        f"inbound: {len(q_in)} pts ({float(t_in[-1]) if len(t_in) else 0.0:.2f}s)"
    )

    print("\n4) Visualizing in PyBullet ...")
    env = PyBulletEnv(
        wheelchair_robot_config,
        visualize=True,
        viz_urdf_path=VIZ_URDF_PATH,
    )
    env.set_configuration(home)

    env.draw_sphere(grasp_pos, radius=0.018, color=(1.0, 0.2, 0.2, 0.85))
    env.draw_sphere(pregrasp_pos, radius=0.016, color=(1.0, 0.65, 0.1, 0.85))
    env.draw_frame(grasp_pos, grasp_rot, size=0.10, radius=0.004)
    env.draw_frame(pregrasp_pos, grasp_rot, size=0.08, radius=0.003)

    print("  Controls: SPACE play/pause | <-/-> step | close window to exit")
    env.animate_path(playback, fps=fps)
    env.wait_for_close()


if __name__ == "__main__":
    Fire(main)
