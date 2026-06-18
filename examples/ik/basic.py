"""Minimal IK example — no visualization, no PyBullet.

Available chains:
    "arm"          7 DOF   (link_base -> link_tcp, the xArm7)
    "whole_body"  10 DOF   (Link_Zero_Point -> link_tcp, planar base + arm)

JOINT_GROUPS (indices into the full 10-DOF config):
    base [0:3]   Virtual_X, Virtual_Y, Virtual_Theta
    arm  [3:10]  joint1 .. joint7
"""

import numpy as np

from wheelchair_planning.kinematics import create_ik_solver
from wheelchair_planning.types import IKConfig, SE3Pose, SolveType
from wheelchair_planning.wheelchair import HOME_JOINTS, JOINT_GROUPS

# Home configuration subset for the arm chain (via JOINT_GROUPS).
G = JOINT_GROUPS
HOME_ARM = HOME_JOINTS[G["arm"]]


def main():
    # --- IKConfig (all fields shown with defaults) ---
    config = IKConfig(
        timeout=0.2,  # seconds per TRAC-IK attempt
        epsilon=1e-5,  # convergence tolerance
        solve_type=SolveType.SPEED,  # SPEED | DISTANCE | MANIP1 | MANIP2
        max_attempts=10,  # random restart attempts
        position_tolerance=1e-4,  # post-solve check (meters)
        orientation_tolerance=1e-4,  # post-solve check (radians)
    )

    # --- Create solver ---
    solver = create_ik_solver("arm", config=config)
    print(
        f"Chain: {solver.base_frame} -> {solver.ee_frame} ({solver.num_joints} joints)"
    )

    # Forward kinematics: end-effector pose at the home configuration.
    home_pose = solver.fk(HOME_ARM)
    print(f"Home EE position: {home_pose.position}")

    # Define a target pose: small offset, keep the same orientation.
    target = SE3Pose(
        position=home_pose.position + np.array([0.05, 0.0, -0.05]),
        rotation=home_pose.rotation,
    )

    # Solve IK (seed is optional; if None, uses random within joint limits).
    result = solver.solve(target, seed=HOME_ARM)
    print(f"IK status: {result.status.value}")
    print(f"  position error:    {result.position_error:.6f} m")
    print(f"  orientation error: {result.orientation_error:.6f} rad")

    if result.success:
        print(f"  solution: {np.round(result.joint_positions, 4)}")
        achieved = solver.fk(result.joint_positions)
        print(f"  achieved position: {np.round(achieved.position, 4)}")
    else:
        print("  IK failed to find a valid solution.")


if __name__ == "__main__":
    main()
