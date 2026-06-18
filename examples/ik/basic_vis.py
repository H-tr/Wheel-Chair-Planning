"""IK solver example using TRAC-IK with PyBullet visualization.

Solves IK for the 7-DOF xArm7 arm chain (``link_base`` -> ``link_tcp``)
and visualizes the home pose, the target frame, and the solution.
"""

import time

import numpy as np
import pybullet as pb

from wheelchair_planning.envs.pybullet_env import PyBulletEnv
from wheelchair_planning.kinematics import create_ik_solver
from wheelchair_planning.types import IKConfig, SE3Pose, SolveType
from wheelchair_planning.wheelchair import (
    CHAIN_CONFIGS,
    HOME_JOINTS,
    JOINT_GROUPS,
    wheelchair_robot_config,
)

# Home configuration subset matching the arm chain's joint ordering.
G = JOINT_GROUPS
HOME_ARM = HOME_JOINTS[G["arm"]]

CHAIN_SEEDS = {
    "arm": HOME_ARM,
}

# Mapping from chain solution indices to the actuated (non-base) joint indices
# the visualization sets. The wheelchair's actuated joints are joint1..joint7,
# i.e. HOME_JOINTS[3:], so the arm solution maps 1:1 onto indices 0..6.
CHAIN_TO_BODY = {
    "arm": list(range(0, 7)),
}


def get_ee_link_index(env, link_name):
    """Find PyBullet link index by name."""
    client = env.sim.client
    for i in range(client.getNumJoints(env.sim.skel_id)):
        info = client.getJointInfo(env.sim.skel_id, i)
        if info[12].decode("utf-8") == link_name:
            return i
    return -1


def draw_frame_at_link(env, link_index, length=0.08, width=3):
    """Draw RGB axes at a link's world pose. Returns debug line IDs."""
    client = env.sim.client
    state = client.getLinkState(env.sim.skel_id, link_index)
    pos = np.array(state[0])
    rot = np.array(client.getMatrixFromQuaternion(state[1])).reshape(3, 3)

    line_ids = []
    for axis_idx, color in enumerate([[1, 0, 0], [0, 1, 0], [0, 0, 1]]):
        axis = np.zeros(3)
        axis[axis_idx] = length
        end = (pos + rot @ axis).tolist()
        line_ids.append(
            client.addUserDebugLine(pos.tolist(), end, color, lineWidth=width)
        )
    return line_ids


def wait_key(env, key, msg):
    """Wait for a key press in the PyBullet GUI."""
    client = env.sim.client
    text_id = client.addUserDebugText(
        msg, [0, 0, 1.5], textColorRGB=[0, 0, 0], textSize=1.5
    )
    print(msg)
    while True:
        keys = client.getKeyboardEvents()
        if key in keys and keys[key] & pb.KEY_WAS_TRIGGERED:
            break
        time.sleep(0.01)
    client.removeUserDebugItem(text_id)


def test_chain(env, chain_name):
    """Solve IK for one chain and visualize."""
    print(f"\n{'='*60}")
    print(f"Chain: {chain_name}")
    print(f"{'='*60}")

    config = IKConfig(
        timeout=0.2,
        epsilon=1e-5,
        solve_type=SolveType.DISTANCE,
        max_attempts=10,
    )

    solver = create_ik_solver(chain_name, config=config)
    seed = CHAIN_SEEDS[chain_name]
    ee_link = CHAIN_CONFIGS[chain_name].ee_link
    ee_idx = get_ee_link_index(env, ee_link)

    print(f"  DOF: {solver.num_joints}")
    print(f"  base: {solver.base_frame}")
    print(f"  ee:   {solver.ee_frame}")

    # Show home config and draw current EE frame (actuated joints = arm).
    env.set_joint_states(HOME_JOINTS[3:])
    debug_lines = draw_frame_at_link(env, ee_idx, length=0.06, width=2)

    current_pose = solver.fk(seed)

    # Arm-only: keep orientation, small position offset.
    target_pose = SE3Pose(
        position=current_pose.position + np.array([0.05, 0.0, -0.05]),
        rotation=current_pose.rotation,
    )

    wait_key(env, ord("n"), f"[{chain_name}] Home config. Press 'n' to solve IK.")

    result = solver.solve(target_pose, seed=seed)
    print(
        f"  IK: {result.status.value}, "
        f"pos_err={result.position_error:.6f}m, "
        f"ori_err={result.orientation_error:.6f}rad"
    )

    if result.joint_positions is not None:
        body_joints = HOME_JOINTS[3:].copy()
        for i, bi in enumerate(CHAIN_TO_BODY[chain_name]):
            body_joints[bi] = float(result.joint_positions[i])
        env.set_joint_states(body_joints)
        debug_lines += draw_frame_at_link(env, ee_idx, length=0.05, width=2)

    wait_key(env, ord("n"), f"[{chain_name}] Solution shown. Press 'n' for next.")

    for lid in debug_lines:
        env.sim.client.removeUserDebugItem(lid)


def main():
    print("TRAC-IK Solver Example")
    print("=" * 60)

    env = PyBulletEnv(wheelchair_robot_config, visualize=True)

    for chain_name in ["arm"]:
        try:
            test_chain(env, chain_name)
        except Exception as e:
            print(f"  ERROR on {chain_name}: {e}")
            import traceback

            traceback.print_exc()

    wait_key(env, ord("q"), "All chains done. Press 'q' to quit.")
    print("\nDone.")


if __name__ == "__main__":
    main()
