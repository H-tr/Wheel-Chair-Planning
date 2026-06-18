"""TRAC-IK stress test with PyBullet visualization.

Same target poses as constrained_vis.py but using the unconstrained
TRAC-IK backend — no camera stabilization, no collision avoidance.

Controls:
    n — advance to next target
    q — quit

Usage:
    python examples/ik/trac_ik_vis.py
"""

import time

import numpy as np
import pybullet as pb
from scipy.spatial.transform import Rotation

from wheelchair_planning.envs.pybullet_env import PyBulletEnv
from wheelchair_planning.kinematics import create_ik_solver
from wheelchair_planning.types import IKConfig, SE3Pose, SolveType
from wheelchair_planning.wheelchair import (
    CHAIN_CONFIGS,
    HOME_JOINTS,
    JOINT_GROUPS,
    wheelchair_robot_config,
)

G = JOINT_GROUPS
SEED = HOME_JOINTS[G["arm"]]
# The arm chain solution (7 DOF) maps 1:1 onto the actuated joints (joint1..7),
# which are HOME_JOINTS[3:].
CHAIN_TO_BODY = list(range(0, 7))


def get_ee_link_index(env, link_name):
    client = env.sim.client
    for i in range(client.getNumJoints(env.sim.skel_id)):
        info = client.getJointInfo(env.sim.skel_id, i)
        if info[12].decode("utf-8") == link_name:
            return i
    return -1


def draw_frame(env, pos, rot, length=0.08, width=3):
    client = env.sim.client
    ids = []
    for axis_idx, color in enumerate([[1, 0, 0], [0, 1, 0], [0, 0, 1]]):
        axis = np.zeros(3)
        axis[axis_idx] = length
        end = (pos + rot @ axis).tolist()
        ids.append(client.addUserDebugLine(pos.tolist(), end, color, lineWidth=width))
    return ids


def draw_frame_at_link(env, link_index, length=0.08, width=3):
    client = env.sim.client
    state = client.getLinkState(env.sim.skel_id, link_index)
    pos = np.array(state[0])
    rot = np.array(client.getMatrixFromQuaternion(state[1])).reshape(3, 3)
    return draw_frame(env, pos, rot, length, width)


def apply_solution(env, joint_positions):
    body = HOME_JOINTS[3:].copy()
    for i, bi in enumerate(CHAIN_TO_BODY):
        body[bi] = float(joint_positions[i])
    env.set_joint_states(body)


def wait_key(env, key, msg):
    client = env.sim.client
    tid = client.addUserDebugText(
        msg, [0, 0, 1.5], textColorRGB=[0, 0, 0], textSize=1.5
    )
    print(msg)
    while True:
        keys = client.getKeyboardEvents()
        if key in keys and keys[key] & pb.KEY_WAS_TRIGGERED:
            break
        time.sleep(0.01)
    client.removeUserDebugItem(tid)


def clear(env, ids):
    for lid in ids:
        env.sim.client.removeUserDebugItem(lid)


def rx(d):
    return Rotation.from_euler("x", d, degrees=True).as_matrix()


def ry(d):
    return Rotation.from_euler("y", d, degrees=True).as_matrix()


def rz(d):
    return Rotation.from_euler("z", d, degrees=True).as_matrix()


def build_targets(home_pose):
    p, R = home_pose.position, home_pose.rotation
    return [
        ("Front reach (+20cm x)", SE3Pose(p + [0.20, 0.0, 0.0], R)),
        ("High reach (+20cm z)", SE3Pose(p + [0.05, 0.0, 0.20], R)),
        ("Low reach (-20cm z)", SE3Pose(p + [0.10, 0.0, -0.20], rx(45) @ R)),
        ("Side reach (+20cm y)", SE3Pose(p + [0.0, 0.20, 0.0], R)),
        ("Cross-body (-15cm y)", SE3Pose(p + [0.10, -0.15, 0.0], R)),
        ("High front (+15x, +20z)", SE3Pose(p + [0.15, 0.0, 0.20], ry(-20) @ R)),
        (
            "Wrist rotation (45deg Z + 30deg X)",
            SE3Pose(p + [0.10, 0.0, 0.0], rz(45) @ rx(30) @ R),
        ),
    ]


def main():
    print("TRAC-IK Stress Test — PyBullet Visualization")
    print("=" * 60)

    env = PyBulletEnv(wheelchair_robot_config, visualize=True)
    ee_link = CHAIN_CONFIGS["arm"].ee_link
    ee_idx = get_ee_link_index(env, ee_link)

    config = IKConfig(solve_type=SolveType.DISTANCE, max_attempts=20)
    solver = create_ik_solver("arm", config=config)

    home_pose = solver.fk(SEED)
    targets = build_targets(home_pose)
    n = len(targets)
    results = []

    for idx, (name, target) in enumerate(targets):
        print(f"\n{'=' * 60}")
        print(f"[{idx+1}/{n}] {name}")
        print(
            f"  target: [{target.position[0]:.3f}, "
            f"{target.position[1]:.3f}, {target.position[2]:.3f}]"
        )

        env.set_joint_states(HOME_JOINTS[3:])
        debug = draw_frame_at_link(env, ee_idx, length=0.06, width=2)
        debug += draw_frame(env, target.position, target.rotation, length=0.10, width=4)

        wait_key(env, ord("n"), f"[{idx+1}/{n}] {name}. Press 'n' to solve.")

        result = solver.solve(target, seed=SEED)
        s = result.status.value
        print(f"  status:    {s}")
        print(f"  pos error: {result.position_error:.6f} m")
        print(f"  ori error: {result.orientation_error:.6f} rad")
        results.append((name, s, result.position_error, result.orientation_error))

        if result.joint_positions is not None:
            apply_solution(env, result.joint_positions)
            debug += draw_frame_at_link(env, ee_idx, length=0.06, width=2)
            achieved = solver.fk(result.joint_positions)
            print(
                f"  achieved:  [{achieved.position[0]:.3f}, "
                f"{achieved.position[1]:.3f}, {achieved.position[2]:.3f}]"
            )

        wait_key(env, ord("n"), f"[{idx+1}/{n}] Done. Press 'n' for next.")
        clear(env, debug)

    print("\n" + "=" * 80)
    print("SUMMARY (TRAC-IK, unconstrained)")
    print("=" * 80)
    print(f"{'Target':<35} {'Status':<15} {'Pos (mm)':<12} {'Ori (deg)':<12}")
    print("-" * 74)
    ok = 0
    for name, status, pe, oe in results:
        flag = "" if status == "success" else " <--"
        print(f"{name:<35} {status:<15} {pe*1000:<12.2f} {np.rad2deg(oe):<12.2f}{flag}")
        if status == "success":
            ok += 1
    print("-" * 74)
    print(f"Success: {ok}/{n}")

    wait_key(env, ord("q"), "All targets done. Press 'q' to quit.")
    print("\nDone.")


if __name__ == "__main__":
    main()
