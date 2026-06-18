"""The wheelchair + xArm7 robot's bundled description.

This module collects every concrete value that describes the one robot
this project ships: joint groupings, the home pose, the URDF chains
TRAC-IK and Pinocchio operate on, the VAMP planning subgroups, and the
top-level ``RobotConfig`` instance.

The robot is a mobile manipulator: a wheelchair platform (modeled as a
3-DOF planar virtual base) carrying a 7-DOF UFACTORY xArm7 arm, for a
total of 10 planning DOF. The parallel gripper and the four wheels are
frozen to ``fixed`` in the planning URDFs, so they contribute no DOF.

The dataclass *types* themselves live in
:mod:`wheelchair_planning.types.robot` — this file holds *values* of
those types.
"""

from __future__ import annotations

import os

import numpy as np

from wheelchair_planning.types.robot import CameraConfig, ChainConfig, RobotConfig

_PKG_ROOT = os.path.dirname(os.path.abspath(__file__))
_RESOURCES_DIR = os.path.join(_PKG_ROOT, "resources", "robot", "wheelchair")

# Atomic joint groups — indices into the full 10-DOF configuration array.
# Order must match VAMP's URDF tree traversal: base (virtual planar joints)
# first, then the xArm7 arm.
JOINT_GROUPS = {
    "base": slice(0, 3),  # Virtual_X, Virtual_Y, Virtual_Theta
    "arm": slice(3, 10),  # joint1 → joint7 (7 DOF)
}

CHAIN_CONFIGS: dict[str, ChainConfig] = {
    # Fixed-base arm chain: from the xArm7 mounting flange to the tool TCP.
    "arm": ChainConfig(
        base_link="link_base",
        ee_link="link_tcp",
        num_joints=7,
        urdf_path=os.path.join(_RESOURCES_DIR, "wheelchair.urdf"),
    ),
    # Whole-body chain: planar base (3) + arm (7) from the synthetic root
    # ``Link_Zero_Point`` (added by the base-augmented URDF) to the tool TCP.
    "whole_body": ChainConfig(
        base_link="Link_Zero_Point",
        ee_link="link_tcp",
        num_joints=10,
        urdf_path=os.path.join(_RESOURCES_DIR, "wheelchair_base.urdf"),
    ),
}

VIZ_URDF_PATH = os.path.join(_RESOURCES_DIR, "wheelchair_viz.urdf")

# Conservative placeholder limits for time-optimal trajectory generation.
# The base translation joints (x, y) and yaw move slowly; the arm joints can
# be faster. Override per-joint when real robot specs are wired in.
MAX_VELOCITY = np.array(
    [
        0.3,  # base x   (m/s)
        0.3,  # base y   (m/s)
        0.5,  # base yaw (rad/s)
        0.8,  # joint1
        0.8,  # joint2
        0.8,  # joint3
        0.8,  # joint4
        1.0,  # joint5
        1.0,  # joint6
        1.0,  # joint7
    ],
    dtype=np.float64,
)
MAX_ACCELERATION = np.array(
    [
        0.5,  # base x
        0.5,  # base y
        0.8,  # base yaw
        1.0,  # joint1
        1.0,  # joint2
        1.0,  # joint3
        1.0,  # joint4
        1.2,  # joint5
        1.2,  # joint6
        1.2,  # joint7
    ],
    dtype=np.float64,
)

wheelchair_robot_config = RobotConfig(
    urdf_path=os.path.join(_RESOURCES_DIR, "wheelchair.urdf"),
    joint_names=[
        # [0:3] base (virtual planar joints)
        "Joint_Virtual_X",
        "Joint_Virtual_Y",
        "Joint_Virtual_Theta",
        # [3:10] xArm7 arm
        "joint1",
        "joint2",
        "joint3",
        "joint4",
        "joint5",
        "joint6",
        "joint7",
    ],
    camera=CameraConfig(
        # RealSense D435i color optical frame, mounted on the arm wrist
        # (``link_eef``).
        link_name="camera_color_optical_frame",
        width=640,
        height=480,
        fov=60.0,
        near=0.1,
        far=10.0,
    ),
    max_velocity=MAX_VELOCITY,
    max_acceleration=MAX_ACCELERATION,
)

# VAMP subgroup robot names for planning.
# Each maps to a separate VAMP module with the full robot body (all links and
# collision geometry) but only the listed joints movable.
_ARM_JOINTS = [
    "joint1",
    "joint2",
    "joint3",
    "joint4",
    "joint5",
    "joint6",
    "joint7",
]
_BASE_JOINTS = ["Joint_Virtual_X", "Joint_Virtual_Y", "Joint_Virtual_Theta"]
PLANNING_SUBGROUPS = {
    # xArm7 arm only (7 DOF) — base held fixed.
    "wheelchair_arm": {"dof": 7, "joints": _ARM_JOINTS},
    # Mobile base in the ground plane (3 DOF: x, y, yaw).
    "wheelchair_base": {"dof": 3, "joints": _BASE_JOINTS},
    # Whole body: planar base + arm (10 DOF). Equivalent to the full-body
    # ``"wheelchair"`` planner.
    "wheelchair_whole_body": {"dof": 10, "joints": _BASE_JOINTS + _ARM_JOINTS},
}

# Neutral home pose. Base at the origin; xArm7 in a gently folded, upright
# stance that keeps the end-effector clear of the wheelchair chassis. All
# values respect the URDF joint limits (e.g. joint4 lower bound is -0.192).
# Verify collision-free against the spherized model before relying on it.
HOME_JOINTS = np.array(
    [
        # [0:3] base
        0.0,
        0.0,
        0.0,
        # [3:10] xArm7 arm
        0.0,
        -0.2,
        0.0,
        0.6,
        0.0,
        0.8,
        0.0,
    ]
)
