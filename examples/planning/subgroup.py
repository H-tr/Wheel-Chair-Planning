"""Plan every kinematic subgroup at three different base stances.

Demonstrates that the same subgroup name (e.g. ``wheelchair_arm``) can be
planned around any 10-DOF base configuration the caller passes in — no
stance is baked into the planner name.  The three stances below are
*example data*, not part of the planning API: they move the mobile base
(x, y, yaw) so the arm subgroup plans around different platform poses.

    pixi run python examples/planning/subgroup.py
"""

import numpy as np
from fire import Fire

from wheelchair_planning.envs.pybullet_env import PyBulletEnv
from wheelchair_planning.planning import create_planner
from wheelchair_planning.types import PlannerConfig
from wheelchair_planning.wheelchair import HOME_JOINTS, wheelchair_robot_config

# Joint values for three example base stances. Replace this dict with any
# 10-DOF array (e.g. the live state from your env) to plan around an
# arbitrary pose.
STANCES = {
    "origin": {
        "Joint_Virtual_X": 0.0,
        "Joint_Virtual_Y": 0.0,
        "Joint_Virtual_Theta": 0.0,
    },
    "forward": {
        "Joint_Virtual_X": 0.5,
        "Joint_Virtual_Y": 0.0,
        "Joint_Virtual_Theta": 0.0,
    },
    "turned": {
        "Joint_Virtual_X": 0.3,
        "Joint_Virtual_Y": 0.3,
        "Joint_Virtual_Theta": 0.6,
    },
}

SUBGROUPS = [
    "wheelchair_arm",  # 7 DOF: xArm7
    "wheelchair_base",  # 3 DOF: virtual x, y, yaw
    "wheelchair_whole_body",  # 10 DOF: base + arm
]


def base_with_stance(stance: dict[str, float]) -> np.ndarray:
    base = HOME_JOINTS.copy()
    for joint_name, value in stance.items():
        base[wheelchair_robot_config.joint_names.index(joint_name)] = value
    return base


def plan_and_show(
    env, robot_name: str, base: np.ndarray, config: PlannerConfig, label: str
) -> bool:
    """Plan one subgroup against *base* and animate it interactively.

    Returns ``True`` if the user pressed ``n`` to advance to the next demo,
    ``False`` if the user closed the GUI window.
    """
    planner = create_planner(robot_name, config=config, base_config=base)
    start = planner.extract_config(base)
    goal = planner.sample_valid()

    result = planner.plan(start, goal)
    n_wp = result.path.shape[0] if result.path is not None else 0
    print(f"  [{label}] {result.status.value} — {n_wp} waypoints")

    if result.success and result.path is not None:
        return env.animate_path(planner.embed_path(result.path), next_key="n")
    env.wait_key("n", f"[{label}] no path — press 'n' for next")
    return env.sim.client.isConnected()


def main(planner_name: str = "bitstar", time_limit: float = 0.5):
    """Run the subgroup sweep with the chosen OMPL planner.

    Available planner names (pick one and pass as ``--planner_name``):

        RRT family ........... rrtc / rrtconnect, rrt, rrtstar,
                               informed_rrtstar, rrtsharp, rrtxstatic,
                               strrtstar, lbtrrt, trrt, bitrrt
        Informed trees ....... bitstar, abitstar, aitstar, eitstar, blitstar
        FMT .................. fmt, bfmt
        KPIECE ............... kpiece, bkpiece, lbkpiece
        PRM family ........... prm, prmstar, lazyprm, lazyprmstar,
                               spars, spars2
        Exploration-based .... est, biest, sbl, stride, pdst
    """
    env = PyBulletEnv(wheelchair_robot_config, visualize=True)
    config = PlannerConfig(planner_name=planner_name, time_limit=time_limit)

    for stance_name, stance in STANCES.items():
        base = base_with_stance(stance)
        for robot_name in SUBGROUPS:
            cont = plan_and_show(
                env, robot_name, base, config, f"{robot_name} @ {stance_name}"
            )
            if not cont:
                return

    env.wait_for_close()


if __name__ == "__main__":
    Fire(main)
