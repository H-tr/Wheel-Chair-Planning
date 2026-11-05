"""The compiled planner must check the same robot the shipped URDF describes.

``_ompl_vamp`` bakes the spheres from ``ext/ompl_vamp/robot/wheelchair.hh``
in at build time. When the header is regenerated but the extension isn't
rebuilt, the planner silently keeps checking the old sphere set: a build
from before ``lip_1`` was spherized drove the lip 7 cm into a counter while
reporting zero collisions. Rebuild with
``pip install --no-build-isolation --no-deps -e .`` when this fails.
"""

from __future__ import annotations

import xml.etree.ElementTree as ET
from pathlib import Path

import numpy as np
import pytest

import wheelchair_planning
from wheelchair_planning.planning import create_planner
from wheelchair_planning.types import PlannerConfig
from wheelchair_planning.wheelchair import HOME_JOINTS, wheelchair_robot_config

URDF = (
    Path(wheelchair_planning.__file__).parent
    / "resources/robot/wheelchair/wheelchair_spherized.urdf"
)


def _sphere_radii() -> list[float]:
    return [float(s.get("radius")) for s in ET.parse(URDF).getroot().iter("sphere")]


def test_compiled_sphere_radii_match_urdf():
    from wheelchair_planning._ompl_vamp import OmplVampPlanner

    radii = _sphere_radii()
    assert np.allclose(
        OmplVampPlanner().min_max_radii(), (min(radii), max(radii)), atol=1e-6
    )


@pytest.mark.parametrize("link", ["lip_1", "base_link_collision_0", "link_eef"])
def test_obstacle_on_link_is_a_collision(link):
    """A point at a link's first sphere centre must invalidate the home pose."""
    pin = pytest.importorskip("pinocchio")
    model = pin.buildModelFromUrdf(str(URDF))
    data = model.createData()
    q = pin.neutral(model)
    for name, value in zip(wheelchair_robot_config.joint_names, HOME_JOINTS):
        q[model.joints[model.getJointId(name)].idx_q] = value
    pin.framesForwardKinematics(model, data, q)

    link_el = next(
        el for el in ET.parse(URDF).getroot().iter("link") if el.get("name") == link
    )
    origin = link_el.find("collision/origin")
    offset = (
        np.array(origin.get("xyz").split(), dtype=float)
        if origin is not None
        else np.zeros(3)
    )
    point = data.oMf[model.getFrameId(link)].act(offset)

    planner = create_planner(
        "wheelchair_whole_body",
        config=PlannerConfig(point_radius=0.001),
        pointcloud=point[None].astype(np.float32),
    )
    assert not planner.validate(
        HOME_JOINTS
    ), f"{link} is missing from the compiled collision model"
