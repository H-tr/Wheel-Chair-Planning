#!/usr/bin/env python3
"""Build pipeline: convert the raw wheelchair+xArm7 URDF into a
planning-ready robot description.

All paths are provided via CLI arguments — no hardcoded project paths.

Stages:
  1. Preprocess URDF  (rename robot, flatten mesh paths, freeze gripper + wheel joints)
  2. Generate simple URDF  (strip sensor/decorative links)
  3. Generate base URDFs  (add a 3-DOF planar virtual base)
  4. Generate SRDF  (groups, end-effector, collision disables)
  5. Copy meshes  (recursively flatten the multi-folder mesh tree)
  6. Repair collision meshes  (optional, requires foam)
  7. Distribute to the shipped package resources

The robot is a wheelchair platform (modeled as a 3-DOF planar base) carrying a
7-DOF UFACTORY xArm7. The parallel gripper (6 joints) and the four wheels are
frozen to ``fixed`` so they contribute no planning DOF. After adding the planar
base, the planning model has 10 DOF: [x, y, theta, joint1..joint7].
"""

from __future__ import annotations

import argparse
import copy
import math
import shutil
import xml.etree.ElementTree as ET
from pathlib import Path

ROBOT_NAME = "wheelchair"

# Revolute/continuous joints that become ``fixed`` for planning: the xArm
# parallel gripper (1 driver + 5 mimic) and the four wheelchair wheels. Base
# mobility is captured by the planar virtual joints added in stage 3 instead.
FREEZE_JOINTS = {
    # gripper
    "drive_joint",
    "left_finger_joint",
    "left_inner_knuckle_joint",
    "right_outer_knuckle_joint",
    "right_finger_joint",
    "right_inner_knuckle_joint",
    # wheels
    "FR_joint",
    "FL_joint",
    "BR_joint",
    "BL_joint",
}

# Links to strip for the simplified URDF (sensor frames and decorative parts).
# All are attached via fixed joints and irrelevant for collision planning.
STRIP_LINK_PREFIXES = (
    "camera_",  # RealSense optical/frame links
    "laser_",  # lidar boxes
    "lip_",  # front ramp/lip
)

# The synthetic root link inserted above the chassis to host the planar base.
ZERO_POINT_LINK = "Link_Zero_Point"
# The original root link of the raw URDF (the wheelchair chassis).
CHASSIS_LINK = "base_link"

# ── Helpers ──────────────────────────────────────────────────────────────────


def _indent(elem: ET.Element, level: int = 0) -> None:
    """Add pretty-print indentation (for Python < 3.9 compat)."""
    indent = "\n" + "  " * level
    if len(elem):
        if not elem.text or not elem.text.strip():
            elem.text = indent + "  "
        if not elem.tail or not elem.tail.strip():
            elem.tail = indent
        for child in elem:
            _indent(child, level + 1)
        if not child.tail or not child.tail.strip():
            child.tail = indent
    else:
        if level and (not elem.tail or not elem.tail.strip()):
            elem.tail = indent


def _write_xml(tree: ET.ElementTree, path: Path) -> None:
    _indent(tree.getroot())
    tree.write(str(path), encoding="utf-8", xml_declaration=True)
    print(f"  wrote {path}")


# ── Stage 1: Preprocess URDF ────────────────────────────────────────────────


def preprocess_urdf(urdf_path: Path) -> ET.ElementTree:
    """Parse raw URDF, rename robot, flatten mesh paths, freeze fixed-for-planning joints."""
    tree = ET.parse(str(urdf_path))
    root = tree.getroot()

    # Rename robot
    root.set("name", ROBOT_NAME)

    # Flatten mesh paths: package://wheelchair_xarm/meshes/<sub>/X.stl
    #                  -> package://meshes/X.stl
    for mesh in root.iter("mesh"):
        fn = mesh.get("filename", "")
        if "meshes/" in fn:
            basename = fn.split("/")[-1]
            mesh.set("filename", f"package://meshes/{basename}")

    # Freeze the gripper and wheel joints: convert to fixed and drop the
    # motion-related children (limit/axis/mimic/dynamics).
    for joint in root.findall("joint"):
        if joint.get("name", "") in FREEZE_JOINTS and joint.get("type") != "fixed":
            joint.set("type", "fixed")
            for tag in ("limit", "axis", "mimic", "dynamics"):
                for el in joint.findall(tag):
                    joint.remove(el)

    return tree


# ── Stage 2: Generate simple URDF ───────────────────────────────────────────


def _should_strip(name: str) -> bool:
    return any(name.startswith(p) for p in STRIP_LINK_PREFIXES)


def generate_simple_urdf(preprocessed: ET.ElementTree) -> ET.ElementTree:
    """Strip sensor and decorative links for faster collision checking."""
    tree = copy.deepcopy(preprocessed)
    root = tree.getroot()

    links_to_remove = {
        link.get("name")
        for link in root.findall("link")
        if _should_strip(link.get("name", ""))
    }

    for link in list(root.findall("link")):
        if link.get("name") in links_to_remove:
            root.remove(link)

    for joint in list(root.findall("joint")):
        child_el = joint.find("child")
        if child_el is not None and child_el.get("link") in links_to_remove:
            root.remove(joint)

    print(f"  stripped {len(links_to_remove)} non-planning links")
    return tree


# ── Stage 3: Generate base URDF ─────────────────────────────────────────────

_VIRTUAL_LINK_TEMPLATE = """
<link name="{name}">
  <inertial>
    <origin xyz="0 0 0" rpy="0 0 0"/>
    <mass value="0.001"/>
    <inertia ixx="1e-9" ixy="0" ixz="0" iyy="1e-9" iyz="0" izz="1e-9"/>
  </inertial>
</link>
"""


def _make_joint(
    name: str,
    jtype: str,
    parent: str,
    child: str,
    *,
    axis: str | None = None,
    lower: float | None = None,
    upper: float | None = None,
    origin_xyz: str = "0 0 0",
    origin_rpy: str = "0 0 0",
) -> ET.Element:
    joint = ET.Element("joint", name=name, type=jtype)
    ET.SubElement(joint, "origin", xyz=origin_xyz, rpy=origin_rpy)
    ET.SubElement(joint, "parent", link=parent)
    ET.SubElement(joint, "child", link=child)
    if axis is not None:
        ET.SubElement(joint, "axis", xyz=axis)
    if lower is not None and upper is not None:
        ET.SubElement(
            joint,
            "limit",
            lower=str(lower),
            upper=str(upper),
            effort="100",
            velocity="1.0",
        )
    return joint


def generate_base_urdf(preprocessed: ET.ElementTree) -> ET.ElementTree:
    """Insert a 3-DOF planar virtual base (X, Y, Theta) above the chassis.

    The raw URDF roots at the chassis (``base_link``). We add a synthetic root
    ``Link_Zero_Point`` and chain it to the chassis through three virtual joints
    so the mobile base can be planned in the ground plane.
    """
    tree = copy.deepcopy(preprocessed)
    root = tree.getroot()

    # Insert the synthetic root + virtual links.
    for name in [ZERO_POINT_LINK, "Link_Virtual_X", "Link_Virtual_Y", "Link_Virtual_Theta"]:
        root.append(ET.fromstring(_VIRTUAL_LINK_TEMPLATE.format(name=name)))

    pi = math.pi
    root.append(
        _make_joint(
            "Joint_Virtual_X",
            "prismatic",
            ZERO_POINT_LINK,
            "Link_Virtual_X",
            axis="1 0 0",
            lower=-10,
            upper=10,
        )
    )
    root.append(
        _make_joint(
            "Joint_Virtual_Y",
            "prismatic",
            "Link_Virtual_X",
            "Link_Virtual_Y",
            axis="0 1 0",
            lower=-10,
            upper=10,
        )
    )
    root.append(
        _make_joint(
            "Joint_Virtual_Theta",
            "revolute",
            "Link_Virtual_Y",
            "Link_Virtual_Theta",
            axis="0 0 1",
            lower=-pi,
            upper=pi,
        )
    )
    # Rigidly attach the chassis to the planar base at the origin.
    root.append(
        _make_joint(
            "Joint_Base_Fixed",
            "fixed",
            "Link_Virtual_Theta",
            CHASSIS_LINK,
        )
    )

    return tree


# ── Stage 4: Generate SRDF ──────────────────────────────────────────────────

# Kinematic chains / link groupings used to author the collision disables.
_ARM_CHAIN = [
    "link_base",
    "link1",
    "link2",
    "link3",
    "link4",
    "link5",
    "link6",
    "link7",
    "link_eef",
    "xarm_gripper_base_link",
]
_GRIPPER_LINKS = [
    "left_outer_knuckle",
    "left_finger",
    "left_inner_knuckle",
    "right_outer_knuckle",
    "right_finger",
    "right_inner_knuckle",
]
_WHEELS = ["front_right_1", "front_left_1", "back_right_1", "back_left_1"]
_ARM_JOINTS = [f"joint{i}" for i in range(1, 8)]


def generate_srdf(preprocessed: ET.ElementTree) -> ET.ElementTree:
    """Generate SRDF with the arm group, an end-effector, and collision disables."""
    root_urdf = preprocessed.getroot()
    all_links = {link.get("name") for link in root_urdf.findall("link")}

    robot = ET.Element("robot", name=ROBOT_NAME)
    tree = ET.ElementTree(robot)

    # Groups
    _add_group(robot, "arm", "link_base", "link_tcp")

    # Group states
    _add_group_state(robot, "Home", "arm", _ARM_JOINTS)

    # End effector
    ET.SubElement(
        robot,
        "end_effector",
        name="gripper",
        parent_link="link_tcp",
        group="arm",
    )

    # Disable collisions
    _generate_collision_disables(robot, all_links)

    return tree


def _add_group(robot: ET.Element, name: str, base_link: str, tip_link: str) -> None:
    group = ET.SubElement(robot, "group", name=name)
    ET.SubElement(group, "chain", base_link=base_link, tip_link=tip_link)


def _add_group_state(
    robot: ET.Element, state_name: str, group_name: str, joints: list[str]
) -> None:
    gs = ET.SubElement(robot, "group_state", name=state_name, group=group_name)
    for j in joints:
        ET.SubElement(gs, "joint", name=j, value="0")


def _generate_collision_disables(robot: ET.Element, all_links: set[str]) -> None:
    """Author disable_collisions pairs for the spherized model."""
    pairs: set[tuple[str, str]] = set()

    def add(a: str, b: str) -> None:
        if a in all_links and b in all_links and a != b:
            pairs.add((min(a, b), max(a, b)))

    # All intra-arm pairs: a serial spherized arm reports constant false
    # positives between its own (especially adjacent) links.
    for i in range(len(_ARM_CHAIN)):
        for j in range(i + 1, len(_ARM_CHAIN)):
            add(_ARM_CHAIN[i], _ARM_CHAIN[j])

    # Gripper sub-links vs each other and vs the wrist links they sit on.
    _gripper_neighbours = ["link6", "link7", "link_eef", "xarm_gripper_base_link"]
    for i in range(len(_GRIPPER_LINKS)):
        for j in range(i + 1, len(_GRIPPER_LINKS)):
            add(_GRIPPER_LINKS[i], _GRIPPER_LINKS[j])
        for nb in _gripper_neighbours:
            add(_GRIPPER_LINKS[i], nb)

    # Proximal arm vs chassis: link_base is bolted onto the chassis and link1
    # sits in the mount region, so their spheres overlap the chassis envelope.
    # Distal arm links stay collision-checked against the chassis (a swing into
    # the platform is a real fault).
    for proximal in ("link_base", "link1"):
        add(proximal, CHASSIS_LINK)

    # Wheels are frozen, low, static geometry: disable them against the chassis,
    # each other, and the arm mount so spherization artefacts never trip.
    for w in _WHEELS:
        add(w, CHASSIS_LINK)
        add(w, "link_base")
        for w2 in _WHEELS:
            add(w, w2)

    for a, b in sorted(pairs):
        ET.SubElement(robot, "disable_collisions", link1=a, link2=b, reason="Default")

    print(f"  wrote {len(pairs)} disable_collisions pairs")


# ── Stage 5: Copy meshes ────────────────────────────────────────────────────


def copy_meshes(urdf_tree: ET.ElementTree, mesh_src_root: Path, out_dir: Path) -> None:
    """Copy referenced STL files, flattening the multi-folder source tree.

    The raw description scatters meshes across ``meshes/<sub>/...`` folders; the
    preprocessed URDF references them as ``package://meshes/<basename>``. Resolve
    each basename by searching the source tree recursively.
    """
    mesh_out = out_dir / "meshes"
    mesh_out.mkdir(parents=True, exist_ok=True)

    referenced = set()
    for mesh in urdf_tree.getroot().iter("mesh"):
        fn = mesh.get("filename", "")
        if fn.startswith("package://meshes/"):
            referenced.add(fn.split("/")[-1])

    # Build a basename -> path index of the source tree (case-insensitive on ext).
    index: dict[str, list[Path]] = {}
    for p in mesh_src_root.rglob("*"):
        if p.is_file():
            index.setdefault(p.name, []).append(p)

    copied = 0
    for basename in sorted(referenced):
        matches = index.get(basename, [])
        if not matches:
            print(f"  WARNING: mesh not found: {basename}")
            continue
        if len({m.stat().st_size for m in matches}) > 1:
            print(
                f"  WARNING: basename collision for {basename} "
                f"({len(matches)} distinct files); using {matches[0]}"
            )
        shutil.copy2(str(matches[0]), str(mesh_out / basename))
        copied += 1

    print(f"  copied {copied}/{len(referenced)} mesh files")


# ── Stage 6: Repair collision meshes ────────────────────────────────────────


def repair_collision_meshes(simple_urdf_path: Path, method: str = "medial") -> None:
    """Repair meshes so they pass sphere-tree validation (requires foam)."""
    import trimesh
    from foam import smooth_manifold
    from foam.external import check_valid_for_spherization

    tree = ET.parse(str(simple_urdf_path))
    urdf_dir = simple_urdf_path.parent

    mesh_paths: set[Path] = set()
    for mesh_el in tree.getroot().iter("mesh"):
        fn = mesh_el.get("filename", "")
        if fn:
            mesh_paths.add(urdf_dir / fn.replace("package://", ""))

    repaired_count = 0
    for mesh_path in sorted(mesh_paths):
        if not mesh_path.exists():
            continue

        mesh = trimesh.load(str(mesh_path), process=False)

        if check_valid_for_spherization(method, mesh):
            print(f"  OK    {mesh_path.name}")
            continue

        fixed = None
        strategy = ""
        try:
            repaired = smooth_manifold(mesh)
            if check_valid_for_spherization(method, repaired):
                fixed = repaired
                strategy = "smooth manifold"
        except Exception:
            pass

        if fixed is None:
            fixed = mesh.convex_hull
            strategy = "convex hull"

        fixed.export(str(mesh_path))
        repaired_count += 1
        print(
            f"  REPAIRED  {mesh_path.name}  ({strategy}, {len(fixed.vertices)} verts)"
        )

    print(f"  {repaired_count} mesh(es) repaired out of {len(mesh_paths)} total")


# ── Stage 7: Distribute ─────────────────────────────────────────────────────


def distribute(out_dir: Path, dest_dirs: list[Path]) -> None:
    """Copy generated files into the shipped package resource directories."""
    files = [
        "wheelchair.urdf",
        "wheelchair_simple.urdf",
        "wheelchair_base.urdf",
        "wheelchair_base_simple.urdf",
        "wheelchair_viz.urdf",
        "wheelchair.srdf",
        "wheelchair_spherized.urdf",
    ]
    for dest_dir in dest_dirs:
        dest_dir.mkdir(parents=True, exist_ok=True)
        for f in files:
            src = out_dir / f
            if src.exists():
                shutil.copy2(str(src), str(dest_dir / f))
        for subdir in ("meshes", "viz_meshes"):
            src_sub = out_dir / subdir
            dst_sub = dest_dir / subdir
            if src_sub.exists():
                if dst_sub.exists():
                    shutil.rmtree(str(dst_sub))
                shutil.copytree(str(src_sub), str(dst_sub))
        print(f"  distributed to {dest_dir}")


# ── Main ─────────────────────────────────────────────────────────────────────


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--urdf", type=Path, required=True, help="Path to raw input URDF")
    parser.add_argument(
        "--mesh-dir",
        type=Path,
        required=True,
        help="Root directory of source mesh files (searched recursively)",
    )
    parser.add_argument(
        "--output-dir", type=Path, required=True, help="Output directory for generated files"
    )
    parser.add_argument(
        "--distribute-to",
        type=Path,
        nargs="*",
        default=[],
        help="Package resource directories to copy results into",
    )
    parser.add_argument(
        "--repair-meshes",
        action="store_true",
        help="Repair collision meshes for sphere-tree construction (requires foam)",
    )
    args = parser.parse_args()

    urdf_path: Path = args.urdf
    mesh_src_dir: Path = args.mesh_dir
    out_dir: Path = args.output_dir
    dist_dirs: list[Path] = args.distribute_to

    if not urdf_path.exists():
        raise FileNotFoundError(f"URDF not found: {urdf_path}")
    if not mesh_src_dir.exists():
        raise FileNotFoundError(f"Mesh directory not found: {mesh_src_dir}")

    print(f"Input URDF:  {urdf_path}")
    print(f"Mesh source: {mesh_src_dir}")
    print(f"Output dir:  {out_dir}")
    out_dir.mkdir(parents=True, exist_ok=True)

    total_stages = 7 if args.repair_meshes else 6

    print(f"\n[1/{total_stages}] Preprocessing URDF...")
    preprocessed = preprocess_urdf(urdf_path)
    _write_xml(preprocessed, out_dir / "wheelchair.urdf")
    # The wheelchair shares visual and collision meshes, so the visualization
    # URDF is identical to the planning URDF.
    _write_xml(copy.deepcopy(preprocessed), out_dir / "wheelchair_viz.urdf")

    print(f"\n[2/{total_stages}] Generating simple URDF...")
    simple_tree = generate_simple_urdf(preprocessed)
    _write_xml(simple_tree, out_dir / "wheelchair_simple.urdf")

    print(f"\n[3/{total_stages}] Generating base URDFs...")
    base_tree = generate_base_urdf(preprocessed)
    _write_xml(base_tree, out_dir / "wheelchair_base.urdf")
    base_simple_tree = generate_base_urdf(simple_tree)
    _write_xml(base_simple_tree, out_dir / "wheelchair_base_simple.urdf")

    srdf_path = out_dir / "wheelchair.srdf"
    if srdf_path.exists():
        print(
            f"\n[4/{total_stages}] SRDF already exists, keeping existing (hand-edited) version."
        )
    else:
        print(f"\n[4/{total_stages}] Generating SRDF...")
        srdf_tree = generate_srdf(base_simple_tree)
        _write_xml(srdf_tree, srdf_path)

    print(f"\n[5/{total_stages}] Copying meshes...")
    copy_meshes(preprocessed, mesh_src_dir, out_dir)

    if args.repair_meshes:
        print(f"\n[6/{total_stages}] Repairing collision meshes...")
        repair_collision_meshes(out_dir / "wheelchair_base_simple.urdf")

    last = total_stages
    if dist_dirs:
        print(f"\n[{last}/{total_stages}] Distributing to package resources...")
        distribute(out_dir, dist_dirs)
    else:
        print(f"\n[{last}/{total_stages}] No distribute targets specified, skipping.")

    # Verify planning joint count (arm only; the planar base adds 3 more in the
    # base URDFs for a 10-DOF whole-body model).
    rev_joints = [
        j.get("name")
        for j in preprocessed.findall("joint")
        if j.get("type") in ("revolute", "prismatic", "continuous")
        and j.get("name") not in FREEZE_JOINTS
    ]
    print(f"\nArm planning joints ({len(rev_joints)} DOF): {rev_joints}")
    assert len(rev_joints) == 7, f"Expected 7 arm planning DOF, got {len(rev_joints)}"
    print("Whole-body model = 3 (planar base) + 7 (arm) = 10 DOF")
    print("\nDone!")


if __name__ == "__main__":
    main()
