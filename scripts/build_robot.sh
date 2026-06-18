#!/usr/bin/env bash
# Build planning-ready robot description from the wheelchair + xArm7 URDF.
# All project-specific paths live here; the Python script is a pure tool.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
ROOT="$(dirname "$SCRIPT_DIR")"

# The raw description is vendored as a submodule under assets/. We flatten its
# multi-folder mesh tree into resources/robot/wheelchair/meshes and distribute
# the generated URDFs into the shipped package resources so the runtime
# (TRAC-IK / pinocchio / pybullet) loads them straight from the package.
#
# We intentionally do NOT pass --repair-meshes: CoACD runs downstream and emits
# convex (manifold) pieces for foam. We also do not pass --distribute-to the
# third_party submodules — those stay unmodified header-only dependencies.
python "$SCRIPT_DIR/build_robot_description.py" \
    --urdf "$ROOT/assets/wheelchair_xarm_description/urdf/wheelchair_xarm.urdf" \
    --mesh-dir "$ROOT/assets/wheelchair_xarm_description/meshes" \
    --output-dir "$ROOT/resources/robot/wheelchair" \
    --distribute-to "$ROOT/wheelchair_planning/resources/robot/wheelchair" \
    "$@"
