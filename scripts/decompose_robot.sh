#!/usr/bin/env bash
# Convex-decompose the wheelchair collision meshes with CoACD so downstream
# foam spherization sees (approximately) convex input per piece.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
ROOT="$(dirname "$SCRIPT_DIR")"

# The wheelchair chassis (base_link / base_fixed.stl) is a large concave mesh:
# spherizing it directly would fit huge spheres that swallow the arm's
# work-volume and block valid solutions. Decompose only the chassis — finely
# (24 compact pieces) — so foam fits tight spheres to each frame member,
# plate and the arm mount instead of to large hulls that span open space.
# The slender xArm7 links keep their raw meshes. Everything else is left
# untouched. The pattern is anchored: a bare 'base_link' would also match
# xarm_gripper_base_link and shatter the gripper into dozens of pieces.
python -u "$SCRIPT_DIR/decompose_meshes.py" \
    --input   "$ROOT/resources/robot/wheelchair/wheelchair_base_simple.urdf" \
    --output  "$ROOT/resources/robot/wheelchair/wheelchair_base_decomposed.urdf" \
    --parts-dir "$ROOT/resources/robot/wheelchair/meshes/decomposed" \
    --threshold 0.1 \
    --max-convex-hull 24 \
    --include '^base_link$' \
    "$@"
