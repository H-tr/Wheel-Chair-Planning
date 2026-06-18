#!/usr/bin/env bash
# Convex-decompose the wheelchair collision meshes with CoACD so downstream
# foam spherization sees (approximately) convex input per piece.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
ROOT="$(dirname "$SCRIPT_DIR")"

# The wheelchair chassis (base_link / base_fixed.stl) is a large concave mesh:
# spherizing it directly would fit huge spheres that swallow the arm's
# work-volume and block valid solutions. Decompose only the chassis — coarsely —
# so foam fits a tight set of spheres to the platform footprint. The slender
# xArm7 links keep their raw meshes (a handful of spheres each is already
# tight). Everything else is left untouched.
python -u "$SCRIPT_DIR/decompose_meshes.py" \
    --input   "$ROOT/resources/robot/wheelchair/wheelchair_base_simple.urdf" \
    --output  "$ROOT/resources/robot/wheelchair/wheelchair_base_decomposed.urdf" \
    --parts-dir "$ROOT/resources/robot/wheelchair/meshes/decomposed" \
    --threshold 0.3 \
    --max-convex-hull 8 \
    --include 'base_link' \
    "$@"
