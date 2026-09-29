#!/usr/bin/env bash
# Spherize the wheelchair collision model using foam.
# All project-specific paths live here; foam's script is the tool.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
ROOT="$(dirname "$SCRIPT_DIR")"

RES="$ROOT/resources/robot/wheelchair"

# Feed the convex-decomposed URDF to foam so each convex piece is spherized
# independently (tight fit). Fall back to the non-decomposed base URDF if the
# decomposed one is missing.
INPUT_URDF="$RES/wheelchair_base_decomposed.urdf"
if [ ! -f "$INPUT_URDF" ]; then
    INPUT_URDF="$RES/wheelchair_base_simple.urdf"
fi

# Use the 'medial' sphere-tree method (tightest fit). The bundled makeTreeMedial
# binary's --verify mesh-validity check raises false positives on CoACD/trimesh
# convex hulls (float-precision "bad faces" from the OBJ round-trip), which makes
# foam's wrapper give up. The inputs are guaranteed-manifold convex hulls, so we
# disable --verify.
#
# Per-link sphere budgets (``--<link> <factor>`` scales foam's per-mesh branch
# heuristic): the arm links get twice the default so their spheres hug the
# cylindrical housings, while the tiny knuckle/finger links get half (the
# default spends 6-8 spheres on each 5 cm part). xarm_gripper_base_link keeps
# the default: its heuristic budget is 1, and 0.5 would round it to 0.
python "$ROOT/third_party/foam/scripts/generate_sphere_urdf.py" \
    "$INPUT_URDF" \
    --output "$RES/wheelchair_spherized.urdf" \
    --database "$ROOT/third_party/foam/sphere_database.json" \
    --method medial \
    --verify False \
    --threads 16 \
    --link_base 2 --link1 2 --link2 2 --link3 2 --link4 2 --link5 2 --link6 2 \
    --left_outer_knuckle 0.5 --right_outer_knuckle 0.5 \
    --left_inner_knuckle 0.5 --right_inner_knuckle 0.5 \
    --left_finger 0.5 --right_finger 0.5 \
    "$@"

# The chassis carries most of the spheres. Spread them over fixed child links
# so each cluster gets its own tight bounding sphere in the generated collision
# checker; otherwise every check tests every chassis sphere against the arm.
# The spheres and the self-collision pairs are unchanged (the SRDF entries of
# base_link are mirrored onto the new links).
python "$SCRIPT_DIR/split_collision_link.py" \
    --urdf "$RES/wheelchair_spherized.urdf" \
    --srdf "$RES/wheelchair.srdf" \
    --link base_link \
    --clusters 12

# Sync the spherized model and SRDF into the shipped package resources (the
# runtime ships them for reference / regeneration; cricket reads the top-level
# copies directly).
PKG_RES="$ROOT/wheelchair_planning/resources/robot/wheelchair"
if [ -d "$PKG_RES" ]; then
    cp -f "$RES/wheelchair_spherized.urdf" "$PKG_RES/wheelchair_spherized.urdf"
    cp -f "$RES/wheelchair.srdf" "$PKG_RES/wheelchair.srdf"
    echo "Synced wheelchair_spherized.urdf and wheelchair.srdf to package resources."
fi

# The decomposed URDF and its per-piece STLs are purely intermediate inputs to
# foam. The spherized output embeds absolute sphere positions and no longer
# references those meshes, so clean them up.
rm -f "$RES/wheelchair_base_decomposed.urdf"
rm -rf "$RES/meshes/decomposed"
echo "Cleaned intermediate decomposition artefacts."
