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
python "$ROOT/third_party/foam/scripts/generate_sphere_urdf.py" \
    "$INPUT_URDF" \
    --output "$RES/wheelchair_spherized.urdf" \
    --database "$ROOT/third_party/foam/sphere_database.json" \
    --method medial \
    --verify False \
    "$@"

# Sync the spherized model into the shipped package resources (the runtime ships
# it for reference / regeneration; cricket reads the top-level copy directly).
PKG_RES="$ROOT/wheelchair_planning/resources/robot/wheelchair"
if [ -d "$PKG_RES" ]; then
    cp -f "$RES/wheelchair_spherized.urdf" "$PKG_RES/wheelchair_spherized.urdf"
    echo "Synced wheelchair_spherized.urdf to package resources."
fi

# The decomposed URDF and its per-piece STLs are purely intermediate inputs to
# foam. The spherized output embeds absolute sphere positions and no longer
# references those meshes, so clean them up.
rm -f "$RES/wheelchair_base_decomposed.urdf"
rm -rf "$RES/meshes/decomposed"
echo "Cleaned intermediate decomposition artefacts."
