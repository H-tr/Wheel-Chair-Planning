#!/usr/bin/env bash
# Generate the wheelchair FK + collision-checking C++ header via cricket.
# All inputs (urdf/srdf) and outputs live inside the project; third_party
# submodules are only read from, never written.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
ROOT="$(dirname "$SCRIPT_DIR")"

CRICKET_BIN="$ROOT/third_party/cricket/build/fkcc_gen"
CONFIG="$ROOT/ext/ompl_vamp/robot/wheelchair.json"
OUTDIR="$ROOT/ext/ompl_vamp/robot"

if [ ! -x "$CRICKET_BIN" ]; then
    echo "[generate_fk] Building cricket first..." >&2
    cmake --build "$ROOT/third_party/cricket/build"
fi

# fkcc_gen writes its output into CWD using the config's "output" filename
# ("wheelchair.hh"). Run from the target dir so the header lands there directly.
cd "$OUTDIR"
"$CRICKET_BIN" "$CONFIG"

echo "[generate_fk] Wrote $OUTDIR/wheelchair.hh"
