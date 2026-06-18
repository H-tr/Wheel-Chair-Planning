#!/usr/bin/env bash
# Idempotent build-time fixups for pinned submodules, mirroring the carried
# OMPL patch in cmake/patches/ (applied to the working tree; the submodule
# commit pin is unchanged). Safe to re-run.
#
#   cricket: fkcc_gen.cc uses fmt::format but includes only <fmt/core.h>.
#            In fmt >= 11 that symbol moved to <fmt/format.h>, so the pinned
#            commit no longer compiles against the conda fmt 12 toolchain.
#            Rewrite the include.
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"

patched=0
while IFS= read -r f; do
    if grep -q '#include <fmt/core.h>' "$f"; then
        sed -i 's|#include <fmt/core.h>|#include <fmt/format.h>|' "$f"
        echo "[patch_submodules] cricket: fmt/core.h -> fmt/format.h in ${f#$ROOT/}"
        patched=$((patched + 1))
    fi
done < <(grep -rl 'fmt::format' "$ROOT/third_party/cricket/src" 2>/dev/null || true)

echo "[patch_submodules] done (${patched} file(s) patched)"
