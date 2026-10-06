#!/usr/bin/env bash
# Builds TMalign and qTMclust from source (https://github.com/pylelab/USalign)
# for zevals's RNA structure-search oracle (oracle/structure_search/usalign.yaml).
#
# Unlike FoldSeek/MMseqs2, USalign ships no pre-built static binaries — only
# C++ source, compiled with a single g++ invocation per program (no build
# system beyond that; see USalign's own Makefile). This script downloads a
# source tarball (no `git` dependency) and compiles just the two programs
# zevals actually uses.
#
# Usage:
#   scripts/install_usalign.sh [install_dir] [--force]
#
#   install_dir defaults to ~/.local/bin. The script is idempotent — it
#   skips the build if both `TMalign` and `qTMclust` are already on PATH
#   (pass --force as a second arg to rebuild anyway).
#
# Normally invoked via `zevals setup --tools usalign` (zevals/setup.py), not
# directly — this script is the underlying installer that command shells out to.

set -euo pipefail

INSTALL_DIR="${1:-$HOME/.local/bin}"
FORCE="${2:-}"

if command -v TMalign >/dev/null 2>&1 && command -v qTMclust >/dev/null 2>&1 \
    && [ "$FORCE" != "--force" ]; then
    echo "TMalign + qTMclust already on PATH ($(command -v TMalign), $(command -v qTMclust)) — skipping build."
    echo "(pass --force as the second argument to rebuild anyway)"
    TMalign -v || true  # TMalign -v exits 1 on success — not a real failure
    exit 0
fi

if ! command -v g++ >/dev/null 2>&1; then
    echo "Error: g++ not found. USalign has no pre-built binaries — it must" >&2
    echo "be compiled from source. Install a C++ compiler (e.g. 'module load" >&2
    echo "gcc' on a cluster, or 'apt install g++'/'conda install gxx') and" >&2
    echo "re-run." >&2
    exit 1
fi

URL="https://codeload.github.com/pylelab/USalign/tar.gz/refs/heads/master"

WORKDIR="$(mktemp -d)"
trap 'rm -rf "$WORKDIR"' EXIT

echo "Downloading USalign source ..."
wget -q "$URL" -O "$WORKDIR/usalign.tar.gz"

echo "Extracting ..."
tar xzf "$WORKDIR/usalign.tar.gz" -C "$WORKDIR"
SRC_DIR="$(find "$WORKDIR" -maxdepth 1 -type d -name 'USalign-*')"

echo "Compiling TMalign ..."
g++ -O3 -ffast-math "$SRC_DIR/TMalign.cpp" -o "$WORKDIR/TMalign"

echo "Compiling qTMclust ..."
g++ -O3 -ffast-math "$SRC_DIR/qTMclust.cpp" -o "$WORKDIR/qTMclust"

mkdir -p "$INSTALL_DIR"
cp "$WORKDIR/TMalign" "$INSTALL_DIR/TMalign"
cp "$WORKDIR/qTMclust" "$INSTALL_DIR/qTMclust"
chmod +x "$INSTALL_DIR/TMalign" "$INSTALL_DIR/qTMclust"

echo "Installed TMalign + qTMclust to $INSTALL_DIR"
if ! command -v TMalign >/dev/null 2>&1; then
    echo "Warning: $INSTALL_DIR is not on PATH in this shell session." >&2
    echo "Add it (e.g. export PATH=\"$INSTALL_DIR:\$PATH\") or set" >&2
    echo "binary_path explicitly in usalign.yaml." >&2
fi

"$INSTALL_DIR/TMalign" -v || true  # TMalign -v exits 1 on success — not a real failure
