#!/usr/bin/env bash
# Installs the FoldSeek static binary (https://github.com/steineggerlab/foldseek)
# for zevals's structure_search oracle (oracle/structure_search/foldseek.yaml).
#
# Usage:
#   scripts/install_foldseek.sh [install_dir]
#
#   install_dir defaults to ~/.local/bin. The script is idempotent — it
#   skips the download if a `foldseek` binary is already on PATH (pass
#   --force as a second arg to reinstall anyway).
#
# Normally invoked via `zevals setup` (zevals/setup.py), not directly —
# this script is the underlying installer that command shells out to.

set -euo pipefail

INSTALL_DIR="${1:-$HOME/.local/bin}"
FORCE="${2:-}"

if command -v foldseek >/dev/null 2>&1 && [ "$FORCE" != "--force" ]; then
    echo "foldseek already on PATH at $(command -v foldseek) — skipping install."
    echo "(pass --force as the second argument to reinstall anyway)"
    foldseek version
    exit 0
fi

ARCH="$(uname -m)"
if [ "$ARCH" = "aarch64" ] || [ "$ARCH" = "arm64" ]; then
    BUILD="arm64"
elif [ "$ARCH" = "x86_64" ] && grep -qm1 avx2 /proc/cpuinfo 2>/dev/null; then
    BUILD="avx2"
elif [ "$ARCH" = "x86_64" ]; then
    echo "Warning: x86_64 CPU without AVX2 detected; the AVX2 static build" >&2
    echo "requires AVX2 and will crash on older CPUs. Building from source" >&2
    echo "is the fallback in that case (not automated by this script)." >&2
    BUILD="avx2"
else
    echo "Unsupported architecture: $ARCH" >&2
    exit 1
fi

TARBALL="foldseek-linux-${BUILD}.tar.gz"
URL="https://mmseqs.com/foldseek/${TARBALL}"

WORKDIR="$(mktemp -d)"
trap 'rm -rf "$WORKDIR"' EXIT

echo "Downloading ${URL} ..."
wget -q "$URL" -O "$WORKDIR/$TARBALL"

echo "Extracting ..."
tar xzf "$WORKDIR/$TARBALL" -C "$WORKDIR"

mkdir -p "$INSTALL_DIR"
cp "$WORKDIR/foldseek/bin/foldseek" "$INSTALL_DIR/foldseek"
chmod +x "$INSTALL_DIR/foldseek"

echo "Installed foldseek to $INSTALL_DIR/foldseek"
if ! command -v foldseek >/dev/null 2>&1; then
    echo "Warning: $INSTALL_DIR is not on PATH in this shell session." >&2
    echo "Add it (e.g. export PATH=\"$INSTALL_DIR:\$PATH\") or set" >&2
    echo "binary_path: $INSTALL_DIR/foldseek explicitly in foldseek.yaml." >&2
fi

"$INSTALL_DIR/foldseek" version
