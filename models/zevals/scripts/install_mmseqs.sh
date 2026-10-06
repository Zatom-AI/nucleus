#!/usr/bin/env bash
# Installs the MMseqs2 static binary (https://github.com/soedinglab/MMseqs2)
# for zevals's protein/RNA sequence-diversity metrics (mmseqs_bin config field
# in proteins.yaml / rna.yaml).
#
# Usage:
#   scripts/install_mmseqs.sh [install_dir]
#
#   install_dir defaults to ~/.local/bin. The script is idempotent — it
#   skips the download if an `mmseqs` binary is already on PATH (pass
#   --force as a second arg to reinstall anyway).
#
# Mirrors scripts/install_foldseek.sh's shape — same publisher (mmseqs.com),
# same static-tarball convention, confirmed against staging/MMseqs2/README.md.

set -euo pipefail

INSTALL_DIR="${1:-$HOME/.local/bin}"
FORCE="${2:-}"

if command -v mmseqs >/dev/null 2>&1 && [ "$FORCE" != "--force" ]; then
    echo "mmseqs already on PATH at $(command -v mmseqs) — skipping install."
    echo "(pass --force as the second argument to reinstall anyway)"
    mmseqs version
    exit 0
fi

ARCH="$(uname -m)"
if [ "$ARCH" = "x86_64" ] && grep -qm1 avx2 /proc/cpuinfo 2>/dev/null; then
    BUILD="avx2"
elif [ "$ARCH" = "x86_64" ] && grep -qm1 sse4_1 /proc/cpuinfo 2>/dev/null; then
    echo "No AVX2 support detected; falling back to the slower SSE4.1 build." >&2
    BUILD="sse41"
elif [ "$ARCH" = "x86_64" ]; then
    echo "Warning: x86_64 CPU without AVX2/SSE4.1 detected; falling back to" >&2
    echo "the SSE2 build (slowest, for very old systems)." >&2
    BUILD="sse2"
else
    echo "Unsupported architecture for this script: $ARCH" >&2
    echo "MMseqs2 also supports ARM64/PPC64LE, but this script only" >&2
    echo "automates the x86_64 static builds confirmed in staging/MMseqs2/README.md." >&2
    echo "See https://github.com/soedinglab/MMseqs2#installation for other options" >&2
    echo "(conda, Homebrew, build from source)." >&2
    exit 1
fi

TARBALL="mmseqs-linux-${BUILD}.tar.gz"
URL="https://mmseqs.com/latest/${TARBALL}"

WORKDIR="$(mktemp -d)"
trap 'rm -rf "$WORKDIR"' EXIT

echo "Downloading ${URL} ..."
wget -q "$URL" -O "$WORKDIR/$TARBALL"

echo "Extracting ..."
tar xzf "$WORKDIR/$TARBALL" -C "$WORKDIR"

mkdir -p "$INSTALL_DIR"
cp "$WORKDIR/mmseqs/bin/mmseqs" "$INSTALL_DIR/mmseqs"
chmod +x "$INSTALL_DIR/mmseqs"

echo "Installed mmseqs to $INSTALL_DIR/mmseqs"
if ! command -v mmseqs >/dev/null 2>&1; then
    echo "Warning: $INSTALL_DIR is not on PATH in this shell session." >&2
    echo "Add it (e.g. export PATH=\"$INSTALL_DIR:\$PATH\") or set" >&2
    echo "mmseqs_bin: $INSTALL_DIR/mmseqs explicitly in proteins.yaml/rna.yaml." >&2
fi

"$INSTALL_DIR/mmseqs" version
