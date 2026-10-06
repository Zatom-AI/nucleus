#!/usr/bin/env bash
# Downloads and builds a FoldSeek reference database via FoldSeek's own
# `foldseek databases` subcommand — needed for the *novelty* metric
# (oracle/structure_search/foldseek.yaml's target_db /
# ProteinEvalPipeline.target_db / RNAEvalPipeline.target_db). NOT needed for
# structural/sequence diversity (clustering) — that runs against your own
# generated samples only, no reference database required.
#
# Usage:
#   scripts/build_foldseek_db.sh <db_name> <output_dir> [--force]
#
#   db_name    One of FoldSeek's built-in presets. Run `foldseek databases`
#              with no arguments to list them all (PDB, Alphafold/UniProt50-
#              minimal, CATH50, ESMAtlas30, ...). For protein novelty
#              against "PDB + AFDB" (the convention this repo's README
#              documents), build both PDB and a representative AFDB set
#              (e.g. Alphafold/UniProt50-minimal) separately and pass
#              whichever (or both, searched independently) as target_db.
#   output_dir Where to write the built database. Pick a persistent,
#              large-storage location on your cluster (NOT /tmp — these are
#              multi-GB to multi-hundred-GB and should survive across
#              sessions/training runs, not live on ephemeral/purged
#              scratch). The database prefix ends up at
#              output_dir/<slugified db_name>/db.
#
# Idempotent: skips rebuilding if the target database prefix already exists,
# unless --force. Requires `foldseek` on PATH — run install_foldseek.sh (or
# `zevals setup`) first.

set -euo pipefail

DB_NAME="${1:?Usage: build_foldseek_db.sh <db_name> <output_dir> [--force]}"
OUTPUT_DIR="${2:?Usage: build_foldseek_db.sh <db_name> <output_dir> [--force]}"
FORCE="${3:-}"

if ! command -v foldseek >/dev/null 2>&1; then
    echo "foldseek not found on PATH — run scripts/install_foldseek.sh (or 'zevals setup') first." >&2
    exit 1
fi

SLUG="$(echo "$DB_NAME" | tr '/ ' '__')"
DB_DIR="$OUTPUT_DIR/$SLUG"
DB_PREFIX="$DB_DIR/db"

if [ -f "$DB_PREFIX" ] && [ "$FORCE" != "--force" ]; then
    echo "Database already exists at $DB_PREFIX — skipping (pass --force to rebuild)."
    echo "$DB_PREFIX"
    exit 0
fi

mkdir -p "$DB_DIR"
FS_TMPDIR="$(mktemp -d)"
trap 'rm -rf "$FS_TMPDIR"' EXIT

echo "Downloading + building FoldSeek database '$DB_NAME' at $DB_PREFIX ..."
echo "(this can take a while and use significant disk space — run 'foldseek databases'"
echo " with no arguments for the full preset list and descriptions)"
foldseek databases "$DB_NAME" "$DB_PREFIX" "$FS_TMPDIR"

echo
echo "Done. Database prefix: $DB_PREFIX"
echo "$DB_PREFIX"
