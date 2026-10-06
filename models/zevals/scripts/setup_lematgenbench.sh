#!/usr/bin/env bash
# Create the isolated, pinned LeMat-GenBench environment used by zevals.

set -euo pipefail

readonly LEMAT_REPOSITORY="https://github.com/LeMaterial/lemat-genbench.git"
readonly LEMAT_COMMIT="58e6eae3e4a6c87c22171cf069123ecc4e2fa7e6"
readonly LEMAT_LOCK_SHA256="74970323f6697d50e60251de9182e7a559031bce67ab2f693986742e7c286850"

if [[ $# -ne 1 ]]; then
  echo "usage: $0 /absolute/path/to/lemat-genbench" >&2
  exit 2
fi

target_dir="$1"
if [[ "${target_dir}" != /* ]]; then
  echo "target must be an absolute path: ${target_dir}" >&2
  exit 2
fi
if [[ -e "${target_dir}" ]]; then
  echo "refusing to overwrite existing path: ${target_dir}" >&2
  exit 2
fi

command -v git >/dev/null
command -v uv >/dev/null
command -v sha256sum >/dev/null

# Upstream issue #82 reports an exhausted Git LFS quota. The canonical
# structure-matcher configuration does not consume the affected BAWL parquet
# files, so skip LFS smudging during checkout.
export GIT_LFS_SKIP_SMUDGE=1
git clone --filter=blob:none --no-checkout "${LEMAT_REPOSITORY}" "${target_dir}"
git -C "${target_dir}" checkout --detach "${LEMAT_COMMIT}"

# Use the lockfile committed at the pinned revision. Running `uv lock` here
# would mutate the upstream protocol and make the checkout fail the adapter's
# clean-worktree check.
actual_lock_sha256="$(sha256sum "${target_dir}/uv.lock")"
actual_lock_sha256="${actual_lock_sha256%% *}"
if [[ "${actual_lock_sha256}" != "${LEMAT_LOCK_SHA256}" ]]; then
  echo "unexpected upstream uv.lock SHA-256: ${actual_lock_sha256}" >&2
  exit 1
fi
uv sync --directory "${target_dir}" --python 3.11 --frozen

echo "LeMat-GenBench ${LEMAT_COMMIT} is ready at ${target_dir}"
echo "Set ZEVALS_LEMAT_GENBENCH_REPO=${target_dir} before evaluation."
