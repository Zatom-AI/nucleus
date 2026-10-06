#!/usr/bin/env bash
# Isolated FlowMM-compatible CHGNet runtime; does not modify the training env.
set -euo pipefail

if [[ $# -ne 1 || "$1" != /* ]]; then
  echo "usage: $0 /absolute/path/to/chgnet-prerelax-env" >&2
  exit 2
fi
chgnet_env_dir="$1"
if [[ -e "${chgnet_env_dir}" ]]; then
  echo "refusing to overwrite existing path: ${chgnet_env_dir}" >&2
  exit 2
fi
script_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
command -v uv >/dev/null

uv venv --python 3.11 "${chgnet_env_dir}"
# Resolve once into an auditable, hash-locked requirements file. Keep this file
# to recreate the full environment, including transitive dependencies.
uv pip compile --python "${chgnet_env_dir}/bin/python" \
  --quiet --exclude-newer 2024-06-01 \
  --generate-hashes --index-strategy unsafe-best-match \
  --extra-index-url https://download.pytorch.org/whl/cu118 \
  --output-file "${chgnet_env_dir}/requirements.lock" \
  "${script_dir}/chgnet_prerelax.requirements.in"
uv pip sync --python "${chgnet_env_dir}/bin/python" --require-hashes \
  --index-strategy unsafe-best-match \
  --extra-index-url https://download.pytorch.org/whl/cu118 \
  "${chgnet_env_dir}/requirements.lock"

echo "CHGNet pre-relaxation environment ready: ${chgnet_env_dir}"
echo "export ZEVALS_CHGNET_PYTHON=${chgnet_env_dir}/bin/python"
