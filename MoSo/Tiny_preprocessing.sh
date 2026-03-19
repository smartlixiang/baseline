#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
DATA_ROOT="${1:-${SCRIPT_DIR}/data}"
python "${SCRIPT_DIR}/prepare_tiny_imagenet.py" --data_root "${DATA_ROOT}"
