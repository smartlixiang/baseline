#!/usr/bin/env bash
set -euo pipefail

DATA_ROOT="${1:-./data}"
python prepare_tiny_imagenet.py --data_root "${DATA_ROOT}"
