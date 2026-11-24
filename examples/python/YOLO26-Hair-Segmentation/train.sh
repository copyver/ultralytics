#!/usr/bin/env bash
# Ultralytics 🚀 AGPL-3.0 License - https://ultralytics.com/license
set -euo pipefail

recipe_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
export PYTHONPATH="$recipe_dir/../../..${PYTHONPATH:+:$PYTHONPATH}"
python_bin="${PYTHON_BIN:-/picassox/intelligent-cpfs/beauty/lhy9/pyenv/ultralytics/bin/python}"
exec "$python_bin" -c 'from ultralytics.cfg import entrypoint; entrypoint()' \
    cfg="$recipe_dir/train_yolo26x_sem_ade20k.yaml" "$@"
