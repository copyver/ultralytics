#!/usr/bin/env bash
# Ultralytics 🚀 AGPL-3.0 License - https://ultralytics.com/license
set -euo pipefail
if (( $# < 2 )); then
    echo "Usage: $0 MODEL_DIR DATASET_YAML [key=value ...]" >&2
    exit 2
fi
weights="$1/yolo26x-sem-ade20k.pt"
data="$2"
shift 2
for file in "$weights" "$data"; do
    if [[ ! -f "$file" ]]; then
        echo "File not found: $file" >&2
        exit 1
    fi
done

recipe_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
export PYTHONPATH="$recipe_dir/../../..${PYTHONPATH:+:$PYTHONPATH}"
exec "${PYTHON_BIN:-python3}" -c 'from ultralytics.cfg import entrypoint; entrypoint()' \
    cfg="$recipe_dir/train_merged_ade20k.yaml" model="$weights" data="$data" "$@"
