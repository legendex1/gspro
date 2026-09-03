#!/usr/bin/env bash

# Run GaussianPro training, rendering, and metric evaluation.
# Usage:
#   bash run_pipeline.sh <input_data_dir> <output_model_dir>

set -Eeuo pipefail

usage() {
    echo "Usage:"
    echo "  bash run_pipeline.sh <input_data_dir> <output_model_dir>"
    echo
    echo "Example:"
    echo "  bash run_pipeline.sh /data/DL3DV/scene01 /data/GaussianPro/results/scene01"
}

die() {
    echo "Error: $*" >&2
    exit 1
}

if [[ $# -ne 2 ]]; then
    usage
    exit 2
fi

CALLER_DIR="$(pwd)"
SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
SOURCE_ARG="$1"
MODEL_ARG="$2"

[[ -d "$SOURCE_ARG" ]] || die "Input data directory does not exist: $SOURCE_ARG"
SOURCE_DIR="$(cd -- "$SOURCE_ARG" && pwd)"

case "$MODEL_ARG" in
    /*) MODEL_DIR="$MODEL_ARG" ;;
    *)  MODEL_DIR="$CALLER_DIR/$MODEL_ARG" ;;
esac
mkdir -p "$MODEL_DIR"
MODEL_DIR="$(cd -- "$MODEL_DIR" && pwd)"

[[ -d "$SOURCE_DIR/images" ]] || die "Missing images directory: $SOURCE_DIR/images"
[[ -d "$SOURCE_DIR/sparse/0" ]] || die "Missing COLMAP sparse model: $SOURCE_DIR/sparse/0"

cd -- "$SCRIPT_DIR"

# Use the currently active environment. If no environment is active, fall back
# to python or python3 available on PATH.
if [[ -n "${PYTHON_BIN:-}" ]]; then
    PYTHON="$PYTHON_BIN"
elif command -v python >/dev/null 2>&1; then
    PYTHON="python"
elif command -v python3 >/dev/null 2>&1; then
    PYTHON="python3"
else
    die "Python not found. Install and activate the GaussianPro environment first."
fi

command -v "$PYTHON" >/dev/null 2>&1 || die "Cannot execute Python: $PYTHON"

# Give each physical GPU its own GUI port. Set CUDA_VISIBLE_DEVICES before
# starting the script, for example CUDA_VISIBLE_DEVICES=1. GAUSSIANPRO_PORT
# can be set to explicitly override the automatically selected port.
GPU_LIST="${CUDA_VISIBLE_DEVICES:-0}"
PRIMARY_GPU="${GPU_LIST%%,*}"
[[ "$PRIMARY_GPU" =~ ^[0-9]+$ ]] || die "CUDA_VISIBLE_DEVICES must start with a numeric GPU id: $GPU_LIST"
PORT="${GAUSSIANPRO_PORT:-$((1021 + PRIMARY_GPU))}"
[[ "$PORT" =~ ^[0-9]+$ ]] || die "GAUSSIANPRO_PORT must be a number: $PORT"

LOG_FILE="$MODEL_DIR/pipeline_$(date +%Y%m%d_%H%M%S).log"
touch "$LOG_FILE" || die "Cannot write log file: $LOG_FILE"

run_step() {
    local step_name="$1"
    shift

    echo
    echo "============================================================"
    echo "$step_name"
    echo "============================================================"
    echo "Command: $*"
    "$@" 2>&1 | tee -a "$LOG_FILE"
}

echo "GaussianPro pipeline"
echo "Project:  $SCRIPT_DIR"
echo "Input:    $SOURCE_DIR"
echo "Output:   $MODEL_DIR"
echo "Python:   $PYTHON"
echo "GPU:      $GPU_LIST"
echo "Port:     $PORT"
echo "Log file: $LOG_FILE"

# These are the GaussianPro settings used by the project's demo.sh.
# --dataset free is suitable for custom COLMAP video/image sequences.
run_step "1/3 Training GaussianPro" \
    "$PYTHON" train.py \
    -s "$SOURCE_DIR" \
    -m "$MODEL_DIR" \
    --eval \
    --data_device cpu \
    --dataset free \
    --flatten_loss \
    --normal_loss \
    --depth_loss \
    --position_lr_init 0.000016 \
    --scaling_lr 0.001 \
    --percent_dense 0.0005 \
    --propagation_interval 50 \
    --depth_error_min_threshold 0.8 \
    --depth_error_max_threshold 1.0 \
    --propagated_iteration_begin 1000 \
    --propagated_iteration_after 6000 \
    --patch_size 20 \
    --lambda_l1_normal 0.001 \
    --lambda_cos_normal 0.001 \
    --port "$PORT"

run_step "2/3 Rendering train and test sets" \
    "$PYTHON" render.py \
    -m "$MODEL_DIR"

run_step "3/3 Evaluating test metrics" \
    "$PYTHON" metrics.py \
    -m "$MODEL_DIR"

echo
echo "Pipeline completed."
echo "Renders: $MODEL_DIR/test/ours_*/renders"
echo "Color depth maps: $MODEL_DIR/test/ours_*/render_depth"
echo "Metrics: $MODEL_DIR/results.json"
echo "Log: $LOG_FILE"
