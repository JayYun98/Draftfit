#!/usr/bin/env bash
# Reproduce one Ling DSpark 900/98 run on one GPU.
#
# The default source is the already regenerated target-greedy aggregate. Set
# REGENERATE=1 and REGEN_SERVER=host:port to regenerate it from a running
# pinned target server before splitting. Target weights are downloaded to the
# instance and are never copied back by this script.
set -Eeuo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$(dirname "$SCRIPT_DIR")"
# Keep Python helpers importable when this checkout has not been installed.
export PYTHONPATH="$ROOT_DIR${PYTHONPATH:+:$PYTHONPATH}"

RUN_ROOT="${RUN_ROOT:-$ROOT_DIR/../../runs/ling-1k-repro}"
SOURCE_ROWS="${SOURCE_ROWS:-$ROOT_DIR/../../runs/ling-1k-local-input/all_998.jsonl}"
TARGET_REPO="${TARGET_REPO:-inclusionAI/Ling-3.0-tiny}"
TARGET_REVISION="${TARGET_REVISION:-a2ee06c0f2de5b171701aee7f73f70a1da75483b}"
TARGET_MODEL_DIR="${TARGET_MODEL_DIR:-$RUN_ROOT/model/Ling-3.0-tiny}"
DRAFT_CONFIG="${DRAFT_CONFIG:-$ROOT_DIR/configs/ling-3.0-tiny-dspark.json}"
CONFIG="${CONFIG:-$ROOT_DIR/examples/configs/ling-3.0-tiny-dspark-offline.yaml}"
GPU="${GPU:-0}"
GPU_IDS="${GPU_IDS:-$GPU}"
CAPTURE_NPROC="${CAPTURE_NPROC:-1}"
CAPTURE_TP_SIZE="${CAPTURE_TP_SIZE:-$CAPTURE_NPROC}"
TRAIN_NPROC="${TRAIN_NPROC:-1}"
TRAIN_SIZE="${TRAIN_SIZE:-900}"
HOLDOUT_SIZE="${HOLDOUT_SIZE:-98}"
TRAIN_NUM_EPOCHS="${TRAIN_NUM_EPOCHS:-1}"
TRAIN_MAX_STEPS="${TRAIN_MAX_STEPS:-}"
TRAIN_TOTAL_STEPS="${TRAIN_TOTAL_STEPS:-}"
RESUME_FROM="${RESUME_FROM:-}"
SPLIT_SEED="${SPLIT_SEED:-20260830}"
REGENERATE="${REGENERATE:-0}"
REGEN_SERVER="${REGEN_SERVER:-127.0.0.1:30000}"
DRY_RUN="${DRY_RUN:-0}"

case "$DRY_RUN" in 0|1) ;; *) echo "DRY_RUN must be 0 or 1" >&2; exit 2 ;; esac
case "$REGENERATE" in 0|1) ;; *) echo "REGENERATE must be 0 or 1" >&2; exit 2 ;; esac
for value in "$CAPTURE_NPROC" "$CAPTURE_TP_SIZE" "$TRAIN_NPROC" "$TRAIN_SIZE" "$TRAIN_NUM_EPOCHS"; do
    [[ "$value" =~ ^[1-9][0-9]*$ ]] || { echo "process counts and training sizes must be positive integers" >&2; exit 2; }
done
TRAIN_MAX_STEPS="${TRAIN_MAX_STEPS:-$(( (TRAIN_SIZE + TRAIN_NPROC - 1) / TRAIN_NPROC ))}"
[[ "$TRAIN_MAX_STEPS" =~ ^[1-9][0-9]*$ ]] || { echo "TRAIN_MAX_STEPS must be a positive integer" >&2; exit 2; }
TRAIN_TOTAL_STEPS="${TRAIN_TOTAL_STEPS:-$TRAIN_MAX_STEPS}"
[[ "$TRAIN_TOTAL_STEPS" =~ ^[1-9][0-9]*$ ]] || { echo "TRAIN_TOTAL_STEPS must be a positive integer" >&2; exit 2; }
(( CAPTURE_NPROC % CAPTURE_TP_SIZE == 0 )) || { echo "CAPTURE_TP_SIZE must divide CAPTURE_NPROC" >&2; exit 2; }
[[ -f "$CONFIG" ]] || { echo "CONFIG does not exist: $CONFIG" >&2; exit 2; }
[[ -f "$DRAFT_CONFIG" ]] || { echo "DRAFT_CONFIG does not exist: $DRAFT_CONFIG" >&2; exit 2; }
if [[ "$DRY_RUN" == 0 && ! -f "$SOURCE_ROWS" ]]; then
    echo "SOURCE_ROWS does not exist: $SOURCE_ROWS" >&2
    exit 2
fi

DATA_ROOT="$RUN_ROOT/dataset"
RAW_ROWS="$DATA_ROOT/all_998_target_greedy.jsonl"
TRAIN_ROWS="$DATA_ROOT/train.jsonl"
HOLDOUT_ROWS="$DATA_ROOT/holdout.jsonl"
TRAIN_FEATURES="$RUN_ROOT/features/train"
HOLDOUT_FEATURES="$RUN_ROOT/features/holdout"
OUTPUT_DIR="$RUN_ROOT/checkpoints"
EXPORT_DIR="$RUN_ROOT/export"
RUN_ID="${RUN_ID:-ling-3.0-tiny-dspark-1k}"

run_cmd() {
    printf '+'
    printf ' %q' "$@"
    printf '\n'
    if [[ "$DRY_RUN" == 0 ]]; then
        "$@"
    fi
}

run_env_cmd() {
    local key="$1"
    local value="$2"
    shift 2
    printf '+'
    printf ' %q' env "$key=$value" "$@"
    printf '\n'
    if [[ "$DRY_RUN" == 0 ]]; then
        env "$key=$value" "$@"
    fi
}

echo "Ling DSpark 1K: train=${TRAIN_SIZE}, holdout=${HOLDOUT_SIZE}, target=${TARGET_REPO}@${TARGET_REVISION}"
if [[ "$DRY_RUN" == 1 ]]; then
    echo "DRY RUN: no download, split, capture, training, or export will run"
fi

if [[ ! -f "$TARGET_MODEL_DIR/config.json" ]]; then
    run_cmd hf download "$TARGET_REPO" --revision "$TARGET_REVISION" --local-dir "$TARGET_MODEL_DIR"
else
    echo "target already present: $TARGET_MODEL_DIR"
fi

run_cmd mkdir -p "$DATA_ROOT"
if [[ "$REGENERATE" == 1 ]]; then
    run_cmd python "$SCRIPT_DIR/regenerate_train_data.py" \
        --model "$TARGET_MODEL_DIR" \
        --server-address "$REGEN_SERVER" \
        --temperature 0 \
        --reasoning disable \
        --regenerated-by ling-t0 \
        --input-file-path "$SOURCE_ROWS" \
        --output-file-path "$RAW_ROWS" \
        --resume
else
    run_cmd cp "$SOURCE_ROWS" "$RAW_ROWS"
fi

run_cmd python "$SCRIPT_DIR/make_1k_split.py" "$RAW_ROWS" "$DATA_ROOT" \
    --train-size "$TRAIN_SIZE" --holdout-size "$HOLDOUT_SIZE" --seed "$SPLIT_SEED"

capture_args=(
    --strategy dspark
    --target-model-path "$TARGET_MODEL_DIR"
    --trust-remote-code
    --draft-model-config "$DRAFT_CONFIG"
    --chat-template ling-flash-2.0
    --max-length 4096
    --tp-size "$CAPTURE_TP_SIZE"
    --batch-size 1
    --build-dataset-num-proc 1
    --num-workers 0
    --sglang-attention-backend fa3
    --sglang-mem-fraction-static 0.72
    --sglang-context-length 4096
    --sglang-disable-radix-cache
)
run_env_cmd CUDA_VISIBLE_DEVICES "$GPU_IDS" torchrun --nproc_per_node="$CAPTURE_NPROC" \
    "$SCRIPT_DIR/prepare_hidden_states.py" \
    "${capture_args[@]}" --data-path "$TRAIN_ROWS" --output-path "$TRAIN_FEATURES"
run_env_cmd CUDA_VISIBLE_DEVICES "$GPU_IDS" torchrun --nproc_per_node="$CAPTURE_NPROC" \
    "$SCRIPT_DIR/prepare_hidden_states.py" \
    "${capture_args[@]}" --data-path "$HOLDOUT_ROWS" --output-path "$HOLDOUT_FEATURES"

train_args=(
    "model.target_model_path=$TARGET_MODEL_DIR"
    "data.hidden_states_path=$TRAIN_FEATURES"
    "data.eval_hidden_states_path=$HOLDOUT_FEATURES"
    "training.num_epochs=$TRAIN_NUM_EPOCHS"
    "training.max_steps=$TRAIN_MAX_STEPS"
    "training.total_steps=$TRAIN_TOTAL_STEPS"
    "training.eval_interval=100"
    "deployment.trainer.nproc_per_node=$TRAIN_NPROC"
    "run_id=$RUN_ID"
    "output_dir=$OUTPUT_DIR"
)
if [[ -n "$RESUME_FROM" ]]; then
    train_args+=("training.resume_from=$RESUME_FROM")
fi
run_env_cmd CUDA_VISIBLE_DEVICES "$GPU_IDS" specforge train --config "$CONFIG" \
    "${train_args[@]}"

run_cmd specforge export --to hf \
    --checkpoint "$OUTPUT_DIR" \
    --draft-config "$DRAFT_CONFIG" \
    --embedding-source "$TARGET_MODEL_DIR" \
    --embedding-key model.word_embeddings.weight \
    --output-dir "$EXPORT_DIR"

if [[ "$DRY_RUN" == 1 ]]; then
    echo "Plan complete: features=$RUN_ROOT/features checkpoints=$OUTPUT_DIR export=$EXPORT_DIR"
else
    echo "Completed: features=$RUN_ROOT/features checkpoints=$OUTPUT_DIR export=$EXPORT_DIR"
fi
