#!/bin/bash
#SBATCH --job-name=difix_dual
#SBATCH --output=slurm_logs/%A_%a.out
#SBATCH --error=slurm_logs/%A_%a.err
#SBATCH --partition=default_partition
#SBATCH --exclude=lil-compute-05,unicorn-compute-01,bhattacharjee-compute-02,awang-compute-01,kim-compute-03,kim-compute-04,kim-compute-05
#SBATCH --gres=gpu:1
#SBATCH --constraint="gpu-high"
#SBATCH -N 1
#SBATCH -n 2
#SBATCH -t 24:00:00
#SBATCH --mem 64gb
#SBATCH --array=0-1

# Dual training + optional external-eval SLURM job array for Difix3D
# Array task 0: Full training (all images)
# Array task 1: Filtered training (optional match_string)
#
# Usage: sbatch examples/dual_training.slurm DATA_DIR [MATCH_STRING] [EVAL_DIR]
#   - DATA_DIR:       Path to COLMAP dataset root (expects 'images/' and 'sparse/')
#   - MATCH_STRING:   Optional token to filter training images (regex-safe word match)
#   - EVAL_DIR:       Optional path to an existing COLMAP dataset directory for external eval
#                     (expects 'images/' and 'sparse/'). No preprocessing is performed.
#
# The result base is inferred from the basename of DATA_DIR and written to
# results/<basename>/{combined,filtered}. Set RESULT_ROOT to override the
# results directory prefix (defaults to "results").
#
# Example (filtered run on RIGHT images with external eval dataset dir):
#   sbatch examples/dual_training.slurm ../gs7/input_data/vision-pro-dog RIGHT ../gs7/input_data/dog/eval/

source /home/wl757/.bashrc
conda activate gaussian_splatting

# Ensure we operate from the repository root (where sbatch was run)
if [ -n "${SLURM_SUBMIT_DIR:-}" ]; then
    cd "$SLURM_SUBMIT_DIR" || {
        echo "ERROR: Failed to cd into SLURM_SUBMIT_DIR='$SLURM_SUBMIT_DIR'"
        exit 1
    }
fi
REPO_ROOT="$(pwd)"
GSPLAT_DIR="$(realpath "$REPO_ROOT/../gsplat" 2>/dev/null || true)"
if [ -n "$GSPLAT_DIR" ] && [ -d "$GSPLAT_DIR" ]; then
    export PYTHONPATH="$REPO_ROOT:$REPO_ROOT/examples/gsplat:$GSPLAT_DIR/examples${PYTHONPATH:+:$PYTHONPATH}"
else
    GSPLAT_DIR=""
    export PYTHONPATH="$REPO_ROOT:$REPO_ROOT/examples/gsplat${PYTHONPATH:+:$PYTHONPATH}"
fi

TRAINER_SCRIPT="$REPO_ROOT/examples/gsplat/simple_trainer_difix3d.py"

# Get arguments
DATA_DIR_INPUT=$1
MATCH_STRING=${2:-""}
# Optional separate COLMAP dataset directory to evaluate on
EVAL_DIR_INPUT=${3:-""}

# Normalize dataset arguments
if [ -n "$DATA_DIR_INPUT" ]; then
    DATA_DIR="$(realpath "$DATA_DIR_INPUT" 2>/dev/null || true)"
else
    DATA_DIR=""
fi
if [ -n "$EVAL_DIR_INPUT" ]; then
    EVAL_DIR="$(realpath "$EVAL_DIR_INPUT" 2>/dev/null || true)"
else
    EVAL_DIR=""
fi

# Evaluation uses the training match string by default unless overridden.
if [ -n "${EVAL_MATCH_STRING:-}" ]; then
    EVAL_MATCH_STR="$EVAL_MATCH_STRING"
elif [ -n "$EVAL_DIR" ] && [ "$EVAL_DIR" != "$DATA_DIR" ]; then
    EVAL_MATCH_STR=""
else
    EVAL_MATCH_STR="$MATCH_STRING"
fi

# Check required arguments
if [ -z "$DATA_DIR" ]; then
    echo "ERROR: Missing required arguments"
    echo "Usage: sbatch examples/dual_training.slurm DATA_DIR [MATCH_STRING] [EVAL_DIR]"
    exit 1
fi

if [ ! -d "$DATA_DIR" ]; then
    echo "ERROR: Data directory '$DATA_DIR' does not exist."
   exit 1
fi

RESULT_ROOT_ENV=${RESULT_ROOT:-results}
if [ "${RESULT_ROOT_ENV#/}" = "$RESULT_ROOT_ENV" ]; then
    RESULT_ROOT_ABS="$REPO_ROOT/$RESULT_ROOT_ENV"
else
    RESULT_ROOT_ABS="$RESULT_ROOT_ENV"
fi
RESULT_NAMING_DIR="$DATA_DIR"
case "$RESULT_NAMING_DIR" in
    */subsets/train)
        RESULT_NAMING_DIR="${RESULT_NAMING_DIR%/subsets/train}"
        ;;
    */subsets/test)
        RESULT_NAMING_DIR="${RESULT_NAMING_DIR%/subsets/test}"
        ;;
esac
DATA_NAME=$(basename "$RESULT_NAMING_DIR")
if [ "$DATA_NAME" = "train" ] || [ "$DATA_NAME" = "test" ]; then
    parent_dir=$(dirname "$RESULT_NAMING_DIR")
    parent_name=$(basename "$parent_dir")
    grandparent_dir=$(dirname "$parent_dir")
    grandparent_name=$(basename "$grandparent_dir")

    if [ -n "$parent_name" ] && [ "$parent_name" != "." ] && [ "$parent_name" != "/" ]; then
        if [ -n "$grandparent_name" ] && [ "$grandparent_name" != "." ] && [ "$grandparent_name" != "/" ]; then
            DATA_NAME="$grandparent_name/$parent_name"
        else
            DATA_NAME="$parent_name"
        fi
    fi
fi
RESULT_BASE="$RESULT_ROOT_ABS/$DATA_NAME"
COMBINED_RESULT_DIR="$RESULT_BASE/combined"
FILTERED_RESULT_DIR="$RESULT_BASE/filtered"

if ! mkdir -p "$COMBINED_RESULT_DIR" "$FILTERED_RESULT_DIR"; then
    echo "ERROR: Failed to create result directories at '$RESULT_BASE'."
    exit 1
fi

# Set result directory and match string based on array task ID
if [ "$SLURM_ARRAY_TASK_ID" -eq 0 ]; then
    RESULT_DIR="$COMBINED_RESULT_DIR"
    JOB_DATA_DIR="$DATA_DIR"
    MATCH_STR=""
    RUN_TYPE="Full (combined)"
else
    RESULT_DIR="$FILTERED_RESULT_DIR"
    JOB_DATA_DIR="$DATA_DIR"
    MATCH_STR="$MATCH_STRING"
    if [ -n "$MATCH_STR" ]; then
        RUN_TYPE="Filtered (match '$MATCH_STR')"
    else
        RUN_TYPE="Filtered (no match string)"
    fi
fi

echo "=========================================="
echo "DIFIX3D DUAL TRAINING - ARRAY JOB"
echo "=========================================="
echo "Job ID: $SLURM_ARRAY_JOB_ID"
echo "Array Task ID: $SLURM_ARRAY_TASK_ID"
echo "Run type: $RUN_TYPE"
echo "Node: $SLURMD_NODENAME"
echo "GPU: $CUDA_VISIBLE_DEVICES"
echo "Data dir: $JOB_DATA_DIR"
echo "Result base: $RESULT_BASE"
echo "Result dir: $RESULT_DIR"
if [ -n "$MATCH_STR" ]; then
    echo "Match string: $MATCH_STR"
else
    echo "Match string: (none)"
fi
if [ -n "$EVAL_MATCH_STR" ]; then
    echo "Eval match string: $EVAL_MATCH_STR"
else
    echo "Eval match string: (none)"
fi
if [ -n "$EVAL_DIR" ]; then
    echo "Eval dataset: $EVAL_DIR"
else
    echo "Eval dataset: (none)"
fi
if [ -z "$GSPLAT_DIR" ]; then
    echo "WARNING: ../gsplat repository not found; external evaluation will be limited."
else
    echo "Gsplat helpers: $GSPLAT_DIR"
fi
echo "=========================================="
echo ""

# Optionally compute covisible masks for external evaluation only
TRAIN_TEST_EVERY=${TRAIN_TEST_EVERY:-8}
EVAL_TEST_EVERY=${EVAL_TEST_EVERY:-1}
USE_COVISIBLE_EXTERNAL=${USE_COVISIBLE_EXTERNAL:-1}
COVI_BATCH_SIZE=${COVI_BATCH_SIZE:-${COVI_CHUNK:-8}}
COVI_NUM_WORKERS=${COVI_NUM_WORKERS:-${COVI_MICRO_CHUNK:-2}}
COVI_DEVICE=${COVI_DEVICE:-cuda}
EVAL_ALIGN_PATH_USER_SET=0
if [ -n "${EVAL_ALIGN_PATH:-}" ]; then
    EVAL_ALIGN_PATH_USER_SET=1
fi
EVAL_ALIGN_PATH=${EVAL_ALIGN_PATH:-}
EVAL_ALIGN_SOURCE=""
if [ -n "$EVAL_ALIGN_PATH" ]; then
    EVAL_ALIGN_SOURCE="env"
fi
ALIGN_ROOT=${ALIGN_ROOT:-$RESULT_DIR/alignments}

# Build the training command
CMD=(
    python "$TRAINER_SCRIPT" default
    --disable_viewer
    --data_factor 1
    --data_dir "$JOB_DATA_DIR"
    --result_dir "$RESULT_DIR"
    --pose_opt
    --eval_steps 3000 7000 30000
    --save_steps 3000 7000 30000
    --max_steps 30000
    --strategy.reset_every 100000
    --strategy.pause_refine_after_reset 0
    --strategy.prune_scale3d 0.22
    --strategy.prune_scale2d 0.12
    --strategy.prune_opa 0.006
    --strategy.grow_grad2d 0.00035
    --strategy.grow_scale3d 0.012
    --strategy.refine_stop_iter 26000
    --strategy.refine_scale2d_stop_iter 26000
    --scale_reg 0.0005
    --antialiased
    --test_every "$TRAIN_TEST_EVERY"
)

if [ -n "$MATCH_STR" ]; then
    CMD+=(--match_string "$MATCH_STR")
fi

echo "Executing: ${CMD[*]}"
echo ""
"${CMD[@]}"

# Link any precomputed covisible masks from the dataset into this run.
if [ -d "$DATA_DIR/covisible" ]; then
    echo "Linking precomputed covisible caches from $DATA_DIR/covisible"
    mkdir -p "$RESULT_DIR/covisible"
    for subset_path in "$DATA_DIR"/covisible/*; do
        [ -d "$subset_path" ] || continue
        subset_name=$(basename "$subset_path")
        dest="$RESULT_DIR/covisible/$subset_name"
        if [ ! -e "$dest" ]; then
            ln -s "$(realpath "$subset_path")" "$dest"
        fi
    done
fi

# If an eval dataset directory is provided, run eval-only using the latest
# checkpoint produced by this run. Results go to: $RESULT_DIR/eval_on_<eval_dir_basename>
if [ -n "$EVAL_DIR" ]; then
    if [ -z "$GSPLAT_DIR" ]; then
        echo "WARNING: Skipping external evaluation; gsplat helpers not found."
    else
        echo ""
        EVAL_DATA_DIR="$EVAL_DIR"
        EVAL_STEM=$(basename "$EVAL_DATA_DIR")

        if [ -d "$EVAL_DATA_DIR/images" ] && [ -d "$EVAL_DATA_DIR/sparse" ]; then
            echo "Finding latest checkpoint in: $RESULT_DIR/ckpts"
            CKPT=$(ls -1 "$RESULT_DIR"/ckpts/ckpt_*_rank0.pt 2>/dev/null | sort -V | tail -n 1)
            if [ -z "$CKPT" ]; then
                echo "WARNING: No checkpoint found; skipping eval-only."
            else
                EVAL_OUT_DIR="$RESULT_DIR/eval_on_${EVAL_STEM}"
                mkdir -p "$EVAL_OUT_DIR"
                if [ -z "$EVAL_ALIGN_PATH" ]; then
                    mkdir -p "$ALIGN_ROOT"
                    default_align="$ALIGN_ROOT/${EVAL_STEM}_to_train.npz"
                    echo "Computing alignment transform at $default_align"
                    python "$REPO_ROOT/examples/compute_dataset_alignment.py" \
                        --train-dir "$DATA_DIR" \
                        --subset-dir "$EVAL_DATA_DIR" \
                        --train-test-every "$TRAIN_TEST_EVERY" \
                        --eval-test-every "$EVAL_TEST_EVERY" \
                        --output "$default_align"
                    if [ -f "$default_align" ]; then
                        EVAL_ALIGN_PATH="$default_align"
                        EVAL_ALIGN_SOURCE="computed"
                    fi
                fi
                if [ "$USE_COVISIBLE_EXTERNAL" = "1" ]; then
                    COVI_SCRIPT="$GSPLAT_DIR/examples/preprocess_covisible_colmap.py"
                    if [ -f "$COVI_SCRIPT" ]; then
                        COVI_OUTPUT_ROOT="$RESULT_DIR/covisible/$EVAL_STEM"
                        COVI_MASK_DIR="$COVI_OUTPUT_ROOT/1x/val"
                        if [ ! -d "$COVI_MASK_DIR" ]; then
                            echo "Precomputing covisible masks for external eval dataset..."
                            PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
                            python "$COVI_SCRIPT" \
                                --base_dir "$EVAL_DATA_DIR" \
                                --support_dir "$JOB_DATA_DIR" \
                                --factor 1 \
                                --test_every 1 \
                                --support_test_every "$TRAIN_TEST_EVERY" \
                                --batch_size "$COVI_BATCH_SIZE" \
                                --num_workers "$COVI_NUM_WORKERS" \
                                --device "$COVI_DEVICE" \
                                --output_dir "$COVI_OUTPUT_ROOT" \
                                --base_split val \
                                --support_split train
                        else
                            echo "Covisible masks exist at $COVI_MASK_DIR; skipping generation."
                        fi
                        covi_align="$COVI_OUTPUT_ROOT/alignment.npz"
                        if [ -f "$covi_align" ] && [ "$EVAL_ALIGN_PATH_USER_SET" -eq 0 ]; then
                            if [ -z "$EVAL_ALIGN_PATH" ]; then
                                EVAL_ALIGN_PATH="$covi_align"
                                EVAL_ALIGN_SOURCE="covisible"
                                echo "Using covisible metadata alignment at $covi_align"
                            else
                                echo "Keeping dataset alignment $EVAL_ALIGN_PATH (covisible metadata at $covi_align will not override)."
                            fi
                        fi
                    else
                        echo "WARNING: Covisible script not found at $COVI_SCRIPT; skipping covisible preprocessing."
                    fi
                fi
                COVI_ROOT_GS="$RESULT_DIR/covisible/$EVAL_STEM/1x"
                EVAL_CMD=(
                    python "$TRAINER_SCRIPT" default
                    --disable_viewer
                    --data_factor 1
                    --data_dir "$EVAL_DATA_DIR"
                    --result_dir "$EVAL_OUT_DIR"
                    --ckpt "$CKPT"
                    --test_every "$EVAL_TEST_EVERY"
                    --max_steps 30000
                    --eval_only
                )
                if [ -n "$EVAL_ALIGN_PATH" ] && [ -f "$EVAL_ALIGN_PATH" ]; then
                    echo "Using dataset transform: $EVAL_ALIGN_PATH"
                    EVAL_CMD+=(--dataset_transform_path "$EVAL_ALIGN_PATH")
                fi
                if [ "$USE_COVISIBLE_EXTERNAL" = "1" ]; then
                    if [ -d "$COVI_ROOT_GS" ]; then
                        EVAL_CMD+=(--eval_use_covisible --eval_dycheck_metrics --eval_covisible_dir "$COVI_ROOT_GS")
                    else
                        echo "WARNING: Covisible directory $COVI_ROOT_GS missing; skipping covisible eval flags."
                    fi
                fi
                echo "Executing external eval: ${EVAL_CMD[*]}"
                "${EVAL_CMD[@]}"
            fi
        else
            echo "WARNING: Expected eval dataset at '$EVAL_DATA_DIR' not found (need 'images/' and 'sparse/'); skipping eval-only."
        fi
    fi
fi

echo ""
echo "=========================================="
echo "ARRAY TASK $SLURM_ARRAY_TASK_ID COMPLETE"
echo "=========================================="
