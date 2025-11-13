#!/bin/bash
#SBATCH --job-name=difix_eval
#SBATCH --output=slurm_logs/%A_%a.out
#SBATCH --error=slurm_logs/%A_%a.err
#SBATCH --partition=default_partition
#SBATCH --exclude=lil-compute-05,unicorn-compute-01,bhattacharjee-compute-02,awang-compute-01,kim-compute-03,kim-compute-04,kim-compute-05
#SBATCH --gres=gpu:1
#SBATCH --constraint="gpu-high"
#SBATCH -N 1
#SBATCH -n 2
#SBATCH -t 08:00:00
#SBATCH --mem 32gb
#SBATCH --array=0-7
#SBATCH --requeue

# Eval-only SLURM array for Difix3D runs, one job per scene.
# For each scene, evaluates both modalities {iphone, stereo} and variants
# {combined, filtered} if checkpoints are present under results.
# This renders full images (no covisible masking/filling), mirroring gsplat's
# eval-only behavior.
#
# Usage:
#   sbatch examples/eval_only.slurm [DATA_ROOT] [RESULT_ROOT]
#
# Env overrides:
#   TRAIN_SUBPATH      (default: subsets/train)
#   TEST_SUBPATH       (default: subsets/test)
#   TRAIN_TEST_EVERY   (default: 8)
#   EVAL_TEST_EVERY    (default: 1)
#   USE_COVISIBLE_EXTERNAL (default: 1)
#   EVAL_ALIGN_PATH    (default: unset, auto-compute if missing)

set -euo pipefail

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
export PYTHONPATH="$REPO_ROOT:$REPO_ROOT/examples/gsplat${GSPLAT_DIR:+:$GSPLAT_DIR:$GSPLAT_DIR/examples}${PYTHONPATH:+:$PYTHONPATH}"

TRAINER_SCRIPT="$REPO_ROOT/examples/gsplat/simple_trainer_difix3d.py"
ALIGN_SCRIPT="$REPO_ROOT/examples/compute_dataset_alignment.py"

# Arguments
DATA_ROOT_INPUT=${1:-"../gs7/dataset"}
RESULT_ROOT_INPUT=${2:-"results"}

DATA_ROOT="$(realpath "$DATA_ROOT_INPUT")"
if [ "${RESULT_ROOT_INPUT#/}" = "$RESULT_ROOT_INPUT" ]; then
    RESULT_ROOT="$REPO_ROOT/$RESULT_ROOT_INPUT"
else
    RESULT_ROOT="$RESULT_ROOT_INPUT"
fi

TRAIN_SUBPATH=${TRAIN_SUBPATH:-subsets/train}
TEST_SUBPATH=${TEST_SUBPATH:-subsets/test}
TRAIN_TEST_EVERY=${TRAIN_TEST_EVERY:-8}
EVAL_TEST_EVERY=${EVAL_TEST_EVERY:-1}
# Render full images by default (no covisible masking)
USE_COVISIBLE_EXTERNAL=${USE_COVISIBLE_EXTERNAL:-0}
EVAL_ALIGN_SOURCE=""
COVI_BATCH_SIZE=${COVI_BATCH_SIZE:-8}
COVI_NUM_WORKERS=${COVI_NUM_WORKERS:-2}
COVI_DEVICE=${COVI_DEVICE:-cuda}
COVI_MAX_HW=${COVI_MAX_HW:-512}
COVI_SUPPORT_SPLIT=${COVI_SUPPORT_SPLIT:-train}
FORCE_ALIGN_REGEN=${FORCE_ALIGN_REGEN:-0}

SCENES=(
  action-figure
  ball
  chicken
  dog
  espresso
  optics
  salt-pepper
  shelf
)
MODALITIES=(iphone stereo)

SCENE_IDX=${EVAL_SCENE_INDEX:-${SLURM_ARRAY_TASK_ID:-0}}
SCENE=${SCENES[$SCENE_IDX]}

echo "=========================================="
echo "DIFIX3D EVAL-ONLY"
echo "=========================================="
echo "Job ID: ${MANUAL_EVAL_JOB_ID:-${SLURM_ARRAY_JOB_ID:-manual}}"
echo "Array Task ID: $SCENE_IDX"
echo "Scene: $SCENE"
echo "Data root: $DATA_ROOT"
echo "Result root: $RESULT_ROOT"
echo "=========================================="

for modality in "${MODALITIES[@]}"; do
  # Resolve dataset dirs
  train_dir="$DATA_ROOT/$SCENE/$modality/$TRAIN_SUBPATH"
  test_dir="$DATA_ROOT/$SCENE/$modality/$TEST_SUBPATH"
  if [[ ! -d "$train_dir" && "$TRAIN_SUBPATH" == "subsets/train" ]]; then
    alt_train="$DATA_ROOT/$SCENE/$modality/train"
    [[ -d "$alt_train" ]] && train_dir="$alt_train"
  fi
  if [[ ! -d "$test_dir" && "$TEST_SUBPATH" == "subsets/test" ]]; then
    alt_test="$DATA_ROOT/$SCENE/$modality/test"
    [[ -d "$alt_test" ]] && test_dir="$alt_test"
  fi

  if [[ ! -d "$test_dir/images" || ! -d "$test_dir/sparse" ]]; then
    echo "[skip] $SCENE/$modality: missing test dataset at $test_dir"
    continue
  fi

  for variant in combined filtered; do
    RESULT_DIR="$RESULT_ROOT/$SCENE/$modality/$variant"
    CKPT=$(ls -1 "$RESULT_DIR"/ckpts/ckpt_*_rank0.pt 2>/dev/null | sort -V | tail -n 1 || true)
    if [[ -z "$CKPT" ]]; then
      echo "[skip] $SCENE/$modality/$variant: no checkpoints"
      continue
    fi

    # Reset eval outputs to avoid mixing old renders/alignment caches
    rm -rf "$RESULT_DIR/eval_on_test"
    rm -rf "$RESULT_DIR/eval_on_external"
    EVAL_OUT_DIR="$RESULT_DIR/eval_on_test"
    mkdir -p "$EVAL_OUT_DIR"
    CFG_FILE="$RESULT_DIR/cfg.yml"
    POSE_OPT_FLAG=0
    if [[ -f "$CFG_FILE" ]]; then
      POSE_OPT_FLAG=$(
        python - <<'PY' "$CFG_FILE"
import sys
path = sys.argv[1]
val = 0
with open(path, "r") as handle:
    for line in handle:
        stripped = line.strip().lower()
        if stripped.startswith("pose_opt:"):
            token = stripped.split(":", 1)[1].strip()
            if token in ("true", "1", "yes"):
                val = 1
            break
print(val)
PY
      )
    fi

    # Always use dataset alignment. Compute if missing.
    EVAL_ALIGN_SOURCE=""
    DEFAULT_ALIGN="$RESULT_DIR/alignments/test_to_train.npz"
    ALIGN_PATH=""
    if [[ -f "$DEFAULT_ALIGN" && "$FORCE_ALIGN_REGEN" == "1" ]]; then
      echo "Forcing alignment regeneration; removing $DEFAULT_ALIGN"
      rm -f "$DEFAULT_ALIGN"
    fi
    if [[ -f "$DEFAULT_ALIGN" ]]; then
      ALIGN_PATH="$DEFAULT_ALIGN"
      EVAL_ALIGN_SOURCE="cached"
    else
      if [[ -d "$train_dir/images" && -d "$train_dir/sparse" ]]; then
        mkdir -p "$(dirname "$DEFAULT_ALIGN")"
        echo "Computing alignment transform at $DEFAULT_ALIGN"
        python "$ALIGN_SCRIPT" \
          --train-dir "$train_dir" \
          --subset-dir "$test_dir" \
          --train-test-every "$TRAIN_TEST_EVERY" \
          --eval-test-every "$EVAL_TEST_EVERY" \
          --output "$DEFAULT_ALIGN" || true
        if [[ -f "$DEFAULT_ALIGN" ]]; then
          ALIGN_PATH="$DEFAULT_ALIGN"
          EVAL_ALIGN_SOURCE="computed"
        fi
      else
        echo "Warning: cannot compute alignment (missing train dataset); proceeding without alignment."
      fi
    fi

    # Optional covisible usage (disabled by default)
    USE_COVI_FLAGS=()
    if [[ "$USE_COVISIBLE_EXTERNAL" == "1" ]]; then
      COVI_ROOT="$RESULT_DIR/covisible/test/1x"
      COVI_STAGE_DIR="$COVI_ROOT/val"
      if [[ ! -d "$COVI_STAGE_DIR" ]]; then
        if [[ -z "$GSPLAT_DIR" || ! -f "$GSPLAT_DIR/examples/preprocess_covisible_colmap.py" ]]; then
          echo "WARNING: gsplat repo not found; cannot generate covisible masks."
        else
          SUPPORT_DIR="$train_dir"
          if [[ ! -d "$SUPPORT_DIR/images" || ! -d "$SUPPORT_DIR/sparse" ]]; then
            echo "Support dataset missing at $SUPPORT_DIR; falling back to test dataset."
            SUPPORT_DIR="$test_dir"
          fi
          if [[ -d "$SUPPORT_DIR/images" && -d "$SUPPORT_DIR/sparse" ]]; then
            mkdir -p "$RESULT_DIR/covisible/test"
            echo "Generating covisible masks into $RESULT_DIR/covisible/test"
            PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
            python "$GSPLAT_DIR/examples/preprocess_covisible_colmap.py" \
              --base_dir "$test_dir" \
              --support_dir "$SUPPORT_DIR" \
              --factor 1 \
              --test_every "$EVAL_TEST_EVERY" \
              --support_test_every "$TRAIN_TEST_EVERY" \
              --base_split val \
              --support_split "$COVI_SUPPORT_SPLIT" \
              --batch_size "$COVI_BATCH_SIZE" \
              --num_workers "$COVI_NUM_WORKERS" \
              --max_hw "$COVI_MAX_HW" \
              --device "$COVI_DEVICE" \
              --output_dir "$RESULT_DIR/covisible/test"
          else
            echo "WARNING: Unable to locate support dataset; skipping covisible preprocessing."
          fi
      fi
    fi
    if [[ -d "$COVI_STAGE_DIR" ]]; then
      USE_COVI_FLAGS=(--eval_use_covisible --eval_dycheck_metrics --eval_covisible_dir "$COVI_ROOT")
      covi_align="$RESULT_DIR/covisible/test/alignment.npz"
      if [[ -z "$ALIGN_PATH" && -f "$covi_align" ]]; then
        ALIGN_PATH="$covi_align"
        EVAL_ALIGN_SOURCE="covisible"
        echo "Using covisible metadata alignment at $covi_align"
      elif [[ -n "$ALIGN_PATH" && -f "$covi_align" ]]; then
        echo "Keeping dataset alignment $ALIGN_PATH (covisible metadata at $covi_align will not override)."
      fi
    else
      echo "WARNING: Covisible directory $COVI_STAGE_DIR missing; skipping covisible eval flags."
    fi
  fi

    echo "Running eval: $SCENE/$modality/$variant"
    CMD=(
      python "$TRAINER_SCRIPT" default
      --disable_viewer
      --data_factor 1
      --data_dir "$test_dir"
      --result_dir "$EVAL_OUT_DIR"
      --ckpt "$CKPT"
      --test_every "$EVAL_TEST_EVERY"
      --max_steps 30000
      --eval_only
    )
    if [[ -n "$ALIGN_PATH" ]]; then
      CMD+=(--dataset_transform_path "$ALIGN_PATH")
    fi
    if [[ "$POSE_OPT_FLAG" == "1" ]]; then
      CMD+=(--pose_opt)
    fi
    CMD+=("${USE_COVI_FLAGS[@]}")
    echo "Executing: ${CMD[*]}"
    "${CMD[@]}"
  done
done

echo ""
echo "=========================================="
echo "EVAL FOR SCENE '$SCENE' COMPLETE"
echo "=========================================="
