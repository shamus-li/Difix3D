#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$ROOT_DIR"

DATA_ROOT=${DATA_ROOT:-../gs7/dataset}
TRAIN_SUBPATH=${TRAIN_SUBPATH:-subsets/train}
TEST_SUBPATH=${TEST_SUBPATH:-subsets/test}
IPHONE_MATCH=${IPHONE_MATCH:-wide}
STEREO_MATCH=${STEREO_MATCH:-RIGHT}
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
RESULT_ROOT=${RESULT_ROOT:-results}

mkdir -p slurm_logs

for scene in "${SCENES[@]}"; do
  for modality in "${MODALITIES[@]}"; do
    data_dir="${DATA_ROOT}/${scene}/${modality}/${TRAIN_SUBPATH}"
    eval_dir="${DATA_ROOT}/${scene}/${modality}/${TEST_SUBPATH}"

    if [[ ! -d "${data_dir}" && "${TRAIN_SUBPATH}" == "subsets/train" ]]; then
      alt_dir="${DATA_ROOT}/${scene}/${modality}/train"
      if [[ -d "${alt_dir}" ]]; then
        data_dir="${alt_dir}"
      fi
    fi
    if [[ ! -d "${eval_dir}" && "${TEST_SUBPATH}" == "subsets/test" ]]; then
      alt_eval="${DATA_ROOT}/${scene}/${modality}/test"
      if [[ -d "${alt_eval}" ]]; then
        eval_dir="${alt_eval}"
      fi
    fi

    if [[ ! -d "${data_dir}" ]]; then
      echo "Skipping ${scene}/${modality}: missing training dir ${data_dir}" >&2
      continue
    fi
    if [[ ! -d "${eval_dir}/images" || ! -d "${eval_dir}/sparse" ]]; then
      echo "Skipping ${scene}/${modality}: ${eval_dir} missing images/ or sparse/" >&2
      continue
    fi

    if [[ "${modality}" == "iphone" ]]; then
      match_token="${IPHONE_MATCH}"
    else
      match_token="${STEREO_MATCH}"
    fi

    result_base="${RESULT_ROOT}/${scene}/${modality}"
    for split in combined filtered; do
      covisible_dir="${result_base}/${split}/covisible"
      if [[ -d "${covisible_dir}" ]]; then
        echo "Removing stale covisible cache at ${covisible_dir}"
        rm -rf "${covisible_dir}"
      fi
    done

    echo "Launching ${scene}/${modality} (match=${match_token})"
    sbatch --requeue examples/dual_training.slurm "${data_dir}" "${match_token}" "${eval_dir}"
  done
done
