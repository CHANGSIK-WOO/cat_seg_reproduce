#!/usr/bin/env bash

#SBATCH --job-name=cat_seg_reproduce
#SBATCH --nodes=1
#SBATCH --gres=gpu:2
#SBATCH -p batch
#SBATCH -w vgi1                     
#SBATCH --cpus-per-gpu=8
#SBATCH --mem-per-gpu=15G
#SBATCH --time=2-0
#SBATCH -o ./logs/%N_%x_%j.out
#SBATCH -e ./logs/%N_%x_%j.err

set -euo pipefail

#args parsing
CONFIG="configs/train_cat_seg_wrapper_vitb-384_coco-stuff164k.py"
EXTRA_ARGS=("${@:2}")

#DDP Parameter
GPUS="${GPUS:-${SLURM_GPUS_ON_NODE:-2}}"

mkdir -p ./logs
export PYTHONUNBUFFERED=1
TRAIN_PY="./train.py"

echo "[INFO] CONFIG=${CONFIG}"
echo "[INFO] GPUS=${GPUS}"
echo "[INFO] TRAIN_PY=${TRAIN_PY}"
echo "[INFO] EXTRA_ARGS=${EXTRA_ARGS[*]-}"

. /home/$USER/anaconda3/etc/profile.d/conda.sh
conda activate catseg
torchrun \
  --nproc_per_node="${GPUS}" \
  "${TRAIN_PY}" \
  "${CONFIG}" \
  --launcher pytorch \
  "${EXTRA_ARGS[@]}"

  