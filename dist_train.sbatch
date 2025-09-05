#!/usr/bin/env bash
#
# ==== SLURM ====
#SBATCH --job-name=cat_seg_reproduce
#SBATCH --nodes=1
#SBATCH --gres=gpu:2
#SBATCH -p batch
#SBATCH -w vgi1
#SBATCH --cpus-per-gpu=8
#SBATCH --mem=20G
#SBATCH --time=2-0
#SBATCH -o ./logs/%N_%x_%j.out
#SBATCH -e ./logs/%N_%x_%j.err



set -euo pipefail

cd "${SLURM_SUBMIT_DIR}"


# ==== Args ====
CONFIG="${1:-train_cat_seg_wrapper_vitb-384_coco-stuff164k.py}"
EXTRA_ARGS=("${@:2}")

# ==== Dist params (single node) ====
GPUS="${GPUS:-${SLURM_GPUS_ON_NODE:-2}}"   # --gres=gpu:2면 보통 2로 잡힘
NNODES="${NNODES:-${SLURM_NNODES:-1}}"
NODE_RANK="${NODE_RANK:-${SLURM_NODEID:-0}}"
MASTER_ADDR="${MASTER_ADDR:-$(scontrol show hostnames "$SLURM_JOB_NODELIST" | head -n1)}"
PORT="${PORT:-29500}"

# ==== Paths / Env ====
mkdir -p ./logs
export PYTHONUNBUFFERED=1

TRAIN_PY="train.py"   # 필요하면 tools/train.py로 변경

echo "[INFO] CONFIG=${CONFIG}"
echo "[INFO] GPUS=${GPUS}, NNODES=${NNODES}, NODE_RANK=${NODE_RANK}"
echo "[INFO] MASTER_ADDR=${MASTER_ADDR}, PORT=${PORT}"
echo "[INFO] TRAIN_PY=${TRAIN_PY}"
echo "[INFO] EXTRA_ARGS=${EXTRA_ARGS[*]-}"

# ==== Launch ====
. /home/$USER/anaconda3/etc/profile.d/conda.sh
conda activate catseg
python -m torch.distributed.launch \
  --nnodes="${NNODES}" \
  --node_rank="${NODE_RANK}" \
  --master_addr="${MASTER_ADDR}" \
  --nproc_per_node="${GPUS}" \
  --master_port="${PORT}" \
  "${TRAIN_PY}" \
  "${CONFIG}" \
  --launcher pytorch \
  "${EXTRA_ARGS[@]}"
