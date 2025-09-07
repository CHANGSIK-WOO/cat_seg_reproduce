
GPUS="${GPUS:-${SLURM_GPUS_ON_NODE:-2}}"
# NNODES=${NNODES:-1}
# NODE_RANK=${NODE_RANK:-0}
MASTER_PORT=${MASTER_PORT:-29500}
MASTER_ADDR=${MASTER_ADDR:-"127.0.0.1"}

set -euo pipefail

export DETECTRON2_DATASETS="/data/datasets/"

#args parsing
CONFIG="./configs/cat_seg_wrapper_vitb-384_ade20k_150.py"
CHECKPOINT="${2:-}"
EXTRA_ARGS=("${@:3}")
export PYTHONUNBUFFERED=1
TEST_PY="./tools/test.py"

echo "[INFO] CONFIG=${CONFIG}"
echo "[INFO] GPUS=${GPUS}"
echo "[INFO] TRAIN_PY=${TEST_PY}"
echo "[INFO] EXTRA_ARGS=${EXTRA_ARGS[*]-}"

. /home/$USER/anaconda3/etc/profile.d/conda.sh
conda activate catseg
ARGS=("${CONFIG}")
if [[ -n "${CHECKPOINT}" ]]; then
  ARGS+=("${CHECKPOINT}")
fi
torchrun \
  --nproc_per_node="${GPUS}" \
  --master_addr="${MASTER_ADDR}" \
  --master_port="${MASTER_PORT}" \
  "${TEST_PY}" \
  "${ARGS[@]}" \
  --launcher pytorch \
  "${EXTRA_ARGS[@]}"

    
