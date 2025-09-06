
GPUS="${GPUS:-${SLURM_GPUS_ON_NODE:-2}}"
# NNODES=${NNODES:-1}
# NODE_RANK=${NODE_RANK:-0}
# PORT=${PORT:-29500}
# MASTER_ADDR=${MASTER_ADDR:-"127.0.0.1"}

set -euo pipefail

#args parsing
CONFIG="./configs/train_cat_seg_wrapper_vitb-384_coco-stuff164k.py"
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
  "${TEST_PY}" \
  "${ARGS[@]}" \
  --launcher pytorch \
  "${EXTRA_ARGS[@]}"


    
