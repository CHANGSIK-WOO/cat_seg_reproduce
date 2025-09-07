#!/usr/bin/env bash

#SBATCH --job-name=cat_seg_repo
#SBATCH --nodes=1
#SBATCH --gres=gpu:2
#SBATCH -p batch
#SBATCH -w vgi1
#SBATCH --cpus-per-gpu=8
#SBATCH --mem=45G
#SBATCH --time=2-0
#SBATCH -o ./logs/%N_%x_%j.out
#SBATCH -e ./logs/%N_%x_%j.err

set -euo pipefail

# config=$1
# gpus=$2
# output=$3

export DETECTRON2_DATASETS="/data/datasets"


. /home/$USER/anaconda3/etc/profile.d/conda.sh
conda activate catseg

# if [ -z $config ]
# then
#     echo "No config file found! Run with "sh eval.sh [CONFIG_FILE] [NUM_GPUS] [OUTPUT_DIR] [OPTS]""
#     exit 0
# fi

# if [ -z $gpus ]
# then
#     echo "Number of gpus not specified! Run with "sh eval.sh [CONFIG_FILE] [NUM_GPUS] [OUTPUT_DIR] [OPTS]""
#     exit 0
# fi

# if [ -z $output ]
# then
#     echo "No output directory found! Run with "sh eval.sh [CONFIG_FILE] [NUM_GPUS] [OUTPUT_DIR] [OPTS]""
#     exit 0
# fi

# shift 3
# opts=${@}

CONFIG="configs/coco-stuff164k_vitb_384.yaml"
GPUS=2
OUTPUT="./output_250906_coco_stuff_train_reproduce_4_80000iter_real"

python misc/train_net.py --config "${CONFIG}" \
 --num-gpus "${GPUS}" \
 --dist-url "auto" \
 --resume \
 OUTPUT_DIR "${OUTPUT}" \


# sh eval.sh $config $gpus $output $opts