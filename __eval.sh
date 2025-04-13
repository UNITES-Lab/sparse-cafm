#!/bin/bash

DEPTH=6
NUM_HEADS=6
NUM_BLOCKS=6
WINDOW_SIZE=8
DPR=0.1
NORM_LAYER=torch.nn.LayerNorm

WEIGHTS_FP="/playpen/mufan/levi/tianlong-chen-lab/material-super-resolution/__exps__/foundation-models/3. combined/2x/2025-03-03_11-01-10_DS={all}-[X|y]-BS=32-opt=Adam-lr=1e-5/DS={all}-[X|y]-BS=32-opt=Adam-lr=1e-5_best.pth"
LOGS_DIR="__exps__/__logs__"

EXP_NAME="augmentations={horizontal_flip, vertical_flip, random_rotation, elastic_transform}"
EXP_ROOT="/playpen/mufan/levi/tianlong-chen-lab/material-super-resolution/__exps__/ablations/transfer-learning/bto/_evals"

UPSAMPLING_RATIO=2
FORMULATION="y"
CKPT="__weights__/2x/DS={all}-[X|y]-BS=32-opt=Adam-lr=1e-5_best.pth"

# --------------------------------------------------------------------------

# ---- 2x2 == 4x upscaling ----
DATASET="mos2-sef"
EXP_NAME="$DATASET-$UPSAMPLING_RATIO-Y-0-samples"
python expert_evaluation.py \
    --exp_root "$EXP_ROOT" \
    --exp_name "$EXP_NAME" \
    --ckpt_fp "$CKPT" \
    --formulation "$FORMULATION" \
    --dataset "$DATASET" \
    --upsampling_ratio "$UPSAMPLING_RATIO"

DATASET="sapphire"
EXP_NAME="$DATASET-$UPSAMPLING_RATIO-X-1-samples"
python expert_evaluation.py \
    --exp_root "$EXP_ROOT" \
    --exp_name "$EXP_NAME" \
    --ckpt_fp "$CKPT" \
    --formulation "$FORMULATION" \
    --dataset "$DATASET" \
    --upsampling_ratio "$UPSAMPLING_RATIO"

DATASET="silicon"
EXP_NAME="$DATASET-$UPSAMPLING_RATIO-X-1-samples"
python expert_evaluation.py \
    --exp_root "$EXP_ROOT" \
    --exp_name "$EXP_NAME" \
    --ckpt_fp "$CKPT" \
    --formulation "$FORMULATION" \
    --dataset "$DATASET" \
    --upsampling_ratio "$UPSAMPLING_RATIO"

# -------------------------------------------
exit