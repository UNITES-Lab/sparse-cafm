#!/bin/bash

DEPTH=6
NUM_HEADS=6
NUM_BLOCKS=6
WINDOW_SIZE=8
DPR=0.1
NORM_LAYER=torch.nn.LayerNorm

SURROGATE_FP="/playpen/mufan/levi/tianlong-chen-lab/material-super-resolution/__exps__/expert-surrogates/2025-03-02_15-27-59_{average_surface_current, coverage_percentage, total_area_extended_shapes}-ViT-Trainable-BS=64/{average_surface_current, coverage_percentage, total_area_extended_shapes}-ViT-Trainable-BS=64_best_older_surrogate.pth"
WEIGHTS_FP="/playpen/mufan/levi/tianlong-chen-lab/material-super-resolution/__exps__/foundation-models/3. combined/2x/2025-03-03_11-01-10_DS={all}-[X|y]-BS=32-opt=Adam-lr=1e-5/DS={all}-[X|y]-BS=32-opt=Adam-lr=1e-5_best.pth"
LOGS_DIR="__exps__/__logs__"

EXP_NAME="augmentations={horizontal_flip, vertical_flip, random_rotation, elastic_transform}"
EXP_ROOT="/playpen/mufan/levi/tianlong-chen-lab/material-super-resolution/__exps__/ablations/transfer-learning/bto/_evals"

UPSAMPLING_RATIO=8
FORMULATION="X"
DATASET="bto"

# --------------------------------------------------------------------------

# ---- 2x2 == 4x upscaling ----
CKPT="/playpen/mufan/levi/tianlong-chen-lab/material-super-resolution/__exps__/foundation-models/3. combined/8x/2025-03-03_10-56-59_DS={all}-[X|y]-BS=32-opt=Adam-lr=1e-5/DS={all}-[X|y]-BS=32-opt=Adam-lr=1e-5_best.pth"
EXP_NAME="$DATASET-$UPSAMPLING_RATIO-X-0-samples"
python expert_evaluation.py \
    --exp_root "$EXP_ROOT" \
    --exp_name "$EXP_NAME" \
    --ckpt_fp "$CKPT" \
    --formulation "$FORMULATION" \
    --dataset "$DATASET" \
    --upsampling_ratio "$UPSAMPLING_RATIO"

# CKPT="/playpen/mufan/levi/tianlong-chen-lab/material-super-resolution/__exps__/ablations/transfer-learning/bto/2025-03-14_13-14-16_BTO-1-samples/BTO-1-samples_best.pth"
# EXP_NAME="$DATASET-$UPSAMPLING_RATIO-X-1-samples"
# python expert_evaluation.py \
#     --exp_root "$EXP_ROOT" \
#     --exp_name "$EXP_NAME" \
#     --ckpt_fp "$CKPT" \
#     --formulation "$FORMULATION" \
#     --dataset "$DATASET" \
#     --upsampling_ratio "$UPSAMPLING_RATIO"

# CKPT="/playpen/mufan/levi/tianlong-chen-lab/material-super-resolution/__exps__/ablations/transfer-learning/bto/2025-03-14_13-12-52_BTO-2-samples/BTO-2-samples_best.pth"
# EXP_NAME="$DATASET-$UPSAMPLING_RATIO-X-2-samples"
# python expert_evaluation.py \
#     --exp_root "$EXP_ROOT" \
#     --exp_name "$EXP_NAME" \
#     --ckpt_fp "$CKPT" \
#     --formulation "$FORMULATION" \
#     --dataset "$DATASET" \
#     --upsampling_ratio "$UPSAMPLING_RATIO"

# CKPT="/playpen/mufan/levi/tianlong-chen-lab/material-super-resolution/__exps__/ablations/transfer-learning/bto/2025-03-14_13-10-40_BTO-3-samples/BTO-3-samples_best.pth"
# EXP_NAME="$DATASET-$UPSAMPLING_RATIO-X-3-samples"
# python expert_evaluation.py \
#     --exp_root "$EXP_ROOT" \
#     --exp_name "$EXP_NAME" \
#     --ckpt_fp "$CKPT" \
#     --formulation "$FORMULATION" \
#     --dataset "$DATASET" \
#     --upsampling_ratio "$UPSAMPLING_RATIO"

# -------------------------------------------
exit