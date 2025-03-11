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
EXP_ROOT_DIR="/playpen/mufan/levi/tianlong-chen-lab/material-super-resolution/__exps__/ablations/augmentations/_evals"
EXP_NAME="augmentations={horizontal_flip, vertical_flip, random_rotation, elastic_transform}"

UPSAMPLE_FACTOR=2
FORMULATION="both"
DATASET="mos2-sef"

# --------------------------------------------------------------------------

LOGS_DIR="__exps__/__logs__"
EXP_ROOT="/playpen/mufan/levi/tianlong-chen-lab/material-super-resolution/__exps__/ablations/augmentations/_evals"

export CUDA_VISIBLE_DEVICES=2
CKPT="/playpen/mufan/levi/tianlong-chen-lab/material-super-resolution/__exps__/ablations/augmentations/2025-03-09_10-20-10_augmentations={None}/augmentations={None}_best.pth"
EXP_NAME="{None}"
python test.py \
    --root "$EXP_ROOT_DIR" \
    --exp_name "$EXP_NAME" \
    --upsample_factor $UPSAMPLE_FACTOR \
    --dataset $DATASET \
    --formulation $FORMULATION \
    --weights "$CKPT" \
    --surrogate_weights "$SURROGATE_FP" \
    --num_heads $NUM_HEADS \
    --num_blocks $NUM_BLOCKS \
    --window_size $WINDOW_SIZE \
    --drop_path_rate $DPR \
    --norm_layer $NORM_LAYER \
    > "$LOGS_DIR/_$EXP_NAME.out" 2>&1 &

export CUDA_VISIBLE_DEVICES=3
CKPT="/playpen/mufan/levi/tianlong-chen-lab/material-super-resolution/__exps__/ablations/augmentations/2025-03-09_10-20-33_augmentations={horizontal_flip}/augmentations={horizontal_flip}_best.pth"
EXP_NAME="{horizontal_flip, }"
python test.py \
    --root "$EXP_ROOT_DIR" \
    --exp_name "$EXP_NAME" \
    --upsample_factor $UPSAMPLE_FACTOR \
    --dataset $DATASET \
    --formulation $FORMULATION \
    --weights "$CKPT" \
    --surrogate_weights "$SURROGATE_FP" \
    --num_heads $NUM_HEADS \
    --num_blocks $NUM_BLOCKS \
    --window_size $WINDOW_SIZE \
    --drop_path_rate $DPR \
    --norm_layer $NORM_LAYER \
    > "$LOGS_DIR/_$EXP_NAME.out" 2>&1 &

export CUDA_VISIBLE_DEVICES=4
CKPT="/playpen/mufan/levi/tianlong-chen-lab/material-super-resolution/__exps__/ablations/augmentations/2025-03-09_10-20-50_augmentations={horizontal_flip, vertical_flip}/augmentations={horizontal_flip, vertical_flip}_best.pth"
EXP_NAME="{horizontal_flip, vertical_flip}"
python test.py \
    --root "$EXP_ROOT_DIR" \
    --exp_name "$EXP_NAME" \
    --upsample_factor $UPSAMPLE_FACTOR \
    --dataset $DATASET \
    --formulation $FORMULATION \
    --weights "$CKPT" \
    --surrogate_weights "$SURROGATE_FP" \
    --num_heads $NUM_HEADS \
    --num_blocks $NUM_BLOCKS \
    --window_size $WINDOW_SIZE \
    --drop_path_rate $DPR \
    --norm_layer $NORM_LAYER \
    > "$LOGS_DIR/_$EXP_NAME.out" 2>&1 &

export CUDA_VISIBLE_DEVICES=5
CKPT="/playpen/mufan/levi/tianlong-chen-lab/material-super-resolution/__exps__/ablations/augmentations/2025-03-09_10-21-02_augmentations={horizontal_flip, vertical_flip, rotate}/augmentations={horizontal_flip, vertical_flip, rotate}_best.pth"
EXP_NAME="{horizontal_flip, vertical_flip, random_rotation}"
python test.py \
    --root "$EXP_ROOT_DIR" \
    --exp_name "$EXP_NAME" \
    --upsample_factor $UPSAMPLE_FACTOR \
    --dataset $DATASET \
    --formulation $FORMULATION \
    --weights "$CKPT" \
    --surrogate_weights "$SURROGATE_FP" \
    --num_heads $NUM_HEADS \
    --num_blocks $NUM_BLOCKS \
    --window_size $WINDOW_SIZE \
    --drop_path_rate $DPR \
    --norm_layer $NORM_LAYER \
    > "$LOGS_DIR/_$EXP_NAME.out" 2>&1 &
    
CKPT="/playpen/mufan/levi/tianlong-chen-lab/material-super-resolution/__exps__/ablations/augmentations/2025-03-09_10-21-19_augmentations={horizontal_flip, vertical_flip, rotate, elastic_transform}/augmentations={horizontal_flip, vertical_flip, rotate, elastic_transform}_best.pth"
EXP_NAME="{horizontal_flip, vertical_flip, random_rotation, elastic_transform}"
python test.py \
    --root "$EXP_ROOT_DIR" \
    --exp_name "$EXP_NAME" \
    --upsample_factor $UPSAMPLE_FACTOR \
    --dataset $DATASET \
    --formulation $FORMULATION \
    --weights "$CKPT" \
    --surrogate_weights "$SURROGATE_FP" \
    --num_heads $NUM_HEADS \
    --num_blocks $NUM_BLOCKS \
    --window_size $WINDOW_SIZE \
    --drop_path_rate $DPR \
    --norm_layer $NORM_LAYER \
    > "$LOGS_DIR/_$EXP_NAME.out" 2>&1 &

# ---- surface current ----
# FORMULATION=y
# EXP_ROOT="/playpen/mufan/levi/tianlong-chen-lab/material-super-resolution/__exps__/foundation-models/4. combined-evals"

# ---- 2x2 == 4x upscaling ----
# CKPT="/playpen/mufan/levi/tianlong-chen-lab/material-super-resolution/__exps__/foundation-models/3. combined/2x/2025-03-04_11-14-17_DS={all}-[X|y]-BS=32-opt=Adam-lr=1e-5/DS={all}-[X|y]-BS=32-opt=Adam-lr=1e-5_best.pth"
# UPSAMPLING_RATIO=2

# EXP_NAME="mos2-$UPSAMPLING_RATIO-X"
# DATASET=mos2-sef
# python expert_evaluation.py \
#     --exp_root "$EXP_ROOT" \
#     --exp_name "$EXP_NAME" \
#     --ckpt_fp "$CKPT" \
#     --formulation "$FORMULATION" \
#     --dataset "$DATASET" \
#     --upsampling_ratio "$UPSAMPLING_RATIO"

# EXP_NAME="sapphire-$UPSAMPLING_RATIO-X"
# DATASET=sapphire
# python expert_evaluation.py \
#     --exp_root "$EXP_ROOT" \
#     --exp_name "$EXP_NAME" \
#     --ckpt_fp "$CKPT" \
#     --formulation "$FORMULATION" \
#     --dataset "$DATASET" \
#     --upsampling_ratio "$UPSAMPLING_RATIO"

# EXP_NAME="silicon-$UPSAMPLING_RATIO-X"
# DATASET=silicon
# python expert_evaluation.py \
#     --exp_root "$EXP_ROOT" \
#     --exp_name "$EXP_NAME" \
#     --ckpt_fp "$CKPT" \
#     --formulation "$FORMULATION" \
#     --dataset "$DATASET" \
#     --upsampling_ratio "$UPSAMPLING_RATIO"

export CUDA_VISIBLE_DEVICES=7

# ---- 4x4 == 16x upscaling ----
CKPT="/playpen/mufan/levi/tianlong-chen-lab/material-super-resolution/__exps__/foundation-models/3. combined/4x/2025-03-03_10-59-32_DS={all}-[X|y]-BS=32-opt=Adam-lr=1e-5/DS={all}-[X|y]-BS=32-opt=Adam-lr=1e-5_best.pth"
UPSAMPLING_RATIO=4

# EXP_NAME="mos2-$UPSAMPLING_RATIO-X"
# DATASET=mos2-sef
# python expert_evaluation.py \
#     --exp_root "$EXP_ROOT" \
#     --exp_name "$EXP_NAME" \
#     --ckpt_fp "$CKPT" \
#     --formulation "$FORMULATION" \
#     --dataset "$DATASET" \
#     --upsampling_ratio "$UPSAMPLING_RATIO" \
#     > "$LOGS_DIR/_$EXP_NAME.out" 2>&1 &

# EXP_NAME="sapphire-$UPSAMPLING_RATIO-X"
# DATASET=sapphire
# python expert_evaluation.py \
#     --exp_root "$EXP_ROOT" \
#     --exp_name "$EXP_NAME" \
#     --ckpt_fp "$CKPT" \
#     --formulation "$FORMULATION" \
#     --dataset "$DATASET" \
#     --upsampling_ratio "$UPSAMPLING_RATIO" \
#     > "$LOGS_DIR/_$EXP_NAME.out" 2>&1 &

# EXP_NAME="silicon-$UPSAMPLING_RATIO-X"
# DATASET=silicon
# python expert_evaluation.py \
#     --exp_root "$EXP_ROOT" \
#     --exp_name "$EXP_NAME" \
#     --ckpt_fp "$CKPT" \
#     --formulation "$FORMULATION" \
#     --dataset "$DATASET" \
#     --upsampling_ratio "$UPSAMPLING_RATIO" \
#     > "$LOGS_DIR/_$EXP_NAME.out" 2>&1 &

# # ---- 8x8 == 64x upscaling ----
# CKPT="/playpen/mufan/levi/tianlong-chen-lab/material-super-resolution/__exps__/foundation-models/3. combined/8x/2025-03-03_10-56-59_DS={all}-[X|y]-BS=32-opt=Adam-lr=1e-5/DS={all}-[X|y]-BS=32-opt=Adam-lr=1e-5_best.pth"
# UPSAMPLING_RATIO=8

# EXP_NAME="mos2-$UPSAMPLING_RATIO-X"
# DATASET=mos2-sef
# python expert_evaluation.py \
#     --exp_root "$EXP_ROOT" \
#     --exp_name "$EXP_NAME" \
#     --ckpt_fp "$CKPT" \
#     --formulation "$FORMULATION" \
#     --dataset "$DATASET" \
#     --upsampling_ratio "$UPSAMPLING_RATIO" \
#     > "$LOGS_DIR/_$EXP_NAME.out" 2>&1 &

# EXP_NAME="sapphire-$UPSAMPLING_RATIO-X"
# DATASET=sapphire
# python expert_evaluation.py \
#     --exp_root "$EXP_ROOT" \
#     --exp_name "$EXP_NAME" \
#     --ckpt_fp "$CKPT" \
#     --formulation "$FORMULATION" \
#     --dataset "$DATASET" \
#     --upsampling_ratio "$UPSAMPLING_RATIO" \
#     > "$LOGS_DIR/_$EXP_NAME.out" 2>&1 &

# EXP_NAME="silicon-$UPSAMPLING_RATIO-X"
# DATASET=silicon
# python expert_evaluation.py \
#     --exp_root "$EXP_ROOT" \
#     --exp_name "$EXP_NAME" \
#     --ckpt_fp "$CKPT" \
#     --formulation "$FORMULATION" \
#     --dataset "$DATASET" \
#     --upsampling_ratio "$UPSAMPLING_RATIO" \
#     > "$LOGS_DIR/_$EXP_NAME.out" 2>&1 &

# -------------------------------------------
exit