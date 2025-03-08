#!/bin/bash
LOGS_DIR="/playpen/mufan/levi/tianlong-chen-lab/material-super-resolution/__exps__/__logs__"
export CUDA_VISIBLE_DEVICES=0

# ---- surface current ----
FORMULATION=y
EXP_ROOT="/playpen/mufan/levi/tianlong-chen-lab/material-super-resolution/__exps__/foundation-models/4. combined-evals"

# ---- 2x2 == 4x upscaling ----
CKPT="/playpen/mufan/levi/tianlong-chen-lab/material-super-resolution/__exps__/foundation-models/3. combined/2x/2025-03-04_11-14-17_DS={all}-[X|y]-BS=32-opt=Adam-lr=1e-5/DS={all}-[X|y]-BS=32-opt=Adam-lr=1e-5_best.pth"
UPSAMPLING_RATIO=2

EXP_NAME="mos2-$UPSAMPLING_RATIO-X"
DATASET=mos2-sef
python expert_evaluation.py \
    --exp_root "$EXP_ROOT" \
    --exp_name "$EXP_NAME" \
    --ckpt_fp "$CKPT" \
    --formulation "$FORMULATION" \
    --dataset "$DATASET" \
    --upsampling_ratio "$UPSAMPLING_RATIO"

EXP_NAME="sapphire-$UPSAMPLING_RATIO-X"
DATASET=sapphire
python expert_evaluation.py \
    --exp_root "$EXP_ROOT" \
    --exp_name "$EXP_NAME" \
    --ckpt_fp "$CKPT" \
    --formulation "$FORMULATION" \
    --dataset "$DATASET" \
    --upsampling_ratio "$UPSAMPLING_RATIO"

EXP_NAME="silicon-$UPSAMPLING_RATIO-X"
DATASET=silicon
python expert_evaluation.py \
    --exp_root "$EXP_ROOT" \
    --exp_name "$EXP_NAME" \
    --ckpt_fp "$CKPT" \
    --formulation "$FORMULATION" \
    --dataset "$DATASET" \
    --upsampling_ratio "$UPSAMPLING_RATIO"

# ---- 4x4 == 16x upscaling ----
CKPT="/playpen/mufan/levi/tianlong-chen-lab/material-super-resolution/__exps__/foundation-models/3. combined/4x/2025-03-03_10-59-32_DS={all}-[X|y]-BS=32-opt=Adam-lr=1e-5/DS={all}-[X|y]-BS=32-opt=Adam-lr=1e-5_best.pth"
UPSAMPLING_RATIO=4

EXP_NAME="mos2-$UPSAMPLING_RATIO-X"
DATASET=mos2-sef
python expert_evaluation.py \
    --exp_root "$EXP_ROOT" \
    --exp_name "$EXP_NAME" \
    --ckpt_fp "$CKPT" \
    --formulation "$FORMULATION" \
    --dataset "$DATASET" \
    --upsampling_ratio "$UPSAMPLING_RATIO"

EXP_NAME="sapphire-$UPSAMPLING_RATIO-X"
DATASET=sapphire
python expert_evaluation.py \
    --exp_root "$EXP_ROOT" \
    --exp_name "$EXP_NAME" \
    --ckpt_fp "$CKPT" \
    --formulation "$FORMULATION" \
    --dataset "$DATASET" \
    --upsampling_ratio "$UPSAMPLING_RATIO"

EXP_NAME="silicon-$UPSAMPLING_RATIO-X"
DATASET=silicon
python expert_evaluation.py \
    --exp_root "$EXP_ROOT" \
    --exp_name "$EXP_NAME" \
    --ckpt_fp "$CKPT" \
    --formulation "$FORMULATION" \
    --dataset "$DATASET" \
    --upsampling_ratio "$UPSAMPLING_RATIO"

# ---- 8x8 == 64x upscaling ----
CKPT="/playpen/mufan/levi/tianlong-chen-lab/material-super-resolution/__exps__/foundation-models/3. combined/8x/2025-03-03_10-56-59_DS={all}-[X|y]-BS=32-opt=Adam-lr=1e-5/DS={all}-[X|y]-BS=32-opt=Adam-lr=1e-5_best.pth"
UPSAMPLING_RATIO=8

EXP_NAME="mos2-$UPSAMPLING_RATIO-X"
DATASET=mos2-sef
python expert_evaluation.py \
    --exp_root "$EXP_ROOT" \
    --exp_name "$EXP_NAME" \
    --ckpt_fp "$CKPT" \
    --formulation "$FORMULATION" \
    --dataset "$DATASET" \
    --upsampling_ratio "$UPSAMPLING_RATIO"

EXP_NAME="sapphire-$UPSAMPLING_RATIO-X"
DATASET=sapphire
python expert_evaluation.py \
    --exp_root "$EXP_ROOT" \
    --exp_name "$EXP_NAME" \
    --ckpt_fp "$CKPT" \
    --formulation "$FORMULATION" \
    --dataset "$DATASET" \
    --upsampling_ratio "$UPSAMPLING_RATIO"

EXP_NAME="silicon-$UPSAMPLING_RATIO-X"
DATASET=silicon
python expert_evaluation.py \
    --exp_root "$EXP_ROOT" \
    --exp_name "$EXP_NAME" \
    --ckpt_fp "$CKPT" \
    --formulation "$FORMULATION" \
    --dataset "$DATASET" \
    --upsampling_ratio "$UPSAMPLING_RATIO"
# -------------------------------------------