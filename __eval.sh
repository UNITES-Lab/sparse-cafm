#!/bin/bash
LOGS_DIR="/playpen/mufan/levi/tianlong-chen-lab/material-super-resolution/__exps__/__logs__"

export CUDA_VISIBLE_DEVICES=7
FORMULATION=y

# ----- average surface current -----
# CKPT="__exps__/substrates/mos2-sef/2x/2025-02-28_16-15-47_expert-eval-optim=avg_surface_current-dataset=mos2-sef-[y]-BS=8-optim=Adam-lr=1e-5/expert-eval-optim=avg_surface_current-dataset=mos2-sef-[y]-BS=8-optim=Adam-lr=1e-5_best.pth"

# ----- coverage percentage -----
# CKPT="__exps__/substrates/mos2-sef/2x/2025-03-01_15-26-26_metric=coverage-percentage-ds=mos2-sef-[y]-BS=8-opt=Adam-lr=1e-5/metric=coverage-percentage-ds=mos2-sef-[y]-BS=8-opt=Adam-lr=1e-5_best.pth"

# ----- total area extended shapes -----
DATASET=mos2-sef
CKPT="/playpen/mufan/levi/tianlong-chen-lab/material-super-resolution/__exps__/substrates/mos2-sef/2x/2025-03-02_10-47-29_DS={all}-[X|y]-BS=8-opt=Adam-lr=1e-5/DS={all}-[X|y]-BS=8-opt=Adam-lr=1e-5_best.pth"
UPSAMPLING_RATIO=2
python expert_evaluation.py \
    --ckpt_fp "$CKPT" \
    --formulation "$FORMULATION" \
    --dataset "$DATASET" \
    --upsampling_ratio "$UPSAMPLING_RATIO"

DATASET=sapphire
python expert_evaluation.py \
    --ckpt_fp "$CKPT" \
    --formulation "$FORMULATION" \
    --dataset "$DATASET" \
    --upsampling_ratio "$UPSAMPLING_RATIO"

DATASET=silicon
python expert_evaluation.py \
    --ckpt_fp "$CKPT" \
    --formulation "$FORMULATION" \
    --dataset "$DATASET" \
    --upsampling_ratio "$UPSAMPLING_RATIO"
# -------------------------------------------