#!/bin/bash
LOGS_DIR="/playpen/mufan/levi/tianlong-chen-lab/material-super-resolution/__exps__/__logs__"

# CKPT="__exps__/substrates/mos2-sef/2x/2025-02-28_16-15-47_expert-eval-optim=avg_surface_current-dataset=mos2-sef-[y]-BS=8-optim=Adam-lr=1e-5/expert-eval-optim=avg_surface_current-dataset=mos2-sef-[y]-BS=8-optim=Adam-lr=1e-5_best.pth"
CKPT="/playpen/mufan/levi/tianlong-chen-lab/material-super-resolution/__exps__/foundation-models/1. surface-morphology/2x/2025-02-27_15-27-17_SwinIR-SR-2X-DS=MoS2-synth-downsampling=bicubic-s64-bs-1-adam-lr=1e-6-loss=L1/SwinIR-SR-2X-DS=MoS2-synth-downsampling=bicubic-s64-bs-1-adam-lr=1e-6-loss=L1_best.pth"
FORMULATION=X
DATASET=mos2-sef
UPSAMPLING_RATIO=2

export CUDA_VISIBLE_DEVICES=0
python expert_evaluation.py \
    --ckpt_fp "$CKPT" \
    --formulation "$FORMULATION" \
    --dataset "$DATASET" \
    --upsampling_ratio "$UPSAMPLING_RATIO"