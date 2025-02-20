#!/bin/bash
LOGS_DIR="/playpen/mufan/levi/tianlong-chen-lab/material-super-resolution/__exps__/__logs__"

# ---- eval SwinIR runs -----

# export CUDA_VISIBLE_DEVICES=0
# SRC_DIR="/playpen/mufan/levi/tianlong-chen-lab/material-super-resolution/__exps__/y-task-formulations/p(y | y_sparse)/a. train-runs/2025-02-18_16-29-36_SwinIR-loss=L1(raw-out, y)"
# EXP_NAME="SwinIR-loss=L1(raw-out, y)"
# WEIGHTS_FP="${SRC_DIR}/${EXP_NAME}_best.pth"
# python test.py \
#     --exp_name "$EXP_NAME" \
#     --model_weights_path "$WEIGHTS_FP" \
#     # > "$LOGS_DIR/_$EXP_NAME.out" 2>&1 &

# ---- batch job -----

ROOT_DIR="/playpen/mufan/levi/tianlong-chen-lab/material-super-resolution/__exps__/y-task-formulations/p(y | y_sparse)/a. train-runs"
device=0
for SRC_DIR in "$ROOT_DIR"/*02-20*; do
    if [ -d "$SRC_DIR" ]; then
        BEST_FILE=$(find "$SRC_DIR" -maxdepth 1 -type f -name "*_best.pth" | head -n 1)
        if [ -z "$BEST_FILE" ]; then
            echo "No _best.pth file found in $SRC_DIR. Skipping."
            continue
        fi
        EXP_NAME=$(basename "$BEST_FILE")
        EXP_NAME=${EXP_NAME%_best.pth}
        WEIGHTS_FP="$BEST_FILE"
        echo "Running test.py for experiment: $EXP_NAME on GPU $device in directory: $SRC_DIR"
        CUDA_VISIBLE_DEVICES=$device nohup python test.py \
            --exp_name "$EXP_NAME" \
            --model_weights_path "$WEIGHTS_FP" \
            > "$LOGS_DIR/_${EXP_NAME}.out" 2>&1 &
        
        device=$(( (device + 1) % 8 ))
    fi
done

# ---------------------------------------------
# exit