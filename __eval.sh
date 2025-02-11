LOGS_DIR="/playpen/mufan/levi/tianlong-chen-lab/material-super-resolution/__exps__/__logs__"

# ---- eval SwinIR runs -----
export CUDA_VISIBLE_DEVICES=0
SRC_DIR="/playpen/mufan/levi/tianlong-chen-lab/material-super-resolution/__exps__/y-task-formulations/p(y | y_sparse)/a. train-runs/2025-02-11_12-33-51_swinir-loss=OLDER-Perceptual-Multi-Layer+L1(raw-out, y)-lambda=1e-0"
EXP_NAME="swinir-loss=OLDER-Perceptual-Multi-Layer+L1(raw-out, y)-lambda=1e-0"
WEIGHTS_FP="${SRC_DIR}/${EXP_NAME}_best.pth"
nohup python test.py \
    --exp_name "$EXP_NAME" \
    --model_weights_path "$WEIGHTS_FP" \
    > "$LOGS_DIR/_$EXP_NAME.out" 2>&1 &

# ---------------------------------------------
exit