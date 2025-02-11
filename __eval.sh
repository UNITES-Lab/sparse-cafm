LOGS_DIR="/playpen/mufan/levi/tianlong-chen-lab/material-super-resolution/__exps__/__logs__"

# ---- eval SwinIR runs -----
export CUDA_VISIBLE_DEVICES=0
SRC_DIR="/playpen/mufan/levi/tianlong-chen-lab/material-super-resolution/__exps__/y-task-formulations/p(y | y_sparse)/a. train-runs/2025-02-10_10-55-33_swinir-loss=OLDER-Perceptual(raw-out, y)+L1(raw-out, y)"
EXP_NAME="swinir-loss=OLDER-Perceptual(raw-out, y)+L1(raw-out, y)"
WEIGHTS_FP="${SRC_DIR}/${EXP_NAME}_best.pth"
nohup python test.py \
    --exp_name "$EXP_NAME" \
    --model_weights_path "$WEIGHTS_FP" \
    > "$LOGS_DIR/_$EXP_NAME.out" 2>&1 &

export CUDA_VISIBLE_DEVICES=1
SRC_DIR="/playpen/mufan/levi/tianlong-chen-lab/material-super-resolution/__exps__/y-task-formulations/p(y | y_sparse)/a. train-runs/2025-02-10_10-53-35_swinir-loss=OLDER-Perceptual(raw-out, y)"
EXP_NAME="swinir-loss=OLDER-Perceptual(raw-out, y)"
WEIGHTS_FP="${SRC_DIR}/${EXP_NAME}_best.pth"
nohup python test.py \
    --exp_name "$EXP_NAME" \
    --model_weights_path "$WEIGHTS_FP" \
    > "$LOGS_DIR/_$EXP_NAME.out" 2>&1 &

# ---------------------------------------------
exit