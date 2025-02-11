LOGS_DIR="/playpen/mufan/levi/tianlong-chen-lab/material-super-resolution/__exps__/__logs__"

# ---- eval SwinIR runs -----
export CUDA_VISIBLE_DEVICES=0
SRC_DIR="/playpen/mufan/levi/tianlong-chen-lab/material-super-resolution/__exps__/y-task-formulations/p(y | y_sparse)/a. train-runs/2025-02-11_09-23-13_swinir-loss=OLDER-Perceptual+L1(raw-out, y)-lambda=1e-1"
EXP_NAME="swinir-loss=OLDER-Perceptual+L1(raw-out, y)-lambda=1e-1"
WEIGHTS_FP="${SRC_DIR}/${EXP_NAME}_best.pth"
nohup python test.py \
    --exp_name "$EXP_NAME" \
    --model_weights_path "$WEIGHTS_FP" \
    > "$LOGS_DIR/_$EXP_NAME.out" 2>&1 &

export CUDA_VISIBLE_DEVICES=1
SRC_DIR="/playpen/mufan/levi/tianlong-chen-lab/material-super-resolution/__exps__/y-task-formulations/p(y | y_sparse)/a. train-runs/2025-02-11_09-23-13_swinir-loss=OLDER-Perceptual+L1(raw-out, y)-lambda=1e-2"
EXP_NAME="swinir-loss=OLDER-Perceptual+L1(raw-out, y)-lambda=1e-2"
WEIGHTS_FP="${SRC_DIR}/${EXP_NAME}_best.pth"
nohup python test.py \
    --exp_name "$EXP_NAME" \
    --model_weights_path "$WEIGHTS_FP" \
    > "$LOGS_DIR/_$EXP_NAME.out" 2>&1 &

export CUDA_VISIBLE_DEVICES=2
SRC_DIR="/playpen/mufan/levi/tianlong-chen-lab/material-super-resolution/__exps__/y-task-formulations/p(y | y_sparse)/a. train-runs/2025-02-11_09-24-27_swinir-loss=OLDER-Perceptual+L1(raw-out, y)-lambda=1e-3"
EXP_NAME="swinir-loss=OLDER-Perceptual+L1(raw-out, y)-lambda=1e-3"
WEIGHTS_FP="${SRC_DIR}/${EXP_NAME}_best.pth"
nohup python test.py \
    --exp_name "$EXP_NAME" \
    --model_weights_path "$WEIGHTS_FP" \
    > "$LOGS_DIR/_$EXP_NAME.out" 2>&1 &

export CUDA_VISIBLE_DEVICES=4
SRC_DIR="/playpen/mufan/levi/tianlong-chen-lab/material-super-resolution/__exps__/y-task-formulations/p(y | y_sparse)/a. train-runs/2025-02-11_09-25-30_swinir-loss=OLDER-Perceptual+L1(raw-out, y)-lambda=1e-4"
EXP_NAME="swinir-loss=OLDER-Perceptual+L1(raw-out, y)-lambda=1e-4"
WEIGHTS_FP="${SRC_DIR}/${EXP_NAME}_best.pth"
nohup python test.py \
    --exp_name "$EXP_NAME" \
    --model_weights_path "$WEIGHTS_FP" \
    > "$LOGS_DIR/_$EXP_NAME.out" 2>&1 &

export CUDA_VISIBLE_DEVICES=5
SRC_DIR="/playpen/mufan/levi/tianlong-chen-lab/material-super-resolution/__exps__/y-task-formulations/p(y | y_sparse)/a. train-runs/2025-02-11_09-27-08_swinir-loss=OLDER-Perceptual+L1(raw-out, y)-lambda=1e-6"
EXP_NAME="swinir-loss=OLDER-Perceptual+L1(raw-out, y)-lambda=1e-6"
WEIGHTS_FP="${SRC_DIR}/${EXP_NAME}_best.pth"
nohup python test.py \
    --exp_name "$EXP_NAME" \
    --model_weights_path "$WEIGHTS_FP" \
    > "$LOGS_DIR/_$EXP_NAME.out" 2>&1 &

export CUDA_VISIBLE_DEVICES=6
SRC_DIR="/playpen/mufan/levi/tianlong-chen-lab/material-super-resolution/__exps__/y-task-formulations/p(y | y_sparse)/a. train-runs/2025-02-11_09-27-09_swinir-loss=OLDER-Perceptual+L1(raw-out, y)-lambda=1e-5"
EXP_NAME="swinir-loss=OLDER-Perceptual+L1(raw-out, y)-lambda=1e-5"
WEIGHTS_FP="${SRC_DIR}/${EXP_NAME}_best.pth"
nohup python test.py \
    --exp_name "$EXP_NAME" \
    --model_weights_path "$WEIGHTS_FP" \
    > "$LOGS_DIR/_$EXP_NAME.out" 2>&1 &

export CUDA_VISIBLE_DEVICES=7
SRC_DIR="/playpen/mufan/levi/tianlong-chen-lab/material-super-resolution/__exps__/y-task-formulations/p(y | y_sparse)/a. train-runs/2025-02-11_09-28-30_swinir-loss=OLDER-Perceptual+L1(raw-out, y)-lambda=1e-7"
EXP_NAME="swinir-loss=OLDER-Perceptual+L1(raw-out, y)-lambda=1e-7"
WEIGHTS_FP="${SRC_DIR}/${EXP_NAME}_best.pth"
nohup python test.py \
    --exp_name "$EXP_NAME" \
    --model_weights_path "$WEIGHTS_FP" \
    > "$LOGS_DIR/_$EXP_NAME.out" 2>&1 &

# ---------------------------------------------
exit