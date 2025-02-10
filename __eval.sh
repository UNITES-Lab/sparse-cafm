LOGS_DIR="/playpen/mufan/levi/tianlong-chen-lab/material-super-resolution/__exps__/__logs__"

# ---- eval SwinIR runs -----
export CUDA_VISIBLE_DEVICES=0
SRC_DIR="/playpen/mufan/levi/tianlong-chen-lab/material-super-resolution/__exps__/y-task-formulations/p(y | y_sparse)/a. train-runs/2025-02-09/2025-02-09_13-17-15_swinir-loss=L1(raw_out,y)-surrogate_mode=train"
EXP_NAME="swinir-loss=L1(raw_out,y)-surrogate_mode=train"
WEIGHTS_FP="${SRC_DIR}/${EXP_NAME}_best.pth"
nohup python test.py \
    --exp_name "$EXP_NAME" \
    --model_weights_path "$WEIGHTS_FP" \
    > "$LOGS_DIR/_$EXP_NAME.out" 2>&1 &

export CUDA_VISIBLE_DEVICES=1
SRC_DIR="/playpen/mufan/levi/tianlong-chen-lab/material-super-resolution/__exps__/y-task-formulations/p(y | y_sparse)/a. train-runs/2025-02-09/2025-02-09_13-18-08_swinir-loss=OLDER(raw_out,y)-surrogate_mode=train"
EXP_NAME="swinir-loss=OLDER(raw_out,y)-surrogate_mode=train"
WEIGHTS_FP="${SRC_DIR}/${EXP_NAME}_best.pth"
nohup python test.py \
    --exp_name "$EXP_NAME" \
    --model_weights_path "$WEIGHTS_FP" \
    > "$LOGS_DIR/_$EXP_NAME.out" 2>&1 &

export CUDA_VISIBLE_DEVICES=2
SRC_DIR="/playpen/mufan/levi/tianlong-chen-lab/material-super-resolution/__exps__/y-task-formulations/p(y | y_sparse)/a. train-runs/2025-02-09/2025-02-09_13-19-11_swinir-loss=OLDER+L1(raw_out,y)-surrogate_mode=train"
EXP_NAME="swinir-loss=OLDER+L1(raw_out,y)-surrogate_mode=train"
WEIGHTS_FP="${SRC_DIR}/${EXP_NAME}_best.pth"
nohup python test.py \
    --exp_name "$EXP_NAME" \
    --model_weights_path "$WEIGHTS_FP" \
    > "$LOGS_DIR/_$EXP_NAME.out" 2>&1 &

export CUDA_VISIBLE_DEVICES=3
SRC_DIR="/playpen/mufan/levi/tianlong-chen-lab/material-super-resolution/__exps__/y-task-formulations/p(y | y_sparse)/a. train-runs/2025-02-09/2025-02-09_13-23-47_swinir-loss=L1(y_hat, y)-surrogate_mode=eval"
EXP_NAME="swinir-loss=L1(y_hat, y)-surrogate_mode=eval"
WEIGHTS_FP="${SRC_DIR}/${EXP_NAME}_best.pth"
nohup python test.py \
    --exp_name "$EXP_NAME" \
    --model_weights_path "$WEIGHTS_FP" \
    > "$LOGS_DIR/_$EXP_NAME.out" 2>&1 &

export CUDA_VISIBLE_DEVICES=4
SRC_DIR="/playpen/mufan/levi/tianlong-chen-lab/material-super-resolution/__exps__/y-task-formulations/p(y | y_sparse)/a. train-runs/2025-02-09/2025-02-09_13-24-07_swinir-loss=OLDER(y_hat, y)-surrogate_mode=eval"
EXP_NAME="swinir-loss=OLDER(y_hat, y)-surrogate_mode=eval"
WEIGHTS_FP="${SRC_DIR}/${EXP_NAME}_best.pth"
nohup python test.py \
    --exp_name "$EXP_NAME" \
    --model_weights_path "$WEIGHTS_FP" \
    > "$LOGS_DIR/_$EXP_NAME.out" 2>&1 &

export CUDA_VISIBLE_DEVICES=5
SRC_DIR="/playpen/mufan/levi/tianlong-chen-lab/material-super-resolution/__exps__/y-task-formulations/p(y | y_sparse)/a. train-runs/2025-02-09/2025-02-09_13-24-39_swinir-loss=OLDER+L1(y_hat, y)-surrogate_mode=eval"
EXP_NAME="swinir-loss=OLDER+L1(y_hat, y)-surrogate_mode=eval"
WEIGHTS_FP="${SRC_DIR}/${EXP_NAME}_best.pth"
nohup python test.py \
    --exp_name "$EXP_NAME" \
    --model_weights_path "$WEIGHTS_FP" \
    > "$LOGS_DIR/_$EXP_NAME.out" 2>&1 &

# ---------------------------------------------
exit