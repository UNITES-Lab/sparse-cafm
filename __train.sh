# ---- train standalone-OLDER surrogate model ----
# export CUDA_VISIBLE_DEVICES=2
# nohup python train_multi_head_surrogate.py \
#     --exp_name "older_surrogate_mh-ViT-L16-l[-2]+layernorm-adamW-lr=1e-4" \
#     --depths $DEPTH \
#     > "_older_surrogate_mh-ViT-L16-l[-2]+layernorm-adamW-lr=1e-4.out" 2>&1 &

# ---- train in-filling model ----
# SwinIR hyperparameters
DEPTH=6
NUM_HEADS=6
NUM_BLOCKS=6
WINDOW_SIZE=8
DPR=0.1
NORM_LAYER=torch.nn.LayerNorm

export CUDA_VISIBLE_DEVICES=2
LOGS_DIR="/playpen/mufan/levi/tianlong-chen-lab/material-super-resolution/__exps__/__logs__"
EXP_ROOT_DIR="__exps__/y-task-formulations/p(y | y_sparse)/a. train-runs"
EXP_NAME="swinir-loss=OLDER+L1"
SURROGATE_CKPT="/playpen/mufan/levi/tianlong-chen-lab/material-super-resolution/__exps__/y-task-formulations/p(y | y_sparse)/e. surrogate standalone train-runs/2025-02-08_12-36-35_older_surrogate_mh-ViT-l[-2]+layernorm/older_surrogate_mh-ViT-l[-2]+layernorm_latest_older_surrogate.pth"
nohup python train.py \
    --exp_name "$EXP_NAME" \
    --root "$EXP_ROOT_DIR" \
    --surrogate_weights_file_path "$SURROGATE_CKPT" \
    --depths $DEPTH \
    --num_heads $NUM_HEADS \
    --num_blocks $NUM_BLOCKS \
    --window_size $WINDOW_SIZE \
    --drop_path_rate $DPR \
    --norm_layer $NORM_LAYER \
    > "$LOGS_DIR/_$EXP_NAME.out" 2>&1 &
# -------------------------------
exit