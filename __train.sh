# ---- train standalone-OLDER surrogate model ----

# DEPTH=6
# NUM_HEADS=6
# NUM_BLOCKS=6
# WINDOW_SIZE=8
# DPR=0.1
# NORM_LAYER=torch.nn.LayerNorm

# LOGS_DIR="/playpen/mufan/levi/tianlong-chen-lab/material-super-resolution/__exps__/__logs__"
# EXP_NAME="older_surrogate_mh-VGG-l[-2]+layernorm-adamW-lr=1e-4-augs=True-iii"

# export CUDA_VISIBLE_DEVICES=7
# nohup python train_multi_head_surrogate.py \
#     --exp_name "$EXP_NAME" \
#     --depths $DEPTH \
#     > "$LOGS_DIR/$EXP_NAME.out" 2>&1 &

# ---- train in-filling model ----

DEPTH=6
NUM_HEADS=6
NUM_BLOCKS=6
WINDOW_SIZE=8
DPR=0.1
NORM_LAYER=torch.nn.LayerNorm

LOGS_DIR="/playpen/mufan/levi/tianlong-chen-lab/material-super-resolution/__exps__/__logs__"
EXP_ROOT_DIR="__exps__/y-task-formulations/p(y | y_sparse)/a. train-runs"
SURROGATE_CKPT="/playpen/mufan/levi/tianlong-chen-lab/material-super-resolution/__exps__/y-task-formulations/p(y | y_sparse)/e. surrogate standalone train-runs/2025-02-12_10-49-50_older_surrogate_mh-VGG-l[-2]+layernorm-adamW-lr=1e-4-augs=True-iii/older_surrogate_mh-VGG-l[-2]+layernorm-adamW-lr=1e-4-augs=True-iii_best_older_surrogate.pth"

# export CUDA_VISIBLE_DEVICES=4
# LAMBDA=1e-0
# EXP_NAME="swinir-loss=OLDER-Perceptual-VGG-Multi-Layer+L1(raw-out, y)-lambda=${LAMBDA}"
# python train.py \
#     --exp_name "$EXP_NAME" \
#     --root "$EXP_ROOT_DIR" \
#     --surrogate_weights_file_path "$SURROGATE_CKPT" \
#     --surrogate_loss_mixin $LAMBDA \
#     --depths $DEPTH \
#     --num_heads $NUM_HEADS \
#     --num_blocks $NUM_BLOCKS \
#     --window_size $WINDOW_SIZE \
#     --drop_path_rate $DPR \
#     --norm_layer $NORM_LAYER \
#     > "$LOGS_DIR/_$EXP_NAME.out" 2>&1 &

for i in {1..7}; do
    LAMBDA="1e-${i}"
    device=$(( (i - 1) % 8 ))
    EXP_NAME="swinir-loss=OLDER-Perceptual-VGG-Multi-Layer+L1(raw-out, y)-lambda=${LAMBDA}"
    echo "Launching job for lambda ${LAMBDA} on CUDA device ${device}"
    CUDA_VISIBLE_DEVICES=${device} python train.py \
        --exp_name "$EXP_NAME" \
        --root "$EXP_ROOT_DIR" \
        --surrogate_weights_file_path "$SURROGATE_CKPT" \
        --surrogate_loss_mixin $LAMBDA \
        --depths $DEPTH \
        --num_heads $NUM_HEADS \
        --num_blocks $NUM_BLOCKS \
        --window_size $WINDOW_SIZE \
        --drop_path_rate $DPR \
        --norm_layer $NORM_LAYER \
        > "$LOGS_DIR/_$EXP_NAME.out" 2>&1 &
done

# -------------------------------
exit