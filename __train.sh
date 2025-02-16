# ---- train ControlNet ----

# EXP_NAME="ControlNet-MR=16"
# cd _ControlNet
# python train.py > "$LOGS_DIR/$EXP_NAME.out" 2>&1 &

# cd _ControlNet
# for i in {2..7}; do
#     export CUDA_VISIBLE_DEVICES=${i}
#     LOGS_DIR="/playpen/mufan/levi/tianlong-chen-lab/material-super-resolution/__exps__/__logs__"
#     EXP_NAME="MOS2-SEF-ControlNet-synthetic-dataset-device=${i}"
#     echo "Launching job for lambda ${LAMBDA} on CUDA device ${i}"
#     python generate_synth_dataset.py > "$LOGS_DIR/$EXP_NAME.out" 2>&1 &
# done

# ---- train standalone-OLDER surrogate model ----

DEPTH=6
NUM_HEADS=6
NUM_BLOCKS=6
WINDOW_SIZE=8
DPR=0.1
NORM_LAYER=torch.nn.LayerNorm

LOGS_DIR="/playpen/mufan/levi/tianlong-chen-lab/material-super-resolution/__exps__/__logs__"
EXP_NAME="DS=Synth-OLDER-surrogate-BB=VGG-19-Frozen"

export CUDA_VISIBLE_DEVICES=7
nohup python train_multi_head_surrogate.py \
    --exp_name "$EXP_NAME" \
    --depths $DEPTH \
    > "$LOGS_DIR/$EXP_NAME.out" 2>&1 &

# ---- train in-filling model ----

# DEPTH=6
# NUM_HEADS=6
# NUM_BLOCKS=6
# WINDOW_SIZE=8
# DPR=0.1
# NORM_LAYER=torch.nn.LayerNorm

# LOGS_DIR="/playpen/mufan/levi/tianlong-chen-lab/material-super-resolution/__exps__/__logs__"
# EXP_ROOT_DIR="__exps__/y-task-formulations/p(y | y_sparse)/a. train-runs"
# SURROGATE_CKPT="/playpen/mufan/levi/tianlong-chen-lab/material-super-resolution/__exps__/y-task-formulations/p(y | y_sparse)/e. surrogate standalone train-runs/2025-02-13_11-37-56_OLDERW_val_loss_fix_surrogate_mh-VGG-layers[-2]+layernorm-adamW-lr=1e-5-augs=False/OLDERW_val_loss_fix_surrogate_mh-VGG-layers[-2]+layernorm-adamW-lr=1e-5-augs=False_best_older_surrogate.pth"

# for i in {0..8}; do
#     LAMBDA="1e-${i}"
#     device=$(( (i) % 8 ))
#     EXP_NAME="SwinIR-loss=OLDER-W-Perceptual-VGG-aug=F-Multi-Layer+L1(raw-out, y)-lambda=${LAMBDA}"
#     echo "Launching job for lambda ${LAMBDA} on CUDA device ${device}"
#     CUDA_VISIBLE_DEVICES=${device} python train.py \
#         --exp_name "$EXP_NAME" \
#         --root "$EXP_ROOT_DIR" \
#         --surrogate_weights_file_path "$SURROGATE_CKPT" \
#         --surrogate_loss_mixin $LAMBDA \
#         --depths $DE`PTH \
#         --num_heads $NUM_HEADS \
#         --num_blocks $NUM_BLOCKS \
#         --window_size $WINDOW_SIZE \
#         --drop_path_rate $DPR \
#         --norm_layer $NORM_LAYER \
#         > "$LOGS_DIR/_$EXP_NAME.out" 2>&1 &
# done

# -------------------------------
exit