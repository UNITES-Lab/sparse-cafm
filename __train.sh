# ---- train ControlNet ----

# EXP_NAME="ControlNet-MR=16"
# cd _ControlNet
# for i in {5..7}; do
#     export CUDA_VISIBLE_DEVICES=${i}
#     LOGS_DIR="/playpen/mufan/levi/tianlong-chen-lab/material-super-resolution/__exps__/__logs__"
#     EXP_NAME="MOS2-SEF-ControlNet-synthetic-dataset-device=${i}"
#     echo "Launching job for lambda ${LAMBDA} on CUDA device ${i}"
#     python generate_synth_dataset.py > "$LOGS_DIR/$EXP_NAME.out" 2>&1 &
# done

# ---- train standalone-OLDER surrogate model ----

# 1. Predict OLDER using a synthetic dataset #
# LOGS_DIR="/playpen/mufan/levi/tianlong-chen-lab/material-super-resolution/__exps__/__logs__"
# EXP_NAME="DS=Real-BB-VGG-19-Trainable-Optimal-4-Features"
# CUDA_VISIBLE_DEVICES=6 python train_multi_head_surrogate.py \
#     --exp_name "$EXP_NAME" \
#     > "$LOGS_DIR/$EXP_NAME.out" 2>&1 &
 
# ---- train in-filling model ----

DEPTH=6
NUM_HEADS=6
NUM_BLOCKS=6
WINDOW_SIZE=8
DPR=0.1
NORM_LAYER=torch.nn.LayerNorm
LOGS_DIR="/playpen/mufan/levi/tianlong-chen-lab/material-super-resolution/__exps__/__logs__"
EXP_ROOT_DIR="/playpen/mufan/levi/tianlong-chen-lab/material-super-resolution/__exps__/y-task-formulations/8x-sr"

##### baseline #####

export CUDA_VISIBLE_DEVICES=3
EXP_NAME="SwinIR-8x-sr-s48-bs-8-adamw-lr=1e-6-more-augs"
python train.py \
        --exp_name "$EXP_NAME" \
        --root "$EXP_ROOT_DIR" \
        --num_heads $NUM_HEADS \
        --num_blocks $NUM_BLOCKS \
        --window_size $WINDOW_SIZE \
        --drop_path_rate $DPR \
        --norm_layer $NORM_LAYER \
        > "$LOGS_DIR/_$EXP_NAME.out" 2>&1 &


##### mixin ablation #####

# for i in {0..8}; do
#     LAMBDA="1e-${i}"
#     device=$(( (i) % 8 ))
#     EXP_NAME="SwinIR-loss=OLDER-Perc-NEW-VGG+L1(raw-out, y)-lambda=${LAMBDA}"
#     echo "Launching job for lambda ${LAMBDA} on CUDA device ${device}"
#     CUDA_VISIBLE_DEVICES=${device} python train.py \
#         --exp_name "$EXP_NAME" \
#         --root "$EXP_ROOT_DIR" \
#         --surrogate_weights_file_path "$SURROGATE_CKPT" \
#         --surrogate_loss_mixin $LAMBDA \
#         --vgg_feature_layer 4 \
#         --depths $DEPTH \
#         --num_heads $NUM_HEADS \
#         --num_blocks $NUM_BLOCKS \
#         --window_size $WINDOW_SIZE \
#         --drop_path_rate $DPR \
#         --norm_layer $NORM_LAYER \
#         > "$LOGS_DIR/_$EXP_NAME.out" 2>&1 &
# done

##### optimal feature layers ablation #####

# LAYERS=(4 11 24 37 50)
# FIXED_MIXIN=1.0

# for i in "${!LAYERS[@]}"; do
#     layer=${LAYERS[$i]}
#     device=$(( i % 8 ))
#     EXP_NAME="SwinIR-loss=OLDER-Perceptual-VGG-19(raw-out, y)-vgg_layer=${layer}"
#     echo "Launching job for vgg_feature_layer ${layer} on CUDA device ${device}"
#     CUDA_VISIBLE_DEVICES=${device} python train.py \
#         --exp_name "$EXP_NAME" \
#         --root "$EXP_ROOT_DIR" \
#         --surrogate_weights_file_path "$SURROGATE_CKPT" \
#         --surrogate_loss_mixin ${FIXED_MIXIN} \
#         --vgg_feature_layer ${layer} \
#         --depths $DEPTH \
#         --num_heads $NUM_HEADS \
#         --num_blocks $NUM_BLOCKS \
#         --window_size $WINDOW_SIZE \
#         --drop_path_rate $DPR \
#         --norm_layer $NORM_LAYER \
#         > "$LOGS_DIR/_$EXP_NAME.out" 2>&1 &
# done

##########################

##### optimal learning rate ablation  #####

# LRS=(5e-7 1e-6 5e-6 1e-5 5e-5 1e-4 5e-4 1e-3)
# FIXED_MIXIN=1.0

# for i in "${!LRS[@]}"; do
#     LR=${LRS[$i]}
#     device=$(( i % 8 ))
#     EXP_NAME="SwinIR-loss=OLDER-Perceptual-VGG-19(raw-out, y)-LR=${LR}"
#     echo "Launching job for learning_rate ${LR} on CUDA device ${device}"
#     CUDA_VISIBLE_DEVICES=${device} python train.py \
#         --exp_name "$EXP_NAME" \
#         --root "$EXP_ROOT_DIR" \
#         --surrogate_weights_file_path "$SURROGATE_CKPT" \
#         --surrogate_loss_mixin ${FIXED_MIXIN} \
#         --learning_rate ${LR} \
#         --vgg_feature_layer 50 \
#         --depths $DEPTH \
#         --num_heads $NUM_HEADS \
#         --num_blocks $NUM_BLOCKS \
#         --window_size $WINDOW_SIZE \
#         --drop_path_rate $DPR \
#         --norm_layer $NORM_LAYER \
#         > "$LOGS_DIR/_$EXP_NAME.out" 2>&1 &
# done

##########################

##### optimal batch-size ablation  #####

# BSS=(2 3 4 5 6)
# FIXED_MIXIN=1.0

# for i in "${!BSS[@]}"; do
#     BS=${BSS[$i]}
#     device=$(( i % 8 ))
#     EXP_NAME="SwinIR-loss=OLDER-Perceptual-VGG-19(raw-out, y)-BS=${BS}"
#     echo "Launching job for batch_size=${BS} on CUDA device ${device}"
#     CUDA_VISIBLE_DEVICES=${device} python train.py \
#         --exp_name "$EXP_NAME" \
#         --root "$EXP_ROOT_DIR" \
#         --surrogate_weights_file_path "$SURROGATE_CKPT" \
#         --surrogate_loss_mixin ${FIXED_MIXIN} \
#         --learning_rate 1e-4 \
#         --batch_size ${BS} \
#         --vgg_feature_layer 50 \
#         --depths $DEPTH \
#         --num_heads $NUM_HEADS \
#         --num_blocks $NUM_BLOCKS \
#         --window_size $WINDOW_SIZE \
#         --drop_path_rate $DPR \
#         --norm_layer $NORM_LAYER \
#         > "$LOGS_DIR/_$EXP_NAME.out" 2>&1 &
# done

##########################

# -------------------------------
exit