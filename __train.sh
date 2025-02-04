# ---- train surrogate model ----
# export CUDA_VISIBLE_DEVICES=6
# DEPTH=8
# nohup python train_surrogate.py \
#     --exp_name surrogate-depth=$DEPTH \
#     --depths $DEPTH \
#     > _surrogate=$DEPTH.out 2>&1 &
# -------------------------------

# ---- train in-filling model ----
DEPTH=32
WINDOW_SIZE=8
NUM_HEADS=6
NUM_BLOCKS=6
DPR=0.1
NORM_LAYER=torch.nn.LayerNorm

export CUDA_VISIBLE_DEVICES=2

nohup python train.py \
    --exp_name "swinir-depth=32" \
    --depths $DEPTH \
    --num_heads $NUM_HEADS \
    --num_blocks $NUM_BLOCKS \
    --window_size $WINDOW_SIZE \
    --drop_path_rate $DPR \
    --norm_layer $NORM_LAYER \
    > "_swinir-move-depth=32.out" 2>&1 &
# -------------------------------

exit