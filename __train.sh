# ---- train surrogate model ----
# export CUDA_VISIBLE_DEVICES=6
# DEPTH=8
# nohup python train_surrogate.py \
#     --exp_name surrogate-depth=$DEPTH \
#     --depths $DEPTH \
#     > _surrogate=$DEPTH.out 2>&1 &
# -------------------------------

# ---- train in-filling model ----
export CUDA_VISIBLE_DEVICES=0
DEPTH=32
NUM_BLOCKS=1
nohup python train.py \
    --exp_name swinir-depth=$DEPTH \
    --depths $DEPTH \
    --num_blocks $NUM_BLOCKS \
    > _swinir_depths=$DEPTH.out 2>&1 &

export CUDA_VISIBLE_DEVICES=1
DEPTH=32
NUM_BLOCKS=3
nohup python train.py \
    --exp_name swinir-depth=$DEPTH \
    --depths $DEPTH \
    --num_blocks $NUM_BLOCKS \
    > _swinir_depths=$DEPTH.out 2>&1 &

exit