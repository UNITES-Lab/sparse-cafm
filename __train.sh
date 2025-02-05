# ---- train infilling + OLDER surrogate model ----
DEPTH=6
WINDOW_SIZE=8
NUM_HEADS=6
NUM_BLOCKS=6
DPR=0.1
NORM_LAYER=torch.nn.LayerNorm

export CUDA_VISIBLE_DEVICES=3
nohup python train_surrogate.py \
    --exp_name "surrogate-swinir-sigmoid-sig(older)+L1" \
    --depths $DEPTH \
    > "_surrogate-swinir-sigmoid-sig(older)+L1.out" 2>&1 &
# ------------------------------------------------

# ---- train in-filling model ----
# export CUDA_VISIBLE_DEVICES=0
# nohup python train.py \
#     --exp_name "swinir-depth=32-sigmoid" \
#     --depths $DEPTH \
#     --num_heads $NUM_HEADS \
#     --num_blocks $NUM_BLOCKS \
#     --window_size $WINDOW_SIZE \
#     --drop_path_rate $DPR \
#     --norm_layer $NORM_LAYER \
#     > "_swinir-depth=32.out" 2>&1 &

# -------------------------------
exit