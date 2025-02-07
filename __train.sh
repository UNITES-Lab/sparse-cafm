DEPTH=32
WINDOW_SIZE=8
NUM_HEADS=6
NUM_BLOCKS=6
DPR=0.1
NORM_LAYER=torch.nn.LayerNorm

# ---- train standalone-OLDER surrogate model ----
export CUDA_VISIBLE_DEVICES=2
nohup python train_multi_head_surrogate.py \
    --exp_name "older_surrogate_mh-1x-head-layer" \
    --depths $DEPTH \
    > "_older_surrogate_mh-1x-head-layer.out" 2>&1 &

# ---- train in-filling model ----
# export CUDA_VISIBLE_DEVICES=0
# nohup python train.py \
#     --exp_name "swinir-depth=32" \
#     --root "__exps__/y-task-formulations/p(y | y_sparse)/a. train-runs" \
#     --depths $DEPTH \
#     --num_heads $NUM_HEADS \
#     --num_blocks $NUM_BLOCKS \
#     --window_size $WINDOW_SIZE \
#     --drop_path_rate $DPR \
#     --norm_layer $NORM_LAYER \
#     > "_swinir-depth=32-p(y|y_sparse).out" 2>&1 &
# -------------------------------
exit