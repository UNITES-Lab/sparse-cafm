DEPTH=32
WINDOW_SIZE=8
NUM_HEADS=6
NUM_BLOCKS=6
DPR=0.1
NORM_LAYER=torch.nn.LayerNorm

# ---- train infilling + OLDER surrogate model ----
export CUDA_VISIBLE_DEVICES=3
nohup python train_surrogate.py \
    --exp_name "infiller-swinir-D=32-loss=sig(OLDER)+L1" \
    --depths $DEPTH \
    > "_infiller-swinir-D=32-loss=sig(OLDER)+L1.out" 2>&1 &
# ------------------------------------------------

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