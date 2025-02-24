# ---- train in-filling model ----

DEPTH=6
NUM_HEADS=6
NUM_BLOCKS=6
WINDOW_SIZE=8
DPR=0.1
NORM_LAYER=torch.nn.LayerNorm
LOGS_DIR="/playpen/mufan/levi/tianlong-chen-lab/material-super-resolution/__exps__/__logs__"
EXP_ROOT_DIR="/playpen/mufan/levi/tianlong-chen-lab/material-super-resolution/__exps__/substrates/mos2-sef/2x"

##### baseline #####

SR_FACTOR=2
EXP_NAME="SwinIR-2x-sr-s64-bs-1-adamw-lr=1e-6-more-augs"
export CUDA_VISIBLE_DEVICES=1
python train.py \
        --exp_name "$EXP_NAME" \
        --root "$EXP_ROOT_DIR" \
        --upsampling_factor $SR_FACTOR \
        --num_heads $NUM_HEADS \
        --num_blocks $NUM_BLOCKS \
        --window_size $WINDOW_SIZE \
        --drop_path_rate $DPR \
        --norm_layer $NORM_LAYER \
        > "$LOGS_DIR/_$EXP_NAME.out" 2>&1 &

# -------------------------------
exit