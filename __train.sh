# ---- train in-filling model ----

DEPTH=6
NUM_HEADS=6
NUM_BLOCKS=6
WINDOW_SIZE=8
DPR=0.1
NORM_LAYER=torch.nn.LayerNorm
LOGS_DIR="/playpen/mufan/levi/tianlong-chen-lab/material-super-resolution/__exps__/__logs__"
EXP_ROOT_DIR="/playpen/mufan/levi/tianlong-chen-lab/material-super-resolution/__exps__/substrates/mos2-sef/8x"

##### baseline #####

SR_FACTOR=8
EXP_NAME="SwinIR-SR-8X-DS=MoS2-sef-downsampling=bicubic-s48-bs-1-adam-lr=1e-6-loss=L1"
export CUDA_VISIBLE_DEVICES=2
python train.py \
        --exp_name "$EXP_NAME" \
        --root "$EXP_ROOT_DIR" \
        --upsample_factor $SR_FACTOR \
        --num_heads $NUM_HEADS \
        --num_blocks $NUM_BLOCKS \
        --window_size $WINDOW_SIZE \
        --drop_path_rate $DPR \
        --norm_layer $NORM_LAYER \
        > "$LOGS_DIR/_$EXP_NAME.out" 2>&1 &

# -------------------------------
exit