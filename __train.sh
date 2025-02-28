
DEPTH=6
NUM_HEADS=6
NUM_BLOCKS=6
WINDOW_SIZE=8
DPR=0.1
NORM_LAYER=torch.nn.LayerNorm

# ---- train super-resolution model ----

# EXP_ROOT_DIR="/playpen/mufan/levi/tianlong-chen-lab/material-super-resolution/__exps__/substrates/mos2-sef/2x"
# LOGS_DIR="/playpen/mufan/levi/tianlong-chen-lab/material-super-resolution/__exps__/__logs__"
# EXP_NAME="OPTIM=avg_surface_current"

# UPSAMPLE_FACTOR=2
# FORMULATION="y"
# DATASET="mos2-sef"

# python train.py \
#   --exp_name "$EXP_NAME" \
#   --root "$EXP_ROOT_DIR" \
#   --upsample_factor $SR_FACTOR \
#   --dataset $DATASET \
#   --formulation $FORMULATION \
#   --weights $WEIGHTS_FP \
#   --num_heads $NUM_HEADS \
#   --num_blocks $NUM_BLOCKS \
#   --window_size $WINDOW_SIZE \
#   --drop_path_rate $DPR \
#   --norm_layer $NORM_LAYER \
#   > "$LOGS_DIR/_$EXP_NAME.out" 2>&1 &

# ---------------------------------------

EXP_ROOT_DIR="/playpen/mufan/levi/tianlong-chen-lab/material-super-resolution/__exps__/expert-surrogates"
LOGS_DIR="/playpen/mufan/levi/tianlong-chen-lab/material-super-resolution/__exps__/__logs__"
EXP_NAME="average_surface_current_bb=ViT-Trainable"

export CUDA_VISIBLE_DEVICES=5
python train_multi_head_surrogate.py \
  --exp_name "$EXP_NAME" \
  --num_heads $NUM_HEADS \
  --num_blocks $NUM_BLOCKS \
  --window_size $WINDOW_SIZE \
  --drop_path_rate $DPR \
  --norm_layer $NORM_LAYER \
  > "$LOGS_DIR/_$EXP_NAME.out" 2>&1 &

# ---------------------------------------
exit