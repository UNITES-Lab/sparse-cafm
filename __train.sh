# ---- Model Hyperparams ----

DEPTH=6
NUM_HEADS=6
NUM_BLOCKS=6
WINDOW_SIZE=8
DPR=0.1
NORM_LAYER=torch.nn.LayerNorm

SURROGATE_FP="__exps__/expert-surrogates/2025-03-02_15-27-59_{average_surface_current, coverage_percentage, total_area_extended_shapes}-ViT-Trainable-BS=64/{average_surface_current, coverage_percentage, total_area_extended_shapes}-ViT-Trainable-BS=64_best_older_surrogate.pth"
EXP_ROOT_DIR="__exps__/scratch-runs/mos2-sef/current-map/2x"
WEIGHTS_FP="__weights__/8x/DS={all}-[X|y]-BS=32-opt=Adam-lr=1e-5_best.pth"

LOGS_DIR="__exps__/__logs__"
EXP_NAME="mos2-y-2x-UNet"

UPSAMPLE_FACTOR=2
FORMULATION="y"
DATASET="mos2-sef"

export CUDA_VISIBLE_DEVICES=6
python train.py \
  --exp_name "$EXP_NAME" \
  --root "$EXP_ROOT_DIR" \
  --upsample_factor $UPSAMPLE_FACTOR \
  --dataset $DATASET \
  --formulation $FORMULATION \
  --weights "$WEIGHTS_FP" \
  --surrogate_weights "$SURROGATE_FP" \
  --num_heads $NUM_HEADS \
  --num_blocks $NUM_BLOCKS \
  --window_size $WINDOW_SIZE \
  --drop_path_rate $DPR \
  --norm_layer $NORM_LAYER \
  > "$LOGS_DIR/_$EXP_NAME.out" 2>&1 &

# ---------------------------------------
exit