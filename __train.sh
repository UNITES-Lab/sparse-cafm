# ---- Model Hyperparams ----

DEPTH=6
NUM_HEADS=6
NUM_BLOCKS=6
WINDOW_SIZE=8
DPR=0.1
NORM_LAYER=torch.nn.LayerNorm

SURROGATE_FP="/playpen/mufan/levi/tianlong-chen-lab/material-super-resolution/__exps__/expert-surrogates/2025-03-02_15-27-59_{average_surface_current, coverage_percentage, total_area_extended_shapes}-ViT-Trainable-BS=64/{average_surface_current, coverage_percentage, total_area_extended_shapes}-ViT-Trainable-BS=64_best_older_surrogate.pth"
WEIGHTS_FP="/playpen/mufan/levi/tianlong-chen-lab/material-super-resolution/__exps__/foundation-models/3. combined/2x/2025-03-03_11-01-10_DS={all}-[X|y]-BS=32-opt=Adam-lr=1e-5/DS={all}-[X|y]-BS=32-opt=Adam-lr=1e-5_best.pth"
LOGS_DIR="__exps__/__logs__"
EXP_ROOT_DIR="/playpen/mufan/levi/tianlong-chen-lab/material-super-resolution/__exps__/ablations/transfer-learning/pre-trained-models"
EXP_NAME="4x-DS={mos2-sef, silicon}"

UPSAMPLE_FACTOR=4
FORMULATION="both"
DATASET="all"

export CUDA_VISIBLE_DEVICES=2
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