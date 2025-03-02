DEPTH=6
NUM_HEADS=6
NUM_BLOCKS=6
WINDOW_SIZE=8
DPR=0.1
NORM_LAYER=torch.nn.LayerNorm

# ---- train super-resolution model ----

# WEIGHTS_FP=_SwinIR/__weights__/001_classicalSR_DF2K_s64w8_SwinIR-M_x2.pth
WEIGHTS_FP="/playpen/mufan/levi/tianlong-chen-lab/material-super-resolution/__exps__/foundation-models/2. surface-conductivity/2x/2025-02-28_13-59-45_dataset=synth-BS=16-optim=Adam-lr=1e-5/dataset=synth-BS=16-optim=Adam-lr=1e-5_best.pth"

# ---- metric: avg_surface_current ----
# SURROGATE_FP="__exps__/expert-surrogates/2025-02-28_13-03-05_average_surface_current_bb=ViT-Trainable-BS=64/average_surface_current_bb=ViT-Trainable-BS=64_best_older_surrogate.pth"

# ---- metric: coverage_percentage ----
# SURROGATE_FP="__exps__/expert-surrogates/2025-03-01_10-18-29_coverage_percent_bb=ViT-Trainable-BS=64/coverage_percent_bb=ViT-Trainable-BS=64_best_older_surrogate.pth"

# ---- metric: total area extended shapes ----
# SURROGATE_FP="__exps__/expert-surrogates/2025-03-01_10-18-29_coverage_percent_bb=ViT-Trainable-BS=64/coverage_percent_bb=ViT-Trainable-BS=64_best_older_surrogate.pth"

# LOGS_DIR="/playpen/mufan/levi/tianlong-chen-lab/material-super-resolution/__exps__/__logs__"
# EXP_ROOT_DIR="__exps__/substrates/mos2-sef/2x"
# EXP_NAME="DS={all}-[X|y]-BS=8-opt=Adam-lr=1e-5"

# UPSAMPLE_FACTOR=2
# FORMULATION="both"
# DATASET="all"

# export CUDA_VISIBLE_DEVICES=2
# python train.py \
#   --exp_name "$EXP_NAME" \
#   --root "$EXP_ROOT_DIR" \
#   --upsample_factor $UPSAMPLE_FACTOR \
#   --dataset $DATASET \
#   --formulation $FORMULATION \
#   --weights "$WEIGHTS_FP" \
#   --surrogate_weights $SURROGATE_FP \
#   --num_heads $NUM_HEADS \
#   --num_blocks $NUM_BLOCKS \
#   --window_size $WINDOW_SIZE \
#   --drop_path_rate $DPR \
#   --norm_layer $NORM_LAYER \
#   > "$LOGS_DIR/_$EXP_NAME.out" 2>&1 &

# ---------------------------------------

EXP_ROOT_DIR="/playpen/mufan/levi/tianlong-chen-lab/material-super-resolution/__exps__/expert-surrogates"
LOGS_DIR="/playpen/mufan/levi/tianlong-chen-lab/material-super-resolution/__exps__/__logs__"
EXP_NAME="{average_surface_current, coverage_percentage, total_area_extended_shapes}-ViT-Trainable-BS=64"

export CUDA_VISIBLE_DEVICES=3
python train_multi_head_surrogate.py \
  --exp_name "$EXP_NAME" \
  --num_heads $NUM_HEADS \
  --num_blocks $NUM_BLOCKS \
  --window_size $WINDOW_SIZE \
  --drop_path_rate $DPR \
  --norm_layer $NORM_LAYER \
  --average_surface_current \
  --coverage_percentage \
  --total_area_extended_shapes \
  > "$LOGS_DIR/_$EXP_NAME.out" 2>&1 &

# ---------------------------------------
exit