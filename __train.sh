# ---- train in-filling model ----

DEPTH=6
NUM_HEADS=6
NUM_BLOCKS=6
WINDOW_SIZE=8
DPR=0.1
NORM_LAYER=torch.nn.LayerNorm
LOGS_DIR="/playpen/mufan/levi/tianlong-chen-lab/material-super-resolution/__exps__/__logs__"

UPSAMPLE_FACTOR=2
FORMULATION="y"
DATASET="synth"

python train.py \
  --exp_name "$EXP_NAME" \
  --root "$EXP_ROOT_DIR" \
  --upsample_factor $SR_FACTOR \
  --dataset $DATASET \
  --formulation $FORMULATION \
  --weights $WEIGHTS_FP \
  --num_heads $NUM_HEADS \
  --num_blocks $NUM_BLOCKS \
  --window_size $WINDOW_SIZE \
  --drop_path_rate $DPR \
  --norm_layer $NORM_LAYER \
  > "$LOGS_DIR/_$EXP_NAME.out" 2>&1 &

# ##### baseline #####

# # Datasets: ['mos2-sef', 'sapphire', 'silicon']
# # ---------------------------------------------

# FORMULATION=X
# WEIGHTS_FP=/playpen/mufan/levi/tianlong-chen-lab/material-super-resolution/__exps__/foundation-models/1. surface-morphology/2x/2025-02-27_11-15-53_SwinIR-SR-2X-DS=MoS2-synth-downsampling=bicubic-s64-bs-1-adam-lr=1e-6-loss=L1/SwinIR-SR-2X-DS=MoS2-synth-downsampling=bicubic-s64-bs-1-adam-lr=1e-6-loss=L1_best.pth
# # WEIGHTS_FP=/playpen/mufan/levi/tianlong-chen-lab/material-super-resolution/__exps__/foundation-models/1. surface-morphology/4x/2025-02-27_11-16-52_SwinIR-SR-4X-DS=MoS2-synth-downsampling=bicubic-s64-bs-1-adam-lr=1e-6-loss=L1/SwinIR-SR-4X-DS=MoS2-synth-downsampling=bicubic-s64-bs-1-adam-lr=1e-6-loss=L1_best.pth
# # WEIGHTS_FP=/playpen/mufan/levi/tianlong-chen-lab/material-super-resolution/__exps__/foundation-models/1. surface-morphology/8x/2025-02-27_11-18-12_SwinIR-SR-8X-DS=MoS2-synth-downsampling=bicubic-s48-bs-1-adam-lr=1e-6-loss=L1/SwinIR-SR-8X-DS=MoS2-synth-downsampling=bicubic-s48-bs-1-adam-lr=1e-6-loss=L1_best.pth

# # FORMULATION=y
# # WEIGHTS_FP=/playpen/mufan/levi/tianlong-chen-lab/material-super-resolution/__exps__/foundation-models/2. surface-conductivity/2x/2025-02-27_11-10-31_SwinIR-SR-2X-DS=MoS2-synth-downsampling=bicubic-s64-bs-1-adam-lr=1e-6-loss=L1/SwinIR-SR-2X-DS=MoS2-synth-downsampling=bicubic-s64-bs-1-adam-lr=1e-6-loss=L1_best.pth
# # WEIGHTS_FP=/playpen/mufan/levi/tianlong-chen-lab/material-super-resolution/__exps__/foundation-models/2. surface-conductivity/4x/2025-02-27_11-12-01_SwinIR-SR-4X-DS=MoS2-synth-downsampling=bicubic-s64-bs-1-adam-lr=1e-6-loss=L1/SwinIR-SR-4X-DS=MoS2-synth-downsampling=bicubic-s64-bs-1-adam-lr=1e-6-loss=L1_best.pth
# # WEIGHTS_FP=/playpen/mufan/levi/tianlong-chen-lab/material-super-resolution/__exps__/foundation-models/2. surface-conductivity/8x/2025-02-27_11-13-29_SwinIR-SR-8X-DS=MoS2-synth-downsampling=bicubic-s48-bs-1-adam-lr=1e-6-loss=L1/SwinIR-SR-8X-DS=MoS2-synth-downsampling=bicubic-s48-bs-1-adam-lr=1e-6-loss=L1_best.pth

# # ---------------------------------------------

# SR_FACTOR=2
# DATASET="mos2-sef"
# FORMULATION=y

# EXP_ROOT_DIR="/playpen/mufan/levi/tianlong-chen-lab/material-super-resolution/__exps__/foundation-models/1. surface-morphology/2x"
# EXP_NAME="SwinIR-SR-2X-DS=MoS2-synth-downsampling=bicubic-s64-bs-1-adam-lr=1e-6-loss=L1"
# WEIGHTS_FP="/playpen/mufan/levi/tianlong-chen-lab/material-super-resolution/__exps__/foundation-models/1. surface-morphology/2x/2025-02-27_11-15-53_SwinIR-SR-2X-DS=MoS2-synth-downsampling=bicubic-s64-bs-1-adam-lr=1e-6-loss=L1/SwinIR-SR-2X-DS=MoS2-synth-downsampling=bicubic-s64-bs-1-adam-lr=1e-6-loss=L1_best.pth"

# export CUDA_VISIBLE_DEVICES=0
# python train.py \
#         --exp_name "$EXP_NAME" \
#         --root "$EXP_ROOT_DIR" \
#         --upsample_factor $SR_FACTOR \
#         --dataset $DATASET \
#         --formulation $FORMULATION \
#         --weights $WEIGHTS_FP \
#         --num_heads $NUM_HEADS \
#         --num_blocks $NUM_BLOCKS \
#         --window_size $WINDOW_SIZE \
#         --drop_path_rate $DPR \
#         --norm_layer $NORM_LAYER \
#         > "$LOGS_DIR/_$EXP_NAME.out" 2>&1 &

# # -------------------------------
# exit

#!/bin/bash

# ---- train in-filling model ----

# DEPTH=6
# NUM_HEADS=6
# NUM_BLOCKS=6
# WINDOW_SIZE=8
# DPR=0.1
# NORM_LAYER=torch.nn.LayerNorm
# LOGS_DIR="/playpen/mufan/levi/tianlong-chen-lab/material-super-resolution/__exps__/__logs__"

# # Datasets to iterate over
# datasets=("mos2-sef" "sapphire" "silicon")

# # Formulations: "X" and "y"
# formulations=("X" "y")

# # We'll iterate over SR factors: 2, 4, 8.
# # Note: The exact EXP_ROOT_DIR, EXP_NAME, and WEIGHTS_FP values come directly from the provided file paths.

# counter=0
# for dataset in "${datasets[@]}"; do
#   for formulation in "${formulations[@]}"; do
#     for sr in 2 4 8; do
      
#       # Set configuration based on formulation and SR factor
#       if [ "$formulation" == "X" ]; then
#         if [ "$sr" -eq 2 ]; then
#           EXP_ROOT_DIR="/playpen/mufan/levi/tianlong-chen-lab/material-super-resolution/__exps__/foundation-models/1. surface-morphology/2x"
#           EXP_NAME="SwinIR-SR-2X-DS=MoS2-synth-downsampling=bicubic-s64-bs-1-adam-lr=1e-6-loss=L1"
#           WEIGHTS_FP="/playpen/mufan/levi/tianlong-chen-lab/material-super-resolution/__exps__/foundation-models/1. surface-morphology/2x/2025-02-27_11-15-53_SwinIR-SR-2X-DS=MoS2-synth-downsampling=bicubic-s64-bs-1-adam-lr=1e-6-loss=L1/SwinIR-SR-2X-DS=MoS2-synth-downsampling=bicubic-s64-bs-1-adam-lr=1e-6-loss=L1_best.pth"
#         elif [ "$sr" -eq 4 ]; then
#           EXP_ROOT_DIR="/playpen/mufan/levi/tianlong-chen-lab/material-super-resolution/__exps__/foundation-models/1. surface-morphology/4x"
#           EXP_NAME="SwinIR-SR-4X-DS=MoS2-synth-downsampling=bicubic-s64-bs-1-adam-lr=1e-6-loss=L1"
#           WEIGHTS_FP="/playpen/mufan/levi/tianlong-chen-lab/material-super-resolution/__exps__/foundation-models/1. surface-morphology/4x/2025-02-27_11-16-52_SwinIR-SR-4X-DS=MoS2-synth-downsampling=bicubic-s64-bs-1-adam-lr=1e-6-loss=L1/SwinIR-SR-4X-DS=MoS2-synth-downsampling=bicubic-s64-bs-1-adam-lr=1e-6-loss=L1_best.pth"
#         elif [ "$sr" -eq 8 ]; then
#           EXP_ROOT_DIR="/playpen/mufan/levi/tianlong-chen-lab/material-super-resolution/__exps__/foundation-models/1. surface-morphology/8x"
#           EXP_NAME="SwinIR-SR-8X-DS=MoS2-synth-downsampling=bicubic-s48-bs-1-adam-lr=1e-6-loss=L1"
#           WEIGHTS_FP="/playpen/mufan/levi/tianlong-chen-lab/material-super-resolution/__exps__/foundation-models/1. surface-morphology/8x/2025-02-27_11-18-12_SwinIR-SR-8X-DS=MoS2-synth-downsampling=bicubic-s48-bs-1-adam-lr=1e-6-loss=L1/SwinIR-SR-8X-DS=MoS2-synth-downsampling=bicubic-s48-bs-1-adam-lr=1e-6-loss=L1_best.pth"
#         fi
#       elif [ "$formulation" == "y" ]; then
#         if [ "$sr" -eq 2 ]; then
#           EXP_ROOT_DIR="/playpen/mufan/levi/tianlong-chen-lab/material-super-resolution/__exps__/foundation-models/2. surface-conductivity/2x"
#           EXP_NAME="SwinIR-SR-2X-DS=MoS2-synth-downsampling=bicubic-s64-bs-1-adam-lr=1e-6-loss=L1"
#           WEIGHTS_FP="/playpen/mufan/levi/tianlong-chen-lab/material-super-resolution/__exps__/foundation-models/2. surface-conductivity/2x/2025-02-27_11-10-31_SwinIR-SR-2X-DS=MoS2-synth-downsampling=bicubic-s64-bs-1-adam-lr=1e-6-loss=L1/SwinIR-SR-2X-DS=MoS2-synth-downsampling=bicubic-s64-bs-1-adam-lr=1e-6-loss=L1_best.pth"
#         elif [ "$sr" -eq 4 ]; then
#           EXP_ROOT_DIR="/playpen/mufan/levi/tianlong-chen-lab/material-super-resolution/__exps__/foundation-models/2. surface-conductivity/4x"
#           EXP_NAME="SwinIR-SR-4X-DS=MoS2-synth-downsampling=bicubic-s64-bs-1-adam-lr=1e-6-loss=L1"
#           WEIGHTS_FP="/playpen/mufan/levi/tianlong-chen-lab/material-super-resolution/__exps__/foundation-models/2. surface-conductivity/4x/2025-02-27_11-12-01_SwinIR-SR-4X-DS=MoS2-synth-downsampling=bicubic-s64-bs-1-adam-lr=1e-6-loss=L1/SwinIR-SR-4X-DS=MoS2-synth-downsampling=bicubic-s64-bs-1-adam-lr=1e-6-loss=L1_best.pth"
#         elif [ "$sr" -eq 8 ]; then
#           EXP_ROOT_DIR="/playpen/mufan/levi/tianlong-chen-lab/material-super-resolution/__exps__/foundation-models/2. surface-conductivity/8x"
#           EXP_NAME="SwinIR-SR-8X-DS=MoS2-synth-downsampling=bicubic-s48-bs-1-adam-lr=1e-6-loss=L1"
#           WEIGHTS_FP="/playpen/mufan/levi/tianlong-chen-lab/material-super-resolution/__exps__/foundation-models/2. surface-conductivity/8x/2025-02-27_11-13-29_SwinIR-SR-8X-DS=MoS2-synth-downsampling=bicubic-s48-bs-1-adam-lr=1e-6-loss=L1/SwinIR-SR-8X-DS=MoS2-synth-downsampling=bicubic-s48-bs-1-adam-lr=1e-6-loss=L1_best.pth"
#         fi
#       fi
      
#       # Select GPU device (round-robin over 0-7)
#       gpu_id=$(( counter % 8 ))
#       counter=$(( counter + 1 ))
      
#       echo "Running experiment: dataset=${dataset}, formulation=${formulation}, upsample_factor=${sr} on GPU ${gpu_id}"
      
#       export CUDA_VISIBLE_DEVICES=$gpu_id
#       python train.py \
#           --exp_name "$EXP_NAME" \
#           --root "$EXP_ROOT_DIR" \
#           --upsample_factor $sr \
#           --dataset $dataset \
#           --formulation $formulation \
#           --weights "$WEIGHTS_FP" \
#           --num_heads $NUM_HEADS \
#           --num_blocks $NUM_BLOCKS \
#           --window_size $WINDOW_SIZE \
#           --drop_path_rate $DPR \
#           --norm_layer $NORM_LAYER \
#           > "$LOGS_DIR/_${EXP_NAME}_${dataset}_${formulation}_${sr}.out" 2>&1 &
#     done
#   done
# done

# exit