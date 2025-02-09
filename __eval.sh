# ---- eval SwinIR runs -----
# export CUDA_VISIBLE_DEVICES=0
# nohup python test.py \
#     --exp_name "swinir_L1-BEST" \
#     --model_weights_path "/playpen/mufan/levi/tianlong-chen-lab/material-super-resolution/__exps__/y-task-formulations/p(y | y_sparse)/a. train-runs/2025-02-08_15-41-51_swinir-loss=L1/swinir-loss=L1_best.pth" \
#     > "_swinir_L1-BEST.out" 2>&1 &

# export CUDA_VISIBLE_DEVICES=1
# nohup python test.py \
#     --exp_name "swinir_L1-LATEST" \
#     --model_weights_path "/playpen/mufan/levi/tianlong-chen-lab/material-super-resolution/__exps__/y-task-formulations/p(y | y_sparse)/a. train-runs/2025-02-08_15-41-51_swinir-loss=L1/swinir-loss=L1_latest.pth" \
#     > "_swinir_L1-LATEST.out" 2>&1 &

# export CUDA_VISIBLE_DEVICES=2
# nohup python test.py \
#     --exp_name "swinir_OLDER-BEST" \
#     --model_weights_path "/playpen/mufan/levi/tianlong-chen-lab/material-super-resolution/__exps__/y-task-formulations/p(y | y_sparse)/a. train-runs/2025-02-08_15-42-54_swinir-loss=OLDER/swinir-loss=OLDER_best.pth" \
#     > "_swinir_OLDER-BEST.out" 2>&1 &

# export CUDA_VISIBLE_DEVICES=3
# nohup python test.py \
#     --exp_name "swinir_OLDER-LATEST" \
#     --model_weights_path "/playpen/mufan/levi/tianlong-chen-lab/material-super-resolution/__exps__/y-task-formulations/p(y | y_sparse)/a. train-runs/2025-02-08_15-42-54_swinir-loss=OLDER/swinir-loss=OLDER_latest.pth" \
#     > "_swinir_OLDER-LATEST.out" 2>&1 &

# export CUDA_VISIBLE_DEVICES=4
# nohup python test.py \
#     --exp_name "swinir_OLDER+L1-BEST" \
#     --model_weights_path "/playpen/mufan/levi/tianlong-chen-lab/material-super-resolution/__exps__/y-task-formulations/p(y | y_sparse)/a. train-runs/2025-02-08_15-44-36_swinir-loss=OLDER+L1/swinir-loss=OLDER+L1_best.pth" \
#     > "_swinir_OLDER+L1-BEST.out" 2>&1 &

# export CUDA_VISIBLE_DEVICES=5
# nohup python test.py \
#     --exp_name "swinir_OLDER+L1-LATEST" \
#     --model_weights_path "/playpen/mufan/levi/tianlong-chen-lab/material-super-resolution/__exps__/y-task-formulations/p(y | y_sparse)/a. train-runs/2025-02-08_15-44-36_swinir-loss=OLDER+L1/swinir-loss=OLDER+L1_latest.pth" \
#     > "_swinir_OLDER+L1-LATEST.out" 2>&1 &

export CUDA_VISIBLE_DEVICES=1
nohup python test.py \
    --exp_name "nn_interpolation-baseline" \
    > "_nn_interpolation-baseline.out" 2>&1 &
# ---------------------------------------------
exit