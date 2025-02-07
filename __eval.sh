# ---- eval SwinIR runs -----
# export CUDA_VISIBLE_DEVICES=0
# nohup python test.py \
#     --exp_name "swinir_p(y | X_sparse, y_sparse)" \
#     --model_weights_path "/playpen/mufan/levi/tianlong-chen-lab/material-super-resolution/__exps__/y-task-formulations/p(y | X_sparse, y_sparse)/a. train-runs/2025-02-06_08-08-50_swinir-depth=32/swinir-depth=32_best.pth" \
#     > "_swinir_p(y | X_sparse, y_sparse).out" 2>&1 &

# export CUDA_VISIBLE_DEVICES=1
# nohup python test.py \
#     --exp_name "swinir_p(y | X_sparse)" \
#     --model_weights_path "/playpen/mufan/levi/tianlong-chen-lab/material-super-resolution/__exps__/y-task-formulations/p(y | X_sparse)/a. train-runs/2025-02-06_08-15-51_swinir-depth=32/swinir-depth=32_best.pth" \
#     > "_swinir_p(y | X_sparse).out" 2>&1 &

# export CUDA_VISIBLE_DEVICES=2
# nohup python test.py \
#     --exp_name "swinir_p(y | y_sparse)" \
#     --model_weights_path "/playpen/mufan/levi/tianlong-chen-lab/material-super-resolution/__exps__/y-task-formulations/p(y | y_sparse)/a. train-runs/2025-02-06_08-19-21_swinir-depth=32/swinir-depth=32_best.pth" \
#     > "_swinir_p(y | y_sparse).out" 2>&1 &

# ---- eval surrogate runs -----
export CUDA_VISIBLE_DEVICES=4
nohup python test_surrogate.py \
    --exp_name "older-surrogate-loss=older" \
    --older_surrogate_model_weights_path "__exps__/y-task-formulations/p(y | y_sparse)/6c. surrogate-train-runs/2025-02-06_10-27-07_infiller-swinir-D=32-loss=OLDER/infiller-swinir-D=32-loss=OLDER_latest_older_surrogate.pth" \
    --infilling_model_weights_path "__exps__/y-task-formulations/p(y | y_sparse)/6c. surrogate-train-runs/2025-02-06_10-27-07_infiller-swinir-D=32-loss=OLDER/infiller-swinir-D=32-loss=OLDER_latest_infilling_model.pth" \
   > "_older-surrogate-loss=older.out" 2>&1 &

export CUDA_VISIBLE_DEVICES=5
nohup python test_surrogate.py \
    --exp_name "older-surrogate-loss=sig(older)" \
    --older_surrogate_model_weights_path "/playpen/mufan/levi/tianlong-chen-lab/material-super-resolution/__exps__/y-task-formulations/p(y | y_sparse)/6c. surrogate-train-runs/2025-02-06_10-27-43_infiller-swinir-D=32-loss=sig(OLDER)/infiller-swinir-D=32-loss=sig(OLDER)_latest_older_surrogate.pth" \
    --infilling_model_weights_path "/playpen/mufan/levi/tianlong-chen-lab/material-super-resolution/__exps__/y-task-formulations/p(y | y_sparse)/6c. surrogate-train-runs/2025-02-06_10-27-43_infiller-swinir-D=32-loss=sig(OLDER)/infiller-swinir-D=32-loss=sig(OLDER)_latest_infilling_model.pth" \
   > "_older-surrogate-loss=sig(older).out" 2>&1 &

export CUDA_VISIBLE_DEVICES=6
nohup python test_surrogate.py \
    --exp_name "older-surrogate-loss=older+L1" \
    --older_surrogate_model_weights_path "/playpen/mufan/levi/tianlong-chen-lab/material-super-resolution/__exps__/y-task-formulations/p(y | y_sparse)/6c. surrogate-train-runs/2025-02-06_10-28-33_infiller-swinir-D=32-loss=OLDER+L1/infiller-swinir-D=32-loss=OLDER+L1_latest_older_surrogate.pth" \
    --infilling_model_weights_path "/playpen/mufan/levi/tianlong-chen-lab/material-super-resolution/__exps__/y-task-formulations/p(y | y_sparse)/6c. surrogate-train-runs/2025-02-06_10-28-33_infiller-swinir-D=32-loss=OLDER+L1/infiller-swinir-D=32-loss=OLDER+L1_latest_infilling_model.pth" \
   > "_older-surrogate-loss=older+L1.out" 2>&1 &

export CUDA_VISIBLE_DEVICES=7
nohup python test_surrogate.py \
    --exp_name "older-surrogate-loss=sig(older)+L1" \
    --older_surrogate_model_weights_path "/playpen/mufan/levi/tianlong-chen-lab/material-super-resolution/__exps__/y-task-formulations/p(y | y_sparse)/6c. surrogate-train-runs/2025-02-06_10-29-49_infiller-swinir-D=32-loss=sig(OLDER)+L1/infiller-swinir-D=32-loss=sig(OLDER)+L1_latest_older_surrogate.pth" \
    --infilling_model_weights_path "/playpen/mufan/levi/tianlong-chen-lab/material-super-resolution/__exps__/y-task-formulations/p(y | y_sparse)/6c. surrogate-train-runs/2025-02-06_10-29-49_infiller-swinir-D=32-loss=sig(OLDER)+L1/infiller-swinir-D=32-loss=sig(OLDER)+L1_latest_infilling_model.pth" \
   > "_older-surrogate-loss=sig(older)+L1.out" 2>&1 &
# ---------------
exit