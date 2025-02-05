# ---- eval -----
# export CUDA_VISIBLE_DEVICES=2
# nohup python test.py \
#     --exp_name "_swinir->unet.out" \
#     --model_weights_path "/playpen/mufan/levi/tianlong-chen-lab/material-super-resolution/__exps__/y-task-formulations/p(y | y_sparse)/6a. train-runs/2025-02-05_08-05-15_swinir-depth=32-sigmoid/swinir-depth=32-sigmoid_best.pth" \
#     > "_swinir-depth=32.out" 2>&1 &

export CUDA_VISIBLE_DEVICES=2
python test_surrogate.py \
    --exp_name "surrogate-eval-test" \
    --denoising_model_weights_path "/playpen/mufan/levi/tianlong-chen-lab/material-super-resolution/__exps__/y-task-formulations/p(y | y_sparse)/6c. surrogate-train-runs/2025-02-05_08-30-24_surrogate-swinir-sigmoid/surrogate-swinir-sigmoid_best_older_surrogate.pth" \
    --older_surrogate_model_weights_path "/playpen/mufan/levi/tianlong-chen-lab/material-super-resolution/__exps__/y-task-formulations/p(y | y_sparse)/6c. surrogate-train-runs/2025-02-05_08-30-24_surrogate-swinir-sigmoid/surrogate-swinir-sigmoid_best_infilling_model.pth" \
#    > "_surrogate-eval-test.out" 2>&1 &
# ---------------

# exit