export CUDA_VISIBLE_DEVICES=1

# ---- eval -----
# nohup python test.py \
#     --exp_name "_swinir->unet.out" \
#     --model_weights_path "/playpen/mufan/levi/tianlong-chen-lab/material-super-resolution/__exps__/y-task-formulations/p(y | y_sparse)/6a. train-runs/2025-02-03_10-40-32_swinir->unet-with-zero-conv-full-ds [GOD RUN]/swinir->unet-with-zero-conv-full-ds_best.pth" \
#     > "_swinir->unet.out" 2>&1 &

nohup python test_surrogate.py \
    --exp_name "surrogate_initial_eval_test" \
    --older_surrogate_model_weights_path "/playpen/mufan/levi/tianlong-chen-lab/material-super-resolution/__exps__/y-task-formulations/p(y | y_sparse)/6c. surrogate-train-runs/2025-02-02_06-10-31_surrogate-depth=8/surrogate-depth=8_best_denoiser.pth" \
    --denoising_model_weights_path "/playpen/mufan/levi/tianlong-chen-lab/material-super-resolution/__exps__/y-task-formulations/p(y | y_sparse)/6c. surrogate-train-runs/2025-02-02_06-10-31_surrogate-depth=8/surrogate-depth=8_best_denoiser.pth" \
    > "_surrogate_initial_eval_test.out" 2>&1 &
# ---------------
exit