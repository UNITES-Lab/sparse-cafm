export CUDA_VISIBLE_DEVICES=1

# ---- eval -----
nohup python test.py \
    --exp_name "_swinir->unet.out" \
    --model_weights_path "/playpen/mufan/levi/tianlong-chen-lab/material-super-resolution/__exps__/y-task-formulations/p(y | y_sparse)/6a. train-runs/2025-02-03_10-40-32_swinir->unet-with-zero-conv-full-ds [GOD RUN]/swinir->unet-with-zero-conv-full-ds_best.pth" \
    > "_swinir->unet.out" 2>&1 &
# ---------------
exit