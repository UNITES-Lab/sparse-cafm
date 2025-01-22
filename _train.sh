# 1. P(y|y_sparse)
export CUDA_VISIBLE_DEVICES=3
nohup python train.py > unet_128.out 2>&1 &
exit