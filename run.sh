# 1. P(y | y_sparse)
export CUDA_VISIBLE_DEVICES=6
nohup python train.py > best.out 2>&1 &
exit