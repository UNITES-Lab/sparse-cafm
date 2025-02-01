# 1. P(y|y_sparse)
export CUDA_VISIBLE_DEVICES=0
nohup python train.py > _swinir_baseline.out 2>&1 &
exit