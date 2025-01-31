# 1. P(y|y_sparse)
export CUDA_VISIBLE_DEVICES=5
nohup python train.py > _swinir_d32.out 2>&1 &
exit