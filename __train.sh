# 1. P(y|y_sparse)
export CUDA_VISIBLE_DEVICES=1
nohup python train.py > _swinir_lr=1e-5.out 2>&1 &
exit