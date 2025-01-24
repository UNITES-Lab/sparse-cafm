# 1. P(y|y_sparse)
export CUDA_VISIBLE_DEVICES=7
nohup python train.py > _ours_128x128.out 2>&1 &
exit