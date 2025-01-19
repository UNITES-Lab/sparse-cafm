# 1. P(y|y_sparse)
export CUDA_VISIBLE_DEVICES=7
nohup python train.py > need-weights_ii.out 2>&1 &
exit