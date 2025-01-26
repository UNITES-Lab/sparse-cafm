# 1. P(y|y_sparse)
export CUDA_VISIBLE_DEVICES=1
nohup python train.py > _swinir_sigmoid_128x128.out 2>&1 &
exit