# 1. P(y|y_sparse)
export CUDA_VISIBLE_DEVICES=0
nohup python train.py > _swinir_no-sigmoid_old-channel-downsample.out 2>&1 &
exit