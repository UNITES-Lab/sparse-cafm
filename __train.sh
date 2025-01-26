# 1. P(y|y_sparse)
export CUDA_VISIBLE_DEVICES=6

nohup python train.py > _unet_sigmoid_128x128_logger_no_renorm.out 2>&1 &
exit