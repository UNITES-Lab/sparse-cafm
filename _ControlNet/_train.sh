export CUDA_VISIBLE_DEVICES=1
nohup python train.py > _cn_unconditional_384_fixed?.out 2>&1 &
exit