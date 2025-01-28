export CUDA_VISIBLE_DEVICES=3
nohup python test.py > _bicubic-test.out 2>&1 &
exit