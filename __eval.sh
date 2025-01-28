export CUDA_VISIBLE_DEVICES=4
nohup python test.py > _bicubic-test.out 2>&1 &
exit