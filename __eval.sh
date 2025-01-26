export CUDA_VISIBLE_DEVICES=1
nohup python test.py > test_eval.out 2>&1 &
exit