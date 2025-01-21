export CUDA_VISIBLE_DEVICES=2
nohup python eval.py > test_eval.out 2>&1 &
exit