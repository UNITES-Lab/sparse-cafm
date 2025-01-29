export CUDA_VISIBLE_DEVICES=5
nohup python test.py > _swinir-best-eval.out 2>&1 &
exit