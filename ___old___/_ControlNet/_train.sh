export CUDA_VISIBLE_DEVICES=4
nohup python train.py > _cn_unconditional_384_topo-maps.out 2>&1 &
exit