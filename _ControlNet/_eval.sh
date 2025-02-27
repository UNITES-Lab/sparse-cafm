export CUDA_VISIBLE_DEVICES=7
nohup python generate_synth_dataset.py > _topo-synth-10k.out 2>&1 &
exit