export CUDA_VISIBLE_DEVICES=1
CONFIG_FP="_test.yaml"

python train_simple.py \
    --config $CONFIG_FP