export CUDA_VISIBLE_DEVICES=1
DEPTH=12
nohup python train.py \
    --exp_name swinir-depth=$DEPTH \
    --depths $DEPTH \
    > _swinir_depths=$DEPTH.out 2>&1 &

export CUDA_VISIBLE_DEVICES=2
DEPTH=16
nohup python train.py \
    --exp_name swinir-depth=$DEPTH \
    --depths $DEPTH \
    > _swinir_depths=$DEPTH.out 2>&1 &

export CUDA_VISIBLE_DEVICES=3
DEPTH=20
nohup python train.py \
    --exp_name swinir-depth=$DEPTH \
    --depths $DEPTH \
    > _swinir_depths=$DEPTH.out 2>&1 &

export CUDA_VISIBLE_DEVICES=4
DEPTH=28
nohup python train.py \
    --exp_name swinir-depth=$DEPTH \
    --depths $DEPTH \
    > _swinir_depths=$DEPTH.out 2>&1 &

export CUDA_VISIBLE_DEVICES=5
DEPTH=32
nohup python train.py \
    --exp_name swinir-depth=$DEPTH \
    --depths $DEPTH \
    > _swinir_depths=$DEPTH.out 2>&1 &
exit