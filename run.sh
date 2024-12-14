# python train_f1.py
cd __repos__/ControlNet

# nohup python train.py /dev/null 2>&1 &
# exit

export CUDA_VISIBLE_DEVICES=2
nohup python sample.py /dev/null 2>&1 &
exit