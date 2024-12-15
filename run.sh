# # 1. P(z | X)
# export CUDA_VISIBLE_DEVICES=0
# nohup python train_z_bar_X.py /dev/null 2>&1 &
# exit

# # 2. P(z | y_hat)
# export CUDA_VISIBLE_DEVICES=1
# nohup python train_z_bar_y_hat.py /dev/null 2>&1 &
# exit

# cd __repos__/ControlNet
# export CUDA_VISIBLE_DEVICES=2
# nohup python sample.py /dev/null 2>&1 &
# exit