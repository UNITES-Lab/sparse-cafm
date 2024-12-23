# 1. P(y | y_sparse)
export CUDA_VISIBLE_DEVICES=7
nohup python train_y_bar_y_sparse.py > unet.out 2>&1 &
exit

# 2. P(y | y_sparse) – VAE flavored
# export CUDA_VISIBLE_DEVICES=7
# nohup python train_y_bar_y_sparse_vae.py > vae_small_loss.out 2>&1 &
# exit

# cd ControlNet
# export CUDA_VISIBLE_DEVICES=7
# nohup python train.py /dev/null 2>&1 &
# exit