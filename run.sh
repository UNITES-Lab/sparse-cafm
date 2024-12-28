# # 1. P(y | y_sparse)
# export CUDA_VISIBLE_DEVICES=2
# nohup python train_y_bar_y_sparse.py > swinir.out 2>&1 &
# exit

# 2. P(y | y_sparse) – VAE flavored
# export CUDA_VISIBLE_DEVICES=7
# nohup python train_y_bar_y_sparse_vae.py > vae_small_loss.out 2>&1 &
# exit

# cd ControlNet
# export CUDA_VISIBLE_DEVICES=2
# python train.py

# nohup python train.py > control_net.out 2>&1 &
# exit

cd /playpen/mufan/levi/tianlong-chen-lab/material-super-resolution/__benchmarks__/places-365
wget http://data.csail.mit.edu/places/places365/train_large_places365standard.tar
wget http://data.csail.mit.edu/places/places365/val_large.tar
wget http://data.csail.mit.edu/places/places365/test_large.tar