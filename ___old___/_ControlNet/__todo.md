1. Directly denoise y_sparse
2. Change all data-norm + model out-activations to Sigmoid -> [0, 1]
3. Change standard eps loss -> {x_0, ...}
4. Play with classifier-free guidance scales