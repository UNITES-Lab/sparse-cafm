# **Experiment Set**: 4-Task Formulations

Goal: figure out which of the four task formulations is most promising.

Definitions
- $X$ : topology-map with shape $\{H, W, D\}$
- $y$ : current-map with shape $\{H, W, C\}$
- $\hat{y}$ : predicted current-map with shape $\{H, W, C\}$
- $g_{\theta}$ : ControlNet `(N, H, W, D) -> (N, H, W, C)`
    - $g_{\theta}(X) = \hat{y} \approx y$

Preparation
- Train a model $g_{\theta}(X) = \hat{y} \approx y$
- Develop a pre-processed dataset of $\hat{y}$ 
    - Splice training and validation set original samples using **fixed-grid sampling**
    - `1x(512, 512) -> 64x(64, 64)`
    - Train and validation samples completely seperate (i.e., **4 train 1 val**)

Four task formulations

1. $P(z | X)$
    - Directly regress scalar value $z$ from $X$
    - Train a model $f_{\theta}(X) = \hat{z} \approx z$
2. $P(z | \hat{y})$
    - Train a model $f_{\theta}(\hat{y}) = \hat{z} \approx z$
3. $P(z | X, \hat{y})$
    - Train a model $f_{\theta}(X, \hat{y}) = \hat{z} \approx z$
4.  $P(z | X) + P(z | \hat{y})$
    - Predict $z = (z_1 + z_2) / 2$

Table 1.

| Formulation | Eval L1 Loss |
| :---: | :---: | 
| --- | --- |
| --- | --- |
| --- | --- |
| --- | --- |