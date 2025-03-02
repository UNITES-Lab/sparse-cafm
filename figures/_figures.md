# **Qualitative**
---

1. [Spider Chart]
    - 1a. cherry pick the best dataset: [MoS2-Sef, Silicon, Sapphire]
    - 1b. compare method performance on **electrical characterization metrics** @[2x, 4x, 8x] sparsity vs [linear, bicubic, nn]
2. [Image Grid] Main Results (electrical characterization)
    - Show cherry picked model outputs @[2x, 4x, 8x] levels of sparsity for **electrical maps**
3. [Diagram] Method
    - Show a high-level overview of how our method works
    - Include a diagram showing the sparse sampling patterns used to obtain downsampled maps
        - e.g., Fig.1 [G. Han, B. Lin /Ultramicroscopy 189 (2018) 85–94]
4. [Image Grid] Main Results (surface morphology)
    - Show cherry picked model outputs @[2x, 4x, 8x] levels of sparsity for **topological maps**
5. [Bar Chart] Performance of Surrogate Model (or before/after surrogate model loss?)
    - Show that the surrogate model **helps**; maybe report before/after scores on electircal characterization metrics with L1 vs surrogate
6. [Diagram] Model Architecture
    - Side by side digram showing designs of 1.) SwinIR/super-resolution model, 2.) eletrical-characterization surrogate

# **Quantitative**
---

1. [Table] Main Results (electrical characterization)
    - Large table showing performance of our model on **electrical characterization metrics** on different datasets/levels of sparisty
    - Use *downsampled inputs* as **baseline**
    - NOTE: add "estimated redution in time-to-data" as a column
2. [Line Plot] Model Performance vs # Training Samples
    - Q: how quickly can our method adapt to new datasets?
    - Show that our model can learn very quickly
3. [Table] Model Performance on Real vs Synthetic Data
    - Q: how well does our model transfer from training on synthetically downsampled data to real, downsampled data?
4. [Table] Main Results (AFM-recovery/surface morphology)
    - Report surface roughness (any other metrics we can report here)?