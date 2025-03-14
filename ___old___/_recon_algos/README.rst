README
======

This deposition accompanies the research article "Reconstruction
Algorithms in Undersampled AFM Imaging"
(http://dx.doi.org/10.1109/JSTSP.2015.2500363) and contains the
following files:

* Simulation
  * `reconstruction_goblet_128.py` 
  * `reconstruction_goblet_256.py` 
  * `aspmci_reconstructions_128.py` 
  * `aspmci_reconstructions_256.py` 
  * `amp_bgg_solver.py`
  * `gamp_reconstructions.py`
  * `interp_reconstructions.py`
  * `it_reconstructions.py`
  * `optim_reconstructions.py`
  * `utils.py`

* Data extraction/analysis
  * `analyse_data.ipynb`
  * `extract_data_128.py` 
  * `extract_data_256.py` 
  * `extract_originals.py`
  * `extract_patterns.py`
  * `extract_recon_imgs_128.py` 
  * `extract_recon_imgs_256.py` 
  * `extract_recon_imgs_other_128.py` 
  * `extract_recon_imgs_other_256.py` 

* Meta-data
  * `LICENSE`
  * `MD5SUMS`
  * `README.rst`
  * `SHA256SUMS` 


File descriptions:

* `reconstruction_goblet_256.py`
  The main script for running the simulations on 256×256 images. This
  script depends on:
  - The magni Python package from http://dx.doi.org/10.5278/VBN/MISC/Magni
  - The PyUNLocBox Python package from:
    https://github.com/epfl-lts2/pyunlocbox/

    The specific version of this package used for the original
    numerical experiments is available at:
    https://github.com/epfl-lts2/pyunlocbox/tree/f9fafb070df125c38da86b346a330743e8065910
  - The PyWavelets Python package from: https://github.com/PyWavelets/pywt
    
    The specific version of this package used for the original
    numerical experiments is available at:
    https://github.com/PyWavelets/pywt/tree/9343316c10f7a5e189f333eec28b06696ece9dc1
  - The original AFM image files available at:
    http://dx.doi.org/10.5281/zenodo.17573
  - Various other Python packages available through standard channels
    (see imports in the Python script).

* `reconstruction_goblet_128.py`
  Same as above, working on 128×128 images.
* `aspmci_reconstructions_256.py`
  Script for running further simulations on 256×256 images. Only the
  results for interpolation and BG AMP are used in this context. This
  script has the same dependencies as `reconstruction_goblet_256.py`.
* `aspmci_reconstructions_128.py`
  Script for running further simulations on 128×128 images. Only the
  results for interpolation and BG AMP are used in this context. This
  script has the same dependencies as `reconstruction_goblet_256.py`.
* `amp_bgg_solver.py`
  Module implementing one of the reconstruction methods
  (Bernoulli-Gaussian approximate message passing) used in the
  research article accompanying this
  software. `reconstruction_goblet_128.py` and
  `aspmci_reconstructions_128.py` use this module.
* `gamp_reconstructions.py`
  Module implementing L1 AMP. Used by `reconstruction_goblet_256.py`,
  `reconstruction_goblet_128.py`, `aspmci_reconstructions_256.py`, and
  `aspmci_reconstructions_128.py`.
* `interp_reconstructions.py`
  Module providing interfaces to interpolation functions. Used by
  `reconstruction_goblet_256.py`, `reconstruction_goblet_128.py`,
  `aspmci_reconstructions_256.py`, and
  `aspmci_reconstructions_128.py`.
* `it_reconstructions.py`
  Module providing interfaces to iterative thresholding algorithms
  from the Magni package. Used by `reconstruction_goblet_256.py`,
  `reconstruction_goblet_128.py`, `aspmci_reconstructions_256.py`, and
  `aspmci_reconstructions_128.py`.
* `optim_reconstructions.py`
  Module providing interfaces to convex optimisation algorithms
  implemented via PyUNLocBox. Used by `reconstruction_goblet_256.py`,
  `reconstruction_goblet_128.py`, `aspmci_reconstructions_256.py`, and
  `aspmci_reconstructions_128.py`.
* `utils.py`
  Module providing common functionality to several of the above files.
* `analyse_data.ipynb`
  This a Jupyter notebook in which various numerical results are
  extracted from the simulation result database (available at
  http://doi.org/10.5281/zenodo.32958). Jupyter notebooks can be
  opened using Jupyter (see http://jupyter.org/).
* `extract_data_256.py`
  Script used for extracting and calculating statistics of the
  simulation results (for 256×256 images) and plotting these in a
  figure shown as Figure 3 in the research article accompanying this
  software.
* `extract_data_128.py`
  Script used for extracting and calculating statistics of the
  simulation results (for 128×128 images) and plotting these in a
  figure shown as Figure 4 in the research article accompanying this
  software.
* `extract_originals.py`
  Script for extracting the original images from MI files. This
  generates most of Figure 2 in the research article accompanying this
  software. The script also generates the color bars shown in Figure 1
  in the mentioned article.
* `extract_patterns.py`
  Script for extracting and visualising an example sampling pattern
  which is shown in Figure 2 in the research article accompanying this
  software.
* `extract_recon_imgs_256.py`
  Script for extracting the reconstructed images from the result
  file(s) (for 256×256 images). A selection of these are shown in
  Figures 5-8 in the research article accompanying this software.
* `extract_recon_imgs_other_256.py`
  Script for extracting additional reconstructed images from the
  result file(s) (for 256×256 images). These are the images
  reconstructed using interpolation. A selection of these are shown in
  Figures 5-8 in the research article accompanying this software.
* `extract_recon_imgs_128.py`
  Same as `extract_recon_imgs_256.py`, for 128×128 images. Images of
  this resolution are not shown in the research article accompanying
  this software.
* `extract_recon_imgs_other_128.py`
  Same as `extract_recon_imgs_other_256.py`, for 128×128
  images. Images of this resolution are not shown in the research
  article accompanying this software.
* `MD5SUMS`, `SHA256SUMS`
  Checksums of the enclosed files that can be used for verifying the
  integrity of the data after download.
