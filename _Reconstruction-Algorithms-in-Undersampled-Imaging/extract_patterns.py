"""
This script extracts example sampling pattern(s) for the ASPMCI
paper from a specified experiment results HDF5 file.
"""

from __future__ import division
import itertools

import matplotlib.pyplot as plt
import numpy as np
import tables as tb
from os import makedirs
import errno

import magni

# Setup - the following path can be changed if the result file is in a
# different location. This should correspond to the `result_folder`
# variable in 'aspmci_reconstructions.py', line 488:
data_path = 'data/'
hdf_database_name = '5699_0_reconstruction_goblet_ID_0_of_4.hdf5'

imsize = 256

# Extract a delta = 0.1 example sampling pattern, plot, and save
with tb.File(data_path + hdf_database_name, mode='r') as h5_file:
    try:
        makedirs('../figures/fig2/')
    except OSError, e:
        if e.errno != errno.EEXIST:
            throw
        unique_coords = (h5_file.get_node('/simulation_results/image_0_mi/rect_spiral/DCT/delta__0_1/ell_1_optim/param__0_1',
                                       name='unique_coords').read())
        # figsize chosen arbitrarily, but match it to dpi in line 48
        plt.figure(figsize=(4,4))
        magni.imaging.measurements.plot_pixel_mask(imsize, imsize,
                                                   unique_coords)
        # Axis modification suggested in
        # http://stackoverflow.com/a/26610602/865169
        plt.axis('off')
        plt.gcf().axes[0].get_xaxis().set_visible(False)
        plt.gcf().axes[0].get_yaxis().set_visible(False)
        # dpi was chosen to match figzise in line 42 to obtain 256x256
        # resolution
        plt.savefig('../figures/fig2/pattern-rect_spiral-0_1.png',
                    bbox_inches='tight', pad_inches=0, dpi=81.75)
h5_file.close()
