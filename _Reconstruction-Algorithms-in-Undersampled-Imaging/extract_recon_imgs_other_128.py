"""
This script extracts reconstructed images for the ASPMCI paper
from a specified experiment results HDF5 file.

"""

from __future__ import division
import itertools

import matplotlib.pyplot as plt
import numpy as np
import tables as tb
import pandas as pd
from os import makedirs
import errno

import magni

# Setup - the following path can be changed if the result file is in a
# different location. This should correspond to the `result_folder`
# variable in 'aspmci_reconstructions.py', line 488:
data_path = 'data/'
hdf_database_names = ['5668_0_aspmci_reconstructions.hdf5',
                      '5668_0_aspmci_reconstructions.hdf5']

images = tuple(['image_{}.mi'.format(k) for k in range(7)])
sampling_patterns = ('rect_spiral', 'uniform_lines')
undersampling_ratios = np.linspace(0.1, 0.3, 3)
reconstruction_algorithms = ('cubic_interpolation', 'bg_amp')
# Open data file
hdf_stores = []
reconstruction_metrics = []
for hdf_database_name in hdf_database_names:
    hdf_stores.append(pd.HDFStore(data_path + hdf_database_name))
    reconstruction_metrics.append(hdf_stores[-1].select('/simulation_results/metrics'))

# Extract a selection of reconstructed images: one for each original
# undersampling rate, image, algorithm, and sampling pattern
try:
    makedirs('../figures_test/')
except OSError, e:
    if e.errno != errno.EEXIST:
        throw
for idx, hdf_database_name in enumerate(hdf_database_names):
    with tb.File(data_path + hdf_database_name, mode='r') as h5_file:
        for ur in undersampling_ratios:
            for image in images:
                for algorithm in reconstruction_algorithms:
                    for pattern in sampling_patterns:
                        try:
                            recon_img = (h5_file.get_node('/simulation_results/{}/{}/d{}/{}/'.format(
                                image.replace('.', '_'), pattern,
                                str(ur).replace('.', '_'), algorithm),
                                                          name='reconstructed_img_vec').read())
                            img_shape = (h5_file.get_node('/simulation_results/{}/{}/d{}/{}/'.format(
                                image.replace('.', '_'), pattern,
                                str(ur).replace('.', '_'), algorithm),
                                                          name='img_shape').read())
                        except:
                            # Assume the image is in one of the other files
                            pass
                        else:
                            disp_img = magni.imaging.vec2mat(recon_img, img_shape)
                            plt.figure(figsize=(4,4)) # figsize chosen arbitrarily, but match it to dpi in line 78
                            select_criterion = np.logical_and(np.logical_and(reconstruction_metrics[idx]['delta'] == ur,
                                                                             reconstruction_metrics[idx]['image'] == image),
                                                              np.logical_and(reconstruction_metrics[idx]['reconstruction_algorithm'] == algorithm,
                                                                             reconstruction_metrics[idx]['sampling_pattern'] == pattern))
                            plt.title('PSNR: {:.2f} dB / SSIM: {:.2f}'.format(reconstruction_metrics[idx][select_criterion]['psnr'].iat[0],
                                                                              reconstruction_metrics[idx][select_criterion]['ssim'].iat[0]))
                            fig = magni.imaging.visualisation.imshow(disp_img, show_axis='none')
                            # Axis modification suggested in http://stackoverflow.com/a/26610602/865169
                            plt.axis('off')
                            fig.axes.get_xaxis().set_visible(False)
                            fig.axes.get_yaxis().set_visible(False)
                            plt.savefig('../figures_test/recon-{}-{}-{}-{}-128px.pdf'.format(image.replace('.mi',''),
                                                                                  algorithm,
                                                                                  pattern,
                                                                                  str(ur).replace('.','_')),
                                        bbox_inches='tight',
                                        pad_inches=0,
                                        dpi=41.5) # dpi was chosen to match figzise in line 60 to obtain 128x128 resolution
                            plt.close()
    h5_file.close()

for hdf_store in hdf_stores:
    hdf_store.close()
