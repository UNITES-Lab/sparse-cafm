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

# Setup - the following path can be changed if the result files are in
# a different location. This should correspond to the `result_folder`
# variable in 'aspmci_reconstructions.py', line 488:
data_path = 'data/'
hdf_database_names = ['5694_0_reconstruction_goblet_ID_0_of_4.hdf5',
                      '5694_1_reconstruction_goblet_ID_1_of_4.hdf5',
                      '5694_2_reconstruction_goblet_ID_2_of_4.hdf5',
                      '5694_3_reconstruction_goblet_ID_3_of_4.hdf5']

images = tuple(['image_{}.mi'.format(k) for k in range(7)])
sampling_patterns = ('rect_spiral', 'uniform_lines')
undersampling_ratios = np.linspace(0.1, 0.3, 3)
reconstruction_algorithms = ('iht_fixed', 'ell_1_optim', 'ell_1_amp',
                             'ist_fixed', 'ell_1_dwt_db',
                             'ell_1_dwt_dmey', 'ell_1_dct_overc2',
                             'ell_1_dct_overc3', 'ell_1_dwt_sym',
                             'tv_optim')

# Extract a selection of reconstructed images: one for each original
# undersampling rate, image, algorithm, and sampling pattern
try:
    makedirs('../figures_test/')
except OSError, e:
    if e.errno != errno.EEXIST:
        throw
for ur in undersampling_ratios:
    for image in images:
        for algorithm in reconstruction_algorithms:
            for pattern in sampling_patterns:
                image_found = False
                for idx, hdf_database_name in enumerate(hdf_database_names):
                    with pd.HDFStore(data_path + hdf_database_name) as hdf_store:
                        reconstruction_metrics = hdf_store.select('/simulation_results/metrics')
                    hdf_store.close()
                    with tb.File(data_path + hdf_database_name, mode='r') as h5_file:
                        id_string = '/simulation_results/{}/{}/DCT/delta__{}/{}'.format(
                            image.replace('.', '_'), pattern,
                            str(ur).replace('.', '_'), algorithm)
                        if h5_file.__contains__(id_string):
                            image_found = True
                        try:
                            df_current = reconstruction_metrics.query('delta == @ur and image == @image and reconstruction_algorithm == @algorithm and sampling_pattern == @pattern')
                            best_idx = df_current['psnr'].idxmax()
                            reg_param = df_current['reconstruction_parameter'].loc[best_idx]
                            # Handle unfortunate encoding of some floats and ints
                            if (reg_param%1 == 0) and (reg_param >= 10):
                                reg_param = int(reg_param)
                            if algorithm == 'ell_1_amp':
                                assert reg_param == 1
                                reg_param = int(reg_param)
                            recon_img = (
                                h5_file.get_node(
                                    id_string + '/param__' + str(reg_param).replace('.', '_').replace('-', '__'),
                                    name='reconstructed_img_vec').read())
                            img_shape = (
                                h5_file.get_node(
                                    id_string + '/param__' + str(reg_param).replace('.', '_').replace('-', '__'),
                                    name='img_shape').read())
                        except ValueError as err:
                            # Assume the image is in one of the other files
                            assert not image_found, 'Probably trouble with indexing in a DataFrame'
                        except tb.exceptions.NoSuchNodeError as err:
                            # Assume the image is in one of the other files
                            assert not image_found, 'Probably trouble with formatting of reg_param'
                        else:
                            image_found = True
                            disp_img = magni.imaging.vec2mat(recon_img, img_shape)
                            plt.figure(figsize=(4,4)) # figsize chosen arbitrarily, but match it to dpi in line 78
                            #select_criterion = np.logical_and(np.logical_and(reconstruction_metrics[idx]['delta'] == ur,
                            #                                                 reconstruction_metrics[idx]['image'] == image),
                            #                                  np.logical_and(reconstruction_metrics[idx]['reconstruction_algorithm'] == algorithm,
                            #                                                 reconstruction_metrics[idx]['sampling_pattern'] == pattern))
                            plt.title('PSNR: {:.2f} dB / SSIM: {:.2f}'.format(df_current['psnr'].loc[best_idx],
                                                                              df_current['ssim'].loc[best_idx]))
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
                            break
                        finally:
                            h5_file.close()
                if not image_found:
                    print('Could not locate: ' + id_string)
