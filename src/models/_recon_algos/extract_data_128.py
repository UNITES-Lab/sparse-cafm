"""
This script extracts data and calculates and plots statistics for
the ASPMCI paper from a specified experiment results HDF5 file.

"""

from __future__ import division
import itertools

import matplotlib as mpl
import matplotlib.pyplot as plt
plt.style.use('ggplot')
import numpy as np
import pandas as pd
from os import makedirs
import errno
from collections import OrderedDict

import magni

# Setup - the following path can be changed if the result file is in a
# different location. This should correspond to the `result_folder`
# variable in 'aspmci_reconstructions.py', line 488:
data_path = 'data/'
hdf_database_name = '5694_merged_hdf_reconstruction_goblet_ID_.hdf5'
hdf_additional_names = ['5668_0_aspmci_reconstructions.hdf5',
                        '5668_1_aspmci_reconstructions.hdf5']
try:
    mpl.rc('text', usetex=False)
except AttributeError:
    None

undersampling_ratios = np.linspace(0.1, 0.3, 9)

# Load merged metrics database
hdf_stores = []
hdf_store = pd.HDFStore(data_path + hdf_database_name)
reconstruction_metrics = hdf_store.select('/merged_metrics')
reconstruction_metrics.drop('array_job', axis=1, inplace=True)
assert len(reconstruction_metrics) == len(reconstruction_metrics.drop_duplicates())  # Assert no duplicate rows
hdf_stores.append(hdf_store)
reconstruction_metrics_add = pd.DataFrame()
for filename in hdf_additional_names:
    hdf_store = pd.HDFStore(data_path + filename)
    reconstruction_metrics_add = reconstruction_metrics_add.append(
        hdf_store.select('/simulation_results/metrics'),
        ignore_index=True)
    hdf_stores.append(hdf_store)
additional_metrics = reconstruction_metrics_add.query('reconstruction_algorithm == \'cubic_interpolation\' or reconstruction_algorithm == \'bg_amp\'')
reconstruction_metrics = reconstruction_metrics.append(
    additional_metrics, ignore_index=True)

pattern_names = {'rect_spiral': 'spiral', 'uniform_lines': 'raster'}
recon_names = OrderedDict([('ell_1_optim', '$\ell_1$ (DCT)'),
                           ('ell_1_dwt_db', '$\ell_1$ (DWT - Daubechies)'),
                           ('ell_1_dwt_dmey', '$\ell_1$ (DWT - Meyer)'),
                           ('ell_1_dwt_sym', '$\ell_1$ (DWT - Symlet)'),
                           ('ell_1_dct_overc2', '$\ell_1$ (DCT - $2\\times\ 2$ overc.)'),
                           ('ell_1_dct_overc3', '$\ell_1$ (DCT - $3\\times\ 3$ overc.)'),
                           ('ell_1_amp', 'Laplace AMP (DCT)'),
                           ('iht_fixed', 'IHT (DCT)'),
                           ('ist_fixed', 'IST (DCT)'),
                           ('tv_optim', 'TV'),
                           ('cubic_interpolation', 'Interpolation'),
                           ('bg_amp', 'Bernoulli-Gauss AMP')])
line_specs = ('-','--')
magni.utils.plotting.setup_matplotlib({'figure': {'figsize': (10, 3)},
                                       'axes': {'color_cycle':
                                                ((0.2081, 0.1663, 0.5292),
                                                 (0.1403, 0.3147, 0.8168),
                                                 (0.0410, 0.4502, 0.8685),
                                                 (0.0734, 0.5410, 0.8257),
                                                 (0.0232, 0.6407, 0.7925),
                                                 (0.1024, 0.6984, 0.6934),
                                                 (0.3187, 0.7395, 0.5625),
                                                 (0.5745, 0.7484, 0.4479),
                                                 (0.7798, 0.7361, 0.3658),
                                                 (0.9613, 0.7281, 0.2774),
                                                 (0.9763, 0.8328, 0.1590),
                                                 (0.9763, 0.9831, 0.0538)) }});

for idx, samp_patt in enumerate(reconstruction_metrics['sampling_pattern'].unique()):
    fig1, axes1 = plt.subplots(1, 3)
    for k, rec_algo in enumerate(reconstruction_metrics['reconstruction_algorithm'].unique()):
        # Carve out a DataFrame of values for the current reconstruction algorithm and sampling pattern
        df_current = reconstruction_metrics.query('reconstruction_algorithm == @rec_algo and sampling_pattern == @samp_patt')
        # This will average over the images
        if rec_algo in ['cubic_interpolation', 'bg_amp']:
            df_current_mean = df_current.groupby('delta').mean()
        else:
            df_current_mean = df_current.groupby(['delta','reconstruction_parameter']).mean()
        # Locate the results with best SSIM
        ssim_idx = df_current_mean['ssim'].groupby(level=0).idxmax().values
        # Locate the results with best PSNR
        psnr_idx = df_current_mean['psnr'].groupby(level=0).idxmax().values

        # Plot curves
        axes1[0].plot(undersampling_ratios,
                      df_current_mean.loc[psnr_idx]['psnr'],
                      line_specs[k%len(line_specs)],
                      label=recon_names[rec_algo])
        axes1[1].plot(undersampling_ratios,
                      df_current_mean.loc[psnr_idx]['ssim'],
                      line_specs[k%len(line_specs)],
                      label=recon_names[rec_algo])
        # Skip BG AMP because it takes so long - it makes the other
        # curves in the figure too hard to distinguish
        if rec_algo == 'bg_amp':
            axes1[2].plot(undersampling_ratios,
                          np.nan * np.ones_like(df_current_mean.loc[psnr_idx]['time'].values),
                          line_specs[k%len(line_specs)],
                          label=recon_names[rec_algo])
        else:
            axes1[2].plot(undersampling_ratios,
                          df_current_mean.loc[psnr_idx]['time'],
                          line_specs[k%len(line_specs)],
                          label=recon_names[rec_algo])
    axes1[2].set_yscale('log')
    axes1[0].set_ylim([10,45])
    axes1[1].set_ylim([0,1])
    axes1[0].set_xlabel('Undersampling ratio, $\delta$')
    axes1[1].set_xlabel('Undersampling ratio, $\delta$')
    axes1[2].set_xlabel('Undersampling ratio, $\delta$')
    axes1[0].set_ylabel('PSNR [dB]')
    axes1[1].set_ylabel('SSIM')
    axes1[2].set_ylabel('Time [s]')
    if idx == 1:
        axes1[1].legend(loc='lower center', ncol = 4, borderaxespad=-9,
                       borderpad=.6)
    try:
        makedirs('../figures/fig3/')
    except OSError, e:
        if e.errno != errno.EEXIST:
            throw
    plt.tight_layout()
    fig1.savefig('../figures/fig3/graphs-psnr-ssim-time-{}-128.pdf'.format(samp_patt),
                 bbox_inches='tight', pad_inches=0)
    plt.close(fig1)

for hdf_store in hdf_stores:
    hdf_store.close()
