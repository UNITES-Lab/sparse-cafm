"""
This script extracts the original images for the ASPMCI paper from
a specified experiment results HDF5 file.

"""

from __future__ import division
import itertools

#import matplotlib as mpl
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
img_folder = './orig_images/'

# Load image
image_size = 256
decimation = int(512 / image_size)
image_names = tuple(['image_{}.mi'.format(k) for k in range(7)])
images = []
for image_name in image_names:
    mi_img = magni.afm.io.read_mi_file(
        img_folder + image_name).get_buffer('Topography')[0]
    mi_img_data = mi_img.data
    if image_name == 'image_0.mi':
        # Remove single outlier
        mi_img_data = mi_img_data.copy()
        mi_img_data[511, 0] = mi_img_data[510, 1]
    images.append(magni.imaging.visualisation.stretch_image(
        mi_img_data[::decimation, ::decimation], 1.0))
    assert images[-1].shape == (image_size, image_size)
    assert np.allclose(images[-1].min(), 0.0)
    assert np.allclose(images[-1].max(), 1.0)
    h, w = images[-1].shape

# Plot images
try:
    makedirs('../figures/fig2/')
except OSError, e:
    if e.errno != errno.EEXIST:
        throw
for image, image_name in zip(images, image_names):
    # figsize chosen arbitrarily, but match it to dpi in line 52
    plt.figure(figsize=(4,4))
    fig = magni.imaging.visualisation.imshow(image, show_axis='none')
    # Axis modifications suggested in http://stackoverflow.com/a/26610602/865169
    plt.axis('off')
    fig.axes.get_xaxis().set_visible(False)
    fig.axes.get_yaxis().set_visible(False)
    # dpi was chosen to match figzise in line 53 to obtain 128x128
    # resolution
    plt.savefig('../figures/fig2/' + image_name.replace('mi','png'),
                bbox_inches='tight', pad_inches=0, dpi=82.75)

# Generate colorbar for cool-warm colormap
y = np.array([0, 1])
x = np.linspace(0,1,1024)
X, Y = np.meshgrid(x, y)
fig, axes = plt.subplots(1, 1)
p_mesh = axes.pcolormesh(X, Y, X, vmin=0, vmax=1, edgecolor='face',
                         cmap='coolwarm')
plt.axis('off')
fig.axes[0].get_xaxis().set_visible(False)
fig.axes[0].get_yaxis().set_visible(False)
try:
    makedirs('../figures/fig1/')
except OSError, e:
    if e.errno != errno.EEXIST:
        throw
plt.savefig('../figures/fig1/colorbar-coolwarm.png',
            bbox_inches='tight', pad_inches=0)
p_mesh = axes.pcolormesh(X, Y, X, vmin=0, vmax=1, edgecolor='face',
                         cmap='afmhot')
plt.savefig('../figures/fig1/colorbar-afmhot.png',
            bbox_inches='tight', pad_inches=0)
