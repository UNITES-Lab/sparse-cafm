"""
Copyright (c) 2015,
Christian Schou Oxvig, Thomas Arildsen, and Torben Larsen
Aalborg University, Department of Electronic Systems, Signal and Information
Processing, Fredrik Bajers Vej 7, DK-9220 Aalborg, Denmark.

All rights reserved.

Redistribution and use in source and binary forms, with or without
modification, are permitted provided that the following conditions are met:

1. Redistributions of source code must retain the above copyright notice, this
list of conditions and the following disclaimer.

2. Redistributions in binary form must reproduce the above copyright notice,
this list of conditions and the following disclaimer in the documentation
and/or other materials provided with the distribution.

THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS" AND
ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED
WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.


A goblet of reconstructions of undersampled atomic force microscopy images.

This module provides interpolation like reconstruction methods.

Routine listings
----------------
cubic_interpolation(var)
    Cubic interpolation by triangulation.
linear_interpolation(var)
    Linear interpolation by triangulation.
naturaln_interpolation(var)
    Natural neighbour interpolation.
nearestn_interpolation(var)
    Nearest neighbour interpolation.

"""

from __future__ import division
import time
import warnings

import magni
import numpy as np
from scipy import interpolate
from matplotlib import mlab

import utils as _utils


def cubic_interpolation(var):
    """Cubic interpolation by triangulation."""

    xx, yy = np.meshgrid(*map(np.arange, (var["h"], var["w"])))

    t0 = time.time()
    reconstructed_img = interpolate.griddata(
        var["unique_coords"],
        var["measurements"].ravel(),
        (xx, yy),
        method="cubic",
        fill_value=var["measurements"].mean(),
    )
    t1 = time.time()

    reconstructed_img_vec = magni.imaging.mat2vec(
        magni.imaging.visualisation.stretch_image(reconstructed_img, 1.0)
    )
    reconstruction_time = t1 - t0
    reconstructed_coefficients_vec = var["Psi"].T.dot(reconstructed_img_vec)

    return (
        reconstructed_img_vec,
        reconstruction_time,
        reconstructed_coefficients_vec,
        None,
    )


def linear_interpolation(var):
    """Linear interpolation by triangulation."""

    xx, yy = np.meshgrid(*map(np.arange, (var["h"], var["w"])))

    t0 = time.time()
    reconstructed_img = interpolate.griddata(
        var["unique_coords"],
        var["measurements"].ravel(),
        (xx, yy),
        method="linear",
        fill_value=var["measurements"].mean(),
    )
    t1 = time.time()

    reconstructed_img_vec = magni.imaging.mat2vec(
        magni.imaging.visualisation.stretch_image(reconstructed_img, 1.0)
    )
    reconstruction_time = t1 - t0
    reconstructed_coefficients_vec = var["Psi"].T.dot(reconstructed_img_vec)

    return (
        reconstructed_img_vec,
        reconstruction_time,
        reconstructed_coefficients_vec,
        None,
    )


def naturaln_interpolation(var):
    """Natural neighbour interpolation."""

    xx, yy = np.meshgrid(*map(np.arange, (var["h"], var["w"])))

    # The natural neighbour interpolation implemented in Matplotlib uses either
    # matplotlibs own delaunay module (which is deprecated in
    # matplotlib==1.4.0) or the external mpl_toolkits.natgrid.
    # See also:
    # https://github.com/matplotlib/matplotlib/commit/
    #         a65066d72411e682043652afd1e51d404b85ee68

    try:
        t0 = time.time()
        reconstructed_img = mlab.griddata(
            var["unique_coords"][:, 0],
            var["unique_coords"][:, 1],
            var["measurements"].ravel(),
            xx,
            yy,
            interp="nn",
        )
        t1 = time.time()
    except RuntimeError as e:
        t0 = 0
        t1 = 0
        reconstructed_img = np.zeros((var["h"], var["w"]))
        reconstructed_img[0, 0] = 1
        warnings.warn("NN interpolation failed\n{}".format(e.args[0]), RuntimeWarning)
    else:
        if isinstance(reconstructed_img, np.ma.MaskedArray):
            reconstructed_img = reconstructed_img.filled(var["measurements"].mean())

    reconstructed_img_vec = magni.imaging.mat2vec(
        magni.imaging.visualisation.stretch_image(reconstructed_img, 1.0)
    )
    reconstruction_time = t1 - t0
    reconstructed_coefficients_vec = var["Psi"].T.dot(reconstructed_img_vec)

    return (
        reconstructed_img_vec,
        reconstruction_time,
        reconstructed_coefficients_vec,
        None,
    )


def nearestn_interpolation(var):
    """Nearest neighbour interpolation."""

    xx, yy = np.meshgrid(*map(np.arange, (var["h"], var["w"])))

    t0 = time.time()
    reconstructed_img = interpolate.griddata(
        var["unique_coords"],
        var["measurements"].ravel(),
        (xx, yy),
        method="nearest",
        fill_value=var["measurements"].mean(),
    )
    t1 = time.time()

    reconstructed_img_vec = magni.imaging.mat2vec(
        magni.imaging.visualisation.stretch_image(reconstructed_img, 1.0)
    )
    reconstruction_time = t1 - t0
    reconstructed_coefficients_vec = var["Psi"].T.dot(reconstructed_img_vec)

    return (
        reconstructed_img_vec,
        reconstruction_time,
        reconstructed_coefficients_vec,
        None,
    )
