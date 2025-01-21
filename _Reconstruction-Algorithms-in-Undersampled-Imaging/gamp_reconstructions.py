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

This module provides generalized approximate message passing like
reconstruction methods.

Routine listings
----------------
_common_post_processing(t0, alpha, t1, Psi)
    Post processing common to all gamp reconstruction algorithms.
ell_1_amp(var)
    Basic approximate message passing.

"""

from __future__ import division
import time

import numpy as np
import magni

import utils as _utils


def _common_post_processing(t0, alpha, t1, Psi):
    """Post processing common to all gamp reconstruction algorithms."""

    reconstructed_img_vec = magni.imaging.visualisation.stretch_image(
        Psi.dot(alpha), 1.0)
    reconstruction_time = t1 - t0
    reconstructed_coefficients_vec = alpha

    return (reconstructed_img_vec, reconstruction_time,
            reconstructed_coefficients_vec)


def ell_1_amp(var):
    """Basic approximate message passing."""
    #FIXME: Insert more useful docstring.
    A = var['A']
    alpha = np.zeros((var['A'].shape[1], 1))
    y = var['measurements']
    z = y.copy()

    t0 = time.time()
    for it in range(300):
        alpha_z = alpha + A.T.dot(z)

        thres = np.sort(np.abs(alpha_z).ravel())[-A.shape[0]]
        a_t_p = (alpha_z > thres)
        a_t_m = (alpha_z < -thres)
        alpha = (alpha_z - thres) * a_t_p + (alpha_z + thres) * a_t_m

        r = y - A.dot(alpha)

        z = r + z * 1/A.shape[0] * np.sum(a_t_p + a_t_m)

        if np.linalg.norm(r) < 1e-3 * np.linalg.norm(y):
            break

    t1 = time.time()

    result = _common_post_processing(t0, alpha, t1, var['Psi']) + tuple([None])

    return result
