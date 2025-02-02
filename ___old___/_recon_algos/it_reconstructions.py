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

This module provides iterative thresholding like reconstruction methods.

Routine listings
----------------
_common_post_processing(t0, alpha, t1, Psi)
    Post processing common to all IT reconstruction algorithms.
_fit_weights_model(var)
    Fit a model of weights.
iht_far(var)
    Iterative Hard Tresholding with False Alarm Rate Heuristic.
iht_fixed(var)
    Iterative Hard Tresholding with fixed sparsity level.
w_iht(var)
    Weighted Iterative Hard Therhsolding.
ist_far(var)
    Iterative Soft Tresholding with False Alarm Rate Heuristic.
ist_fixed(var)
    Iterative Soft Tresholding with fixed sparsity level.
w_ist(var)
    Weighted Iterative Soft Therhsolding.

"""

from __future__ import division
import time

import numpy as np
import magni

import utils as _utils


def _common_post_processing(t0, alpha, t1, Psi):
    """Post processing common to all IT reconstruction algorithms."""

    reconstructed_img_vec = magni.imaging.visualisation.stretch_image(
        Psi.dot(alpha), 1.0)
    reconstruction_time = t1 - t0
    reconstructed_coefficients_vec = alpha

    return (reconstructed_img_vec, reconstruction_time,
            reconstructed_coefficients_vec)


def _fit_weights_model(var):
    """Fit a model of weights."""
    training_coefs_vecs = [var['Psi'].T.dot(vec)
                           for vec in var['training_img_vecs']]
    training_abs_coefs = [magni.imaging.visualisation.stretch_image(
        np.abs(magni.imaging.vec2mat(vec, (var['h'], var['w']))), 1)
        for vec in training_coefs_vecs]

    if var['task']['dictionary'] == 'DCT':
        gaus_model = _utils.fit_gaussian_model(
            training_abs_coefs, var['h'], var['w'],
            a=var['fixed_quantities']['gaussian_model_a_parameter'])
        W = magni.imaging.mat2vec(
            magni.imaging.visualisation.stretch_image(
                gaus_model, 1.0, min_val=1e-3))
    else:
        raise ValueError('No model is known for the {!r} dictionary.'.format(
            var['task']['Dictionary']))

    return W


def iht_far(var):
    """Iterative Hard Tresholding with False Alarm Rate Heuristic."""

    t0 = time.time()
    alpha = magni.cs.reconstruction.it.run(var['measurements'], var['A'])
    t1 = time.time()
    magni.cs.reconstruction.it.config.reset()

    result = _common_post_processing(t0, alpha, t1, var['Psi']) + tuple([None])

    return result


def iht_fixed(var):
    """Iterative Hard Tresholding with fixed sparsity level."""

    threshold_fixed = int(
        var['task']['reconstruction_parameter'] * var['h'] * var['w'])

    magni.cs.reconstruction.it.config.update(
        {'threshold': 'fixed', 'threshold_fixed': threshold_fixed})
    t0 = time.time()
    alpha = magni.cs.reconstruction.it.run(var['measurements'], var['A'])
    t1 = time.time()
    magni.cs.reconstruction.it.config.reset()

    result = _common_post_processing(t0, alpha, t1, var['Psi']) + tuple([None])

    return result


def w_iht(var):
    """Weighted Iterative Hard Therhsolding."""

    threshold_fixed = int(
        var['task']['reconstruction_parameter'] * var['h'] * var['w'])
    W = _fit_weights_model(var)

    magni.cs.reconstruction.it.config.update(
        {'threshold': 'fixed', 'threshold_fixed': threshold_fixed,
         'threshold_operator': 'weighted_hard', 'threshold_weights': W})

    t0 = time.time()
    alpha = magni.cs.reconstruction.it.run(var['measurements'], var['A'])
    t1 = time.time()
    magni.cs.reconstruction.it.config.reset()

    result = _common_post_processing(t0, alpha, t1, var['Psi']) + tuple([W])

    return result


def ist_far(var):
    """Iterative Soft Tresholding with False Alarm Rate Heuristic."""

    magni.cs.reconstruction.it.config.update({
        'kappa_fixed': 0.6, 'threshold_operator': 'soft'})
    t0 = time.time()
    alpha = magni.cs.reconstruction.it.run(var['measurements'], var['A'])
    t1 = time.time()
    magni.cs.reconstruction.it.config.reset()

    result = _common_post_processing(t0, alpha, t1, var['Psi']) + tuple([None])

    return result


def ist_fixed(var):
    """Iterative Soft Tresholding with fixed sparsity level."""

    threshold_fixed = int(
        var['task']['reconstruction_parameter'] * var['h'] * var['w'])

    magni.cs.reconstruction.it.config.update(
        {'kappa_fixed': 0.6, 'threshold_operator': 'soft',
         'threshold': 'fixed', 'threshold_fixed': threshold_fixed})
    t0 = time.time()
    alpha = magni.cs.reconstruction.it.run(var['measurements'], var['A'])
    t1 = time.time()
    magni.cs.reconstruction.it.config.reset()

    result = _common_post_processing(t0, alpha, t1, var['Psi']) + tuple([None])

    return result


def w_ist(var):
    """Weighted Iterative Soft Therhsolding."""

    threshold_fixed = int(
        var['task']['reconstruction_parameter'] * var['h'] * var['w'])
    W = _fit_weights_model(var)

    magni.cs.reconstruction.it.config.update(
        {'kappa_fixed': 0.6, 'threshold': 'fixed',
         'threshold_fixed': threshold_fixed,
         'threshold_operator': 'weighted_soft', 'threshold_weights': W})
    t0 = time.time()
    alpha = magni.cs.reconstruction.it.run(var['measurements'], var['A'])
    t1 = time.time()
    magni.cs.reconstruction.it.config.reset()

    result = _common_post_processing(t0, alpha, t1, var['Psi']) + tuple([W])

    return result
