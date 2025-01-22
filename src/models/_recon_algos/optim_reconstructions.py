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

This module provides convex optimization like reconstruction methods.

Routine listings
----------------
_common_post_processing(t0, alpha, t1, Psi)
    Post processing common to all optimization reconstruction algorithms.
ell_1_optim(var)
    Minimization of squared error subject to ell_1 sparsity constraint.
tv_optim(var)
    Minimization of squared error subject to total variation constraint.

"""

from __future__ import division
import time

import numpy as np
import magni
import pyunlocbox as ulb
import pywt

import utils as _utils

WAVELET_LEVELS = 5


def _common_post_processing(t0, alpha, t1, Psi):
    """Post processing common to all optimization reconstruction algorithms."""

    reconstructed_img_vec = magni.imaging.visualisation.stretch_image(
        Psi.dot(alpha), 1.0)
    reconstruction_time = t1 - t0
    reconstructed_coefficients_vec = alpha

    return (reconstructed_img_vec, reconstruction_time,
            reconstructed_coefficients_vec)


def ell_1_optim(var):
    """Minimization of squared error subject to ell_1 sparsity
    constraint.  Uses DCT dictionary.

    """

    def Afunc(x):
        return var['A'].dot(x)

    def Atfunc(x):
        return var['A'].T.dot(x)

    epsilon = var['task']['reconstruction_parameter']
    #FIXME: 1e-3*np.linalg.norm(var['measurements'])

    f1 = ulb.functions.norm_l1()
    f2 = ulb.functions.proj_b2(epsilon=epsilon, y=var['measurements'],
                               A=Afunc, At=Atfunc, tight=False)
    solver = ulb.solvers.douglas_rachford()
    alpha0 = np.zeros((var['A'].shape[1], 1))

    t0 = time.time()
    solution = ulb.solvers.solve([f1, f2], alpha0, solver, rtol=1e-9,
                                 maxit=2000)
    alpha = solution['sol']
    t1 = time.time()

    result = _common_post_processing(t0, alpha, t1, var['Psi']) + tuple([None])

    return result


def ell_1_dct_overcomplete2(var):
    """Minimization of squared error subject to ell_1 sparsity
    constraint.  Uses 2x2-overcomplete DCT dictionary.

    """

    h_larger = int(2*var['h'])
    w_larger = int(2*var['w'])
    Psi2 = magni.imaging.dictionaries.utils.get_function_handle(
        'matrix', 'DCT')((h_larger, w_larger))
    def Afunc(x):
        if len(x.shape) == 1:
            x = x.reshape(x.shape[0],1)
        return var['Phi'].dot(_zero_unpad(Psi2.dot(x), (var['h'], var['w']),
                                          (h_larger, w_larger)))
    def Atfunc(x):
        if len(x.shape) == 1:
            x = x.reshape(x.shape[0],1)
        return Psi2.T.dot(_zero_pad(var['Phi'].T.dot(x), (var['h'], var['w']),
                                    (h_larger, w_larger)))
    epsilon = var['task']['reconstruction_parameter']

    f1 = ulb.functions.norm_l1()
    f2 = ulb.functions.proj_b2(epsilon=epsilon, y=var['measurements'],
                               A=Afunc, At=Atfunc, tight=False)
    solver = ulb.solvers.douglas_rachford()
    alpha0 = np.zeros((h_larger*w_larger,1))

    t0 = time.time()
    solution = ulb.solvers.solve([f1, f2], alpha0, solver, rtol=1e-9,
                                 maxit=2000)
    alpha = solution['sol']
    t1 = time.time()

    reconstruction_time = t1 - t0
    reconstructed_img_vec = _zero_unpad(Psi2.dot(alpha), (var['h'], var['w']),
                                        (h_larger, w_larger))
    # Caution: only part of alpha is returned because the caller of
    # this function does not support the fact that it is overcomplete,
    # i.e. has more entries than the corresponding image.
    return (reconstructed_img_vec, reconstruction_time,
            alpha[:var['h'] * var['w']]) + tuple([None])


def ell_1_dct_overcomplete3(var):
    """Minimization of squared error subject to ell_1 sparsity
    constraint.  Uses 3x3-overcomplete DCT dictionary.

    """

    h_larger = int(3*var['h'])
    w_larger = int(3*var['w'])
    Psi2 = magni.imaging.dictionaries.utils.get_function_handle(
        'matrix', 'DCT')((h_larger, w_larger))
    def Afunc(x):
        if len(x.shape) == 1:
            x = x.reshape(x.shape[0],1)
        return var['Phi'].dot(_zero_unpad(Psi2.dot(x), (var['h'], var['w']),
                                          (h_larger, w_larger)))
    def Atfunc(x):
        if len(x.shape) == 1:
            x = x.reshape(x.shape[0],1)
        return Psi2.T.dot(_zero_pad(var['Phi'].T.dot(x), (var['h'], var['w']),
                                    (h_larger, w_larger)))
    epsilon = var['task']['reconstruction_parameter']

    f1 = ulb.functions.norm_l1()
    f2 = ulb.functions.proj_b2(epsilon=epsilon, y=var['measurements'],
                               A=Afunc, At=Atfunc, tight=False)
    solver = ulb.solvers.douglas_rachford()
    alpha0 = np.zeros((h_larger*w_larger,1))

    t0 = time.time()
    solution = ulb.solvers.solve([f1, f2], alpha0, solver, rtol=1e-9,
                                 maxit=2000)
    alpha = solution['sol']
    t1 = time.time()

    reconstruction_time = t1 - t0
    reconstructed_img_vec = _zero_unpad(Psi2.dot(alpha), (var['h'], var['w']),
                                        (h_larger, w_larger))
    # Caution: only part of alpha is returned because the caller of
    # this function does not support the fact that it is overcomplete,
    # i.e. has more entries than the corresponding image.
    return (reconstructed_img_vec, reconstruction_time,
            alpha[:var['h'] * var['w']]) + tuple([None])


def ell_1_dwt_dmey(var):
    """Minimization of squared error subject to ell_1 sparsity
    constraint.  Uses dicretised Meyer DWT dictionary.

    """
    mask = np.zeros((var['h'], var['w']), dtype=np.bool_)
    mask[var['unique_coords'][:, 1], var['unique_coords'][:, 0]] = True

    meas_mat = magni.imaging.vec2mat(var['Phi'].T.dot(var['measurements']), (var['h'], var['w']))

    def Afunc(X):
        return mask * pywt.waverec2(_pywt_unpack_wavelets2(X, WAVELET_LEVELS), 'dmey', 'per')

    def Atfunc(X):
        coeffs = pywt.wavedec2(mask * X, 'dmey', 'per', level=WAVELET_LEVELS)
        return _pywt_pack_wavelets2(coeffs)
    epsilon = var['task']['reconstruction_parameter']

    f1 = ulb.functions.norm_l1(dim=2)
    f2 = ulb.functions.proj_b2(epsilon=epsilon, y=meas_mat, A=Afunc,
                               At=Atfunc)
    solver = ulb.solvers.douglas_rachford()
    alpha0 = np.zeros_like(meas_mat)

    t0 = time.time()
    solution = ulb.solvers.solve([f1, f2], alpha0, solver, rtol=1e-9,
                                 maxit=2000)
    alpha = solution['sol']
    t1 = time.time()

    reconstruction_time = t1 - t0
    reconstructed_img_vec = magni.imaging.mat2vec( pywt.waverec2(
        _pywt_unpack_wavelets2( alpha, WAVELET_LEVELS), 'dmey', 'per'))
    return (reconstructed_img_vec, reconstruction_time,
            magni.imaging.mat2vec(alpha)) + tuple([None])


def ell_1_dwt_sym(var):
    """Minimization of squared error subject to ell_1 sparsity
    constraint.  Uses dicretised Meyer DWT dictionary.

    """
    mask = np.zeros((var['h'], var['w']), dtype=np.bool_)
    mask[var['unique_coords'][:, 1], var['unique_coords'][:, 0]] = True

    meas_mat = magni.imaging.vec2mat(var['Phi'].T.dot(var['measurements']), (var['h'], var['w']))

    def Afunc(X):
        return mask * pywt.waverec2(_pywt_unpack_wavelets2(X, WAVELET_LEVELS), 'sym20', 'per')

    def Atfunc(X):
        coeffs = pywt.wavedec2(mask * X, 'sym20', 'per', level=WAVELET_LEVELS)
        return _pywt_pack_wavelets2(coeffs)
    epsilon = var['task']['reconstruction_parameter']

    f1 = ulb.functions.norm_l1(dim=2)
    f2 = ulb.functions.proj_b2(epsilon=epsilon, y=meas_mat, A=Afunc,
                               At=Atfunc)
    solver = ulb.solvers.douglas_rachford()
    alpha0 = np.zeros_like(meas_mat)

    t0 = time.time()
    solution = ulb.solvers.solve([f1, f2], alpha0, solver, rtol=1e-9,
                                 maxit=2000)
    alpha = solution['sol']
    t1 = time.time()

    reconstruction_time = t1 - t0
    reconstructed_img_vec = magni.imaging.mat2vec( pywt.waverec2(
        _pywt_unpack_wavelets2( alpha, WAVELET_LEVELS), 'sym20', 'per'))
    return (reconstructed_img_vec, reconstruction_time,
            magni.imaging.mat2vec(alpha)) + tuple([None])


def ell_1_dwt_db(var):
    """Minimization of squared error subject to ell_1 sparsity
    constraint.  Uses dicretised Meyer DWT dictionary.

    """
    mask = np.zeros((var['h'], var['w']), dtype=np.bool_)
    mask[var['unique_coords'][:, 1], var['unique_coords'][:, 0]] = True

    meas_mat = magni.imaging.vec2mat(var['Phi'].T.dot(var['measurements']), (var['h'], var['w']))

    def Afunc(X):
        return mask * pywt.waverec2(_pywt_unpack_wavelets2(X, WAVELET_LEVELS), 'db20', 'per')

    def Atfunc(X):
        coeffs = pywt.wavedec2(mask * X, 'db20', 'per', level=WAVELET_LEVELS)
        return _pywt_pack_wavelets2(coeffs)
    epsilon = var['task']['reconstruction_parameter']

    f1 = ulb.functions.norm_l1(dim=2)
    f2 = ulb.functions.proj_b2(epsilon=epsilon, y=meas_mat, A=Afunc,
                               At=Atfunc)
    solver = ulb.solvers.douglas_rachford()
    alpha0 = np.zeros_like(meas_mat)

    t0 = time.time()
    solution = ulb.solvers.solve([f1, f2], alpha0, solver, rtol=1e-9,
                                 maxit=2000)
    alpha = solution['sol']
    t1 = time.time()

    reconstruction_time = t1 - t0
    reconstructed_img_vec = magni.imaging.mat2vec( pywt.waverec2(
        _pywt_unpack_wavelets2( alpha, WAVELET_LEVELS), 'db20', 'per'))
    return (reconstructed_img_vec, reconstruction_time,
            magni.imaging.mat2vec(alpha)) + tuple([None])


def tv_optim(var):
    """Minimization of squared error subject to total variation constraint."""

    meas_mat = magni.imaging.vec2mat(
        var['Phi'].T.dot(var['measurements']), (var['h'], var['w']))

    def Afunc(x):
        return var['measurement_mask'] * x

    epsilon = var['task']['reconstruction_parameter']
    #FIXME: 1e-3*np.linalg.norm(var['measurements'])

    f1 = ulb.functions.norm_tv()
    f2 = ulb.functions.proj_b2(epsilon=epsilon, y=meas_mat, A=Afunc, At=Afunc)
    solver = ulb.solvers.douglas_rachford(step=1e-2)
    x0 = meas_mat

    t0 = time.time()
    solution = ulb.solvers.solve([f1, f2], x0, solver, rtol=1e-9, maxit=2000)
    x = magni.imaging.mat2vec(solution['sol'])
    t1 = time.time()

    alpha = var['Psi'].T.dot(x)

    result = _common_post_processing(t0, alpha, t1, var['Psi']) + tuple([None])

    return result


def _zero_pad(x, shape_orig, shape_padded):
    """
    Takes a vector as input, reshapes it to a matrix, zero-pads it,
    and reshapes the result back into a vector.

    """

    X_unpadded = magni.imaging.vec2mat(x, shape_orig)
    X_padded = np.pad(X_unpadded, ((0, shape_padded[0]-shape_orig[0]), (0, shape_padded[1]-shape_orig[1])), mode = 'constant')
    return magni.imaging.mat2vec(X_padded)


def _zero_unpad(x, shape_orig, shape_padded):
    """
    Takes a vector as input, reshapes it to a matrix, removes zero-pading,
    and reshapes the result back into a vector.

    """

    X_padded = magni.imaging.vec2mat(x, shape_padded)
    X_unpadded = X_padded[0:shape_orig[0], 0:shape_orig[1]]
    return magni.imaging.mat2vec(X_unpadded)


def _pywt_pack_wavelets2(coeffs):
    """
    Takes a list of wavelet coefficients returned by `pywt`'s
    `wavedec2` function (must use 'per' mode) and packs them into a
    matrix with the classic multi-level structure.
    """

    coeff_matrix = coeffs[0]
    for level in range(1,len(coeffs)):
        assert(len(coeffs[level])==3)
        coeff_matrix = np.vstack((np.hstack((coeff_matrix, coeffs[level][0])), np.hstack((coeffs[level][1], coeffs[level][2]))))
    return coeff_matrix


def _pywt_unpack_wavelets2(coeff_matrix, levels):
    """
    Takes a matrix of wavelet coefficients with the classic
    multi-level structure and unpacks it into the list of coefficient
    matrices required by `pywt`'s `waverec2` function (must use 'per'
    mode).

    Assumes square matrices!
    """

    assert(not np.ptp(coeff_matrix.shape)) # Ensure square matrix
    assert(coeff_matrix.shape[0] >= 2**levels)
    size = coeff_matrix.shape[0]/(2**levels)
    coeffs = [coeff_matrix[:size,:size]]
    for level in range(1,levels+1):
        coeffs.append((coeff_matrix[:size,size:2*size],
                       coeff_matrix[size:2*size,:size],
                       coeff_matrix[size:2*size,size:2*size]))
        size *= 2
    return coeffs
