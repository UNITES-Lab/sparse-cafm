# -*- coding: utf-8 -*-
"""
Script for running the simulations described in the paper:

Reconstruction Algorithms in Undersampled AFM Imaging

Copyright (c) 2015, Thomas Arildsen, Christian Schou Oxvig,
    Patrick Steffen Pedersen, Jan Østergaard, and Torben Larsen
All rights reserved.

**Combinations that are tested**

- 7 images
- 2 sampling patterns: Rect Spiral, Uniform lines
- 9 undersampling ratios: 0.10 0.125 0.15 0.175 0.20 0.225 0.25 0.275 0.30
- 1 Dictionary: DCT, DWT (for ell_1 reconstruction)
- 7 Reconstruction algorithms: ell_1, TV, IHT, IST, Interpolation, AMP-L1,
  AMP-BG
- 2 Evaluation indicators: PSNR, SSIM

See the bottom of the script for more details about the specific combinations.

**Requirements**

To run the script, the following must be available:

- the seven AFM images
- magni : http://vbn.aau.dk/en/publications/
          magni(194fc193-7913-4b88-85d6-25570aa43ae1).html, magni_1.3.0
- pyunlockbox : https://github.com/epfl-lts2/pyunlocbox,
  commit:f9fafb070df125c38da86b346a330743e8065910
- amp_bgg_solver.py : bundled with this script

**Output***

All output is placed in the "aspmci_reconstructions" folder. The output is:

- A HDF database ("aspmci_reconstructions.hdf5") containing all results.
- An overview pdf of each reconstruction

"""


from __future__ import division
import json
import os
import time
import warnings

import matplotlib as mpl; mpl.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import scipy.misc
from scipy import interpolate
from skimage.measure import structural_similarity as calc_ssim
import tables as tb
import pandas as pd

import pyunlocbox as ulb
import pywt
import amp_bgg_solver
import magni


def run_simulation_task(img_folder=None, result_folder=None, h5_name=None,
                        task=None, parameters=None):
    """
    Function to run for each simulation task.

    The following elements are part of this simulation:

    * Load and downsample image
    * Scan / sample image
    * Setup reconstruction
    * Reconstruct image
    * Evaluate reconstruction result
    * Save results

    Parameters
    ----------
    img_folder : str
        The path to the folder containing the mi-files.
    result_folder : str
        The path to the root folder to save result figures to.
    h5_name : str
        The name of the HDF5 file in which the results are saved.
    task : dict
        The simulation task specification.

    """

    # Load image
    image_size = 128
    downsampling = int(512 / image_size)
    mi_img = magni.afm.io.read_mi_file(
        img_folder + task['image']).get_buffer('Topography')[0]
    mi_img_data = mi_img.data
    if task['image'] == 'image_0.mi':
        # Remove single outlier
        mi_img_data = mi_img_data.copy()
        mi_img_data[511, 0] = mi_img_data[510, 1]
    img = magni.imaging.visualisation.stretch_image(
        mi_img_data[::downsampling, ::downsampling], 1.0)
    assert img.shape == (image_size, image_size)
    assert np.allclose(img.min(), 0.0)
    assert np.allclose(img.max(), 1.0)
    h, w = img.shape

    # Scanning setup
    img_coords, Phi = get_sampling_setup(
        task['sampling_pattern'], task['delta'], h, w)
    unique_pixels = magni.imaging.measurements.unique_pixels(img_coords)

    # De-tilt based on samples
    scan_mask = np.zeros((h, w), dtype=np.bool_)
    scan_mask[unique_pixels[:, 1], unique_pixels[:, 0]] = True
    img_detilt, tilt = magni.imaging.preprocessing.detilt(
        img, mask=scan_mask, return_tilt=True)

    # Convert image to vector
    img_vec = magni.imaging.mat2vec(img)
    img_detilt_vec = magni.imaging.mat2vec(
        magni.imaging.visualisation.stretch_image(img_detilt, 1.0))
    tilt_vec = magni.imaging.mat2vec(tilt)
    assert np.allclose(img_detilt_vec.min(), 0.0)
    assert np.allclose(img_detilt_vec.max(), 1.0)

    # Reconstruction setup
    Psi = magni.imaging.dictionaries.utils.get_function_handle(
        'matrix', 'DCT')((h, w))
    domain = magni.imaging.domains.MultiDomainImage(Phi, Psi)
    domain.measurements = Phi.dot(img_detilt_vec)

    # Get reconstruction parameter
    if type(task['reconstruction_algorithm']) == tuple:
        assert(len(task['reconstruction_algorithm']) == 2)
        if task['reconstruction_algorithm'][0] == 'ell_1_dct_overc':
            assert task['reconstruction_algorithm'][1] in ['2','3'], 'Invalid identifier \'{}\' + \'{}\''.format(task['reconstruction_algorithm'][0], task['reconstruction_algorithm'][1])
            reconstruction_parameter = parameters.loc[
                task['reconstruction_algorithm'][0] +
                task['reconstruction_algorithm'][1],
                task['sampling_pattern'], task['delta']].values
        elif task['reconstruction_algorithm'][0] == 'ell_1_dwt':
            reconstruction_parameter = parameters.loc[
                task['reconstruction_algorithm'][0] + '_' +
                task['reconstruction_algorithm'][1].replace('20',''),
                task['sampling_pattern'], task['delta']].values
        else:
            raise ValueError('Don\'t know how to combine \'{}\' and \'{}\' to a valid algorithm identifier.'.format(task['reconstruction_algorithm'][0],
                                                                                                                    task['reconstruction_algorithm'][1]))
    elif task['reconstruction_algorithm'] not in ['cubic_interpolation', 'ell_1_amp', 'bg_amp']:
        reconstruction_parameter = parameters.loc[
            task['reconstruction_algorithm'],
            task['sampling_pattern'], task['delta']].values
    else:
        reconstruction_parameter = None
    if reconstruction_parameter:
        assert reconstruction_parameter.size == 1
        reconstruction_parameter = float(reconstruction_parameter)

    # Reconstruction
    reconstructed_img_vec, reconstruction_time, rec_coefs = reconstruct_image(
        task['reconstruction_algorithm'], domain.measurements, Phi, Psi,
        img_coords, h, w, reconstruction_parameter)

    # Evaluation
    psnr = magni.imaging.evaluation.calculate_psnr(
        img_detilt_vec, reconstructed_img_vec, 1.0)

    with warnings.catch_warnings():
        # Ignore irrelevant copy warning
        warnings.simplefilter('ignore')
        ssim = calc_ssim(
            magni.imaging.vec2mat(img_detilt_vec, (h, w)),
            magni.imaging.vec2mat(reconstructed_img_vec, (h, w)),
            dynamic_range=1)

    # Save results in database
    h5_file = result_folder + '/' + h5_name
    with magni.utils.multiprocessing.File(h5_file, mode='a') as h5file:
        # Save metrics
        row = h5file.root.simulation_results.metrics.row
        row['image'] = task['image']
        row['sampling_pattern'] = task['sampling_pattern']
        row['delta'] = task['delta']
        if type(task['reconstruction_algorithm']) == tuple:
            row['reconstruction_algorithm'] = '{0}_{1}'.format(task['reconstruction_algorithm'][0],
                                                               _fix_str_representation(task['reconstruction_algorithm'][1]))
        else:
            row['reconstruction_algorithm'] = task['reconstruction_algorithm']
        row['psnr'] = psnr
        row['ssim'] = ssim
        row['time'] = reconstruction_time
        row.append()

    save_path = '/'.join(['/simulation_results',
                          task['image'],
                          task['sampling_pattern'],
                          'd' + str(task['delta'])])

    save_path = _fix_str_representation(save_path)

    with magni.utils.multiprocessing.File(h5_file, mode='a') as h5file:
        # Save arrays and tasks
        if type(task['reconstruction_algorithm']) == tuple:
            db_group = h5file.create_group(
                save_path,
                '{0}_{1}'.format(task['reconstruction_algorithm'][0],
                                 _fix_str_representation(task['reconstruction_algorithm'][1])),
                createparents=True)
        else:
            db_group = h5file.create_group(
                save_path, task['reconstruction_algorithm'],
                createparents=True)

        h5file.create_array(db_group, 'img_vec', obj=img_vec)
        h5file.create_array(db_group, 'img_detilt_vec', obj=img_detilt_vec)
        h5file.create_array(db_group, 'tilt_vec', obj=tilt_vec)
        h5file.create_array(db_group, 'domain_measurements',
                            obj=domain.measurements)
        h5file.create_array(db_group, 'img_coords', obj=img_coords)
        h5file.create_array(db_group, 'reconstructed_coefficients_vec',
                            obj=rec_coefs)
        h5file.create_array(db_group, 'reconstructed_img_vec',
                            obj=reconstructed_img_vec)
        h5file.create_array(db_group, 'task', obj=json.dumps(task).encode())
        h5file.create_array(db_group, 'img_shape', obj=(h, w))

    # Save summary figures
    measurement_img = magni.imaging.mat2vec(
        magni.imaging.visualisation.mask_img_from_coords(
            img, magni.imaging.measurements.unique_pixels(img_coords)))

    figs = [magni.imaging.vec2mat(fig, (h, w))
            for fig in [measurement_img, img_detilt_vec, reconstructed_img_vec]
            ]
    titles = ['Measurements', 'Original', 'Reconstruction']

    for colormap in ['coolwarm', 'afmhot']:
        magni.utils.plotting.setup_matplotlib(
            {'figure': {'figsize': (20, 12)}}, cmap=colormap)
        fig = magni.imaging.visualisation.imsubplot(figs, 1, titles=titles)

        if type(task['reconstruction_algorithm']) == tuple:
            out_dir = (result_folder + save_path + '/' +
                       '{0}_{1}'.format(task['reconstruction_algorithm'][0],
                                        _fix_str_representation(task['reconstruction_algorithm'][1])))
        else:
            out_dir = (result_folder + save_path + '/' +
                       task['reconstruction_algorithm'])

        fig.suptitle(
            '{}\n PSNR: {:.2f} dB, SSIM: {:.2f}, time: {:.2f} s'.format(
                out_dir, psnr, ssim, reconstruction_time), fontsize=20)

        if not os.path.isdir(out_dir):
            os.makedirs(out_dir)

        plt.savefig(out_dir + '/summary_{}.png'.format(colormap))
        plt.close(fig)


def get_sampling_setup(sampling_pattern, delta, h, w):
    """
    Return image coordinates and matrix representation for sampling pattern.

    Parameters
    ----------
    sampling_pattern : str
        The sampling pattern to use.
    delta : float
        The undersampling ratio.
    h : int
        The image height in pixels.
    w : int
        The image width in pixels.

    Returns
    -------
    img_coords : ndarray
        The Pixel coordinates used in the image sampling.
    Phi : magni.utils.matrices.Matrix
        The sampling matrix operator.

    """

    scan_length = delta * 2 * h * w
    num_points = 10 * int(scan_length)  # Make sure to have enough points

    if sampling_pattern == 'rect_spiral':
        img_coords = magni.imaging.measurements.spiral_sample_image(
            h, w, scan_length, num_points, rect_area=True)

    elif sampling_pattern == 'uniform_lines':
        img_coords = magni.imaging.measurements.uniform_line_sample_image(
            h, w, scan_length, num_points)

    else:
        raise ValueError('Invalid sampling pattern: {!r}'.format(
            sampling_pattern))

    Phi = magni.imaging.measurements.construct_measurement_matrix(
        img_coords, h, w)

    return img_coords, Phi


def reconstruct_image(algorithm, measurements, Phi, Psi, img_coords,
                      h, w, parameter=None):
    """
    Return a reconstructed image along with the reconstruction time.

    Parameters
    ----------
    algorithm : str
        The reconstruction algorithm to use.
    measurements : ndarray
        The m x 1 vector of measurments.
    Phi : magni.utils.matrices.Matrix
        The measurements matrix operator.
    Psi : magni.utils.matrices.Matrix
        The dictionary matrix operator.
    img_coords : ndarray
        The m x 2 array of measurment coordinates.
    h : int
        The image height in pixels.
    w : int
        The image width in pixels.

    Returns
    -------
    reconstructed_img_vec : ndarray
        The n x 1 vector representing the reconstructed image.
    reconstrution_time : float
        The time in seconds it took to do the reconstruction.
    reconstructed_coefficients : ndarray
        The sparse coefficients in the reconstruction.


    """

    WAVELET_LEVELS = 5

    if type(algorithm) == tuple:
        assert(len(algorithm) == 2)
        variant = algorithm[1]
        algorithm = algorithm[0]

    if algorithm == 'iht_fixed':
        assert not parameter == None
        A = magni.utils.matrices.MatrixCollection((Phi, Psi))
        threshold_fixed = int(parameter * h * w)
        magni.cs.reconstruction.it.config.update(
            {'threshold': 'fixed', 'threshold_fixed': threshold_fixed})

        t0 = time.time()
        alpha = magni.cs.reconstruction.it.run(measurements, A)
        x = Psi.dot(alpha)
        t1 = time.time()

    elif algorithm == 'ist_fixed':
        assert not parameter == None
        A = magni.utils.matrices.MatrixCollection((Phi, Psi))
        threshold_fixed = int(parameter * h * w)
        magni.cs.reconstruction.it.config.update(
            {'threshold': 'fixed', 'threshold_fixed': threshold_fixed})
        magni.cs.reconstruction.it.config.update(
            {'kappa_fixed': 0.6, 'threshold_operator': 'soft'})
        t0 = time.time()
        alpha = magni.cs.reconstruction.it.run(measurements, A)
        x = Psi.dot(alpha)
        t1 = time.time()
        magni.cs.reconstruction.it.config.reset()

    elif algorithm == 'cubic_interpolation':
        xx, yy = np.meshgrid(*map(np.arange, (h, w)))
        unique_pixels = magni.imaging.measurements.unique_pixels(img_coords)
        t0 = time.time()
        recon = interpolate.griddata(unique_pixels, measurements.ravel(),
                                     (xx, yy), method='cubic',
                                     fill_value=measurements.mean())
        x = magni.imaging.mat2vec(recon)
        t1 = time.time()
        alpha = Psi.T.dot(x)

    elif algorithm == 'ell_1_optim':
        assert not parameter == None
        A = magni.utils.matrices.MatrixCollection((Phi, Psi))

        def Afunc(x):
            return A.dot(x)

        def Atfunc(x):
            return A.T.dot(x)

        f1 = ulb.functions.norm_l1()
        f2 = ulb.functions.proj_b2(epsilon=parameter, y=measurements,
                                   A=Afunc, At=Atfunc, tight=False)
        solver = ulb.solvers.douglas_rachford()
        x0 = np.zeros((A.shape[1], 1))

        t0 = time.time()
        solution = ulb.solvers.solve([f1, f2], x0, solver, rtol=1e-9,
                                     maxit=300)
        alpha = solution['sol']
        x = Psi.dot(alpha)
        t1 = time.time()

    elif algorithm == 'ell_1_dct_overc':
        assert not parameter == None
        h_larger = int(variant) * h
        w_larger = int(variant) * w
        Psi2 = magni.imaging.dictionaries.utils.get_function_handle(
            'matrix', 'DCT')((h_larger, w_larger))
        def Afunc(x):
            if len(x.shape) == 1:
                x = x.reshape(x.shape[0],1)
            return Phi.dot(_zero_unpad_corner(Psi2.dot(x), (h, w), (h_larger, w_larger)))
        def Atfunc(x):
            if len(x.shape) == 1:
                x = x.reshape(x.shape[0],1)
            return Psi2.T.dot(_zero_pad_corner(Phi.T.dot(x), (h, w), (h_larger, w_larger)))
        t0 = time.time()
        # The following lines set up a solver from PyUNLocBox
        f1 = ulb.functions.norm_l1()
        f2 = ulb.functions.proj_b2(epsilon=parameter, y=measurements,
                                   A=Afunc, At=Atfunc, tight=False)
        solver = ulb.solvers.douglas_rachford()
        x0 = np.zeros((h_larger*w_larger,1))
        solution = ulb.solvers.solve([f1, f2], x0, solver, rtol=1e-9,
                                     maxit=500) 
        alpha = solution['sol']
        x = _zero_unpad_corner(Psi2.dot(solution['sol']), (h, w), (h_larger, w_larger))
        t1 = time.time()

    elif algorithm == 'ell_1_dct_overc_other':
        assert not parameter == None
        h_larger = int(variant) * h
        w_larger = int(variant) * w
        Psi2 = magni.imaging.dictionaries.utils.get_function_handle(
            'matrix', 'DCT')((h_larger, w_larger))
        def Afunc(x):
            if len(x.shape) == 1:
                x = x.reshape(x.shape[0],1)
            return Phi.dot(_zero_unpad_center(Psi2.dot(x), (h, w), (h_larger, w_larger)))
        def Atfunc(x):
            if len(x.shape) == 1:
                x = x.reshape(x.shape[0],1)
            return Psi2.T.dot(_zero_pad_center(Phi.T.dot(x), (h, w), (h_larger, w_larger)))
        t0 = time.time()
        # The following lines set up a solver from PyUNLocBox
        f1 = ulb.functions.norm_l1()
        f2 = ulb.functions.proj_b2(epsilon=parameter, y=measurements,
                                   A=Afunc, At=Atfunc, tight=False)
        solver = ulb.solvers.douglas_rachford()
        x0 = np.zeros((h_larger*w_larger,1))
        solution = ulb.solvers.solve([f1, f2], x0, solver, rtol=1e-9,
                                     maxit=500) 
        alpha = solution['sol']
        x = _zero_unpad_center(Psi2.dot(solution['sol']), (h, w), (h_larger, w_larger))
        t1 = time.time()

    elif algorithm == 'tv_optim':
        assert not parameter == None
        unique_pixels = magni.imaging.measurements.unique_pixels(img_coords)
        mask = np.zeros((h, w), dtype=np.bool_)
        mask[unique_pixels[:, 1], unique_pixels[:, 0]] = True

        meas_mat = magni.imaging.vec2mat(Phi.T.dot(measurements), (h, w))

        def Afunc(x):
            return mask * x

        f1 = ulb.functions.norm_tv()
        f2 = ulb.functions.proj_b2(epsilon=parameter, y=meas_mat,
                                   A=Afunc, At=Afunc)
        solver = ulb.solvers.douglas_rachford()
        x0 = meas_mat

        t0 = time.time()
        solution = ulb.solvers.solve([f1, f2], x0, solver, rtol=1e-9,
                                     maxit=300)
        x = magni.imaging.mat2vec(solution['sol'])
        alpha = Psi.T.dot(x)
        t1 = time.time()

    elif algorithm == 'ell_1_dwt':
        assert not parameter == None
        unique_pixels = magni.imaging.measurements.unique_pixels(img_coords)
        mask = np.zeros((h, w), dtype=np.bool_)
        mask[unique_pixels[:, 1], unique_pixels[:, 0]] = True

        meas_mat = magni.imaging.vec2mat(Phi.T.dot(measurements), (h, w))

        def Afunc(X):
            return mask * pywt.waverec2(_pywt_unpack_wavelets2(X, WAVELET_LEVELS), variant, 'per')

        def Atfunc(X):
            coeffs = pywt.wavedec2(mask * X, variant, 'per', level=WAVELET_LEVELS)
            return _pywt_pack_wavelets2(coeffs)

        f1 = ulb.functions.norm_l1(dim=2)
        f2 = ulb.functions.proj_b2(epsilon=parameter, y=meas_mat,
                                   A=Afunc, At=Atfunc)
        solver = ulb.solvers.douglas_rachford()
        x0 = np.zeros_like(meas_mat)

        t0 = time.time()
        solution = ulb.solvers.solve([f1, f2], x0, solver, rtol=1e-9,
                                     maxit=300)
        alpha = solution['sol']
        x = magni.imaging.mat2vec(pywt.waverec2(_pywt_unpack_wavelets2(alpha, WAVELET_LEVELS), variant, 'per'))
        t1 = time.time()

    elif algorithm == 'ell_1_swt_old':
        unique_pixels = magni.imaging.measurements.unique_pixels(img_coords)
        mask = np.zeros((h, w), dtype=np.bool_)
        mask[unique_pixels[:, 1], unique_pixels[:, 0]] = True

        meas_mat = magni.imaging.vec2mat(Phi.T.dot(measurements), (h, w))

        def Afunc(X):
            return_val = mask * pywt.iswt2(_pywt_unpack_swt2_redundant(X, WAVELET_LEVELS), 'bior3.9')
            return return_val

        def Atfunc(X):
            coeffs = pywt.swt2(mask * X, variant, level=WAVELET_LEVELS)
            return _pywt_pack_swt2_redundant(coeffs)

        f1 = ulb.functions.norm_l1(dim=2)
        f2 = ulb.functions.proj_b2(epsilon=1e-3*np.linalg.norm(measurements),
                                   y=meas_mat, A=Afunc, At=Atfunc)
        solver = ulb.solvers.douglas_rachford(step=1e-2)
        x0 = np.zeros((2 * mask.shape[0], WAVELET_LEVELS * 2 * mask.shape[1]))

        t0 = time.time()
        solution = ulb.solvers.solve([f1, f2], x0, solver, maxit=500)
        alpha = solution['sol']
        x = magni.imaging.mat2vec(pywt.iswt2(_pywt_unpack_swt2_redundant(alpha, WAVELET_LEVELS), variant))
        t1 = time.time()

    elif algorithm == 'ell_1_swt':
        unique_pixels = magni.imaging.measurements.unique_pixels(img_coords)
        mask = np.zeros((h, w), dtype=np.bool_)
        mask[unique_pixels[:, 1], unique_pixels[:, 0]] = True

        meas_mat = magni.imaging.vec2mat(Phi.T.dot(measurements), (h, w))

        def Afunc(X):
            return_val = mask * pywt.iswt2(_pywt_unpack_swt2(X, WAVELET_LEVELS), wavelet_variant)
            return return_val

        def Atfunc(X):
            coeffs = pywt.swt2(mask * X, wavelet_variant, level=WAVELET_LEVELS)
            return _pywt_pack_swt2(coeffs)

        f1 = ulb.functions.norm_l1(dim=2)
        f2 = ulb.functions.proj_b2(epsilon=1e-3*np.linalg.norm(measurements),
                                   y=meas_mat, A=Afunc, At=Atfunc)
        solver = ulb.solvers.douglas_rachford(step=1e-2)
        x0 = np.zeros(((WAVELET_LEVELS * 3 + 1) * mask.shape[0], mask.shape[1]))

        t0 = time.time()
        solution = ulb.solvers.solve([f1, f2], x0, solver, maxit=500)
        alpha = solution['sol']
        x = magni.imaging.mat2vec(pywt.iswt2(_pywt_unpack_swt2(alpha, WAVELET_LEVELS), wavelet_variant))
        t1 = time.time()

    elif algorithm == 'ell_1_amp':
        A = magni.utils.matrices.MatrixCollection((Phi, Psi))
        alpha = np.zeros((A.shape[1], 1))
        y = measurements
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

        x = Psi.dot(alpha)
        t1 = time.time()

    elif algorithm == 'bg_amp':
        A = magni.utils.matrices.MatrixCollection((Phi, Psi)).A
        rho = 0.3
        theta_bar = 0.0
        theta_hat = 1.0
        d_s = 1.0E-6
        T = 300
        prior_prmts = {'rho': rho, 'theta_bar': theta_bar,
                       'theta_hat': theta_hat}
        y = measurements.flatten()

        t0 = time.time()
        alpha, _dummy = amp_bgg_solver.camp(y, A, d_s, prior_prmts, T, 1.0E-3)
        x = Psi.dot(alpha)
        t1 = time.time()

    else:
        raise ValueError('Invalid reconstruction algorithm: {!r}'.format(
            algorithm))

    reconstructed_img_vec = magni.imaging.visualisation.stretch_image(x, 1.0)
    reconstruction_time = t1 - t0
    reconstructed_coefficients = alpha

    return (reconstructed_img_vec, reconstruction_time,
            reconstructed_coefficients)


def get_tasks(job_id=None):
    """
    The task setup, i.e. the list of dicts of tasks for the workers.

    """

    import random
    import math

    ASSUMED_NO_NODES = 4

    # Combinations to test
    images = tuple(['image_{}.mi'.format(k) for k in range(7)])
    #images = ('image_2.mi',)
    sampling_patterns = ('rect_spiral', 'uniform_lines')
    #sampling_patterns = ('rect_spiral',)
    undersampling_ratios = np.linspace(0.1, 0.3, 9)
    #undersampling_ratios = (0.2,)
    reconstruction_algorithms = (
        'iht_fixed',
        'ist_fixed',
        'cubic_interpolation',
        'tv_optim',
        'bg_amp',
        'ell_1_optim',
        'ell_1_amp',
        ('ell_1_dct_overc', '2'),
        ('ell_1_dct_overc', '3'),
        ('ell_1_dwt', 'db20'),
        ('ell_1_dwt', 'sym20'),
        ('ell_1_dwt', 'dmey'))

    # Resulting tasks
    tasks = [{'image': image,
              'sampling_pattern': sampling_pattern,
              'delta': undersampling_ratio,
              'reconstruction_algorithm': reconstruction_algorithm}
             for image in images
             for sampling_pattern in sampling_patterns
             for undersampling_ratio in undersampling_ratios
             for reconstruction_algorithm in reconstruction_algorithms]

    # Group structure
    group_structure = tuple(['image', 'sampling_pattern',
                             'undersampling_ratio', 'reconstruction_algorithm']
                            )

    if job_id is not None:
        random.seed(123)
        random.shuffle(tasks)  # Shuffle tasks for "stochastic load balancing"
        chunk_size = int(math.ceil(len(tasks)/ASSUMED_NO_NODES))
        return group_structure, tasks[job_id * chunk_size:(job_id + 1) * chunk_size]
    else:
        return group_structure, tasks


def create_database(h5_path, group_structure):
    """
    Create an empty database for storing the results.

    """

    class ReconMetrics(tb.IsDescription):
        """
        Table description for table to contain reconstruction performance
        metrics.

        """

        image = tb.StringCol(itemsize=10, pos=0)
        sampling_pattern = tb.StringCol(itemsize=20, pos=1)
        delta = tb.Float64Col(pos=2)
        reconstruction_algorithm = tb.StringCol(itemsize=30, pos=3)
        psnr = tb.Float64Col(pos=4)
        ssim = tb.Float64Col(pos=5)
        time = tb.Float64Col(pos=6)

    magni.reproducibility.io.create_database(h5_path)
    with magni.utils.multiprocessing.File(h5_path, mode='a') as h5file:
        sim_group = h5file.create_group('/', 'simulation_results')
        h5file.create_array(sim_group, 'group_structure',
                            obj=json.dumps(group_structure).encode())
        h5file.create_table(sim_group, 'metrics', description=ReconMetrics,
                            expectedrows=1000)


def _fix_str_representation(string):
    return string.replace('.', '_').replace('-', '_')


def _zero_pad_corner(x, shape_orig, shape_padded):
    """
    Takes a vector as input, reshapes it to a matrix, zero-pads it
    with the matrix in the upper-left corner, and reshapes the result
    back into a vector.

    """

    X_unpadded = magni.imaging.vec2mat(x, shape_orig)
    X_padded = np.pad(X_unpadded, ((0, shape_padded[0]-shape_orig[0]), (0, shape_padded[1]-shape_orig[1])), mode = 'constant')
    return magni.imaging.mat2vec(X_padded)


def _zero_unpad_corner(x, shape_orig, shape_padded):
    """
    Takes a vector as input, reshapes it to a matrix, removes
    zero-pading around the upper-left corner, and reshapes the result
    back into a vector.

    """

    X_padded = magni.imaging.vec2mat(x, shape_padded)
    X_unpadded = X_padded[0:shape_orig[0], 0:shape_orig[1]]
    return magni.imaging.mat2vec(X_unpadded)


def _zero_pad_center(x, shape_orig, shape_padded):
    """
    Takes a vector as input, reshapes it to a matrix, zero-pads it
    with the matrix in the center, and reshapes the result back into a
    vector.

    """

    X_unpadded = magni.imaging.vec2mat(x, shape_orig)
    X_padded = np.pad(X_unpadded, (((shape_padded[0]-shape_orig[0])/2, (shape_padded[0]-shape_orig[0])/2), ((shape_padded[1]-shape_orig[1])/2, (shape_padded[1]-shape_orig[1])/2)), mode = 'constant')
    return magni.imaging.mat2vec(X_padded)


def _zero_unpad_center(x, shape_orig, shape_padded):
    """
    Takes a vector as input, reshapes it to a matrix, removes
    zero-pading around the center, and reshapes the result back into a
    vector.

    """

    X_padded = magni.imaging.vec2mat(x, shape_padded)
    X_unpadded = X_padded[(shape_padded[0]-shape_orig[0])/2:(shape_padded[0]-shape_orig[0])/2+shape_orig[0], (shape_padded[1]-shape_orig[1])/2:(shape_padded[1]-shape_orig[1])/2+shape_orig[1]]
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


def _pywt_pack_swt2_redundant(coeffs):
    """
    Takes a list of wavelet coefficients returned by `pywt`'s `swt2`
    function and packs them into a matrix.

    """

    inner_shape = coeffs[0][0].shape
    coeff_matrix = np.empty((2 * inner_shape[0], len(coeffs) * 2 * inner_shape[1]))
    for idx, inner_coeffs in enumerate(coeffs):
        coeff_matrix[:,(idx * 2 * inner_shape[1]):((idx+1) * 2 * inner_shape[1])] = np.vstack((np.hstack((inner_coeffs[0], inner_coeffs[1][0])), np.hstack((inner_coeffs[1][1], inner_coeffs[1][2]))))
    return coeff_matrix


def _pywt_unpack_swt2_redundant(coeff_matrix, levels):
    """
    Takes a matrix of wavelet coefficients (as returned by
    `_pywt_pack_swt2_redundant`) and unpacks it into the list of
    tuples of coefficient matrices required by `pywt`'s `iswt2`
    function.
    """

    matrix_per_level = np.hsplit(coeff_matrix, levels)
    coeffs = []
    for level_matrix in matrix_per_level:
        top, bottom = np.vsplit(level_matrix, 2)
        cA, cH = np.hsplit(top, 2)
        cV, cD = np.hsplit(bottom, 2)
        coeffs.append((cA, (cH, cV, cD)))
    return coeffs


def _pywt_pack_swt2(coeffs):
    """
    Takes a list of wavelet coefficients returned by `pywt`'s `swt2`
    function and packs them into a matrix.

    """

    inner_shape = coeffs[0][0].shape
    coeff_matrix = np.empty(((len(coeffs) * 3 + 1) * inner_shape[0], inner_shape[1]))
    for idx, inner_coeffs in enumerate(coeffs):
        coeff_matrix[(idx * 3 * inner_shape[0]):((idx+1) * 3 * inner_shape[0]),:] = np.vstack((inner_coeffs[1][0], inner_coeffs[1][1], inner_coeffs[1][2]))
    coeff_matrix[((idx+1) * 3 * inner_shape[0]):(((idx+1) * 3 + 1) * inner_shape[0]),:] = inner_coeffs[0]
    return coeff_matrix


def _pywt_unpack_swt2(coeff_matrix, levels):
    """
    Takes a matrix of wavelet coefficients (as returned by
    `_pywt_pack_swt2`) and unpacks it into the list of tuples of
    coefficient matrices required by `pywt`'s `iswt2` function.
    """

    matrices = np.vsplit(coeff_matrix, 3 * levels + 1)
    coeffs = []
    for level in range(levels):
        if level == levels - 1:
            cA = matrices[level * 3 + 3]
        else:
            cA = np.nan * np.ones_like(matrices[0])
        cH = matrices[level * 3]
        cV = matrices[level * 3 + 1]
        cD = matrices[level * 3 + 2]
        coeffs.append((cA, (cH, cV, cD)))
    return coeffs


# Run the simulation
if __name__ == '__main__':
    import sys
    try:
        JOB_ID = int(sys.argv[1])
    except:
        JOB_ID = None

    use_multiprocessing = True
    result_folder = './data/'
    if not os.path.isdir(result_folder):
        os.makedirs(result_folder)

    img_folder = './orig_images/'
    h5_name = 'aspmci_reconstructions.hdf5'

    parameter_store = pd.HDFStore(img_folder + 'best_parameters_128.hdf5')
    best_parameters = parameter_store['reconstruction_parameters']

    group_structure, tasks = get_tasks(JOB_ID)
    print('Number of tasks: {}'.format(len(tasks)))  # Debugging
    h5_path = result_folder + h5_name
    create_database(h5_path, group_structure)

    if use_multiprocessing:
        kwargs = [{'img_folder': img_folder, 'result_folder':
                   result_folder, 'h5_name': h5_name, 'task': task,
                   'parameters': best_parameters} for task in tasks]

        magni.utils.multiprocessing.config.update(workers=8)
        magni.utils.multiprocessing.process(
            run_simulation_task, kwargs_list=kwargs, maxtasks=1)
    else:
        for task in tasks:
            run_simulation_task(img_folder=img_folder,
                                result_folder=result_folder,
                                h5_name=h5_name, task=task,
                                parameters=best_parameters)

    parameter_store.close()
