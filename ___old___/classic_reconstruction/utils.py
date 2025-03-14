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

This module provides common utility functions used across all reconstructions.

Routine listings
----------------
create_database(h5_path, group_structure, expected_rows)
    Function to creating and HDF database to store the results.
decode_str_representation(string)
    Function to decode string representation for use with PyTables.
encode_str_representation(string)
    Function to encode string representation for use with PyTables.
evaluate_reconstruction(original_vec, reconstructed_vec, ssim_dynamic_range,
    h, w)
    Evaluate a reconstruction.
fit_gaussian_model(abs_coefficients, h, w, a=0.0015,
    initial_guess=(0.005,  0.01,  0.01))
    Function to fit a Gaussian model to a 2D array of absolute oefficients.
get_image_subsamples(sampling_pattern, delta, h, w)
    Return image coordinates for pixels included in sampling pattern.
jack_knife_input_images(all_images, test_image, img_folder, downsampling)
    Function to divide images into training samples and test image and load
    them.
save_summary_figure(result_folder, task, var)
    Save a summary figure of a reconstruction result.
store_results(result_folder, h5_name, task, var)
    Store a reconstruction result in an HDF5 database.

"""

from __future__ import division
import json
import os
import warnings

import magni
from magni.utils.validation import decorate_validation as _decorate_validation
from magni.utils.validation import validate_generic as _generic
from magni.utils.validation import validate_levels as _levels
from magni.utils.validation import validate_numeric as _numeric
import numpy as np
from scipy import optimize
from skimage.measure._structural_similarity import structural_similarity as calc_ssim
import tables as tb
import matplotlib.pyplot as plt
import copy


def create_database(h5_path, group_structure, task_setup_summary,
                    expected_rows):
    """
    Create an empty HDF5 database for storing the results.

    The created database has an HDF table for storing reconstruction metrics,
    an array with the group structure hierarchy description, and
    reproducibility chases and annotations.

    Parameters
    ----------
    h5_path : str
        The full path to the database to create.
    group_structure : tuple
        The group structure hierarchy description.
    task_setup_summary : dict
        The dict of all elements to test.
    expected_rows : int
        The number of expected rows in the HDF table.

    """

    @_decorate_validation
    def validate_input():
        _generic('h5_path', 'string')
        _levels('group_structure',
                (_generic(None, tuple),
                 _generic(None, 'string')))
        _generic('task_setup_summary', 'mapping')
        _numeric('expected_rows', 'integer', range_='[1;inf)')

    validate_input()

    class ReconMetrics(tb.IsDescription):
        """
        Table description for table to contain reconstruction performance
        metrics.

        """

        image = tb.StringCol(itemsize=20, pos=0)
        sampling_pattern = tb.StringCol(itemsize=20, pos=1)
        dictionary = tb.StringCol(itemsize=20, pos=2)
        delta = tb.Float64Col(pos=3)
        reconstruction_algorithm = tb.StringCol(itemsize=20, pos=4)
        reconstruction_parameter = tb.Float64Col(pos=5)
        ssim = tb.Float64Col(pos=6)
        psnr = tb.Float64Col(pos=7)
        time = tb.Float64Col(pos=8)

    magni.reproducibility.io.create_database(h5_path)
    with magni.utils.multiprocessing.File(h5_path, mode='a') as h5file:
        sim_group = h5file.create_group('/', 'simulation_results')
        h5file.create_array(sim_group, 'group_structure',
                            obj=json.dumps(group_structure).encode())
        h5file.create_array(sim_group, 'task_setup_summary',
                            obj=json.dumps(task_setup_summary).encode())
        h5file.create_table(sim_group, 'metrics', description=ReconMetrics,
                            expectedrows=expected_rows)


def decode_str_representation(string):
    """Decode string representation for use with PyTables."""
    @_decorate_validation
    def validate_input():
        _generic('string', 'string')

    validate_input()

    return string.replace('__', '-').replace('_', '.')


def encode_str_representation(string):
    """Encode string representation for use with PyTables."""
    @_decorate_validation
    def validate_input():
        _generic('string', 'string')

    validate_input()

    return string.replace('.', '_').replace('-', '__')


def evaluate_reconstruction(original_vec, reconstructed_vec,
                            ssim_dynamic_range, h, w):
    """
    Evaluate a reconstruction.

    The reconstruction is evaluated using the performance indicators
    peak-signal-to-noise-ratio (PSNR) and structural similarity index measure
    (SSIM).

    Parameters
    ----------
    origignal_vec : ndarray
        The original image as a n x 1 vector.
    reconstructed_vec : ndarray
        The reconstructed image as a n x 1 vector.
    ssim_dynamic_range : float or int
        The dynamic range parameter to use in calculating SSIM.
    h : int
        The height in pixels of the coefficient ndarrays.
    w : int
        The width in pixels of the coefficient ndarrays.

    Returns
    -------
    psnr : float
        The calculated PSNR.
    ssim : float
        The calculated SSIM.

    """

    @_decorate_validation
    def validate_input():
        _numeric('original_vec', ('integer', 'floating'), shape=(-1, 1))
        _numeric('reconstructed_vec', ('integer', 'floating'), shape=(-1, 1))
        _numeric('ssim_dynamic_range', 'integer', range_='[1;inf)')
        _numeric('h', 'integer', range_='[1;inf)')
        _numeric('w', 'integer', range_='[1;inf)')

    validate_input()

    psnr = magni.imaging.evaluation.calculate_psnr(
        original_vec, reconstructed_vec, 1.0)
    with warnings.catch_warnings():
        # Ignore warning about need to memcpy
        warnings.simplefilter('ignore')
        ssim = calc_ssim(magni.imaging.vec2mat(original_vec, (h, w)),
                         magni.imaging.vec2mat(reconstructed_vec, (h, w)),
                         dynamic_range=ssim_dynamic_range)

    return psnr, ssim


def fit_gaussian_model(abs_coefficients, h, w, a=0.0015,
                       initial_guess=(0.005,  0.01,  0.01)):
    """
    Fit a Gaussian model to a 2D array of absolute oefficients.

    The fit is based on the average of the absolute coefficients. A sum of
    squares cost function is used in the fit. Powells method is used as the
    solver.

    Parameters
    ----------
    abs_coefficients : list
        The list of absolute coefficients as 2D ndarrays.
    h : int
        The height in pixels of the coefficient ndarrays.
    w : int
        The width in pixels of the coefficient ndarrays.
    a : float
        The scale factor in the Gaussian model.
    initial_guess: tuple
        The initial parameter guess to pass to the optimization solver.

    Returns
    -------
    gaussian_model : ndarray
        The fitted model.

    """

    @_decorate_validation
    def validate_input():
        _numeric('h', 'integer', range_='[1;inf)')
        _numeric('w', 'integer', range_='[1;inf)')
        _levels('abs_coefficients',
                (_generic(None, list),
                 _numeric(None, ('integer', 'floating'), shape=(h, w))))
        _numeric('a', ('integer', 'floating'), range_='(0;inf)')
        _levels('initial_guess',
                (_generic(None, tuple, len_=3),
                 _numeric(None, ('integer', 'floating'))))

    validate_input()

    mean_coefs = np.dstack(abs_coefficients).mean(axis=2)
    xx, yy = np.meshgrid(*map(lambda n: np.linspace(0, 1, n), (h, w)))

    x = magni.imaging.mat2vec(xx)
    y = magni.imaging.mat2vec(yy)
    X = np.hstack([x, y]).T

    # 45 degree rotation matrix to align principle axes of ellipsis
    R = np.array([[np.sqrt(2) / 2, -np.sqrt(2) / 2],
                  [np.sqrt(2) / 2, np.sqrt(2) / 2]])

    def f_model(theta):
        """Gaussian function with "mean" b and "variances" c1, c2."""

        b, c1, c2 = theta

        b = np.array([[b], [b]])
        C_hat_inv = np.array([[1/c1, 0],
                              [0, 1/c2]])
        C_inv = R.dot(C_hat_inv).dot(R.T)

        gaus_model = a * (np.exp(-((X - b) * C_inv.dot((X - b))).sum(axis=0))
                          ).reshape(h, w)

        return gaus_model

    def c_fun(theta):
        """Optimization cost function (sum of squared errors)."""

        return np.linalg.norm(mean_coefs - f_model(theta))**2

    res = optimize.minimize(c_fun, initial_guess, method='Powell')
    assert res.success

    return f_model(res.x)


def get_image_subsamples(sampling_pattern, delta, h, w):
    """
    Return image coordinates for pixels included in sampling pattern.

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
    unique_pixels : ndarray
        The pixel coordinates used in the image sampling.

    """

    @_decorate_validation
    def validate_input():
        _generic('sampling_pattern', 'string')
        _numeric('delta', 'floating', range_='(0;1)')
        _numeric('h', 'integer', range_='[1;inf)')
        _numeric('w', 'integer', range_='[1;inf)')

    validate_input()

    scan_length = delta * 2 * h * w
    num_points = 10 * int(scan_length)  # Make sure to have enough points

    # FIXME: THIS IF-ELSE SHOULD BE HANDLED IN magni.imaging.measurements
    if sampling_pattern == 'rect_spiral':
        img_coords = magni.imaging.measurements.spiral_sample_image(
            h, w, scan_length, num_points, rect_area=True)
    elif sampling_pattern == 'uniform_lines':
        img_coords = magni.imaging.measurements.uniform_line_sample_image(
            h, w, scan_length, num_points)
    else:
        raise ValueError('Invalid sampling pattern: {!r}'.format(
            sampling_pattern))

    unique_pixels = magni.imaging.measurements.unique_pixels(img_coords)

    return unique_pixels


def jack_knife_input_images(all_images, test_image, img_folder, downsampling):
    """
    Divide images into training samples and test image and load them.

    Parameters
    ----------
    all_images : list or tuple
        The list of the names (as strings) of all images used.
    test_image : str
        The name of the image to use as test image.
    img_folder : str
        The path to the folder containing all the images.
    downsampling : int
        The downsampling factor to use when loading the images.

    Returns
    -------
    test_img : ndarray
        The possibly downsampled test image.
    training_imgs : list
        The list of possibly downsampled training images (as ndarrays).

    """

    @_decorate_validation
    def validate_input():
        _levels('all_images',
                (_generic(None, 'explicit collection'),
                 _generic(None, 'string')))
        _generic('test_image', 'string')
        _generic('img_folder', 'string')
        _numeric('downsampling', 'integer', range_='[1;inf)')

    validate_input()

    training_images = copy.copy(list(all_images))
    training_images.remove(test_image)
    training_mi_imgs = [magni.afm.io.read_mi_file(
        img_folder + image).get_buffer('Topography')[0].data
        for image in training_images]
    test_mi_img = magni.afm.io.read_mi_file(
        img_folder + test_image).get_buffer('Topography')[0].data

    # Handle broken images
    if 'image_0.mi' in training_images:
        ix = training_images.index('image_0.mi')
        mod_img = training_mi_imgs[ix]
    elif 'image_0.mi' == test_image:
        mod_img = test_mi_img
    # Remove single outlier
    mod_img = copy.copy(mod_img)  # MI data array is read-only
    mod_img[511, 0] = mod_img[510, 1]

    training_imgs = [magni.imaging.visualisation.stretch_image(
        magni.imaging.preprocessing.detilt(
            mi_img[::downsampling, ::downsampling]), 1.0)
        for mi_img in training_mi_imgs]
    test_img = magni.imaging.visualisation.stretch_image(
        test_mi_img[::downsampling, ::downsampling], 1.0)

    for img in training_imgs:
        assert img.shape == test_img.shape
        assert np.allclose(img.min(), 0.0)
        assert np.allclose(img.max(), 1.0)

    return test_img, training_imgs


def save_summary_figure(result_folder, task, var):
    """
    Save a summary figure of a reconstruction result.

    Parameters
    ----------
    result_folder : str
        The path to the root folder to save result figures to.
    task : dict
        The simulation task specification.
    var : dict
        The local scope of the simulation task to save summary figure for.

    """

    @_decorate_validation
    def validate_input():
        _generic('result_folder', 'string')
        _generic('task', 'mapping')
        _generic('var', 'mapping')

    validate_input()

    measurement_img_vec = magni.imaging.mat2vec(
        magni.imaging.visualisation.mask_img_from_coords(
            var['test_img'], var['unique_coords']))

    reconstruction_log_vec = magni.imaging.visualisation.stretch_image(
        np.log10(magni.imaging.visualisation.stretch_image(
            np.abs(var['reconstructed_coefficients_vec']), 1) + 1e-5), 1)

    if (isinstance(var['algorithm_save_vec'], np.ndarray) and
            var['algorithm_save_vec'].shape == var['test_img_vec'].shape):
        alg_save_vec = magni.imaging.mat2vec(var['algorithm_save_vec'])
    else:
        alg_save_vec = np.zeros_like(var['test_img_vec'])

    figs = [magni.imaging.vec2mat(vec, (var['h'], var['w'])) for vec in
            [var['test_img_vec'], var['test_img_detilt_vec'], var['tilt_vec'],
             alg_save_vec, measurement_img_vec, var['reconstructed_img_vec'],
             var['test_img_detilt_vec'], reconstruction_log_vec]
            ]
    titles = ['Original', 'Detilted', 'Tilt', 'Algorithm Save',
              'Measurements', 'Reconstruction', 'Detilted',
              'Dictionary coefficients\nin reconstruction']

    magni.utils.plotting.setup_matplotlib({'figure': {'figsize': (20, 12)}})
    fig = magni.imaging.visualisation.imsubplot(figs, 2, titles=titles)

    save_path = '/'.join(['/simulation_results',
                          task['test_image'],
                          task['sampling_pattern'],
                          task['dictionary'],
                          'delta-' + str(task['delta']),
                          task['reconstruction_algorithm']])
    save_path = encode_str_representation(save_path)
    out_dir = (result_folder + save_path + '/' +
               encode_str_representation(
                   'param-' + str(task['reconstruction_parameter'])))

    fig.suptitle('{}\n PSNR: {:.2f} dB, SSIM: {:.2f}, time: {:.2f} s'.format(
        out_dir, var['psnr'], var['ssim'], var['reconstruction_time']),
                 fontsize=20)

    if not os.path.isdir(out_dir):
        os.makedirs(out_dir)

    plt.savefig(out_dir + '/summary.png')
    plt.close(fig)


def store_results(result_folder, h5_name, task, var):
    """
    Store a reconstruction result in an HDF5 database.

    The database is stored as `result_folder`/`h5name`. Reconstruction metrics
    from `var` are stored in the database table whereas all arrays from `var`
    are stored in the group structure.

    Parameters
    ----------
    result_folder : str
        The path to the root folder to save result database to.
    h5_name : str
        The name of the HDF5 file in which the results are saved.
    task : dict
        The simulation task specification.
    var : dict
        The local scope of the simulation task to store results from.

    """

    @_decorate_validation
    def validate_input():
        _generic('result_folder', 'string')
        _generic('h5_name', 'string')
        _generic('task', 'mapping')
        _generic('var', 'mapping')

    validate_input()

    h5_file = result_folder + '/' + h5_name
    with magni.utils.multiprocessing.File(h5_file, mode='a') as h5file:
        # Save metrics
        row = h5file.root.simulation_results.metrics.row
        row['image'] = task['test_image']
        row['sampling_pattern'] = task['sampling_pattern']
        row['dictionary'] = task['dictionary']
        row['delta'] = task['delta']
        row['reconstruction_algorithm'] = task['reconstruction_algorithm']
        row['reconstruction_parameter'] = task['reconstruction_parameter']
        row['psnr'] = var['psnr']
        row['ssim'] = var['ssim']
        row['time'] = var['reconstruction_time']
        row.append()

    save_path = '/'.join(['/simulation_results',
                          task['test_image'],
                          task['sampling_pattern'],
                          task['dictionary'],
                          'delta-' + str(task['delta']),
                          task['reconstruction_algorithm']])
    save_path = encode_str_representation(save_path)

    carray_tb_filters = tb.Filters(complevel=1, complib='zlib',
                                   fletcher32=True)
    with magni.utils.multiprocessing.File(h5_file, mode='a') as h5file:
        # Save arrays and tasks
        db_group = h5file.create_group(
            save_path,
            encode_str_representation(
                'param-' + str(task['reconstruction_parameter'])),
            createparents=True)

        h5file.create_carray(db_group, 'test_img_vec', obj=var['test_img_vec'],
                             filters=carray_tb_filters)
        h5file.create_carray(db_group, 'test_img_detilt_vec',
                             obj=var['test_img_detilt_vec'],
                             filters=carray_tb_filters)
        h5file.create_carray(db_group, 'tilt_vec', obj=var['tilt_vec'],
                             filters=carray_tb_filters)
        h5file.create_carray(db_group, 'measurements', obj=var['measurements'],
                             filters=carray_tb_filters)
        h5file.create_carray(db_group, 'unique_coords',
                             obj=var['unique_coords'],
                             filters=carray_tb_filters)
        h5file.create_carray(db_group, 'reconstructed_img_vec',
                             obj=var['reconstructed_img_vec'],
                             filters=carray_tb_filters)
        h5file.create_array(db_group, 'task', obj=json.dumps(task).encode())
        h5file.create_array(db_group, 'img_shape', obj=(var['h'], var['w']))

        if var['reconstructed_coefficients_vec'] is not None:
            h5file.create_carray(db_group, 'reconstructed_coefficients_vec',
                                 obj=var['reconstructed_coefficients_vec'],
                                 filters=carray_tb_filters)

        if var['algorithm_save_vec'] is not None:
            # Custom algorithm saves, e.g., weights in w_ist
            h5file.create_carray(db_group, 'algorithm_save_vec',
                                 obj=var['algorithm_save_vec'],
                                 filters=carray_tb_filters)

if __name__ == "__main__":
    pass