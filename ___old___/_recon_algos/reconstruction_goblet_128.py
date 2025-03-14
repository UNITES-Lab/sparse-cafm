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

This module provides the main functionality of the reconstruction goblet.

Routine listings
----------------
get_tasks(ID, nodes, algorithms)
    Get the task setup, i.e. the list of dicts of tasks for the workers.
run_simulation_task(img_folder=None, result_folder=None, h5_name=None,
    task=None, fixed_quantities=None):
    Run a simulation task.

Notes
-----
Local scope parameters passed in "var" to reconstruction algorithm

Reconstruction algorithm return

    reconstructed_img_vec = reconstruction_result[0]
        The n x 1 vector representing the reconstructed image.
    reconstruction_time = reconstruction_result[1]
        The time in seconds it took to do the reconstruction.
    reconstructed_coefficients_vec = reconstruction_result[2]
        The sparse coefficients in the reconstruction.
    algorithm_save_vec = reconstruction_result[3]
        The algorithm specific details needed in the reconstruction.

"""

from __future__ import division
import argparse
from collections import OrderedDict
import os
from pprint import pprint
import random

import matplotlib as mpl; mpl.use('Agg')
import magni
from magni.utils.validation import decorate_validation as _decorate_validation
from magni.utils.validation import validate_generic as _generic
from magni.utils.validation import validate_levels as _levels
from magni.utils.validation import validate_numeric as _numeric
import numpy as np
import psutil
import copy

import gamp_reconstructions as _gamp_recon
import interp_reconstructions as _interp_recon
import it_reconstructions as _it_recon
import optim_reconstructions as _optim_recon
import utils as _utils

RECONSTRUCTION_ALGORITHM_HANDLES = OrderedDict([
    ('iht_fixed', _it_recon.iht_fixed),
    ('ist_fixed', _it_recon.ist_fixed),
    ('ell_1_optim', _optim_recon.ell_1_optim),
    ('tv_optim', _optim_recon.tv_optim),
    ('ell_1_amp', _gamp_recon.ell_1_amp),
    ('ell_1_dwt_dmey', _optim_recon.ell_1_dwt_dmey),
    ('ell_1_dwt_sym', _optim_recon.ell_1_dwt_sym),
    ('ell_1_dwt_db', _optim_recon.ell_1_dwt_db),
    ('ell_1_dct_overc2', _optim_recon.ell_1_dct_overcomplete2),
    ('ell_1_dct_overc3', _optim_recon.ell_1_dct_overcomplete3)])


def get_tasks(ID, nodes, algorithms, shuffle=True):
    """
    Get the task setup, i.e. the list of dicts of tasks for the workers.

    **Combinations that are tested**
    - 7 images
    - 2 sampling pattern: Rect Spiral, uniform lines
    - 1 Dictionary: DCT (plus some others "hidden" as other recon. algs.
    - 9 undersampling ratios: np.linspace(0.10, 0.30, 9)
    - 25 Reconstruction algorithm parameters
    - 10 Reconstruction algorithms: IHT, IST, ell_1 DCT, ell_1 (2-/3-)overcomplete DCT,
                                    ell_1 DWT (dmey, sym20, db20)
    - 2 Quality indicators: PSNR, SSIM

    Parameters
    ----------
    ID : int
        The compute node ID.
    nodes : int
        The total number of compute nodes.
    algorithms : str or list
        The reconstruction algorithms to test. Must be the string 'all' or a
        list of specific algorithms (as strings).
    shuffle : bool
        The indicator of whether or not to shuffle the list of tasks.

    Returns
    -------
    group_structure : list
        The group structure used in the task set.
    task_setup_summary : dict
        The dict of all elements to test.
    tasks : list
        The the list of simulation tasks for the given node.

    """

    # Input validation
    @_decorate_validation
    def validate_input():
        _numeric('ID', 'integer', range_='[0;inf)')
        _numeric('nodes', 'integer', range_='[1;inf)')
        if algorithms != 'all':
            _levels('algorithms',
                    (_generic(None, 'explicit collection'),
                     _generic(None, 'string',
                              value_in=RECONSTRUCTION_ALGORITHM_HANDLES.keys()
                              )))

    validate_input()

    # Combinations to test
    all_images = tuple(['image_{}.mi'.format(k) for k in range(7)])
    sampling_patterns = tuple(['rect_spiral', 'uniform_lines'])
    # Other dictionaries are included as well, but they are
    # implemented as separate reconstruction algorithms:
    dictionaries = tuple(['DCT'])
    undersampling_ratios = np.linspace(0.1, 0.3, 9)
    if algorithms == 'all':
        reconstruction_algorithms = list(
            # RECONSTRUCTION_ALGORITHM_HANDLES *must* be ordered
            RECONSTRUCTION_ALGORITHM_HANDLES.keys())
    else:
        reconstruction_algorithms = algorithms

    optim_epsilon = sorted(
        [1e-9, 1e-8, 1e-7, 1e-6, 1e-5, 1e-4, 1e-3, 5e-3, 1e-2, 5e-2, 1e-1,
         0.2, 0.4, 0.6, 0.8, 1.0, 1.5, 2.0, 3.0, 4.0, 5.0, 10, 15, 20, 30])
    assert len(optim_epsilon) == 25
    reconstruction_parameters = {'cubic_interp': [1],
                                 'linear_interp': [1],
                                 'naturaln_interp': [1],
                                 'nearestn_interp': [1],
                                 'iht_far': [1],
                                 'iht_fixed': np.linspace(0.05, 0.40, 25),
                                 'w_iht': np.linspace(0.05, 0.40, 25),
                                 'ist_far': [1],
                                 'ist_fixed': np.linspace(0.05, 0.40, 25),
                                 'w_ist': np.linspace(0.05, 0.40, 25),
                                 'ell_1_optim': optim_epsilon,
                                 'tv_optim': optim_epsilon,
                                 'ell_1_amp': [1],
                                 'ell_1_dwt_dmey': optim_epsilon,
                                 'ell_1_dwt_sym': optim_epsilon,
                                 'ell_1_dwt_db': optim_epsilon,
                                 'ell_1_dct_overc2': optim_epsilon,
                                 'ell_1_dct_overc3': optim_epsilon}


    for algorithm in reconstruction_algorithms:
        assert algorithm in reconstruction_parameters

    # All tasks
    # MAKE SURE THIS LIST IS FULLY DETERMINISTIC
    all_tasks = [{'all_images': all_images,
                  'test_image': test_image,
                  'sampling_pattern': sampling_pattern,
                  'dictionary': dictionary,
                  'delta': undersampling_ratio,
                  'reconstruction_algorithm': reconstruction_algorithm,
                  'reconstruction_parameter': reconstruction_parameter}
                 for test_image in all_images
                 for sampling_pattern in sampling_patterns
                 for dictionary in dictionaries
                 for reconstruction_algorithm in reconstruction_algorithms
                 for reconstruction_parameter in reconstruction_parameters[
                         reconstruction_algorithm]
                 for undersampling_ratio in undersampling_ratios]  # Least sig.

    # Random subset of all tasks
    if shuffle:
        print('Now shuffling task array')
        random.seed(6021)
        true_state = (2147483648, 1618339863, 519244099, 1595962633,
                      1576686576, 4132336775, 1122096546, 3605922522, 90459623,
                      796897237, 1627086794, 496464278, 3157893, 3701555684,
                      2225625825, 2236838988, 1104720552, 1671259614,
                      4118423168, 624)
        current_state = random.getstate()
        test_state = current_state[1][:10] + current_state[1][-10:]
        for k, elem in enumerate(test_state):
            # Assert correct state of random generator
            assert elem == true_state[k]

        random.shuffle(all_tasks)  # Shuffle task list

    list_of_tasks = [all_tasks[k::nodes] for k in range(nodes)]
    tasks = list_of_tasks[ID]

    # Group structure
    group_structure = tuple(['test_image', 'sampling_pattern', 'dictionary',
                             'undersampling_ratio', 'reconstruction_algorithm',
                             'reconstruction_parameter'])

    # Task setup summary
    if isinstance(undersampling_ratios, np.ndarray):
        undersamp_ratios = undersampling_ratios.tolist()
    else:
        undersamp_ratios = undersampling_ratios

    recon_params = copy.copy(reconstruction_parameters)
    for key, val in recon_params.items():
        if isinstance(val, np.ndarray):
            recon_params[key] = val.tolist()

    task_setup_summary = OrderedDict([
        ('all_images', all_images),
        ('sampling_patterns', sampling_patterns),
        ('dictionaries', dictionaries),
        ('undersampling_ratios', undersamp_ratios),
        ('reconstruction_algorithms', reconstruction_algorithms),
        ('reconstruction_parameters', recon_params)])

    return group_structure, task_setup_summary, tasks


def run_simulation_task(img_folder=None, result_folder=None, h5_name=None,
                        task=None, fixed_quantities=None):
    """
    Run a simulation task.

    The following elements are part of this simulation:

    * Load, downsample, and detilt image
    * Fit model of DCT domain
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

    # Input validation
    @_decorate_validation
    def validate_input():
        _generic('img_folder', 'string')
        _generic('result_folder', 'string')
        _generic('h5_name', 'string')
        _generic('task', 'mapping')
        _generic('fixed_quantities', 'mapping')

    validate_input()

    # 1. Load test image (and training images)
    img_base_shape = magni.afm.io.read_mi_file(
        img_folder + task['test_image']).get_buffer('Topography')[0].data.shape
    assert img_base_shape[0] == img_base_shape[1]
    downsampling = int(img_base_shape[0] / fixed_quantities['image_size'])
    assert downsampling >= 1
    test_img, training_imgs = _utils.jack_knife_input_images(
        task['all_images'], task['test_image'], img_folder, downsampling)

    # 2. Get system setup
    h, w = test_img.shape  # Height and width of image
    assert [h, w] == [fixed_quantities['image_size']] * 2
    unique_coords = _utils.get_image_subsamples(
        task['sampling_pattern'], task['delta'], h, w)  # measurment coords
    Phi = magni.imaging.measurements.construct_measurement_matrix(
        unique_coords, h, w)  # Sampling operator
    Psi = magni.imaging.dictionaries.utils.get_function_handle(
        'matrix', task['dictionary'])((h, w))  # Dictionary
    A = magni.utils.matrices.MatrixCollection((Phi, Psi))

    # 3. De-tilt test image based on samples
    measurement_mask = magni.imaging.measurements.construct_pixel_mask(
            h, w, unique_coords)
    test_img_detilt, tilt = magni.imaging.preprocessing.detilt(
        test_img, mask=measurement_mask, return_tilt=True)

    # 4. Convert images to vectors
    training_img_vecs = [magni.imaging.mat2vec(img) for img in training_imgs]
    test_img_vec = magni.imaging.mat2vec(test_img)
    test_img_detilt_vec = magni.imaging.mat2vec(
        magni.imaging.visualisation.stretch_image(test_img_detilt, 1.0))
    tilt_vec = magni.imaging.mat2vec(tilt)
    assert np.allclose(test_img_detilt_vec.min(), 0.0)
    assert np.allclose(test_img_detilt_vec.max(), 1.0)

    # 5. Take measurements
    measurements = Phi.dot(test_img_detilt_vec)
    assert measurements.shape[0] == unique_coords.shape[0]

    # 6. Reconstruction of image
    reconstruction_result = RECONSTRUCTION_ALGORITHM_HANDLES[
        task['reconstruction_algorithm']](locals())
    reconstructed_img_vec = reconstruction_result[0]
    reconstruction_time = reconstruction_result[1]
    reconstructed_coefficients_vec = reconstruction_result[2]
    algorithm_save_vec = reconstruction_result[3]
    assert reconstructed_img_vec.shape == test_img_vec.shape
    assert reconstruction_time >= 0

    # 7. Evaluation
    psnr, ssim = _utils.evaluate_reconstruction(
        test_img_detilt_vec, reconstructed_img_vec,
        fixed_quantities['ssim_dynamic_range'], h, w)

    # 8. Save results in database
    _utils.store_results(result_folder, h5_name, task, locals())

    # 9. Save summary figure
    _utils.save_summary_figure(result_folder, task, locals())

    # Log finish of task
    pprint('Finished task:')
    pprint(task)


# Run the simulation
if __name__ == '__main__':
    """
    FIXME: ADD DESCRIPTION 
    **Configuration**
    Fixed quantities are configured below.
    Simulation combinations are configured in the `get_tasks` function above.

    **Requirements**

    To run the script, the following must be available:

    - The seven AFM cell images from http://dx.doi.org/10.5281/zenodo.17573
    IMAGE       MD5SUM / SHA256SUM
    image_0.mi  f80ef9f86036271640603980a8f2b903 /
      9cc4586865cb9ca32bbad51d0370f2fbfdc4e478b5af7661c429de0a60712f0d
    image_1.mi  76634621003a2745e7e9efc513321371 /
      1d0ff7d0bd97335d68ddfdaac3068f418d007d78c4e57abdbf75c08a902e2226
    image_2.mi  d2176dd6ec617107311d32961f040253 /
      8a0667f40464e190edb569412c43ce1fb9b9f1a7c8b604d404ee753983b10144
    image_3.mi  1ab520563ac6648d02a1999cff739c3b /
      3e8eaa89cda0f9626f44e9b36d5220f5a720b8ea23a5843a4a87c40aef704cdb
    image_4.mi  0b17690bcd33f9461d2547e57fbfe582 /
      8f707180538c72d5af5d89248e2b3be6812a867de52af7f3a9619a4afab3cb7b
    image_5.mi  dd16745c58604a2b206bd22a69545314 /
      bb04c99e6aae4a8fa65e6594a66f0cfd44f42f314d846e7f09d365260f6aae46
    image_6.mi  b8d685a210f151f11b11348f7ad5115a /
      b2a6e00b8f237987b832e5da1fe3731cc63a3918bf9af5f5d0bfcd423dbf66ba

    **Test run**
    python reconstruction_goblet.py 0 1 --test

    """

    # Argument parsing
    arg_parser = argparse.ArgumentParser(description='Reconstruction Goblet')
    arg_parser.add_argument(
        'ID', action='store', type=int, help='Worker ID')
    arg_parser.add_argument(
        'Nodes', action='store', type=int,
        help='Number of nodes used in simulation.')
    arg_parser.add_argument(
        '--test', action='store_true',
        help='Only run a small test batch of simulation tasks.')
    arg_parser.add_argument(
        '--shuffle', action='store_true',
        help='Shuffle task list.')
    arg_parser.add_argument(
        '--algorithms', action='store', type=str, nargs='*', default='all',
        choices=RECONSTRUCTION_ALGORITHM_HANDLES.keys(),
        help='Only run simulation for the specified algorithms.')
    args = arg_parser.parse_args()

    # Setup
    fixed_quantities = {
        'image_size': 128,  # Image size to use in simulations
        'ssim_dynamic_range': 1,  # Dynamic range used in SSIM calculation.
        'gaussian_model_a_parameter': 0.0015  # Scale factor in Gassian model
        }

    result_folder = './data/'
    if not os.path.isdir(result_folder):
        os.makedirs(result_folder)

    img_folder = './orig_images/'
    h5_name = 'reconstruction_goblet_ID_{}_of_{}.hdf5'.format(args.ID,
                                                              args.Nodes)

    pprint('SETUP')
    pprint('='*80)
    pprint('Fixed quantities:')
    pprint(fixed_quantities)
    pprint('Algorithms to test:')
    pprint(args.algorithms)
    pprint('ID: {}'.format(args.ID))
    pprint('Nodes: {}'.format(args.Nodes))
    pprint('Results folder: {}'.format(result_folder))
    pprint('Image folder: {}'.format(img_folder))
    pprint('Database name: {}'.format(h5_name))
    pprint('Task array shuffled: {}'.format(args.shuffle))

    # Custom annotations
    #FIXME: USE CUSTOM ANNOTATIONS HERE

    # Get tasks
    group_structure, task_setup_summary, tasks = get_tasks(
        args.ID, args.Nodes, args.algorithms, args.shuffle)
    expected_rows = len(tasks)
    h5_path = result_folder + h5_name
    _utils.create_database(h5_path, group_structure, task_setup_summary,
                           expected_rows)

    kwargs = [{'img_folder': img_folder, 'result_folder': result_folder,
               'h5_name': h5_name, 'task': task,
               'fixed_quantities': fixed_quantities} for task in tasks]
    pprint('Total number of simulation tasks: {}'.format(len(tasks)))

    # Run simulation
    if int(psutil.__version__[0]) >= 3:
        # Logical cpu_count only equals number of cores for psutil > 3.0.0
        # https://github.com/giampaolo/psutil/pull/641
        workers = psutil.cpu_count(logical=False)
    else:
        workers = int(psutil.cpu_count() / 2)  # Assume 2 HW threads per core

    magni.utils.multiprocessing.config.update(workers=workers)

    if args.test:
        for kwarg in kwargs[:10]:
            run_simulation_task(**kwarg)
    else:
        magni.utils.multiprocessing.process(
            run_simulation_task, kwargs_list=kwargs, maxtasks=1)

        # Assert that all results are stored in database
        h5_file = result_folder + '/' + h5_name
        with magni.utils.multiprocessing.File(h5_file, mode='a') as h5file:
            metric_table = h5file.get_node('/simulation_results/metrics')
            assert metric_table.nrows == len(tasks)
