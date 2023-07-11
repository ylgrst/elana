import pytest
from elana.anisotropic_stiffness_tensor import AnisotropicStiffnessTensor, _compute_4th_order_tensor_from_6x6_matrix

import numpy as np


def test_convert_to_tensor_given_diagonal_eye_matrix_must_return_corresponding_tensor() -> None:
    stiffness_matrix = AnisotropicStiffnessTensor()
    stiffness_matrix.matrix = np.eye(6)

    tensor = _compute_4th_order_tensor_from_6x6_matrix(stiffness_matrix.matrix)

    target_tensor = np.array([[[[1., 0., 0.],
                                [0., 0., 0.],
                                [0., 0., 0.]],
                               [[0., 0.25, 0.],
                                [0.25, 0., 0.],
                                [0., 0., 0.]],
                               [[0., 0., 0.25],
                                [0., 0., 0.],
                                [0.25, 0., 0.]]],
                              [[[0., 0.25, 0.],
                                [0.25, 0., 0.],
                                [0., 0., 0.]],
                               [[0., 0., 0.],
                                [0., 1., 0.],
                                [0., 0., 0.]],
                               [[0., 0., 0.],
                                [0., 0., 0.25],
                                [0., 0.25, 0.]]],
                              [[[0., 0., 0.25],
                                [0., 0., 0.],
                                [0.25, 0., 0.]],
                               [[0., 0., 0.],
                                [0., 0., 0.25],
                                [0., 0.25, 0.]],
                               [[0., 0., 0.],
                                [0., 0., 0.],
                                [0., 0., 1.]]]])

    # ones_list = [tensor[0, 0, 0, 0], tensor[1, 1, 1, 1], tensor[2, 2, 2, 2], tensor[0, 1, 0, 1], tensor[1, 0, 1, 0],
    # tensor[1, 0, 0, 1], tensor[0, 1, 1, 0], tensor[0, 2, 0, 2], tensor[2, 0, 2, 0], tensor[2, 0, 0, 2],
    # tensor[0, 2, 2, 0], tensor[1, 2, 1, 2], tensor[2, 1, 2, 1], tensor[2, 1, 1, 2], tensor[1, 2, 2, 1]]

    # TODO find a way to list sets of indices

    assert (tensor == target_tensor).all()


def test_voigt_averages_compare_with_reference_must_return_true(trigonal_matrix_filename: str) -> None:
    # arrange

    trigonal_matrix = AnisotropicStiffnessTensor()
    trigonal_matrix.read_matrix_from_txt(trigonal_matrix_filename)

    reference_voigt_average_bulk = 38.233
    reference_voigt_average_shear = 47.93
    reference_voigt_average_young = 101.41
    reference_voigt_average_poisson = 0.057923
    reference_voigt_averages = np.asarray(
        [reference_voigt_average_bulk, reference_voigt_average_shear, reference_voigt_average_young,
         reference_voigt_average_poisson])

    # act
    voigt_averages = np.asarray(trigonal_matrix.voigt_averages())

    # assert

    assert np.isclose(voigt_averages, reference_voigt_averages, rtol=1e-3).all()


def test_reuss_averages_compare_with_reference_must_return_true(trigonal_matrix_filename: str) -> None:
    # arrange

    trigonal_matrix = AnisotropicStiffnessTensor()
    trigonal_matrix.read_matrix_from_txt(trigonal_matrix_filename)

    reference_reuss_average_bulk = 37.724
    reference_reuss_average_shear = 41.683
    reference_reuss_average_young = 91.389
    reference_reuss_average_poisson = 0.096239
    reference_reuss_averages = np.asarray(
        [reference_reuss_average_bulk, reference_reuss_average_shear, reference_reuss_average_young,
         reference_reuss_average_poisson])

    # act
    reuss_averages = np.asarray(trigonal_matrix.reuss_averages())

    # assert

    assert np.isclose(reuss_averages, reference_reuss_averages, rtol=1e-3).all()


def test_hill_averages_compare_with_reference_must_return_true(trigonal_matrix_filename: str) -> None:
    # arrange

    trigonal_matrix = AnisotropicStiffnessTensor()
    trigonal_matrix.read_matrix_from_txt(trigonal_matrix_filename)

    reference_hill_average_bulk = 37.979
    reference_hill_average_shear = 44.806
    reference_hill_average_young = 96.478
    reference_hill_average_poisson = 0.076612
    reference_hill_averages = np.asarray(
        [reference_hill_average_bulk, reference_hill_average_shear, reference_hill_average_young,
         reference_hill_average_poisson])

    # act
    hill_averages = np.asarray(trigonal_matrix.hill_averages())

    # assert

    assert np.isclose(hill_averages, reference_hill_averages, rtol=1e-3).all()
