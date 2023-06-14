import pytest
from elana import StiffnessTensor

import numpy as np

from pathlib import Path


@pytest.fixture(scope="function")
def tmp_dir(tmp_path_factory: pytest.TempPathFactory) -> Path:
    tmp_dir_name = "test_tmp_dir"
    return tmp_path_factory.mktemp(tmp_dir_name)


@pytest.fixture(scope="function")
def tmp_matrix_txt_filename(tmp_dir: Path) -> str:
    return (tmp_dir / "tensor_ones.txt").as_posix()


@pytest.fixture(scope="function")
def trigonal_matrix_filename() -> str:
    return 'trigonal_stiffness_matrix.txt'


def tensor_ones(tmp_matrix_txt_filename) -> str:
    ones = np.ones((6, 6))
    np.savetxt(tmp_matrix_txt_filename, ones)
    return tmp_matrix_txt_filename


def test_trigonal_matrix_read_matrix_from_txt_compare_with_reference_must_return_true(
        trigonal_matrix_filename: str) -> None:
    # arrange

    trigonal_matrix = StiffnessTensor()
    reference_trigonal_matrix = np.array([[87.64, 6.99, 11.91, -17.19, 0.0, 0.0],
                                          [6.99, 87.64, 11.91, 17.19, 0.0, 0.0],
                                          [11.91, 11.91, 107.2, 0.0, 0.0, 0.0],
                                          [-17.19, 17.19, 0.0, 57.94, 0.0, 0.0],
                                          [0.0, 0.0, 0.0, 0.0, 57.94, -17.19],
                                          [0.0, 0.0, 0.0, 0.0, -17.19, 39.88]])

    # act

    trigonal_matrix.read_matrix_from_txt(trigonal_matrix_filename)

    # assert

    assert (trigonal_matrix.matrix == reference_trigonal_matrix).all()


def test_voigt_averages_compare_with_reference_must_return_true(trigonal_matrix_filename: str) -> None:
    # arrange

    trigonal_matrix = StiffnessTensor()
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

    trigonal_matrix = StiffnessTensor()
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

    trigonal_matrix = StiffnessTensor()
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
