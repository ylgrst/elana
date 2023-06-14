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


def test_read_from_mesh_file_must_return_tensor_with_only_ones(tmp_matrix_txt_filename: str) -> None:
    # Arrange

    stiffness_tensor = StiffnessTensor()
    ones = tensor_ones(tmp_matrix_txt_filename)

    # Act

    stiffness_tensor.read_matrix_from_txt(ones)

    # Assert

    assert (stiffness_tensor.matrix == 1.0).all()


def test_trigonal_matrix_read_matrix_from_txt_compare_with_target_must_return_true(trigonal_matrix_filename) -> None:
    # arrange

    trigonal_matrix = StiffnessTensor()
    target_trigonal_matrix = np.array([[87.64, 6.99, 11.91, -17.19, 0.0, 0.0],
                                       [6.99, 87.64, 11.91, 17.19, 0.0, 0.0],
                                       [11.91, 11.91, 107.2, 0.0, 0.0, 0.0],
                                       [-17.19, 17.19, 0.0, 57.94, 0.0, 0.0],
                                       [0.0, 0.0, 0.0, 0.0, 57.94, -17.19],
                                       [0.0, 0.0, 0.0, 0.0, -17.19, 39.88]])

    # act

    trigonal_matrix.read_matrix_from_txt(trigonal_matrix_filename)

    # assert

    assert (trigonal_matrix.matrix == target_trigonal_matrix).all()

def test_voigt_averages_compare_with_target_must_return_true(trigonal_matrix_filename) -> None:
    # arrange

    trigonal_matrix = StiffnessTensor()
    trigonal_matrix.read_matrix_from_txt(trigonal_matrix_filename)
    target_voigt_averages = [38.233, 47.93, 101.41, 0.057923]

    # act
    print(trigonal_matrix)
    voigt_averages = trigonal_matrix.voigt_averages()

    # assert

    assert voigt_averages == target_voigt_averages

def test_reuss_averages_compare_with_target_must_return_true(trigonal_matrix_filename) -> None:
    # arrange

    trigonal_matrix = StiffnessTensor()
    trigonal_matrix.read_matrix_from_txt(trigonal_matrix_filename)
    target_reuss_averages = [37.724, 41.683, 91.389, 0.096239]

    # act
    reuss_averages = trigonal_matrix.reuss_averages()

    # assert

    assert reuss_averages == target_reuss_averages

def test_hill_averages_compare_with_target_must_return_true(trigonal_matrix_filename) -> None:
    # arrange

    trigonal_matrix = StiffnessTensor()
    trigonal_matrix.read_matrix_from_txt(trigonal_matrix_filename)
    target_hill_averages = [37.979, 44.806, 96.478, 0.076612]

    # act
    hill_averages = trigonal_matrix.hill_averages()

    # assert

    assert hill_averages == target_hill_averages
