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

def tensor_ones(tmp_matrix_txt_filename) -> str:
    ones = np.ones((6,6))
    np.savetxt(tmp_matrix_txt_filename, ones)
    return tmp_matrix_txt_filename

def test_read_from_mesh_file_must_return_tensor_with_only_ones(tmp_matrix_txt_filename : str) -> None:
    # Arrange

    stiffness_tensor = StiffnessTensor()
    ones = tensor_ones(tmp_matrix_txt_filename)

    # Act

    stiffness_tensor.read_matrix_from_txt(ones)

    # Assert

    assert (stiffness_tensor.matrix == 1.0).all()
