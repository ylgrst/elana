import pytest
from elana.anisotropic_stiffness_tensor import AnisotropicStiffnessTensor
from elana.plotter import *
import numpy as np
import numpy.typing as npt

@pytest.fixture
def _reference_trigonal_stiffness_matrix():
    reference_trigonal_stiffness_matrix : npt.NDArray[np.float_] = np.array([[87.64, 6.99, 11.91, -17.19, 0.0, 0.0],
                                                    [6.99, 87.64, 11.91, 17.19, 0.0, 0.0],
                                                    [11.91, 11.91, 107.2, 0.0, 0.0, 0.0],
                                                    [-17.19, 17.19, 0.0, 57.94, 0.0, 0.0],
                                                    [0.0, 0.0, 0.0, 0.0, 57.94, -17.19],
                                                    [0.0, 0.0, 0.0, 0.0, -17.19, 39.88]])

    return reference_trigonal_stiffness_matrix

def test_young_2d(_reference_trigonal_stiffness_matrix) -> None:
    stiffness_matrix = AnisotropicStiffnessTensor(_reference_trigonal_stiffness_matrix)

    plot_young_2d(stiffness_matrix)

    assert 1

def test_linear_compressibility_2d(_reference_trigonal_stiffness_matrix) -> None:
    stiffness_matrix = AnisotropicStiffnessTensor(_reference_trigonal_stiffness_matrix)

    plot_linear_compressibility_2d(stiffness_matrix)

    assert 1

def test_shear_2d(_reference_trigonal_stiffness_matrix) -> None:
    stiffness_matrix = AnisotropicStiffnessTensor(_reference_trigonal_stiffness_matrix)

    plot_shear_modulus_2d(stiffness_matrix)

    assert 1

def test_poisson_2d(_reference_trigonal_stiffness_matrix) -> None:
    stiffness_matrix = AnisotropicStiffnessTensor(_reference_trigonal_stiffness_matrix)

    plot_poisson_2d(stiffness_matrix)

    assert 1

def test_young_3d(_reference_trigonal_stiffness_matrix) -> None:
    stiffness_matrix = AnisotropicStiffnessTensor(_reference_trigonal_stiffness_matrix)

    plot_young_3d(stiffness_matrix)

    assert 1

def test_linear_compressibility_3d(_reference_trigonal_stiffness_matrix) -> None:
    stiffness_matrix = AnisotropicStiffnessTensor(_reference_trigonal_stiffness_matrix)

    plot_linear_compressibility_3d(stiffness_matrix)

    assert 1

def test_shear_modulus_3d(_reference_trigonal_stiffness_matrix) -> None:
    stiffness_matrix = AnisotropicStiffnessTensor(_reference_trigonal_stiffness_matrix)

    plot_shear_modulus_3d(stiffness_matrix)

    assert 1

def test_poisson_3d(_reference_trigonal_stiffness_matrix) -> None:
    stiffness_matrix = AnisotropicStiffnessTensor(_reference_trigonal_stiffness_matrix)

    plot_poisson_3d(stiffness_matrix)

    assert 1

