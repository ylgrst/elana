import pytest
from elana.anisotropic_stiffness_tensor import AnisotropicStiffnessTensor
from elana.plotter import *
import numpy as np

def test_young_2d() -> None:
    stiffness_matrix = AnisotropicStiffnessTensor()
    matrix_file = 'trigonal_stiffness_matrix.txt'
    stiffness_matrix.read_matrix_from_txt(matrix_file)

    plot_young_2d(stiffness_matrix)

    assert 1

def test_linear_compressibility_2d() -> None:
    stiffness_matrix = AnisotropicStiffnessTensor()
    matrix_file = 'trigonal_stiffness_matrix.txt'
    stiffness_matrix.read_matrix_from_txt(matrix_file)

    plot_linear_compressibility_2d(stiffness_matrix)

    assert 1

def test_shear_2d() -> None:
    stiffness_matrix = AnisotropicStiffnessTensor()
    matrix_file = 'trigonal_stiffness_matrix.txt'
    stiffness_matrix.read_matrix_from_txt(matrix_file)

    plot_shear_modulus_2d(stiffness_matrix)

    assert 1

def test_poisson_2d() -> None:
    stiffness_matrix = AnisotropicStiffnessTensor()
    matrix_file = 'trigonal_stiffness_matrix.txt'
    stiffness_matrix.read_matrix_from_txt(matrix_file)

    plot_poisson_2d(stiffness_matrix)

    assert 1

def test_young_3d() -> None:
    stiffness_matrix = AnisotropicStiffnessTensor()
    matrix_file = 'trigonal_stiffness_matrix.txt'
    stiffness_matrix.read_matrix_from_txt(matrix_file)

    plot_young_3d(stiffness_matrix)

    assert 1

def test_linear_compressibility_3d() -> None:
    stiffness_matrix = AnisotropicStiffnessTensor()
    matrix_file = 'trigonal_stiffness_matrix.txt'
    stiffness_matrix.read_matrix_from_txt(matrix_file)

    plot_linear_compressibility_3d(stiffness_matrix)

    assert 1

def test_shear_modulus_3d() -> None:
    stiffness_matrix = AnisotropicStiffnessTensor()
    matrix_file = 'trigonal_stiffness_matrix.txt'
    stiffness_matrix.read_matrix_from_txt(matrix_file)

    plot_shear_modulus_3d(stiffness_matrix)

    assert 1

def test_poisson_3d() -> None:
    stiffness_matrix = AnisotropicStiffnessTensor()
    matrix_file = 'trigonal_stiffness_matrix.txt'
    stiffness_matrix.read_matrix_from_txt(matrix_file)

    plot_poisson_3d(stiffness_matrix)

    assert 1

