import numpy as np
import math as m
from typing import Callable
from scipy import optimize
import numpy.typing as npt
np.float_ = np.float64

_VOIGT_MATRIX: npt.NDArray[int] = np.array([[0, 5, 4], [5, 1, 3], [4, 3, 2]])


def _compute_voigt_coefficient(p: int, q: int) -> float:
    return 1. / ((1 + p // 3) * (1 + q // 3))


def _compute_4th_order_tensor_from_6x6_matrix(matrix: npt.NDArray[np.float_]) -> npt.NDArray[np.float_]:
    output_tensor = [[[[_compute_voigt_coefficient(_VOIGT_MATRIX[i, j], _VOIGT_MATRIX[k, l]) *
                        matrix[_VOIGT_MATRIX[i, j]][_VOIGT_MATRIX[k, l]]
                        for i in range(3)] for j in range(3)] for k in range(3)] for l in range(3)]

    return np.asarray(output_tensor)

def compute_direction_vector_spherical_to_cartesian(theta : float, phi : float, chi : float = None) -> list[float]:
    if chi is None:
        return [m.sin(theta)*m.cos(phi), m.sin(theta)*m.sin(phi), m.cos(theta)]
    else:
        return [m.cos(theta)*m.cos(phi)*m.cos(chi) - m.sin(phi)*m.sin(chi),
             m.cos(theta)*m.sin(phi)*m.cos(chi) + m.cos(phi)*m.sin(chi),
             - m.sin(theta)*m.cos(chi)]

def minimize_elastic_constant_function(elastic_constant_function: Callable[[], float], dimension : int):
    if dimension == 2:
        optimize_range = ((0, np.pi), (0, np.pi))
        n_grid_points = 25
    elif dimension == 3:
        optimize_range = ((0, np.pi), (0, np.pi), (0, np.pi))
        n_grid_points = 10

    return optimize.brute(elastic_constant_function, optimize_range, Ns = n_grid_points, full_output = True, finish = optimize.fmin)[0:2]

def maximize_elastic_constant_function(elastic_constant_function: Callable[[], float], dimension : int):
    temp = minimize_elastic_constant_function(lambda x: -elastic_constant_function(x), dimension)
    return temp[0], -temp[1]

def make_planar_plot_data(data_x: npt.NDArray[np.float_], data_y: npt.NDArray[np.float_]) -> tuple[
    npt.NDArray[np.float_], npt.NDArray[np.float_]]:
    """Prepares 2d plot data for planar plotting"""
    data_x = np.append(data_x, -data_x)
    data_y = np.append(data_y, -data_y)
    return data_x, data_y

#TODO Change minimize and maximize to use tuples when calling function to maximize or minimize

def symmetrize_matrix(matrix: npt.NDArray[np.float_]) -> npt.NDArray[np.float_]:
    """Ensures a nearly symmetrix 6x6 stiffness matrix is symmetric (in case of float value discrepancy)"""
    for i in range(6):
        for j in range(i + 1, 6):
            if np.isclose(matrix[i,j], matrix[j,i]):
                matrix[j,i] = matrix[i,j]
            else:
                raise ValueError("Stiffness matrix cannot be symmetrized")

    return matrix
