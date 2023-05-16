import numpy as np
import math as m
from typing import Callable
from scipy import optimize

def compute_direction_vector_spherical_to_cartesian(theta : float, phi : float, chi : float = None) -> list[float]:
    if chi is None:
        return [m.sin(theta)*m.cos(phi), m.sin(theta)]
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



