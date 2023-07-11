import numpy as np
from abstract_stiffness_tensor import AbstractStiffnessTensor

def is_orthorhombic(stiffness_matrix : AbstractStiffnessTensor) -> bool:
    stiffness_matrix_orthorhombic_coefficient_list: list[float] = [stiffness_matrix.matrix[0][3], stiffness_matrix.matrix[0][4],
                                                                   stiffness_matrix.matrix[0][5], stiffness_matrix.matrix[1][3],
                                                                   stiffness_matrix.matrix[1][4], stiffness_matrix.matrix[1][5],
                                                                   stiffness_matrix.matrix[2][3], stiffness_matrix.matrix[2][4],
                                                                   stiffness_matrix.matrix[2][5], stiffness_matrix.matrix[3][4],
                                                                   stiffness_matrix.matrix[3][5], stiffness_matrix.matrix[4][5]]

    return np.isclose(stiffness_matrix_orthorhombic_coefficient_list, [0.0] * 12)


def is_cubic(stiffness_matrix : AbstractStiffnessTensor) -> bool:
    stiffness_matrix_cubic_coefficient_list: list[float] = [stiffness_matrix.matrix[0][3], stiffness_matrix.matrix[0][4], stiffness_matrix.matrix[0][5],
                                                            stiffness_matrix.matrix[1][3], stiffness_matrix.matrix[1][4], stiffness_matrix.matrix[1][5],
                                                            stiffness_matrix.matrix[2][3], stiffness_matrix.matrix[2][4], stiffness_matrix.matrix[2][5],
                                                            stiffness_matrix.matrix[3][4], stiffness_matrix.matrix[3][5], stiffness_matrix.matrix[4][5],
                                                            stiffness_matrix.matrix[0][0] - stiffness_matrix.matrix[1][1],
                                                            stiffness_matrix.matrix[0][0] - stiffness_matrix.matrix[2][2],
                                                            stiffness_matrix.matrix[3][3] - stiffness_matrix.matrix[4][4],
                                                            stiffness_matrix.matrix[3][3] - stiffness_matrix.matrix[5][5],
                                                            stiffness_matrix.matrix[0][1] - stiffness_matrix.matrix[0][2],
                                                            stiffness_matrix.matrix[0][1] - stiffness_matrix.matrix[1][2]]

    return np.isclose(stiffness_matrix_cubic_coefficient_list, [0.0] * 18)
