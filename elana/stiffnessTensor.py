import numpy as np
import numpy.typing as npt
import math as m


class StiffnessTensor:
    """
    Class to manage stiffness tensors (6x6 matrix)
    """

    def __init__(self,
                 matrix: npt.NDArray[np.float_] = np.zeros((6, 6)),
                 flexibility_matrix: npt.NDArray[np.float_] = np.zeros((6, 6))) -> None:
        self.matrix = matrix
        self.flexibility_matrix = flexibility_matrix

    def read_matrix_from_txt(self, matrix_file: str) -> None:
        matrix: npt.NDArray[np.float_] = np.loadtxt(matrix_file)

        if matrix.shape != (6, 6):
            raise ValueError("Matrix shape should be 6x6")
        if np.linalg.norm(matrix - matrix.transpose()) > 1e-3:
            raise ValueError("Matrix should be symmetric")

        self.matrix = matrix

        try:
            self.flexibility_matrix = np.linalg.inv(matrix)
        except:
            raise ValueError("Matrix is singular")

    def is_orthorhombic(self) -> bool:
        stiffness_matrix_orthorhombic_coefficient_list: list[float] = [self.matrix[0][3], self.matrix[0][4],
                                                                       self.matrix[0][5], self.matrix[1][3],
                                                                       self.matrix[1][4], self.matrix[1][5],
                                                                       self.matrix[2][3], self.matrix[2][4],
                                                                       self.matrix[2][5], self.matrix[3][4],
                                                                       self.matrix[3][5], self.matrix[4][5]]

        return np.isclose(stiffness_matrix_orthorhombic_coefficient_list, [0.0] * 12)

    def is_cubic(self) -> bool:
        stiffness_matrix_cubic_coefficient_list: list[float] = [self.matrix[0][3], self.matrix[0][4], self.matrix[0][5],
                                                                self.matrix[1][3], self.matrix[1][4], self.matrix[1][5],
                                                                self.matrix[2][3], self.matrix[2][4], self.matrix[2][5],
                                                                self.matrix[3][4], self.matrix[3][5], self.matrix[4][5],
                                                                self.matrix[0][0] - self.matrix[1][1],
                                                                self.matrix[0][0] - self.matrix[2][2],
                                                                self.matrix[3][3] - self.matrix[4][4],
                                                                self.matrix[3][3] - self.matrix[5][5],
                                                                self.matrix[0][1] - self.matrix[0][2],
                                                                self.matrix[0][1] - self.matrix[1][2]]

        return np.isclose(stiffness_matrix_cubic_coefficient_list, [0.0] * 18)
