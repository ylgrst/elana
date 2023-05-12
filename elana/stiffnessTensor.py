import numpy as np
import numpy.typing as npt
import math as m


class StiffnessTensor:
    """
    Class to manage stiffness tensors in Voigt notation (6x6 matrix)
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
